"""Tests for the persistent admission rate limit on asynchronous rewrap jobs.

Contract under test for ``POST /v1/rewrap-jobs``:

* at most five requests that genuinely create a job per
  (tenant_id, workload_id) per UTC natural minute are admitted; the
  budget is persistent, atomic and isolated per scope, and shares nothing
  with the grant, challenge-issuance or verification budgets;
* only requests whose body, cursor and idempotency-key checks all passed
  and that are neither a keyed replay nor a same-key conflict consume a
  slot — every earlier judgement (422 validation, a 500 keyring failure,
  a stored-response replay and a 409 key conflict) is unchanged and
  spends nothing;
* the reservation commits in the same transaction as the queued job, its
  submission event and (on the keyed path) the idempotency record, so a
  crash can leave neither a counter without its job nor a job without its
  counter;
* an over-budget request gets a 429 whose detail is exactly
  ``rewrap job rate limit exceeded`` with a whole-seconds ``Retry-After``
  header, and writes nothing: no counter, no job, no event, no
  idempotency record, no audit row;
* the new budget is not surfaced through ``GET /v1/observability/
  rate-limits`` or the metrics exposition, whose fields, key order,
  metric names and value conventions are unchanged.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    ChallengeIssuanceCounter,
    RateLimitCounter,
    RewrapJob,
    RewrapJobAdmissionCounter,
    RewrapJobEvent,
    RewrapJobIdempotencyRecord,
    VerificationAdmissionCounter,
)
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V1 = b64url_encode(KEY_V1_BYTES)

IDEMPOTENCY_HEADER = "Idempotency-Key"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = create_app(f"sqlite:///{tmp_path}/job-limits.db")
    yield application
    # No lifespan was entered, so no runner pool runs; just drop the engine.
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _submit(client, *, tenant=TENANT, workload=WORKLOAD, limit=None, key=None):
    body: dict = {"tenant_id": tenant, "workload_id": workload}
    if limit is not None:
        body["limit"] = limit
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else None
    return client.post("/v1/rewrap-jobs", json=body, headers=headers)


def _counter_rows(app):
    with app.state.session_factory() as session:
        return session.scalars(select(RewrapJobAdmissionCounter)).all()


# ---------------------------------------------------------------------------
# Basic shape and the 5-per-minute ceiling
# ---------------------------------------------------------------------------


def test_first_new_job_initializes_current_minute(app, client):
    response = _submit(client)
    assert response.status_code == 202
    assert response.json()["status"] == "queued"

    rows = _counter_rows(app)
    assert len(rows) == 1
    row = rows[0]
    assert (row.tenant_id, row.workload_id) == (TENANT, WORKLOAD)
    assert row.count == 1
    now = datetime.now(timezone.utc)
    assert row.window_start == now.replace(second=0, microsecond=0)
    with app.state.session_factory() as session:
        # The rewrap-job budget uses its own table; none of the other
        # admission budgets is touched.
        assert session.scalars(select(RateLimitCounter)).all() == []
        assert session.scalars(select(ChallengeIssuanceCounter)).all() == []
        assert session.scalars(select(VerificationAdmissionCounter)).all() == []


def test_five_jobs_admitted_sixth_rejected(app, client):
    responses = [_submit(client) for _ in range(5)]
    assert [r.status_code for r in responses] == [202] * 5

    sixth = _submit(client)
    assert sixth.status_code == 429
    assert sixth.json() == {"detail": "rewrap job rate limit exceeded"}
    retry_after = sixth.headers["retry-after"]
    assert retry_after.isdigit()
    assert 1 <= int(retry_after) <= 60

    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == app_module.REWRAP_JOB_ADMISSION_BUDGET_PER_MINUTE


def test_retry_after_counts_down_to_next_utc_minute(app, monkeypatch):
    fixed = datetime(2026, 10, 3, 12, 30, 0, 500_000, tzinfo=timezone.utc)
    client = TestClient(app)
    with monkeypatch.context() as patch:
        patch.setattr(app_module, "_utcnow", lambda: fixed)
        for _ in range(5):
            assert _submit(client).status_code == 202
        at_window_start = _submit(client)
        patch.setattr(
            app_module,
            "_utcnow",
            lambda: fixed.replace(second=45, microsecond=900_000),
        )
        late_in_window = _submit(client)

    assert at_window_start.status_code == 429
    assert late_in_window.status_code == 429
    assert at_window_start.headers["retry-after"] == "60"
    assert late_in_window.headers["retry-after"] == "15"


def test_concurrent_burst_admits_exactly_five(app):
    def call(_):
        return TestClient(app).post(
            "/v1/rewrap-jobs",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        responses = list(pool.map(call, range(10)))
    statuses = [r.status_code for r in responses]

    assert statuses.count(202) == 5
    assert statuses.count(429) == 5
    for response in responses:
        if response.status_code == 429:
            assert response.json() == {"detail": "rewrap job rate limit exceeded"}
            assert response.headers["retry-after"].isdigit()
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 5
    with app.state.session_factory() as session:
        # Exactly the five admitted jobs exist, each with its submission
        # event; no half-admitted request left a row behind.
        assert session.query(RewrapJob).count() == 5
        assert session.query(RewrapJobEvent).count() == 5


def test_429_repeats_do_not_decrement_or_extend(app, client):
    for _ in range(5):
        assert _submit(client).status_code == 202
    assert [_submit(client).status_code for _ in range(4)] == [429] * 4
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 5


def test_429_writes_no_state(app, client):
    for _ in range(5):
        assert _submit(client).status_code == 202
    with app.state.session_factory() as session:
        jobs_before = {
            row.job_id: (row.status, row.next_cursor)
            for row in session.scalars(select(RewrapJob))
        }
        events_before = {row.event_id for row in session.scalars(select(RewrapJobEvent))}
        audits_before = {row.event_id for row in session.scalars(select(AuditEvent))}

    rejected = _submit(client, key="over-budget-key")
    assert rejected.status_code == 429
    assert "job_id" not in rejected.text

    with app.state.session_factory() as session:
        # No new job, no event, no idempotency record, no audit row; the
        # admitted jobs are untouched, cursors included.
        assert {
            row.job_id: (row.status, row.next_cursor)
            for row in session.scalars(select(RewrapJob))
        } == jobs_before
        assert {
            row.event_id for row in session.scalars(select(RewrapJobEvent))
        } == events_before
        assert {
            row.event_id for row in session.scalars(select(AuditEvent))
        } == audits_before
        assert session.scalars(select(RewrapJobIdempotencyRecord)).all() == []


# ---------------------------------------------------------------------------
# Earlier judgements never consume the budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"workload_id": ""},
        {"limit": 0},
        {"limit": 201},
        {"limit": "50"},
        {"cursor": "not a cursor"},
        {"cursor": "forged-but-well-formed-AAAA"},
    ],
)
def test_invalid_bodies_are_422_and_never_consume_budget(app, client, overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    body.update(overrides)
    for _ in range(app_module.REWRAP_JOB_ADMISSION_BUDGET_PER_MINUTE + 2):
        assert client.post("/v1/rewrap-jobs", json=body).status_code == 422
    assert _counter_rows(app) == []
    # The next fully valid request is still the first admitted.
    assert _submit(client).status_code == 202


@pytest.mark.parametrize(
    "key",
    ["", "has space", "tab\there", "x" * 65],
)
def test_invalid_idempotency_key_is_422_and_not_counted(app, client, key):
    for _ in range(7):
        response = client.post(
            "/v1/rewrap-jobs",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
            headers={IDEMPOTENCY_HEADER: key},
        )
        assert response.status_code == 422
    assert _counter_rows(app) == []


def test_non_ascii_idempotency_key_is_422_and_not_counted(app):
    # httpx cannot carry a non-ASCII header value, so the request is
    # driven through a raw ASGI call, exactly as the idempotency-header
    # validation tests do.
    import asyncio

    async def _call():
        payload = json.dumps(
            {"tenant_id": TENANT, "workload_id": WORKLOAD}
        ).encode()
        status: dict = {}
        chunks: list[bytes] = []

        async def receive():
            return {
                "type": "http.request",
                "body": payload,
                "more_body": False,
            }

        async def send(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))

        path = "/v1/rewrap-jobs"
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
                (b"idempotency-key", "non-ascii-🔐".encode("utf-8")),
            ],
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
        await app(scope, receive, send)
        return status["code"]

    for _ in range(7):
        assert asyncio.run(_call()) == 422
    assert _counter_rows(app) == []


def test_keyring_failure_is_500_and_not_counted(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    for _ in range(7):
        response = _submit(client)
        assert response.status_code == 500
        assert response.json() == {"detail": "encryption unavailable"}
    assert _counter_rows(app) == []
    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 0


def test_keyring_failure_on_keyed_path_is_500_and_not_counted(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    for _ in range(7):
        response = _submit(client, key="keyed-500")
        assert response.status_code == 500
        assert response.json() == {"detail": "encryption unavailable"}
    assert _counter_rows(app) == []
    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 0
        assert session.scalars(select(RewrapJobIdempotencyRecord)).all() == []


# ---------------------------------------------------------------------------
# Idempotency interactions
# ---------------------------------------------------------------------------


def test_keyed_submission_consumes_exactly_one_slot(app, client):
    first = _submit(client, key="job-key-1")
    assert first.status_code == 202
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 1


def test_keyed_replay_returns_stored_202_without_budget(app, client):
    first = _submit(client, key="job-key-1")
    assert first.status_code == 202
    for _ in range(10):
        replay = _submit(client, key="job-key-1")
        assert replay.status_code == 202
        assert replay.content == first.content
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 1
    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 1
        assert session.query(RewrapJobEvent).count() == 1


def test_keyed_replay_is_admitted_even_when_budget_is_full(app, client):
    first = _submit(client, key="job-key-1")
    assert first.status_code == 202
    # Fill the minute's budget with four more distinct new jobs.
    for n in range(4):
        assert _submit(client, key=f"job-key-{n + 2}").status_code == 202
    assert _submit(client).status_code == 429

    # A verbatim replay of the stored key is still the saved 202, never a
    # 429, and consumes nothing.
    replay = _submit(client, key="job-key-1")
    assert replay.status_code == 202
    assert replay.content == first.content
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 5
    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 5


def test_same_key_different_request_is_409_and_not_counted(app, client):
    assert _submit(client, key="job-key-1", limit=10).status_code == 202
    for _ in range(7):
        conflict = _submit(client, key="job-key-1", limit=11)
        assert conflict.status_code == 409
        assert conflict.json() == {
            "detail": "idempotency key reused with a different request"
        }
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 1
    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 1


def test_429_on_keyed_submission_stores_no_record(app, client, monkeypatch):
    # Exhaust the budget with keyless jobs, then a keyed submission is
    # rejected and stores nothing: the same key is a fresh first
    # submission once the quota recovers.
    fixed = datetime(2026, 10, 3, 12, 30, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(app_module, "_utcnow", lambda: fixed)
    for _ in range(5):
        assert _submit(client).status_code == 202
    rejected = _submit(client, key="late-key")
    assert rejected.status_code == 429
    with app.state.session_factory() as session:
        assert session.scalars(select(RewrapJobIdempotencyRecord)).all() == []

    monkeypatch.setattr(
        app_module, "_utcnow", lambda: fixed + timedelta(minutes=1)
    )
    recovered = _submit(client, key="late-key")
    assert recovered.status_code == 202
    replay = _submit(client, key="late-key")
    assert replay.status_code == 202
    assert replay.content == recovered.content


# ---------------------------------------------------------------------------
# Scope isolation, UTC minute recovery and restart durability
# ---------------------------------------------------------------------------


def test_scopes_are_strictly_isolated_with_no_borrowing(app, client):
    for _ in range(5):
        assert _submit(client).status_code == 202
    assert _submit(client).status_code == 429

    assert _submit(client, tenant="tenant-b").status_code == 202
    assert _submit(client, workload="workload-2").status_code == 202

    counts = {
        (row.tenant_id, row.workload_id): row.count for row in _counter_rows(app)
    }
    assert counts == {
        (TENANT, WORKLOAD): 5,
        ("tenant-b", WORKLOAD): 1,
        (TENANT, "workload-2"): 1,
    }


def test_quota_recovers_at_next_utc_minute(app, client, monkeypatch):
    # Pin the clock so the test cannot straddle a real minute boundary.
    fixed = datetime(2026, 10, 3, 12, 30, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(app_module, "_utcnow", lambda: fixed)
    for _ in range(5):
        assert _submit(client).status_code == 202
    assert _submit(client).status_code == 429

    # The next UTC minute has an independent budget.
    monkeypatch.setattr(
        app_module, "_utcnow", lambda: fixed + timedelta(minutes=1)
    )
    assert _submit(client).status_code == 202
    rows = {row.window_start: row.count for row in _counter_rows(app)}
    assert rows == {fixed: 5, fixed + timedelta(minutes=1): 1}


def test_previous_minute_counter_does_not_limit_new_minute(app, client):
    with app.state.session_factory() as session:
        session.add(
            RewrapJobAdmissionCounter(
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                window_start=datetime.now(timezone.utc).replace(
                    second=0, microsecond=0
                )
                - timedelta(minutes=1),
                count=500,
            )
        )
        session.commit()

    assert _submit(client).status_code == 202


def test_budget_persists_across_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart-job-limits.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    for _ in range(5):
        assert _submit(client1).status_code == 202
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    # The sixth new job in the same UTC minute is still refused.
    response = _submit(client2)
    assert response.status_code == 429
    assert response.json() == {"detail": "rewrap job rate limit exceeded"}
    app2.state.engine.dispose()


def test_old_database_opens_and_submits_without_migration(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/legacy-jobs.db"
    first = create_app(url)
    first.state.engine.dispose()

    import sqlite3

    with sqlite3.connect(tmp_path / "legacy-jobs.db") as conn:
        conn.execute("DROP TABLE rewrap_job_admission_counters")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "rewrap_jobs" in tables
    assert "rewrap_job_admission_counters" not in tables

    # Startup on the old database is additive: the table is created, no
    # caller-side migration is required, and submission works.
    app = create_app(url)
    response = _submit(TestClient(app))
    assert response.status_code == 202
    app.state.engine.dispose()


# ---------------------------------------------------------------------------
# Independence from the other budgets and from the batch endpoint
# ---------------------------------------------------------------------------


def test_job_budget_does_not_touch_other_budgets(app, client):
    for _ in range(5):
        assert _submit(client).status_code == 202
    assert _submit(client).status_code == 429

    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []
        assert session.scalars(select(ChallengeIssuanceCounter)).all() == []
        assert session.scalars(select(VerificationAdmissionCounter)).all() == []

    # Challenge issuance still has its own full budget...
    for _ in range(5):
        created = client.post(
            "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
        )
        assert created.status_code == 201
    with app.state.session_factory() as session:
        # ...and issuing challenges never draws from the rewrap-job
        # budget, which remains exactly the five consumed slots.
        row = session.scalar(select(RewrapJobAdmissionCounter))
        assert row.count == 5


def test_rewrap_batches_are_not_limited_and_do_not_consume(app, client):
    for _ in range(5):
        assert _submit(client).status_code == 202
    assert _submit(client).status_code == 429

    # The synchronous batch endpoint has no admission budget of its own
    # and is unaffected by an exhausted job budget.
    for _ in range(7):
        batch = client.post(
            "/v1/rewrap-batches",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
        assert batch.status_code == 200
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 5


# ---------------------------------------------------------------------------
# Counter failure
# ---------------------------------------------------------------------------


def test_counter_unavailable_returns_500_and_creates_nothing(app, client):
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_job_admission_counters"))

    response = _submit(client)
    assert response.status_code == 500
    assert response.json() == {"detail": "rewrap job unavailable"}

    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 0
        assert session.query(RewrapJobEvent).count() == 0
        assert session.scalars(select(RewrapJobIdempotencyRecord)).all() == []
        assert session.scalars(select(AuditEvent)).all() == []

    # Once the counter is available again, the request is the first
    # admitted slot of the minute.
    RewrapJobAdmissionCounter.__table__.create(app.state.engine)
    recovered = _submit(client)
    assert recovered.status_code == 202
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 1


def test_counter_unavailable_on_keyed_path_returns_500_and_stores_nothing(
    app, client
):
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_job_admission_counters"))

    response = _submit(client, key="unavailable-key")
    assert response.status_code == 500
    assert response.json() == {"detail": "rewrap job unavailable"}

    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 0
        assert session.scalars(select(RewrapJobIdempotencyRecord)).all() == []

    # The key was never recorded: after recovery the same key is a fresh
    # first submission, not a replay or a conflict.
    RewrapJobAdmissionCounter.__table__.create(app.state.engine)
    recovered = _submit(client, key="unavailable-key")
    assert recovered.status_code == 202


# ---------------------------------------------------------------------------
# Observability compatibility: the new budget is not surfaced
# ---------------------------------------------------------------------------


def test_rate_limits_observability_is_unchanged_by_job_budget(app, client):
    for _ in range(5):
        assert _submit(client).status_code == 202
    assert _submit(client).status_code == 429

    response = client.get(
        "/v1/observability/rate-limits",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    data = json.loads(response.content)
    # Exactly the pre-existing fields, in the pre-existing order; the
    # rewrap-job budget appears nowhere.
    assert list(data) == [
        "tenant_id",
        "workload_id",
        "window_start",
        "reset_at",
        "challenge_issuance",
        "verification",
        "grant_actions",
    ]
    for budget in ("challenge_issuance", "verification", "grant_actions"):
        assert data[budget] == {"limit": 5, "used": 0, "remaining": 5}


def test_metrics_are_unchanged_by_job_budget(app, client):
    for _ in range(5):
        assert _submit(client).status_code == 202
    assert _submit(client).status_code == 429

    response = client.get(
        "/v1/observability/metrics",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    text_body = response.text
    # The budget label set is exactly the three pre-existing budgets.
    budget_labels = {
        line.split('budget="')[1].split('"')[0]
        for line in text_body.splitlines()
        if 'budget="' in line
    }
    assert budget_labels == {"challenge_issuance", "grant_actions", "verification"}
    for line in text_body.splitlines():
        if line.startswith("proof_release_rate_limit_budget_used"):
            assert line.endswith(" 0")
