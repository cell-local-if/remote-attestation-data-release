"""Tests for the persistent admission rate limit on ``POST /v1/rewrap-jobs``.

Contract under test:

* only requests that genuinely create a job count against a budget of
  five new jobs per UTC natural minute per (tenant_id, workload_id);
* the existing 422 responses for body/cursor/idempotency-key shapes, the
  keyed 202 replay, the 409 same-key-different-request conflict and the
  500 keyring failure are unchanged and never consume the budget;
* the sixth new-job request in a minute gets a 429 whose body carries
  only the fixed detail ``rewrap job rate limit exceeded`` and whose
  ``Retry-After`` header carries the whole seconds to the next UTC
  minute;
* under concurrent submission exactly five distinct new requests get a
  202 and a queued job with its submitted event; the rest get 429;
* a rejected request creates no job, event, idempotency record or audit
  row and changes no existing job or envelope cursor;
* scopes are strictly isolated with no borrowing, the quota recovers at
  the next UTC minute, and the current window count survives a process
  restart;
* admission is independent of the challenge-issuance, verification and
  one-time-grant budgets, and the observability summary and metrics keep
  their existing three-budget shape;
* a counter failure is a 500 and leaves neither a half job nor a half
  count.

As in test_rewrap_jobs.py the default app is built without entering its
lifespan so no worker pool runs and the durable rows stay exactly as
committed.
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
from proof_release.envelopes import MasterKeyError, b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"

KEY_V1 = b64url_encode(b"0123456789abcdef0123456789abcdef")

IDEMPOTENCY_HEADER = "Idempotency-Key"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = create_app(f"sqlite:///{tmp_path}/rewrap-job-limits.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _submit(client, *, tenant=TENANT, workload=WORKLOAD, key=None, **overrides):
    body = {"tenant_id": tenant, "workload_id": workload}
    body.update(overrides)
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else None
    return client.post("/v1/rewrap-jobs", json=body, headers=headers)


def _counter_rows(app):
    with app.state.session_factory() as session:
        return session.scalars(select(RewrapJobAdmissionCounter)).all()


# ---------------------------------------------------------------------------
# Basic shape and the 5-per-minute ceiling
# ---------------------------------------------------------------------------


def test_first_new_job_initializes_current_minute_with_one(app, client):
    response = _submit(client)
    assert response.status_code == 202

    rows = _counter_rows(app)
    assert len(rows) == 1
    row = rows[0]
    assert (row.tenant_id, row.workload_id) == (TENANT, WORKLOAD)
    assert row.count == 1
    now = datetime.now(timezone.utc)
    assert row.window_start == now.replace(second=0, microsecond=0)
    # The admission budget uses its own table; no other budget row is
    # ever created by a rewrap-job submission.
    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []
        assert session.scalars(select(ChallengeIssuanceCounter)).all() == []
        assert session.scalars(select(VerificationAdmissionCounter)).all() == []


def test_five_jobs_are_accepted_and_queued(client):
    responses = [_submit(client) for _ in range(5)]
    assert [r.status_code for r in responses] == [202] * 5
    assert len({r.json()["job_id"] for r in responses}) == 5
    assert all(r.json()["status"] == "queued" for r in responses)


def test_sixth_request_returns_429_with_retry_after(client):
    for _ in range(5):
        assert _submit(client).status_code == 202

    sixth = _submit(client)
    assert sixth.status_code == 429
    assert sixth.json() == {"detail": "rewrap job rate limit exceeded"}
    retry_after = sixth.headers["retry-after"]
    assert retry_after.isdigit()
    assert 1 <= int(retry_after) <= 60


def test_retry_after_counts_down_to_next_utc_minute(app, monkeypatch):
    fixed = datetime(2026, 10, 3, 12, 30, 0, 500_000, tzinfo=timezone.utc)
    client = TestClient(app)
    with monkeypatch.context() as patch:
        # All requests share the fixed minute window 12:30; only the
        # response instant of the rejected requests varies.
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


def test_ten_concurrent_new_requests_admit_exactly_five(app):
    def call(index):
        return TestClient(app).post(
            "/v1/rewrap-jobs",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": 10 + index},
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

    with app.state.session_factory() as session:
        row = session.scalar(select(RewrapJobAdmissionCounter))
        assert row.count == 5
        jobs = session.scalars(
            select(RewrapJob).where(
                RewrapJob.tenant_id == TENANT,
                RewrapJob.workload_id == WORKLOAD,
            )
        ).all()
        assert len(jobs) == 5
        assert all(job.status == "queued" for job in jobs)
        # Every admitted job was born with exactly its submitted event.
        events = session.scalars(select(RewrapJobEvent)).all()
        assert len(events) == 5
        assert {event.job_id for event in events} == {job.job_id for job in jobs}


def test_429_writes_nothing_and_changes_nothing(app, client):
    accepted = [_submit(client).json() for _ in range(5)]
    rejected = [_submit(client).status_code for _ in range(4)]
    assert rejected == [429, 429, 429, 429]

    with app.state.session_factory() as session:
        row = session.scalar(select(RewrapJobAdmissionCounter))
        assert row.count == app_module.REWRAP_JOB_BUDGET_PER_MINUTE
        jobs = session.scalars(select(RewrapJob)).all()
        assert {job.job_id for job in jobs} == {j["job_id"] for j in accepted}
        assert all(job.status == "queued" for job in jobs)
        assert all(job.next_cursor == job.cursor for job in jobs)
        # No extra events, no idempotency records, no audit rows.
        events = session.scalars(select(RewrapJobEvent)).all()
        assert len(events) == 5
        assert session.scalars(select(RewrapJobIdempotencyRecord)).all() == []
        assert session.scalars(select(AuditEvent)).all() == []


# ---------------------------------------------------------------------------
# Idempotency interplay: replays and conflicts never touch the budget
# ---------------------------------------------------------------------------


def test_keyed_submission_consumes_one_slot_and_replay_consumes_none(app, client):
    first = _submit(client, key="key-1")
    assert first.status_code == 202
    for _ in range(4):
        assert _submit(client).status_code == 202
    assert _submit(client).status_code == 429

    # The verbatim replay is answered from the stored record even though
    # the minute's budget is exhausted: same bytes, no new job, no quota.
    replay = _submit(client, key="key-1")
    assert replay.status_code == 202
    assert replay.content == first.content

    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 5
    with app.state.session_factory() as session:
        assert len(session.scalars(select(RewrapJob)).all()) == 5
        records = session.scalars(select(RewrapJobIdempotencyRecord)).all()
        assert len(records) == 1


def test_keyed_submissions_fill_the_budget_and_the_sixth_is_429(app, client):
    for index in range(5):
        assert _submit(client, key=f"key-{index}").status_code == 202

    sixth = _submit(client, key="key-6")
    assert sixth.status_code == 429
    assert sixth.json() == {"detail": "rewrap job rate limit exceeded"}

    with app.state.session_factory() as session:
        # The rejected keyed request left no idempotency record behind.
        records = session.scalars(select(RewrapJobIdempotencyRecord)).all()
        assert len(records) == 5
        assert all(record.idempotency_key != "key-6" for record in records)
        assert len(session.scalars(select(RewrapJob)).all()) == 5


def test_same_key_different_request_is_409_and_consumes_no_quota(app, client):
    assert _submit(client, key="key-1", limit=10).status_code == 202
    for _ in range(4):
        assert _submit(client).status_code == 202
    assert _submit(client).status_code == 429

    # The conflict check still runs with an exhausted budget: a stable
    # 409 that creates nothing and spends nothing.
    conflict = _submit(client, key="key-1", limit=99)
    assert conflict.status_code == 409

    rows = _counter_rows(app)
    assert rows[0].count == 5
    with app.state.session_factory() as session:
        assert len(session.scalars(select(RewrapJob)).all()) == 5
        assert len(session.scalars(select(RewrapJobIdempotencyRecord)).all()) == 1


# ---------------------------------------------------------------------------
# 422 and 500 precede the counter: no budget consumed, no state written
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"tenant_id": "", "workload_id": "w"},
        {"tenant_id": "t"},
        {"tenant_id": "t", "workload_id": ""},
        {"tenant_id": 1, "workload_id": "w"},
        {"tenant_id": "t", "workload_id": "w", "limit": 0},
        {"tenant_id": "t", "workload_id": "w", "limit": "10"},
        {"tenant_id": "t", "workload_id": "w", "cursor": "forged-cursor"},
    ],
)
def test_invalid_requests_are_422_and_never_consume_budget(app, client, body):
    for _ in range(app_module.REWRAP_JOB_BUDGET_PER_MINUTE + 2):
        assert client.post("/v1/rewrap-jobs", json=body).status_code == 422

    assert _counter_rows(app) == []
    with app.state.session_factory() as session:
        assert session.scalars(select(RewrapJob)).all() == []

    # The next well-formed request is still the first admitted.
    assert _submit(client).status_code == 202


def test_invalid_idempotency_key_is_422_and_not_counted(app, client):
    for _ in range(7):
        response = client.post(
            "/v1/rewrap-jobs",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
            headers={IDEMPOTENCY_HEADER: "k" * 65},
        )
        assert response.status_code == 422
    assert _counter_rows(app) == []


def test_keyring_failure_is_500_and_consumes_no_quota(app, client, monkeypatch):
    real_load = app_module.load_keyring

    def failing_load():
        raise MasterKeyError("keyring vanished")

    monkeypatch.setattr(app_module, "load_keyring", failing_load)
    for _ in range(7):
        response = _submit(client)
        assert response.status_code == 500
        assert response.json() == {"detail": "encryption unavailable"}
    # A keyed submission fails the same way before any reservation.
    keyed = _submit(client, key="key-1")
    assert keyed.status_code == 500

    assert _counter_rows(app) == []
    with app.state.session_factory() as session:
        assert session.scalars(select(RewrapJob)).all() == []
        assert session.scalars(select(RewrapJobIdempotencyRecord)).all() == []

    monkeypatch.setattr(app_module, "load_keyring", real_load)
    assert _submit(client).status_code == 202
    assert _submit(client, key="key-1").status_code == 202


# ---------------------------------------------------------------------------
# Scope isolation, UTC minute recovery and restart durability
# ---------------------------------------------------------------------------


def test_scopes_are_strictly_isolated_with_no_borrowing(app, client):
    for _ in range(5):
        assert _submit(client).status_code == 202
    assert _submit(client).status_code == 429

    # A different tenant on the same workload has its own full budget.
    assert _submit(client, tenant="tenant-b").status_code == 202
    # A different workload in the same tenant is likewise independent.
    assert _submit(client, workload="workload-2").status_code == 202

    rows = _counter_rows(app)
    counts = {(r.tenant_id, r.workload_id): r.count for r in rows}
    assert counts == {
        (TENANT, WORKLOAD): 5,
        ("tenant-b", WORKLOAD): 1,
        (TENANT, "workload-2"): 1,
    }


def test_quota_recovers_at_next_utc_minute(app, client):
    accepted = [_submit(client).json()["job_id"] for _ in range(5)]
    assert _submit(client).status_code == 429

    # Move the exhausted row into the previous minute: the current minute
    # has no admission counter yet.
    with app.state.session_factory() as session:
        row = session.scalar(select(RewrapJobAdmissionCounter))
        row.window_start = row.window_start - timedelta(minutes=1)
        session.commit()

    recovered = _submit(client)
    assert recovered.status_code == 202
    assert recovered.json()["job_id"] not in accepted
    with app.state.session_factory() as session:
        rows = {
            r.window_start: r.count
            for r in session.scalars(select(RewrapJobAdmissionCounter)).all()
        }
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        assert rows[now] == 1
        assert rows[now - timedelta(minutes=1)] == 5
        assert len(session.scalars(select(RewrapJob)).all()) == 6


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
    url = f"sqlite:///{tmp_path}/restart-rewrap-limits.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    accepted = [
        client1.post(
            "/v1/rewrap-jobs", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
        )
        for _ in range(5)
    ]
    assert [r.status_code for r in accepted] == [202] * 5
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    # The sixth new job in the same UTC minute is still refused...
    response = client2.post(
        "/v1/rewrap-jobs", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert response.status_code == 429
    assert response.json() == {"detail": "rewrap job rate limit exceeded"}
    # ...and the five jobs accepted before the restart are intact.
    with app2.state.session_factory() as session:
        jobs = session.scalars(select(RewrapJob)).all()
        assert {job.job_id for job in jobs} == {
            r.json()["job_id"] for r in accepted
        }
        assert all(job.status == "queued" for job in jobs)
    app2.state.engine.dispose()


def test_old_database_opens_and_admits_without_migration(tmp_path, monkeypatch):
    # Build a database with a previous deployment, then remove the new
    # admission table as an older deployment would never have created it.
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/legacy.db"
    first = create_app(url)
    first.state.engine.dispose()

    import sqlite3

    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
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
    # caller-side migration is required, and admission works.
    app = create_app(url)
    response = TestClient(app).post(
        "/v1/rewrap-jobs", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert response.status_code == 202
    app.state.engine.dispose()


# ---------------------------------------------------------------------------
# Independence from the other budgets and the observability surface
# ---------------------------------------------------------------------------


def test_admission_budget_is_independent_of_other_budgets(app, client):
    for _ in range(5):
        assert _submit(client).status_code == 202
    assert _submit(client).status_code == 429

    # The exhausted rewrap-job budget shares nothing with challenge
    # issuance: a fully valid challenge request is still admitted.
    challenge = client.post(
        "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert challenge.status_code == 201

    with app.state.session_factory() as session:
        # Rewrap submissions never created rows in the other budgets, and
        # the challenge issuance touched only its own counter.
        assert session.scalars(select(RateLimitCounter)).all() == []
        assert session.scalars(select(VerificationAdmissionCounter)).all() == []
        challenge_row = session.scalar(select(ChallengeIssuanceCounter))
        assert challenge_row.count == 1


def test_observability_summary_keeps_its_three_budget_shape(app, client):
    for _ in range(5):
        assert _submit(client).status_code == 202
    assert _submit(client).status_code == 429

    response = client.get(
        "/v1/observability/rate-limits",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    data = json.loads(response.content)
    # The new budget is not surfaced here: the summary keeps exactly its
    # existing keys and reports the three original budgets as unused.
    assert set(data) == {
        "tenant_id",
        "workload_id",
        "window_start",
        "reset_at",
        "challenge_issuance",
        "verification",
        "grant_actions",
    }
    for key in ("challenge_issuance", "verification", "grant_actions"):
        assert data[key]["used"] == 0
        assert data[key]["remaining"] == data[key]["limit"]


def test_metrics_exposition_has_no_rewrap_admission_series(app, client):
    for _ in range(5):
        assert _submit(client).status_code == 202

    response = client.get(
        "/v1/observability/metrics",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    budget_lines = [
        line
        for line in response.text.splitlines()
        if line.startswith("proof_release_rate_limit_budget_")
    ]
    assert budget_lines
    assert all("rewrap" not in line for line in budget_lines)


# ---------------------------------------------------------------------------
# Counter failure
# ---------------------------------------------------------------------------


def test_counter_unavailable_returns_500_and_creates_nothing(app, client):
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_job_admission_counters"))

    response = _submit(client)
    assert response.status_code == 500

    with app.state.session_factory() as session:
        assert session.scalars(select(RewrapJob)).all() == []
        assert session.scalars(select(RewrapJobEvent)).all() == []
        assert session.scalars(select(RewrapJobIdempotencyRecord)).all() == []
        assert session.scalars(select(AuditEvent)).all() == []

    # Once the counter is available again, the request is the first
    # admitted slot of the minute.
    RewrapJobAdmissionCounter.__table__.create(app.state.engine)
    recovered = _submit(client)
    assert recovered.status_code == 202
    with app.state.session_factory() as session:
        row = session.scalar(select(RewrapJobAdmissionCounter))
        assert row.count == 1
        jobs = session.scalars(select(RewrapJob)).all()
        assert len(jobs) == 1
        assert jobs[0].job_id == recovered.json()["job_id"]
