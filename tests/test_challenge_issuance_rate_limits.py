"""Tests for the persistent issuance rate limit on ``POST /v1/challenges``.

Contract under test:

* only fully validated creation requests count against a budget of five
  issued challenges per UTC natural minute per (tenant_id, workload_id);
* the existing 422 responses for missing/blank/ill-typed fields and an
  out-of-range ttl_seconds are unchanged and never consume the budget;
* the budget is attributed solely to tenant_id and workload_id: the
  challenge id and nonce never enter the window;
* the sixth request in a minute gets a compact 429 carrying only a
  positive-integer ``retry_after_seconds`` followed by one newline;
* under concurrent submission exactly five legal requests get a 201 and a
  consumable pending challenge; the rest get 429;
* a rejected request creates no challenge, mints/returns/persists no
  nonce, changes no existing challenge, and writes no audit;
* scopes are strictly isolated with no borrowing, the quota recovers at
  the next UTC minute, and the current window count survives a process
  restart;
* issuance is independent of the one-time-grant (consume/revoke/release)
  budget;
* a counter read/write failure is a 500 with the fixed detail
  ``rate limit unavailable`` and leaves neither a half challenge nor a
  half count.
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
    Challenge,
    ChallengeIssuanceCounter,
    RateLimitCounter,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/challenge-limits.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, **overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    body.update(overrides)
    return client.post("/v1/challenges", json=body)


def _consume(client, challenge_id, nonce, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": nonce,
    }
    body.update(overrides)
    return client.post(f"/v1/challenges/{challenge_id}/consume", json=body)


# ---------------------------------------------------------------------------
# Basic shape and the 5-per-minute ceiling
# ---------------------------------------------------------------------------


def test_first_valid_request_initializes_current_minute_with_one(app, client):
    response = _create(client)
    assert response.status_code == 201

    with app.state.session_factory() as session:
        rows = session.scalars(select(ChallengeIssuanceCounter)).all()
        assert len(rows) == 1
        row = rows[0]
        assert (row.tenant_id, row.workload_id) == (TENANT, WORKLOAD)
        assert row.count == 1
        now = datetime.now(timezone.utc)
        assert row.window_start == now.replace(second=0, microsecond=0)
        # The issuance budget uses its own table; the grant budget row is
        # never created by a challenge issuance.
        assert session.scalars(select(RateLimitCounter)).all() == []


def test_five_challenges_are_issued_and_pending(client):
    responses = [_create(client) for _ in range(5)]
    assert [r.status_code for r in responses] == [201] * 5
    assert len({r.json()["challenge_id"] for r in responses}) == 5
    assert all(r.json()["status"] == "pending" for r in responses)


def test_sixth_serial_request_returns_429(client):
    for _ in range(5):
        assert _create(client).status_code == 201

    sixth = _create(client)
    assert sixth.status_code == 429
    assert sixth.headers["content-type"] == "application/json"
    raw = sixth.content
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw and b": " not in raw
    data = json.loads(raw)
    assert set(data) == {"retry_after_seconds"}
    value = data["retry_after_seconds"]
    assert isinstance(value, int) and not isinstance(value, bool)
    assert 1 <= value <= 60
    # The body is exactly the compact one-field object plus newline.
    assert raw == (
        json.dumps(
            {"retry_after_seconds": value}, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )


def test_retry_after_decreases_with_response_time(app, monkeypatch):
    fixed = datetime(2026, 10, 3, 12, 30, 0, 500_000, tzinfo=timezone.utc)
    client = TestClient(app)
    with monkeypatch.context() as patch:
        # All requests share the fixed minute window 12:30; only the
        # response instant of the rejected requests varies.
        patch.setattr(app_module, "_utcnow", lambda: fixed)
        for _ in range(5):
            assert _create(client).status_code == 201
        at_window_start = _create(client)
        patch.setattr(
            app_module,
            "_utcnow",
            lambda: fixed.replace(second=45, microsecond=900_000),
        )
        late_in_window = _create(client)

    assert at_window_start.status_code == 429
    assert late_in_window.status_code == 429
    assert at_window_start.json()["retry_after_seconds"] == 60
    assert late_in_window.json()["retry_after_seconds"] == 15


def test_six_concurrent_requests_admit_exactly_five(app):
    client = TestClient(app)

    def call():
        return TestClient(app).post(
            "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: call(), range(6)))

    statuses = [r.status_code for r in responses]
    assert statuses.count(201) == 5
    assert statuses.count(429) == 1

    with app.state.session_factory() as session:
        row = session.scalar(select(ChallengeIssuanceCounter))
        assert row.count == 5
        challenges = session.scalars(
            select(Challenge).where(
                Challenge.tenant_id == TENANT,
                Challenge.workload_id == WORKLOAD,
            )
        ).all()
        assert len(challenges) == 5
        assert all(c.status == "pending" for c in challenges)


def test_larger_concurrent_burst_admits_exactly_five(app):
    def call():
        return TestClient(app).post(
            "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        statuses = list(
            pool.map(
                lambda _: call().status_code,
                range(32),
            )
        )

    assert statuses.count(201) == 5
    assert statuses.count(429) == 27


def test_429_repeats_do_not_decrement_or_extend(app, client):
    issued = [_create(client).json() for _ in range(5)]
    rejected = [_create(client).status_code for _ in range(4)]
    assert rejected == [429, 429, 429, 429]

    with app.state.session_factory() as session:
        row = session.scalar(select(ChallengeIssuanceCounter))
        assert row.count == app_module.CHALLENGE_ISSUANCE_BUDGET_PER_MINUTE
        challenges = session.scalars(select(Challenge)).all()
        assert {c.challenge_id for c in challenges} == {
            c["challenge_id"] for c in issued
        }


def test_429_carries_no_nonce_and_touches_nothing(app, client):
    for _ in range(5):
        _create(client)

    rejected = _create(client)
    assert rejected.status_code == 429
    assert "nonce" not in rejected.text
    assert "challenge_id" not in rejected.text

    with app.state.session_factory() as session:
        # Exactly five challenges, all untouched and still pending.
        challenges = session.scalars(select(Challenge)).all()
        assert len(challenges) == 5
        assert all(c.status == "pending" for c in challenges)
        # Challenge issuance appends no audit rows.
        assert session.scalars(select(AuditEvent)).all() == []


def test_admitted_challenges_are_consumable(client):
    issued = [_create(client).json() for _ in range(5)]
    for created in issued:
        consumed = _consume(client, created["challenge_id"], created["nonce"])
        assert consumed.status_code == 200
        assert consumed.json()["status"] == "consumed"


# ---------------------------------------------------------------------------
# 422 precedes the counter: no budget consumed, no state written
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"tenant_id": "", "workload_id": "w"},
        {"tenant_id": "   ", "workload_id": "w"},
        {"tenant_id": "t"},
        {"tenant_id": "t", "workload_id": ""},
        {"tenant_id": 1, "workload_id": "w"},
        {"tenant_id": "t", "workload_id": "w", "ttl_seconds": 29},
        {"tenant_id": "t", "workload_id": "w", "ttl_seconds": 901},
        {"tenant_id": "t", "workload_id": "w", "ttl_seconds": "300"},
        {"tenant_id": "t", "workload_id": "w", "ttl_seconds": 300.5},
        {"tenant_id": "t", "workload_id": "w", "ttl_seconds": True},
    ],
)
def test_invalid_requests_are_422_and_never_consume_budget(app, client, body):
    for _ in range(app_module.CHALLENGE_ISSUANCE_BUDGET_PER_MINUTE + 2):
        assert client.post("/v1/challenges", json=body).status_code == 422

    with app.state.session_factory() as session:
        assert session.scalars(select(ChallengeIssuanceCounter)).all() == []
        assert session.scalars(select(Challenge)).all() == []

    # The next well-formed request is still the first admitted.
    assert _create(client).status_code == 201


def test_non_object_body_is_422_and_not_counted(app, client):
    for _ in range(7):
        assert client.post("/v1/challenges", json=["not", "an", "object"]).status_code == 422
    with app.state.session_factory() as session:
        assert session.scalars(select(ChallengeIssuanceCounter)).all() == []


# ---------------------------------------------------------------------------
# Scope isolation, UTC minute recovery and restart durability
# ---------------------------------------------------------------------------


def test_scopes_are_strictly_isolated_with_no_borrowing(app, client):
    for _ in range(5):
        assert _create(client).status_code == 201
    assert _create(client).status_code == 429

    # A different tenant on the same workload has its own full budget.
    other_tenant = _create(client, tenant_id="tenant-b")
    assert other_tenant.status_code == 201
    # A different workload in the same tenant is likewise independent.
    other_workload = _create(client, workload_id="workload-2")
    assert other_workload.status_code == 201

    with app.state.session_factory() as session:
        rows = session.scalars(select(ChallengeIssuanceCounter)).all()
        counts = {(r.tenant_id, r.workload_id): r.count for r in rows}
        assert counts == {
            (TENANT, WORKLOAD): 5,
            ("tenant-b", WORKLOAD): 1,
            (TENANT, "workload-2"): 1,
        }


def test_quota_recovers_at_next_utc_minute(app, client):
    issued = [_create(client).json()["challenge_id"] for _ in range(5)]
    assert _create(client).status_code == 429

    # Move the exhausted row into the previous minute: the current minute
    # has no issuance counter yet.
    with app.state.session_factory() as session:
        row = session.scalar(select(ChallengeIssuanceCounter))
        row.window_start = row.window_start - timedelta(minutes=1)
        session.commit()

    recovered = _create(client)
    assert recovered.status_code == 201
    with app.state.session_factory() as session:
        rows = {
            r.window_start: r.count
            for r in session.scalars(select(ChallengeIssuanceCounter)).all()
        }
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        assert rows[now] == 1
        assert rows[now - timedelta(minutes=1)] == 5
        # Five old plus one new challenge; the old ones are untouched.
        challenges = session.scalars(select(Challenge)).all()
        assert len(challenges) == 6
        assert recovered.json()["challenge_id"] not in issued


def test_previous_minute_counter_does_not_limit_new_minute(app, client):
    with app.state.session_factory() as session:
        session.add(
            ChallengeIssuanceCounter(
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

    assert _create(client).status_code == 201


def test_budget_persists_across_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart-challenge-limits.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    issued = [client1.post(
        "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    ) for _ in range(5)]
    assert [r.status_code for r in issued] == [201] * 5
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    # The sixth issuance in the same UTC minute is still refused.
    response = client2.post(
        "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert response.status_code == 429
    # The five challenges minted before the restart remain pending and are
    # consumable with their originally returned nonces.
    for created in issued:
        consumed = client2.post(
            f"/v1/challenges/{created.json()['challenge_id']}/consume",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "nonce": created.json()["nonce"],
            },
        )
        assert consumed.status_code == 200
    app2.state.engine.dispose()


def test_old_database_opens_and_issues_without_migration(tmp_path):
    # Build a database with a previous deployment, then remove the new
    # issuance table as an older deployment would never have created it.
    url = f"sqlite:///{tmp_path}/legacy.db"
    first = create_app(url)
    first.state.engine.dispose()

    import sqlite3

    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE challenge_issuance_counters")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "challenges" in tables
    assert "challenge_issuance_counters" not in tables

    # Startup on the old database is additive: the table is created, no
    # caller-side migration is required, and issuance works.
    app = create_app(url)
    response = TestClient(app).post(
        "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert response.status_code == 201
    app.state.engine.dispose()


# ---------------------------------------------------------------------------
# Independence from the one-time-grant budget
# ---------------------------------------------------------------------------


def test_grant_budget_actions_do_not_share_issuance_budget(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", "unit-test-secret")
    monkeypatch.setenv(
        "PROOF_RELEASE_MASTER_KEY",
        "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY",
    )
    # Five challenges exhaust the issuance budget...
    created = [_create(client).json() for _ in range(5)]
    assert _create(client).status_code == 429

    # ...but that does not touch the one-time-grant counter: consuming a
    # challenge (not a grant) needs no grant slot; assert at the table
    # level that the grant budget has no row at all.
    consumed = _consume(client, created[0]["challenge_id"], created[0]["nonce"])
    assert consumed.status_code == 200
    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []


def test_issuance_budget_does_not_consume_grant_budget(app, client):
    for _ in range(5):
        assert _create(client).status_code == 201
    with app.state.session_factory() as session:
        grant_rows = session.scalars(select(RateLimitCounter)).all()
        assert grant_rows == []


# ---------------------------------------------------------------------------
# Counter failure
# ---------------------------------------------------------------------------


def test_counter_unavailable_returns_500_and_creates_nothing(app, client):
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE challenge_issuance_counters"))

    response = _create(client)
    assert response.status_code == 500
    assert response.json() == {"detail": "rate limit unavailable"}

    with app.state.session_factory() as session:
        assert session.scalars(select(Challenge)).all() == []
        assert session.scalars(select(AuditEvent)).all() == []

    # Once the counter is available again, the request is the first
    # admitted slot of the minute.
    ChallengeIssuanceCounter.__table__.create(app.state.engine)
    recovered = _create(client)
    assert recovered.status_code == 201
    with app.state.session_factory() as session:
        row = session.scalar(select(ChallengeIssuanceCounter))
        assert row.count == 1
        challenges = session.scalars(select(Challenge)).all()
        assert len(challenges) == 1
        assert challenges[0].challenge_id == recovered.json()["challenge_id"]
