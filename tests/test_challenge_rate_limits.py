"""Tests for the persistent per-tenant/workload issuance rate limit on
POST /v1/challenges.

Contract under test:

* at most five fully validated challenge creations per (tenant, workload)
  per UTC natural minute; the budget is attributed solely by the two
  scope fields, never by the minted challenge id or nonce;
* field/type/ttl failures (422) never consume budget and write no state;
* the sixth request in a minute gets a compact 429 carrying only a
  positive-integer ``retry_after_seconds`` followed by one newline, and
  leaves no challenge, no nonce and no counter change behind;
* the counter row and the challenge row commit in one transaction, the
  count survives restarts, the quota recovers at the next UTC minute,
  scopes are strictly isolated, and the challenge budget is independent
  of the shared grant consume/revoke/release budget;
* a counter read/write failure is a 500 (``rate limit unavailable``)
  with no partial state.
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
from proof_release.db import Challenge, ChallengeRateLimitCounter, RateLimitCounter

TENANT = "tenant-a"
WORKLOAD = "workload-1"
ZERO_UUID = "00000000-0000-0000-0000-000000000000"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/issue-limits.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, **overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    body.update(overrides)
    return client.post("/v1/challenges", json=body)


def _consume_grant_budget_slot(client):
    """Spend one shared grant-budget slot with a well-formed 404."""
    return client.post(
        f"/v1/release-grants/{ZERO_UUID}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": "A" * 43,
        },
    )


# ---------------------------------------------------------------------------
# Basic shape and the 5-per-minute ceiling
# ---------------------------------------------------------------------------


def test_first_valid_request_initializes_current_minute_with_one(app, client):
    assert _create(client).status_code == 201

    with app.state.session_factory() as session:
        rows = session.scalars(select(ChallengeRateLimitCounter)).all()
        assert len(rows) == 1
        row = rows[0]
        assert (row.tenant_id, row.workload_id) == (TENANT, WORKLOAD)
        assert row.count == 1
        now = datetime.now(timezone.utc)
        assert row.window_start == now.replace(second=0, microsecond=0)


def test_sixth_serial_request_returns_compact_429(app, client):
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

    with app.state.session_factory() as session:
        row = session.scalar(select(ChallengeRateLimitCounter))
        assert row.count == app_module.CHALLENGE_ISSUE_BUDGET_PER_MINUTE


def test_sixth_concurrent_request_returns_429_with_five_admitted(app):
    def call():
        return TestClient(app).post(
            "/v1/challenges",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: call(), range(6)))

    statuses = [r.status_code for r in responses]
    assert statuses.count(201) == 5
    assert statuses.count(429) == 1

    # Exactly the five admitted challenges exist, all pending, each with a
    # distinct id and nonce digest.
    with app.state.session_factory() as session:
        challenges = session.scalars(select(Challenge)).all()
        assert len(challenges) == 5
        assert {c.status for c in challenges} == {"pending"}
        assert len({c.challenge_id for c in challenges}) == 5
        assert len({c.nonce_digest for c in challenges}) == 5
        row = session.scalar(select(ChallengeRateLimitCounter))
        assert row.count == 5


def test_429_repeats_do_not_decrement_or_extend(app, client):
    for _ in range(5):
        assert _create(client).status_code == 201

    rejected = [_create(client).status_code for _ in range(4)]
    assert rejected == [429, 429, 429, 429]

    with app.state.session_factory() as session:
        row = session.scalar(select(ChallengeRateLimitCounter))
        assert row.count == app_module.CHALLENGE_ISSUE_BUDGET_PER_MINUTE
        # No challenge rows beyond the five admitted ones.
        assert len(session.scalars(select(Challenge)).all()) == 5


def test_retry_after_shrinks_within_the_minute(client, monkeypatch):
    # A fully controlled clock kept inside one UTC minute.
    clock = {"now": datetime(2026, 10, 3, 12, 0, 10, tzinfo=timezone.utc)}
    monkeypatch.setattr(app_module, "_utcnow", lambda: clock["now"])

    for _ in range(5):
        assert _create(client).status_code == 201

    first = _create(client)
    clock["now"] = clock["now"] + timedelta(seconds=20)
    second = _create(client)

    assert first.status_code == 429
    assert second.status_code == 429
    assert first.json()["retry_after_seconds"] == 50
    assert second.json()["retry_after_seconds"] == 30


# ---------------------------------------------------------------------------
# 422 precedes the counter: no budget consumed, no state written
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"workload_id": "   "},
        {"tenant_id": 1},
        {"tenant_id": True},
        {"workload_id": 2},
        {"ttl_seconds": 29},
        {"ttl_seconds": 901},
        {"ttl_seconds": "300"},
        {"ttl_seconds": 300.5},
        {"ttl_seconds": True},
    ],
)
def test_422_does_not_consume_budget(app, client, overrides):
    for _ in range(app_module.CHALLENGE_ISSUE_BUDGET_PER_MINUTE + 2):
        assert _create(client, **overrides).status_code == 422

    with app.state.session_factory() as session:
        assert session.scalars(select(ChallengeRateLimitCounter)).all() == []
        assert session.scalars(select(Challenge)).all() == []
    # A well-formed request is still the first admitted request.
    assert _create(client).status_code == 201


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"tenant_id": TENANT},
        {"workload_id": WORKLOAD},
        ["not", "an", "object"],
    ],
)
def test_missing_fields_and_non_object_body_are_not_counted(app, client, body):
    for _ in range(7):
        assert client.post("/v1/challenges", json=body).status_code == 422

    with app.state.session_factory() as session:
        assert session.scalars(select(ChallengeRateLimitCounter)).all() == []
    assert _create(client).status_code == 201


# ---------------------------------------------------------------------------
# Rejection leaves no challenge, no nonce, no audit side effects
# ---------------------------------------------------------------------------


def test_429_creates_no_challenge_and_leaks_no_nonce(app, client):
    nonces = set()
    for _ in range(5):
        created = _create(client)
        assert created.status_code == 201
        nonces.add(created.json()["nonce"])

    rejected = _create(client)
    assert rejected.status_code == 429
    body = rejected.text
    for nonce in nonces:
        assert nonce not in body
    assert "challenge_id" not in body
    assert "nonce" not in body

    with app.state.session_factory() as session:
        challenges = session.scalars(select(Challenge)).all()
        assert len(challenges) == 5
        # The admitted challenges are untouched: still pending, still
        # consumable after the rejection.
        assert all(c.status == "pending" for c in challenges)
        assert all(c.consumed_at is None for c in challenges)


def test_admitted_challenges_remain_consumable(client):
    created = [_create(client).json() for _ in range(5)]
    assert _create(client).status_code == 429

    for challenge in created:
        consumed = client.post(
            f"/v1/challenges/{challenge['challenge_id']}/consume",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "nonce": challenge["nonce"],
            },
        )
        assert consumed.status_code == 200
        assert consumed.json()["status"] == "consumed"
        # One-time semantics are unchanged: a replay is a 409.
        replay = client.post(
            f"/v1/challenges/{challenge['challenge_id']}/consume",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "nonce": challenge["nonce"],
            },
        )
        assert replay.status_code == 409


# ---------------------------------------------------------------------------
# Scope isolation, minute recovery and restart persistence
# ---------------------------------------------------------------------------


def test_scopes_are_strictly_isolated_with_no_borrowing(app, client):
    for _ in range(5):
        assert _create(client).status_code == 201
    assert _create(client).status_code == 429

    # A different tenant on the same workload has its own full budget.
    assert _create(client, tenant_id="tenant-b").status_code == 201
    # A different workload in the same tenant is likewise independent.
    assert _create(client, workload_id="workload-2").status_code == 201

    with app.state.session_factory() as session:
        rows = session.scalars(select(ChallengeRateLimitCounter)).all()
        counts = {(r.tenant_id, r.workload_id): r.count for r in rows}
        assert counts == {
            (TENANT, WORKLOAD): 5,
            ("tenant-b", WORKLOAD): 1,
            (TENANT, "workload-2"): 1,
        }


def test_quota_recovers_at_next_utc_minute(app, client):
    for _ in range(5):
        assert _create(client).status_code == 201
    assert _create(client).status_code == 429

    # Simulate crossing the UTC minute boundary by moving the exhausted
    # row into the previous minute: the current minute has no counter yet.
    with app.state.session_factory() as session:
        row = session.scalar(select(ChallengeRateLimitCounter))
        row.window_start = row.window_start - timedelta(minutes=1)
        session.commit()

    assert _create(client).status_code == 201
    with app.state.session_factory() as session:
        rows = {
            r.window_start: r.count
            for r in session.scalars(select(ChallengeRateLimitCounter)).all()
        }
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        assert rows[now] == 1
        assert rows[now - timedelta(minutes=1)] == 5


def test_previous_minute_counter_does_not_limit_new_minute(app, client):
    # Seed a saturated counter for a past minute directly.
    with app.state.session_factory() as session:
        session.add(
            ChallengeRateLimitCounter(
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
    url = f"sqlite:///{tmp_path}/restart-issue-limits.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    for _ in range(5):
        assert _create(client1).status_code == 201
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    # The sixth issuance in the same UTC minute is still refused.
    assert _create(client2).status_code == 429
    app2.state.engine.dispose()


def test_old_database_without_counter_table_starts(tmp_path):
    # A database written by a deployment that predates the limiter lacks
    # the counter table; opening it must add the table without touching
    # any existing challenge.
    url = f"sqlite:///{tmp_path}/upgrade.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    created = _create(client1).json()
    with app1.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE challenge_rate_limit_counters"))
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    # The pre-existing challenge is still there and consumable, and the
    # recreated counter starts a fresh minute window.
    with app2.state.session_factory() as session:
        challenge = session.get(Challenge, created["challenge_id"])
        assert challenge is not None
        assert challenge.status == "pending"
    assert _create(client2).status_code == 201
    app2.state.engine.dispose()


# ---------------------------------------------------------------------------
# Independence from the shared grant consume/revoke/release budget
# ---------------------------------------------------------------------------


def test_challenge_budget_does_not_touch_grant_budget(app, client):
    for _ in range(5):
        assert _create(client).status_code == 201
    assert _create(client).status_code == 429

    # The grant limiter's table was never written...
    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []
    # ...and a grant-path request still enters judgement (a plain 404 for
    # the unknown grant, never a 429).
    assert _consume_grant_budget_slot(client).status_code == 404


def test_grant_budget_does_not_touch_challenge_budget(app, client):
    # Exhaust the shared grant budget with five well-formed 404s.
    for _ in range(5):
        assert _consume_grant_budget_slot(client).status_code == 404
    assert _consume_grant_budget_slot(client).status_code == 429

    # Challenge issuance is unaffected and draws on its own counter only.
    for _ in range(5):
        assert _create(client).status_code == 201
    assert _create(client).status_code == 429

    with app.state.session_factory() as session:
        rows = session.scalars(select(ChallengeRateLimitCounter)).all()
        assert len(rows) == 1
        assert rows[0].count == 5


# ---------------------------------------------------------------------------
# Counter failure
# ---------------------------------------------------------------------------


def test_counter_unavailable_returns_500_and_writes_nothing(app, client):
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE challenge_rate_limit_counters"))

    response = _create(client)
    assert response.status_code == 500
    assert response.json() == {"detail": "rate limit unavailable"}

    # No half state: no challenge row and no counter row.
    with app.state.session_factory() as session:
        assert session.scalars(select(Challenge)).all() == []
    with app.state.engine.connect() as conn:
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
    assert ("challenge_rate_limit_counters",) not in tables

    ChallengeRateLimitCounter.__table__.create(app.state.engine)
    recovered = _create(client)
    assert recovered.status_code == 201
    with app.state.session_factory() as session:
        row = session.scalar(select(ChallengeRateLimitCounter))
        assert row.count == 1
