"""Tests for the persistent verification rate limit on
``POST /v1/evidence/{evidence_id}/verify``.

Contract under test:

* only requests whose identity, challenge binding, nonce, digest and
  format all pass against evidence still in ``received`` count against a
  budget of five verifier admissions per UTC natural minute per
  (tenant_id, workload_id);
* 404/401/422 rejections and settled-evidence replays end before the
  verifier starts and never consume the budget;
* the sixth admitted request in a minute gets a 429 with the fixed detail
  ``verification rate limit exceeded`` and a ``Retry-After`` header with
  the whole seconds remaining in the current UTC minute; it changes no
  evidence, challenge, lifecycle-event or audit state;
* under concurrency exactly five requests win a reservation; the count is
  durable across restarts and scopes/minutes never borrow;
* the budget is independent of the challenge-issuance budget and the
  one-time-grant budget;
* a counter read/write failure is a 500 with detail
  ``verification rate limit unavailable`` and the evidence stays
  ``received``; a verifier or registry failure after admission keeps the
  consumed slot (no refund) and also leaves the evidence ``received``.
"""

from __future__ import annotations

import hashlib
import uuid
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
    Evidence,
    ProofLifecycleEvent,
    RateLimitCounter,
    VerificationRateLimitCounter,
)
from proof_release.verifiers import (
    VerificationResult,
    Verifier,
    VerifierRegistry,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
NONCE = "nonce-abc"
EVIDENCE = "evidence-blob"
FORMAT = "accepting"


class AcceptingVerifier(Verifier):
    """Verifier that accepts everything and counts its invocations."""

    format_name = FORMAT

    def __init__(self):
        self.calls = 0

    def verify(self, context) -> VerificationResult:
        self.calls += 1
        return VerificationResult(accepted=True)


class RaisingVerifier(Verifier):
    """Verifier that always fails, mapped to a 500 by the service."""

    format_name = FORMAT

    def __init__(self):
        self.calls = 0

    def verify(self, context) -> VerificationResult:
        self.calls += 1
        raise RuntimeError("internal verifier fault")


def _build_app(tmp_path, verifier=None, name="verify-limits.db"):
    verifier = verifier or AcceptingVerifier()
    registry = VerifierRegistry()
    registry.register(verifier)
    application = create_app(
        f"sqlite:///{tmp_path}/{name}", verifier_registry=registry
    )
    return application, verifier, registry


@pytest.fixture()
def app(tmp_path):
    application, verifier, _ = _build_app(tmp_path)
    application.state.test_verifier = verifier
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _insert_received(
    app,
    *,
    tenant_id=TENANT,
    workload_id=WORKLOAD,
    nonce=NONCE,
    evidence=EVIDENCE,
    evidence_format=FORMAT,
):
    """Insert a consumed challenge and a received evidence row directly."""
    now = datetime.now(timezone.utc)
    challenge_id = str(uuid.uuid4())
    evidence_id = str(uuid.uuid4())
    with app.state.session_factory() as session:
        session.add(
            Challenge(
                challenge_id=challenge_id,
                tenant_id=tenant_id,
                workload_id=workload_id,
                nonce_digest=hashlib.sha256(nonce.encode("ascii")).hexdigest(),
                status="consumed",
                issued_at=now - timedelta(seconds=1),
                expires_at=now + timedelta(minutes=5),
                consumed_at=now - timedelta(seconds=1),
            )
        )
        session.add(
            Evidence(
                evidence_id=evidence_id,
                challenge_id=challenge_id,
                tenant_id=tenant_id,
                workload_id=workload_id,
                evidence_format=evidence_format,
                status="received",
                received_at=now,
                evidence_sha256=hashlib.sha256(
                    evidence.encode("utf-8")
                ).hexdigest(),
                verified_at=None,
                verification_result=None,
            )
        )
        session.commit()
    return evidence_id


def _verify(client, evidence_id, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": NONCE,
        "evidence": EVIDENCE,
    }
    body.update(overrides)
    return client.post(f"/v1/evidence/{evidence_id}/verify", json=body)


def _counter_rows(app):
    with app.state.session_factory() as session:
        return session.scalars(select(VerificationRateLimitCounter)).all()


# ---------------------------------------------------------------------------
# Basic shape and the 5-per-minute ceiling
# ---------------------------------------------------------------------------


def test_first_admitted_verification_initializes_current_minute(app, client):
    evidence_id = _insert_received(app)

    response = _verify(client, evidence_id)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"

    rows = _counter_rows(app)
    assert len(rows) == 1
    row = rows[0]
    assert (row.tenant_id, row.workload_id) == (TENANT, WORKLOAD)
    assert row.count == 1
    now = datetime.now(timezone.utc)
    assert row.window_start == now.replace(second=0, microsecond=0)
    # The verification budget uses its own table; neither the grant budget
    # nor the challenge-issuance budget is touched by a verification.
    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []
        assert session.scalars(select(ChallengeIssuanceCounter)).all() == []


def test_five_verifications_admitted_sixth_is_429(app, client):
    evidence_ids = [_insert_received(app) for _ in range(6)]

    for evidence_id in evidence_ids[:5]:
        assert _verify(client, evidence_id).status_code == 200

    sixth = _verify(client, evidence_ids[5])
    assert sixth.status_code == 429
    assert sixth.json() == {"detail": "verification rate limit exceeded"}
    retry_after = sixth.headers["Retry-After"]
    assert retry_after.isdigit()
    assert 1 <= int(retry_after) <= 60

    # The rejected request never entered the verifier and settled nothing.
    assert app.state.test_verifier.calls == 5
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_ids[5])
        assert record.status == "received"
        assert record.verified_at is None
        assert record.verification_result is None


def test_retry_after_counts_down_to_next_utc_minute(app, monkeypatch):
    evidence_ids = [_insert_received(app) for _ in range(7)]
    fixed = datetime(2026, 10, 3, 12, 30, 0, 500_000, tzinfo=timezone.utc)
    with monkeypatch.context() as patch:
        patch.setattr(app_module, "_utcnow", lambda: fixed)
        for evidence_id in evidence_ids[:5]:
            assert _verify(TestClient(app), evidence_id).status_code == 200
        at_window_start = _verify(TestClient(app), evidence_ids[5])
        patch.setattr(
            app_module,
            "_utcnow",
            lambda: fixed.replace(second=45, microsecond=900_000),
        )
        late_in_window = _verify(TestClient(app), evidence_ids[6])

    assert at_window_start.status_code == 429
    assert late_in_window.status_code == 429
    assert at_window_start.headers["Retry-After"] == "60"
    assert late_in_window.headers["Retry-After"] == "15"
    assert (
        at_window_start.json()
        == late_in_window.json()
        == {"detail": "verification rate limit exceeded"}
    )


def test_concurrent_burst_admits_exactly_five(app):
    evidence_ids = [_insert_received(app) for _ in range(8)]

    def call(evidence_id):
        return TestClient(app).post(
            f"/v1/evidence/{evidence_id}/verify",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "nonce": NONCE,
                "evidence": EVIDENCE,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(call, evidence_ids))

    statuses = [r.status_code for r in responses]
    assert statuses.count(200) == 5
    assert statuses.count(429) == 3
    for response in responses:
        if response.status_code == 429:
            assert response.json() == {
                "detail": "verification rate limit exceeded"
            }
            assert response.headers["Retry-After"].isdigit()

    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 5
    assert app.state.test_verifier.calls == 5
    with app.state.session_factory() as session:
        settled = session.scalars(
            select(Evidence).where(Evidence.status == "verified")
        ).all()
        pending = session.scalars(
            select(Evidence).where(Evidence.status == "received")
        ).all()
        assert len(settled) == 5
        assert len(pending) == 3


def test_429_repeats_do_not_decrement_extend_or_write(app, client):
    evidence_ids = [_insert_received(app) for _ in range(6)]
    for evidence_id in evidence_ids[:5]:
        assert _verify(client, evidence_id).status_code == 200

    rejected = [_verify(client, evidence_ids[5]).status_code for _ in range(4)]
    assert rejected == [429, 429, 429, 429]

    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == app_module.VERIFICATION_BUDGET_PER_MINUTE
    with app.state.session_factory() as session:
        # Only the five settlements' lifecycle events exist; the 429s
        # appended none, and verification writes no audit rows at all.
        events = session.scalars(select(ProofLifecycleEvent)).all()
        assert len(events) == 5
        assert all(e.event_type == "proof-verified" for e in events)
        assert session.scalars(select(AuditEvent)).all() == []
        record = session.get(Evidence, evidence_ids[5])
        assert record.status == "received"


# ---------------------------------------------------------------------------
# Pre-validator outcomes never consume the budget
# ---------------------------------------------------------------------------


def test_pre_validator_failures_never_consume_budget(app, client):
    evidence_id = _insert_received(app)
    unsupported_id = _insert_received(app, evidence_format="mystery")
    unknown_id = "00000000-0000-0000-0000-000000000000"
    wrong_nonce = "nonce-abd"

    for _ in range(app_module.VERIFICATION_BUDGET_PER_MINUTE + 2):
        assert _verify(client, unknown_id).status_code == 404
        assert (
            _verify(client, evidence_id, tenant_id="tenant-b").status_code
            == 404
        )
        assert (
            _verify(client, evidence_id, workload_id="workload-2").status_code
            == 404
        )
        assert _verify(client, evidence_id, nonce=wrong_nonce).status_code == 401
        assert (
            _verify(client, evidence_id, evidence=EVIDENCE + " ").status_code
            == 422
        )
        assert _verify(client, unsupported_id).status_code == 422
        assert _verify(client, evidence_id, nonce="").status_code == 422
        assert _verify(client, evidence_id, nonce="with=padding").status_code == 422
        assert _verify(client, evidence_id, tenant_id="").status_code == 422
        assert _verify(client, evidence_id, evidence="").status_code == 422

    assert _counter_rows(app) == []
    assert app.state.test_verifier.calls == 0

    # The budget is untouched: five fresh verifications are all admitted.
    for _ in range(5):
        fresh_id = _insert_received(app)
        assert _verify(client, fresh_id).status_code == 200
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 5


def test_settled_replay_never_consumes_budget(app, client):
    evidence_id = _insert_received(app)

    first = _verify(client, evidence_id)
    assert first.status_code == 200
    assert first.json()["status"] == "verified"

    for _ in range(10):
        replay = _verify(client, evidence_id)
        assert replay.status_code == 200
        assert replay.json() == first.json()

    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 1
    assert app.state.test_verifier.calls == 1

    # Four more fresh verifications fill the minute exactly; a sixth
    # admission is refused.
    for _ in range(4):
        fresh_id = _insert_received(app)
        assert _verify(client, fresh_id).status_code == 200
    overflow_id = _insert_received(app)
    assert _verify(client, overflow_id).status_code == 429


# ---------------------------------------------------------------------------
# Scope isolation, UTC minute recovery and restart durability
# ---------------------------------------------------------------------------


def test_scopes_are_strictly_isolated_with_no_borrowing(app, client):
    for _ in range(5):
        assert _verify(client, _insert_received(app)).status_code == 200
    assert _verify(client, _insert_received(app)).status_code == 429

    # A different tenant on the same workload has its own full budget.
    other_tenant_id = _insert_received(app, tenant_id="tenant-b")
    assert (
        _verify(client, other_tenant_id, tenant_id="tenant-b").status_code
        == 200
    )
    # A different workload in the same tenant is likewise independent.
    other_workload_id = _insert_received(app, workload_id="workload-2")
    assert (
        _verify(client, other_workload_id, workload_id="workload-2").status_code
        == 200
    )

    counts = {
        (row.tenant_id, row.workload_id): row.count for row in _counter_rows(app)
    }
    assert counts == {
        (TENANT, WORKLOAD): 5,
        ("tenant-b", WORKLOAD): 1,
        (TENANT, "workload-2"): 1,
    }


def test_quota_recovers_at_next_utc_minute(app, client):
    for _ in range(5):
        assert _verify(client, _insert_received(app)).status_code == 200
    assert _verify(client, _insert_received(app)).status_code == 429

    # Move the exhausted row into the previous minute: the current minute
    # has no verification counter yet.
    with app.state.session_factory() as session:
        row = session.scalar(select(VerificationRateLimitCounter))
        row.window_start = row.window_start - timedelta(minutes=1)
        session.commit()

    recovered_id = _insert_received(app)
    assert _verify(client, recovered_id).status_code == 200
    rows = {
        row.window_start: row.count for row in _counter_rows(app)
    }
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    assert rows[now] == 1
    assert rows[now - timedelta(minutes=1)] == 5


def test_previous_minute_counter_does_not_limit_new_minute(app, client):
    with app.state.session_factory() as session:
        session.add(
            VerificationRateLimitCounter(
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

    assert _verify(client, _insert_received(app)).status_code == 200


def test_budget_persists_across_restart(tmp_path):
    url_name = "restart-verify-limits.db"
    app1, _, _ = _build_app(tmp_path, name=url_name)
    client1 = TestClient(app1)
    for _ in range(5):
        evidence_id = _insert_received(app1)
        assert _verify(client1, evidence_id).status_code == 200
    pending_id = _insert_received(app1)
    app1.state.engine.dispose()

    app2, verifier2, _ = _build_app(tmp_path, name=url_name)
    client2 = TestClient(app2)
    # The sixth admission in the same UTC minute is still refused.
    response = _verify(client2, pending_id)
    assert response.status_code == 429
    assert response.json() == {"detail": "verification rate limit exceeded"}
    assert verifier2.calls == 0
    with app2.state.session_factory() as session:
        record = session.get(Evidence, pending_id)
        assert record.status == "received"
    app2.state.engine.dispose()


def test_old_database_opens_and_verifies_without_migration(tmp_path):
    # Build a database with a previous deployment, then remove the new
    # verification table as an older deployment would never have created it.
    url_name = "legacy-verify-limits.db"
    first, _, _ = _build_app(tmp_path, name=url_name)
    first.state.engine.dispose()

    import sqlite3

    db_path = tmp_path / url_name
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE verification_rate_limit_counters")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "evidence" in tables
    assert "verification_rate_limit_counters" not in tables

    # Startup on the old database is additive: the table is created, no
    # caller-side migration is required, and verification works.
    app, _, _ = _build_app(tmp_path, name=url_name)
    evidence_id = _insert_received(app)
    response = _verify(TestClient(app), evidence_id)
    assert response.status_code == 200
    assert len(_counter_rows(app)) == 1
    app.state.engine.dispose()


# ---------------------------------------------------------------------------
# Independence from the challenge-issuance and one-time-grant budgets
# ---------------------------------------------------------------------------


def test_verification_budget_does_not_touch_other_budgets(app, client):
    for _ in range(5):
        assert _verify(client, _insert_received(app)).status_code == 200
    assert _verify(client, _insert_received(app)).status_code == 429

    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []
        assert session.scalars(select(ChallengeIssuanceCounter)).all() == []

    # Challenge issuance still has its own full budget in the same scope.
    issued = client.post(
        "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert issued.status_code == 201


def test_challenge_issuance_budget_does_not_limit_verification(app, client):
    # Exhaust the challenge-issuance budget in this scope.
    for _ in range(5):
        issued = client.post(
            "/v1/challenges",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
        assert issued.status_code == 201
    assert (
        client.post(
            "/v1/challenges",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).status_code
        == 429
    )

    # Verification has its own independent budget.
    for _ in range(5):
        assert _verify(client, _insert_received(app)).status_code == 200
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 5


# ---------------------------------------------------------------------------
# Counter failure and post-admission failure semantics
# ---------------------------------------------------------------------------


def test_counter_unavailable_returns_500_and_settles_nothing(app, client):
    evidence_id = _insert_received(app)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE verification_rate_limit_counters"))

    response = _verify(client, evidence_id)
    assert response.status_code == 500
    assert response.json() == {"detail": "verification rate limit unavailable"}
    assert app.state.test_verifier.calls == 0

    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verified_at is None
        assert session.scalars(select(ProofLifecycleEvent)).all() == []

    # Once the counter is available again, the same request is the first
    # admitted slot of the minute.
    VerificationRateLimitCounter.__table__.create(app.state.engine)
    recovered = _verify(client, evidence_id)
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "verified"
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 1


def test_verifier_failure_consumes_slot_without_refund(tmp_path):
    raising = RaisingVerifier()
    app, _, registry = _build_app(tmp_path, verifier=raising, name="raising.db")
    client = TestClient(app)
    evidence_id = _insert_received(app)

    response = _verify(client, evidence_id)
    assert response.status_code == 500

    # The slot was reserved before the verifier ran and is never refunded;
    # the evidence stays received.
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 1
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verified_at is None

    # After recovery the retry succeeds — spending a second slot.
    accepting = AcceptingVerifier()
    registry.register(accepting)
    retried = _verify(client, evidence_id)
    assert retried.status_code == 200
    assert retried.json()["status"] == "verified"
    assert accepting.calls == 1
    assert raising.calls == 1
    rows = _counter_rows(app)
    assert rows[0].count == 2
    app.state.engine.dispose()
