"""Tests for the persistent admission rate limit on evidence verification.

Contract under test for ``POST /v1/evidence/{evidence_id}/verify``:

* at most five requests per (tenant_id, workload_id) per UTC natural
  minute may enter the verifier; the budget is persistent, atomic and
  isolated per scope, and shares nothing with the challenge-issuance or
  one-time-grant budgets;
* only requests whose evidence is still ``received`` and whose identity,
  challenge binding, nonce, digest and format checks all passed consume a
  slot — every earlier judgement (422/404/401 and idempotent replays of a
  settled evidence) is unchanged and spends nothing;
* the sixth admitted candidate in a minute gets a 429 whose detail is
  exactly ``verification rate limit exceeded`` with a whole-seconds
  ``Retry-After`` header, and writes nothing: no counter, no settlement,
  no lifecycle event, no audit row;
* a verifier plugin failure or an unavailable X.509 revocation registry
  keeps the consumed slot (500, evidence stays received);
* a counter read/write failure is a 500 with the fixed detail
  ``verification rate limit unavailable`` and leaves the evidence
  received for a later retry.
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
    VerificationAdmissionCounter,
)
from proof_release.verifiers import (
    VerificationContext,
    VerificationResult,
    Verifier,
    VerifierRegistry,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
NONCE = "test-nonce"
EVIDENCE = "opaque-blob"


class AcceptVerifier(Verifier):
    format_name = "accept"

    def __init__(self):
        self.calls = 0

    def verify(self, context: VerificationContext) -> VerificationResult:
        self.calls += 1
        return VerificationResult(accepted=True)


class RaisingVerifier(Verifier):
    format_name = "raising"

    def __init__(self):
        self.calls = 0

    def verify(self, context: VerificationContext) -> VerificationResult:
        self.calls += 1
        raise RuntimeError("internal verifier fault")


def _registry(*verifiers):
    registry = VerifierRegistry()
    for verifier in verifiers:
        registry.register(verifier)
    return registry


@pytest.fixture()
def app(tmp_path):
    application = create_app(
        f"sqlite:///{tmp_path}/verification-limits.db",
        verifier_registry=_registry(AcceptVerifier()),
    )
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _seed_evidence(
    app,
    *,
    tenant_id=TENANT,
    workload_id=WORKLOAD,
    evidence_format="accept",
    evidence=EVIDENCE,
    nonce=NONCE,
):
    """Insert a consumed challenge plus a received evidence directly.

    Seeding bypasses the API so the independent challenge-issuance budget
    can never interfere with a verification-budget test.
    """
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
                issued_at=now,
                expires_at=now + timedelta(seconds=300),
                consumed_at=now,
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
        return session.scalars(select(VerificationAdmissionCounter)).all()


# ---------------------------------------------------------------------------
# Basic shape and the 5-per-minute ceiling
# ---------------------------------------------------------------------------


def test_first_admitted_verification_initializes_current_minute(app, client):
    evidence_id = _seed_evidence(app)
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
    with app.state.session_factory() as session:
        # The verification budget uses its own table; neither the grant
        # budget nor the challenge-issuance budget is touched.
        assert session.scalars(select(RateLimitCounter)).all() == []
        assert session.scalars(select(ChallengeIssuanceCounter)).all() == []


def test_five_verifications_admitted_sixth_rejected(app, client):
    evidence_ids = [_seed_evidence(app) for _ in range(6)]
    responses = [_verify(client, evidence_id) for evidence_id in evidence_ids[:5]]
    assert [r.status_code for r in responses] == [200] * 5

    sixth = _verify(client, evidence_ids[5])
    assert sixth.status_code == 429
    assert sixth.json() == {"detail": "verification rate limit exceeded"}
    retry_after = sixth.headers["retry-after"]
    assert retry_after.isdigit()
    assert 1 <= int(retry_after) <= 60

    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == app_module.VERIFICATION_BUDGET_PER_MINUTE


def test_retry_after_counts_down_to_next_utc_minute(app, monkeypatch):
    fixed = datetime(2026, 10, 3, 12, 30, 0, 500_000, tzinfo=timezone.utc)
    client = TestClient(app)
    evidence_ids = [_seed_evidence(app) for _ in range(7)]
    with monkeypatch.context() as patch:
        patch.setattr(app_module, "_utcnow", lambda: fixed)
        for evidence_id in evidence_ids[:5]:
            assert _verify(client, evidence_id).status_code == 200
        at_window_start = _verify(client, evidence_ids[5])
        patch.setattr(
            app_module,
            "_utcnow",
            lambda: fixed.replace(second=45, microsecond=900_000),
        )
        late_in_window = _verify(client, evidence_ids[6])

    assert at_window_start.status_code == 429
    assert late_in_window.status_code == 429
    assert at_window_start.headers["retry-after"] == "60"
    assert late_in_window.headers["retry-after"] == "15"


def test_concurrent_burst_admits_exactly_five(app):
    evidence_ids = [_seed_evidence(app) for _ in range(12)]

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

    with ThreadPoolExecutor(max_workers=12) as pool:
        statuses = list(pool.map(lambda eid: call(eid).status_code, evidence_ids))

    assert statuses.count(200) == 5
    assert statuses.count(429) == 7
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 5


def test_429_repeats_do_not_decrement_or_extend(app, client):
    evidence_ids = [_seed_evidence(app) for _ in range(9)]
    for evidence_id in evidence_ids[:5]:
        assert _verify(client, evidence_id).status_code == 200
    rejected = [_verify(client, evidence_id).status_code for evidence_id in evidence_ids[5:]]
    assert rejected == [429, 429, 429, 429]
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 5


def test_429_modifies_no_state(app, client):
    evidence_ids = [_seed_evidence(app) for _ in range(6)]
    for evidence_id in evidence_ids[:5]:
        assert _verify(client, evidence_id).status_code == 200
    with app.state.session_factory() as session:
        events_before = {
            row.event_id for row in session.scalars(select(ProofLifecycleEvent))
        }
        audits_before = {row.event_id for row in session.scalars(select(AuditEvent))}

    rejected = _verify(client, evidence_ids[5])
    assert rejected.status_code == 429
    assert "evidence_id" not in rejected.text
    assert "challenge_id" not in rejected.text

    with app.state.session_factory() as session:
        # The rejected evidence is untouched and still verifiable later.
        record = session.get(Evidence, evidence_ids[5])
        assert record.status == "received"
        assert record.verified_at is None
        assert record.verification_result is None
        # No lifecycle event and no audit row was appended.
        assert {
            row.event_id for row in session.scalars(select(ProofLifecycleEvent))
        } == events_before
        assert {
            row.event_id for row in session.scalars(select(AuditEvent))
        } == audits_before
        # The five settled evidences keep their conclusions.
        settled = {
            session.get(Evidence, evidence_id).status
            for evidence_id in evidence_ids[:5]
        }
        assert settled == {"verified"}


# ---------------------------------------------------------------------------
# Pre-verifier judgements never consume the budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"workload_id": ""},
        {"nonce": ""},
        {"nonce": "not base64!!!"},
        {"nonce": "with=padding"},
        {"evidence": ""},
        {"evidence": 42},
        {"evidence": None},
    ],
)
def test_invalid_fields_are_422_and_never_consume_budget(app, client, overrides):
    evidence_id = _seed_evidence(app)
    for _ in range(app_module.VERIFICATION_BUDGET_PER_MINUTE + 2):
        assert _verify(client, evidence_id, **overrides).status_code == 422
    assert _counter_rows(app) == []
    # The next fully valid request is still the first admitted.
    assert _verify(client, evidence_id).status_code == 200


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "nonce", "evidence"])
def test_missing_fields_are_422_and_not_counted(app, client, missing):
    evidence_id = _seed_evidence(app)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": NONCE,
        "evidence": EVIDENCE,
    }
    del body[missing]
    for _ in range(7):
        response = client.post(f"/v1/evidence/{evidence_id}/verify", json=body)
        assert response.status_code == 422
    assert _counter_rows(app) == []


def test_unknown_and_cross_scope_evidence_is_404_and_not_counted(app, client):
    evidence_id = _seed_evidence(app)
    unknown = str(uuid.uuid4())
    for _ in range(7):
        assert _verify(client, unknown).status_code == 404
        assert _verify(client, evidence_id, tenant_id="tenant-b").status_code == 404
        assert _verify(client, evidence_id, workload_id="workload-2").status_code == 404
    assert _counter_rows(app) == []
    assert _verify(client, evidence_id).status_code == 200


def test_nonce_mismatch_is_401_and_not_counted(app, client):
    evidence_id = _seed_evidence(app)
    wrong = ("B" if NONCE[0] != "B" else "C") + NONCE[1:]
    for _ in range(7):
        assert _verify(client, evidence_id, nonce=wrong).status_code == 401
    assert _counter_rows(app) == []
    assert _verify(client, evidence_id).status_code == 200


def test_digest_mismatch_is_422_and_not_counted(app, client):
    evidence_id = _seed_evidence(app)
    for _ in range(7):
        assert _verify(client, evidence_id, evidence=EVIDENCE + " ").status_code == 422
    assert _counter_rows(app) == []
    assert _verify(client, evidence_id).status_code == 200


def test_unregistered_format_is_422_and_not_counted(app, client):
    evidence_id = _seed_evidence(app, evidence_format="mystery")
    for _ in range(7):
        response = _verify(client, evidence_id)
        assert response.status_code == 422
        assert response.json()["detail"] == "unsupported evidence format"
    assert _counter_rows(app) == []


def test_settled_replays_return_stored_conclusion_without_budget(app, client):
    evidence_id = _seed_evidence(app)
    first = _verify(client, evidence_id)
    assert first.status_code == 200
    for _ in range(10):
        replay = _verify(client, evidence_id)
        assert replay.status_code == 200
        assert replay.json() == first.json()
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 1


# ---------------------------------------------------------------------------
# Failures after the reservation keep the consumed slot
# ---------------------------------------------------------------------------


def test_verifier_failure_keeps_the_slot(tmp_path):
    raising = RaisingVerifier()
    registry = _registry(raising)
    application = create_app(
        f"sqlite:///{tmp_path}/raising-limits.db", verifier_registry=registry
    )
    client = TestClient(application)
    evidence_ids = [_seed_evidence(application, evidence_format="raising") for _ in range(6)]

    for evidence_id in evidence_ids[:5]:
        response = _verify(client, evidence_id)
        assert response.status_code == 500
        assert EVIDENCE not in response.text

    # Five verifier entries were admitted and consumed five slots even
    # though every one of them failed; the sixth legal request is refused.
    assert raising.calls == 5
    rows = _counter_rows(application)
    assert len(rows) == 1
    assert rows[0].count == 5
    sixth = _verify(client, evidence_ids[5])
    assert sixth.status_code == 429
    assert sixth.json() == {"detail": "verification rate limit exceeded"}

    with application.state.session_factory() as session:
        for evidence_id in evidence_ids:
            record = session.get(Evidence, evidence_id)
            assert record.status == "received"
            assert record.verified_at is None
            assert record.verification_result is None
    application.state.engine.dispose()


def test_recovered_verifier_settles_after_failures(tmp_path):
    raising = RaisingVerifier()
    accept = AcceptVerifier()
    accept.format_name = "raising"
    registry = _registry(raising)
    application = create_app(
        f"sqlite:///{tmp_path}/recovery-limits.db", verifier_registry=registry
    )
    client = TestClient(application)
    evidence_ids = [_seed_evidence(application, evidence_format="raising") for _ in range(3)]

    for evidence_id in evidence_ids[:2]:
        assert _verify(client, evidence_id).status_code == 500

    # Once the verifier recovers, the remaining budget still admits.
    registry.register(accept)
    assert _verify(client, evidence_ids[2]).status_code == 200
    rows = _counter_rows(application)
    assert len(rows) == 1
    assert rows[0].count == 3
    application.state.engine.dispose()


# ---------------------------------------------------------------------------
# Scope isolation, UTC minute recovery and restart durability
# ---------------------------------------------------------------------------


def test_scopes_are_strictly_isolated_with_no_borrowing(app, client):
    for _ in range(5):
        assert _verify(client, _seed_evidence(app)).status_code == 200
    assert _verify(client, _seed_evidence(app)).status_code == 429

    other_tenant = _seed_evidence(app, tenant_id="tenant-b")
    assert _verify(client, other_tenant, tenant_id="tenant-b").status_code == 200
    other_workload = _seed_evidence(app, workload_id="workload-2")
    assert _verify(client, other_workload, workload_id="workload-2").status_code == 200

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
        assert _verify(client, _seed_evidence(app)).status_code == 200
    assert _verify(client, _seed_evidence(app)).status_code == 429

    # The next UTC minute has an independent budget.
    monkeypatch.setattr(
        app_module, "_utcnow", lambda: fixed + timedelta(minutes=1)
    )
    recovered = _seed_evidence(app)
    assert _verify(client, recovered).status_code == 200
    rows = {row.window_start: row.count for row in _counter_rows(app)}
    assert rows == {fixed: 5, fixed + timedelta(minutes=1): 1}


def test_previous_minute_counter_does_not_limit_new_minute(app, client):
    with app.state.session_factory() as session:
        session.add(
            VerificationAdmissionCounter(
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

    assert _verify(client, _seed_evidence(app)).status_code == 200


def test_budget_persists_across_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart-verification-limits.db"
    app1 = create_app(url, verifier_registry=_registry(AcceptVerifier()))
    client1 = TestClient(app1)
    for _ in range(5):
        assert _verify(client1, _seed_evidence(app1)).status_code == 200
    app1.state.engine.dispose()

    app2 = create_app(url, verifier_registry=_registry(AcceptVerifier()))
    client2 = TestClient(app2)
    # The sixth admission in the same UTC minute is still refused.
    response = _verify(client2, _seed_evidence(app2))
    assert response.status_code == 429
    assert response.json() == {"detail": "verification rate limit exceeded"}
    app2.state.engine.dispose()


def test_old_database_opens_and_verifies_without_migration(tmp_path):
    url = f"sqlite:///{tmp_path}/legacy-verification.db"
    first = create_app(url, verifier_registry=_registry(AcceptVerifier()))
    first.state.engine.dispose()

    import sqlite3

    with sqlite3.connect(tmp_path / "legacy-verification.db") as conn:
        conn.execute("DROP TABLE verification_admission_counters")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "evidence" in tables
    assert "verification_admission_counters" not in tables

    # Startup on the old database is additive: the table is created, no
    # caller-side migration is required, and verification works.
    app = create_app(url, verifier_registry=_registry(AcceptVerifier()))
    response = _verify(TestClient(app), _seed_evidence(app))
    assert response.status_code == 200
    app.state.engine.dispose()


# ---------------------------------------------------------------------------
# Independence from the challenge-issuance and one-time-grant budgets
# ---------------------------------------------------------------------------


def test_verification_budget_does_not_touch_other_budgets(app, client):
    for _ in range(5):
        assert _verify(client, _seed_evidence(app)).status_code == 200
    assert _verify(client, _seed_evidence(app)).status_code == 429

    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []
        assert session.scalars(select(ChallengeIssuanceCounter)).all() == []

    # Challenge issuance still has its own full budget.
    for _ in range(5):
        created = client.post(
            "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
        )
        assert created.status_code == 201
    with app.state.session_factory() as session:
        # ...and issuing challenges never draws from the verification
        # budget, which remains exactly the five consumed slots.
        row = session.scalar(select(VerificationAdmissionCounter))
        assert row.count == 5


# ---------------------------------------------------------------------------
# Counter failure
# ---------------------------------------------------------------------------


def test_counter_unavailable_returns_500_and_settles_nothing(app, client):
    evidence_id = _seed_evidence(app)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE verification_admission_counters"))

    response = _verify(client, evidence_id)
    assert response.status_code == 500
    assert response.json() == {"detail": "verification rate limit unavailable"}

    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verified_at is None
        assert record.verification_result is None
        assert session.scalars(select(ProofLifecycleEvent)).all() == []
        assert session.scalars(select(AuditEvent)).all() == []

    # Once the counter is available again, the request is the first
    # admitted slot of the minute.
    VerificationAdmissionCounter.__table__.create(app.state.engine)
    recovered = _verify(client, evidence_id)
    assert recovered.status_code == 200
    rows = _counter_rows(app)
    assert len(rows) == 1
    assert rows[0].count == 1
