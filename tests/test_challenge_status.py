"""Tests for the read-only challenge status query.

``GET /v1/challenges/{challenge_id}`` reports the derived lifecycle phase
of one scoped challenge without writing anything: no expiry migration, no
events, no rate-limit counters and no idempotency records.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select

from proof_release.app import CHALLENGE_ISSUANCE_BUDGET_PER_MINUTE, create_app
from proof_release.db import (
    Challenge,
    ChallengeIssuanceCounter,
    Evidence,
    VerificationAdmissionCounter,
)
from proof_release.verifiers import (
    VerificationResult,
    Verifier,
    VerifierRegistry,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"


class _AcceptVerifier(Verifier):
    format_name = "always-accept"

    def verify(self, context):
        return VerificationResult(accepted=True)


class _RejectVerifier(Verifier):
    format_name = "always-reject"

    def verify(self, context):
        return VerificationResult(accepted=False)


@pytest.fixture()
def app(tmp_path):
    registry = VerifierRegistry()
    registry.register(_AcceptVerifier())
    registry.register(_RejectVerifier())
    return create_app(f"sqlite:///{tmp_path}/test.db", verifier_registry=registry)


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, **overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    body.update(overrides)
    return client.post("/v1/challenges", json=body)


def _consume(client, created):
    return client.post(
        f"/v1/challenges/{created['challenge_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
        },
    )


def _submit(client, created, evidence_format="always-accept", evidence="cXVvdGU="):
    return client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": evidence_format,
            "evidence": evidence,
        },
    )


def _verify(client, evidence_id, created, evidence="cXVvdGU="):
    return client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )


def _status(client, challenge_id, **params):
    query = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    query.update(params)
    return client.get(f"/v1/challenges/{challenge_id}", params=query)


def test_pending_challenge(client):
    created = _create(client).json()

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["challenge_id"] == created["challenge_id"]
    assert data["status"] == "pending"
    assert data["issued_at"] == created["issued_at"]
    assert data["expires_at"] == created["expires_at"]
    assert data["changed_at"] == created["issued_at"]
    assert data["evidence_id"] is None
    assert data["evidence_format"] is None
    assert data["verification_result"] is None
    for field in ("issued_at", "expires_at", "changed_at"):
        parsed = datetime.fromisoformat(data[field])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)


def test_expired_challenge(client, app):
    created = _create(client, ttl_seconds=30).json()
    with app.state.session_factory() as session:
        challenge = session.get(Challenge, created["challenge_id"])
        challenge.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "expired"
    assert data["changed_at"] == data["expires_at"]
    assert data["evidence_id"] is None
    assert data["verification_result"] is None
    # The expiry is derived at read time: the stored row is untouched.
    with app.state.session_factory() as session:
        challenge = session.get(Challenge, created["challenge_id"])
        assert challenge.status == "pending"
        assert challenge.consumed_at is None


def test_consumed_challenge(client):
    created = _create(client).json()
    consumed = _consume(client, created).json()

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "consumed"
    assert data["changed_at"] == consumed["consumed_at"]
    assert data["evidence_id"] is None
    assert data["evidence_format"] is None
    assert data["verification_result"] is None


def test_evidence_received(client):
    created = _create(client).json()
    submitted = _submit(client, created).json()

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "evidence_received"
    assert data["changed_at"] == submitted["received_at"]
    assert data["evidence_id"] == submitted["evidence_id"]
    assert data["evidence_format"] == "always-accept"
    assert data["verification_result"] is None


def test_verified_evidence(client):
    created = _create(client).json()
    submitted = _submit(client, created).json()
    verified = _verify(client, submitted["evidence_id"], created).json()
    assert verified["status"] == "verified"

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "verified"
    assert data["changed_at"] == verified["verified_at"]
    assert data["evidence_id"] == submitted["evidence_id"]
    assert data["evidence_format"] == "always-accept"
    assert data["verification_result"] == "accepted"


def test_rejected_evidence(client):
    created = _create(client).json()
    submitted = _submit(client, created, evidence_format="always-reject").json()
    verified = _verify(client, submitted["evidence_id"], created).json()
    assert verified["status"] == "rejected"

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["changed_at"] == verified["verified_at"]
    assert data["verification_result"] == "rejected"


def test_response_never_carries_secret_material(client):
    created = _create(client).json()
    submitted = _submit(client, created).json()
    _verify(client, submitted["evidence_id"], created)

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "challenge_id",
        "issued_at",
        "expires_at",
        "status",
        "changed_at",
        "evidence_id",
        "evidence_format",
        "verification_result",
    }
    assert created["nonce"] not in response.text
    assert "cXVvdGU=" not in response.text


@pytest.mark.parametrize(
    "challenge_id",
    [
        "not-a-uuid",
        "00000000-0000-0000-0000-00000000000",  # too short
        "00000000-0000-0000-0000-0000000000000",  # too long
        "00000000-0000-0000-0000-00000000000g",
        "000000000000-0000-0000-0000-000000000000",
        "",
    ],
)
def test_non_uuid_path_identifier_is_422(client, challenge_id):
    response = _status(client, challenge_id or " ")

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid challenge query"


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": TENANT},  # workload_id missing
        {"workload_id": WORKLOAD},  # tenant_id missing
        {},  # both missing
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "   ", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": "  "},
        {"tenant_id": f" {TENANT}", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": f"{WORKLOAD} "},
    ],
)
def test_invalid_scope_parameters_are_422(client, params):
    created = _create(client).json()

    response = client.get(f"/v1/challenges/{created['challenge_id']}", params=params)

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid challenge query"


def test_unknown_challenge_is_404(client):
    response = _status(client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json()["detail"] == "challenge not found"


def test_cross_scope_is_indistinguishable_404(client):
    created = _create(client).json()

    other_tenant = _status(client, created["challenge_id"], tenant_id="tenant-b")
    assert other_tenant.status_code == 404
    assert other_tenant.json()["detail"] == "challenge not found"

    other_workload = _status(client, created["challenge_id"], workload_id="workload-2")
    assert other_workload.status_code == 404
    assert other_workload.json()["detail"] == "challenge not found"


def test_query_is_stable_and_writes_nothing(client, app):
    created = _create(client).json()
    with app.state.session_factory() as session:
        # The issuance itself reserved exactly one budget slot.
        counter = session.scalars(select(ChallengeIssuanceCounter)).one()
        count_before = counter.count

    first = _status(client, created["challenge_id"])
    second = _status(client, created["challenge_id"])

    assert first.status_code == 200
    assert first.json() == second.json()
    with app.state.session_factory() as session:
        # Neither read moved any counter or challenge state.
        counter = session.scalars(select(ChallengeIssuanceCounter)).one()
        assert counter.count == count_before
        assert session.scalars(select(VerificationAdmissionCounter)).all() == []
        challenge = session.get(Challenge, created["challenge_id"])
        assert challenge.status == "pending"
        assert challenge.consumed_at is None
        assert session.scalars(select(Evidence)).all() == []


def test_query_does_not_consume_issuance_budget(client, app):
    # Saturate the scope's issuance budget, then confirm the read-only
    # query still succeeds and the counter did not move.
    created = _create(client).json()
    with app.state.session_factory() as session:
        counter = session.scalars(select(ChallengeIssuanceCounter)).one()
        counter.count = CHALLENGE_ISSUANCE_BUDGET_PER_MINUTE
        session.commit()

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    assert response.json()["status"] == "pending"
    with app.state.session_factory() as session:
        counter = session.scalars(select(ChallengeIssuanceCounter)).one()
        assert counter.count == CHALLENGE_ISSUANCE_BUDGET_PER_MINUTE


def test_storage_failure_is_500(client, app):
    created = _create(client).json()
    engine = app.state.engine

    def fail_select(conn, cursor, statement, parameters, context, executemany):
        if "FROM challenges" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_select)
    try:
        response = _status(client, created["challenge_id"])
        assert response.status_code == 500
        assert response.json()["detail"] == "challenge status unavailable"
    finally:
        event.remove(engine, "before_cursor_execute", fail_select)

    # The failure left the challenge intact and observable.
    recovered = _status(client, created["challenge_id"])
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "pending"
