"""Tests for POST /v1/evidence/{evidence_id}/decisions."""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from proof_release.app import create_app
from proof_release.db import Decision
from proof_release.verifiers import (
    ATTESTED_NONCE_JSON,
    VerificationResult,
    Verifier,
    VerifierRegistry,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    return create_app(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


def _mac(nonce: str, claims: dict, tenant_id: str = TENANT, workload_id: str = WORKLOAD) -> str:
    key = hmac.new(
        SECRET.encode("utf-8"),
        f"{tenant_id}:{workload_id}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _attested_evidence(nonce: str, claims: dict) -> str:
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)}
    )


def _verified(client, claims, tenant_id=TENANT, workload_id=WORKLOAD):
    """Run the full challenge -> submit -> verify flow; return context."""
    created = client.post(
        "/v1/challenges", json={"tenant_id": tenant_id, "workload_id": workload_id}
    ).json()
    evidence = _attested_evidence(created["nonce"], claims)
    evidence_id = client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant_id,
            "workload_id": workload_id,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": ATTESTED_NONCE_JSON,
            "evidence": evidence,
        },
    ).json()["evidence_id"]
    verified = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": tenant_id,
            "workload_id": workload_id,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.status_code == 200
    assert verified.json()["status"] == "verified"
    return created, evidence, evidence_id


def _policy(client, rule, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "name": "release-policy",
        "rule": rule,
    }
    body.update(overrides)
    response = client.post("/v1/policies", json=body)
    assert response.status_code == 201
    return response.json()


def _decide(client, evidence_id, created, evidence, policy_id, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
        "policy_id": policy_id,
    }
    body.update(overrides)
    return client.post(f"/v1/evidence/{evidence_id}/decisions", json=body)


CLAIMS = {"measurement": "abc", "debug": False, "level": 3}


@pytest.fixture()
def decided(client):
    created, evidence, evidence_id = _verified(client, CLAIMS)
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})
    return created, evidence, evidence_id, policy


def test_satisfied_rule_grants(decided, client):
    created, evidence, evidence_id, policy = decided

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["decision_id"]
    assert data["evidence_id"] == evidence_id
    assert data["policy_id"] == policy["policy_id"]
    assert data["policy_version"] == 1
    assert data["status"] == "granted"
    decided_at = datetime.fromisoformat(data["decided_at"])
    assert decided_at.utcoffset() == timedelta(0)


def test_unsatisfied_rule_denies(client):
    created, evidence, evidence_id = _verified(client, CLAIMS)
    policy = _policy(client, {"claim": "measurement", "equals": "other"})

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 200
    assert response.json()["status"] == "denied"


def test_missing_claim_denies(client):
    created, evidence, evidence_id = _verified(client, CLAIMS)
    policy = _policy(client, {"claim": "absent", "equals": "abc"})

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 200
    assert response.json()["status"] == "denied"


@pytest.mark.parametrize(
    "rule, expected",
    [
        ({"claim": "debug", "equals": False}, "granted"),
        ({"claim": "debug", "equals": 0}, "denied"),  # bool is not a number
        ({"claim": "level", "equals": 3}, "granted"),
        ({"claim": "level", "equals": "3"}, "denied"),
        (
            {
                "all": [
                    {"claim": "measurement", "equals": "abc"},
                    {"claim": "level", "equals": 3},
                ]
            },
            "granted",
        ),
        (
            {
                "all": [
                    {"claim": "measurement", "equals": "abc"},
                    {"claim": "level", "equals": 4},
                ]
            },
            "denied",
        ),
        (
            {
                "any": [
                    {"claim": "measurement", "equals": "zzz"},
                    {"claim": "level", "equals": 3},
                ]
            },
            "granted",
        ),
        ({"not": {"claim": "debug", "equals": True}}, "granted"),
        ({"not": {"claim": "debug", "equals": False}}, "denied"),
    ],
)
def test_rule_evaluation(client, rule, expected):
    created, evidence, evidence_id = _verified(client, CLAIMS)
    policy = _policy(client, rule)

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 200
    assert response.json()["status"] == expected


def test_unknown_evidence_returns_404(decided, client):
    created, evidence, _, policy = decided

    response = _decide(
        client,
        "00000000-0000-0000-0000-000000000000",
        created,
        evidence,
        policy["policy_id"],
    )

    assert response.status_code == 404


def test_cross_scope_evidence_returns_404(decided, client):
    created, evidence, evidence_id, policy = decided

    assert (
        _decide(
            client, evidence_id, created, evidence, policy["policy_id"],
            tenant_id="tenant-b",
        ).status_code
        == 404
    )
    assert (
        _decide(
            client, evidence_id, created, evidence, policy["policy_id"],
            workload_id="workload-2",
        ).status_code
        == 404
    )


def test_unknown_policy_returns_404(decided, client):
    created, evidence, evidence_id, _ = decided

    response = _decide(
        client, evidence_id, created, evidence,
        "00000000-0000-0000-0000-000000000000",
    )

    assert response.status_code == 404


def test_cross_scope_policy_returns_404(decided, client):
    created, evidence, evidence_id, _ = decided
    foreign = _policy(
        client,
        {"claim": "measurement", "equals": "abc"},
        tenant_id="tenant-b",
    )

    response = _decide(
        client, evidence_id, created, evidence, foreign["policy_id"]
    )

    assert response.status_code == 404


def test_wrong_nonce_returns_401(decided, client):
    created, evidence, evidence_id, policy = decided
    wrong = "B" + created["nonce"][1:]

    response = _decide(
        client, evidence_id, created, evidence, policy["policy_id"], nonce=wrong
    )

    assert response.status_code == 401


def test_unverified_evidence_returns_409(client):
    created = client.post(
        "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    ).json()
    evidence = _attested_evidence(created["nonce"], CLAIMS)
    evidence_id = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": ATTESTED_NONCE_JSON,
            "evidence": evidence,
        },
    ).json()["evidence_id"]
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 409


def test_rejected_evidence_returns_409(client):
    created = client.post(
        "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    ).json()
    document = {"nonce": created["nonce"], "claims": CLAIMS, "mac": "0" * 64}
    evidence = json.dumps(document)
    evidence_id = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": ATTESTED_NONCE_JSON,
            "evidence": evidence,
        },
    ).json()["evidence_id"]
    verified = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.json()["status"] == "rejected"
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 409


def test_digest_mismatch_returns_422(decided, client):
    created, evidence, evidence_id, policy = decided

    response = _decide(
        client, evidence_id, created, evidence + " ", policy["policy_id"]
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"nonce": ""},
        {"nonce": "not base64!!!"},
        {"nonce": "with=padding"},
        {"evidence": ""},
        {"evidence": 42},
        {"policy_id": ""},
        {"policy_id": "  "},
        {"policy_id": None},
    ],
)
def test_invalid_fields_return_422(decided, client, overrides):
    created, evidence, evidence_id, policy = decided
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
        "policy_id": policy["policy_id"],
    }
    body.update(overrides)

    response = client.post(f"/v1/evidence/{evidence_id}/decisions", json=body)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "missing", ["tenant_id", "workload_id", "nonce", "evidence", "policy_id"]
)
def test_missing_fields_return_422(decided, client, missing):
    created, evidence, evidence_id, policy = decided
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
        "policy_id": policy["policy_id"],
    }
    del body[missing]

    response = client.post(f"/v1/evidence/{evidence_id}/decisions", json=body)

    assert response.status_code == 422


class AcceptAllVerifier(Verifier):
    """Accepts any evidence payload, for formats without JSON claims."""

    format_name = "opaque"

    def verify(self, context):
        return VerificationResult(accepted=True)


def _verified_opaque(application, client, evidence, evidence_format="opaque"):
    created = client.post(
        "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    ).json()
    evidence_id = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": evidence_format,
            "evidence": evidence,
        },
    ).json()["evidence_id"]
    verified = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.json()["status"] == "verified"
    return created, evidence_id


@pytest.fixture()
def opaque_app(tmp_path):
    registry = VerifierRegistry()
    registry.register(AcceptAllVerifier())
    application = create_app(
        f"sqlite:///{tmp_path}/opaque.db", verifier_registry=registry
    )
    return application, TestClient(application)


def test_non_builtin_format_returns_422(opaque_app):
    application, client = opaque_app
    created, evidence_id = _verified_opaque(application, client, "opaque-blob")
    policy = _policy(client, {"claim": "m", "equals": 1})

    response = _decide(client, evidence_id, created, "opaque-blob", policy["policy_id"])

    assert response.status_code == 422


@pytest.mark.parametrize(
    "evidence",
    [
        "not json",
        "[]",
        json.dumps({"nonce": "x", "claims": [1, 2]}),
    ],
)
def test_unparseable_builtin_evidence_returns_422(tmp_path, evidence):
    # An accept-all verifier registered under the built-in format name lets
    # malformed documents reach the decision endpoint in verified state.
    registry = VerifierRegistry()
    verifier = AcceptAllVerifier()
    verifier.format_name = ATTESTED_NONCE_JSON
    registry.register(verifier)
    application = create_app(
        f"sqlite:///{tmp_path}/builtin.db", verifier_registry=registry
    )
    client = TestClient(application)
    created, evidence_id = _verified_opaque(
        application, client, evidence, evidence_format=ATTESTED_NONCE_JSON
    )
    policy = _policy(client, {"claim": "m", "equals": 1})

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 422


def test_retry_returns_the_same_decision(decided, client, app):
    created, evidence, evidence_id, policy = decided

    first = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    second = _decide(client, evidence_id, created, evidence, policy["policy_id"])

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    with app.state.session_factory() as session:
        decisions = session.scalars(select(Decision)).all()
    assert len(decisions) == 1


def test_concurrent_decisions_settle_once(app, decided, client):
    created, evidence, evidence_id, policy = decided

    def decide():
        return TestClient(app).post(
            f"/v1/evidence/{evidence_id}/decisions",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "nonce": created["nonce"],
                "evidence": evidence,
                "policy_id": policy["policy_id"],
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: decide(), range(16)))

    assert all(r.status_code == 200 for r in responses)
    assert len({r.json()["decision_id"] for r in responses}) == 1
    assert len({r.json()["decided_at"] for r in responses}) == 1
    assert {r.json()["status"] for r in responses} == {"granted"}
    with app.state.session_factory() as session:
        decisions = session.scalars(select(Decision)).all()
    assert len(decisions) == 1


def test_each_policy_version_gets_its_own_decision(client, app):
    created, evidence, evidence_id = _verified(client, CLAIMS)
    policy_v1 = _policy(client, {"claim": "measurement", "equals": "abc"})
    policy_v2 = _policy(client, {"claim": "measurement", "equals": "zzz"})

    first = _decide(client, evidence_id, created, evidence, policy_v1["policy_id"])
    second = _decide(client, evidence_id, created, evidence, policy_v2["policy_id"])

    assert first.json()["status"] == "granted"
    assert first.json()["policy_version"] == 1
    assert second.json()["status"] == "denied"
    assert second.json()["policy_version"] == 2
    assert first.json()["decision_id"] != second.json()["decision_id"]
    with app.state.session_factory() as session:
        decisions = session.scalars(select(Decision)).all()
    assert len(decisions) == 2


def test_decision_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    created, evidence, evidence_id = _verified(client1, CLAIMS)
    policy = _policy(client1, {"claim": "measurement", "equals": "abc"})
    first = _decide(client1, evidence_id, created, evidence, policy["policy_id"])
    assert first.status_code == 200
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    repeat = _decide(client2, evidence_id, created, evidence, policy["policy_id"])
    assert repeat.status_code == 200
    assert repeat.json() == first.json()


def test_decision_never_persists_or_returns_claims_or_evidence(
    decided, client, app
):
    created, evidence, evidence_id, policy = decided

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 200
    assert evidence not in response.text
    assert "abc" not in response.text
    assert "claims" not in response.text
    with app.state.session_factory() as session:
        decision = session.scalars(select(Decision)).one()
        columns = {c.name: getattr(decision, c.name) for c in decision.__table__.columns}
    for name, value in columns.items():
        assert evidence not in str(value), f"evidence leaked into column {name}"
        assert "abc" not in str(value), f"claim value leaked into column {name}"
