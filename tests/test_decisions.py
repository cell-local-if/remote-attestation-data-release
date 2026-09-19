"""Tests for POST /v1/evidence/{evidence_id}/decisions."""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Decision, Evidence, Policy
from proof_release.verifiers import X509_ATTESTED_NONCE_JSON

from x509_helpers import make_evidence, make_intermediate, make_leaf, make_root, pem

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    return create_app(f"sqlite:///{tmp_path}/decisions.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


def _mac(nonce: str, claims: dict) -> str:
    key = hmac.new(
        SECRET.encode("utf-8"),
        f"{TENANT}:{WORKLOAD}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _evidence(nonce: str, claims: dict | None = None) -> str:
    claims = claims if claims is not None else {}
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)}
    )


def _challenge(client):
    return client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()


def _receive_and_verify(client, claims=None):
    created = _challenge(client)
    evidence = _evidence(created["nonce"], claims)
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    verified = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.status_code == 200
    assert verified.json()["status"] == "verified"
    return created, evidence, evidence_id


def _receive_only(client, claims=None):
    created = _challenge(client)
    evidence = _evidence(created["nonce"], claims)
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201
    return created, evidence, submitted.json()["evidence_id"]


def _policy(client, rule, name="release", **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "name": name,
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


def test_allow_decision_when_claims_satisfy_policy(client):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc", "tier": 2}
    )
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["decision_id"]
    assert data["evidence_id"] == evidence_id
    assert data["policy_id"] == policy["policy_id"]
    assert data["policy_version"] == 1
    assert data["status"] == "allowed"
    decided_at = datetime.fromisoformat(data["decided_at"])
    assert decided_at.utcoffset() == timedelta(0)


def test_deny_decision_when_claims_do_not_satisfy_policy(client):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    policy = _policy(client, {"claim": "measurement", "equals": "xyz"})

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 200
    assert response.json()["status"] == "denied"


def test_compound_rules_evaluate(client):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc", "tier": 2, "enabled": True}
    )
    satisfied = _policy(
        client,
        {
            "all": [
                {"claim": "measurement", "equals": "abc"},
                {"any": [
                    {"claim": "tier", "equals": 3},
                    {"claim": "enabled", "equals": True},
                ]},
                {"not": {"claim": "measurement", "equals": "zzz"}},
            ]
        },
        name="compound-yes",
    )
    assert (
        _decide(client, evidence_id, created, evidence, satisfied["policy_id"])
        .json()["status"]
        == "allowed"
    )

    unsatisfied = _policy(
        client,
        {"all": [
            {"claim": "measurement", "equals": "abc"},
            {"not": {"claim": "enabled", "equals": True}},
        ]},
        name="compound-no",
    )
    assert (
        _decide(client, evidence_id, created, evidence, unsatisfied["policy_id"])
        .json()["status"]
        == "denied"
    )


def test_missing_claim_denies(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"a": 1})
    policy = _policy(client, {"claim": "b", "equals": None}, name="missing")
    # No claim "b" present; null is not the same as an absent key.
    assert (
        _decide(client, evidence_id, created, evidence, policy["policy_id"])
        .json()["status"]
        == "denied"
    )

    created2, evidence2, evidence_id2 = _receive_and_verify(client, {"b": None})
    assert (
        _decide(client, evidence_id2, created2, evidence2, policy["policy_id"])
        .json()["status"]
        == "allowed"
    )


def test_decision_before_verification_returns_409(client):
    created, evidence, evidence_id = _receive_only(client)
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 409


def test_decision_on_rejected_evidence_returns_409(client):
    # Receive evidence, settle it as rejected (bad MAC), then attempt a
    # decision with the same bytes.
    created = _challenge(client)
    evidence = json.dumps(
        {"nonce": created["nonce"], "claims": {}, "mac": "0" * 64}
    )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": evidence,
        },
    )
    evidence_id = submitted.json()["evidence_id"]
    rejected = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert rejected.json()["status"] == "rejected"
    policy = _policy(client, {"claim": "x", "equals": 1})

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 409


def test_unknown_evidence_returns_404(client):
    created = _challenge(client)
    policy = _policy(client, {"claim": "x", "equals": 1})
    response = _decide(
        client,
        "00000000-0000-0000-0000-000000000000",
        created,
        "{}",
        policy["policy_id"],
    )
    assert response.status_code == 404


def test_cross_scope_evidence_returns_404(client):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})

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


def test_unknown_and_cross_scope_policy_return_404(client):
    created, evidence, evidence_id = _receive_and_verify(client)

    assert (
        _decide(
            client,
            evidence_id,
            created,
            evidence,
            "00000000-0000-0000-0000-000000000000",
        ).status_code
        == 404
    )

    foreign = _policy(
        client, {"claim": "x", "equals": 1}, tenant_id="tenant-b"
    )
    assert (
        _decide(
            client, evidence_id, created, evidence, foreign["policy_id"]
        ).status_code
        == 404
    )


def test_wrong_nonce_returns_422(client):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})
    wrong = "B" + created["nonce"][1:]

    response = _decide(
        client, evidence_id, created, evidence, policy["policy_id"], nonce=wrong
    )
    assert response.status_code == 422


def test_digest_mismatch_returns_422(client):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})

    response = _decide(
        client, evidence_id, created, evidence + " ", policy["policy_id"]
    )
    assert response.status_code == 422


def test_non_json_builtin_evidence_format_returns_422(client):
    # Evidence received/verified through a plugin format the service cannot
    # extract built-in JSON claims from cannot be evaluated.
    created = _challenge(client)
    opaque = "opaque-verified-blob"
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "tpm-quote",
            "evidence": opaque,
        },
    )
    evidence_id = submitted.json()["evidence_id"]
    # Settle directly in storage as verified (simulating an accepting plugin)
    # while the format stays outside the built-in JSON family.
    with client.app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        record.status = "verified"
        record.verified_at = datetime.now(tz=None)
        session.commit()

    policy = _policy(client, {"claim": "x", "equals": 1})
    response = _decide(client, evidence_id, created, opaque, policy["policy_id"])
    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "nonce", "evidence", "policy_id"])
def test_decision_requires_all_fields(client, missing):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})
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


@pytest.mark.parametrize(
    "overrides",
    [
        {"nonce": "not base64!!!"},
        {"nonce": "with=padding"},
        {"evidence": ""},
        {"policy_id": ""},
        {"policy_id": "   "},
        {"tenant_id": 1},
    ],
)
def test_decision_rejects_invalid_fields(client, overrides):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})
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


def test_decision_is_idempotent_and_returns_same_result(client, app):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})

    first = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert first.status_code == 200
    second = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert second.status_code == 200
    assert second.json() == first.json()

    # A policy version change does not retroactively change the decision.
    new_version = _policy(client, {"claim": "measurement", "equals": "other"})
    other = _decide(
        client, evidence_id, created, evidence, new_version["policy_id"]
    )
    assert other.status_code == 200
    assert other.json()["policy_version"] == 2
    assert other.json()["status"] == "denied"
    # Original audit row untouched.
    repeat = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert repeat.json() == first.json()

    with app.state.session_factory() as session:
        rows = session.query(Decision).all()
        assert len(rows) == 2
        assert {(r.evidence_id, r.policy_id, r.status) for r in rows} == {
            (evidence_id, policy["policy_id"], "allowed"),
            (evidence_id, new_version["policy_id"], "denied"),
        }


def test_old_policy_versions_remain_usable_and_immutable(client):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    v1 = _policy(client, {"claim": "measurement", "equals": "abc"})
    v2 = _policy(client, {"claim": "measurement", "equals": "zzz"})

    assert (
        _decide(client, evidence_id, created, evidence, v1["policy_id"])
        .json()["status"]
        == "allowed"
    )
    assert (
        _decide(client, evidence_id, created, evidence, v2["policy_id"])
        .json()["status"]
        == "denied"
    )


def test_concurrent_decisions_return_one_identical_result(app):
    client = TestClient(app)
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})
    payload = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
        "policy_id": policy["policy_id"],
    }

    def decide():
        return TestClient(app).post(
            f"/v1/evidence/{evidence_id}/decisions", json=payload
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: decide(), range(16)))

    assert all(r.status_code == 200 for r in responses)
    bodies = {r.text for r in responses}
    assert len(bodies) == 1
    with app.state.session_factory() as session:
        rows = session.query(Decision).all()
        assert len(rows) == 1


def test_audit_row_never_stores_raw_evidence_or_claims(client, app):
    secret_value = "super-secret-claim-value-987654"
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc", "secret": secret_value}
    )
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 200
    assert evidence not in response.text
    assert secret_value not in response.text

    with app.state.session_factory() as session:
        decision = session.query(Decision).one()
        policy_row = session.get(Policy, policy["policy_id"])
        for target in (decision, policy_row):
            columns = {
                c.name: getattr(target, c.name) for c in target.__table__.columns
            }
            for name, value in columns.items():
                assert evidence not in str(value), f"evidence leaked in {name}"
                assert secret_value not in str(value), f"claim leaked in {name}"


def test_decision_evaluates_x509_builtin_json_claims(client):
    # The second built-in format carries the same claims envelope; a
    # verified X.509 evidence must drive decisions without re-running any
    # verifier.
    root_key, root_cert = make_root()
    intermediate_key, intermediate_cert = make_intermediate(root_cert, root_key)
    leaf_key, leaf_cert = make_leaf(intermediate_cert, intermediate_key)
    trust = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(root_cert),
        },
    )
    assert trust.status_code == 201

    created = _challenge(client)
    claims = {"measurement": "abc"}
    evidence = make_evidence(
        created["nonce"],
        leaf_key,
        [leaf_cert, intermediate_cert, root_cert],
        claims,
    )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": X509_ATTESTED_NONCE_JSON,
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    verified = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.status_code == 200
    assert verified.json()["status"] == "verified"

    policy = _policy(client, {"claim": "measurement", "equals": "abc"})
    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 200
    assert response.json()["status"] == "allowed"
