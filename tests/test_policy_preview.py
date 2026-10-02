"""Tests for POST /v1/evidence/{evidence_id}/policy-preview.

The preview endpoint is the read-only twin of the formal decision: it
applies the same scope, challenge-binding, nonce and evidence-digest
checks and evaluates the same immutable rule snapshots against the same
parsed claims, but it never invokes a verifier, never writes a decision,
proof event, evaluation node, grant or audit row, and never returns a
decision_id. Results come back in request order as compact JSON carrying
only identifiers, the fixed allowed/denied codes and node
positions/types/booleans — never evidence, nonces, claim names or
values, comparison targets or keys.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    Decision,
    DecisionEvaluationNode,
    Evidence,
    ProofLifecycleEvent,
)
from proof_release.verifiers import X509_ATTESTED_NONCE_JSON

from x509_helpers import make_evidence, make_intermediate, make_leaf, make_root, pem

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"
UPPER_UUID = "ABCDEF12-3456-7890-ABCD-EF1234567890"

SECRET_CLAIM = "super-secret-claim-value-987654"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    application = create_app(f"sqlite:///{tmp_path}/policy_preview.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


# --- lifecycle builders ----------------------------------------------------


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


def _receive(client, evidence, created, evidence_format="attested-nonce-json"):
    submitted = client.post(
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
    assert submitted.status_code == 201, submitted.text
    return submitted.json()["evidence_id"]


def _receive_and_verify(client, claims=None):
    created = _challenge(client)
    evidence = _evidence(created["nonce"], claims)
    evidence_id = _receive(client, evidence, created)
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
    return created, evidence, _receive(client, evidence, created)


def _settle_verified(client, evidence, evidence_format="attested-nonce-json"):
    """Receive raw bytes and settle them as verified directly in storage.

    Simulates an accepting verifier for content the built-in JSON claim
    reader cannot parse, without invoking any plugin.
    """
    created = _challenge(client)
    evidence_id = _receive(client, evidence, created, evidence_format)
    with client.app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        record.status = "verified"
        record.verified_at = datetime.now(tz=None)
        session.commit()
    return created, evidence_id


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


def _preview(client, evidence_id, created, evidence, policy_ids, **overrides):
    if isinstance(policy_ids, str):
        policy_ids = [policy_ids]
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
        "policy_ids": policy_ids,
    }
    body.update(overrides)
    return client.post(f"/v1/evidence/{evidence_id}/policy-preview", json=body)


# --- happy path ------------------------------------------------------------


def test_preview_allows_and_denies_in_request_order(client):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc", "tier": 2}
    )
    yes = _policy(client, {"claim": "measurement", "equals": "abc"}, name="yes")
    no = _policy(client, {"claim": "tier", "equals": 3}, name="no")

    response = _preview(
        client, evidence_id, created, evidence, [no["policy_id"], yes["policy_id"]]
    )

    assert response.status_code == 200
    data = response.json()
    assert data["evidence_id"] == evidence_id
    assert set(data.keys()) == {"evidence_id", "results"}
    assert [r["policy_id"] for r in data["results"]] == [
        no["policy_id"],
        yes["policy_id"],
    ]
    assert [r["status"] for r in data["results"]] == ["denied", "allowed"]
    for result, policy in zip(data["results"], (no, yes)):
        assert set(result.keys()) == {
            "policy_id",
            "policy_version",
            "status",
            "evaluation",
        }
        assert result["policy_version"] == policy["version"]
        evaluation = result["evaluation"]
        assert evaluation["evaluation_version"] == 1
        nodes = evaluation["nodes"]
        assert len(nodes) == 1
        (node,) = nodes
        assert set(node.keys()) == {"node_index", "rule_path", "node_type", "outcome"}
        assert node["node_index"] == 0
        assert node["rule_path"] == []
        assert node["node_type"] == "leaf"
        assert node["outcome"] is (result["status"] == "allowed")


def test_preview_explains_compound_rule_node_by_node(client):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc", "tier": 2, "enabled": True}
    )
    policy = _policy(
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
    )

    response = _preview(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 200
    (result,) = response.json()["results"]
    assert result["status"] == "allowed"
    assert result["evaluation"]["nodes"] == [
        {"node_index": 0, "rule_path": [], "node_type": "all", "outcome": True},
        {"node_index": 1, "rule_path": [0], "node_type": "leaf", "outcome": True},
        {"node_index": 2, "rule_path": [1], "node_type": "any", "outcome": True},
        {"node_index": 3, "rule_path": [1, 0], "node_type": "leaf", "outcome": False},
        {"node_index": 4, "rule_path": [1, 1], "node_type": "leaf", "outcome": True},
        {"node_index": 5, "rule_path": [2], "node_type": "not", "outcome": True},
        {"node_index": 6, "rule_path": [2, 0], "node_type": "leaf", "outcome": False},
    ]


def test_preview_response_is_compact_json(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"m": "x"})
    policy = _policy(client, {"claim": "m", "equals": "x"})

    response = _preview(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.text.endswith("\n")
    assert ", " not in response.text
    assert '": ' not in response.text
    assert json.loads(response.text) == response.json()


def test_preview_is_repeatable_and_concurrent_safe(client, app):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})
    payload = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
        "policy_ids": [policy["policy_id"]],
    }

    first = TestClient(app).post(
        f"/v1/evidence/{evidence_id}/policy-preview", json=payload
    )
    assert first.status_code == 200

    def preview():
        return TestClient(app).post(
            f"/v1/evidence/{evidence_id}/policy-preview", json=payload
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: preview(), range(16)))

    assert all(r.status_code == 200 for r in responses)
    bodies = {r.text for r in responses} | {first.text}
    assert len(bodies) == 1


def test_preview_writes_no_state(client, app):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})

    response = _preview(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 200
    assert "decision_id" not in response.text

    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 0
        assert session.query(DecisionEvaluationNode).count() == 0
        assert session.query(AuditEvent).count() == 0
        # Only the reception/verification lifecycle events exist; the
        # preview adds no decision event.
        assert {
            row.event_type for row in session.query(ProofLifecycleEvent).all()
        } <= {"proof-received", "proof-verified"}


def test_preview_after_formal_decision_creates_nothing(client, app):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})
    decided = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert decided.status_code == 200

    response = _preview(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 200
    (result,) = response.json()["results"]
    assert result["status"] == decided.json()["status"] == "allowed"
    assert "decision_id" not in response.text
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 1


def test_preview_uses_each_versions_immutable_rule_snapshot(client):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    v1 = _policy(client, {"claim": "measurement", "equals": "abc"})
    v2 = _policy(client, {"claim": "measurement", "equals": "zzz"})

    response = _preview(
        client, evidence_id, created, evidence, [v1["policy_id"], v2["policy_id"]]
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert [(r["policy_version"], r["status"]) for r in results] == [
        (1, "allowed"),
        (2, "denied"),
    ]


def test_preview_accepts_sixteen_policies(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"m": "x"})
    policies = [
        _policy(client, {"claim": "m", "equals": "x"}, name=f"p{index}")
        for index in range(16)
    ]

    response = _preview(
        client, evidence_id, created, evidence, [p["policy_id"] for p in policies]
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 16
    assert [r["policy_id"] for r in response.json()["results"]] == [
        p["policy_id"] for p in policies
    ]


def test_preview_evaluates_x509_builtin_json_claims(client):
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
    evidence = make_evidence(
        created["nonce"],
        leaf_key,
        [leaf_cert, intermediate_cert, root_cert],
        {"measurement": "abc"},
    )
    evidence_id = _receive(client, evidence, created, X509_ATTESTED_NONCE_JSON)
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

    policy = _policy(client, {"claim": "measurement", "equals": "abc"})
    response = _preview(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "allowed"


# --- not found -------------------------------------------------------------


def test_unknown_evidence_returns_404(client):
    created = _challenge(client)
    policy = _policy(client, {"claim": "x", "equals": 1})
    response = _preview(client, ZERO_UUID, created, "{}", policy["policy_id"])
    assert response.status_code == 404


def test_cross_scope_evidence_returns_404(client):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})

    assert (
        _preview(
            client, evidence_id, created, evidence, policy["policy_id"],
            tenant_id="tenant-b",
        ).status_code
        == 404
    )
    assert (
        _preview(
            client, evidence_id, created, evidence, policy["policy_id"],
            workload_id="workload-2",
        ).status_code
        == 404
    )


def test_unknown_and_cross_scope_policy_return_404(client):
    created, evidence, evidence_id = _receive_and_verify(client)

    assert (
        _preview(client, evidence_id, created, evidence, ZERO_UUID).status_code
        == 404
    )

    foreign = _policy(client, {"claim": "x", "equals": 1}, tenant_id="tenant-b")
    assert (
        _preview(client, evidence_id, created, evidence, foreign["policy_id"])
        .status_code
        == 404
    )


def test_any_unknown_policy_fails_the_whole_request(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"m": "x"})
    policy = _policy(client, {"claim": "m", "equals": "x"})

    response = _preview(
        client, evidence_id, created, evidence, [policy["policy_id"], ZERO_UUID]
    )

    assert response.status_code == 404


# --- client errors ---------------------------------------------------------


def test_wrong_nonce_returns_422(client):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})
    wrong = ("B" if created["nonce"][0] != "B" else "C") + created["nonce"][1:]

    response = _preview(
        client, evidence_id, created, evidence, policy["policy_id"], nonce=wrong
    )
    assert response.status_code == 422


def test_digest_mismatch_returns_422(client):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})

    response = _preview(
        client, evidence_id, created, evidence + " ", policy["policy_id"]
    )
    assert response.status_code == 422


def test_unsupported_evidence_format_returns_422(client):
    created, evidence_id = _settle_verified(
        client, "opaque-verified-blob", evidence_format="tpm-quote"
    )
    policy = _policy(client, {"claim": "x", "equals": 1})

    response = _preview(
        client, evidence_id, created, "opaque-verified-blob", policy["policy_id"]
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "evidence",
    [
        "not json at all",
        "[1, 2]",
        json.dumps({"nonce": "n", "claims": [1, 2], "mac": "0" * 64}),
    ],
)
def test_unparseable_json_or_claims_return_422(client, evidence):
    created, evidence_id = _settle_verified(client, evidence)
    policy = _policy(client, {"claim": "x", "equals": 1})

    response = _preview(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 422


# --- conflicts -------------------------------------------------------------


def test_unverified_evidence_returns_409(client):
    created, evidence, evidence_id = _receive_only(client)
    policy = _policy(client, {"claim": "x", "equals": 1})

    response = _preview(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 409


def test_rejected_evidence_returns_409(client):
    created = _challenge(client)
    evidence = json.dumps(
        {"nonce": created["nonce"], "claims": {}, "mac": "0" * 64}
    )
    evidence_id = _receive(client, evidence, created)
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

    response = _preview(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 409


def test_retired_policy_returns_409(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"m": "x"})
    policy = _policy(client, {"claim": "m", "equals": "x"})
    retired = client.post(
        f"/v1/policies/{policy['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200

    response = _preview(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 409


def test_policy_not_found_precedes_unverified_evidence(client):
    # Check order mirrors the formal decision: policy 404 before the
    # evidence-settlement 409.
    created, evidence, evidence_id = _receive_only(client)
    response = _preview(client, evidence_id, created, evidence, ZERO_UUID)
    assert response.status_code == 404


# --- request shape ---------------------------------------------------------


@pytest.mark.parametrize(
    "missing", ["tenant_id", "workload_id", "nonce", "evidence", "policy_ids"]
)
def test_preview_requires_all_fields(client, missing):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
        "policy_ids": [policy["policy_id"]],
    }
    del body[missing]
    response = client.post(f"/v1/evidence/{evidence_id}/policy-preview", json=body)
    assert response.status_code == 422


def test_unknown_field_is_rejected(client):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})

    response = _preview(
        client, evidence_id, created, evidence, policy["policy_id"], extra="nope"
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "policy_ids",
    [
        [],
        [ZERO_UUID] * 17,
        [ZERO_UUID, ZERO_UUID],
        [UPPER_UUID],
        ["not-a-uuid"],
        [""],
        ["  " + ZERO_UUID],
        [ZERO_UUID + " "],
        [1],
        [None],
    ],
)
def test_invalid_policy_ids_are_rejected(client, policy_ids):
    created, evidence, evidence_id = _receive_and_verify(client)
    response = _preview(client, evidence_id, created, evidence, policy_ids)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"policy_ids": "not-a-list"},
        {"policy_ids": ZERO_UUID},
        {"nonce": "not base64!!!"},
        {"nonce": "with=padding"},
        {"nonce": ""},
        {"evidence": ""},
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"tenant_id": 1},
        {"evidence": 1},
    ],
)
def test_preview_rejects_invalid_fields(client, overrides):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
        "policy_ids": [policy["policy_id"]],
    }
    body.update(overrides)
    response = client.post(f"/v1/evidence/{evidence_id}/policy-preview", json=body)
    assert response.status_code == 422


# --- confidentiality -------------------------------------------------------


def test_response_never_carries_evidence_nonce_or_claim_material(client):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc", "secret": SECRET_CLAIM}
    )
    policy = _policy(
        client, {"claim": "secret", "equals": SECRET_CLAIM}, name="secret-rule"
    )

    response = _preview(client, evidence_id, created, evidence, policy["policy_id"])

    assert response.status_code == 200
    assert evidence not in response.text
    assert created["nonce"] not in response.text
    assert SECRET_CLAIM not in response.text
    # The claim name and the comparison target appear nowhere either:
    # nodes carry positions, structural types and booleans only.
    assert "secret" not in response.text
    assert "claim" not in response.text
