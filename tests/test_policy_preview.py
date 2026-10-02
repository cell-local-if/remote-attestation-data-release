"""Tests for POST /v1/evidence/{evidence_id}/policy-preview."""

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
    ReleaseGrant,
)
from proof_release.verifiers import X509_ATTESTED_NONCE_JSON

from x509_helpers import make_evidence, make_intermediate, make_leaf, make_root, pem

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    return create_app(f"sqlite:///{tmp_path}/preview.db")


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


def _preview(client, evidence_id, created, evidence, policy_ids, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
        "policy_ids": list(policy_ids),
    }
    body.update(overrides)
    return client.post(f"/v1/evidence/{evidence_id}/policy-preview", json=body)


def test_preview_allows_and_denies_in_request_order(client):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc", "tier": 2}
    )
    yes = _policy(client, {"claim": "measurement", "equals": "abc"}, name="yes")
    no = _policy(client, {"claim": "tier", "equals": 3}, name="no")

    response = _preview(
        client, evidence_id, created, evidence,
        [no["policy_id"], yes["policy_id"]],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["evidence_id"] == evidence_id
    assert [item["policy_id"] for item in data["results"]] == [
        no["policy_id"],
        yes["policy_id"],
    ]
    assert [item["status"] for item in data["results"]] == ["denied", "allowed"]
    assert [item["policy_version"] for item in data["results"]] == [1, 1]


def test_preview_response_is_compact_json_with_trailing_newline(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"m": "abc"})
    policy = _policy(client, {"claim": "m", "equals": "abc"})

    response = _preview(client, evidence_id, created, evidence, [policy["policy_id"]])

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.content.endswith(b"\n")
    assert b"\n" not in response.content[:-1]
    assert b": " not in response.content  # compact separators


def test_preview_evaluation_matches_decision_explanation_shape(client):
    claims = {"measurement": "abc", "tier": 2, "enabled": True}
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    rule = {
        "all": [
            {"claim": "measurement", "equals": "abc"},
            {"any": [
                {"claim": "tier", "equals": 3},
                {"claim": "enabled", "equals": True},
            ]},
            {"not": {"claim": "measurement", "equals": "zzz"}},
        ]
    }
    policy = _policy(client, rule)

    response = _preview(client, evidence_id, created, evidence, [policy["policy_id"]])

    assert response.status_code == 200
    (item,) = response.json()["results"]
    assert item["status"] == "allowed"
    evaluation = item["evaluation"]
    # Pre-order nodes: root all, leaf, any, leaf, leaf, not, leaf.
    assert [node["node_index"] for node in evaluation] == list(range(7))
    assert [node["node_type"] for node in evaluation] == [
        "all", "leaf", "any", "leaf", "leaf", "not", "leaf",
    ]
    assert [node["rule_path"] for node in evaluation] == [
        [], [0], [1], [1, 0], [1, 1], [2], [2, 0],
    ]
    assert [node["outcome"] for node in evaluation] == [
        True, True, True, False, True, True, False,
    ]
    # Only positions, structural types and booleans: no claim names,
    # comparison values or actual values anywhere in the evaluation.
    for node in evaluation:
        assert set(node) == {"node_index", "rule_path", "node_type", "outcome"}
    assert "measurement" not in json.dumps(item["evaluation"])
    assert "abc" not in json.dumps(item["evaluation"])

    # The formal decision taken afterwards persists the identical
    # explanation shape.
    decision = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert decision.status_code == 200
    stored = client.get(
        f"/v1/decisions/{decision.json()['decision_id']}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert stored.status_code == 200
    assert stored.json()["nodes"] == evaluation


def test_preview_denied_leaf_outcomes_stay_boolean_only(client):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    policy = _policy(client, {"claim": "measurement", "equals": "xyz"})

    response = _preview(client, evidence_id, created, evidence, [policy["policy_id"]])

    (item,) = response.json()["results"]
    assert item["status"] == "denied"
    assert item["evaluation"] == [
        {"node_index": 0, "rule_path": [], "node_type": "leaf", "outcome": False}
    ]


def test_preview_writes_no_decision_event_grant_or_audit(client, app):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})

    response = _preview(client, evidence_id, created, evidence, [policy["policy_id"]])
    assert response.status_code == 200

    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 0
        assert session.query(DecisionEvaluationNode).count() == 0
        assert session.query(ReleaseGrant).count() == 0
        assert session.query(AuditEvent).count() == 0
        # Only the reception and verification proof events exist; the
        # preview added none.
        events = session.query(ProofLifecycleEvent).all()
        assert {event.event_type for event in events} == {
            "proof-received",
            "proof-verified",
        }


def test_preview_is_repeatable_and_concurrently_deterministic(client, app):
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
    assert {r.text for r in responses} == {first.text}
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 0


def test_preview_after_formal_decision_creates_no_new_decision(client, app):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})
    decision = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert decision.status_code == 200

    response = _preview(client, evidence_id, created, evidence, [policy["policy_id"]])

    assert response.status_code == 200
    (item,) = response.json()["results"]
    assert item["status"] == "allowed"
    # The preview never mints or exposes a decision/grant identifier.
    assert "decision_id" not in response.text
    assert "grant" not in response.text
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 1
        assert session.query(ReleaseGrant).count() == 0


def test_preview_reads_the_immutable_version_rule_snapshot(client):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    v1 = _policy(client, {"claim": "measurement", "equals": "abc"})
    v2 = _policy(client, {"claim": "measurement", "equals": "zzz"})

    response = _preview(
        client, evidence_id, created, evidence,
        [v1["policy_id"], v2["policy_id"]],
    )

    assert response.status_code == 200
    items = response.json()["results"]
    assert [(i["policy_version"], i["status"]) for i in items] == [
        (1, "allowed"),
        (2, "denied"),
    ]


def test_preview_supports_up_to_sixteen_policies(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"m": "abc"})
    policies = [
        _policy(client, {"claim": "m", "equals": "abc"}, name=f"p{index}")
        for index in range(16)
    ]

    response = _preview(
        client, evidence_id, created, evidence,
        [policy["policy_id"] for policy in policies],
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 16
    assert all(item["status"] == "allowed" for item in response.json()["results"])


def test_preview_before_verification_returns_409(client):
    created, evidence, evidence_id = _receive_only(client)
    policy = _policy(client, {"claim": "x", "equals": 1})

    response = _preview(client, evidence_id, created, evidence, [policy["policy_id"]])

    assert response.status_code == 409


def test_preview_on_rejected_evidence_returns_409(client):
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

    response = _preview(client, evidence_id, created, evidence, [policy["policy_id"]])
    assert response.status_code == 409


def test_retired_policy_returns_409(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"m": "abc"})
    policy = _policy(client, {"claim": "m", "equals": "abc"})
    retired = client.post(
        f"/v1/policies/{policy['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200

    response = _preview(client, evidence_id, created, evidence, [policy["policy_id"]])

    assert response.status_code == 409


def test_unknown_evidence_returns_404(client):
    created = _challenge(client)
    policy = _policy(client, {"claim": "x", "equals": 1})

    response = _preview(
        client,
        "00000000-0000-0000-0000-000000000000",
        created,
        "{}",
        [policy["policy_id"]],
    )
    assert response.status_code == 404


def test_cross_scope_evidence_returns_404(client):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})

    assert (
        _preview(
            client, evidence_id, created, evidence, [policy["policy_id"]],
            tenant_id="tenant-b",
        ).status_code
        == 404
    )
    assert (
        _preview(
            client, evidence_id, created, evidence, [policy["policy_id"]],
            workload_id="workload-2",
        ).status_code
        == 404
    )


def test_unknown_and_cross_scope_policy_return_404(client):
    created, evidence, evidence_id = _receive_and_verify(client)

    assert (
        _preview(
            client, evidence_id, created, evidence,
            ["00000000-0000-0000-0000-000000000000"],
        ).status_code
        == 404
    )

    foreign = _policy(client, {"claim": "x", "equals": 1}, tenant_id="tenant-b")
    assert (
        _preview(
            client, evidence_id, created, evidence, [foreign["policy_id"]]
        ).status_code
        == 404
    )


def test_wrong_nonce_returns_422(client):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})
    wrong = ("B" if created["nonce"][0] != "B" else "C") + created["nonce"][1:]

    response = _preview(
        client, evidence_id, created, evidence, [policy["policy_id"]], nonce=wrong
    )
    assert response.status_code == 422


def test_digest_mismatch_returns_422(client):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})

    response = _preview(
        client, evidence_id, created, evidence + " ", [policy["policy_id"]]
    )
    assert response.status_code == 422


def test_unsupported_evidence_format_returns_422(client, app):
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
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        record.status = "verified"
        record.verified_at = datetime.now(tz=None)
        session.commit()

    policy = _policy(client, {"claim": "x", "equals": 1})
    response = _preview(client, evidence_id, created, opaque, [policy["policy_id"]])
    assert response.status_code == 422


def test_non_json_evidence_returns_422(client, app):
    # Verified attested-nonce-json evidence whose stored bytes are not
    # JSON (settled directly, as if an earlier deployment accepted it).
    created = _challenge(client)
    blob = "not-json-at-all"
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": blob,
        },
    )
    evidence_id = submitted.json()["evidence_id"]
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        record.status = "verified"
        record.verified_at = datetime.now(tz=None)
        session.commit()

    policy = _policy(client, {"claim": "x", "equals": 1})
    response = _preview(client, evidence_id, created, blob, [policy["policy_id"]])
    assert response.status_code == 422


def test_non_object_claims_returns_422(client, app):
    created = _challenge(client)
    blob = json.dumps({"nonce": created["nonce"], "claims": [1, 2, 3]})
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": blob,
        },
    )
    evidence_id = submitted.json()["evidence_id"]
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        record.status = "verified"
        record.verified_at = datetime.now(tz=None)
        session.commit()

    policy = _policy(client, {"claim": "x", "equals": 1})
    response = _preview(client, evidence_id, created, blob, [policy["policy_id"]])
    assert response.status_code == 422


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


@pytest.mark.parametrize(
    "overrides",
    [
        {"nonce": "not base64!!!"},
        {"nonce": "with=padding"},
        {"nonce": ""},
        {"evidence": ""},
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"workload_id": 2},
        {"policy_ids": "not-a-list"},
        {"policy_ids": []},
        {"policy_ids": [1]},
        {"policy_ids": [None]},
        {"policy_ids": ["not-a-uuid"]},
        # Uppercase and whitespace-padded ids are not canonical.
        {"policy_ids": ["A0000000-0000-0000-0000-000000000000"]},
        {"policy_ids": [" 00000000-0000-0000-0000-000000000000"]},
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


def test_preview_rejects_more_than_sixteen_policy_ids(client):
    created, evidence, evidence_id = _receive_and_verify(client)
    ids = [f"00000000-0000-0000-0000-{index:012d}" for index in range(1, 18)]

    response = _preview(client, evidence_id, created, evidence, ids)
    assert response.status_code == 422


def test_preview_rejects_duplicate_policy_ids(client):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})

    response = _preview(
        client, evidence_id, created, evidence,
        [policy["policy_id"], policy["policy_id"]],
    )
    assert response.status_code == 422


def test_preview_rejects_unknown_fields(client):
    created, evidence, evidence_id = _receive_and_verify(client)
    policy = _policy(client, {"claim": "x", "equals": 1})

    response = _preview(
        client, evidence_id, created, evidence, [policy["policy_id"]],
        policy_id=policy["policy_id"],
    )
    assert response.status_code == 422


def test_preview_never_returns_evidence_nonce_or_claims(client, app):
    secret_value = "super-secret-claim-value-123456"
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc", "secret": secret_value}
    )
    policy = _policy(client, {"claim": "secret", "equals": secret_value})

    response = _preview(client, evidence_id, created, evidence, [policy["policy_id"]])

    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "allowed"
    assert evidence not in response.text
    assert secret_value not in response.text
    assert created["nonce"] not in response.text
    # The rule's comparison value and claim name never appear either.
    assert "secret" not in response.text


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
    response = _preview(client, evidence_id, created, evidence, [policy["policy_id"]])
    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "allowed"


def test_preview_failure_leaves_no_partial_state(client, app):
    # A retired policy among the ids fails the whole request; nothing is
    # written and a later valid preview still works.
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    active = _policy(client, {"claim": "measurement", "equals": "abc"}, name="a")
    retired = _policy(client, {"claim": "measurement", "equals": "abc"}, name="r")
    assert client.post(
        f"/v1/policies/{retired['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).status_code == 200

    failed = _preview(
        client, evidence_id, created, evidence,
        [active["policy_id"], retired["policy_id"]],
    )
    assert failed.status_code == 409

    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 0
        assert session.query(ProofLifecycleEvent).count() == 2

    ok = _preview(client, evidence_id, created, evidence, [active["policy_id"]])
    assert ok.status_code == 200
    assert ok.json()["results"][0]["status"] == "allowed"
