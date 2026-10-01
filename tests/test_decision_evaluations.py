"""Tests for GET /v1/decisions/{decision_id}/evaluation.

The evaluation endpoint returns the complete rule explanation that was
persisted in the decision's own transaction: every rule node in
depth-first pre-order with its consecutive ``node_index``, its
``rule_path`` of child positions from the root, its ``node_type``
(leaf/all/any/not) and its boolean ``outcome``, where the root outcome
equals the decision's status. The explanation records only node
positions, kinds and booleans — never claim names, claim values,
evidence, nonces, capabilities, payloads or keys.

The endpoint is strictly read-only. Every request-shape failure is a 422
returned before any state is read; an unknown or out-of-scope decision
is one indistinguishable 404; a decision settled before explanations
were recorded answers 409; a storage or integrity failure (including a
partial or inconsistent explanation) is a 500 with no half-built result.
Retries and concurrent creators of the same evidence/policy version
leave exactly one decision and exactly one explanation, and later policy
retirement, restarts or other lifecycle changes never rewrite it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from proof_release.app import create_app
from proof_release.db import (
    Decision,
    DecisionEvaluationNode,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
SECRET = "unit-test-secret"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"
UPPER_UUID = "ABCDEF12-3456-7890-ABCD-EF1234567890"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    application = create_app(f"sqlite:///{tmp_path}/decision_evaluations.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


# --- lifecycle builders ----------------------------------------------------


def _mac_for(nonce: str, claims: dict, *, tenant=TENANT, workload=WORKLOAD) -> str:
    key = hmac.new(
        SECRET.encode("utf-8"),
        f"{tenant}:{workload}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _submit_verified(client, *, claims=None, tenant=TENANT, workload=WORKLOAD):
    claims = {"m": "x"} if claims is None else claims
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": claims,
            "mac": _mac_for(
                created["nonce"], claims, tenant=tenant, workload=workload
            ),
        }
    )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201, submitted.text
    evidence_id = submitted.json()["evidence_id"]
    verified = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.status_code == 200, verified.text
    return created, evidence, evidence_id


def _policy(client, rule=None, *, name="r", tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/policies",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "name": name,
            "rule": rule if rule is not None else {"claim": "m", "equals": "x"},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _decide(client, created, evidence, evidence_id, policy_id, *,
            tenant=TENANT, workload=WORKLOAD):
    return client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy_id,
        },
    )


def _one_decision(client, *, claims=None, rule=None, name="r"):
    created, evidence, evidence_id = _submit_verified(client, claims=claims)
    policy = _policy(client, rule, name=name)
    decided = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    assert decided.status_code == 200, decided.text
    return created, evidence, evidence_id, policy, decided.json()


def _evaluation(client, decision_id, *, tenant=TENANT, workload=WORKLOAD,
                **extra):
    params = {"tenant_id": tenant, "workload_id": workload, **extra}
    return client.get(f"/v1/decisions/{decision_id}/evaluation", params=params)


# --- request validation (all 422, before any state is read) ----------------


def test_evaluation_requires_scope_parameters(client):
    assert client.get(f"/v1/decisions/{ZERO_UUID}/evaluation").status_code == 422
    assert (
        client.get(
            f"/v1/decisions/{ZERO_UUID}/evaluation",
            params={"workload_id": WORKLOAD},
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/v1/decisions/{ZERO_UUID}/evaluation",
            params={"tenant_id": TENANT},
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": "\t"},
        {"workload_id": ""},
        {"workload_id": "  "},
        {"bogus": "value"},
        {"cursor": "abc"},
        {"status": "allowed"},
        {"policy_id": ZERO_UUID},
        {"evaluation_version": "1"},
    ],
)
def test_evaluation_rejects_invalid_or_unknown_parameters(client, params):
    response = client.get(
        f"/v1/decisions/{ZERO_UUID}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD, **params},
    )
    assert response.status_code == 422, response.text


def test_evaluation_rejects_repeated_parameters(client):
    response = client.get(
        f"/v1/decisions/{ZERO_UUID}/evaluation"
        f"?tenant_id={TENANT}&tenant_id={TENANT}&workload_id={WORKLOAD}"
    )
    assert response.status_code == 422, response.text
    response = client.get(
        f"/v1/decisions/{ZERO_UUID}/evaluation"
        f"?tenant_id={TENANT}&workload_id={WORKLOAD}&workload_id={WORKLOAD}"
    )
    assert response.status_code == 422, response.text


@pytest.mark.parametrize(
    "decision_id",
    [
        UPPER_UUID,
        f" {ZERO_UUID}",
        f"{ZERO_UUID} ",
        "not-a-uuid",
        ZERO_UUID.replace("-", ""),
        "0" * 36,
    ],
)
def test_evaluation_rejects_non_canonical_identifier(client, decision_id):
    response = _evaluation(client, decision_id)
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "invalid decision identifier"


def test_evaluation_rejects_empty_path_segment(client):
    response = client.get(
        "/v1/decisions//evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "invalid decision identifier"


@pytest.mark.parametrize("body", [b"{}", b" ", b"\n", b"null", b"x"])
def test_evaluation_rejects_non_empty_body(client, body):
    response = client.request(
        "GET",
        f"/v1/decisions/{ZERO_UUID}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=body,
    )
    assert response.status_code == 422, response.text


# --- unknown and cross-scope decisions share one 404 -----------------------


def test_evaluation_unknown_decision_is_404(client):
    response = _evaluation(client, ZERO_UUID)
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "decision not found"


def test_evaluation_cross_scope_is_the_same_404(client):
    *_, decision = _one_decision(client)
    for tenant, workload in (
        (OTHER_TENANT, WORKLOAD),
        (TENANT, OTHER_WORKLOAD),
        (OTHER_TENANT, OTHER_WORKLOAD),
    ):
        response = _evaluation(
            client, decision["decision_id"], tenant=tenant, workload=workload
        )
        assert response.status_code == 404, response.text
        assert response.json()["detail"] == "decision not found"


# --- the persisted explanation ----------------------------------------------


def test_evaluation_leaf_allowed(client):
    *_, policy, decision = _one_decision(
        client, claims={"m": "x"}, rule={"claim": "m", "equals": "x"}
    )
    response = _evaluation(client, decision["decision_id"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {
        "decision_id": decision["decision_id"],
        "policy_version": policy["version"],
        "status": "allowed",
        "decided_at": decision["decided_at"],
        "evaluation_version": 1,
        "nodes": [
            {
                "node_index": 0,
                "rule_path": [],
                "node_type": "leaf",
                "outcome": True,
            }
        ],
    }


def test_evaluation_leaf_denied_root_outcome_matches_status(client):
    *_, decision = _one_decision(
        client, claims={"m": "other"}, rule={"claim": "m", "equals": "x"}
    )
    assert decision["status"] == "denied"
    response = _evaluation(client, decision["decision_id"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "denied"
    assert body["nodes"][0]["outcome"] is False


def test_evaluation_compound_tree_is_complete_and_ordered(client):
    rule = {
        "all": [
            {"claim": "a", "equals": 1},
            {
                "any": [
                    {"claim": "b", "in": ["x", "y"]},
                    {"not": {"path": ["meta", "tag"], "exists": True}},
                ]
            },
        ]
    }
    claims = {"a": 1, "b": "y", "meta": {}}
    *_, decision = _one_decision(client, claims=claims, rule=rule)
    assert decision["status"] == "allowed"
    response = _evaluation(client, decision["decision_id"])
    assert response.status_code == 200, response.text
    nodes = response.json()["nodes"]
    assert nodes == [
        {"node_index": 0, "rule_path": [], "node_type": "all", "outcome": True},
        {"node_index": 1, "rule_path": [0], "node_type": "leaf", "outcome": True},
        {"node_index": 2, "rule_path": [1], "node_type": "any", "outcome": True},
        {"node_index": 3, "rule_path": [1, 0], "node_type": "leaf", "outcome": True},
        {"node_index": 4, "rule_path": [1, 1], "node_type": "not", "outcome": True},
        {"node_index": 5, "rule_path": [1, 1, 0], "node_type": "leaf", "outcome": False},
    ]


def test_evaluation_compound_denied_propagates_to_root(client):
    rule = {
        "all": [
            {"claim": "a", "equals": 1},
            {"not": {"claim": "b", "gte": 10}},
        ]
    }
    # a matches but b >= 10, so the "not" fails and the root is denied.
    *_, decision = _one_decision(client, claims={"a": 1, "b": 12}, rule=rule)
    assert decision["status"] == "denied"
    response = _evaluation(client, decision["decision_id"])
    assert response.status_code == 200, response.text
    nodes = response.json()["nodes"]
    assert nodes == [
        {"node_index": 0, "rule_path": [], "node_type": "all", "outcome": False},
        {"node_index": 1, "rule_path": [0], "node_type": "leaf", "outcome": True},
        {"node_index": 2, "rule_path": [1], "node_type": "not", "outcome": False},
        {"node_index": 3, "rule_path": [1, 0], "node_type": "leaf", "outcome": True},
    ]


def test_evaluation_response_shape_is_fixed(client):
    *_, decision = _one_decision(client)
    response = _evaluation(client, decision["decision_id"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert list(body.keys()) == [
        "decision_id",
        "policy_version",
        "status",
        "decided_at",
        "evaluation_version",
        "nodes",
    ]
    assert isinstance(body["evaluation_version"], int)
    assert body["evaluation_version"] == 1
    assert isinstance(body["policy_version"], int)
    for node in body["nodes"]:
        assert set(node.keys()) == {
            "node_index",
            "rule_path",
            "node_type",
            "outcome",
        }
        assert isinstance(node["node_index"], int)
        assert isinstance(node["outcome"], bool)
        assert node["node_type"] in {"leaf", "all", "any", "not"}
        assert all(
            isinstance(position, int) and not isinstance(position, bool)
            for position in node["rule_path"]
        )


def test_evaluation_never_returns_claims_values_or_evidence(client, app):
    secret_value = "super-secret-claim-value-abcdef"
    secret_claim = "secret_claim_name_xyz"
    claims = {"m": "x", secret_claim: secret_value}
    rule = {
        "all": [
            {"claim": "m", "equals": "x"},
            {"claim": secret_claim, "equals": secret_value},
        ]
    }
    created, evidence, evidence_id = _submit_verified(client, claims=claims)
    policy = _policy(client, rule)
    decided = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    assert decided.status_code == 200, decided.text

    response = _evaluation(client, decided.json()["decision_id"])
    assert response.status_code == 200, response.text
    assert secret_value not in response.text
    assert secret_claim not in response.text
    assert evidence not in response.text
    assert created["nonce"] not in response.text

    # Nothing but positions, kinds and booleans is persisted either.
    with app.state.session_factory() as session:
        rows = session.query(DecisionEvaluationNode).all()
        assert len(rows) == 3
        for row in rows:
            columns = {
                c.name: getattr(row, c.name) for c in row.__table__.columns
            }
            stored = json.dumps(columns, default=str)
            assert secret_value not in stored
            assert secret_claim not in stored
            assert created["nonce"] not in stored


# --- idempotency and concurrency --------------------------------------------


def test_evaluation_retry_returns_the_same_explanation(client, app):
    created, evidence, evidence_id = _submit_verified(client, claims={"m": "x"})
    policy = _policy(client, {"claim": "m", "equals": "x"})
    first = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    second = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()

    decision_id = first.json()["decision_id"]
    first_read = _evaluation(client, decision_id)
    second_read = _evaluation(client, decision_id)
    assert first_read.status_code == second_read.status_code == 200
    assert first_read.text == second_read.text
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 1
        assert session.query(DecisionEvaluationNode).count() == 1


def test_concurrent_decisions_leave_one_decision_and_one_explanation(app):
    client = TestClient(app)
    created, evidence, evidence_id = _submit_verified(client, claims={"m": "x"})
    policy = _policy(
        client,
        {"all": [{"claim": "m", "equals": "x"}, {"claim": "m", "exists": True}]},
    )
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
    assert len({r.text for r in responses}) == 1
    decision_id = responses[0].json()["decision_id"]
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 1
        nodes = (
            session.query(DecisionEvaluationNode)
            .filter_by(decision_id=decision_id)
            .all()
        )
        # Exactly one explanation: the root plus the two leaves, once.
        assert len(nodes) == 3
        assert sorted(node.node_index for node in nodes) == [0, 1, 2]
    read = _evaluation(client, decision_id)
    assert read.status_code == 200, read.text
    assert len(read.json()["nodes"]) == 3


# --- pre-upgrade decisions and failure handling -----------------------------


def _insert_legacy_decision(app, decision_id=ZERO_UUID):
    """Insert a decision row with no explanation, as a pre-upgrade
    deployment would have left it."""
    with app.state.session_factory() as session:
        session.add(
            Decision(
                decision_id=decision_id,
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                evidence_id="11111111-1111-4111-8111-111111111111",
                policy_id="22222222-2222-4222-8222-222222222222",
                policy_version=1,
                status="allowed",
                decided_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                commit_seq=None,
            )
        )
        session.commit()


def test_evaluation_missing_for_pre_upgrade_decision_is_409(client, app):
    _insert_legacy_decision(app)
    response = _evaluation(client, ZERO_UUID)
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "evaluation not recorded"


def test_evaluation_partial_explanation_is_500(client, app):
    *_, decision = _one_decision(
        client,
        rule={"all": [{"claim": "m", "equals": "x"}, {"claim": "m", "exists": True}]},
    )
    with app.state.session_factory() as session:
        victim = session.scalar(
            select(DecisionEvaluationNode).where(
                DecisionEvaluationNode.node_index == 1
            )
        )
        session.delete(victim)
        session.commit()
    response = _evaluation(client, decision["decision_id"])
    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "decision evaluation unavailable"


def test_evaluation_inconsistent_root_outcome_is_500(client, app):
    *_, decision = _one_decision(client)
    with app.state.session_factory() as session:
        root = session.get(DecisionEvaluationNode, (decision["decision_id"], 0))
        root.outcome = False
        session.commit()
    response = _evaluation(client, decision["decision_id"])
    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "decision evaluation unavailable"


def test_evaluation_corrupt_node_kind_is_500(client, app):
    *_, decision = _one_decision(client)
    with app.state.session_factory() as session:
        root = session.get(DecisionEvaluationNode, (decision["decision_id"], 0))
        root.node_type = "bogus"
        session.commit()
    response = _evaluation(client, decision["decision_id"])
    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "decision evaluation unavailable"


def test_evaluation_storage_failure_is_500(client, app):
    *_, decision = _one_decision(client)
    DecisionEvaluationNode.__table__.drop(app.state.engine)
    try:
        response = _evaluation(client, decision["decision_id"])
        assert response.status_code == 500, response.text
        assert response.json()["detail"] == "decision evaluation unavailable"
    finally:
        DecisionEvaluationNode.__table__.create(app.state.engine)


# --- immutability across lifecycle changes and restarts ---------------------


def test_evaluation_survives_policy_retirement(client):
    created, evidence, evidence_id = _submit_verified(client, claims={"m": "x"})
    policy = _policy(client, {"claim": "m", "equals": "x"})
    decided = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    assert decided.status_code == 200, decided.text
    decision_id = decided.json()["decision_id"]
    before = _evaluation(client, decision_id)
    assert before.status_code == 200, before.text

    retired = client.post(
        f"/v1/policies/{policy['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200, retired.text

    after = _evaluation(client, decision_id)
    assert after.status_code == 200, after.text
    assert after.text == before.text


def test_evaluation_survives_restart(app, tmp_path, monkeypatch):
    client = TestClient(app)
    *_, decision = _one_decision(client)
    decision_id = decision["decision_id"]
    before = _evaluation(client, decision_id)
    assert before.status_code == 200, before.text
    app.state.engine.dispose()

    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    reopened = create_app(f"sqlite:///{tmp_path}/decision_evaluations.db")
    try:
        restarted = TestClient(reopened)
        after = _evaluation(restarted, decision_id)
        assert after.status_code == 200, after.text
        assert after.text == before.text
    finally:
        reopened.state.engine.dispose()


def test_evaluation_is_read_only(client, app):
    *_, decision = _one_decision(client)
    decision_id = decision["decision_id"]
    with app.state.session_factory() as session:
        decisions_before = session.query(Decision).count()
        nodes_before = session.query(DecisionEvaluationNode).count()
    response = _evaluation(client, decision_id)
    assert response.status_code == 200, response.text
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == decisions_before
        assert session.query(DecisionEvaluationNode).count() == nodes_before
