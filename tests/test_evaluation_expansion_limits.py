"""Tests for the evaluation-time expansion resource limits.

Both POST /v1/policies/{policy_id}/evaluate and
POST /v1/evidence/{evidence_id}/decisions evaluate a policy version's
rule under the same bounds: a path leaf may not expand past 4096
candidates at any segment, the path leaves of the whole tree may not
yield more than 4096 candidates in one evaluation, and a
contains/contains_any/contains_all comparison never scans past the
4097th element of a located array. Exactly reaching a bound keeps the
ordinary verdict and complete explanation; crossing one is a 422 with
``detail == "policy evaluation expansion too large"`` — the trial
endpoint returns no verdict, timestamp or partial evaluation, and the
decision endpoint creates no decision, explanation node, audit event,
proof-lifecycle event or grant state.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    Decision,
    DecisionEvaluationNode,
    Policy,
    PolicyCommitCounter,
    PolicyIdempotencyRecord,
    ProofLifecycleEvent,
    ReleaseGrant,
    ReleaseGrantEvent,
)
from proof_release.policies import (
    EvaluationTooLarge,
    evaluate_rule,
    explain_rule,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"

LIMIT = 4096
WILDCARD = {"wildcard": True}

EXPANSION_DETAIL = "policy evaluation expansion too large"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    application = create_app(f"sqlite:///{tmp_path}/expansion_limits.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create_policy(client, rule, *, name="release"):
    response = client.post(
        "/v1/policies",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "name": name,
            "rule": rule,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _evaluate(client, policy_id, claims):
    return client.post(
        f"/v1/policies/{policy_id}/evaluate",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "claims": claims,
        },
    )


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


def _receive_and_verify(client, claims):
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": claims,
            "mac": _mac(created["nonce"], claims),
        }
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
    assert submitted.status_code == 201, submitted.text
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
    assert verified.status_code == 200, verified.text
    return created, evidence, evidence_id


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


def _wildcard_rule(equals=0):
    return {"path": ["items", WILDCARD, "id"], "equals": equals}


def _items(count, *, id_value=0):
    return {"items": [{"id": id_value}] * count}


# --- pure evaluator bounds ----------------------------------------------------


def test_evaluator_per_leaf_limit_exactly_reached():
    rule = _wildcard_rule(equals=LIMIT - 1)
    claims = {"items": [{"id": i} for i in range(LIMIT)]}
    assert evaluate_rule(rule, claims) is True
    nodes = explain_rule(rule, claims)
    assert len(nodes) == 1 and nodes[0]["outcome"] is True


def test_evaluator_per_leaf_limit_crossed():
    rule = _wildcard_rule()
    claims = _items(LIMIT + 1)
    with pytest.raises(EvaluationTooLarge):
        evaluate_rule(rule, claims)
    with pytest.raises(EvaluationTooLarge):
        explain_rule(rule, claims)


def test_evaluator_intermediate_segment_limit_crossed():
    # 100 x 100 nested arrays: the second wildcard expands to 10000
    # candidates even though the leaf's final count would be smaller.
    rule = {"path": ["a", WILDCARD, "b", WILDCARD, "c"], "exists": True}
    claims = {"a": [{"b": [{"c": 1}] * 100} for _ in range(100)]}
    with pytest.raises(EvaluationTooLarge):
        evaluate_rule(rule, claims)


def test_evaluator_tree_total_limit():
    leaf_match = _wildcard_rule(equals=0)
    leaf_miss = _wildcard_rule(equals=999999)
    rule = {"all": [leaf_match, leaf_miss]}
    # 2048 + 2048 == 4096 exactly: ordinary semantics.
    at_limit = _items(LIMIT // 2)
    assert evaluate_rule(rule, at_limit) is False
    assert [n["outcome"] for n in explain_rule(rule, at_limit)] == [
        False,
        True,
        False,
    ]
    # 3000 + 3000 > 4096: too large.
    over = _items(3000)
    with pytest.raises(EvaluationTooLarge):
        evaluate_rule(rule, over)
    with pytest.raises(EvaluationTooLarge):
        explain_rule(rule, over)


def test_evaluator_contains_scan_limit():
    big = [0] * (LIMIT + 1)
    for rule in (
        {"claim": "arr", "contains": 1},
        {"claim": "arr", "contains_any": [1]},
        {"claim": "arr", "contains_all": [1]},
    ):
        with pytest.raises(EvaluationTooLarge):
            evaluate_rule(rule, {"arr": big})
    # Exactly 4096 scanned elements keeps the ordinary result.
    assert evaluate_rule({"claim": "arr", "contains": 1}, {"arr": [0] * LIMIT}) is False
    # A match inside the first 4096 elements stops the scan before the
    # 4097th, so the oversized tail is never touched.
    assert evaluate_rule({"claim": "arr", "contains": 1}, {"arr": [1] + big}) is True


# --- trial evaluation endpoint -------------------------------------------------


def test_evaluate_at_limit_returns_full_result(client):
    created = _create_policy(client, _wildcard_rule(equals=LIMIT - 1))
    claims = {"items": [{"id": i} for i in range(LIMIT)]}
    response = _evaluate(client, created["policy_id"], claims)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["allowed"] is True
    assert data["evaluation"][0]["outcome"] is True
    assert data["checked_at"]


def test_evaluate_per_leaf_expansion_too_large(client):
    created = _create_policy(client, _wildcard_rule())
    response = _evaluate(client, created["policy_id"], _items(LIMIT + 1))
    assert response.status_code == 422
    assert response.json() == {"detail": EXPANSION_DETAIL}
    assert b"allowed" not in response.content
    assert b"checked_at" not in response.content


def test_evaluate_intermediate_segment_too_large(client):
    rule = {"path": ["a", WILDCARD, "b", WILDCARD, "c"], "exists": True}
    created = _create_policy(client, rule)
    claims = {"a": [{"b": [{"c": 1}] * 100} for _ in range(100)]}
    response = _evaluate(client, created["policy_id"], claims)
    assert response.status_code == 422
    assert response.json() == {"detail": EXPANSION_DETAIL}


def test_evaluate_tree_total_too_large(client):
    rule = {"all": [_wildcard_rule(equals=0), _wildcard_rule(equals=999999)]}
    created = _create_policy(client, rule)
    response = _evaluate(client, created["policy_id"], _items(3000))
    assert response.status_code == 422
    assert response.json() == {"detail": EXPANSION_DETAIL}


def test_evaluate_tree_total_at_limit_succeeds(client):
    rule = {"all": [_wildcard_rule(equals=0), _wildcard_rule(equals=999999)]}
    created = _create_policy(client, rule)
    response = _evaluate(client, created["policy_id"], _items(LIMIT // 2))
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["allowed"] is False
    assert [n["outcome"] for n in data["evaluation"]] == [False, True, False]


def test_evaluate_contains_scan_too_large(client):
    created = _create_policy(client, {"claim": "arr", "contains": 1})
    response = _evaluate(client, created["policy_id"], {"arr": [0] * (LIMIT + 1)})
    assert response.status_code == 422
    assert response.json() == {"detail": EXPANSION_DETAIL}


def test_evaluate_contains_at_limit_succeeds(client):
    created = _create_policy(client, {"claim": "arr", "contains": 1})
    response = _evaluate(client, created["policy_id"], {"arr": [0] * LIMIT})
    assert response.status_code == 200, response.text
    assert response.json()["allowed"] is False


def test_evaluate_no_wildcard_rule_unaffected(client):
    created = _create_policy(client, {"claim": "measurement", "equals": "abc"})
    response = _evaluate(
        client, created["policy_id"], {"measurement": "abc"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["allowed"] is True


def test_evaluate_expansion_failure_writes_no_state(app, client):
    created = _create_policy(client, _wildcard_rule())
    with app.state.session_factory() as session:
        before = {
            model: session.query(model).count()
            for model in (
                Policy,
                PolicyCommitCounter,
                PolicyIdempotencyRecord,
                Decision,
                DecisionEvaluationNode,
                ProofLifecycleEvent,
                AuditEvent,
            )
        }
    response = _evaluate(client, created["policy_id"], _items(LIMIT + 1))
    assert response.status_code == 422
    with app.state.session_factory() as session:
        for model, count in before.items():
            assert session.query(model).count() == count, model.__name__


# --- formal decision endpoint ---------------------------------------------------


def test_decision_at_limit_succeeds(client):
    claims = {"items": [{"id": i} for i in range(LIMIT)]}
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(client, _wildcard_rule(equals=LIMIT - 1))

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "allowed"


def test_decision_per_leaf_expansion_too_large_creates_nothing(app, client):
    created, evidence, evidence_id = _receive_and_verify(client, _items(LIMIT + 1))
    policy = _create_policy(client, _wildcard_rule())

    models = (
        Decision,
        DecisionEvaluationNode,
        ProofLifecycleEvent,
        AuditEvent,
        ReleaseGrant,
        ReleaseGrantEvent,
    )
    with app.state.session_factory() as session:
        before = {model: session.query(model).count() for model in models}

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 422
    assert response.json() == {"detail": EXPANSION_DETAIL}

    with app.state.session_factory() as session:
        for model, count in before.items():
            assert session.query(model).count() == count, model.__name__


def test_decision_tree_total_too_large(client):
    created, evidence, evidence_id = _receive_and_verify(client, _items(3000))
    rule = {"all": [_wildcard_rule(equals=0), _wildcard_rule(equals=999999)]}
    policy = _create_policy(client, rule)

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 422
    assert response.json() == {"detail": EXPANSION_DETAIL}


def test_decision_contains_scan_too_large(client):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"arr": [0] * (LIMIT + 1)}
    )
    policy = _create_policy(client, {"claim": "arr", "contains": 1})

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 422
    assert response.json() == {"detail": EXPANSION_DETAIL}


def test_decision_expansion_check_follows_nonce_and_digest_checks(client):
    created, evidence, evidence_id = _receive_and_verify(client, _items(LIMIT + 1))
    policy = _create_policy(client, _wildcard_rule())

    # A wrong nonce keeps its own 422 — the expansion limit is only
    # reached after the nonce, digest and replay checks.
    wrong_nonce = _decide(
        client, evidence_id, created, evidence, policy["policy_id"],
        nonce="0" * 64,
    )
    assert wrong_nonce.status_code == 422
    assert wrong_nonce.json() == {"detail": "invalid nonce"}

    tampered = _decide(
        client,
        evidence_id,
        created,
        evidence.replace('"id": 0', '"id": 1', 1),
        policy["policy_id"],
    )
    assert tampered.status_code == 422
    assert tampered.json() == {"detail": "evidence digest mismatch"}


def test_decision_retry_after_expansion_failure_still_evaluates_fresh(client):
    # A 422 expansion failure records nothing, so a later request against
    # a policy whose expansion fits decides normally for the same evidence.
    created, evidence, evidence_id = _receive_and_verify(client, _items(LIMIT + 1))
    too_big = _create_policy(client, _wildcard_rule(), name="too-big")
    assert (
        _decide(client, evidence_id, created, evidence, too_big["policy_id"])
        .status_code
        == 422
    )

    fits = _create_policy(client, {"claim": "items", "exists": True}, name="fits")
    response = _decide(client, evidence_id, created, evidence, fits["policy_id"])
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "allowed"


def test_existing_decision_retry_returns_original(client):
    claims = _items(10)
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(client, _wildcard_rule())

    first = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert first.status_code == 200, first.text
    replay = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert replay.status_code == 200
    assert replay.json() == first.json()
