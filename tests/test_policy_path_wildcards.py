"""Tests for wildcard array traversal in policy rule paths.

A ``path`` segment may be the object ``{"wildcard": true}``, which
expands every element of a JSON array at that position so the remaining
segments apply to each element; multiple wildcards expand left to right
into all candidate paths. Leaf comparisons keep their type-strict
semantics and hold when *any* complete candidate satisfies them;
``exists`` holds when at least one complete candidate is readable. Empty
arrays, non-arrays at a wildcard, missing fields and scalar/non-object
values mid-path all yield zero candidates and fail the leaf. The
wildcard object must carry exactly the one key with the JSON boolean
``true`` — any other object form is a 422 that allocates no version and
writes no state. Plain string segments (including the literal ``"*"``),
claim leaves, compound nodes, explanations and persisted history are
unchanged.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Policy

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"

WILDCARD = {"wildcard": True}
WILDCARD_LEAF = {"path": ["findings", WILDCARD, "severity"], "equals": "high"}

CLAIMS = {
    "findings": [
        {"severity": "low", "count": 1},
        {"severity": "high", "count": 3},
    ],
    "tags": [],
    "scalar": 5,
    "nested": {"items": [[{"v": 1}], [{"v": 2}]]},
    "*": {"literal": True},
}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    application = create_app(f"sqlite:///{tmp_path}/wildcards.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, rule=WILDCARD_LEAF, name="release", **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "name": name,
        "rule": rule,
    }
    body.update(overrides)
    return client.post("/v1/policies", json=body)


def _mac_for(nonce, claims):
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


def _decide_with_claims(client, rule, claims):
    """Create a policy from ``rule`` and decide it against ``claims``."""
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": claims,
            "mac": _mac_for(created["nonce"], claims),
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
    policy = _create(client, rule=rule)
    assert policy.status_code == 201, policy.text
    decided = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy.json()["policy_id"],
        },
    )
    assert decided.status_code == 200, decided.text
    return decided.json()


# --- creation, validation and persistence -----------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        WILDCARD_LEAF,
        # Wildcard as the final segment: candidates are the elements.
        {"path": ["findings", WILDCARD], "exists": True},
        # Multiple wildcards expand left to right.
        {
            "path": ["nested", "items", WILDCARD, WILDCARD, "v"],
            "equals": 2,
        },
        # Wildcards mix freely with every comparison key.
        {"path": ["findings", WILDCARD, "severity"], "in": ["high", "mid"]},
        {"path": ["findings", WILDCARD, "tags"], "contains": "x"},
        {"path": ["findings", WILDCARD, "count"], "lt": 10},
        {"path": ["findings", WILDCARD, "count"], "lte": 3},
        {"path": ["findings", WILDCARD, "count"], "gt": 0},
        {"path": ["findings", WILDCARD, "count"], "gte": 3},
        # Wildcard segments count toward the eight-segment bound.
        {"path": [WILDCARD] * 8, "exists": True},
        # Wildcards nest inside compound nodes.
        {
            "all": [
                WILDCARD_LEAF,
                {"not": {"path": ["findings", WILDCARD, "blocked"], "exists": True}},
            ]
        },
        {"any": [WILDCARD_LEAF, {"claim": "m", "equals": "x"}]},
        {"not": WILDCARD_LEAF},
    ],
)
def test_wildcard_rules_are_accepted(client, rule):
    response = _create(client, rule=rule)
    assert response.status_code == 201, response.text
    # The rule is echoed back with its structure exactly as submitted.
    assert response.json()["rule"] == rule


@pytest.mark.parametrize(
    "rule",
    [
        # The wildcard marker must be exactly {"wildcard": true}.
        {"path": [{"wildcard": 1}], "equals": 1},  # 1 is not true
        {"path": [{"wildcard": "true"}], "equals": 1},
        {"path": [{"wildcard": False}], "equals": 1},
        {"path": [{"wildcard": None}], "equals": 1},
        {"path": [{}], "equals": 1},  # empty object
        {"path": [{"wildcard": True, "extra": 1}], "equals": 1},  # extra key
        {"path": [{"Wildcard": True}], "equals": 1},  # wrong key
        {"path": [{"wildcard": True}, {"wildcard": 1}], "equals": 1},
        # Other non-string segment types stay invalid.
        {"path": [["wildcard"]], "equals": 1},
        {"path": [True], "equals": 1},
        # Wildcard segments count toward the segment bound.
        {"path": [WILDCARD] * 9, "exists": True},
        {"path": ["a", "b", "c", "d", WILDCARD, "e", "f", "g", "h"], "equals": 1},
    ],
)
def test_malformed_wildcard_segments_are_rejected_without_state(client, app, rule):
    assert _create(client, rule={"claim": "m", "equals": "x"}).status_code == 201

    rejected = _create(client, rule=rule)
    assert rejected.status_code == 422

    # No version was allocated and no row was written for the rejection.
    after = _create(client, rule=WILDCARD_LEAF)
    assert after.status_code == 201
    assert after.json()["version"] == 2
    with app.state.session_factory() as session:
        rows = session.query(Policy).order_by(Policy.version).all()
        assert [r.version for r in rows] == [1, 2]


def test_wildcard_rule_versions_and_queries(client):
    first = _create(client, rule=WILDCARD_LEAF)
    assert first.status_code == 201
    assert first.json()["version"] == 1
    second = _create(client, rule={"path": ["tags", WILDCARD], "contains": "a"})
    assert second.json()["version"] == 2

    listed = client.get(
        "/v1/policies", params={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert listed.status_code == 200
    rules = [item["rule"] for item in listed.json()["policies"]]
    assert WILDCARD_LEAF in rules
    assert {"path": ["tags", WILDCARD], "contains": "a"} in rules


def test_wildcard_rule_persists_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/wildcard-restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    first = _create(client1, rule=WILDCARD_LEAF)
    assert first.status_code == 201
    policy_id = first.json()["policy_id"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        row = session.get(Policy, policy_id)
        assert row is not None
        assert json.loads(row.rule_json) == WILDCARD_LEAF
    app2.state.engine.dispose()


# --- decision semantics ------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        # Any candidate satisfying the comparison makes the leaf true.
        {"path": ["findings", WILDCARD, "severity"], "equals": "high"},
        {"path": ["findings", WILDCARD, "severity"], "in": ["mid", "high"]},
        {"path": ["findings", WILDCARD, "count"], "gte": 3},
        {"path": ["findings", WILDCARD, "count"], "gt": 2},
        {"path": ["findings", WILDCARD, "count"], "lte": 1},
        {"path": ["findings", WILDCARD, "count"], "lt": 2},
        {"path": ["findings", WILDCARD, "severity"], "exists": True},
        # Multiple wildcards expand into all candidate paths.
        {"path": ["nested", "items", WILDCARD, WILDCARD, "v"], "equals": 2},
        # The string "*" remains a literal field name.
        {"path": ["*", "literal"], "equals": True},
        # Compound nodes compose wildcard leaves as usual.
        {
            "all": [
                {"path": ["findings", WILDCARD, "severity"], "equals": "high"},
                {"claim": "scalar", "equals": 5},
            ]
        },
        {"not": {"path": ["findings", WILDCARD, "severity"], "equals": "mid"}},
    ],
)
def test_wildcard_rules_allow_when_any_candidate_matches(client, rule):
    assert _decide_with_claims(client, rule, CLAIMS)["status"] == "allowed"


@pytest.mark.parametrize(
    "rule",
    [
        # No candidate satisfies the comparison.
        {"path": ["findings", WILDCARD, "severity"], "equals": "critical"},
        {"path": ["findings", WILDCARD, "severity"], "in": ["mid", "low2"]},
        {"path": ["findings", WILDCARD, "count"], "gt": 3},
        # Strict typing is kept per candidate: 3 does not equal "3".
        {"path": ["findings", WILDCARD, "count"], "equals": "3"},
        {"path": ["findings", WILDCARD, "count"], "equals": True},
        # Missing intermediate field: zero candidates.
        {"path": ["findings", WILDCARD, "missing"], "exists": True},
        # Empty array expands to zero candidates.
        {"path": ["tags", WILDCARD], "exists": True},
        # Non-array at the wildcard position: zero candidates.
        {"path": ["scalar", WILDCARD], "exists": True},
        # Scalar where further descent is needed: zero candidates.
        {"path": ["scalar", WILDCARD, "x"], "exists": True},
        {"path": ["findings", WILDCARD, "severity", "x"], "exists": True},
        # contains needs a candidate that is itself an array.
        {"path": ["findings", WILDCARD, "severity"], "contains": "high"},
        {"path": ["tags", WILDCARD], "contains": "a"},
    ],
)
def test_wildcard_rules_deny_when_no_candidate_matches(client, rule):
    assert _decide_with_claims(client, rule, CLAIMS)["status"] == "denied"


def test_wildcard_contains_matches_array_candidates(client):
    claims = {"rows": [{"vals": [1, 2]}, {"vals": [3]}]}
    rule = {"path": ["rows", WILDCARD, "vals"], "contains": 3}
    assert _decide_with_claims(client, rule, claims)["status"] == "allowed"
    rule = {"path": ["rows", WILDCARD, "vals"], "contains": 4}
    assert _decide_with_claims(client, rule, claims)["status"] == "denied"


def test_wildcard_explanation_keeps_shape_and_hides_values(client):
    rule = {
        "all": [
            {"path": ["findings", WILDCARD, "severity"], "equals": "high"},
            {"not": {"path": ["tags", WILDCARD], "exists": True}},
        ]
    }
    decision = _decide_with_claims(client, rule, CLAIMS)
    assert decision["status"] == "allowed"

    evaluation = client.get(
        f"/v1/decisions/{decision['decision_id']}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert evaluation.status_code == 200, evaluation.text
    nodes = evaluation.json()["nodes"]
    assert [n["node_type"] for n in nodes] == ["all", "leaf", "not", "leaf"]
    assert [n["rule_path"] for n in nodes] == [[], [0], [1], [1, 0]]
    assert [n["node_index"] for n in nodes] == [0, 1, 2, 3]
    assert [n["outcome"] for n in nodes] == [True, True, True, False]
    # The explanation carries no claim names, paths or values.
    payload = json.dumps(evaluation.json())
    for leaked in ("findings", "severity", "wildcard", "high", "tags"):
        assert leaked not in payload
