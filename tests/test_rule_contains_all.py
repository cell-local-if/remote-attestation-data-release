"""Tests for the ``contains_all`` array-coverage rule leaf.

Covers POST /v1/policies validation (422 with no version allocated and
no effect on a later creation of the same name), canonical persistence
and snapshot rendering, decision evaluation of the new leaf against
verified evidence claims (the located array must cover the whole
candidate set under type-strict scalar equality), composition with
all/any/not and the other comparison keys, wildcard paths, the
read-only evaluate endpoint, the compare endpoint's leaf-change
detection, explanation shape, and the unchanged decision boundary
codes.
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
    Evidence,
    Policy,
    PolicyCommitCounter,
    PolicyIdempotencyRecord,
    ProofLifecycleEvent,
)
from proof_release.policies import evaluate_rule, explain_rule, rule_structure

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"

WILD = {"wildcard": True}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    return create_app(f"sqlite:///{tmp_path}/contains_all.db")


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


def _evidence(nonce: str, claims: dict) -> str:
    return json.dumps({"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)})


def _receive_and_verify(client, claims):
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
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


def _policy(client, rule, name="release", **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "name": name,
        "rule": rule,
    }
    body.update(overrides)
    return client.post("/v1/policies", json=body)


def _create_policy(client, rule, name="release", **overrides):
    response = _policy(client, rule, name=name, **overrides)
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


def _status_for(client, claims, rule, name="release"):
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(client, rule, name=name)
    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 200
    return response.json()["status"]


def _evaluate(client, policy_id, claims, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "claims": claims,
    }
    body.update(overrides)
    return client.post(f"/v1/policies/{policy_id}/evaluate", json=body)


def _compare(client, left_policy_id, right_policy_id):
    return client.get(
        "/v1/policies/compare",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "left_policy_id": left_policy_id,
            "right_policy_id": right_policy_id,
        },
    )


# --- pure semantics --------------------------------------------------------


@pytest.mark.parametrize(
    "actual,candidates",
    [
        # Exact cover, and coverage with order shuffled.
        (["gold", "silver"], ["gold", "silver"]),
        (["silver", "gold"], ["gold", "silver"]),
        # Extra elements do not matter.
        (["a", "b", "c"], ["a", "c"]),
        (["a", "b", "c"], ["c"]),
        # Repeated elements do not matter.
        (["a", "a", "b"], ["a", "b"]),
        # Candidates need not be adjacent.
        ([1, 2, 3, 4], [1, 4]),
        ([1, 2, 3], [3, 1]),
        # A single candidate behaves like membership.
        (["gold", "silver"], ["gold"]),
        ([1, 2, 3], [2]),
        # Numeric equality by value.
        ([1.0, 2.0], [1, 2]),
        ([-0.0, 1], [0]),
        # 1 and true are distinct; with both present the set is covered.
        ([True, 1], [1, True]),
        ([False, 0], [0, False]),
        # Null among the elements.
        ([None, "x"], [None, "x"]),
        # Objects/arrays among the elements do not disturb the scan.
        (["a", {"k": "v"}, ["nested"], 1], [1, "a"]),
        # Mixed scalar types in the candidate set.
        ([3, "a", None, True], ["a", None, True, 3]),
        # The 32-candidate boundary, fully covered.
        (list(range(32)), list(range(32))),
        # The array may be far larger than the candidate set.
        (list(range(100)), [0, 50, 99]),
    ],
)
def test_contains_all_pure_match(actual, candidates):
    assert evaluate_rule(
        {"claim": "a", "contains_all": candidates}, {"a": actual}
    )


@pytest.mark.parametrize(
    "actual,candidates",
    [
        # Empty, non-array, absent-shaped values.
        ([], ["x"]),
        ("gold", ["gold"]),
        (1, [1]),
        (True, [True]),
        (None, [None]),
        ({"gold": 1}, ["gold"]),
        # One candidate short.
        (["gold"], ["gold", "silver"]),
        (["gold", "bronze"], ["gold", "silver"]),
        # Type-strict misses: true is not 1, numeric strings are not
        # numbers, null only equals null.
        ([1], [1, True]),
        ([True], [True, 1]),
        (["2"], [2, "2"]),
        ([2], [2, "2"]),
        ([0], [0, False]),
        ([False], [0, False]),
        ([None], [None, "x"]),
        ([""], [None, ""]),
        # Non-scalar elements never equal a scalar candidate.
        ([["gold"]], ["gold"]),
        ([{"x": 1}], [1]),
        ([["gold"], "silver"], ["gold", "silver"]),
        # Repeats in the array do not cover distinct candidates.
        (["gold", "gold"], ["gold", "silver"]),
        ([1, 1], [1, 2]),
    ],
)
def test_contains_all_pure_miss(actual, candidates):
    assert not evaluate_rule(
        {"claim": "a", "contains_all": candidates}, {"a": actual}
    )


def test_contains_all_pure_missing_claim_and_path():
    assert not evaluate_rule(
        {"claim": "a", "contains_all": [1]}, {"other": [1]}
    )
    assert not evaluate_rule(
        {"path": ["t", "tags"], "contains_all": [1]}, {"t": {}}
    )
    # A non-object on the descent path is a miss, not an error.
    assert not evaluate_rule(
        {"path": ["t", "tags"], "contains_all": [1]}, {"t": ["x"]}
    )
    # A terminal array under a path covers element-wise.
    assert evaluate_rule(
        {"path": ["t", "tags"], "contains_all": [1, 2]},
        {"t": {"tags": [1, 2]}},
    )
    # A terminal array missing one candidate fails.
    assert not evaluate_rule(
        {"path": ["t", "tags"], "contains_all": [1, 2]},
        {"t": {"tags": [1]}},
    )


def test_contains_all_pure_wildcard_any_candidate_covers():
    rule = {"path": ["items", WILD, "tags"], "contains_all": ["x", "y"]}
    # One complete path candidate covers both.
    assert evaluate_rule(
        rule, {"items": [{"tags": ["a"]}, {"tags": ["y", "x"]}]}
    )
    assert evaluate_rule(
        rule,
        {"items": [{"tags": ["x"]}, {"tags": ["x", "y", "z"]}]},
    )
    # No single complete candidate covers both, even if their union would.
    assert not evaluate_rule(
        rule, {"items": [{"tags": ["x"]}, {"tags": ["y"]}]}
    )
    assert not evaluate_rule(rule, {"items": [{"tags": ["x", "z"]}]})
    # Zero candidates: empty expansion, non-array terminal, missing field.
    assert not evaluate_rule(rule, {"items": []})
    assert not evaluate_rule(rule, {"items": [{"tags": "y"}]})
    assert not evaluate_rule(rule, {"items": [{}]})


# --- creation: accepted and read back verbatim -----------------------------


@pytest.mark.parametrize(
    "rule",
    [
        {"claim": "tier", "contains_all": ["gold"]},
        {"claim": "tier", "contains_all": ["gold", "silver"]},
        {"claim": "level", "contains_all": [3]},
        {"claim": "level", "contains_all": [3, 3.5]},
        {"claim": "flag", "contains_all": [True]},
        {"claim": "note", "contains_all": [None]},
        # 1 and true are distinct scalars, so they may coexist.
        {"claim": "mixed", "contains_all": [1, True]},
        {"claim": "mixed", "contains_all": [1, "1", None, False]},
        # The 32-candidate boundary.
        {"claim": "many", "contains_all": list(range(32))},
        {"path": ["tenant", "regions"], "contains_all": ["eu", "us"]},
        {"path": ["scores"], "contains_all": [0]},
        {
            "all": [
                {"claim": "tiers", "contains_all": ["gold", "silver"]},
                {"any": [
                    {"claim": "scores", "contains_all": [5, 6]},
                    {"not": {"path": ["tenant", "blocked"], "contains_all": [True]}},
                ]},
            ]
        },
    ],
)
def test_create_policy_accepts_contains_all_leaf(client, rule):
    response = _policy(client, rule)

    assert response.status_code == 201
    assert response.json()["rule"] == rule


def test_contains_all_rule_persists_canonically_and_lists_identically(client, app):
    rule = {
        "all": [
            {"claim": "tiers", "contains_all": ["gold", "silver"]},
            {"path": ["tenant", "regions"], "contains_all": ["eu"]},
        ]
    }
    created = _create_policy(client, rule)

    with app.state.session_factory() as session:
        row = session.get(Policy, created["policy_id"])
        assert row.rule_json == json.dumps(
            rule, sort_keys=True, separators=(",", ":")
        )

    fetched = client.get(
        "/v1/policies",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "policy_id": created["policy_id"],
        },
    )
    assert fetched.status_code == 200
    assert fetched.json()["policies"][0]["rule"] == rule

    listed = client.get(
        "/v1/policies", params={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert listed.status_code == 200
    assert listed.json()["policies"][0]["rule"] == rule


# --- creation: invalid leaves are 422, no version, no state ----------------


@pytest.mark.parametrize(
    "rule",
    [
        # The candidate list itself must be a list of 1..32 scalars.
        {"claim": "a"},  # missing comparison
        {"contains_all": [1]},  # missing locator
        {"claim": "a", "contains_all": "gold"},  # scalar, not a list
        {"claim": "a", "contains_all": 1},
        {"claim": "a", "contains_all": True},
        {"claim": "a", "contains_all": None},
        {"claim": "a", "contains_all": {"x": 1}},  # object, not a list
        {"claim": "a", "contains_all": []},  # empty list
        {"claim": "a", "contains_all": list(range(33))},  # over the bound
        # Non-scalar candidates.
        {"claim": "a", "contains_all": [["gold"]]},
        {"claim": "a", "contains_all": [{"x": 1}]},
        {"claim": "a", "contains_all": ["gold", ["silver"]]},
        {"claim": "a", "contains_all": [1, {"x": 1}]},
        # Repeating candidates under type-strict equality.
        {"claim": "a", "contains_all": ["gold", "gold"]},
        {"claim": "a", "contains_all": [1, 1.0]},
        {"claim": "a", "contains_all": [1.0, 1]},
        {"claim": "a", "contains_all": [True, True]},
        {"claim": "a", "contains_all": [None, None]},
        {"claim": "a", "contains_all": [0, -0.0]},
        {"claim": "a", "contains_all": ["a", 1, "a"]},
        # Mixed locators and multiple comparison keys.
        {"claim": "a", "path": ["a"], "contains_all": [1]},
        {"claim": "a", "contains_all": [1], "equals": 1},
        {"claim": "a", "contains_all": [1], "in": [1]},
        {"claim": "a", "contains_all": [1], "contains": 1},
        {"claim": "a", "contains_all": [1], "contains_any": [1]},
        {"path": ["a"], "contains_all": [1], "gte": 0},
        # Unknown sibling / node keys.
        {"claim": "a", "contains_all": [1], "extra": 2},
        {"claim": "a", "foo": 1},
        {"contains_all": [1], "all": []},
        # Malformed path siblings.
        {"path": ["a"], "contains_all": "x"},
        {"path": [], "contains_all": [1]},
        {"path": "a", "contains_all": [1]},
        {"path": [""], "contains_all": [1]},
        {"path": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
         "contains_all": [1]},
    ],
)
def test_create_policy_rejects_invalid_contains_all_leaves(client, app, rule):
    response = _policy(client, rule)

    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == 0


@pytest.mark.parametrize(
    "rule_json",
    [
        '{"claim": "a", "contains_all": [NaN]}',
        '{"claim": "a", "contains_all": [Infinity]}',
        '{"claim": "a", "contains_all": [-Infinity]}',
        '{"claim": "a", "contains_all": [1, NaN]}',
        '{"claim": "a", "contains_all": NaN}',
    ],
)
def test_non_finite_contains_all_value_never_persists(client, app, rule_json):
    body = (
        '{"tenant_id": "%s", "workload_id": "%s", "name": "release", '
        '"rule": %s}' % (TENANT, WORKLOAD, rule_json)
    )
    response = client.post(
        "/v1/policies", content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == 0


def test_rejected_contains_all_allocates_no_version_and_name_is_reusable(client, app):
    assert _create_policy(client, {"claim": "a", "contains_all": ["x"]})["version"] == 1

    rejected = _policy(client, {"claim": "a", "contains_all": []})
    assert rejected.status_code == 422
    rejected = _policy(client, {"claim": "a", "contains_all": [1, 1.0]})
    assert rejected.status_code == 422

    # A subsequent legal create still takes exactly version 2: the rejected
    # requests neither allocated a version nor inserted a row.
    created = _create_policy(client, {"claim": "a", "contains_all": ["y"]})
    assert created["version"] == 2
    with app.state.session_factory() as session:
        rows = session.query(Policy).order_by(Policy.version).all()
        assert [r.version for r in rows] == [1, 2]


def test_defensive_bounds_still_apply_to_contains_all(client):
    deep = leaf = {"claim": "a", "contains_all": [1]}
    for _ in range(33):
        deep = {"not": deep}
    assert _policy(client, deep).status_code == 422

    wide = {"all": [{"claim": "a", "contains_all": [1]}] * 257}
    assert _policy(client, wide).status_code == 422
    assert leaf is not None


# --- decision evaluation ---------------------------------------------------


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        # Every candidate is present, across scalar types.
        ({"tiers": ["gold", "silver"]},
         {"claim": "tiers", "contains_all": ["gold", "silver"]}, "allowed"),
        ({"tiers": ["silver", "gold", "bronze"]},
         {"claim": "tiers", "contains_all": ["gold", "silver"]}, "allowed"),
        ({"levels": [1, 2, 3]},
         {"claim": "levels", "contains_all": [3, 1]}, "allowed"),
        ({"levels": [1.0, 2.0, 3.0]},
         {"claim": "levels", "contains_all": [1, 2]}, "allowed"),
        ({"flags": [True, False]},
         {"claim": "flags", "contains_all": [False, True]}, "allowed"),
        ({"notes": [None, "x"]},
         {"claim": "notes", "contains_all": [None, "x"]}, "allowed"),
        # Missing one or more candidates denies.
        ({"tiers": ["bronze", "gold"]},
         {"claim": "tiers", "contains_all": ["gold", "silver"]}, "denied"),
        ({"levels": [2]},
         {"claim": "levels", "contains_all": [1, "2"]}, "denied"),
        ({"levels": ["2"]},
         {"claim": "levels", "contains_all": [2, "2"]}, "denied"),
        ({"flags": [True]},
         {"claim": "flags", "contains_all": [1, True]}, "denied"),
        ({"levels": [1]},
         {"claim": "levels", "contains_all": [True, 1]}, "denied"),
        ({"levels": [0]},
         {"claim": "levels", "contains_all": [0, False]}, "denied"),
        ({"notes": [None, False, 0, ""]},
         {"claim": "notes", "contains_all": [None, "x"]}, "denied"),
        # Empty array, non-array value, missing claim all deny.
        ({"tiers": []},
         {"claim": "tiers", "contains_all": ["gold"]}, "denied"),
        ({"tiers": "gold"},
         {"claim": "tiers", "contains_all": ["gold"]}, "denied"),
        ({},
         {"claim": "tiers", "contains_all": ["gold"]}, "denied"),
        # A scalar terminal is not searched.
        ({"word": "ingot"},
         {"claim": "word", "contains_all": ["go", "ing"]}, "denied"),
        # Nested arrays/objects are opaque elements, not scalar matches.
        ({"tiers": [["gold"], "silver"]},
         {"claim": "tiers", "contains_all": ["gold", "silver"]}, "denied"),
        ({"levels": [{"v": 1}]},
         {"claim": "levels", "contains_all": [1]}, "denied"),
        # Path-located arrays.
        (
            {"tenant": {"regions": ["eu", "us", "ap"]}},
            {"path": ["tenant", "regions"], "contains_all": ["us", "eu"]},
            "allowed",
        ),
        (
            {"tenant": {"regions": ["ap"]}},
            {"path": ["tenant", "regions"], "contains_all": ["eu", "us"]},
            "denied",
        ),
        (
            {"tenant": {}},
            {"path": ["tenant", "regions"], "contains_all": ["eu"]},
            "denied",
        ),
        (
            {"tenant": {"regions": "eu"}},
            {"path": ["tenant", "regions"], "contains_all": ["eu"]},
            "denied",
        ),
        # Wildcard paths: one complete candidate array must cover all.
        (
            {"items": [{"tags": ["a"]}, {"tags": ["b", "y", "x"]}]},
            {"path": ["items", WILD, "tags"], "contains_all": ["x", "y"]},
            "allowed",
        ),
        (
            # The union covers both but no single array does.
            {"items": [{"tags": ["x"]}, {"tags": ["y"]}]},
            {"path": ["items", WILD, "tags"], "contains_all": ["x", "y"]},
            "denied",
        ),
        (
            {"items": []},
            {"path": ["items", WILD, "tags"], "contains_all": ["x"]},
            "denied",
        ),
    ],
)
def test_contains_all_leaf_decision_evaluates_type_strictly(
    client, claims, rule, expected
):
    assert _status_for(client, claims, rule) == expected


def test_contains_all_composes_with_all_any_not_and_other_keys(client):
    claims = {
        "tiers": ["gold", "silver", "bronze"],
        "score": 7,
        "tenant": {"regions": ["eu", "us"]},
    }
    satisfied = {
        "all": [
            {"claim": "tiers", "contains_all": ["gold", "silver"]},
            {"any": [
                {"claim": "score", "gte": 5},
                {"claim": "score", "lt": 0},
            ]},
            {"not": {"path": ["tenant", "blocked"], "contains_all": [True]}},
            {"path": ["tenant", "regions"], "contains_all": ["us", "eu"]},
        ]
    }
    assert _status_for(client, claims, satisfied) == "allowed"

    unsatisfied = {
        "all": [
            {"claim": "tiers", "contains_all": ["platinum", "gold"]},
            {"claim": "score", "lt": 5},
        ]
    }
    assert _status_for(client, claims, unsatisfied) == "denied"


def test_contains_all_under_not_allows_on_miss_and_non_array(client):
    rule = {"not": {"claim": "tiers", "contains_all": ["gold", "silver"]}}
    # Missing a candidate -> the leaf is false -> not is true.
    assert _status_for(client, {"tiers": ["bronze", "gold"]}, rule) == "allowed"
    assert _status_for(client, {}, rule) == "allowed"
    assert _status_for(client, {"tiers": "gold"}, rule) == "allowed"
    # Full coverage -> the leaf is true -> not is false.
    assert (
        _status_for(client, {"tiers": ["gold", "silver"]}, rule) == "denied"
    )
    assert (
        _status_for(client, {"tiers": ["silver", "gold", "x"]}, rule)
        == "denied"
    )


def test_contains_all_decision_is_deterministic_and_scope_isolated(client):
    claims = {"tiers": ["gold", "silver"]}
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(
        client, {"claim": "tiers", "contains_all": ["gold", "silver"]}
    )

    first = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    second = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert first.status_code == 200
    assert second.json() == first.json()
    assert first.json()["status"] == "allowed"

    cross = _decide(
        client,
        evidence_id,
        created,
        evidence,
        policy["policy_id"],
        tenant_id="tenant-b",
    )
    assert cross.status_code == 404


def test_contains_all_trace_and_evaluation_leak_no_locator_or_values(client):
    rule = {
        "all": [
            {"claim": "tiers", "contains_all": ["gold", "silver"]},
            {"not": {"path": ["meta", "flags"], "contains_all": [True]}},
        ]
    }
    claims = {"tiers": ["bronze", "gold"], "meta": {"flags": [False, True]}}
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(client, rule)

    decided = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert decided.status_code == 200
    assert decided.json()["status"] == "denied"
    decision_id = decided.json()["decision_id"]

    evaluation = client.get(
        f"/v1/decisions/{decision_id}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert evaluation.status_code == 200
    nodes = evaluation.json()["nodes"]
    assert [n["node_type"] for n in nodes] == ["all", "leaf", "not", "leaf"]
    assert [n["rule_path"] for n in nodes] == [[], [0], [1], [1, 0]]
    # First leaf misses "silver"; not-child covers [True] so it is true.
    assert [n["outcome"] for n in nodes] == [False, False, False, True]
    rendered = json.dumps(nodes)
    for secret in ("tiers", "meta", "flags", "gold", "silver", "bronze",
                   "contains_all"):
        assert secret not in rendered

    trace = client.get(
        f"/v1/decisions/{decision_id}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert trace.status_code == 200
    assert trace.json()["policy_version"]["rule"] == rule


def test_contains_all_explanation_matches_structure_and_root():
    rule = {"any": [
        {"all": [
            {"claim": "a", "contains_all": [1, 2]},
            {"not": {"path": ["x"], "contains_all": [None]}},
        ]},
        {"not": {"claim": "b", "contains_all": [1, 2]}},
    ]}
    claims = {"a": [1, 2], "x": [None], "b": 9}
    nodes = explain_rule(rule, claims)
    skeleton = rule_structure(rule)
    assert len(nodes) == len(skeleton)
    for node, (path, node_type) in zip(nodes, skeleton):
        assert tuple(node["rule_path"]) == path
        assert node["node_type"] == node_type
    assert nodes[0]["outcome"] is True
    assert nodes[0]["outcome"] == evaluate_rule(rule, claims)


# --- trial evaluation endpoint ----------------------------------------------


def test_evaluate_endpoint_allowed_matches_root_outcome(client):
    rule = {
        "all": [
            {"claim": "tiers", "contains_all": ["gold", "silver"]},
            {"claim": "level", "gte": 2},
        ]
    }
    policy = _create_policy(client, rule)

    allowed = _evaluate(
        client,
        policy["policy_id"],
        {"tiers": ["silver", "gold", "bronze"], "level": 3},
    )
    assert allowed.status_code == 200
    data = allowed.json()
    assert data["allowed"] is True
    assert data["evaluation"][0]["outcome"] is True
    assert data["allowed"] == data["evaluation"][0]["outcome"]
    assert [n["node_type"] for n in data["evaluation"]] == [
        "all", "leaf", "leaf",
    ]

    denied = _evaluate(
        client, policy["policy_id"], {"tiers": ["bronze", "gold"], "level": 3}
    )
    assert denied.status_code == 200
    data = denied.json()
    assert data["allowed"] is False
    assert data["evaluation"][0]["outcome"] is False
    assert [n["outcome"] for n in data["evaluation"]] == [False, False, True]


def test_evaluate_endpoint_repeated_claims_are_stable(client):
    policy = _create_policy(
        client, {"claim": "tiers", "contains_all": ["gold"]}
    )
    first = _evaluate(client, policy["policy_id"], {"tiers": ["gold"]})
    second = _evaluate(client, policy["policy_id"], {"tiers": ["gold"]})
    assert first.status_code == second.status_code == 200
    assert first.json()["allowed"] == second.json()["allowed"] is True
    assert first.json()["evaluation"] == second.json()["evaluation"]

    denied_first = _evaluate(client, policy["policy_id"], {"tiers": []})
    denied_second = _evaluate(client, policy["policy_id"], {"tiers": []})
    assert denied_first.json()["allowed"] is False
    assert denied_first.json()["allowed"] == denied_second.json()["allowed"]
    assert denied_first.json()["evaluation"] == denied_second.json()["evaluation"]


def test_evaluate_endpoint_leaks_no_contains_all_material(client):
    policy = _create_policy(
        client,
        {"claim": "secret-claim-name", "contains_all": ["secret-value"]},
    )
    response = _evaluate(
        client,
        policy["policy_id"],
        {"secret-claim-name": ["secret-value"]},
    )
    assert response.status_code == 200
    assert response.json()["allowed"] is True
    assert "secret-claim-name" not in response.text
    assert "secret-value" not in response.text
    assert "contains_all" not in response.text


def test_evaluate_endpoint_writes_no_state(client, app):
    policy = _create_policy(
        client, {"claim": "tiers", "contains_all": ["gold"]}
    )
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

    response = _evaluate(client, policy["policy_id"], {"tiers": ["gold"]})
    assert response.status_code == 200
    assert response.json()["allowed"] is True

    with app.state.session_factory() as session:
        for model, count in before.items():
            assert session.query(model).count() == count, model.__name__


def test_retired_contains_all_version_still_evaluates(client):
    policy = _create_policy(
        client, {"claim": "tiers", "contains_all": ["gold"]}
    )
    retired = client.post(
        f"/v1/policies/{policy['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200

    response = _evaluate(client, policy["policy_id"], {"tiers": ["gold"]})
    assert response.status_code == 200
    assert response.json()["allowed"] is True

    denied = _evaluate(client, policy["policy_id"], {"tiers": ["bronze"]})
    assert denied.status_code == 200
    assert denied.json()["allowed"] is False


# --- compare endpoint --------------------------------------------------------


def test_compare_detects_contains_all_candidate_list_change(client):
    left = _create_policy(
        client, {"claim": "tiers", "contains_all": ["gold", "silver"]},
        name="left",
    )
    right = _create_policy(
        client,
        {"claim": "tiers", "contains_all": ["gold"]},
        name="right",
    )
    response = _compare(client, left["policy_id"], right["policy_id"])
    assert response.status_code == 200
    data = response.json()
    assert data["identical"] is False
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [],
            "left": {"claim": "tiers", "contains_all": ["gold", "silver"]},
            "right": {"claim": "tiers", "contains_all": ["gold"]},
        }
    ]


def test_compare_detects_contains_all_vs_contains_any_change(client):
    left = _create_policy(
        client, {"claim": "tiers", "contains_all": ["gold"]}, name="left"
    )
    right = _create_policy(
        client, {"claim": "tiers", "contains_any": ["gold"]}, name="right"
    )
    data = _compare(client, left["policy_id"], right["policy_id"]).json()
    assert data["identical"] is False
    assert len(data["changes"]) == 1
    assert data["changes"][0]["change"] == "changed"
    assert data["changes"][0]["rule_path"] == []


def test_compare_contains_all_candidate_order_is_positional(client):
    left = _create_policy(
        client, {"claim": "n", "contains_all": [1, 2]}, name="left"
    )
    right = _create_policy(
        client, {"claim": "n", "contains_all": [2, 1]}, name="right"
    )
    data = _compare(client, left["policy_id"], right["policy_id"]).json()
    assert data["identical"] is False
    assert len(data["changes"]) == 1
    assert data["changes"][0]["change"] == "changed"
    assert data["changes"][0]["rule_path"] == []


def test_compare_identical_contains_all_rules(client):
    rule = {
        "all": [
            {"claim": "tiers", "contains_all": ["gold", "silver"]},
            {"path": ["tenant", "regions"], "contains_all": ["eu"]},
        ]
    }
    left = _create_policy(client, rule, name="one")
    right = _create_policy(client, rule, name="two")
    data = _compare(client, left["policy_id"], right["policy_id"]).json()
    assert data["identical"] is True
    assert data["changes"] == []


def test_compare_contains_all_numeric_candidates_by_value(client):
    left = _create_policy(
        client, {"claim": "n", "contains_all": [1, 2]}, name="left"
    )
    right = _create_policy(
        client, {"claim": "n", "contains_all": [1.0, 2.0]}, name="right"
    )
    data = _compare(client, left["policy_id"], right["policy_id"]).json()
    assert data["identical"] is True
    assert data["changes"] == []

    other = _create_policy(
        client, {"claim": "n", "contains_all": [True, 2]}, name="other"
    )
    data = _compare(client, left["policy_id"], other["policy_id"]).json()
    assert data["identical"] is False


# --- backward compatibility ------------------------------------------------


def test_contains_any_and_in_semantics_unchanged_next_to_contains_all(client):
    # contains_any: at least one hit; contains_all requires every one.
    assert (
        _status_for(
            client,
            {"t": ["gold"]},
            {"claim": "t", "contains_any": ["gold", "silver"]},
        )
        == "allowed"
    )
    assert (
        _status_for(
            client,
            {"t": ["gold"]},
            {"claim": "t", "contains_all": ["gold", "silver"]},
        )
        == "denied"
    )
    # in: scalar actual value against a scalar candidate set; an array
    # terminal value still never matches an in candidate.
    assert (
        _status_for(client, {"t": "gold"}, {"claim": "t", "in": ["gold"]})
        == "allowed"
    )
    assert (
        _status_for(client, {"t": ["gold"]}, {"claim": "t", "in": ["gold"]})
        == "denied"
    )
    # The existing operators' validation is unaffected by the new key.
    assert _policy(client, {"claim": "a", "in": [1, 1.0]}).status_code == 422
    assert _policy(client, {"claim": "a", "in": []}).status_code == 422
    assert (
        _policy(client, {"claim": "a", "contains_any": []}).status_code == 422
    )
    assert (
        _policy(client, {"claim": "a", "contains": ["x"]}).status_code == 422
    )


def test_equals_semantics_unchanged_next_to_contains_all(client):
    assert (
        _status_for(client, {"m": ["x"]}, {"claim": "m", "equals": "x"})
        == "denied"
    )
    assert (
        _status_for(client, {"m": True}, {"claim": "m", "equals": 1}) == "denied"
    )
    assert _status_for(client, {}, {"claim": "m", "equals": None}) == "denied"


# --- decision boundary codes stay intact with the new leaf -----------------


def test_digest_mismatch_with_contains_all_policy_is_422(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"a": [1]})
    policy = _create_policy(client, {"claim": "a", "contains_all": [1]})

    response = _decide(client, evidence_id, created, evidence + " ", policy["policy_id"])
    assert response.status_code == 422
    assert response.json()["detail"] == "evidence digest mismatch"


def test_unverified_evidence_with_contains_all_policy_is_409(client, app):
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    evidence = _evidence(created["nonce"], {"a": [1]})
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
    policy = _create_policy(client, {"claim": "a", "contains_all": [1]})

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 409
    assert response.json()["detail"] == "evidence is not verified"
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 0


def test_unknown_policy_with_contains_all_shape_is_404(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"a": [1]})
    # A well-formed UUID the service never minted is simply not found.
    missing = "00000000-0000-0000-0000-000000000000"
    response = _decide(client, evidence_id, created, evidence, missing)
    assert response.status_code == 404
    assert response.json()["detail"] == "policy not found"

    foreign = _create_policy(
        client, {"claim": "a", "contains_all": [1]}, tenant_id="tenant-b"
    )
    cross = _decide(
        client, evidence_id, created, evidence, foreign["policy_id"]
    )
    assert cross.status_code == 404
    assert cross.json()["detail"] == "policy not found"


def test_retired_contains_all_policy_is_409(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"a": [1]})
    policy = _create_policy(client, {"claim": "a", "contains_all": [1]})
    retired = client.post(
        f"/v1/policies/{policy['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 409
    assert response.json()["detail"] == "policy is retired"


def test_repeated_decision_with_contains_all_records_first_once(client, app):
    created, evidence, evidence_id = _receive_and_verify(client, {"a": [1, 2]})
    # v1 denies; it is recorded first. Repeating the same policy returns
    # that first record and inserts no second row, even after a later
    # policy version has been evaluated against the same evidence.
    v1 = _create_policy(client, {"claim": "a", "contains_all": [1, 9]})
    first = _decide(client, evidence_id, created, evidence, v1["policy_id"])
    assert first.status_code == 200
    assert first.json()["status"] == "denied"

    repeat = _decide(client, evidence_id, created, evidence, v1["policy_id"])
    assert repeat.status_code == 200
    assert repeat.json() == first.json()

    # A distinct policy version gets its own one row; the v1 record stays
    # immutable afterwards.
    v2 = _create_policy(client, {"claim": "a", "contains_all": [1, 2]})
    other = _decide(client, evidence_id, created, evidence, v2["policy_id"])
    assert other.status_code == 200
    assert other.json()["status"] == "allowed"

    repeat_after = _decide(
        client, evidence_id, created, evidence, v1["policy_id"]
    )
    assert repeat_after.json() == first.json()

    with app.state.session_factory() as session:
        rows = session.query(Decision).all()
        assert len(rows) == 2
        assert {(r.policy_id, r.status) for r in rows} == {
            (v1["policy_id"], "denied"),
            (v2["policy_id"], "allowed"),
        }


def test_contains_all_persists_neither_claims_nor_evidence(client, app):
    secret_value = "super-secret-claim-value"
    created, evidence, evidence_id = _receive_and_verify(
        client, {"tags": [secret_value]}
    )
    policy = _create_policy(
        client, {"claim": "tags", "contains_all": [secret_value]}
    )
    decided = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert decided.status_code == 200

    with app.state.session_factory() as session:
        decision = session.query(Decision).one()
        evidence_row = session.get(Evidence, evidence_id)
        policy_row = session.get(Policy, policy["policy_id"])
        for target in (decision, evidence_row, policy_row):
            columns = {
                c.name: getattr(target, c.name) for c in target.__table__.columns
            }
            for name, value in columns.items():
                assert evidence not in str(value), f"evidence leaked in {name}"
        # The expected scalars legitimately live in the immutable rule
        # snapshot, so they are not asserted absent from the policy row; the
        # evidence and nonce material must be absent everywhere.
        assert created["nonce"] not in str(
            {c.name: getattr(decision, c.name) for c in decision.__table__.columns}
        )
