"""Tests for the ``contains_any`` array/intersection rule leaf.

A ``contains_any`` leaf pairs one ``claim`` or ``path`` locator with a
list of 1 to 32 unique JSON scalars and is true when the located value is
itself a JSON array holding at least one element type-strictly equal to
at least one listed candidate. These tests cover the pure engine
semantics, POST /v1/policies validation (422 with no version allocated
and the name reusable), canonical persistence, the read-only
POST /v1/policies/{policy_id}/evaluate trial, formal
POST /v1/evidence/{evidence_id}/decisions (including the explanation
shape and its no-leakage guarantee), GET /v1/policies/compare leaf
diffs, wildcard paths, and the surrounding boundary semantics.
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
    Policy,
    PolicyCommitCounter,
    PolicyIdempotencyRecord,
    ProofLifecycleEvent,
)
from proof_release.policies import (
    InvalidRule,
    evaluate_rule,
    explain_rule,
    rule_structure,
    validate_rule,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    return create_app(f"sqlite:///{tmp_path}/contains_any.db")


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
    assert response.status_code == 201, response.text
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
    assert response.status_code == 200, response.text
    return response.json()["status"]


def _evaluate(client, policy_id, claims, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "claims": claims,
    }
    body.update(overrides)
    return client.post(f"/v1/policies/{policy_id}/evaluate", json=body)


# --- pure validation --------------------------------------------------------


@pytest.mark.parametrize(
    "candidates",
    [
        ["gold"],
        ["gold", "silver", 1, 2.5, True, False, None],
        [1, True],
        [0, False, None, ""],
        list(range(32)),
    ],
)
def test_validate_accepts_well_formed_candidate_sets(candidates):
    validate_rule({"claim": "a", "contains_any": list(candidates)})
    validate_rule({"path": ["a", "b"], "contains_any": list(candidates)})


@pytest.mark.parametrize(
    "candidates",
    [
        [],
        [1, 1.0],
        [1.0, 1],
        [2.0, 2],
        [True, True],
        [False, False],
        [None, None],
        ["a", "a"],
        [1, 2, 1],
        list(range(32)) + [0],
    ],
)
def test_validate_rejects_empty_oversized_or_repeating(candidates):
    with pytest.raises(InvalidRule):
        validate_rule({"claim": "a", "contains_any": candidates})


@pytest.mark.parametrize(
    "value",
    [
        None,
        "gold",
        1,
        1.5,
        True,
        {"x": 1},
        [["gold"]],
        [{"x": 1}],
        [None, ["x"]],
        [None, {}],
    ],
)
def test_validate_rejects_non_list_or_non_scalar_entries(value):
    with pytest.raises(InvalidRule):
        validate_rule({"claim": "a", "contains_any": value})


@pytest.mark.parametrize(
    "rule",
    [
        # A leaf needs exactly one locator and one comparison.
        {"contains_any": ["a"]},
        {"claim": "a"},
        {"claim": "a", "path": ["a"], "contains_any": ["a"]},
        {"claim": "a", "contains_any": ["a"], "equals": "a"},
        {"claim": "a", "contains_any": ["a"], "contains": "a"},
        {"claim": "a", "contains_any": ["a"], "in": ["a"]},
        {"path": ["a"], "contains_any": ["a"], "gte": 0},
        {"claim": "a", "contains_any": ["a"], "extra": 1},
        {"claim": "a", "foo": 1},
        # Malformed path siblings.
        {"path": [], "contains_any": ["a"]},
        {"path": "a", "contains_any": ["a"]},
        {"path": [""], "contains_any": ["a"]},
        {"path": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
         "contains_any": ["a"]},
    ],
)
def test_validate_rejects_malformed_leaves(rule):
    with pytest.raises(InvalidRule):
        validate_rule(rule)


# --- pure evaluation --------------------------------------------------------


@pytest.mark.parametrize(
    "actual,candidates",
    [
        (["gold", "silver"], ["gold"]),
        (["gold", "silver"], ["bronze", "silver"]),
        ([1, 2, 3], [2]),
        ([1, 2, 3], [2.0]),
        ([1.0, 2.0], [1]),
        ([-0.0], [0]),
        ([True, False], [False]),
        ([None, "x"], [None]),
        (["x", None, 1], [None, 9, 8]),
        # Objects/arrays among the elements do not disturb the scan.
        (["a", {"k": "v"}, ["nested"], 1], [1]),
    ],
)
def test_evaluate_pure_match(actual, candidates):
    assert evaluate_rule(
        {"claim": "a", "contains_any": candidates}, {"a": actual}
    )


@pytest.mark.parametrize(
    "actual,candidates",
    [
        # Empty, non-array, absent.
        ([], ["x"]),
        ("gold", ["gold"]),
        (1, [1]),
        (True, [True]),
        (None, [None]),
        ({"gold": 1}, ["gold"]),
        # Type-strict misses.
        ([1], [True]),
        ([True], [1]),
        (["2"], [2]),
        ([2], ["2"]),
        ([0], [False]),
        ([False], [0]),
        ([None], [False]),
        ([None], [0]),
        ([None], [""]),
        ([""], [None]),
        # Non-scalar elements never equal a scalar candidate.
        ([["gold"]], ["gold"]),
        ([{"x": 1}], [1]),
        # Candidate set present but nothing in common.
        (["a", "b"], ["c", "d"]),
        ([1, 2], [3, 4]),
    ],
)
def test_evaluate_pure_miss(actual, candidates):
    assert not evaluate_rule(
        {"claim": "a", "contains_any": candidates}, {"a": actual}
    )


def test_evaluate_missing_claim_and_path_descent():
    assert not evaluate_rule(
        {"claim": "a", "contains_any": [1]}, {"other": [1]}
    )
    assert not evaluate_rule(
        {"path": ["t", "tags"], "contains_any": [1]}, {"t": {}}
    )
    # A non-object on the descent path is a miss, not an error.
    assert not evaluate_rule(
        {"path": ["t", "tags"], "contains_any": [1]}, {"t": ["x"]}
    )
    # A terminal array under a path matches element-wise.
    assert evaluate_rule(
        {"path": ["t", "tags"], "contains_any": [1, 2]},
        {"t": {"tags": [5, 2]}},
    )
    # A terminal non-array under a path is a miss even when equal-looking.
    assert not evaluate_rule(
        {"path": ["t", "tags"], "contains_any": ["eu"]},
        {"t": {"tags": "eu"}},
    )


def test_evaluate_wildcard_path_any_complete_candidate():
    rule = {
        "path": ["tenants", {"wildcard": True}, "regions"],
        "contains_any": ["eu", "us"],
    }
    assert evaluate_rule(
        rule,
        {"tenants": [
            {"regions": ["ap"]},
            {"regions": ["ca", "us"]},
        ]},
    )
    assert not evaluate_rule(
        rule,
        {"tenants": [{"regions": ["ap"]}, {"regions": []}]},
    )
    # Empty array at a wildcard yields no candidates.
    assert not evaluate_rule(rule, {"tenants": []})
    # Non-array at a wildcard yields no candidates.
    assert not evaluate_rule(rule, {"tenants": {}})


def test_explain_shape_and_root_outcome():
    rule = {
        "all": [
            {"claim": "tiers", "contains_any": ["gold", "platinum"]},
            {"not": {"path": ["meta", "flags"], "contains_any": [True]}},
        ]
    }
    claims = {"tiers": ["silver", "platinum"], "meta": {"flags": [False]}}
    nodes = explain_rule(rule, claims)
    skeleton = rule_structure(rule)
    assert len(nodes) == len(skeleton)
    for node, (path, node_type) in zip(nodes, skeleton):
        assert tuple(node["rule_path"]) == path
        assert node["node_type"] == node_type
    assert [n["node_type"] for n in nodes] == ["all", "leaf", "not", "leaf"]
    assert [n["rule_path"] for n in nodes] == [[], [0], [1], [1, 0]]
    assert [n["outcome"] for n in nodes] == [True, True, True, False]
    assert nodes[0]["outcome"] == evaluate_rule(rule, claims)
    # The explanation carries only positions, types and booleans.
    rendered = json.dumps(nodes)
    for secret in ("tiers", "meta", "flags", "gold", "platinum"):
        assert secret not in rendered


# --- POST /v1/policies ------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        {"claim": "tier", "contains_any": ["gold"]},
        {"claim": "level", "contains_any": [3]},
        {"claim": "level", "contains_any": [3.5, 4]},
        {"claim": "flag", "contains_any": [True]},
        {"claim": "note", "contains_any": [None]},
        {"claim": "mixed", "contains_any": ["a", 1, True, None, 2.5]},
        {"path": ["tenant", "regions"], "contains_any": ["eu", "us"]},
        {
            "all": [
                {"claim": "tiers", "contains_any": ["gold"]},
                {"any": [
                    {"claim": "scores", "contains_any": [5, 6]},
                    {"not": {"path": ["tenant", "blocked"],
                             "contains_any": [True]}},
                ]},
            ]
        },
    ],
)
def test_create_policy_accepts_contains_any_leaf(client, rule):
    response = _policy(client, rule)
    assert response.status_code == 201, response.text
    assert response.json()["rule"] == rule


def test_contains_any_persists_canonically(client, app):
    rule = {
        "all": [
            {"claim": "tiers", "contains_any": ["gold", "silver"]},
            {"path": ["tenant", "regions"], "contains_any": ["eu"]},
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


@pytest.mark.parametrize(
    "rule",
    [
        # Missing / wrong-typed candidate list.
        {"claim": "a", "contains_any": []},
        {"claim": "a", "contains_any": "gold"},
        {"claim": "a", "contains_any": 1},
        {"claim": "a", "contains_any": {"x": 1}},
        # Non-scalar or repeating entries.
        {"claim": "a", "contains_any": [["gold"]]},
        {"claim": "a", "contains_any": [{"x": 1}]},
        {"claim": "a", "contains_any": [1, 1.0]},
        {"claim": "a", "contains_any": [True, True]},
        {"claim": "a", "contains_any": [None, None]},
        {"claim": "a", "contains_any": ["a", "a"]},
        {"claim": "a", "contains_any": list(range(33))},
        # Locator/composition shape errors.
        {"claim": "a"},
        {"contains_any": [1]},
        {"claim": "a", "path": ["a"], "contains_any": [1]},
        {"claim": "a", "contains_any": [1], "equals": 1},
        {"claim": "a", "contains_any": [1], "in": [1]},
        {"path": ["a"], "contains_any": [1], "gte": 0},
        {"claim": "a", "contains_any": [1], "extra": 2},
        {"path": [], "contains_any": [1]},
        {"path": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
         "contains_any": [1]},
    ],
)
def test_create_policy_rejects_invalid_contains_any(client, app, rule):
    response = _policy(client, rule)
    assert response.status_code == 422, response.text
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == 0
        assert session.query(PolicyCommitCounter).count() == 0
        assert session.query(PolicyIdempotencyRecord).count() == 0


@pytest.mark.parametrize(
    "rule_json",
    [
        '{"claim": "a", "contains_any": [NaN]}',
        '{"claim": "a", "contains_any": [Infinity]}',
        '{"claim": "a", "contains_any": [1, -Infinity]}',
    ],
)
def test_non_finite_candidate_never_persists(client, app, rule_json):
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


def test_rejected_contains_any_allocates_no_version_and_name_reusable(client):
    assert _policy(client, {"claim": "a", "contains_any": []}).status_code == 422
    assert (
        _policy(client, {"claim": "a", "contains_any": [1, 1.0]}).status_code
        == 422
    )
    created = _create_policy(client, {"claim": "a", "contains_any": ["x"]})
    assert created["version"] == 1


def test_one_and_true_are_distinct_candidates(client):
    response = _policy(client, {"claim": "a", "contains_any": [1, True]})
    assert response.status_code == 201, response.text


def test_defensive_bounds_still_apply_to_contains_any(client):
    deep = leaf = {"claim": "a", "contains_any": [1]}
    for _ in range(33):
        deep = {"not": deep}
    assert _policy(client, deep).status_code == 422

    wide = {"all": [{"claim": "a", "contains_any": [1]}] * 257}
    assert _policy(client, wide).status_code == 422
    assert leaf is not None


# --- formal decision --------------------------------------------------------


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        # At least one array element hits at least one candidate.
        ({"tiers": ["gold", "silver"]},
         {"claim": "tiers", "contains_any": ["gold", "bronze"]}, "allowed"),
        ({"levels": [1, 2, 3]}, {"claim": "levels", "contains_any": [2.0]},
         "allowed"),
        ({"flags": [True, False]}, {"claim": "flags", "contains_any": [False]},
         "allowed"),
        ({"notes": [None, "x"]}, {"claim": "notes", "contains_any": [None]},
         "allowed"),
        # No intersection.
        ({"tiers": ["bronze"]},
         {"claim": "tiers", "contains_any": ["gold", "silver"]}, "denied"),
        ({"levels": [2]}, {"claim": "levels", "contains_any": ["2"]}, "denied"),
        ({"levels": ["2"]}, {"claim": "levels", "contains_any": [2]}, "denied"),
        ({"flags": [True]}, {"claim": "flags", "contains_any": [1]}, "denied"),
        ({"levels": [1]}, {"claim": "levels", "contains_any": [True]}, "denied"),
        ({"notes": [0, False, ""]},
         {"claim": "notes", "contains_any": [None]}, "denied"),
        # Empty array, non-array value and missing claim all deny.
        ({"tiers": []}, {"claim": "tiers", "contains_any": ["gold"]}, "denied"),
        ({"tiers": "gold"}, {"claim": "tiers", "contains_any": ["gold"]},
         "denied"),
        ({}, {"claim": "tiers", "contains_any": ["gold"]}, "denied"),
        # Nested arrays/objects are opaque elements.
        ({"tiers": [["gold"]]},
         {"claim": "tiers", "contains_any": ["gold"]}, "denied"),
        ({"levels": [{"v": 1}]},
         {"claim": "levels", "contains_any": [1]}, "denied"),
        # Path-located arrays.
        (
            {"tenant": {"regions": ["eu", "us"]}},
            {"path": ["tenant", "regions"], "contains_any": ["us", "ap"]},
            "allowed",
        ),
        (
            {"tenant": {"regions": ["ap"]}},
            {"path": ["tenant", "regions"], "contains_any": ["eu", "us"]},
            "denied",
        ),
        (
            {"tenant": {}},
            {"path": ["tenant", "regions"], "contains_any": ["eu"]},
            "denied",
        ),
        (
            {"tenant": {"regions": "eu"}},
            {"path": ["tenant", "regions"], "contains_any": ["eu"]},
            "denied",
        ),
    ],
)
def test_contains_any_leaf_decision_evaluates_type_strictly(
    client, claims, rule, expected
):
    assert _status_for(client, claims, rule) == expected


def test_contains_any_composes_with_all_any_not(client):
    claims = {
        "tiers": ["gold", "silver"],
        "score": 7,
        "tenant": {"regions": ["eu"]},
    }
    satisfied = {
        "all": [
            {"claim": "tiers", "contains_any": ["gold", "platinum"]},
            {"any": [
                {"claim": "score", "gte": 5},
                {"claim": "score", "lt": 0},
            ]},
            {"not": {"path": ["tenant", "blocked"], "contains_any": [True]}},
            {"path": ["tenant", "regions"], "contains_any": ["eu", "us"]},
        ]
    }
    assert _status_for(client, claims, satisfied) == "allowed"

    unsatisfied = {
        "all": [
            {"claim": "tiers", "contains_any": ["platinum"]},
            {"claim": "score", "lt": 5},
        ]
    }
    assert _status_for(client, claims, unsatisfied) == "denied"


def test_contains_any_under_not_allows_on_miss_and_non_array(client):
    rule = {"not": {"claim": "tiers", "contains_any": ["gold"]}}
    assert _status_for(client, {"tiers": ["bronze"]}, rule) == "allowed"
    assert _status_for(client, {}, rule) == "allowed"
    assert _status_for(client, {"tiers": ["gold", "bronze"]}, rule) == "denied"


def test_decision_deterministic_and_scope_isolated(client):
    claims = {"tiers": ["gold"]}
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(
        client, {"claim": "tiers", "contains_any": ["gold", "silver"]}
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


def test_decision_trace_and_evaluation_leak_no_locator_or_values(client):
    rule = {
        "all": [
            {"claim": "tiers", "contains_any": ["gold", "platinum"]},
            {"not": {"path": ["meta", "flags"], "contains_any": [True]}},
        ]
    }
    claims = {"tiers": ["bronze"], "meta": {"flags": [False]}}
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
    assert [n["outcome"] for n in nodes] == [False, False, True, False]
    rendered = json.dumps(nodes)
    for secret in ("tiers", "meta", "flags", "gold", "platinum", "bronze"):
        assert secret not in rendered

    trace = client.get(
        f"/v1/decisions/{decision_id}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert trace.status_code == 200
    assert trace.json()["policy_version"]["rule"] == rule


def test_decision_against_retired_contains_any_policy_is_409(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"a": [1]})
    policy = _create_policy(client, {"claim": "a", "contains_any": [1]})
    retired = client.post(
        f"/v1/policies/{policy['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200
    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 409
    assert response.json()["detail"] == "policy is retired"


def test_digest_mismatch_with_contains_any_policy_is_422(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"a": [1]})
    policy = _create_policy(client, {"claim": "a", "contains_any": [1]})
    response = _decide(
        client, evidence_id, created, evidence + " ", policy["policy_id"]
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "evidence digest mismatch"


# --- read-only trial evaluation ---------------------------------------------


def test_trial_evaluate_matches_root_and_is_read_only(client, app):
    rule = {
        "any": [
            {"claim": "tiers", "contains_any": ["gold"]},
            {"path": ["meta", "flags"], "contains_any": [True]},
        ]
    }
    policy = _create_policy(client, rule)

    claims = {"tiers": ["bronze"], "meta": {"flags": [True]}}
    response = _evaluate(client, policy["policy_id"], claims)
    assert response.status_code == 200, response.text
    data = response.json()
    assert set(data.keys()) == {
        "policy_id",
        "policy_version",
        "tenant_id",
        "workload_id",
        "allowed",
        "checked_at",
        "evaluation",
    }
    assert data["allowed"] is True
    assert data["allowed"] == data["evaluation"][0]["outcome"]
    assert [n["node_type"] for n in data["evaluation"]] == [
        "any",
        "leaf",
        "leaf",
    ]
    assert [n["outcome"] for n in data["evaluation"]] == [True, False, True]

    with app.state.session_factory() as session:
        # The trial wrote no business or audit state of any kind.
        assert session.query(Decision).count() == 0
        assert session.query(AuditEvent).count() == 0
        assert session.query(ProofLifecycleEvent).count() == 0


def test_trial_evaluate_denied_misses_are_false(client):
    policy = _create_policy(
        client, {"claim": "tiers", "contains_any": ["gold", "silver"]}
    )
    for claims, expected in (
        ({"tiers": ["bronze"]}, False),
        ({"tiers": []}, False),
        ({"tiers": "gold"}, False),
        ({}, False),
        ({"tiers": ["silver"]}, True),
    ):
        response = _evaluate(client, policy["policy_id"], claims)
        assert response.status_code == 200
        assert response.json()["allowed"] is expected


def test_trial_evaluate_retired_version_still_reads_immutable_rule(client):
    v1 = _create_policy(client, {"claim": "a", "contains_any": [1]})
    retired = client.post(
        f"/v1/policies/{v1['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200
    response = _evaluate(client, v1["policy_id"], {"a": [1, 2]})
    assert response.status_code == 200
    assert response.json()["allowed"] is True
    response = _evaluate(client, v1["policy_id"], {"a": [9]})
    assert response.json()["allowed"] is False


def test_trial_evaluate_scope_and_identifier_boundaries(client):
    policy = _create_policy(client, {"claim": "a", "contains_any": [1]})

    cross = _evaluate(
        client, policy["policy_id"], {"a": [1]}, tenant_id="tenant-b"
    )
    assert cross.status_code == 404
    bad_path = client.post(
        "/v1/policies/not-a-uuid/evaluate",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "claims": {"a": [1]},
        },
    )
    assert bad_path.status_code == 422


# --- compare ----------------------------------------------------------------


def test_compare_identifies_candidate_list_change(client):
    left_rule = {"claim": "tiers", "contains_any": ["gold", "silver"]}
    right_rule = {"claim": "tiers", "contains_any": ["gold", "platinum"]}
    left, right, response = _compare_pair(client, left_rule, right_rule)

    data = response.json()
    assert data["identical"] is False
    assert len(data["changes"]) == 1
    change = data["changes"][0]
    assert change["change"] == "changed"
    assert change["rule_path"] == []
    assert change["left"] == left_rule
    assert change["right"] == right_rule
    assert left["policy_id"] != right["policy_id"]


def test_compare_same_candidates_is_identical(client):
    rule = {"path": ["t", "regions"], "contains_any": ["eu", "us"]}
    _, _, response = _compare_pair(client, rule, rule)
    assert response.json()["identical"] is True
    assert response.json()["changes"] == []


def test_compare_detects_contains_any_inside_compound_tree(client):
    left_rule = {
        "all": [
            {"claim": "tiers", "contains_any": ["gold"]},
            {"claim": "score", "gte": 5},
        ]
    }
    right_rule = {
        "all": [
            {"claim": "tiers", "contains_any": ["platinum"]},
            {"claim": "score", "gte": 5},
        ]
    }
    _, _, response = _compare_pair(client, left_rule, right_rule)
    data = response.json()
    assert data["identical"] is False
    assert len(data["changes"]) == 1
    assert data["changes"][0]["rule_path"] == [0]
    assert data["changes"][0]["change"] == "changed"


def _compare_pair(client, left_rule, right_rule):
    left = _create_policy(client, left_rule, name="left")
    right = _create_policy(client, right_rule, name="right")
    response = client.get(
        "/v1/policies/compare",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "left_policy_id": left["policy_id"],
            "right_policy_id": right["policy_id"],
        },
    )
    assert response.status_code == 200, response.text
    return left, right, response


# --- backward compatibility -------------------------------------------------


def test_contains_semantics_unchanged_next_to_contains_any(client):
    # contains still takes one scalar and still matches element-wise.
    assert (
        _status_for(client, {"tiers": ["gold"]},
                    {"claim": "tiers", "contains": "gold"})
        == "allowed"
    )
    # An array value is not searched by in; and in's own validation and
    # the 32-item bound are untouched.
    assert _policy(client, {"claim": "a", "in": [1, 1.0]}).status_code == 422
    assert _policy(
        client, {"claim": "a", "in": list(range(33))}
    ).status_code == 422


def test_old_persisted_rules_evaluate_unchanged(client):
    # A rule persisted without any contains_any key keeps its verdict.
    rule = {"all": [
        {"claim": "tier", "in": ["gold", "silver"]},
        {"path": ["x"], "exists": True},
    ]}
    assert _status_for(client, {"tier": "gold", "x": 1}, rule) == "allowed"
    assert _status_for(client, {"tier": "bronze", "x": 1}, rule) == "denied"
