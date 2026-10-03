"""Tests for the ``contains`` rule leaf (array membership of a scalar).

A ``contains`` leaf locates a value with a top-level ``claim`` name or an
object ``path`` exactly like the other leaf forms; its expected value must
be one JSON scalar, and evaluation is true only when the located value is a
JSON array with at least one element equal to that scalar under the
type-strict equality used by ``equals``. Covers POST /v1/policies
validation (422 with no version allocated), canonical persistence and
snapshot rendering, decision evaluation against verified evidence claims,
explanation shape, and composition with the other rule forms.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Policy
from proof_release.policies import evaluate_rule, explain_rule

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    return create_app(f"sqlite:///{tmp_path}/contains.db")


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


# --- creation: accepted forms round-trip verbatim -------------------------


@pytest.mark.parametrize(
    "rule",
    [
        {"claim": "tags", "contains": "gold"},
        {"claim": "levels", "contains": 1},
        {"claim": "levels", "contains": 1.5},
        {"claim": "flags", "contains": True},
        {"claim": "flags", "contains": False},
        {"claim": "notes", "contains": None},
        {"path": ["tenant", "tags"], "contains": "eu"},
        {"path": ["quota", "owners"], "contains": 0},
        # Composition with the compound forms and the other comparisons.
        {
            "all": [
                {"claim": "tags", "contains": "gold"},
                {"any": [
                    {"claim": "score", "gte": 5},
                    {"not": {"claim": "tags", "contains": "blocked"}},
                ]},
            ]
        },
    ],
)
def test_create_policy_accepts_contains_leaves(client, rule):
    response = _policy(client, rule)

    assert response.status_code == 201
    assert response.json()["rule"] == rule


def test_contains_rule_persists_canonically_and_renders_identically(client, app):
    rule = {
        "all": [
            {"claim": "tags", "contains": "gold"},
            {"path": ["tenant", "tags"], "contains": "eu"},
        ]
    }
    created = _create_policy(client, rule)

    with app.state.session_factory() as session:
        row = session.get(Policy, created["policy_id"])
        assert row.rule_json == json.dumps(rule, sort_keys=True, separators=(",", ":"))

    listed = client.get(
        "/v1/policies", params={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert listed.status_code == 200
    assert listed.json()["policies"][0]["rule"] == rule


def test_trace_renders_the_contains_rule_snapshot(client):
    rule = {"claim": "tags", "contains": "gold"}
    claims = {"tags": ["bronze", "gold"]}
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(client, rule)

    decided = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert decided.status_code == 200
    assert decided.json()["status"] == "allowed"

    trace = client.get(
        f"/v1/decisions/{decided.json()['decision_id']}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert trace.status_code == 200
    assert trace.json()["policy_version"]["rule"] == rule
    assert trace.json()["policy_version"]["version"] == 1


# --- creation: invalid leaves are 422 and allocate no version -------------


@pytest.mark.parametrize(
    "rule",
    [
        # The expected member must be one scalar.
        {"claim": "a", "contains": ["x"]},
        {"claim": "a", "contains": {"x": 1}},
        {"claim": "a", "contains": [1]},
        {"claim": "a", "contains": []},
        # Exactly one comparison key.
        {"claim": "a", "contains": 1, "equals": 1},
        {"claim": "a", "contains": 1, "in": [1]},
        {"claim": "a", "contains": "x", "extra": 2},
        # Exactly one locator.
        {"claim": "a", "path": ["a"], "contains": 1},
        {"contains": 1},
        {"path": [], "contains": 1},
        {"path": "a", "contains": 1},
        # Unknown nodes stay unknown.
        {"contains": 1, "all": []},
        {"path": ["a"], "contains": ["x"]},
    ],
)
def test_create_policy_rejects_invalid_contains_leaves(client, app, rule):
    response = _policy(client, rule)

    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == 0


def test_rejected_contains_allocates_no_version_and_does_not_block_name(client):
    assert _policy(client, {"claim": "a", "contains": ["x"]}).status_code == 422
    assert _policy(client, {"claim": "a", "path": ["a"], "contains": 1}).status_code == 422

    # The rejected submissions allocated nothing: the first valid policy of
    # this name is still version 1.
    created = _create_policy(client, {"claim": "a", "contains": 1})
    assert created["version"] == 1

    assert _policy(client, {"claim": "a", "contains": ["y"]}, name="release").status_code == 422
    second = _create_policy(client, {"claim": "a", "contains": 2}, name="release")
    assert second["version"] == 2


@pytest.mark.parametrize(
    "rule_json",
    [
        # Non-finite numbers are not valid JSON scalars.
        '{"claim": "a", "contains": NaN}',
        '{"claim": "a", "contains": Infinity}',
        '{"claim": "a", "contains": -Infinity}',
    ],
)
def test_non_finite_contains_value_never_persists(client, app, rule_json):
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


def test_defensive_bounds_still_apply_to_contains_leaves(client):
    deep = rule = {"claim": "a", "contains": 1}
    for _ in range(33):
        rule = {"not": rule}
    assert _policy(client, rule).status_code == 422

    wide = {"all": [{"claim": "a", "contains": 1}] * 257}
    assert _policy(client, wide).status_code == 422

    assert (
        _policy(
            client, {"path": ["s" * 9 for _ in range(9)], "contains": 1}
        ).status_code
        == 422
    )
    assert deep is not None


# --- evaluation: array membership under equals scalar semantics -----------


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        # Each scalar type can be the expected member.
        ({"tags": ["gold", "silver"]}, {"claim": "tags", "contains": "gold"}, "allowed"),
        ({"levels": [1, 2, 3]}, {"claim": "levels", "contains": 2}, "allowed"),
        ({"flags": [True, False]}, {"claim": "flags", "contains": False}, "allowed"),
        ({"notes": [None, "x"]}, {"claim": "notes", "contains": None}, "allowed"),
        ({"vs": [1.5]}, {"claim": "vs", "contains": 1.5}, "allowed"),
        # Members present but not matching: false.
        ({"tags": ["bronze"]}, {"claim": "tags", "contains": "gold"}, "denied"),
        ({"levels": [2, 3]}, {"claim": "levels", "contains": 1}, "denied"),
        # Type-strict: true is not 1, string digits are not numbers, null
        # only equals null, false is not 0.
        ({"v": [True]}, {"claim": "v", "contains": 1}, "denied"),
        ({"v": [1]}, {"claim": "v", "contains": True}, "denied"),
        ({"v": ["1"]}, {"claim": "v", "contains": 1}, "denied"),
        ({"v": [1]}, {"claim": "v", "contains": "1"}, "denied"),
        ({"v": [0]}, {"claim": "v", "contains": False}, "denied"),
        ({"v": [False]}, {"claim": "v", "contains": 0}, "denied"),
        ({"v": ["", 0, False]}, {"claim": "v", "contains": None}, "denied"),
        ({"v": [None]}, {"claim": "v", "contains": False}, "denied"),
        # Numbers of different width but equal value match.
        ({"v": [1]}, {"claim": "v", "contains": 1.0}, "allowed"),
        ({"v": [1.0]}, {"claim": "v", "contains": 1}, "allowed"),
        ({"v": [2, 3.0]}, {"claim": "v", "contains": 3}, "allowed"),
        # Missing, non-array, and empty arrays are all false.
        ({}, {"claim": "tags", "contains": "gold"}, "denied"),
        ({"other": []}, {"claim": "tags", "contains": "gold"}, "denied"),
        ({"tags": []}, {"claim": "tags", "contains": "gold"}, "denied"),
        ({"tags": "gold"}, {"claim": "tags", "contains": "gold"}, "denied"),
        ({"tags": 1}, {"claim": "tags", "contains": 1}, "denied"),
        ({"tags": None}, {"claim": "tags", "contains": None}, "denied"),
        ({"tags": {"x": 1}}, {"claim": "tags", "contains": 1}, "denied"),
        ({"tags": True}, {"claim": "tags", "contains": True}, "denied"),
        # Arrays are not flattened or compared element-wise: a nested array
        # element is not the scalar, and an object element never equals one.
        ({"v": [[1]]}, {"claim": "v", "contains": 1}, "denied"),
        ({"v": [{"x": 1}]}, {"claim": "v", "contains": 1}, "denied"),
        # A matching element anywhere in the array satisfies.
        ({"v": ["a", "b", "c", "b"]}, {"claim": "v", "contains": "c"}, "allowed"),
        # Path-located arrays.
        (
            {"tenant": {"tags": ["eu", "us"]}},
            {"path": ["tenant", "tags"], "contains": "eu"},
            "allowed",
        ),
        (
            {"tenant": {"tags": ["ap"]}},
            {"path": ["tenant", "tags"], "contains": "eu"},
            "denied",
        ),
        (
            {"tenant": {"tags": []}},
            {"path": ["tenant", "tags"], "contains": None},
            "denied",
        ),
        (
            {"tenant": {}},
            {"path": ["tenant", "tags"], "contains": "eu"},
            "denied",
        ),
        (
            {"tenant": {"tags": "eu"}},
            {"path": ["tenant", "tags"], "contains": "eu"},
            "denied",
        ),
        (
            {"tenant": [{"tags": ["eu"]}]},
            {"path": ["tenant", "tags"], "contains": "eu"},
            "denied",
        ),
    ],
)
def test_contains_leaf_evaluates_array_membership(client, claims, rule, expected):
    assert _status_for(client, claims, rule) == expected


def test_contains_against_actual_null_array_with_null_member():
    # An explicit null *element* matches an expected null, while a null
    # terminal value itself is not an array.
    assert evaluate_rule({"claim": "v", "contains": None}, {"v": [None]}) is True
    assert evaluate_rule({"claim": "v", "contains": None}, {"v": None}) is False


# --- composition ----------------------------------------------------------


def test_contains_composes_with_all_any_not_and_other_comparisons(client):
    claims = {
        "tier": "gold",
        "score": 7,
        "tags": ["release", "eu"],
        "tenant": {"regions": ["eu", "us"]},
    }
    satisfied = {
        "all": [
            {"claim": "tier", "equals": "gold"},
            {"claim": "tags", "contains": "release"},
            {"any": [
                {"claim": "score", "gte": 5},
                {"claim": "tags", "contains": "override"},
            ]},
            {"not": {"path": ["tenant", "regions"], "contains": "blocked"}},
        ]
    }
    assert _status_for(client, claims, satisfied) == "allowed"

    unsatisfied = {
        "all": [
            {"claim": "tags", "contains": "release"},
            {"claim": "tags", "contains": "missing"},
        ]
    }
    assert _status_for(client, claims, unsatisfied) == "denied"


def test_contains_denied_decision_is_denied_and_idempotent(client):
    claims = {"tags": ["a"]}
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(client, {"claim": "tags", "contains": "b"})

    first = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert first.status_code == 200
    assert first.json()["status"] == "denied"

    second = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert second.status_code == 200
    assert second.json() == first.json()


def test_contains_explanation_is_preorder_leaves_with_positions_only():
    rule = {"all": [
        {"claim": "tags", "contains": "x"},
        {"not": {"path": ["t", "tags"], "contains": "y"}},
    ]}

    hit = explain_rule(rule, {"tags": ["x"], "t": {"tags": ["z"]}})
    assert [n["node_type"] for n in hit] == ["all", "leaf", "not", "leaf"]
    assert [n["rule_path"] for n in hit] == [[], [0], [1], [1, 0]]
    assert [n["node_index"] for n in hit] == [0, 1, 2, 3]
    assert [n["outcome"] for n in hit] == [True, True, True, False]
    # Only the documented structural fields are ever emitted.
    for node in hit:
        assert set(node) == {"node_index", "rule_path", "node_type", "outcome"}

    miss = explain_rule(rule, {"tags": [], "t": {"tags": ["y"]}})
    assert [n["outcome"] for n in miss] == [False, False, False, True]
    assert miss[0]["outcome"] == evaluate_rule(rule, {"tags": [], "t": {"tags": ["y"]}})


# --- existing boundaries and semantics are unchanged ----------------------


def test_in_leaf_still_does_not_match_array_terminal(client):
    # The new operator must not change in's scalar-candidate semantics: an
    # array terminal value never equals a scalar candidate.
    assert (
        _status_for(
            client,
            {"tags": ["eu"]},
            {"claim": "tags", "in": ["eu", "us"]},
        )
        == "denied"
    )
    assert (
        _status_for(
            client,
            {"tags": "eu"},
            {"claim": "tags", "contains": "eu"},
        )
        == "denied"
    )


def test_contains_decision_error_boundaries(client):
    # A contains policy goes through the same gates as every other leaf.
    created, evidence, evidence_id = _receive_and_verify(client, {"tags": ["x"]})
    policy = _create_policy(client, {"claim": "tags", "contains": "x"})

    # A cross-scope policy against in-scope verified evidence: 404 policy
    # not found.
    foreign = _create_policy(
        client, {"claim": "tags", "contains": "x"}, tenant_id="tenant-b"
    )
    cross_policy = _decide(
        client, evidence_id, created, evidence, foreign["policy_id"]
    )
    assert cross_policy.status_code == 404
    assert cross_policy.json()["detail"] == "policy not found"

    # A request scoped to another tenant never even reaches the policy:
    # the evidence is out of scope.
    cross_scope = _decide(
        client,
        evidence_id,
        created,
        evidence,
        policy["policy_id"],
        tenant_id="tenant-b",
    )
    assert cross_scope.status_code == 404
    assert cross_scope.json()["detail"] == "evidence not found"

    # Tampered evidence bytes: digest mismatch before any evaluation.
    tampered = _evidence(created["nonce"], {"tags": ["y"]})
    bad_digest = _decide(
        client, evidence_id, created, tampered, policy["policy_id"]
    )
    assert bad_digest.status_code == 422
    assert bad_digest.json()["detail"] == "evidence digest mismatch"
