"""Tests for the ``containsAll`` set-membership rule leaf.

Covers POST /v1/policies validation (422 with no version allocated),
canonical persistence and snapshot rendering, and decision evaluation of
the new leaf form — the located value must be a JSON array containing
every candidate under the same type-strict scalar equality ``in`` uses.
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


# --- creation: the new leaf is accepted and round-trips verbatim ---------


@pytest.mark.parametrize(
    "rule",
    [
        {"claim": "labels", "containsAll": ["measured", "boot"]},
        {"claim": "levels", "containsAll": [1, 2, 3]},
        {"claim": "flags", "containsAll": [True, False]},
        {"claim": "notes", "containsAll": [None]},
        {"claim": "mix", "containsAll": ["a", 1, 1.5, True, None]},
        {"path": ["meta", "labels"], "containsAll": ["eu", "tier-1"]},
        {"path": ["caps"], "containsAll": ["release"]},
        # The widest candidate set still fits.
        {"claim": "n", "containsAll": list(range(32))},
        # A single candidate is the smallest legal set.
        {"claim": "labels", "containsAll": ["only"]},
        # New leaves compose with the existing forms.
        {
            "all": [
                {"claim": "labels", "containsAll": ["measured"]},
                {"any": [
                    {"claim": "level", "gte": 5},
                    {"not": {"path": ["meta", "blocked"], "exists": True}},
                ]},
                {"not": {"claim": "tier", "in": ["bronze"]}},
            ]
        },
    ],
)
def test_create_policy_accepts_contains_all_leaves(client, rule):
    response = _policy(client, rule)

    assert response.status_code == 201
    assert response.json()["rule"] == rule


def test_contains_all_rule_persists_canonically_and_renders_identically(client, app):
    rule = {
        "all": [
            {"claim": "labels", "containsAll": ["gold", "silver"]},
            {"path": ["meta", "tags"], "containsAll": [1, 2]},
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


# --- creation: invalid leaves are 422 and allocate no version ------------


@pytest.mark.parametrize(
    "rule",
    [
        # Candidate set: empty, oversized, not a list, non-scalar members.
        {"claim": "a", "containsAll": []},
        {"claim": "a", "containsAll": list(range(33))},
        {"claim": "a", "containsAll": "abc"},
        {"claim": "a", "containsAll": "gold"},
        {"claim": "a", "containsAll": 1},
        {"claim": "a", "containsAll": [["a"]]},
        {"claim": "a", "containsAll": [{"x": 1}]},
        {"claim": "a", "containsAll": [["a", "b"], "c"]},
        {"claim": "a", "containsAll": [None, [1]]},
        # Repeated candidates under type-strict equality.
        {"claim": "a", "containsAll": [1, 1]},
        {"claim": "a", "containsAll": [1, 1.0]},
        {"claim": "a", "containsAll": ["x", "x"]},
        {"claim": "a", "containsAll": [None, None]},
        {"claim": "a", "containsAll": [True, True]},
        {"claim": "a", "containsAll": [1.0, 1]},
        # A leaf still carries exactly one comparison.
        {"claim": "a", "containsAll": [1], "equals": 1},
        {"claim": "a", "containsAll": [1], "in": [1]},
        {"claim": "a", "containsAll": [1], "exists": True},
        {"claim": "a", "containsAll": [1], "lt": 2},
        {"claim": "a", "containsAll": [1], "extra": 2},
        {"claim": "a", "containsAll": [1], "all": []},
        # The comparison never appears without a locator.
        {"containsAll": [1]},
        {"containsAll": []},
        # Path leaves share the same validation.
        {"path": ["a"], "containsAll": []},
        {"path": ["a"], "containsAll": list(range(33))},
        {"path": ["a"], "containsAll": [1, 1.0]},
        {"path": ["a"], "containsAll": [{"x": 1}]},
        {"path": ["a"], "containsAll": [1], "equals": 1},
        {"path": [], "containsAll": [1]},
        {"path": "tenant", "containsAll": [1]},
    ],
)
def test_create_policy_rejects_invalid_contains_all_leaves(client, app, rule):
    response = _policy(client, rule)

    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == 0


def test_rejected_contains_all_allocates_no_version(client):
    assert _policy(client, {"claim": "a", "containsAll": []}).status_code == 422
    assert _policy(
        client, {"claim": "a", "containsAll": [1, 1.0]}
    ).status_code == 422

    created = _create_policy(client, {"claim": "a", "containsAll": [1]})
    assert created["version"] == 1


def test_defensive_bounds_still_apply_to_contains_all(client):
    deep = rule = {"claim": "a", "containsAll": [1]}
    for _ in range(33):
        rule = {"not": rule}
    assert _policy(client, rule).status_code == 422

    wide = {"all": [{"claim": "a", "containsAll": [1]}] * 257}
    assert _policy(client, wide).status_code == 422
    assert deep is not None


# --- evaluation: array containment ---------------------------------------


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        # Every candidate present allows; order and extra members do not matter.
        (
            {"labels": ["measured", "boot", "debug"]},
            {"claim": "labels", "containsAll": ["boot", "measured"]},
            "allowed",
        ),
        (
            {"labels": ["a", "b", "c"]},
            {"claim": "labels", "containsAll": ["c", "a", "b"]},
            "allowed",
        ),
        # Missing any one candidate denies.
        (
            {"labels": ["measured", "debug"]},
            {"claim": "labels", "containsAll": ["measured", "boot"]},
            "denied",
        ),
        # An empty actual array contains nothing (a candidate set is never empty).
        (
            {"labels": []},
            {"claim": "labels", "containsAll": ["measured"]},
            "denied",
        ),
        # Numbers compare by value across integer/decimal widths.
        ({"n": [1, 2, 3]}, {"claim": "n", "containsAll": [1.0, 2.0]}, "allowed"),
        ({"n": [1.0]}, {"claim": "n", "containsAll": [1]}, "allowed"),
        ({"n": [1, 2]}, {"claim": "n", "containsAll": [2, 3]}, "denied"),
        ({"n": [1.5, 2.5]}, {"claim": "n", "containsAll": [1.5]}, "allowed"),
        # Booleans are not numbers, on either side.
        ({"f": [True]}, {"claim": "f", "containsAll": [1]}, "denied"),
        ({"f": [1]}, {"claim": "f", "containsAll": [True]}, "denied"),
        ({"f": [False]}, {"claim": "f", "containsAll": [0]}, "denied"),
        ({"f": [True, False]}, {"claim": "f", "containsAll": [True, False]}, "allowed"),
        # Strings never match numbers; null matches only an explicit null.
        ({"s": ["2"]}, {"claim": "s", "containsAll": [2]}, "denied"),
        ({"s": [2]}, {"claim": "s", "containsAll": ["2"]}, "denied"),
        ({"x": [None, "y"]}, {"claim": "x", "containsAll": [None]}, "allowed"),
        ({"x": [None]}, {"claim": "x", "containsAll": [None, "y"]}, "denied"),
        ({"x": [""]}, {"claim": "x", "containsAll": [None]}, "denied"),
        ({"x": [0, False]}, {"claim": "x", "containsAll": [None]}, "denied"),
        # A single array member satisfies at most one type-strict candidate.
        ({"x": [1]}, {"claim": "x", "containsAll": [1, True]}, "denied"),
        # Repeated actual members are harmless.
        ({"x": ["a", "a", "b"]}, {"claim": "x", "containsAll": ["a", "b"]}, "allowed"),
        # Object/array members never match a scalar candidate.
        ({"x": [["a"]]}, {"claim": "x", "containsAll": ["a"]}, "denied"),
        ({"x": [{"a": 1}]}, {"claim": "x", "containsAll": [1]}, "denied"),
        ({"x": [["a", "b"], "a"]}, {"claim": "x", "containsAll": ["a", "b"]}, "denied"),
        # Missing, null or non-array located values all fail, never raise.
        ({}, {"claim": "labels", "containsAll": ["measured"]}, "denied"),
        ({"labels": None}, {"claim": "labels", "containsAll": ["measured"]}, "denied"),
        ({"labels": "measured"}, {"claim": "labels", "containsAll": ["measured"]}, "denied"),
        ({"labels": 5}, {"claim": "labels", "containsAll": [5]}, "denied"),
        ({"labels": True}, {"claim": "labels", "containsAll": [True]}, "denied"),
        ({"labels": {"a": 1}}, {"claim": "labels", "containsAll": ["a"]}, "denied"),
        # A missing claim is not equal to a candidate set containing null.
        ({}, {"claim": "labels", "containsAll": [None]}, "denied"),
    ],
)
def test_contains_all_leaf_evaluates_type_strictly(client, claims, rule, expected):
    assert _status_for(client, claims, rule) == expected


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        # Path descent ends at a terminal array.
        (
            {"meta": {"labels": ["eu", "tier-1"]}},
            {"path": ["meta", "labels"], "containsAll": ["eu", "tier-1"]},
            "allowed",
        ),
        (
            {"meta": {"labels": ["eu"]}},
            {"path": ["meta", "labels"], "containsAll": ["eu", "tier-1"]},
            "denied",
        ),
        # A single-segment path reads the top-level claim as an array.
        (
            {"caps": ["release", "rewrap"]},
            {"path": ["caps"], "containsAll": ["release"]},
            "allowed",
        ),
        # Missing at any level is a plain miss.
        ({}, {"path": ["meta", "labels"], "containsAll": ["eu"]}, "denied"),
        ({"meta": {}}, {"path": ["meta", "labels"], "containsAll": ["eu"]}, "denied"),
        # An array mid-path is never indexed through; no array subscript syntax.
        (
            {"meta": [{"labels": ["eu"]}]},
            {"path": ["meta", "labels"], "containsAll": ["eu"]},
            "denied",
        ),
        # A scalar/null terminal value is not an array.
        (
            {"meta": {"labels": None}},
            {"path": ["meta", "labels"], "containsAll": [None]},
            "denied",
        ),
        (
            {"meta": {"labels": "eu"}},
            {"path": ["meta", "labels"], "containsAll": ["eu"]},
            "denied",
        ),
        (
            {"meta": None},
            {"path": ["meta", "labels"], "containsAll": ["eu"]},
            "denied",
        ),
    ],
)
def test_contains_all_path_leaf_walks_objects_only(client, claims, rule, expected):
    assert _status_for(client, claims, rule) == expected


# --- composition, traces and decision semantics --------------------------


def test_contains_all_composes_with_all_any_not(client):
    claims = {
        "labels": ["measured", "boot"],
        "level": 7,
        "meta": {"region": "eu"},
    }
    satisfied = {
        "all": [
            {"claim": "labels", "containsAll": ["boot", "measured"]},
            {"any": [
                {"claim": "level", "gte": 5},
                {"claim": "level", "lt": 0},
            ]},
            {"not": {"path": ["meta", "blocked"], "exists": True}},
        ]
    }
    assert _status_for(client, claims, satisfied) == "allowed"

    unsatisfied = {
        "all": [
            {"claim": "labels", "containsAll": ["measured"]},
            {"claim": "labels", "containsAll": ["debug"]},
        ]
    }
    assert _status_for(client, claims, unsatisfied) == "denied"


def test_trace_renders_the_contains_all_rule_snapshot(client):
    rule = {
        "all": [
            {"claim": "labels", "containsAll": ["measured", "boot"]},
            {"path": ["meta", "tags"], "containsAll": [1]},
        ]
    }
    claims = {"labels": ["boot", "measured"], "meta": {"tags": [1, 2]}}
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
    body = trace.json()
    assert body["policy_version"]["rule"] == rule
    assert body["policy_version"]["version"] == 1

    evaluation = client.get(
        f"/v1/decisions/{decided.json()['decision_id']}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert evaluation.status_code == 200
    nodes = evaluation.json()["nodes"]
    # One all node with two leaf children, every outcome agreeing.
    assert [node["node_type"] for node in nodes] == ["all", "leaf", "leaf"]
    assert all(node["outcome"] for node in nodes)
    # The trace never leaks the comparison key, candidates or claim values.
    rendered = json.dumps(nodes)
    assert "containsAll" not in rendered
    assert "measured" not in rendered
    assert "labels" not in rendered


def test_decision_isolation_determinism_and_first_wins_with_contains_all(client):
    claims = {"labels": ["measured", "boot"]}
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    rule = {"claim": "labels", "containsAll": ["measured", "boot"]}
    policy = _create_policy(client, rule)

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

    foreign = _create_policy(
        client,
        {"claim": "labels", "containsAll": ["measured"]},
        tenant_id="tenant-b",
    )
    assert (
        _decide(client, evidence_id, created, evidence, foreign["policy_id"])
        .status_code
        == 404
    )


def test_first_decision_persists_even_when_contains_all_would_flip(client):
    # The stored first decision is returned unchanged on repeat, even
    # though the same rule denies for evidence lacking the candidates —
    # decisions attach to one concrete evidence, and this merely confirms
    # the existing first-wins path is untouched for the new leaf.
    created, evidence, evidence_id = _receive_and_verify(
        client, {"labels": ["measured"]}
    )
    policy = _create_policy(
        client, {"claim": "labels", "containsAll": ["measured"]}
    )
    first = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert first.status_code == 200
    assert first.json()["status"] == "allowed"
    second = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert second.json() == first.json()


def test_existing_comparisons_unchanged_next_to_contains_all(client):
    # Neighboring leaves keep their exact semantics with the new key present.
    assert (
        _status_for(
            client, {"m": ["a"]}, {"claim": "m", "containsAll": ["a"]}
        )
        == "allowed"
    )
    # equals still needs a scalar actual; a scalar actual still fails containsAll.
    assert (
        _status_for(client, {"m": "a"}, {"claim": "m", "equals": "a"})
        == "allowed"
    )
    assert (
        _status_for(client, {"m": "a"}, {"claim": "m", "containsAll": ["a"]})
        == "denied"
    )
    # in still compares the actual scalar against the candidate set.
    assert (
        _status_for(client, {"m": "a"}, {"claim": "m", "in": ["a", "b"]})
        == "allowed"
    )
    assert (
        _status_for(
            client, {"m": True}, {"claim": "m", "equals": 1}
        )
        == "denied"
    )


@pytest.mark.parametrize(
    "rule_json",
    [
        # Non-finite numbers are not valid JSON scalars: the server-side
        # parser accepts the literals, and validation must reject them.
        '{"claim": "a", "containsAll": [NaN]}',
        '{"claim": "a", "containsAll": [Infinity]}',
        '{"claim": "a", "containsAll": [-Infinity]}',
        '{"claim": "a", "containsAll": [1, NaN]}',
    ],
)
def test_non_finite_contains_all_candidates_never_persist(client, app, rule_json):
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
