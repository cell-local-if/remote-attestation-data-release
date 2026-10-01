"""Tests for the extended rule leaf comparisons (in/lt/lte/gt/gte/exists).

Covers POST /v1/policies validation (422 with no version allocated),
canonical persistence and snapshot rendering, and decision evaluation of
the new leaf forms against verified evidence claims.
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
    return create_app(f"sqlite:///{tmp_path}/comparisons.db")


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


# --- creation: new leaf forms are accepted and round-trip verbatim -------


@pytest.mark.parametrize(
    "rule",
    [
        {"claim": "tier", "in": ["gold", "silver"]},
        {"claim": "level", "in": [1, 2, 3]},
        {"claim": "flag", "in": [True, False]},
        {"claim": "note", "in": [None]},
        {"claim": "mix", "in": ["a", 1, 1.5, True, None]},
        {"claim": "score", "lt": 10},
        {"claim": "score", "lte": 10.5},
        {"claim": "score", "gt": 0},
        {"claim": "score", "gte": -2},
        {"claim": "region", "exists": True},
        {"path": ["tenant", "region"], "in": ["eu", "us"]},
        {"path": ["quota", "used"], "lt": 100},
        {"path": ["quota", "used"], "lte": 100},
        {"path": ["quota", "used"], "gt": 0},
        {"path": ["quota", "used"], "gte": 0},
        {"path": ["tenant"], "exists": True},
        # The widest in set still fits.
        {"claim": "n", "in": list(range(32))},
        # New leaves compose with the compound forms.
        {
            "all": [
                {"claim": "tier", "in": ["gold"]},
                {"any": [
                    {"claim": "score", "gte": 5},
                    {"not": {"path": ["quota", "used"], "exists": True}},
                ]},
            ]
        },
    ],
)
def test_create_policy_accepts_new_leaf_forms(client, rule):
    response = _policy(client, rule)

    assert response.status_code == 201
    assert response.json()["rule"] == rule


def test_new_leaf_rule_persists_canonically_and_renders_identically(client, app):
    rule = {
        "all": [
            {"claim": "tier", "in": ["gold", "silver"]},
            {"path": ["quota", "used"], "lt": 100},
            {"claim": "region", "exists": True},
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
        # in: empty, oversized, non-scalar or repeating candidates.
        {"claim": "a", "in": []},
        {"claim": "a", "in": list(range(33))},
        {"claim": "a", "in": "abc"},
        {"claim": "a", "in": [[1]]},
        {"claim": "a", "in": [{"x": 1}]},
        {"claim": "a", "in": [1, 1]},
        {"claim": "a", "in": [1, 1.0]},
        {"claim": "a", "in": ["x", "x"]},
        {"claim": "a", "in": [None, None]},
        {"claim": "a", "in": [True, True]},
        # Ordinal bounds must be finite JSON numbers, never booleans.
        {"claim": "a", "lt": True},
        {"claim": "a", "lte": False},
        {"claim": "a", "gt": "5"},
        {"claim": "a", "gte": None},
        {"claim": "a", "lt": [1]},
        # exists must be exactly true.
        {"claim": "a", "exists": False},
        {"claim": "a", "exists": None},
        {"claim": "a", "exists": 1},
        {"claim": "a", "exists": "true"},
        # Missing, extra, or unknown comparison keys.
        {"claim": "a"},
        {"claim": "a", "equals": 1, "in": [1]},
        {"claim": "a", "in": [1], "extra": 2},
        {"claim": "a", "foo": 1},
        {"claim": "a", "lt": 1, "gt": 2},
        {"path": ["a"], "exists": False},
        {"path": ["a"], "in": []},
        {"path": ["a"], "lt": "x"},
        {"path": ["a"], "in": [1], "equals": 1},
        # Unknown nodes stay unknown.
        {"claim": "a", "in": [1], "all": []},
        {"exists": True},
        {"in": [1]},
    ],
)
def test_create_policy_rejects_invalid_new_leaves(client, app, rule):
    response = _policy(client, rule)

    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == 0


def test_rejected_new_leaf_allocates_no_version(client):
    assert _policy(client, {"claim": "a", "in": []}).status_code == 422
    assert _policy(client, {"claim": "a", "exists": False}).status_code == 422

    created = _create_policy(client, {"claim": "a", "in": [1]})
    assert created["version"] == 1


def test_defensive_bounds_still_apply_to_new_leaves(client):
    deep = rule = {"claim": "a", "in": [1]}
    for _ in range(33):
        rule = {"not": rule}
    assert _policy(client, rule).status_code == 422

    wide = {"all": [{"claim": "a", "gte": 1}] * 257}
    assert _policy(client, wide).status_code == 422
    assert deep is not None


# --- evaluation: in -------------------------------------------------------


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        # Scalar membership, type-strict.
        ({"tier": "gold"}, {"claim": "tier", "in": ["gold", "silver"]}, "allowed"),
        ({"tier": "bronze"}, {"claim": "tier", "in": ["gold", "silver"]}, "denied"),
        ({"level": 2}, {"claim": "level", "in": [1, 2, 3]}, "allowed"),
        ({"level": 2.0}, {"claim": "level", "in": [1, 2, 3]}, "allowed"),
        ({"flag": True}, {"claim": "flag", "in": [True, False]}, "allowed"),
        ({"note": None}, {"claim": "note", "in": [None, "x"]}, "allowed"),
        # Type-strict misses: booleans are not numbers, strings are not
        # numbers, null only matches null.
        ({"flag": True}, {"claim": "flag", "in": [1]}, "denied"),
        ({"level": 1}, {"claim": "level", "in": [True]}, "denied"),
        ({"level": "2"}, {"claim": "level", "in": [2]}, "denied"),
        ({"level": 2}, {"claim": "level", "in": ["2"]}, "denied"),
        ({"note": None}, {"claim": "note", "in": ["", 0, False]}, "denied"),
        ({"note": "x"}, {"claim": "note", "in": [None]}, "denied"),
        # A missing claim never hits the set.
        ({}, {"claim": "tier", "in": ["gold", None]}, "denied"),
        # Path-located membership.
        (
            {"tenant": {"region": "eu"}},
            {"path": ["tenant", "region"], "in": ["eu", "us"]},
            "allowed",
        ),
        (
            {"tenant": {"region": "ap"}},
            {"path": ["tenant", "region"], "in": ["eu", "us"]},
            "denied",
        ),
        (
            {"tenant": {"region": None}},
            {"path": ["tenant", "region"], "in": [None]},
            "allowed",
        ),
        # A missing path never hits the set.
        ({"tenant": {}}, {"path": ["tenant", "region"], "in": [None]}, "denied"),
        # An array terminal value is not a scalar and never matches.
        (
            {"tenant": {"region": ["eu"]}},
            {"path": ["tenant", "region"], "in": ["eu"]},
            "denied",
        ),
    ],
)
def test_in_leaf_evaluates_type_strictly(client, claims, rule, expected):
    assert _status_for(client, claims, rule) == expected


# --- evaluation: lt/lte/gt/gte --------------------------------------------


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        ({"score": 5}, {"claim": "score", "lt": 6}, "allowed"),
        ({"score": 5}, {"claim": "score", "lt": 5}, "denied"),
        ({"score": 5}, {"claim": "score", "lte": 5}, "allowed"),
        ({"score": 5}, {"claim": "score", "lte": 4}, "denied"),
        ({"score": 5}, {"claim": "score", "gt": 4}, "allowed"),
        ({"score": 5}, {"claim": "score", "gt": 5}, "denied"),
        ({"score": 5}, {"claim": "score", "gte": 5}, "allowed"),
        ({"score": 5}, {"claim": "score", "gte": 6}, "denied"),
        # Integers and decimals compare by value across widths.
        ({"score": 5}, {"claim": "score", "lt": 5.5}, "allowed"),
        ({"score": 2.5}, {"claim": "score", "gte": 2}, "allowed"),
        ({"score": -1}, {"claim": "score", "lt": 0}, "allowed"),
        # Booleans are not numbers, on either side of the comparison.
        ({"score": True}, {"claim": "score", "lt": 2}, "denied"),
        ({"score": 1}, {"claim": "score", "lt": 2}, "allowed"),
        # Strings and null never order against a number.
        ({"score": "5"}, {"claim": "score", "lt": 6}, "denied"),
        ({"score": None}, {"claim": "score", "lt": 6}, "denied"),
        # A missing claim fails the comparison.
        ({}, {"claim": "score", "lt": 6}, "denied"),
        # Path-located ordering.
        (
            {"quota": {"used": 7}},
            {"path": ["quota", "used"], "lte": 10},
            "allowed",
        ),
        (
            {"quota": {"used": 11}},
            {"path": ["quota", "used"], "lte": 10},
            "denied",
        ),
        ({"quota": {}}, {"path": ["quota", "used"], "lte": 10}, "denied"),
        (
            {"quota": {"used": "7"}},
            {"path": ["quota", "used"], "lte": 10},
            "denied",
        ),
    ],
)
def test_order_leaf_evaluates_numbers_only(client, claims, rule, expected):
    assert _status_for(client, claims, rule) == expected


# --- evaluation: exists ---------------------------------------------------


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        # A present claim exists, whatever its scalar value.
        ({"region": "eu"}, {"claim": "region", "exists": True}, "allowed"),
        ({"region": None}, {"claim": "region", "exists": True}, "allowed"),
        ({"region": False}, {"claim": "region", "exists": True}, "allowed"),
        ({"region": 0}, {"claim": "region", "exists": True}, "allowed"),
        # An absent claim does not exist.
        ({}, {"claim": "region", "exists": True}, "denied"),
        ({"other": 1}, {"claim": "region", "exists": True}, "denied"),
        # A full object path exists when every segment is readable.
        (
            {"tenant": {"region": "eu"}},
            {"path": ["tenant", "region"], "exists": True},
            "allowed",
        ),
        # A null terminal value still exists.
        (
            {"tenant": {"region": None}},
            {"path": ["tenant", "region"], "exists": True},
            "allowed",
        ),
        # A single-segment path reads the top-level claim.
        ({"tenant": {"region": "eu"}}, {"path": ["tenant"], "exists": True}, "allowed"),
        # Missing at any level means absent.
        ({}, {"path": ["tenant", "region"], "exists": True}, "denied"),
        ({"tenant": {}}, {"path": ["tenant", "region"], "exists": True}, "denied"),
        # Arrays and scalars block descent; they are never expanded.
        (
            {"tenant": [{"region": "eu"}]},
            {"path": ["tenant", "region"], "exists": True},
            "denied",
        ),
        (
            {"tenant": "eu"},
            {"path": ["tenant", "region"], "exists": True},
            "denied",
        ),
        (
            {"tenant": None},
            {"path": ["tenant", "region"], "exists": True},
            "denied",
        ),
        # A terminal array or object value still exists as a field.
        (
            {"tenant": {"tags": ["a"]}},
            {"path": ["tenant", "tags"], "exists": True},
            "allowed",
        ),
    ],
)
def test_exists_leaf_checks_presence_only(client, claims, rule, expected):
    assert _status_for(client, claims, rule) == expected


# --- composition, snapshots and trace -------------------------------------


def test_new_leaves_compose_with_all_any_not(client):
    claims = {
        "tier": "gold",
        "score": 7,
        "tenant": {"region": "eu"},
    }
    satisfied = {
        "all": [
            {"claim": "tier", "in": ["gold", "silver"]},
            {"any": [
                {"claim": "score", "gte": 5},
                {"claim": "score", "lt": 0},
            ]},
            {"not": {"path": ["tenant", "blocked"], "exists": True}},
            {"path": ["tenant", "region"], "in": ["eu", "us"]},
        ]
    }
    assert _status_for(client, claims, satisfied) == "allowed"

    unsatisfied = {
        "all": [
            {"claim": "tier", "in": ["gold", "silver"]},
            {"claim": "score", "lt": 5},
        ]
    }
    assert _status_for(client, claims, unsatisfied) == "denied"


def test_trace_renders_the_same_rule_snapshot(client):
    rule = {
        "all": [
            {"claim": "tier", "in": ["gold"]},
            {"claim": "score", "gte": 5},
            {"path": ["tenant", "region"], "exists": True},
        ]
    }
    claims = {"tier": "gold", "score": 9, "tenant": {"region": "eu"}}
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


def test_decision_isolation_and_determinism_with_new_leaves(client):
    # The same evidence drives set, boundary and path-existence policies
    # to deterministic results, and scopes stay isolated.
    claims = {"tier": "gold", "score": 7, "tenant": {"region": "eu"}}
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    rule = {"claim": "score", "gte": 5}
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
        client, {"claim": "score", "lt": 5}, tenant_id="tenant-b"
    )
    assert (
        _decide(client, evidence_id, created, evidence, foreign["policy_id"])
        .status_code
        == 404
    )


def test_equals_semantics_unchanged(client):
    # The pre-existing leaf keeps its exact behavior next to the new keys.
    assert (
        _status_for(client, {"m": "abc"}, {"claim": "m", "equals": "abc"})
        == "allowed"
    )
    assert (
        _status_for(client, {"m": True}, {"claim": "m", "equals": 1}) == "denied"
    )
    assert (
        _status_for(client, {}, {"claim": "m", "equals": None}) == "denied"
    )
    assert (
        _status_for(
            client,
            {"tenant": {"region": None}},
            {"path": ["tenant", "region"], "equals": None},
        )
        == "allowed"
    )


@pytest.mark.parametrize(
    "rule_json",
    [
        # Non-finite numbers are not valid JSON scalars: the server-side
        # parser accepts the literals, and validation must reject them.
        '{"claim": "a", "lt": NaN}',
        '{"claim": "a", "gte": Infinity}',
        '{"claim": "a", "gt": -Infinity}',
        '{"claim": "a", "in": [NaN]}',
        '{"claim": "a", "in": [Infinity]}',
        '{"claim": "a", "equals": NaN}',
    ],
)
def test_non_finite_comparison_values_never_persist(client, app, rule_json):
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
