"""Tests for the ``contains`` array-membership rule leaf.

Covers POST /v1/policies validation (422 with no version allocated and
no effect on a later creation of the same name), canonical persistence
and snapshot rendering, decision evaluation of the new leaf against
verified evidence claims (type-strict scalar equality against array
elements), composition with all/any/not and the other comparison keys,
explanation shape, and the unchanged decision boundary codes.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Decision, Evidence, Policy
from proof_release.policies import evaluate_rule, explain_rule, rule_structure

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


# --- pure semantics --------------------------------------------------------


@pytest.mark.parametrize(
    "actual,expected",
    [
        (["gold", "silver"], "gold"),
        (["gold", "silver"], "silver"),
        ([1, 2, 3], 2),
        ([1, 2, 3], 2.0),
        ([1.0, 2.0], 1),
        ([-0.0], 0),
        ([True, False], False),
        ([None, "x"], None),
        (["x", None, 1], None),
        # Objects/arrays among the elements do not disturb the scan.
        (["a", {"k": "v"}, ["nested"], 1], 1),
    ],
)
def test_contains_pure_match(actual, expected):
    assert evaluate_rule({"claim": "a", "contains": expected}, {"a": actual})


@pytest.mark.parametrize(
    "actual,expected",
    [
        # Empty, non-array, absent.
        ([], "x"),
        ("gold", "gold"),
        (1, 1),
        (True, True),
        (None, None),
        ({"gold": 1}, "gold"),
        # Type-strict misses: true is not 1, numeric strings are not
        # numbers, null only equals null.
        ([1], True),
        ([True], 1),
        (["2"], 2),
        ([2], "2"),
        ([0], False),
        ([False], 0),
        ([None], False),
        ([None], 0),
        ([None], ""),
        ([""], None),
        # Non-scalar elements never equal a scalar expectation.
        ([["gold"]], "gold"),
        ([{"x": 1}], 1),
    ],
)
def test_contains_pure_miss(actual, expected):
    assert not evaluate_rule(
        {"claim": "a", "contains": expected}, {"a": actual}
    )


def test_contains_pure_missing_claim_and_path():
    assert not evaluate_rule(
        {"claim": "a", "contains": 1}, {"other": [1]}
    )
    assert not evaluate_rule(
        {"path": ["t", "tags"], "contains": 1}, {"t": {}}
    )
    # A non-object on the descent path is a miss, not an error.
    assert not evaluate_rule(
        {"path": ["t", "tags"], "contains": 1}, {"t": ["x"]}
    )
    # A terminal array under a path matches element-wise.
    assert evaluate_rule(
        {"path": ["t", "tags"], "contains": 1}, {"t": {"tags": [1, 2]}}
    )


# --- creation: accepted and read back verbatim -----------------------------


@pytest.mark.parametrize(
    "rule",
    [
        {"claim": "tier", "contains": "gold"},
        {"claim": "level", "contains": 3},
        {"claim": "level", "contains": 3.5},
        {"claim": "flag", "contains": True},
        {"claim": "note", "contains": None},
        {"path": ["tenant", "regions"], "contains": "eu"},
        {"path": ["scores"], "contains": 0},
        {
            "all": [
                {"claim": "tiers", "contains": "gold"},
                {"any": [
                    {"claim": "scores", "contains": 5},
                    {"not": {"path": ["tenant", "blocked"], "contains": True}},
                ]},
            ]
        },
    ],
)
def test_create_policy_accepts_contains_leaf(client, rule):
    response = _policy(client, rule)

    assert response.status_code == 201
    assert response.json()["rule"] == rule


def test_contains_rule_persists_canonically_and_lists_identically(client, app):
    rule = {
        "all": [
            {"claim": "tiers", "contains": "gold"},
            {"path": ["tenant", "regions"], "contains": "eu"},
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


def test_policy_query_returns_contains_rule_stably(client):
    rule = {"claim": "tiers", "contains": "gold"}
    created = _create_policy(client, rule)
    for _ in range(2):
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
        assert fetched.json()["policies"][0]["version"] == 1


# --- creation: invalid leaves are 422, no version, no state ----------------


@pytest.mark.parametrize(
    "rule",
    [
        # The expected value must be a scalar.
        {"claim": "a", "contains": ["gold"]},
        {"claim": "a", "contains": {"x": 1}},
        # Missing expected value / bare locator or operator.
        {"claim": "a"},
        {"contains": 1},
        # Mixed locators and multiple comparison keys.
        {"claim": "a", "path": ["a"], "contains": 1},
        {"claim": "a", "contains": 1, "equals": 1},
        {"claim": "a", "contains": 1, "in": [1]},
        {"path": ["a"], "contains": 1, "gte": 0},
        # Unknown sibling / node keys.
        {"claim": "a", "contains": 1, "extra": 2},
        {"claim": "a", "foo": 1},
        {"contains": 1, "all": []},
        {"path": ["a"], "contains": [1]},
        {"path": [], "contains": 1},
        {"path": "a", "contains": 1},
        {"path": [""], "contains": 1},
        {"path": ["a", "b", "c", "d", "e", "f", "g", "h", "i"], "contains": 1},
    ],
)
def test_create_policy_rejects_invalid_contains_leaves(client, app, rule):
    response = _policy(client, rule)

    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == 0


@pytest.mark.parametrize(
    "rule_json",
    [
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


def test_rejected_contains_allocates_no_version_and_name_is_reusable(client):
    assert _policy(client, {"claim": "a", "contains": ["x"]}).status_code == 422
    assert _policy(client, {"claim": "a", "contains": {"x": 1}}).status_code == 422

    created = _create_policy(client, {"claim": "a", "contains": "x"})
    assert created["version"] == 1


def test_defensive_bounds_still_apply_to_contains(client):
    deep = leaf = {"claim": "a", "contains": 1}
    for _ in range(33):
        deep = {"not": deep}
    assert _policy(client, deep).status_code == 422

    wide = {"all": [{"claim": "a", "contains": 1}] * 257}
    assert _policy(client, wide).status_code == 422
    assert leaf is not None


# --- decision evaluation ---------------------------------------------------


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        # Element matches, of every scalar type.
        ({"tiers": ["gold", "silver"]}, {"claim": "tiers", "contains": "gold"}, "allowed"),
        ({"levels": [1, 2, 3]}, {"claim": "levels", "contains": 2}, "allowed"),
        ({"levels": [1, 2, 3]}, {"claim": "levels", "contains": 2.0}, "allowed"),
        ({"flags": [True, False]}, {"claim": "flags", "contains": False}, "allowed"),
        ({"notes": [None, "x"]}, {"claim": "notes", "contains": None}, "allowed"),
        # Misses: no equal element.
        ({"tiers": ["bronze"]}, {"claim": "tiers", "contains": "gold"}, "denied"),
        ({"levels": [2]}, {"claim": "levels", "contains": "2"}, "denied"),
        ({"levels": ["2"]}, {"claim": "levels", "contains": 2}, "denied"),
        ({"flags": [True]}, {"claim": "flags", "contains": 1}, "denied"),
        ({"levels": [1]}, {"claim": "levels", "contains": True}, "denied"),
        ({"levels": [0]}, {"claim": "levels", "contains": False}, "denied"),
        ({"notes": [0, False, ""]}, {"claim": "notes", "contains": None}, "denied"),
        # Empty array, non-array value, missing claim all deny.
        ({"tiers": []}, {"claim": "tiers", "contains": "gold"}, "denied"),
        ({"tiers": "gold"}, {"claim": "tiers", "contains": "gold"}, "denied"),
        ({}, {"claim": "tiers", "contains": "gold"}, "denied"),
        # A scalar terminal is not searched: contains never string-matches.
        ({"word": "ingot"}, {"claim": "word", "contains": "go"}, "denied"),
        # Nested arrays/objects are opaque elements, not scalar matches.
        ({"tiers": [["gold"]]}, {"claim": "tiers", "contains": "gold"}, "denied"),
        ({"levels": [{"v": 1}]}, {"claim": "levels", "contains": 1}, "denied"),
        # Path-located arrays.
        (
            {"tenant": {"regions": ["eu", "us"]}},
            {"path": ["tenant", "regions"], "contains": "us"},
            "allowed",
        ),
        (
            {"tenant": {"regions": ["ap"]}},
            {"path": ["tenant", "regions"], "contains": "eu"},
            "denied",
        ),
        (
            {"tenant": {}},
            {"path": ["tenant", "regions"], "contains": "eu"},
            "denied",
        ),
        (
            {"tenant": {"regions": "eu"}},
            {"path": ["tenant", "regions"], "contains": "eu"},
            "denied",
        ),
    ],
)
def test_contains_leaf_decision_evaluates_type_strictly(client, claims, rule, expected):
    assert _status_for(client, claims, rule) == expected


def test_contains_composes_with_all_any_not_and_other_keys(client):
    claims = {
        "tiers": ["gold", "silver"],
        "score": 7,
        "tenant": {"regions": ["eu"]},
    }
    satisfied = {
        "all": [
            {"claim": "tiers", "contains": "gold"},
            {"any": [
                {"claim": "score", "gte": 5},
                {"claim": "score", "lt": 0},
            ]},
            {"not": {"path": ["tenant", "blocked"], "contains": True}},
            {"path": ["tenant", "regions"], "contains": "eu"},
        ]
    }
    assert _status_for(client, claims, satisfied) == "allowed"

    unsatisfied = {
        "all": [
            {"claim": "tiers", "contains": "platinum"},
            {"claim": "score", "lt": 5},
        ]
    }
    assert _status_for(client, claims, unsatisfied) == "denied"


def test_contains_under_not_allows_on_miss_and_non_array(client):
    claims = {"tiers": ["bronze"]}
    rule = {"not": {"claim": "tiers", "contains": "gold"}}
    assert _status_for(client, claims, rule) == "allowed"

    assert _status_for(client, {}, rule) == "allowed"
    assert (
        _status_for(
            client, {"tiers": ["gold", "bronze"]}, rule
        )
        == "denied"
    )


def test_contains_decision_is_deterministic_and_scope_isolated(client):
    claims = {"tiers": ["gold"]}
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(client, {"claim": "tiers", "contains": "gold"})

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


def test_contains_trace_and_evaluation_leak_no_locator_or_values(client):
    rule = {
        "all": [
            {"claim": "tiers", "contains": "gold"},
            {"not": {"path": ["meta", "flags"], "contains": True}},
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
    for secret in ("tiers", "meta", "flags", "gold", "bronze"):
        assert secret not in rendered

    trace = client.get(
        f"/v1/decisions/{decision_id}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert trace.status_code == 200
    assert trace.json()["policy_version"]["rule"] == rule


def test_contains_explanation_matches_structure_and_root():
    rule = {"any": [
        {"all": [
            {"claim": "a", "contains": 1},
            {"not": {"path": ["x"], "contains": None}},
        ]},
        {"not": {"claim": "b", "in": [1, 2]}},
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


# --- backward compatibility ------------------------------------------------


def test_in_semantics_unchanged_next_to_contains(client):
    # in: scalar actual value against a scalar candidate set. An array
    # terminal value still never matches an in candidate.
    assert (
        _status_for(client, {"tier": "gold"}, {"claim": "tier", "in": ["gold"]})
        == "allowed"
    )
    assert (
        _status_for(
            client, {"tier": ["gold"]}, {"claim": "tier", "in": ["gold"]}
        )
        == "denied"
    )
    # in's candidate set is unaffected by the new operator key.
    assert _policy(
        client, {"claim": "a", "in": [1, 1.0]}
    ).status_code == 422


def test_equals_semantics_unchanged_next_to_contains(client):
    # equals keeps comparing one scalar: an array claim never equals a
    # scalar element, and type-strict equality is unchanged.
    assert (
        _status_for(client, {"m": ["x"]}, {"claim": "m", "equals": "x"})
        == "denied"
    )
    assert (
        _status_for(client, {"m": True}, {"claim": "m", "equals": 1}) == "denied"
    )
    assert _status_for(client, {}, {"claim": "m", "equals": None}) == "denied"


# --- decision boundary codes stay intact with the new leaf -----------------


def test_digest_mismatch_with_contains_policy_is_422(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"a": [1]})
    policy = _create_policy(client, {"claim": "a", "contains": 1})

    response = _decide(client, evidence_id, created, evidence + " ", policy["policy_id"])
    assert response.status_code == 422
    assert response.json()["detail"] == "evidence digest mismatch"


def test_unverified_evidence_with_contains_policy_is_409(client, app):
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
    policy = _create_policy(client, {"claim": "a", "contains": 1})

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 409
    assert response.json()["detail"] == "evidence is not verified"
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 0


def test_unknown_policy_with_contains_shape_is_404(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"a": [1]})
    # A well-formed UUID the service never minted is simply not found.
    missing = "00000000-0000-0000-0000-000000000000"
    response = _decide(client, evidence_id, created, evidence, missing)
    assert response.status_code == 404
    assert response.json()["detail"] == "policy not found"

    foreign = _create_policy(
        client, {"claim": "a", "contains": 1}, tenant_id="tenant-b"
    )
    cross = _decide(
        client, evidence_id, created, evidence, foreign["policy_id"]
    )
    assert cross.status_code == 404
    assert cross.json()["detail"] == "policy not found"


def test_retired_contains_policy_is_409(client):
    created, evidence, evidence_id = _receive_and_verify(client, {"a": [1]})
    policy = _create_policy(client, {"claim": "a", "contains": 1})
    retired = client.post(
        f"/v1/policies/{policy['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200

    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 409
    assert response.json()["detail"] == "policy is retired"


def test_repeated_decision_with_contains_records_first_once(client, app):
    created, evidence, evidence_id = _receive_and_verify(client, {"a": [1]})
    # v1 denies; it is recorded first. Repeating the same policy returns
    # that first record and inserts no second row, even after a later
    # policy version has been evaluated against the same evidence.
    v1 = _create_policy(client, {"claim": "a", "contains": 9})
    first = _decide(client, evidence_id, created, evidence, v1["policy_id"])
    assert first.status_code == 200
    assert first.json()["status"] == "denied"

    repeat = _decide(client, evidence_id, created, evidence, v1["policy_id"])
    assert repeat.status_code == 200
    assert repeat.json() == first.json()

    # A distinct policy version gets its own one row; the v1 record stays
    # immutable afterwards.
    v2 = _create_policy(client, {"claim": "a", "contains": 1})
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


def test_contains_persists_neither_claims_nor_evidence(client, app):
    secret_value = "super-secret-claim-value"
    created, evidence, evidence_id = _receive_and_verify(
        client, {"tags": [secret_value]}
    )
    policy = _create_policy(
        client, {"claim": "tags", "contains": secret_value}
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
        # The expected scalar legitimately lives in the immutable rule
        # snapshot, so it is not asserted absent from the policy row; the
        # evidence and nonce material must be absent everywhere.
        assert created["nonce"] not in str(
            {c.name: getattr(decision, c.name) for c in decision.__table__.columns}
        )
