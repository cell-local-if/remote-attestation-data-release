"""Tests for wildcard array-traversal segments in path rule leaves.

A path segment may be the exact object ``{"wildcard": True}`` in
addition to a plain string. A wildcard expands the JSON array found at
that position, several wildcards expand left to right, and a leaf is
true when any complete candidate satisfies its type-strict comparison.

Covers POST /v1/policies validation (201, 422 with no version allocated
and no state written), canonical persistence, versioning and the
cursor-paged listing, restart durability, decision evaluation of
wildcard paths against verified evidence claims, composition with
all/any/not, and the unchanged explanation shape with no leak of claim
names, locators, expected or actual values.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Policy
from proof_release.policies import (
    MAX_PATH_SEGMENTS,
    evaluate_rule,
    explain_rule,
    rule_structure,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"

WILD = {"wildcard": True}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    return create_app(f"sqlite:///{tmp_path}/wildcards.db")


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


def _decide(client, evidence_id, created, evidence, policy_id):
    return client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy_id,
        },
    )


def _status_for(client, claims, rule, name="release"):
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(client, rule, name=name)
    response = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert response.status_code == 200, response.text
    return response.json()["status"]


# --- pure evaluator semantics ----------------------------------------------


@pytest.mark.parametrize(
    "claims,rule",
    [
        # One matching element among several.
        (
            {"items": [{"region": "us"}, {"region": "eu"}]},
            {"path": ["items", WILD, "region"], "equals": "eu"},
        ),
        # First element matches.
        (
            {"items": [{"region": "eu"}, {"region": "us"}]},
            {"path": ["items", WILD, "region"], "equals": "eu"},
        ),
        # Wildcard directly after the root and as the final segment.
        ({"tags": ["a", "b", "c"]}, {"path": ["tags", WILD], "equals": "b"}),
        # Wildcard as the only segment (the claims object is not an
        # array, so this is a miss — covered in the deny table; here an
        # array-typed top-level value under a leading field).
        (
            {"groups": [[{"ok": True}], [{"ok": False}]]},
            {"path": ["groups", WILD, WILD, "ok"], "equals": True},
        ),
        # Nested arrays: Cartesian product, a deep element matches.
        (
            {"a": [{"b": [{"c": 1}, {"c": 2}]}, {"b": [{"c": 3}]}]},
            {"path": ["a", WILD, "b", WILD, "c"], "equals": 3},
        ),
        # Terminal null candidate.
        ({"items": [{"v": 1}, {"v": None}]}, {"path": ["items", WILD, "v"], "equals": None}),
        # Terminal object candidate never equals a scalar; this row uses
        # exists which an object candidate satisfies.
        (
            {"items": [{"meta": {"k": "v"}}, {}]},
            {"path": ["items", WILD, "meta"], "exists": True},
        ),
        # Terminal array candidate used with contains.
        (
            {"items": [{"tags": []}, {"tags": ["x", "y"]}]},
            {"path": ["items", WILD, "tags"], "contains": "y"},
        ),
        # Any-semantics for in and the numeric orderings.
        (
            {"items": [{"n": 1}, {"n": 5}]},
            {"path": ["items", WILD, "n"], "in": [5, 6]},
        ),
        (
            {"items": [{"n": 1}, {"n": 5}]},
            {"path": ["items", WILD, "n"], "gt": 4},
        ),
        (
            {"items": [{"n": 5}, {"n": 1}]},
            {"path": ["items", WILD, "n"], "lte": 1},
        ),
        # Numeric equality across widths still type-strict.
        (
            {"items": [{"n": 1.0}]},
            {"path": ["items", WILD, "n"], "equals": 1},
        ),
        # An element missing the field does not spoil the other matches.
        (
            {"items": [{"other": 1}, {"region": "eu"}]},
            {"path": ["items", WILD, "region"], "equals": "eu"},
        ),
    ],
)
def test_wildcard_leaf_is_true_when_any_candidate_matches(claims, rule):
    assert evaluate_rule(rule, claims)


@pytest.mark.parametrize(
    "claims,rule",
    [
        # No element compares equal.
        (
            {"items": [{"region": "us"}, {"region": "ap"}]},
            {"path": ["items", WILD, "region"], "equals": "eu"},
        ),
        # Empty array: zero candidates.
        ({"items": []}, {"path": ["items", WILD, "region"], "equals": "eu"}),
        ({"items": []}, {"path": ["items", WILD], "exists": True}),
        # Non-array at a wildcard position.
        ({"items": {"region": "eu"}}, {"path": ["items", WILD, "region"], "equals": "eu"}),
        ({"items": "eu"}, {"path": ["items", WILD], "equals": "eu"}),
        ({"items": None}, {"path": ["items", WILD], "exists": True}),
        # Missing intermediate field.
        ({}, {"path": ["items", WILD, "region"], "equals": "eu"}),
        ({"items": [{}]}, {"path": ["items", WILD, "region"], "exists": True}),
        # Scalar or non-object met while another segment still has to
        # descend.
        ({"items": ["eu"]}, {"path": ["items", WILD, "region"], "equals": "eu"}),
        (
            {"items": [["eu"]]},
            {"path": ["items", WILD, "region"], "equals": "eu"},
        ),
        # Type-strict: true is not 1, strings are not numbers.
        (
            {"items": [{"n": True}]},
            {"path": ["items", WILD, "n"], "equals": 1},
        ),
        (
            {"items": [{"n": 1}]},
            {"path": ["items", WILD, "n"], "equals": True},
        ),
        (
            {"items": [{"n": "1"}]},
            {"path": ["items", WILD, "n"], "gt": 0},
        ),
        # Explicit null candidate is not equal to a string.
        (
            {"items": [{"v": None}]},
            {"path": ["items", WILD, "v"], "equals": "eu"},
        ),
        # Terminal object/array candidates never equal a scalar.
        (
            {"items": [{"v": {"region": "eu"}}]},
            {"path": ["items", WILD, "v"], "equals": "eu"},
        ),
        (
            {"items": [{"v": ["eu"]}]},
            {"path": ["items", WILD, "v"], "equals": "eu"},
        ),
        # contains still needs an array terminal on every candidate.
        (
            {"items": [{"tags": "x"}]},
            {"path": ["items", WILD, "tags"], "contains": "x"},
        ),
        # A later wildcard finds only empty arrays even though the first
        # expanded.
        (
            {"a": [{"b": []}, {"b": []}]},
            {"path": ["a", WILD, "b", WILD], "exists": True},
        ),
    ],
)
def test_wildcard_leaf_is_false_with_zero_or_mismatched_candidates(claims, rule):
    assert not evaluate_rule(rule, claims)


def test_literal_asterisk_segment_still_names_a_field():
    # The string "*" is a literal field name and never expands arrays.
    claims = {"*": [{"region": "eu"}]}
    assert evaluate_rule({"path": ["*"], "exists": True}, claims)
    assert not evaluate_rule(
        {"path": ["*", "region"], "equals": "eu"}, claims
    )


def test_wildcard_composes_with_all_any_not():
    claims = {
        "items": [{"region": "eu", "tier": 2}, {"region": "us", "tier": 9}],
        "blocked": [{"flag": True}],
    }
    rule = {
        "all": [
            {"path": ["items", WILD, "region"], "in": ["eu", "us"]},
            {"any": [
                {"path": ["items", WILD, "tier"], "gt": 5},
                {"claim": "override", "equals": True},
            ]},
            {"not": {"path": ["missing", WILD], "exists": True}},
        ]
    }
    assert evaluate_rule(rule, claims)

    # The not flips a matching any-candidate leaf.
    assert not evaluate_rule(
        {"not": {"path": ["items", WILD, "region"], "equals": "eu"}},
        claims,
    )


# --- validation through POST /v1/policies ----------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        {"path": ["items", WILD, "region"], "equals": "eu"},
        {"path": [WILD], "exists": True},
        {"path": ["items", WILD], "equals": 1},
        # Two wildcards, and wildcards at both the first and last
        # position.
        {"path": [WILD, "b", WILD], "equals": None},
        {"path": ["a", WILD, "b", WILD, "c", WILD, "d", WILD], "exists": True},
        # The eight-segment bound counts wildcard segments like any
        # other.
        {"path": [WILD, WILD, WILD, WILD, WILD, WILD, WILD, WILD], "equals": 1},
        # A long string segment next to a wildcard is still within the
        # 128-character segment bound.
        {"path": ["x" * 128, WILD, "y"], "equals": True},
        # Inside compounds, mixed with claim leaves.
        {
            "all": [
                {"claim": "measurement", "equals": "abc"},
                {"path": ["items", WILD, "region"], "equals": "eu"},
                {"not": {"path": ["items", WILD], "exists": True}},
            ]
        },
        # The literal string "*" remains an ordinary, valid segment.
        {"path": ["*"], "equals": 1},
    ],
)
def test_create_policy_accepts_wildcard_rules(client, rule):
    response = _policy(client, rule=rule)
    assert response.status_code == 201, response.text
    # The rule is echoed back structurally unchanged.
    assert response.json()["rule"] == rule


@pytest.mark.parametrize(
    "segment",
    [
        {"wildcard": 1},  # true must not be written as 1
        {"wildcard": 1.0},
        {"wildcard": False},
        {"wildcard": "true"},
        {"wildcard": None},
        {"wildcard": []},
        {},  # bare object is not a wildcard
        {"foo": True},  # unknown key
        {"wildcard": True, "extra": 1},  # extra key
        [],  # wrong type entirely
        7,
        True,
    ],
)
def test_create_policy_rejects_malformed_wildcard_segments(client, segment):
    rule = {"path": ["items", segment, "region"], "equals": "eu"}
    assert _policy(client, rule=rule).status_code == 422


def test_create_policy_rejects_wildcard_path_over_the_segment_bound(client):
    rule = {"path": ["a", WILD, "c", "d", "e", "f", "g", "h", "i"], "equals": 1}
    assert len(rule["path"]) == MAX_PATH_SEGMENTS + 1
    assert _policy(client, rule=rule).status_code == 422


def test_rejected_wildcard_rule_allocates_no_version_and_writes_no_state(
    client, app
):
    first = _policy(client, rule={"claim": "measurement", "equals": "abc"})
    assert first.status_code == 201

    rejected = _policy(
        client, rule={"path": ["items", {"wildcard": 1}], "equals": "eu"}
    )
    assert rejected.status_code == 422

    after = _create_policy(
        client, {"path": ["items", WILD, "region"], "equals": "eu"}
    )
    assert after["version"] == 2
    with app.state.session_factory() as session:
        rows = session.query(Policy).order_by(Policy.version).all()
        assert [r.version for r in rows] == [1, 2]


# --- persistence, versioning and the paged listing --------------------------


def test_wildcard_rule_persists_across_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart-wildcard.db"
    rule = {
        "all": [
            {"path": ["items", WILD, "region"], "in": ["eu", "us"]},
            {"not": {"path": ["blocked", WILD, "flag"], "equals": True}},
        ]
    }
    app1 = create_app(url)
    client1 = TestClient(app1)
    created = _policy(client1, rule=rule)
    assert created.status_code == 201
    policy_id = created.json()["policy_id"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        row = session.get(Policy, policy_id)
        assert row is not None
        assert json.loads(row.rule_json) == rule
    app2.state.engine.dispose()


def test_wildcard_rule_appears_in_version_listing_and_pagination(
    client, monkeypatch
):
    from proof_release import app as app_module

    for index in range(4):
        assert (
            _policy(
                client,
                rule={
                    "path": ["items", WILD, "region"],
                    "equals": f"r{index}",
                },
                name=f"policy-{index}",
            ).status_code
            == 201
        )

    # Walk the listing with a deliberately small page size and confirm
    # the wildcard rule structure survives pagination untouched.
    monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", 2)
    seen = []
    token = None
    for _ in range(10):
        params = {"tenant_id": TENANT, "workload_id": WORKLOAD}
        if token is not None:
            params["cursor"] = token
        page = client.get("/v1/policies", params=params)
        assert page.status_code == 200, page.text
        data = page.json()
        seen.extend(data["policies"])
        if data["complete"]:
            assert data["next_cursor"] == ""
            break
        token = data["next_cursor"]
        assert token
    else:  # pragma: no cover
        raise AssertionError("pagination never completed")

    rendered = {row["name"]: row["rule"] for row in seen}
    for index in range(4):
        assert rendered[f"policy-{index}"] == {
            "path": ["items", WILD, "region"],
            "equals": f"r{index}",
        }


# --- end-to-end decisions against verified evidence -------------------------


@pytest.mark.parametrize(
    "claims,rule",
    [
        (
            {"items": [{"region": "us"}, {"region": "eu"}]},
            {"path": ["items", WILD, "region"], "equals": "eu"},
        ),
        (
            {"items": [{"n": 1}, {"n": 5}]},
            {"path": ["items", WILD, "n"], "gte": 5},
        ),
        (
            {"items": [{"tags": ["a"]}, {"tags": ["b", "c"]}]},
            {"path": ["items", WILD, "tags"], "contains": "c"},
        ),
    ],
)
def test_decision_allows_when_a_wildcard_candidate_matches(client, claims, rule):
    assert _status_for(client, claims, rule) == "allowed"


@pytest.mark.parametrize(
    "claims,rule",
    [
        # Empty array and a non-array at the wildcard position.
        ({"items": []}, {"path": ["items", WILD, "region"], "equals": "eu"}),
        (
            {"items": {"region": "eu"}},
            {"path": ["items", WILD, "region"], "equals": "eu"},
        ),
        # Path without a wildcard still treats an array as missing.
        (
            {"items": [{"region": "eu"}]},
            {"path": ["items", "region"], "equals": "eu"},
        ),
        # No matching candidate.
        (
            {"items": [{"region": "us"}]},
            {"path": ["items", WILD, "region"], "equals": "eu"},
        ),
    ],
)
def test_decision_denies_on_wildcard_miss_without_raising(client, claims, rule):
    assert _status_for(client, claims, rule) == "denied"


def test_decision_explanation_shape_is_unchanged_and_leaks_nothing(client):
    secret = "super-secret-region-value-123456"
    rule = {
        "all": [
            {"path": ["items", WILD, "region"], "equals": secret},
            {"not": {"path": ["items", WILD, "blocked"], "equals": True}},
        ]
    }
    claims = {
        "items": [
            {"region": secret, "blocked": False},
            {"region": "other", "blocked": False},
        ]
    }
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(client, rule)
    decided = _decide(
        client, evidence_id, created, evidence, policy["policy_id"]
    ).json()
    assert decided["status"] == "allowed"

    response = client.get(
        f"/v1/decisions/{decided['decision_id']}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200, response.text
    nodes = response.json()["nodes"]

    # Shape: one entry per node, pre-order, fixed keys and domains.
    assert [node["node_index"] for node in nodes] == list(range(len(nodes)))
    for node in nodes:
        assert set(node) == {"node_index", "rule_path", "node_type", "outcome"}
        assert node["node_type"] in ("leaf", "all", "any", "not")
        assert isinstance(node["outcome"], bool)
    assert [(node["rule_path"], node["node_type"]) for node in nodes] == [
        ([], "all"),
        ([0], "leaf"),
        ([1], "not"),
        ([1, 0], "leaf"),
    ]
    assert nodes[0]["outcome"] is True
    assert nodes[1]["outcome"] is True
    assert nodes[2]["outcome"] is True
    assert nodes[3]["outcome"] is False

    text = response.text
    for token in (secret, "items", "region", "blocked", "wildcard"):
        assert token not in text
    assert '"path"' not in text and '"equals"' not in text
    assert evidence not in text

    # rule_structure agrees with the persisted shape and exposes only
    # positions and structural types.
    assert rule_structure(rule) == [
        ((), "all"),
        ((0,), "leaf"),
        ((1,), "not"),
        ((1, 0), "leaf"),
    ]
    assert explain_rule(rule, claims)[0]["outcome"] is True


def test_wildcard_leaf_counts_as_a_single_rule_node(client):
    # Wildcards are path segments, not rule nodes: the node and depth
    # budgets are unchanged. An any node plus 255 wildcard leaves is 256
    # nodes (at the limit); one more leaf is over it.
    at_limit = {
        "any": [
            {"path": ["items", WILD, "region"], "equals": index}
            for index in range(255)
        ]
    }
    assert _policy(client, rule=at_limit).status_code == 201

    over_limit = {
        "any": [
            {"path": ["items", WILD, "region"], "equals": index}
            for index in range(256)
        ]
    }
    assert _policy(client, rule=over_limit).status_code == 422
