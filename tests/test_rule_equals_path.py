"""Tests for the ``equals_path`` rule leaf.

An ``equals_path`` leaf locates two values inside the same verified
claims — one with its ``claim``/``path`` locator and one with a second
read-only path — and is true when any complete candidate on one side is
type-strictly equal to any complete candidate on the other under the
existing scalar semantics. These tests cover:

* POST /v1/policies acceptance, 422 shapes (no version, no writes),
  canonical persistence and idempotent creation;
* POST /v1/policies/{policy_id}/evaluate read-only trial evaluation,
  including repeat stability and the node-only explanation;
* policy-driven decisions (allowed/denied, deterministic replay), the
  persisted evaluation nodes and the trace rule snapshot;
* GET /v1/policies/compare detecting a target-path change;
* composition with all/any/not and stability across a restart.
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
    PolicyIdempotencyRecord,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    return create_app(f"sqlite:///{tmp_path}/equals_path.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


# --- helpers ---------------------------------------------------------------


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
    assert verified.json()["status"] == "verified"
    return created, evidence, evidence_id


def _policy(client, rule, name="release", key=None, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "name": name,
        "rule": rule,
    }
    body.update(overrides)
    headers = {"Idempotency-Key": key} if key is not None else None
    return client.post("/v1/policies", json=body, headers=headers)


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


# --- creation: accepted shapes round-trip and persist canonically ----------


@pytest.mark.parametrize(
    "rule",
    [
        {"claim": "region", "equals_path": ["policy", "region"]},
        {"path": ["workload", "region"], "equals_path": ["allowed", "region"]},
        # The target path may be a single segment.
        {"claim": "region", "equals_path": ["region"]},
        # Wildcards are legal on either side, including both sides.
        {
            "path": ["items", {"wildcard": True}, "region"],
            "equals_path": ["allowed", "region"],
        },
        {
            "path": ["a"],
            "equals_path": ["items", {"wildcard": True}, "region"],
        },
        {
            "path": ["xs", {"wildcard": True}],
            "equals_path": ["ys", {"wildcard": True}],
        },
        # The widest target path (eight segments) is accepted.
        {"claim": "a", "equals_path": ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]},
        # equals_path leaves compose with the compound forms.
        {
            "all": [
                {"claim": "region", "equals_path": ["policy", "region"]},
                {"any": [
                    {"path": ["workload", "region"],
                     "equals_path": ["allowed", "region"]},
                    {"not": {"claim": "frozen", "exists": True}},
                ]},
            ]
        },
    ],
)
def test_create_policy_accepts_equals_path(client, rule):
    response = _policy(client, rule)

    assert response.status_code == 201, response.text
    assert response.json()["rule"] == rule


def test_equals_path_persists_canonically_and_lists_identically(client, app):
    rule = {
        "equals_path": ["policy", "region"],
        "claim": "region",
    }
    canonical = '{"claim":"region","equals_path":["policy","region"]}'
    created = _create_policy(client, rule)

    with app.state.session_factory() as session:
        row = session.get(Policy, created["policy_id"])
        assert row.rule_json == canonical

    listed = client.get(
        "/v1/policies", params={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert listed.status_code == 200
    assert listed.json()["policies"][0]["rule"] == {
        "claim": "region",
        "equals_path": ["policy", "region"],
    }


def test_equals_path_rule_is_stable_across_restart(client, tmp_path, monkeypatch):
    rule = {"claim": "region", "equals_path": ["policy", "region"]}
    created = _create_policy(client, rule)

    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    reopened = create_app(f"sqlite:///{tmp_path}/equals_path.db")
    reopened_client = TestClient(reopened)
    try:
        listed = reopened_client.get(
            "/v1/policies",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
        assert listed.status_code == 200
        assert listed.json()["policies"][0]["rule"] == rule
        assert listed.json()["policies"][0]["version"] == created["version"]
    finally:
        reopened.state.engine.dispose()


# --- creation: invalid shapes are 422 and write nothing --------------------


@pytest.mark.parametrize(
    "rule",
    [
        # The target must be a valid path of the same kind as ``path``.
        {"claim": "a", "equals_path": []},
        {"claim": "a", "equals_path": "region"},
        {"claim": "a", "equals_path": None},
        {"claim": "a", "equals_path": True},
        {"claim": "a", "equals_path": 1},
        {"claim": "a", "equals_path": {}},
        {"claim": "a", "equals_path": [""]},
        {"claim": "a", "equals_path": ["  "]},
        {"claim": "a", "equals_path": [7]},
        {"claim": "a", "equals_path": [None]},
        {"claim": "a", "equals_path": [["x"]]},
        {"claim": "a", "equals_path": [{"wildcard": True}] + ["s"] * 8},
        {"claim": "a", "equals_path": [{"wildcard": 1}]},
        {"claim": "a", "equals_path": [{"wildcard": True, "x": 1}]},
        {"claim": "a", "equals_path": [{"wildcard": False}]},
        {"claim": "a", "equals_path": ["x" * 129]},
        # No comparison value, no rule node, no extra or double keys.
        {"path": ["x"], "equals_path": ["y"], "equals": "z"},
        {"path": ["x"], "equals_path": ["y"], "exists": True},
        {"claim": "a", "equals_path": ["b"], "in": ["b"]},
        {"path": ["x"], "equals_path": ["y"], "extra": 1},
        # A locator is still mandatory.
        {"equals_path": ["x"]},
        # equals_path is a leaf comparison key, never a node of its own.
        {"all": [{"claim": "a", "equals_path": ["b"]}], "equals_path": ["x"]},
        # And the claim locator must still be a string.
        {"claim": 7, "equals_path": ["b"]},
    ],
)
def test_create_policy_rejects_invalid_equals_path(client, app, rule):
    response = _policy(client, rule)

    assert response.status_code == 422, response.text
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == 0
        assert session.query(PolicyIdempotencyRecord).count() == 0


def test_rejected_equals_path_allocates_no_version(client):
    bad = {"claim": "a", "equals_path": []}
    assert _policy(client, bad).status_code == 422
    assert _policy(client, bad).status_code == 422

    created = _create_policy(client, {"claim": "a", "equals_path": ["b"]})
    assert created["version"] == 1


def test_invalid_equals_path_with_idempotency_key_persists_nothing(client, app):
    # Body validation runs before the key is consumed: no record and no
    # policy survive the 422, so the key stays free afterwards.
    response = _policy(
        client,
        {"claim": "a", "equals_path": []},
        key="same-key",
    )
    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == 0
        assert session.query(PolicyIdempotencyRecord).count() == 0

    good = _policy(
        client,
        {"claim": "a", "equals_path": ["b"]},
        key="same-key",
    )
    assert good.status_code == 201


def test_equals_path_idempotent_creation_replays_first_response(client):
    rule = {"claim": "region", "equals_path": ["policy", "region"]}
    first = _policy(client, rule, key="ep-key")
    assert first.status_code == 201
    replay = _policy(client, rule, key="ep-key")
    assert replay.status_code == 201
    assert replay.text == first.text

    # A same-key request whose target path differs is a stable 409.
    conflict = _policy(
        client,
        {"claim": "region", "equals_path": ["policy", "zone"]},
        key="ep-key",
    )
    assert conflict.status_code == 409


# --- decision evaluation semantics -----------------------------------------


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        # Matching and differing string fields.
        (
            {"region": "eu", "policy": {"region": "eu"}},
            {"claim": "region", "equals_path": ["policy", "region"]},
            "allowed",
        ),
        (
            {"region": "eu", "policy": {"region": "us"}},
            {"claim": "region", "equals_path": ["policy", "region"]},
            "denied",
        ),
        # Path-located left side.
        (
            {"workload": {"region": "eu"}, "allowed": {"region": "eu"}},
            {"path": ["workload", "region"], "equals_path": ["allowed", "region"]},
            "allowed",
        ),
        (
            {"workload": {"region": "ap"}, "allowed": {"region": "eu"}},
            {"path": ["workload", "region"], "equals_path": ["allowed", "region"]},
            "denied",
        ),
        # Numbers compare by value across widths.
        ({"a": 1, "b": 1}, {"claim": "a", "equals_path": ["b"]}, "allowed"),
        ({"a": 1, "b": 1.0}, {"claim": "a", "equals_path": ["b"]}, "allowed"),
        ({"a": 1.5, "b": 1.5}, {"claim": "a", "equals_path": ["b"]}, "allowed"),
        ({"a": 1, "b": 2}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        # Booleans are not numbers and match only booleans.
        ({"a": True, "b": True}, {"claim": "a", "equals_path": ["b"]}, "allowed"),
        ({"a": False, "b": False}, {"claim": "a", "equals_path": ["b"]}, "allowed"),
        ({"a": True, "b": 1}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        ({"a": 1, "b": True}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        # Strings compare only with strings.
        ({"a": "1", "b": "1"}, {"claim": "a", "equals_path": ["b"]}, "allowed"),
        ({"a": "1", "b": 1}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        ({"a": "x", "b": None}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        # null equals only an explicit null.
        ({"a": None, "b": None}, {"claim": "a", "equals_path": ["b"]}, "allowed"),
        ({"a": None, "b": "x"}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        ({"a": "x", "b": None}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        # Arrays and objects are never equal, even structurally equal ones.
        ({"a": [1], "b": [1]}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        ({"a": [], "b": []}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        (
            {"a": {"x": 1}, "b": {"x": 1}},
            {"claim": "a", "equals_path": ["b"]},
            "denied",
        ),
        ({"a": [1], "b": 1}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        ({"a": {"x": 1}, "b": "x"}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        # Missing or unreadable sides deny, never raise.
        ({}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        ({"a": 1}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        ({"b": 1}, {"claim": "a", "equals_path": ["b"]}, "denied"),
        (
            {"region": "eu", "policy": None},
            {"claim": "region", "equals_path": ["policy", "region"]},
            "denied",
        ),
        (
            {"region": "eu", "policy": "eu"},
            {"claim": "region", "equals_path": ["policy", "region"]},
            "denied",
        ),
        (
            {"region": "eu", "policy": {}},
            {"claim": "region", "equals_path": ["policy", "region"]},
            "denied",
        ),
        # An explicit null on one side is not the same as a missing field.
        (
            {"region": None, "policy": {}},
            {"claim": "region", "equals_path": ["policy", "region"]},
            "denied",
        ),
        # Wildcard candidate sets: any equal pair satisfies the leaf.
        (
            {"items": [{"r": "eu"}, {"r": "ap"}], "allowed": {"r": "ap"}},
            {"path": ["items", {"wildcard": True}, "r"],
             "equals_path": ["allowed", "r"]},
            "allowed",
        ),
        (
            {"items": [{"r": "eu"}], "allowed": {"r": "ap"}},
            {"path": ["items", {"wildcard": True}, "r"],
             "equals_path": ["allowed", "r"]},
            "denied",
        ),
        (
            {"items": [], "allowed": {"r": "ap"}},
            {"path": ["items", {"wildcard": True}, "r"],
             "equals_path": ["allowed", "r"]},
            "denied",
        ),
        (
            {"items": "not-an-array", "allowed": {"r": "ap"}},
            {"path": ["items", {"wildcard": True}, "r"],
             "equals_path": ["allowed", "r"]},
            "denied",
        ),
        # Wildcards on both sides take the Cartesian product.
        (
            {"xs": [1, 2, 3], "ys": [3, 4]},
            {"path": ["xs", {"wildcard": True}],
             "equals_path": ["ys", {"wildcard": True}]},
            "allowed",
        ),
        (
            {"xs": [1, 2], "ys": [3, 4]},
            {"path": ["xs", {"wildcard": True}],
             "equals_path": ["ys", {"wildcard": True}]},
            "denied",
        ),
        # The target wildcard must itself meet an array at that position.
        (
            {"xs": [1], "ys": 1},
            {"path": ["xs", {"wildcard": True}],
             "equals_path": ["ys", {"wildcard": True}]},
            "denied",
        ),
        # A target side whose terminal is an array never scalar-matches.
        (
            {"a": "eu", "b": ["eu"]},
            {"claim": "a", "equals_path": ["b"]},
            "denied",
        ),
    ],
)
def test_equals_path_decision_semantics(client, claims, rule, expected):
    assert _status_for(client, claims, rule) == expected


def test_equals_path_composes_inside_compound_nodes(client):
    claims = {
        "region": "eu",
        "policy": {"region": "eu"},
        "workload": {"region": "eu"},
        "allowed": {"region": "us"},
    }
    satisfied = {
        "all": [
            {"claim": "region", "equals_path": ["policy", "region"]},
            {"any": [
                {"path": ["workload", "region"],
                 "equals_path": ["allowed", "region"]},
                {"path": ["workload", "region"],
                 "equals_path": ["policy", "region"]},
            ]},
            {"not": {"claim": "frozen", "equals_path": ["region"]}},
        ]
    }
    assert _status_for(client, claims, satisfied) == "allowed"

    unsatisfied = {
        "all": [
            {"claim": "region", "equals_path": ["policy", "region"]},
            {"path": ["workload", "region"],
             "equals_path": ["allowed", "region"]},
        ]
    }
    assert _status_for(client, claims, unsatisfied) == "denied"


def test_decision_replay_is_stable_with_equals_path(client):
    claims = {"region": "eu", "policy": {"region": "eu"}}
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(
        client, {"claim": "region", "equals_path": ["policy", "region"]}
    )

    first = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    second = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert first.status_code == 200
    assert second.json() == first.json()
    assert first.json()["status"] == "allowed"
    assert first.json()["policy_version"] == policy["version"]


# --- read-only trial evaluation --------------------------------------------


def _trial(client, policy_id, claims):
    return client.post(
        f"/v1/policies/{policy_id}/evaluate",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "claims": claims,
        },
    )


def test_evaluate_equals_path_outcomes_and_explanation(client, app):
    rule = {
        "all": [
            {"claim": "region", "equals_path": ["policy", "region"]},
            {"claim": "level", "equals_path": ["expected", "level"]},
        ]
    }
    policy = _create_policy(client, rule)

    matching = {"region": "eu", "policy": {"region": "eu"},
                "level": 3, "expected": {"level": 3}}
    response = _trial(client, policy["policy_id"], matching)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["allowed"] is True
    # all + two leaves, pre-order, booleans only.
    assert [node["node_type"] for node in data["evaluation"]] == [
        "all", "leaf", "leaf"
    ]
    assert [node["outcome"] for node in data["evaluation"]] == [
        True, True, True
    ]
    for node in data["evaluation"]:
        assert set(node) == {"node_index", "rule_path", "node_type", "outcome"}

    differing = dict(matching, level=4)
    denied = _trial(client, policy["policy_id"], differing)
    assert denied.status_code == 200
    body = denied.json()
    assert body["allowed"] is False
    assert [node["outcome"] for node in body["evaluation"]] == [
        False, True, False
    ]

    # The trial wrote nothing: no decision, audit event or policy change.
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 0
        assert session.query(AuditEvent).count() == 0
        assert session.query(Policy).count() == 1


def test_evaluate_equals_path_repeats_stable_except_checked_at(client):
    rule = {"claim": "region", "equals_path": ["policy", "region"]}
    policy = _create_policy(client, rule)
    claims = {"region": "eu", "policy": {"region": "eu"}}

    first = _trial(client, policy["policy_id"], claims)
    second = _trial(client, policy["policy_id"], claims)
    assert first.status_code == second.status_code == 200
    first_body = first.json()
    second_body = second.json()
    assert first_body["checked_at"] != second_body["checked_at"]
    del first_body["checked_at"]
    del second_body["checked_at"]
    assert first_body == second_body


def test_evaluate_equals_path_missing_side_is_false_without_leak(client):
    rule = {"claim": "region", "equals_path": ["policy", "region"]}
    policy = _create_policy(client, rule)

    response = _trial(client, policy["policy_id"], {"region": "eu"})
    assert response.status_code == 200
    assert response.json()["allowed"] is False
    assert response.json()["evaluation"] == [
        {"node_index": 0, "rule_path": [], "node_type": "leaf", "outcome": False}
    ]


# --- persisted evaluation nodes and trace snapshot -------------------------


def test_decision_evaluation_nodes_and_trace_for_equals_path(client):
    rule = {
        "any": [
            {"claim": "region", "equals_path": ["policy", "region"]},
            {"path": ["workload", "region"],
             "equals_path": ["allowed", "region"]},
        ]
    }
    claims = {
        "region": "us",
        "policy": {"region": "eu"},
        "workload": {"region": "ap"},
        "allowed": {"region": "ap"},
    }
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(client, rule)
    decided = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert decided.status_code == 200
    assert decided.json()["status"] == "allowed"
    decision_id = decided.json()["decision_id"]

    evaluation = client.get(
        f"/v1/decisions/{decision_id}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert evaluation.status_code == 200, evaluation.text
    nodes = evaluation.json()["nodes"]
    # any + two leaves: the first leaf denies, the second allows.
    assert [node["node_type"] for node in nodes] == ["any", "leaf", "leaf"]
    assert [node["outcome"] for node in nodes] == [True, False, True]
    assert [node["rule_path"] for node in nodes] == [[], [0], [1]]
    for node in nodes:
        assert set(node) == {"node_index", "rule_path", "node_type", "outcome"}

    trace = client.get(
        f"/v1/decisions/{decision_id}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert trace.status_code == 200, trace.text
    snapshot = trace.json()["policy_version"]
    assert snapshot["rule"] == rule
    assert snapshot["version"] == policy["version"]
    assert snapshot["policy_id"] == policy["policy_id"]


# --- compare ---------------------------------------------------------------


def _compare(client, left_id, right_id):
    return client.get(
        "/v1/policies/compare",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "left_policy_id": left_id,
            "right_policy_id": right_id,
        },
    )


def test_compare_detects_equals_path_target_change(client):
    left_rule = {"claim": "region", "equals_path": ["policy", "region"]}
    right_rule = {"claim": "region", "equals_path": ["policy", "zone"]}
    left = _create_policy(client, left_rule, name="pair")
    right = _create_policy(client, right_rule, name="pair")

    response = _compare(client, left["policy_id"], right["policy_id"])
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["identical"] is False
    assert len(data["changes"]) == 1
    change = data["changes"][0]
    assert change["change"] == "changed"
    assert change["rule_path"] == []
    assert change["left"] == left_rule
    assert change["right"] == right_rule


def test_compare_detects_locator_change_around_equals_path(client):
    left_rule = {"path": ["workload", "region"],
                 "equals_path": ["allowed", "region"]}
    right_rule = {"path": ["workload", "zone"],
                  "equals_path": ["allowed", "region"]}
    left = _create_policy(client, left_rule, name="pair2")
    right = _create_policy(client, right_rule, name="pair2")

    data = _compare(client, left["policy_id"], right["policy_id"]).json()
    assert data["identical"] is False
    assert [change["change"] for change in data["changes"]] == ["changed"]


def test_compare_identical_equals_path_rules(client):
    left_rule = {"equals_path": ["allowed", "region"],
                 "path": ["workload", "region"]}
    right_rule = {"path": ["workload", "region"],
                  "equals_path": ["allowed", "region"]}
    left = _create_policy(client, left_rule, name="pair3")
    right = _create_policy(client, right_rule, name="pair3")

    data = _compare(client, left["policy_id"], right["policy_id"]).json()
    assert data["identical"] is True
    assert data["changes"] == []
