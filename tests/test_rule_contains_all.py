"""Tests for the containsAll rule leaf comparison.

Covers POST /v1/policies validation (422 with no version allocated and no
state written), canonical persistence and snapshot rendering, and
decision evaluation of the set-membership leaf against verified evidence
claims: the located value must be a JSON array holding every candidate
under the same type-strict scalar equality as equals/in.
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
    return create_app(f"sqlite:///{tmp_path}/contains-all.db")


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


# --- creation: the containsAll leaf is accepted and round-trips verbatim --


@pytest.mark.parametrize(
    "rule",
    [
        {"claim": "tags", "containsAll": ["pii", "eu"]},
        {"claim": "levels", "containsAll": [1, 2, 3]},
        {"claim": "flags", "containsAll": [True]},
        {"claim": "notes", "containsAll": [None]},
        {"claim": "mix", "containsAll": ["a", 1, 1.5, True, None]},
        {"path": ["data", "classes"], "containsAll": ["confidential"]},
        # The widest candidate set still fits.
        {"claim": "n", "containsAll": list(range(32))},
        # containsAll composes with the other leaf and compound forms.
        {
            "all": [
                {"claim": "tags", "containsAll": ["eu"]},
                {"any": [
                    {"claim": "score", "gte": 5},
                    {"not": {"claim": "tags", "containsAll": ["blocked"]}},
                ]},
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
            {"claim": "tags", "containsAll": ["pii", "eu"]},
            {"path": ["data", "classes"], "containsAll": ["confidential", 2, None]},
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


# --- creation: invalid containsAll leaves are 422 and allocate no version --


@pytest.mark.parametrize(
    "rule",
    [
        # Empty, oversized, non-scalar or repeating candidates.
        {"claim": "a", "containsAll": []},
        {"claim": "a", "containsAll": list(range(33))},
        {"claim": "a", "containsAll": "abc"},
        {"claim": "a", "containsAll": [[1]]},
        {"claim": "a", "containsAll": [{"x": 1}]},
        {"claim": "a", "containsAll": [1, 1]},
        {"claim": "a", "containsAll": [1, 1.0]},
        {"claim": "a", "containsAll": ["x", "x"]},
        {"claim": "a", "containsAll": [None, None]},
        {"claim": "a", "containsAll": [True, True]},
        # A locator without the set, or mixed with other comparisons.
        {"claim": "a", "containsAll": [1], "equals": 1},
        {"claim": "a", "containsAll": [1], "in": [1]},
        {"claim": "a", "containsAll": [1], "extra": 2},
        {"path": ["a"], "containsAll": []},
        {"path": ["a"], "containsAll": [[1]]},
        {"path": ["a"], "containsAll": [1], "exists": True},
        {"containsAll": [1]},
    ],
)
def test_create_policy_rejects_invalid_contains_all_leaves(client, app, rule):
    response = _policy(client, rule)

    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == 0


def test_rejected_contains_all_leaf_allocates_no_version(client):
    assert _policy(client, {"claim": "a", "containsAll": []}).status_code == 422
    assert _policy(client, {"claim": "a", "containsAll": [1, 1.0]}).status_code == 422

    created = _create_policy(client, {"claim": "a", "containsAll": [1]})
    assert created["version"] == 1


# --- evaluation: containsAll ----------------------------------------------


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        # Every candidate present, type-strictly.
        (
            {"tags": ["pii", "eu", "internal"]},
            {"claim": "tags", "containsAll": ["pii", "eu"]},
            "allowed",
        ),
        # Exact set and superset both satisfy; order never matters.
        ({"tags": ["eu", "pii"]}, {"claim": "tags", "containsAll": ["pii", "eu"]}, "allowed"),
        ({"tags": ["pii"]}, {"claim": "tags", "containsAll": ["pii", "eu"]}, "denied"),
        ({"tags": []}, {"claim": "tags", "containsAll": ["pii"]}, "denied"),
        # Numbers of either width match by value.
        ({"levels": [1, 2.0]}, {"claim": "levels", "containsAll": [1, 2]}, "allowed"),
        ({"levels": [1.5, 2]}, {"claim": "levels", "containsAll": [1.5]}, "allowed"),
        # Booleans are not numbers, null only matches null.
        ({"flags": [True]}, {"claim": "flags", "containsAll": [1]}, "denied"),
        ({"flags": [1]}, {"claim": "flags", "containsAll": [True]}, "denied"),
        ({"notes": [None, "x"]}, {"claim": "notes", "containsAll": [None]}, "allowed"),
        ({"notes": ["", 0, False]}, {"claim": "notes", "containsAll": [None]}, "denied"),
        # Strings never match numbers.
        ({"levels": ["2"]}, {"claim": "levels", "containsAll": [2]}, "denied"),
        ({"levels": [2]}, {"claim": "levels", "containsAll": ["2"]}, "denied"),
        # A missing claim, null, scalar or object actual value is a plain
        # miss, never an error.
        ({}, {"claim": "tags", "containsAll": ["pii"]}, "denied"),
        ({"tags": None}, {"claim": "tags", "containsAll": ["pii"]}, "denied"),
        ({"tags": "pii"}, {"claim": "tags", "containsAll": ["pii"]}, "denied"),
        ({"tags": 1}, {"claim": "tags", "containsAll": [1]}, "denied"),
        ({"tags": {"0": "pii"}}, {"claim": "tags", "containsAll": ["pii"]}, "denied"),
        # Non-scalar array elements never match a candidate.
        ({"tags": [["pii"]]}, {"claim": "tags", "containsAll": ["pii"]}, "denied"),
        ({"tags": [{"x": 1}]}, {"claim": "tags", "containsAll": [1]}, "denied"),
        # Path-located arrays.
        (
            {"data": {"classes": ["confidential", "eu"]}},
            {"path": ["data", "classes"], "containsAll": ["confidential"]},
            "allowed",
        ),
        (
            {"data": {"classes": ["public"]}},
            {"path": ["data", "classes"], "containsAll": ["confidential"]},
            "denied",
        ),
        (
            {"data": {"classes": "confidential"}},
            {"path": ["data", "classes"], "containsAll": ["confidential"]},
            "denied",
        ),
        ({"data": {}}, {"path": ["data", "classes"], "containsAll": [None]}, "denied"),
        ({}, {"path": ["data", "classes"], "containsAll": [None]}, "denied"),
    ],
)
def test_contains_all_leaf_evaluates_type_strictly(client, claims, rule, expected):
    assert _status_for(client, claims, rule) == expected


def test_contains_all_composes_with_all_any_not(client):
    claims = {
        "tags": ["pii", "eu"],
        "score": 7,
        "data": {"classes": ["confidential"]},
    }
    satisfied = {
        "all": [
            {"claim": "tags", "containsAll": ["pii", "eu"]},
            {"any": [
                {"claim": "score", "gte": 5},
                {"claim": "tags", "containsAll": ["internal"]},
            ]},
            {"not": {"claim": "tags", "containsAll": ["blocked"]}},
            {"path": ["data", "classes"], "containsAll": ["confidential"]},
        ]
    }
    assert _status_for(client, claims, satisfied) == "allowed"

    unsatisfied = {
        "all": [
            {"claim": "tags", "containsAll": ["pii"]},
            {"claim": "tags", "containsAll": ["internal"]},
        ]
    }
    assert _status_for(client, claims, unsatisfied) == "denied"


def test_contains_all_decision_repeats_return_the_first_result(client):
    claims = {"tags": ["pii", "eu"]}
    created, evidence, evidence_id = _receive_and_verify(client, claims)
    policy = _create_policy(client, {"claim": "tags", "containsAll": ["pii"]})

    first = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    second = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    assert first.json()["status"] == "allowed"
