"""Tests for the extended policy leaf operators.

Covers ``in`` membership, ``lt``/``lte``/``gt``/``gte`` numeric bounds and
``exists`` presence checks, on both ``claim`` and ``path`` locators:
validation at POST /v1/policies (422, no version allocated) and type-strict
evaluation at POST /v1/evidence/{evidence_id}/decisions.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    return create_app(f"sqlite:///{tmp_path}/operators.db")


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


def _verified_evidence(client, claims):
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    evidence = json.dumps(
        {"nonce": created["nonce"], "claims": claims, "mac": _mac(created["nonce"], claims)}
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
    return created, evidence, evidence_id


def _policy(client, rule, name="release"):
    return client.post(
        "/v1/policies",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "name": name,
            "rule": rule,
        },
    )


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


# --------------------------------------------------------------------- #
# POST /v1/policies validation
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rule",
    [
        {"claim": "r", "in": ["eu", "us"]},
        {"path": ["a", "b"], "in": [1, 2, 3]},
        {"claim": "n", "lt": 10},
        {"claim": "n", "lte": 10.5},
        {"claim": "n", "gt": -1},
        {"claim": "n", "gte": 0},
        {"path": ["a", "n"], "lt": 1.5},
        {"claim": "r", "exists": True},
        {"path": ["a", "b"], "exists": True},
        # Every allowed candidate kind together, up to the 32-item bound.
        {"claim": "r", "in": [None, True, False, "s", 1, 1.5] + list(range(26, 52))},
        {
            "all": [
                {"claim": "r", "in": ["eu", "us"]},
                {"any": [
                    {"path": ["p", "v"], "gte": 3},
                    {"claim": "flag", "exists": True},
                ]},
                {"not": {"claim": "blocked", "lt": 0}},
            ]
        },
    ],
)
def test_extended_operators_are_accepted(client, rule):
    assert _policy(client, rule).status_code == 201


@pytest.mark.parametrize(
    "rule",
    [
        # in: empty / over-long / non-scalar / duplicate (type-strict).
        {"claim": "r", "in": []},
        {"claim": "r", "in": "not-a-list"},
        {"claim": "r", "in": ["x"] * 33},
        {"claim": "r", "in": [["a"]]},
        {"claim": "r", "in": [{"a": 1}]},
        {"claim": "r", "in": ["a", "a"]},
        {"claim": "r", "in": [1, 1.0]},
        {"claim": "r", "in": [True, True]},
        {"claim": "r", "in": [None, None]},
        # ordering bounds must be finite numbers, never booleans/null/str.
        {"claim": "n", "lt": True},
        {"claim": "n", "lte": False},
        {"claim": "n", "gt": "5"},
        {"claim": "n", "gte": None},
        # Non-finite floats cannot cross the JSON wire; the validator's
        # isfinite guard is covered at unit level.
        {"path": ["a"], "gt": [1]},
        # exists accepts exactly the literal true.
        {"claim": "r", "exists": False},
        {"claim": "r", "exists": 1},
        {"claim": "r", "exists": "true"},
        {"claim": "r", "exists": None},
        # Missing or extra comparison keys, two locators, unknown nodes.
        {"claim": "r"},
        {"in": ["a"]},
        {"lt": 1},
        {"exists": True},
        {"claim": "r", "equals": 1, "in": [1]},
        {"claim": "r", "path": ["a"], "exists": True},
        {"claim": "r", "bogus": 1},
        {"unknown": 1},
        # Malformed path sibling on a new leaf.
        {"path": [], "exists": True},
        {"path": ["a", "b", "c", "d", "e", "f", "g", "h", "i"], "lt": 1},
        {"path": ["a"], "in": []},
    ],
)
def test_extended_operators_rejected_with_422(client, rule):
    assert _policy(client, rule, name=f"bad-{json.dumps(rule, default=str)}").status_code == 422


def test_rejected_extended_rule_allocates_no_version(client):
    assert _policy(client, {"claim": "r", "in": ["a"]}).status_code == 201
    assert _policy(client, {"claim": "r", "in": ["a", "a"]}).status_code == 422
    after = _policy(client, {"claim": "r", "exists": True})
    assert after.status_code == 201
    assert after.json()["version"] == 2


def test_extended_rule_round_trips_in_create_response(client):
    rule = {"all": [
        {"claim": "r", "in": ["eu", "us"]},
        {"path": ["p", "v"], "gte": 3},
    ]}
    response = _policy(client, rule)
    assert response.status_code == 201
    assert response.json()["rule"] == rule


# --------------------------------------------------------------------- #
# in evaluation
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        ({"r": "eu"}, {"claim": "r", "in": ["eu", "us"]}, "allowed"),
        ({"r": "ap"}, {"claim": "r", "in": ["eu", "us"]}, "denied"),
        # Booleans are not numbers.
        ({"n": 1}, {"claim": "n", "in": [True, False]}, "denied"),
        ({"n": True}, {"claim": "n", "in": [1, 0]}, "denied"),
        ({"n": True}, {"claim": "n", "in": [True]}, "allowed"),
        # Numbers of either width share one type.
        ({"n": 1}, {"claim": "n", "in": [1.0]}, "allowed"),
        # null only matches null; an absent key never matches.
        ({"n": None}, {"claim": "n", "in": [None]}, "allowed"),
        ({}, {"claim": "n", "in": [None]}, "denied"),
        # Structured actuals are never members.
        ({"n": [1]}, {"claim": "n", "in": [1]}, "denied"),
        ({"n": {"a": 1}}, {"claim": "n", "in": ["a"]}, "denied"),
        # Path locator with a mid-path miss.
        ({"a": {"b": "x"}}, {"path": ["a", "b"], "in": ["x", "y"]}, "allowed"),
        ({"a": [{"b": "x"}]}, {"path": ["a", "b"], "in": ["x"]}, "denied"),
        ({"a": {}}, {"path": ["a", "b"], "in": [None]}, "denied"),
    ],
)
def test_in_evaluation(client, claims, rule, expected):
    created, evidence, evidence_id = _verified_evidence(client, claims)
    policy = _policy(client, rule, name="in")
    response = _decide(client, evidence_id, created, evidence, policy.json()["policy_id"])
    assert response.status_code == 200
    assert response.json()["status"] == expected


# --------------------------------------------------------------------- #
# ordering evaluation
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        ({"n": 9}, {"claim": "n", "lt": 10}, "allowed"),
        ({"n": 10}, {"claim": "n", "lt": 10}, "denied"),
        ({"n": 10}, {"claim": "n", "lte": 10}, "allowed"),
        ({"n": 0}, {"claim": "n", "gt": -1}, "allowed"),
        ({"n": 1.5}, {"claim": "n", "gte": 1.5}, "allowed"),
        # Inclusive boundaries over a path.
        ({"p": {"v": 3}}, {"path": ["p", "v"], "gt": 3}, "denied"),
        ({"p": {"v": 3}}, {"path": ["p", "v"], "gte": 3}, "allowed"),
        # Missing, null, boolean, string and structured values never order.
        ({}, {"claim": "n", "lt": 5}, "denied"),
        ({"n": None}, {"claim": "n", "lt": 5}, "denied"),
        ({"n": True}, {"claim": "n", "lt": 5}, "denied"),
        ({"n": "4"}, {"claim": "n", "lt": 5}, "denied"),
        ({"n": [4]}, {"claim": "n", "lt": 5}, "denied"),
        ({"a": "x"}, {"path": ["a", "v"], "lt": 5}, "denied"),
    ],
)
def test_ordering_evaluation(client, claims, rule, expected):
    created, evidence, evidence_id = _verified_evidence(client, claims)
    policy = _policy(client, rule, name="ord")
    response = _decide(client, evidence_id, created, evidence, policy.json()["policy_id"])
    assert response.status_code == 200
    assert response.json()["status"] == expected


# --------------------------------------------------------------------- #
# exists evaluation
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "claims,rule,expected",
    [
        # Readable keys exist even with null / structured values.
        ({"r": None}, {"claim": "r", "exists": True}, "allowed"),
        ({"r": False}, {"claim": "r", "exists": True}, "allowed"),
        ({"r": [1]}, {"claim": "r", "exists": True}, "allowed"),
        ({}, {"claim": "r", "exists": True}, "denied"),
        # A complete object path exists; a mid-path miss does not.
        ({"a": {"b": None}}, {"path": ["a", "b"], "exists": True}, "allowed"),
        ({"a": {}}, {"path": ["a", "b"], "exists": True}, "denied"),
        ({"a": [{}]}, {"path": ["a", "b"], "exists": True}, "denied"),
        ({"a": "x"}, {"path": ["a", "b"], "exists": True}, "denied"),
        # A terminal array is present; arrays only block *further* descent.
        ({"a": [1, 2]}, {"path": ["a"], "exists": True}, "allowed"),
        # Absence is expressible with not + exists.
        ({}, {"not": {"claim": "r", "exists": True}}, "allowed"),
        ({"r": 1}, {"not": {"claim": "r", "exists": True}}, "denied"),
    ],
)
def test_exists_evaluation(client, claims, rule, expected):
    created, evidence, evidence_id = _verified_evidence(client, claims)
    policy = _policy(client, rule, name="exists")
    response = _decide(client, evidence_id, created, evidence, policy.json()["policy_id"])
    assert response.status_code == 200
    assert response.json()["status"] == expected


# --------------------------------------------------------------------- #
# Snapshot identity across create / list / trace
# --------------------------------------------------------------------- #


def test_rule_snapshot_identical_in_create_list_and_trace(client):
    claims = {"r": "eu", "p": {"v": 4}, "flag": True}
    created, evidence, evidence_id = _verified_evidence(client, claims)
    rule = {
        "all": [
            {"claim": "r", "in": ["eu", "us"]},
            {"path": ["p", "v"], "gte": 3},
            {"claim": "flag", "exists": True},
        ]
    }
    created_response = _policy(client, rule, name="snap")
    assert created_response.status_code == 201
    policy_id = created_response.json()["policy_id"]
    snapshot = created_response.json()["rule"]

    listing = client.get(
        "/v1/policies",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD, "name": "snap"},
    )
    assert listing.status_code == 200
    listed = [p for p in listing.json()["policies"] if p["policy_id"] == policy_id]
    assert len(listed) == 1
    assert listed[0]["rule"] == snapshot

    decision = _decide(client, evidence_id, created, evidence, policy_id)
    assert decision.json()["status"] == "allowed"
    trace = client.get(
        f"/v1/decisions/{decision.json()['decision_id']}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert trace.status_code == 200
    assert trace.json()["policy_version"]["rule"] == snapshot


def test_recorded_decision_is_not_re_evaluated_after_new_version(client):
    created, evidence, evidence_id = _verified_evidence(client, {"r": "eu"})
    v1 = _policy(client, {"claim": "r", "in": ["eu"]}, name="moving")
    first = _decide(client, evidence_id, created, evidence, v1.json()["policy_id"])
    assert first.json()["status"] == "allowed"

    v2 = _policy(client, {"claim": "r", "lt": 0}, name="moving")
    assert v2.json()["version"] == 2
    # The stored v1 conclusion is replayed verbatim, never re-evaluated
    # against either the new code or the new version.
    replay = _decide(client, evidence_id, created, evidence, v1.json()["policy_id"])
    assert replay.json() == first.json()
