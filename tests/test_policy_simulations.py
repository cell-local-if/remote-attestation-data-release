"""Tests for POST /v1/policies/{policy_id}/simulate — the read-only
simulation entry point that confirms the conclusion a fixed policy
version reaches for a caller-supplied claim set:

* strict request shape (exact fields, no duplicates, no NaN/Infinity,
  canonical path UUID) rejected with 422 before any state is touched;
* indistinguishable 404 for unknown or cross-scope policies;
* full-tree depth-first evaluation consistent with all/any/not semantics;
* active and retired versions both simulate, repeat calls are identical;
* no decision, grant, proof/audit event, persisted claims or rate-limit
  budget is ever produced, and a storage failure is a clean 500.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select, func

from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    Decision,
    DecisionEvaluationNode,
    Policy,
    ProofLifecycleEvent,
    RateLimitCounter,
    ReleaseGrant,
    ReleaseGrantEvent,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

UNKNOWN_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", "unit-test-secret")
    monkeypatch.setenv(
        "PROOF_RELEASE_MASTER_KEY",
        "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY",
    )
    application = create_app(f"sqlite:///{tmp_path}/simulate.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


# --- helpers ---------------------------------------------------------------


def _policy(client, rule=None, *, name="release", tenant=TENANT,
            workload=WORKLOAD):
    response = client.post(
        "/v1/policies",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "name": name,
            "rule": rule or {"claim": "measurement", "equals": "abc"},
        },
    )
    assert response.status_code == 201
    return response.json()


def _simulate(client, policy_id, *, tenant=TENANT, workload=WORKLOAD,
              claims=None):
    return client.post(
        f"/v1/policies/{policy_id}/simulate",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "claims": {} if claims is None else claims,
        },
    )


def _raw_simulate(client, policy_id, body: bytes):
    return client.post(
        f"/v1/policies/{policy_id}/simulate",
        content=body,
        headers={"content-type": "application/json"},
    )


# --- happy path ------------------------------------------------------------


def test_simulate_allowed_leaf(client):
    policy = _policy(client)
    response = _simulate(client, policy["policy_id"],
                         claims={"measurement": "abc"})
    assert response.status_code == 200
    # Compact JSON terminated by exactly one newline.
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert b", " not in response.content and b'": ' not in response.content
    # Fixed top-level key order.
    assert list(response.json().keys()) == [
        "tenant_id",
        "workload_id",
        "policy_id",
        "policy_name",
        "policy_version",
        "policy_status",
        "allowed",
        "evaluation",
    ]
    payload = response.json()
    assert payload["tenant_id"] == TENANT
    assert payload["workload_id"] == WORKLOAD
    assert payload["policy_id"] == policy["policy_id"]
    assert payload["policy_name"] == "release"
    assert payload["policy_version"] == 1
    assert payload["policy_status"] == "active"
    assert payload["allowed"] is True
    assert payload["evaluation"] == [
        {"node_index": 0, "rule_path": [], "node_type": "leaf", "outcome": True}
    ]


def test_simulate_denied_when_claim_missing(client):
    policy = _policy(client)
    response = _simulate(client, policy["policy_id"], claims={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["allowed"] is False
    assert payload["evaluation"][0]["outcome"] is False


def test_simulate_denied_when_value_differs(client):
    policy = _policy(client)
    response = _simulate(client, policy["policy_id"],
                         claims={"measurement": "other"})
    assert response.status_code == 200
    assert response.json()["allowed"] is False


def test_retired_version_still_simulates(client):
    policy = _policy(client)
    retired = client.post(
        f"/v1/policies/{policy['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200
    response = _simulate(client, policy["policy_id"],
                         claims={"measurement": "abc"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["policy_status"] == "retired"
    assert payload["allowed"] is True


def test_repeat_simulation_is_identical(client):
    policy = _policy(client)
    first = _simulate(client, policy["policy_id"],
                      claims={"measurement": "abc"})
    second = _simulate(client, policy["policy_id"],
                       claims={"measurement": "abc"})
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


# --- evaluation tree -------------------------------------------------------


def test_compound_tree_is_explained_depth_first(client):
    rule = {
        "all": [
            {"any": [
                {"claim": "level", "gte": 3},
                {"not": {"claim": "debug", "exists": True}},
            ]},
            {"path": ["image", "signer"], "in": ["ca-1", "ca-2"]},
            {"claim": "revoked", "equals": False},
        ]
    }
    policy = _policy(client, rule)
    claims = {
        "level": 5,
        "debug": True,
        "image": {"signer": "ca-2"},
        "revoked": False,
    }
    response = _simulate(client, policy["policy_id"], claims=claims)
    assert response.status_code == 200
    payload = response.json()
    assert payload["allowed"] is True
    assert payload["evaluation"] == [
        {"node_index": 0, "rule_path": [], "node_type": "all", "outcome": True},
        {"node_index": 1, "rule_path": [0], "node_type": "any", "outcome": True},
        {"node_index": 2, "rule_path": [0, 0], "node_type": "leaf",
         "outcome": True},
        {"node_index": 3, "rule_path": [0, 1], "node_type": "not",
         "outcome": False},
        {"node_index": 4, "rule_path": [0, 1, 0], "node_type": "leaf",
         "outcome": True},
        {"node_index": 5, "rule_path": [1], "node_type": "leaf",
         "outcome": True},
        {"node_index": 6, "rule_path": [2], "node_type": "leaf",
         "outcome": True},
    ]


def test_compound_outcomes_follow_all_any_not(client):
    rule = {
        "any": [
            {"all": [
                {"claim": "a", "equals": 1},
                {"claim": "b", "lt": 2},
            ]},
            {"not": {"claim": "c", "exists": True}},
        ]
    }
    policy = _policy(client, rule)
    # First disjunct fails on "b", second fails because "c" exists.
    response = _simulate(client, policy["policy_id"],
                         claims={"a": 1, "b": 5, "c": "x"})
    payload = response.json()
    assert payload["allowed"] is False
    outcomes = {tuple(n["rule_path"]): n["outcome"]
                for n in payload["evaluation"]}
    assert outcomes[()] is False        # any
    assert outcomes[(0,)] is False      # all
    assert outcomes[(0, 0)] is True     # a == 1
    assert outcomes[(0, 1)] is False    # b < 2
    assert outcomes[(1,)] is False      # not
    assert outcomes[(1, 0)] is True     # c exists
    # Every compound node agrees with its children.
    nodes = payload["evaluation"]
    by_path = {tuple(n["rule_path"]): n for n in nodes}
    for node in nodes:
        path = tuple(node["rule_path"])
        if node["node_type"] in ("all", "any"):
            children = [
                n for n in nodes
                if n["rule_path"][:-1] == list(path)
                and len(n["rule_path"]) == len(path) + 1
            ]
            expected = (
                all(c["outcome"] for c in children)
                if node["node_type"] == "all"
                else any(c["outcome"] for c in children)
            )
            assert node["outcome"] is expected
        elif node["node_type"] == "not":
            child = by_path[path + (0,)]
            assert node["outcome"] is (not child["outcome"])


def test_path_descends_objects_only(client):
    rule = {"path": ["outer", "inner"], "equals": "x"}
    policy = _policy(client, rule)
    # Object descent reaches the value.
    assert _simulate(client, policy["policy_id"],
                     claims={"outer": {"inner": "x"}}).json()["allowed"] is True
    # An array is never expanded and never indexed into.
    assert _simulate(client, policy["policy_id"],
                     claims={"outer": [{"inner": "x"}]}).json()["allowed"] is False
    assert _simulate(client, policy["policy_id"],
                     claims={"outer": {"0": {"inner": "x"}}}).json()[
        "allowed"] is False
    # A missing intermediate field is simply not satisfied.
    assert _simulate(client, policy["policy_id"],
                     claims={"outer": {}}).json()["allowed"] is False
    assert _simulate(client, policy["policy_id"],
                     claims={}).json()["allowed"] is False


def test_comparison_operators(client):
    cases = [
        ({"claim": "v", "lt": 3}, {"v": 2}, True),
        ({"claim": "v", "lt": 3}, {"v": 3}, False),
        ({"claim": "v", "lte": 3}, {"v": 3}, True),
        ({"claim": "v", "gt": 3}, {"v": 4}, True),
        ({"claim": "v", "gte": 3}, {"v": 3}, True),
        ({"claim": "v", "gte": 3}, {"v": True}, False),  # bool is not a number
        ({"claim": "v", "in": ["a", "b"]}, {"v": "b"}, True),
        ({"claim": "v", "in": ["a", "b"]}, {"v": "c"}, False),
        ({"claim": "v", "in": [1, 2]}, {"v": True}, False),  # type-strict
        ({"claim": "v", "exists": True}, {"v": None}, True),
        ({"claim": "v", "exists": True}, {}, False),
        ({"claim": "v", "equals": None}, {"v": None}, True),
        ({"claim": "v", "equals": None}, {}, False),
        ({"claim": "v", "equals": 1}, {"v": True}, False),
        ({"claim": "v", "equals": 1.5}, {"v": 1.5}, True),
    ]
    for index, (rule, claims, expected) in enumerate(cases):
        policy = _policy(client, rule, name=f"case-{index}")
        response = _simulate(client, policy["policy_id"], claims=claims)
        assert response.status_code == 200
        assert response.json()["allowed"] is expected, rule


# --- response hygiene ------------------------------------------------------


def test_response_carries_no_claims_or_values(client):
    rule = {"all": [
        {"claim": "secretclaim", "equals": "secretvalue"},
        {"path": ["nested", "token"], "gte": 42},
    ]}
    policy = _policy(client, rule)
    response = _simulate(
        client, policy["policy_id"],
        claims={"secretclaim": "secretvalue", "nested": {"token": 99}},
    )
    assert response.status_code == 200
    text = response.content.decode("utf-8")
    for leaked in ("secretclaim", "secretvalue", "nested", "token", "42",
                   "99", "claims"):
        assert leaked not in text
    # No floats anywhere in the response document.
    def _no_floats(value):
        if isinstance(value, float):
            return False
        if isinstance(value, dict):
            return all(_no_floats(v) for v in value.values())
        if isinstance(value, list):
            return all(_no_floats(v) for v in value)
        return True
    assert _no_floats(json.loads(text))


# --- 422 request shape -----------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},                                        # missing everything
        {"tenant_id": TENANT, "workload_id": WORKLOAD},          # no claims
        {"tenant_id": TENANT, "claims": {}},       # missing workload_id
        {"workload_id": WORKLOAD, "claims": {}},   # missing tenant_id
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": {},
         "extra": 1},                              # unknown field
        {"tenant_id": "", "workload_id": WORKLOAD, "claims": {}},
        {"tenant_id": "   ", "workload_id": WORKLOAD, "claims": {}},
        {"tenant_id": TENANT, "workload_id": "", "claims": {}},
        {"tenant_id": TENANT, "workload_id": "  ", "claims": {}},
        {"tenant_id": 1, "workload_id": WORKLOAD, "claims": {}},
        {"tenant_id": None, "workload_id": WORKLOAD, "claims": {}},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": []},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": "x"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": None},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": 1},
    ],
)
def test_invalid_body_shapes_are_422(client, body):
    policy = _policy(client)
    response = client.post(
        f"/v1/policies/{policy['policy_id']}/simulate", json=body
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "raw",
    [
        b"",                                       # empty body
        b"not json",
        b"[1,2]",
        b'"text"',
        b"null",
        b"42",
        # duplicate top-level field
        b'{"tenant_id":"a","tenant_id":"a","workload_id":"w","claims":{}}',
        # duplicate field nested inside claims
        b'{"tenant_id":"a","workload_id":"w","claims":{"k":1,"k":2}}',
        # non-finite numbers at any depth
        b'{"tenant_id":"a","workload_id":"w","claims":{"k":NaN}}',
        b'{"tenant_id":"a","workload_id":"w","claims":{"k":Infinity}}',
        b'{"tenant_id":"a","workload_id":"w","claims":{"k":-Infinity}}',
        b'{"tenant_id":"a","workload_id":"w","claims":{"k":1e999}}',
        b'{"tenant_id":"a","workload_id":"w","claims":NaN}',
    ],
)
def test_invalid_raw_bodies_are_422(client, raw):
    policy = _policy(client)
    response = _raw_simulate(client, policy["policy_id"], raw)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "policy_id",
    [
        "ABCDEF01-2345-6789-ABCD-EF0123456789",   # uppercase
        " 11111111-1111-1111-1111-1111111111",    # padded
        "11111111-1111-1111-1111-11111111111",    # short
        "not-a-uuid",
        "11111111111111111111111111111111",       # no dashes
    ],
)
def test_non_canonical_path_identifier_is_422(client, policy_id):
    policy = _policy(client)
    response = _simulate(client, policy_id)
    assert response.status_code == 422


def test_missing_path_identifier_is_422(client):
    response = client.post(
        "/v1/policies//simulate",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": {}},
    )
    assert response.status_code == 422


def test_shape_failures_read_no_state(client, app):
    # A 422 is produced before storage is touched: even with the database
    # made unusable, the rejection stands.
    policy = _policy(client)
    engine = app.state.engine

    def fail_all(conn, cursor, statement, parameters, context, executemany):
        raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_all)
    try:
        response = _simulate(client, policy["policy_id"], tenant=" ")
        assert response.status_code == 422
        response = _simulate(client, "not-a-uuid")
        assert response.status_code == 422
        response = _raw_simulate(
            client, policy["policy_id"],
            b'{"tenant_id":"a","tenant_id":"a","workload_id":"w","claims":{}}',
        )
        assert response.status_code == 422
    finally:
        event.remove(engine, "before_cursor_execute", fail_all)


# --- 404 not found ---------------------------------------------------------


def test_unknown_policy_is_404(client):
    response = _simulate(client, UNKNOWN_ID)
    assert response.status_code == 404


def test_cross_scope_policy_is_indistinguishable_404(client):
    policy = _policy(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    unknown = _simulate(client, UNKNOWN_ID)
    wrong_tenant = _simulate(client, policy["policy_id"])
    wrong_workload = _simulate(client, policy["policy_id"],
                               tenant=OTHER_TENANT)
    assert unknown.status_code == 404
    assert wrong_tenant.status_code == 404
    assert wrong_workload.status_code == 404
    # Unknown id and cross-scope id are byte-identical.
    assert unknown.content == wrong_tenant.content == wrong_workload.content


# --- read-only guarantees --------------------------------------------------


def _table_counts(app):
    engine = app.state.engine
    from sqlalchemy.orm import sessionmaker

    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        return {
            model.__tablename__: session.scalar(
                select(func.count()).select_from(model)
            )
            for model in (
                Decision,
                DecisionEvaluationNode,
                ReleaseGrant,
                ReleaseGrantEvent,
                ProofLifecycleEvent,
                AuditEvent,
                RateLimitCounter,
            )
        }


def test_simulation_writes_no_state_and_spends_no_budget(client, app):
    policy = _policy(client)
    before = _table_counts(app)
    for _ in range(8):
        response = _simulate(client, policy["policy_id"],
                             claims={"measurement": "abc"})
        assert response.status_code == 200
    after = _table_counts(app)
    assert before == after
    # In particular: no decision, grant, proof/audit event, and the shared
    # business rate-limit budget was never touched.
    assert all(count == 0 for count in after.values())


def test_simulation_does_not_persist_claims(client, app):
    policy = _policy(client)
    response = _simulate(client, policy["policy_id"],
                         claims={"measurement": "abc", "extra": "sensitive"})
    assert response.status_code == 200
    engine = app.state.engine
    from sqlalchemy.orm import sessionmaker

    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        # The only policy state is the immutable rule text, unchanged.
        stored = session.get(Policy, policy["policy_id"])
        assert "sensitive" not in stored.rule_json
        assert stored.status == "active"


def test_simulation_does_not_change_policy(client, app):
    policy = _policy(client)
    _simulate(client, policy["policy_id"], claims={"measurement": "abc"})
    listed = client.get(
        "/v1/policies",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert listed.status_code == 200
    (entry,) = listed.json()["policies"]
    assert entry["status"] == "active"
    assert entry["retired_at"] is None


# --- failure semantics -----------------------------------------------------


def test_storage_failure_is_500(client, app):
    policy = _policy(client)
    engine = app.state.engine

    def fail_all(conn, cursor, statement, parameters, context, executemany):
        raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_all)
    try:
        response = _simulate(client, policy["policy_id"],
                             claims={"measurement": "abc"})
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_all)
    # The same request succeeds after recovery.
    response = _simulate(client, policy["policy_id"],
                         claims={"measurement": "abc"})
    assert response.status_code == 200


def test_damaged_rule_is_500_not_partial(client, app):
    policy = _policy(client)
    engine = app.state.engine
    from sqlalchemy.orm import sessionmaker

    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        stored = session.get(Policy, policy["policy_id"])
        stored.rule_json = "{not json"
        session.commit()
    response = _simulate(client, policy["policy_id"],
                         claims={"measurement": "abc"})
    assert response.status_code == 500
