"""Tests for POST /v1/policies/{policy_id}/evaluate.

Read-only trial evaluation of one immutable policy version against
caller-supplied claims, inside one tenant/workload scope. The body
carries exactly the two non-blank scope strings and the ``claims`` JSON
object; any shape error is a 422 raised before storage is read, and an
unknown or out-of-scope identifier is one indistinguishable 404. The
endpoint never writes state — no decision, audit event, lifecycle event,
idempotency record or counter — and a retired version still evaluates
against its own immutable rule.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event

from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    Decision,
    DecisionEvaluationNode,
    Policy,
    PolicyCommitCounter,
    PolicyIdempotencyRecord,
    ProofLifecycleEvent,
)
from proof_release.policies import explain_rule, rule_structure

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"

COMPOUND_RULE = {
    "all": [
        {"claim": "measurement", "equals": "abc"},
        {"any": [
            {"claim": "level", "gte": 3},
            {"not": {"claim": "debug", "exists": True}},
        ]},
    ]
}


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/policies_evaluate.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, rule=None, *, name="release", tenant=TENANT,
            workload=WORKLOAD):
    body = {
        "tenant_id": tenant,
        "workload_id": workload,
        "name": name,
        "rule": rule or {"claim": "measurement", "equals": "abc"},
    }
    response = client.post("/v1/policies", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _retire(client, policy_id_value, *, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        f"/v1/policies/{policy_id_value}/retire",
        json={"tenant_id": tenant, "workload_id": workload},
    )
    assert response.status_code == 200, response.text
    return response.json()


_UNSET = object()


def _evaluate(client, policy_id, *, tenant=TENANT, workload=WORKLOAD,
              claims=_UNSET, **extra):
    body = {
        "tenant_id": tenant,
        "workload_id": workload,
        "claims": {"measurement": "abc"} if claims is _UNSET else claims,
    }
    body.update(extra)
    return client.post(f"/v1/policies/{policy_id}/evaluate", json=body)


# --- success -----------------------------------------------------------------


def test_evaluate_allowed(client):
    created = _create(client)
    response = _evaluate(client, created["policy_id"])
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
    assert data["policy_id"] == created["policy_id"]
    assert data["policy_version"] == 1
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["allowed"] is True
    assert isinstance(data["checked_at"], str) and data["checked_at"]


def test_evaluate_denied(client):
    created = _create(client)
    response = _evaluate(
        client, created["policy_id"], claims={"measurement": "different"}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["allowed"] is False
    assert data["evaluation"][0]["outcome"] is False


def test_evaluation_covers_complete_rule_tree(client):
    created = _create(client, rule=COMPOUND_RULE)
    claims = {"measurement": "abc", "level": 5, "debug": True}
    response = _evaluate(client, created["policy_id"], claims=claims)
    assert response.status_code == 200, response.text
    data = response.json()

    expected = explain_rule(COMPOUND_RULE, claims)
    assert data["evaluation"] == expected
    # One node per rule-tree node, pre-order, with exactly the four fields.
    assert len(data["evaluation"]) == len(rule_structure(COMPOUND_RULE))
    for index, node in enumerate(data["evaluation"]):
        assert set(node.keys()) == {
            "node_index",
            "rule_path",
            "node_type",
            "outcome",
        }
        assert node["node_index"] == index
        assert isinstance(node["outcome"], bool)
    # The root outcome is the overall verdict.
    assert data["evaluation"][0]["outcome"] == data["allowed"]


def test_evaluation_node_outcomes_follow_compound_semantics(client):
    # all: leaf true, any: gte false but not-exists true -> any true.
    created = _create(client, rule=COMPOUND_RULE)
    claims = {"measurement": "abc", "level": 1}
    data = _evaluate(client, created["policy_id"], claims=claims).json()
    assert data["allowed"] is True
    nodes = data["evaluation"]
    assert [node["node_type"] for node in nodes] == [
        "all", "leaf", "any", "leaf", "not", "leaf",
    ]
    assert [node["rule_path"] for node in nodes] == [
        [], [0], [1], [1, 0], [1, 1], [1, 1, 0],
    ]
    assert [node["outcome"] for node in nodes] == [
        True, True, True, False, True, False,
    ]


def test_response_contains_no_claim_or_rule_material(client):
    created = _create(
        client, rule={"claim": "secret-claim-name", "equals": "secret-value"}
    )
    response = _evaluate(
        client,
        created["policy_id"],
        claims={"secret-claim-name": "secret-value"},
    )
    assert response.status_code == 200, response.text
    assert "secret-claim-name" not in response.text
    assert "secret-value" not in response.text


def test_numbers_and_booleans_are_native_json(client):
    created = _create(client, rule={"claim": "level", "gte": 3})
    response = _evaluate(
        client, created["policy_id"], claims={"level": 4, "flag": True}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["allowed"] is True
    assert isinstance(data["policy_version"], int)
    assert all(
        isinstance(node["node_index"], int) for node in data["evaluation"]
    )
    # No float ever appears in the serialized body.
    assert "." not in response.text.split('"checked_at"', 1)[0].replace(
        created["policy_id"], ""
    )


def test_repeat_evaluation_is_identical_except_checked_at(client):
    created = _create(client, rule=COMPOUND_RULE)
    claims = {"measurement": "abc", "level": 7}
    first = _evaluate(client, created["policy_id"], claims=claims).json()
    second = _evaluate(client, created["policy_id"], claims=claims).json()
    assert first.keys() == second.keys()
    first.pop("checked_at")
    second.pop("checked_at")
    assert first == second


def test_retired_version_still_evaluates(client):
    created = _create(client)
    _retire(client, created["policy_id"])
    response = _evaluate(client, created["policy_id"])
    assert response.status_code == 200, response.text
    assert response.json()["allowed"] is True


def test_evaluation_reflects_the_named_version_not_the_latest(client):
    first = _create(client, rule={"claim": "measurement", "equals": "old"})
    second = _create(client, rule={"claim": "measurement", "equals": "new"})
    assert second["version"] == first["version"] + 1

    claims = {"measurement": "old"}
    response = _evaluate(client, first["policy_id"], claims=claims)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["policy_version"] == first["version"]
    assert data["allowed"] is True

    latest = _evaluate(client, second["policy_id"], claims=claims)
    assert latest.json()["allowed"] is False


# --- 404: unknown or out-of-scope, indistinguishable --------------------------


def test_unknown_policy_is_404(client):
    response = _evaluate(client, ZERO_UUID)
    assert response.status_code == 404
    assert response.json() == {"detail": "policy not found"}


def test_out_of_scope_policy_shares_one_indistinguishable_404(client):
    created = _create(client)
    unknown = _evaluate(client, ZERO_UUID)
    cross_tenant = _evaluate(client, created["policy_id"], tenant=OTHER_TENANT)
    cross_workload = _evaluate(
        client, created["policy_id"], workload=OTHER_WORKLOAD
    )
    for response in (cross_tenant, cross_workload):
        assert response.status_code == 404
        assert response.content == unknown.content


# --- 422: request shape, before any storage access ----------------------------


@pytest.mark.parametrize(
    "policy_id",
    [
        "not-a-uuid",
        ZERO_UUID[:-1] + "z",
        "  " + ZERO_UUID,
        ZERO_UUID + "  ",
        "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
    ],
)
def test_non_canonical_path_identifier_is_422(client, policy_id):
    assert _evaluate(client, policy_id).status_code == 422


def test_empty_path_identifier_is_422(client):
    response = client.post(
        "/v1/policies//evaluate",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "claims": {},
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "claims"])
def test_missing_body_field_is_422(client, missing):
    created = _create(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "claims": {"measurement": "abc"},
    }
    del body[missing]
    response = client.post(
        f"/v1/policies/{created['policy_id']}/evaluate", json=body
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_id", ""),
        ("tenant_id", "   "),
        ("tenant_id", 7),
        ("workload_id", ""),
        ("workload_id", "\t"),
        ("workload_id", ["workload-1"]),
    ],
)
def test_blank_or_wrong_typed_scope_is_422(client, field, value):
    created = _create(client)
    assert _evaluate(client, created["policy_id"], **{field: value}).status_code == 422


def test_unknown_body_field_is_422(client):
    created = _create(client)
    assert (
        _evaluate(client, created["policy_id"], unexpected="x").status_code
        == 422
    )


@pytest.mark.parametrize("claims", [
    ["measurement", "abc"],
    "measurement",
    7,
    True,
    None,
])
def test_non_object_claims_is_422(client, claims):
    created = _create(client)
    assert (
        _evaluate(client, created["policy_id"], claims=claims).status_code
        == 422
    )


def test_shape_errors_never_touch_storage(app, client):
    created = _create(client)
    with app.state.session_factory() as session:
        policies_before = session.query(Policy).count()
        counters_before = session.query(PolicyCommitCounter).count()
    response = client.post(
        f"/v1/policies/{created['policy_id']}/evaluate",
        json={"tenant_id": TENANT, "claims": {}},
    )
    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == policies_before
        assert session.query(PolicyCommitCounter).count() == counters_before


# --- side-effect free ---------------------------------------------------------


def test_evaluation_writes_no_state(app, client):
    created = _create(client, rule=COMPOUND_RULE)
    with app.state.session_factory() as session:
        before = {
            model: session.query(model).count()
            for model in (
                Policy,
                PolicyCommitCounter,
                PolicyIdempotencyRecord,
                Decision,
                DecisionEvaluationNode,
                ProofLifecycleEvent,
                AuditEvent,
            )
        }
        status_before = session.get(Policy, created["policy_id"]).status

    response = _evaluate(
        client, created["policy_id"], claims={"measurement": "abc", "level": 9}
    )
    assert response.status_code == 200, response.text

    with app.state.session_factory() as session:
        for model, count in before.items():
            assert session.query(model).count() == count, model.__name__
        row = session.get(Policy, created["policy_id"])
        assert row.status == status_before
        assert row.retired_at is None


# --- storage failure ----------------------------------------------------------


def test_read_failure_returns_500_without_partial_evaluation(app, client):
    created = _create(client)

    def fail_read(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().startswith("SELECT policies.policy_id"):
            raise RuntimeError("super-secret-failure-detail")

    event.listen(app.state.engine, "before_cursor_execute", fail_read)
    try:
        response = _evaluate(client, created["policy_id"])
        assert response.status_code == 500
        assert b"super-secret-failure-detail" not in response.content
        assert b"evaluation" not in response.content
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_read)

    # After recovery the same request evaluates normally.
    recovered = _evaluate(client, created["policy_id"])
    assert recovered.status_code == 200
    assert recovered.json()["allowed"] is True
