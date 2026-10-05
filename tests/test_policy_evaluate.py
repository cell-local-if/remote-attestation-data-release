"""Tests for POST /v1/policies/{policy_id}/evaluate.

Read-only trial evaluation of one immutable policy version against a
caller-supplied claims object. The path identifier must be a canonical
lowercase UUID and the body carries exactly the two non-blank scope
strings and the claims object; every shape error is a 422 raised before
storage is read, and an unknown or out-of-scope policy is one
indistinguishable 404. The endpoint evaluates the exact version named in
the path — active or retired — with the same rule semantics and the same
complete depth-first explanation as a formal decision, and never creates
a decision, audit record, lifecycle event, idempotency record or counter
entry, nor changes the policy's state.
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
    DecisionCommitCounter,
    DecisionEvaluationNode,
    Policy,
    PolicyCommitCounter,
    PolicyIdempotencyRecord,
    ProofLifecycleEvent,
)
from proof_release.policies import evaluate_rule, explain_rule

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"

RULE = {
    "all": [
        {"claim": "measurement", "equals": "expected-measurement"},
        {"any": [
            {"claim": "level", "gte": 2},
            {"not": {"claim": "debug", "exists": True}},
        ]},
    ]
}

CLAIMS_OK = {"measurement": "expected-measurement", "level": 3}
CLAIMS_DENY = {"measurement": "expected-measurement", "level": 1, "debug": True}


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
        "rule": rule if rule is not None else RULE,
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


def _evaluate(client, policy_id_value, *, tenant=TENANT, workload=WORKLOAD,
              claims=_UNSET, **_extra):
    body = {
        "tenant_id": tenant,
        "workload_id": workload,
        "claims": CLAIMS_OK if claims is _UNSET else claims,
    }
    body.update(_extra)
    return client.post(f"/v1/policies/{policy_id_value}/evaluate", json=body)


# --- success shape ------------------------------------------------------------


def test_evaluate_allowed_response_shape(client):
    policy = _create(client)
    response = _evaluate(client, policy["policy_id"])
    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload) == {
        "policy_id",
        "policy_version",
        "tenant_id",
        "workload_id",
        "allowed",
        "checked_at",
        "evaluation",
    }
    assert payload["policy_id"] == policy["policy_id"]
    assert payload["policy_version"] == policy["version"]
    assert payload["tenant_id"] == TENANT
    assert payload["workload_id"] == WORKLOAD
    assert payload["allowed"] is True
    assert isinstance(payload["checked_at"], str) and payload["checked_at"]


def test_evaluate_denied(client):
    policy = _create(client)
    response = _evaluate(client, policy["policy_id"], claims=CLAIMS_DENY)
    assert response.status_code == 200, response.text
    assert response.json()["allowed"] is False


def test_evaluation_matches_decision_explanation(client):
    policy = _create(client)
    response = _evaluate(client, policy["policy_id"], claims=CLAIMS_DENY)
    assert response.status_code == 200, response.text
    evaluation = response.json()["evaluation"]

    expected = explain_rule(RULE, CLAIMS_DENY)
    assert evaluation == expected
    # The complete tree is covered in pre-order with continuous indices.
    assert [node["node_index"] for node in evaluation] == list(
        range(len(evaluation))
    )
    assert evaluation[0]["rule_path"] == []
    # Every node carries exactly the four structural fields.
    for node in evaluation:
        assert set(node) == {"node_index", "rule_path", "node_type", "outcome"}
        assert node["node_type"] in {"leaf", "all", "any", "not"}
        assert isinstance(node["outcome"], bool)
        assert all(isinstance(step, int) for step in node["rule_path"])
    # The root outcome equals the verdict.
    assert evaluation[0]["outcome"] is False


def test_evaluation_node_types_cover_compound_tree(client):
    policy = _create(client)
    response = _evaluate(client, policy["policy_id"])
    evaluation = response.json()["evaluation"]
    types = [node["node_type"] for node in evaluation]
    assert types == ["all", "leaf", "any", "leaf", "not", "leaf"]
    paths = [node["rule_path"] for node in evaluation]
    assert paths == [[], [0], [1], [1, 0], [1, 1], [1, 1, 0]]


def test_allowed_equals_evaluate_rule(client):
    policy = _create(client)
    for claims in (CLAIMS_OK, CLAIMS_DENY, {}, {"measurement": "expected-measurement"}):
        response = _evaluate(client, policy["policy_id"], claims=claims)
        assert response.status_code == 200, response.text
        assert response.json()["allowed"] is evaluate_rule(RULE, claims)


def test_response_contains_no_claim_material(client):
    policy = _create(client)
    response = _evaluate(client, policy["policy_id"])
    assert response.status_code == 200, response.text
    raw = response.text
    for leaked in ("measurement", "level", "debug", "expected-measurement", "claims"):
        assert leaked not in raw


def test_numeric_and_boolean_claims_keep_json_types(client):
    rule = {
        "all": [
            {"claim": "score", "gte": 2},
            {"claim": "flag", "equals": True},
        ]
    }
    policy = _create(client, rule=rule)
    response = _evaluate(
        client, policy["policy_id"], claims={"score": 2, "flag": True}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["allowed"] is True
    # The response carries no floats: every non-string scalar is an int
    # (indices, version) or a bool (outcomes, allowed).
    for node in payload["evaluation"]:
        assert isinstance(node["node_index"], int)
        assert isinstance(node["outcome"], bool)


def test_repeat_evaluation_is_deterministic_except_checked_at(client):
    policy = _create(client)
    first = _evaluate(client, policy["policy_id"]).json()
    second = _evaluate(client, policy["policy_id"]).json()
    assert first.keys() == second.keys()
    for key in first:
        if key == "checked_at":
            continue
        assert first[key] == second[key]


# --- request validation (all 422, before any storage read) --------------------


def test_evaluate_requires_all_body_fields(client):
    policy = _create(client)
    url = f"/v1/policies/{policy['policy_id']}/evaluate"
    assert client.post(url, json={}).status_code == 422
    base = {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": {}}
    for missing in base:
        body = {key: value for key, value in base.items() if key != missing}
        response = client.post(url, json=body)
        assert response.status_code == 422, (missing, response.text)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_id", ""),
        ("tenant_id", "   "),
        ("tenant_id", 7),
        ("tenant_id", None),
        ("workload_id", ""),
        ("workload_id", "\t"),
        ("workload_id", 7),
        ("workload_id", None),
    ],
)
def test_evaluate_rejects_bad_scope_fields(client, field, value):
    policy = _create(client)
    response = _evaluate(client, policy["policy_id"], **{field: value})
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("claims", [[], ["a"], "x", 1, 1.5, True, None])
def test_evaluate_rejects_non_object_claims(client, claims):
    policy = _create(client)
    response = _evaluate(client, policy["policy_id"], claims=claims)
    assert response.status_code == 422, response.text


def test_evaluate_rejects_unknown_body_fields(client):
    policy = _create(client)
    response = _evaluate(client, policy["policy_id"], extra="nope")
    assert response.status_code == 422, response.text


@pytest.mark.parametrize(
    "identifier",
    [
        "not-a-uuid",
        ZERO_UUID[:-1] + "z",
        "  " + ZERO_UUID,
        ZERO_UUID + "  ",
        "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
        "0A000000-0000-0000-0000-000000000000",
        "",
    ],
)
def test_evaluate_rejects_non_canonical_path_identifier(client, identifier):
    response = _evaluate(client, identifier)
    assert response.status_code == 422, (identifier, response.text)


def test_evaluate_empty_path_segment_is_422(client):
    response = client.post(
        "/v1/policies//evaluate",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": {}},
    )
    assert response.status_code == 422, response.text


def test_validation_errors_read_no_state(client, app):
    policy = _create(client)
    with app.state.session_factory() as session:
        policies_before = session.query(Policy).count()
    for response in (
        _evaluate(client, "not-a-uuid"),
        _evaluate(client, policy["policy_id"], tenant=""),
        _evaluate(client, policy["policy_id"], claims=[1, 2]),
        _evaluate(client, policy["policy_id"], unknown="x"),
    ):
        assert response.status_code == 422, response.text
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == policies_before


# --- scoping: one indistinguishable 404 ---------------------------------------


def test_evaluate_unknown_policy_is_404(client):
    response = _evaluate(client, ZERO_UUID)
    assert response.status_code == 404, response.text


def test_evaluate_cross_tenant_and_workload_are_indistinguishable_404(client):
    policy = _create(client)
    unknown = _evaluate(client, ZERO_UUID)
    cross_tenant = _evaluate(client, policy["policy_id"], tenant=OTHER_TENANT)
    cross_workload = _evaluate(
        client, policy["policy_id"], workload=OTHER_WORKLOAD
    )
    for response in (unknown, cross_tenant, cross_workload):
        assert response.status_code == 404, response.text
    assert unknown.text == cross_tenant.text == cross_workload.text


def test_evaluate_policy_in_other_scope_is_404(client):
    other = _create(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    response = _evaluate(client, other["policy_id"])
    assert response.status_code == 404, response.text


# --- immutable version semantics ----------------------------------------------


def test_evaluate_targets_the_exact_version_not_the_latest(client):
    first = _create(client, rule={"claim": "tag", "equals": "old"})
    second = _create(client, rule={"claim": "tag", "equals": "new"})
    assert second["version"] == first["version"] + 1

    claims = {"tag": "old"}
    response_first = _evaluate(client, first["policy_id"], claims=claims)
    response_second = _evaluate(client, second["policy_id"], claims=claims)
    assert response_first.status_code == 200, response_first.text
    assert response_second.status_code == 200, response_second.text
    assert response_first.json()["allowed"] is True
    assert response_first.json()["policy_version"] == first["version"]
    assert response_second.json()["allowed"] is False
    assert response_second.json()["policy_version"] == second["version"]


def test_evaluate_retired_version_still_evaluates(client):
    policy = _create(client)
    _retire(client, policy["policy_id"])

    response = _evaluate(client, policy["policy_id"])
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["allowed"] is True
    assert payload["policy_version"] == policy["version"]

    denied = _evaluate(client, policy["policy_id"], claims=CLAIMS_DENY)
    assert denied.status_code == 200, denied.text
    assert denied.json()["allowed"] is False


# --- side-effect freedom --------------------------------------------------------


def _state_counts(session):
    return {
        model: session.query(model).count()
        for model in (
            Policy,
            PolicyCommitCounter,
            PolicyIdempotencyRecord,
            Decision,
            DecisionCommitCounter,
            DecisionEvaluationNode,
            ProofLifecycleEvent,
            AuditEvent,
        )
    }


def test_evaluate_writes_no_state(client, app):
    policy = _create(client)
    _retire(client, policy["policy_id"])
    with app.state.session_factory() as session:
        before = _state_counts(session)
        status_before = session.get(Policy, policy["policy_id"]).status

    for claims in (CLAIMS_OK, CLAIMS_DENY):
        response = _evaluate(client, policy["policy_id"], claims=claims)
        assert response.status_code == 200, response.text

    with app.state.session_factory() as session:
        assert _state_counts(session) == before
        row = session.get(Policy, policy["policy_id"])
        assert row.status == status_before
        assert row.retired_at is not None


def test_evaluate_404_and_422_write_no_state(client, app):
    policy = _create(client)
    with app.state.session_factory() as session:
        before = _state_counts(session)

    responses = [
        _evaluate(client, ZERO_UUID),
        _evaluate(client, policy["policy_id"], tenant=OTHER_TENANT),
        _evaluate(client, "not-a-uuid"),
        _evaluate(client, policy["policy_id"], claims="nope"),
    ]
    assert [r.status_code for r in responses] == [404, 404, 422, 422]

    with app.state.session_factory() as session:
        assert _state_counts(session) == before


# --- storage failure ------------------------------------------------------------


def test_evaluate_storage_failure_is_500(client, app):
    policy = _create(client)

    def fail_read(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().startswith("SELECT"):
            raise RuntimeError("simulated storage failure")

    event.listen(app.state.engine, "before_cursor_execute", fail_read)
    try:
        response = _evaluate(client, policy["policy_id"])
        assert response.status_code == 500, response.text
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_read)
    # No partial evaluation leaks into the error body.
    assert "outcome" not in response.text
    assert "allowed" not in response.text


def test_evaluate_does_not_depend_on_request_json_key_order(client):
    policy = _create(client)
    body_one = json.dumps(
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": CLAIMS_OK}
    )
    body_two = json.dumps(
        {"claims": CLAIMS_OK, "workload_id": WORKLOAD, "tenant_id": TENANT}
    )
    first = client.post(
        f"/v1/policies/{policy['policy_id']}/evaluate",
        content=body_one,
        headers={"content-type": "application/json"},
    )
    second = client.post(
        f"/v1/policies/{policy['policy_id']}/evaluate",
        content=body_two,
        headers={"content-type": "application/json"},
    )
    assert first.status_code == second.status_code == 200
    one, two = first.json(), second.json()
    one.pop("checked_at")
    two.pop("checked_at")
    assert one == two
