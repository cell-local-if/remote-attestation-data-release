"""Tests for GET /v1/policies/compare.

Read-only structural comparison of two persisted policy versions in one
tenant/workload scope. Only the immutable rule trees participate in
``identical`` — name, version, created_at and status never do — under the
rule evaluator's numeric equality (1 equals 1.0, true does not equal 1),
with object key order and JSON whitespace insignificant and arrays
positional. The endpoint never writes state, appends no audit record and
takes no decision; comparing a retired version is legal and changes
nothing.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"
ONE_UUID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/policies_compare.db")
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


def _compare(client, left, right, *, tenant=TENANT, workload=WORKLOAD,
             **extra):
    query = {
        "tenant_id": tenant,
        "workload_id": workload,
        "left_policy_id": left,
        "right_policy_id": right,
    }
    query.update(extra)
    return client.get("/v1/policies/compare", params=query)


# --- request validation (all 422, no policy is read) ------------------------


def test_compare_requires_all_four_parameters(client):
    assert client.get("/v1/policies/compare").status_code == 422
    base = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "left_policy_id": ZERO_UUID,
        "right_policy_id": ONE_UUID,
    }
    for missing in base:
        params = {key: value for key, value in base.items() if key != missing}
        assert (
            client.get("/v1/policies/compare", params=params).status_code == 422
        ), missing


def test_compare_rejects_unknown_parameter(client):
    response = _compare(client, ZERO_UUID, ONE_UUID, cursor="abc")
    assert response.status_code == 422
    response = _compare(client, ZERO_UUID, ONE_UUID, policy_id=ZERO_UUID)
    assert response.status_code == 422


def test_compare_rejects_repeated_parameter(client):
    query = (
        f"tenant_id={TENANT}&workload_id={WORKLOAD}"
        f"&left_policy_id={ZERO_UUID}&left_policy_id={ONE_UUID}"
        f"&right_policy_id={ONE_UUID}"
    )
    assert client.get(f"/v1/policies/compare?{query}").status_code == 422
    query = (
        f"tenant_id={TENANT}&tenant_id={OTHER_TENANT}&workload_id={WORKLOAD}"
        f"&left_policy_id={ZERO_UUID}&right_policy_id={ONE_UUID}"
    )
    assert client.get(f"/v1/policies/compare?{query}").status_code == 422


def test_compare_rejects_blank_scope(client):
    assert _compare(client, ZERO_UUID, ONE_UUID, tenant="").status_code == 422
    assert _compare(client, ZERO_UUID, ONE_UUID, tenant="  ").status_code == 422
    assert _compare(client, ZERO_UUID, ONE_UUID, workload="").status_code == 422


@pytest.mark.parametrize(
    "bad_id",
    [
        "",
        " ",
        "not-a-uuid",
        "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa".replace("-", ""),
        f" {ZERO_UUID}",
        f"{ZERO_UUID} ",
        ZERO_UUID[:-1] + "g",
    ],
)
def test_compare_rejects_non_canonical_uuid(client, bad_id):
    assert _compare(client, bad_id, ONE_UUID).status_code == 422
    assert _compare(client, ZERO_UUID, bad_id).status_code == 422


def test_compare_rejects_non_empty_body(client):
    query = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "left_policy_id": ZERO_UUID,
        "right_policy_id": ONE_UUID,
    }
    response = client.request(
        "GET", "/v1/policies/compare", params=query, content=b"{}"
    )
    assert response.status_code == 422
    response = client.request(
        "GET", "/v1/policies/compare", params=query, content=b" "
    )
    assert response.status_code == 422


def test_compare_validation_precedes_any_read(client):
    # A malformed request against an empty database is still a 422, never
    # a 404: the shape is judged before any policy is read.
    assert _compare(client, "nope", ONE_UUID).status_code == 422
    assert (
        _compare(client, ZERO_UUID, ONE_UUID, unknown="x").status_code == 422
    )


# --- existence and scope ----------------------------------------------------


def test_compare_unknown_identifiers_are_404(client):
    created = _create(client)
    assert _compare(client, created["policy_id"], ZERO_UUID).status_code == 404
    assert _compare(client, ZERO_UUID, created["policy_id"]).status_code == 404
    assert _compare(client, ZERO_UUID, ONE_UUID).status_code == 404


def test_compare_cross_scope_is_indistinguishable_404(client):
    created = _create(client)
    other = _create(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    # A persisted id addressed under the wrong scope is the same 404 as an
    # unknown id; existence is never revealed.
    for left, right, scope in [
        (created["policy_id"], other["policy_id"], {}),
        (created["policy_id"], created["policy_id"], {"tenant": OTHER_TENANT}),
        (created["policy_id"], created["policy_id"], {"workload": OTHER_WORKLOAD}),
        (other["policy_id"], other["policy_id"], {}),
    ]:
        response = _compare(client, left, right, **scope)
        assert response.status_code == 404
        assert response.json()["detail"] == "policy not found"


# --- comparison semantics ---------------------------------------------------


def test_identical_rules_ignore_metadata(client):
    rule = {"all": [{"claim": "m", "equals": "abc"}, {"claim": "v", "gte": 2}]}
    first = _create(client, rule, name="release")
    second = _create(client, rule, name="release")
    assert second["version"] == first["version"] + 1
    response = _compare(client, first["policy_id"], second["policy_id"])
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["identical"] is True
    assert data["changes"] == []
    # Metadata still describes each side faithfully.
    assert data["left"]["policy_id"] == first["policy_id"]
    assert data["left"]["version"] == first["version"]
    assert data["right"]["version"] == second["version"]


def test_self_comparison_is_legal_and_identical(client):
    created = _create(client)
    response = _compare(client, created["policy_id"], created["policy_id"])
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["identical"] is True
    assert data["changes"] == []
    assert data["left"] == data["right"]


def test_key_order_and_whitespace_are_insignificant(client):
    first = _create(client, {"claim": "m", "equals": "abc"})
    # The same leaf submitted with the keys in another order persists the
    # same canonical rule.
    response = client.post(
        "/v1/policies",
        content=b'{"tenant_id":"tenant-a","workload_id":"workload-1",'
        b'"name":"release","rule":{"equals":"abc",  "claim":"m"}}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 201, response.text
    second = response.json()
    data = _compare(client, first["policy_id"], second["policy_id"]).json()
    assert data["identical"] is True
    assert data["changes"] == []


def test_numbers_compare_by_value(client):
    first = _create(client, {"claim": "v", "gte": 1})
    second = _create(client, {"claim": "v", "gte": 1.0})
    data = _compare(client, first["policy_id"], second["policy_id"]).json()
    assert data["identical"] is True
    assert data["changes"] == []


def test_boolean_is_not_a_number(client):
    first = _create(client, {"claim": "v", "equals": True})
    second = _create(client, {"claim": "v", "equals": 1})
    data = _compare(client, first["policy_id"], second["policy_id"]).json()
    assert data["identical"] is False
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [],
            "left": {"claim": "v", "equals": True},
            "right": {"claim": "v", "equals": 1},
        }
    ]


def test_null_and_string_are_strictly_distinct(client):
    first = _create(client, {"claim": "v", "equals": None})
    second = _create(client, {"claim": "v", "equals": "null"})
    data = _compare(client, first["policy_id"], second["policy_id"]).json()
    assert data["identical"] is False
    assert len(data["changes"]) == 1
    assert data["changes"][0]["change"] == "changed"


def test_changed_leaf_records_both_nodes(client):
    first = _create(client, {"claim": "m", "equals": "abc"})
    second = _create(client, {"claim": "m", "equals": "def"})
    data = _compare(client, first["policy_id"], second["policy_id"]).json()
    assert data["identical"] is False
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [],
            "left": {"claim": "m", "equals": "abc"},
            "right": {"claim": "m", "equals": "def"},
        }
    ]


def test_only_first_differing_child_is_recorded(client):
    first = _create(
        client,
        {"all": [{"claim": "a", "equals": 1}, {"claim": "b", "equals": 2}]},
    )
    second = _create(
        client,
        {"all": [{"claim": "a", "equals": 9}, {"claim": "b", "equals": 8}]},
    )
    data = _compare(client, first["policy_id"], second["policy_id"]).json()
    assert data["identical"] is False
    # Both children differ, but only the first differing child is
    # descended into and recorded.
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [0],
            "left": {"claim": "a", "equals": 1},
            "right": {"claim": "a", "equals": 9},
        }
    ]


def test_nested_first_difference_uses_explain_paths(client):
    first = _create(
        client,
        {
            "any": [
                {"claim": "a", "equals": 1},
                {"not": {"claim": "b", "in": ["x", "y"]}},
            ]
        },
    )
    second = _create(
        client,
        {
            "any": [
                {"claim": "a", "equals": 1},
                {"not": {"claim": "b", "in": ["x", "z"]}},
            ]
        },
    )
    data = _compare(client, first["policy_id"], second["policy_id"]).json()
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [1, 0],
            "left": {"claim": "b", "in": ["x", "y"]},
            "right": {"claim": "b", "in": ["x", "z"]},
        }
    ]


def test_form_difference_is_changed_at_current_path(client):
    first = _create(client, {"all": [{"claim": "a", "equals": 1}]})
    second = _create(client, {"any": [{"claim": "a", "equals": 1}]})
    data = _compare(client, first["policy_id"], second["policy_id"]).json()
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [],
            "left": {"all": [{"claim": "a", "equals": 1}]},
            "right": {"any": [{"claim": "a", "equals": 1}]},
        }
    ]


def test_trailing_children_are_itemized_added(client):
    first = _create(client, {"all": [{"claim": "a", "equals": 1}]})
    second = _create(
        client,
        {"all": [{"claim": "a", "equals": 1}, {"claim": "b", "equals": 2}]},
    )
    data = _compare(client, first["policy_id"], second["policy_id"]).json()
    assert data["identical"] is False
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [],
            "left": {"all": [{"claim": "a", "equals": 1}]},
            "right": {
                "all": [
                    {"claim": "a", "equals": 1},
                    {"claim": "b", "equals": 2},
                ]
            },
        },
        {
            "change": "added",
            "rule_path": [1],
            "left": None,
            "right": {"claim": "b", "equals": 2},
        },
    ]


def test_trailing_children_are_itemized_removed(client):
    first = _create(
        client,
        {
            "any": [
                {"claim": "a", "equals": 1},
                {"claim": "b", "equals": 2},
                {"not": {"claim": "c", "exists": True}},
            ]
        },
    )
    second = _create(client, {"any": [{"claim": "a", "equals": 1}]})
    data = _compare(client, first["policy_id"], second["policy_id"]).json()
    assert data["identical"] is False
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [],
            "left": {
                "any": [
                    {"claim": "a", "equals": 1},
                    {"claim": "b", "equals": 2},
                    {"not": {"claim": "c", "exists": True}},
                ]
            },
            "right": {"any": [{"claim": "a", "equals": 1}]},
        },
        {
            "change": "removed",
            "rule_path": [1],
            "left": {"claim": "b", "equals": 2},
            "right": None,
        },
        {
            "change": "removed",
            "rule_path": [2],
            "left": {"not": {"claim": "c", "exists": True}},
            "right": None,
        },
    ]


def test_arrays_compare_positionally(client):
    first = _create(client, {"claim": "v", "in": ["a", "b"]})
    second = _create(client, {"claim": "v", "in": ["b", "a"]})
    data = _compare(client, first["policy_id"], second["policy_id"]).json()
    assert data["identical"] is False
    assert len(data["changes"]) == 1


# --- response shape ---------------------------------------------------------


def test_response_key_order_and_compact_framing(client):
    first = _create(client, {"claim": "m", "equals": "abc"})
    second = _create(client, {"claim": "m", "equals": "def"})
    response = _compare(client, first["policy_id"], second["policy_id"])
    assert response.status_code == 200
    text = response.text
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert "\n" not in text[:-1]
    assert '": "' not in text  # compact separators
    data = json.loads(text)
    assert list(data.keys()) == [
        "tenant_id",
        "workload_id",
        "left",
        "right",
        "identical",
        "changes",
    ]
    for side in (data["left"], data["right"]):
        assert list(side.keys()) == [
            "policy_id",
            "name",
            "version",
            "status",
            "created_at",
            "retired_at",
        ]
    assert list(data["changes"][0].keys()) == [
        "change",
        "rule_path",
        "left",
        "right",
    ]
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD


def test_rule_numbers_round_trip_verbatim(client):
    first = _create(client, {"claim": "v", "gte": 1.5})
    second = _create(client, {"claim": "v", "gte": 2.5})
    response = _compare(client, first["policy_id"], second["policy_id"])
    assert '"gte":1.5' in response.text
    assert '"gte":2.5' in response.text


# --- lifecycle interaction --------------------------------------------------


def test_retired_versions_compare_without_state_change(client):
    first = _create(client, {"claim": "m", "equals": "abc"})
    second = _create(client, {"claim": "m", "equals": "def"})
    retired = _retire(client, first["policy_id"])
    response = _compare(client, first["policy_id"], second["policy_id"])
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["identical"] is False
    assert data["left"]["status"] == "retired"
    assert data["left"]["retired_at"] == retired["retired_at"]
    assert data["right"]["status"] == "active"
    assert data["right"]["retired_at"] is None
    # The comparison is read-only: the recorded retirement is untouched
    # and a repeated retire still conflicts exactly once.
    again = client.post(
        f"/v1/policies/{first['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert again.status_code == 409
    listing = client.get(
        "/v1/policies",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "policy_id": first["policy_id"],
        },
    ).json()
    assert listing["policies"][0]["retired_at"] == retired["retired_at"]


def test_compare_does_not_create_or_reactivate(client):
    first = _create(client, {"claim": "m", "equals": "abc"})
    second = _create(client, {"claim": "m", "equals": "abc"})
    _retire(client, first["policy_id"])
    _compare(client, first["policy_id"], second["policy_id"])
    listing = client.get(
        "/v1/policies", params={"tenant_id": TENANT, "workload_id": WORKLOAD}
    ).json()
    assert len(listing["policies"]) == 2
    statuses = {row["policy_id"]: row["status"] for row in listing["policies"]}
    assert statuses[first["policy_id"]] == "retired"
    assert statuses[second["policy_id"]] == "active"
