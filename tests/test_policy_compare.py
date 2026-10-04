"""Tests for GET /v1/policies/compare.

Read-only comparison of the immutable rule trees of two persisted policy
versions within one tenant/workload scope. The request carries exactly
four query parameters and no body; any shape error is a 422 raised before
storage is read, and an unknown or out-of-scope identifier on either side
is one indistinguishable 404. Only the rule trees determine ``identical``
and ``changes`` — name, version, status and timestamps are reported but
never compared — and the endpoint never writes state, appends audit or
influences decisions.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Policy, PolicyCommitCounter
from proof_release.policies import diff_rules, rules_equal

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"


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


def _compare(client, *, tenant=TENANT, workload=WORKLOAD,
             left=None, right=None, **extra):
    query = {
        "tenant_id": tenant,
        "workload_id": workload,
        "left_policy_id": left if left is not None else ZERO_UUID,
        "right_policy_id": right if right is not None else ZERO_UUID,
    }
    query.update(extra)
    return client.get("/v1/policies/compare", params=query)


def _pair(client, left_rule, right_rule, *, left_name="left",
          right_name="right"):
    left = _create(client, rule=left_rule, name=left_name)
    right = _create(client, rule=right_rule, name=right_name)
    response = _compare(
        client, left=left["policy_id"], right=right["policy_id"]
    )
    assert response.status_code == 200, response.text
    return left, right, response


# --- request validation (all 422, before any storage read) -----------------


def test_compare_requires_all_four_parameters(client):
    assert client.get("/v1/policies/compare").status_code == 422
    base = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "left_policy_id": ZERO_UUID,
        "right_policy_id": ZERO_UUID,
    }
    for missing in base:
        params = {key: value for key, value in base.items() if key != missing}
        response = client.get("/v1/policies/compare", params=params)
        assert response.status_code == 422, (missing, response.text)


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"workload_id": "\t"},
        {"left_policy_id": ""},
        {"left_policy_id": "   "},
        {"left_policy_id": "not-a-uuid"},
        {"left_policy_id": ZERO_UUID[:-1] + "z"},
        {"left_policy_id": "  " + ZERO_UUID},
        {"left_policy_id": ZERO_UUID + "  "},
        {"left_policy_id": "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"},
        {"right_policy_id": ""},
        {"right_policy_id": "not-a-uuid"},
        {"right_policy_id": "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"},
        {"bogus": "value"},
        {"policy_id": ZERO_UUID},
        {"name": "release"},
    ],
)
def test_compare_rejects_invalid_parameters(client, params):
    response = _compare(client, **params)
    assert response.status_code == 422, response.text


def test_repeated_parameter_is_422(client):
    response = client.get(
        "/v1/policies/compare",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("left_policy_id", ZERO_UUID),
            ("left_policy_id", ZERO_UUID),
            ("right_policy_id", ZERO_UUID),
        ],
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"{}", b"  ", b"not json", b"\x00"])
def test_non_empty_body_is_422_before_state_read(app, client, body):
    def fail_read(conn, cursor, statement, parameters, context, executemany):
        if "FROM policies" in statement:
            raise AssertionError("storage must not be read for a bodied query")

    from sqlalchemy import event

    event.listen(app.state.engine, "before_cursor_execute", fail_read)
    try:
        response = client.request(
            "GET",
            "/v1/policies/compare",
            params={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "left_policy_id": ZERO_UUID,
                "right_policy_id": ZERO_UUID,
            },
            content=body,
        )
        assert response.status_code == 422
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_read)


def test_invalid_parameters_write_no_state(app, client):
    _create(client, name="one")
    _compare(client, left="nope")
    _compare(client, unknown="x")
    _compare(client, tenant="   ")
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == 1


# --- 404 semantics -----------------------------------------------------------


def test_unknown_identifiers_return_404(client):
    response = _compare(client)
    assert response.status_code == 404


def test_one_unknown_side_is_the_same_404(client):
    created = _create(client, name="release")
    unknown_left = _compare(client, left=ZERO_UUID,
                            right=created["policy_id"])
    unknown_right = _compare(client, left=created["policy_id"],
                             right=ZERO_UUID)
    assert unknown_left.status_code == 404
    assert unknown_right.status_code == 404
    assert unknown_left.text == unknown_right.text


def test_cross_scope_identifier_is_indistinguishable_404(client):
    created = _create(client, name="release")
    other = _create(client, name="release", tenant=OTHER_TENANT,
                    workload=OTHER_WORKLOAD)
    unknown = _compare(client)
    for response in (
        _compare(client, tenant=OTHER_TENANT, left=created["policy_id"],
                 right=created["policy_id"]),
        _compare(client, workload=OTHER_WORKLOAD, left=created["policy_id"],
                 right=created["policy_id"]),
        _compare(client, left=created["policy_id"], right=other["policy_id"]),
        _compare(client, left=other["policy_id"], right=created["policy_id"]),
    ):
        assert response.status_code == 404
        assert response.text == unknown.text


# --- identical comparisons ---------------------------------------------------


def test_self_comparison_is_identical(client):
    created = _create(client, name="release")
    response = _compare(client, left=created["policy_id"],
                        right=created["policy_id"])
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["identical"] is True
    assert data["changes"] == []
    assert data["left"] == data["right"]
    assert data["left"]["policy_id"] == created["policy_id"]


def test_metadata_never_affects_identity(client):
    rule = {"all": [{"claim": "m", "equals": "abc"}, {"claim": "n", "gte": 2}]}
    left = _create(client, rule=rule, name="one")
    # Same rule under a different name (and therefore version lineage).
    right = _create(client, rule=rule, name="two")
    response = _compare(client, left=left["policy_id"],
                        right=right["policy_id"])
    data = response.json()
    assert data["identical"] is True
    assert data["changes"] == []
    assert data["left"]["name"] == "one"
    assert data["right"]["name"] == "two"


def test_object_key_order_is_insignificant(client):
    left = _create(client, rule={"claim": "m", "equals": "abc"}, name="one")
    # Same leaf, different key order in the submitted JSON.
    response = client.post(
        "/v1/policies",
        content=(
            b'{"tenant_id":"%s","workload_id":"%s","name":"two",'
            b'"rule":{"equals":"abc","claim":"m"}}'
            % (TENANT.encode(), WORKLOAD.encode())
        ),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 201, response.text
    reordered = response.json()
    response = _compare(client, left=left["policy_id"],
                        right=reordered["policy_id"])
    assert response.json()["identical"] is True
    assert response.json()["changes"] == []


def test_numbers_compare_by_value_booleans_never(client):
    _, _, response = _pair(
        client,
        {"claim": "n", "equals": 1},
        {"claim": "n", "equals": 1.0},
    )
    assert response.json()["identical"] is True

    _, _, response = _pair(
        client,
        {"claim": "n", "equals": True},
        {"claim": "n", "equals": 1},
    )
    data = response.json()
    assert data["identical"] is False
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [],
            "left": {"claim": "n", "equals": True},
            "right": {"claim": "n", "equals": 1},
        }
    ]


def test_in_candidate_order_is_positional(client):
    _, _, response = _pair(
        client,
        {"claim": "n", "in": [1, 2]},
        {"claim": "n", "in": [2, 1]},
    )
    data = response.json()
    assert data["identical"] is False
    assert len(data["changes"]) == 1
    assert data["changes"][0]["change"] == "changed"
    assert data["changes"][0]["rule_path"] == []


# --- change recording --------------------------------------------------------


def test_leaf_change_is_recorded_at_root(client):
    _, _, response = _pair(
        client,
        {"claim": "m", "equals": "abc"},
        {"claim": "m", "equals": "def"},
    )
    data = response.json()
    assert data["identical"] is False
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [],
            "left": {"claim": "m", "equals": "abc"},
            "right": {"claim": "m", "equals": "def"},
        }
    ]


def test_different_node_forms_record_changed_without_descending(client):
    _, _, response = _pair(
        client,
        {"all": [{"claim": "m", "equals": "abc"}]},
        {"any": [{"claim": "m", "equals": "abc"}]},
    )
    data = response.json()
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [],
            "left": {"all": [{"claim": "m", "equals": "abc"}]},
            "right": {"any": [{"claim": "m", "equals": "abc"}]},
        }
    ]


def test_only_the_first_differing_child_is_recorded(client):
    left_rule = {
        "all": [
            {"claim": "a", "equals": 1},
            {"claim": "b", "equals": 2},
        ]
    }
    right_rule = {
        "all": [
            {"claim": "a", "equals": 9},
            {"claim": "b", "equals": 8},
        ]
    }
    _, _, response = _pair(client, left_rule, right_rule)
    data = response.json()
    # Both children differ; only the first is reported.
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [0],
            "left": {"claim": "a", "equals": 1},
            "right": {"claim": "a", "equals": 9},
        }
    ]


def test_nested_difference_records_the_first_differing_child_only(client):
    shared = {"claim": "shared", "exists": True}
    left_rule = {
        "all": [
            {"any": [{"claim": "a", "equals": 1}]},
            shared,
        ]
    }
    right_rule = {
        "all": [
            {"any": [{"claim": "a", "equals": 2}]},
            shared,
        ]
    }
    _, _, response = _pair(client, left_rule, right_rule)
    data = response.json()
    # Same form and child count at every level above the leaf: the diff
    # descends the first differing child and records where the leaf
    # values finally differ.
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [0, 0],
            "left": {"claim": "a", "equals": 1},
            "right": {"claim": "a", "equals": 2},
        },
    ]


def test_not_descends_into_its_single_child(client):
    _, _, response = _pair(
        client,
        {"not": {"claim": "a", "equals": 1}},
        {"not": {"claim": "a", "equals": 2}},
    )
    data = response.json()
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [0],
            "left": {"claim": "a", "equals": 1},
            "right": {"claim": "a", "equals": 2},
        }
    ]


def test_extra_right_children_are_itemized_added(client):
    left_rule = {"all": [{"claim": "a", "equals": 1}]}
    right_rule = {
        "all": [
            {"claim": "a", "equals": 1},
            {"claim": "b", "equals": 2},
            {"claim": "c", "equals": 3},
        ]
    }
    _, _, response = _pair(client, left_rule, right_rule)
    data = response.json()
    assert data["identical"] is False
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [],
            "left": left_rule,
            "right": right_rule,
        },
        {
            "change": "added",
            "rule_path": [1],
            "left": None,
            "right": {"claim": "b", "equals": 2},
        },
        {
            "change": "added",
            "rule_path": [2],
            "left": None,
            "right": {"claim": "c", "equals": 3},
        },
    ]


def test_extra_left_children_are_itemized_removed(client):
    left_rule = {
        "any": [
            {"claim": "a", "equals": 1},
            {"claim": "b", "equals": 2},
        ]
    }
    right_rule = {"any": [{"claim": "a", "equals": 1}]}
    _, _, response = _pair(client, left_rule, right_rule)
    data = response.json()
    assert data["changes"] == [
        {
            "change": "changed",
            "rule_path": [],
            "left": left_rule,
            "right": right_rule,
        },
        {
            "change": "removed",
            "rule_path": [1],
            "left": {"claim": "b", "equals": 2},
            "right": None,
        },
    ]


# --- response shape ------------------------------------------------------------


def test_response_root_and_side_key_order(client):
    left = _create(client, name="one")
    right = _create(client, name="two")
    response = _compare(client, left=left["policy_id"],
                        right=right["policy_id"])
    assert response.status_code == 200
    data = response.json()
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
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD


def test_change_entry_key_order(client):
    _, _, response = _pair(
        client,
        {"claim": "m", "equals": "abc"},
        {"claim": "m", "equals": "def"},
    )
    entry = response.json()["changes"][0]
    assert list(entry.keys()) == ["change", "rule_path", "left", "right"]


def test_response_is_compact_json_with_single_newline(client):
    created = _create(client, name="release")
    response = _compare(client, left=created["policy_id"],
                        right=created["policy_id"])
    raw = response.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert raw[:-1] == json.dumps(
        response.json(), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def test_side_metadata_reports_lifecycle_fields(client):
    left = _create(client, name="one")
    right = _create(client, name="two")
    response = _compare(client, left=left["policy_id"],
                        right=right["policy_id"])
    data = response.json()
    assert data["left"]["version"] == 1
    assert data["left"]["status"] == "active"
    assert data["left"]["retired_at"] is None
    assert data["left"]["created_at"] == left["created_at"]
    assert data["right"]["name"] == "two"


# --- retired versions and read-only behaviour --------------------------------


def test_retired_version_compares_without_changing_state(app, client):
    left = _create(client, name="one")
    right = _create(client, rule={"claim": "m", "equals": "def"}, name="two")
    retired = _retire(client, right["policy_id"])

    response = _compare(client, left=left["policy_id"],
                        right=right["policy_id"])
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["right"]["status"] == "retired"
    assert data["right"]["retired_at"] == retired["retired_at"]
    assert data["identical"] is False

    # The comparison is read-only: the retirement stands, nothing was
    # rewritten, and a repeat retire still conflicts.
    again = client.post(
        f"/v1/policies/{right['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert again.status_code == 409
    with app.state.session_factory() as session:
        row = session.get(Policy, right["policy_id"])
        assert row.status == "retired"


def test_comparison_writes_no_state(app, client):
    left = _create(client, name="one")
    right = _create(client, rule={"claim": "m", "equals": "def"}, name="two")
    with app.state.session_factory() as session:
        counter_before = session.query(PolicyCommitCounter).count()
        policies_before = session.query(Policy).count()

    first = _compare(client, left=left["policy_id"], right=right["policy_id"])
    second = _compare(client, left=left["policy_id"], right=right["policy_id"])
    assert first.status_code == second.status_code == 200
    assert first.content == second.content

    with app.state.session_factory() as session:
        assert session.query(Policy).count() == policies_before
        assert session.query(PolicyCommitCounter).count() == counter_before


# --- diff_rules / rules_equal unit semantics -----------------------------------


def test_rules_equal_scalar_semantics():
    assert rules_equal({"claim": "n", "equals": 1}, {"claim": "n", "equals": 1.0})
    assert not rules_equal(
        {"claim": "n", "equals": True}, {"claim": "n", "equals": 1}
    )
    assert not rules_equal(
        {"claim": "n", "equals": None}, {"claim": "n", "equals": False}
    )
    assert rules_equal(
        {"a": {"claim": "x", "equals": "v"}}["a"],
        {"claim": "x", "equals": "v"},
    )


def test_diff_rules_empty_when_equal():
    rule = {"all": [{"claim": "a", "equals": 1}, {"not": {"claim": "b", "exists": True}}]}
    assert diff_rules(rule, json.loads(json.dumps(rule))) == []


def test_diff_rules_entries_are_stably_sorted():
    left = {
        "all": [
            {"claim": "a", "equals": 1},
            {"claim": "b", "equals": 2},
            {"claim": "c", "equals": 3},
        ]
    }
    right = {"all": [{"claim": "a", "equals": 1}]}
    changes = diff_rules(left, right)
    assert [entry["rule_path"] for entry in changes] == [[], [1], [2]]
    assert [entry["change"] for entry in changes] == [
        "changed",
        "removed",
        "removed",
    ]
