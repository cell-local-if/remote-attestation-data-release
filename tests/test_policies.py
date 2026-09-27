"""Tests for POST /v1/policies (versioned, isolated policies)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Policy

TENANT = "tenant-a"
WORKLOAD = "workload-1"

LEAF = {"claim": "measurement", "equals": "abc"}
PATH_LEAF = {"path": ["tenant", "region"], "equals": "eu"}


@pytest.fixture()
def app(tmp_path):
    return create_app(f"sqlite:///{tmp_path}/policies.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, rule=LEAF, name="release", **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "name": name,
        "rule": rule,
    }
    body.update(overrides)
    return client.post("/v1/policies", json=body)


def test_create_policy_returns_201_with_version_and_timestamp(client):
    response = _create(client)

    assert response.status_code == 201
    data = response.json()
    assert data["policy_id"]
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["name"] == "release"
    assert data["version"] == 1
    assert data["rule"] == LEAF
    created_at = datetime.fromisoformat(data["created_at"])
    assert created_at.utcoffset() == timedelta(0)


def test_same_scope_and_name_increments_version_and_keeps_old(client, app):
    first = _create(client, rule=LEAF)
    assert first.status_code == 201
    second_rule = {"all": [LEAF, {"claim": "tier", "equals": 2}]}
    second = _create(client, rule=second_rule)
    assert second.status_code == 201

    assert second.json()["version"] == 2
    assert second.json()["rule"] == second_rule
    assert second.json()["policy_id"] != first.json()["policy_id"]

    with app.state.session_factory() as session:
        rows = session.query(Policy).order_by(Policy.version).all()
        assert [(r.name, r.version) for r in rows] == [("release", 1), ("release", 2)]


def test_other_name_or_scope_starts_at_version_1(client):
    assert _create(client, name="a").json()["version"] == 1
    assert _create(client, name="b").json()["version"] == 1
    assert _create(
        client, name="a", tenant_id="tenant-b", rule=LEAF
    ).json()["version"] == 1
    assert _create(
        client, name="a", workload_id="workload-2", rule=LEAF
    ).json()["version"] == 1
    assert _create(client, name="a").json()["version"] == 2


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "name", "rule"])
def test_create_policy_requires_all_fields(client, missing):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "name": "release",
        "rule": LEAF,
    }
    del body[missing]
    assert client.post("/v1/policies", json=body).status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"name": ""},
        {"name": "  "},
        {"tenant_id": 1},
    ],
)
def test_create_policy_rejects_blank_or_typed_fields(client, overrides):
    assert _create(client, **overrides).status_code == 422


@pytest.mark.parametrize(
    "rule",
    [
        "not-an-object",
        42,
        ["claim", "measurement"],
        True,
        None,
        {},
        {"claim": "measurement"},  # missing equals
        {"equals": "abc"},  # missing claim
        {"claim": "", "equals": "abc"},  # empty claim name
        {"claim": "measurement", "equals": [1, 2]},  # non-scalar equals
        {"claim": "measurement", "equals": {"x": 1}},
        {"claim": 1, "equals": "abc"},  # non-string claim
        {"all": []},  # empty list
        {"any": []},
        {"all": {}},  # not a list
        {"not": []},  # not a rule
        {"all": "x"},
        {"foo": 1},  # unknown form
        {"claim": "a", "equals": 1, "extra": 2},  # extra key on leaf
        {"path": ["a"], "equals": 1, "extra": 2},  # extra key on path leaf
        {"all": [LEAF], "any": [LEAF]},  # two forms at once
        {"not": {"not": {"claim": "x", "equals": 1, "bogus": 2}}},
        # Path leaf validation: missing/empty/blank/wrong-typed path.
        {"equals": "eu"},  # missing path (and claim)
        {"path": ["tenant"]},  # missing equals
        {"path": [], "equals": "eu"},  # empty path
        {"path": "tenant", "equals": "eu"},  # path not a list
        {"path": ["tenant", 7], "equals": "eu"},  # non-string segment
        {"path": ["tenant", None], "equals": "eu"},  # null segment
        {"path": ["tenant", ""], "equals": "eu"},  # empty segment
        {"path": ["tenant", "  "], "equals": "eu"},  # whitespace segment
        {"path": [[]], "equals": "eu"},  # segment of the wrong type
        {"path": [{}], "equals": "eu"},
        # Over the eight-segment bound.
        {"path": ["a", "b", "c", "d", "e", "f", "g", "h", "i"], "equals": 1},
        # A single segment longer than 128 characters.
        {"path": ["x" * 129], "equals": 1},
        # equals must stay a scalar on path leaves.
        {"path": ["tenant"], "equals": {"region": "eu"}},
        {"path": ["tenant"], "equals": [1, 2]},
        # A path cannot be mixed with a claim on the same node, nor carry
        # a nested rule where equals/segments belong.
        {"path": ["tenant"], "claim": "tenant", "equals": "eu"},
        {"all": [{"path": ["a"]}]},  # path leaf missing equals inside all
    ],
)
def test_create_policy_rejects_rules_outside_the_four_forms(client, rule):
    response = _create(client, rule=rule)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "rule",
    [
        {"claim": "measurement", "equals": "abc"},
        {"claim": "count", "equals": 3},
        {"claim": "enabled", "equals": True},
        {"claim": "note", "equals": None},
        {"all": [LEAF]},
        {"any": [LEAF, {"claim": "x", "equals": 1}]},
        {"not": LEAF},
        {
            "all": [
                {"any": [{"claim": "a", "equals": 1}, {"claim": "b", "equals": 2}]},
                {"not": {"claim": "c", "equals": False}},
            ]
        },
        # Path leaves, including the segment-count and segment-length
        # boundaries, a dotted segment (one literal field name), scalar
        # types other than strings, and mixing with the old leaf form.
        PATH_LEAF,
        {"path": ["tenant"], "equals": "eu"},
        {"path": ["a", "b", "c", "d", "e", "f", "g", "h"], "equals": None},
        {"path": ["x" * 128], "equals": 3},
        {"path": ["a.b"], "equals": True},
        {"path": ["nested", "deep", "value"], "equals": -1.5},
        {
            "all": [
                {"claim": "measurement", "equals": "abc"},
                {"path": ["tenant", "region"], "equals": "eu"},
                {"not": {"path": ["tenant", "blocked"], "equals": True}},
            ]
        },
    ],
)
def test_create_policy_accepts_the_four_rule_forms(client, rule):
    assert _create(client, rule=rule).status_code == 201


def test_policies_persist_across_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    first = _create(client1)
    assert first.status_code == 201
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    second = _create(client2)
    assert second.json()["version"] == 2
    app2.state.engine.dispose()


def test_concurrent_policy_creation_never_reuses_versions(app):
    def create():
        return TestClient(app).post(
            "/v1/policies",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "name": "release",
                "rule": LEAF,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: create(), range(20)))

    assert all(r.status_code == 201 for r in responses)
    versions = sorted(r.json()["version"] for r in responses)
    assert versions == list(range(1, 21))
    ids = [r.json()["policy_id"] for r in responses]
    assert len(set(ids)) == 20


@pytest.mark.parametrize(
    "rule",
    [
        {"path": [], "equals": "eu"},
        {"path": ["a", "b", "c", "d", "e", "f", "g", "h", "i"], "equals": 1},
        {"path": ["x" * 129], "equals": 1},
        {"path": ["ok"], "equals": {"nested": 1}},
        {"path": ["ok", ""], "equals": 1},
    ],
)
def test_invalid_path_rule_allocates_no_version_and_writes_no_state(
    client, app, rule
):
    # A valid first version exists so a silent allocation of version 2
    # would be observable.
    assert _create(client, rule=LEAF).status_code == 201

    rejected = _create(client, rule=rule)
    assert rejected.status_code == 422

    # A subsequent legal create still takes exactly version 2: the rejected
    # request neither allocated a version nor inserted a row.
    after = _create(client, rule=PATH_LEAF)
    assert after.status_code == 201
    assert after.json()["version"] == 2
    with app.state.session_factory() as session:
        rows = session.query(Policy).order_by(Policy.version).all()
        assert [r.version for r in rows] == [1, 2]


def test_path_rule_persists_across_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart-path.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    first = _create(client1, rule=PATH_LEAF)
    assert first.status_code == 201
    policy_id = first.json()["policy_id"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        row = session.get(Policy, policy_id)
        assert row is not None
        assert json.loads(row.rule_json) == PATH_LEAF
    app2.state.engine.dispose()
