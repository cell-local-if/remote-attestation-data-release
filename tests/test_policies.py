"""Tests for POST /v1/policies (versioned, isolated policies)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Policy

TENANT = "tenant-a"
WORKLOAD = "workload-1"

LEAF = {"claim": "measurement", "equals": "abc"}


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
        {"all": [LEAF], "any": [LEAF]},  # two forms at once
        {"not": {"not": {"claim": "x", "equals": 1, "bogus": 2}}},
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
