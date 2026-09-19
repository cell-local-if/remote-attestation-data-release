"""Tests for POST /v1/policies."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from proof_release.app import create_app
from proof_release.db import Policy

TENANT = "tenant-a"
WORKLOAD = "workload-1"


@pytest.fixture()
def app(tmp_path):
    return create_app(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "name": "release-policy",
        "rule": {"claim": "measurement", "equals": "abc"},
    }
    body.update(overrides)
    return client.post("/v1/policies", json=body)


def test_create_policy_returns_201_with_scope_version_rule_and_time(client):
    response = _create(client)

    assert response.status_code == 201
    data = response.json()
    assert data["policy_id"]
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["name"] == "release-policy"
    assert data["version"] == 1
    assert data["rule"] == {"claim": "measurement", "equals": "abc"}
    created_at = datetime.fromisoformat(data["created_at"])
    assert created_at.utcoffset() == timedelta(0)


def test_same_scope_and_name_increments_version_and_keeps_old(client, app):
    first = _create(client).json()
    second = _create(client, rule={"claim": "measurement", "equals": "def"}).json()
    third = _create(client).json()

    assert second["version"] == 2
    assert third["version"] == 3
    assert len({first["policy_id"], second["policy_id"], third["policy_id"]}) == 3

    with app.state.session_factory() as session:
        policies = session.scalars(
            select(Policy).where(Policy.name == "release-policy")
        ).all()
    assert sorted(p.version for p in policies) == [1, 2, 3]
    # Old versions are retained with their original rules.
    by_version = {p.version: p for p in policies}
    assert by_version[1].policy_id == first["policy_id"]
    assert '"def"' in by_version[2].rule


def test_version_is_scoped_per_tenant_workload_and_name(client):
    assert _create(client).json()["version"] == 1
    assert _create(client, name="other-policy").json()["version"] == 1
    assert _create(client, tenant_id="tenant-b").json()["version"] == 1
    assert _create(client, workload_id="workload-2").json()["version"] == 1
    assert _create(client).json()["version"] == 2


@pytest.mark.parametrize(
    "rule",
    [
        {"claim": "m", "equals": "abc"},
        {"claim": "m", "equals": 42},
        {"claim": "m", "equals": 4.2},
        {"claim": "m", "equals": True},
        {"claim": "m", "equals": None},
        {"all": [{"claim": "a", "equals": 1}, {"claim": "b", "equals": "x"}]},
        {"any": [{"claim": "a", "equals": 1}]},
        {"not": {"claim": "a", "equals": 1}},
        {"all": [{"any": [{"not": {"claim": "a", "equals": 1}}]}]},
    ],
)
def test_valid_rules_are_accepted(client, rule):
    response = _create(client, rule=rule)

    assert response.status_code == 201
    assert response.json()["rule"] == rule


@pytest.mark.parametrize(
    "rule",
    [
        None,
        "claim",
        42,
        [],
        {},
        {"claim": "m"},  # missing equals
        {"equals": 1},  # missing claim
        {"claim": "", "equals": 1},
        {"claim": "   ", "equals": 1},
        {"claim": 7, "equals": 1},
        {"claim": "m", "equals": [1]},  # non-scalar equals
        {"claim": "m", "equals": {"x": 1}},
        {"claim": "m", "equals": 1, "extra": 2},
        {"all": []},  # empty subrule list
        {"any": []},
        {"all": "not-a-list"},
        {"all": [{"claim": "m", "equals": 1}, "junk"]},
        {"any": [{"bad": 1}]},
        {"not": {}},
        {"not": [{"claim": "m", "equals": 1}]},
        {"unknown": {"claim": "m", "equals": 1}},
        {"all": [{"claim": "m", "equals": 1}], "any": [{"claim": "m", "equals": 2}]},
    ],
)
def test_invalid_rules_return_422(client, rule):
    response = _create(client, rule=rule)

    assert response.status_code == 422


def test_deeply_nested_rule_returns_422(client):
    rule = {"claim": "m", "equals": 1}
    for _ in range(40):
        rule = {"not": rule}

    response = _create(client, rule=rule)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"workload_id": ""},
        {"workload_id": "  "},
        {"name": ""},
        {"name": "  "},
        {"name": 3},
    ],
)
def test_invalid_fields_return_422(client, overrides):
    response = _create(client, **overrides)

    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "name", "rule"])
def test_missing_fields_return_422(client, missing):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "name": "release-policy",
        "rule": {"claim": "m", "equals": 1},
    }
    del body[missing]

    response = client.post("/v1/policies", json=body)

    assert response.status_code == 422


def test_policies_survive_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    created = _create(TestClient(app1)).json()
    assert created["version"] == 1
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    followup = _create(client2).json()
    assert followup["version"] == 2
    with app2.state.session_factory() as session:
        policies = session.scalars(select(Policy)).all()
    assert sorted(p.version for p in policies) == [1, 2]


def test_concurrent_creates_never_share_a_version(app):
    def create():
        return TestClient(app).post(
            "/v1/policies",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "name": "contended",
                "rule": {"claim": "m", "equals": 1},
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: create(), range(16)))

    assert all(r.status_code == 201 for r in responses)
    versions = sorted(r.json()["version"] for r in responses)
    assert versions == list(range(1, 17))
    assert len({r.json()["policy_id"] for r in responses}) == 16
