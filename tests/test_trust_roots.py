"""Tests for POST /v1/trust-roots."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import TrustRoot

from x509_helpers import generate_key, make_leaf, make_root, pem, private_key_pem

TENANT = "tenant-a"
WORKLOAD = "workload-1"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/test.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def root_pem():
    _, certificate = make_root()
    return pem(certificate)


def _create(client, root_pem_value, **overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD, "root_pem": root_pem_value}
    body.update(overrides)
    return client.post("/v1/trust-roots", json=body)


def test_create_trust_root_returns_201_with_fields(client, root_pem):
    response = _create(client, root_pem, name="primary-root")

    assert response.status_code == 201
    data = response.json()
    assert data["root_id"]
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["name"] == "primary-root"
    created_at = datetime.fromisoformat(data["created_at"])
    assert created_at.utcoffset() == timedelta(0)


def test_create_trust_root_name_is_optional(client, root_pem):
    response = _create(client, root_pem)

    assert response.status_code == 201
    assert response.json()["name"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"workload_id": ""},
        {"workload_id": "   "},
        {"root_pem": ""},
        {"root_pem": "   "},
        {"root_pem": 42},
        {"name": ""},
        {"name": "   "},
        {"name": 42},
    ],
)
def test_create_trust_root_rejects_invalid_fields(client, root_pem, overrides):
    response = _create(client, root_pem, **overrides)

    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "root_pem"])
def test_create_trust_root_requires_fields(client, root_pem, missing):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD, "root_pem": root_pem}
    del body[missing]

    response = client.post("/v1/trust-roots", json=body)

    assert response.status_code == 422


def test_unparseable_pem_returns_422(client):
    response = _create(client, "this is not a pem block")

    assert response.status_code == 422


def test_private_key_pem_returns_422_and_stores_nothing(client, app):
    key_pem = private_key_pem(generate_key())

    response = _create(client, key_pem)

    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.query(TrustRoot).count() == 0


def test_non_ca_certificate_returns_422(client):
    root_key, root_cert = make_root()
    _, leaf = make_leaf(root_cert, root_key)

    response = _create(client, pem(leaf))

    assert response.status_code == 422


def test_duplicate_trust_root_returns_409(client, root_pem):
    first = _create(client, root_pem)
    assert first.status_code == 201

    second = _create(client, root_pem)
    assert second.status_code == 409


def test_same_certificate_allowed_for_other_tenant_or_workload(client, root_pem):
    assert _create(client, root_pem).status_code == 201

    other_tenant = _create(client, root_pem, tenant_id="tenant-b")
    assert other_tenant.status_code == 201
    other_workload = _create(client, root_pem, workload_id="workload-2")
    assert other_workload.status_code == 201

    # But the exact same tenant/workload pair still conflicts.
    assert _create(client, root_pem).status_code == 409


def test_distinct_certificates_for_same_workload_get_distinct_ids(client):
    _, first = make_root("root-one")
    _, second = make_root("root-two")

    first_response = _create(client, pem(first))
    second_response = _create(client, pem(second))

    assert first_response.status_code == 201
    assert second_response.status_code == 201
    assert first_response.json()["root_id"] != second_response.json()["root_id"]


def test_trust_roots_persist_across_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    _, certificate = make_root()
    created = _create(client1, pem(certificate), name="durable-root")
    assert created.status_code == 201
    root_id = created.json()["root_id"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        record = session.get(TrustRoot, root_id)
        assert record is not None
        assert record.tenant_id == TENANT
        assert record.workload_id == WORKLOAD
        assert record.name == "durable-root"
        assert record.created_at is not None

    # A duplicate is still rejected after the restart.
    client2 = TestClient(app2)
    assert _create(client2, pem(certificate)).status_code == 409
    app2.state.engine.dispose()


def test_no_private_key_material_is_stored_or_returned(client, app, root_pem):
    response = _create(client, root_pem)

    assert response.status_code == 201
    assert "PRIVATE KEY" not in response.text
    assert set(response.json()) == {
        "root_id",
        "tenant_id",
        "workload_id",
        "name",
        "created_at",
    }
    with app.state.session_factory() as session:
        record = session.get(TrustRoot, response.json()["root_id"])
        columns = {c.name: getattr(record, c.name) for c in record.__table__.columns}
    for name, value in columns.items():
        assert "PRIVATE KEY" not in str(value), f"key material in column {name}"
