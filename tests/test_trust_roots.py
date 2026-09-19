"""Tests for POST /v1/trust-roots."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import TrustRoot

from x509util import generate_key, make_leaf, make_root_ca, pem

TENANT = "tenant-a"
WORKLOAD = "workload-1"


@pytest.fixture()
def app(tmp_path):
    return create_app(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def root_pem():
    _, cert = make_root_ca()
    return pem(cert)


def _create(client, pem_value=None, **overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD, "root_pem": pem_value}
    body.update(overrides)
    return client.post("/v1/trust-roots", json=body)


def test_create_trust_root_returns_201_with_metadata(client, root_pem):
    response = _create(client, root_pem, name="primary-root")

    assert response.status_code == 201
    data = response.json()
    assert data["root_id"]
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["name"] == "primary-root"
    created_at = datetime.fromisoformat(data["created_at"])
    assert created_at.utcoffset() == timedelta(0)
    # The PEM itself is not echoed back.
    assert root_pem not in response.text


def test_create_trust_root_name_is_optional(client, root_pem):
    response = _create(client, root_pem)

    assert response.status_code == 201
    assert response.json()["name"] is None


def test_create_trust_root_persists(client, app, root_pem):
    created = _create(client, root_pem, name="stored").json()

    with app.state.session_factory() as session:
        record = session.get(TrustRoot, created["root_id"])
        assert record is not None
        assert record.tenant_id == TENANT
        assert record.workload_id == WORKLOAD
        assert record.name == "stored"
        assert record.root_pem == root_pem
        assert len(record.fingerprint_sha256) == 64


def test_duplicate_certificate_same_tenant_workload_returns_409(client, root_pem):
    assert _create(client, root_pem).status_code == 201

    duplicate = _create(client, root_pem)

    assert duplicate.status_code == 409


def test_same_certificate_allowed_for_other_tenant_or_workload(client, root_pem):
    assert _create(client, root_pem).status_code == 201

    assert _create(client, root_pem, tenant_id="tenant-b").status_code == 201
    assert _create(client, root_pem, workload_id="workload-2").status_code == 201


def test_different_certificate_same_scope_returns_201(client, root_pem):
    assert _create(client, root_pem).status_code == 201
    _, other = make_root_ca("other-root")

    assert _create(client, pem(other)).status_code == 201


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"workload_id": ""},
        {"workload_id": "  "},
        {"root_pem": ""},
        {"root_pem": "   "},
        {"root_pem": 42},
        {"root_pem": "not a pem"},
        {"name": ""},
        {"name": "   "},
        {"name": 7},
    ],
)
def test_invalid_fields_return_422(client, root_pem, overrides):
    response = _create(client, root_pem, **overrides)

    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "root_pem"])
def test_missing_required_fields_return_422(client, root_pem, missing):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD, "root_pem": root_pem}
    del body[missing]

    response = client.post("/v1/trust-roots", json=body)

    assert response.status_code == 422


def test_non_ca_certificate_returns_422(client):
    # An end-entity certificate (CA:FALSE) cannot anchor a chain.
    root_key, root_cert = make_root_ca()
    _, leaf = make_leaf("leaf", root_cert, root_key)

    response = _create(client, pem(leaf))

    assert response.status_code == 422


def test_private_key_pem_returns_422(client):
    from cryptography.hazmat.primitives import serialization

    key_pem = (
        generate_key()
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode("ascii")
    )

    response = _create(client, key_pem)

    assert response.status_code == 422


def test_trust_roots_survive_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"
    _, cert = make_root_ca()
    root_pem = pem(cert)
    app1 = create_app(url)
    client1 = TestClient(app1)
    created = _create(client1, root_pem, name="durable")
    assert created.status_code == 201
    first_body = created.json()
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    # The same certificate is still registered: a duplicate attempt conflicts.
    duplicate = _create(client2, root_pem)
    assert duplicate.status_code == 409
    with app2.state.session_factory() as session:
        record = session.get(TrustRoot, first_body["root_id"])
        assert record is not None
        assert record.tenant_id == TENANT
        assert record.workload_id == WORKLOAD
        assert record.name == "durable"
        assert record.root_pem == root_pem
