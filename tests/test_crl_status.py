"""Tests for the read-only CRL freshness status endpoint
GET /v1/crls/status."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select

from proof_release.app import create_app
from proof_release.db import (
    CertificateRevocationList,
    CrlRevokedCertificate,
)

from x509_helpers import make_crl, make_root, pem

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/test.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def chain():
    root_key, root_cert = make_root()
    return {"root_key": root_key, "root_cert": root_cert}


@pytest.fixture()
def root_id(client, chain):
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    assert response.status_code == 201
    return response.json()["root_id"]


def _register_crl(client, root_id_value, crl_pem, **scope):
    payload = {
        "tenant_id": scope.get("tenant_id", TENANT),
        "workload_id": scope.get("workload_id", WORKLOAD),
        "trust_root_id": root_id_value,
        "crl_pem": crl_pem,
    }
    return client.post("/v1/crls", json=payload)


def _get_status(client, **params):
    query = {
        "tenant_id": params.pop("tenant_id", TENANT),
        "workload_id": params.pop("workload_id", WORKLOAD),
        "trust_root_id": params.pop("trust_root_id", None),
    }
    query.update(params)
    query = {key: value for key, value in query.items() if value is not None}
    return client.get("/v1/crls/status", params=query)


@pytest.fixture()
def registered(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [(1001, now - timedelta(hours=1))],
        number=3,
    )
    created = _register_crl(client, root_id, crl_pem)
    assert created.status_code == 201
    return {
        "crl_id": created.json()["crl_id"],
        "root_id": root_id,
        "this_update": created.json()["this_update"],
        "next_update": created.json()["next_update"],
    }


# --- success path -----------------------------------------------------------


def test_status_missing_when_root_has_no_crl(client, root_id):
    response = _get_status(client, trust_root_id=root_id)
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == [
        "tenant_id",
        "workload_id",
        "trust_root_id",
        "observed_at",
        "state",
        "current_crl_id",
        "crl_number",
        "this_update",
        "next_update",
    ]
    assert body["tenant_id"] == TENANT
    assert body["workload_id"] == WORKLOAD
    assert body["trust_root_id"] == root_id
    assert body["observed_at"].endswith("+00:00")
    assert body["state"] == "missing"
    assert body["current_crl_id"] is None
    assert body["crl_number"] is None
    assert body["this_update"] is None
    assert body["next_update"] is None

    # Compact JSON terminated by exactly one newline.
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert b", " not in response.content
    assert b": " not in response.content


def test_status_fresh_with_current_snapshot(client, registered):
    response = _get_status(client, trust_root_id=registered["root_id"])
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "fresh"
    assert body["current_crl_id"] == registered["crl_id"]
    assert body["crl_number"] == 3
    assert isinstance(body["crl_number"], int)
    assert body["this_update"] == registered["this_update"]
    assert body["next_update"] == registered["next_update"]
    assert body["observed_at"] < body["next_update"]

    # No CRL body, entries, digest or certificate material is exposed.
    assert "BEGIN X509 CRL" not in response.text
    assert "BEGIN CERTIFICATE" not in response.text
    assert "crl_sha256" not in body
    assert "entries" not in body
    assert "1001" not in response.text


def test_status_stale_once_next_update_passes(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [],
        number=1,
        next_update=now + timedelta(seconds=2),
    )
    created = _register_crl(client, root_id, crl_pem)
    assert created.status_code == 201

    fresh = _get_status(client, trust_root_id=root_id)
    assert fresh.status_code == 200
    assert fresh.json()["state"] == "fresh"

    threading.Event().wait(2.5)
    stale = _get_status(client, trust_root_id=root_id)
    assert stale.status_code == 200
    body = stale.json()
    assert body["state"] == "stale"
    assert body["current_crl_id"] == created.json()["crl_id"]
    assert body["crl_number"] == 1
    assert body["observed_at"] >= body["next_update"]


def test_status_uses_highest_crl_number_snapshot(client, root_id, chain):
    now = datetime.now(timezone.utc)
    first = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    assert _register_crl(client, root_id, first).status_code == 201
    second = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [],
        number=2,
        this_update=now - timedelta(minutes=5),
    )
    created = _register_crl(client, root_id, second)
    assert created.status_code == 201

    response = _get_status(client, trust_root_id=root_id)
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "fresh"
    assert body["current_crl_id"] == created.json()["crl_id"]
    assert body["crl_number"] == 2


def test_status_is_read_only_and_repeatable(client, app, registered):
    first = _get_status(client, trust_root_id=registered["root_id"])
    second = _get_status(client, trust_root_id=registered["root_id"])
    assert first.status_code == second.status_code == 200
    assert first.json()["state"] == second.json()["state"] == "fresh"
    with app.state.session_factory() as session:
        snapshots = session.scalars(select(CertificateRevocationList)).all()
        entries = session.scalars(select(CrlRevokedCertificate)).all()
    assert len(snapshots) == 1
    assert len(entries) == 1

    # The snapshot query endpoint is unaffected by the status reads.
    snapshot = client.get(
        f"/v1/crls/{registered['crl_id']}",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": registered["root_id"],
        },
    )
    assert snapshot.status_code == 200
    assert snapshot.json()["crl_number"] == 3


# --- request shape (422) ----------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_id", ""),
        ("tenant_id", "   "),
        ("tenant_id", f" {TENANT}"),
        ("tenant_id", f"{TENANT} "),
        ("workload_id", ""),
        ("workload_id", "\t"),
        ("workload_id", f" {WORKLOAD}"),
        ("workload_id", f"{WORKLOAD}\t"),
        ("trust_root_id", ""),
        ("trust_root_id", "   "),
        ("trust_root_id", "not-a-uuid"),
        ("trust_root_id", "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"),
    ],
)
def test_invalid_scope_parameters_are_422(client, registered, field, value):
    params = {"trust_root_id": registered["root_id"], field: value}
    response = _get_status(client, **params)
    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "trust_root_id"])
def test_missing_scope_parameter_is_422(client, registered, missing):
    params = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": registered["root_id"],
    }
    del params[missing]
    response = client.get("/v1/crls/status", params=params)
    assert response.status_code == 422


def test_extra_query_parameter_is_422(client, registered):
    response = _get_status(
        client, trust_root_id=registered["root_id"], include_entries="true"
    )
    assert response.status_code == 422


def test_repeated_query_parameter_is_422(client, registered):
    response = client.get(
        f"/v1/crls/status?tenant_id={TENANT}&tenant_id={TENANT}"
        f"&workload_id={WORKLOAD}&trust_root_id={registered['root_id']}"
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"{}", b" ", b"not-json"])
def test_non_empty_body_is_422(client, registered, body):
    response = client.request(
        "GET",
        "/v1/crls/status",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": registered["root_id"],
        },
        content=body,
    )
    assert response.status_code == 422


def test_shape_failures_read_no_state(client, app, registered):
    # A 422 response is produced before storage is touched: even with the
    # database made unusable, the rejection stands.
    engine = app.state.engine

    def fail_all(conn, cursor, statement, parameters, context, executemany):
        raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_all)
    try:
        response = _get_status(
            client, trust_root_id=registered["root_id"], tenant_id=" padded "
        )
        assert response.status_code == 422
    finally:
        event.remove(engine, "before_cursor_execute", fail_all)


# --- not found (indistinguishable 404) --------------------------------------


def test_unknown_trust_root_is_404(client, registered):
    response = _get_status(
        client, trust_root_id="11111111-1111-1111-1111-111111111111"
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "trust root not found"


@pytest.mark.parametrize(
    "override",
    [
        {"tenant_id": OTHER_TENANT},
        {"workload_id": OTHER_WORKLOAD},
    ],
)
def test_cross_scope_is_indistinguishable_404(client, registered, override):
    response = _get_status(client, trust_root_id=registered["root_id"], **override)
    assert response.status_code == 404
    assert response.json()["detail"] == "trust root not found"


def test_root_of_other_tenant_is_404(client, chain):
    roots = {}
    for tenant in (TENANT, OTHER_TENANT):
        response = client.post(
            "/v1/trust-roots",
            json={
                "tenant_id": tenant,
                "workload_id": WORKLOAD,
                "root_pem": pem(chain["root_cert"]),
            },
        )
        assert response.status_code == 201
        roots[tenant] = response.json()["root_id"]

    # The other tenant cannot observe the root's status, even though the
    # same certificate is configured under its own scope.
    response = _get_status(
        client, tenant_id=OTHER_TENANT, trust_root_id=roots[TENANT]
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "trust root not found"

    # Each tenant observes only its own root.
    own = _get_status(client, tenant_id=OTHER_TENANT, trust_root_id=roots[OTHER_TENANT])
    assert own.status_code == 200
    assert own.json()["state"] == "missing"


# --- storage failure --------------------------------------------------------


def test_storage_failure_is_500(app, client, registered):
    engine = app.state.engine

    def fail_select(conn, cursor, statement, parameters, context, executemany):
        if "FROM certificate_revocation_lists" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_select)
    try:
        response = _get_status(client, trust_root_id=registered["root_id"])
        assert response.status_code == 500
        assert response.json()["detail"] == "CRL registry unavailable"
    finally:
        event.remove(engine, "before_cursor_execute", fail_select)

    # The failure left the registered snapshot intact and observable.
    recovered = _get_status(client, trust_root_id=registered["root_id"])
    assert recovered.status_code == 200
    assert recovered.json()["state"] == "fresh"
