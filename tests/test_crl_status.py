"""Tests for the read-only CRL freshness status endpoint
GET /v1/crls/status."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import CertificateRevocationList

from x509_helpers import make_crl, pem, make_root

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


def _status(client, **params):
    query = {
        "tenant_id": params.pop("tenant_id", TENANT),
        "workload_id": params.pop("workload_id", WORKLOAD),
        "trust_root_id": params.pop("trust_root_id"),
    }
    query.update(params)
    return client.get("/v1/crls/status", params=query)


def _register_crl(client, root_id_value, crl_pem, **scope):
    payload = {
        "tenant_id": scope.get("tenant_id", TENANT),
        "workload_id": scope.get("workload_id", WORKLOAD),
        "trust_root_id": root_id_value,
        "crl_pem": crl_pem,
    }
    return client.post("/v1/crls", json=payload)


def _insert_snapshot(
    app,
    *,
    root_id_value,
    number,
    this_update,
    next_update,
    tenant_id=TENANT,
    workload_id=WORKLOAD,
    crl_id=None,
):
    """Insert an immutable snapshot row directly, bypassing receipt-time
    validation so that stale and exact-boundary windows are constructable."""
    import uuid

    crl_id = crl_id or str(uuid.uuid4())
    with app.state.session_factory() as session:
        session.add(
            CertificateRevocationList(
                crl_id=crl_id,
                tenant_id=tenant_id,
                workload_id=workload_id,
                trust_root_id=root_id_value,
                crl_number=number,
                crl_sha256=f"{number:0{64}x}",
                this_update=this_update,
                next_update=next_update,
                revoked_count=0,
                created_at=datetime.now(timezone.utc),
            )
        )
        session.commit()
    return crl_id


# --- success path -----------------------------------------------------------


def test_missing_state_when_root_has_no_crl(client, root_id):
    response = _status(client, trust_root_id=root_id)
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
    assert body["state"] == "missing"
    assert body["current_crl_id"] is None
    assert body["crl_number"] is None
    assert body["this_update"] is None
    assert body["next_update"] is None
    observed = datetime.fromisoformat(body["observed_at"])
    assert observed.tzinfo is not None
    # Compact JSON terminated by exactly one newline.
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert b", " not in response.content
    assert b": " not in response.content
    assert b"null" in response.content


def test_fresh_state_after_registration(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [(1001, now - timedelta(hours=1))],
        number=1,
    )
    created = _register_crl(client, root_id, crl_pem)
    assert created.status_code == 201
    expected = created.json()

    response = _status(client, trust_root_id=root_id)
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "fresh"
    assert body["current_crl_id"] == expected["crl_id"]
    assert body["crl_number"] == 1
    assert isinstance(body["crl_number"], int)
    assert body["this_update"] == expected["this_update"]
    assert body["next_update"] == expected["next_update"]

    # No CRL body, entries, digest, count or certificate material leaks.
    for forbidden in (
        "BEGIN X509 CRL",
        "BEGIN CERTIFICATE",
        "crl_sha256",
        "revoked_count",
        "entries",
        "1001",
    ):
        assert forbidden not in response.text


def test_status_uses_highest_crl_number_snapshot(app, client, root_id, chain):
    now = datetime.now(timezone.utc)
    # Registration refuses an already-expired CRL, so insert the expired
    # lower snapshot directly.
    _insert_snapshot(
        app,
        root_id_value=root_id,
        number=1,
        this_update=now - timedelta(days=2),
        next_update=now - timedelta(days=1),
    )
    second = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [],
        number=2,
        this_update=now - timedelta(minutes=5),
    )
    created = _register_crl(client, root_id, second)
    assert created.status_code == 201

    response = _status(client, trust_root_id=root_id)
    assert response.status_code == 200
    body = response.json()
    # Highest CRLNumber wins even though an older expired snapshot exists,
    # and the fresh highest snapshot makes the root fresh.
    assert body["crl_number"] == 2
    assert body["current_crl_id"] == created.json()["crl_id"]
    assert body["state"] == "fresh"


def test_stale_state_when_highest_snapshot_expired(app, client, root_id):
    now = datetime.now(timezone.utc)
    crl_id = _insert_snapshot(
        app,
        root_id_value=root_id,
        number=7,
        this_update=now - timedelta(days=10),
        next_update=now - timedelta(seconds=1),
    )
    response = _status(client, trust_root_id=root_id)
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "stale"
    assert body["current_crl_id"] == crl_id
    assert body["crl_number"] == 7
    assert body["this_update"] is not None
    assert body["next_update"] is not None


def test_next_update_boundary_is_stale_when_equal(app, client, root_id, monkeypatch):
    instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(app_module, "_utcnow", lambda: instant)
    _insert_snapshot(
        app,
        root_id_value=root_id,
        number=1,
        this_update=instant - timedelta(hours=1),
        next_update=instant,
    )
    response = _status(client, trust_root_id=root_id)
    assert response.status_code == 200
    assert response.json()["state"] == "stale"


def test_just_before_next_update_is_fresh(app, client, root_id, monkeypatch):
    instant = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(app_module, "_utcnow", lambda: instant)
    _insert_snapshot(
        app,
        root_id_value=root_id,
        number=1,
        this_update=instant - timedelta(hours=1),
        next_update=instant + timedelta(microseconds=1),
    )
    response = _status(client, trust_root_id=root_id)
    assert response.status_code == 200
    assert response.json()["state"] == "fresh"


def test_retired_root_still_reports_status(client, root_id):
    retire = client.post(
        f"/v1/trust-roots/{root_id}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retire.status_code in (200, 204)
    response = _status(client, trust_root_id=root_id)
    assert response.status_code == 200
    assert response.json()["state"] == "missing"


def test_status_is_read_only_and_repeatable(app, client, root_id):
    first = _status(client, trust_root_id=root_id)
    second = _status(client, trust_root_id=root_id)
    assert first.status_code == second.status_code == 200
    # observed_at may tick between calls, but the state and null fields are
    # stable; no snapshot or other row is created.
    assert first.json()["state"] == second.json()["state"] == "missing"
    with app.state.session_factory() as session:
        assert session.scalars(select(CertificateRevocationList)).all() == []


# --- request shape (422) ----------------------------------------------------


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "trust_root_id"])
def test_missing_parameter_is_422(client, root_id, missing):
    params = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
    }
    del params[missing]
    response = client.get("/v1/crls/status", params=params)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_id", ""),
        ("tenant_id", "   "),
        ("tenant_id", "\t"),
        ("tenant_id", f" {TENANT}"),
        ("tenant_id", f"{TENANT} "),
        ("workload_id", ""),
        ("workload_id", "  "),
        ("workload_id", f" {WORKLOAD}"),
        ("workload_id", f"{WORKLOAD}\n"),
        ("trust_root_id", ""),
        ("trust_root_id", "   "),
        ("trust_root_id", "not-a-uuid"),
        ("trust_root_id", "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"),
        ("trust_root_id", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa"),
        ("trust_root_id", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaZ"),
        ("trust_root_id", " 11111111-1111-1111-1111-111111111111"),
    ],
)
def test_invalid_parameters_are_422(client, root_id, field, value):
    params = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        field: value,
    }
    response = client.get("/v1/crls/status", params=params)
    assert response.status_code == 422


def test_extra_query_parameter_is_422(client, root_id):
    response = _status(client, trust_root_id=root_id, state="fresh")
    assert response.status_code == 422


def test_repeated_query_parameter_is_422(client, root_id):
    response = client.get(
        "/v1/crls/status"
        f"?tenant_id={TENANT}&tenant_id={TENANT}"
        f"&workload_id={WORKLOAD}&trust_root_id={root_id}"
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"{}", b"x", b" ", b"\n"])
def test_non_empty_body_is_422(client, root_id, body):
    response = client.request(
        "GET",
        "/v1/crls/status"
        f"?tenant_id={TENANT}&workload_id={WORKLOAD}&trust_root_id={root_id}",
        content=body,
    )
    assert response.status_code == 422


def test_status_route_is_not_captured_by_crl_id_route(client, root_id):
    # A valid request must reach the status route (200), proving it is
    # registered before the /v1/crls/{crl_id} catch-all.
    response = _status(client, trust_root_id=root_id)
    assert response.status_code == 200


# --- not found (indistinguishable 404) --------------------------------------


def test_unknown_trust_root_is_404(client, root_id):
    response = _status(
        client, trust_root_id="33333333-3333-3333-3333-333333333333"
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "trust root not found"


@pytest.mark.parametrize(
    "override",
    [
        {"tenant_id": OTHER_TENANT},
        {"workload_id": OTHER_WORKLOAD},
        {"trust_root_id": "44444444-4444-4444-4444-444444444444"},
    ],
)
def test_cross_scope_is_indistinguishable_404(client, root_id, override):
    params = {"trust_root_id": root_id}
    params.update(override)
    response = _status(client, **params)
    assert response.status_code == 404
    assert response.json()["detail"] == "trust root not found"


def test_root_of_other_tenant_is_404(client, chain):
    for tenant in (TENANT, OTHER_TENANT):
        created = client.post(
            "/v1/trust-roots",
            json={
                "tenant_id": tenant,
                "workload_id": WORKLOAD,
                "root_pem": pem(chain["root_cert"]),
            },
        )
        assert created.status_code == 201
    # Asking with tenant B's scope and a UUID that exists under tenant A
    # must not reveal the A root.
    response = client.get(
        "/v1/crls/status",
        params={
            "tenant_id": OTHER_TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": "00000000-0000-0000-0000-000000000000",
        },
    )
    assert response.status_code == 404


# --- storage failure --------------------------------------------------------


def test_storage_failure_is_500(app, client, root_id):
    engine = app.state.engine

    def fail_select(conn, cursor, statement, parameters, context, executemany):
        if "FROM trust_roots" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_select)
    try:
        response = _status(client, trust_root_id=root_id)
        assert response.status_code == 500
        assert response.json()["detail"] == "CRL status unavailable"
    finally:
        event.remove(engine, "before_cursor_execute", fail_select)

    recovered = _status(client, trust_root_id=root_id)
    assert recovered.status_code == 200
