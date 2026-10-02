"""Tests for the read-only CRL snapshot query endpoint
GET /v1/crls/{crl_id}."""

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

from x509_helpers import make_crl, make_intermediate, make_leaf, make_root, pem

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
    intermediate_key, intermediate_cert = make_intermediate(root_cert, root_key)
    leaf_key, leaf_cert = make_leaf(intermediate_cert, intermediate_key)
    return {
        "root_key": root_key,
        "root_cert": root_cert,
        "intermediate_key": intermediate_key,
        "intermediate_cert": intermediate_cert,
        "leaf_key": leaf_key,
        "leaf_cert": leaf_cert,
    }


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


def _get_crl(client, crl_id, **params):
    query = {
        "tenant_id": params.pop("tenant_id", TENANT),
        "workload_id": params.pop("workload_id", WORKLOAD),
        "trust_root_id": params.pop("trust_root_id", None),
    }
    query.update(params)
    query = {key: value for key, value in query.items() if value is not None}
    return client.get(f"/v1/crls/{crl_id}", params=query)


@pytest.fixture()
def registered(client, root_id, chain):
    now = datetime.now(timezone.utc)
    revoked = [
        (1003, now - timedelta(hours=1)),
        (1001, now - timedelta(hours=3)),
        (1002, now - timedelta(hours=2)),
        (1004, now + timedelta(days=3)),
    ]
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], revoked, number=3)
    created = _register_crl(client, root_id, crl_pem)
    assert created.status_code == 201
    return {
        "crl_id": created.json()["crl_id"],
        "root_id": root_id,
        "serials": [1001, 1002, 1003, 1004],
    }


# --- success path -----------------------------------------------------------


def test_get_crl_returns_snapshot_and_all_entries(client, app, registered):
    response = _get_crl(
        client, registered["crl_id"], trust_root_id=registered["root_id"]
    )
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == [
        "crl_id",
        "tenant_id",
        "workload_id",
        "trust_root_id",
        "crl_number",
        "crl_sha256",
        "this_update",
        "next_update",
        "revoked_count",
        "created_at",
        "entries",
    ]
    assert body["crl_id"] == registered["crl_id"]
    assert body["tenant_id"] == TENANT
    assert body["workload_id"] == WORKLOAD
    assert body["trust_root_id"] == registered["root_id"]
    assert body["crl_number"] == 3
    assert isinstance(body["crl_number"], int)
    assert isinstance(body["revoked_count"], int)
    assert len(body["crl_sha256"]) == 64
    assert body["crl_sha256"] == body["crl_sha256"].lower()
    int(body["crl_sha256"], 16)
    for field in ("this_update", "next_update", "created_at"):
        assert body[field].endswith("+00:00")

    # The not-yet-effective entry (1004) is listed, but revoked_count keeps
    # the registration-time arrived count of 3.
    assert body["revoked_count"] == 3
    assert len(body["entries"]) == 4
    for entry in body["entries"]:
        assert list(entry.keys()) == [
            "entry_id",
            "issuer_dn",
            "serial_number",
            "revocation_date",
        ]
        assert entry["issuer_dn"] == "CN=test-root"
        assert entry["revocation_date"].endswith("+00:00")
    assert [int(e["serial_number"]) for e in body["entries"]] == [1001, 1002, 1003, 1004]
    # Serial numbers are canonical decimal text.
    assert [e["serial_number"] for e in body["entries"]] == [
        "1001",
        "1002",
        "1003",
        "1004",
    ]

    # The CRL body and certificate material are never exposed.
    assert "BEGIN X509 CRL" not in response.text
    assert "BEGIN CERTIFICATE" not in response.text
    # Compact JSON: no whitespace between tokens.
    assert b", " not in response.content
    assert b": " not in response.content


def test_entries_ordered_by_revocation_date_then_entry_id(
    client, app, root_id, chain
):
    now = datetime.now(timezone.utc)
    moment = now - timedelta(hours=2)
    revoked = [
        (2003, moment),
        (2001, now - timedelta(hours=5)),
        (2002, moment),
        (2004, now - timedelta(hours=1)),
    ]
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], revoked, number=1)
    created = _register_crl(client, root_id, crl_pem)
    assert created.status_code == 201
    crl_id = created.json()["crl_id"]

    response = _get_crl(client, crl_id, trust_root_id=root_id)
    assert response.status_code == 200
    entries = response.json()["entries"]

    with app.state.session_factory() as session:
        rows = session.scalars(
            select(CrlRevokedCertificate).where(
                CrlRevokedCertificate.crl_id == crl_id
            )
        ).all()
    expected = sorted(rows, key=lambda r: (r.revocation_date, r.entry_id))
    assert [e["entry_id"] for e in entries] == [r.entry_id for r in expected]
    assert [e["serial_number"] for e in entries] == [
        r.serial_number for r in expected
    ]


def test_revoked_count_is_not_recomputed_after_registration(
    client, root_id, chain
):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [
            (3001, now - timedelta(hours=1)),
            (3002, now + timedelta(seconds=2)),
        ],
        number=1,
    )
    created = _register_crl(client, root_id, crl_pem)
    assert created.status_code == 201
    assert created.json()["revoked_count"] == 1
    crl_id = created.json()["crl_id"]

    # Once the second entry's revocationDate has arrived, the stored count
    # still reflects registration time; the entry list is unchanged too.
    threading.Event().wait(2.5)
    response = _get_crl(client, crl_id, trust_root_id=root_id)
    assert response.status_code == 200
    body = response.json()
    assert body["revoked_count"] == 1
    assert [e["serial_number"] for e in body["entries"]] == ["3001", "3002"]


def test_replaced_snapshot_remains_queryable(client, root_id, chain):
    now = datetime.now(timezone.utc)
    first = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [(4001, now - timedelta(hours=1))],
        number=1,
    )
    created = _register_crl(client, root_id, first)
    assert created.status_code == 201
    first_id = created.json()["crl_id"]
    second = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [],
        number=2,
        this_update=now - timedelta(minutes=5),
    )
    assert _register_crl(client, root_id, second).status_code == 201

    response = _get_crl(client, first_id, trust_root_id=root_id)
    assert response.status_code == 200
    body = response.json()
    assert body["crl_number"] == 1
    assert [e["serial_number"] for e in body["entries"]] == ["4001"]


def test_repeated_queries_are_identical_and_read_only(client, app, registered):
    first = _get_crl(
        client, registered["crl_id"], trust_root_id=registered["root_id"]
    )
    second = _get_crl(
        client, registered["crl_id"], trust_root_id=registered["root_id"]
    )
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    with app.state.session_factory() as session:
        snapshots = session.scalars(select(CertificateRevocationList)).all()
        entries = session.scalars(select(CrlRevokedCertificate)).all()
    assert len(snapshots) == 1
    assert len(entries) == 4


# --- request shape (422) ----------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    [
        "not-a-uuid",
        "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa",
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaZ",
        "",
    ],
)
def test_non_canonical_crl_id_is_422(client, registered, bad_id):
    response = _get_crl(client, bad_id, trust_root_id=registered["root_id"])
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_id", ""),
        ("tenant_id", "   "),
        ("workload_id", ""),
        ("workload_id", "\t"),
        ("trust_root_id", ""),
        ("trust_root_id", "   "),
        ("trust_root_id", "not-a-uuid"),
        ("trust_root_id", "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"),
    ],
)
def test_invalid_scope_parameters_are_422(client, registered, field, value):
    params = {"trust_root_id": registered["root_id"], field: value}
    response = _get_crl(client, registered["crl_id"], **params)
    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "trust_root_id"])
def test_missing_scope_parameter_is_422(client, registered, missing):
    params = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": registered["root_id"],
    }
    del params[missing]
    response = client.get(f"/v1/crls/{registered['crl_id']}", params=params)
    assert response.status_code == 422


def test_extra_query_parameter_is_422(client, registered):
    response = _get_crl(
        client,
        registered["crl_id"],
        trust_root_id=registered["root_id"],
        include_entries="true",
    )
    assert response.status_code == 422


def test_repeated_query_parameter_is_422(client, registered):
    response = client.get(
        f"/v1/crls/{registered['crl_id']}"
        f"?tenant_id={TENANT}&tenant_id={TENANT}"
        f"&workload_id={WORKLOAD}&trust_root_id={registered['root_id']}"
    )
    assert response.status_code == 422


# --- not found (indistinguishable 404) --------------------------------------


def test_unknown_crl_id_is_404(client, registered):
    response = _get_crl(
        client,
        "11111111-1111-1111-1111-111111111111",
        trust_root_id=registered["root_id"],
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "crl not found"


@pytest.mark.parametrize(
    "override",
    [
        {"tenant_id": OTHER_TENANT},
        {"workload_id": OTHER_WORKLOAD},
        {"trust_root_id": "22222222-2222-2222-2222-222222222222"},
    ],
)
def test_cross_scope_is_indistinguishable_404(client, registered, override):
    params = {"trust_root_id": registered["root_id"]}
    params.update(override)
    response = _get_crl(client, registered["crl_id"], **params)
    assert response.status_code == 404
    assert response.json()["detail"] == "crl not found"


def test_crl_of_other_tenant_is_404(client, chain):
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
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [(5001, now - timedelta(hours=1))],
        number=1,
    )
    created = _register_crl(client, roots[TENANT], crl_pem)
    assert created.status_code == 201
    crl_id = created.json()["crl_id"]

    # The other tenant cannot see the snapshot, even naming its own root.
    response = _get_crl(
        client, crl_id, tenant_id=OTHER_TENANT, trust_root_id=roots[OTHER_TENANT]
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "crl not found"


# --- storage failure --------------------------------------------------------


def test_storage_failure_is_500(app, client, registered):
    engine = app.state.engine

    def fail_select(conn, cursor, statement, parameters, context, executemany):
        if "FROM certificate_revocation_lists" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_select)
    try:
        response = _get_crl(
            client, registered["crl_id"], trust_root_id=registered["root_id"]
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "CRL registry unavailable"
    finally:
        event.remove(engine, "before_cursor_execute", fail_select)

    # The failure left the snapshot intact and queryable.
    recovered = _get_crl(
        client, registered["crl_id"], trust_root_id=registered["root_id"]
    )
    assert recovered.status_code == 200
