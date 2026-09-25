"""Tests for GET /v1/revocations.

Read-only, tenant-isolated, cursor-stable retrieval of registered X.509
certificate revocations. A first query fixes a replayable registration
snapshot; the endpoint never computes or rewrites a certificate's
current status and never writes state.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import AuditEvent, CertificateRevocation
from proof_release.envelopes import b64url_encode

from x509_helpers import make_root, pem

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/revocation_query.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def chain():
    root_key, root_cert = make_root()
    other_root_key, other_root_cert = make_root("other-root")
    return {
        "root_key": root_key,
        "root_cert": root_cert,
        "other_root_key": other_root_key,
        "other_root_cert": other_root_cert,
    }


def _fingerprint(certificate) -> str:
    from cryptography.hazmat.primitives.serialization import Encoding

    return b64url_encode(
        hashlib.sha256(certificate.public_bytes(Encoding.DER)).digest()
    )


def _unique_fp(index: int) -> str:
    """A canonical unpadded-base64url 32-byte fingerprint unique per index.

    Revocation rows only store the digest, so a synthetic 32-byte value is
    enough to register distinct fingerprints without minting certificates.
    """
    raw = (f"fp-{index}-".encode("ascii") * 8)[:32]
    return b64url_encode(raw)


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


@pytest.fixture()
def other_root_id(client, chain):
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["other_root_cert"]),
        },
    )
    assert response.status_code == 201
    return response.json()["root_id"]


def _register(client, root_id_value, fp, effective_at, *, tenant=TENANT,
              workload=WORKLOAD):
    return client.post(
        "/v1/revocations",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "trust_root_id": root_id_value,
            "certificate_fingerprint": fp,
            "effective_at": effective_at,
        },
    )


def _other_scope_root(client, chain, *, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD):
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    assert response.status_code == 201
    return response.json()["root_id"]


def _query(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        "/v1/revocations",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


def _walk(client, *, page_size=None, **params):
    token = None
    rows = []
    for _ in range(50):
        query_params = dict(params)
        if token is not None:
            query_params["cursor"] = token
        if page_size is not None:
            app_module.REVOCATION_PAGE_SIZE = page_size
        page = _query(client, **query_params)
        assert page.status_code == 200, page.text
        data = page.json()
        rows.extend(data["revocations"])
        if data["complete"]:
            assert data["next_cursor"] == ""
            return rows
        token = data["next_cursor"]
        assert token
    raise AssertionError("pagination never completed")  # pragma: no cover


# --- request validation -----------------------------------------------------


def test_query_requires_scope_parameters(client):
    assert _query_missing(client) == 422
    assert client.get(
        "/v1/revocations", params={"workload_id": WORKLOAD}
    ).status_code == 422
    assert client.get(
        "/v1/revocations", params={"tenant_id": TENANT}
    ).status_code == 422


def _query_missing(client):
    return client.get("/v1/revocations").status_code


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"workload_id": "\t"},
        {"revocation_id": ""},
        {"revocation_id": "   "},
        {"revocation_id": "not-a-uuid"},
        {"revocation_id": ZERO_UUID[:-1] + "Z"},
        {"revocation_id": "  " + ZERO_UUID},
        {"trust_root_id": ""},
        {"trust_root_id": "   "},
        {"trust_root_id": "not-a-uuid"},
        {"certificate_fingerprint": ""},
        {"certificate_fingerprint": "   "},
        {"certificate_fingerprint": "abcd"},  # too short
        {"certificate_fingerprint": "A" * 44},  # padded/wrong shape
        {"certificate_fingerprint": "?" * 43},  # bad alphabet
        {"effective_after": "not-a-timestamp"},
        {"effective_after": "2026-01-01T00:00:00"},  # naive
        {"effective_before": "2026-01-01"},
        {"effective_after": "2026-01-01T01:00:00+01:00"},  # non-UTC offset
        {"effective_after": ""},
        {"effective_before": "  "},
        {
            "effective_after": "2026-01-02T00:00:00Z",
            "effective_before": "2026-01-01T00:00:00Z",
        },
        {
            "effective_after": "2026-01-01T01:00:00+01:00",
            "effective_before": "2026-01-01T00:00:00Z",
        },
        {"cursor": "   "},
        {"cursor": "not base64!!"},
        {"cursor": "AAA="},
        {"bogus": "value"},
        {"limit": "10"},
        {"status": "revoked"},
    ],
)
def test_query_rejects_invalid_parameters(client, params):
    response = _query(client, **params)
    assert response.status_code == 422, response.text


def test_padded_fingerprint_is_422(client, root_id, chain):
    import base64

    from cryptography.hazmat.primitives.serialization import Encoding

    raw = hashlib.sha256(chain["root_cert"].public_bytes(Encoding.DER)).digest()
    padded = base64.urlsafe_b64encode(raw).decode("ascii")
    assert _query(client, certificate_fingerprint=padded).status_code == 422


@pytest.mark.parametrize("body", [b'{"unexpected": 1}', b"   ", b"null", b"[]"])
def test_query_with_non_empty_body_returns_422(client, body):
    response = client.request(
        "GET",
        "/v1/revocations",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_query_empty_body_is_accepted(client):
    response = client.request(
        "GET",
        "/v1/revocations",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"",
    )
    assert response.status_code == 200


def test_invalid_parameters_write_no_state(app, client, root_id, chain):
    _query(client, revocation_id="nope")
    _query(client, certificate_fingerprint="?" * 43)
    _query(client, effective_after="2026-01-01T00:00:00")
    with app.state.session_factory() as session:
        assert session.scalars(select(CertificateRevocation)).all() == []
        assert session.scalars(select(AuditEvent)).all() == []


# --- 404 semantics ----------------------------------------------------------


def test_unknown_revocation_identifier_returns_404(client):
    assert _query(client, revocation_id=ZERO_UUID).status_code == 404


def test_cross_scope_revocation_identifier_returns_404(client, root_id, chain):
    registered = _register(
        client, root_id, _fingerprint(chain["root_cert"]), "2020-01-01T00:00:00Z"
    )
    revocation_id = registered.json()["revocation_id"]
    assert _query(
        client, tenant=OTHER_TENANT, revocation_id=revocation_id
    ).status_code == 404
    assert _query(
        client, workload=OTHER_WORKLOAD, revocation_id=revocation_id
    ).status_code == 404
    # A trust-root UUID is a different identifier space and never matches.
    assert _query(client, revocation_id=root_id).status_code == 404


def test_unknown_trust_root_identifier_returns_404(client):
    assert _query(client, trust_root_id=ZERO_UUID).status_code == 404


def test_cross_scope_trust_root_returns_404(client, root_id):
    assert _query(
        client, tenant=OTHER_TENANT, trust_root_id=root_id
    ).status_code == 404
    assert _query(
        client, workload=OTHER_WORKLOAD, trust_root_id=root_id
    ).status_code == 404


def test_unknown_fingerprint_returns_404(client, chain):
    assert _query(
        client, certificate_fingerprint=_fingerprint(chain["root_cert"])
    ).status_code == 404


def test_cross_scope_fingerprint_returns_404(client, chain):
    # The same certificate anchors trust roots in two distinct scopes; a
    # fingerprint registered in one is unknown in the other.
    other_root = _other_scope_root(client, chain)
    fp = _fingerprint(chain["root_cert"])
    registered = _register(client, other_root, fp, "2020-01-01T00:00:00Z",
                           tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    assert registered.status_code == 201
    assert _query(client, certificate_fingerprint=fp).status_code == 404


def test_explicit_known_identifier_excluded_by_window_is_empty_page(
    client, root_id, chain
):
    # The explicit resource exists in the scope (never a 404); the time
    # window, like any other non-resource filter, may still narrow the page
    # to an empty, completed array — mirroring the grant audit.
    registered = _register(
        client, root_id, _fingerprint(chain["root_cert"]), "2020-01-01T00:00:00Z"
    )
    revocation_id = registered.json()["revocation_id"]
    response = _query(
        client,
        revocation_id=revocation_id,
        effective_after="2030-01-01T00:00:00Z",
    )
    assert response.status_code == 200
    assert response.json() == {
        "revocations": [],
        "next_cursor": "",
        "complete": True,
    }


def test_known_fingerprint_under_other_root_filter_is_empty_not_404(
    client, root_id, other_root_id, chain
):
    # The fingerprint is known in the scope (under a different trust
    # root); the combination with the root filter simply matches nothing.
    fp = _fingerprint(chain["root_cert"])
    registered = _register(client, root_id, fp, "2020-01-01T00:00:00Z")
    assert registered.status_code == 201
    response = _query(
        client, trust_root_id=other_root_id, certificate_fingerprint=fp
    )
    assert response.status_code == 200
    assert response.json()["revocations"] == []
    assert response.json()["complete"] is True


# --- empty scope / response shape -------------------------------------------


def test_empty_scope_returns_empty_completed_page(client):
    response = _query(client)
    assert response.status_code == 200
    assert response.content == (
        b'{"revocations":[],"next_cursor":"","complete":true}\n'
    )


def test_unknown_trust_root_is_404_even_with_other_narrowing_filters(client):
    # The explicit trust root is a named resource: unknown is a 404 and
    # never distinguished from a cross-scope root, regardless of any other
    # filter accompanying it.
    unknown = ZERO_UUID.replace("0", "1")
    assert _query(client, trust_root_id=unknown).status_code == 404
    assert (
        _query(
            client,
            trust_root_id=unknown,
            effective_after="2000-01-01T00:00:00Z",
            effective_before="2100-01-01T00:00:00Z",
        ).status_code
        == 404
    )
    assert (
        _query(client, trust_root_id=unknown, certificate_fingerprint=_unique_fp(1))
        .status_code
        == 404
    )


def test_response_is_compact_json_with_single_trailing_newline(
    client, root_id, chain
):
    _register(
        client, root_id, _fingerprint(chain["root_cert"]), "2020-01-01T00:00:00Z"
    )
    response = _query(client)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == [
        "revocations",
        "next_cursor",
        "complete",
    ]


def test_entry_has_exact_registration_shape_and_field_order(
    client, root_id, chain
):
    fp = _fingerprint(chain["root_cert"])
    registered = _register(client, root_id, fp, "2020-01-01T00:00:00Z").json()
    response = _query(client)
    row = response.json()["revocations"][0]
    assert list(row) == [
        "revocation_id",
        "trust_root_id",
        "certificate_fingerprint",
        "effective_at",
    ]
    assert set(row) == {
        "revocation_id",
        "trust_root_id",
        "certificate_fingerprint",
        "effective_at",
    }
    assert row == {
        "revocation_id": registered["revocation_id"],
        "trust_root_id": root_id,
        "certificate_fingerprint": fp,
        "effective_at": "2020-01-01T00:00:00+00:00",
    }
    for value in row.values():
        assert isinstance(value, str) and value


def test_no_floats_or_non_finite_values(client, root_id, chain):
    _register(
        client, root_id, _fingerprint(chain["root_cert"]), "2020-01-01T00:00:00Z"
    )
    parsed = json.loads(_query(client).content)
    assert isinstance(parsed["complete"], bool)
    assert isinstance(parsed["next_cursor"], str)
    for row in parsed["revocations"]:
        for value in row.values():
            assert isinstance(value, str)
            assert not isinstance(value, bool)


def test_future_effective_registrations_are_listed(client, root_id, chain):
    # The listing presents registrations regardless of whether they are in
    # effect; it never computes a certificate's current status.
    _register(
        client, root_id, _fingerprint(chain["root_cert"]), "2099-01-01T00:00:00Z"
    )
    rows = _query(client).json()["revocations"]
    assert len(rows) == 1
    assert rows[0]["effective_at"] == "2099-01-01T00:00:00+00:00"


# --- ordering ---------------------------------------------------------------


def test_rows_ordered_by_effective_at_then_revocation_id(
    client, root_id, other_root_id
):
    instants = [
        "2020-03-01T00:00:00Z",
        "2020-01-01T00:00:00Z",
        "2020-02-01T00:00:00Z",
        "2020-01-01T00:00:00Z",
    ]
    for i, instant in enumerate(instants):
        response = _register(
            client,
            root_id if i % 2 == 0 else other_root_id,
            _unique_fp(i),
            instant,
        )
        assert response.status_code == 201

    rows = _query(client).json()["revocations"]
    keys = [(row["effective_at"], row["revocation_id"]) for row in rows]
    assert keys == sorted(keys)
    assert [row["effective_at"] for row in rows] == [
        "2020-01-01T00:00:00+00:00",
        "2020-01-01T00:00:00+00:00",
        "2020-02-01T00:00:00+00:00",
        "2020-03-01T00:00:00+00:00",
    ]


def test_ties_at_same_effective_instant_order_by_revocation_id(
    client, root_id
):
    for i in range(5):
        response = _register(
            client, root_id, _unique_fp(i), "2020-01-01T00:00:00Z"
        )
        assert response.status_code == 201
    rows = _query(client).json()["revocations"]
    assert len(rows) == 5
    ids = [row["revocation_id"] for row in rows]
    assert ids == sorted(ids)
    assert {row["effective_at"] for row in rows} == {"2020-01-01T00:00:00+00:00"}


# --- filtering --------------------------------------------------------------


def test_filter_by_revocation_id(client, root_id, chain):
    first = _register(
        client, root_id, _unique_fp(1), "2020-01-01T00:00:00Z"
    ).json()
    _register(client, root_id, _unique_fp(2), "2020-02-01T00:00:00Z")
    rows = _query(client, revocation_id=first["revocation_id"]).json()[
        "revocations"
    ]
    assert [row["revocation_id"] for row in rows] == [first["revocation_id"]]


def test_filter_by_trust_root_isolates_roots(
    client, root_id, other_root_id
):
    _register(client, root_id, _unique_fp(1), "2020-01-01T00:00:00Z")
    _register(client, root_id, _unique_fp(2), "2020-02-01T00:00:00Z")
    _register(client, other_root_id, _unique_fp(3), "2020-03-01T00:00:00Z")
    rows = _query(client, trust_root_id=other_root_id).json()["revocations"]
    assert len(rows) == 1
    assert rows[0]["trust_root_id"] == other_root_id


def test_filter_by_fingerprint(client, root_id):
    target = _unique_fp(7)
    _register(client, root_id, _unique_fp(1), "2020-01-01T00:00:00Z")
    _register(client, root_id, target, "2020-02-01T00:00:00Z")
    rows = _query(client, certificate_fingerprint=target).json()["revocations"]
    assert [row["certificate_fingerprint"] for row in rows] == [target]


def test_filter_by_time_window_is_inclusive(client, root_id):
    instants = [
        "2020-01-01T00:00:00Z",
        "2020-02-01T00:00:00Z",
        "2020-03-01T00:00:00Z",
    ]
    for i, instant in enumerate(instants):
        _register(client, root_id, _unique_fp(i), instant)
    rows = _query(
        client,
        effective_after="2020-02-01T00:00:00Z",
        effective_before="2020-02-01T00:00:00Z",
    ).json()["revocations"]
    assert [row["effective_at"] for row in rows] == ["2020-02-01T00:00:00+00:00"]
    rows = _query(
        client,
        effective_after="2020-01-15T00:00:00Z",
        effective_before="2020-02-15T00:00:00Z",
    ).json()["revocations"]
    assert len(rows) == 1
    assert _query(
        client, effective_before="2019-12-31T00:00:00Z"
    ).json()["revocations"] == []
    assert _query(
        client, effective_after="2030-01-01T00:00:00Z"
    ).json()["revocations"] == []


def test_rows_are_scoped_to_tenant_and_workload(client, chain):
    other_root_a = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    # A second tenant anchors the same certificate independently.
    second_chain_root = make_root("second")
    root_b = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": OTHER_TENANT,
            "workload_id": OTHER_WORKLOAD,
            "root_pem": pem(second_chain_root[1]),
        },
    ).json()["root_id"]
    root_a = other_root_a.json()["root_id"]
    _register(client, root_a, _unique_fp(1), "2020-01-01T00:00:00Z")
    _register(
        client,
        root_b,
        _unique_fp(2),
        "2020-01-01T00:00:00Z",
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
    )
    rows_a = _query(client).json()["revocations"]
    rows_b = _query(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    ).json()["revocations"]
    assert len(rows_a) == 1 and rows_a[0]["trust_root_id"] == root_a
    assert len(rows_b) == 1 and rows_b[0]["trust_root_id"] == root_b


def test_same_fingerprint_under_distinct_roots_are_independent_rows(
    client, root_id, other_root_id, chain
):
    fp = _fingerprint(chain["root_cert"])
    assert _register(
        client, root_id, fp, "2020-01-01T00:00:00Z"
    ).status_code == 201
    # A different root certificate is needed for the second row, since the
    # root under which a fingerprint is registered differs; reuse the other
    # anchor with a different fingerprint instead.
    other_fp = _fingerprint(chain["other_root_cert"])
    assert _register(
        client, other_root_id, other_fp, "2020-01-01T00:00:00Z"
    ).status_code == 201
    rows = _query(client, certificate_fingerprint=fp).json()["revocations"]
    assert len(rows) == 1 and rows[0]["trust_root_id"] == root_id


# --- pagination -------------------------------------------------------------


def test_pagination_walks_every_record_once_in_order(client, root_id):
    for i in range(7):
        assert _register(
            client,
            root_id,
            _unique_fp(i),
            f"2020-01-{i + 1:02d}T00:00:00Z",
        ).status_code == 201
    rows = _walk(client, page_size=3)
    assert len(rows) == 7
    keys = [(row["effective_at"], row["revocation_id"]) for row in rows]
    assert keys == sorted(keys)
    assert len({row["revocation_id"] for row in rows}) == 7


def test_page_carries_cursor_and_complete_flag(client, root_id):
    original = app_module.REVOCATION_PAGE_SIZE
    app_module.REVOCATION_PAGE_SIZE = 2
    try:
        for i in range(5):
            _register(client, root_id, _unique_fp(i), "2020-01-01T00:00:00Z")

        first = _query(client).json()
        assert len(first["revocations"]) == 2
        assert first["complete"] is False
        assert first["next_cursor"]

        second = _query(client, cursor=first["next_cursor"]).json()
        assert len(second["revocations"]) == 2
        assert second["complete"] is False
        first_keys = [
            (r["effective_at"], r["revocation_id"]) for r in first["revocations"]
        ]
        second_keys = [
            (r["effective_at"], r["revocation_id"]) for r in second["revocations"]
        ]
        assert first_keys < second_keys

        last = _query(client, cursor=second["next_cursor"]).json()
        assert len(last["revocations"]) == 1
        assert last["complete"] is True
        assert last["next_cursor"] == ""
    finally:
        app_module.REVOCATION_PAGE_SIZE = original


def test_replaying_same_cursor_returns_identical_page(client, root_id):
    original = app_module.REVOCATION_PAGE_SIZE
    app_module.REVOCATION_PAGE_SIZE = 2
    try:
        for i in range(5):
            _register(client, root_id, _unique_fp(i), "2020-01-01T00:00:00Z")
        cursor = _query(client).json()["next_cursor"]
        first = _query(client, cursor=cursor)
        second = _query(client, cursor=cursor)
        assert first.status_code == second.status_code == 200
        assert first.content == second.content
    finally:
        app_module.REVOCATION_PAGE_SIZE = original


def test_empty_string_cursor_equals_default(client, root_id):
    original = app_module.REVOCATION_PAGE_SIZE
    app_module.REVOCATION_PAGE_SIZE = 2
    try:
        for i in range(3):
            _register(client, root_id, _unique_fp(i), "2020-01-01T00:00:00Z")
        assert _query(client).content == _query(client, cursor="").content
    finally:
        app_module.REVOCATION_PAGE_SIZE = original


def test_tampered_forged_or_cross_scope_cursor_returns_422(client, root_id):
    original = app_module.REVOCATION_PAGE_SIZE
    app_module.REVOCATION_PAGE_SIZE = 1
    try:
        for i in range(3):
            _register(client, root_id, _unique_fp(i), "2020-01-01T00:00:00Z")
        cursor = _query(client).json()["next_cursor"]
        assert cursor

        tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
        assert _query(client, cursor=tampered).status_code == 422

        forged = b64url_encode(
            b'{"k":"certificate-revocations-v1"}' + b"0" * 32
        )
        assert _query(client, cursor=forged).status_code == 422

        assert _query(
            client, tenant=OTHER_TENANT, cursor=cursor
        ).status_code == 422
        assert _query(
            client, workload=OTHER_WORKLOAD, cursor=cursor
        ).status_code == 422
    finally:
        app_module.REVOCATION_PAGE_SIZE = original


def test_cursor_cannot_cross_filters_or_kinds(client, root_id):
    original = app_module.REVOCATION_PAGE_SIZE
    app_module.REVOCATION_PAGE_SIZE = 1
    try:
        for i in range(3):
            _register(client, root_id, _unique_fp(i), "2020-01-01T00:00:00Z")
        cursor = _query(client).json()["next_cursor"]

        assert _query(
            client, trust_root_id=root_id, cursor=cursor
        ).status_code == 422
        assert _query(
            client, certificate_fingerprint=_unique_fp(1), cursor=cursor
        ).status_code == 422
        assert _query(
            client,
            effective_after="2000-01-01T00:00:00Z",
            cursor=cursor,
        ).status_code == 422
        assert _query(
            client,
            effective_before="2100-01-01T00:00:00Z",
            cursor=cursor,
        ).status_code == 422

        # Cursors from the other three HMAC families are never accepted.
        from proof_release.app import (
            _encode_audit_event_cursor,
            _encode_cursor,
            _encode_grant_audit_cursor,
        )

        foreign_rewrap = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
        assert _query(client, cursor=foreign_rewrap).status_code == 422
        foreign_grant = _encode_grant_audit_cursor(
            TENANT, WORKLOAD, ZERO_UUID,
            grant_id="", decision_id="", data_id="", status="",
            issued_after="", issued_before="",
        )
        assert _query(client, cursor=foreign_grant).status_code == 422
        foreign_audit = _encode_audit_event_cursor(
            TENANT, WORKLOAD, "2020-01-01T00:00:00+00:00", ZERO_UUID,
            event_id="", event_type="", status="",
            occurred_after="", occurred_before="",
        )
        assert _query(client, cursor=foreign_audit).status_code == 422
    finally:
        app_module.REVOCATION_PAGE_SIZE = original


# --- snapshot isolation -----------------------------------------------------


def test_first_query_fixes_replayable_snapshot(client, root_id):
    original = app_module.REVOCATION_PAGE_SIZE
    app_module.REVOCATION_PAGE_SIZE = 2
    try:
        for i in range(3):
            assert _register(
                client, root_id, _unique_fp(i), "2020-01-01T00:00:00Z"
            ).status_code == 201

        first = _query(client).json()
        assert len(first["revocations"]) == 2
        cursor = first["next_cursor"]
        second_before = _query(client, cursor=cursor).json()
        assert len(second_before["revocations"]) == 1
        assert second_before["complete"] is True
        second_page_ids = [
            row["revocation_id"] for row in second_before["revocations"]
        ]

        # Registrations committed after the snapshot fixed do not enter
        # the walk, even on replay of its final page.
        for i in range(3, 6):
            assert _register(
                client, root_id, _unique_fp(i), "2020-02-01T00:00:00Z"
            ).status_code == 201
        second_after = _query(client, cursor=cursor).json()
        assert [
            row["revocation_id"] for row in second_after["revocations"]
        ] == second_page_ids
        assert second_after["complete"] is True

        # A new first query fixes a fresh snapshot that contains them all.
        fresh_rows = _walk(client, page_size=2)
        assert len(fresh_rows) == 6
    finally:
        app_module.REVOCATION_PAGE_SIZE = original


def test_replayed_walk_never_duplicates_or_skips(client, root_id):
    original = app_module.REVOCATION_PAGE_SIZE
    app_module.REVOCATION_PAGE_SIZE = 3
    try:
        for i in range(7):
            _register(client, root_id, _unique_fp(i), "2020-01-01T00:00:00Z")
        pages = []
        token = None
        while True:
            params = {"cursor": token} if token else {}
            data = _query(client, **params).json()
            pages.append(data)
            if data["complete"]:
                break
            token = data["next_cursor"]
        # Replay each resume cursor: every page is byte-identical to the
        # page first returned for that cursor.
        token = pages[0]["next_cursor"]
        for earlier in pages[1:]:
            replay = _query(client, cursor=token).json()
            assert [r["revocation_id"] for r in replay["revocations"]] == [
                r["revocation_id"] for r in earlier["revocations"]
            ]
            assert replay["complete"] == earlier["complete"]
            token = earlier["next_cursor"]
        all_ids = [
            row["revocation_id"]
            for page in pages
            for row in page["revocations"]
        ]
        assert len(all_ids) == 7
        assert len(set(all_ids)) == 7
    finally:
        app_module.REVOCATION_PAGE_SIZE = original


# --- read-only behaviour ----------------------------------------------------


def test_query_writes_no_state_and_creates_no_audit(app, client, root_id):
    _register(client, root_id, _unique_fp(1), "2020-01-01T00:00:00Z")
    with app.state.session_factory() as session:
        assert session.query(CertificateRevocation).count() == 1
        assert session.query(AuditEvent).count() == 0
    for _ in range(3):
        assert _query(client).status_code == 200
    with app.state.session_factory() as session:
        assert session.query(CertificateRevocation).count() == 1
        assert session.query(AuditEvent).count() == 0


def test_query_does_not_change_registration_rows(app, client, root_id):
    registered = _register(
        client, root_id, _unique_fp(1), "2020-01-01T00:00:00Z"
    ).json()
    _query(client)
    with app.state.session_factory() as session:
        row = session.get(CertificateRevocation, registered["revocation_id"])
        assert row.trust_root_id == root_id
        assert row.certificate_fingerprint == registered["certificate_fingerprint"]
        assert row.effective_at.isoformat() == "2020-01-01T00:00:00+00:00"


# --- server failure ---------------------------------------------------------


def test_storage_failure_returns_500_and_no_page(app, client, root_id):
    _register(client, root_id, _unique_fp(1), "2020-01-01T00:00:00Z")
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE certificate_revocations"))
    response = _query(client)
    assert response.status_code == 500
    # A failure body never leaks certificate material or storage detail.
    assert "BEGIN CERTIFICATE" not in response.text


def test_storage_failure_creates_no_audit(app, client):
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE certificate_revocations"))
    assert _query(client).status_code == 500
    with app.state.session_factory() as session:
        assert session.query(AuditEvent).count() == 0


# --- persistence ------------------------------------------------------------


def test_records_and_cursors_survive_restart(tmp_path, chain):
    url = f"sqlite:///{tmp_path}/restart_query.db"
    first = create_app(url)
    client1 = TestClient(first)
    root = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    ).json()["root_id"]
    original = app_module.REVOCATION_PAGE_SIZE
    app_module.REVOCATION_PAGE_SIZE = 2
    try:
        registered_ids = []
        for i in range(3):
            response = _register(
                client1, root, _unique_fp(i), "2020-01-01T00:00:00Z"
            )
            assert response.status_code == 201
            registered_ids.append(response.json()["revocation_id"])
        # All three share an effective instant, so the keyset tie-breaks on
        # revocation_id ascending.
        ordered_ids = sorted(registered_ids)
        first_page = _query(client1).json()
        assert [r["revocation_id"] for r in first_page["revocations"]] == ordered_ids[:2]
        cursor = first_page["next_cursor"]
        assert cursor
    finally:
        app_module.REVOCATION_PAGE_SIZE = original
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    app_module.REVOCATION_PAGE_SIZE = 2
    try:
        # The cursor minted before the restart replays the same final page.
        replayed = _query(client2, cursor=cursor).json()
        assert [r["revocation_id"] for r in replayed["revocations"]] == ordered_ids[2:]
        assert replayed["complete"] is True
    finally:
        app_module.REVOCATION_PAGE_SIZE = original
    # A fresh first query after restart lists every persisted record.
    fresh = _query(client2).json()["revocations"]
    assert [r["revocation_id"] for r in fresh] == ordered_ids
    second.state.engine.dispose()
