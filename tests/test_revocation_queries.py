"""Tests for GET /v1/revocations.

Read-only retrieval of trust-root-scoped X.509 certificate revocation
registrations. The query is ranged by tenant/workload/trust root, narrowed
by an optional registration id, certificate fingerprint and an inclusive
effective-time window, and paginated with opaque, scope/filter/snapshot-
bound cursors. The endpoint never writes state and never computes a
certificate's current status.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from proof_release import app as app_module
from proof_release.app import (
    create_app,
    _encode_audit_event_cursor,
    _encode_cursor,
    _encode_grant_audit_cursor,
)
from proof_release.db import AuditEvent, CertificateRevocation
from proof_release.envelopes import b64url_encode

from cryptography.hazmat.primitives.serialization import Encoding

from x509_helpers import (
    make_intermediate,
    make_leaf,
    make_root,
    pem,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"


def fingerprint(certificate) -> str:
    return b64url_encode(
        hashlib.sha256(certificate.public_bytes(Encoding.DER)).digest()
    )


def random_fingerprint() -> str:
    return b64url_encode(os.urandom(32))


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/revocations_query.db")
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


@pytest.fixture()
def other_root_id(client, chain):
    """The same root certificate configured for another tenant/workload."""
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": OTHER_TENANT,
            "workload_id": OTHER_WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    assert response.status_code == 201
    return response.json()["root_id"]


def _register(client, root_id_value, fp, effective_at, **scope):
    payload = {
        "tenant_id": scope.get("tenant_id", TENANT),
        "workload_id": scope.get("workload_id", WORKLOAD),
        "trust_root_id": root_id_value,
        "certificate_fingerprint": fp,
        "effective_at": effective_at,
    }
    return client.post("/v1/revocations", json=payload)


def _query(client, *, tenant=TENANT, workload=WORKLOAD, root=None, **params):
    root_value = root if root is not None else params.pop("trust_root_id", None)
    query = {"tenant_id": tenant, "workload_id": workload}
    if root_value is not None:
        query["trust_root_id"] = root_value
    query.update(params)
    return client.get("/v1/revocations", params=query)


def _register_set(client, root_id_value, entries):
    """Register (fingerprint, effective_at) pairs; return list of bodies."""
    bodies = []
    for fp, effective_at in entries:
        response = _register(client, root_id_value, fp, effective_at)
        assert response.status_code == 201, response.text
        bodies.append(response.json())
    return bodies


def _walk(client, *, page_size=None, monkeypatch=None, **params):
    token = None
    rows = []
    for _ in range(50):
        query_params = dict(params)
        if token is not None:
            query_params["cursor"] = token
        if page_size is not None:
            monkeypatch.setattr(app_module, "REVOCATION_PAGE_SIZE", page_size)
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


# --- request validation ----------------------------------------------------


def test_query_requires_scope_parameters(client, root_id):
    assert client.get("/v1/revocations").status_code == 422
    assert (
        client.get("/v1/revocations", params={"workload_id": WORKLOAD}).status_code
        == 422
    )
    assert (
        client.get("/v1/revocations", params={"tenant_id": TENANT}).status_code == 422
    )
    assert (
        client.get(
            "/v1/revocations",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"workload_id": "\t"},
        {"trust_root_id": ""},
        {"trust_root_id": "   "},
        {"trust_root_id": "not-a-uuid"},
        {"trust_root_id": ZERO_UUID[:-1] + "Z"},
        {"trust_root_id": "  " + ZERO_UUID},
        {"revocation_id": ""},
        {"revocation_id": "   "},
        {"revocation_id": "not-a-uuid"},
        {"revocation_id": "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"},  # uppercase
        {"certificate_fingerprint": ""},
        {"certificate_fingerprint": "   "},
        {"certificate_fingerprint": "abcd"},  # too short
        {"certificate_fingerprint": "A" * 44},  # padded/wrong shape
        {"certificate_fingerprint": "?" * 43},  # bad alphabet
        {"effective_after": ""},
        {"effective_after": "   "},
        {"effective_after": "not-a-timestamp"},
        {"effective_after": "2026-01-01T00:00:00"},  # naive
        {"effective_before": "2026-01-01"},
        {"effective_after": "2026-01-01T00:00:00+02:00"},  # non-UTC offset
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
        {"status": "pending"},
    ],
)
def test_query_rejects_invalid_parameters(client, root_id, params):
    response = _query(client, root=root_id, **params)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("body", [b"{}", b"  ", b"not json", b"\x00"])
def test_non_empty_body_is_422_before_state_read(app, client, root_id, body):
    def fail_read(conn, cursor, statement, parameters, context, executemany):
        if "FROM certificate_revocations" in statement or "FROM trust_roots" in statement:
            raise AssertionError("storage must not be read for a bodied query")

    from sqlalchemy import event

    event.listen(app.state.engine, "before_cursor_execute", fail_read)
    try:
        response = client.request(
            "GET",
            "/v1/revocations",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD,
                    "trust_root_id": root_id},
            content=body,
        )
        assert response.status_code == 422
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_read)


def test_padded_fingerprint_filter_is_422(client, root_id, chain):
    import base64

    raw = hashlib.sha256(chain["leaf_cert"].public_bytes(Encoding.DER)).digest()
    padded = base64.urlsafe_b64encode(raw).decode("ascii")
    assert _query(client, root=root_id, certificate_fingerprint=padded).status_code == 422


def test_invalid_parameters_write_no_state(app, client, root_id, chain):
    _register(client, root_id, fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z")
    _query(client, root=root_id, revocation_id="nope")
    _query(client, root=root_id, effective_after="2020-01-01T00:00:00+03:00")
    _query(client, root=root_id, unknown="x")
    with app.state.session_factory() as session:
        assert session.query(AuditEvent).count() == 0
        rows = session.scalars(select(CertificateRevocation)).all()
        assert len(rows) == 1


# --- 404 semantics ---------------------------------------------------------


def test_unknown_trust_root_returns_404(client):
    assert _query(client, root=ZERO_UUID).status_code == 404


def test_cross_scope_trust_root_returns_404(client, root_id):
    assert _query(client, tenant=OTHER_TENANT, root=root_id).status_code == 404
    assert _query(client, workload=OTHER_WORKLOAD, root=root_id).status_code == 404


def test_unknown_revocation_identifier_returns_404(client, root_id):
    assert _query(client, root=root_id, revocation_id=ZERO_UUID).status_code == 404


def test_cross_scope_revocation_identifier_is_indistinguishable_404(
    client, root_id, other_root_id, chain
):
    registered = _register(
        client,
        other_root_id,
        fingerprint(chain["leaf_cert"]),
        "2020-01-01T00:00:00Z",
        tenant_id=OTHER_TENANT,
        workload_id=OTHER_WORKLOAD,
    )
    assert registered.status_code == 201
    # The id exists, but under another scope/root: same 404 as an unknown.
    response = _query(
        client, root=root_id, revocation_id=registered.json()["revocation_id"]
    )
    assert response.status_code == 404
    response = _query(
        client,
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
        root=root_id,
        revocation_id=registered.json()["revocation_id"],
    )
    assert response.status_code == 404


def test_unknown_or_cross_scope_fingerprint_returns_404(client, root_id, other_root_id, chain):
    fp = fingerprint(chain["leaf_cert"])
    # Registered only under the other scope/root.
    registered = _register(
        client,
        other_root_id,
        fp,
        "2020-01-01T00:00:00Z",
        tenant_id=OTHER_TENANT,
        workload_id=OTHER_WORKLOAD,
    )
    assert registered.status_code == 201

    # Unknown fingerprint: 404.
    assert _query(client, root=root_id,
                  certificate_fingerprint=random_fingerprint()).status_code == 404
    # Known fingerprint, wrong scope/root: indistinguishable 404.
    assert _query(client, root=root_id, certificate_fingerprint=fp).status_code == 404


def test_explicit_unknown_resources_return_404_even_with_other_matching_rows(
    client, root_id, chain
):
    _register(client, root_id, fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z")
    assert _query(client, root=root_id, revocation_id=ZERO_UUID).status_code == 404
    assert (
        _query(client, root=root_id,
               certificate_fingerprint=random_fingerprint()).status_code
        == 404
    )


# --- empty range / response shape ------------------------------------------


def test_empty_range_returns_empty_completed_page(client, root_id):
    response = _query(client, root=root_id)
    assert response.status_code == 200
    assert response.json() == {
        "revocations": [],
        "next_cursor": "",
        "complete": True,
    }


def test_known_explicit_resource_excluded_only_by_window_returns_empty_page(
    client, root_id, chain
):
    # The explicit resource resolves in scope (so not a 404); the window
    # simply narrows it away, yielding an empty completed page.
    fp = fingerprint(chain["leaf_cert"])
    registered = _register(
        client, root_id, fp, "2020-01-01T00:00:00Z"
    ).json()
    by_id = _query(
        client, root=root_id,
        revocation_id=registered["revocation_id"],
        effective_after="2030-01-01T00:00:00Z",
    )
    assert by_id.status_code == 200
    assert by_id.json() == {"revocations": [], "next_cursor": "", "complete": True}
    by_fp = _query(
        client, root=root_id,
        certificate_fingerprint=fp,
        effective_before="2019-01-01T00:00:00Z",
    )
    assert by_fp.status_code == 200
    assert by_fp.json() == {"revocations": [], "next_cursor": "", "complete": True}


def test_window_without_matches_returns_empty_completed_page(client, root_id, chain):
    _register(client, root_id, fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z")
    response = _query(
        client, root=root_id, effective_after="2030-01-01T00:00:00Z"
    )
    assert response.status_code == 200
    assert response.json() == {
        "revocations": [],
        "next_cursor": "",
        "complete": True,
    }


def test_response_is_compact_json_with_single_trailing_newline(
    client, root_id, chain
):
    _register(client, root_id, fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z")
    response = _query(client, root=root_id)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    # Container key order mirrors the compliance audit query.
    assert list(json.loads(raw)) == ["revocations", "next_cursor", "complete"]


def test_entry_has_exact_shape_and_field_order(client, root_id, chain):
    fp = fingerprint(chain["leaf_cert"])
    registered = _register(
        client, root_id, fp, "2020-01-01T00:00:00Z"
    ).json()
    raw = _query(client, root=root_id).content
    entry = json.loads(raw)["revocations"][0]
    # Only the four registration fields, in the registration response order.
    assert list(entry) == [
        "revocation_id",
        "trust_root_id",
        "certificate_fingerprint",
        "effective_at",
    ]
    assert set(entry) == {
        "revocation_id",
        "trust_root_id",
        "certificate_fingerprint",
        "effective_at",
    }
    assert entry["revocation_id"] == registered["revocation_id"]
    assert entry["trust_root_id"] == root_id
    assert entry["certificate_fingerprint"] == fp
    assert entry["effective_at"] == "2020-01-01T00:00:00+00:00"
    # Every entry value is a JSON string.
    for value in entry.values():
        assert isinstance(value, str) and value


def test_no_floats_or_non_finite_values(client, root_id, chain):
    _register_set(
        client,
        root_id,
        [
            (fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z"),
            (fingerprint(chain["intermediate_cert"]), "2020-01-02T00:00:00Z"),
        ],
    )
    parsed = json.loads(_query(client, root=root_id).content)
    assert isinstance(parsed["complete"], bool)
    assert isinstance(parsed["next_cursor"], str)

    def _check(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            assert not isinstance(value, bool)
            return
        if isinstance(value, float):  # pragma: no cover - structural guard
            raise AssertionError("float present")
        assert value is None or isinstance(value, str)

    for row in parsed["revocations"]:
        for value in row.values():
            _check(value)


def test_future_effective_registration_is_listed_unchanged(client, root_id, chain):
    # The query presents the recorded effective time verbatim; it never
    # computes or rewrites whether the revocation is currently in effect.
    _register(client, root_id, fingerprint(chain["leaf_cert"]), "2099-01-01T00:00:00Z")
    rows = _query(client, root=root_id).json()["revocations"]
    assert len(rows) == 1
    assert rows[0]["effective_at"] == "2099-01-01T00:00:00+00:00"


# --- ordering --------------------------------------------------------------


def test_results_ordered_by_effective_at_then_revocation_id(client, root_id, chain):
    fps = [
        fingerprint(chain["leaf_cert"]),
        fingerprint(chain["intermediate_cert"]),
        fingerprint(chain["root_cert"]),
        random_fingerprint(),
        random_fingerprint(),
    ]
    # Register deliberately out of effective-time order, including a tie.
    bodies = _register_set(
        client,
        root_id,
        [
            (fps[0], "2020-01-03T00:00:00Z"),
            (fps[1], "2020-01-01T00:00:00Z"),
            (fps[2], "2020-01-02T00:00:00Z"),
            (fps[3], "2020-01-01T00:00:00Z"),
            (fps[4], "2020-01-01T00:00:00Z"),
        ],
    )
    effective_by_id = {b["revocation_id"]: b["effective_at"] for b in bodies}

    rows = _query(client, root=root_id).json()["revocations"]
    keys = [(row["effective_at"], row["revocation_id"]) for row in rows]
    assert keys == sorted(keys)
    assert [row[0] for row in keys] == [
        "2020-01-01T00:00:00+00:00",
        "2020-01-01T00:00:00+00:00",
        "2020-01-01T00:00:00+00:00",
        "2020-01-02T00:00:00+00:00",
        "2020-01-03T00:00:00+00:00",
    ]
    for row in rows:
        assert effective_by_id[row["revocation_id"]] == row["effective_at"]


# --- filtering -------------------------------------------------------------


def test_filter_by_revocation_id_returns_exactly_that_registration(
    client, root_id, chain
):
    bodies = _register_set(
        client,
        root_id,
        [
            (fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z"),
            (fingerprint(chain["intermediate_cert"]), "2020-01-02T00:00:00Z"),
        ],
    )
    target = bodies[0]["revocation_id"]
    rows = _query(client, root=root_id, revocation_id=target).json()["revocations"]
    assert [row["revocation_id"] for row in rows] == [target]


def test_filter_by_fingerprint_returns_exactly_that_registration(
    client, root_id, chain
):
    fp = fingerprint(chain["leaf_cert"])
    _register_set(
        client,
        root_id,
        [
            (fp, "2020-01-01T00:00:00Z"),
            (fingerprint(chain["intermediate_cert"]), "2020-01-02T00:00:00Z"),
        ],
    )
    rows = _query(client, root=root_id, certificate_fingerprint=fp).json()[
        "revocations"
    ]
    assert len(rows) == 1
    assert rows[0]["certificate_fingerprint"] == fp


def test_effective_window_is_inclusive_on_both_ends(client, root_id):
    bodies = _register_set(
        client,
        root_id,
        [
            (random_fingerprint(), "2020-01-01T00:00:00Z"),
            (random_fingerprint(), "2020-01-02T00:00:00Z"),
            (random_fingerprint(), "2020-01-03T00:00:00Z"),
        ],
    )
    middle = bodies[1]["revocation_id"]
    rows = _query(
        client,
        root=root_id,
        effective_after="2020-01-02T00:00:00Z",
        effective_before="2020-01-02T00:00:00Z",
    ).json()["revocations"]
    assert [row["revocation_id"] for row in rows] == [middle]

    rows = _query(
        client,
        root=root_id,
        effective_after="2020-01-01T12:00:00Z",
        effective_before="2020-01-02T12:00:00Z",
    ).json()["revocations"]
    assert [row["revocation_id"] for row in rows] == [middle]

    assert (
        _query(
            client, root=root_id, effective_before="2019-12-31T00:00:00Z"
        ).json()["revocations"]
        == []
    )
    assert (
        _query(
            client, root=root_id, effective_after="2030-01-01T00:00:00Z"
        ).json()["revocations"]
        == []
    )


def test_equivalent_utc_spellings_share_one_cursor_domain(
    client, root_id, monkeypatch
):
    _register_set(
        client,
        root_id,
        [(random_fingerprint(), f"2020-01-0{i}T00:00:00Z") for i in range(1, 4)],
    )
    monkeypatch.setattr(app_module, "REVOCATION_PAGE_SIZE", 1)
    zed = _query(client, root=root_id, effective_after="2020-01-01T00:00:00Z")
    cursor = zed.json()["next_cursor"]
    offset = _query(
        client,
        root=root_id,
        effective_after="2020-01-01T00:00:00+00:00",
        cursor=cursor,
    )
    assert offset.status_code == 200
    replayed = _query(
        client, root=root_id, effective_after="2020-01-01T00:00:00Z", cursor=cursor
    )
    assert offset.content == replayed.content


def test_results_are_scoped_to_tenant_workload_and_root(
    client, root_id, other_root_id, chain
):
    _register(client, root_id, fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z")
    _register(
        client,
        other_root_id,
        fingerprint(chain["intermediate_cert"]),
        "2020-01-01T00:00:00Z",
        tenant_id=OTHER_TENANT,
        workload_id=OTHER_WORKLOAD,
    )
    rows_a = _query(client, root=root_id).json()["revocations"]
    rows_b = _query(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD, root=other_root_id
    ).json()["revocations"]
    assert len(rows_a) == 1
    assert rows_a[0]["certificate_fingerprint"] == fingerprint(chain["leaf_cert"])
    assert len(rows_b) == 1
    assert rows_b[0]["certificate_fingerprint"] == fingerprint(
        chain["intermediate_cert"]
    )


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_registration_once_in_order(
    client, root_id, monkeypatch
):
    entries = [(random_fingerprint(), f"2020-01-{i:02d}T00:00:00Z") for i in range(1, 8)]
    bodies = _register_set(client, root_id, entries)
    rows = _walk(client, page_size=3, monkeypatch=monkeypatch, root=root_id)
    assert len(rows) == 7
    keys = [(row["effective_at"], row["revocation_id"]) for row in rows]
    assert keys == sorted(keys)
    assert {row["revocation_id"] for row in rows} == {
        b["revocation_id"] for b in bodies
    }


def test_page_carries_cursor_and_complete_flag(client, root_id, monkeypatch):
    monkeypatch.setattr(app_module, "REVOCATION_PAGE_SIZE", 2)
    _register_set(
        client,
        root_id,
        [(random_fingerprint(), f"2020-01-0{i}T00:00:00Z") for i in range(1, 6)],
    )

    first = _query(client, root=root_id).json()
    assert len(first["revocations"]) == 2
    assert first["complete"] is False
    assert first["next_cursor"]

    second = _query(client, root=root_id, cursor=first["next_cursor"]).json()
    assert len(second["revocations"]) == 2
    assert second["complete"] is False
    first_keys = [
        (r["effective_at"], r["revocation_id"]) for r in first["revocations"]
    ]
    second_keys = [
        (r["effective_at"], r["revocation_id"]) for r in second["revocations"]
    ]
    assert first_keys < second_keys

    last = _query(client, root=root_id, cursor=second["next_cursor"]).json()
    assert len(last["revocations"]) == 1
    assert last["complete"] is True
    assert last["next_cursor"] == ""


def test_replaying_same_cursor_returns_identical_page(client, root_id, monkeypatch):
    monkeypatch.setattr(app_module, "REVOCATION_PAGE_SIZE", 2)
    _register_set(
        client,
        root_id,
        [(random_fingerprint(), f"2020-01-0{i}T00:00:00Z") for i in range(1, 6)],
    )
    cursor = _query(client, root=root_id).json()["next_cursor"]
    first = _query(client, root=root_id, cursor=cursor)
    second = _query(client, root=root_id, cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_string_cursor_equals_default(client, root_id, monkeypatch):
    monkeypatch.setattr(app_module, "REVOCATION_PAGE_SIZE", 2)
    _register_set(
        client,
        root_id,
        [(random_fingerprint(), f"2020-01-0{i}T00:00:00Z") for i in range(1, 4)],
    )
    assert _query(client, root=root_id).content == _query(
        client, root=root_id, cursor=""
    ).content


def test_tampered_or_forged_cursor_returns_422(client, root_id, monkeypatch):
    monkeypatch.setattr(app_module, "REVOCATION_PAGE_SIZE", 1)
    _register_set(
        client,
        root_id,
        [(random_fingerprint(), f"2020-01-0{i}T00:00:00Z") for i in range(1, 4)],
    )
    cursor = _query(client, root=root_id).json()["next_cursor"]
    assert cursor

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _query(client, root=root_id, cursor=tampered).status_code == 422

    forged = b64url_encode(b'{"k":"certificate-revocations-v1"}' + b"0" * 32)
    assert _query(client, root=root_id, cursor=forged).status_code == 422


def test_cross_scope_or_cross_root_cursor_returns_422(
    client, root_id, other_root_id, monkeypatch
):
    monkeypatch.setattr(app_module, "REVOCATION_PAGE_SIZE", 1)
    _register_set(
        client,
        root_id,
        [(random_fingerprint(), f"2020-01-0{i}T00:00:00Z") for i in range(1, 4)],
    )
    cursor = _query(client, root=root_id).json()["next_cursor"]
    assert _query(client, tenant=OTHER_TENANT, root=root_id, cursor=cursor).status_code == 422
    assert _query(client, workload=OTHER_WORKLOAD, root=root_id, cursor=cursor).status_code == 422
    assert _query(client, root=other_root_id, cursor=cursor).status_code == 422


def test_cursor_cannot_cross_filters(client, root_id, monkeypatch):
    monkeypatch.setattr(app_module, "REVOCATION_PAGE_SIZE", 1)
    entries = [
        (random_fingerprint(), "2020-01-01T00:00:00Z"),
        (random_fingerprint(), "2020-01-02T00:00:00Z"),
        (random_fingerprint(), "2020-01-03T00:00:00Z"),
    ]
    bodies = _register_set(client, root_id, entries)
    cursor = _query(client, root=root_id).json()["next_cursor"]

    assert (
        _query(
            client, root=root_id, revocation_id=bodies[0]["revocation_id"], cursor=cursor
        ).status_code
        == 422
    )
    assert (
        _query(
            client, root=root_id,
            certificate_fingerprint=entries[0][0], cursor=cursor
        ).status_code
        == 422
    )
    assert (
        _query(
            client, root=root_id,
            effective_after="2000-01-01T00:00:00Z", cursor=cursor
        ).status_code
        == 422
    )
    assert (
        _query(
            client, root=root_id,
            effective_before="2100-01-01T00:00:00Z", cursor=cursor
        ).status_code
        == 422
    )


def test_cursor_rejects_other_cursor_families(client, root_id):
    foreign_rewrap = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _query(client, root=root_id, cursor=foreign_rewrap).status_code == 422

    foreign_grant = _encode_grant_audit_cursor(
        TENANT, WORKLOAD, ZERO_UUID,
        grant_id="", decision_id="", data_id="", status="",
        issued_after="", issued_before="",
    )
    assert _query(client, root=root_id, cursor=foreign_grant).status_code == 422

    foreign_audit = _encode_audit_event_cursor(
        TENANT, WORKLOAD, "2020-01-01T00:00:00+00:00", ZERO_UUID,
        event_id="", event_type="", status="",
        occurred_after="", occurred_before="",
    )
    assert _query(client, root=root_id, cursor=foreign_audit).status_code == 422


# --- snapshot stability ----------------------------------------------------


def test_first_query_snapshot_is_replayable_and_new_first_query_advances(
    client, root_id, monkeypatch
):
    monkeypatch.setattr(app_module, "REVOCATION_PAGE_SIZE", 2)
    bodies = _register_set(
        client,
        root_id,
        [(random_fingerprint(), f"2020-01-0{i}T00:00:00Z") for i in range(1, 4)],
    )

    # First query fixes the replayable snapshot.
    first = _query(client, root=root_id).json()
    assert [r["revocation_id"] for r in first["revocations"]] == [
        bodies[0]["revocation_id"],
        bodies[1]["revocation_id"],
    ]
    cursor = first["next_cursor"]

    # A registration committed afterwards (with an effective time that
    # would otherwise sort into the middle of the listing) never enters
    # the snapshot's pages: walking the snapshot reaches its end having
    # seen exactly the original three rows once.
    late = _register(
        client, root_id, random_fingerprint(), "2020-01-02T12:00:00Z"
    ).json()
    second = _query(client, root=root_id, cursor=cursor).json()
    assert [r["revocation_id"] for r in second["revocations"]] == [
        bodies[2]["revocation_id"]
    ]
    assert second["complete"] is True
    assert second["next_cursor"] == ""

    # The cursor replays byte-for-byte and still omits the later row.
    replayed = _query(client, root=root_id, cursor=cursor)
    assert replayed.json() == second

    # A fresh first query establishes a new snapshot that includes the
    # later commit.
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch, root=root_id)
    fresh_keys = [(row["effective_at"], row["revocation_id"]) for row in fresh]
    assert fresh_keys == sorted(fresh_keys)
    assert len(fresh) == 4
    assert late["revocation_id"] in {row["revocation_id"] for row in fresh}


def test_snapshot_is_stable_under_concurrent_registrations(app, client, root_id):
    # Small pages, original snapshot of six rows, four more committing
    # while the cursor is walked.
    app_module.REVOCATION_PAGE_SIZE = 3
    try:
        original = _register_set(
            client,
            root_id,
            [
                (random_fingerprint(), f"2020-01-{i:02d}T00:00:00Z")
                for i in range(1, 7)
            ],
        )
        first = _query(client, root=root_id).json()
        assert len(first["revocations"]) == 3
        cursor = first["next_cursor"]

        def register_late(index):
            fp = random_fingerprint()
            # Mid-listing effective times make these candidates for
            # positions the snapshot walk passes through.
            response = TestClient(app).post(
                "/v1/revocations",
                json={
                    "tenant_id": TENANT,
                    "workload_id": WORKLOAD,
                    "trust_root_id": root_id,
                    "certificate_fingerprint": fp,
                    "effective_at": f"2020-01-0{index + 2}T12:00:00Z",
                },
            )
            assert response.status_code == 201
            return response.json()["revocation_id"]

        with ThreadPoolExecutor(max_workers=4) as pool:
            late_ids = list(pool.map(register_late, range(4)))

        snapshot_rows = []
        token = cursor
        for _ in range(10):
            data = _query(client, root=root_id, cursor=token).json()
            snapshot_rows.extend(data["revocations"])
            if data["complete"]:
                break
            token = data["next_cursor"]

        # Exactly the remaining original rows: no duplicates, no skips,
        # and none of the concurrently committed registrations.
        snapshot_ids = [row["revocation_id"] for row in snapshot_rows]
        original_ids = [b["revocation_id"] for b in original]
        assert [r["revocation_id"] for r in first["revocations"]] == original_ids[:3]
        assert snapshot_ids == original_ids[3:]
        assert not (set(snapshot_ids) & set(late_ids))

        # A fresh first query sees all ten committed registrations.
        def walk_fresh():
            rows = []
            token = None
            for _ in range(20):
                params = {"root": root_id}
                if token is not None:
                    params["cursor"] = token
                data = _query(client, **params).json()
                rows.extend(data["revocations"])
                if data["complete"]:
                    return rows
                token = data["next_cursor"]
            raise AssertionError("pagination never completed")  # pragma: no cover

        fresh_ids = {row["revocation_id"] for row in walk_fresh()}
        assert fresh_ids == set(original_ids) | set(late_ids)
    finally:
        app_module.REVOCATION_PAGE_SIZE = 100


# --- read-only behaviour ---------------------------------------------------


def test_query_creates_no_audit_and_changes_no_registration(
    app, client, root_id, chain
):
    registered = _register(
        client, root_id, fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z"
    ).json()
    before = _query(client, root=root_id).content
    for _ in range(3):
        assert _query(client, root=root_id).status_code == 200
        assert _query(client, root=root_id,
                      revocation_id=registered["revocation_id"]).status_code == 200
    assert _query(client, root=root_id).content == before
    with app.state.session_factory() as session:
        assert session.query(AuditEvent).count() == 0
        rows = session.query(CertificateRevocation).all()
        assert len(rows) == 1
        assert rows[0].revocation_id == registered["revocation_id"]


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_half_page(app, client, root_id, chain):
    from sqlalchemy import event, text

    _register(client, root_id, fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z")

    def fail_scan(conn, cursor, statement, parameters, context, executemany):
        if "FROM certificate_revocations" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(app.state.engine, "before_cursor_execute", fail_scan)
    try:
        response = _query(client, root=root_id)
        assert response.status_code == 500
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_scan)

    # No state was written by the failed read; a recovered read returns
    # the still-complete range.
    recovered = _query(client, root=root_id)
    assert recovered.status_code == 200
    assert len(recovered.json()["revocations"]) == 1
    with app.state.session_factory() as session:
        assert session.query(AuditEvent).count() == 0


# --- persistence -----------------------------------------------------------


def test_registrations_queryable_after_restart_and_cursor_replays(tmp_path, chain):
    url = f"sqlite:///{tmp_path}/restart.db"
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
    bodies = _register_set(
        client1,
        root,
        [
            (fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z"),
            (fingerprint(chain["intermediate_cert"]), "2020-01-02T00:00:00Z"),
        ],
    )
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    data = client2.get(
        "/v1/revocations",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD, "trust_root_id": root},
    ).json()
    assert [row["revocation_id"] for row in data["revocations"]] == [
        b["revocation_id"] for b in bodies
    ]
    assert data["complete"] is True
    second.state.engine.dispose()
