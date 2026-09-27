"""Tests for GET /v1/trust-roots.

Read-only lifecycle query over registered trust roots. The range is fixed
by the mandatory tenant/workload and narrowed by an optional registration
id, exact name, status and an inclusive created-at window; pagination is
stable ``(created_at, root_id)`` ascending with opaque, scope/filter/
snapshot-bound cursors. The endpoint never writes state: it never creates
or retires a trust root and never stores new fields.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select

from proof_release import app as app_module
from proof_release.app import (
    create_app,
    _encode_audit_event_cursor,
    _encode_cursor,
    _encode_grant_audit_cursor,
)
from proof_release.db import AuditEvent, TrustRoot
from proof_release.envelopes import b64url_encode

from x509_helpers import make_root, pem

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/trust_roots_query.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


_root_counter = 0


def _new_pem() -> str:
    global _root_counter
    _root_counter += 1
    _, certificate = make_root(f"root-{_root_counter}")
    return pem(certificate)


def _create(client, *, name=None, tenant=TENANT, workload=WORKLOAD, pem_value=None):
    body = {"tenant_id": tenant, "workload_id": workload, "root_pem": pem_value or _new_pem()}
    if name is not None:
        body["name"] = name
    return client.post("/v1/trust-roots", json=body)


def _create_at(
    app,
    *,
    created_at: datetime,
    name=None,
    tenant=TENANT,
    workload=WORKLOAD,
    status="active",
    retired_at=None,
):
    """Create a trust root row directly with an exact created_at."""
    import hashlib
    import uuid

    from cryptography.hazmat.primitives.serialization import Encoding

    _, certificate = make_root(f"seed-{uuid.uuid4()}")
    der = certificate.public_bytes(Encoding.DER)
    root_id = str(uuid.uuid4())
    with app.state.session_factory() as session:
        session.add(
            TrustRoot(
                root_id=root_id,
                tenant_id=tenant,
                workload_id=workload,
                name=name,
                root_pem=certificate.public_bytes(Encoding.PEM).decode("ascii"),
                cert_sha256=hashlib.sha256(der).hexdigest(),
                status=status,
                created_at=created_at,
                retired_at=retired_at,
            )
        )
        session.commit()
    return {
        "root_id": root_id,
        "tenant_id": tenant,
        "workload_id": workload,
        "name": name,
        "created_at": _iso(created_at),
    }


def _retire(client, root, *, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        f"/v1/trust-roots/{root}/retire",
        json={"tenant_id": tenant, "workload_id": workload},
    )


def _query(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    query = {"tenant_id": tenant, "workload_id": workload}
    query.update(params)
    return client.get("/v1/trust-roots", params=query)


def _walk(client, *, page_size=None, monkeypatch=None, **params):
    token = None
    rows = []
    for _ in range(50):
        query_params = dict(params)
        if token is not None:
            query_params["cursor"] = token
        if page_size is not None:
            monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", page_size)
        page = _query(client, **query_params)
        assert page.status_code == 200, page.text
        data = page.json()
        rows.extend(data["trust_roots"])
        if data["complete"]:
            assert data["next_cursor"] == ""
            return rows
        token = data["next_cursor"]
        assert token
    raise AssertionError("pagination never completed")  # pragma: no cover


# --- request validation ----------------------------------------------------


def test_query_requires_scope_parameters(client):
    assert client.get("/v1/trust-roots").status_code == 422
    assert (
        client.get("/v1/trust-roots", params={"workload_id": WORKLOAD}).status_code
        == 422
    )
    assert (
        client.get("/v1/trust-roots", params={"tenant_id": TENANT}).status_code == 422
    )


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"workload_id": "\t"},
        {"root_id": ""},
        {"root_id": "   "},
        {"root_id": "not-a-uuid"},
        {"root_id": ZERO_UUID[:-1] + "Z"},
        {"root_id": "  " + ZERO_UUID},
        {"root_id": "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"},  # uppercase
        {"name": ""},
        {"name": "   "},
        {"name": "\t"},
        {"status": ""},
        {"status": "   "},
        {"status": "ACTIVE"},
        {"status": "Retired"},
        {"status": "pending"},
        {"created_after": ""},
        {"created_after": "   "},
        {"created_after": "not-a-timestamp"},
        {"created_after": "2026-01-01T00:00:00"},  # naive
        {"created_before": "2026-01-01"},
        {"created_after": "2026-01-01T00:00:00+02:00"},  # non-UTC offset
        {
            "created_after": "2026-01-02T00:00:00Z",
            "created_before": "2026-01-01T00:00:00Z",
        },
        {
            "created_after": "2026-01-01T01:00:00+01:00",
            "created_before": "2026-01-01T00:00:00Z",
        },
        {"cursor": "   "},
        {"cursor": "not base64!!"},
        {"cursor": "AAA="},
        {"bogus": "value"},
        {"limit": "10"},
    ],
)
def test_query_rejects_invalid_parameters(client, params):
    response = _query(client, **params)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("param", ["tenant_id", "status", "name"])
def test_repeated_parameter_is_422(client, param):
    response = client.get(
        "/v1/trust-roots",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            (param, "first"),
            (param, "second"),
        ],
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"{}", b"  ", b"not json", b"\x00"])
def test_non_empty_body_is_422_before_state_read(app, client, body):
    def fail_read(conn, cursor, statement, parameters, context, executemany):
        if "FROM trust_roots" in statement:
            raise AssertionError("storage must not be read for a bodied query")

    event.listen(app.state.engine, "before_cursor_execute", fail_read)
    try:
        response = client.request(
            "GET",
            "/v1/trust-roots",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            content=body,
        )
        assert response.status_code == 422
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_read)


def test_invalid_parameters_write_no_state(app, client):
    created = _create(client, name="keep")
    _query(client, root_id="nope")
    _query(client, status="deleted")
    _query(client, created_after="2020-01-01T00:00:00+03:00")
    _query(client, unknown="x")
    with app.state.session_factory() as session:
        assert session.query(AuditEvent).count() == 0
        rows = session.scalars(select(TrustRoot)).all()
        assert [r.root_id for r in rows] == [created.json()["root_id"]]
        assert rows[0].status == "active"
        assert rows[0].retired_at is None


# --- 404 semantics ---------------------------------------------------------


def test_unknown_root_identifier_returns_404(client):
    assert _query(client, root_id=ZERO_UUID).status_code == 404


def test_cross_scope_root_identifier_is_indistinguishable_404(client):
    created = _create(client).json()
    assert _query(client, tenant=OTHER_TENANT, root_id=created["root_id"]).status_code == 404
    assert (
        _query(client, workload=OTHER_WORKLOAD, root_id=created["root_id"]).status_code
        == 404
    )
    assert (
        _query(
            client,
            tenant=OTHER_TENANT,
            workload=OTHER_WORKLOAD,
            root_id=created["root_id"],
        ).status_code
        == 404
    )


def test_explicit_unknown_identifier_returns_404_even_with_other_matching_rows(client):
    _create(client, name="present")
    assert _query(client, root_id=ZERO_UUID).status_code == 404


# --- empty range / response shape ------------------------------------------


def test_empty_range_returns_empty_completed_page(client):
    response = _query(client)
    assert response.status_code == 200
    assert response.json() == {"trust_roots": [], "next_cursor": "", "complete": True}


def test_name_miss_returns_empty_range_not_404(client):
    _create(client, name="present")
    response = _query(client, name="absent")
    assert response.status_code == 200
    assert response.json() == {"trust_roots": [], "next_cursor": "", "complete": True}


def test_status_without_matches_returns_empty_completed_page(client):
    _create(client)
    response = _query(client, status="retired")
    assert response.status_code == 200
    assert response.json() == {"trust_roots": [], "next_cursor": "", "complete": True}


def test_known_explicit_identifier_excluded_only_by_window_returns_empty_page(client):
    created = _create(client).json()
    by_window = _query(
        client,
        root_id=created["root_id"],
        created_after="2030-01-01T00:00:00Z",
    )
    assert by_window.status_code == 200
    assert by_window.json() == {
        "trust_roots": [],
        "next_cursor": "",
        "complete": True,
    }
    by_status = _query(client, root_id=created["root_id"], status="retired")
    assert by_status.status_code == 200
    assert by_status.json() == {
        "trust_roots": [],
        "next_cursor": "",
        "complete": True,
    }


def test_response_is_compact_json_with_single_trailing_newline(client):
    _create(client)
    response = _query(client)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == ["trust_roots", "next_cursor", "complete"]


def test_entry_has_exact_shape_and_field_order(client):
    created = _create(client, name="primary-root").json()
    raw = _query(client).content
    entry = json.loads(raw)["trust_roots"][0]
    assert list(entry) == [
        "root_id",
        "tenant_id",
        "workload_id",
        "name",
        "created_at",
        "status",
        "retired_at",
    ]
    assert set(entry) == {
        "root_id",
        "tenant_id",
        "workload_id",
        "name",
        "created_at",
        "status",
        "retired_at",
    }
    assert entry["root_id"] == created["root_id"]
    assert entry["tenant_id"] == TENANT
    assert entry["workload_id"] == WORKLOAD
    assert entry["name"] == "primary-root"
    assert entry["created_at"] == created["created_at"]
    assert entry["status"] == "active"
    assert entry["retired_at"] is None
    # No certificate material, evidence or free-form text.
    assert "BEGIN CERTIFICATE" not in raw.decode("utf-8")


def test_unnamed_root_has_null_name(client):
    _create(client)
    entry = _query(client).json()["trust_roots"][0]
    assert entry["name"] is None


def test_retired_root_carries_status_and_retired_at(client):
    created = _create(client, name="doomed").json()
    retired = _retire(client, created["root_id"]).json()
    entry = _query(client).json()["trust_roots"][0]
    assert entry["status"] == "retired"
    assert entry["retired_at"] == retired["retired_at"]
    parsed = datetime.fromisoformat(entry["retired_at"])
    assert parsed.utcoffset() == timedelta(0)


def test_all_entry_values_are_strings_null_or_bool(client):
    created = _create(client, name="one").json()
    _retire(client, created["root_id"])
    _create(client, name="two")
    parsed = json.loads(_query(client).content)
    assert isinstance(parsed["complete"], bool)
    assert isinstance(parsed["next_cursor"], str)

    def _check(value):
        if isinstance(value, bool):
            return
        assert value is None or isinstance(value, str)
        assert not isinstance(value, float)

    for row in parsed["trust_roots"]:
        for value in row.values():
            _check(value)


# --- ordering --------------------------------------------------------------


def test_results_ordered_by_created_at_then_root_id(app, client):
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    bodies = [
        _create_at(app, created_at=base + timedelta(days=3), name="c"),
        _create_at(app, created_at=base + timedelta(days=1), name="a"),
        _create_at(app, created_at=base + timedelta(days=2), name="b"),
    ]
    # Two more rows sharing the earliest created_at break the tie by
    # root_id ascending.
    tie_a = _create_at(app, created_at=base + timedelta(days=1), name="a2")
    tie_b = _create_at(app, created_at=base + timedelta(days=1), name="a3")

    rows = _query(client).json()["trust_roots"]
    keys = [(row["created_at"], row["root_id"]) for row in rows]
    assert keys == sorted(keys)
    ordered_ids = [
        *sorted([bodies[1]["root_id"], tie_a["root_id"], tie_b["root_id"]]),
        bodies[2]["root_id"],
        bodies[0]["root_id"],
    ]
    assert [row["root_id"] for row in rows] == ordered_ids


# --- filtering -------------------------------------------------------------


def test_filter_by_root_id_returns_exactly_that_registration(client):
    first = _create(client, name="one").json()
    _create(client, name="two")
    rows = _query(client, root_id=first["root_id"]).json()["trust_roots"]
    assert [row["root_id"] for row in rows] == [first["root_id"]]


def test_filter_by_name_is_exact_text(client):
    target = _create(client, name="exact").json()
    _create(client, name="exact-two")
    _create(client, name="EXACT")
    rows = _query(client, name="exact").json()["trust_roots"]
    assert [row["root_id"] for row in rows] == [target["root_id"]]


def test_filter_by_status_distinguishes_active_and_retired(client):
    active = _create(client, name="live").json()
    retired = _create(client, name="old").json()
    _retire(client, retired["root_id"])

    active_rows = _query(client, status="active").json()["trust_roots"]
    assert [r["root_id"] for r in active_rows] == [active["root_id"]]
    assert all(r["status"] == "active" for r in active_rows)

    retired_rows = _query(client, status="retired").json()["trust_roots"]
    assert [r["root_id"] for r in retired_rows] == [retired["root_id"]]
    assert all(r["status"] == "retired" for r in retired_rows)
    assert retired_rows[0]["retired_at"] is not None


def test_created_window_is_inclusive_on_both_ends(app, client):
    at = datetime(2020, 6, 15, 12, 0, tzinfo=timezone.utc)
    before = _create_at(app, created_at=at - timedelta(days=1))
    middle = _create_at(app, created_at=at)
    after = _create_at(app, created_at=at + timedelta(days=1))

    rows = _query(
        client,
        created_after=_iso(at),
        created_before=_iso(at),
    ).json()["trust_roots"]
    assert [r["root_id"] for r in rows] == [middle["root_id"]]

    rows = _query(
        client,
        created_after=_iso(at - timedelta(days=1)),
        created_before=_iso(at + timedelta(days=1)),
    ).json()["trust_roots"]
    assert [r["root_id"] for r in rows] == [
        before["root_id"],
        middle["root_id"],
        after["root_id"],
    ]


def test_equivalent_utc_spellings_share_one_cursor_domain(app, client, monkeypatch):
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for i in range(3):
        _create_at(app, created_at=base + timedelta(days=i))
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 1)
    zed = _query(client, created_after="2020-01-01T00:00:00Z")
    cursor = zed.json()["next_cursor"]
    offset = _query(
        client,
        created_after="2020-01-01T00:00:00+00:00",
        cursor=cursor,
    )
    assert offset.status_code == 200
    replayed = _query(client, created_after="2020-01-01T00:00:00Z", cursor=cursor)
    assert offset.content == replayed.content


def test_results_are_scoped_to_tenant_and_workload(client):
    in_scope = _create(client).json()
    other_tenant = _create(client, tenant=OTHER_TENANT).json()
    other_workload = _create(client, workload=OTHER_WORKLOAD).json()

    rows_a = _query(client).json()["trust_roots"]
    assert [r["root_id"] for r in rows_a] == [in_scope["root_id"]]
    rows_b = _query(client, tenant=OTHER_TENANT).json()["trust_roots"]
    assert [r["root_id"] for r in rows_b] == [other_tenant["root_id"]]
    rows_c = _query(client, workload=OTHER_WORKLOAD).json()["trust_roots"]
    assert [r["root_id"] for r in rows_c] == [other_workload["root_id"]]


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_registration_once_in_order(app, client, monkeypatch):
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    bodies = [_create_at(app, created_at=base + timedelta(days=i)) for i in range(7)]
    rows = _walk(client, page_size=3, monkeypatch=monkeypatch)
    assert len(rows) == 7
    keys = [(row["created_at"], row["root_id"]) for row in rows]
    assert keys == sorted(keys)
    assert {row["root_id"] for row in rows} == {b["root_id"] for b in bodies}


def test_page_carries_cursor_and_complete_flag(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 2)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for i in range(5):
        _create_at(app, created_at=base + timedelta(days=i))

    first = _query(client).json()
    assert len(first["trust_roots"]) == 2
    assert first["complete"] is False
    assert first["next_cursor"]

    second = _query(client, cursor=first["next_cursor"]).json()
    assert len(second["trust_roots"]) == 2
    assert second["complete"] is False
    first_keys = [(r["created_at"], r["root_id"]) for r in first["trust_roots"]]
    second_keys = [(r["created_at"], r["root_id"]) for r in second["trust_roots"]]
    assert first_keys < second_keys

    last = _query(client, cursor=second["next_cursor"]).json()
    assert len(last["trust_roots"]) == 1
    assert last["complete"] is True
    assert last["next_cursor"] == ""


def test_replaying_same_cursor_returns_identical_page(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 2)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for i in range(5):
        _create_at(app, created_at=base + timedelta(days=i))
    cursor = _query(client).json()["next_cursor"]
    first = _query(client, cursor=cursor)
    second = _query(client, cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_string_cursor_equals_default(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 2)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for i in range(3):
        _create_at(app, created_at=base + timedelta(days=i))
    assert _query(client).content == _query(client, cursor="").content


def test_tampered_or_forged_cursor_returns_422(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 1)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for i in range(3):
        _create_at(app, created_at=base + timedelta(days=i))
    cursor = _query(client).json()["next_cursor"]
    assert cursor

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _query(client, cursor=tampered).status_code == 422

    forged = b64url_encode(b'{"k":"trust-roots-v1"}' + b"0" * 32)
    assert _query(client, cursor=forged).status_code == 422


def test_cross_scope_cursor_returns_422(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 1)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for i in range(3):
        _create_at(app, created_at=base + timedelta(days=i))
    cursor = _query(client).json()["next_cursor"]
    assert _query(client, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert _query(client, workload=OTHER_WORKLOAD, cursor=cursor).status_code == 422


def test_cursor_cannot_cross_filters(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 1)
    named = _create_at(
        app, created_at=datetime(2020, 1, 1, tzinfo=timezone.utc), name="one"
    )
    _create_at(app, created_at=datetime(2020, 1, 2, tzinfo=timezone.utc), name="two")
    cursor = _query(client).json()["next_cursor"]

    assert _query(client, root_id=named["root_id"], cursor=cursor).status_code == 422
    assert _query(client, name="one", cursor=cursor).status_code == 422
    assert _query(client, status="retired", cursor=cursor).status_code == 422
    assert (
        _query(
            client, created_after="2000-01-01T00:00:00Z", cursor=cursor
        ).status_code
        == 422
    )
    assert (
        _query(
            client, created_before="2100-01-01T00:00:00Z", cursor=cursor
        ).status_code
        == 422
    )


def test_cursor_rejects_other_cursor_families(app, client):
    _create_at(app, created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
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


# --- snapshot stability ----------------------------------------------------


def test_first_query_snapshot_is_replayable_and_new_first_query_advances(
    app, client, monkeypatch
):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 2)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    bodies = [_create_at(app, created_at=base + timedelta(days=i)) for i in range(3)]

    # First query fixes the replayable snapshot.
    first = _query(client).json()
    assert [r["root_id"] for r in first["trust_roots"]] == [
        bodies[0]["root_id"],
        bodies[1]["root_id"],
    ]
    cursor = first["next_cursor"]

    # A registration committed afterwards gets the server's current
    # created_at (later than every snapshotted row), so it sorts beyond the
    # high-water mark and never enters the snapshot's pages.
    late = _create(client, name="late")
    second = _query(client, cursor=cursor).json()
    assert [r["root_id"] for r in second["trust_roots"]] == [bodies[2]["root_id"]]
    assert second["complete"] is True
    assert second["next_cursor"] == ""

    # The cursor replays byte-for-byte and still omits the later row.
    replayed = _query(client, cursor=cursor)
    assert replayed.json() == second

    # A fresh first query establishes a new snapshot that includes it.
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert len(fresh) == 4
    assert late.json()["root_id"] in {row["root_id"] for row in fresh}


def test_snapshot_is_stable_under_concurrent_registrations(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 3)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    original = [
        _create_at(app, created_at=base + timedelta(days=i)) for i in range(6)
    ]
    first = _query(client).json()
    assert len(first["trust_roots"]) == 3
    cursor = first["next_cursor"]

    def register_late(index):
        other = TestClient(app)
        response = other.post(
            "/v1/trust-roots",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "root_pem": _new_pem(),
                "name": f"late-{index}",
            },
        )
        assert response.status_code == 201
        return response.json()["root_id"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        late_ids = list(pool.map(register_late, range(4)))

    snapshot_rows = []
    token = cursor
    for _ in range(10):
        data = _query(client, cursor=token).json()
        snapshot_rows.extend(data["trust_roots"])
        if data["complete"]:
            break
        token = data["next_cursor"]

    snapshot_ids = [row["root_id"] for row in snapshot_rows]
    original_ids = [b["root_id"] for b in original]
    assert [r["root_id"] for r in first["trust_roots"]] == original_ids[:3]
    assert snapshot_ids == original_ids[3:]
    assert not (set(snapshot_ids) & set(late_ids))

    # A fresh first query sees every committed registration.
    fresh = _walk(client, page_size=3, monkeypatch=monkeypatch)
    assert {row["root_id"] for row in fresh} == set(original_ids) | set(late_ids)


def test_replay_unaffected_by_concurrent_retirement(app, client, monkeypatch):
    # Retiring a row changes only its current status/retired_at; it never
    # moves the (created_at, root_id) position fixed by the snapshot. A row
    # retired after a page was minted stays in its page, now carrying the
    # retired state, and replaying the cursor returns the same page.
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 2)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    bodies = [_create_at(app, created_at=base + timedelta(days=i)) for i in range(4)]
    first = _query(client).json()
    assert len(first["trust_roots"]) == 2
    cursor = first["next_cursor"]

    # Retire a row that appears later in the snapshot walk, and register a
    # brand new root concurrently: its server-assigned created_at is beyond
    # the snapshot high-water mark, so it must not enter either page.
    _retire(client, bodies[3]["root_id"])
    late = _create(client, name="late")

    second = _query(client, cursor=cursor).json()
    assert [r["root_id"] for r in second["trust_roots"]] == [
        bodies[2]["root_id"],
        bodies[3]["root_id"],
    ]
    assert second["trust_roots"][1]["status"] == "retired"
    assert second["trust_roots"][1]["retired_at"] is not None

    # Same cursor replay: byte-for-byte identical page, no dupes or skips.
    replayed = _query(client, cursor=cursor)
    assert replayed.status_code == 200
    assert replayed.json() == second

    # A fresh first query sees all five committed registrations, with the
    # late one (created "now") sorting after the seeded rows.
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert [r["root_id"] for r in fresh] == [
        bodies[0]["root_id"],
        bodies[1]["root_id"],
        bodies[2]["root_id"],
        bodies[3]["root_id"],
        late.json()["root_id"],
    ]
    assert {r["status"] for r in fresh if r["root_id"] == bodies[3]["root_id"]} == {
        "retired"
    }


def test_snapshot_filtered_by_status_is_stable_after_retirement(
    app, client, monkeypatch
):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 1)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    bodies = [_create_at(app, created_at=base + timedelta(days=i)) for i in range(3)]

    # Snapshot of the active range fixes its high-water mark over the three
    # active rows; the first page presents only the earliest.
    first = _query(client, status="active").json()
    assert [r["root_id"] for r in first["trust_roots"]] == [bodies[0]["root_id"]]
    cursor = first["next_cursor"]

    # Retire the middle row after the active snapshot was fixed. Its
    # position is still bounded by the snapshot, but the row no longer
    # satisfies the status predicate, so the continued walk skips it
    # without any other row taking its place and without duplicating the
    # row already shown.
    _retire(client, bodies[1]["root_id"])
    seen = [r["root_id"] for r in first["trust_roots"]]
    token = cursor
    for _ in range(10):
        data = _query(client, status="active", cursor=token).json()
        seen.extend(r["root_id"] for r in data["trust_roots"])
        if data["complete"]:
            break
        token = data["next_cursor"]
    assert seen == [bodies[0]["root_id"], bodies[2]["root_id"]]

    # A fresh active first query (new snapshot) reflects post-retirement
    # state exactly: the two surviving active rows in stable order.
    fresh = _walk(client, page_size=1, monkeypatch=monkeypatch, status="active")
    assert [r["root_id"] for r in fresh] == [
        bodies[0]["root_id"],
        bodies[2]["root_id"],
    ]
    assert all(r["status"] == "active" for r in fresh)

    # And the retired row is exactly what the retired filter presents.
    retired_rows = _walk(
        client, page_size=1, monkeypatch=monkeypatch, status="retired"
    )
    assert [r["root_id"] for r in retired_rows] == [bodies[1]["root_id"]]
    assert retired_rows[0]["retired_at"] is not None


def test_empty_first_page_is_complete_and_a_later_commit_needs_a_new_first_query(
    client,
):
    # An empty range has no high-water mark to carry, so the first (and
    # only) page is an empty, completed range with the beginning marker.
    first = _query(client)
    assert first.json() == {"trust_roots": [], "next_cursor": "", "complete": True}

    # An explicit empty cursor is the same beginning-of-range request and
    # therefore also a fresh first query.
    assert _query(client, cursor="").json() == {
        "trust_roots": [],
        "next_cursor": "",
        "complete": True,
    }

    # After a later commit, a new first query establishes a new snapshot
    # and sees the new registration; an empty page had no opaque cursor to
    # pin.
    late = _create(client).json()
    fresh = _query(client)
    assert [r["root_id"] for r in fresh.json()["trust_roots"]] == [late["root_id"]]
    assert _query(client, cursor="").json() == fresh.json()


# --- read-only behaviour ---------------------------------------------------


def test_query_changes_no_state_and_creates_no_audit(app, client):
    created = _create(client, name="ro").json()
    before = _query(client).content
    for _ in range(3):
        assert _query(client).status_code == 200
        assert _query(client, root_id=created["root_id"]).status_code == 200
        assert _query(client, name="ro", status="active").status_code == 200
    assert _query(client).content == before
    with app.state.session_factory() as session:
        assert session.query(AuditEvent).count() == 0
        rows = session.query(TrustRoot).all()
        assert len(rows) == 1
        assert rows[0].root_id == created["root_id"]
        assert rows[0].status == "active"
        assert rows[0].retired_at is None


def test_response_and_storage_carry_no_new_secret_fields(app, client):
    response = _create(client, name="field-check")
    root_id = response.json()["root_id"]
    listing = _query(client).content
    # Only the documented container keys and entry keys are present.
    parsed = json.loads(listing)
    assert set(parsed) == {"trust_roots", "next_cursor", "complete"}
    assert set(parsed["trust_roots"][0]) == {
        "root_id",
        "tenant_id",
        "workload_id",
        "name",
        "created_at",
        "status",
        "retired_at",
    }
    # No new columns were introduced on the storage model.
    with app.state.session_factory() as session:
        record = session.get(TrustRoot, root_id)
        assert set(record.__table__.columns.keys()) == {
            "root_id",
            "tenant_id",
            "workload_id",
            "name",
            "root_pem",
            "cert_sha256",
            "status",
            "created_at",
            "retired_at",
        }


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_half_page(app, client):
    _create(client, name="one")

    def fail_scan(conn, cursor, statement, parameters, context, executemany):
        if "FROM trust_roots" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(app.state.engine, "before_cursor_execute", fail_scan)
    try:
        response = _query(client)
        assert response.status_code == 500
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_scan)

    recovered = _query(client)
    assert recovered.status_code == 200
    assert len(recovered.json()["trust_roots"]) == 1
    with app.state.session_factory() as session:
        assert session.query(AuditEvent).count() == 0


# --- persistence -----------------------------------------------------------


def test_registrations_queryable_after_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    created = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": _new_pem(),
            "name": "durable-root",
        },
    ).json()
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    data = client2.get(
        "/v1/trust-roots",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    assert [r["root_id"] for r in data["trust_roots"]] == [created["root_id"]]
    assert data["trust_roots"][0]["name"] == "durable-root"
    assert data["complete"] is True
    second.state.engine.dispose()


def test_retired_state_queryable_after_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart_retired.db"
    first = create_app(url)
    client1 = TestClient(first)
    created = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": _new_pem(),
        },
    ).json()
    retired = client1.post(
        f"/v1/trust-roots/{created['root_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    entry = client2.get(
        "/v1/trust-roots",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()["trust_roots"][0]
    assert entry["status"] == "retired"
    assert entry["retired_at"] == retired["retired_at"]
    second.state.engine.dispose()
