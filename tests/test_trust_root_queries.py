"""Tests for GET /v1/trust-roots.

Read-only lifecycle query of configured attestation trust roots. The query
is ranged by tenant/workload, narrowed by an optional registration id,
exact name, lifecycle status and an inclusive creation-time window, and
paginated with opaque, scope/filter/snapshot-bound cursors. The snapshot
is fixed at a per-scope commit boundary shared by creation and
retirement, so a replayed page is stable even under concurrent
retirements and new registrations. The endpoint never writes state and
never returns certificate or key material.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.app import (
    create_app,
    _encode_audit_event_cursor,
    _encode_cursor,
    _encode_decision_cursor,
    _encode_grant_audit_cursor,
    _encode_revocation_cursor,
)
from proof_release.db import TrustRoot, TrustRootCommitCounter

from x509_helpers import make_root, pem

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/trust_roots_query.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, *, name=None, tenant=TENANT, workload=WORKLOAD, common_name=None):
    root_key, root_cert = make_root(common_name=common_name or f"root-{name}")
    body = {"tenant_id": tenant, "workload_id": workload, "root_pem": pem(root_cert)}
    if name is not None:
        body["name"] = name
    response = client.post("/v1/trust-roots", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _retire(client, root_id_value, *, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        f"/v1/trust-roots/{root_id_value}/retire",
        json={"tenant_id": tenant, "workload_id": workload},
    )
    assert response.status_code == 200, response.text
    return response.json()


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
        {"status": "pending"},
        {"status": "RETIRED"},
        {"status": "retired "},
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
        {"cursor": "   "},
        {"cursor": "not base64!!"},
        {"cursor": "AAA="},
        {"bogus": "value"},
        {"limit": "10"},
        {"trust_root_id": ZERO_UUID},
    ],
)
def test_query_rejects_invalid_parameters(client, params):
    response = _query(client, **params)
    assert response.status_code == 422, response.text


def test_repeated_parameter_is_422(client):
    response = client.get(
        "/v1/trust-roots",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("status", "active"),
            ("status", "retired"),
        ],
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"{}", b"  ", b"not json", b"\x00"])
def test_non_empty_body_is_422_before_state_read(app, client, body):
    def fail_read(conn, cursor, statement, parameters, context, executemany):
        if "FROM trust_roots" in statement:
            raise AssertionError("storage must not be read for a bodied query")

    from sqlalchemy import event

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
    _create(client, name="one")
    _query(client, root_id="nope")
    _query(client, created_after="2020-01-01T00:00:00+03:00")
    _query(client, unknown="x")
    with app.state.session_factory() as session:
        assert session.query(TrustRoot).count() == 1


# --- 404 semantics ---------------------------------------------------------


def test_unknown_root_identifier_returns_404(client):
    assert _query(client, root_id=ZERO_UUID).status_code == 404


def test_cross_scope_root_identifier_is_indistinguishable_404(client):
    created = _create(client, name="root")
    assert (
        _query(client, tenant=OTHER_TENANT, root_id=created["root_id"]).status_code
        == 404
    )
    assert (
        _query(client, workload=OTHER_WORKLOAD, root_id=created["root_id"]).status_code
        == 404
    )


def test_explicit_unknown_identifier_404_even_with_other_matching_rows(client):
    _create(client, name="one")
    assert _query(client, root_id=ZERO_UUID).status_code == 404


# --- empty range / response shape ------------------------------------------


def test_empty_range_returns_empty_completed_page(client):
    response = _query(client)
    assert response.status_code == 200
    assert response.json() == {"trust_roots": [], "next_cursor": "", "complete": True}


def test_name_with_no_match_returns_empty_range_not_404(client):
    _create(client, name="one")
    response = _query(client, name="no-such-name")
    assert response.status_code == 200
    assert response.json() == {"trust_roots": [], "next_cursor": "", "complete": True}


def test_window_with_no_matches_returns_empty_completed_page(client):
    _create(client, name="one")
    response = _query(client, created_after="2030-01-01T00:00:00Z")
    assert response.status_code == 200
    assert response.json() == {"trust_roots": [], "next_cursor": "", "complete": True}


def test_response_is_compact_json_with_single_trailing_newline(client):
    _create(client, name="one")
    response = _query(client)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == ["trust_roots", "next_cursor", "complete"]


def test_entry_has_exact_shape_and_field_order(client):
    created = _create(client, name="primary")
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
    assert entry["root_id"] == created["root_id"]
    assert entry["tenant_id"] == TENANT
    assert entry["workload_id"] == WORKLOAD
    assert entry["name"] == "primary"
    assert entry["created_at"] == created["created_at"]
    assert entry["status"] == "active"
    assert entry["retired_at"] is None
    # No certificate PEM or other material leaks.
    assert "root_pem" not in entry
    assert "cert_sha256" not in entry
    for key, value in entry.items():
        assert value is None or isinstance(value, str)


def test_retired_entry_carries_status_and_retirement_time(client):
    created = _create(client, name="primary")
    retired = _retire(client, created["root_id"])
    entry = _query(client).json()["trust_roots"][0]
    assert entry["status"] == "retired"
    assert entry["retired_at"] == retired["retired_at"]
    assert entry["created_at"] == created["created_at"]


def test_name_absent_is_preserved_as_null(client):
    created = _create(client)  # no name
    entry = _query(client).json()["trust_roots"][0]
    assert entry["name"] is None
    assert entry["root_id"] == created["root_id"]


def test_no_floats_or_non_finite_values(client):
    _create(client, name="one")
    parsed = json.loads(_query(client).content)
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

    for row in parsed["trust_roots"]:
        for value in row.values():
            _check(value)


# --- ordering and scoping --------------------------------------------------


def test_results_ordered_by_created_at_then_root_id(client):
    created = [_create(client, name=f"root-{i}") for i in range(5)]
    rows = _query(client).json()["trust_roots"]
    keys = [(row["created_at"], row["root_id"]) for row in rows]
    assert keys == sorted(keys)
    assert [row["root_id"] for row in rows] == [c["root_id"] for c in created]


def test_results_are_scoped_to_tenant_and_workload(client):
    a = _create(client, name="a")
    b = _create(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD, name="b")
    rows_a = _query(client).json()["trust_roots"]
    rows_b = _query(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()[
        "trust_roots"
    ]
    assert [r["root_id"] for r in rows_a] == [a["root_id"]]
    assert [r["root_id"] for r in rows_b] == [b["root_id"]]


# --- filtering -------------------------------------------------------------


def test_filter_by_root_id_returns_exactly_that_registration(client):
    bodies = [_create(client, name=f"root-{i}") for i in range(3)]
    target = bodies[1]
    rows = _query(client, root_id=target["root_id"]).json()["trust_roots"]
    assert [r["root_id"] for r in rows] == [target["root_id"]]


def test_filter_by_name_is_exact_text(client):
    _create(client, name="alpha")
    _create(client, name="alpha-two")
    rows = _query(client, name="alpha").json()["trust_roots"]
    assert len(rows) == 1
    assert rows[0]["name"] == "alpha"


def test_name_matches_verbatim_including_surrounding_spaces(client):
    # A non-blank name with surrounding visible whitespace is a distinct
    # exact value; only an all-whitespace filter is a 422.
    _create(client, name="pad")
    _create(client, name=" pad ")
    assert _query(client, name="pad").json()["trust_roots"][0]["name"] == "pad"
    spaced = _query(client, name=" pad ").json()["trust_roots"]
    assert len(spaced) == 1 and spaced[0]["name"] == " pad "


def test_filter_by_status(client):
    active_a = _create(client, name="a")
    retiring = _create(client, name="b")
    _retire(client, retiring["root_id"])
    active_c = _create(client, name="c")

    active_rows = _query(client, status="active").json()["trust_roots"]
    assert {r["root_id"] for r in active_rows} == {
        active_a["root_id"],
        active_c["root_id"],
    }
    retired_rows = _query(client, status="retired").json()["trust_roots"]
    assert [r["root_id"] for r in retired_rows] == [retiring["root_id"]]
    assert all(r["status"] == "active" for r in active_rows)
    assert all(r["status"] == "retired" for r in retired_rows)


def _get_id(client, name):
    rows = _query(client, name=name).json()["trust_roots"]
    assert len(rows) == 1
    return rows[0]["root_id"]


def test_creation_window_is_inclusive_on_both_ends(client):
    # All three roots share effectively the same creation day; anchor on
    # the recorded timestamps of the first and last.
    first = _create(client, name="first")
    _create(client, name="middle")
    last = _create(client, name="last")
    ta = first["created_at"]
    tb = last["created_at"]
    if ta == tb:
        # Degenerate identical timestamps: the single-instant window must
        # include every root created at that instant.
        rows = _query(
            client, created_after=ta, created_before=tb
        ).json()["trust_roots"]
        assert {r["root_id"] for r in rows} == {
            first["root_id"],
            _get_id(client, "middle"),
            last["root_id"],
        }
        return
    middle_id = _get_id(client, "middle")
    rows = _query(client, created_after=ta, created_before=tb).json()["trust_roots"]
    assert {r["root_id"] for r in rows} == {
        first["root_id"],
        middle_id,
        last["root_id"],
    }
    only_first = _query(client, created_after=ta, created_before=ta).json()[
        "trust_roots"
    ]
    assert [r["root_id"] for r in only_first] == [first["root_id"]]


def test_equivalent_utc_spellings_share_one_cursor_domain(client, monkeypatch):
    for i in range(3):
        _create(client, name=f"root-{i}")
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 1)
    zed = _query(client, created_after="2000-01-01T00:00:00Z")
    cursor = zed.json()["next_cursor"]
    offset = _query(
        client, created_after="2000-01-01T00:00:00+00:00", cursor=cursor
    )
    assert offset.status_code == 200
    replayed = _query(client, created_after="2000-01-01T00:00:00Z", cursor=cursor)
    assert offset.content == replayed.content


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_root_once_in_order(client, monkeypatch):
    bodies = [_create(client, name=f"root-{i}") for i in range(7)]
    rows = _walk(client, page_size=3, monkeypatch=monkeypatch)
    assert len(rows) == 7
    keys = [(r["created_at"], r["root_id"]) for r in rows]
    assert keys == sorted(keys)
    assert {r["root_id"] for r in rows} == {b["root_id"] for b in bodies}


def test_page_carries_cursor_and_complete_flag(client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 2)
    bodies = [_create(client, name=f"root-{i}") for i in range(5)]

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
    walked = [r["root_id"] for r in first["trust_roots"]]
    walked += [r["root_id"] for r in second["trust_roots"]]
    walked += [r["root_id"] for r in last["trust_roots"]]
    assert walked == [b["root_id"] for b in bodies]


def test_replaying_same_cursor_returns_identical_page(client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 2)
    for i in range(5):
        _create(client, name=f"root-{i}")
    cursor = _query(client).json()["next_cursor"]
    first = _query(client, cursor=cursor)
    second = _query(client, cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_string_cursor_equals_default(client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 2)
    for i in range(3):
        _create(client, name=f"root-{i}")
    assert _query(client).content == _query(client, cursor="").content


def test_tampered_or_forged_cursor_returns_422(client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 1)
    for i in range(3):
        _create(client, name=f"root-{i}")
    cursor = _query(client).json()["next_cursor"]
    assert cursor

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _query(client, cursor=tampered).status_code == 422

    forged = app_module.b64url_encode(
        b'{"k":"trust-roots-v1"}' + b"0" * 32
    )
    assert _query(client, cursor=forged).status_code == 422


def test_cross_scope_cursor_returns_422(client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 1)
    for i in range(3):
        _create(client, name=f"root-{i}")
    cursor = _query(client).json()["next_cursor"]
    assert _query(client, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert _query(client, workload=OTHER_WORKLOAD, cursor=cursor).status_code == 422


def test_cursor_cannot_cross_filters(client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 1)
    bodies = [_create(client, name=f"root-{i}") for i in range(3)]
    cursor = _query(client).json()["next_cursor"]

    assert _query(client, root_id=bodies[0]["root_id"], cursor=cursor).status_code == 422
    assert _query(client, name="root-0", cursor=cursor).status_code == 422
    assert _query(client, status="active", cursor=cursor).status_code == 422
    assert (
        _query(client, created_after="2000-01-01T00:00:00Z", cursor=cursor).status_code
        == 422
    )
    assert (
        _query(client, created_before="2100-01-01T00:00:00Z", cursor=cursor).status_code
        == 422
    )


def test_cursor_rejects_other_cursor_families(client):
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

    foreign_revocation = _encode_revocation_cursor(
        TENANT, WORKLOAD, ZERO_UUID,
        "2020-01-01T00:00:00+00:00", ZERO_UUID,
        revocation_id="", certificate_fingerprint="",
        effective_after="", effective_before="",
        snapshot_at="2020-01-01T00:00:00+00:00", snapshot_id=ZERO_UUID,
    )
    assert _query(client, cursor=foreign_revocation).status_code == 422

    foreign_decision = _encode_decision_cursor(
        TENANT, WORKLOAD, "2020-01-01T00:00:00+00:00", ZERO_UUID,
        decision_id="", evidence_id="", policy_id="", status="",
        decided_after="", decided_before="", snapshot_seq=1,
    )
    assert _query(client, cursor=foreign_decision).status_code == 422


# --- snapshot stability ----------------------------------------------------


def test_first_query_snapshot_excludes_later_registrations(client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 2)
    bodies = [_create(client, name=f"root-{i}") for i in range(3)]

    first = _query(client).json()
    assert [r["root_id"] for r in first["trust_roots"]] == [
        b["root_id"] for b in bodies[:2]
    ]
    cursor = first["next_cursor"]

    # A registration committed after the first query never enters the
    # snapshot's later pages.
    late = _create(client, name="late")
    second = _query(client, cursor=cursor).json()
    assert [r["root_id"] for r in second["trust_roots"]] == [bodies[2]["root_id"]]
    assert second["complete"] is True
    assert second["next_cursor"] == ""

    # Replaying the cursor stays byte-for-byte stable.
    assert _query(client, cursor=cursor).json() == second

    # A fresh first query includes the later registration.
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert {r["root_id"] for r in fresh} == {
        b["root_id"] for b in bodies
    } | {late["root_id"]}


def test_snapshot_is_stable_under_concurrent_retirement(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 2)
    bodies = [_create(client, name=f"root-{i}") for i in range(6)]
    first = _query(client).json()
    assert len(first["trust_roots"]) == 2
    cursor = first["next_cursor"]

    # Retire every root after the snapshot was fixed. The replayed walk
    # must keep presenting all six as active with null retired_at and
    # never duplicate or skip one.
    def retire_later(root_id_value):
        response = TestClient(app).post(
            f"/v1/trust-roots/{root_id_value}/retire",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
        assert response.status_code == 200

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(retire_later, [b["root_id"] for b in bodies]))

    snapshot_rows = []
    token = cursor
    for _ in range(10):
        data = _query(client, cursor=token).json()
        snapshot_rows.extend(data["trust_roots"])
        if data["complete"]:
            break
        token = data["next_cursor"]

    assert [r["root_id"] for r in first["trust_roots"]] == [
        b["root_id"] for b in bodies[:2]
    ]
    snapshot_ids = [r["root_id"] for r in snapshot_rows]
    assert snapshot_ids == [b["root_id"] for b in bodies[2:]]
    for row in first["trust_roots"] + snapshot_rows:
        assert row["status"] == "active"
        assert row["retired_at"] is None

    # A fresh first query sees every root retired.
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert len(fresh) == 6
    assert all(r["status"] == "retired" for r in fresh)
    assert all(r["retired_at"] for r in fresh)


def test_snapshot_status_filter_stable_under_concurrent_retirement(client):
    bodies = [_create(client, name=f"root-{i}") for i in range(3)]
    # Fix the snapshot while all three are active, filtered to active.
    first = _query(client, status="active").json()
    assert {r["root_id"] for r in first["trust_roots"]} == {
        b["root_id"] for b in bodies
    }

    for created in bodies:
        _retire(client, created["root_id"])

    # The snapshot-less first page is already complete: the same filter
    # parameters from a fresh query now return an empty active range and a
    # complete retired range, while nothing about the old first page changes.
    fresh_active = _query(client, status="active").json()
    assert fresh_active["trust_roots"] == []
    fresh_retired = _query(client, status="retired").json()
    assert {r["root_id"] for r in fresh_retired["trust_roots"]} == {
        b["root_id"] for b in bodies
    }
    assert first["trust_roots"][0]["status"] == "active"


def test_retirement_between_pages_does_not_change_snapshot(client, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 1)
    bodies = [_create(client, name=f"root-{i}") for i in range(3)]
    first = _query(client).json()
    cursor = first["next_cursor"]

    # Retire the very root the next page will display.
    _retire(client, bodies[1]["root_id"])
    second = _query(client, cursor=cursor).json()
    assert [r["root_id"] for r in second["trust_roots"]] == [bodies[1]["root_id"]]
    # The snapshot still presents it as active with no retirement time.
    assert second["trust_roots"][0]["status"] == "active"
    assert second["trust_roots"][0]["retired_at"] is None

    last = _query(client, cursor=second["next_cursor"]).json()
    assert [r["root_id"] for r in last["trust_roots"]] == [bodies[2]["root_id"]]
    assert last["complete"] is True

    # A fresh query shows the middle root retired.
    fresh = _walk(client, page_size=1, monkeypatch=monkeypatch)
    middle = next(r for r in fresh if r["root_id"] == bodies[1]["root_id"])
    assert middle["status"] == "retired"
    assert middle["retired_at"]


# --- read-only behaviour ---------------------------------------------------


def test_query_changes_no_state_and_creates_no_counters_or_retirement(
    app, client
):
    created = _create(client, name="one")
    before = _query(client).content
    for _ in range(3):
        assert _query(client).status_code == 200
        assert _query(client, root_id=created["root_id"]).status_code == 200
    assert _query(client).content == before
    with app.state.session_factory() as session:
        root = session.get(TrustRoot, created["root_id"])
        assert root.status == "active"
        assert root.retired_at is None
        assert root.retired_seq is None
        # Reading never mints or advances the lifecycle counter beyond the
        # one sequence consumed by creation.
        counter = session.get(TrustRootCommitCounter, (TENANT, WORKLOAD))
        assert counter.last_seq == 1


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_half_page(app, client):
    for i in range(3):
        _create(client, name=f"root-{i}")

    def fail_scan(conn, cursor, statement, parameters, context, executemany):
        if "FROM trust_roots" in statement:
            raise RuntimeError("simulated storage failure")

    from sqlalchemy import event

    event.listen(app.state.engine, "before_cursor_execute", fail_scan)
    try:
        response = _query(client)
        # A storage failure is a 500 with no partial list.
        assert response.status_code == 500
        assert b'"trust_roots"' not in response.content
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_scan)

    recovered = _query(client)
    assert recovered.status_code == 200
    assert len(recovered.json()["trust_roots"]) == 3
    with app.state.session_factory() as session:
        assert session.query(TrustRoot).count() == 3


# --- persistence -----------------------------------------------------------


def test_roots_queryable_after_restart_and_cursor_replays(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 1)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    bodies = []
    for i in range(3):
        root_key, root_cert = make_root(common_name=f"restart-root-{i}")
        response = client1.post(
            "/v1/trust-roots",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "root_pem": pem(root_cert),
                "name": f"root-{i}",
            },
        )
        assert response.status_code == 201
        bodies.append(response.json())
    cursor = client1.get(
        "/v1/trust-roots",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()["next_cursor"]
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    page = client2.get(
        "/v1/trust-roots",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "cursor": cursor,
        },
    )
    assert page.status_code == 200
    assert [r["root_id"] for r in page.json()["trust_roots"]] == [
        bodies[1]["root_id"]
    ]

    monkeypatch.setattr(app_module, "TRUST_ROOT_PAGE_SIZE", 100)
    all_rows = client2.get(
        "/v1/trust-roots",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()["trust_roots"]
    assert [r["root_id"] for r in all_rows] == [b["root_id"] for b in bodies]
    second.state.engine.dispose()


# --- legacy migration ------------------------------------------------------


def test_legacy_sqlite_database_backfills_sequences_and_counters(
    tmp_path,
):
    from sqlalchemy import create_engine, text

    from proof_release.db import Base

    url = f"sqlite:///{tmp_path}/legacy.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # Build a pre-lifecycle-sequence database: drop the new columns,
        # the new indexes and the counter table.
        conn.execute(text("DROP INDEX IF EXISTS ix_trust_roots_scope_commit_seq"))
        conn.execute(text("DROP INDEX IF EXISTS ix_trust_roots_scope_created"))
        conn.execute(text("ALTER TABLE trust_roots DROP COLUMN retired_seq"))
        conn.execute(text("ALTER TABLE trust_roots DROP COLUMN commit_seq"))
        conn.execute(text("DROP TABLE trust_root_commit_counters"))
        conn.execute(
            text(
                "INSERT INTO trust_roots "
                "(root_id, tenant_id, workload_id, name, root_pem, "
                "cert_sha256, status, created_at, retired_at) VALUES "
                "('11111111-1111-4111-8111-111111111111', :t, :w, 'a', 'p', 'h1', "
                "'retired', '2026-01-01T00:00:01+00:00', '2026-01-03T00:00:00+00:00'),"
                "('22222222-2222-4222-8222-222222222222', :t, :w, 'b', 'p', 'h2', "
                "'active', '2026-01-02T00:00:00+00:00', NULL)"
            ),
            {"t": TENANT, "w": WORKLOAD},
        )
    engine.dispose()

    application = create_app(url)
    client = TestClient(application)
    data = client.get(
        "/v1/trust-roots",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    ids = [r["root_id"] for r in data["trust_roots"]]
    assert ids == [
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    ]
    by_id = {r["root_id"]: r for r in data["trust_roots"]}
    assert by_id["11111111-1111-4111-8111-111111111111"]["status"] == "retired"
    assert by_id["22222222-2222-4222-8222-222222222222"]["status"] == "active"

    # The next creation/retirement allocates past the seeded maximum (2
    # creations + 1 retirement = 3) without colliding.
    with application.state.session_factory() as session:
        counter = session.get(TrustRootCommitCounter, (TENANT, WORKLOAD))
        assert counter.last_seq == 3
    application.state.engine.dispose()
