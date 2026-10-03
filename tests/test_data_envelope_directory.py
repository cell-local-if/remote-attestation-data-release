"""Tests for GET /v1/data-envelopes (read-only envelope directory).

The directory inventories persisted envelopes by tenant/workload without
touching ciphertext material: it is narrowed by an optional explicit
data_id and an inclusive UTC creation-time window, ordered by data_id,
and paginated with opaque, scope/filter/snapshot-bound cursors. The
snapshot is fixed at a per-scope creation commit boundary, so envelopes
committed after a range began never enter its later pages even when
their data_id sorts earlier, and a concurrent rewrap changes only the
reported key_version, never ordering or membership. The endpoint never
writes, rewraps, audits or consumes rate-limit budget and returns only
metadata.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.app import (
    _encode_audit_event_cursor,
    _encode_cursor,
    _encode_grant_audit_cursor,
)
from proof_release.db import AuditEvent, DataEnvelope, DataEnvelopeCommitCounter
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
PAYLOAD = "super-secret-attestation-payload"

MASTER_KEY_BYTES = b"0123456789abcdef0123456789abcdef"
MASTER_KEY = b64url_encode(MASTER_KEY_BYTES)

KEY_V1 = b64url_encode(b"11111111111111111111111111111111")
KEY_V2 = b64url_encode(b"22222222222222222222222222222222")
KEYRING_V1 = json.dumps({"current_version": 1, "keys": {"1": KEY_V1}})
KEYRING_V1_V2 = json.dumps(
    {"current_version": 2, "keys": {"1": KEY_V1, "2": KEY_V2}}
)

ZERO_UUID = "00000000-0000-0000-0000-000000000000"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = app_module.create_app(f"sqlite:///{tmp_path}/directory.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, data_id, *, tenant=TENANT, workload=WORKLOAD, payload=PAYLOAD):
    response = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": payload,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _query(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    query = {"tenant_id": tenant, "workload_id": workload}
    query.update(params)
    return client.get("/v1/data-envelopes", params=query)


def _walk(client, *, page_size=None, monkeypatch=None, **params):
    token = None
    rows = []
    for _ in range(50):
        query_params = dict(params)
        if token is not None:
            query_params["cursor"] = token
        if page_size is not None:
            monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", page_size)
        page = _query(client, **query_params)
        assert page.status_code == 200, page.text
        data = page.json()
        rows.extend(data["envelopes"])
        if data["complete"]:
            assert data["next_cursor"] == ""
            return rows
        token = data["next_cursor"]
        assert token
    raise AssertionError("pagination never completed")  # pragma: no cover


# --- request validation ----------------------------------------------------


def test_query_requires_scope_parameters(client):
    assert client.get("/v1/data-envelopes").status_code == 422
    assert (
        client.get(
            "/v1/data-envelopes", params={"workload_id": WORKLOAD}
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/v1/data-envelopes", params={"tenant_id": TENANT}
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
        {"data_id": ""},
        {"data_id": "   "},
        {"created_after": ""},
        {"created_after": "   "},
        {"created_before": "\t"},
        {"cursor": "   "},
    ],
)
def test_blank_values_are_422(client, params):
    assert _query(client, **params).status_code == 422


def test_unknown_parameter_is_422(client):
    assert (
        _query(client, unexpected="x").status_code == 422
    )


def test_repeated_parameter_is_422(app, client):
    def fail_read(conn, cursor, statement, parameters, context, executemany):
        if "FROM data_envelopes" in statement:
            raise AssertionError("storage must not be read for a malformed query")

    from sqlalchemy import event

    event.listen(app.state.engine, "before_cursor_execute", fail_read)
    try:
        response = client.get(
            "/v1/data-envelopes",
            params=[
                ("tenant_id", TENANT),
                ("workload_id", WORKLOAD),
                ("data_id", "a"),
                ("data_id", "b"),
            ],
        )
        assert response.status_code == 422
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_read)


@pytest.mark.parametrize("body", [b"{}", b"  ", b"not json", b"\x00"])
def test_non_empty_body_is_422_before_state_read(app, client, body):
    def fail_read(conn, cursor, statement, parameters, context, executemany):
        if "FROM data_envelopes" in statement:
            raise AssertionError("storage must not be read for a bodied query")

    from sqlalchemy import event

    event.listen(app.state.engine, "before_cursor_execute", fail_read)
    try:
        response = client.request(
            "GET",
            "/v1/data-envelopes",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            content=body,
        )
        assert response.status_code == 422
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_read)


@pytest.mark.parametrize(
    "value",
    [
        "2020-01-01T00:00:00",          # naive, no offset
        "2020-01-01T00:00:00+03:00",   # non-UTC offset
        "2020-01-01T00:00:00-05:00",   # non-UTC offset
        "2020-01-01",                  # date only
        "not-a-timestamp",
        "2020-01-01T00:00:00+0000",    # RFC3339 needs the colon in the offset
    ],
)
def test_non_utc_or_malformed_timestamp_is_422(client, value):
    assert _query(client, created_after=value).status_code == 422
    assert _query(client, created_before=value).status_code == 422


@pytest.mark.parametrize("value", ["2020-01-01T00:00:00Z", "2020-01-01T00:00:00+00:00"])
def test_utc_timestamps_are_accepted(client, value):
    assert _query(client, created_after=value).status_code == 200
    assert _query(client, created_before=value).status_code == 200


def test_inverted_window_is_422(client):
    response = _query(
        client,
        created_after="2020-02-01T00:00:00Z",
        created_before="2020-01-01T00:00:00Z",
    )
    assert response.status_code == 422


def test_equal_window_bounds_are_valid(client):
    assert (
        _query(
            client,
            created_after="2020-01-01T00:00:00Z",
            created_before="2020-01-01T00:00:00Z",
        ).status_code
        == 200
    )


# --- 404 semantics ---------------------------------------------------------


def test_explicit_unknown_data_id_returns_404(client):
    assert _query(client, data_id="missing").status_code == 404


def test_explicit_cross_scope_data_id_is_indistinguishable_404(client):
    _create(client, "item")
    assert (
        _query(client, tenant=OTHER_TENANT, data_id="item").status_code == 404
    )
    assert (
        _query(client, workload=OTHER_WORKLOAD, data_id="item").status_code == 404
    )


def test_unknown_data_id_404_even_with_other_rows_present(client):
    _create(client, "present")
    assert _query(client, data_id="absent").status_code == 404


# --- empty range / response shape ------------------------------------------


def test_empty_scope_returns_empty_completed_page(client):
    response = _query(client)
    assert response.status_code == 200
    assert response.json() == {
        "envelopes": [],
        "next_cursor": "",
        "complete": True,
    }


def test_window_with_no_matches_returns_empty_range_not_404(client):
    _create(client, "item")
    response = _query(client, created_after="2030-01-01T00:00:00Z")
    assert response.status_code == 200
    assert response.json() == {
        "envelopes": [],
        "next_cursor": "",
        "complete": True,
    }


def test_response_is_compact_json_with_single_trailing_newline(client):
    _create(client, "item")
    raw = _query(client).content
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == ["envelopes", "next_cursor", "complete"]


def test_entry_has_exact_metadata_shape_and_field_order(client):
    created = _create(client, "item")
    entry = _query(client).json()["envelopes"][0]
    assert list(entry) == [
        "data_id",
        "tenant_id",
        "workload_id",
        "key_version",
        "created_at",
    ]
    assert entry == {
        "data_id": "item",
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "key_version": 1,
        "created_at": created["created_at"],
    }
    # No material or secret field is ever returned.
    for name in (
        "ciphertext",
        "iv",
        "tag",
        "wrapped_key",
        "payload",
        "data_key",
        "master_key",
    ):
        assert name not in entry
    assert PAYLOAD.encode() not in _query(client).content


def test_results_ordered_by_data_id_regardless_of_creation_order(client):
    for data_id in ("c", "a", "b", "A", "B"):
        _create(client, data_id)
    rows = _query(client).json()["envelopes"]
    assert [r["data_id"] for r in rows] == ["A", "B", "a", "b", "c"]
    for row in rows:
        assert row["tenant_id"] == TENANT
        assert row["workload_id"] == WORKLOAD


def test_scope_isolation(client):
    _create(client, "a", tenant=TENANT, workload=WORKLOAD)
    _create(client, "b", tenant=OTHER_TENANT, workload=WORKLOAD)
    _create(client, "c", tenant=TENANT, workload=OTHER_WORKLOAD)

    own = [r["data_id"] for r in _query(client).json()["envelopes"]]
    other_tenant = [
        r["data_id"]
        for r in _query(client, tenant=OTHER_TENANT).json()["envelopes"]
    ]
    other_workload = [
        r["data_id"]
        for r in _query(client, workload=OTHER_WORKLOAD).json()["envelopes"]
    ]
    assert own == ["a"]
    assert other_tenant == ["b"]
    assert other_workload == ["c"]


def test_creation_window_is_inclusive_on_both_ends(client):
    first = _create(client, "a")
    _create(client, "b")
    instant = first["created_at"]
    rows = _query(
        client, created_after=instant, created_before=instant
    ).json()["envelopes"]
    assert [r["data_id"] for r in rows] == ["a"]


def test_explicit_data_id_returns_only_that_envelope(client):
    _create(client, "a")
    _create(client, "b")
    rows = _query(client, data_id="b").json()["envelopes"]
    assert [r["data_id"] for r in rows] == ["b"]


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_envelope_once_in_order(client, monkeypatch):
    for i in range(5):
        _create(client, f"id-{i:02d}")
    rows = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert [r["data_id"] for r in rows] == [f"id-{i:02d}" for i in range(5)]


def test_page_carries_cursor_and_complete_flag(client, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 2)
    for i in range(5):
        _create(client, f"id-{i}")

    first = _query(client).json()
    assert len(first["envelopes"]) == 2
    assert first["complete"] is False
    assert first["next_cursor"]

    second = _query(client, cursor=first["next_cursor"]).json()
    assert len(second["envelopes"]) == 2
    assert second["complete"] is False

    last = _query(client, cursor=second["next_cursor"]).json()
    assert len(last["envelopes"]) == 1
    assert last["complete"] is True
    assert last["next_cursor"] == ""


def test_exact_multiple_of_page_size_reports_completion(client, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 2)
    for i in range(4):
        _create(client, f"id-{i}")

    first = _query(client).json()
    assert len(first["envelopes"]) == 2 and first["complete"] is False
    second = _query(client, cursor=first["next_cursor"]).json()
    # The extra-row probe distinguishes "full last page" from "more".
    assert len(second["envelopes"]) == 2
    assert second["complete"] is True
    assert second["next_cursor"] == ""


def test_replaying_same_cursor_returns_identical_page(client, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 2)
    for i in range(5):
        _create(client, f"id-{i}")
    cursor = _query(client).json()["next_cursor"]
    first = _query(client, cursor=cursor)
    second = _query(client, cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_string_cursor_equals_default(client, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 2)
    for i in range(3):
        _create(client, f"id-{i}")
    assert _query(client).content == _query(client, cursor="").content


# --- cursor security -------------------------------------------------------


def test_tampered_or_forged_cursor_returns_422(client, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 1)
    for i in range(3):
        _create(client, f"id-{i}")
    cursor = _query(client).json()["next_cursor"]
    assert cursor

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _query(client, cursor=tampered).status_code == 422

    forged = app_module.b64url_encode(
        b'{"k":"data-envelopes-v1"}' + b"0" * 32
    )
    assert _query(client, cursor=forged).status_code == 422


def test_cross_scope_cursor_returns_422(client, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 1)
    for i in range(3):
        _create(client, f"id-{i}")
    cursor = _query(client).json()["next_cursor"]
    assert _query(client, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert _query(client, workload=OTHER_WORKLOAD, cursor=cursor).status_code == 422


def test_cursor_cannot_cross_filters(client, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 1)
    _create(client, "a")
    _create(client, "b")
    cursor = _query(client).json()["next_cursor"]

    assert _query(client, data_id="a", cursor=cursor).status_code == 422
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


def test_cursor_rejects_other_cursor_families(client):
    foreign_rewrap = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _query(client, cursor=foreign_rewrap).status_code == 422

    foreign_grant = _encode_grant_audit_cursor(
        TENANT,
        WORKLOAD,
        ZERO_UUID,
        grant_id="",
        decision_id="",
        data_id="",
        status="",
        issued_after="",
        issued_before="",
    )
    assert _query(client, cursor=foreign_grant).status_code == 422

    foreign_audit = _encode_audit_event_cursor(
        TENANT,
        WORKLOAD,
        "2020-01-01T00:00:00+00:00",
        ZERO_UUID,
        event_id="",
        event_type="",
        status="",
        occurred_after="",
        occurred_before="",
    )
    assert _query(client, cursor=foreign_audit).status_code == 422


# --- snapshot stability ----------------------------------------------------


def test_first_query_snapshot_excludes_later_creations(client, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 2)
    _create(client, "m1")
    _create(client, "m2")
    _create(client, "m3")

    first = _query(client).json()
    assert [r["data_id"] for r in first["envelopes"]] == ["m1", "m2"]
    cursor = first["next_cursor"]

    # A late envelope whose data_id sorts BEFORE both members of the
    # snapshot still cannot enter it: the snapshot is commit-bounded, not
    # merely key-bounded.
    _create(client, "a0-late")
    second = _query(client, cursor=cursor).json()
    assert [r["data_id"] for r in second["envelopes"]] == ["m3"]
    assert second["complete"] is True
    assert second["next_cursor"] == ""

    # Replaying stays byte-for-byte stable.
    assert _query(client, cursor=cursor).json() == second

    # A fresh first query sees the late envelope.
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert [r["data_id"] for r in fresh] == ["a0-late", "m1", "m2", "m3"]


def test_creation_between_pages_does_not_duplicate_or_skip(client, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 2)
    ids = [f"id-{i}" for i in range(4)]
    for data_id in ids:
        _create(client, data_id)

    cursor = _query(client).json()["next_cursor"]
    # New rows (some sorting inside the remaining window) commit mid-walk.
    _create(client, "id-1-late")
    _create(client, "zzz-late")

    snapshot_ids = []
    token = cursor
    for _ in range(10):
        data = _query(client, cursor=token).json()
        snapshot_ids.extend(r["data_id"] for r in data["envelopes"])
        if data["complete"]:
            break
        token = data["next_cursor"]
    # Later pages of the fixed snapshot contain exactly the original tail,
    # once each, in order — neither late row leaks in and nothing is
    # skipped or duplicated.
    assert snapshot_ids == ["id-2", "id-3"]

    # A fresh walk (new snapshot) contains every committed envelope once.
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert [r["data_id"] for r in fresh] == [
        "id-0",
        "id-1",
        "id-1-late",
        "id-2",
        "id-3",
        "zzz-late",
    ]


def test_rewrap_changes_only_key_version_not_order_or_membership(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1)
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    for i in range(3):
        _create(client, f"id-{i}")
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 2)

    first = _query(client).json()
    assert {r["key_version"] for r in first["envelopes"]} == {1}
    cursor = first["next_cursor"]

    # Rotate the keyring and rewrap every envelope after the snapshot was
    # fixed. Ordering and membership must not move; key_version reflects
    # the committed current version.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    for i in range(3):
        rewrapped = client.post(
            f"/v1/data-envelopes/id-{i}/rewrap",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
        assert rewrapped.status_code == 200, rewrapped.text
        assert rewrapped.json()["key_version"] == 2

    second = _query(client, cursor=cursor).json()
    assert [r["data_id"] for r in second["envelopes"]] == ["id-2"]
    assert second["envelopes"][0]["key_version"] == 2
    first_replayed = _query(client).json()
    assert [r["data_id"] for r in first_replayed["envelopes"]] == [
        "id-0",
        "id-1",
    ]
    assert {r["key_version"] for r in first_replayed["envelopes"]} == {2}

    # A fresh full walk keeps the same members, all at version 2.
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert [r["data_id"] for r in fresh] == ["id-0", "id-1", "id-2"]
    assert {r["key_version"] for r in fresh} == {2}


def test_equivalent_utc_spellings_share_one_cursor_domain(client, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 1)
    _create(client, "a")
    _create(client, "b")
    cursor = _query(
        client, created_after="2000-01-01T00:00:00Z"
    ).json()["next_cursor"]
    # The +00:00 spelling is the same instant and must not mint a new
    # cursor domain: the cursor replays rather than 422.
    replayed = _query(
        client, created_after="2000-01-01T00:00:00+00:00", cursor=cursor
    )
    assert replayed.status_code == 200


# --- read-only guarantees --------------------------------------------------


def test_directory_works_without_keyring(app, client, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    # Insert a row directly: the read path neither loads nor needs the
    # keyring, unlike create/rewrap.
    with app.state.session_factory() as session:
        from datetime import datetime, timezone

        session.add(
            DataEnvelope(
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                data_id="keyless",
                key_version=1,
                ciphertext=b"c",
                iv=b"i" * 12,
                tag=b"t" * 16,
                wrapped_key=b"k" * 40,
                created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                commit_seq=1,
            )
        )
        # The snapshot high-water mark the read path bounds membership by.
        session.add(
            DataEnvelopeCommitCounter(
                tenant_id=TENANT, workload_id=WORKLOAD, last_seq=1
            )
        )
        session.commit()
    response = _query(client)
    assert response.status_code == 200
    assert response.json()["envelopes"][0]["data_id"] == "keyless"


def test_directory_writes_no_audit_and_no_state(app, client):
    _create(client, "a")
    with app.state.session_factory() as session:
        audits_before = session.query(AuditEvent).count()
        counters_before = session.query(DataEnvelopeCommitCounter).count()
        envelopes_before = session.query(DataEnvelope).count()

    for _ in range(3):
        assert _query(client).status_code == 200

    with app.state.session_factory() as session:
        assert session.query(AuditEvent).count() == audits_before
        assert session.query(DataEnvelopeCommitCounter).count() == counters_before
        assert session.query(DataEnvelope).count() == envelopes_before


def test_storage_failure_returns_500_and_no_half_page(app, client):
    for i in range(3):
        _create(client, f"id-{i}")

    def fail_scan(conn, cursor, statement, parameters, context, executemany):
        if "FROM data_envelopes" in statement:
            raise RuntimeError("simulated storage failure")

    from sqlalchemy import event

    event.listen(app.state.engine, "before_cursor_execute", fail_scan)
    try:
        response = _query(client)
        assert response.status_code == 500
        assert b'"envelopes"' not in response.content
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_scan)

    recovered = _query(client)
    assert recovered.status_code == 200
    assert [r["data_id"] for r in recovered.json()["envelopes"]] == [
        "id-0",
        "id-1",
        "id-2",
    ]


# --- persistence -----------------------------------------------------------


def test_envelopes_queryable_after_restart_and_cursor_replays(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 1)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = app_module.create_app(url)
    client1 = TestClient(first)
    for i in range(3):
        _create(client1, f"id-{i}")
    cursor = _query(client1).json()["next_cursor"]
    first.state.engine.dispose()

    second = app_module.create_app(url)
    client2 = TestClient(second)
    replayed = _query(client2, cursor=cursor)
    assert replayed.status_code == 200
    assert replayed.json()["envelopes"][0]["data_id"] == "id-1"
    rows = _walk(client2, page_size=1, monkeypatch=monkeypatch)
    assert [r["data_id"] for r in rows] == ["id-0", "id-1", "id-2"]
    second.state.engine.dispose()
