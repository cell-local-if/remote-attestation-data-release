"""Tests for GET /v1/data-envelopes (read-only envelope directory)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event as sa_event, text

from proof_release import app as app_module
from proof_release.db import (
    AuditEvent,
    DataEnvelope,
    DataEnvelopeCommitCounter,
)
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
PAYLOAD = "directory-secret-payload 🔐"

# 32-byte fixed master keys for tests, rendered as unpadded base64url.
KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V2_BYTES = b"fedcba9876543210fedcba9876543210"
KEY_V1 = b64url_encode(KEY_V1_BYTES)
KEY_V2 = b64url_encode(KEY_V2_BYTES)


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


KEYRING_V1_V2 = _keyring(2, {1: KEY_V1, 2: KEY_V2})


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = app_module.create_app(f"sqlite:///{tmp_path}/directory.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, data_id, *, tenant=TENANT, workload=WORKLOAD, payload=PAYLOAD):
    return client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": payload,
        },
    )


def _seed(client, data_ids, **kwargs):
    for data_id in data_ids:
        assert _create(client, data_id, **kwargs).status_code == 201


def _query(client, *, cursor=None, tenant=TENANT, workload=WORKLOAD, **params):
    if cursor is not None:
        params["cursor"] = cursor
    return client.get(
        "/v1/data-envelopes",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


def _walk(client, *, page_size=None, monkeypatch=None, **params):
    if page_size is not None:
        assert monkeypatch is not None
        monkeypatch.setattr(
            app_module, "DATA_ENVELOPE_DIRECTORY_PAGE_SIZE", page_size
        )
    seen = []
    cursor = None
    while True:
        data = _query(client, cursor=cursor, **params).json()
        seen.extend(data["envelopes"])
        if data["complete"]:
            assert data["next_cursor"] == ""
            break
        cursor = data["next_cursor"]
        assert cursor
    return seen


def _insert_direct(app, data_id, created_at, *, tenant=TENANT, workload=WORKLOAD,
                   key_version=1):
    """Insert an envelope at a fixed time with dummy material columns."""
    with app.state.session_factory() as session:
        commit_seq = app_module._next_data_envelope_commit_seq(
            session, tenant, workload
        )
        session.add(
            DataEnvelope(
                tenant_id=tenant,
                workload_id=workload,
                data_id=data_id,
                key_version=key_version,
                ciphertext=b"\x01" * 8,
                iv=b"\x02" * 12,
                tag=b"\x03" * 16,
                wrapped_key=b"\x04" * 40,
                created_at=created_at,
                commit_seq=commit_seq,
            )
        )
        session.commit()


# --- basic shape -----------------------------------------------------------


def test_empty_scope_returns_empty_range(client):
    response = _query(client)
    assert response.status_code == 200
    assert response.json() == {"envelopes": [], "next_cursor": "", "complete": True}


def test_entries_are_metadata_only_in_fixed_key_order(client):
    created = _create(client, "item-1")
    assert created.status_code == 201
    created_at = created.json()["created_at"]

    response = _query(client)
    assert response.status_code == 200
    # Outer keys, in order; entry keys, in order.
    assert list(json.loads(response.content)) == [
        "envelopes",
        "next_cursor",
        "complete",
    ]
    (entry,) = response.json()["envelopes"]
    assert list(entry) == [
        "data_id",
        "tenant_id",
        "workload_id",
        "key_version",
        "created_at",
    ]
    assert entry == {
        "data_id": "item-1",
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "key_version": 1,
        "created_at": created_at,
    }
    # No material field name, payload or serialized material ever appears.
    single = client.get(
        "/v1/data-envelopes/item-1",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    for field in ("ciphertext", "iv", "tag", "wrapped_key"):
        assert single[field] not in response.text
        assert field not in entry
    assert "payload" not in response.text
    assert PAYLOAD not in response.text


def test_body_is_compact_json_terminated_by_one_newline(client):
    _seed(client, ["a", "b"])
    response = _query(client)
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    # Compact separators: no whitespace after commas or colons.
    assert b", " not in response.content
    assert b": " not in response.content


def test_results_are_ordered_by_data_id_ascending(client):
    _seed(client, ["zeta", "alpha", "mid", "Beta", "000-first"])
    ids = [e["data_id"] for e in _walk(client)]
    assert ids == ["000-first", "Beta", "alpha", "mid", "zeta"]


def test_scope_isolation(client):
    _create(client, "a", tenant=TENANT, workload=WORKLOAD)
    _create(client, "b", tenant=OTHER_TENANT, workload=WORKLOAD)
    _create(client, "c", tenant=TENANT, workload=OTHER_WORKLOAD)

    assert [e["data_id"] for e in _walk(client)] == ["a"]
    assert [e["data_id"] for e in _walk(client, tenant=OTHER_TENANT)] == ["b"]
    assert [
        e["data_id"] for e in _walk(client, workload=OTHER_WORKLOAD)
    ] == ["c"]
    # A scope that never existed is an empty range, not an error.
    assert _query(client, tenant="nobody", workload="nothing").json() == {
        "envelopes": [],
        "next_cursor": "",
        "complete": True,
    }


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_envelope_once_in_order(client, monkeypatch):
    _seed(client, [f"item-{i:03d}" for i in range(7)])
    seen = _walk(client, page_size=3, monkeypatch=monkeypatch)
    assert [e["data_id"] for e in seen] == [
        f"item-{i:03d}" for i in range(7)
    ]


def test_page_carries_cursor_and_complete_flag(client, monkeypatch):
    _seed(client, ["a", "b", "c", "d", "e"])
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_DIRECTORY_PAGE_SIZE", 2)

    first = _query(client).json()
    assert len(first["envelopes"]) == 2
    assert first["complete"] is False
    assert first["next_cursor"]

    second = _query(client, cursor=first["next_cursor"]).json()
    assert [e["data_id"] for e in second["envelopes"]] == ["c", "d"]
    assert second["complete"] is False

    last = _query(client, cursor=second["next_cursor"]).json()
    assert [e["data_id"] for e in last["envelopes"]] == ["e"]
    assert last["complete"] is True
    assert last["next_cursor"] == ""


def test_replaying_same_cursor_returns_identical_page(client, monkeypatch):
    _seed(client, ["a", "b", "c", "d"])
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_DIRECTORY_PAGE_SIZE", 2)
    cursor = _query(client).json()["next_cursor"]

    first = _query(client, cursor=cursor)
    second = _query(client, cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_string_cursor_equals_default(client, monkeypatch):
    _seed(client, ["a", "b", "c"])
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_DIRECTORY_PAGE_SIZE", 1)
    default = _query(client).json()
    explicit_empty = _query(client, cursor="").json()
    assert default == explicit_empty


# --- fixed snapshot semantics ----------------------------------------------


def test_new_envelopes_never_enter_a_walk_already_started(client, monkeypatch):
    _seed(client, ["a", "b", "c", "d", "e"])
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_DIRECTORY_PAGE_SIZE", 2)

    first = _query(client).json()
    cursor = first["next_cursor"]
    # Concurrently committed envelopes: none may enter this walk.
    _create(client, "f-later")
    _create(client, "aa-early-sort-position")

    seen = list(first["envelopes"])
    while True:
        page = _query(client, cursor=cursor).json()
        seen.extend(page["envelopes"])
        if page["complete"]:
            break
        cursor = page["next_cursor"]
    assert sorted(e["data_id"] for e in seen) == ["a", "b", "c", "d", "e"]

    # A fresh, cursor-less query fixes a new snapshot and sees everything.
    fresh = [e["data_id"] for e in _walk(client)]
    assert fresh == ["a", "aa-early-sort-position", "b", "c", "d", "e", "f-later"]


def test_empty_first_page_snapshot_stays_empty_until_fresh_query(client):
    # First query against the empty scope fixes an empty snapshot.
    first = _query(client).json()
    assert first == {"envelopes": [], "next_cursor": "", "complete": True}
    _create(client, "later")
    # There is no cursor to replay; a repeat parameter shape is itself a
    # fresh first query and therefore sees the now-committed envelope.
    assert [e["data_id"] for e in _walk(client)] == ["later"]


def test_rewrap_changes_only_committed_key_version_not_membership(
    app, client, monkeypatch
):
    _seed(client, ["a", "b", "c"])
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_DIRECTORY_PAGE_SIZE", 2)
    first = _query(client).json()
    cursor = first["next_cursor"]

    # Rotate an envelope that belongs to the remaining snapshot page.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    rewrapped = client.post(
        "/v1/data-envelopes/c/rewrap",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert rewrapped.status_code == 200
    assert rewrapped.json()["key_version"] == 2

    second = _query(client, cursor=cursor).json()
    assert [e["data_id"] for e in second["envelopes"]] == ["c"]
    # The fixed snapshot page reports the committed current key version.
    assert second["envelopes"][0]["key_version"] == 2
    assert second["complete"] is True

    # The rewrap allocated no creation sequence: order is data_id order and
    # the counter still stands at the number of creations (3).
    with app.state.session_factory() as session:
        counter = session.get(DataEnvelopeCommitCounter, (TENANT, WORKLOAD))
        assert counter.last_seq == 3


# --- filters ---------------------------------------------------------------


def test_explicit_data_id_returns_just_that_envelope(client):
    _seed(client, ["a", "b", "c"])
    response = _query(client, data_id="b")
    assert response.status_code == 200
    assert [e["data_id"] for e in response.json()["envelopes"]] == ["b"]
    assert response.json()["complete"] is True


def test_explicit_data_id_outside_window_returns_empty_range(client):
    _create(client, "a")
    response = _query(
        client,
        data_id="a",
        created_after="2099-01-01T00:00:00Z",
    )
    assert response.status_code == 200
    assert response.json() == {"envelopes": [], "next_cursor": "", "complete": True}


def test_created_window_is_inclusive_on_both_ends(app, client):
    t = datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)
    _insert_direct(app, "at-t", t)
    _insert_direct(app, "before", datetime(2026, 3, 4, 11, 59, 59, tzinfo=timezone.utc))
    _insert_direct(app, "after", datetime(2026, 3, 4, 12, 0, 1, tzinfo=timezone.utc))

    instant = "2026-03-04T12:00:00Z"
    ids = [
        e["data_id"]
        for e in _walk(client, created_after=instant, created_before=instant)
    ]
    assert ids == ["at-t"]

    windowed = _query(
        client,
        created_after="2026-03-04T12:00:00+00:00",
        created_before="2026-03-04T12:00:00+00:00",
    )
    assert [e["data_id"] for e in windowed.json()["envelopes"]] == ["at-t"]


def test_window_filters_match_nothing_as_empty_range(app, client):
    _insert_direct(app, "a", datetime(2026, 1, 1, tzinfo=timezone.utc))
    response = _query(client, created_after="2027-01-01T00:00:00Z")
    assert response.status_code == 200
    assert response.json() == {"envelopes": [], "next_cursor": "", "complete": True}


def test_equivalent_utc_spellings_share_one_cursor_domain(client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_DIRECTORY_PAGE_SIZE", 1)

    zed = _query(client, created_after="2000-01-01T00:00:00Z")
    assert zed.status_code == 200
    cursor = zed.json()["next_cursor"]
    offset = _query(
        client, cursor=cursor, created_after="2000-01-01T00:00:00+00:00"
    )
    assert offset.status_code == 200
    assert offset.json()["complete"] is True


# --- 404 -------------------------------------------------------------------


def test_explicit_unknown_or_cross_scope_data_id_returns_404(client):
    _create(client, "a")
    assert _query(client, data_id="missing").status_code == 404
    assert _query(client, data_id="a", tenant=OTHER_TENANT).status_code == 404
    assert _query(client, data_id="a", workload=OTHER_WORKLOAD).status_code == 404
    # A 404 leaks no material.
    response = _query(client, data_id="missing")
    assert "ciphertext" not in response.text


# --- 422 request shape ------------------------------------------------------


def test_requires_scope_parameters(client):
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
    ],
)
def test_blank_values_return_422(client, params):
    base = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    base.update(params)
    assert client.get("/v1/data-envelopes", params=base).status_code == 422


def test_unknown_query_parameter_returns_422(client):
    response = _query(client, unexpected="x")
    assert response.status_code == 422


def test_repeated_query_parameter_returns_422(client):
    response = client.get(
        "/v1/data-envelopes",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("tenant_id", OTHER_TENANT),
        ],
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-01T00:00:00",  # naive
        "2026-01-01T00:00:00+01:00",  # non-UTC offset
        "2026-01-01T00:00:00-05:00",
        "not-a-timestamp",
        "2026-13-40T99:00:00Z",
        "   ",
    ],
)
def test_invalid_created_after_returns_422(client, value):
    assert _query(client, created_after=value).status_code == 422


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-01T00:00:00",
        "2026-01-01T00:00:00-00:01",
        "garbage",
        "  ",
    ],
)
def test_invalid_created_before_returns_422(client, value):
    assert _query(client, created_before=value).status_code == 422


def test_inverted_window_returns_422(client):
    response = _query(
        client,
        created_after="2026-02-02T00:00:00Z",
        created_before="2026-02-01T00:00:00Z",
    )
    assert response.status_code == 422


def test_non_empty_body_returns_422(client):
    url = f"/v1/data-envelopes?tenant_id={TENANT}&workload_id={WORKLOAD}"
    assert client.request("GET", url, content=b"{}").status_code == 422
    assert client.request("GET", url, content=b" ").status_code == 422
    # An explicitly empty body is fine.
    assert client.request("GET", url, content=b"").status_code == 200


# --- 422 cursors -----------------------------------------------------------


def test_tampered_or_forged_cursor_returns_422(client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_DIRECTORY_PAGE_SIZE", 1)
    cursor = _query(client).json()["next_cursor"]

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _query(client, cursor=tampered).status_code == 422
    assert _query(client, cursor="not-base64!!!").status_code == 422
    assert _query(client, cursor="   ").status_code == 422
    assert _query(client, cursor=b64url_encode(b"forged")).status_code == 422


def test_cursor_from_another_query_family_returns_422(client):
    _create(client, "a")
    # A rewrap-batch cursor is HMAC-authenticated with the same secret but
    # belongs to a different family.
    foreign = app_module._encode_cursor(TENANT, WORKLOAD, "a")
    assert _query(client, cursor=foreign).status_code == 422
    # A grant-audit cursor likewise cannot be replayed here.
    foreign_grant = app_module._encode_grant_audit_cursor(
        TENANT,
        WORKLOAD,
        "grant-id",
        grant_id="grant-id",
        decision_id="",
        data_id="",
        status="",
        issued_after="",
        issued_before="",
    )
    assert _query(client, cursor=foreign_grant).status_code == 422


def test_cross_scope_cursor_returns_422(client, monkeypatch):
    _seed(client, ["a", "a2"])
    _create(client, "b", tenant=OTHER_TENANT)
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_DIRECTORY_PAGE_SIZE", 1)
    cursor = _query(client).json()["next_cursor"]
    assert cursor

    assert _query(client, cursor=cursor, tenant=OTHER_TENANT).status_code == 422
    assert _query(client, cursor=cursor, workload=OTHER_WORKLOAD).status_code == 422


def test_cursor_cannot_cross_filters(client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_DIRECTORY_PAGE_SIZE", 1)
    cursor = _query(
        client, created_after="2000-01-01T00:00:00Z"
    ).json()["next_cursor"]

    # Same scope, different effective filters: the signed cursor must not
    # move across filter domains.
    assert _query(client, cursor=cursor).status_code == 422
    assert (
        _query(
            client,
            cursor=cursor,
            created_after="2001-01-01T00:00:00Z",
        ).status_code
        == 422
    )
    assert (
        _query(
            client,
            cursor=cursor,
            created_after="2000-01-01T00:00:00Z",
            created_before="2030-01-01T00:00:00Z",
        ).status_code
        == 422
    )
    assert _query(client, cursor=cursor, data_id="a").status_code == 422


def test_cursor_signed_with_another_secret_is_invalid(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_DIRECTORY_PAGE_SIZE", 1)
    cursor = _query(client).json()["next_cursor"]

    monkeypatch.setenv("PROOF_RELEASE_CURSOR_SECRET", "a-different-secret")
    assert _query(client, cursor=cursor).status_code == 422


# --- read-only behaviour ----------------------------------------------------


def test_directory_writes_no_audit(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert (
        client.post(
            "/v1/data-envelopes/a/rewrap",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).status_code
        == 200
    )

    def _audit_count():
        with app.state.session_factory() as session:
            return session.query(AuditEvent).count()

    after_rewrap = _audit_count()
    assert after_rewrap >= 1  # the rewrap itself writes one audit row
    for _ in range(3):
        assert _query(client).status_code == 200
        assert _query(client, data_id="a").status_code == 200
    # The directory appends nothing.
    assert _audit_count() == after_rewrap


# --- server failure ---------------------------------------------------------


def test_storage_failure_returns_500_and_no_half_page(app, client):
    _seed(client, ["a", "b", "c"])

    def fail_scan(conn, cursor, statement, parameters, context, executemany):
        if "FROM data_envelopes" in statement:
            raise RuntimeError("simulated storage failure")

    sa_event.listen(app.state.engine, "before_cursor_execute", fail_scan)
    try:
        response = _query(client)
        assert response.status_code == 500
        assert b'"envelopes"' not in response.content
    finally:
        sa_event.remove(app.state.engine, "before_cursor_execute", fail_scan)

    recovered = _query(client)
    assert recovered.status_code == 200
    assert len(recovered.json()["envelopes"]) == 3


# --- legacy migration -------------------------------------------------------


def test_legacy_sqlite_database_backfills_sequences_and_counters(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    from sqlalchemy import create_engine, text

    from proof_release.db import Base

    url = f"sqlite:///{tmp_path}/legacy-envelopes.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # Build a pre-sequence database: drop the new column, index and
        # counter table, then write legacy rows out of data_id order.
        conn.execute(
            text("DROP INDEX IF EXISTS ix_data_envelopes_scope_commit_seq")
        )
        conn.execute(text("ALTER TABLE data_envelopes DROP COLUMN commit_seq"))
        conn.execute(text("DROP TABLE data_envelope_commit_counters"))
        conn.execute(
            text(
                "INSERT INTO data_envelopes "
                "(tenant_id, workload_id, data_id, key_version, ciphertext, "
                "iv, tag, wrapped_key, created_at) VALUES "
                "(:t, :w, 'zeta', 1, x'01', x'02', x'03', x'04', "
                "'2026-01-03T00:00:00+00:00'),"
                "(:t, :w, 'alpha', 1, x'01', x'02', x'03', x'04', "
                "'2026-01-01T00:00:00+00:00'),"
                "(:ot, :w, 'only-other-tenant', 1, x'01', x'02', x'03', x'04', "
                "'2026-01-02T00:00:00+00:00')"
            ),
            {"t": TENANT, "w": WORKLOAD, "ot": OTHER_TENANT},
        )
    engine.dispose()

    application = app_module.create_app(url)
    client = TestClient(application)
    ids = [e["data_id"] for e in _walk(client)]
    # Backfill numbers by the directory key: alpha=1, zeta=2 in the scope.
    assert ids == ["alpha", "zeta"]

    with application.state.session_factory() as session:
        counter = session.get(DataEnvelopeCommitCounter, (TENANT, WORKLOAD))
        assert counter.last_seq == 2
        other = session.get(
            DataEnvelopeCommitCounter, (OTHER_TENANT, WORKLOAD)
        )
        assert other.last_seq == 1
        by_id = {
            row.data_id: row.commit_seq
            for row in session.query(DataEnvelope)
            .filter(
                DataEnvelope.tenant_id == TENANT,
                DataEnvelope.workload_id == WORKLOAD,
            )
            .all()
        }
        assert by_id == {"alpha": 1, "zeta": 2}

    # The next creation allocates past the seeded maximum without colliding.
    response = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": "new-after-upgrade",
            "payload": PAYLOAD,
        },
    )
    assert response.status_code == 201
    with application.state.session_factory() as session:
        row = session.get(
            DataEnvelope, (TENANT, WORKLOAD, "new-after-upgrade")
        )
        assert row.commit_seq == 3
    application.state.engine.dispose()
