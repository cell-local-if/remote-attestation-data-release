"""Tests for scoped, auditable, recoverable batch envelope rewrapping:

POST /v1/rewrap-batches and GET /v1/rewrap-batches/{batch_id}.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import DataEnvelope, RewrapBatch, RewrapBatchItem
from proof_release.envelopes import (
    MasterKeyError,
    b64url_decode,
    b64url_encode,
)

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap

TENANT = "tenant-a"
WORKLOAD = "workload-1"
PAYLOAD = "batch-secret-payload 🔐"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V2_BYTES = b"fedcba9876543210fedcba9876543210"
KEY_V1 = b64url_encode(KEY_V1_BYTES)
KEY_V2 = b64url_encode(KEY_V2_BYTES)


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


KEYRING_V1_V2 = _keyring(2, {1: KEY_V1, 2: KEY_V2})
KEYRING_V2_ONLY = _keyring(2, {2: KEY_V2})


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = app_module.create_app(f"sqlite:///{tmp_path}/batches.db")
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


def _start_batch(client, *, cursor=None, limit=None, tenant=TENANT, workload=WORKLOAD):
    body: dict = {"tenant_id": tenant, "workload_id": workload}
    if cursor is not None:
        body["cursor"] = cursor
    if limit is not None:
        body["limit"] = limit
    return client.post("/v1/rewrap-batches", json=body)


def _get_envelope(client, data_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/data-envelopes/{data_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _get_batch(client, batch_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/rewrap-batches/{batch_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _unwrap_and_decrypt(master_key: bytes, fields: dict) -> bytes:
    data_key = aes_key_unwrap(master_key, b64url_decode(fields["wrapped_key"]))
    iv = b64url_decode(fields["iv"])
    sealed = b64url_decode(fields["ciphertext"]) + b64url_decode(fields["tag"])
    return AESGCM(data_key).decrypt(iv, sealed, None)


# --- request validation ----------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"workload_id": WORKLOAD},
        {"tenant_id": TENANT},
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "   ", "workload_id": WORKLOAD},
        {"tenant_id": 7, "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": "\t"},
        {"tenant_id": TENANT, "workload_id": 3},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": 0},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": 201},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": -1},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": "10"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": 1.5},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": True},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": None},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "cursor": 9},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "cursor": "   "},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "cursor": "bad token!!"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "cursor": "AAA="},
        {},
    ],
)
def test_create_batch_rejects_invalid_requests(client, body):
    response = client.post("/v1/rewrap-batches", json=body)
    assert response.status_code == 422, response.text


def test_create_batch_rejects_non_json_body(client):
    response = client.post(
        "/v1/rewrap-batches",
        content="not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_forged_or_cross_scope_cursor_returns_422(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _start_batch(client, limit=1).json()
    good_cursor = first["next_cursor"]
    assert good_cursor

    # Tampered token.
    tampered = good_cursor[:-2] + ("AA" if good_cursor[-2:] != "AA" else "BB")
    assert _start_batch(client, cursor=tampered).status_code == 422
    # Token minted for another tenant/workload.
    assert (
        _start_batch(client, cursor=good_cursor, tenant="tenant-b").status_code == 422
    )
    assert (
        _start_batch(client, cursor=good_cursor, workload="workload-2").status_code
        == 422
    )
    # Opaque garbage that still parses as base64url.
    forged = b64url_encode(b'{"t":"tenant-a","w":"workload-1","d":"x"}' + b"0" * 32)
    assert _start_batch(client, cursor=forged).status_code == 422


# --- happy path, pagination, counting --------------------------------------


def test_empty_scope_completes_immediately(client):
    response = _start_batch(client)
    assert response.status_code == 200
    data = response.json()
    assert data["processed"] == 0
    assert data["rewrapped"] == 0
    assert data["skipped"] == 0
    assert data["failed"] == 0
    assert data["next_cursor"] == ""
    assert data["complete"] is True
    summary = _get_batch(client, data["batch_id"]).json()
    assert summary["audits"] == []
    assert summary["complete"] is True


def test_batch_rewraps_in_data_id_order_and_only_changes_wrapping(
    app, client, monkeypatch
):
    data_ids = ["zeta", "alpha", "mid", "001", "beta"]
    _seed(client, data_ids)
    before = {d: _get_envelope(client, d).json() for d in data_ids}
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    response = _start_batch(client)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "batch_id",
        "processed",
        "rewrapped",
        "skipped",
        "failed",
        "next_cursor",
        "complete",
    }
    assert data["processed"] == 5
    assert data["rewrapped"] == 5
    assert data["skipped"] == 0
    assert data["failed"] == 0
    assert data["next_cursor"] == ""
    assert data["complete"] is True

    ordered = sorted(data_ids)
    summary = _get_batch(client, data["batch_id"]).json()
    assert [item["data_id"] for item in summary["audits"]] == ordered
    for item, d in zip(summary["audits"], ordered):
        assert item["result"] == "rewrapped"
        assert item["old_key_version"] == 1
        assert item["new_key_version"] == 2
        stamped = datetime.fromisoformat(item["audited_at"])
        assert stamped.utcoffset() == timedelta(0)
    for d in data_ids:
        after = _get_envelope(client, d).json()
        assert after["key_version"] == 2
        assert after["wrapped_key"] != before[d]["wrapped_key"]
        # Nothing but the wrapping key changes.
        for field in ("ciphertext", "iv", "tag", "created_at", "data_id"):
            assert after[field] == before[d][field]
        assert _unwrap_and_decrypt(KEY_V2_BYTES, after) == PAYLOAD.encode("utf-8")


def test_batch_default_limit_is_50(app, client, monkeypatch):
    _seed(client, [f"item-{i:03d}" for i in range(51)])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    data = _start_batch(client).json()
    assert data["processed"] == 50
    assert data["complete"] is False
    assert data["next_cursor"] != ""


@pytest.mark.parametrize("limit", [1, 2, 200])
def test_batch_limit_bounds_accepted(app, client, monkeypatch, limit):
    _seed(client, [f"x{i}" for i in range(3)])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _start_batch(client, limit=limit)
    assert response.status_code == 200
    assert response.json()["processed"] == min(limit, 3)


def test_pagination_walks_the_whole_scope(app, client, monkeypatch):
    data_ids = [f"id-{i:02d}" for i in range(7)]
    _seed(client, data_ids)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    pages, seen = [], []
    cursor = None
    for _ in range(10):
        page = _start_batch(client, cursor=cursor, limit=3).json()
        pages.append(page)
        summary = _get_batch(client, page["batch_id"]).json()
        seen.extend(item["data_id"] for item in summary["audits"])
        if page["complete"]:
            break
        cursor = page["next_cursor"]
    else:  # pragma: no cover - sentinel for a cursor loop that never ends
        raise AssertionError("pagination did not complete")

    assert seen == sorted(data_ids)
    assert [p["processed"] for p in pages] == [3, 3, 1]
    assert pages[0]["complete"] is False and pages[1]["complete"] is False
    assert pages[-1]["complete"] is True
    assert pages[-1]["next_cursor"] == ""
    # Each page has its own batch and independent audit rows.
    batch_ids = {p["batch_id"] for p in pages}
    assert len(batch_ids) == 3
    with app.state.session_factory() as session:
        assert session.query(RewrapBatch).count() == 3
        assert session.query(RewrapBatchItem).count() == 7


def test_batch_is_scoped_to_tenant_and_workload(client, monkeypatch):
    _seed(client, ["a1", "a2"], tenant=TENANT, workload=WORKLOAD)
    _seed(client, ["b1"], tenant="tenant-b", workload=WORKLOAD)
    _seed(client, ["c1"], tenant=TENANT, workload="workload-9")
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    data = _start_batch(client).json()
    assert data["processed"] == 2
    other_tenant = _get_envelope(client, "b1", tenant="tenant-b").json()
    other_workload = _get_envelope(
        client, "c1", tenant=TENANT, workload="workload-9"
    ).json()
    assert other_tenant["key_version"] == 1
    assert other_workload["key_version"] == 1


def test_current_version_envelopes_are_skipped(app, client, monkeypatch):
    _seed(client, ["old-1", "new-1", "old-2", "new-2"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    # Rotate two envelopes to the current version before the batch.
    for data_id in ("new-1", "new-2"):
        assert (
            client.post(
                f"/v1/data-envelopes/{data_id}/rewrap",
                json={"tenant_id": TENANT, "workload_id": WORKLOAD},
            ).status_code
            == 200
        )
    current_before = {
        d: _get_envelope(client, d).json() for d in ("new-1", "new-2")
    }

    data = _start_batch(client).json()
    assert data["processed"] == 4
    assert data["rewrapped"] == 2
    assert data["skipped"] == 2
    assert data["failed"] == 0

    summary = _get_batch(client, data["batch_id"]).json()
    results = {item["data_id"]: item["result"] for item in summary["audits"]}
    assert results == {
        "new-1": "skipped",
        "new-2": "skipped",
        "old-1": "rewrapped",
        "old-2": "rewrapped",
    }
    for item in summary["audits"]:
        if item["result"] == "skipped":
            assert item["old_key_version"] == item["new_key_version"] == 2
    # Skipped envelopes are byte-identical afterwards.
    for d in ("new-1", "new-2"):
        assert _get_envelope(client, d).json() == current_before[d]


def test_retrying_next_cursor_never_rewraps_twice(app, client, monkeypatch):
    _seed(client, [f"i{n}" for n in range(6)])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _start_batch(client, limit=6).json()
    assert first["rewrapped"] == 6
    after_first = {f"i{n}": _get_envelope(client, f"i{n}").json() for n in range(6)}

    # Replaying from the beginning only observes current-version results.
    retry = _start_batch(client, limit=6).json()
    assert retry["rewrapped"] == 0
    assert retry["skipped"] == 6
    assert retry["failed"] == 0
    assert retry["complete"] is True
    for n in range(6):
        assert _get_envelope(client, f"i{n}").json() == after_first[f"i{n}"]


# --- failure handling ------------------------------------------------------


def test_invalid_keyring_returns_500_creates_nothing_and_changes_nothing(
    app, client, monkeypatch
):
    _seed(client, ["a", "b"])
    before = {d: _get_envelope(client, d).json() for d in ("a", "b")}
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")

    response = _start_batch(client)
    assert response.status_code == 500
    with app.state.session_factory() as session:
        assert session.query(RewrapBatch).count() == 0
        assert session.query(RewrapBatchItem).count() == 0
        assert session.query(DataEnvelope).count() == 2
    for d in ("a", "b"):
        assert _get_envelope(client, d).json() == before[d]


def test_missing_historical_key_stops_page_and_keeps_failed_envelope(
    app, client, monkeypatch
):
    data_ids = ["a0", "a1", "a2", "a3"]
    _seed(client, data_ids)
    before = {d: _get_envelope(client, d).json() for d in data_ids}
    # Keyring without the historical version 1: the first envelope fails.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)

    data = _start_batch(client, limit=10).json()
    assert data["processed"] == 1
    assert data["rewrapped"] == 0
    assert data["skipped"] == 0
    assert data["failed"] == 1
    assert data["complete"] is False
    # Failure on the first item leaves the cursor at the scope beginning.
    assert data["next_cursor"] == ""

    summary = _get_batch(client, data["batch_id"]).json()
    assert len(summary["audits"]) == 1
    audit = summary["audits"][0]
    assert audit["data_id"] == "a0"
    assert audit["result"] == "missing-key"
    assert audit["old_key_version"] == audit["new_key_version"] == 1
    # Nothing after the failed item was visited or modified.
    for d in data_ids:
        assert _get_envelope(client, d).json() == before[d]

    # Restore the historical key and resume: the stopped page advances.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    resumed = _start_batch(client, limit=10).json()
    assert resumed["failed"] == 0
    assert resumed["rewrapped"] == 4
    assert resumed["complete"] is True


def test_rewrap_failure_mid_page_commits_prior_audits_and_stops(
    app, client, monkeypatch
):
    data_ids = ["a0", "a1", "a2", "a3"]
    _seed(client, data_ids)
    before = {d: _get_envelope(client, d).json() for d in data_ids}
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    real_rewrap = app_module.rewrap_data_key
    calls = {"n": 0}

    def failing_rewrap(unwrapping_key, wrapping_key, wrapped_key):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("crypto backend exploded")
        return real_rewrap(unwrapping_key, wrapping_key, wrapped_key)

    monkeypatch.setattr(app_module, "rewrap_data_key", failing_rewrap)
    data = _start_batch(client, limit=10).json()
    monkeypatch.setattr(app_module, "rewrap_data_key", real_rewrap)

    assert data["processed"] == 3
    assert data["rewrapped"] == 2
    assert data["skipped"] == 0
    assert data["failed"] == 1
    assert data["complete"] is False

    summary = _get_batch(client, data["batch_id"]).json()
    results = [(item["data_id"], item["result"]) for item in summary["audits"]]
    assert results == [("a0", "rewrapped"), ("a1", "rewrapped"), ("a2", "rewrap")]
    failed_audit = summary["audits"][-1]
    assert failed_audit["old_key_version"] == failed_audit["new_key_version"] == 1

    # The failed envelope and everything after it are untouched.
    for d in ("a2", "a3"):
        assert _get_envelope(client, d).json() == before[d]

    # Resume from the returned cursor: a2 now rewraps, a3 is then reached.
    resumed = _start_batch(client, cursor=data["next_cursor"], limit=10).json()
    assert resumed["failed"] == 0
    assert resumed["rewrapped"] == 2
    assert resumed["complete"] is True
    assert _get_envelope(client, "a2").json()["key_version"] == 2


def test_keyring_becoming_invalid_mid_page_stops_with_keyring_result(
    app, client, monkeypatch
):
    _seed(client, ["a0", "a1", "a2"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    real_load = app_module.load_keyring
    calls = {"n": 0}

    def failing_load():
        calls["n"] += 1
        # Call 1 is the up-front check; envelope 1 uses call 2; envelope 2
        # (call 3) observes the keyring going bad mid-page.
        if calls["n"] == 3:
            raise MasterKeyError("keyring vanished")
        return real_load()

    monkeypatch.setattr(app_module, "load_keyring", failing_load)
    data = _start_batch(client, limit=10).json()

    assert data["processed"] == 2
    assert data["rewrapped"] == 1
    assert data["failed"] == 1
    assert data["complete"] is False
    summary = _get_batch(client, data["batch_id"]).json()
    assert [item["result"] for item in summary["audits"]] == [
        "rewrapped",
        "keyring",
    ]
    # The stopped envelope is unchanged.
    assert _get_envelope(client, "a1").json()["key_version"] == 1
    assert _get_envelope(client, "a2").json()["key_version"] == 1


# --- concurrency -----------------------------------------------------------


def test_concurrent_batches_advance_each_envelope_at_most_once(
    app, client, monkeypatch
):
    count = 8
    _seed(client, [f"e{n}" for n in range(count)])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(
            pool.map(
                lambda _: _start_batch(client, limit=50),
                range(8),
            )
        )
    assert all(response.status_code == 200 for response in responses)
    totals = {
        key: sum(response.json()[key] for response in responses)
        for key in ("rewrapped", "skipped", "failed")
    }
    assert totals["rewrapped"] == count
    assert totals["failed"] == 0
    # Every envelope processed after the winning batch observes the
    # already-current result: 8 batches x 8 envelopes - 8 wins.
    assert totals["skipped"] == count * 8 - count

    for n in range(count):
        stored = _get_envelope(client, f"e{n}").json()
        assert stored["key_version"] == 2
        assert _unwrap_and_decrypt(KEY_V2_BYTES, stored) == PAYLOAD.encode("utf-8")

    # Every losing batch records its envelope as already current
    # (old == new == current version), never as a rewrap.
    losing_audits = []
    for response in responses:
        summary = _get_batch(client, response.json()["batch_id"]).json()
        for item in summary["audits"]:
            if item["result"] == "skipped":
                losing_audits.append(item)
                assert item["old_key_version"] == item["new_key_version"] == 2
    assert len(losing_audits) == count * 8 - count


# --- batch lookup ----------------------------------------------------------


def test_get_unknown_or_cross_scope_batch_returns_404(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    batch_id = _start_batch(client).json()["batch_id"]

    unknown = "00000000-0000-0000-0000-000000000000"
    assert _get_batch(client, unknown).status_code == 404
    # Syntactically valid uppercase UUID is normalized, then not found.
    assert (
        _get_batch(client, "11111111-1111-1111-1111-111111111111").status_code
        == 404
    )
    # The actual id works even if the client uppercases it.
    assert _get_batch(client, batch_id.upper()).status_code == 200
    assert _get_batch(client, batch_id, tenant="tenant-b").status_code == 404
    assert (
        _get_batch(client, batch_id, workload="workload-2").status_code == 404
    )


@pytest.mark.parametrize(
    "batch_id",
    [
        "not-a-uuid",
        "abc123",
        "00000000-0000-0000-0000-00000000000Z",
        "  00000000-0000-0000-0000-000000000000",
    ],
)
def test_get_batch_rejects_malformed_identifier(client, batch_id):
    response = client.get(
        f"/v1/rewrap-batches/{batch_id}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_get_batch_rejects_blank_or_missing_scope_query(client):
    batch_id = "00000000-0000-0000-0000-000000000000"
    assert (
        client.get(
            f"/v1/rewrap-batches/{batch_id}",
            params={"tenant_id": "  ", "workload_id": WORKLOAD},
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/v1/rewrap-batches/{batch_id}",
            params={"tenant_id": TENANT, "workload_id": ""},
        ).status_code
        == 422
    )
    assert (
        client.get(f"/v1/rewrap-batches/{batch_id}").status_code == 422
    )


def test_get_batch_without_identifier_returns_422(client):
    response = client.get(
        "/v1/rewrap-batches/",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_get_batch_failure_returns_500(app, client, monkeypatch):
    _seed(client, ["a"])
    batch_id = _start_batch(client).json()["batch_id"]

    # Simulate a storage/service failure on the audit read: the batch row
    # is still reachable but loading its items raises.
    from sqlalchemy import text

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_batch_items"))
    assert _get_batch(client, batch_id).status_code == 500


# --- persistence, wire format ----------------------------------------------


def test_batches_and_audits_survive_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, [f"r{n}" for n in range(4)])
    app1.state.engine.dispose()

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    page = _start_batch(client2, limit=2).json()
    assert page["rewrapped"] == 2
    batch_id, cursor = page["batch_id"], page["next_cursor"]
    assert page["complete"] is False and cursor
    app2.state.engine.dispose()

    app3 = app_module.create_app(url)
    client3 = TestClient(app3)
    # The finished page and its audits are queryable after restart.
    summary = _get_batch(client3, batch_id).json()
    assert summary["processed"] == 2
    assert [item["data_id"] for item in summary["audits"]] == ["r0", "r1"]
    # The cursor remains valid after restart: it points past r1, so the
    # next page rewraps the untouched tail r2/r3.
    tail = _start_batch(client3, cursor=cursor, limit=10).json()
    assert tail["rewrapped"] == 2
    assert tail["skipped"] == 0
    assert tail["complete"] is True
    # Replaying any cursor now only observes already-current results.
    replay = _start_batch(client3, limit=10).json()
    assert replay["rewrapped"] == 0
    assert replay["skipped"] == 4
    assert replay["complete"] is True
    app3.state.engine.dispose()


def test_batch_responses_are_compact_json_without_trailing_newline_or_floats(
    client, monkeypatch
):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    response = _start_batch(client)
    raw = response.content
    assert raw[-1:] != b"\n"
    assert b", " not in raw
    assert b": " not in raw
    assert response.headers["content-type"] == "application/json"
    parsed = json.loads(raw)
    assert isinstance(parsed["batch_id"], str)
    assert isinstance(parsed["next_cursor"], str)
    assert isinstance(parsed["complete"], bool)
    for key in ("processed", "rewrapped", "skipped", "failed"):
        assert isinstance(parsed[key], int) and not isinstance(parsed[key], bool)

    summary_response = _get_batch(client, parsed["batch_id"])
    summary_raw = summary_response.content
    assert summary_raw[-1:] != b"\n"
    assert b", " not in summary_raw
    summary = json.loads(summary_raw)
    assert isinstance(summary["complete"], bool)
    assert isinstance(summary["limit"], int)
    for item in summary["audits"]:
        assert isinstance(item["old_key_version"], int)
        assert isinstance(item["new_key_version"], int)
        assert isinstance(item["data_id"], str)
        assert set(item) == {
            "data_id",
            "old_key_version",
            "new_key_version",
            "result",
            "audited_at",
        }


def test_batch_never_exposes_payload_or_key_material(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _start_batch(client)
    for secret in (PAYLOAD, KEY_V1, KEY_V2, "wrapped_key", "ciphertext", "payload"):
        assert secret not in response.text
    summary_response = _get_batch(client, response.json()["batch_id"])
    for secret in (PAYLOAD, KEY_V1, KEY_V2, "wrapped_key", "ciphertext", "payload"):
        assert secret not in summary_response.text
    with app.state.session_factory() as session:
        item = session.query(RewrapBatchItem).one()
        assert not hasattr(item, "wrapped_key")
