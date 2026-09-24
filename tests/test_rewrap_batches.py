"""Tests for scoped, auditable, resumable batch envelope rewrapping:

POST /v1/rewrap-batches and GET /v1/rewrap-batches/{batch_id}.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import DataEnvelope, RewrapBatch, RewrapBatchAudit
from proof_release.envelopes import (
    MasterKeyError,
    b64url_decode,
    b64url_encode,
    load_keyring,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap

TENANT = "tenant-a"
WORKLOAD = "workload-1"
PAYLOAD = "super-secret-batch-payload 🔐"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V2_BYTES = b"fedcba9876543210fedcba9876543210"
KEY_V1 = b64url_encode(KEY_V1_BYTES)
KEY_V2 = b64url_encode(KEY_V2_BYTES)

KEYRING_V1_V2 = json.dumps(
    {"current_version": 2, "keys": {"1": KEY_V1, "2": KEY_V2}}
)
KEYRING_V2_ONLY = json.dumps({"current_version": 2, "keys": {"2": KEY_V2}})


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


def _create_many(client, ids, **kwargs):
    for data_id in ids:
        response = _create(client, data_id, **kwargs)
        assert response.status_code == 201, response.text


def _start_batch(client, *, cursor=None, limit=None, tenant=TENANT, workload=WORKLOAD):
    body: dict = {"tenant_id": tenant, "workload_id": workload}
    if cursor is not None:
        body["cursor"] = cursor
    if limit is not None:
        body["limit"] = limit
    return client.post("/v1/rewrap-batches", json=body)


def _get_batch(client, batch_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/rewrap-batches/{batch_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _get_envelope(client, data_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/data-envelopes/{data_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _unwrap_and_decrypt(master_key: bytes, fields: dict) -> bytes:
    data_key = aes_key_unwrap(master_key, b64url_decode(fields["wrapped_key"]))
    iv = b64url_decode(fields["iv"])
    sealed = b64url_decode(fields["ciphertext"]) + b64url_decode(fields["tag"])
    return AESGCM(data_key).decrypt(iv, sealed, None)


# --- response shape --------------------------------------------------------


def test_batch_rewraps_a_page_and_reports_exact_shape(app, client, monkeypatch):
    _create_many(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    response = _start_batch(client)
    assert response.status_code == 200, response.text
    data = response.json()
    assert set(data) == {
        "batch_id",
        "processed",
        "rewrapped",
        "skipped",
        "failed",
        "next_cursor",
        "done",
    }
    assert data["processed"] == 2
    assert data["rewrapped"] == 2
    assert data["skipped"] == 0
    assert data["failed"] == 0
    assert data["done"] is True
    assert data["next_cursor"] == ""
    # Counts are JSON integers, done is a boolean; nothing is a float.
    for name in ("processed", "rewrapped", "skipped", "failed"):
        assert isinstance(data[name], int) and not isinstance(data[name], bool)
    assert isinstance(data["done"], bool)
    assert isinstance(data["batch_id"], str) and isinstance(data["next_cursor"], str)
    # Compact JSON with no trailing newline and no non-finite number spellings.
    assert response.content == json.dumps(
        data, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert not response.content.endswith(b"\n")
    assert b"-0.0" not in response.content
    # No payload or key material on the wire.
    assert PAYLOAD not in response.text
    assert KEY_V1 not in response.text and KEY_V2 not in response.text


def test_batch_audit_rows_record_only_scope_ids_versions_code_and_time(
    app, client, monkeypatch
):
    _create_many(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    batch_id = _start_batch(client).json()["batch_id"]
    detail = _get_batch(client, batch_id)
    assert detail.status_code == 200
    data = detail.json()
    assert data["batch_id"] == batch_id
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["limit"] == 50
    assert data["processed"] == 2
    assert data["rewrapped"] == 2
    assert data["skipped"] == 0
    assert data["failed"] == 0
    assert data["done"] is True
    assert data["next_cursor"] == ""
    assert data["completed_at"] is not None
    assert datetime.fromisoformat(data["created_at"]).utcoffset() == timedelta(0)
    assert datetime.fromisoformat(data["completed_at"]).utcoffset() == timedelta(0)

    assert [audit["data_id"] for audit in data["audits"]] == ["a", "b"]
    for audit in data["audits"]:
        assert set(audit) == {
            "tenant_id",
            "workload_id",
            "data_id",
            "old_key_version",
            "new_key_version",
            "result",
            "occurred_at",
        }
        assert audit["tenant_id"] == TENANT
        assert audit["workload_id"] == WORKLOAD
        assert audit["result"] == "rewrapped"
        assert audit["old_key_version"] == 1
        assert audit["new_key_version"] == 2
        assert audit["occurred_at"].endswith("+00:00")
        assert datetime.fromisoformat(audit["occurred_at"]).utcoffset() == timedelta(0)
    # GET responses are compact JSON too.
    assert detail.content == json.dumps(
        data, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert not detail.content.endswith(b"\n")


# --- ordering, pagination, resumability ------------------------------------


def test_batch_pages_in_stable_data_id_order_with_exclusive_cursor(
    app, client, monkeypatch
):
    # Inserted out of order; selection must follow data_id, not insertion.
    _create_many(client, ["z", "a", "m"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    first = _start_batch(client, limit=2)
    assert first.status_code == 200
    page1 = first.json()
    assert page1["processed"] == 2
    assert page1["rewrapped"] == 2
    assert page1["done"] is False
    assert page1["next_cursor"]
    # The opaque token is random and bears no structural encoding of the
    # boundary position (ids themselves are never sent in cursors).
    assert len(page1["next_cursor"]) == 32

    audits1 = _get_batch(client, page1["batch_id"]).json()["audits"]
    assert [a["data_id"] for a in audits1] == ["a", "m"]

    second = _start_batch(client, cursor=page1["next_cursor"], limit=2)
    page2 = second.json()
    assert second.status_code == 200
    assert page2["processed"] == 1
    assert page2["rewrapped"] == 1
    assert page2["done"] is True
    assert page2["next_cursor"] == ""
    audits2 = _get_batch(client, page2["batch_id"]).json()["audits"]
    assert [a["data_id"] for a in audits2] == ["z"]


def test_default_limit_is_fifty(app, client, monkeypatch):
    _create_many(client, [f"item-{i:03d}" for i in range(51)])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    first = _start_batch(client)
    page1 = first.json()
    assert page1["processed"] == 50
    assert page1["rewrapped"] == 50
    assert page1["done"] is False

    second = _start_batch(client, cursor=page1["next_cursor"])
    page2 = second.json()
    assert page2["processed"] == 1
    assert page2["rewrapped"] == 1
    assert page2["done"] is True
    assert page2["next_cursor"] == ""


def test_empty_range_completes_immediately(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _start_batch(client)
    assert response.status_code == 200
    data = response.json()
    assert data["processed"] == 0
    assert data["rewrapped"] == 0
    assert data["skipped"] == 0
    assert data["failed"] == 0
    assert data["done"] is True
    assert data["next_cursor"] == ""
    detail = _get_batch(client, data["batch_id"]).json()
    assert detail["audits"] == []


def test_current_version_envelopes_are_skipped_not_rewrapped(app, client, monkeypatch):
    _create_many(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _start_batch(client).json()["rewrapped"] == 2

    # A second batch over the same range changes nothing and counts skips.
    before = _get_envelope(client, "a").json()
    again = _start_batch(client)
    assert again.status_code == 200
    data = again.json()
    assert data["processed"] == 2
    assert data["rewrapped"] == 0
    assert data["skipped"] == 2
    assert data["failed"] == 0
    assert data["done"] is True
    assert _get_envelope(client, "a").json() == before
    audits = _get_batch(client, data["batch_id"]).json()["audits"]
    assert [a["result"] for a in audits] == ["skipped", "skipped"]
    for audit in audits:
        assert audit["old_key_version"] == audit["new_key_version"] == 2


def test_replaying_a_cursor_does_not_advance_handled_envelopes_again(
    app, client, monkeypatch
):
    _create_many(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    page1 = _start_batch(client, limit=2).json()
    # Replay the *start* of the same page: both envelopes are current now.
    replay = _start_batch(client, limit=2)
    assert replay.json()["rewrapped"] == 0
    assert replay.json()["skipped"] == 2
    assert replay.json()["done"] is False
    # Continuing with the issued cursor lands on the one remaining envelope.
    cont = _start_batch(client, cursor=page1["next_cursor"])
    assert cont.json() == {
        "batch_id": cont.json()["batch_id"],
        "processed": 1,
        "rewrapped": 1,
        "skipped": 0,
        "failed": 0,
        "next_cursor": "",
        "done": True,
    }


def test_batch_only_changes_the_wrapped_key_and_preserves_payload(
    app, client, monkeypatch
):
    _create(client, "a")
    before = _get_envelope(client, "a").json()
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    assert _start_batch(client).status_code == 200
    after = _get_envelope(client, "a").json()
    assert after["key_version"] == 2
    assert after["wrapped_key"] != before["wrapped_key"]
    for field in ("ciphertext", "iv", "tag", "created_at", "data_id"):
        assert after[field] == before[field]
    assert _unwrap_and_decrypt(KEY_V2_BYTES, after) == PAYLOAD.encode("utf-8")


def test_batch_is_scoped_to_tenant_and_workload(app, client, monkeypatch):
    _create_many(client, ["a", "b"], tenant=TENANT, workload=WORKLOAD)
    _create_many(client, ["a"], tenant="tenant-b", workload=WORKLOAD)
    _create_many(client, ["a"], tenant=TENANT, workload="workload-2")
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    data = _start_batch(client).json()
    assert data["processed"] == 2
    assert _get_envelope(client, "a", tenant="tenant-b").json()["key_version"] == 1
    assert (
        _get_envelope(client, "a", tenant=TENANT, workload="workload-2").json()[
            "key_version"
        ]
        == 1
    )


# --- request validation ----------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"tenant_id": True},
        {"workload_id": ""},
        {"workload_id": "  "},
        {"workload_id": 9},
        {"limit": 0},
        {"limit": 201},
        {"limit": -1},
        {"limit": True},
        {"limit": 1.0},
        {"limit": 1.5},
        {"limit": "50"},
        {"limit": None},
        {"cursor": ""},
        {"cursor": "   "},
        {"cursor": 7},
        {"cursor": True},
    ],
)
def test_start_batch_rejects_invalid_fields(client, overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    body.update(overrides)
    response = client.post("/v1/rewrap-batches", json=body)
    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id"])
def test_start_batch_requires_scope_fields(client, missing):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    del body[missing]
    assert client.post("/v1/rewrap-batches", json=body).status_code == 422


def test_forged_cursor_is_rejected_with_422(app, client, monkeypatch):
    _create_many(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    # Well-formed opaque shape, but no such cursor exists.
    forged = b64url_encode(b"x" * 24)
    response = _start_batch(client, cursor=forged)
    assert response.status_code == 422


def test_cursor_from_another_scope_is_rejected_with_422(app, client, monkeypatch):
    _create_many(client, ["a", "b"], tenant=TENANT, workload=WORKLOAD)
    _create_many(client, ["a", "b"], tenant="tenant-b", workload=WORKLOAD)
    _create_many(client, ["a", "b"], tenant=TENANT, workload="workload-2")
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    token = _start_batch(client, limit=1).json()["next_cursor"]
    # Different tenant and different workload are both cross-scope uses.
    assert (
        _start_batch(
            client, cursor=token, tenant="tenant-b", workload=WORKLOAD
        ).status_code
        == 422
    )
    assert (
        _start_batch(
            client, cursor=token, tenant=TENANT, workload="workload-2"
        ).status_code
        == 422
    )
    # The legitimate scope still accepts it.
    assert _start_batch(client, cursor=token, limit=10).status_code == 200


# --- keyring failures ------------------------------------------------------


def test_invalid_keyring_preflight_returns_500_with_no_batch_or_changes(
    app, client, monkeypatch
):
    _create_many(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not valid json")

    response = _start_batch(client)
    assert response.status_code == 500
    with app.state.session_factory() as session:
        assert session.query(RewrapBatch).count() == 0
        assert session.query(RewrapBatchAudit).count() == 0
        assert session.query(DataEnvelope).filter_by(key_version=1).count() == 2
    assert _get_envelope(client, "a").json()["key_version"] == 1


def test_missing_historical_key_stops_page_with_200_and_keeps_envelope(
    app, client, monkeypatch
):
    _create_many(client, ["a", "b", "c"])
    before = _get_envelope(client, "a").json()
    # The current keyring dropped version 1 entirely.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)

    response = _start_batch(client, limit=3)
    assert response.status_code == 200
    data = response.json()
    assert data["processed"] == 0
    assert data["rewrapped"] == 0
    assert data["skipped"] == 0
    assert data["failed"] == 1
    assert data["done"] is False
    assert data["next_cursor"]

    audits = _get_batch(client, data["batch_id"]).json()["audits"]
    assert audits == [
        {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": "a",
            "old_key_version": 1,
            "new_key_version": 2,
            "result": "missing-key",
            "occurred_at": audits[0]["occurred_at"],
        }
    ]
    # The failing envelope and everything after it are untouched.
    assert _get_envelope(client, "a").json() == before
    assert _get_envelope(client, "b").json()["key_version"] == 1
    assert _get_envelope(client, "c").json()["key_version"] == 1


def test_keyring_failure_mid_page_is_audited_and_resumable(app, client, monkeypatch):
    _create_many(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    calls = {"n": 0}

    def flaky_keyring():
        calls["n"] += 1
        # preflight + first envelope succeed; the keyring vanishes after.
        if calls["n"] <= 2:
            return load_keyring()
        raise MasterKeyError("keyring removed mid page")

    monkeypatch.setattr(app_module, "load_keyring", flaky_keyring)
    response = _start_batch(client, limit=3)
    assert response.status_code == 200
    data = response.json()
    assert data["processed"] == 1
    assert data["rewrapped"] == 1
    assert data["failed"] == 1
    assert data["done"] is False
    audits = _get_batch(client, data["batch_id"]).json()["audits"]
    assert [(a["data_id"], a["result"]) for a in audits] == [
        ("a", "rewrapped"),
        ("b", "keyring"),
    ]
    assert audits[1]["old_key_version"] == 1
    assert audits[1]["new_key_version"] == 2
    for audit in audits:
        assert isinstance(audit["new_key_version"], int)

    # After restoring the keyring, the cursor resumes at the failed envelope.
    # Restore only the function (the KEYRING_V1_V2 env var stays in effect).
    monkeypatch.setattr(app_module, "load_keyring", load_keyring)
    resume = _start_batch(client, cursor=data["next_cursor"], limit=3)
    page2 = resume.json()
    assert resume.status_code == 200
    assert page2["rewrapped"] == 2
    assert page2["failed"] == 0
    assert page2["done"] is True
    assert page2["next_cursor"] == ""
    audits2 = _get_batch(client, page2["batch_id"]).json()["audits"]
    assert [a["data_id"] for a in audits2] == ["b", "c"]
    assert _get_envelope(client, "a").json()["key_version"] == 2
    for data_id in ("b", "c"):
        stored = _get_envelope(client, data_id).json()
        assert stored["key_version"] == 2
        assert _unwrap_and_decrypt(KEY_V2_BYTES, stored) == PAYLOAD.encode("utf-8")


def test_rewrap_crypto_failure_stops_page_and_leaves_envelope(
    app, client, monkeypatch
):
    _create_many(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    before = _get_envelope(client, "a").json()

    def boom(*args, **kwargs):
        raise RuntimeError("crypto backend exploded")

    monkeypatch.setattr(app_module, "rewrap_data_key", boom)
    response = _start_batch(client, limit=2)
    assert response.status_code == 200
    data = response.json()
    assert data["processed"] == 0
    assert data["failed"] == 1
    assert data["done"] is False
    audits = _get_batch(client, data["batch_id"]).json()["audits"]
    assert audits[0]["result"] == "rewrap"
    assert audits[0]["old_key_version"] == 1
    assert audits[0]["new_key_version"] == 2
    assert _get_envelope(client, "a").json() == before
    assert _get_envelope(client, "b").json()["key_version"] == 1

    from proof_release.envelopes import rewrap_data_key

    monkeypatch.setattr(app_module, "rewrap_data_key", rewrap_data_key)
    resume = _start_batch(client, cursor=data["next_cursor"], limit=2)
    assert resume.json()["rewrapped"] == 2
    assert resume.json()["done"] is True


# --- concurrency -----------------------------------------------------------


def test_concurrent_batches_advance_each_envelope_at_most_once(
    app, client, monkeypatch
):
    _create_many(client, [f"x{i:02d}" for i in range(8)])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(
            pool.map(lambda _: _start_batch(client, limit=8), range(8))
        )
    assert all(response.status_code == 200 for response in responses)
    # Exactly one batch performs the eight advances; the others observe the
    # envelopes as already current.
    assert sum(response.json()["rewrapped"] for response in responses) == 8
    assert (
        sum(response.json()["processed"] for response in responses) == 8 * 8
    )
    for i in range(8):
        stored = _get_envelope(client, f"x{i:02d}").json()
        assert stored["key_version"] == 2
        assert _unwrap_and_decrypt(KEY_V2_BYTES, stored) == PAYLOAD.encode("utf-8")


def test_batch_losing_race_to_single_rewrap_records_skip(app, client, monkeypatch):
    _create_many(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    # Rotate "a" through the single-envelope endpoint first; a batch started
    # afterwards must record the already-current result instead of changing
    # material, and still rotate "b".
    single = client.post(
        "/v1/data-envelopes/a/rewrap",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert single.status_code == 200
    batch = _start_batch(client)
    data = batch.json()
    assert data["rewrapped"] == 1
    assert data["skipped"] == 1
    audits = _get_batch(client, data["batch_id"]).json()["audits"]
    assert [(a["data_id"], a["result"]) for a in audits] == [
        ("a", "skipped"),
        ("b", "rewrapped"),
    ]


# --- GET validation --------------------------------------------------------


def test_get_batch_unknown_or_cross_scope_returns_404(app, client, monkeypatch):
    _create_many(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    batch_id = _start_batch(client).json()["batch_id"]

    assert _get_batch(client, "00000000-0000-0000-0000-000000000000").status_code == 404
    assert (
        _get_batch(client, batch_id, tenant="tenant-b").status_code == 404
    )
    assert (
        _get_batch(client, batch_id, workload="workload-2").status_code == 404
    )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"workload_id": WORKLOAD},
        {"tenant_id": TENANT},
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "   ", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": "  "},
    ],
)
def test_get_batch_requires_valid_scope_query_params(client, params):
    response = client.get(
        "/v1/rewrap-batches/00000000-0000-0000-0000-000000000000", params=params
    )
    assert response.status_code == 422


@pytest.mark.parametrize("batch_id", ["not-a-uuid", "12345", "   ", "%20%20"])
def test_get_batch_rejects_malformed_identifier(client, batch_id):
    response = client.get(
        f"/v1/rewrap-batches/{batch_id}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


# --- durability ------------------------------------------------------------


def test_batches_audits_and_cursors_survive_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart-batches.db"

    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _create_many(client1, ["a", "b", "c"])
    app1.state.engine.dispose()

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    page1 = _start_batch(client2, limit=2).json()
    app2.state.engine.dispose()

    # Restart again: the batch, its per-envelope audits and the cursor are
    # all still queryable, and the cursor continues the range exactly once.
    app3 = app_module.create_app(url)
    client3 = TestClient(app3)
    detail = _get_batch(client3, page1["batch_id"])
    assert detail.status_code == 200
    assert [a["data_id"] for a in detail.json()["audits"]] == ["a", "b"]
    assert detail.json()["next_cursor"] == page1["next_cursor"]

    page2 = _start_batch(client3, cursor=page1["next_cursor"], limit=2).json()
    assert page2["rewrapped"] == 1
    assert page2["done"] is True
    assert page2["next_cursor"] == ""
    assert [
        a["data_id"] for a in _get_batch(client3, page2["batch_id"]).json()["audits"]
    ] == ["c"]
    for data_id in ("a", "b", "c"):
        stored = _get_envelope(client3, data_id).json()
        assert stored["key_version"] == 2
        assert _unwrap_and_decrypt(KEY_V2_BYTES, stored) == PAYLOAD.encode("utf-8")
    app3.state.engine.dispose()


def test_stopped_batch_and_its_cursor_survive_restart_and_resume(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart-stopped.db"

    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _create_many(client1, ["a", "b", "c"])
    app1.state.engine.dispose()

    # Run with a keyring missing historical version 1: the page stops on the
    # first envelope, recording one failed audit and a continuation cursor.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    stopped = _start_batch(client2, limit=3).json()
    assert stopped["failed"] == 1 and stopped["processed"] == 0
    assert stopped["done"] is False and stopped["next_cursor"]
    app2.state.engine.dispose()

    # After restart with the historical key restored, the failed batch and
    # its audit are still readable and its cursor resumes without skipping
    # or repeating anything.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    app3 = app_module.create_app(url)
    client3 = TestClient(app3)
    detail = _get_batch(client3, stopped["batch_id"]).json()
    assert detail["failed"] == 1
    assert [a["data_id"] for a in detail["audits"]] == ["a"]
    assert detail["audits"][0]["result"] == "missing-key"

    resumed = _start_batch(
        client3, cursor=stopped["next_cursor"], limit=3
    ).json()
    assert resumed["rewrapped"] == 3
    assert resumed["failed"] == 0
    assert resumed["done"] is True
    assert resumed["next_cursor"] == ""
    for data_id in ("a", "b", "c"):
        assert _get_envelope(client3, data_id).json()["key_version"] == 2
    app3.state.engine.dispose()
