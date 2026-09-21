"""Tests for master key rotation: the keyring configuration and
POST /v1/data-envelopes/{data_id}/rewrap."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import DataEnvelope
from proof_release.envelopes import b64url_decode, b64url_encode

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "item-1"
PAYLOAD = "super-secret-attestation-payload 🔐"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V2_BYTES = b"fedcba9876543210fedcba9876543210"
KEY_V3_BYTES = b"a" * 32
KEY_V1 = b64url_encode(KEY_V1_BYTES)
KEY_V2 = b64url_encode(KEY_V2_BYTES)
KEY_V3 = b64url_encode(KEY_V3_BYTES)

KEYRING_V1_V2 = json.dumps(
    {"current_version": 2, "keys": {"1": KEY_V1, "2": KEY_V2}}
)
KEYRING_V2_ONLY = json.dumps({"current_version": 2, "keys": {"2": KEY_V2}})


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = app_module.create_app(f"sqlite:///{tmp_path}/rotation.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "payload": PAYLOAD,
    }
    body.update(overrides)
    return client.post("/v1/data-envelopes", json=body)


def _get(client, data_id=DATA_ID, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/data-envelopes/{data_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _rewrap(client, data_id=DATA_ID, **overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    body.update(overrides)
    return client.post(f"/v1/data-envelopes/{data_id}/rewrap", json=body)


def _unwrap_and_decrypt(master_key: bytes, fields: dict) -> bytes:
    data_key = aes_key_unwrap(master_key, b64url_decode(fields["wrapped_key"]))
    iv = b64url_decode(fields["iv"])
    sealed = b64url_decode(fields["ciphertext"]) + b64url_decode(fields["tag"])
    return AESGCM(data_key).decrypt(iv, sealed, None)


# --- keyring configuration -------------------------------------------------


def test_keyring_current_version_is_used_for_new_envelopes(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _create(client)
    assert response.status_code == 201
    assert response.json()["key_version"] == 2

    stored = _get(client).json()
    assert stored["key_version"] == 2
    assert _unwrap_and_decrypt(KEY_V2_BYTES, stored) == PAYLOAD.encode("utf-8")


def test_legacy_master_key_is_version_one_when_keyring_unset(client):
    response = _create(client)
    assert response.status_code == 201
    assert response.json()["key_version"] == 1
    stored = _get(client).json()
    assert _unwrap_and_decrypt(KEY_V1_BYTES, stored) == PAYLOAD.encode("utf-8")


def test_keyring_takes_precedence_over_legacy_variable(app, client, monkeypatch):
    # The legacy variable is deliberately invalid: if it were consulted,
    # the request would fail.
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", "not-base64!!!")
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    response = _create(client)
    assert response.status_code == 201
    assert response.json()["key_version"] == 2


@pytest.mark.parametrize(
    "keyring",
    [
        "not json",
        "[]",
        "{}",
        # current_version must be a positive integer.
        json.dumps({"current_version": 0, "keys": {"1": KEY_V1}}),
        json.dumps({"current_version": -1, "keys": {"1": KEY_V1}}),
        json.dumps({"current_version": "1", "keys": {"1": KEY_V1}}),
        json.dumps({"current_version": True, "keys": {"1": KEY_V1}}),
        json.dumps({"current_version": 1.5, "keys": {"1": KEY_V1}}),
        # keys must be a non-empty object of decimal positive integers.
        json.dumps({"current_version": 1, "keys": {}}),
        json.dumps({"current_version": 1, "keys": []}),
        json.dumps({"current_version": 1, "keys": {"0": KEY_V1}}),
        json.dumps({"current_version": 1, "keys": {"-1": KEY_V1}}),
        json.dumps({"current_version": 1, "keys": {"01": KEY_V1}}),
        json.dumps({"current_version": 1, "keys": {"one": KEY_V1}}),
        # Key material must be unpadded base64url of exactly 32 bytes.
        json.dumps({"current_version": 1, "keys": {"1": "not-base64!!!"}}),
        json.dumps({"current_version": 1, "keys": {"1": KEY_V1 + "="}}),
        json.dumps({"current_version": 1, "keys": {"1": b64url_encode(b"a" * 31)}}),
        json.dumps({"current_version": 1, "keys": {"1": b64url_encode(b"a" * 33)}}),
        json.dumps({"current_version": 1, "keys": {"1": 42}}),
        # current_version must have a configured key.
        json.dumps({"current_version": 2, "keys": {"1": KEY_V1}}),
    ],
)
def test_invalid_keyring_returns_500_and_writes_nothing(
    tmp_path, monkeypatch, keyring
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", keyring)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = app_module.create_app(f"sqlite:///{tmp_path}/badring.db")
    client = TestClient(application)

    response = _create(client)
    assert response.status_code == 500
    assert PAYLOAD not in response.text
    with application.state.session_factory() as session:
        assert session.query(DataEnvelope).count() == 0
    application.state.engine.dispose()


# --- rewrap ----------------------------------------------------------------


def test_rewrap_rotates_to_current_version_and_preserves_material(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    before = _get(client).json()
    assert before["key_version"] == 1

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _rewrap(client)

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "data_id",
        "tenant_id",
        "workload_id",
        "key_version",
        "rotated_at",
    }
    assert data["data_id"] == DATA_ID
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["key_version"] == 2
    rotated_at = data["rotated_at"]
    assert rotated_at.endswith("+00:00")
    assert datetime.fromisoformat(rotated_at).utcoffset() == timedelta(0)
    # No key or payload material on the response.
    assert PAYLOAD not in response.text
    for secret_name in ("ciphertext", "iv", "tag", "wrapped_key", "payload"):
        assert secret_name not in data

    after = _get(client).json()
    assert after["key_version"] == 2
    # Only the wrapping changed; everything else is byte-identical.
    for field in ("ciphertext", "iv", "tag", "created_at", "data_id"):
        assert after[field] == before[field]
    assert after["wrapped_key"] != before["wrapped_key"]
    # The rotated material unwraps under the new current key to the payload.
    assert _unwrap_and_decrypt(KEY_V2_BYTES, after) == PAYLOAD.encode("utf-8")


def test_rewrap_retry_at_current_version_changes_no_material(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _rewrap(client).status_code == 200
    settled = _get(client).json()

    again = _rewrap(client)
    assert again.status_code == 200
    assert again.json()["key_version"] == 2
    assert _get(client).json() == settled


def test_rewrap_unknown_or_cross_scope_returns_404(app, client, monkeypatch):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    assert _rewrap(client, data_id="missing").status_code == 404
    assert _rewrap(client, tenant_id="tenant-b").status_code == 404
    assert _rewrap(client, workload_id="workload-2").status_code == 404


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"workload_id": ""},
        {"workload_id": "   "},
        {"workload_id": 7},
    ],
)
def test_rewrap_rejects_invalid_fields(client, overrides):
    assert _rewrap(client, **overrides).status_code == 422


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id"])
def test_rewrap_requires_fields(client, missing):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    del body[missing]
    assert (
        client.post(f"/v1/data-envelopes/{DATA_ID}/rewrap", json=body).status_code
        == 422
    )


def test_rewrap_without_stored_version_key_returns_500_and_keeps_row(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    before = _get(client).json()

    # The keyring no longer carries version 1, so the row cannot be unwrapped.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    response = _rewrap(client)
    assert response.status_code == 500
    assert PAYLOAD not in response.text
    assert _get(client).json() == before


def test_rewrap_with_invalid_keyring_returns_500_and_keeps_row(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    before = _get(client).json()

    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING",
        json.dumps({"current_version": 3, "keys": {"1": KEY_V1}}),
    )
    assert _rewrap(client).status_code == 500
    assert _get(client).json() == before


def test_rewrap_failure_returns_500_and_keeps_row(app, client, monkeypatch):
    assert _create(client).status_code == 201
    before = _get(client).json()
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    def boom(*args, **kwargs):
        raise RuntimeError("crypto backend exploded")

    monkeypatch.setattr(app_module, "rewrap_data_key", boom)
    response = _rewrap(client)
    assert response.status_code == 500
    assert PAYLOAD not in response.text
    monkeypatch.undo()
    assert _get(client).json() == before


def test_rewrap_survives_restart_and_rotated_material_decrypts(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart-rotation.db"
    app1 = app_module.create_app(url)
    assert _create(TestClient(app1)).status_code == 201
    app1.state.engine.dispose()

    # Restart with a rotated keyring: historical version 1 is still
    # supported, new envelopes use version 2.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    assert _get(client2).json()["key_version"] == 1
    assert _rewrap(client2).json()["key_version"] == 2
    app2.state.engine.dispose()

    # Restart again with only the current key: the rotated row still reads.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    app3 = app_module.create_app(url)
    client3 = TestClient(app3)
    stored = _get(client3).json()
    assert stored["key_version"] == 2
    assert _unwrap_and_decrypt(KEY_V2_BYTES, stored) == PAYLOAD.encode("utf-8")
    app3.state.engine.dispose()


def test_concurrent_rewraps_leave_current_version_material(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _rewrap(client), range(8)))
    assert all(r.status_code == 200 for r in results)
    assert all(r.json()["key_version"] == 2 for r in results)

    stored = _get(client).json()
    assert stored["key_version"] == 2
    assert _unwrap_and_decrypt(KEY_V2_BYTES, stored) == PAYLOAD.encode("utf-8")


def test_rewrap_response_and_row_never_expose_key_material(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _rewrap(client)
    assert response.status_code == 200
    # Neither master key nor the plaintext data key appears anywhere.
    for key_text in (KEY_V1, KEY_V2):
        assert key_text not in response.text
    with app.state.session_factory() as session:
        record = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        data_key = aes_key_unwrap(KEY_V2_BYTES, record.wrapped_key)
        assert data_key not in record.wrapped_key
        assert record.ciphertext != PAYLOAD.encode("utf-8")


def test_multi_step_rotation_chain(tmp_path, monkeypatch):
    # v1 -> v2 -> v3 across restarts, each step preserving the payload.
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/chain.db"
    app1 = app_module.create_app(url)
    assert _create(TestClient(app1)).status_code == 201
    app1.state.engine.dispose()

    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(2, {1: KEY_V1, 2: KEY_V2, 3: KEY_V3})
    )
    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    assert _rewrap(client2).json()["key_version"] == 2
    app2.state.engine.dispose()

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", _keyring(3, {2: KEY_V2, 3: KEY_V3}))
    app3 = app_module.create_app(url)
    client3 = TestClient(app3)
    assert _rewrap(client3).json()["key_version"] == 3
    stored = _get(client3).json()
    assert stored["key_version"] == 3
    assert _unwrap_and_decrypt(KEY_V3_BYTES, stored) == PAYLOAD.encode("utf-8")
    app3.state.engine.dispose()
