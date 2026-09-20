"""Tests for POST/GET /v1/data-envelopes (envelope encryption)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import DataEnvelope
from proof_release.envelopes import (
    KEY_VERSION,
    b64url_decode,
    b64url_encode,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap, InvalidUnwrap

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "item-1"
PAYLOAD = "super-secret-attestation-payload 🔐"

# 32-byte fixed master key for tests, rendered as unpadded base64url.
MASTER_KEY_BYTES = b"0123456789abcdef0123456789abcdef"
MASTER_KEY = b64url_encode(MASTER_KEY_BYTES)


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = app_module.create_app(f"sqlite:///{tmp_path}/envelopes.db")
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


def _unwrap_and_decrypt(master_key: bytes, fields: dict) -> bytes:
    data_key = aes_key_unwrap(master_key, b64url_decode(fields["wrapped_key"]))
    iv = b64url_decode(fields["iv"])
    sealed = b64url_decode(fields["ciphertext"]) + b64url_decode(fields["tag"])
    return AESGCM(data_key).decrypt(iv, sealed, None)


def test_create_envelope_returns_201_metadata_only(client):
    response = _create(client)

    assert response.status_code == 201
    data = response.json()
    assert set(data) == {
        "data_id",
        "tenant_id",
        "workload_id",
        "key_version",
        "created_at",
    }
    assert data["data_id"] == DATA_ID
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["key_version"] == KEY_VERSION == 1
    created_at = data["created_at"]
    assert created_at.endswith("+00:00")
    assert datetime.fromisoformat(created_at).utcoffset() == timedelta(0)
    # The payload and any key/ciphertext material never appear on creation.
    assert PAYLOAD not in response.text
    for secret_name in ("ciphertext", "iv", "tag", "wrapped_key", "payload"):
        assert secret_name not in data


def test_get_envelope_returns_material_that_decrypts_and_authenticates(client):
    created = _create(client)
    assert created.status_code == 201
    created_at = created.json()["created_at"]

    response = client.get(
        f"/v1/data-envelopes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "data_id",
        "tenant_id",
        "workload_id",
        "key_version",
        "created_at",
        "ciphertext",
        "iv",
        "tag",
        "wrapped_key",
    }
    assert data["data_id"] == DATA_ID
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["key_version"] == 1
    assert data["created_at"] == created_at

    # All four material fields are strict unpadded base64url.
    for field in ("ciphertext", "iv", "tag", "wrapped_key"):
        raw = b64url_decode(data[field])
        assert "=" not in data[field]
        assert data[field] == b64url_encode(raw)
    assert len(b64url_decode(data["iv"])) == 12
    assert len(b64url_decode(data["tag"])) == 16
    # AES-KW of a 32-byte key is 40 bytes.
    assert len(b64url_decode(data["wrapped_key"])) == 40

    # The material unwraps with this master key version and the
    # authenticated decryption yields the exact payload.
    plaintext = _unwrap_and_decrypt(MASTER_KEY_BYTES, data)
    assert plaintext == PAYLOAD.encode("utf-8")


def test_ciphertext_is_randomized_and_does_not_contain_payload(client):
    first = _create(client)
    second = _create(client, data_id="item-2")
    assert first.status_code == 201 and second.status_code == 201

    one = client.get(
        f"/v1/data-envelopes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    two = client.get(
        "/v1/data-envelopes/item-2",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()

    assert one["wrapped_key"] != two["wrapped_key"]
    assert one["iv"] != two["iv"]
    assert b64url_decode(one["ciphertext"]) != PAYLOAD.encode("utf-8")
    assert PAYLOAD.encode("utf-8") not in b64url_decode(one["ciphertext"])


def test_tampered_ciphertext_fails_authentication(client):
    _create(client)
    data = client.get(
        f"/v1/data-envelopes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()

    tampered = bytearray(b64url_decode(data["ciphertext"]))
    tampered[0] ^= 0x01
    data["ciphertext"] = b64url_encode(bytes(tampered))
    with pytest.raises(Exception):
        _unwrap_and_decrypt(MASTER_KEY_BYTES, data)


def test_wrapped_key_does_not_unwrap_under_another_master_key(client):
    _create(client)
    data = client.get(
        f"/v1/data-envelopes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()

    other_key = b"ffffffffffffffffffffffffffffffff"
    with pytest.raises(InvalidUnwrap):
        aes_key_unwrap(other_key, b64url_decode(data["wrapped_key"]))


def test_duplicate_data_id_same_scope_returns_409(client):
    assert _create(client).status_code == 201
    duplicate = _create(client)
    assert duplicate.status_code == 409
    assert PAYLOAD not in duplicate.text


def test_same_data_id_is_distinct_per_scope(client):
    assert _create(client).status_code == 201
    other_tenant = _create(client, tenant_id="tenant-b")
    other_workload = _create(client, workload_id="workload-2")
    assert other_tenant.status_code == 201
    assert other_workload.status_code == 201

    # The two other-scope envelopes decrypt independently to the payload.
    for params in (
        {"tenant_id": "tenant-b", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": "workload-2"},
    ):
        response = client.get(
            f"/v1/data-envelopes/{DATA_ID}", params=params
        )
        assert response.status_code == 200
        assert (
            _unwrap_and_decrypt(MASTER_KEY_BYTES, response.json())
            == PAYLOAD.encode("utf-8")
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"workload_id": ""},
        {"workload_id": "   "},
        {"data_id": ""},
        {"data_id": "   "},
        {"data_id": 7},
        {"payload": ""},
        {"payload": 42},
    ],
)
def test_create_envelope_rejects_invalid_fields(client, overrides):
    assert _create(client, **overrides).status_code == 422


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "data_id", "payload"])
def test_create_envelope_requires_fields(client, missing):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "payload": PAYLOAD,
    }
    del body[missing]
    assert client.post("/v1/data-envelopes", json=body).status_code == 422


def test_get_unknown_or_cross_scope_returns_404(client):
    assert _create(client).status_code == 201

    assert (
        client.get(
            "/v1/data-envelopes/missing",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).status_code
        == 404
    )
    # An existing data_id probed from another scope is indistinguishable
    # from a missing one.
    assert (
        client.get(
            f"/v1/data-envelopes/{DATA_ID}",
            params={"tenant_id": "tenant-b", "workload_id": WORKLOAD},
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/v1/data-envelopes/{DATA_ID}",
            params={"tenant_id": TENANT, "workload_id": "workload-2"},
        ).status_code
        == 404
    )


def test_get_requires_query_parameters(client):
    assert _create(client).status_code == 201
    base = f"/v1/data-envelopes/{DATA_ID}"

    assert client.get(base).status_code == 422
    assert (
        client.get(base, params={"workload_id": WORKLOAD}).status_code == 422
    )
    assert (
        client.get(base, params={"tenant_id": TENANT}).status_code == 422
    )
    assert (
        client.get(
            base, params={"tenant_id": "", "workload_id": WORKLOAD}
        ).status_code
        == 422
    )
    assert (
        client.get(
            base, params={"tenant_id": "  ", "workload_id": WORKLOAD}
        ).status_code
        == 422
    )
    assert (
        client.get(
            base, params={"tenant_id": TENANT, "workload_id": " "}
        ).status_code
        == 422
    )


def test_envelopes_persist_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = app_module.create_app(url)
    created = TestClient(app1).post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": DATA_ID,
            "payload": PAYLOAD,
        },
    )
    assert created.status_code == 201
    app1.state.engine.dispose()

    app2 = app_module.create_app(url)
    response = TestClient(app2).get(
        f"/v1/data-envelopes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    assert (
        _unwrap_and_decrypt(MASTER_KEY_BYTES, response.json())
        == PAYLOAD.encode("utf-8")
    )
    # Duplicate rejected after restart as well.
    duplicate = TestClient(app2).post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": DATA_ID,
            "payload": PAYLOAD,
        },
    )
    assert duplicate.status_code == 409
    app2.state.engine.dispose()


def test_no_plaintext_payload_or_data_key_is_persisted(app, client):
    assert _create(client).status_code == 201
    with app.state.session_factory() as session:
        record = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        assert record is not None
        columns = {
            c.name: getattr(record, c.name) for c in record.__table__.columns
        }

    payload_bytes = PAYLOAD.encode("utf-8")
    for name, value in columns.items():
        blob = value if isinstance(value, bytes) else str(value).encode("utf-8")
        assert payload_bytes not in blob, f"plaintext payload in column {name}"

    # The stored wrapped key is not the raw data key; the plaintext data
    # key is irrecoverable without the master key.
    assert columns["wrapped_key"] != b""
    with pytest.raises(InvalidUnwrap):
        aes_key_unwrap(b"x" * 32, columns["wrapped_key"])


@pytest.mark.parametrize(
    "bad_key",
    [
        None,  # unset
        "",
        "not-base64!!!",
        # 32 bytes of valid base64url text but padded form is rejected.
        b64url_encode(MASTER_KEY_BYTES) + "=",
        # Decodes to 31 bytes.
        b64url_encode(b"a" * 31),
        # Decodes to 33 bytes.
        b64url_encode(b"a" * 33),
    ],
)
def test_bad_master_key_returns_500_and_writes_nothing(
    tmp_path, monkeypatch, bad_key
):
    if bad_key is None:
        monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    else:
        monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", bad_key)
    application = app_module.create_app(f"sqlite:///{tmp_path}/badkey.db")
    client = TestClient(application)

    response = _create(client)
    assert response.status_code == 500
    with application.state.session_factory() as session:
        assert session.query(DataEnvelope).count() == 0
    application.state.engine.dispose()


def test_encryption_failure_returns_500_and_leaves_no_record(
    app, client, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("crypto backend exploded")

    monkeypatch.setattr(app_module, "encrypt_payload", boom)

    response = _create(client)
    assert response.status_code == 500
    assert PAYLOAD not in response.text
    with app.state.session_factory() as session:
        assert session.query(DataEnvelope).count() == 0


def test_write_failure_returns_500_and_leaves_no_record(app, client, monkeypatch):
    from sqlalchemy.orm import Session

    def raise_commit(self):
        raise OSError("disk full")

    monkeypatch.setattr(Session, "commit", raise_commit)

    response = _create(client)
    assert response.status_code == 500

    # Restore before verifying so the check sees the real database state.
    monkeypatch.undo()
    with app.state.session_factory() as session:
        assert session.query(DataEnvelope).count() == 0


def test_material_is_written_atomically_in_one_row(app, client):
    assert _create(client).status_code == 201
    with app.state.session_factory() as session:
        record = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        assert record.ciphertext
        assert len(record.iv) == 12
        assert len(record.tag) == 16
        assert len(record.wrapped_key) == 40
        assert record.key_version == 1
        # Direct round trip against the stored columns.
        data_key = aes_key_unwrap(MASTER_KEY_BYTES, record.wrapped_key)
        plaintext = AESGCM(data_key).decrypt(
            record.iv, record.ciphertext + record.tag, None
        )
        assert plaintext == PAYLOAD.encode("utf-8")
