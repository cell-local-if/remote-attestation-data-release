"""Tests for POST /v1/data-envelopes and GET /v1/data-envelopes/{data_id}."""

from __future__ import annotations

import base64
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap

from proof_release.app import create_app
from proof_release.db import DataEnvelope

TENANT = "tenant-a"
WORKLOAD = "workload-1"

#: 32 raw bytes rendered as unpadded base64url (43 characters).
MASTER_KEY_RAW = bytes(range(32))
MASTER_KEY_B64URL = base64.urlsafe_b64encode(MASTER_KEY_RAW).rstrip(b"=").decode(
    "ascii"
)

_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _unseal(envelope: dict, master_key: bytes = MASTER_KEY_RAW) -> str:
    """Mirror the service: AES-KW unwrap the data key, then AES-GCM decrypt."""
    data_key = aes_key_unwrap(master_key, _b64url_decode(envelope["wrapped_key"]))
    sealed = _b64url_decode(envelope["ciphertext"]) + _b64url_decode(
        envelope["tag"]
    )
    plaintext = AESGCM(data_key).decrypt(
        _b64url_decode(envelope["iv"]), sealed, None
    )
    return plaintext.decode("utf-8")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY_B64URL)
    return create_app(f"sqlite:///{tmp_path}/envelopes.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": "data-1",
        "payload": "top secret payload",
    }
    body.update(overrides)
    return client.post("/v1/data-envelopes", json=body)


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
    assert data["data_id"] == "data-1"
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["key_version"] == 1
    created_at = datetime.fromisoformat(data["created_at"])
    assert created_at.utcoffset() == timedelta(0)
    # The plaintext and all sealed material must stay off the create response.
    assert "payload" not in response.text
    assert "top secret payload" not in response.text
    for field in ("ciphertext", "iv", "tag", "wrapped_key"):
        assert field not in data


def test_created_envelope_is_encrypt_envelope(client):
    created = _create(client, payload="unicode: é中文 \U0001f512")
    fetched = client.get(
        f"/v1/data-envelopes/data-1",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert fetched.status_code == 200
    envelope = fetched.json()

    # Every sealed field is unpadded base64url of the expected length.
    for field in ("ciphertext", "iv", "tag", "wrapped_key"):
        assert _B64URL_RE.fullmatch(envelope[field])
        assert "=" not in envelope[field]
    plaintext = "unicode: é中文 \U0001f512".encode("utf-8")
    assert len(_b64url_decode(envelope["iv"])) == 12
    assert len(_b64url_decode(envelope["tag"])) == 16
    # AES-KW of a 32-byte key is 40 bytes (8-byte integrity prefix).
    assert len(_b64url_decode(envelope["wrapped_key"])) == 40
    assert len(_b64url_decode(envelope["ciphertext"])) == len(plaintext)

    # The sealed envelope unwraps with the version-1 master key and
    # authenticated-decrypts back to exactly the submitted payload.
    assert _unseal(envelope) == "unicode: é中文 \U0001f512"


def test_each_envelope_gets_a_fresh_data_key_and_iv(client):
    first = _create(client, data_id="data-a").json()
    second = _create(client, data_id="data-b", payload="top secret payload")

    def fetch(data_id):
        return client.get(
            f"/v1/data-envelopes/{data_id}",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).json()

    a, b = fetch("data-a"), fetch("data-b")
    assert a["iv"] != b["iv"]
    assert a["wrapped_key"] != b["wrapped_key"]
    assert a["ciphertext"] != b["ciphertext"]


def test_duplicate_data_id_in_same_scope_returns_409(client):
    assert _create(client).status_code == 201
    duplicate = _create(client, payload="a different secret")
    assert duplicate.status_code == 409
    assert "payload" not in duplicate.text
    assert "different secret" not in duplicate.text

    # The original envelope is untouched and still decrypts correctly.
    envelope = client.get(
        "/v1/data-envelopes/data-1",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    assert _unseal(envelope) == "top secret payload"


def test_same_data_id_in_other_scope_is_independent(client):
    assert _create(client).status_code == 201
    other_tenant = _create(client, tenant_id="tenant-b", payload="tenant b secret")
    assert other_tenant.status_code == 201
    other_workload = _create(
        client, workload_id="workload-2", payload="workload 2 secret"
    )
    assert other_workload.status_code == 201


def test_concurrent_creates_only_one_succeeds(app):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": "race",
        "payload": "race payload",
    }

    def create():
        return TestClient(app).post("/v1/data-envelopes", json=body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: create(), range(8)))

    assert statuses.count(201) == 1
    assert statuses.count(409) == 7
    with app.state.session_factory() as session:
        assert session.query(DataEnvelope).count() == 1


@pytest.mark.parametrize(
    "missing", ["tenant_id", "workload_id", "data_id", "payload"]
)
def test_create_requires_all_fields(client, missing):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": "data-1",
        "payload": "top secret payload",
    }
    del body[missing]
    assert client.post("/v1/data-envelopes", json=body).status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"workload_id": "\t"},
        {"data_id": ""},
        {"data_id": "   "},
        {"payload": ""},
        {"payload": "   "},
        {"tenant_id": 1},
        {"data_id": 42},
        {"payload": 0},
        {"payload": ["not", "a", "string"]},
        {"payload": None},
    ],
)
def test_create_rejects_invalid_fields(client, overrides):
    assert _create(client, **overrides).status_code == 422


@pytest.mark.parametrize(
    "bad_key",
    [
        None,  # unset
        "",
        "   ",
        # 32 characters of base64url text decodes to 24 bytes, not 32.
        "A" * 32,
        # 44 characters: padded base64 is rejected (must be unpadded).
        MASTER_KEY_B64URL + "=",
        # 43 characters but outside the base64url alphabet.
        "!" + "A" * 42,
        # Correct length and alphabet but non-canonical for length? 43 chars
        # always decode to 32 bytes; instead corrupt the final char so the
        # 31-byte payload region cannot be reached — urlsafe decode of 43
        # chars still yields 32 bytes, so validate length strictly with a
        # 42-char value below.
        "A" * 42,
    ],
)
def test_bad_master_key_returns_500_and_writes_nothing(
    tmp_path, monkeypatch, bad_key
):
    url = f"sqlite:///{tmp_path}/badkey.db"
    app = create_app(url)
    client = TestClient(app)
    if bad_key is None:
        monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    else:
        monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", bad_key)

    response = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": "data-1",
            "payload": "top secret payload",
        },
    )
    assert response.status_code == 500
    with app.state.session_factory() as session:
        assert session.query(DataEnvelope).count() == 0


def test_get_envelope_returns_sealed_fields(client):
    _create(client, data_id="data-7", payload="sealed")
    response = client.get(
        "/v1/data-envelopes/data-7",
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
    assert data["data_id"] == "data-7"
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["key_version"] == 1
    assert datetime.fromisoformat(data["created_at"]).utcoffset() == timedelta(0)
    # GET returns sealed material only, never plaintext.
    assert "sealed" not in response.text


def test_get_unknown_or_cross_scope_returns_404(client):
    _create(client)

    unknown = client.get(
        "/v1/data-envelopes/missing",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert unknown.status_code == 404

    other_tenant = client.get(
        "/v1/data-envelopes/data-1",
        params={"tenant_id": "tenant-b", "workload_id": WORKLOAD},
    )
    assert other_tenant.status_code == 404

    other_workload = client.get(
        "/v1/data-envelopes/data-1",
        params={"tenant_id": TENANT, "workload_id": "workload-2"},
    )
    assert other_workload.status_code == 404


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id"])
def test_get_requires_scope_query_params(client, missing):
    _create(client)
    params = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    del params[missing]
    response = client.get("/v1/data-envelopes/data-1", params=params)
    assert response.status_code == 422


@pytest.mark.parametrize("blank", ["", "   "])
def test_get_rejects_blank_scope_query_params(client, blank):
    _create(client)
    response = client.get(
        "/v1/data-envelopes/data-1",
        params={"tenant_id": blank, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422
    response = client.get(
        "/v1/data-envelopes/data-1",
        params={"tenant_id": TENANT, "workload_id": blank},
    )
    assert response.status_code == 422


def test_get_does_not_require_master_key_or_decrypt(tmp_path, monkeypatch):
    # Sealing needs the master key; reading sealed bytes back must not.
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY_B64URL)
    url = f"sqlite:///{tmp_path}/getnokey.db"
    app = create_app(url)
    TestClient(app).post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": "data-1",
            "payload": "sealed secret",
        },
    )
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY")

    response = TestClient(app).get(
        "/v1/data-envelopes/data-1",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    envelope = response.json()
    assert "sealed secret" not in response.text
    # Restore the key out-of-band and confirm the stored fields are intact.
    assert _unseal(envelope) == "sealed secret"


def test_tampered_ciphertext_fails_authenticated_decryption(client):
    _create(client)
    envelope = client.get(
        "/v1/data-envelopes/data-1",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    tampered = dict(envelope)
    tampered["ciphertext"] = ("A" if envelope["ciphertext"][0] != "A" else "B") + (
        envelope["ciphertext"][1:]
    )
    with pytest.raises(InvalidTag):
        _unseal(tampered)


def test_envelope_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY_B64URL)
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    created = TestClient(app1).post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": "data-1",
            "payload": "persisted secret",
        },
    )
    assert created.status_code == 201
    app1.state.engine.dispose()

    response = TestClient(create_app(url)).get(
        "/v1/data-envelopes/data-1",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    envelope = response.json()
    assert envelope["key_version"] == 1
    assert _unseal(envelope) == "persisted secret"


def test_plaintext_is_never_persisted(client, app):
    marker = "PLAINTEXT-MARKER-" + os.urandom(8).hex()
    assert _create(client, data_id="data-9", payload=marker).status_code == 201

    with app.state.session_factory() as session:
        row = session.get(DataEnvelope, (TENANT, WORKLOAD, "data-9"))
        assert row is not None
        columns = {c.name: getattr(row, c.name) for c in row.__table__.columns}
        for name, value in columns.items():
            assert marker not in str(value), f"payload leaked into column {name}"
        # The stored key_version is the master key version, never key bytes.
        assert row.key_version == 1
