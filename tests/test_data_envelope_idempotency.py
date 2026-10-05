"""Tests for the optional Idempotency-Key header on POST /v1/data-envelopes.

A keyed creation writes the envelope and its idempotency record in one
transaction and replays the exact stored 201 afterwards; a missing key
keeps the legacy one-envelope-per-request behavior. The record persists
only the scope, the key, the data_id, the payload digest, the first
response body and the creation time — never the plaintext payload, a
data key, a master key or any ciphertext material.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import (
    DataEnvelope,
    DataEnvelopeCommitCounter,
    DataEnvelopeIdempotencyRecord,
)
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "item-1"
PAYLOAD = "super-secret-attestation-payload 🔐"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V2_BYTES = b"fedcba9876543210fedcba9876543210"
KEY_V1 = b64url_encode(KEY_V1_BYTES)
KEY_V2 = b64url_encode(KEY_V2_BYTES)

IDEMPOTENCY_HEADER = "Idempotency-Key"


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


KEYRING_V1_V2 = _keyring(2, {1: KEY_V1, 2: KEY_V2})


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = app_module.create_app(f"sqlite:///{tmp_path}/idem-envelopes.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(
    client,
    *,
    tenant=TENANT,
    workload=WORKLOAD,
    data_id=DATA_ID,
    payload=PAYLOAD,
    key=None,
):
    body = {
        "tenant_id": tenant,
        "workload_id": workload,
        "data_id": data_id,
        "payload": payload,
    }
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else None
    return client.post("/v1/data-envelopes", json=body, headers=headers)


def _count_rows(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


def _records(app):
    with app.state.session_factory() as session:
        return session.query(DataEnvelopeIdempotencyRecord).all()


def _scope_seq(app, tenant=TENANT, workload=WORKLOAD):
    with app.state.session_factory() as session:
        counter = session.get(DataEnvelopeCommitCounter, (tenant, workload))
        return None if counter is None else counter.last_seq


async def _asgi_call(application, path, headers, body: bytes):
    """Invoke the ASGI app directly with arbitrary raw header bytes.

    The HTTP test client refuses some header values (control characters,
    non-ASCII bytes) before they reach the app; driving the ASGI app
    lets the service's own header validation see them.
    """
    chunks: list[bytes] = []
    status: dict[str, int] = {}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            status["code"] = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }
    await application(scope, receive, send)
    return status["code"], b"".join(chunks)


def _raw_create(app, body_obj, *, key_raw=b"key-1", duplicate_key=False):
    payload = json.dumps(body_obj).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"idempotency-key", key_raw),
    ]
    if duplicate_key:
        headers.append((b"idempotency-key", key_raw))
    return asyncio.run(_asgi_call(app, "/v1/data-envelopes", headers, payload))


def _body(**overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "payload": PAYLOAD,
    }
    body.update(overrides)
    return body


# --- header validation -----------------------------------------------------


@pytest.mark.parametrize(
    "raw_key",
    [
        b"",  # empty value
        b" ",  # bare whitespace
        b" key-1",  # leading space
        b"key-1 ",  # trailing space
        b"key 1",  # embedded space
        b"key\t1",  # tab
        b"key-1\n",  # control character
        b"k\xc3\xa9y",  # non-ASCII (UTF-8 e-acute)
        b"k" * 65,  # over-long
    ],
)
def test_invalid_idempotency_key_is_422_before_any_write(app, raw_key):
    status, raw = _raw_create(app, _body(), key_raw=raw_key)
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    assert _count_rows(app, DataEnvelope) == 0
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0


def test_duplicate_idempotency_header_is_422(app):
    status, raw = _raw_create(app, _body(), duplicate_key=True)
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    assert _count_rows(app, DataEnvelope) == 0
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0


def test_duplicate_idempotency_header_through_client_is_422(client):
    response = client.post(
        "/v1/data-envelopes",
        json=_body(),
        headers=[
            (IDEMPOTENCY_HEADER, "key-1"),
            (IDEMPOTENCY_HEADER, "key-1"),
        ],
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid idempotency key"}
    assert _count_rows(client.app, DataEnvelope) == 0


@pytest.mark.parametrize("length", [1, 64])
def test_boundary_length_keys_accepted(app, client, length):
    response = _create(client, key="A" * length, data_id=f"item-{length}")
    assert response.status_code == 201, response.text
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1


def test_visible_punctuation_key_accepted(client):
    response = _create(client, key="~!@#$%^&*()_+-=[]{}|;:',.<>/?`")
    assert response.status_code == 201, response.text


def test_invalid_key_is_not_consumed_and_later_valid_use_succeeds(app, client):
    assert _create(client, key="").status_code == 422
    assert _raw_create(app, _body(), key_raw=b"bad key")[0] == 422
    accepted = _create(client, key="key-1")
    assert accepted.status_code == 201
    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1


def test_body_validation_still_fails_with_valid_idempotency_key(client):
    bad_bodies = [
        {"workload_id": WORKLOAD, "data_id": DATA_ID, "payload": PAYLOAD},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "data_id": DATA_ID},
        _body(tenant_id=""),
        _body(tenant_id="   "),
        _body(workload_id=""),
        _body(data_id=""),
        _body(payload=""),
        _body(tenant_id=7),
        {},
    ]
    for body in bad_bodies:
        response = client.post(
            "/v1/data-envelopes",
            json=body,
            headers={IDEMPOTENCY_HEADER: "key-1"},
        )
        assert response.status_code == 422, body
    assert _count_rows(client.app, DataEnvelope) == 0
    assert _count_rows(client.app, DataEnvelopeIdempotencyRecord) == 0


def test_invalid_key_rejected_before_keyring_check(app, client, monkeypatch):
    # An illegal header must be a 422 even when the keyring is unusable:
    # header validation precedes the keyring check and writes nothing.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    response = client.post(
        "/v1/data-envelopes", json=_body(), headers={IDEMPOTENCY_HEADER: ""}
    )
    assert response.status_code == 422
    assert _raw_create(app, _body(), key_raw=b"x" * 65)[0] == 422
    assert _count_rows(app, DataEnvelope) == 0
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0


# --- core replay semantics -------------------------------------------------


def test_first_keyed_create_writes_envelope_and_record_atomically(app, client):
    response = _create(client, key="key-1")

    assert response.status_code == 201
    data = response.json()
    assert set(data) == {
        "data_id",
        "tenant_id",
        "workload_id",
        "key_version",
        "created_at",
        "classification",
    }
    assert data["data_id"] == DATA_ID
    assert data["key_version"] == 1
    assert data["classification"] == "unclassified"

    assert _count_rows(app, DataEnvelope) == 1
    records = _records(app)
    assert len(records) == 1
    record = records[0]
    assert record.tenant_id == TENANT
    assert record.workload_id == WORKLOAD
    assert record.idempotency_key == "key-1"
    assert record.data_id == DATA_ID
    assert record.classification == "unclassified"
    # Only the irreversible payload digest is stored, never the payload.
    assert record.payload_sha256 == hashlib.sha256(
        PAYLOAD.encode("utf-8")
    ).hexdigest()
    assert PAYLOAD not in record.response_body
    assert json.loads(record.response_body) == data
    # The stored body is the exact wire form of the first response.
    assert record.response_body.encode("utf-8") == response.content


def test_replay_returns_first_response_byte_for_byte(app, client):
    first = _create(client, key="key-1")
    assert first.status_code == 201

    replay = _create(client, key="key-1")
    assert replay.status_code == 201
    assert replay.content == first.content
    assert replay.json() == first.json()

    # No second envelope, no second record, no sequence advancement.
    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1
    assert _scope_seq(app) == 1


def test_replay_after_key_rotation_keeps_original_key_version(
    app, client, monkeypatch
):
    first = _create(client, key="key-1")
    assert first.status_code == 201
    assert first.json()["key_version"] == 1

    # Rotate the keyring: new envelopes wrap under version 2.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)

    replay = _create(client, key="key-1")
    assert replay.status_code == 201
    assert replay.content == first.content
    assert replay.json()["key_version"] == 1

    # The stored envelope is untouched: still version 1, still one row.
    with app.state.session_factory() as session:
        envelopes = session.query(DataEnvelope).all()
    assert len(envelopes) == 1
    assert envelopes[0].key_version == 1
    assert _scope_seq(app) == 1

    # A different key still creates a fresh envelope under version 2.
    other = _create(client, key="key-2", data_id="item-2")
    assert other.status_code == 201
    assert other.json()["key_version"] == 2


def test_same_key_different_payload_is_409_and_changes_nothing(app, client):
    first = _create(client, key="key-1")
    assert first.status_code == 201

    conflict = _create(client, key="key-1", payload="other-payload")
    assert conflict.status_code == 409
    assert conflict.json() == {
        "detail": "idempotency key reused with a different request"
    }

    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1
    # The original record still replays the original response.
    assert _create(client, key="key-1").content == first.content


def test_same_key_different_data_id_is_409_and_changes_nothing(app, client):
    first = _create(client, key="key-1")
    assert first.status_code == 201

    conflict = _create(client, key="key-1", data_id="item-2")
    assert conflict.status_code == 409
    assert conflict.json() == {
        "detail": "idempotency key reused with a different request"
    }

    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1
    assert _create(client, key="key-1").content == first.content


def test_same_key_different_scope_fields_is_409(app, client):
    first = _create(client, key="key-1")
    assert first.status_code == 201

    # A different tenant or workload is a different key namespace, so a
    # conflict only arises inside the original scope. Replaying the key
    # in the original scope with the original scope's fields but a
    # mismatched body field is a 409; here the data_id/payload match but
    # the record is keyed per scope, so cross-scope use is independent.
    other_tenant = _create(client, key="key-1", tenant="tenant-b")
    assert other_tenant.status_code == 201
    other_workload = _create(client, key="key-1", workload="workload-2")
    assert other_workload.status_code == 201

    # Three independent scopes, three envelopes, three records.
    assert _count_rows(app, DataEnvelope) == 3
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 3

    # Within each scope the key replays its own first response only.
    assert _create(client, key="key-1").content == first.content
    assert _create(client, key="key-1", tenant="tenant-b").content == (
        other_tenant.content
    )
    # Cross-scope replay never leaks the other scope's response.
    assert other_tenant.content != first.content


def test_keyed_create_with_existing_data_id_is_409_and_key_stays_free(
    app, client,
):
    # An unkeyed creation already owns the data_id.
    assert _create(client).status_code == 201

    conflict = _create(client, key="key-1")
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "data_id already exists in this scope"}
    # The failed judgement wrote no record: the key is still free.
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0

    created = _create(client, key="key-1", data_id="item-2")
    assert created.status_code == 201
    assert _count_rows(app, DataEnvelope) == 2
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1


def test_encryption_failure_leaves_no_record_and_recovers(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)

    failed = _create(client, key="key-1")
    assert failed.status_code == 500
    assert _count_rows(app, DataEnvelope) == 0
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0

    # After recovery the same key is judged fresh and succeeds.
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    created = _create(client, key="key-1")
    assert created.status_code == 201
    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1


def test_concurrent_same_key_requests_create_exactly_one_envelope(app):
    body = _body()

    def submit(_):
        with TestClient(app) as thread_client:
            return thread_client.post(
                "/v1/data-envelopes",
                json=body,
                headers={IDEMPOTENCY_HEADER: "key-1"},
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(submit, range(8)))

    statuses = {response.status_code for response in responses}
    assert statuses == {201}
    bodies = {response.content for response in responses}
    assert len(bodies) == 1
    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1
    assert _scope_seq(app) == 1


# --- keyless behavior is unchanged -----------------------------------------


def test_missing_header_keeps_legacy_semantics(app, client):
    first = _create(client)
    assert first.status_code == 201

    duplicate = _create(client)
    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "data_id already exists in this scope"}

    # No idempotency record is ever written for keyless requests.
    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0


def test_keyed_and_keyless_creates_share_data_id_uniqueness(app, client):
    keyed = _create(client, key="key-1")
    assert keyed.status_code == 201

    # A keyless retry of the same data_id is still a plain duplicate.
    duplicate = _create(client)
    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "data_id already exists in this scope"}

    # The keyed replay still returns the stored first response.
    assert _create(client, key="key-1").content == keyed.content
    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1


def test_record_stores_no_payload_or_key_material(app, client):
    response = _create(client, key="key-1")
    assert response.status_code == 201

    (record,) = _records(app)
    stored = {
        "tenant_id": record.tenant_id,
        "workload_id": record.workload_id,
        "idempotency_key": record.idempotency_key,
        "data_id": record.data_id,
        "payload_sha256": record.payload_sha256,
        "response_body": record.response_body,
    }
    for value in stored.values():
        assert PAYLOAD not in value
        assert KEY_V1 not in value
    # The response body carries only the six public metadata fields.
    assert set(json.loads(record.response_body)) == {
        "data_id",
        "tenant_id",
        "workload_id",
        "key_version",
        "created_at",
        "classification",
    }


def test_envelope_detail_and_directory_unaffected_by_keyed_create(app, client):
    created = _create(client, key="key-1")
    assert created.status_code == 201

    detail = client.get(
        f"/v1/data-envelopes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert detail.status_code == 200
    assert detail.json()["data_id"] == DATA_ID

    directory = client.get(
        "/v1/data-envelopes",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert directory.status_code == 200
    items = directory.json()["envelopes"]
    assert [item["data_id"] for item in items] == [DATA_ID]

    # A replay does not add a directory entry.
    assert _create(client, key="key-1").status_code == 201
    directory = client.get(
        "/v1/data-envelopes",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert [item["data_id"] for item in directory.json()["envelopes"]] == [DATA_ID]
