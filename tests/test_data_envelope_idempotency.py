"""Tests for the optional Idempotency-Key header on envelope creation
(POST /v1/data-envelopes).

A keyed creation writes one envelope and one idempotency record per
(tenant, workload, key) in a single transaction and replays the exact
stored 201 afterwards; a missing key keeps the legacy
one-envelope-per-request behavior.
"""

from __future__ import annotations

import asyncio
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
PAYLOAD = "idempotent-secret-payload 🔐"

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
    application = app_module.create_app(f"sqlite:///{tmp_path}/idem.db")
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
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else None
    return client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": payload,
        },
        headers=headers,
    )


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


def _valid_body(**overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "payload": PAYLOAD,
    }
    body.update(overrides)
    return body


def _count_rows(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


def _scope_seq(app, tenant=TENANT, workload=WORKLOAD):
    with app.state.session_factory() as session:
        return session.get(DataEnvelopeCommitCounter, (tenant, workload))


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
    status, raw = _raw_create(app, _valid_body(), key_raw=raw_key)
    assert status == 422, raw
    assert json.loads(raw)["detail"] == "invalid idempotency key"
    assert _count_rows(app, DataEnvelope) == 0
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0


def test_duplicate_idempotency_header_is_422(app):
    status, raw = _raw_create(app, _valid_body(), duplicate_key=True)
    assert status == 422, raw
    assert json.loads(raw)["detail"] == "invalid idempotency key"
    assert _count_rows(app, DataEnvelope) == 0
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0


def test_duplicate_idempotency_header_through_client_is_422(client):
    response = client.post(
        "/v1/data-envelopes",
        json=_valid_body(),
        headers=[
            (IDEMPOTENCY_HEADER, "key-1"),
            (IDEMPOTENCY_HEADER, "key-1"),
        ],
    )
    assert response.status_code == 422
    assert _count_rows(client.app, DataEnvelope) == 0


def test_empty_idempotency_header_through_client_is_422(client):
    response = _create(client, key="")
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid idempotency key"
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
    assert _raw_create(app, _valid_body(), key_raw=b"bad key")[0] == 422
    accepted = _create(client, key="key-1")
    assert accepted.status_code == 201
    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1


def test_body_validation_still_fails_with_valid_idempotency_key(client):
    bad_bodies = [
        {"workload_id": WORKLOAD, "data_id": DATA_ID, "payload": PAYLOAD},
        {"tenant_id": "", "workload_id": WORKLOAD, "data_id": DATA_ID, "payload": PAYLOAD},
        {"tenant_id": "  ", "workload_id": WORKLOAD, "data_id": DATA_ID, "payload": PAYLOAD},
        {"tenant_id": 7, "workload_id": WORKLOAD, "data_id": DATA_ID, "payload": PAYLOAD},
        _valid_body(workload_id=""),
        _valid_body(data_id=""),
        _valid_body(data_id="   "),
        _valid_body(payload=""),
        _valid_body(payload=42),
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
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    assert _create(client, key="").status_code == 422
    assert _raw_create(app, _valid_body(), key_raw=b"x" * 65)[0] == 422
    assert _count_rows(app, DataEnvelope) == 0
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0


# --- core replay semantics -------------------------------------------------


def test_first_keyed_create_persists_envelope_and_record(app, client):
    response = _create(client, key="env-key-1")
    assert response.status_code == 201
    data = response.json()
    assert list(data) == [
        "data_id",
        "tenant_id",
        "workload_id",
        "key_version",
        "created_at",
    ]
    assert data["data_id"] == DATA_ID
    assert data["key_version"] == 1
    assert _count_rows(app, DataEnvelope) == 1
    with app.state.session_factory() as session:
        record = session.query(DataEnvelopeIdempotencyRecord).one()
        assert record.tenant_id == TENANT
        assert record.workload_id == WORKLOAD
        assert record.idempotency_key == "env-key-1"
        assert record.data_id == DATA_ID
        assert len(record.payload_sha256) == 64
        assert record.response_body.encode() == response.content


def test_replay_returns_saved_201_verbatim_and_creates_nothing(app, client):
    first = _create(client, key="env-key-1")
    assert first.status_code == 201

    second = _create(client, key="env-key-1")
    third = _create(client, key="env-key-1")
    assert second.status_code == 201 and third.status_code == 201
    # Byte-for-byte identical: same data_id, key_version and created_at.
    assert second.content == first.content
    assert third.content == first.content
    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1
    # The directory commit sequence advanced exactly once (for the first
    # creation); replays never move it.
    assert _scope_seq(app).last_seq == 1


def test_replay_never_re_encrypts(app, client, monkeypatch):
    first = _create(client, key="env-key-1")
    assert first.status_code == 201

    def boom(*args, **kwargs):
        raise AssertionError("replay must not encrypt")

    monkeypatch.setattr(app_module, "encrypt_payload", boom)
    replay = _create(client, key="env-key-1")
    assert replay.status_code == 201
    assert replay.content == first.content


def test_replay_after_key_rotation_keeps_original_key_version(
    app, client, monkeypatch
):
    first = _create(client, key="env-key-1")
    assert first.status_code == 201
    assert first.json()["key_version"] == 1

    # Rotate: current version becomes 2. A replay still returns the
    # original body and the stored envelope is untouched.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    replay = _create(client, key="env-key-1")
    assert replay.status_code == 201
    assert replay.content == first.content
    assert replay.json()["key_version"] == 1
    with app.state.session_factory() as session:
        envelope = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        assert envelope.key_version == 1
        assert envelope.commit_seq == 1

    # A different key creates a new envelope under the new version.
    other = _create(client, key="env-key-2", data_id="item-2")
    assert other.status_code == 201
    assert other.json()["key_version"] == 2


def test_same_key_different_payload_is_409_and_changes_nothing(app, client):
    first = _create(client, key="dup")
    assert first.status_code == 201

    conflict = _create(client, key="dup", payload="other-payload")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency key reused with a different request"
    # Original envelope and saved response are untouched; exact replay works.
    assert _create(client, key="dup").content == first.content
    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1


def test_same_key_different_data_id_is_409_and_creates_nothing(app, client):
    first = _create(client, key="dup")
    assert first.status_code == 201

    conflict = _create(client, key="dup", data_id="item-2")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency key reused with a different request"
    assert _create(client, key="dup").content == first.content
    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1


def test_409_takes_precedence_over_later_keyring_failure(
    app, client, monkeypatch
):
    assert _create(client, key="dup").status_code == 201
    # Keyring later becomes unusable; a mismatched replay is still a 409
    # (a stored-record judgement never re-checks the keyring)...
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    conflict = _create(client, key="dup", payload="changed")
    assert conflict.status_code == 409
    # ...and an exact replay is still the verbatim 201, never a 500.
    replay = _create(client, key="dup")
    assert replay.status_code == 201


# --- scope isolation -------------------------------------------------------


def test_same_key_in_other_tenant_or_workload_is_independent(app, client):
    r1 = _create(client, key="shared-key")
    r2 = _create(client, key="shared-key", tenant="tenant-b")
    r3 = _create(client, key="shared-key", workload="workload-9")
    assert {r.status_code for r in (r1, r2, r3)} == {201}
    assert _count_rows(app, DataEnvelope) == 3
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 3
    # Replay in each scope resolves to that scope's own record.
    assert _create(client, key="shared-key").content == r1.content
    assert _create(client, key="shared-key", tenant="tenant-b").content == r2.content
    assert (
        _create(client, key="shared-key", workload="workload-9").content
        == r3.content
    )
    # A payload that differs only across a scope boundary is no conflict.
    assert (
        _create(client, key="shared-key", tenant="tenant-b", payload="changed").status_code
        == 409
    )
    assert _create(client, key="shared-key").status_code == 201


# --- interaction with the keyless path -------------------------------------


def test_missing_key_keeps_legacy_duplicate_data_id_semantics(client):
    assert _create(client).status_code == 201
    duplicate = _create(client)
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "data_id already exists in this scope"
    assert _count_rows(client.app, DataEnvelopeIdempotencyRecord) == 0


def test_keyed_create_with_data_id_taken_keylessly_is_409_and_frees_key(
    app, client
):
    assert _create(client).status_code == 201  # keyless creation
    conflict = _create(client, key="env-key-1")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "data_id already exists in this scope"
    # No idempotency record was written: the same key is still usable for
    # a fresh data_id, and then replays that creation.
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0
    created = _create(client, key="env-key-1", data_id="item-2")
    assert created.status_code == 201
    assert _create(client, key="env-key-1", data_id="item-2").content == created.content
    assert _count_rows(app, DataEnvelope) == 2
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1


def test_keyless_create_after_keyed_create_same_data_id_is_409(client):
    assert _create(client, key="env-key-1").status_code == 201
    duplicate = _create(client)
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "data_id already exists in this scope"
    assert _count_rows(client.app, DataEnvelope) == 1


# --- failure rollback ------------------------------------------------------


def test_invalid_keyring_with_valid_key_returns_500_and_rolls_back(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    response = _create(client, key="env-key-1")
    assert response.status_code == 500
    assert _count_rows(app, DataEnvelope) == 0
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0

    # The failed attempt consumed no key: once the keyring is fixed the
    # same request creates normally.
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    accepted = _create(client, key="env-key-1")
    assert accepted.status_code == 201
    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1


def test_encryption_failure_returns_500_and_leaves_no_record(
    app, client, monkeypatch
):
    real_encrypt = app_module.encrypt_payload

    def boom(*args, **kwargs):
        raise RuntimeError("crypto backend exploded")

    monkeypatch.setattr(app_module, "encrypt_payload", boom)
    response = _create(client, key="env-key-1")
    assert response.status_code == 500
    assert PAYLOAD not in response.text
    assert _count_rows(app, DataEnvelope) == 0
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0

    # Recovery: the same key re-judges and succeeds.
    monkeypatch.setattr(app_module, "encrypt_payload", real_encrypt)
    assert _create(client, key="env-key-1").status_code == 201


def test_envelope_write_failure_rolls_back_idempotency_record(app, client):
    from sqlalchemy import text

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE data_envelopes"))
    response = _create(client, key="env-key-1")
    assert response.status_code == 500
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0


def test_idempotency_table_write_failure_returns_500_with_no_envelope(
    app, client
):
    from sqlalchemy import text

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE data_envelope_idempotency_records"))
    response = _create(client, key="env-key-1")
    assert response.status_code == 500
    assert _count_rows(app, DataEnvelope) == 0
    # A keyless request is unaffected by the missing table.
    assert _create(client).status_code == 201


# --- concurrency -----------------------------------------------------------


def test_concurrent_same_key_creations_settle_as_one_envelope(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/concurrent-idem.db"
    application = app_module.create_app(url)
    client = TestClient(application)
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: _create(client, key="race-key"), range(8)))
    assert all(r.status_code == 201 for r in responses), [
        r.status_code for r in responses
    ]
    bodies = {r.content for r in responses}
    assert len(bodies) == 1
    with application.state.session_factory() as session:
        assert session.query(DataEnvelope).count() == 1
        assert session.query(DataEnvelopeIdempotencyRecord).count() == 1
        assert session.get(DataEnvelopeCommitCounter, (TENANT, WORKLOAD)).last_seq == 1
    application.state.engine.dispose()


# --- restart persistence ---------------------------------------------------


def test_idempotency_record_survives_restart_and_replays_original(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart-idem.db"
    app1 = app_module.create_app(url)
    accepted = _create(TestClient(app1), key="durable-key")
    assert accepted.status_code == 201
    app1.state.engine.dispose()

    # A fresh process (even with the keyring later altered) still has the
    # record and answers replays with the original 201.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    replay = _create(client2, key="durable-key")
    assert replay.status_code == 201
    assert replay.content == accepted.content
    conflict = _create(client2, key="durable-key", payload="changed")
    assert conflict.status_code == 409
    with app2.state.session_factory() as session:
        assert session.query(DataEnvelope).count() == 1
        assert session.query(DataEnvelopeIdempotencyRecord).count() == 1
    app2.state.engine.dispose()


# --- read paths and secrecy ------------------------------------------------


def test_keyed_creation_and_replay_leave_read_paths_consistent(app, client):
    created = _create(client, key="env-key-1")
    assert created.status_code == 201
    assert _create(client, key="env-key-1").status_code == 201

    detail = client.get(
        f"/v1/data-envelopes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert detail.status_code == 200
    assert detail.json()["created_at"] == created.json()["created_at"]

    listing = client.get(
        "/v1/data-envelopes",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert listing.status_code == 200
    assert [e["data_id"] for e in listing.json()["envelopes"]] == [DATA_ID]


def test_keyed_path_never_exposes_material(app, client):
    created = _create(client, key="env-key-1")
    replay = _create(client, key="env-key-1")
    for response in (created, replay):
        assert response.status_code == 201
        for secret in (
            PAYLOAD,
            KEY_V1,
            "wrapped_key",
            "ciphertext",
            "payload",
        ):
            assert secret not in response.text
    with app.state.session_factory() as session:
        record = session.query(DataEnvelopeIdempotencyRecord).one()
        columns = {
            c.name: getattr(record, c.name) for c in record.__table__.columns
        }
        payload_bytes = PAYLOAD.encode("utf-8")
        for name, value in columns.items():
            blob = str(value).encode("utf-8")
            assert payload_bytes not in blob, f"plaintext payload in column {name}"
            assert KEY_V1 not in str(value), f"master key in column {name}"
        # Only the irreversible digest of the payload is stored.
        import hashlib

        assert (
            columns["payload_sha256"]
            == hashlib.sha256(payload_bytes).hexdigest()
        )
