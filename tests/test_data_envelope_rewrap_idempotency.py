"""Tests for the optional Idempotency-Key header on the single-envelope
public rewrap entry point: ``POST
/v1/data-envelopes/{data_id}/rewrap``.

A keyed rewrap rotates the envelope's ``key_version``/``wrapped_key`` and
writes its idempotency record (plus the rewrap audit event) in one
transaction and replays the exact stored 200 afterwards; a missing key
keeps the legacy rewrap-on-every-call behavior. The record is a separate
table from envelope creation, rewrap batches and rewrap jobs, so the same
key across those operations never conflicts. It persists only the scope,
the key, the data_id, a digest of the request identity, the first
response body and the creation time — never a master key, data key,
ciphertext, iv, tag or wrapped key.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import (
    AuditEvent,
    DataEnvelope,
    DataEnvelopeRewrapIdempotencyRecord,
)
from proof_release.envelopes import b64url_encode

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

IDEMPOTENCY_HEADER = "Idempotency-Key"


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


KEYRING_V1_V2 = _keyring(2, {1: KEY_V1, 2: KEY_V2})
KEYRING_V2_ONLY = json.dumps({"current_version": 2, "keys": {"2": KEY_V2}})


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = app_module.create_app(f"sqlite:///{tmp_path}/idem-rewrap.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, data_id=DATA_ID, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": PAYLOAD,
        },
    )


def _rewrap(
    client,
    data_id=DATA_ID,
    *,
    tenant=TENANT,
    workload=WORKLOAD,
    key=None,
):
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else None
    return client.post(
        f"/v1/data-envelopes/{data_id}/rewrap",
        json={"tenant_id": tenant, "workload_id": workload},
        headers=headers,
    )


def _get(client, data_id=DATA_ID, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/data-envelopes/{data_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _count_rows(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


def _records(app):
    with app.state.session_factory() as session:
        return session.query(DataEnvelopeRewrapIdempotencyRecord).all()


def _envelope(app, data_id=DATA_ID, tenant=TENANT, workload=WORKLOAD):
    with app.state.session_factory() as session:
        return session.get(DataEnvelope, (tenant, workload, data_id))


def _audit_count(app, data_id=DATA_ID, tenant=TENANT, workload=WORKLOAD):
    with app.state.session_factory() as session:
        return (
            session.query(AuditEvent)
            .filter(
                AuditEvent.tenant_id == tenant,
                AuditEvent.workload_id == workload,
                AuditEvent.data_id == data_id,
            )
            .count()
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


def _raw_rewrap(app, *, key_raw=b"key-1", duplicate_key=False, data_id=DATA_ID):
    payload = json.dumps({"tenant_id": TENANT, "workload_id": WORKLOAD}).encode()
    path = f"/v1/data-envelopes/{data_id}/rewrap"
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"idempotency-key", key_raw),
    ]
    if duplicate_key:
        headers.append((b"idempotency-key", key_raw))
    return asyncio.run(_asgi_call(app, path, headers, payload))


# --- header validation (422 before any keyring or envelope IO) -------------


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
def test_invalid_idempotency_key_is_422_before_any_read_or_write(
    app, client, monkeypatch, raw_key
):
    assert _create(client).status_code == 201
    wrapped_before = _envelope(app).wrapped_key
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    status, raw = _raw_rewrap(app, key_raw=raw_key)
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 0
    # No keyring use and no envelope mutation.
    row = _envelope(app)
    assert row.key_version == 1
    assert row.wrapped_key == wrapped_before
    assert _audit_count(app) == 0


def test_duplicate_idempotency_header_is_422(app, client, monkeypatch):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    status, raw = _raw_rewrap(app, duplicate_key=True)
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 0
    assert _envelope(app).key_version == 1


def test_duplicate_idempotency_header_through_client_is_422(client):
    assert _create(client).status_code == 201
    response = client.post(
        f"/v1/data-envelopes/{DATA_ID}/rewrap",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers=[
            (IDEMPOTENCY_HEADER, "key-1"),
            (IDEMPOTENCY_HEADER, "key-1"),
        ],
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid idempotency key"}
    assert _count_rows(client.app, DataEnvelopeRewrapIdempotencyRecord) == 0


@pytest.mark.parametrize("length", [1, 64])
def test_boundary_length_keys_accepted(app, client, monkeypatch, length):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _rewrap(client, key="A" * length)
    assert response.status_code == 200, response.text
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1


def test_visible_punctuation_key_accepted(client):
    assert _create(client).status_code == 201
    response = _rewrap(client, key="~!@#$%^&*()_+-=[]{}|;:',.<>/?`")
    assert response.status_code == 200, response.text


def test_invalid_key_rejected_before_keyring_check(app, client, monkeypatch):
    # An illegal header must be a 422 even when the keyring is unusable:
    # header validation precedes the keyring check and writes nothing.
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    response = client.post(
        f"/v1/data-envelopes/{DATA_ID}/rewrap",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers={IDEMPOTENCY_HEADER: ""},
    )
    assert response.status_code == 422
    assert _raw_rewrap(app, key_raw=b"x" * 65)[0] == 422
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 0
    assert _envelope(app).key_version == 1


def test_invalid_key_is_not_consumed_and_later_valid_use_succeeds(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _rewrap(client, key="").status_code == 422
    assert _raw_rewrap(app, key_raw=b"bad key")[0] == 422
    accepted = _rewrap(client, key="key-1")
    assert accepted.status_code == 200
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1
    assert _envelope(app).key_version == 2


def test_body_validation_still_fails_with_valid_idempotency_key(client):
    assert _create(client).status_code == 201
    bad_bodies = [
        {"workload_id": WORKLOAD},
        {"tenant_id": TENANT},
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "   ", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": 7, "workload_id": WORKLOAD},
        {},
    ]
    for body in bad_bodies:
        response = client.post(
            f"/v1/data-envelopes/{DATA_ID}/rewrap",
            json=body,
            headers={IDEMPOTENCY_HEADER: "key-1"},
        )
        assert response.status_code == 422, body
    assert _count_rows(client.app, DataEnvelopeRewrapIdempotencyRecord) == 0


def test_blank_path_data_id_with_key_is_422_and_writes_nothing(client):
    response = client.post(
        "/v1/data-envelopes/%20/rewrap",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers={IDEMPOTENCY_HEADER: "key-1"},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid scope parameters"}
    assert _count_rows(client.app, DataEnvelopeRewrapIdempotencyRecord) == 0


# --- core replay semantics -------------------------------------------------


def test_first_keyed_rewrap_rotates_and_writes_record_atomically(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    response = _rewrap(client, key="key-1")
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

    row = _envelope(app)
    assert row.key_version == 2
    records = _records(app)
    assert len(records) == 1
    record = records[0]
    assert record.tenant_id == TENANT
    assert record.workload_id == WORKLOAD
    assert record.idempotency_key == "key-1"
    assert record.data_id == DATA_ID
    assert json.loads(record.response_body) == data
    # The stored body is the exact wire form of the first response.
    assert record.response_body.encode("utf-8") == response.content
    # Exactly one rewrap audit event, committed with the rotation.
    assert _audit_count(app) == 1


def test_replay_returns_first_response_byte_for_byte(app, client, monkeypatch):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _rewrap(client, key="key-1")
    assert first.status_code == 200

    replay = _rewrap(client, key="key-1")
    assert replay.status_code == 200
    assert replay.content == first.content
    assert replay.json() == first.json()

    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1
    # A replay rewraps nothing and appends no second audit event.
    assert _audit_count(app) == 1
    assert _envelope(app).key_version == 2


def test_replay_after_further_rotation_keeps_original_response(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(2, {1: KEY_V1, 2: KEY_V2})
    )
    first = _rewrap(client, key="key-1")
    assert first.status_code == 200
    assert first.json()["key_version"] == 2

    # The keyring advances to version 3 and the envelope is rotated
    # there out-of-band (keyless). The keyed replay must still return
    # the stored first 200, frozen at version 2.
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(3, {1: KEY_V1, 2: KEY_V2, 3: KEY_V3})
    )
    later = _rewrap(client)
    assert later.status_code == 200
    assert later.json()["key_version"] == 3

    replay = _rewrap(client, key="key-1")
    assert replay.status_code == 200
    assert replay.content == first.content
    assert replay.json()["key_version"] == 2
    # Replay changed nothing about the stored envelope.
    assert _envelope(app).key_version == 3
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1


def test_replay_does_not_touch_broken_keyring(app, client, monkeypatch):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _rewrap(client, key="key-1")
    assert first.status_code == 200

    # A replay succeeds verbatim even though the keyring is now unusable.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    replay = _rewrap(client, key="key-1")
    assert replay.status_code == 200
    assert replay.content == first.content


def test_replay_on_already_current_envelope_still_records(app, client):
    # The envelope is already at the sole/current version: the first
    # keyed call changes no material (no audit event) but still records
    # so the retry replays the exact response.
    assert _create(client).status_code == 201
    first = _rewrap(client, key="key-1")
    assert first.status_code == 200
    assert first.json()["key_version"] == 1
    assert _audit_count(app) == 0

    replay = _rewrap(client, key="key-1")
    assert replay.status_code == 200
    assert replay.content == first.content
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1
    assert _audit_count(app) == 0


# --- conflict: same key, different request identity -----------------------


def test_same_key_different_data_id_is_409_and_changes_nothing(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    assert _create(client, data_id="item-2").status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _rewrap(client, key="key-1")
    assert first.status_code == 200
    wrapped2_before = _envelope(app, data_id="item-2").wrapped_key

    conflict = _rewrap(client, data_id="item-2", key="key-1")
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "idempotency key conflict"}

    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1
    # The other envelope was never touched.
    assert _envelope(app, data_id="item-2").key_version == 1
    assert _envelope(app, data_id="item-2").wrapped_key == wrapped2_before
    # The original record still replays its own first response.
    assert _rewrap(client, key="key-1").content == first.content


def test_same_key_different_scope_is_409(app, client):
    assert _create(client).status_code == 201
    assert _create(client, tenant="tenant-b").status_code == 201
    first = _rewrap(client, key="key-1")
    assert first.status_code == 200

    conflict = _rewrap(client, tenant="tenant-b", key="key-1")
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "idempotency key conflict"}

    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1
    assert _envelope(app, tenant="tenant-b").key_version == 1
    assert _rewrap(client, key="key-1").content == first.content


def test_same_key_different_workload_is_409(app, client):
    assert _create(client).status_code == 201
    assert _create(client, workload="workload-2").status_code == 201
    first = _rewrap(client, key="key-1")
    assert first.status_code == 200

    conflict = _rewrap(client, workload="workload-2", key="key-1")
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "idempotency key conflict"}
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1
    assert _envelope(app, workload="workload-2").key_version == 1


def test_replay_with_non_ascii_identifiers_does_not_raise(app, client):
    # Scope/data identifiers may be non-ASCII; conflict detection must
    # not call a constant-time compare on non-ASCII strings (it uses the
    # ASCII-hex request fingerprint).
    unicode_id = "item-🔐"
    assert _create(client, data_id=unicode_id).status_code == 201
    first = _rewrap(client, data_id=unicode_id, key="key-1")
    assert first.status_code == 200
    replay = _rewrap(client, data_id=unicode_id, key="key-1")
    assert replay.status_code == 200
    assert replay.content == first.content

    # A different non-ASCII data_id under the same key is a clean 409.
    assert _create(client, data_id="other-🔑").status_code == 201
    conflict = _rewrap(client, data_id="other-🔑", key="key-1")
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "idempotency key conflict"}


# --- failure paths leave the key free and the envelope untouched ----------


def test_rewrap_unknown_data_id_with_key_is_404_and_key_stays_free(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    missing = _rewrap(client, data_id="missing", key="key-1")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "data envelope not found"}
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 0

    # The key is still free for the real envelope.
    accepted = _rewrap(client, key="key-1")
    assert accepted.status_code == 200
    assert accepted.json()["key_version"] == 2
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1


def test_broken_keyring_with_key_is_500_and_key_stays_free(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    before = _get(client).json()
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")

    failed = _rewrap(client, key="key-1")
    assert failed.status_code == 500
    assert failed.json() == {"detail": "encryption unavailable"}
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 0
    assert _envelope(app).key_version == 1

    # After recovery the same key judges fresh and succeeds.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    accepted = _rewrap(client, key="key-1")
    assert accepted.status_code == 200
    assert accepted.json()["key_version"] == 2
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1
    assert _get(client).json()["key_version"] == 2
    assert before["wrapped_key"] != _get(client).json()["wrapped_key"]


def test_missing_historical_key_with_key_is_500_and_key_stays_free(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    before = _get(client).json()
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)

    failed = _rewrap(client, key="key-1")
    assert failed.status_code == 500
    assert failed.json() == {"detail": "encryption unavailable"}
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 0
    assert _get(client).json() == before

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    accepted = _rewrap(client, key="key-1")
    assert accepted.status_code == 200
    assert accepted.json()["key_version"] == 2
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1


def test_rewrap_crypto_failure_with_key_is_500_and_key_stays_free(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    before = _get(client).json()
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    original = app_module.rewrap_data_key

    def boom(*args, **kwargs):
        raise RuntimeError("crypto backend exploded")

    monkeypatch.setattr(app_module, "rewrap_data_key", boom)
    failed = _rewrap(client, key="key-1")
    assert failed.status_code == 500
    assert failed.json() == {"detail": "encryption failed"}
    # Restore the crypto backend without touching the keyring env.
    monkeypatch.setattr(app_module, "rewrap_data_key", original)

    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 0
    assert _get(client).json() == before
    # The key is reusable after recovery.
    accepted = _rewrap(client, key="key-1")
    assert accepted.status_code == 200
    assert accepted.json()["key_version"] == 2
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1


# --- concurrency -----------------------------------------------------------


def test_concurrent_same_key_requests_rotate_exactly_once(app, client, monkeypatch):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}

    def submit(_):
        with TestClient(app) as thread_client:
            return thread_client.post(
                f"/v1/data-envelopes/{DATA_ID}/rewrap",
                json=body,
                headers={IDEMPOTENCY_HEADER: "key-1"},
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(submit, range(8)))

    statuses = {response.status_code for response in responses}
    assert statuses == {200}
    bodies = {response.content for response in responses}
    assert len(bodies) == 1
    assert all(response.json()["key_version"] == 2 for response in responses)
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1
    assert _envelope(app).key_version == 2
    # Exactly one rotation -> exactly one rewrap audit event.
    assert _audit_count(app) == 1


# --- keyless semantics and cross-operation isolation ----------------------


def test_missing_header_keeps_legacy_rewrap_every_call_semantics(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    first = _rewrap(client)
    assert first.status_code == 200
    assert first.json()["key_version"] == 2
    # A second keyless call still executes a rewrap judgement (a
    # current-version no-op, reporting fresh state) and records nothing.
    second = _rewrap(client)
    assert second.status_code == 200
    assert second.json()["key_version"] == 2
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 0
    assert _audit_count(app) == 1


def test_keyed_and_keyless_rewraps_do_not_record_for_keyless(
    app, client, monkeypatch
):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    keyed = _rewrap(client, key="key-1")
    assert keyed.status_code == 200
    # A keyless rewrap afterwards neither consumes nor creates a record.
    assert _rewrap(client).status_code == 200
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1
    # The keyed replay still returns the stored first response.
    assert _rewrap(client, key="key-1").content == keyed.content


def test_same_key_does_not_conflict_with_envelope_creation(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    # An idempotency-keyed envelope creation owns "shared-key" in the
    # creation table; the rewrap table is independent.
    created = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": DATA_ID,
            "payload": PAYLOAD,
        },
        headers={IDEMPOTENCY_HEADER: "shared-key"},
    )
    assert created.status_code == 201
    assert created.json()["key_version"] == 2

    rewrapped = _rewrap(client, key="shared-key")
    assert rewrapped.status_code == 200
    # The envelope was born current (v2); the rewrap is a no-op but
    # recorded independently.
    assert rewrapped.json()["key_version"] == 2
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1

    # And the creation key still replays its own 201.
    replay_create = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": DATA_ID,
            "payload": PAYLOAD,
        },
        headers={IDEMPOTENCY_HEADER: "shared-key"},
    )
    assert replay_create.status_code == 201
    assert replay_create.content == created.content


def test_same_key_does_not_conflict_with_rewrap_batch(app, client, monkeypatch):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    # The batch endpoint ignores Idempotency-Key entirely.
    batch = client.post(
        "/v1/rewrap-batches",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers={IDEMPOTENCY_HEADER: "shared-key"},
    )
    assert batch.status_code == 200

    rewrapped = _rewrap(client, key="shared-key")
    assert rewrapped.status_code == 200
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1


def test_same_key_does_not_conflict_with_rewrap_job(app, client, monkeypatch):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job = client.post(
        "/v1/rewrap-jobs",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers={IDEMPOTENCY_HEADER: "shared-key"},
    )
    assert job.status_code == 202

    rewrapped = _rewrap(client, key="shared-key")
    assert rewrapped.status_code == 200
    assert _count_rows(app, DataEnvelopeRewrapIdempotencyRecord) == 1
    # The job key independently replays its own 202.
    replay_job = client.post(
        "/v1/rewrap-jobs",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers={IDEMPOTENCY_HEADER: "shared-key"},
    )
    assert replay_job.status_code == 202
    assert replay_job.content == job.content


# --- persistence and material hygiene -------------------------------------


def test_record_persists_across_restart(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/restart-idem-rewrap.db"
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    assert _create(client1).status_code == 201
    app1.state.engine.dispose()

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    first = _rewrap(client2, key="key-1")
    assert first.status_code == 200
    assert first.json()["key_version"] == 2
    app2.state.engine.dispose()

    # A replay after a full restart still returns the stored first 200
    # without rewrapping, even with the historical key removed.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    app3 = app_module.create_app(url)
    client3 = TestClient(app3)
    replay = _rewrap(client3, key="key-1")
    assert replay.status_code == 200
    assert replay.content == first.content
    assert replay.json()["key_version"] == 2
    app3.state.engine.dispose()


def test_record_stores_no_key_or_envelope_material(app, client, monkeypatch):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _rewrap(client, key="key-1")
    assert response.status_code == 200

    (record,) = _records(app)
    stored_text = json.dumps(
        {
            "tenant_id": record.tenant_id,
            "workload_id": record.workload_id,
            "idempotency_key": record.idempotency_key,
            "data_id": record.data_id,
            "request_fingerprint": record.request_fingerprint,
            "response_body": record.response_body,
        }
    )
    for secret in (KEY_V1, KEY_V2, PAYLOAD):
        assert secret not in stored_text
    # The response body carries only the five public metadata fields.
    assert set(json.loads(record.response_body)) == {
        "data_id",
        "tenant_id",
        "workload_id",
        "key_version",
        "rotated_at",
    }
    # No wrapped_key/iv/tag/ciphertext column exists on the record model.
    assert not {
        "wrapped_key",
        "iv",
        "tag",
        "ciphertext",
        "payload_sha256",
    } & set(record.__table__.columns.keys())


def test_detail_directory_and_rotation_status_unaffected(app, client, monkeypatch):
    assert _create(client).status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _rewrap(client, key="key-1").status_code == 200

    detail = _get(client)
    assert detail.status_code == 200
    assert detail.json()["key_version"] == 2

    directory = client.get(
        "/v1/data-envelopes",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert directory.status_code == 200
    assert [item["data_id"] for item in directory.json()["envelopes"]] == [DATA_ID]

    status = client.get(
        "/v1/key-rotation/status",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert status.status_code == 200

    # A replay adds neither a directory entry nor anything else.
    assert _rewrap(client, key="key-1").status_code == 200
    directory = client.get(
        "/v1/data-envelopes",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert [item["data_id"] for item in directory.json()["envelopes"]] == [DATA_ID]
