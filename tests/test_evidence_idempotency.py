"""Tests for the optional Idempotency-Key header on POST /v1/evidence.

A keyed submission commits the challenge consumption, the evidence
row, its proof-received event and the idempotency record in one
transaction and replays the exact stored 201 afterwards; a missing
key keeps the legacy one-evidence-per-challenge behavior. The record
persists only the scope, the key, a fingerprint over non-sensitive
request identity (challenge id, format and the nonce/evidence
digests), the first response body and the creation time — never the
plaintext nonce or evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import (
    Challenge,
    Evidence,
    EvidenceIdempotencyRecord,
    ProofEventCommitCounter,
    ProofLifecycleEvent,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
IDEMPOTENCY_HEADER = "Idempotency-Key"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/idem-evidence.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create_challenge(client, *, tenant=TENANT, workload=WORKLOAD, ttl_seconds=None):
    body = {"tenant_id": tenant, "workload_id": workload}
    if ttl_seconds is not None:
        body["ttl_seconds"] = ttl_seconds
    response = client.post("/v1/challenges", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _submit_body(challenge, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "challenge_id": challenge["challenge_id"],
        "nonce": challenge["nonce"],
        "evidence_format": "tpm-quote",
        "evidence": "cXVvdGU=",
    }
    body.update(overrides)
    return body


def _submit(client, challenge, *, key=None, **overrides):
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else None
    return client.post(
        "/v1/evidence", json=_submit_body(challenge, **overrides), headers=headers
    )


def _count_rows(application, model) -> int:
    with application.state.session_factory() as session:
        return session.query(model).count()


def _records(application):
    with application.state.session_factory() as session:
        return session.query(EvidenceIdempotencyRecord).all()


async def _asgi_call(application, path, headers, body: bytes):
    """Invoke the ASGI app directly with arbitrary raw header bytes.

    The HTTP test client refuses some header values (control characters,
    non-ASCII bytes) before they reach the app; driving the ASGI app lets
    the service's own header validation see them.
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


def _raw_submit(application, body_obj, *, key_raw=b"key-1", duplicate_key=False):
    payload = json.dumps(body_obj).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"idempotency-key", key_raw),
    ]
    if duplicate_key:
        headers.append((b"idempotency-key", key_raw))
    return asyncio.run(_asgi_call(application, "/v1/evidence", headers, payload))


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
def test_invalid_idempotency_key_is_422_before_any_write(app, client, raw_key):
    challenge = _create_challenge(client)
    status, raw = _raw_submit(app, _submit_body(challenge), key_raw=raw_key)
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    assert _count_rows(app, Evidence) == 0
    assert _count_rows(app, EvidenceIdempotencyRecord) == 0
    with app.state.session_factory() as session:
        assert session.get(Challenge, challenge["challenge_id"]).status == "pending"


def test_duplicate_idempotency_header_is_422(app, client):
    challenge = _create_challenge(client)
    status, raw = _raw_submit(
        app, _submit_body(challenge), duplicate_key=True
    )
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    assert _count_rows(app, Evidence) == 0
    assert _count_rows(app, EvidenceIdempotencyRecord) == 0


def test_duplicate_idempotency_header_through_client_is_422(client, app):
    challenge = _create_challenge(client)
    response = client.post(
        "/v1/evidence",
        json=_submit_body(challenge),
        headers=[
            (IDEMPOTENCY_HEADER, "key-1"),
            (IDEMPOTENCY_HEADER, "key-1"),
        ],
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid idempotency key"}
    assert _count_rows(app, EvidenceIdempotencyRecord) == 0


@pytest.mark.parametrize("length", [1, 64])
def test_boundary_length_keys_accepted(app, client, length):
    challenge = _create_challenge(client)
    response = _submit(client, challenge, key="A" * length)
    assert response.status_code == 201, response.text
    assert _count_rows(app, EvidenceIdempotencyRecord) == 1


def test_visible_punctuation_key_accepted(client):
    challenge = _create_challenge(client)
    response = _submit(client, challenge, key="~!@#$%^&*()_+-=[]{}|;:',.<>/?`")
    assert response.status_code == 201, response.text


def test_invalid_key_is_not_consumed_and_later_valid_use_succeeds(app, client):
    challenge = _create_challenge(client)
    assert _raw_submit(app, _submit_body(challenge), key_raw=b"")[0] == 422
    assert _raw_submit(app, _submit_body(challenge), key_raw=b"bad key")[0] == 422
    accepted = _submit(client, challenge, key="key-1")
    assert accepted.status_code == 201
    assert _count_rows(app, Evidence) == 1
    assert _count_rows(app, EvidenceIdempotencyRecord) == 1


def test_body_validation_still_fails_with_valid_idempotency_key(app, client):
    challenge = _create_challenge(client)
    bad_bodies = [
        _submit_body(challenge, tenant_id=""),
        _submit_body(challenge, workload_id="   "),
        _submit_body(challenge, challenge_id=""),
        _submit_body(challenge, nonce=""),
        _submit_body(challenge, nonce="not base64!!!"),
        _submit_body(challenge, evidence_format=""),
        _submit_body(challenge, evidence=""),
        _submit_body(challenge, evidence=42),
    ]
    for bad_body in bad_bodies:
        response = client.post(
            "/v1/evidence",
            json=bad_body,
            headers={IDEMPOTENCY_HEADER: "key-1"},
        )
        assert response.status_code == 422, bad_body
    assert _count_rows(app, Evidence) == 0
    assert _count_rows(app, EvidenceIdempotencyRecord) == 0
    # The key was never consumed: the same key now succeeds.
    accepted = _submit(client, challenge, key="key-1")
    assert accepted.status_code == 201
    assert _count_rows(app, EvidenceIdempotencyRecord) == 1


# --- core replay semantics -------------------------------------------------


def test_first_keyed_submission_writes_all_state_atomically(app, client):
    challenge = _create_challenge(client)

    response = _submit(client, challenge, key="key-1")

    assert response.status_code == 201
    data = response.json()
    assert set(data) == {"evidence_id", "challenge_id", "status", "received_at"}
    assert data["challenge_id"] == challenge["challenge_id"]
    assert data["status"] == "received"
    received_at = datetime.fromisoformat(data["received_at"])
    assert received_at.utcoffset() == timedelta(0)

    with app.state.session_factory() as session:
        stored_challenge = session.get(Challenge, challenge["challenge_id"])
        assert stored_challenge.status == "consumed"
        evidence = session.get(Evidence, data["evidence_id"])
        assert evidence is not None
        events = (
            session.query(ProofLifecycleEvent)
            .filter_by(evidence_id=data["evidence_id"])
            .all()
        )
        assert len(events) == 1
        assert events[0].event_type == "proof-received"
        counter = session.get(ProofEventCommitCounter, (TENANT, WORKLOAD))
        assert counter.last_seq == 1

    records = _records(app)
    assert len(records) == 1
    record = records[0]
    assert record.tenant_id == TENANT
    assert record.workload_id == WORKLOAD
    assert record.idempotency_key == "key-1"
    assert len(record.request_fingerprint) == 64
    assert json.loads(record.response_body) == data
    # The stored body is the exact wire form of the first response.
    assert record.response_body.encode("utf-8") == response.content


def test_replay_returns_first_response_byte_for_byte(app, client):
    challenge = _create_challenge(client)
    first = _submit(client, challenge, key="key-1")
    assert first.status_code == 201

    replay = _submit(client, challenge, key="key-1")
    assert replay.status_code == 201
    assert replay.content == first.content
    assert replay.json()["received_at"] == first.json()["received_at"]
    assert replay.json()["evidence_id"] == first.json()["evidence_id"]

    assert _count_rows(app, Evidence) == 1
    assert _count_rows(app, ProofLifecycleEvent) == 1
    assert _count_rows(app, EvidenceIdempotencyRecord) == 1


def test_replay_after_challenge_expired_still_returns_first_201(app, client):
    challenge = _create_challenge(client, ttl_seconds=30)
    first = _submit(client, challenge, key="key-1")
    assert first.status_code == 201

    with app.state.session_factory() as session:
        stored = session.get(Challenge, challenge["challenge_id"])
        stored.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    replay = _submit(client, challenge, key="key-1")
    assert replay.status_code == 201
    assert replay.content == first.content
    assert _count_rows(app, Evidence) == 1


def test_repeated_replays_all_identical(app, client):
    challenge = _create_challenge(client)
    first = _submit(client, challenge, key="key-1")
    for _ in range(5):
        assert _submit(client, challenge, key="key-1").content == first.content
    assert _count_rows(app, Evidence) == 1
    assert _count_rows(app, EvidenceIdempotencyRecord) == 1


@pytest.mark.parametrize(
    "field",
    ["challenge_id", "nonce", "evidence_format", "evidence"],
)
def test_same_key_different_field_is_422_and_changes_nothing(app, client, field):
    challenge = _create_challenge(client)
    first = _submit(client, challenge, key="key-1")
    assert first.status_code == 201

    other_challenge = _create_challenge(client)
    overrides = {
        "challenge_id": other_challenge["challenge_id"],
        "nonce": other_challenge["nonce"],
        "evidence_format": "snake-oil",
        "evidence": "b3RoZXI=",
    }
    # Override only the one field under test; keep every other field equal.
    overrides = {field: overrides[field]}
    conflict = _submit(client, challenge, key="key-1", **overrides)
    assert conflict.status_code == 422
    assert conflict.json() == {
        "detail": "idempotency key reused with a different request"
    }

    # The conflict consumed no challenge, evidence, event or record.
    assert _count_rows(app, Evidence) == 1
    assert _count_rows(app, ProofLifecycleEvent) == 1
    assert _count_rows(app, EvidenceIdempotencyRecord) == 1
    with app.state.session_factory() as session:
        other = session.get(Challenge, other_challenge["challenge_id"])
        if field == "challenge_id":
            # The request named an unconsumed other challenge: nothing spent.
            assert other.status == "pending"
    # The original record still replays the original response.
    assert _submit(client, challenge, key="key-1").content == first.content


def test_same_key_unknown_challenge_on_replay_is_still_422(app, client):
    # The stored record is checked before challenge judgement, so a
    # replay-time request whose challenge id now points nowhere is the
    # same-key mismatch (422), never a 404.
    challenge = _create_challenge(client)
    first = _submit(client, challenge, key="key-1")
    conflict = _submit(
        client,
        challenge,
        key="key-1",
        challenge_id="00000000-0000-0000-0000-000000000000",
    )
    assert conflict.status_code == 422
    assert conflict.json() == {
        "detail": "idempotency key reused with a different request"
    }
    assert _submit(client, challenge, key="key-1").content == first.content


def test_same_key_in_other_scopes_is_independent(app, client):
    challenge_a = _create_challenge(client)
    challenge_b = _create_challenge(client, tenant="tenant-b")
    challenge_w = _create_challenge(client, workload="workload-2")

    first = _submit(client, challenge_a, key="shared-key")
    other_tenant = _submit(
        client, challenge_b, key="shared-key", tenant_id="tenant-b"
    )
    other_workload = _submit(
        client, challenge_w, key="shared-key", workload_id="workload-2"
    )
    assert first.status_code == 201
    assert other_tenant.status_code == 201
    assert other_workload.status_code == 201

    # Three independent scopes, three records and three evidences.
    assert _count_rows(app, Evidence) == 3
    assert _count_rows(app, EvidenceIdempotencyRecord) == 3

    # Within each scope the key replays its own first response only.
    assert _submit(client, challenge_a, key="shared-key").content == first.content
    assert (
        _submit(
            client, challenge_b, key="shared-key", tenant_id="tenant-b"
        ).content
        == other_tenant.content
    )
    assert first.content != other_tenant.content
    assert first.content != other_workload.content


# --- judgement failures never consume the key ------------------------------


def test_unknown_challenge_with_key_is_404_and_key_stays_free(app, client):
    challenge = _create_challenge(client)
    response = _submit(
        client,
        challenge,
        key="key-1",
        challenge_id="00000000-0000-0000-0000-000000000000",
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "challenge not found"}
    assert _count_rows(app, EvidenceIdempotencyRecord) == 0

    # The key was never consumed: the same key now succeeds.
    assert _submit(client, challenge, key="key-1").status_code == 201


def test_cross_tenant_challenge_with_key_is_404(app, client):
    challenge = _create_challenge(client, tenant="tenant-b")
    response = _submit(client, challenge, key="key-1", tenant_id=TENANT)
    assert response.status_code == 404
    assert response.json() == {"detail": "challenge not found"}
    assert _count_rows(app, EvidenceIdempotencyRecord) == 0


def test_wrong_nonce_with_key_is_401_and_key_stays_free(app, client):
    challenge = _create_challenge(client)
    other = _create_challenge(client)
    response = _submit(client, challenge, key="key-1", nonce=other["nonce"])
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid nonce"}
    assert _count_rows(app, EvidenceIdempotencyRecord) == 0
    assert _submit(client, challenge, key="key-1").status_code == 201


def test_consumed_challenge_with_key_is_409_and_writes_no_record(app, client):
    challenge = _create_challenge(client)
    assert _submit(client, challenge).status_code == 201

    keyed = _submit(client, challenge, key="key-1")
    assert keyed.status_code == 409
    assert keyed.json() == {"detail": "challenge already consumed"}
    assert _count_rows(app, Evidence) == 1
    assert _count_rows(app, EvidenceIdempotencyRecord) == 0
    # A later replay still judges normally: the key holds no stored result.
    assert _submit(client, challenge, key="key-1").status_code == 409


def test_expired_challenge_with_key_is_410_and_key_stays_free(app, client):
    challenge = _create_challenge(client, ttl_seconds=30)
    with app.state.session_factory() as session:
        stored = session.get(Challenge, challenge["challenge_id"])
        stored.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = _submit(client, challenge, key="key-1")
    assert response.status_code == 410
    assert response.json() == {"detail": "challenge expired"}
    assert _count_rows(app, EvidenceIdempotencyRecord) == 0

    # A fresh challenge under the same key (a different challenge_id, but
    # no record was ever stored) succeeds normally.
    fresh = _create_challenge(client)
    assert _submit(client, fresh, key="key-1").status_code == 201


# --- concurrency -----------------------------------------------------------


def test_concurrent_same_key_same_request_create_exactly_one_evidence(app):
    client = TestClient(app)
    challenge = _create_challenge(client)
    body = _submit_body(challenge)

    def submit(_):
        with TestClient(app) as thread_client:
            return thread_client.post(
                "/v1/evidence",
                json=body,
                headers={IDEMPOTENCY_HEADER: "key-1"},
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(submit, range(16)))

    assert {r.status_code for r in responses} == {201}
    bodies = {r.content for r in responses}
    assert len(bodies) == 1
    evidence_ids = {r.json()["evidence_id"] for r in responses}
    assert len(evidence_ids) == 1
    assert _count_rows(app, Evidence) == 1
    assert _count_rows(app, ProofLifecycleEvent) == 1
    assert _count_rows(app, EvidenceIdempotencyRecord) == 1
    with app.state.session_factory() as session:
        counter = session.get(ProofEventCommitCounter, (TENANT, WORKLOAD))
        assert counter.last_seq == 1


def test_concurrent_different_fields_same_key_one_201_rest_422(app):
    client = TestClient(app)
    challenge = _create_challenge(client)

    def submit(index):
        with TestClient(app) as thread_client:
            response = thread_client.post(
                "/v1/evidence",
                json=_submit_body(challenge, evidence=f"cXVvdGU{index:02d}="),
                headers={IDEMPOTENCY_HEADER: "race-key"},
            )
            return index, response

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(8)))

    successes = [(i, r) for i, r in results if r.status_code == 201]
    conflicts = [r for _, r in results if r.status_code == 422]
    assert len(successes) == 1
    winner_index, winner_response = successes[0]
    assert len(conflicts) == 7
    assert all(
        r.json() == {"detail": "idempotency key reused with a different request"}
        for r in conflicts
    )
    assert _count_rows(app, Evidence) == 1
    assert _count_rows(app, ProofLifecycleEvent) == 1
    assert _count_rows(app, EvidenceIdempotencyRecord) == 1
    # The single stored record carries the winning response body, and the
    # winning request replays its own original bytes.
    (record,) = _records(app)
    assert json.loads(record.response_body) == winner_response.json()
    replay = _submit(
        client, challenge, key="race-key", evidence=f"cXVvdGU{winner_index:02d}="
    )
    assert replay.status_code == 201
    assert replay.content == winner_response.content
    # A losing field still gets the stable 422.
    loser_index = (winner_index + 1) % 8
    assert (
        _submit(
            client,
            challenge,
            key="race-key",
            evidence=f"cXVvdGU{loser_index:02d}=",
        ).status_code
        == 422
    )


def test_concurrent_different_keys_same_challenge_one_wins_rest_409(app):
    client = TestClient(app)
    challenge = _create_challenge(client)

    def submit(index):
        with TestClient(app) as thread_client:
            return thread_client.post(
                "/v1/evidence",
                json=_submit_body(challenge),
                headers={IDEMPOTENCY_HEADER: f"key-{index}"},
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(submit, range(8)))

    assert [r.status_code for r in responses].count(201) == 1
    assert [r.status_code for r in responses].count(409) == 7
    assert all(
        r.json() == {"detail": "challenge already consumed"}
        for r in responses
        if r.status_code == 409
    )
    assert _count_rows(app, Evidence) == 1
    assert _count_rows(app, ProofLifecycleEvent) == 1
    # Only the winning key stored a record; the winner replays its 201 and
    # every losing key keeps judging the now-consumed challenge as 409.
    assert _count_rows(app, EvidenceIdempotencyRecord) == 1
    (record,) = _records(app)
    winner_key = record.idempotency_key
    assert (
        _submit(client, challenge, key=winner_key).content
        == next(r.content for r in responses if r.status_code == 201)
    )
    for index in range(8):
        key = f"key-{index}"
        if key != winner_key:
            assert _submit(client, challenge, key=key).status_code == 409
    # The losing keys never stored anything.
    assert _count_rows(app, EvidenceIdempotencyRecord) == 1


# --- keyless behavior is unchanged -----------------------------------------


def test_missing_header_keeps_legacy_behavior(app, client):
    challenge = _create_challenge(client)
    first = _submit(client, challenge)
    second = _submit(client, challenge)
    assert first.status_code == 201
    assert second.status_code == 409
    assert _count_rows(app, Evidence) == 1
    assert _count_rows(app, EvidenceIdempotencyRecord) == 0


def test_keyless_submission_then_keyed_is_409_without_record(app, client):
    challenge = _create_challenge(client)
    keyless = _submit(client, challenge)
    assert keyless.status_code == 201

    keyed = _submit(client, challenge, key="key-1")
    assert keyed.status_code == 409
    assert keyed.json() == {"detail": "challenge already consumed"}
    assert _count_rows(app, Evidence) == 1
    assert _count_rows(app, EvidenceIdempotencyRecord) == 0


def test_keyed_submission_then_keyless_is_409(app, client):
    challenge = _create_challenge(client)
    assert _submit(client, challenge, key="key-1").status_code == 201
    assert _submit(client, challenge).status_code == 409
    assert _count_rows(app, EvidenceIdempotencyRecord) == 1


def test_keyless_and_keyed_use_independent_challenges(app, client):
    first_challenge = _create_challenge(client)
    second_challenge = _create_challenge(client)
    assert _submit(client, first_challenge).status_code == 201
    keyed = _submit(client, second_challenge, key="key-1")
    assert keyed.status_code == 201
    assert _count_rows(app, Evidence) == 2
    assert _count_rows(app, EvidenceIdempotencyRecord) == 1


# --- storage failures and atomicity ----------------------------------------


def test_idempotency_table_write_failure_returns_500_with_no_state(
    app, client
):
    from sqlalchemy import text

    challenge = _create_challenge(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE evidence_idempotency_records"))

    response = _submit(client, challenge, key="key-1")
    assert response.status_code == 500
    assert response.json() == {"detail": "evidence submission failed"}

    # No half state: the challenge is unconsumed and nothing was written.
    assert _count_rows(app, Evidence) == 0
    assert _count_rows(app, ProofLifecycleEvent) == 0
    with app.state.session_factory() as session:
        assert session.get(Challenge, challenge["challenge_id"]).status == "pending"
    # The keyless path does not touch idempotency storage.
    assert _submit(client, challenge).status_code == 201


def test_evidence_table_write_failure_returns_500_with_no_record(app, client):
    from sqlalchemy import text

    challenge = _create_challenge(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE evidence"))

    response = _submit(client, challenge, key="key-1")
    assert response.status_code == 500
    assert response.json() == {"detail": "evidence submission failed"}
    assert _count_rows(app, EvidenceIdempotencyRecord) == 0
    with app.state.session_factory() as session:
        # The whole transaction rolled back: the consumption did not stick.
        assert session.get(Challenge, challenge["challenge_id"]).status == "pending"


def test_failed_write_leaves_key_free_and_recovers_on_retry(tmp_path):
    from sqlalchemy import text

    url = f"sqlite:///{tmp_path}/recover-evidence-idem.db"
    application = create_app(url)
    client = TestClient(application)
    challenge = _create_challenge(client)

    with application.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE evidence_idempotency_records"))
    failed = _submit(client, challenge, key="key-1")
    assert failed.status_code == 500
    application.state.engine.dispose()

    # A fresh process re-creates the missing table additively and the same
    # legal request now succeeds under the same key.
    application = create_app(url)
    client = TestClient(application)
    created = _submit(client, challenge, key="key-1")
    assert created.status_code == 201
    with application.state.session_factory() as session:
        assert session.query(Evidence).count() == 1
        assert session.query(EvidenceIdempotencyRecord).count() == 1
        assert session.get(Challenge, challenge["challenge_id"]).status == "consumed"
    application.state.engine.dispose()


# --- restart persistence ----------------------------------------------------


def test_idempotency_record_survives_restart_and_replays_original(tmp_path):
    url = f"sqlite:///{tmp_path}/restart-evidence-idem.db"

    app1 = create_app(url)
    client1 = TestClient(app1)
    challenge = _create_challenge(client1)
    accepted = _submit(client1, challenge, key="durable-key")
    assert accepted.status_code == 201
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    replay = _submit(client2, challenge, key="durable-key")
    assert replay.status_code == 201
    assert replay.content == accepted.content
    assert replay.json()["evidence_id"] == accepted.json()["evidence_id"]
    assert replay.json()["received_at"] == accepted.json()["received_at"]
    with app2.state.session_factory() as session:
        assert session.query(Evidence).count() == 1
        assert session.query(EvidenceIdempotencyRecord).count() == 1
    app2.state.engine.dispose()


# --- additive migration on an old database ----------------------------------


def test_old_database_creates_idempotency_table_on_first_open(tmp_path):
    from sqlalchemy import create_engine, inspect, text

    from proof_release.db import Base

    url = f"sqlite:///{tmp_path}/evidence-legacy.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE evidence_idempotency_records"))
    assert "evidence_idempotency_records" not in inspect(engine).get_table_names()
    engine.dispose()

    application = create_app(url)
    assert "evidence_idempotency_records" in inspect(
        application.state.engine
    ).get_table_names()
    client = TestClient(application)
    challenge = _create_challenge(client)

    keyed = _submit(client, challenge, key="legacy-key")
    assert keyed.status_code == 201
    assert _submit(client, challenge, key="legacy-key").content == keyed.content
    with application.state.session_factory() as session:
        assert session.query(EvidenceIdempotencyRecord).count() == 1

    application.state.engine.dispose()
    application = create_app(url)
    client = TestClient(application)
    assert _submit(client, challenge, key="legacy-key").status_code == 201
    application.state.engine.dispose()


# --- secrecy: only digests and documented columns ---------------------------


def test_record_stores_no_plaintext_nonce_or_evidence(app, client):
    challenge = _create_challenge(client)
    evidence = "cXVvdGU="
    response = _submit(client, challenge, key="key-1", evidence=evidence)
    assert response.status_code == 201

    (record,) = _records(app)
    table = type(record).metadata.tables["evidence_idempotency_records"]
    assert {column.name for column in table.columns} == {
        "record_id",
        "tenant_id",
        "workload_id",
        "idempotency_key",
        "request_fingerprint",
        "response_body",
        "created_at",
    }
    # The plaintext nonce and evidence never appear in any stored column.
    assert challenge["nonce"] not in record.request_fingerprint
    assert evidence not in record.request_fingerprint
    assert challenge["nonce"] not in record.response_body
    assert evidence not in record.response_body
    # The fingerprint really is derived from the two digests: recompute it.
    from proof_release.app import _evidence_request_fingerprint

    expected = _evidence_request_fingerprint(
        challenge["challenge_id"],
        hashlib.sha256(challenge["nonce"].encode("ascii")).hexdigest(),
        "tpm-quote",
        hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
    )
    assert record.request_fingerprint == expected
    # The response body is only the four documented public fields.
    assert set(json.loads(record.response_body)) == {
        "evidence_id",
        "challenge_id",
        "status",
        "received_at",
    }
