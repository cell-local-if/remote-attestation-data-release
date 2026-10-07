"""Tests for the optional Idempotency-Key header on POST /v1/evidence.

Contract under test:

* the header is optional; a missing header keeps the existing
  one-submission-per-challenge semantics (a repeated submission is
  still 409);
* a present header must occur exactly once and carry 1..64 visible ASCII
  characters; an empty value, surrounding or embedded whitespace, a
  control or non-ASCII character, an over-long value or a duplicated
  header line is an indistinguishable 422 raised after body validation
  and before any challenge read or write;
* the first successful keyed submission atomically consumes the
  challenge, persists the evidence (digest only) and the proof-received
  event and writes the idempotency record with the exact first 201 body;
* a same-key same-scope replay of the same fields returns that stored
  201 byte-for-byte (including the original received_at), consumes no
  challenge and adds no evidence or event — even after the challenge
  has since expired;
* a same-key request with any differing field (challenge_id, nonce,
  evidence_format or evidence) in the same scope is a stable 422 with
  detail "idempotency key reused with a different request"; the same
  key in another tenant or workload is independent;
* only a successful submission occupies the key: 404/401/409/410/500
  save no record, so a recovered request re-judges normally;
* concurrent same-key retries settle exactly once: one claim, one
  evidence row, one event, and every retry reads the same 201;
  concurrent different keys contend for the challenge exactly as
  without idempotency (one 201, the rest 409);
* any persistence failure on the first creation fully rolls back and is
  a 500 with detail "evidence submission failed", leaving no half state
  and not occupying the key, so the identical request retries
  successfully once storage recovers;
* the record survives restarts, is created additively on databases that
  predate it, and never stores the plaintext nonce or evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import (
    Challenge,
    Evidence,
    EvidenceSubmissionIdempotencyRecord,
    ProofLifecycleEvent,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
IDEMPOTENCY_HEADER = "Idempotency-Key"
EVIDENCE = "cXVvdGU="
FORMAT = "tpm-quote"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/evidence-idem.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, **overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    body.update(overrides)
    return client.post("/v1/challenges", json=body).json()


def _submit_body(created, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "challenge_id": created["challenge_id"],
        "nonce": created["nonce"],
        "evidence_format": FORMAT,
        "evidence": EVIDENCE,
    }
    body.update(overrides)
    return body


def _submit(client, created, *, key=..., **overrides):
    headers = (
        None
        if key is ...
        else ({IDEMPOTENCY_HEADER: key} if key is not None else None)
    )
    return client.post(
        "/v1/evidence", json=_submit_body(created, **overrides), headers=headers
    )


async def _asgi_call(application, headers, body: bytes):
    """Invoke the ASGI app directly with arbitrary raw header bytes.

    The HTTP test client refuses some header values (empty, control
    characters, non-ASCII bytes) before they reach the app; driving the
    ASGI app lets the service's own header validation see them.
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
        "path": "/v1/evidence",
        "raw_path": b"/v1/evidence",
        "query_string": b"",
        "headers": headers,
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }
    await application(scope, receive, send)
    return status["code"], b"".join(chunks)


def _raw_submit(app, body_obj, *, key_raw=b"key-1", duplicate_key=False):
    payload = json.dumps(body_obj).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"idempotency-key", key_raw),
    ]
    if duplicate_key:
        headers.append((b"idempotency-key", key_raw))
    return asyncio.run(_asgi_call(app, headers, payload))


def _count(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


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
def test_invalid_idempotency_key_is_422_before_challenge_access(
    app, raw_key
):
    client = TestClient(app)
    created = _create(client)
    status, raw = _raw_submit(app, _submit_body(created), key_raw=raw_key)
    assert status == 422, raw
    assert json.loads(raw)["detail"] == "invalid idempotency key"
    # Nothing was read for judgement or written.
    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 0
    assert _count(app, Evidence) == 0
    assert _count(app, ProofLifecycleEvent) == 0
    with app.state.session_factory() as session:
        row = session.get(Challenge, created["challenge_id"])
        assert row.status == "pending"
        assert row.consumed_at is None


def test_duplicate_idempotency_header_is_422(app):
    client = TestClient(app)
    created = _create(client)
    status, raw = _raw_submit(
        app, _submit_body(created), key_raw=b"key-1", duplicate_key=True
    )
    assert status == 422, raw
    assert json.loads(raw)["detail"] == "invalid idempotency key"
    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 0


def test_invalid_key_with_unknown_challenge_is_still_422_for_header(app):
    body = _submit_body(
        {
            "challenge_id": "00000000-0000-0000-0000-000000000000",
            "nonce": "AAAA",
        }
    )
    status, raw = _raw_submit(app, body, key_raw=b" ")
    assert status == 422
    assert json.loads(raw)["detail"] == "invalid idempotency key"


@pytest.mark.parametrize("length", [1, 64])
def test_boundary_length_keys_accepted(client, length):
    created = _create(client)
    response = _submit(client, created, key="A" * length)
    assert response.status_code == 201, response.text
    assert _count(client.app, EvidenceSubmissionIdempotencyRecord) == 1


def test_visible_punctuation_key_accepted(client):
    created = _create(client)
    response = _submit(
        client, created, key="~!@#$%^&*()_+-=[]{}|;:',.<>/?`"
    )
    assert response.status_code == 201, response.text


# --- first keyed submission ------------------------------------------------


def test_first_keyed_submission_persists_atomically(client, app):
    created = _create(client)
    response = _submit(client, created, key="submit-key-1")
    assert response.status_code == 201
    data = response.json()
    assert set(data) == {
        "evidence_id",
        "challenge_id",
        "status",
        "received_at",
    }
    assert data["challenge_id"] == created["challenge_id"]
    assert data["status"] == "received"
    assert data["evidence_id"]
    received_at = datetime.fromisoformat(data["received_at"])
    assert received_at.utcoffset() == timedelta(0)
    assert EVIDENCE not in response.text and created["nonce"] not in response.text

    with app.state.session_factory() as session:
        challenge = session.get(Challenge, created["challenge_id"])
        assert challenge.status == "consumed"
        assert challenge.consumed_at is not None
        evidence = session.get(Evidence, data["evidence_id"])
        assert evidence is not None
        assert evidence.evidence_sha256 == hashlib.sha256(
            EVIDENCE.encode("utf-8")
        ).hexdigest()
        events = session.scalars(
            select(ProofLifecycleEvent).where(
                ProofLifecycleEvent.evidence_id == data["evidence_id"]
            )
        ).all()
        assert len(events) == 1
        record = session.query(EvidenceSubmissionIdempotencyRecord).one()
        assert record.tenant_id == TENANT
        assert record.workload_id == WORKLOAD
        assert record.idempotency_key == "submit-key-1"
        assert record.challenge_id == created["challenge_id"]
        assert record.evidence_id == data["evidence_id"]
        assert record.evidence_format == FORMAT
        assert record.nonce_digest == hashlib.sha256(
            created["nonce"].encode("ascii")
        ).hexdigest()
        assert record.response_body.encode() == response.content
        # The record commits in the same instant as the claim.
        assert record.created_at == challenge.consumed_at
        assert record.created_at == evidence.received_at


def test_missing_header_keeps_legacy_duplicate_409(client, app):
    created = _create(client)
    first = _submit(client, created, key=None)
    assert first.status_code == 201
    second = _submit(client, created, key=None)
    assert second.status_code == 409
    assert second.json()["detail"] == "challenge already consumed"
    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 0


# --- replay ----------------------------------------------------------------


def test_replay_returns_saved_201_verbatim_and_appends_nothing(client, app):
    created = _create(client)
    first = _submit(client, created, key="replay-key")
    assert first.status_code == 201
    original_received_at = first.json()["received_at"]

    second = _submit(client, created, key="replay-key")
    third = _submit(client, created, key="replay-key")
    assert second.status_code == 201 and third.status_code == 201
    # Byte-for-byte identical: same identifiers and the original time.
    assert second.content == first.content
    assert third.content == first.content
    assert second.json()["received_at"] == original_received_at

    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 1
    assert _count(app, Evidence) == 1
    assert _count(app, ProofLifecycleEvent) == 1


def test_replay_returns_original_201_after_challenge_expired(client, app):
    created = _create(client, ttl_seconds=30)
    first = _submit(client, created, key="replay-key")
    assert first.status_code == 201
    with app.state.session_factory() as session:
        row = session.get(Challenge, created["challenge_id"])
        original_consumed_at = row.consumed_at
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    replay = _submit(client, created, key="replay-key")
    assert replay.status_code == 201
    assert replay.content == first.content
    with app.state.session_factory() as session:
        row = session.get(Challenge, created["challenge_id"])
        assert row.status == "consumed"
        assert row.consumed_at == original_consumed_at
        assert session.query(ProofLifecycleEvent).count() == 1


@pytest.mark.parametrize(
    "overrides",
    [
        lambda other: {
            "challenge_id": other["challenge_id"],
            "nonce": other["nonce"],
        },
        {"evidence_format": "x509-attested-nonce-json"},
        {"evidence": "b3RoZXI="},
    ],
)
def test_same_key_different_field_is_422_and_changes_nothing(
    client, app, overrides
):
    created = _create(client)
    first = _submit(client, created, key="shared")
    assert first.status_code == 201

    second = _create(client)
    if callable(overrides):
        field_overrides = overrides(second)
    else:
        field_overrides = dict(overrides)
    conflict = _submit(client, created, key="shared", **field_overrides)
    assert conflict.status_code == 422, conflict.text
    assert conflict.json()["detail"] == (
        "idempotency key reused with a different request"
    )

    # Nothing changed: the (possibly different) challenge is untouched,
    # still exactly one evidence/event/record, and an exact replay still
    # answers with the stored 201.
    assert _count(app, Evidence) == 1
    assert _count(app, ProofLifecycleEvent) == 1
    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 1
    with app.state.session_factory() as session:
        second_row = session.get(Challenge, second["challenge_id"])
        assert second_row.status == "pending"
        assert second_row.consumed_at is None
    replay = _submit(client, created, key="shared")
    assert replay.status_code == 201
    assert replay.content == first.content


def test_same_key_wrong_nonce_is_422_not_401_or_409(client, app):
    # The stored field comparison precedes any challenge re-judgement: a
    # replay carrying a different nonce against the now-consumed
    # challenge is the idempotency 422, never 401/409.
    created = _create(client)
    assert _submit(client, created, key="shared").status_code == 201
    wrong = ("B" if created["nonce"][0] != "B" else "C") + created["nonce"][1:]
    conflict = _submit(client, created, key="shared", nonce=wrong)
    assert conflict.status_code == 422
    assert conflict.json()["detail"] == (
        "idempotency key reused with a different request"
    )
    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 1


def test_same_key_non_ascii_format_is_422_not_500(client, app):
    # The replay comparison must not raise on non-ASCII free-form field
    # values (hmac.compare_digest rejects non-ASCII str); a differing
    # non-ASCII format is the same stable 422.
    created = _create(client)
    assert _submit(client, created, key="shared").status_code == 201
    conflict = _submit(client, created, key="shared", evidence_format="tpm-quöté")
    assert conflict.status_code == 422
    assert conflict.json()["detail"] == (
        "idempotency key reused with a different request"
    )
    # An exact replay still answers with the stored 201.
    replay = _submit(client, created, key="shared")
    assert replay.status_code == 201


def test_same_key_in_other_tenant_or_workload_is_independent(client, app):
    created_a = _create(client)
    created_b = _create(client, tenant_id="tenant-b")
    created_c = _create(client, workload_id="workload-9")

    r1 = _submit(client, created_a, key="shared-key")
    r2 = _submit(
        client,
        created_b,
        key="shared-key",
        tenant_id="tenant-b",
    )
    r3 = _submit(
        client,
        created_c,
        key="shared-key",
        workload_id="workload-9",
    )
    assert {r.status_code for r in (r1, r2, r3)} == {201}
    bodies = {r.content for r in (r1, r2, r3)}
    assert len(bodies) == 3
    assert _count(app, Evidence) == 3
    assert _count(app, ProofLifecycleEvent) == 3
    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 3
    # Each scope's replay resolves to its own stored submission.
    assert (
        _submit(client, created_b, key="shared-key", tenant_id="tenant-b").content
        == r2.content
    )
    assert (
        _submit(
            client, created_c, key="shared-key", workload_id="workload-9"
        ).content
        == r3.content
    )


# --- failures do not occupy the key ----------------------------------------


def test_unknown_challenge_failure_does_not_occupy_key(client, app):
    missing = {
        "challenge_id": "00000000-0000-0000-0000-000000000000",
        "nonce": "AAAA",
    }
    failed = _submit(client, missing, key="retry-key")
    assert failed.status_code == 404
    assert failed.json()["detail"] == "challenge not found"
    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 0
    created = _create(client)
    recovered = _submit(client, created, key="retry-key")
    assert recovered.status_code == 201
    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 1


def test_cross_scope_challenge_is_404_and_does_not_occupy_key(client, app):
    created = _create(client)
    failed = _submit(client, created, key="retry-key", tenant_id="tenant-b")
    assert failed.status_code == 404
    assert failed.json()["detail"] == "challenge not found"
    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 0
    recovered = _submit(client, created, key="retry-key")
    assert recovered.status_code == 201


def test_wrong_nonce_failure_does_not_occupy_key(client, app):
    created = _create(client)
    wrong = ("B" if created["nonce"][0] != "B" else "C") + created["nonce"][1:]
    failed = _submit(client, created, key="retry-key", nonce=wrong)
    assert failed.status_code == 401
    assert failed.json()["detail"] == "invalid nonce"
    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 0
    recovered = _submit(client, created, key="retry-key")
    assert recovered.status_code == 201


def test_already_consumed_failure_does_not_occupy_key(client, app):
    created = _create(client)
    # Settled keyless: a later keyed submission is the existing 409 and
    # must not save the key.
    assert _submit(client, created, key=None).status_code == 201
    failed = _submit(client, created, key="retry-key")
    assert failed.status_code == 409
    assert failed.json()["detail"] == "challenge already consumed"
    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 0
    # The key stays free for a fresh challenge.
    second = _create(client)
    recovered = _submit(client, second, key="retry-key")
    assert recovered.status_code == 201
    assert recovered.json()["challenge_id"] == second["challenge_id"]
    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 1


def test_expired_challenge_failure_does_not_occupy_key(client, app):
    expired = _create(client, ttl_seconds=30)
    fresh = _create(client)
    with app.state.session_factory() as session:
        row = session.get(Challenge, expired["challenge_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    failed = _submit(client, expired, key="retry-key")
    assert failed.status_code == 410
    assert failed.json()["detail"] == "challenge expired"
    assert _count(app, EvidenceSubmissionIdempotencyRecord) == 0
    recovered = _submit(client, fresh, key="retry-key")
    assert recovered.status_code == 201
    assert recovered.json()["challenge_id"] == fresh["challenge_id"]


# --- concurrency -----------------------------------------------------------


def test_concurrent_same_key_retries_settle_once(app):
    client = TestClient(app)
    created = _create(client)
    body = _submit_body(created)
    headers = {IDEMPOTENCY_HEADER: "race-key"}

    def submit():
        return TestClient(app).post("/v1/evidence", json=body, headers=headers)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: submit(), range(8)))

    assert [r.status_code for r in responses].count(201) == 8
    assert len({r.content for r in responses}) == 1
    with app.state.session_factory() as session:
        row = session.get(Challenge, created["challenge_id"])
        assert row.status == "consumed"
        assert session.query(Evidence).count() == 1
        assert session.query(ProofLifecycleEvent).count() == 1
        assert session.query(EvidenceSubmissionIdempotencyRecord).count() == 1


def test_concurrent_different_keys_contend_like_keyless(app):
    client = TestClient(app)
    created = _create(client)
    body = _submit_body(created)

    def submit(i):
        return TestClient(app).post(
            "/v1/evidence",
            json=body,
            headers={IDEMPOTENCY_HEADER: f"race-key-{i}"},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(submit, range(8)))

    statuses = [r.status_code for r in responses]
    assert statuses.count(201) == 1
    assert statuses.count(409) == 7
    assert all(
        r.json()["detail"] == "challenge already consumed"
        for r in responses
        if r.status_code == 409
    )
    with app.state.session_factory() as session:
        assert session.query(Evidence).count() == 1
        assert session.query(ProofLifecycleEvent).count() == 1
        # Only the winner's key is occupied.
        assert session.query(EvidenceSubmissionIdempotencyRecord).count() == 1
        winning_key = (
            session.query(EvidenceSubmissionIdempotencyRecord).one().idempotency_key
        )
    winner = next(r for r in responses if r.status_code == 201)
    winner_index = next(
        i for i, r in enumerate(responses) if r.status_code == 201
    )
    assert winning_key == f"race-key-{winner_index}"
    # A losing key stayed free: reused against a fresh challenge it
    # succeeds normally and keeps that challenge's original response.
    second = _create(client)
    loser_key = "race-key-0" if winning_key != "race-key-0" else "race-key-1"
    recovered = _submit(client, second, key=loser_key)
    assert recovered.status_code == 201
    assert recovered.content != winner.content
    assert recovered.json()["challenge_id"] == second["challenge_id"]
    with app.state.session_factory() as session:
        assert session.query(EvidenceSubmissionIdempotencyRecord).count() == 2


# --- persistence failure rolls back everything ------------------------------


def _fail_event_seq(*_args, **_kwargs):
    raise RuntimeError("simulated storage failure")


def test_keyed_persistence_failure_rolls_back_and_frees_key(client, app, monkeypatch):
    created = _create(client)
    monkeypatch.setattr(app_module, "_next_proof_event_commit_seq", _fail_event_seq)
    failed = _submit(client, created, key="retry-key")
    assert failed.status_code == 500
    assert failed.json()["detail"] == "evidence submission failed"
    monkeypatch.undo()

    with app.state.session_factory() as session:
        row = session.get(Challenge, created["challenge_id"])
        assert row.status == "pending"
        assert row.consumed_at is None
        assert session.query(Evidence).count() == 0
        assert session.query(ProofLifecycleEvent).count() == 0
        assert session.query(EvidenceSubmissionIdempotencyRecord).count() == 0

    # The identical request (same key) succeeds once storage recovers.
    recovered = _submit(client, created, key="retry-key")
    assert recovered.status_code == 201
    with app.state.session_factory() as session:
        assert session.get(Challenge, created["challenge_id"]).status == "consumed"
        assert session.query(Evidence).count() == 1
        assert session.query(ProofLifecycleEvent).count() == 1
        assert session.query(EvidenceSubmissionIdempotencyRecord).count() == 1


def test_keyless_persistence_failure_rolls_back_completely(client, app, monkeypatch):
    created = _create(client)
    monkeypatch.setattr(app_module, "_next_proof_event_commit_seq", _fail_event_seq)
    failed = _submit(client, created, key=None)
    assert failed.status_code == 500
    assert failed.json()["detail"] == "evidence submission failed"
    monkeypatch.undo()

    with app.state.session_factory() as session:
        row = session.get(Challenge, created["challenge_id"])
        assert row.status == "pending"
        assert session.query(Evidence).count() == 0
        assert session.query(ProofLifecycleEvent).count() == 0

    recovered = _submit(client, created, key=None)
    assert recovered.status_code == 201


# --- restart and additive schema -------------------------------------------


def test_idempotency_record_survives_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart-evidence-idem.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    created = _create(client1)
    first = _submit(client1, created, key="durable-key")
    assert first.status_code == 201
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    # Even after the challenge has expired post-restart, the replay
    # answers with the original 201.
    with app2.state.session_factory() as session:
        row = session.get(Challenge, created["challenge_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
    replay = _submit(client2, created, key="durable-key")
    assert replay.status_code == 201
    assert replay.content == first.content
    with app2.state.session_factory() as session:
        assert session.query(EvidenceSubmissionIdempotencyRecord).count() == 1
    app2.state.engine.dispose()


def test_idempotency_table_is_created_additively_on_old_database(tmp_path):
    # A database written by a deployment that predates the table (and
    # which already holds challenges/evidence) gains it on open, and a
    # keyed submission works against the upgraded database.
    url = f"sqlite:///{tmp_path}/legacy-evidence.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    created = _create(client1)
    assert _submit(client1, created, key=None).status_code == 201
    app1.state.engine.dispose()

    # Open with the new build once, then remove just the new table to
    # simulate a deployment that predates it.
    app2 = create_app(url)
    with app2.state.engine.begin() as conn:
        conn.execute(
            text("DROP TABLE evidence_submission_idempotency_records")
        )
    app2.state.engine.dispose()

    app3 = create_app(url)
    client3 = TestClient(app3)
    second = _create(client3)
    response = _submit(client3, second, key="after-upgrade")
    assert response.status_code == 201
    with app3.state.session_factory() as session:
        assert session.query(EvidenceSubmissionIdempotencyRecord).count() == 1
        assert session.get(Challenge, second["challenge_id"]).status == "consumed"
    app3.state.engine.dispose()


# --- secrecy ---------------------------------------------------------------


def test_keyed_submission_never_persists_plaintext_nonce_or_evidence(
    client, app
):
    created = _create(client)
    nonce = created["nonce"]
    first = _submit(client, created, key="secret-key")
    replay = _submit(client, created, key="secret-key")
    assert nonce not in first.text and EVIDENCE not in first.text
    assert nonce not in replay.text and EVIDENCE not in replay.text
    with app.state.session_factory() as session:
        record = session.query(EvidenceSubmissionIdempotencyRecord).one()
        values = {
            c.name: getattr(record, c.name) for c in record.__table__.columns
        }
        for name, value in values.items():
            assert nonce not in str(value), f"nonce leaked in {name}"
            assert EVIDENCE not in str(value), f"evidence leaked in {name}"
        # The stored nonce column is the same digest the challenge holds,
        # not a hash of some canonical request envelope.
        assert record.nonce_digest == hashlib.sha256(
            nonce.encode("ascii")
        ).hexdigest()
        assert record.evidence_sha256 == hashlib.sha256(
            EVIDENCE.encode("utf-8")
        ).hexdigest()
        assert not hasattr(record, "nonce")
        assert not hasattr(record, "evidence")
