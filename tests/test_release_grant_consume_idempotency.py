"""Tests for the optional Idempotency-Key header on release-grant
consumption (POST /v1/release-grants/{grant_id}/consume).

A keyed consume saves the first successful 200 response atomically with
the pending -> consumed settlement and replays it byte-for-byte on
same-key same-shape retries; a missing key keeps the legacy one-shot
semantics exactly. As in test_release_grants.py the default app is built
without entering its lifespan; restart and concurrency checks create
dedicated apps/clients.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    RateLimitCounter,
    ReleaseGrant,
    ReleaseGrantConsumeIdempotencyRecord,
    ReleaseGrantEvent,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"

IDEMPOTENCY_HEADER = "Idempotency-Key"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    application = create_app(f"sqlite:///{tmp_path}/consume-idem.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _mac(nonce: str, claims: dict, tenant=TENANT, workload=WORKLOAD) -> str:
    key = hmac.new(
        SECRET.encode("utf-8"),
        f"{tenant}:{workload}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _evidence(nonce: str, tenant=TENANT, workload=WORKLOAD) -> str:
    claims = {"m": "x"}
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims, tenant, workload)}
    )


def _grant(client, *, tenant=TENANT, workload=WORKLOAD, data_id="data-1"):
    """Drive the full flow and return one pending grant with its capability."""
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    evidence = _evidence(created["nonce"], tenant, workload)
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    verified = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.status_code == 200
    policy = client.post(
        "/v1/policies",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "name": "release",
            "rule": {"claim": "m", "equals": "x"},
        },
    ).json()
    decided = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert decided.status_code == 200
    grant = client.post(
        "/v1/release-grants",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "decision_id": decided.json()["decision_id"],
            "data_id": data_id,
        },
    )
    assert grant.status_code == 201
    return grant.json()


def _consume(
    client,
    grant_id,
    capability,
    *,
    tenant=TENANT,
    workload=WORKLOAD,
    key=None,
):
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else None
    return client.post(
        f"/v1/release-grants/{grant_id}/consume",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "capability": capability,
        },
        headers=headers,
    )


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


def _raw_consume(app, grant_id, body_obj, *, key_raw=b"key-1", duplicate_key=False):
    payload = json.dumps(body_obj).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"idempotency-key", key_raw),
    ]
    if duplicate_key:
        headers.append((b"idempotency-key", key_raw))
    return asyncio.run(
        _asgi_call(app, f"/v1/release-grants/{grant_id}/consume", headers, payload)
    )


def _count_rows(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


def _budget_used(app, tenant=TENANT, workload=WORKLOAD) -> int:
    with app.state.session_factory() as session:
        return (
            session.query(RateLimitCounter)
            .filter(
                RateLimitCounter.tenant_id == tenant,
                RateLimitCounter.workload_id == workload,
            )
            .count()
        )


def _grant_row(app, grant_id) -> ReleaseGrant:
    with app.state.session_factory() as session:
        return session.get(ReleaseGrant, grant_id)


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
def test_invalid_idempotency_key_is_422_without_budget_or_grant_access(app, raw_key):
    client = TestClient(app)
    grant = _grant(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }
    status, _ = _raw_consume(app, grant["grant_id"], body, key_raw=raw_key)
    assert status == 422
    # No budget slot was spent and the grant was never read or written.
    assert _budget_used(app) == 0
    row = _grant_row(app, grant["grant_id"])
    assert row.status == "pending"
    assert row.consumed_at is None
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 0
    # Only the issuance audit exists; the rejected consume appended none.
    with app.state.session_factory() as session:
        assert [a.status for a in session.query(AuditEvent).all()] == ["pending"]


def test_duplicate_idempotency_header_is_422(app):
    client = TestClient(app)
    grant = _grant(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }
    status, _ = _raw_consume(
        app, grant["grant_id"], body, key_raw=b"key-1", duplicate_key=True
    )
    assert status == 422
    assert _budget_used(app) == 0
    assert _grant_row(app, grant["grant_id"]).status == "pending"
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 0


def test_duplicate_idempotency_header_through_client_is_422(app, client):
    grant = _grant(client)
    response = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
        headers=[
            (IDEMPOTENCY_HEADER, "key-1"),
            (IDEMPOTENCY_HEADER, "key-1"),
        ],
    )
    assert response.status_code == 422
    assert _budget_used(app) == 0
    assert _grant_row(app, grant["grant_id"]).status == "pending"


def test_empty_idempotency_header_through_client_is_422(app, client):
    grant = _grant(client)
    response = _consume(client, grant["grant_id"], grant["capability"], key="")
    assert response.status_code == 422
    assert _budget_used(app) == 0
    assert _grant_row(app, grant["grant_id"]).status == "pending"


@pytest.mark.parametrize("length", [1, 64])
def test_boundary_length_keys_accepted(app, client, length):
    grant = _grant(client)
    response = _consume(
        client, grant["grant_id"], grant["capability"], key="A" * length
    )
    assert response.status_code == 200, response.text
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 1


def test_invalid_key_rejected_even_when_budget_exhausted(app, client):
    # Header validation precedes the budget: an illegal key is a 422 (and
    # spends nothing) even after the minute's budget is gone.
    grant = _grant(client)
    for _ in range(5):
        assert (
            _consume(client, grant["grant_id"], "wrong-capability").status_code == 401
        )
    assert _budget_used(app) == 1  # one window row
    with app.state.session_factory() as session:
        counter = session.query(RateLimitCounter).one()
        assert counter.count == 5
    status, _ = _raw_consume(
        app,
        grant["grant_id"],
        {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
        key_raw=b"bad key",
    )
    assert status == 422
    with app.state.session_factory() as session:
        assert session.query(RateLimitCounter).one().count == 5


# --- core replay semantics -------------------------------------------------


def test_first_keyed_consume_persists_record_atomically(app, client):
    grant = _grant(client)
    response = _consume(client, grant["grant_id"], grant["capability"], key="k-1")
    assert response.status_code == 200, response.text
    data = response.json()
    assert list(data) == ["grant_id", "decision_id", "data_id", "consumed", "consumed_at"]
    assert data["grant_id"] == grant["grant_id"]
    assert data["consumed"] is True

    row = _grant_row(app, grant["grant_id"])
    assert row.status == "consumed"
    assert row.consumed_at is not None

    with app.state.session_factory() as session:
        record = session.query(ReleaseGrantConsumeIdempotencyRecord).one()
        assert record.tenant_id == TENANT
        assert record.workload_id == WORKLOAD
        assert record.idempotency_key == "k-1"
        assert record.grant_id == grant["grant_id"]
        assert len(record.request_fingerprint) == 64
        assert record.response_body.encode() == response.content
        # The settlement, its timeline event, the audit row and the record
        # all exist together, exactly once.
        events = (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
            .all()
        )
        assert [(e.old_status, e.new_status, e.reason) for e in events] == [
            (None, "pending", "issued"),
            ("pending", "consumed", "consume"),
        ]
        audits = (
            session.query(AuditEvent)
            .filter(AuditEvent.grant_id == grant["grant_id"])
            .all()
        )
        assert [a.status for a in audits] == ["pending", "consumed"]


def test_replay_returns_saved_200_verbatim_and_changes_nothing(app, client):
    grant = _grant(client)
    first = _consume(client, grant["grant_id"], grant["capability"], key="k-1")
    assert first.status_code == 200

    second = _consume(client, grant["grant_id"], grant["capability"], key="k-1")
    third = _consume(client, grant["grant_id"], grant["capability"], key="k-1")
    assert second.status_code == 200 and third.status_code == 200
    assert second.content == first.content
    assert third.content == first.content
    assert second.json()["consumed_at"] == first.json()["consumed_at"]
    # Replays append nothing: one record, one settlement event pair, one
    # consumed audit row, and the grant's consumed_at is untouched.
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 1
    with app.state.session_factory() as session:
        assert (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
            .count()
            == 2
        )
        assert (
            session.query(AuditEvent)
            .filter(
                AuditEvent.grant_id == grant["grant_id"],
                AuditEvent.status == "consumed",
            )
            .count()
            == 1
        )


def test_replay_after_grant_expired_still_returns_original_200(app, client):
    grant = _grant(client)
    first = _consume(client, grant["grant_id"], grant["capability"], key="k-1")
    assert first.status_code == 200

    # The grant's expiry passes after the successful consumption.
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    replay = _consume(client, grant["grant_id"], grant["capability"], key="k-1")
    assert replay.status_code == 200
    assert replay.content == first.content
    assert replay.json()["consumed_at"] == first.json()["consumed_at"]


def test_replay_consumes_budget_before_idempotency_judgement(app, client):
    grant = _grant(client)
    first = _consume(client, grant["grant_id"], grant["capability"], key="k-1")
    assert first.status_code == 200
    # One slot for the consume plus four replays exhaust the minute.
    for _ in range(4):
        replay = _consume(client, grant["grant_id"], grant["capability"], key="k-1")
        assert replay.status_code == 200
    with app.state.session_factory() as session:
        assert session.query(RateLimitCounter).one().count == 5
    # The sixth admitted-shape request in the minute is a 429 even though
    # an idempotency record exists: the budget is judged first.
    limited = _consume(client, grant["grant_id"], grant["capability"], key="k-1")
    assert limited.status_code == 429
    assert set(limited.json()) == {"retry_after_seconds"}


def test_same_key_different_grant_is_409_and_changes_nothing(app, client):
    first_grant = _grant(client)
    second_grant = _grant(client, data_id="data-2")
    first = _consume(
        client, first_grant["grant_id"], first_grant["capability"], key="dup"
    )
    assert first.status_code == 200

    conflict = _consume(
        client, second_grant["grant_id"], second_grant["capability"], key="dup"
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency key reused with different request"
    # The second grant was never touched and its key is still free.
    assert _grant_row(app, second_grant["grant_id"]).status == "pending"
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 1
    # An exact replay of the first request still works.
    replay = _consume(
        client, first_grant["grant_id"], first_grant["capability"], key="dup"
    )
    assert replay.status_code == 200
    assert replay.content == first.content
    # The second grant consumes fine under its own key.
    other = _consume(
        client, second_grant["grant_id"], second_grant["capability"], key="other"
    )
    assert other.status_code == 200


def test_same_key_different_capability_is_409(app, client):
    grant = _grant(client)
    first = _consume(client, grant["grant_id"], grant["capability"], key="dup")
    assert first.status_code == 200

    conflict = _consume(client, grant["grant_id"], "some-other-capability", key="dup")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency key reused with different request"
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 1


def test_same_key_different_scope_is_independent(app, client):
    grant_a = _grant(client)
    grant_b = _grant(client, tenant="tenant-b")
    grant_c = _grant(client, workload="workload-9")

    r1 = _consume(client, grant_a["grant_id"], grant_a["capability"], key="shared")
    r2 = _consume(
        client,
        grant_b["grant_id"],
        grant_b["capability"],
        tenant="tenant-b",
        key="shared",
    )
    r3 = _consume(
        client,
        grant_c["grant_id"],
        grant_c["capability"],
        workload="workload-9",
        key="shared",
    )
    assert {r.status_code for r in (r1, r2, r3)} == {200}
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 3
    # Each scope's replay resolves to its own stored response.
    assert (
        _consume(client, grant_a["grant_id"], grant_a["capability"], key="shared").content
        == r1.content
    )
    assert (
        _consume(
            client,
            grant_b["grant_id"],
            grant_b["capability"],
            tenant="tenant-b",
            key="shared",
        ).content
        == r2.content
    )


# --- failed attempts never claim the key -----------------------------------


def test_failed_attempts_do_not_claim_the_key(app, client):
    grant = _grant(client)
    # 401: well-formed but wrong capability.
    wrong = _consume(client, grant["grant_id"], "wrong-capability", key="k-1")
    assert wrong.status_code == 401
    # 404: unknown grant.
    missing = _consume(
        client,
        "00000000-0000-0000-0000-000000000000",
        grant["capability"],
        key="k-1",
    )
    assert missing.status_code == 404
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 0
    # The key is still free: the corrected request consumes normally.
    accepted = _consume(client, grant["grant_id"], grant["capability"], key="k-1")
    assert accepted.status_code == 200
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 1


def test_expired_grant_attempt_does_not_claim_the_key(app, client):
    grant = _grant(client)
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
    expired = _consume(client, grant["grant_id"], grant["capability"], key="k-1")
    assert expired.status_code == 410
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 0
    # The same key remains usable on another grant.
    other = _grant(client, data_id="data-2")
    accepted = _consume(client, other["grant_id"], other["capability"], key="k-1")
    assert accepted.status_code == 200


def test_keyed_consume_on_keyless_consumed_grant_is_409_and_claims_nothing(
    app, client
):
    grant = _grant(client)
    # A keyless consume settles the grant first.
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200
    # A keyed retry of the same request observes the legacy 409: no record
    # existed, so there is nothing to replay and the key is not claimed.
    keyed = _consume(client, grant["grant_id"], grant["capability"], key="k-1")
    assert keyed.status_code == 409
    assert keyed.json()["detail"] == "grant already consumed"
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 0


# --- missing key preserves legacy semantics --------------------------------


def test_missing_key_keeps_one_shot_semantics(app, client):
    grant = _grant(client)
    first = _consume(client, grant["grant_id"], grant["capability"])
    assert first.status_code == 200
    second = _consume(client, grant["grant_id"], grant["capability"])
    assert second.status_code == 409
    assert second.json()["detail"] == "grant already consumed"
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 0


# --- concurrency -----------------------------------------------------------


def test_concurrent_same_key_consumes_settle_exactly_once(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setattr(app_module, "GRANT_BUDGET_PER_MINUTE", 64)
    application = create_app(f"sqlite:///{tmp_path}/concurrent-consume-idem.db")
    client = TestClient(application)
    grant = _grant(client)
    url = f"/v1/release-grants/{grant['grant_id']}/consume"
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }

    def consume():
        return TestClient(application).post(
            url, json=body, headers={IDEMPOTENCY_HEADER: "race-key"}
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: consume(), range(16)))

    # Exactly one request migrates the grant; every concurrent retry reads
    # the same stored result and also returns 200.
    assert all(r.status_code == 200 for r in responses), [
        r.status_code for r in responses
    ]
    assert len({r.content for r in responses}) == 1
    with application.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"
        assert session.query(ReleaseGrantConsumeIdempotencyRecord).count() == 1
        # One settlement: exactly one consume event and one consumed audit.
        assert (
            session.query(ReleaseGrantEvent)
            .filter(
                ReleaseGrantEvent.grant_id == grant["grant_id"],
                ReleaseGrantEvent.new_status == "consumed",
            )
            .count()
            == 1
        )
        assert (
            session.query(AuditEvent)
            .filter(
                AuditEvent.grant_id == grant["grant_id"],
                AuditEvent.status == "consumed",
            )
            .count()
            == 1
        )
    application.state.engine.dispose()


# --- restart persistence and incremental schema ----------------------------


def test_idempotency_record_survives_restart_and_replays_original(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart-consume-idem.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    grant = _grant(client1)
    first = _consume(client1, grant["grant_id"], grant["capability"], key="durable")
    assert first.status_code == 200
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    replay = _consume(client2, grant["grant_id"], grant["capability"], key="durable")
    assert replay.status_code == 200
    assert replay.content == first.content
    with app2.state.session_factory() as session:
        assert session.query(ReleaseGrantConsumeIdempotencyRecord).count() == 1
        # No new event or audit row was appended by the replay.
        assert (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
            .count()
            == 2
        )
    app2.state.engine.dispose()


def test_table_is_created_incrementally_on_preexisting_database(
    tmp_path, monkeypatch
):
    from sqlalchemy import text

    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/upgrade-consume-idem.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    grant = _grant(client1)
    # Simulate a database written by a deployment that predates the
    # idempotency table.
    with app1.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE release_grant_consume_idempotency_records"))
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    response = _consume(client2, grant["grant_id"], grant["capability"], key="k-1")
    assert response.status_code == 200
    replay = _consume(client2, grant["grant_id"], grant["capability"], key="k-1")
    assert replay.status_code == 200
    assert replay.content == response.content
    app2.state.engine.dispose()


# --- secrecy ---------------------------------------------------------------


def test_keyed_path_never_persists_or_leaks_plaintext_capability(app, client):
    grant = _grant(client)
    capability = grant["capability"]
    first = _consume(client, grant["grant_id"], capability, key="k-1")
    replay = _consume(client, grant["grant_id"], capability, key="k-1")
    assert first.status_code == 200 and replay.status_code == 200
    assert capability not in first.text
    assert capability not in replay.text
    with app.state.session_factory() as session:
        record = session.query(ReleaseGrantConsumeIdempotencyRecord).one()
        assert capability not in record.request_fingerprint
        assert capability not in record.response_body
        assert not hasattr(record, "capability")
