"""Tests for the optional Idempotency-Key header on
POST /v1/release-grants/{grant_id}/consume.

Contract under test:

* the header is optional; a missing header keeps the existing one-time
  consume semantics (a repeated presentation is still 409);
* a present header must occur exactly once and carry 1..64 visible ASCII
  characters; an empty value, surrounding whitespace, a control or
  non-ASCII character, an over-long value or a duplicated header line is
  an indistinguishable 422 raised before the budget reservation and
  before any grant read or write;
* an admitted keyed request draws the same shared per-scope minute slot
  as every other grant request, and only then judges idempotency;
* the first successful keyed consume atomically settles the grant,
  appends the timeline and audit events and persists the idempotency
  record with the exact first 200 body;
* a same-key same-scope replay returns that stored 200 byte-for-byte —
  even when the grant has since expired or changed otherwise — appends
  no events, and changes neither state nor timestamps;
* a same-key request with a different normalized grant id or capability
  in the same scope is a stable 409; the same key in another tenant or
  workload is independent;
* only a successful consume occupies the key: 404/401/409/410/500 save
  no record, so a recovered request re-judges normally;
* concurrent same-key retries settle exactly once: one migration, one
  event, one audit row, and every retry reads the same 200;
* the record survives restarts, is created additively on databases that
  predate it, and never stores the plaintext capability.
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


def _evidence(nonce: str, claims: dict | None = None, tenant=TENANT,
              workload=WORKLOAD) -> str:
    claims = claims if claims is not None else {"m": "x"}
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(
            nonce, claims, tenant, workload)}
    )


def _decision(client, tenant=TENANT, workload=WORKLOAD):
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    evidence = _evidence(created["nonce"], tenant=tenant, workload=workload)
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
    return decided.json()


def _grant(client, decision_id, *, data_id="data-1", tenant=TENANT,
           workload=WORKLOAD, ttl_seconds=None):
    body = {
        "tenant_id": tenant,
        "workload_id": workload,
        "decision_id": decision_id,
        "data_id": data_id,
    }
    if ttl_seconds is not None:
        body["ttl_seconds"] = ttl_seconds
    return client.post("/v1/release-grants", json=body)


def _consume(client, grant_id, capability, *, key=..., **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": capability,
    }
    body.update(fields)
    headers = None if key is ... else ({IDEMPOTENCY_HEADER: key} if key else None)
    return client.post(
        f"/v1/release-grants/{grant_id}/consume", json=body, headers=headers
    )


async def _asgi_call(application, path, headers, body: bytes):
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
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }
    await application(scope, receive, send)
    return status["code"], b"".join(chunks)


def _raw_consume(app, grant_id, body_obj, *, key_raw=b"key-1",
                 duplicate_key=False):
    path = f"/v1/release-grants/{grant_id}/consume"
    payload = json.dumps(body_obj).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"idempotency-key", key_raw),
    ]
    if duplicate_key:
        headers.append((b"idempotency-key", key_raw))
    return asyncio.run(_asgi_call(app, path, headers, payload))


def _count_rows(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


def _settled_fixture(client, *, ttl_seconds=None):
    decision = _decision(client)
    grant = _grant(
        client, decision["decision_id"], data_id="data-7",
        ttl_seconds=ttl_seconds,
    ).json()
    return decision, grant


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
def test_invalid_idempotency_key_is_422_before_budget_and_grant(
    app, raw_key
):
    client = TestClient(app)
    decision, grant = _settled_fixture(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }
    status, raw = _raw_consume(app, grant["grant_id"], body, key_raw=raw_key)
    assert status == 422, raw
    # No budget spent, no grant state touched, no record written.
    assert _count_rows(app, RateLimitCounter) == 0
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 0
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.consumed_at is None


def test_duplicate_idempotency_header_is_422(app):
    client = TestClient(app)
    _, grant = _settled_fixture(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }
    status, raw = _raw_consume(
        app, grant["grant_id"], body, key_raw=b"key-1", duplicate_key=True
    )
    assert status == 422, raw
    assert _count_rows(app, RateLimitCounter) == 0
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 0


def test_duplicate_idempotency_header_through_client_is_422(client):
    _, grant = _settled_fixture(client)
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
    assert _count_rows(client.app, RateLimitCounter) == 0


def test_invalid_header_returns_422_even_when_budget_exhausted(app):
    # Header validation precedes the budget reservation: an illegal key
    # must surface as 422, never 429, and spend nothing.
    client = TestClient(app)
    _, grant = _settled_fixture(client)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    with app.state.session_factory() as session:
        session.add(
            RateLimitCounter(
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                window_start=now,
                count=app_module.GRANT_BUDGET_PER_MINUTE,
            )
        )
        session.commit()
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }
    status, _ = _raw_consume(app, grant["grant_id"], body, key_raw=b"")
    assert status == 422
    with app.state.session_factory() as session:
        row = session.query(RateLimitCounter).one()
        assert row.count == app_module.GRANT_BUDGET_PER_MINUTE


@pytest.mark.parametrize("length", [1, 64])
def test_boundary_length_keys_accepted(client, length):
    _, grant = _settled_fixture(client)
    response = _consume(client, grant["grant_id"], grant["capability"],
                        key="A" * length)
    assert response.status_code == 200, response.text
    assert _count_rows(client.app, ReleaseGrantConsumeIdempotencyRecord) == 1


def test_visible_punctuation_key_accepted(client):
    _, grant = _settled_fixture(client)
    response = _consume(
        client, grant["grant_id"], grant["capability"],
        key="~!@#$%^&*()_+-=[]{}|;:',.<>/?`",
    )
    assert response.status_code == 200, response.text


# --- first consumption -----------------------------------------------------


def test_first_keyed_consume_settles_and_persists_record(client, app):
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"], data_id="data-7").json()

    response = _consume(client, grant["grant_id"], grant["capability"],
                        key="consume-key-1")
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "grant_id",
        "decision_id",
        "data_id",
        "consumed",
        "consumed_at",
    }
    assert data["grant_id"] == grant["grant_id"]
    assert data["decision_id"] == decision["decision_id"]
    assert data["data_id"] == "data-7"
    assert data["consumed"] is True
    assert "capability" not in response.text

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"
        assert row.consumed_at is not None
        record = session.query(ReleaseGrantConsumeIdempotencyRecord).one()
        assert record.tenant_id == TENANT
        assert record.workload_id == WORKLOAD
        assert record.idempotency_key == "consume-key-1"
        assert record.grant_id == grant["grant_id"]
        assert len(record.request_fingerprint) == 64
        assert record.response_body.encode() == response.content
        # Exactly one consume timeline event (seq 2, after the issued
        # birth event) and one consumed audit row.
        events = (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
            .order_by(ReleaseGrantEvent.seq)
            .all()
        )
        assert [(e.seq, e.reason, e.new_status) for e in events] == [
            (1, "issued", "pending"),
            (2, "consume", "consumed"),
        ]
        # Issuance appends the pending audit; the winning consume appends
        # exactly one consumed audit and a replay appends neither.
        audits = (
            session.query(AuditEvent)
            .filter(
                AuditEvent.grant_id == grant["grant_id"],
                AuditEvent.status == "consumed",
            )
            .all()
        )
        assert len(audits) == 1
        assert record.created_at == row.consumed_at


def test_missing_header_keeps_legacy_duplicate_409(client):
    _, grant = _settled_fixture(client)
    first = _consume(client, grant["grant_id"], grant["capability"], key=None)
    assert first.status_code == 200
    second = _consume(client, grant["grant_id"], grant["capability"], key=None)
    assert second.status_code == 409
    assert _count_rows(client.app, ReleaseGrantConsumeIdempotencyRecord) == 0


# --- replay ----------------------------------------------------------------


def test_replay_returns_saved_200_verbatim_and_appends_nothing(client, app):
    _, grant = _settled_fixture(client)
    first = _consume(client, grant["grant_id"], grant["capability"],
                     key="replay-key")
    assert first.status_code == 200
    original_consumed_at = first.json()["consumed_at"]

    second = _consume(client, grant["grant_id"], grant["capability"],
                      key="replay-key")
    third = _consume(client, grant["grant_id"], grant["capability"],
                     key="replay-key")
    assert second.status_code == 200 and third.status_code == 200
    # Byte-for-byte identical: same identifiers and the original time.
    assert second.content == first.content
    assert third.content == first.content
    assert second.json()["consumed_at"] == original_consumed_at

    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 1
    with app.state.session_factory() as session:
        events = (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
            .count()
        )
        assert events == 2
        assert (
            session.query(AuditEvent)
            .filter(
                AuditEvent.grant_id == grant["grant_id"],
                AuditEvent.status == "consumed",
            )
            .count()
            == 1
        )
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"


def test_replay_returns_original_200_after_grant_expired(client, app):
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"], ttl_seconds=30).json()
    first = _consume(client, grant["grant_id"], grant["capability"],
                     key="replay-key")
    assert first.status_code == 200
    # The grant expires long after the successful consumption.
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        original_consumed_at = row.consumed_at
        session.commit()

    replay = _consume(client, grant["grant_id"], grant["capability"],
                      key="replay-key")
    assert replay.status_code == 200
    assert replay.content == first.content
    # No status/time rewrite, no new event.
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"
        assert row.consumed_at == original_consumed_at
        assert (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
            .count()
            == 2
        )


def test_replay_returns_original_200_after_later_state_change(client, app):
    _, grant = _settled_fixture(client)
    first = _consume(client, grant["grant_id"], grant["capability"],
                     key="replay-key")
    assert first.status_code == 200

    # Simulate an out-of-band later change the API itself can never make
    # (the grant is already terminal): the replay must still answer with
    # the stored 200 and must not rewrite the row back.
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.status = "revoked"
        tampered_at = datetime.now(timezone.utc) - timedelta(hours=2)
        row.consumed_at = tampered_at
        session.commit()

    replay = _consume(client, grant["grant_id"], grant["capability"],
                      key="replay-key")
    assert replay.status_code == 200
    assert replay.content == first.content
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        # Untouched by the replay.
        assert row.status == "revoked"
        assert row.consumed_at == tampered_at
        assert (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
            .count()
            == 2
        )


def test_replay_consumes_shared_budget_before_judging_idempotency(
    client, app
):
    # Admitted keyed requests draw one shared slot even when they turn
    # out to be replays: the first consume plus four replays use the
    # five-per-minute quota, and a fifth replay is a 429 that changes
    # nothing.
    _, grant = _settled_fixture(client)
    first = _consume(client, grant["grant_id"], grant["capability"],
                     key="budget-key")
    assert first.status_code == 200
    for _ in range(4):
        assert (
            _consume(client, grant["grant_id"], grant["capability"],
                     key="budget-key").status_code
            == 200
        )
    exhausted = _consume(client, grant["grant_id"], grant["capability"],
                         key="budget-key")
    assert exhausted.status_code == 429
    assert "retry_after_seconds" in exhausted.json()
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 1


# --- same-key conflicts and scope isolation --------------------------------


def test_same_key_different_grant_is_409_and_changes_nothing(client, app):
    decision = _decision(client)
    grant_a = _grant(client, decision["decision_id"], data_id="data-a").json()
    grant_b = _grant(client, decision["decision_id"], data_id="data-b").json()

    first = _consume(client, grant_a["grant_id"], grant_a["capability"],
                     key="shared")
    assert first.status_code == 200
    conflict = _consume(client, grant_b["grant_id"], grant_b["capability"],
                        key="shared")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == (
        "idempotency key reused with different request"
    )
    # The original consumption and saved response are untouched; grant B
    # stays pending and an exact replay still returns the stored 200.
    with app.state.session_factory() as session:
        row_b = session.get(ReleaseGrant, grant_b["grant_id"])
        assert row_b.status == "pending"
        assert (
            session.query(ReleaseGrantConsumeIdempotencyRecord).count() == 1
        )
    replay = _consume(client, grant_a["grant_id"], grant_a["capability"],
                      key="shared")
    assert replay.status_code == 200
    assert replay.content == first.content


def test_same_key_different_capability_is_409(client, app):
    decision = _decision(client)
    grant_a = _grant(client, decision["decision_id"], data_id="data-a").json()
    grant_b = _grant(client, decision["decision_id"], data_id="data-b").json()

    assert (
        _consume(client, grant_a["grant_id"], grant_a["capability"],
                 key="shared").status_code
        == 200
    )
    # A well-formed but different capability against the same grant: the
    # stored-record mismatch is judged before the capability would be
    # re-checked, so it is the idempotency 409 rather than 401, and
    # nothing changes.
    conflict = _consume(client, grant_a["grant_id"], grant_b["capability"],
                        key="shared")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == (
        "idempotency key reused with different request"
    )
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 1


def test_same_key_in_other_tenant_or_workload_is_independent(client, app):
    decision_a = _decision(client, TENANT, WORKLOAD)
    decision_b = _decision(client, "tenant-b", WORKLOAD)
    decision_c = _decision(client, TENANT, "workload-9")
    grant_a = _grant(client, decision_a["decision_id"], data_id="a").json()
    grant_b = _grant(
        client, decision_b["decision_id"], data_id="a",
        tenant="tenant-b",
    ).json()
    grant_c = _grant(
        client, decision_c["decision_id"], data_id="a",
        workload="workload-9",
    ).json()

    r1 = _consume(client, grant_a["grant_id"], grant_a["capability"],
                  key="shared-key")
    r2 = _consume(
        client, grant_b["grant_id"], grant_b["capability"], key="shared-key",
        tenant_id="tenant-b",
    )
    r3 = _consume(
        client, grant_c["grant_id"], grant_c["capability"], key="shared-key",
        workload_id="workload-9",
    )
    assert {r.status_code for r in (r1, r2, r3)} == {200}
    bodies = {r.content for r in (r1, r2, r3)}
    assert len(bodies) == 3
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 3
    # Each scope's replay resolves to its own stored consumption.
    assert (
        _consume(client, grant_b["grant_id"], grant_b["capability"],
                 key="shared-key", tenant_id="tenant-b").content
        == r2.content
    )
    assert (
        _consume(client, grant_c["grant_id"], grant_c["capability"],
                 key="shared-key", workload_id="workload-9").content
        == r3.content
    )


# --- failures do not occupy the key ----------------------------------------


def test_wrong_capability_failure_does_not_occupy_key(client, app):
    _, grant = _settled_fixture(client)
    wrong = ("B" if grant["capability"][0] != "B" else "C") + \
        grant["capability"][1:]
    failed = _consume(client, grant["grant_id"], wrong, key="retry-key")
    assert failed.status_code == 401
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 0
    # The key is free: the recovered, correct request succeeds normally.
    recovered = _consume(client, grant["grant_id"], grant["capability"],
                         key="retry-key")
    assert recovered.status_code == 200
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 1
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"
        assert (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
            .count()
            == 2
        )


def test_expired_grant_failure_does_not_occupy_key(client, app):
    decision = _decision(client)
    expired = _grant(client, decision["decision_id"], data_id="old").json()
    fresh = _grant(client, decision["decision_id"], data_id="new").json()
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, expired["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    failed = _consume(client, expired["grant_id"], expired["capability"],
                      key="retry-key")
    assert failed.status_code == 410
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 0
    # A 410 saved no key: the same key against a different (fresh) grant
    # is judged as a first request and succeeds.
    recovered = _consume(client, fresh["grant_id"], fresh["capability"],
                         key="retry-key")
    assert recovered.status_code == 200
    assert recovered.json()["grant_id"] == fresh["grant_id"]
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 1


def test_already_consumed_failure_does_not_occupy_key(client, app):
    decision = _decision(client)
    first_grant = _grant(client, decision["decision_id"], data_id="a").json()
    second_grant = _grant(client, decision["decision_id"], data_id="b").json()
    # Settled keyless: a later keyed presentation of the same grant is the
    # existing 409 and must not save the key.
    assert (
        _consume(client, first_grant["grant_id"],
                 first_grant["capability"], key=None).status_code
        == 200
    )
    assert (
        _consume(client, first_grant["grant_id"],
                 first_grant["capability"], key="retry-key").status_code
        == 409
    )
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 0
    # The key stays free for the still-pending second grant.
    recovered = _consume(client, second_grant["grant_id"],
                         second_grant["capability"], key="retry-key")
    assert recovered.status_code == 200
    assert recovered.json()["grant_id"] == second_grant["grant_id"]


def test_unknown_grant_failure_does_not_occupy_key(client, app):
    response = _consume(
        client,
        "00000000-0000-0000-0000-000000000000",
        "A" * 43,
        key="retry-key",
    )
    assert response.status_code == 404
    assert _count_rows(app, ReleaseGrantConsumeIdempotencyRecord) == 0


def test_existing_error_ordering_preserved_for_first_keyed_request(client):
    # 404 / 401 / 409 (consumed, then revoked) / 410 keep their existing
    # trigger conditions and order on the first keyed request.
    decision = _decision(client)

    missing = _consume(
        client, "00000000-0000-0000-0000-000000000000", "A" * 43, key="k"
    )
    assert missing.status_code == 404

    grant = _grant(client, decision["decision_id"], data_id="x").json()
    wrong = ("B" if grant["capability"][0] != "B" else "C") + \
        grant["capability"][1:]
    assert _consume(client, grant["grant_id"], wrong, key="k").status_code == 401

    consumed = _grant(client, decision["decision_id"], data_id="c").json()
    _consume(client, consumed["grant_id"], consumed["capability"], key=None)
    assert (
        _consume(client, consumed["grant_id"], consumed["capability"],
                 key="other-key").status_code
        == 409
    )

    expired = _grant(client, decision["decision_id"], data_id="e").json()
    with client.app.state.session_factory() as session:
        row = session.get(ReleaseGrant, expired["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
    assert (
        _consume(client, expired["grant_id"], expired["capability"],
                 key="yet-another-key").status_code
        == 410
    )


# --- concurrency -----------------------------------------------------------


def test_concurrent_same_key_retries_settle_once(app, monkeypatch):
    # Exercise the one-time settlement under same-key retries, not the
    # shared per-minute budget; raise the limiter so all eight bursts are
    # admitted and only atomic settlement/idempotency decide outcomes.
    monkeypatch.setattr(app_module, "GRANT_BUDGET_PER_MINUTE", 64)
    client = TestClient(app)
    _, grant = _settled_fixture(client)
    url = f"/v1/release-grants/{grant['grant_id']}/consume"
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }
    headers = {IDEMPOTENCY_HEADER: "race-key"}

    def consume():
        return TestClient(app).post(url, json=body, headers=headers)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: consume(), range(8)))

    assert [r.status_code for r in responses].count(200) == 8
    bodies = {r.content for r in responses}
    assert len(bodies) == 1
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"
        assert session.query(ReleaseGrant).count() == 1
        assert session.query(ReleaseGrantConsumeIdempotencyRecord).count() == 1
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


# --- restart and additive schema -------------------------------------------


def test_idempotency_record_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart-consume-idem.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    decision = _decision(client1)
    grant = _grant(client1, decision["decision_id"]).json()
    first = _consume(client1, grant["grant_id"], grant["capability"],
                     key="durable-key")
    assert first.status_code == 200
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    # Even after the grant has expired post-restart, the replay answers
    # with the original 200.
    with app2.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
    replay = _consume(client2, grant["grant_id"], grant["capability"],
                      key="durable-key")
    assert replay.status_code == 200
    assert replay.content == first.content
    with app2.state.session_factory() as session:
        assert session.query(ReleaseGrantConsumeIdempotencyRecord).count() == 1
    app2.state.engine.dispose()


def test_idempotency_table_is_created_additively_on_old_database(
    tmp_path, monkeypatch
):
    # A database written by a deployment that predates the table (and
    # which already holds grants) gains it on open, and a keyed consume
    # works against the upgraded database.
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/legacy.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    decision = _decision(client1)
    grant = _grant(client1, decision["decision_id"]).json()
    app1.state.engine.dispose()

    from sqlalchemy import text

    # Open the database with the new build once, then remove just the new
    # table to simulate a deployment that predates it.
    app2 = create_app(url)
    with app2.state.engine.begin() as conn:
        conn.execute(
            text("DROP TABLE release_grant_consume_idempotency_records")
        )
    app2.state.engine.dispose()

    app3 = create_app(url)
    client3 = TestClient(app3)
    response = _consume(client3, grant["grant_id"], grant["capability"],
                        key="after-upgrade")
    assert response.status_code == 200
    with app3.state.session_factory() as session:
        assert session.query(ReleaseGrantConsumeIdempotencyRecord).count() == 1
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"
    app3.state.engine.dispose()


# --- secrecy ---------------------------------------------------------------


def test_keyed_consume_never_persists_plaintext_capability(client, app):
    _, grant = _settled_fixture(client)
    capability = grant["capability"]
    first = _consume(client, grant["grant_id"], capability, key="secret-key")
    replay = _consume(client, grant["grant_id"], capability, key="secret-key")
    assert capability not in first.text
    assert capability not in replay.text
    with app.state.session_factory() as session:
        record = session.query(ReleaseGrantConsumeIdempotencyRecord).one()
        values = {
            c.name: getattr(record, c.name) for c in record.__table__.columns
        }
        for name, value in values.items():
            assert capability not in str(value), f"capability leaked in {name}"
        # The fingerprint matches the digest-based shape, not the
        # plaintext: hashing the capability itself must not reproduce it.
        assert record.request_fingerprint != hashlib.sha256(
            capability.encode()
        ).hexdigest()
        assert not hasattr(record, "capability")
