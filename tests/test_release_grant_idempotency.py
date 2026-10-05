"""Tests for the optional Idempotency-Key header on POST /v1/release-grants.

Contract under test:

* the header is optional; a missing header keeps the existing
  one-grant-per-request semantics (multiple independent grants may
  still be minted for the same decision);
* a present header must occur exactly once and carry 1..64 visible
  ASCII characters; an empty value, surrounding whitespace, a control
  or non-ASCII character, an over-long value or a duplicated header
  line is an indistinguishable 422 detail "invalid idempotency key"
  raised after the body validation and before any decision read or
  write, so it never mints a grant, generates a capability or writes a
  record;
* the idempotency scope is (tenant_id, workload_id, key); the first
  legal request — after the decision and TTL checks pass — atomically
  writes the grant, its pending lifecycle event, its compliance audit
  event and the idempotency record, persisting only the capability's
  SHA-256 digest, and returns the capability exactly once;
* every later same-scope same-key request is a stable 409 detail
  "release grant already issued": it never mints a second grant, never
  generates or returns a new capability, appends no event or audit row
  and spends no issuance budget — even after the first grant was
  consumed, revoked or expired;
* a same-scope same-key request whose normalized content (decision_id,
  data_id, ttl_seconds or a scope field) differs is a distinct stable
  409 detail "idempotency key conflict"; the same key in another tenant
  or workload is independent;
* a failed first attempt (404/409 judgement or a write failure) leaves
  neither a grant nor a record, so the same legal request later
  succeeds normally;
* concurrent same-key requests settle with exactly one 201, every
  loser receiving a 409;
* the record survives restarts and stores only the scope, key, request
  fingerprint, grant id and creation time — never the capability
  plaintext, its digest, a response body or any payload/key material;
* the keyless create, query, consume, revoke, release, audit and trace
  entry points are unchanged.
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
from sqlalchemy import select

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    RateLimitCounter,
    ReleaseGrant,
    ReleaseGrantEvent,
    ReleaseGrantIdempotencyRecord,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
IDEMPOTENCY_HEADER = "Idempotency-Key"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    application = create_app(f"sqlite:///{tmp_path}/grant-idem.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _mac(scope: str, nonce: str, claims: dict) -> str:
    key = hmac.new(
        SECRET.encode("utf-8"),
        scope.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _decision(client, *, tenant=TENANT, workload=WORKLOAD, satisfied=True):
    """Drive challenge -> evidence -> verify -> decision; return its id."""
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    claims = {"m": "x"} if satisfied else {"m": "other"}
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": claims,
            "mac": _mac(f"{tenant}:{workload}", created["nonce"], claims),
        }
    )
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
    rule = (
        {"claim": "m", "equals": "x"}
        if satisfied
        else {"claim": "m", "equals": "blocked"}
    )
    policy = client.post(
        "/v1/policies",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "name": "release",
            "rule": rule,
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
    return decided.json()["decision_id"]


def _grant(client, decision_id, *, key=None, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision_id,
        "data_id": DATA_ID,
    }
    body.update(fields)
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else None
    return client.post("/v1/release-grants", json=body, headers=headers)


def _consume(client, grant_id, capability, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release-grants/{grant_id}/consume", json=body)


def _count(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


def _records(app):
    with app.state.session_factory() as session:
        return session.query(ReleaseGrantIdempotencyRecord).all()


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


def _raw_grant(app, body_obj, *, key_raw=b"key-1", duplicate_key=False):
    payload = json.dumps(body_obj).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"idempotency-key", key_raw),
    ]
    if duplicate_key:
        headers.append((b"idempotency-key", key_raw))
    return asyncio.run(_asgi_call(app, "/v1/release-grants", headers, payload))


def _valid_body(decision_id):
    return {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision_id,
        "data_id": DATA_ID,
    }


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
def test_invalid_idempotency_key_is_422_before_any_state(app, raw_key):
    body = _valid_body("00000000-0000-0000-0000-000000000000")
    status, raw = _raw_grant(app, body, key_raw=raw_key)
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    assert _count(app, ReleaseGrant) == 0
    assert _count(app, ReleaseGrantIdempotencyRecord) == 0
    assert _count(app, AuditEvent) == 0
    assert _count(app, RateLimitCounter) == 0


def test_duplicate_idempotency_header_is_422(app):
    body = _valid_body("00000000-0000-0000-0000-000000000000")
    status, raw = _raw_grant(app, body, duplicate_key=True)
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    assert _count(app, ReleaseGrant) == 0
    assert _count(app, ReleaseGrantIdempotencyRecord) == 0


def test_duplicate_idempotency_header_through_client_is_422(client, app):
    decision_id = _decision(client)
    response = client.post(
        "/v1/release-grants",
        json=_valid_body(decision_id),
        headers=[
            (IDEMPOTENCY_HEADER, "key-1"),
            (IDEMPOTENCY_HEADER, "key-1"),
        ],
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid idempotency key"}
    assert _count(app, ReleaseGrant) == 0


@pytest.mark.parametrize("length", [1, 64])
def test_boundary_length_keys_accepted(app, client, length):
    decision_id = _decision(client)
    response = _grant(client, decision_id, key="A" * length, data_id=f"d-{length}")
    assert response.status_code == 201, response.text
    assert _count(app, ReleaseGrantIdempotencyRecord) == 1


def test_visible_punctuation_key_accepted(client):
    decision_id = _decision(client)
    response = _grant(
        client, decision_id, key="~!@#$%^&*()_+-=[]{}|;:',.<>/?`"
    )
    assert response.status_code == 201, response.text


def test_invalid_key_is_not_consumed_and_later_valid_use_succeeds(app, client):
    decision_id = _decision(client)
    body = _valid_body(decision_id)
    assert _raw_grant(app, body, key_raw=b"")[0] == 422
    assert _raw_grant(app, body, key_raw=b"bad key")[0] == 422
    accepted = _grant(client, decision_id, key="key-1")
    assert accepted.status_code == 201
    assert _count(app, ReleaseGrant) == 1
    assert _count(app, ReleaseGrantIdempotencyRecord) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"decision_id": ""},
        {"data_id": ""},
        {"tenant_id": 1},
        {"ttl_seconds": 29},
        {"ttl_seconds": 901},
        {"ttl_seconds": "300"},
        {"ttl_seconds": True},
    ],
)
def test_body_validation_still_fails_with_valid_idempotency_key(
    app, client, overrides
):
    decision_id = _decision(client)
    body = _valid_body(decision_id)
    body.update(overrides)
    response = client.post(
        "/v1/release-grants",
        json=body,
        headers={IDEMPOTENCY_HEADER: "key-1"},
    )
    assert response.status_code == 422
    assert _count(app, ReleaseGrant) == 0
    assert _count(app, ReleaseGrantIdempotencyRecord) == 0


def test_invalid_key_with_denied_decision_is_still_header_422(app, client):
    # Header validation precedes every business judgement: the decision
    # is never read when the key is malformed.
    _decision(client, satisfied=False)
    body = _valid_body("00000000-0000-0000-0000-000000000000")
    status, raw = _raw_grant(app, body, key_raw=b"bad key")
    assert status == 422
    assert json.loads(raw) == {"detail": "invalid idempotency key"}


# --- first keyed issuance --------------------------------------------------


def test_first_keyed_grant_returns_201_with_capability_once(app, client):
    decision_id = _decision(client)
    response = _grant(client, decision_id, key="key-1")

    assert response.status_code == 201
    data = response.json()
    assert set(data) == {
        "grant_id",
        "decision_id",
        "data_id",
        "capability",
        "pending",
        "issued_at",
        "expires_at",
    }
    assert data["decision_id"] == decision_id
    assert data["data_id"] == DATA_ID
    assert data["pending"] is True
    capability = data["capability"]
    assert len(capability) == 43
    issued = datetime.fromisoformat(data["issued_at"])
    expires = datetime.fromisoformat(data["expires_at"])
    assert (expires - issued) == timedelta(seconds=300)

    assert _count(app, ReleaseGrant) == 1
    (record,) = _records(app)
    assert record.tenant_id == TENANT
    assert record.workload_id == WORKLOAD
    assert record.idempotency_key == "key-1"
    assert record.grant_id == data["grant_id"]
    assert record.request_fingerprint == (
        app_module._release_grant_request_fingerprint(
            TENANT, WORKLOAD, decision_id, DATA_ID, 300
        )
    )
    # Exactly one pending lifecycle event and one pending audit event,
    # both committed together with the grant and the record.
    with app.state.session_factory() as session:
        grant = session.get(ReleaseGrant, data["grant_id"])
        assert grant.status == "pending"
        assert grant.capability_digest == hashlib.sha256(
            capability.encode("ascii")
        ).hexdigest()
        events = session.scalars(select(ReleaseGrantEvent)).all()
        assert [(e.seq, e.new_status, e.reason) for e in events] == [
            (1, "pending", "issued")
        ]
        audits = session.scalars(select(AuditEvent)).all()
        assert len(audits) == 1
        assert audits[0].status == "pending"


def test_record_columns_hold_no_capability_or_secret(app, client):
    decision_id = _decision(client)
    response = _grant(client, decision_id, key="key-1")
    capability = response.json()["capability"]

    (record,) = _records(app)
    assert {c.name for c in record.__table__.columns} == {
        "record_id",
        "tenant_id",
        "workload_id",
        "idempotency_key",
        "grant_id",
        "request_fingerprint",
        "created_at",
    }
    values = {
        c.name: getattr(record, c.name) for c in record.__table__.columns
    }
    for name, value in values.items():
        assert capability not in str(value), f"capability leaked in {name}"
    # The fingerprint is the normalized request digest, not the
    # capability digest.
    assert record.request_fingerprint != hashlib.sha256(
        capability.encode("ascii")
    ).hexdigest()


def test_distinct_keys_mint_distinct_grants(app, client):
    decision_id = _decision(client)
    first = _grant(client, decision_id, key="key-1", data_id="d-a")
    second = _grant(client, decision_id, key="key-2", data_id="d-b")
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["grant_id"] != second.json()["grant_id"]
    assert first.json()["capability"] != second.json()["capability"]
    assert _count(app, ReleaseGrant) == 2
    assert _count(app, ReleaseGrantIdempotencyRecord) == 2


# --- at-most-once replay semantics -----------------------------------------


def test_same_key_replay_is_409_already_issued_and_mints_nothing(app, client):
    decision_id = _decision(client)
    first = _grant(client, decision_id, key="key-1")
    assert first.status_code == 201

    replay = _grant(client, decision_id, key="key-1")
    assert replay.status_code == 409
    assert replay.json() == {"detail": "release grant already issued"}
    # The one-time capability never appears again.
    assert b"capability" not in replay.content
    assert first.json()["capability"] not in replay.text

    # No second grant, record, lifecycle event or audit row.
    assert _count(app, ReleaseGrant) == 1
    assert _count(app, ReleaseGrantIdempotencyRecord) == 1
    assert _count(app, ReleaseGrantEvent) == 1
    assert _count(app, AuditEvent) == 1


def test_repeated_replays_all_stay_409(app, client):
    decision_id = _decision(client)
    assert _grant(client, decision_id, key="key-1").status_code == 201
    for _ in range(6):
        replay = _grant(client, decision_id, key="key-1")
        assert replay.status_code == 409
        assert replay.json() == {"detail": "release grant already issued"}
    assert _count(app, ReleaseGrant) == 1


def test_replay_is_409_after_grant_consumed(app, client):
    decision_id = _decision(client)
    created = _grant(client, decision_id, key="key-1").json()
    assert _consume(client, created["grant_id"], created["capability"]).status_code == 200

    replay = _grant(client, decision_id, key="key-1")
    assert replay.status_code == 409
    assert replay.json() == {"detail": "release grant already issued"}
    # Consumption added its own event/audit, but the replays added none.
    assert _count(app, ReleaseGrant) == 1
    assert _count(app, ReleaseGrantIdempotencyRecord) == 1
    assert _count(app, ReleaseGrantEvent) == 2
    assert _count(app, AuditEvent) == 2


def test_replay_is_409_after_grant_revoked(app, client):
    decision_id = _decision(client)
    created = _grant(client, decision_id, key="key-1").json()
    revoked = client.post(
        f"/v1/release-grants/{created['grant_id']}/revoke",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": created["capability"],
        },
    )
    assert revoked.status_code == 200

    replay = _grant(client, decision_id, key="key-1")
    assert replay.status_code == 409
    assert replay.json() == {"detail": "release grant already issued"}


def test_replay_is_409_after_grant_expired(app, client):
    decision_id = _decision(client)
    created = _grant(client, decision_id, key="key-1").json()
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, created["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    replay = _grant(client, decision_id, key="key-1")
    assert replay.status_code == 409
    assert replay.json() == {"detail": "release grant already issued"}


# --- same-key content conflicts --------------------------------------------


def test_same_key_different_data_id_is_409_conflict(app, client):
    decision_id = _decision(client)
    assert _grant(client, decision_id, key="key-1", data_id="d-a").status_code == 201

    conflict = _grant(client, decision_id, key="key-1", data_id="d-b")
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "idempotency key conflict"}
    assert b"capability" not in conflict.content
    assert _count(app, ReleaseGrant) == 1
    assert _count(app, ReleaseGrantIdempotencyRecord) == 1

    # The original request still reports "already issued" and nothing
    # was written by either replay.
    same = _grant(client, decision_id, key="key-1", data_id="d-a")
    assert same.status_code == 409
    assert same.json() == {"detail": "release grant already issued"}
    assert _count(app, ReleaseGrant) == 1


def test_same_key_different_ttl_is_409_conflict(app, client):
    decision_id = _decision(client)
    assert _grant(client, decision_id, key="key-1").status_code == 201

    conflict = _grant(client, decision_id, key="key-1", ttl_seconds=60)
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "idempotency key conflict"}
    assert _count(app, ReleaseGrant) == 1
    assert _count(app, ReleaseGrantEvent) == 1


def test_same_key_different_decision_is_409_conflict(app, client):
    decision_a = _decision(client)
    decision_b = _decision(client)
    assert _grant(client, decision_a, key="key-1").status_code == 201

    # Fingerprint comparison precedes the decision judgement, so even a
    # valid allowed second decision is a stable conflict.
    conflict = _grant(client, decision_b, key="key-1")
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "idempotency key conflict"}
    assert _count(app, ReleaseGrant) == 1

    # An unknown decision id under the same key is likewise the conflict
    # 409, not the decision-not-found 404.
    unknown = _grant(
        client, "00000000-0000-0000-0000-000000000000", key="key-1"
    )
    assert unknown.status_code == 409
    assert unknown.json() == {"detail": "idempotency key conflict"}


def test_same_key_in_another_scope_is_independent(app, client):
    decision_a = _decision(client, tenant=TENANT, workload=WORKLOAD)
    decision_b = _decision(client, tenant="tenant-b", workload=WORKLOAD)

    first = _grant(client, decision_a, key="key-1")
    assert first.status_code == 201
    other = _grant(
        client,
        decision_b,
        key="key-1",
        tenant_id="tenant-b",
    )
    assert other.status_code == 201
    assert other.json()["grant_id"] != first.json()["grant_id"]

    # Two independent scopes, two grants and two records.
    assert _count(app, ReleaseGrant) == 2
    assert _count(app, ReleaseGrantIdempotencyRecord) == 2

    # Each scope replays its own key as "already issued".
    assert _grant(client, decision_a, key="key-1").status_code == 409
    assert (
        _grant(client, decision_b, key="key-1", tenant_id="tenant-b").status_code
        == 409
    )


# --- judgement failures leave the key free ---------------------------------


def test_unknown_decision_leaves_no_record_and_key_stays_free(app, client):
    response = _grant(
        client, "00000000-0000-0000-0000-000000000000", key="key-1"
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "decision not found"}
    assert _count(app, ReleaseGrant) == 0
    assert _count(app, ReleaseGrantIdempotencyRecord) == 0

    # Once a real allowed decision exists, the same key is judged fresh
    # and succeeds.
    decision_id = _decision(client)
    created = _grant(client, decision_id, key="key-1")
    assert created.status_code == 201
    assert _count(app, ReleaseGrant) == 1
    assert _count(app, ReleaseGrantIdempotencyRecord) == 1


def test_denied_decision_leaves_no_record_and_writes_nothing(app, client):
    decision_id = _decision(client, satisfied=False)
    response = _grant(client, decision_id, key="key-1")
    assert response.status_code == 409
    assert response.json() == {"detail": "decision is not allowed"}
    assert b"capability" not in response.content
    assert _count(app, ReleaseGrant) == 0
    assert _count(app, ReleaseGrantIdempotencyRecord) == 0
    assert _count(app, AuditEvent) == 0


def test_cross_scope_decision_leaves_no_record(app, client):
    decision_id = _decision(client)
    response = _grant(client, decision_id, key="key-1", tenant_id="tenant-b")
    assert response.status_code == 404
    assert _count(app, ReleaseGrantIdempotencyRecord) == 0


# --- no rate-limit budget is spent by issuance or replays ------------------


def test_keyed_issuance_and_replays_write_no_rate_limit_counter(app, client):
    decision_id = _decision(client)
    assert _grant(client, decision_id, key="key-1").status_code == 201
    for _ in range(7):
        assert _grant(client, decision_id, key="key-1").status_code == 409
    # Issuance has never shared the consume/revoke/release budget.
    assert _count(app, RateLimitCounter) == 0


# --- concurrency -----------------------------------------------------------


def test_concurrent_same_key_requests_mint_exactly_one_grant(app):
    with TestClient(app) as setup_client:
        decision_id = _decision(setup_client)
    body = _valid_body(decision_id)

    def submit(_):
        with TestClient(app) as thread_client:
            return thread_client.post(
                "/v1/release-grants",
                json=body,
                headers={IDEMPOTENCY_HEADER: "key-1"},
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(submit, range(8)))

    statuses = [response.status_code for response in responses]
    assert statuses.count(201) == 1
    assert statuses.count(409) == 7
    for response in responses:
        if response.status_code == 409:
            assert response.json() == {"detail": "release grant already issued"}
    winners = [r for r in responses if r.status_code == 201]
    assert len({r.json()["grant_id"] for r in winners}) == 1
    # Only the winner's response carries a capability.
    assert sum(b"capability" in r.content for r in responses) == 1

    assert _count(app, ReleaseGrant) == 1
    assert _count(app, ReleaseGrantIdempotencyRecord) == 1
    assert _count(app, ReleaseGrantEvent) == 1
    assert _count(app, AuditEvent) == 1


def test_concurrent_requests_with_mixed_content_settle_one_201(app):
    with TestClient(app) as setup_client:
        decision_id = _decision(setup_client)
    base = _valid_body(decision_id)
    divergent = dict(base, data_id="d-b")

    def submit(index):
        chosen = divergent if index % 2 else base
        with TestClient(app) as thread_client:
            return thread_client.post(
                "/v1/release-grants",
                json=chosen,
                headers={IDEMPOTENCY_HEADER: "key-1"},
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(submit, range(8)))

    statuses = [response.status_code for response in responses]
    assert statuses.count(201) == 1
    assert statuses.count(409) == 7
    details = {response.json()["detail"] for response in responses
               if response.status_code == 409}
    assert details <= {
        "release grant already issued",
        "idempotency key conflict",
    }
    assert _count(app, ReleaseGrant) == 1
    assert _count(app, ReleaseGrantIdempotencyRecord) == 1


# --- restart ---------------------------------------------------------------


def test_idempotency_record_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart-grant-idem.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    decision_id = _decision(client1)
    created = _grant(client1, decision_id, key="key-1")
    assert created.status_code == 201
    grant = created.json()
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    # The verdict after restart is unchanged and still returns no
    # capability; the record needs no migration to remain authoritative.
    replay = _grant(client2, decision_id, key="key-1")
    assert replay.status_code == 409
    assert replay.json() == {"detail": "release grant already issued"}
    conflict = _grant(client2, decision_id, key="key-1", data_id="d-b")
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "idempotency key conflict"}

    # The original grant is still fully operational with the capability
    # returned exactly once, before the restart.
    consumed = _consume(client2, grant["grant_id"], grant["capability"])
    assert consumed.status_code == 200
    app2.state.engine.dispose()


# --- keyless and downstream behavior is unchanged --------------------------


def test_missing_header_keeps_legacy_multi_grant_semantics(app, client):
    decision_id = _decision(client)
    first = _grant(client, decision_id, data_id="d-a")
    second = _grant(client, decision_id, data_id="d-b")
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["grant_id"] != second.json()["grant_id"]
    assert first.json()["capability"] != second.json()["capability"]
    assert _count(app, ReleaseGrant) == 2
    # Keyless issuance never writes an idempotency record.
    assert _count(app, ReleaseGrantIdempotencyRecord) == 0


def test_keyed_and_keyless_grants_coexist(app, client):
    decision_id = _decision(client)
    keyed = _grant(client, decision_id, key="key-1", data_id="d-a")
    keyless = _grant(client, decision_id, data_id="d-b")
    assert keyed.status_code == 201
    assert keyless.status_code == 201
    # The keyed replay still mints nothing; keyless issuance stays free.
    assert _grant(client, decision_id, key="key-1", data_id="d-a").status_code == 409
    assert _grant(client, decision_id, data_id="d-c").status_code == 201
    assert _count(app, ReleaseGrant) == 3
    assert _count(app, ReleaseGrantIdempotencyRecord) == 1


def test_keyed_grant_consume_revoke_release_and_queries_unchanged(app, client):
    decision_id = _decision(client)
    created = _grant(client, decision_id, key="key-1", data_id="d-7").json()

    # Listing, trace and events see the keyed grant exactly like any
    # other grant.
    listing = client.get(
        "/v1/release-grants",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert listing.status_code == 200
    assert [g["grant_id"] for g in listing.json()["grants"]] == [
        created["grant_id"]
    ]
    events = client.get(
        f"/v1/release-grants/{created['grant_id']}/events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert events.status_code == 200

    # Consumption with the one-time capability settles the grant
    # normally, exactly as on the keyless path.
    consumed = _consume(client, created["grant_id"], created["capability"])
    assert consumed.status_code == 200
    assert consumed.json()["grant_id"] == created["grant_id"]
    again = _consume(client, created["grant_id"], created["capability"])
    assert again.status_code == 409


def test_independent_keyed_grant_can_still_be_revoked(app, client):
    decision_id = _decision(client)
    created = _grant(client, decision_id, key="key-1").json()
    response = client.post(
        f"/v1/release-grants/{created['grant_id']}/revoke",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": created["capability"],
        },
    )
    assert response.status_code == 200
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, created["grant_id"])
        assert row.status == "revoked"
        assert row.revoked_at is not None
