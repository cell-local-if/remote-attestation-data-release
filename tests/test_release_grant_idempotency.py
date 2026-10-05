"""Tests for the optional Idempotency-Key header on POST /v1/release-grants.

Contract under test:

* the header is optional; a missing header keeps the existing
  one-grant-per-request semantics (every unkeyed request mints a grant);
* a present header must occur exactly once and carry 1..64 visible ASCII
  characters; an empty value, surrounding whitespace, a control or
  non-ASCII character, an over-long value or a duplicated header line is
  an indistinguishable 422 raised before any state is read or written;
* the first legal keyed request mints the grant, its pending timeline
  event, its pending audit event and the idempotency record in one
  atomic commit, persists only the capability digest and returns the
  capability exactly once (201);
* a same-key same-scope replay of the same normalized request (decision,
  data, TTL and scope) is a stable 409 "release grant already issued":
  no second grant, no new capability, no event or audit row, and the
  original grant is untouched — even after it was consumed or revoked;
* a same-key request with different normalized content is a stable 409
  "idempotency key conflict"; the same key in another tenant or
  workload is independent;
* only a successful issuance occupies the key: 404/409 judgements save
  no record, so a recovered request re-judges normally;
* concurrent same-key submissions mint at most one grant: exactly one
  201, every loser a 409, one event and one audit row;
* the record survives restarts, is created additively on databases that
  predate it, and stores only the scope, the key, the request
  fingerprint and the grant id — never the capability.
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

from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    ReleaseGrant,
    ReleaseGrantEvent,
    ReleaseGrantIdempotencyRecord,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"
IDEMPOTENCY_HEADER = "Idempotency-Key"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    application = create_app(f"sqlite:///{tmp_path}/create-idem.db")
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


def _decision(client, tenant=TENANT, workload=WORKLOAD, satisfied=True):
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
            "rule": {"claim": "m", "equals": "x" if satisfied else "y"},
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


def _create(client, decision_id, *, key=..., tenant=TENANT,
            workload=WORKLOAD, data_id="data-1", ttl_seconds=None):
    body = {
        "tenant_id": tenant,
        "workload_id": workload,
        "decision_id": decision_id,
        "data_id": data_id,
    }
    if ttl_seconds is not None:
        body["ttl_seconds"] = ttl_seconds
    headers = None if key is ... else ({IDEMPOTENCY_HEADER: key} if key else None)
    return client.post("/v1/release-grants", json=body, headers=headers)


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


def _raw_create(app, body_obj, *, key_raw=b"key-1", duplicate_key=False):
    payload = json.dumps(body_obj).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"idempotency-key", key_raw),
    ]
    if duplicate_key:
        headers.append((b"idempotency-key", key_raw))
    return asyncio.run(_asgi_call(app, "/v1/release-grants", headers, payload))


def _count_rows(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


def _body(decision_id, *, tenant=TENANT, workload=WORKLOAD,
          data_id="data-1", ttl_seconds=300):
    return {
        "tenant_id": tenant,
        "workload_id": workload,
        "decision_id": decision_id,
        "data_id": data_id,
        "ttl_seconds": ttl_seconds,
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
def test_invalid_idempotency_key_is_422_and_writes_nothing(app, raw_key):
    client = TestClient(app)
    decision = _decision(client)
    status, raw = _raw_create(app, _body(decision["decision_id"]),
                              key_raw=raw_key)
    assert status == 422, raw
    assert json.loads(raw)["detail"] == "invalid idempotency key"
    # No grant, no record, no event, no audit row.
    assert _count_rows(app, ReleaseGrant) == 0
    assert _count_rows(app, ReleaseGrantIdempotencyRecord) == 0
    assert _count_rows(app, ReleaseGrantEvent) == 0
    assert _count_rows(app, AuditEvent) == 0


def test_duplicate_idempotency_header_is_422(app):
    client = TestClient(app)
    decision = _decision(client)
    status, raw = _raw_create(
        app, _body(decision["decision_id"]), key_raw=b"key-1",
        duplicate_key=True,
    )
    assert status == 422, raw
    assert json.loads(raw)["detail"] == "invalid idempotency key"
    assert _count_rows(app, ReleaseGrant) == 0
    assert _count_rows(app, ReleaseGrantIdempotencyRecord) == 0


def test_duplicate_idempotency_header_through_client_is_422(client):
    decision = _decision(client)
    response = client.post(
        "/v1/release-grants",
        json=_body(decision["decision_id"]),
        headers=[
            (IDEMPOTENCY_HEADER, "key-1"),
            (IDEMPOTENCY_HEADER, "key-1"),
        ],
    )
    assert response.status_code == 422
    assert _count_rows(client.app, ReleaseGrant) == 0
    assert _count_rows(client.app, ReleaseGrantIdempotencyRecord) == 0


@pytest.mark.parametrize("length", [1, 64])
def test_boundary_length_keys_accepted(client, length):
    decision = _decision(client)
    response = _create(client, decision["decision_id"], key="A" * length)
    assert response.status_code == 201, response.text
    assert _count_rows(client.app, ReleaseGrantIdempotencyRecord) == 1


def test_visible_punctuation_key_accepted(client):
    decision = _decision(client)
    response = _create(
        client, decision["decision_id"],
        key="~!@#$%^&*()_+-=[]{}|;:',.<>/?`",
    )
    assert response.status_code == 201, response.text


# --- first issuance ----------------------------------------------------------


def test_first_keyed_create_mints_and_persists_record(client, app):
    decision = _decision(client)
    response = _create(
        client, decision["decision_id"], key="create-key-1",
        data_id="data-7", ttl_seconds=120,
    )
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
    assert data["decision_id"] == decision["decision_id"]
    assert data["data_id"] == "data-7"
    assert data["pending"] is True
    assert len(data["capability"]) == 43

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, data["grant_id"])
        assert row.status == "pending"
        record = session.query(ReleaseGrantIdempotencyRecord).one()
        assert record.tenant_id == TENANT
        assert record.workload_id == WORKLOAD
        assert record.idempotency_key == "create-key-1"
        assert record.grant_id == data["grant_id"]
        assert len(record.request_fingerprint) == 64
        # Only scope, key, fingerprint and grant id are stored.
        assert set(c.name for c in record.__table__.columns) == {
            "record_id",
            "tenant_id",
            "workload_id",
            "idempotency_key",
            "grant_id",
            "request_fingerprint",
            "created_at",
        }
        # Exactly one issued timeline event and one pending audit row.
        events = (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.grant_id == data["grant_id"])
            .all()
        )
        assert [(e.seq, e.reason, e.new_status) for e in events] == [
            (1, "issued", "pending")
        ]
        audits = (
            session.query(AuditEvent)
            .filter(
                AuditEvent.grant_id == data["grant_id"],
                AuditEvent.status == "pending",
            )
            .all()
        )
        assert len(audits) == 1


def test_missing_header_keeps_legacy_unkeyed_semantics(client):
    decision = _decision(client)
    first = _create(client, decision["decision_id"], key=None)
    second = _create(client, decision["decision_id"], key=None)
    assert first.status_code == 201 and second.status_code == 201
    # Every unkeyed request mints its own grant and capability.
    assert first.json()["grant_id"] != second.json()["grant_id"]
    assert first.json()["capability"] != second.json()["capability"]
    assert _count_rows(client.app, ReleaseGrant) == 2
    assert _count_rows(client.app, ReleaseGrantIdempotencyRecord) == 0


# --- replay -------------------------------------------------------------------


def test_replay_same_content_is_409_and_mints_nothing(client, app):
    decision = _decision(client)
    first = _create(client, decision["decision_id"], key="replay-key")
    assert first.status_code == 201

    second = _create(client, decision["decision_id"], key="replay-key")
    third = _create(client, decision["decision_id"], key="replay-key")
    assert second.status_code == 409
    assert second.json()["detail"] == "release grant already issued"
    assert third.status_code == 409
    assert third.json()["detail"] == "release grant already issued"
    # No capability is ever returned again.
    assert "capability" not in second.text
    assert first.json()["capability"] not in second.text

    assert _count_rows(app, ReleaseGrant) == 1
    assert _count_rows(app, ReleaseGrantIdempotencyRecord) == 1
    with app.state.session_factory() as session:
        assert (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.grant_id == first.json()["grant_id"])
            .count()
            == 1
        )
        assert (
            session.query(AuditEvent)
            .filter(AuditEvent.grant_id == first.json()["grant_id"])
            .count()
            == 1
        )
        row = session.get(ReleaseGrant, first.json()["grant_id"])
        assert row.status == "pending"


def test_replay_is_409_even_after_grant_is_consumed(client, app):
    decision = _decision(client)
    first = _create(client, decision["decision_id"], key="replay-key")
    assert first.status_code == 201
    grant = first.json()
    consumed = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )
    assert consumed.status_code == 200

    replay = _create(client, decision["decision_id"], key="replay-key")
    assert replay.status_code == 409
    assert replay.json()["detail"] == "release grant already issued"
    # The settled grant is untouched by the replay.
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"
        assert (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
            .count()
            == 2
        )


def test_replay_is_409_even_after_grant_is_revoked(client, app):
    decision = _decision(client)
    first = _create(client, decision["decision_id"], key="replay-key")
    grant = first.json()
    revoked = client.post(
        f"/v1/release-grants/{grant['grant_id']}/revoke",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )
    assert revoked.status_code == 200

    replay = _create(client, decision["decision_id"], key="replay-key")
    assert replay.status_code == 409
    assert replay.json()["detail"] == "release grant already issued"
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "revoked"


# --- same-key conflicts and scope isolation -----------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("data_id", "data-other"),
        ("ttl_seconds", 600),
    ],
)
def test_same_key_different_content_is_409_conflict(client, app, field, value):
    decision = _decision(client)
    first = _create(client, decision["decision_id"], key="shared")
    assert first.status_code == 201

    conflict = _create(
        client, decision["decision_id"], key="shared", **{field: value}
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency key conflict"
    assert _count_rows(app, ReleaseGrant) == 1
    assert _count_rows(app, ReleaseGrantIdempotencyRecord) == 1


def test_same_key_different_decision_is_409_conflict(client, app):
    decision_a = _decision(client)
    decision_b = _decision(client)
    first = _create(client, decision_a["decision_id"], key="shared")
    assert first.status_code == 201

    conflict = _create(client, decision_b["decision_id"], key="shared")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency key conflict"
    assert _count_rows(app, ReleaseGrant) == 1


def test_same_key_in_other_tenant_or_workload_is_independent(client, app):
    decision_a = _decision(client, TENANT, WORKLOAD)
    decision_b = _decision(client, "tenant-b", WORKLOAD)
    decision_c = _decision(client, TENANT, "workload-9")

    r1 = _create(client, decision_a["decision_id"], key="shared-key")
    r2 = _create(client, decision_b["decision_id"], key="shared-key",
                 tenant="tenant-b")
    r3 = _create(client, decision_c["decision_id"], key="shared-key",
                 workload="workload-9")
    assert {r.status_code for r in (r1, r2, r3)} == {201}
    grant_ids = {r.json()["grant_id"] for r in (r1, r2, r3)}
    assert len(grant_ids) == 3
    assert _count_rows(app, ReleaseGrantIdempotencyRecord) == 3
    # Each scope's replay resolves to its own stored issuance.
    replay = _create(client, decision_b["decision_id"], key="shared-key",
                     tenant="tenant-b")
    assert replay.status_code == 409
    assert replay.json()["detail"] == "release grant already issued"


# --- failures do not occupy the key -------------------------------------------


def test_unknown_decision_failure_does_not_occupy_key(client, app):
    failed = _create(
        client, "00000000-0000-0000-0000-000000000000", key="retry-key"
    )
    assert failed.status_code == 404
    assert failed.json()["detail"] == "decision not found"
    assert _count_rows(app, ReleaseGrantIdempotencyRecord) == 0
    # The key is free: the recovered, legal request succeeds normally.
    decision = _decision(client)
    recovered = _create(client, decision["decision_id"], key="retry-key")
    assert recovered.status_code == 201
    assert _count_rows(app, ReleaseGrantIdempotencyRecord) == 1


def test_denied_decision_failure_does_not_occupy_key(client, app):
    denied = _decision(client, satisfied=False)
    assert denied["status"] == "denied"
    failed = _create(client, denied["decision_id"], key="retry-key")
    assert failed.status_code == 409
    assert failed.json()["detail"] == "decision is not allowed"
    assert _count_rows(app, ReleaseGrantIdempotencyRecord) == 0
    # The key stays free for a legal request.
    allowed = _decision(client)
    recovered = _create(client, allowed["decision_id"], key="retry-key")
    assert recovered.status_code == 201
    assert _count_rows(app, ReleaseGrantIdempotencyRecord) == 1


def test_cross_scope_decision_failure_does_not_occupy_key(client, app):
    decision = _decision(client)
    failed = _create(
        client, decision["decision_id"], key="retry-key", tenant="tenant-b"
    )
    assert failed.status_code == 404
    assert _count_rows(app, ReleaseGrantIdempotencyRecord) == 0
    recovered = _create(client, decision["decision_id"], key="retry-key")
    assert recovered.status_code == 201


def test_existing_error_ordering_preserved_for_first_keyed_request(client):
    # 404 (unknown decision) and 409 (denied decision) keep their existing
    # trigger conditions on the first keyed request.
    missing = _create(
        client, "00000000-0000-0000-0000-000000000000", key="k"
    )
    assert missing.status_code == 404

    denied = _decision(client, satisfied=False)
    assert _create(client, denied["decision_id"], key="k").status_code == 409

    allowed = _decision(client)
    assert _create(client, allowed["decision_id"], key="k").status_code == 201


# --- concurrency ---------------------------------------------------------------


def test_concurrent_same_key_requests_mint_at_most_one_grant(app):
    client = TestClient(app)
    decision = _decision(client)
    body = _body(decision["decision_id"])
    headers = {IDEMPOTENCY_HEADER: "race-key"}

    def create():
        return TestClient(app).post(
            "/v1/release-grants", json=body, headers=headers
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: create(), range(8)))

    assert [r.status_code for r in responses].count(201) == 1
    assert [r.status_code for r in responses].count(409) == 7
    capabilities = {
        r.json()["capability"] for r in responses if r.status_code == 201
    }
    assert len(capabilities) == 1
    assert all(
        r.json()["detail"] == "release grant already issued"
        for r in responses
        if r.status_code == 409
    )
    with app.state.session_factory() as session:
        assert session.query(ReleaseGrant).count() == 1
        assert session.query(ReleaseGrantIdempotencyRecord).count() == 1
        assert session.query(ReleaseGrantEvent).count() == 1
        assert session.query(AuditEvent).count() == 1


# --- restart and additive schema ------------------------------------------------


def test_idempotency_record_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart-create-idem.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    decision = _decision(client1)
    first = _create(client1, decision["decision_id"], key="durable-key")
    assert first.status_code == 201
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    replay = _create(client2, decision["decision_id"], key="durable-key")
    assert replay.status_code == 409
    assert replay.json()["detail"] == "release grant already issued"
    conflict = _create(
        client2, decision["decision_id"], key="durable-key",
        data_id="data-other",
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency key conflict"
    with app2.state.session_factory() as session:
        assert session.query(ReleaseGrant).count() == 1
        assert session.query(ReleaseGrantIdempotencyRecord).count() == 1
    app2.state.engine.dispose()


def test_idempotency_table_is_created_additively_on_old_database(
    tmp_path, monkeypatch
):
    # A database written by a deployment that predates the table (and
    # which already holds grants and audit rows) gains it on open, and a
    # keyed create works against the upgraded database.
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/legacy.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    decision = _decision(client1)
    unkeyed = _create(client1, decision["decision_id"], key=None)
    assert unkeyed.status_code == 201
    app1.state.engine.dispose()

    from sqlalchemy import text

    # Open the database with the new build once, then remove just the new
    # table to simulate a deployment that predates it.
    app2 = create_app(url)
    with app2.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE release_grant_idempotency_records"))
    app2.state.engine.dispose()

    app3 = create_app(url)
    client3 = TestClient(app3)
    response = _create(client3, decision["decision_id"], key="after-upgrade")
    assert response.status_code == 201
    with app3.state.session_factory() as session:
        assert session.query(ReleaseGrant).count() == 2
        assert session.query(ReleaseGrantIdempotencyRecord).count() == 1
        # The pre-existing grant and its audit row survived the upgrade.
        assert (
            session.query(AuditEvent)
            .filter(AuditEvent.grant_id == unkeyed.json()["grant_id"])
            .count()
            == 1
        )
    app3.state.engine.dispose()


# --- secrecy -------------------------------------------------------------------


def test_keyed_create_never_persists_plaintext_capability(client, app):
    decision = _decision(client)
    first = _create(client, decision["decision_id"], key="secret-key")
    assert first.status_code == 201
    capability = first.json()["capability"]
    replay = _create(client, decision["decision_id"], key="secret-key")
    assert capability not in replay.text
    with app.state.session_factory() as session:
        record = session.query(ReleaseGrantIdempotencyRecord).one()
        values = {
            c.name: getattr(record, c.name) for c in record.__table__.columns
        }
        for name, value in values.items():
            assert capability not in str(value), f"capability leaked in {name}"
        # The fingerprint covers the request identity, not the
        # capability: hashing the capability itself must not reproduce it.
        assert record.request_fingerprint != hashlib.sha256(
            capability.encode()
        ).hexdigest()
        assert not hasattr(record, "capability")
