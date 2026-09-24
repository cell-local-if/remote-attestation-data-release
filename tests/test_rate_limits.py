"""Tests for the per-scope per-minute admission budget.

The release-grant consume, revoke and payload-release endpoints share a
budget of five field-valid requests per (tenant, workload) scope and UTC
minute. Requests that fail field validation (422) never consume budget;
every syntactically valid, scoped request consumes exactly one unit
regardless of its business outcome (404/401/409/410/500). The sixth
valid request in the same window is rejected with 429 and a compact
``retry_after_seconds`` body, writing no state and no audit.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import proof_release.app as app_module
from proof_release.app import create_app
from proof_release.db import AuditEvent, DataEnvelope, ReleaseGrant
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
MASTER_KEY = b64url_encode(b"0123456789abcdef0123456789abcdef")
PAYLOAD = "rate limited secret payload"

#: A well-formed capability that matches nothing.
UNKNOWN_CAPABILITY = "A" * 43
#: A well-formed grant identifier that matches nothing.
UNKNOWN_GRANT = "00000000-0000-0000-0000-000000000000"


@pytest.fixture()
def clock(monkeypatch):
    """A controllable UTC clock, started on a minute boundary."""
    box = {"now": datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc)}
    monkeypatch.setattr(app_module, "_utcnow", lambda: box["now"])
    return box


@pytest.fixture()
def app(tmp_path, monkeypatch, clock):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/rate.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _mac(nonce: str, claims: dict) -> str:
    key = hmac.new(
        SECRET.encode("utf-8"),
        f"{TENANT}:{WORKLOAD}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _evidence(nonce: str) -> str:
    claims = {"m": "x"}
    return json.dumps({"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)})


def _decision(client, tenant=TENANT, workload=WORKLOAD):
    created = client.post(
        "/v1/challenges", json={"tenant_id": tenant, "workload_id": workload}
    ).json()
    evidence = _evidence(created["nonce"])
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


def _envelope(client, data_id=DATA_ID, payload=PAYLOAD):
    response = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": data_id,
            "payload": payload,
        },
    )
    assert response.status_code == 201
    return response.json()


def _grant(client, decision_id, data_id=DATA_ID, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision_id,
        "data_id": data_id,
    }
    body.update(fields)
    response = client.post("/v1/release-grants", json=body)
    assert response.status_code == 201
    return response.json()


def _consume(client, grant_id=UNKNOWN_GRANT, capability=UNKNOWN_CAPABILITY, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release-grants/{grant_id}/consume", json=body)


def _revoke(client, grant_id=UNKNOWN_GRANT, capability=UNKNOWN_CAPABILITY, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release-grants/{grant_id}/revoke", json=body)


def _release(client, grant_id=UNKNOWN_GRANT, capability=UNKNOWN_CAPABILITY, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release/{grant_id}", json=body)


# ---------------------------------------------------------------------------
# Admission and budget accounting
# ---------------------------------------------------------------------------


def test_five_valid_requests_admitted_sixth_limited(client):
    # Unknown grants: every request is syntactically valid and scoped, so
    # each consumes one budget unit even though the outcome is 404.
    for _ in range(5):
        assert _consume(client).status_code == 404
    response = _consume(client)
    assert response.status_code == 429


def test_every_business_outcome_consumes_budget(client, clock):
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])
    # 200: a winning consume.
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200
    # 409: the same grant is now settled.
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 409
    # 401: a well-formed but wrong capability on a fresh grant.
    other = _grant(client, decision["decision_id"])
    assert _consume(client, other["grant_id"]).status_code == 401
    # 404: an unknown grant.
    assert _consume(client).status_code == 404
    # 410: a grant minted with the minimum TTL, observed after expiry but
    # within the same UTC minute window.
    expiring = _grant(client, decision["decision_id"], ttl_seconds=30)
    clock["now"] += timedelta(seconds=31)
    assert _consume(client, expiring["grant_id"], expiring["capability"]).status_code == 410
    # The sixth valid request in the window is limited.
    assert _consume(client).status_code == 429


def test_server_error_outcome_consumes_budget(app, client):
    decision = _decision(client)
    _envelope(client)
    grant = _grant(client, decision["decision_id"])
    # Corrupt the envelope so every release attempt fails decryption with
    # a 500 while the grant stays pending.
    with app.state.session_factory() as session:
        envelope = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        tampered = bytearray(envelope.ciphertext)
        tampered[0] ^= 0x01
        envelope.ciphertext = bytes(tampered)
        session.commit()

    for _ in range(5):
        response = _release(client, grant["grant_id"], grant["capability"])
        assert response.status_code == 500
    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 429


def test_budget_shared_across_consume_revoke_and_release(client):
    _decision(client)
    assert _consume(client).status_code == 404
    assert _revoke(client).status_code == 404
    assert _release(client).status_code == 404
    assert _consume(client).status_code == 404
    assert _revoke(client).status_code == 404
    # The five admissions are spent across the three endpoints; the next
    # valid request on any of them is limited.
    assert _release(client).status_code == 429
    assert _consume(client).status_code == 429
    assert _revoke(client).status_code == 429


def test_budget_is_strictly_per_scope(client):
    for _ in range(5):
        assert _consume(client).status_code == 404
    assert _consume(client).status_code == 429
    # A different tenant or workload has its own untouched budget.
    assert _consume(client, tenant_id=OTHER_TENANT).status_code == 404
    assert _consume(client, workload_id=OTHER_WORKLOAD).status_code == 404
    assert _revoke(client, tenant_id=OTHER_TENANT).status_code == 404
    assert _release(client, workload_id=OTHER_WORKLOAD).status_code == 404


def test_field_validation_failures_do_not_consume_budget(client):
    # Missing, blank, wrong-typed and malformed fields are all 422 and
    # never touch the budget, on all three endpoints.
    for _ in range(8):
        assert _consume(client, capability="").status_code == 422
        assert _consume(client, capability="   ").status_code == 422
        assert _consume(client, capability="not base64!!!").status_code == 422
        assert _consume(client, tenant_id="").status_code == 422
        assert _revoke(client, capability="with=padding").status_code == 422
        assert _release(client, data_id="").status_code == 422
        assert client.post(
            f"/v1/release-grants/{UNKNOWN_GRANT}/consume",
            json={"tenant_id": 1},
        ).status_code == 422
        # A syntactically illegal path grant identifier on revoke is a
        # 422 field error, not a lookup, and consumes nothing either.
        assert _revoke(client, "not-a-uuid").status_code == 422
    # The full budget is still available afterwards.
    for _ in range(5):
        assert _consume(client).status_code == 404
    assert _consume(client).status_code == 429


# ---------------------------------------------------------------------------
# 429 response contract
# ---------------------------------------------------------------------------


def test_429_response_is_compact_retry_after_json(client, clock):
    clock["now"] += timedelta(seconds=10)
    for _ in range(5):
        assert _consume(client).status_code == 404
    response = _consume(client)
    assert response.status_code == 429
    assert response.headers["content-type"].startswith("application/json")
    # Exactly one field, a positive integer, compact separators and a
    # single terminating newline. At second 10 the window has 50 whole
    # seconds left.
    assert response.text == '{"retry_after_seconds":50}\n'
    payload = response.json()
    assert list(payload) == ["retry_after_seconds"]
    assert isinstance(payload["retry_after_seconds"], int)
    assert payload["retry_after_seconds"] == 50


def test_retry_after_counts_down_to_window_end(client, clock):
    clock["now"] += timedelta(seconds=59)
    for _ in range(5):
        assert _consume(client).status_code == 404
    response = _consume(client)
    assert response.status_code == 429
    # One whole second remains; the value never reaches zero or floats.
    assert response.json()["retry_after_seconds"] == 1


def test_repeated_429_is_stable_and_consumes_nothing(client, clock):
    clock["now"] += timedelta(seconds=20)
    for _ in range(5):
        assert _consume(client).status_code == 404
    first = _consume(client)
    assert first.status_code == 429
    assert first.json()["retry_after_seconds"] == 40
    # Repeating the limited request changes nothing: same remaining
    # seconds, no extra budget consumed, window not extended.
    for _ in range(5):
        again = _consume(client)
        assert again.status_code == 429
        assert again.text == first.text
    # The next UTC minute restores the full budget of five.
    clock["now"] += timedelta(seconds=41)
    for _ in range(5):
        assert _consume(client).status_code == 404
    assert _consume(client).status_code == 429


def test_window_recovers_on_the_next_utc_minute(client, clock):
    for _ in range(5):
        assert _consume(client).status_code == 404
    assert _consume(client).status_code == 429
    clock["now"] += timedelta(seconds=60)
    for _ in range(5):
        assert _revoke(client).status_code == 404
    assert _revoke(client).status_code == 429


# ---------------------------------------------------------------------------
# Isolation from state, audit and payloads
# ---------------------------------------------------------------------------


def test_429_writes_no_state_no_audit_and_releases_nothing(app, client, clock):
    decision = _decision(client)
    _envelope(client)
    grant = _grant(client, decision["decision_id"])

    # Spend the whole window on unknown grants.
    for _ in range(5):
        assert _consume(client).status_code == 404

    # The limited release never decrypts, never settles the grant and
    # writes no audit event.
    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 429
    assert PAYLOAD not in response.text
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.consumed_at is None
        assert row.revoked_at is None
        events = session.query(AuditEvent).all()
        # Only the issuance event exists; the 429 added nothing.
        assert len(events) == 1
        assert events[0].status == "pending"

    # The capability was not consumed: once the window rolls over, the
    # same release succeeds.
    clock["now"] += timedelta(seconds=60)
    released = _release(client, grant["grant_id"], grant["capability"])
    assert released.status_code == 200
    assert released.text == (
        json.dumps({"payload": PAYLOAD}, separators=(",", ":")) + "\n"
    )


def test_429_does_not_block_unlimited_endpoints(client):
    for _ in range(5):
        assert _consume(client).status_code == 404
    assert _consume(client).status_code == 429
    # Health, grant minting, envelope creation and the audit listings are
    # not part of the limited action set.
    assert client.get("/health").status_code == 200
    decision = _decision(client)
    minted = _grant(client, decision["decision_id"])
    assert minted["pending"] is True
    assert (
        client.get(
            "/v1/release-grants",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/v1/compliance/audit-events",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).status_code
        == 200
    )


# ---------------------------------------------------------------------------
# Durability and concurrency
# ---------------------------------------------------------------------------


def test_budget_survives_process_restart(tmp_path, monkeypatch, clock):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/rate-restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    for _ in range(5):
        assert _consume(client1).status_code == 404
    app1.state.engine.dispose()

    # A fresh process against the same database sees the spent window.
    app2 = create_app(url)
    client2 = TestClient(app2)
    assert _consume(client2).status_code == 429
    clock["now"] += timedelta(seconds=60)
    assert _consume(client2).status_code == 404
    app2.state.engine.dispose()


def test_concurrent_requests_admit_exactly_five(app):
    def call(_):
        return TestClient(app).post(
            f"/v1/release-grants/{UNKNOWN_GRANT}/consume",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "capability": UNKNOWN_CAPABILITY,
            },
        ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(call, range(12)))

    # Exactly five winners enter business judgement (404 here); no race
    # can mint extra admissions.
    assert statuses.count(404) == 5
    assert statuses.count(429) == 7
