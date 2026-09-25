"""Tests for the shared per-tenant/workload rate limit on the three
one-time-grant entry points: grant consumption, grant revocation and
authorized payload release.

Contract under test:

* the three paths share one budget of five business judgements per UTC
  natural minute per (tenant, workload);
* field/type/format/path-identifier failures (422) never consume budget
  and write no state;
* once basic validation passes, a slot is durably reserved *before* the
  business judgement, so 404/401/409/410/500 still spend the slot;
* the sixth request in a minute gets a compact 429 carrying only a
  positive-integer ``retry_after_seconds`` followed by one newline;
* a 429 changes no grant, payload or audit state, repeats never decrement
  the counter, and the quota recovers at the next UTC minute with strict
  scope isolation and no borrowing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import AuditEvent, DataEnvelope, RateLimitCounter, ReleaseGrant
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
MASTER_KEY = b64url_encode(b"0123456789abcdef0123456789abcdef")
PAYLOAD = "rate-limited secret payload"
ZERO_UUID = "00000000-0000-0000-0000-000000000000"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/limits.db")
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
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)}
    )


def _decision(client):
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    evidence = _evidence(created["nonce"])
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
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
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.status_code == 200
    policy = client.post(
        "/v1/policies",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "name": "release",
            "rule": {"claim": "m", "equals": "x"},
        },
    ).json()
    decided = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert decided.status_code == 200
    return decided.json()["decision_id"]


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


def _grant(client, decision_id, data_id=DATA_ID):
    response = client.post(
        "/v1/release-grants",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "decision_id": decision_id,
            "data_id": data_id,
        },
    )
    assert response.status_code == 201
    return response.json()


def _setup(client, data_id=DATA_ID, payload=PAYLOAD):
    decision_id = _decision(client)
    _envelope(client, data_id=data_id, payload=payload)
    return _grant(client, decision_id, data_id=data_id)


def _consume(client, grant_id, capability, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release-grants/{grant_id}/consume", json=body)


def _revoke(client, grant_id, capability, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release-grants/{grant_id}/revoke", json=body)


def _release(client, grant_id, capability, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release/{grant_id}", json=body)


# ---------------------------------------------------------------------------
# Basic shape and the 5-per-minute ceiling
# ---------------------------------------------------------------------------


def test_first_valid_request_initializes_current_minute_with_one(app, client):
    grant = _setup(client)
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200

    with app.state.session_factory() as session:
        rows = session.scalars(select(RateLimitCounter)).all()
        assert len(rows) == 1
        row = rows[0]
        assert (row.tenant_id, row.workload_id) == (TENANT, WORKLOAD)
        assert row.count == 1
        now = datetime.now(timezone.utc)
        assert row.window_start == now.replace(second=0, microsecond=0)


def test_sixth_serial_request_returns_429(client):
    grant = _setup(client)

    statuses = [
        _consume(client, grant["grant_id"], grant["capability"]).status_code
        for _ in range(5)
    ]
    # First call settles; the next four observe the consumed state.
    assert statuses == [200, 409, 409, 409, 409]

    sixth = _consume(client, grant["grant_id"], grant["capability"])
    assert sixth.status_code == 429
    assert sixth.headers["content-type"] == "application/json"
    raw = sixth.content
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw and b": " not in raw
    data = json.loads(raw)
    assert set(data) == {"retry_after_seconds"}
    value = data["retry_after_seconds"]
    assert isinstance(value, int) and not isinstance(value, bool)
    assert 1 <= value <= 60
    # The body is exactly the compact one-field object plus newline.
    assert raw == (
        json.dumps(
            {"retry_after_seconds": value}, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )


def test_sixth_concurrent_request_returns_429_with_five_admitted(app):
    client = TestClient(app)
    grant = _setup(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }
    url = f"/v1/release-grants/{grant['grant_id']}/consume"

    def call():
        return TestClient(app).post(url, json=body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: call(), range(6)))

    assert statuses.count(429) == 1
    # Exactly five entered business judgement: one 200 and four 409 losers.
    assert statuses.count(200) == 1
    assert statuses.count(409) == 4


def test_429_repeats_do_not_decrement_or_extend(app, client):
    grant = _setup(client)
    for _ in range(5):
        _consume(client, grant["grant_id"], grant["capability"])

    rejected = [
        _consume(client, grant["grant_id"], grant["capability"]).status_code
        for _ in range(4)
    ]
    assert rejected == [429, 429, 429, 429]

    with app.state.session_factory() as session:
        row = session.scalar(select(RateLimitCounter))
        assert row.count == app_module.GRANT_BUDGET_PER_MINUTE


# ---------------------------------------------------------------------------
# Shared budget across all three entry points
# ---------------------------------------------------------------------------


def test_consume_revoke_release_share_one_budget(client):
    first = _setup(client, data_id="d-1")
    second = _setup(client, data_id="d-2")
    third = _setup(client, data_id="d-3")
    fourth = _setup(client, data_id="d-4")
    fifth = _setup(client, data_id="d-5")
    sixth = _setup(client, data_id="d-6")

    assert _consume(client, first["grant_id"], first["capability"]).status_code == 200
    assert _revoke(client, second["grant_id"], second["capability"]).status_code == 200
    assert (
        _release(client, third["grant_id"], third["capability"], data_id="d-3").status_code
        == 200
    )
    assert _consume(client, fourth["grant_id"], fourth["capability"]).status_code == 200
    assert _revoke(client, fifth["grant_id"], fifth["capability"]).status_code == 200

    # The sixth business request, regardless of which path it uses, is 429.
    response = _release(
        client, sixth["grant_id"], sixth["capability"], data_id="d-6"
    )
    assert response.status_code == 429
    assert PAYLOAD not in response.text
    with client.app.state.session_factory() as session:
        assert session.get(ReleaseGrant, sixth["grant_id"]).status == "pending"


# ---------------------------------------------------------------------------
# 422 precedes the counter: no budget consumed, no state written
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"capability": ""},
        {"capability": "   "},
        {"capability": "not base64!!!"},
        {"capability": 9},
        {"capability": True},
        {"tenant_id": 1},
    ],
)
def test_consume_422_does_not_consume_budget(client, overrides):
    grant = _setup(client)
    url = f"/v1/release-grants/{grant['grant_id']}/consume"
    for _ in range(app_module.GRANT_BUDGET_PER_MINUTE + 2):
        body = {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        }
        body.update(overrides)
        response = client.post(url, json=body)
        assert response.status_code == 422

    # A well-formed request is still the first admitted request.
    ok = _consume(client, grant["grant_id"], grant["capability"])
    assert ok.status_code == 200


def test_release_422_does_not_consume_budget(client):
    grant = _setup(client)
    for _ in range(7):
        response = client.post(
            f"/v1/release/{grant['grant_id']}",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "data_id": DATA_ID,
                "capability": "bad alphabet!!!",
            },
        )
        assert response.status_code == 422
    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/v1/release-grants/not-a-uuid/consume",
        "/v1/release-grants//consume",
        "/v1/release/not-a-uuid",
        "/v1/release/",
    ],
)
def test_invalid_path_identifier_is_422_and_not_counted(app, client, path):
    grant = _setup(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "capability": grant["capability"],
    }
    for _ in range(6):
        assert client.post(path, json=body).status_code == 422

    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []
    # The budget is untouched: the real grant still settles on slot one.
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200


def test_non_object_body_is_422_and_not_counted(app, client):
    grant = _setup(client)
    for _ in range(6):
        assert (
            client.post(
                f"/v1/release-grants/{grant['grant_id']}/revoke",
                json=["not", "an", "object"],
            ).status_code
            == 422
        )
    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []


# ---------------------------------------------------------------------------
# Business errors still consume the budget
# ---------------------------------------------------------------------------


def test_404_consumes_budget(client):
    _decision(client)
    # Five requests against an unknown (but well-formed) grant: all 404.
    for _ in range(5):
        assert _consume(client, ZERO_UUID, "A" * 43).status_code == 404
    assert _consume(client, ZERO_UUID, "A" * 43).status_code == 429


def test_401_consumes_budget_and_leaves_grant_pending(client, app):
    grant = _setup(client)
    wrong = ("B" if grant["capability"][0] != "B" else "C") + grant["capability"][1:]
    for _ in range(5):
        assert _revoke(client, grant["grant_id"], wrong).status_code == 401

    # The sixth request, this time with the correct capability, is refused
    # before the capability is ever checked.
    sixth = _revoke(client, grant["grant_id"], grant["capability"])
    assert sixth.status_code == 429
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.revoked_at is None
        # Only the issuance audit; the 401s and the 429 wrote nothing.
        events = session.scalars(select(AuditEvent)).all()
        assert len(events) == 1
        assert events[0].status == "pending"


def test_410_consumes_budget(client):
    grant = _setup(client)
    with client.app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
    for _ in range(5):
        assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 410
    assert (
        _consume(client, grant["grant_id"], grant["capability"]).status_code == 429
    )


def test_429_writes_no_grant_payload_or_audit_state(app, client):
    grant = _setup(client)
    # Five wrong-capability releases: 401 each, grant stays pending.
    wrong = ("B" if grant["capability"][0] != "B" else "C") + grant["capability"][1:]
    for _ in range(5):
        assert _release(client, grant["grant_id"], wrong).status_code == 401

    limited = _release(client, grant["grant_id"], grant["capability"])
    assert limited.status_code == 429
    assert PAYLOAD not in limited.text

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.consumed_at is None
        # Exactly the pending issuance event; no consumption event.
        events = session.scalars(select(AuditEvent)).all()
        assert len(events) == 1
        assert events[0].status == "pending"


def test_decryption_500_retains_consumed_slot(app, client):
    # Four well-formed requests spend four slots (the first settles, the
    # next three are 409). A fifth request fails decryption with a 500 and
    # still spends its slot; the sixth well-formed request is a 429 rather
    # than reaching the grant (which remains pending).
    grant = _setup(client)
    with app.state.session_factory() as session:
        envelope = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        tampered = bytearray(envelope.tag)
        tampered[0] ^= 0x01
        envelope.tag = bytes(tampered)
        session.commit()

    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 500
    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 500
    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 500
    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 500
    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 500
    sixth = _release(client, grant["grant_id"], grant["capability"])
    assert sixth.status_code == 429

    with app.state.session_factory() as session:
        assert session.get(ReleaseGrant, grant["grant_id"]).status == "pending"


# ---------------------------------------------------------------------------
# Scope isolation and minute recovery
# ---------------------------------------------------------------------------


def test_scopes_are_strictly_isolated_with_no_borrowing(app, client):
    grant = _setup(client)
    # Exhaust tenant-a/workload-1.
    for _ in range(5):
        _consume(client, grant["grant_id"], grant["capability"])
    assert (
        _consume(client, grant["grant_id"], grant["capability"]).status_code == 429
    )

    # A different tenant on the same workload has its own full budget. A
    # well-formed request enters judgement and is a plain 404 (unknown
    # grant in that scope), never a 429.
    other_tenant = _consume(
        client, ZERO_UUID, "A" * 43, tenant_id="tenant-b"
    )
    assert other_tenant.status_code == 404

    # A different workload in the same tenant is likewise independent.
    other_workload = _consume(
        client, ZERO_UUID, "A" * 43, workload_id="workload-2"
    )
    assert other_workload.status_code == 404

    with app.state.session_factory() as session:
        rows = session.scalars(select(RateLimitCounter)).all()
        counts = {(r.tenant_id, r.workload_id): r.count for r in rows}
        assert counts == {
            (TENANT, WORKLOAD): 5,
            ("tenant-b", WORKLOAD): 1,
            (TENANT, "workload-2"): 1,
        }


def test_quota_recovers_at_next_utc_minute(app, client):
    grant = _setup(client)
    for _ in range(5):
        _consume(client, grant["grant_id"], grant["capability"])
    assert (
        _consume(client, grant["grant_id"], grant["capability"]).status_code == 429
    )

    # Simulate crossing the UTC minute boundary by moving the exhausted
    # row into the previous minute: the current minute has no counter yet.
    with app.state.session_factory() as session:
        row = session.scalar(select(RateLimitCounter))
        row.window_start = row.window_start - timedelta(minutes=1)
        session.commit()

    # A fresh minute initializes a new counter and admits the request (the
    # grant is already consumed, so it observes 409 — not 429).
    response = _consume(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 409
    with app.state.session_factory() as session:
        rows = {
            r.window_start: r.count
            for r in session.scalars(select(RateLimitCounter)).all()
        }
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        assert rows[now] == 1
        assert rows[now - timedelta(minutes=1)] == 5


def test_previous_minute_counter_does_not_limit_new_minute(app, client):
    grant = _setup(client)
    # Seed a saturated counter for a past minute directly.
    with app.state.session_factory() as session:
        session.add(
            RateLimitCounter(
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                window_start=datetime.now(timezone.utc).replace(
                    second=0, microsecond=0
                )
                - timedelta(minutes=1),
                count=500,
            )
        )
        session.commit()

    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200


def test_budget_persists_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart-limits.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    grant = _setup(client1)
    for _ in range(5):
        _consume(client1, grant["grant_id"], grant["capability"])
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    # The sixth judgement in the same UTC minute is still refused.
    response = _consume(client2, grant["grant_id"], grant["capability"])
    assert response.status_code == 429
    app2.state.engine.dispose()


# ---------------------------------------------------------------------------
# Counter failure
# ---------------------------------------------------------------------------


def test_counter_unavailable_returns_500_and_does_not_judge(app, client):
    grant = _setup(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rate_limit_counters"))

    response = _consume(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 500
    # No business judgement ran: the grant is still pending and consumable
    # once the counter is available again.
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.consumed_at is None

    RateLimitCounter.__table__.create(app.state.engine)
    recovered = _consume(client, grant["grant_id"], grant["capability"])
    assert recovered.status_code == 200


# ---------------------------------------------------------------------------
# Scope of the limiter: grant issuance itself is not limited
# ---------------------------------------------------------------------------


def test_grant_issuance_is_not_rate_limited(client):
    decision_id = _decision(client)
    # More than five grants in the same minute all succeed; only the three
    # one-time-grant actions share the budget.
    for index in range(7):
        response = client.post(
            "/v1/release-grants",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "decision_id": decision_id,
                "data_id": f"unlimited-{index}",
            },
        )
        assert response.status_code == 201
