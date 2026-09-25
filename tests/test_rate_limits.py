"""Tests for the shared per-scope, per-minute rate limit on the grant
consume, grant revoke and payload release endpoints.

The three paths share one persisted budget: at most five validated
requests per (tenant, workload) per UTC natural minute. Validation
failures (422) never consume budget; every other outcome does.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from proof_release.app import create_app
from proof_release.db import AuditEvent, RateLimitWindow, ReleaseGrant
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
MASTER_KEY = b64url_encode(b"0123456789abcdef0123456789abcdef")
PAYLOAD = "rate limited secret payload"

UNKNOWN_GRANT = "00000000-0000-0000-0000-000000000000"
#: Well-formed unpadded base64url capability that matches no grant.
CAPABILITY = "A" * 43

#: The 429 body is exactly one positive-integer field, compact, newline
#: terminated — no floats, -0.0 or non-finite values can appear.
RATE_LIMIT_BODY_RE = re.compile(rb'^\{"retry_after_seconds":[1-9][0-9]*\}\n$')


@pytest.fixture()
def app(tmp_path, monkeypatch):
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
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)}
    )


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
    return decided.json()["decision_id"]


def _grant(client, decision_id, tenant=TENANT, workload=WORKLOAD, data_id=DATA_ID):
    response = client.post(
        "/v1/release-grants",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "decision_id": decision_id,
            "data_id": data_id,
        },
    )
    assert response.status_code == 201
    return response.json()


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


def _consume(client, grant_id=UNKNOWN_GRANT, capability=CAPABILITY, **scope):
    body = {
        "tenant_id": scope.get("tenant_id", TENANT),
        "workload_id": scope.get("workload_id", WORKLOAD),
        "capability": capability,
    }
    return client.post(f"/v1/release-grants/{grant_id}/consume", json=body)


def _revoke(client, grant_id=UNKNOWN_GRANT, capability=CAPABILITY, **scope):
    body = {
        "tenant_id": scope.get("tenant_id", TENANT),
        "workload_id": scope.get("workload_id", WORKLOAD),
        "capability": capability,
    }
    return client.post(f"/v1/release-grants/{grant_id}/revoke", json=body)


def _release(client, grant_id=UNKNOWN_GRANT, capability=CAPABILITY, **scope):
    body = {
        "tenant_id": scope.get("tenant_id", TENANT),
        "workload_id": scope.get("workload_id", WORKLOAD),
        "data_id": scope.get("data_id", DATA_ID),
        "capability": capability,
    }
    return client.post(f"/v1/release/{grant_id}", json=body)


def _assert_429(response):
    assert response.status_code == 429
    assert RATE_LIMIT_BODY_RE.fullmatch(response.content), response.content
    retry_after = response.json()["retry_after_seconds"]
    assert isinstance(retry_after, int)
    assert 1 <= retry_after <= 60


# --- shared budget and 429 contract -----------------------------------------


def test_sixth_validated_request_in_minute_is_429(client):
    # Unknown-but-well-formed grants reach business judgement (404) and
    # each consumes one slot of the shared minute budget.
    for _ in range(5):
        assert _consume(client).status_code == 404
    response = _consume(client)
    _assert_429(response)


def test_budget_is_shared_across_consume_revoke_and_release(client):
    assert _consume(client).status_code == 404
    assert _revoke(client).status_code == 404
    assert _release(client).status_code == 404
    assert _consume(client).status_code == 404
    assert _revoke(client).status_code == 404
    # The sixth validated request on any of the three paths is rejected.
    _assert_429(_release(client))
    _assert_429(_consume(client))
    _assert_429(_revoke(client))


def test_429_body_is_compact_json_with_trailing_newline(client):
    for _ in range(5):
        _consume(client)
    response = _consume(client)
    assert response.headers["content-type"].startswith("application/json")
    assert response.content.endswith(b"\n")
    assert b", " not in response.content
    assert b": " not in response.content
    assert set(response.json().keys()) == {"retry_after_seconds"}


def test_repeated_429_recomputes_delay_without_extending_window(client):
    for _ in range(5):
        _consume(client)
    first = _consume(client)
    second = _consume(client)
    _assert_429(first)
    _assert_429(second)
    # Neither rejection consumed budget or extended the window: the
    # counter is still exactly the five admitted requests.
    with client.app.state.session_factory() as session:
        rows = session.scalars(select(RateLimitWindow)).all()
    assert len(rows) == 1
    assert rows[0].count == 5


def test_concurrent_requests_admit_exactly_five(app):
    client = TestClient(app)

    def call(_):
        return TestClient(app).post(
            f"/v1/release-grants/{UNKNOWN_GRANT}/consume",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "capability": CAPABILITY,
            },
        ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(call, range(12)))
    assert statuses.count(404) == 5
    assert statuses.count(429) == 7
    with app.state.session_factory() as session:
        rows = session.scalars(select(RateLimitWindow)).all()
    assert len(rows) == 1
    assert rows[0].count == 5


# --- validation failures never consume budget --------------------------------


def test_422_field_errors_do_not_consume_budget(client):
    bad_bodies = [
        {"tenant_id": "", "workload_id": WORKLOAD, "capability": CAPABILITY},
        {"tenant_id": TENANT, "workload_id": "  ", "capability": CAPABILITY},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "capability": ""},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "capability": "bad!!"},
        {"tenant_id": 1, "workload_id": WORKLOAD, "capability": CAPABILITY},
        {"tenant_id": TENANT, "workload_id": WORKLOAD},
    ]
    for body in bad_bodies * 2:
        response = client.post(
            f"/v1/release-grants/{UNKNOWN_GRANT}/consume", json=body
        )
        assert response.status_code == 422
    # None of the 422s touched the budget: five validated requests still
    # enter judgement, and only the sixth is rejected.
    for _ in range(5):
        assert _consume(client).status_code == 404
    _assert_429(_consume(client))


def test_invalid_path_grant_id_is_422_and_free_on_all_endpoints(client):
    for path in (
        "/v1/release-grants/not-a-uuid/consume",
        "/v1/release-grants/not-a-uuid/revoke",
        "/v1/release/not-a-uuid",
        "/v1/release-grants//consume",
        "/v1/release-grants//revoke",
        "/v1/release/",
    ):
        for _ in range(2):
            response = client.post(
                path,
                json={
                    "tenant_id": TENANT,
                    "workload_id": WORKLOAD,
                    "data_id": DATA_ID,
                    "capability": CAPABILITY,
                },
            )
            assert response.status_code == 422, path
    with client.app.state.session_factory() as session:
        assert session.scalars(select(RateLimitWindow)).all() == []
    for _ in range(5):
        assert _consume(client).status_code == 404
    _assert_429(_consume(client))


# --- business outcomes keep their consumption --------------------------------


def test_business_failures_still_consume_budget(client):
    decision_id = _decision(client)
    grant = _grant(client, decision_id)
    # 401: wrong capability on a real grant.
    wrong = ("B" if grant["capability"][0] != "B" else "C") + grant["capability"][1:]
    assert _consume(client, grant["grant_id"], wrong).status_code == 401
    # 404: unknown grant.
    assert _consume(client).status_code == 404
    # 200: the real consume.
    assert (
        _consume(client, grant["grant_id"], grant["capability"]).status_code == 200
    )
    # 409: already consumed.
    assert (
        _consume(client, grant["grant_id"], grant["capability"]).status_code == 409
    )
    # 404 again: fifth slot.
    assert _revoke(client).status_code == 404
    # Budget exhausted even though only one request succeeded.
    _assert_429(_consume(client))


def test_429_writes_no_grant_state_payload_or_audit(app):
    client = TestClient(app)
    decision_id = _decision(client)
    _envelope(client)
    grant = _grant(client, decision_id)

    # Burn the budget on unknown grants.
    for _ in range(5):
        assert _consume(client).status_code == 404

    response = _release(client, grant["grant_id"], grant["capability"])
    _assert_429(response)
    assert PAYLOAD not in response.text

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.consumed_at is None and row.revoked_at is None
        # Only the mint event exists; the rejections wrote no audit.
        events = session.scalars(select(AuditEvent)).all()
        assert len(events) == 1
        assert events[0].status == "pending"


# --- isolation, recovery and persistence -------------------------------------


def test_scopes_are_strictly_isolated(client):
    for _ in range(5):
        assert _consume(client).status_code == 404
    _assert_429(_consume(client))
    # Another tenant and another workload each have their own full budget.
    assert _consume(client, tenant_id="tenant-b").status_code == 404
    assert _consume(client, workload_id="workload-2").status_code == 404
    with client.app.state.session_factory() as session:
        rows = session.scalars(select(RateLimitWindow)).all()
    assert len(rows) == 3


def test_budget_recovers_at_next_utc_minute(app, client):
    for _ in range(5):
        assert _consume(client).status_code == 404
    _assert_429(_consume(client))

    # Move the persisted window into the previous minute, simulating the
    # arrival of the next UTC minute boundary.
    previous = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(
        minutes=1
    )
    with app.state.session_factory() as session:
        row = session.scalar(select(RateLimitWindow))
        row.window_start = previous
        session.commit()

    # The new minute is a fresh window: the request enters judgement.
    assert _consume(client).status_code == 404
    with app.state.session_factory() as session:
        rows = session.scalars(select(RateLimitWindow)).all()
    assert len(rows) == 2
    assert sorted(r.count for r in rows) == [1, 5]


def test_budget_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/rate-restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    for _ in range(5):
        assert _consume(client1).status_code == 404
    app1.state.engine.dispose()

    app2 = create_app(url)
    try:
        _assert_429(_consume(TestClient(app2)))
    finally:
        app2.state.engine.dispose()


# --- counter failure ---------------------------------------------------------


def test_unavailable_counter_returns_500_and_skips_judgement(app):
    client = TestClient(app)
    decision_id = _decision(client)
    grant = _grant(client, decision_id)

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rate_limit_windows"))

    response = _consume(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 500
    # The request never reached judgement: the grant is untouched.
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.consumed_at is None

    # Once the counter is back, the same request proceeds normally.
    RateLimitWindow.__table__.create(app.state.engine)
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200
