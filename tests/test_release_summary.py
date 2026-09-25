"""Tests for the read-only release-chain operational summary:

GET /v1/observability/release-summary.

Contract under test:

* the body is always empty and the only query parameters are the two
  mandatory non-blank scope strings; any missing/blank/extra parameter
  or non-empty body is a 422 raised before any state is read;
* the response is a fixed twelve-field object (scope strings, grant
  counts by status, live/expired pending split, current-minute rate
  budget, envelope counts split by current/historical key version) as
  compact JSON with a single trailing newline and integer counts only;
* an untouched scope reports zero counts and the full five-slot budget;
* the query never consumes rate budget, writes no audit event and
  observes only committed state in one consistent snapshot;
* a storage or master-keyring failure is a 500, never a partial summary.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from proof_release.app import GRANT_BUDGET_PER_MINUTE, create_app
from proof_release.db import AuditEvent, RateLimitCounter, ReleaseGrant
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
KEY_V1 = b64url_encode(b"0123456789abcdef0123456789abcdef")
KEY_V2 = b64url_encode(b"fedcba9876543210fedcba9876543210")
KEYRING_V1_V2 = json.dumps(
    {"current_version": 2, "keys": {"1": KEY_V1, "2": KEY_V2}}
)

SUMMARY_URL = "/v1/observability/release-summary"
SCOPE = {"tenant_id": TENANT, "workload_id": WORKLOAD}
FIELD_ORDER = [
    "tenant_id",
    "workload_id",
    "pending",
    "consumed",
    "revoked",
    "live_pending",
    "expired_pending",
    "rate_used",
    "rate_remaining",
    "envelopes",
    "current_key_envelopes",
    "historical_key_envelopes",
]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = create_app(f"sqlite:///{tmp_path}/summary.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _summary(client, **params):
    return client.get(SUMMARY_URL, params=params or SCOPE)


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
    created = client.post("/v1/challenges", json=SCOPE).json()
    evidence = _evidence(created["nonce"])
    submitted = client.post(
        "/v1/evidence",
        json={
            **SCOPE,
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
        json={**SCOPE, "nonce": created["nonce"], "evidence": evidence},
    )
    assert verified.status_code == 200
    policy = client.post(
        "/v1/policies",
        json={**SCOPE, "name": "release", "rule": {"claim": "m", "equals": "x"}},
    ).json()
    decided = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            **SCOPE,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert decided.status_code == 200
    return decided.json()["decision_id"]


def _envelope(client, data_id=DATA_ID):
    response = client.post(
        "/v1/data-envelopes",
        json={**SCOPE, "data_id": data_id, "payload": "summary secret payload"},
    )
    assert response.status_code == 201
    return response.json()


def _grant(client, decision_id, data_id=DATA_ID):
    response = client.post(
        "/v1/release-grants",
        json={**SCOPE, "decision_id": decision_id, "data_id": data_id},
    )
    assert response.status_code == 201
    return response.json()


def _expire_grant(app, grant_id):
    with app.state.session_factory() as session:
        grant = session.get(ReleaseGrant, grant_id)
        grant.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()


# ---------------------------------------------------------------------------
# Response shape and the empty scope


def test_empty_scope_reports_zero_counts_and_full_budget(client):
    response = _summary(client)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    # Compact serialization with exactly one trailing newline.
    assert response.content == json.dumps(
        response.json(), separators=(",", ":")
    ).encode("utf-8") + b"\n"
    assert list(response.json().keys()) == FIELD_ORDER
    assert response.json() == {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "pending": 0,
        "consumed": 0,
        "revoked": 0,
        "live_pending": 0,
        "expired_pending": 0,
        "rate_used": 0,
        "rate_remaining": GRANT_BUDGET_PER_MINUTE,
        "envelopes": 0,
        "current_key_envelopes": 0,
        "historical_key_envelopes": 0,
    }
    # Every count is a JSON integer, never a float.
    for field in FIELD_ORDER[2:]:
        assert type(response.json()[field]) is int


def test_summary_is_scoped_to_tenant_and_workload(client):
    _envelope(client)
    _grant(client, _decision(client))

    other = _summary(
        client, tenant_id="tenant-b", workload_id="workload-9"
    ).json()
    assert other["pending"] == 0
    assert other["envelopes"] == 0

    mine = _summary(client).json()
    assert mine["pending"] == 1
    assert mine["envelopes"] == 1


# ---------------------------------------------------------------------------
# Request validation (all 422, before any state read)


def test_missing_scope_parameters_are_422(client):
    assert client.get(SUMMARY_URL).status_code == 422
    assert (
        client.get(SUMMARY_URL, params={"tenant_id": TENANT}).status_code == 422
    )
    assert (
        client.get(SUMMARY_URL, params={"workload_id": WORKLOAD}).status_code
        == 422
    )


def test_blank_scope_parameters_are_422(client):
    assert (
        _summary(client, tenant_id="  ", workload_id=WORKLOAD).status_code == 422
    )
    assert (
        _summary(client, tenant_id=TENANT, workload_id="").status_code == 422
    )


def test_extra_query_parameter_is_422(client):
    assert (
        _summary(client, tenant_id=TENANT, workload_id=WORKLOAD, cursor="x")
        .status_code
        == 422
    )
    assert (
        _summary(client, tenant_id=TENANT, workload_id=WORKLOAD, status="pending")
        .status_code
        == 422
    )


def test_non_empty_body_is_422(client):
    assert (
        client.request("GET", SUMMARY_URL, params=SCOPE, json={}).status_code
        == 422
    )
    assert (
        client.request(
            "GET",
            SUMMARY_URL,
            params=SCOPE,
            content=b"not-json",
            headers={"content-type": "application/json"},
        ).status_code
        == 422
    )


# ---------------------------------------------------------------------------
# Grant status counts and the live/expired pending split


def test_grant_status_counts(client, app):
    decision_id = _decision(client)
    first = _grant(client, decision_id, data_id="data-1")
    second = _grant(client, decision_id, data_id="data-2")
    third = _grant(client, decision_id, data_id="data-3")

    consumed = client.post(
        f"/v1/release-grants/{first['grant_id']}/consume",
        json={**SCOPE, "capability": first["capability"]},
    )
    assert consumed.status_code == 200
    revoked = client.post(
        f"/v1/release-grants/{second['grant_id']}/revoke",
        json={**SCOPE, "capability": second["capability"]},
    )
    assert revoked.status_code == 200

    summary = _summary(client).json()
    assert summary["pending"] == 1
    assert summary["consumed"] == 1
    assert summary["revoked"] == 1
    assert summary["live_pending"] == 1
    assert summary["expired_pending"] == 0

    _expire_grant(app, third["grant_id"])
    summary = _summary(client).json()
    assert summary["pending"] == 1
    assert summary["live_pending"] == 0
    assert summary["expired_pending"] == 1


# ---------------------------------------------------------------------------
# Rate-limit budget reporting


def test_rate_budget_reflects_committed_counter(client, app):
    # A well-formed request that fails business judgement still spent a
    # durable slot on the shared counter.
    response = client.post(
        "/v1/release-grants/00000000-0000-0000-0000-000000000000/consume",
        json={**SCOPE, "capability": "a" * 43},
    )
    assert response.status_code == 404

    summary = _summary(client).json()
    assert summary["rate_used"] == 1
    assert summary["rate_remaining"] == GRANT_BUDGET_PER_MINUTE - 1


def test_summary_query_never_consumes_budget_or_writes_audit(client, app):
    first = _summary(client)
    assert first.status_code == 200
    second = _summary(client)
    assert second.status_code == 200
    assert second.json()["rate_used"] == 0
    assert second.json()["rate_remaining"] == GRANT_BUDGET_PER_MINUTE

    with app.state.session_factory() as session:
        assert session.scalar(select(RateLimitCounter)) is None
        assert session.scalar(select(AuditEvent)) is None


# ---------------------------------------------------------------------------
# Envelope counts and the key-version split


def test_envelope_counts_split_by_key_version(client, app, monkeypatch):
    _envelope(client, data_id="data-1")
    _envelope(client, data_id="data-2")

    summary = _summary(client).json()
    assert summary["envelopes"] == 2
    assert summary["current_key_envelopes"] == 2
    assert summary["historical_key_envelopes"] == 0

    # Rotate the keyring: both envelopes are now wrapped under a
    # historical version until rewrapped.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    summary = _summary(client).json()
    assert summary["envelopes"] == 2
    assert summary["current_key_envelopes"] == 0
    assert summary["historical_key_envelopes"] == 2

    rewrapped = client.post("/v1/data-envelopes/data-1/rewrap", json=SCOPE)
    assert rewrapped.status_code == 200
    summary = _summary(client).json()
    assert summary["envelopes"] == 2
    assert summary["current_key_envelopes"] == 1
    assert summary["historical_key_envelopes"] == 1


# ---------------------------------------------------------------------------
# Failure handling


def test_missing_master_key_is_500(client, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)

    response = _summary(client)
    assert response.status_code == 500


def test_storage_failure_is_500_and_partial_summary_is_never_returned(
    client, app
):
    _envelope(client)

    with app.state.engine.begin() as connection:
        connection.execute(text("DROP TABLE release_grants"))

    response = _summary(client)
    assert response.status_code == 500
