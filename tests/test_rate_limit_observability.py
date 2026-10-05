"""Tests for GET /v1/observability/rate-limits.

A read-only view of the current UTC natural minute's three independent
per-scope admission budgets — challenge issuance, evidence verification
and one-time grant actions — each persisted in its own counter table and
never sharing rows, quota or lock traffic. The endpoint never writes: it
mints no challenge, verifies no evidence, consumes no budget, appends no
audit row, and returns no nonce, evidence, payload, capability or key
material.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    ChallengeIssuanceCounter,
    RateLimitCounter,
    VerificationAdmissionCounter,
)
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
MASTER_KEY = b64url_encode(b"0123456789abcdef0123456789abcdef")
RATE_LIMITS_PATH = "/v1/observability/rate-limits"

EXPECTED_FIELD_ORDER = [
    "tenant_id",
    "workload_id",
    "window_start",
    "reset_at",
    "challenge_issuance",
    "verification",
    "grant_actions",
]
BUDGET_FIELDS = ["limit", "used", "remaining"]
BUDGET_NAMES = ["challenge_issuance", "verification", "grant_actions"]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/rate_limits.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _rate_limits(client, *, tenant=TENANT, workload=WORKLOAD, **kwargs):
    params = {"tenant_id": tenant, "workload_id": workload}
    params.update(kwargs.pop("params", {}))
    return client.request("GET", RATE_LIMITS_PATH, params=params, **kwargs)


def _mac(nonce: str, claims: dict, *, tenant=TENANT, workload=WORKLOAD) -> str:
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


def _evidence(nonce: str, *, tenant=TENANT, workload=WORKLOAD) -> str:
    claims = {"m": "x"}
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims, tenant=tenant, workload=workload)}
    )


def _challenge(client, *, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    )
    assert response.status_code == 201
    return response.json()


def _verified_evidence(client, *, tenant=TENANT, workload=WORKLOAD):
    """Drive one challenge issuance and one verification admission."""
    created = _challenge(client, tenant=tenant, workload=workload)
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
    return evidence_id, created["nonce"], evidence


def _grant_and_consume(client, *, tenant=TENANT, workload=WORKLOAD):
    """Drive one grant-action admission (a consume) end to end."""
    evidence_id, nonce, evidence = _verified_evidence(
        client, tenant=tenant, workload=workload
    )
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
            "nonce": nonce,
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert decided.status_code == 200
    envelope = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": DATA_ID,
            "payload": "summarized secret payload",
        },
    )
    assert envelope.status_code == 201
    grant = client.post(
        "/v1/release-grants",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "decision_id": decided.json()["decision_id"],
            "data_id": DATA_ID,
        },
    )
    assert grant.status_code == 201
    consume = client.post(
        f"/v1/release-grants/{grant.json()['grant_id']}/consume",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "capability": grant.json()["capability"],
        },
    )
    assert consume.status_code == 200


# ---------------------------------------------------------------------------
# Empty scope and wire format
# ---------------------------------------------------------------------------


def test_empty_scope_reports_full_budgets(client):
    response = _rate_limits(client)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    raw = response.content
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    assert b", " not in raw and b": " not in raw

    data = json.loads(raw)
    assert list(data) == EXPECTED_FIELD_ORDER
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    for name in BUDGET_NAMES:
        assert list(data[name]) == BUDGET_FIELDS
        assert data[name] == {"limit": 5, "used": 0, "remaining": 5}


def test_window_timestamps_are_utc_z_and_one_minute_apart(client):
    before = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    data = _rate_limits(client).json()
    after = datetime.now(timezone.utc)

    window_start = datetime.fromisoformat(data["window_start"].replace("Z", "+00:00"))
    reset_at = datetime.fromisoformat(data["reset_at"].replace("Z", "+00:00"))
    assert data["window_start"].endswith("Z")
    assert data["reset_at"].endswith("Z")
    assert reset_at - window_start == timedelta(minutes=1)
    assert window_start == window_start.replace(second=0, microsecond=0)
    # The reported window is the minute containing the request.
    assert before <= window_start <= after
    assert window_start <= after < reset_at + timedelta(minutes=1)


def test_every_budget_figure_is_a_json_integer_never_bool_or_float(client):
    data = _rate_limits(client).json()
    for name in BUDGET_NAMES:
        for field in BUDGET_FIELDS:
            value = data[name][field]
            assert isinstance(value, int) and not isinstance(value, bool), (
                name,
                field,
            )


# ---------------------------------------------------------------------------
# Request shape: every rejection is 422 before any state is read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"workload_id": WORKLOAD},                       # missing tenant
        {"tenant_id": TENANT},                           # missing workload
        {"tenant_id": "", "workload_id": WORKLOAD},      # empty tenant
        {"tenant_id": "   ", "workload_id": WORKLOAD},   # blank tenant
        {"tenant_id": "\t\n", "workload_id": WORKLOAD},  # whitespace tenant
        {"tenant_id": TENANT, "workload_id": ""},        # empty workload
        {"tenant_id": TENANT, "workload_id": "  "},      # blank workload
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "unexpected": "1"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "cursor": "x"},
    ],
)
def test_bad_query_parameters_are_422(app, client, params):
    response = client.request("GET", RATE_LIMITS_PATH, params=params)
    assert response.status_code == 422
    # No state was read or written: no counter exists even after many
    # rejected calls.
    with app.state.session_factory() as session:
        assert session.scalars(select(ChallengeIssuanceCounter)).all() == []
        assert session.scalars(select(VerificationAdmissionCounter)).all() == []
        assert session.scalars(select(RateLimitCounter)).all() == []


def test_duplicate_query_parameter_is_422(client):
    response = client.request(
        "GET",
        RATE_LIMITS_PATH
        + f"?tenant_id={TENANT}&tenant_id=other&workload_id={WORKLOAD}",
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        b"{}",
        b"[]",
        b"null",
        b'"x"',
        b"garbage",
        b" ",
        b"\t",
        b"\n",
        b" {} ",
    ],
)
def test_any_non_empty_body_is_422_before_state_reads(app, client, body):
    for _ in range(3):
        response = client.request(
            "GET",
            RATE_LIMITS_PATH,
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            content=body,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []
        assert session.scalars(select(AuditEvent)).all() == []


# ---------------------------------------------------------------------------
# The three budgets are reported independently
# ---------------------------------------------------------------------------


def test_challenge_issuance_budget_is_reported_independently(client):
    _challenge(client)
    _challenge(client)
    data = _rate_limits(client).json()
    assert data["challenge_issuance"] == {"limit": 5, "used": 2, "remaining": 3}
    assert data["verification"] == {"limit": 5, "used": 0, "remaining": 5}
    assert data["grant_actions"] == {"limit": 5, "used": 0, "remaining": 5}


def test_verification_budget_is_reported_independently(client):
    _verified_evidence(client)
    data = _rate_limits(client).json()
    # The flow issued one challenge and admitted one verification.
    assert data["challenge_issuance"]["used"] == 1
    assert data["verification"] == {"limit": 5, "used": 1, "remaining": 4}
    assert data["grant_actions"] == {"limit": 5, "used": 0, "remaining": 5}


def test_grant_action_budget_is_reported_independently(client):
    _grant_and_consume(client)
    data = _rate_limits(client).json()
    assert data["grant_actions"] == {"limit": 5, "used": 1, "remaining": 4}
    # The grant flow admitted exactly one verification and issued exactly
    # one challenge; neither budget was touched by the consume itself.
    assert data["verification"]["used"] == 1
    assert data["challenge_issuance"]["used"] == 1


def test_saturated_minute_reports_zero_remaining(client):
    for _ in range(5):
        _challenge(client)
    data = _rate_limits(client).json()
    assert data["challenge_issuance"] == {"limit": 5, "used": 5, "remaining": 0}
    # The other budgets are untouched by challenge issuance.
    assert data["verification"]["remaining"] == 5
    assert data["grant_actions"]["remaining"] == 5
    # The sixth issuance is rejected and changes nothing reported.
    rejected = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert rejected.status_code == 429
    assert _rate_limits(client).json()["challenge_issuance"]["used"] == 5


# ---------------------------------------------------------------------------
# The query itself is read-only: no budget, no audit, no state change
# ---------------------------------------------------------------------------


def test_query_consumes_no_budget_and_writes_no_audit(app, client):
    _verified_evidence(client)

    def counters():
        with app.state.session_factory() as session:
            return {
                (r.tenant_id, r.workload_id, r.window_start): r.count
                for model in (
                    ChallengeIssuanceCounter,
                    VerificationAdmissionCounter,
                    RateLimitCounter,
                )
                for r in session.scalars(select(model)).all()
            }

    def audit_count():
        with app.state.session_factory() as session:
            return len(session.scalars(select(AuditEvent)).all())

    before_counters = counters()
    before_audit = audit_count()
    for _ in range(6):
        response = _rate_limits(client)
        assert response.status_code == 200
        assert response.json()["challenge_issuance"]["used"] == 1
        assert response.json()["verification"]["used"] == 1
    assert counters() == before_counters
    assert audit_count() == before_audit


def test_query_never_returns_nonce_evidence_payload_or_material(client):
    created = _challenge(client)
    raw = _rate_limits(client).content
    text = raw.decode("utf-8")
    assert created["nonce"] not in text
    for forbidden in (
        '"nonce"',
        '"evidence"',
        '"payload"',
        '"ciphertext"',
        '"capability"',
        '"capability_sha256"',
        '"wrapped_key"',
        '"key_material"',
        '"certificate"',
    ):
        assert forbidden not in text


# ---------------------------------------------------------------------------
# Scope isolation
# ---------------------------------------------------------------------------


def test_budgets_are_strictly_scoped_per_tenant_and_workload(client):
    _challenge(client)
    _challenge(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)

    data = _rate_limits(client).json()
    assert data["challenge_issuance"]["used"] == 1

    other = _rate_limits(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()
    assert other["challenge_issuance"]["used"] == 1

    # A scope with no traffic reports full budgets, not a 404.
    empty = _rate_limits(client, tenant="nobody", workload="nothing").json()
    for name in BUDGET_NAMES:
        assert empty[name] == {"limit": 5, "used": 0, "remaining": 5}


# ---------------------------------------------------------------------------
# Persistence across restart
# ---------------------------------------------------------------------------


def test_usage_is_read_from_committed_state_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart_rate_limits.db"

    app1 = create_app(url)
    client1 = TestClient(app1)
    _challenge(client1)
    _challenge(client1)
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    data = _rate_limits(client2).json()
    assert data["challenge_issuance"] == {"limit": 5, "used": 2, "remaining": 3}
    app2.state.engine.dispose()


# ---------------------------------------------------------------------------
# Server failures: 500, never a partial summary, never a state change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table",
    [
        "challenge_issuance_counters",
        "verification_admission_counters",
        "rate_limit_counters",
    ],
)
def test_storage_failure_returns_500_without_partial_summary(app, client, table):
    with app.state.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE {table}"))
    response = _rate_limits(client)
    assert response.status_code == 500
    # No fabricated/partial figure reaches the client.
    assert response.content == b'{"detail":"rate limit summary unavailable"}'


# ---------------------------------------------------------------------------
# Existing entry points are unchanged
# ---------------------------------------------------------------------------


def test_release_summary_is_unchanged(client):
    response = client.get(
        "/v1/observability/release-summary",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    assert list(response.json()) == [
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


def test_health_endpoint_unchanged(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
