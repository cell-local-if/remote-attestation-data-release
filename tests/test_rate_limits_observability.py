"""Tests for GET /v1/observability/rate-limits.

A read-only, point-in-time snapshot of the three independent per-(tenant,
workload) UTC-natural-minute rate budgets: challenge issuance, evidence
verification and one-time-grant actions. The endpoint never writes — it
issues no challenge, verifies no evidence, creates or consumes no grant,
changes no counter, appends no audit or lifecycle event, and returns no
nonce, evidence, payload, capability or key material.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    ChallengeIssuanceCounter,
    ProofLifecycleEvent,
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
BUDGET_FIELDS = ["challenge_issuance", "verification", "grant_actions"]
BUDGET_OBJECT_ORDER = ["limit", "used", "remaining"]


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
    """Issue a challenge and verify one piece of evidence against it."""
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
    return created, evidence_id, evidence


def _grant(client, decision_id, data_id, *, tenant=TENANT, workload=WORKLOAD):
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


def _decision(client, *, tenant=TENANT, workload=WORKLOAD):
    created, evidence_id, evidence = _verified_evidence(
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
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert decided.status_code == 200
    return decided.json()["decision_id"]


def _envelope(client, data_id, *, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": "observability secret payload",
        },
    )
    assert response.status_code == 201
    return response.json()


def _all_counters(app):
    with app.state.session_factory() as session:
        return {
            "challenge": [
                (r.tenant_id, r.workload_id, r.window_start, r.count)
                for r in session.scalars(select(ChallengeIssuanceCounter)).all()
            ],
            "verification": [
                (r.tenant_id, r.workload_id, r.window_start, r.count)
                for r in session.scalars(select(VerificationAdmissionCounter)).all()
            ],
            "grant": [
                (r.tenant_id, r.workload_id, r.window_start, r.count)
                for r in session.scalars(select(RateLimitCounter)).all()
            ],
        }


# ---------------------------------------------------------------------------
# Empty scope and wire format
# ---------------------------------------------------------------------------


def test_empty_scope_reports_zero_used_and_full_budgets(client):
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
    for field in BUDGET_FIELDS:
        budget = data[field]
        assert list(budget) == BUDGET_OBJECT_ORDER
        assert budget == {"limit": 5, "used": 0, "remaining": 5}


def test_response_is_exactly_compact_json_with_one_newline(client):
    raw = _rate_limits(client).content
    data = json.loads(raw)
    assert raw == (
        json.dumps(data, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )


def test_window_and_reset_are_utc_z_timestamps_of_the_same_minute(client):
    before = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    data = _rate_limits(client).json()
    after = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    window_start = data["window_start"]
    reset_at = data["reset_at"]
    assert window_start.endswith("Z")
    assert reset_at.endswith("Z")
    parsed_start = datetime.fromisoformat(window_start)
    parsed_reset = datetime.fromisoformat(reset_at)
    # The window is the current UTC natural minute; reset is the next one.
    assert before <= parsed_start <= after
    assert parsed_reset - parsed_start == timedelta(minutes=1)
    assert parsed_start.second == 0 and parsed_start.microsecond == 0
    assert parsed_reset.second == 0 and parsed_reset.microsecond == 0


def test_every_budget_figure_is_a_json_integer_never_bool_or_float(client):
    data = _rate_limits(client).json()
    for field in BUDGET_FIELDS:
        for key in BUDGET_OBJECT_ORDER:
            value = data[field][key]
            assert isinstance(value, int) and not isinstance(value, bool), (
                field,
                key,
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
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": "5"},
    ],
)
def test_bad_query_parameters_are_422(app, client, params):
    response = client.request("GET", RATE_LIMITS_PATH, params=params)
    assert response.status_code == 422
    # No state was read or written: no counter row exists even after many
    # rejected calls.
    assert _all_counters(app) == {"challenge": [], "verification": [], "grant": []}


def test_duplicate_query_parameter_is_422(client):
    response = client.request(
        "GET",
        RATE_LIMITS_PATH
        + f"?tenant_id={TENANT}&tenant_id=other&workload_id={WORKLOAD}",
    )
    assert response.status_code == 422
    response = client.request(
        "GET",
        RATE_LIMITS_PATH
        + f"?tenant_id={TENANT}&workload_id={WORKLOAD}&workload_id=other",
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
    assert _all_counters(app) == {"challenge": [], "verification": [], "grant": []}
    with app.state.session_factory() as session:
        assert session.scalars(select(AuditEvent)).all() == []


# ---------------------------------------------------------------------------
# The three budgets are reported independently from persisted counters
# ---------------------------------------------------------------------------


def test_challenge_issuance_usage_is_reported(client):
    _challenge(client)
    _challenge(client)

    data = _rate_limits(client).json()
    assert data["challenge_issuance"] == {"limit": 5, "used": 2, "remaining": 3}
    # The other two budgets are untouched by issuance.
    assert data["verification"] == {"limit": 5, "used": 0, "remaining": 5}
    assert data["grant_actions"] == {"limit": 5, "used": 0, "remaining": 5}


def test_verification_usage_is_reported(client):
    _verified_evidence(client)

    data = _rate_limits(client).json()
    # One challenge issued and one verification admitted.
    assert data["challenge_issuance"]["used"] == 1
    assert data["verification"] == {"limit": 5, "used": 1, "remaining": 4}
    assert data["grant_actions"] == {"limit": 5, "used": 0, "remaining": 5}


def test_grant_action_usage_is_reported(client):
    decision_id = _decision(client)
    _envelope(client, DATA_ID)
    grant = _grant(client, decision_id, DATA_ID)
    consume = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )
    assert consume.status_code == 200

    data = _rate_limits(client).json()
    assert data["challenge_issuance"]["used"] == 1
    assert data["verification"]["used"] == 1
    assert data["grant_actions"] == {"limit": 5, "used": 1, "remaining": 4}


def test_saturated_budget_reports_zero_remaining(client):
    for _ in range(app_module.CHALLENGE_ISSUANCE_BUDGET_PER_MINUTE):
        response = client.post(
            "/v1/challenges",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
        assert response.status_code == 201

    data = _rate_limits(client).json()
    assert data["challenge_issuance"] == {"limit": 5, "used": 5, "remaining": 0}
    # Saturating issuance never touches the other budgets.
    assert data["verification"] == {"limit": 5, "used": 0, "remaining": 5}
    assert data["grant_actions"] == {"limit": 5, "used": 0, "remaining": 5}


def test_limits_match_the_existing_budget_constants(client):
    data = _rate_limits(client).json()
    assert data["challenge_issuance"]["limit"] == (
        app_module.CHALLENGE_ISSUANCE_BUDGET_PER_MINUTE
    )
    assert data["verification"]["limit"] == app_module.VERIFICATION_BUDGET_PER_MINUTE
    assert data["grant_actions"]["limit"] == app_module.GRANT_BUDGET_PER_MINUTE


# ---------------------------------------------------------------------------
# The query itself is read-only: no budget, no audit, no state change
# ---------------------------------------------------------------------------


def test_query_consumes_no_budget_and_writes_nothing(app, client):
    decision_id = _decision(client)
    _envelope(client, DATA_ID)
    grant = _grant(client, decision_id, DATA_ID)
    assert (
        client.post(
            f"/v1/release-grants/{grant['grant_id']}/consume",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "capability": grant["capability"],
            },
        ).status_code
        == 200
    )

    def audit_and_events():
        with app.state.session_factory() as session:
            return (
                len(session.scalars(select(AuditEvent)).all()),
                len(session.scalars(select(ProofLifecycleEvent)).all()),
            )

    before_counters = _all_counters(app)
    before_audit = audit_and_events()
    for _ in range(8):
        response = _rate_limits(client)
        assert response.status_code == 200
        data = response.json()
        assert data["challenge_issuance"]["used"] == 1
        assert data["verification"]["used"] == 1
        assert data["grant_actions"]["used"] == 1
    assert _all_counters(app) == before_counters
    assert audit_and_events() == before_audit


def test_query_never_returns_nonce_evidence_payload_or_material(client):
    decision_id = _decision(client)
    _envelope(client, DATA_ID)
    grant = _grant(client, decision_id, DATA_ID)
    raw = _rate_limits(client).content
    text = raw.decode("utf-8")
    assert grant["capability"] not in text
    for forbidden in (
        '"nonce"',
        '"evidence"',
        '"payload"',
        '"capability"',
        '"capability_sha256"',
        '"ciphertext"',
        '"wrapped_key"',
        '"key_material"',
        '"key_version"',
        '"certificate"',
        '"secret"',
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
    assert other["tenant_id"] == OTHER_TENANT
    assert other["workload_id"] == OTHER_WORKLOAD
    assert other["challenge_issuance"]["used"] == 1

    # A scope with no traffic is an all-zero snapshot, not a 404.
    empty = _rate_limits(client, tenant="nobody", workload="nothing").json()
    for field in BUDGET_FIELDS:
        assert empty[field] == {"limit": 5, "used": 0, "remaining": 5}


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


def test_release_summary_still_reports_only_the_shared_grant_budget(client):
    decision_id = _decision(client)
    _envelope(client, DATA_ID)
    grant = _grant(client, decision_id, DATA_ID)
    assert (
        client.post(
            f"/v1/release-grants/{grant['grant_id']}/consume",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "capability": grant["capability"],
            },
        ).status_code
        == 200
    )

    summary = client.request(
        "GET",
        "/v1/observability/release-summary",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert summary.status_code == 200
    data = summary.json()
    # release-summary keeps its original twelve-field shape and semantics.
    assert list(data) == [
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
    assert data["rate_used"] == 1
    assert data["rate_remaining"] == 4


def test_health_endpoint_unchanged(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
