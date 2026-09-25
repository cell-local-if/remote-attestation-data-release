"""Tests for GET /v1/observability/release-summary.

A read-only, point-in-time summary of committed release state for one
tenant/workload: grant status counts (pending split into live and
expired), the current UTC minute's shared rate-limit usage, and the
envelope population split by current versus historical master key
version. The endpoint never writes — it consumes no rate-limit budget,
appends no audit row, and returns no capability, payload, evidence or
key material.
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
from proof_release.db import (
    AuditEvent,
    DataEnvelope,
    RateLimitCounter,
    ReleaseGrant,
)
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
MASTER_KEY = b64url_encode(b"0123456789abcdef0123456789abcdef")
PAYLOAD = "summarized secret payload"
SUMMARY_PATH = "/v1/observability/release-summary"

KEY_V1 = b64url_encode(b"0123456789abcdef0123456789abcdef")
KEY_V2 = b64url_encode(b"fedcba9876543210fedcba9876543210")
KEYRING_V1_V2_CURRENT_V2 = json.dumps(
    {"current_version": 2, "keys": {"1": KEY_V1, "2": KEY_V2}}
)

EXPECTED_FIELD_ORDER = [
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
COUNT_FIELDS = EXPECTED_FIELD_ORDER[2:]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/release_summary.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _summary(client, *, tenant=TENANT, workload=WORKLOAD, **kwargs):
    params = {"tenant_id": tenant, "workload_id": workload}
    params.update(kwargs.pop("params", {}))
    return client.request("GET", SUMMARY_PATH, params=params, **kwargs)


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


def _decision(client, *, tenant=TENANT, workload=WORKLOAD):
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
    return decided.json()["decision_id"]


def _envelope(client, data_id, *, payload=PAYLOAD, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": payload,
        },
    )
    assert response.status_code == 201
    return response.json()


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


def _expire_grant(app, grant_id):
    """Force a still-pending grant past its validity window."""
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant_id)
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()


# ---------------------------------------------------------------------------
# Empty range and wire format
# ---------------------------------------------------------------------------


def test_empty_range_reports_zero_counts_and_full_budget(client):
    response = _summary(client)
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
    assert data["pending"] == 0
    assert data["consumed"] == 0
    assert data["revoked"] == 0
    assert data["live_pending"] == 0
    assert data["expired_pending"] == 0
    assert data["rate_used"] == 0
    assert data["rate_remaining"] == app_module.GRANT_BUDGET_PER_MINUTE
    assert data["envelopes"] == 0
    assert data["current_key_envelopes"] == 0
    assert data["historical_key_envelopes"] == 0


def test_response_is_exactly_twelve_compact_fields_with_one_newline(client):
    raw = _summary(client).content
    assert raw == (
        json.dumps(
            {
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "pending": 0,
                "consumed": 0,
                "revoked": 0,
                "live_pending": 0,
                "expired_pending": 0,
                "rate_used": 0,
                "rate_remaining": app_module.GRANT_BUDGET_PER_MINUTE,
                "envelopes": 0,
                "current_key_envelopes": 0,
                "historical_key_envelopes": 0,
            },
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def test_every_count_is_a_json_integer_never_bool_or_float(client):
    data = _summary(client).json()
    for field in COUNT_FIELDS:
        value = data[field]
        assert isinstance(value, int) and not isinstance(value, bool), field
    assert isinstance(data["tenant_id"], str)
    assert isinstance(data["workload_id"], str)


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
    response = client.request("GET", SUMMARY_PATH, params=params)
    assert response.status_code == 422
    # No state was read or written: no rate counter exists even after many
    # rejected calls.
    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []


def test_duplicate_query_parameter_is_422(client):
    response = client.request(
        "GET",
        SUMMARY_PATH
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
            SUMMARY_PATH,
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            content=body,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []
        assert session.scalars(select(AuditEvent)).all() == []


def test_422_rejects_before_state_is_observed(app, client):
    # Populate state; a shape rejection must not need it and must not leak.
    _envelope(client, DATA_ID)
    response = client.request(
        "GET",
        SUMMARY_PATH,
        params={"tenant_id": TENANT},  # workload missing
        content=b'{"tenant_id":"x"}',
    )
    assert response.status_code == 422
    assert PAYLOAD.encode() not in response.content


# ---------------------------------------------------------------------------
# Populated aggregates driven through the real API
# ---------------------------------------------------------------------------


def test_status_counts_pending_split_and_rate_usage(app, client):
    decision_id = _decision(client)
    _envelope(client, DATA_ID)
    _envelope(client, "data-2")

    live_one = _grant(client, decision_id, DATA_ID)
    live_two = _grant(client, decision_id, "data-2")
    expired = _grant(client, decision_id, "data-2")
    consumed = _grant(client, decision_id, DATA_ID)
    revoked = _grant(client, decision_id, "data-2")
    _expire_grant(app, expired["grant_id"])

    # One consume and one revoke reserve two of the five shared slots.
    consume = client.post(
        f"/v1/release-grants/{consumed['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": consumed["capability"],
        },
    )
    assert consume.status_code == 200
    revoke = client.post(
        f"/v1/release-grants/{revoked['grant_id']}/revoke",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": revoked["capability"],
        },
    )
    assert revoke.status_code == 200

    data = _summary(client).json()
    assert data["pending"] == 3
    assert data["consumed"] == 1
    assert data["revoked"] == 1
    assert data["live_pending"] == 2
    assert data["expired_pending"] == 1
    # The pending split is an exact partition of the pending total.
    assert data["live_pending"] + data["expired_pending"] == data["pending"]
    assert data["rate_used"] == 2
    assert data["rate_remaining"] == 3
    assert data["envelopes"] == 2
    assert data["current_key_envelopes"] == 2
    assert data["historical_key_envelopes"] == 0
    # The live grants really are the two unexpired ones.
    assert {live_one["grant_id"], live_two["grant_id"]}  # sanity handles present
    assert expired["grant_id"] not in {
        live_one["grant_id"],
        live_two["grant_id"],
    }


def test_expired_pending_grants_are_still_pending_not_settled(app, client):
    decision_id = _decision(client)
    grant = _grant(client, decision_id, DATA_ID)
    _expire_grant(app, grant["grant_id"])

    data = _summary(client).json()
    assert data["pending"] == 1
    assert data["consumed"] == 0
    assert data["revoked"] == 0
    assert data["live_pending"] == 0
    assert data["expired_pending"] == 1


def test_saturated_minute_reports_zero_remaining(app, client):
    decision_id = _decision(client)
    _envelope(client, DATA_ID)
    grant = _grant(client, decision_id, DATA_ID)
    # Spend all five slots: one settles, four observe the consumed state.
    for _ in range(app_module.GRANT_BUDGET_PER_MINUTE):
        client.post(
            f"/v1/release-grants/{grant['grant_id']}/consume",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "capability": grant["capability"],
            },
        )
    data = _summary(client).json()
    assert data["rate_used"] == 5
    assert data["rate_remaining"] == 0


# ---------------------------------------------------------------------------
# The summary itself is read-only: no budget, no audit, no state change
# ---------------------------------------------------------------------------


def test_summary_consumes_no_rate_budget_and_writes_no_audit(app, client):
    decision_id = _decision(client)
    _envelope(client, DATA_ID)
    grant = _grant(client, decision_id, DATA_ID)
    # Use one slot on a real business request.
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

    def counters():
        with app.state.session_factory() as session:
            return {
                (r.tenant_id, r.workload_id, r.window_start): r.count
                for r in session.scalars(select(RateLimitCounter)).all()
            }

    def audit_count():
        with app.state.session_factory() as session:
            return len(session.scalars(select(AuditEvent)).all())

    before_counters = counters()
    before_audit = audit_count()
    for _ in range(6):
        response = _summary(client)
        assert response.status_code == 200
        assert response.json()["rate_used"] == 1
        assert response.json()["rate_remaining"] == 4
    assert counters() == before_counters
    assert audit_count() == before_audit


def test_summary_never_returns_capability_payload_or_material(client, app):
    decision_id = _decision(client)
    _envelope(client, DATA_ID, payload=PAYLOAD)
    grant = _grant(client, decision_id, DATA_ID)
    raw = _summary(client).content
    text = raw.decode("utf-8")
    assert PAYLOAD not in text
    assert grant["capability"] not in text
    # No material/capability field name is ever emitted (check quoted JSON
    # keys so a harmless substring like "iv" inside "live_pending" does not
    # confuse the assertion).
    for forbidden in (
        '"capability"',
        '"capability_sha256"',
        '"payload"',
        '"ciphertext"',
        '"wrapped_key"',
        '"nonce"',
        '"evidence"',
        '"iv"',
        '"tag"',
        '"key_material"',
        '"key_version"',
    ):
        assert forbidden not in text


# ---------------------------------------------------------------------------
# Envelope migration classification by current master key version
# ---------------------------------------------------------------------------


def test_envelopes_classified_current_vs_historical_key_version(
    app, client, monkeypatch
):
    # Created while v1 is current.
    _envelope(client, "a")
    _envelope(client, "b")
    data = _summary(client).json()
    assert data["envelopes"] == 2
    assert data["current_key_envelopes"] == 2
    assert data["historical_key_envelopes"] == 0

    # Rotate the configured current version to v2 without rewrapping:
    # both stored envelopes are now historical.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2_CURRENT_V2)
    data = _summary(client).json()
    assert data["envelopes"] == 2
    assert data["current_key_envelopes"] == 0
    assert data["historical_key_envelopes"] == 2

    # A freshly created envelope uses v2; the other two stay historical.
    _envelope(client, "c")
    data = _summary(client).json()
    assert data["envelopes"] == 3
    assert data["current_key_envelopes"] == 1
    assert data["historical_key_envelopes"] == 2

    # Rewrap one historical envelope to v2.
    rewrap = client.post(
        "/v1/data-envelopes/a/rewrap",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert rewrap.status_code == 200
    assert rewrap.json()["key_version"] == 2
    data = _summary(client).json()
    assert data["envelopes"] == 3
    assert data["current_key_envelopes"] == 2
    assert data["historical_key_envelopes"] == 1
    # The two groups always partition the whole envelope population.
    assert (
        data["current_key_envelopes"] + data["historical_key_envelopes"]
        == data["envelopes"]
    )


# ---------------------------------------------------------------------------
# Scope isolation
# ---------------------------------------------------------------------------


def test_summary_is_strictly_scoped_per_tenant_and_workload(app, client):
    decision_id = _decision(client)
    _envelope(client, DATA_ID)
    _grant(client, decision_id, DATA_ID)

    other_decision = _decision(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    _envelope(
        client,
        "other",
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
    )
    _grant(
        client,
        other_decision,
        "other",
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
    )

    data = _summary(client).json()
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["pending"] == 1
    assert data["envelopes"] == 1

    other = _summary(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()
    assert other["pending"] == 1
    assert other["envelopes"] == 1

    # A scope with no traffic is an empty range, not a 404.
    empty = _summary(client, tenant="nobody", workload="nothing").json()
    assert empty["pending"] == 0
    assert empty["envelopes"] == 0
    assert empty["rate_remaining"] == app_module.GRANT_BUDGET_PER_MINUTE


# ---------------------------------------------------------------------------
# Persistence across restart
# ---------------------------------------------------------------------------


def test_summary_reads_committed_state_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart_summary.db"

    app1 = create_app(url)
    client1 = TestClient(app1)
    decision_id = _decision(client1)
    _envelope(client1, DATA_ID)
    _grant(client1, decision_id, DATA_ID)
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    data = _summary(client2).json()
    assert data["pending"] == 1
    assert data["live_pending"] == 1
    assert data["envelopes"] == 1
    assert data["current_key_envelopes"] == 1
    app2.state.engine.dispose()


# ---------------------------------------------------------------------------
# Server failures: 500, never a partial summary, never a state change
# ---------------------------------------------------------------------------


def test_storage_failure_returns_500_without_partial_summary(app, client):
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE release_grants"))
    response = _summary(client)
    assert response.status_code == 500
    # No fabricated/partial figure reaches the client.
    assert response.content == b'{"detail":"release summary unavailable"}'


def test_unusable_keyring_returns_500(app, client, monkeypatch):
    _envelope(client, DATA_ID)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "{this is not json")
    response = _summary(client)
    assert response.status_code == 500
    assert response.content == b'{"detail":"release summary unavailable"}'


def test_keyring_missing_current_key_returns_500(app, client, monkeypatch):
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING",
        json.dumps({"current_version": 3, "keys": {"1": KEY_V1}}),
    )
    assert _summary(client).status_code == 500


# ---------------------------------------------------------------------------
# Concurrent commits never produce a half-applied summary
# ---------------------------------------------------------------------------


def test_concurrent_commits_never_show_inconsistent_groups(app, client):
    decision_id = _decision(client)

    def mint_one(index):
        thread_client = TestClient(app)
        _envelope(thread_client, f"d-{index}")
        _grant(thread_client, decision_id, f"d-{index}")

    def summarize():
        thread_client = TestClient(app)
        data = _summary(thread_client).json()
        # Internal partitions must always hold, regardless of how many
        # mints committed between the aggregated subqueries.
        assert (
            data["pending"] + data["consumed"] + data["revoked"]
            >= data["pending"]
        )
        assert data["live_pending"] + data["expired_pending"] == data["pending"]
        assert (
            data["current_key_envelopes"] + data["historical_key_envelopes"]
            == data["envelopes"]
        )
        return data["pending"], data["envelopes"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for index in range(8):
            futures.append(pool.submit(mint_one, index))
            futures.append(pool.submit(summarize))
        for future in futures:
            future.result()

    final = _summary(client).json()
    assert final["pending"] == 8
    assert final["envelopes"] == 8
    assert final["live_pending"] == 8
    assert final["expired_pending"] == 0


# ---------------------------------------------------------------------------
# Existing entry points are unchanged
# ---------------------------------------------------------------------------


def test_health_endpoint_unchanged(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
