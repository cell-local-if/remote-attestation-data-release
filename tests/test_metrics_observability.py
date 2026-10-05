"""Tests for GET /v1/observability/metrics.

A read-only Prometheus text exposition of the same committed state the
JSON observability endpoints report: release-grant status totals (pending
split into live and expired), envelopes split by current versus
historical master key version, and the limit/used/remaining triple of
each per-minute budget. The endpoint never writes — it creates no
challenge, consumes no capability, rotates no key, appends no audit row
and reserves no rate-limit budget — and returns no payload, capability,
nonce, evidence or key material.
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
    DataEnvelope,
    RateLimitCounter,
    ReleaseGrant,
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
PAYLOAD = "scraped secret payload"
METRICS_PATH = "/v1/observability/metrics"

KEY_V1 = b64url_encode(b"0123456789abcdef0123456789abcdef")
KEY_V2 = b64url_encode(b"fedcba9876543210fedcba9876543210")
KEYRING_V1_V2_CURRENT_V2 = json.dumps(
    {"current_version": 2, "keys": {"1": KEY_V1, "2": KEY_V2}}
)

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

GRANT_STATUSES = ["pending", "consumed", "revoked"]
PENDING_STATES = ["live", "expired"]
KEY_CLASSES = ["current", "historical"]
BUDGETS = ["challenge_issuance", "verification", "grant_actions"]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/metrics.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _metrics(client, *, tenant=TENANT, workload=WORKLOAD, **kwargs):
    params = {"tenant_id": tenant, "workload_id": workload}
    params.update(kwargs.pop("params", {}))
    return client.request("GET", METRICS_PATH, params=params, **kwargs)


def _parse_samples(body: bytes) -> dict:
    """Parse sample lines into {(name, frozenset(label items)): value}."""
    samples = {}
    for line in body.decode("utf-8").splitlines():
        if line.startswith("#"):
            continue
        name, rest = line.split("{", 1)
        label_text, value = rest.rsplit("}", 1)
        labels = frozenset(
            tuple(pair.split("=", 1)) for pair in label_text.split(",")
        )
        samples[(name, labels)] = int(value.strip())
    return samples


def _sample(client, name, **labels):
    """Fetch one sample value for the default scope."""
    merged = {"tenant_id": f'"{TENANT}"', "workload_id": f'"{WORKLOAD}"'}
    merged.update({key: f'"{value}"' for key, value in labels.items()})
    samples = _parse_samples(_metrics(client).content)
    return samples[(name, frozenset(merged.items()))]


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
# Wire format
# ---------------------------------------------------------------------------


def test_content_type_and_single_trailing_newline(client):
    response = _metrics(client)
    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE
    raw = response.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")


def test_empty_scope_reports_zero_counts_and_full_budgets(client):
    assert _sample(client, "proof_release_release_grants_total", status="pending") == 0
    assert _sample(client, "proof_release_release_grants_total", status="consumed") == 0
    assert _sample(client, "proof_release_release_grants_total", status="revoked") == 0
    assert _sample(client, "proof_release_pending_release_grants", state="live") == 0
    assert _sample(client, "proof_release_pending_release_grants", state="expired") == 0
    assert _sample(client, "proof_release_data_envelopes_total", key_version="current") == 0
    assert _sample(client, "proof_release_data_envelopes_total", key_version="historical") == 0
    for budget in BUDGETS:
        assert (
            _sample(client, "proof_release_rate_limit_budget_limit", budget=budget)
            == 5
        )
        assert (
            _sample(client, "proof_release_rate_limit_budget_used", budget=budget)
            == 0
        )
        assert (
            _sample(
                client, "proof_release_rate_limit_budget_remaining", budget=budget
            )
            == 5
        )


def test_every_series_carries_both_scope_labels(client):
    samples = _parse_samples(_metrics(client).content)
    assert samples  # the exposition is never empty, even for an empty scope
    for _, labels in samples:
        label_keys = {key for key, _ in labels}
        assert {"tenant_id", "workload_id"} <= label_keys
        assert ("tenant_id", f'"{TENANT}"') in labels
        assert ("workload_id", f'"{WORKLOAD}"') in labels


def test_fixed_series_set_with_fixed_label_values(client):
    samples = _parse_samples(_metrics(client).content)
    expected = set()
    for status in GRANT_STATUSES:
        expected.add(("proof_release_release_grants_total", "status", status))
    for state in PENDING_STATES:
        expected.add(("proof_release_pending_release_grants", "state", state))
    for key_class in KEY_CLASSES:
        expected.add(("proof_release_data_envelopes_total", "key_version", key_class))
    for budget in BUDGETS:
        for kind in ("limit", "used", "remaining"):
            expected.add(
                (f"proof_release_rate_limit_budget_{kind}", "budget", budget)
            )
    actual = set()
    for (name, labels), _ in samples.items():
        extra = {(key, value.strip('"')) for key, value in labels} - {
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
        }
        assert len(extra) == 1
        (label_key, label_value), = extra
        actual.add((name, label_key, label_value))
    assert actual == expected


def test_help_and_type_precede_each_metric_group(client):
    lines = _metrics(client).content.decode("utf-8").splitlines()
    seen_metrics = []
    index = 0
    while index < len(lines):
        assert lines[index].startswith("# HELP ")
        metric = lines[index].split(" ", 3)[2]
        assert lines[index + 1] == f"# TYPE {metric} gauge"
        index += 2
        while index < len(lines) and not lines[index].startswith("#"):
            assert lines[index].startswith(metric + "{")
            index += 1
        seen_metrics.append(metric)
    # Metric groups are emitted in deterministic (name) order.
    assert seen_metrics == sorted(seen_metrics)


def test_output_is_byte_identical_across_scrapes(client):
    first = _metrics(client).content
    second = _metrics(client).content
    assert first == second


def test_sample_values_are_non_negative_decimal_integers(client):
    for line in _metrics(client).content.decode("utf-8").splitlines():
        if line.startswith("#"):
            continue
        value = line.rsplit(" ", 1)[1]
        assert value.isdigit()


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
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "format": "json"},
    ],
)
def test_bad_query_parameters_are_422(app, client, params):
    response = client.request("GET", METRICS_PATH, params=params)
    assert response.status_code == 422
    assert "detail" in response.json()
    # No state was read or written: no counter exists even after rejection.
    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []


def test_duplicate_query_parameter_is_422(client):
    response = client.request(
        "GET",
        METRICS_PATH
        + f"?tenant_id={TENANT}&tenant_id=other&workload_id={WORKLOAD}",
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [b"{}", b"[]", b"null", b'"x"', b"garbage", b" ", b"\t", b"\n", b" {} "],
)
def test_any_non_empty_body_is_422_before_state_reads(app, client, body):
    for _ in range(3):
        response = client.request(
            "GET",
            METRICS_PATH,
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            content=body,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.scalars(select(RateLimitCounter)).all() == []
        assert session.scalars(select(AuditEvent)).all() == []


# ---------------------------------------------------------------------------
# Populated aggregates driven through the real API
# ---------------------------------------------------------------------------


def test_grant_status_totals_and_pending_split(app, client):
    decision_id = _decision(client)
    _envelope(client, DATA_ID)
    _envelope(client, "data-2")

    _grant(client, decision_id, DATA_ID)
    _grant(client, decision_id, "data-2")
    expired = _grant(client, decision_id, "data-2")
    consumed = _grant(client, decision_id, DATA_ID)
    revoked = _grant(client, decision_id, "data-2")
    _expire_grant(app, expired["grant_id"])

    assert (
        client.post(
            f"/v1/release-grants/{consumed['grant_id']}/consume",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "capability": consumed["capability"],
            },
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/v1/release-grants/{revoked['grant_id']}/revoke",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "capability": revoked["capability"],
            },
        ).status_code
        == 200
    )

    assert _sample(client, "proof_release_release_grants_total", status="pending") == 3
    assert _sample(client, "proof_release_release_grants_total", status="consumed") == 1
    assert _sample(client, "proof_release_release_grants_total", status="revoked") == 1
    live = _sample(client, "proof_release_pending_release_grants", state="live")
    expired_count = _sample(
        client, "proof_release_pending_release_grants", state="expired"
    )
    assert live == 2
    assert expired_count == 1
    # The pending split is an exact partition of the pending total.
    assert live + expired_count == 3


def test_envelope_split_by_current_master_key_version(app, client, monkeypatch):
    _envelope(client, "a")
    _envelope(client, "b")
    assert _sample(client, "proof_release_data_envelopes_total", key_version="current") == 2
    assert _sample(client, "proof_release_data_envelopes_total", key_version="historical") == 0

    # Rotate the configured current version to v2 without rewrapping: both
    # stored envelopes are now historical.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2_CURRENT_V2)
    assert _sample(client, "proof_release_data_envelopes_total", key_version="current") == 0
    assert _sample(client, "proof_release_data_envelopes_total", key_version="historical") == 2

    # A freshly created envelope uses v2.
    _envelope(client, "c")
    assert _sample(client, "proof_release_data_envelopes_total", key_version="current") == 1
    assert _sample(client, "proof_release_data_envelopes_total", key_version="historical") == 2


def test_budget_triples_reflect_spent_slots(client):
    # One challenge issuance spends one challenge-issuance slot; the
    # decision flow also spends one verification slot; one grant consume
    # spends one shared grant-actions slot.
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

    for budget, used in (
        ("challenge_issuance", 1),
        ("verification", 1),
        ("grant_actions", 1),
    ):
        assert (
            _sample(client, "proof_release_rate_limit_budget_limit", budget=budget)
            == 5
        )
        assert (
            _sample(client, "proof_release_rate_limit_budget_used", budget=budget)
            == used
        )
        assert (
            _sample(
                client, "proof_release_rate_limit_budget_remaining", budget=budget
            )
            == 5 - used
        )


def test_saturated_minute_reports_zero_remaining(app, client):
    decision_id = _decision(client)
    _envelope(client, DATA_ID)
    grant = _grant(client, decision_id, DATA_ID)
    for _ in range(app_module.GRANT_BUDGET_PER_MINUTE):
        client.post(
            f"/v1/release-grants/{grant['grant_id']}/consume",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "capability": grant["capability"],
            },
        )
    assert (
        _sample(client, "proof_release_rate_limit_budget_used", budget="grant_actions")
        == 5
    )
    assert (
        _sample(
            client, "proof_release_rate_limit_budget_remaining", budget="grant_actions"
        )
        == 0
    )


# ---------------------------------------------------------------------------
# The scrape itself is read-only: no budget, no audit, no state change
# ---------------------------------------------------------------------------


def test_scrape_consumes_no_budget_and_writes_nothing(app, client):
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

    def counters():
        with app.state.session_factory() as session:
            return {
                model.__name__: {
                    (r.tenant_id, r.workload_id, r.window_start): r.count
                    for r in session.scalars(select(model)).all()
                }
                for model in (
                    RateLimitCounter,
                    ChallengeIssuanceCounter,
                    VerificationAdmissionCounter,
                )
            }

    def audit_count():
        with app.state.session_factory() as session:
            return len(session.scalars(select(AuditEvent)).all())

    before_counters = counters()
    before_audit = audit_count()
    for _ in range(6):
        response = _metrics(client)
        assert response.status_code == 200
    assert counters() == before_counters
    assert audit_count() == before_audit
    # The scrape created no challenge, grant, envelope or lifecycle state.
    with app.state.session_factory() as session:
        assert len(session.scalars(select(ReleaseGrant)).all()) == 1
        assert len(session.scalars(select(DataEnvelope)).all()) == 1


def test_scrape_never_returns_capability_payload_or_material(client):
    decision_id = _decision(client)
    _envelope(client, DATA_ID, payload=PAYLOAD)
    grant = _grant(client, decision_id, DATA_ID)
    text = _metrics(client).content.decode("utf-8")
    assert PAYLOAD not in text
    assert grant["capability"] not in text
    for forbidden in (
        "capability",
        "payload",
        "ciphertext",
        "wrapped_key",
        "nonce",
        "evidence",
        "key_material",
    ):
        assert forbidden not in text


# ---------------------------------------------------------------------------
# Label escaping and scope isolation
# ---------------------------------------------------------------------------


def test_label_values_escape_backslash_quote_and_newline(client):
    tenant = 'quo"te\\back\nslash'
    response = _metrics(client, tenant=tenant)
    assert response.status_code == 200
    text = response.content.decode("utf-8")
    # The escaped form appears; the raw characters never break a line.
    assert 'tenant_id="quo\\"te\\\\back\\nslash"' in text
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        assert line.count("{") == 1 and line.count("}") == 1


def test_metrics_are_strictly_scoped_per_tenant_and_workload(app, client):
    decision_id = _decision(client)
    _envelope(client, DATA_ID)
    _grant(client, decision_id, DATA_ID)

    other_decision = _decision(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    _envelope(client, "other", tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    _grant(client, other_decision, "other", tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)

    assert _sample(client, "proof_release_release_grants_total", status="pending") == 1
    assert _sample(client, "proof_release_data_envelopes_total", key_version="current") == 1

    # A scope with no traffic reports zeros, not a 404.
    response = _metrics(client, tenant="nobody", workload="nothing")
    assert response.status_code == 200
    samples = _parse_samples(response.content)
    assert samples
    for (name, labels), value in samples.items():
        if name in (
            "proof_release_rate_limit_budget_limit",
            "proof_release_rate_limit_budget_remaining",
        ):
            # An untouched scope has the full budget still available.
            assert value == 5
        else:
            assert value == 0


# ---------------------------------------------------------------------------
# Server failures: 500, never a partial exposition, never a state change
# ---------------------------------------------------------------------------


def test_storage_failure_returns_500_without_partial_exposition(app, client):
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE release_grants"))
    response = _metrics(client)
    assert response.status_code == 500
    assert response.content == b'{"detail":"metrics unavailable"}'


def test_unusable_keyring_returns_500(app, client, monkeypatch):
    _envelope(client, DATA_ID)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "{this is not json")
    response = _metrics(client)
    assert response.status_code == 500
    assert response.content == b'{"detail":"master key configuration unavailable"}'


def test_keyring_missing_current_key_returns_500(app, client, monkeypatch):
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING",
        json.dumps({"current_version": 3, "keys": {"1": KEY_V1}}),
    )
    response = _metrics(client)
    assert response.status_code == 500
    assert response.content == b'{"detail":"master key configuration unavailable"}'


# ---------------------------------------------------------------------------
# Existing entry points are unchanged
# ---------------------------------------------------------------------------


def test_health_endpoint_unchanged(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_json_observability_endpoints_unchanged(client):
    params = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    summary = client.get("/v1/observability/release-summary", params=params)
    assert summary.status_code == 200
    assert summary.headers["content-type"] == "application/json"
    rate_limits = client.get("/v1/observability/rate-limits", params=params)
    assert rate_limits.status_code == 200
    assert rate_limits.headers["content-type"] == "application/json"
