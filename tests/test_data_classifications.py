"""Tests for immutable data classifications and their release constraints.

Covers:

* ``PUT`` / ``GET /v1/data-classes/{data_id}`` — immutable assignment with
  idempotent same-code replay, ``409`` reclassification rejection, envelope
  existence and scope rules;
* the optional ``classification`` on ``POST /v1/release-grants`` — bound
  grants, ``404``/``409`` preconditions with no grant/capability/audit
  created, and the unchanged legacy path when omitted;
* the atomic classification check on ``POST /v1/release/{grant_id}``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import ReleaseGrant
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
MASTER_KEY_BYTES = b"0123456789abcdef0123456789abcdef"
MASTER_KEY = b64url_encode(MASTER_KEY_BYTES)
PAYLOAD = "super-secret classified payload"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/classes.db")
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


def _envelope(client, data_id=DATA_ID, payload=PAYLOAD, **scope):
    body = {
        "tenant_id": scope.get("tenant_id", TENANT),
        "workload_id": scope.get("workload_id", WORKLOAD),
        "data_id": data_id,
        "payload": payload,
    }
    response = client.post("/v1/data-envelopes", json=body)
    assert response.status_code == 201
    return response.json()


def _put_class(client, code, data_id=DATA_ID, **scope):
    body = {
        "tenant_id": scope.get("tenant_id", TENANT),
        "workload_id": scope.get("workload_id", WORKLOAD),
        "classification": code,
    }
    return client.put(f"/v1/data-classes/{data_id}", json=body)


def _get_class(client, data_id=DATA_ID, **scope):
    return client.get(
        f"/v1/data-classes/{data_id}",
        params={
            "tenant_id": scope.get("tenant_id", TENANT),
            "workload_id": scope.get("workload_id", WORKLOAD),
        },
    )


def _grant(client, decision_id, data_id=DATA_ID, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision_id,
        "data_id": data_id,
    }
    body.update(fields)
    return client.post("/v1/release-grants", json=body)


def _release(client, grant_id, capability, data_id=DATA_ID):
    return client.post(
        f"/v1/release/{grant_id}",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": data_id,
            "capability": capability,
        },
    )


# --- PUT /v1/data-classes -------------------------------------------------


def test_put_classification_returns_200_with_five_fields(client):
    _envelope(client)

    response = _put_class(client, "public")

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "data_id",
        "tenant_id",
        "workload_id",
        "classification",
        "assigned_at",
    }
    assert data["data_id"] == DATA_ID
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["classification"] == "public"
    assigned_at = data["assigned_at"]
    assert assigned_at.endswith("+00:00")
    assert datetime.fromisoformat(assigned_at).utcoffset() == timedelta(0)
    # No envelope material or other metadata ever appears.
    for secret_name in ("payload", "ciphertext", "iv", "tag", "wrapped_key"):
        assert secret_name not in data


def test_get_returns_same_five_fields_without_material(client):
    _envelope(client)
    put = _put_class(client, "restricted-1")
    assert put.status_code == 200

    response = _get_class(client)

    assert response.status_code == 200
    assert response.json() == put.json()
    assert PAYLOAD not in response.text


def test_put_requires_existing_envelope(client):
    response = _put_class(client, "public")
    assert response.status_code == 404
    assert response.json()["detail"] == "data envelope not found"
    # Nothing was assigned.
    assert _get_class(client).status_code == 404


def test_put_unknown_and_cross_scope_envelope_are_indistinguishable_404(client):
    _envelope(client)
    assert _put_class(client, "public", data_id="other-item").status_code == 404
    assert (
        _put_class(client, "public", tenant_id="tenant-b").status_code == 404
    )
    assert (
        _put_class(client, "public", workload_id="workload-2").status_code
        == 404
    )


def test_get_unclassified_envelope_returns_404_classification_not_found(client):
    _envelope(client)
    response = _get_class(client)
    assert response.status_code == 404
    assert response.json()["detail"] == "classification not found"


def test_get_unknown_and_cross_scope_are_404(client):
    _envelope(client)
    assert _put_class(client, "public").status_code == 200
    assert _get_class(client, data_id="other-item").status_code == 404
    assert _get_class(client, tenant_id="tenant-b").status_code == 404
    assert _get_class(client, workload_id="workload-2").status_code == 404


@pytest.mark.parametrize(
    "code",
    [
        "Public",       # no uppercase
        "a_b",          # no underscore
        "a b",          # no spaces
        "a.b",          # no dot
        "-alpha",       # no leading hyphen
        "alpha-",       # no trailing hyphen
        "-",            # hyphen only
        "--",
        "a" * 65,       # too long
        "éclair",       # non-ASCII
        "",             # empty (rejected by model min length as 422)
    ],
)
def test_put_rejects_malformed_classification_codes_with_422(client, code):
    _envelope(client)
    response = _put_class(client, code)
    assert response.status_code == 422
    assert _get_class(client).status_code == 404


def test_internal_consecutive_hyphens_are_allowed(client):
    _envelope(client)
    response = _put_class(client, "alpha--beta")
    assert response.status_code == 200
    assert response.json()["classification"] == "alpha--beta"


def test_classification_allows_single_characters_and_max_length(client):
    _envelope(client, data_id="d1")
    assert _put_class(client, "a", data_id="d1").status_code == 200
    _envelope(client, data_id="d2")
    long_code = "a" * 64
    response = _put_class(client, long_code, data_id="d2")
    assert response.status_code == 200
    assert response.json()["classification"] == long_code
    # A single digit and a digit-only code are legal codes too.
    _envelope(client, data_id="d3")
    assert _put_class(client, "0", data_id="d3").status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"tenant_id": TENANT, "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "classification": "public"},
        {"workload_id": WORKLOAD, "classification": "public"},
        {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "classification": "public",
            "extra": 1,
        },
        {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "classification": 409,
        },
        {
            "tenant_id": "  ",
            "workload_id": WORKLOAD,
            "classification": "public",
        },
        {
            "tenant_id": TENANT,
            "workload_id": "  ",
            "classification": "public",
        },
        {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "classification": "   ",
        },
        {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "classification": None,
        },
    ],
)
def test_put_rejects_invalid_bodies_with_422(client, body):
    _envelope(client)
    response = client.put(f"/v1/data-classes/{DATA_ID}", json=body)
    assert response.status_code == 422


def test_put_empty_path_segment_is_422(client):
    response = client.put(
        "/v1/data-classes/",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "classification": "public",
        },
    )
    assert response.status_code == 422


def test_get_empty_path_segment_is_422(client):
    response = client.get(
        "/v1/data-classes/",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_get_rejects_blank_scope_parameters(client):
    _envelope(client)
    _put_class(client, "public")
    assert (
        client.get(
            f"/v1/data-classes/{DATA_ID}",
            params={"tenant_id": " ", "workload_id": WORKLOAD},
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/v1/data-classes/{DATA_ID}",
            params={"tenant_id": TENANT, "workload_id": " "},
        ).status_code
        == 422
    )


def test_repeated_same_classification_is_idempotent_and_keeps_assigned_at(
    client,
):
    _envelope(client)
    first = _put_class(client, "secret")
    assert first.status_code == 200
    second = _put_class(client, "secret")
    assert second.status_code == 200
    assert second.json() == first.json()
    assert second.json()["assigned_at"] == first.json()["assigned_at"]
    assert _get_class(client).json() == first.json()


def test_different_classification_returns_409_and_changes_nothing(client):
    _envelope(client)
    first = _put_class(client, "secret")
    assert first.status_code == 200

    response = _put_class(client, "public")
    assert response.status_code == 409
    assert response.json()["detail"] == "classification immutable"

    # The original binding is intact byte-for-byte.
    assert _get_class(client).json() == first.json()


def test_concurrent_same_code_assignments_share_one_assigned_at(client):
    _envelope(client)
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(lambda _: _put_class(client, "secret"), range(4))
        )
    assert all(r.status_code == 200 for r in responses)
    timestamps = {r.json()["assigned_at"] for r in responses}
    assert len(timestamps) == 1
    assert _get_class(client).json()["classification"] == "secret"


def test_concurrent_different_code_assignments_settle_one_200_and_409s(client):
    _envelope(client)
    codes = ["alpha", "beta", "gamma", "delta"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda c: _put_class(client, c), codes))
    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200, 409, 409, 409]
    winner = next(r for r in responses if r.status_code == 200)
    assert _get_class(client).json() == winner.json()


def test_classification_persists_across_restarts(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/persist.db"
    first = create_app(url)
    with TestClient(first) as client:
        _envelope(client)
        assigned = _put_class(client, "secret").json()
    first.state.engine.dispose()

    second = create_app(url)
    with TestClient(second) as client:
        assert _get_class(client).json() == assigned
        # Immutability survives restarts as well.
        assert _put_class(client, "public").status_code == 409
    second.state.engine.dispose()


# --- classification on grant issuance -------------------------------------


def test_classified_grant_binds_classification_and_releases(client):
    decision_id = _decision(client)
    _envelope(client)
    _put_class(client, "secret")

    response = _grant(client, decision_id, classification="secret")
    assert response.status_code == 201
    data = response.json()
    assert data["classification"] == "secret"
    assert "capability" in data and data["capability"]

    released = _release(client, data["grant_id"], data["capability"])
    assert released.status_code == 200
    assert released.json() == {"payload": PAYLOAD}


def test_omitted_classification_grant_is_null_and_unconstrained(client):
    decision_id = _decision(client)
    _envelope(client)
    # Classify after the envelope exists but before issuance; an omitted
    # classification keeps the legacy behavior regardless.
    _put_class(client, "secret")

    response = _grant(client, decision_id)
    assert response.status_code == 201
    assert response.json()["classification"] is None

    # Release is unconstrained even though the envelope carries a
    # classification the grant was never bound to.
    data = response.json()
    released = _release(client, data["grant_id"], data["capability"])
    assert released.status_code == 200
    assert released.json() == {"payload": PAYLOAD}


def test_grant_with_classification_requires_envelope(client):
    decision_id = _decision(client)
    response = _grant(client, decision_id, classification="secret")
    assert response.status_code == 404
    assert response.json()["detail"] == "data envelope not found"
    assert "capability" not in response.text
    _assert_no_grants_or_audits(client)


def test_grant_with_classification_cross_scope_envelope_is_404(client):
    decision_id = _decision(client)
    _envelope(client)
    _put_class(client, "secret")
    response = _grant(
        client,
        decision_id,
        data_id="other-item",
        classification="secret",
    )
    assert response.status_code == 404
    _assert_no_grants_or_audits(client)


def test_grant_against_unclassified_envelope_is_409_mismatch(client):
    decision_id = _decision(client)
    _envelope(client)
    response = _grant(client, decision_id, classification="secret")
    assert response.status_code == 409
    assert response.json()["detail"] == "classification mismatch"
    assert "capability" not in response.text
    _assert_no_grants_or_audits(client)


def test_grant_against_differently_classified_envelope_is_409(client):
    decision_id = _decision(client)
    _envelope(client)
    _put_class(client, "public")
    response = _grant(client, decision_id, classification="secret")
    assert response.status_code == 409
    assert response.json()["detail"] == "classification mismatch"
    _assert_no_grants_or_audits(client)


@pytest.mark.parametrize(
    "code",
    ["Public", "-a", "a-", "a_b", "a" * 65],
)
def test_grant_rejects_malformed_classification_with_422(client, code):
    decision_id = _decision(client)
    _envelope(client)
    response = _grant(client, decision_id, classification=code)
    assert response.status_code == 422
    _assert_no_grants_or_audits(client)


def test_explicit_null_classification_is_treated_as_omitted(client):
    decision_id = _decision(client)
    _envelope(client)
    response = _grant(client, decision_id, classification=None)
    assert response.status_code == 201
    assert response.json()["classification"] is None


def test_classified_grant_stored_binding_survives_and_grant_listing_unaffected(
    client, app
):
    decision_id = _decision(client)
    _envelope(client)
    _put_class(client, "secret")
    created = _grant(client, decision_id, classification="secret").json()

    # The binding is persisted on the grant row.
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, created["grant_id"])
        assert row.classification == "secret"

    # The capability is still returned exactly once (not on the audit).
    listing = client.get(
        "/v1/release-grants",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert listing.status_code == 200
    entry = listing.json()["grants"][0]
    assert "capability" not in entry
    assert "classification" not in entry
    assert created["capability"] not in listing.text


# --- classification check on release --------------------------------------


def _force_grant_binding(app, grant_id: str, code: str) -> None:
    """Directly rewrite a grant's bound classification.

    Classifications are immutable on both envelopes and grants, so a
    disagreement cannot arise through the API; simulate one at the storage
    layer to exercise the release-path defence directly.
    """
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant_id)
        row.classification = code
        session.commit()


def test_release_with_disagreeing_binding_returns_409_and_grant_stays_pending(
    client, app
):
    decision_id = _decision(client)
    _envelope(client)
    _put_class(client, "secret")
    grant = _grant(client, decision_id, classification="secret").json()

    _force_grant_binding(app, grant["grant_id"], "public")

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 409
    assert response.json()["detail"] == "classification mismatch"
    assert "payload" not in response.text

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.consumed_at is None

    # Only the issuance audit exists; no consumption audit was written.
    events = client.get(
        "/v1/compliance/audit-events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()["events"]
    statuses = sorted(e["status"] for e in events)
    assert statuses == ["pending"]


def test_release_mismatch_consumes_rate_limit_budget(client, app):
    decision_id = _decision(client)
    _envelope(client)
    _put_class(client, "secret")
    grant = _grant(client, decision_id, classification="secret").json()
    _force_grant_binding(app, grant["grant_id"], "public")

    from proof_release import app as app_module

    original_budget = app_module.GRANT_BUDGET_PER_MINUTE
    app_module.GRANT_BUDGET_PER_MINUTE = 1
    try:
        first = _release(client, grant["grant_id"], grant["capability"])
        assert first.status_code == 409
        second = _release(client, grant["grant_id"], grant["capability"])
        assert second.status_code == 429
        assert set(second.json()) == {"retry_after_seconds"}
    finally:
        app_module.GRANT_BUDGET_PER_MINUTE = original_budget


def test_release_mismatch_precedes_key_loading_and_works_without_keyring(
    client, app, monkeypatch
):
    decision_id = _decision(client)
    _envelope(client)
    _put_class(client, "secret")
    grant = _grant(client, decision_id, classification="secret").json()
    _force_grant_binding(app, grant["grant_id"], "public")

    # The mismatch must be judged before the keyring is consulted, so an
    # unusable keyring cannot turn it into a 500.
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 409
    assert response.json()["detail"] == "classification mismatch"


def test_bound_grant_releases_when_classification_still_matches(client, app):
    decision_id = _decision(client)
    _envelope(client)
    _put_class(client, "secret")
    grant = _grant(client, decision_id, classification="secret").json()
    # Re-reading the binding (same code) must of course still release.
    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 200
    assert response.json() == {"payload": PAYLOAD}


# --- schema upgrade --------------------------------------------------------


def test_legacy_release_grants_table_is_upgraded_with_classification_column(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/legacy.db"
    engine = create_engine(url)
    # A deployment-old release_grants table carrying no classification
    # column; create_all leaves an existing table untouched and the
    # additive migration must add the nullable column.
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE release_grants ("
                "grant_id VARCHAR(36) NOT NULL PRIMARY KEY, "
                "tenant_id VARCHAR(256), workload_id VARCHAR(256), "
                "decision_id VARCHAR(36), data_id VARCHAR(256), "
                "capability_digest VARCHAR(64), status VARCHAR(16), "
                "issued_at DATETIME, expires_at DATETIME, "
                "consumed_at DATETIME, revoked_at DATETIME)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO release_grants "
                "(grant_id, tenant_id, workload_id, decision_id, data_id, "
                "capability_digest, status, issued_at, expires_at) VALUES "
                "('g1', 't', 'w', 'd', 'data-legacy', 'digest', 'pending', "
                "'2026-01-01T00:00:00+00:00', '2026-01-01T00:05:00+00:00')"
            )
        )
    engine.dispose()

    application = create_app(url)
    with application.state.session_factory() as session:
        row = session.get(ReleaseGrant, "g1")
        assert row is not None
        assert row.classification is None
    application.state.engine.dispose()


# --- helpers ---------------------------------------------------------------


def _assert_no_grants_or_audits(client):
    listing = client.get(
        "/v1/release-grants",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert listing.status_code == 200
    assert listing.json()["grants"] == []
    events = client.get(
        "/v1/compliance/audit-events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert events.status_code == 200
    grant_events = [
        e for e in events.json()["events"] if e["event_type"] == "grant"
    ]
    assert grant_events == []
