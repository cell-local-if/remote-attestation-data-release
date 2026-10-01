"""Tests for immutable data classifications and release-binding constraints.

Covers:

* ``PUT``/``GET /v1/data-classes/{data_id}`` — assignment, idempotent
  replay, immutability, envelope-existence scope rule, indistinguishable
  404s, code-format validation, concurrency (one ``assigned_at``) and
  persistence across restarts;
* the optional ``classification`` on ``POST /v1/release-grants`` —
  permanent binding, 404/409 ordering, no grant/capability/audit on
  failure, and the unchanged omitted-classification legacy flow;
* the atomic bound-classification check in ``POST /v1/release/{grant_id}``
  — matching releases succeed, a mismatch is 409 after the existing
  judgements and before key loading, consumes the shared rate-limit
  slot, leaves the grant pending, writes no consumption audit and
  releases no payload.
"""

from __future__ import annotations

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
    DataClassification,
    ReleaseGrant,
    ReleaseGrantEvent,
)
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
MASTER_KEY_BYTES = b"0123456789abcdef0123456789abcdef"
MASTER_KEY = b64url_encode(MASTER_KEY_BYTES)
PAYLOAD = "classified-payload"
CLASS = "secret-level-1"


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
    assert decided.json()["status"] == "allowed"
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
    return response


def _assign(client, data_id, classification, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "classification": classification,
    }
    body.update(fields)
    return client.put(f"/v1/data-classes/{data_id}", json=body)


def _assign_body(client, data_id, body):
    return client.put(f"/v1/data-classes/{data_id}", json=body)


def _grant(client, decision_id, data_id=DATA_ID, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision_id,
        "data_id": data_id,
    }
    body.update(fields)
    return client.post("/v1/release-grants", json=body)


def _release(client, grant_id, capability, data_id=DATA_ID, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": data_id,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release/{grant_id}", json=body)


# ---------------------------------------------------------------------------
# PUT /v1/data-classes/{data_id}
# ---------------------------------------------------------------------------


def test_put_classification_returns_200_with_five_fields(client):
    _envelope(client)

    response = _assign(client, DATA_ID, CLASS)

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
    assert data["classification"] == CLASS
    assigned = datetime.fromisoformat(data["assigned_at"])
    assert assigned.utcoffset() == timedelta(0)
    # No envelope material of any kind appears on the response.
    assert "ciphertext" not in response.text
    assert PAYLOAD not in response.text


def test_put_same_classification_is_idempotent_and_keeps_assigned_at(client):
    _envelope(client)
    first = _assign(client, DATA_ID, CLASS)
    assert first.status_code == 200
    first_at = first.json()["assigned_at"]

    second = _assign(client, DATA_ID, CLASS)
    assert second.status_code == 200
    assert second.json()["assigned_at"] == first_at
    assert second.json()["classification"] == CLASS


def test_put_different_classification_is_409_immutable(client):
    _envelope(client)
    assert _assign(client, DATA_ID, CLASS).status_code == 200

    response = _assign(client, DATA_ID, "other-level")
    assert response.status_code == 409
    assert response.json()["detail"] == "classification immutable"

    # The stored classification and timestamp are untouched.
    follow_up = client.get(
        f"/v1/data-classes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert follow_up.status_code == 200
    assert follow_up.json()["classification"] == CLASS


def test_put_requires_existing_envelope(client):
    # No envelope ever created for this data id.
    response = _assign(client, "missing-data", CLASS)
    assert response.status_code == 404
    assert response.json()["detail"] == "data envelope not found"


def test_put_cross_scope_envelope_is_indistinguishable_404(client):
    _envelope(client)
    assert (
        _assign(client, DATA_ID, CLASS, tenant_id="tenant-b").status_code == 404
    )
    assert (
        _assign(client, DATA_ID, CLASS, workload_id="workload-2").status_code
        == 404
    )


def test_put_empty_path_segment_is_422(client):
    response = client.put(
        "/v1/data-classes/",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "classification": CLASS,
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "code",
    [
        "",
        "   ",
        "-leading",
        "trailing-",
        "-",
        "UPPER",
        "under_score",
        "sp ace",
        "x" * 65,
        "a.b",
    ],
)
def test_put_rejects_malformed_classification_codes(client, code):
    _envelope(client)
    assert _assign(client, DATA_ID, code).status_code == 422


@pytest.mark.parametrize("code", ["a", "0", "a1", "a--b", "x" * 64])
def test_put_accepts_boundary_shaped_codes(client, code):
    _envelope(client)
    assert _assign(client, DATA_ID, code).status_code == 200


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"tenant_id": 1},
        {"classification": 1},
        {"classification": None},
        {"unexpected": "field"},
    ],
)
def test_put_rejects_invalid_bodies(client, overrides):
    _envelope(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "classification": CLASS,
    }
    body.update(overrides)
    assert _assign_body(client, DATA_ID, body).status_code == 422


def test_put_missing_fields_are_422(client):
    _envelope(client)
    for missing in ("tenant_id", "workload_id", "classification"):
        body = {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "classification": CLASS,
        }
        del body[missing]
        assert (
            client.put(f"/v1/data-classes/{DATA_ID}", json=body).status_code
            == 422
        )


def test_concurrent_puts_settle_on_one_assigned_at(client, app):
    _envelope(client)

    def assign():
        return TestClient(app).put(
            f"/v1/data-classes/{DATA_ID}",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "classification": CLASS,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: assign(), range(12)))

    assert all(r.status_code == 200 for r in responses)
    assigned_ats = {r.json()["assigned_at"] for r in responses}
    assert assigned_ats == {responses[0].json()["assigned_at"]}
    with app.state.session_factory() as session:
        assert session.query(DataClassification).count() == 1


def test_concurrent_puts_with_different_codes_one_wins(app, client):
    _envelope(client)

    def assign(code):
        return TestClient(app).put(
            f"/v1/data-classes/{DATA_ID}",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "classification": code,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(assign, [CLASS] * 8 + ["other"] * 8))

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(200) >= 1
    # Every response for a given code is stable: the winning code gets only
    # 200s, the losing code gets only 409s.
    by_code = {CLASS: [], "other": []}
    for code, response in zip([CLASS] * 8 + ["other"] * 8, responses):
        by_code[code].append(response.status_code)
    stored = client.get(
        f"/v1/data-classes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()["classification"]
    for code, codes in by_code.items():
        expected = 200 if code == stored else 409
        assert set(codes) == {expected}


def test_classification_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    _envelope(client1)
    assigned = _assign(client1, DATA_ID, CLASS).json()
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    response = client2.get(
        f"/v1/data-classes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    assert response.json() == assigned
    # Immutability holds across a restart.
    assert _assign(client2, DATA_ID, "new-code").status_code == 409
    app2.state.engine.dispose()


# ---------------------------------------------------------------------------
# GET /v1/data-classes/{data_id}
# ---------------------------------------------------------------------------


def test_get_unclassified_envelope_is_404(client):
    _envelope(client)
    response = client.get(
        f"/v1/data-classes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "classification not found"


def test_get_unknown_and_cross_scope_are_indistinguishable_404(client):
    _envelope(client)
    _assign(client, DATA_ID, CLASS)

    for params in (
        {"tenant_id": TENANT, "workload_id": WORKLOAD},
    ):
        response = client.get(
            "/v1/data-classes/never-existed", params=params
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "classification not found"

    assert (
        client.get(
            f"/v1/data-classes/{DATA_ID}",
            params={"tenant_id": "tenant-b", "workload_id": WORKLOAD},
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/v1/data-classes/{DATA_ID}",
            params={"tenant_id": TENANT, "workload_id": "workload-2"},
        ).status_code
        == 404
    )


def test_get_returns_assigned_classification_without_material(client):
    _envelope(client, payload=PAYLOAD)
    assigned = _assign(client, DATA_ID, CLASS).json()

    response = client.get(
        f"/v1/data-classes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    assert response.json() == assigned
    assert PAYLOAD not in response.text
    assert "ciphertext" not in response.text


def test_get_missing_scope_query_params_is_422(client):
    response = client.get(f"/v1/data-classes/{DATA_ID}")
    assert response.status_code == 422


def test_get_empty_path_segment_is_422(client):
    response = client.get(
        "/v1/data-classes/",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /v1/release-grants — optional classification
# ---------------------------------------------------------------------------


def test_grant_with_matching_classification_is_bound(client):
    _envelope(client)
    _assign(client, DATA_ID, CLASS)
    decision_id = _decision(client)

    response = _grant(client, decision_id, classification=CLASS)

    assert response.status_code == 201
    data = response.json()
    assert data["classification"] == CLASS
    # The capability is still returned exactly once on this response.
    assert len(data["capability"]) == 43


def test_grant_omitting_classification_stays_unconstrained(client):
    _envelope(client)
    _assign(client, DATA_ID, CLASS)
    decision_id = _decision(client)

    response = _grant(client, decision_id)

    assert response.status_code == 201
    assert response.json()["classification"] is None


def test_explicit_null_classification_is_unconstrained(client):
    _envelope(client)
    _assign(client, DATA_ID, CLASS)
    decision_id = _decision(client)

    response = _grant(client, decision_id, classification=None)

    assert response.status_code == 201
    assert response.json()["classification"] is None


def test_grant_without_envelope_and_classification_is_404(client):
    decision_id = _decision(client)
    response = _grant(
        client, decision_id, data_id="no-envelope", classification=CLASS
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "data envelope not found"


def test_grant_cross_scope_classified_envelope_is_404(client):
    _envelope(client)
    _assign(client, DATA_ID, CLASS)
    decision_id = _decision(client)

    assert (
        _grant(
            client,
            decision_id,
            classification=CLASS,
            tenant_id="tenant-b",
        ).status_code
        == 404
    )
    assert (
        _grant(
            client,
            decision_id,
            classification=CLASS,
            workload_id="workload-2",
        ).status_code
        == 404
    )


def test_grant_on_unclassified_envelope_is_409(client):
    _envelope(client)
    decision_id = _decision(client)

    response = _grant(client, decision_id, classification=CLASS)
    assert response.status_code == 409
    assert response.json()["detail"] == "classification mismatch"
    assert "capability" not in response.text


def test_grant_with_wrong_classification_is_409(client):
    _envelope(client)
    _assign(client, DATA_ID, CLASS)
    decision_id = _decision(client)

    response = _grant(client, decision_id, classification="other-level")
    assert response.status_code == 409
    assert response.json()["detail"] == "classification mismatch"
    assert "capability" not in response.text


def test_grant_classification_failures_create_nothing(client, app):
    _envelope(client)
    _assign(client, DATA_ID, CLASS)
    decision_id = _decision(client)

    assert _grant(client, decision_id, classification="other").status_code == 409
    assert (
        _grant(
            client,
            decision_id,
            data_id="missing",
            classification=CLASS,
        ).status_code
        == 404
    )

    with app.state.session_factory() as session:
        assert session.query(ReleaseGrant).count() == 0
        assert (
            session.query(AuditEvent)
            .filter(AuditEvent.event_type == "grant")
            .count()
            == 0
        )
        assert session.query(ReleaseGrantEvent).count() == 0


@pytest.mark.parametrize(
    "code",
    ["-nope", "nope-", "UPPER", "under_score", "x" * 65, 1, True],
)
def test_grant_rejects_malformed_classification_codes(client, code):
    _envelope(client)
    _assign(client, DATA_ID, CLASS)
    decision_id = _decision(client)
    assert _grant(client, decision_id, classification=code).status_code == 422


def test_bound_classification_is_persisted(client, app):
    _envelope(client)
    _assign(client, DATA_ID, CLASS)
    decision_id = _decision(client)
    created = _grant(client, decision_id, classification=CLASS).json()

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, created["grant_id"])
        assert row.classification == CLASS


def test_legacy_omitted_grant_works_before_envelope_exists(client):
    # The pre-classification flow allowed minting a grant before its
    # envelope exists; that ordering must remain unchanged when the caller
    # omits classification.
    decision_id = _decision(client)
    response = _grant(client, decision_id, data_id="future-data")
    assert response.status_code == 201
    assert response.json()["classification"] is None


# ---------------------------------------------------------------------------
# POST /v1/release/{grant_id} — bound classification check
# ---------------------------------------------------------------------------


def test_release_with_matching_classification_succeeds(client):
    _envelope(client)
    _assign(client, DATA_ID, CLASS)
    decision_id = _decision(client)
    grant = _grant(client, decision_id, classification=CLASS).json()

    response = _release(client, grant["grant_id"], grant["capability"])

    assert response.status_code == 200
    assert response.json() == {"payload": PAYLOAD}


def test_release_with_mismatched_classification_is_409_pending_no_audit(
    client, app
):
    _envelope(client)
    _assign(client, DATA_ID, CLASS)
    decision_id = _decision(client)
    grant = _grant(client, decision_id, classification=CLASS).json()

    # The immutable classification can never change through the API, so
    # tamper the grant's permanent binding directly in storage to simulate
    # a mismatched bound grant.
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.classification = "other-level"
        session.commit()

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 409
    assert response.json()["detail"] == "classification mismatch"
    assert PAYLOAD not in response.text

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.consumed_at is None
        # Only the issuance audit/event exist — no consumption record.
        statuses = [
            e.status
            for e in session.query(AuditEvent)
            .filter(AuditEvent.grant_id == grant["grant_id"])
            .all()
        ]
        assert statuses == ["pending"]
        reasons = [
            e.reason
            for e in session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
            .all()
        ]
        assert reasons == ["issued"]


def test_release_mismatch_consumes_rate_limit_budget(client, app):
    _envelope(client)
    _assign(client, DATA_ID, CLASS)
    decision_id = _decision(client)
    grant = _grant(client, decision_id, classification=CLASS).json()
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.classification = "other-level"
        session.commit()

    # Five judgement-bearing mismatch requests spend the whole per-minute
    # budget; the sixth is a 429 that never reaches the classification
    # check. None of them settles the grant or releases the payload.
    for _ in range(5):
        response = _release(client, grant["grant_id"], grant["capability"])
        assert response.status_code == 409
        assert response.json()["detail"] == "classification mismatch"
    sixth = _release(client, grant["grant_id"], grant["capability"])
    assert sixth.status_code == 429
    assert set(sixth.json()) == {"retry_after_seconds"}

    with app.state.session_factory() as session:
        assert session.get(ReleaseGrant, grant["grant_id"]).status == "pending"


def test_release_unconstrained_grant_skips_classification_check(client):
    _envelope(client)
    # Deliberately no classification at all: an omitted-classification
    # grant releases exactly as before.
    decision_id = _decision(client)
    grant = _grant(client, decision_id).json()

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 200
    assert response.json() == {"payload": PAYLOAD}


def test_release_check_runs_after_capability_and_expiry_judgements(
    client, app
):
    _envelope(client)
    _assign(client, DATA_ID, CLASS)
    decision_id = _decision(client)
    grant = _grant(client, decision_id, classification=CLASS).json()
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.classification = "other-level"
        session.commit()

    # Wrong capability is still 401 (check precedes the classification
    # comparison) and leaves the grant pending.
    wrong = (
        "B" if grant["capability"][0] != "B" else "C"
    ) + grant["capability"][1:]
    response = _release(client, grant["grant_id"], wrong)
    assert response.status_code == 401

    # Expiry (410) also precedes the classification mismatch.
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
    expired = _release(client, grant["grant_id"], grant["capability"])
    assert expired.status_code == 410
