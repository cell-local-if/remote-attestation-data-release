"""Tests for POST /v1/revocations and revocation-aware X.509 verification."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.serialization import Encoding
from fastapi.testclient import TestClient
from sqlalchemy import text

from proof_release.app import create_app
from proof_release.db import Evidence, Revocation
from proof_release.verifiers import X509_ATTESTED_NONCE_JSON

from x509_helpers import (
    make_evidence,
    make_intermediate,
    make_leaf,
    make_root,
    pem,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/test.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def chain():
    root_key, root_cert = make_root()
    intermediate_key, intermediate_cert = make_intermediate(root_cert, root_key)
    leaf_key, leaf_cert = make_leaf(intermediate_cert, intermediate_key)
    return {
        "root_key": root_key,
        "root_cert": root_cert,
        "intermediate_key": intermediate_key,
        "intermediate_cert": intermediate_cert,
        "leaf_key": leaf_key,
        "leaf_cert": leaf_cert,
    }


@pytest.fixture()
def root_id(client, chain):
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    assert response.status_code == 201
    return response.json()["root_id"]


def _fingerprint(certificate) -> str:
    der = certificate.public_bytes(Encoding.DER)
    digest = hashlib.sha256(der).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _past() -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()


def _future() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()


def _register(client, root_id_value, fingerprint, tenant=TENANT, workload=WORKLOAD,
              effective_at=None):
    return client.post(
        "/v1/revocations",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "root_id": root_id_value,
            "cert_fingerprint": fingerprint,
            "effective_at": effective_at or _past(),
        },
    )


def _challenge(client, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/challenges", json={"tenant_id": tenant, "workload_id": workload}
    )
    assert response.status_code == 201
    return response.json()


def _full_chain(chain):
    return [chain["leaf_cert"], chain["intermediate_cert"], chain["root_cert"]]


def _submit_and_verify(client, created, evidence, tenant=TENANT, workload=WORKLOAD):
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": X509_ATTESTED_NONCE_JSON,
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
    return evidence_id, verified


def _fresh_evidence(client, chain, tenant=TENANT, workload=WORKLOAD):
    created = _challenge(client, tenant=tenant, workload=workload)
    evidence = make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))
    return created, evidence


# --- registration ---------------------------------------------------------


def test_register_revocation_returns_201_with_compact_fields(client, root_id, chain):
    fingerprint = _fingerprint(chain["leaf_cert"])
    effective = _past()

    response = _register(client, root_id, fingerprint, effective_at=effective)

    assert response.status_code == 201
    assert response.text == json.dumps(
        {
            "revocation_id": response.json()["revocation_id"],
            "root_id": root_id,
            "cert_fingerprint": fingerprint,
            "effective_at": effective,
        },
        separators=(",", ":"),
    )
    data = response.json()
    assert data["revocation_id"]
    effective_at = datetime.fromisoformat(data["effective_at"])
    assert effective_at.utcoffset() == timedelta(0)


def test_registration_is_persisted(client, root_id, chain, app):
    fingerprint = _fingerprint(chain["leaf_cert"])
    response = _register(client, root_id, fingerprint)
    assert response.status_code == 201

    with app.state.session_factory() as session:
        row = session.get(Revocation, response.json()["revocation_id"])
        assert row is not None
        assert row.tenant_id == TENANT
        assert row.workload_id == WORKLOAD
        assert row.root_id == root_id
        assert row.cert_fingerprint == fingerprint


def test_registration_survives_restart(tmp_path, chain):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    created_root = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    ).json()
    fingerprint = _fingerprint(chain["leaf_cert"])
    response = _register(client1, created_root["root_id"], fingerprint)
    assert response.status_code == 201
    app1.state.engine.dispose()

    # After a restart the registration still rejects matching evidence.
    app2 = create_app(url)
    client2 = TestClient(app2)
    created, evidence = _fresh_evidence(client2, chain)
    _, verified = _submit_and_verify(client2, created, evidence)
    assert verified.status_code == 200
    assert verified.json()["status"] == "rejected"
    app2.state.engine.dispose()


def test_duplicate_fingerprint_same_root_is_409(client, root_id, chain):
    fingerprint = _fingerprint(chain["leaf_cert"])
    assert _register(client, root_id, fingerprint).status_code == 201

    response = _register(client, root_id, fingerprint)

    assert response.status_code == 409
    with client.app.state.session_factory() as session:
        rows = session.query(Revocation).filter_by(root_id=root_id).all()
        assert len(rows) == 1


def test_different_fingerprints_are_independent(client, root_id, chain):
    assert (
        _register(client, root_id, _fingerprint(chain["leaf_cert"])).status_code
        == 201
    )
    response = _register(client, root_id, _fingerprint(chain["intermediate_cert"]))
    assert response.status_code == 201


def test_same_fingerprint_under_different_roots_is_independent(client, chain):
    other_root_key, other_root_cert = make_root("other-root")
    ids = []
    for root_cert in (chain["root_cert"], other_root_cert):
        response = client.post(
            "/v1/trust-roots",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "root_pem": pem(root_cert),
            },
        )
        assert response.status_code == 201
        ids.append(response.json()["root_id"])

    fingerprint = _fingerprint(chain["leaf_cert"])
    assert _register(client, ids[0], fingerprint).status_code == 201
    assert _register(client, ids[1], fingerprint).status_code == 201


def test_concurrent_duplicate_registration_only_one_succeeds(app, root_id, chain):
    fingerprint = _fingerprint(chain["leaf_cert"])
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "root_id": root_id,
        "cert_fingerprint": fingerprint,
        "effective_at": _past(),
    }

    def register():
        return TestClient(app).post("/v1/revocations", json=body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: register(), range(16)))

    assert statuses.count(201) == 1
    assert statuses.count(409) == 15
    with app.state.session_factory() as session:
        rows = session.query(Revocation).filter_by(root_id=root_id).all()
        assert len(rows) == 1


# --- validation (422, before any state access) ----------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"root_id": ""},
        {"root_id": "   "},
        {"root_id": "not-a-uuid"},
        {"root_id": "0" * 32},  # missing hyphens
        {"root_id": "{12345678-1234-1234-1234-123456789012}"},
        {"cert_fingerprint": ""},
        {"cert_fingerprint": "!!!"},
        {"cert_fingerprint": "AAAA"},  # decodes to 3 bytes, not 32
        {"cert_fingerprint": "AQID"},  # 2 bytes
        # padded form is rejected
        {"cert_fingerprint": base64.urlsafe_b64encode(b"\x01" * 32).decode()},
        {"effective_at": ""},
        {"effective_at": "not-a-time"},
        {"effective_at": "2026-09-25T10:00:00"},  # naive
        {"effective_at": "2026-09-25T10:00:00+02:00"},  # non-UTC offset
    ],
)
def test_invalid_fields_are_422_and_create_nothing(client, root_id, chain, overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "root_id": root_id,
        "cert_fingerprint": _fingerprint(chain["leaf_cert"]),
        "effective_at": _past(),
    }
    body.update(overrides)

    response = client.post("/v1/revocations", json=body)

    assert response.status_code == 422
    with client.app.state.session_factory() as session:
        assert session.query(Revocation).count() == 0


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"tenant_id": TENANT},
        {"tenant_id": TENANT, "workload_id": WORKLOAD},
        {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_id": "12345678-1234-1234-1234-123456789012",
        },
    ],
)
def test_missing_fields_are_422(client, body):
    response = client.post("/v1/revocations", json=body)
    assert response.status_code == 422


def test_uppercase_root_id_is_normalized(client, root_id, chain):
    response = _register(
        client, root_id.upper(), _fingerprint(chain["leaf_cert"])
    )
    assert response.status_code == 201
    assert response.json()["root_id"] == root_id


# --- trust root scoping (404) ----------------------------------------------


def test_unknown_trust_root_is_404(client, chain):
    response = _register(
        client,
        "12345678-1234-1234-1234-123456789012",
        _fingerprint(chain["leaf_cert"]),
    )
    assert response.status_code == 404


def test_cross_tenant_trust_root_is_404(client, root_id, chain):
    response = _register(
        client, root_id, _fingerprint(chain["leaf_cert"]), tenant="tenant-b"
    )
    assert response.status_code == 404


def test_cross_workload_trust_root_is_404(client, root_id, chain):
    response = _register(
        client, root_id, _fingerprint(chain["leaf_cert"]), workload="workload-2"
    )
    assert response.status_code == 404


# --- verification integration ----------------------------------------------


def test_revoked_leaf_rejects_evidence(client, root_id, chain):
    assert (
        _register(client, root_id, _fingerprint(chain["leaf_cert"])).status_code
        == 201
    )
    created, evidence = _fresh_evidence(client, chain)

    _, verified = _submit_and_verify(client, created, evidence)

    assert verified.status_code == 200
    assert verified.json()["status"] == "rejected"


def test_revoked_intermediate_rejects_evidence(client, root_id, chain):
    assert (
        _register(
            client, root_id, _fingerprint(chain["intermediate_cert"])
        ).status_code
        == 201
    )
    created, evidence = _fresh_evidence(client, chain)

    _, verified = _submit_and_verify(client, created, evidence)

    assert verified.status_code == 200
    assert verified.json()["status"] == "rejected"


def test_revoked_root_rejects_evidence(client, root_id, chain):
    assert (
        _register(client, root_id, _fingerprint(chain["root_cert"])).status_code
        == 201
    )
    created, evidence = _fresh_evidence(client, chain)

    _, verified = _submit_and_verify(client, created, evidence)

    assert verified.status_code == 200
    assert verified.json()["status"] == "rejected"


def test_unrelated_revocation_does_not_affect_evidence(client, root_id, chain):
    # A registered fingerprint that matches no certificate in the chain.
    _, unrelated_cert = make_root("unrelated")
    assert (
        _register(client, root_id, _fingerprint(unrelated_cert)).status_code == 201
    )
    created, evidence = _fresh_evidence(client, chain)

    _, verified = _submit_and_verify(client, created, evidence)

    assert verified.status_code == 200
    assert verified.json()["status"] == "verified"


def test_future_effective_registration_does_not_yet_reject(client, root_id, chain):
    assert (
        _register(
            client, root_id, _fingerprint(chain["leaf_cert"]), effective_at=_future()
        ).status_code
        == 201
    )
    created, evidence = _fresh_evidence(client, chain)

    _, verified = _submit_and_verify(client, created, evidence)

    assert verified.status_code == 200
    assert verified.json()["status"] == "verified"


def test_registration_under_other_tenants_root_does_not_apply(client, chain):
    # The same root certificate configured for two tenants; a revocation
    # under tenant-b's root must not affect tenant-a's evidence.
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": "tenant-b",
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    assert response.status_code == 201
    other_root_id = response.json()["root_id"]
    assert (
        _register(
            client,
            other_root_id,
            _fingerprint(chain["leaf_cert"]),
            tenant="tenant-b",
        ).status_code
        == 201
    )

    # tenant-a configures the same root independently.
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    assert response.status_code == 201
    created, evidence = _fresh_evidence(client, chain)
    _, verified = _submit_and_verify(client, created, evidence)
    assert verified.status_code == 200
    assert verified.json()["status"] == "verified"

    # ...while tenant-b's evidence is rejected.
    created_b, evidence_b = _fresh_evidence(client, chain, tenant="tenant-b")
    _, verified_b = _submit_and_verify(client, created_b, evidence_b, tenant="tenant-b")
    assert verified_b.status_code == 200
    assert verified_b.json()["status"] == "rejected"


def test_settled_evidence_is_not_rewritten_by_later_registration(
    client, root_id, chain
):
    created, evidence = _fresh_evidence(client, chain)
    evidence_id, verified = _submit_and_verify(client, created, evidence)
    assert verified.json()["status"] == "verified"

    # A registration that becomes effective after settlement does not
    # retroactively change the stored conclusion.
    assert (
        _register(client, root_id, _fingerprint(chain["leaf_cert"])).status_code
        == 201
    )
    again = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert again.status_code == 200
    assert again.json() == verified.json()


def test_hmac_format_is_unaffected_by_registrations(client, root_id, chain):
    # The HMAC built-in format knows no certificates; registrations must
    # not change its conclusion.
    secret = os.environ.get(
        "PROOF_RELEASE_ATTESTED_NONCE_SECRET", "dev-only-attested-nonce-secret"
    )
    created = _challenge(client)
    mac_key = hmac.new(
        secret.encode(), f"{TENANT}:{WORKLOAD}".encode(), hashlib.sha256
    ).digest()
    signed = json.dumps(
        {"claims": {}, "nonce": created["nonce"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": {},
            "mac": hmac.new(mac_key, signed, hashlib.sha256).hexdigest(),
        }
    )
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
    assert verified.json()["status"] == "verified"


# --- storage failure --------------------------------------------------------


def test_registration_storage_failure_is_500_and_leaves_nothing(
    client, root_id, chain, app
):
    fingerprint = _fingerprint(chain["leaf_cert"])
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE revocations"))
    try:
        response = _register(client, root_id, fingerprint)
        assert response.status_code == 500
    finally:
        Revocation.__table__.create(app.state.engine)

    # No half registration is visible; the same request now succeeds.
    with app.state.session_factory() as session:
        assert session.query(Revocation).count() == 0
    assert _register(client, root_id, fingerprint).status_code == 201


def test_verification_store_failure_is_500_and_evidence_stays_received(
    client, root_id, chain, app
):
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": X509_ATTESTED_NONCE_JSON,
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE revocations"))
    try:
        response = client.post(
            f"/v1/evidence/{evidence_id}/verify",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "nonce": created["nonce"],
                "evidence": evidence,
            },
        )
        assert response.status_code == 500
        with app.state.session_factory() as session:
            record = session.get(Evidence, evidence_id)
            assert record.status == "received"
            assert record.verification_result is None
    finally:
        Revocation.__table__.create(app.state.engine)

    # Once the store recovers the evidence can be verified again.
    recovered = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "verified"
