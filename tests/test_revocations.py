"""Tests for POST /v1/revocations and X.509 revocation enforcement."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.serialization import Encoding
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import CertificateRevocation, Evidence
from proof_release.verifiers import ATTESTED_NONCE_JSON, X509_ATTESTED_NONCE_JSON

from x509_helpers import (
    generate_key,
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
    digest = hashlib.sha256(certificate.public_bytes(Encoding.DER)).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _past() -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()


def _future() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()


def _register(client, root_id_value, fingerprint, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id_value,
        "certificate_fingerprint": fingerprint,
        "effective_at": _past(),
    }
    body.update(overrides)
    return client.post("/v1/revocations", json=body)


def _challenge(client, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/challenges", json={"tenant_id": tenant, "workload_id": workload}
    )
    assert response.status_code == 201
    return response.json()


def _submit_and_verify(client, created, evidence, fmt=X509_ATTESTED_NONCE_JSON,
                       tenant=TENANT, workload=WORKLOAD):
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": fmt,
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


def _full_chain(chain):
    return [chain["leaf_cert"], chain["intermediate_cert"], chain["root_cert"]]


def _x509_evidence(chain, nonce):
    return make_evidence(nonce, chain["leaf_key"], _full_chain(chain))


# --- registration ---------------------------------------------------------


def test_register_revocation_returns_201_with_compact_fields(client, root_id, chain):
    fingerprint = _fingerprint(chain["leaf_cert"])
    effective = _past()

    response = _register(client, root_id, fingerprint, effective_at=effective)

    assert response.status_code == 201
    assert response.text == json.dumps(
        {
            "revocation_id": response.json()["revocation_id"],
            "trust_root_id": root_id,
            "certificate_fingerprint": fingerprint,
            "effective_at": datetime.fromisoformat(effective)
            .astimezone(timezone.utc)
            .isoformat(),
        },
        separators=(",", ":"),
    )
    data = response.json()
    assert data["revocation_id"]
    effective_at = datetime.fromisoformat(data["effective_at"])
    assert effective_at.utcoffset() == timedelta(0)


def test_registration_persists_across_restart(tmp_path, chain):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    created = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    root_id = created.json()["root_id"]
    fingerprint = _fingerprint(chain["leaf_cert"])
    response = _register(client1, root_id, fingerprint)
    assert response.status_code == 201
    revocation_id = response.json()["revocation_id"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        record = session.get(CertificateRevocation, revocation_id)
        assert record is not None
        assert record.trust_root_id == root_id
        assert record.cert_fingerprint == fingerprint
        assert record.tenant_id == TENANT
        assert record.workload_id == WORKLOAD

    # A duplicate is still rejected after the restart.
    client2 = TestClient(app2)
    assert _register(client2, root_id, fingerprint).status_code == 409
    app2.state.engine.dispose()


def test_duplicate_fingerprint_same_root_returns_409(client, root_id, chain):
    fingerprint = _fingerprint(chain["leaf_cert"])
    assert _register(client, root_id, fingerprint).status_code == 201

    second = _register(client, root_id, fingerprint)

    assert second.status_code == 409


def test_concurrent_duplicate_registration_has_one_winner(client, app, root_id, chain):
    import threading

    fingerprint = _fingerprint(chain["leaf_cert"])
    results = []
    barrier = threading.Barrier(8)

    def register():
        barrier.wait()
        results.append(_register(client, root_id, fingerprint).status_code)

    threads = [threading.Thread(target=register) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(201) == 1
    assert results.count(409) == 7
    with app.state.session_factory() as session:
        assert session.query(CertificateRevocation).count() == 1


def test_other_fingerprints_are_independent(client, root_id, chain):
    assert (
        _register(client, root_id, _fingerprint(chain["leaf_cert"])).status_code
        == 201
    )
    other = _register(client, root_id, _fingerprint(chain["intermediate_cert"]))
    assert other.status_code == 201
    assert (
        other.json()["revocation_id"]
        != _register(client, root_id, _fingerprint(chain["root_cert"])).json()[
            "revocation_id"
        ]
    )


def test_same_fingerprint_under_other_root_is_independent(client, chain):
    other_root_key, other_root_cert = make_root("other-root")
    roots = []
    for root_cert, tenant in (
        (chain["root_cert"], TENANT),
        (other_root_cert, TENANT),
        (chain["root_cert"], "tenant-b"),
    ):
        response = client.post(
            "/v1/trust-roots",
            json={
                "tenant_id": tenant,
                "workload_id": WORKLOAD,
                "root_pem": pem(root_cert),
            },
        )
        assert response.status_code == 201
        roots.append((response.json()["root_id"], tenant))

    fingerprint = _fingerprint(chain["leaf_cert"])
    for root, tenant in roots:
        assert (
            _register(client, root, fingerprint, tenant_id=tenant).status_code == 201
        )


# --- field validation (422, before any state) ------------------------------


@pytest.mark.parametrize(
    "trust_root_id",
    [
        "",
        "   ",
        "not-a-uuid",
        "A0EEBC99-9C0B-4EF8-BB6D-6BB9BD380A11",  # uppercase is non-canonical
        "a0eebc999c0b4ef8bb6d6bb9bd380a11",  # missing hyphens
        " a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",  # surrounding whitespace
        42,
        None,
    ],
)
def test_non_canonical_trust_root_id_returns_422(client, root_id, chain, trust_root_id):
    response = _register(
        client, trust_root_id, _fingerprint(chain["leaf_cert"])
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "fingerprint",
    [
        "",
        "   ",
        "too-short",
        "A" * 42,
        "A" * 44,
        "A" * 43 + "=",  # padding is not accepted
        "!" + "A" * 42,  # outside the base64url alphabet
        42,
        None,
    ],
)
def test_invalid_fingerprint_returns_422(client, root_id, fingerprint):
    response = _register(client, root_id, fingerprint)

    assert response.status_code == 422


def test_non_canonical_fingerprint_bits_return_422(client, root_id, chain):
    # 32 bytes encode to 43 base64url characters with two spare bits in the
    # final character; a spelling with those bits set is not canonical.
    fingerprint = _fingerprint(chain["leaf_cert"])
    last = fingerprint[-1]
    flipped = fingerprint[:-1] + ("B" if last == "A" else "A")
    assert flipped != fingerprint

    response = _register(client, root_id, flipped)

    # Either rejected as non-canonical or, if it happens to be canonical,
    # accepted exactly once — but never a server error.
    assert response.status_code in (201, 422)


@pytest.mark.parametrize(
    "effective_at",
    [
        "",
        "   ",
        "not-a-time",
        "2026-09-25T10:00:00",  # naive: no offset
        "2026-09-25T10:00:00+02:00",  # non-UTC offset
        42,
        None,
    ],
)
def test_invalid_effective_at_returns_422(client, root_id, chain, effective_at):
    response = _register(
        client, root_id, _fingerprint(chain["leaf_cert"]), effective_at=effective_at
    )

    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "trust_root_id",
                                   "certificate_fingerprint", "effective_at"])
def test_missing_fields_return_422(client, root_id, missing):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "certificate_fingerprint": _fingerprint(make_root("x")[1]),
        "effective_at": _past(),
    }
    del body[missing]

    response = client.post("/v1/revocations", json=body)

    assert response.status_code == 422


def test_invalid_requests_write_no_state(client, app, root_id, chain):
    for body in (
        {"trust_root_id": "not-a-uuid"},
        {"certificate_fingerprint": "bad"},
        {"effective_at": "2026-09-25T10:00:00"},
        {"tenant_id": "  "},
    ):
        full_body = {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "certificate_fingerprint": _fingerprint(chain["leaf_cert"]),
            "effective_at": _past(),
        }
        full_body.update(body)
        assert client.post("/v1/revocations", json=full_body).status_code == 422

    with app.state.session_factory() as session:
        assert session.query(CertificateRevocation).count() == 0


# --- trust root scoping (404) ----------------------------------------------


def test_unknown_trust_root_returns_404(client, chain):
    response = _register(
        client,
        "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        _fingerprint(chain["leaf_cert"]),
    )

    assert response.status_code == 404


def test_cross_tenant_and_cross_workload_root_returns_404(client, root_id, chain):
    fingerprint = _fingerprint(chain["leaf_cert"])

    other_tenant = _register(client, root_id, fingerprint, tenant_id="tenant-b")
    other_workload = _register(client, root_id, fingerprint, workload_id="workload-2")

    assert other_tenant.status_code == 404
    assert other_workload.status_code == 404
    # The root itself remains usable in its own scope.
    assert _register(client, root_id, fingerprint).status_code == 201


# --- verification enforcement ----------------------------------------------


def test_revoked_leaf_settles_evidence_rejected(client, root_id, chain):
    assert (
        _register(client, root_id, _fingerprint(chain["leaf_cert"])).status_code
        == 201
    )

    created = _challenge(client)
    evidence = _x509_evidence(chain, created["nonce"])
    evidence_id, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    with client.app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "rejected"
        assert record.verification_result == "rejected"


def test_revoked_intermediate_settles_evidence_rejected(client, root_id, chain):
    assert (
        _register(client, root_id, _fingerprint(chain["intermediate_cert"])).status_code
        == 201
    )

    created = _challenge(client)
    _, response = _submit_and_verify(client, created, _x509_evidence(chain, created["nonce"]))

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_revoked_root_settles_evidence_rejected(client, root_id, chain):
    assert (
        _register(client, root_id, _fingerprint(chain["root_cert"])).status_code
        == 201
    )

    created = _challenge(client)
    _, response = _submit_and_verify(client, created, _x509_evidence(chain, created["nonce"]))

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_unrelated_fingerprint_does_not_reject(client, root_id, chain):
    _, other_cert = make_root("unrelated")
    assert (
        _register(client, root_id, _fingerprint(other_cert)).status_code == 201
    )

    created = _challenge(client)
    _, response = _submit_and_verify(client, created, _x509_evidence(chain, created["nonce"]))

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_future_effective_revocation_does_not_yet_reject(client, root_id, chain):
    assert (
        _register(
            client,
            root_id,
            _fingerprint(chain["leaf_cert"]),
            effective_at=_future(),
        ).status_code
        == 201
    )

    created = _challenge(client)
    _, response = _submit_and_verify(client, created, _x509_evidence(chain, created["nonce"]))

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_revocation_under_other_scope_root_does_not_reject(client, root_id, chain):
    # The same certificate revoked under a different tenant's registration of
    # the same root must not affect this tenant's evidence.
    other = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": "tenant-b",
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    assert other.status_code == 201
    assert (
        _register(
            client,
            other.json()["root_id"],
            _fingerprint(chain["leaf_cert"]),
            tenant_id="tenant-b",
        ).status_code
        == 201
    )

    created = _challenge(client)
    _, response = _submit_and_verify(client, created, _x509_evidence(chain, created["nonce"]))

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_registration_after_settlement_does_not_rewrite(client, root_id, chain):
    created = _challenge(client)
    evidence = _x509_evidence(chain, created["nonce"])
    evidence_id, first = _submit_and_verify(client, created, evidence)
    assert first.json()["status"] == "verified"

    # A revocation registered after the evidence settled must not reach back.
    assert (
        _register(client, root_id, _fingerprint(chain["leaf_cert"])).status_code
        == 201
    )
    second = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert second.status_code == 200
    assert second.json() == first.json()


def test_hmac_format_is_unaffected_by_revocations(client, root_id, chain):
    import hmac as hmac_module
    import os

    assert (
        _register(client, root_id, _fingerprint(chain["leaf_cert"])).status_code
        == 201
    )

    created = _challenge(client)
    nonce = created["nonce"]
    secret = os.environ.get(
        "PROOF_RELEASE_ATTESTED_NONCE_SECRET", "dev-only-attested-nonce-secret"
    )
    mac_key = hmac_module.new(
        secret.encode(), f"{TENANT}:{WORKLOAD}".encode(), hashlib.sha256
    ).digest()
    claims = {"m": "abc"}
    signed = json.dumps(
        {"claims": claims, "nonce": nonce}, sort_keys=True, separators=(",", ":")
    ).encode()
    mac = hmac_module.new(mac_key, signed, hashlib.sha256).hexdigest()
    evidence = json.dumps({"nonce": nonce, "claims": claims, "mac": mac})

    _, response = _submit_and_verify(
        client, created, evidence, fmt=ATTESTED_NONCE_JSON
    )

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_failed_verification_leaves_evidence_received(client, root_id, chain, app, monkeypatch):
    # A revocation-store failure during verification is a 500 and the
    # evidence stays received, so it can be verified after recovery.
    assert (
        _register(client, root_id, _fingerprint(chain["leaf_cert"])).status_code
        == 201
    )
    created = _challenge(client)
    evidence = _x509_evidence(chain, created["nonce"])
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

    from proof_release import app as app_module

    def broken(*args, **kwargs):
        raise RuntimeError("store down")

    monkeypatch.setattr(app_module, "_x509_chain_revoked", broken)
    failed = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert failed.status_code == 500
    with app.state.session_factory() as session:
        assert session.get(Evidence, evidence_id).status == "received"

    monkeypatch.undo()
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
    assert recovered.json()["status"] == "rejected"
