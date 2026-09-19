"""Tests for the built-in ``x509-attested-nonce-json`` evidence format."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Evidence
from proof_release.verifiers import X509_ATTESTED_NONCE_JSON

from x509util import (
    generate_key,
    make_evidence,
    make_leaf,
    make_root_ca,
    pem,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"


@pytest.fixture()
def app(tmp_path):
    return create_app(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def root():
    return make_root_ca()


def _register_root(client, root_cert, **overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD, "root_pem": pem(root_cert)}
    body.update(overrides)
    response = client.post("/v1/trust-roots", json=body)
    assert response.status_code == 201
    return response.json()["root_id"]


def _challenge(client, **overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    body.update(overrides)
    return client.post("/v1/challenges", json=body).json()


def _submit(client, created, evidence, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "challenge_id": created["challenge_id"],
        "nonce": created["nonce"],
        "evidence_format": X509_ATTESTED_NONCE_JSON,
        "evidence": evidence,
    }
    body.update(overrides)
    return client.post("/v1/evidence", json=body)


def _verify(client, evidence_id, created, evidence, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
    }
    body.update(overrides)
    return client.post(f"/v1/evidence/{evidence_id}/verify", json=body)


def _run(client, created, evidence, **verify_overrides):
    """Submit then verify one evidence document; return the verify response."""
    submitted = _submit(client, created, evidence)
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    return _verify(client, evidence_id, created, evidence, **verify_overrides)


def _leaf_chain(root_key, root_cert, **leaf_overrides):
    leaf_key, leaf_cert = make_leaf("workload-leaf", root_cert, root_key, **leaf_overrides)
    return leaf_key, [pem(leaf_cert), pem(root_cert)]


def test_valid_chain_verifies(client, root):
    root_key, root_cert = root
    _register_root(client, root_cert)
    leaf_key, chain = _leaf_chain(root_key, root_cert)
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], {"measurement": "abc"}, chain, leaf_key)

    submitted = _submit(client, created, evidence)
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    response = _verify(client, evidence_id, created, evidence)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "verified"
    assert data["evidence_id"] == evidence_id
    assert data["challenge_id"] == created["challenge_id"]
    verified_at = datetime.fromisoformat(data["verified_at"])
    assert verified_at.utcoffset() == timedelta(0)


def test_valid_chain_with_intermediate_verifies(client, root):
    root_key, root_cert = root
    _register_root(client, root_cert)
    intermediate_key, intermediate_cert = make_leaf(
        "intermediate", root_cert, root_key, ca=True
    )
    leaf_key, leaf_cert = make_leaf("leaf", intermediate_cert, intermediate_key)
    chain = [pem(leaf_cert), pem(intermediate_cert), pem(root_cert)]
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], {}, chain, leaf_key)

    response = _run(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_no_configured_trust_root_rejects(client, root):
    root_key, root_cert = root
    leaf_key, chain = _leaf_chain(root_key, root_cert)
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], {}, chain, leaf_key)

    response = _run(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_chain_anchored_to_other_tenants_root_rejects(client, root):
    root_key, root_cert = root
    # The identical root certificate is registered, but only for another tenant.
    _register_root(client, root_cert, tenant_id="tenant-b")
    leaf_key, chain = _leaf_chain(root_key, root_cert)
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], {}, chain, leaf_key)

    response = _run(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_chain_anchored_to_other_workloads_root_rejects(client, root):
    root_key, root_cert = root
    _register_root(client, root_cert, workload_id="workload-2")
    leaf_key, chain = _leaf_chain(root_key, root_cert)
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], {}, chain, leaf_key)

    response = _run(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_chain_rooted_elsewhere_rejects(client, root):
    _, root_cert = root
    _register_root(client, root_cert)
    # Evidence chains to a different, unregistered root CA.
    foreign_key, foreign_cert = make_root_ca("foreign-root")
    leaf_key, chain = _leaf_chain(foreign_key, foreign_cert)
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], {}, chain, leaf_key)

    response = _run(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_expired_certificate_rejects(client, root):
    root_key, root_cert = root
    _register_root(client, root_cert)
    now = datetime.now(timezone.utc)
    leaf_key, chain = _leaf_chain(
        root_key,
        root_cert,
        not_before=now - timedelta(days=10),
        not_after=now - timedelta(days=1),
    )
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], {}, chain, leaf_key)

    response = _run(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_not_yet_valid_certificate_rejects(client, root):
    root_key, root_cert = root
    _register_root(client, root_cert)
    now = datetime.now(timezone.utc)
    leaf_key, chain = _leaf_chain(
        root_key,
        root_cert,
        not_before=now + timedelta(days=1),
        not_after=now + timedelta(days=10),
    )
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], {}, chain, leaf_key)

    response = _run(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_broken_chain_signature_rejects(client, root):
    root_key, root_cert = root
    _register_root(client, root_cert)
    # Leaf claims the root as issuer but is signed by an unrelated key.
    stranger_key = generate_key()
    leaf_key, leaf_cert = make_leaf("leaf", root_cert, stranger_key)
    chain = [pem(leaf_cert), pem(root_cert)]
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], {}, chain, leaf_key)

    response = _run(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_non_ca_intermediate_rejects(client, root):
    root_key, root_cert = root
    _register_root(client, root_cert)
    # The "intermediate" lacks a CA basic-constraints usage.
    middle_key, middle_cert = make_leaf("not-a-ca", root_cert, root_key)
    leaf_key, leaf_cert = make_leaf("leaf", middle_cert, middle_key)
    chain = [pem(leaf_cert), pem(middle_cert), pem(root_cert)]
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], {}, chain, leaf_key)

    response = _run(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_wrong_leaf_signature_rejects(client, root):
    root_key, root_cert = root
    _register_root(client, root_cert)
    _, chain = _leaf_chain(root_key, root_cert)
    # Sign the payload with a key that is not the leaf's.
    wrong_key = generate_key()
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], {}, chain, wrong_key)

    response = _run(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_tampered_claims_reject(client, root):
    root_key, root_cert = root
    _register_root(client, root_cert)
    leaf_key, chain = _leaf_chain(root_key, root_cert)
    created = _challenge(client)
    document = json.loads(make_evidence(created["nonce"], {"m": "1"}, chain, leaf_key))
    document["claims"] = {"m": "forged"}

    response = _run(client, created, json.dumps(document))

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_wrong_nonce_rejects(client, root):
    root_key, root_cert = root
    _register_root(client, root_cert)
    leaf_key, chain = _leaf_chain(root_key, root_cert)
    created = _challenge(client)
    other = _challenge(client)
    # The document attests a nonce that is not the bound challenge's nonce.
    evidence = make_evidence(other["nonce"], {}, chain, leaf_key)

    response = _run(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


@pytest.mark.parametrize(
    "template",
    [
        "not json",
        "[]",
        json.dumps({"nonce": "{nonce}", "claims": {}, "certificate_chain": ["x"]}),
        json.dumps(
            {
                "nonce": "{nonce}",
                "claims": {},
                "certificate_chain": [],
                "signature": "eA==",
            }
        ),
        json.dumps(
            {
                "nonce": "{nonce}",
                "claims": {},
                "certificate_chain": ["not a pem"],
                "signature": "eA==",
            }
        ),
        json.dumps(
            {
                "nonce": "{nonce}",
                "claims": {},
                "certificate_chain": ["-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n"],
                "signature": "!!!not-base64!!!",
            }
        ),
    ],
)
def test_malformed_documents_reject(client, root, template):
    _, root_cert = root
    _register_root(client, root_cert)
    created = _challenge(client)
    # Bind the real challenge nonce so rejection comes from the malformed
    # document itself, not the nonce check.
    evidence = template.replace("{nonce}", created["nonce"])

    response = _run(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_rejection_settles_atomically_and_leaks_nothing(client, app, root):
    root_key, root_cert = root
    _register_root(client, root_cert)
    leaf_key, chain = _leaf_chain(root_key, root_cert)
    created = _challenge(client)
    document = json.loads(make_evidence(created["nonce"], {}, chain, leaf_key))
    document["signature"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    evidence = json.dumps(document)
    submitted = _submit(client, created, evidence)
    evidence_id = submitted.json()["evidence_id"]

    first = _verify(client, evidence_id, created, evidence)
    assert first.status_code == 200
    assert first.json()["status"] == "rejected"
    second = _verify(client, evidence_id, created, evidence)
    assert second.json() == first.json()

    # Neither the certificates nor the evidence appear in the response or
    # any persisted column.
    for needle in (evidence, chain[0], chain[1]):
        assert needle not in first.text
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "rejected"
        assert record.verification_result == "rejected"
        columns = {c.name: getattr(record, c.name) for c in record.__table__.columns}
    for name, value in columns.items():
        for needle in (evidence, chain[0], chain[1]):
            assert needle not in str(value), f"leaked into column {name}"


def test_trust_root_and_verification_survive_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"
    root_key, root_cert = make_root_ca()
    app1 = create_app(url)
    client1 = TestClient(app1)
    _register_root(client1, root_cert)
    app1.state.engine.dispose()

    # After a restart the registered root still anchors verification.
    app2 = create_app(url)
    client2 = TestClient(app2)
    leaf_key, chain = _leaf_chain(root_key, root_cert)
    created = _challenge(client2)
    evidence = make_evidence(created["nonce"], {"m": "restarted"}, chain, leaf_key)
    response = _run(client2, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_tenant_isolation_of_verification(tmp_path):
    # Two tenants register different roots; each chain only verifies for
    # its own tenant/workload scope.
    url = f"sqlite:///{tmp_path}/isolation.db"
    app = create_app(url)
    client = TestClient(app)
    root_a_key, root_a = make_root_ca("root-a")
    root_b_key, root_b = make_root_ca("root-b")
    _register_root(client, root_a, tenant_id="tenant-a")
    _register_root(client, root_b, tenant_id="tenant-b")

    leaf_a_key, chain_a = _leaf_chain(root_a_key, root_a)
    created = client.post(
        "/v1/challenges", json={"tenant_id": "tenant-b", "workload_id": WORKLOAD}
    ).json()
    # tenant-b presents a chain anchored to tenant-a's root.
    evidence = make_evidence(created["nonce"], {}, chain_a, leaf_a_key)
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": "tenant-b",
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": X509_ATTESTED_NONCE_JSON,
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201
    response = client.post(
        f"/v1/evidence/{submitted.json()['evidence_id']}/verify",
        json={
            "tenant_id": "tenant-b",
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
