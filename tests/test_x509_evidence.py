"""Tests for the built-in ``x509-attested-nonce-json`` evidence format."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Evidence
from proof_release.verifiers import X509_ATTESTED_NONCE_JSON

from x509_helpers import (
    build_certificate,
    evidence_document,
    generate_key,
    make_evidence,
    make_intermediate,
    make_leaf,
    make_root,
    pem,
    sign_payload,
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
def registered(client, chain):
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


def _challenge(client, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/challenges", json={"tenant_id": tenant, "workload_id": workload}
    )
    assert response.status_code == 201
    return response.json()


def _submit(client, created, evidence, tenant=TENANT, workload=WORKLOAD):
    return client.post(
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


def _verify(client, evidence_id, created, evidence, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )


def _submit_and_verify(client, created, evidence, **kwargs):
    submitted = _submit(client, created, evidence, **kwargs)
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    return evidence_id, _verify(client, evidence_id, created, evidence, **kwargs)


def _full_chain(chain):
    return [chain["leaf_cert"], chain["intermediate_cert"], chain["root_cert"]]


def test_valid_chain_is_verified(client, registered, chain):
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain), {"m": "abc"}
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_valid_leaf_signed_directly_by_root_is_verified(client, registered, chain):
    leaf_key, leaf_cert = make_leaf(chain["root_cert"], chain["root_key"])
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], leaf_key, [leaf_cert, chain["root_cert"]]
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_single_certificate_chain_anchored_to_root_is_verified(
    client, registered, chain
):
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["root_key"], [chain["root_cert"]]
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_signature_over_different_claims_is_rejected(client, registered, chain):
    created = _challenge(client)
    # Sign one claims set but present another in the document.
    signature = sign_payload(chain["leaf_key"], created["nonce"], {"m": "real"})
    evidence = evidence_document(
        created["nonce"],
        {"m": "forged"},
        [pem(c) for c in _full_chain(chain)],
        signature,
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_signature_from_wrong_key_is_rejected(client, registered, chain):
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], generate_key(), _full_chain(chain)
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_document_nonce_mismatch_is_rejected(client, registered, chain):
    created = _challenge(client)
    other = _challenge(client)
    # The document attests a different challenge's nonce.
    evidence = make_evidence(other["nonce"], chain["leaf_key"], _full_chain(chain))

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_no_configured_trust_root_is_rejected(client, chain):
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_chain_anchored_to_unconfigured_root_is_rejected(client, registered, chain):
    # A self-consistent chain whose root was never configured for this
    # tenant/workload must not verify.
    other_root_key, other_root_cert = make_root("other-root")
    leaf_key, leaf_cert = make_leaf(other_root_cert, other_root_key)
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], leaf_key, [leaf_cert, other_root_cert]
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_trust_root_of_other_tenant_does_not_anchor(client, chain):
    # The root is configured only for a different tenant; evidence for this
    # tenant must not be anchored by it (tenant isolation).
    other_tenant = "tenant-b"
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": other_tenant,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    assert response.status_code == 201

    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"

    # The same evidence verifies for the tenant that owns the root.
    other_created = _challenge(client, tenant=other_tenant)
    other_evidence = make_evidence(
        other_created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, other_response = _submit_and_verify(
        client, other_created, other_evidence, tenant=other_tenant
    )
    assert other_response.status_code == 200
    assert other_response.json()["status"] == "verified"


def test_expired_leaf_is_rejected(client, registered, chain):
    now = datetime.now(timezone.utc)
    leaf_key, leaf_cert = make_leaf(
        chain["intermediate_cert"],
        chain["intermediate_key"],
        not_before=now - timedelta(days=10),
        not_after=now - timedelta(days=1),
    )
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"],
        leaf_key,
        [leaf_cert, chain["intermediate_cert"], chain["root_cert"]],
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_not_yet_valid_intermediate_is_rejected(client, registered, chain):
    now = datetime.now(timezone.utc)
    intermediate_key, intermediate_cert = make_intermediate(
        chain["root_cert"],
        chain["root_key"],
        not_before=now + timedelta(days=1),
        not_after=now + timedelta(days=10),
    )
    leaf_key, leaf_cert = make_leaf(intermediate_cert, intermediate_key)
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"],
        leaf_key,
        [leaf_cert, intermediate_cert, chain["root_cert"]],
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_non_ca_intermediate_is_rejected(client, registered, chain):
    # The "intermediate" is not a CA certificate, so it may not issue.
    intermediate_key, intermediate_cert = make_intermediate(
        chain["root_cert"], chain["root_key"], ca=False
    )
    leaf_key, leaf_cert = make_leaf(intermediate_cert, intermediate_key)
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"],
        leaf_key,
        [leaf_cert, intermediate_cert, chain["root_cert"]],
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_chain_with_broken_signature_is_rejected(client, registered, chain):
    # The intermediate claims the root as issuer but was signed by another key.
    intermediate_key = generate_key()
    forged_intermediate = build_certificate(
        "forged-intermediate",
        intermediate_key.public_key(),
        chain["root_cert"].subject,
        generate_key(),
        ca=True,
    )
    leaf_key, leaf_cert = make_leaf(forged_intermediate, intermediate_key)
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"],
        leaf_key,
        [leaf_cert, forged_intermediate, chain["root_cert"]],
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_chain_with_wrong_issuer_name_is_rejected(client, registered, chain):
    # The leaf is signed by the intermediate key but names another issuer.
    leaf_key = generate_key()
    misnamed_leaf = build_certificate(
        "misnamed-leaf",
        leaf_key.public_key(),
        make_root("unrelated-root")[1].subject,
        chain["intermediate_key"],
        ca=False,
    )
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"],
        leaf_key,
        [misnamed_leaf, chain["intermediate_cert"], chain["root_cert"]],
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


@pytest.mark.parametrize(
    "document",
    [
        "not json",
        "[]",
        json.dumps({"claims": {}, "signature": "a"}),  # missing nonce/chain
        json.dumps({"nonce": "abc", "claims": {}, "signature": "a"}),  # no chain
        json.dumps(
            {"nonce": "abc", "claims": {}, "certificate_chain": [], "signature": "a"}
        ),  # empty chain
        json.dumps(
            {
                "nonce": "abc",
                "claims": {},
                "certificate_chain": ["not a pem"],
                "signature": "a",
            }
        ),  # unparseable certificate
        json.dumps(
            {
                "nonce": "abc",
                "claims": {},
                "certificate_chain": ["-----BEGIN CERTIFICATE-----"],
                "signature": "a",
            }
        ),  # truncated certificate
    ],
)
def test_malformed_documents_are_rejected(client, registered, document):
    created = _challenge(client)

    _, response = _submit_and_verify(client, created, document)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_unparseable_signature_encoding_is_rejected(client, registered, chain):
    created = _challenge(client)
    evidence = evidence_document(
        created["nonce"],
        {},
        [pem(c) for c in _full_chain(chain)],
        "!!!not-base64url!!!",
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_rejection_is_settled_atomically_and_repeated(client, registered, chain):
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], generate_key(), _full_chain(chain)
    )
    evidence_id, first = _submit_and_verify(client, created, evidence)
    assert first.json()["status"] == "rejected"

    second = _verify(client, evidence_id, created, evidence)
    assert second.status_code == 200
    assert second.json() == first.json()
    with client.app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "rejected"
        assert record.verification_result == "rejected"


def test_trust_root_survives_restart_and_still_anchors(tmp_path, chain):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    response = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    assert response.status_code == 201
    app1.state.engine.dispose()

    # After a restart the configured root still anchors valid evidence.
    app2 = create_app(url)
    client2 = TestClient(app2)
    created = _challenge(client2)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, verify_response = _submit_and_verify(client2, created, evidence)
    assert verify_response.status_code == 200
    assert verify_response.json()["status"] == "verified"
    app2.state.engine.dispose()


def test_certificates_and_evidence_are_never_persisted_or_returned(
    client, registered, chain, app
):
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain), {"m": "abc"}
    )
    leaf_pem = pem(chain["leaf_cert"])
    evidence_id, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert evidence not in response.text
    assert leaf_pem not in response.text
    assert "BEGIN CERTIFICATE" not in response.text
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        columns = {c.name: getattr(record, c.name) for c in record.__table__.columns}
    for name, value in columns.items():
        assert evidence not in str(value), f"evidence leaked into column {name}"
        assert "BEGIN CERTIFICATE" not in str(value), (
            f"certificate leaked into column {name}"
        )
