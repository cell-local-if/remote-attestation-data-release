"""Tests for embedded OCSP revocation evidence in x509-attested-nonce-json.

The optional ``ocsp_responses`` array must stand in one-to-one
correspondence with the chain's non-root certificates; every entry must
be a current, correctly addressed, good-status response signed by the
direct issuing CA or a valid delegated responder. Everything else
settles the proof as rejected through the ordinary verify response.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509 import ocsp
from cryptography.x509.oid import ExtendedKeyUsageOID
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Evidence
from proof_release.verifiers import X509_ATTESTED_NONCE_JSON

from x509_helpers import (
    build_certificate,
    make_delegated_responder,
    make_evidence,
    make_intermediate,
    make_leaf,
    make_ocsp_response_der,
    make_root,
    ocsp_entry,
    ocsp_with_produced_at,
    ocsp_with_two_single_responses,
    pem,
    sign_payload,
    _b64url,
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


def _challenge(client):
    response = client.post(
        "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert response.status_code == 201
    return response.json()


def _submit(client, created, evidence):
    return client.post(
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


def _verify(client, evidence_id, created, evidence):
    return client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )


def _submit_and_verify(client, created, evidence):
    submitted = _submit(client, created, evidence)
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    return evidence_id, _verify(client, evidence_id, created, evidence)


def _full_chain(chain):
    return [chain["leaf_cert"], chain["intermediate_cert"], chain["root_cert"]]


def _good_entries(chain, **kwargs):
    """One good direct-CA OCSP entry per non-root chain certificate."""
    return [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
            **kwargs,
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
            **kwargs,
        ),
    ]


def _evidence_with_ocsp(created, chain, entries, claims=None):
    claims = claims if claims is not None else {}
    return json.dumps(
        {
            "nonce": created["nonce"],
            "claims": claims,
            "certificate_chain": [pem(c) for c in _full_chain(chain)],
            "signature": sign_payload(
                chain["leaf_key"], created["nonce"], claims
            ),
            "ocsp_responses": entries,
        }
    )


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def test_good_direct_ca_responses_are_verified(client, registered, chain):
    created = _challenge(client)
    evidence = _evidence_with_ocsp(created, chain, _good_entries(chain))

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_entries_match_regardless_of_order(client, registered, chain):
    created = _challenge(client)
    entries = _good_entries(chain)
    evidence = _evidence_with_ocsp(created, chain, list(reversed(entries)))

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_good_response_for_leaf_issued_directly_by_root(client, registered, chain):
    leaf_key, leaf_cert = make_leaf(chain["root_cert"], chain["root_key"])
    created = _challenge(client)
    entries = [
        ocsp_entry(leaf_cert, chain["root_cert"], chain["root_key"]),
    ]
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": {},
            "certificate_chain": [pem(leaf_cert), pem(chain["root_cert"])],
            "signature": sign_payload(leaf_key, created["nonce"], {}),
            "ocsp_responses": entries,
        }
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_good_delegated_responder_by_name_is_verified(client, registered, chain):
    responder_key, responder_cert = make_delegated_responder(
        chain["intermediate_cert"], chain["intermediate_key"]
    )
    created = _challenge(client)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            responder_key,
            responder_cert=responder_cert,
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_good_delegated_responder_by_key_is_verified(client, registered, chain):
    responder_key, responder_cert = make_delegated_responder(
        chain["root_cert"], chain["root_key"]
    )
    created = _challenge(client)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            responder_key,
            responder_cert=responder_cert,
            by_key=True,
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_sha256_certid_is_verified(client, registered, chain):
    from cryptography.hazmat.primitives import hashes

    created = _challenge(client)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
            hash_algorithm=hashes.SHA256(),
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
            hash_algorithm=hashes.SHA256(),
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


# ---------------------------------------------------------------------------
# Certificate status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [ocsp.OCSPCertStatus.REVOKED, ocsp.OCSPCertStatus.UNKNOWN],
)
def test_revoked_or_unknown_leaf_status_is_rejected(
    client, registered, chain, status
):
    now = datetime.now(timezone.utc)
    created = _challenge(client)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
            status=status,
            revocation_time=now - timedelta(minutes=10)
            if status == ocsp.OCSPCertStatus.REVOKED
            else None,
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_revoked_intermediate_status_is_rejected(client, registered, chain):
    now = datetime.now(timezone.utc)
    created = _challenge(client)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
            status=ocsp.OCSPCertStatus.REVOKED,
            revocation_time=now - timedelta(minutes=10),
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_empty_ocsp_array_is_rejected(client, registered, chain):
    created = _challenge(client)
    evidence = _evidence_with_ocsp(created, chain, [])

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


@pytest.mark.parametrize("value", [None, "scalar", 3, {"not": "a list"}])
def test_wrong_typed_ocsp_field_is_rejected(client, registered, chain, value):
    created = _challenge(client)
    document = {
        "nonce": created["nonce"],
        "claims": {},
        "certificate_chain": [pem(c) for c in _full_chain(chain)],
        "signature": sign_payload(chain["leaf_key"], created["nonce"], {}),
        "ocsp_responses": value,
    }
    evidence = json.dumps(document)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_missing_entry_is_rejected(client, registered, chain):
    created = _challenge(client)
    # Only the leaf is covered; the intermediate entry is absent.
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_duplicate_target_entry_is_rejected(client, registered, chain):
    created = _challenge(client)
    leaf_entry = ocsp_entry(
        chain["leaf_cert"],
        chain["intermediate_cert"],
        chain["intermediate_key"],
    )
    entries = [
        leaf_entry,
        dict(leaf_entry),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    # Entry count exceeds the number of non-root certificates too.
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_duplicate_target_with_correct_array_size_is_rejected(
    client, registered, chain
):
    created = _challenge(client)
    # Two entries (the correct count) but both address the leaf, so the
    # intermediate is uncovered and the leaf is covered twice.
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
        ),
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_entry_pointing_at_other_certificate_is_rejected(
    client, registered, chain
):
    other_key, other_cert = make_leaf(
        chain["intermediate_cert"], chain["intermediate_key"], "other-leaf"
    )
    created = _challenge(client)
    # The entry's CertID identifies an unrelated certificate (same CA),
    # so no chain certificate is covered by it.
    entries = [
        ocsp_entry(
            other_cert,
            chain["intermediate_cert"],
            chain["intermediate_key"],
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.update(response="!!not-base64url!!") or e,
        lambda e: e.update(response="") or e,
        lambda e: e.update(responder_certificate="not a pem") or e,
        lambda e: e.update(responder_certificate="") or e,
    ],
)
def test_malformed_entry_fields_are_rejected(client, registered, chain, mutate):
    created = _challenge(client)
    entries = _good_entries(chain)
    mutate(entries[0])
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_entry_that_is_not_an_object_is_rejected(client, registered, chain):
    created = _challenge(client)
    evidence = _evidence_with_ocsp(created, chain, ["scalar-entry"])

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_ocsp_array_on_root_only_chain_is_rejected(client, registered, chain):
    # A root-only chain has no non-root certificate, so no non-empty
    # ocsp_responses array can be valid for it.
    created = _challenge(client)
    entry = ocsp_entry(
        chain["root_cert"], chain["root_cert"], chain["root_key"]
    )
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": {},
            "certificate_chain": [pem(chain["root_cert"])],
            "signature": sign_payload(
                chain["root_key"], created["nonce"], {}
            ),
            "ocsp_responses": [entry],
        }
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


# ---------------------------------------------------------------------------
# Response structure and DER parsing
# ---------------------------------------------------------------------------


def test_undecodable_der_is_rejected(client, registered, chain):
    created = _challenge(client)
    entries = _good_entries(chain)
    entries[0]["response"] = _b64url(b"not-an-ocsp-response")
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_non_successful_response_status_is_rejected(client, registered, chain):
    created = _challenge(client)
    unsuccessful = ocsp.OCSPResponseBuilder.build_unsuccessful(
        ocsp.OCSPResponseStatus.TRY_LATER
    ).public_bytes(serialization.Encoding.DER)
    entries = [
        {
            "response": _b64url(unsuccessful),
            "responder_certificate": pem(chain["intermediate_cert"]),
        },
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_two_single_responses_is_rejected(client, registered, chain):
    created = _challenge(client)
    leaf_der = make_ocsp_response_der(
        chain["leaf_cert"],
        chain["intermediate_cert"],
        chain["intermediate_key"],
    )
    doubled = ocsp_with_two_single_responses(
        leaf_der, chain["intermediate_key"]
    )
    entries = [
        {
            "response": _b64url(doubled),
            "responder_certificate": pem(chain["intermediate_cert"]),
        },
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------


def test_this_update_in_the_future_is_rejected(client, registered, chain):
    now = datetime.now(timezone.utc)
    created = _challenge(client)
    future = now + timedelta(hours=2)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
            this_update=future,
            next_update=future + timedelta(hours=1),
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_next_update_in_the_past_is_rejected(client, registered, chain):
    now = datetime.now(timezone.utc)
    created = _challenge(client)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
            this_update=now - timedelta(hours=3),
            next_update=now - timedelta(hours=1),
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_next_update_more_than_seven_days_out_is_rejected(
    client, registered, chain
):
    now = datetime.now(timezone.utc)
    created = _challenge(client)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
            this_update=now - timedelta(hours=1),
            next_update=now + timedelta(days=8),
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_next_update_exactly_seven_days_out_is_verified(
    client, registered, chain
):
    now = datetime.now(timezone.utc)
    created = _challenge(client)
    this_update = now - timedelta(hours=1)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
            this_update=this_update,
            next_update=this_update + timedelta(days=7),
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
            this_update=this_update,
            next_update=this_update + timedelta(days=7),
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_missing_next_update_is_rejected(client, registered, chain):
    now = datetime.now(timezone.utc)
    created = _challenge(client)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
            this_update=now - timedelta(hours=1),
            next_update=None,
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_produced_at_after_now_is_rejected(client, registered, chain):
    now = datetime.now(timezone.utc)
    created = _challenge(client)
    leaf_der = make_ocsp_response_der(
        chain["leaf_cert"],
        chain["intermediate_cert"],
        chain["intermediate_key"],
    )
    future_der = ocsp_with_produced_at(
        leaf_der, chain["intermediate_key"], now + timedelta(hours=1)
    )
    entries = [
        {
            "response": _b64url(future_der),
            "responder_certificate": pem(chain["intermediate_cert"]),
        },
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_produced_at_before_this_update_is_rejected(client, registered, chain):
    now = datetime.now(timezone.utc)
    created = _challenge(client)
    leaf_der = make_ocsp_response_der(
        chain["leaf_cert"],
        chain["intermediate_cert"],
        chain["intermediate_key"],
    )
    stale_der = ocsp_with_produced_at(
        leaf_der, chain["intermediate_key"], now - timedelta(hours=2)
    )
    entries = [
        {
            "response": _b64url(stale_der),
            "responder_certificate": pem(chain["intermediate_cert"]),
        },
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


# ---------------------------------------------------------------------------
# Responder identity, delegation and signature
# ---------------------------------------------------------------------------


def test_wrong_serial_in_certid_is_rejected(client, registered, chain):
    created = _challenge(client)
    # Sign a response for the leaf but overwrite the presented chain
    # entry correspondence by addressing the response at the
    # intermediate's serial (via add_response on a cert with same CA).
    foreign_leaf_key, foreign_leaf_cert = make_leaf(
        chain["intermediate_cert"],
        chain["intermediate_key"],
        "foreign-leaf-serial",
    )
    entries = [
        ocsp_entry(
            foreign_leaf_cert,
            chain["intermediate_cert"],
            chain["intermediate_key"],
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_responder_certificate_mismatch_is_rejected(client, registered, chain):
    # The response is signed by the intermediate (by name) but the entry
    # presents an unrelated certificate as the responder.
    other_key, other_cert = make_leaf(
        chain["intermediate_cert"], chain["intermediate_key"], "not-responder"
    )
    created = _challenge(client)
    entries = [
        {
            "response": _b64url(
                make_ocsp_response_der(
                    chain["leaf_cert"],
                    chain["intermediate_cert"],
                    chain["intermediate_key"],
                )
            ),
            "responder_certificate": pem(other_cert),
        },
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_response_signed_by_wrong_key_is_rejected(client, registered, chain):
    created = _challenge(client)
    # Build a correctly-addressed intermediate response, then corrupt its
    # signature: the presented responder certificate (the direct
    # intermediate) no longer verifies the response.
    valid_der = make_ocsp_response_der(
        chain["leaf_cert"],
        chain["intermediate_cert"],
        chain["intermediate_key"],
    )
    corrupted = bytearray(valid_der)
    corrupted[-1] ^= 0xFF
    entries = [
        {
            "response": _b64url(bytes(corrupted)),
            "responder_certificate": pem(chain["intermediate_cert"]),
        },
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_delegated_responder_without_eku_is_rejected(client, registered, chain):
    # A plain non-CA leaf certificate carries no id-kp-OCSPSigning EKU.
    responder_key, responder_cert = make_leaf(
        chain["intermediate_cert"],
        chain["intermediate_key"],
        "responder-without-eku",
    )
    created = _challenge(client)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            responder_key,
            responder_cert=responder_cert,
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_delegated_responder_that_is_a_ca_is_rejected(client, registered, chain):
    responder_key, responder_cert = make_delegated_responder(
        chain["intermediate_cert"], chain["intermediate_key"]
    )
    # Rebuild the responder certificate with cA TRUE but the same key.
    ca_responder = build_certificate(
        "ca-responder",
        responder_key.public_key(),
        chain["intermediate_cert"].subject,
        chain["intermediate_key"],
        ca=True,
        extra_extensions=[
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.OCSP_SIGNING])
        ],
    )
    created = _challenge(client)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            responder_key,
            responder_cert=ca_responder,
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_expired_delegated_responder_is_rejected(client, registered, chain):
    now = datetime.now(timezone.utc)
    responder_key, responder_cert = make_delegated_responder(
        chain["intermediate_cert"],
        chain["intermediate_key"],
        not_before=now - timedelta(days=10),
        not_after=now - timedelta(days=1),
    )
    created = _challenge(client)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            responder_key,
            responder_cert=responder_cert,
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_delegated_responder_signed_by_other_ca_is_rejected(
    client, registered, chain
):
    # A valid delegated responder certificate, but issued by a different
    # CA than the target certificate's direct issuer.
    other_root_key, other_root_cert = make_root("other-root")
    responder_key, responder_cert = make_delegated_responder(
        other_root_cert, other_root_key
    )
    created = _challenge(client)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            responder_key,
            responder_cert=responder_cert,
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_delegated_responder_for_intermediate_must_chain_to_root(
    client, registered, chain
):
    # A responder for the intermediate must be issued by the root; one
    # issued by the intermediate (the target's peer, not its issuing CA)
    # is invalid.
    responder_key, responder_cert = make_delegated_responder(
        chain["intermediate_cert"], chain["intermediate_key"]
    )
    created = _challenge(client)
    entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            responder_key,
            responder_cert=responder_cert,
        ),
    ]
    evidence = _evidence_with_ocsp(created, chain, entries)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


# ---------------------------------------------------------------------------
# Settlement semantics and non-leakage
# ---------------------------------------------------------------------------


def test_ocsp_failure_rejection_is_settled_and_repeated(
    client, registered, chain
):
    now = datetime.now(timezone.utc)
    created = _challenge(client)
    bad_entries = [
        ocsp_entry(
            chain["leaf_cert"],
            chain["intermediate_cert"],
            chain["intermediate_key"],
            this_update=now - timedelta(hours=3),
            next_update=now - timedelta(hours=1),
        ),
        ocsp_entry(
            chain["intermediate_cert"],
            chain["root_cert"],
            chain["root_key"],
        ),
    ]
    bad_evidence = _evidence_with_ocsp(created, chain, bad_entries)
    evidence_id, first = _submit_and_verify(client, created, bad_evidence)
    assert first.json()["status"] == "rejected"

    # A repeat carrying fresh, valid OCSP must return the first
    # conclusion: the new field is never read or re-validated.
    good_evidence = _evidence_with_ocsp(created, chain, _good_entries(chain))
    second = _verify(client, evidence_id, created, good_evidence)
    assert second.status_code == 200
    assert second.json() == first.json()
    with client.app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "rejected"
        assert record.verification_result == "rejected"


def test_ocsp_material_is_never_persisted_or_returned(
    client, registered, chain, app
):
    created = _challenge(client)
    entries = _good_entries(chain)
    evidence = _evidence_with_ocsp(
        created, chain, entries, claims={"m": "secret"}
    )
    responder_pem = pem(chain["intermediate_cert"])
    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert evidence not in response.text
    assert responder_pem not in response.text
    assert "BEGIN CERTIFICATE" not in response.text
    with app.state.session_factory() as session:
        from proof_release.db import Evidence as _Evidence

        record = session.get(_Evidence, response.json()["evidence_id"])
        columns = {
            c.name: getattr(record, c.name) for c in record.__table__.columns
        }
    for name, value in columns.items():
        assert evidence not in str(value), f"evidence leaked into column {name}"
        assert "BEGIN CERTIFICATE" not in str(value), (
            f"certificate leaked into column {name}"
        )


def test_absent_ocsp_array_keeps_legacy_verification(client, registered, chain):
    # No ocsp_responses field at all: the pre-existing rules apply.
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_good_ocsp_then_bad_leaf_signature_is_rejected(
    client, registered, chain
):
    # OCSP passes but the leaf signature over the claims must still hold.
    created = _challenge(client)
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": {"m": "forged"},
            "certificate_chain": [pem(c) for c in _full_chain(chain)],
            "signature": sign_payload(
                chain["leaf_key"], created["nonce"], {"m": "real"}
            ),
            "ocsp_responses": _good_entries(chain),
        }
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
