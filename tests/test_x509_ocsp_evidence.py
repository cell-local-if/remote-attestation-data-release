"""Tests for embedded OCSP revocation validation in the
``x509-attested-nonce-json`` evidence verifier.

The optional top-level ``ocsp_responses`` array carries one unpadded
base64url DER OCSPResponse plus the PEM responder certificate per
non-root chain certificate. Absent, behavior is unchanged; present,
every entry must be good, fresh, correctly covered and signed by the
direct issuing CA or a valid delegated responder.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Evidence
from proof_release.verifiers import (
    ATTESTED_NONCE_JSON,
    X509_ATTESTED_NONCE_JSON,
)

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509 import ocsp
from cryptography.x509.oid import NameOID

from x509_helpers import (
    b64url,
    build_certificate,
    evidence_document,
    generate_key,
    make_delegated_responder,
    make_intermediate,
    make_leaf,
    make_ocsp_entry,
    make_ocsp_response,
    make_root,
    ocsp_entries_for_chain,
    ocsp_response_resent,
    ocsp_response_with_extra_single_response,
    ocsp_response_with_produced_at,
    pem,
    sign_payload,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
LEAF_URI = "spiffe://example.org/workload/payments"
HMAC_SECRET = "unit-test-secret"


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
    intermediate_key, intermediate_cert = make_intermediate(
        root_cert, root_key
    )
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


def _challenge(client, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/challenges", json={"tenant_id": tenant, "workload_id": workload}
    )
    assert response.status_code == 201
    return response.json()


def _full_chain(chain):
    return [chain["leaf_cert"], chain["intermediate_cert"], chain["root_cert"]]


def _chain_keys(chain):
    return [chain["leaf_key"], chain["intermediate_key"], chain["root_key"]]


def _good_entries(chain, certificates=None):
    """Good direct-CA-signed entries for the standard three-cert chain."""
    certificates = certificates or _full_chain(chain)
    keys = _chain_keys(chain) if len(certificates) == 3 else None
    return ocsp_entries_for_chain(certificates, keys)


def _evidence(nonce, leaf_key, certificates, entries, *, claims=None):
    claims = claims if claims is not None else {}
    chain_pems = [pem(certificate) for certificate in certificates]
    return evidence_document(
        nonce,
        claims,
        chain_pems,
        sign_payload(leaf_key, nonce, claims),
        ocsp_responses=entries,
    )


def _submit(client, created, evidence, tenant=TENANT, workload=WORKLOAD,
            evidence_format=X509_ATTESTED_NONCE_JSON):
    return client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": evidence_format,
            "evidence": evidence,
        },
    )


def _verify(client, evidence_id, created, evidence, tenant=TENANT,
            workload=WORKLOAD):
    return client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )


def _settle(client, created, evidence, *, tenant=TENANT, workload=WORKLOAD,
            evidence_format=X509_ATTESTED_NONCE_JSON):
    submitted = _submit(
        client, created, evidence, tenant, workload, evidence_format
    )
    assert submitted.status_code == 201, submitted.text
    evidence_id = submitted.json()["evidence_id"]
    response = _verify(client, evidence_id, created, evidence, tenant, workload)
    return evidence_id, response


# ---------------------------------------------------------------------------
# Accepted paths
# ---------------------------------------------------------------------------


def test_good_direct_ca_responses_are_verified(client, root_id, chain):
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"],
        chain["leaf_key"],
        _full_chain(chain),
        _good_entries(chain),
        claims={"m": "abc"},
    )

    _, response = _settle(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_good_responses_in_reversed_order_are_verified(client, root_id, chain):
    # Entries map to certificates by CertID, not by array position.
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"],
        chain["leaf_key"],
        _full_chain(chain),
        list(reversed(_good_entries(chain))),
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "verified"


def test_sha256_certid_hash_algorithm_is_accepted(client, root_id, chain):
    created = _challenge(client)
    certificates = _full_chain(chain)
    entries = [
        make_ocsp_entry(
            certificates[0], certificates[1], certificates[1],
            chain["intermediate_key"], algorithm=hashes.SHA256(),
        ),
        make_ocsp_entry(
            certificates[1], certificates[2], certificates[2],
            chain["root_key"], algorithm=hashes.SHA256(),
        ),
    ]
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "verified"


def test_by_name_responder_id_is_accepted(client, root_id, chain):
    created = _challenge(client)
    certificates = _full_chain(chain)
    entries = [
        make_ocsp_entry(
            certificates[0], certificates[1], certificates[1],
            chain["intermediate_key"],
            responder_encoding=ocsp.OCSPResponderEncoding.NAME,
        ),
        make_ocsp_entry(
            certificates[1], certificates[2], certificates[2],
            chain["root_key"],
            responder_encoding=ocsp.OCSPResponderEncoding.NAME,
        ),
    ]
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "verified"


def test_delegated_responders_are_verified(client, root_id, chain):
    # Each non-root certificate is vouched for by a distinct delegated
    # responder issued by its direct CA.
    leaf_delegate_key, leaf_delegate = make_delegated_responder(
        chain["intermediate_cert"], chain["intermediate_key"], "leaf-ocsp"
    )
    int_delegate_key, int_delegate = make_delegated_responder(
        chain["root_cert"], chain["root_key"], "int-ocsp"
    )
    certificates = _full_chain(chain)
    entries = [
        make_ocsp_entry(
            certificates[0], certificates[1], leaf_delegate,
            leaf_delegate_key,
        ),
        make_ocsp_entry(
            certificates[1], certificates[2], int_delegate, int_delegate_key
        ),
    ]
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "verified"


def test_embedded_responder_certificate_in_response_is_harmless(
    client, root_id, chain
):
    # The responder cert may also ride inside the OCSPResponse DER; the
    # out-of-band responder_certificate field still governs validation.
    delegate_key, delegate = make_delegated_responder(
        chain["intermediate_cert"], chain["intermediate_key"]
    )
    certificates = _full_chain(chain)
    leaf_der = make_ocsp_response(
        certificates[0], certificates[1], delegate, delegate_key,
        embedded_certificates=[delegate],
    )
    entries = [
        {"response": b64url(leaf_der), "responder_certificate": pem(delegate)},
        make_ocsp_entry(
            certificates[1], certificates[2], certificates[2],
            chain["root_key"],
        ),
    ]
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "verified"


def test_leaf_directly_signed_by_root_with_one_response(
    client, root_id, chain
):
    leaf_key, leaf_cert = make_leaf(chain["root_cert"], chain["root_key"])
    certificates = [leaf_cert, chain["root_cert"]]
    entries = ocsp_entries_for_chain(
        certificates, [leaf_key, chain["root_key"]]
    )
    created = _challenge(client)
    evidence = _evidence(created["nonce"], leaf_key, certificates, entries)

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "verified"


def test_rsa_chain_with_ocsp_is_verified(client):
    # OCSP signature verification must follow the responder key type;
    # exercise the RSA PKCS#1 v1.5 path end to end.
    root_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    intermediate_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    )
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    root_subject = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "rsa-root")]
    )
    root_cert = build_certificate(
        "rsa-root", root_key.public_key(), root_subject, root_key, ca=True
    )
    intermediate_cert = build_certificate(
        "rsa-intermediate", intermediate_key.public_key(),
        root_cert.subject, root_key, ca=True,
    )
    leaf_cert = build_certificate(
        "rsa-leaf", leaf_key.public_key(),
        intermediate_cert.subject, intermediate_key, ca=False,
    )
    registered = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(root_cert),
        },
    )
    assert registered.status_code == 201

    created = _challenge(client)
    certificates = [leaf_cert, intermediate_cert, root_cert]
    entries = ocsp_entries_for_chain(
        certificates, [leaf_key, intermediate_key, root_key]
    )
    payload = json.dumps(
        {"claims": {}, "nonce": created["nonce"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = leaf_key.sign(
        payload, asym_padding.PKCS1v15(), hashes.SHA256()
    )
    evidence = evidence_document(
        created["nonce"],
        {},
        [pem(c) for c in certificates],
        b64url(signature),
        ocsp_responses=entries,
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "verified"


def test_absent_ocsp_field_remains_verified(client, root_id, chain):
    # No ocsp_responses key at all: the original rules apply unchanged.
    created = _challenge(client)
    chain_pems = [pem(c) for c in _full_chain(chain)]
    evidence = evidence_document(
        created["nonce"],
        {},
        chain_pems,
        sign_payload(chain["leaf_key"], created["nonce"], {}),
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "verified"


def test_good_responses_with_matching_identity_profile_are_verified(
    client, root_id, chain
):
    leaf_key, leaf_cert = make_leaf(
        chain["intermediate_cert"],
        chain["intermediate_key"],
        uri=LEAF_URI,
    )
    certificates = [leaf_cert, chain["intermediate_cert"], chain["root_cert"]]
    keys = [leaf_key, chain["intermediate_key"], chain["root_key"]]
    registered = client.post(
        "/v1/workload-identities",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "claims": [
                {
                    "issuer": leaf_cert.issuer.rfc4514_string(),
                    "subject": leaf_cert.subject.rfc4514_string(),
                    "uri": LEAF_URI,
                }
            ],
        },
    )
    assert registered.status_code == 201
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], leaf_key, certificates,
        ocsp_entries_for_chain(certificates, keys),
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "verified"


# ---------------------------------------------------------------------------
# CertStatus rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cert_status",
    [ocsp.OCSPCertStatus.REVOKED, ocsp.OCSPCertStatus.UNKNOWN],
)
def test_revoked_and_unknown_statuses_are_rejected(
    client, root_id, chain, cert_status
):
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    entries[0] = make_ocsp_entry(
        certificates[0], certificates[1], certificates[1],
        chain["intermediate_key"], status=cert_status,
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_unsuccessful_response_status_is_rejected(client, root_id, chain):
    certificates = _full_chain(chain)
    unsuccessful = ocsp.OCSPResponseBuilder.build_unsuccessful(
        ocsp.OCSPResponseStatus.TRY_LATER
    ).public_bytes(Encoding.DER)
    entries = _good_entries(chain)
    entries[0] = {
        "response": b64url(unsuccessful),
        "responder_certificate": pem(chain["intermediate_cert"]),
    }
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_two_single_responses_are_rejected(client, root_id, chain):
    certificates = _full_chain(chain)
    leaf_der = make_ocsp_response(
        certificates[0], certificates[1], certificates[1],
        chain["intermediate_key"],
    )
    doubled = ocsp_response_with_extra_single_response(
        leaf_der, chain["intermediate_key"]
    )
    entries = _good_entries(chain)
    entries[0] = {
        "response": b64url(doubled),
        "responder_certificate": pem(chain["intermediate_cert"]),
    }
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


# ---------------------------------------------------------------------------
# Structural / coverage rejections
# ---------------------------------------------------------------------------


def test_empty_array_is_rejected(client, root_id, chain):
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain), []
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_root_only_chain_with_empty_array_is_rejected(client, root_id, chain):
    # The root is the only (and therefore anchor) certificate: there are
    # no non-root certificates to cover, but an explicitly present empty
    # array is still malformed.
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["root_key"], [chain["root_cert"]], []
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_root_only_chain_with_response_about_root_is_rejected(
    client, root_id, chain
):
    created = _challenge(client)
    entries = [
        make_ocsp_entry(
            chain["root_cert"], chain["root_cert"], chain["root_cert"],
            chain["root_key"],
        )
    ]
    evidence = _evidence(
        created["nonce"], chain["root_key"], [chain["root_cert"]], entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_missing_entry_is_rejected(client, root_id, chain):
    certificates = _full_chain(chain)
    entries = [
        make_ocsp_entry(
            certificates[0], certificates[1], certificates[1],
            chain["intermediate_key"],
        )
    ]
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_extra_entry_is_rejected(client, root_id, chain):
    created = _challenge(client)
    entries = _good_entries(chain)
    entries.append(
        make_ocsp_entry(
            chain["leaf_cert"], chain["intermediate_cert"],
            chain["intermediate_cert"], chain["intermediate_key"],
        )
    )
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain), entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_duplicate_target_is_rejected(client, root_id, chain):
    # Two entries for the leaf and none for the intermediate.
    certificates = _full_chain(chain)
    entries = [
        make_ocsp_entry(
            certificates[0], certificates[1], certificates[1],
            chain["intermediate_key"],
        ),
        make_ocsp_entry(
            certificates[0], certificates[1], certificates[1],
            chain["intermediate_key"],
        ),
    ]
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_entry_about_foreign_certificate_is_rejected(client, root_id, chain):
    # A valid response for a certificate that is not part of the chain
    # cannot stand in for a missing entry.
    _, foreign_leaf = make_leaf(
        chain["intermediate_cert"], chain["intermediate_key"]
    )
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    entries[0] = make_ocsp_entry(
        foreign_leaf, certificates[1], certificates[1],
        chain["intermediate_key"],
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_entry_with_wrong_issuer_hashes_is_rejected(client, root_id, chain):
    # The response was built as if the root issued the leaf: the CertID
    # issuer hashes do not match the leaf's actual (intermediate) issuer.
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    entries[0] = make_ocsp_entry(
        certificates[0], certificates[2], certificates[2], chain["root_key"]
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(ocsp_responses="nope"),
        lambda d: d.update(ocsp_responses=1),
        lambda d: d.update(ocsp_responses={}),
        lambda d: d.update(ocsp_responses=[1]),
        lambda d: d.update(ocsp_responses=[["x"]]),
        lambda d: d.update(ocsp_responses=[None, None]),
    ],
)
def test_wrong_typed_array_is_rejected(client, root_id, chain, mutate):
    created = _challenge(client)
    document = {
        "nonce": created["nonce"],
        "claims": {},
        "certificate_chain": [pem(c) for c in _full_chain(chain)],
        "signature": sign_payload(chain["leaf_key"], created["nonce"], {}),
    }
    mutate(document)

    _, response = _settle(client, created, json.dumps(document))

    assert response.json()["status"] == "rejected"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda entries: entries[0].update(response=1),
        lambda entries: entries[0].update(response=""),
        lambda entries: entries[0].pop("response"),
        lambda entries: entries[0].update(responder_certificate=1),
        lambda entries: entries[0].update(responder_certificate=""),
        lambda entries: entries[0].pop("responder_certificate"),
        lambda entries: entries[0].update(response="not base64url!"),
        lambda entries: entries[0].update(
            response=base64.urlsafe_b64encode(b"x").decode()
        ),
    ],
)
def test_malformed_entries_are_rejected(client, root_id, chain, mutate):
    created = _challenge(client)
    entries = _good_entries(chain)
    mutate(entries)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain), entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_padded_base64url_response_is_rejected(client, root_id, chain):
    created = _challenge(client)
    entries = _good_entries(chain)
    raw = base64.urlsafe_b64decode(entries[0]["response"] + "==")
    entries[0]["response"] = base64.urlsafe_b64encode(raw).decode("ascii")
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain), entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_garbage_response_der_is_rejected(client, root_id, chain):
    created = _challenge(client)
    entries = _good_entries(chain)
    entries[0]["response"] = b64url(b"\x30\x06garbage")
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain), entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_unparseable_responder_certificate_is_rejected(
    client, root_id, chain
):
    created = _challenge(client)
    entries = _good_entries(chain)
    entries[0]["responder_certificate"] = "-----BEGIN CERTIFICATE-----\nAAAA\n"
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain), entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


# ---------------------------------------------------------------------------
# Freshness window rejections
# ---------------------------------------------------------------------------


def test_future_this_update_is_rejected(client, root_id, chain):
    now = datetime.now(timezone.utc)
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    entries[0] = make_ocsp_entry(
        certificates[0], certificates[1], certificates[1],
        chain["intermediate_key"],
        this_update=now + timedelta(minutes=30),
        next_update=now + timedelta(days=2),
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_expired_next_update_is_rejected(client, root_id, chain):
    now = datetime.now(timezone.utc)
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    entries[0] = make_ocsp_entry(
        certificates[0], certificates[1], certificates[1],
        chain["intermediate_key"],
        this_update=now - timedelta(days=3),
        next_update=now - timedelta(minutes=1),
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_window_longer_than_seven_days_is_rejected(client, root_id, chain):
    now = datetime.now(timezone.utc)
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    this_update = now - timedelta(hours=1)
    entries[0] = make_ocsp_entry(
        certificates[0], certificates[1], certificates[1],
        chain["intermediate_key"],
        this_update=this_update,
        next_update=this_update + timedelta(days=7, seconds=1),
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_window_exactly_seven_days_is_verified(client, root_id, chain):
    now = datetime.now(timezone.utc)
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    this_update = now - timedelta(hours=1)
    entries[0] = make_ocsp_entry(
        certificates[0], certificates[1], certificates[1],
        chain["intermediate_key"],
        this_update=this_update,
        next_update=this_update + timedelta(days=7),
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "verified"


def test_future_produced_at_is_rejected(client, root_id, chain):
    now = datetime.now(timezone.utc)
    certificates = _full_chain(chain)
    leaf_der = make_ocsp_response(
        certificates[0], certificates[1], certificates[1],
        chain["intermediate_key"],
    )
    modified = ocsp_response_with_produced_at(
        leaf_der, now + timedelta(hours=2), chain["intermediate_key"]
    )
    entries = _good_entries(chain)
    entries[0] = {
        "response": b64url(modified),
        "responder_certificate": pem(chain["intermediate_cert"]),
    }
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_produced_at_before_this_update_is_rejected(client, root_id, chain):
    now = datetime.now(timezone.utc)
    certificates = _full_chain(chain)
    this_update = now - timedelta(minutes=30)
    leaf_der = make_ocsp_response(
        certificates[0], certificates[1], certificates[1],
        chain["intermediate_key"],
        this_update=this_update,
        next_update=now + timedelta(days=1),
    )
    modified = ocsp_response_with_produced_at(
        leaf_der, now - timedelta(hours=2), chain["intermediate_key"]
    )
    entries = _good_entries(chain)
    entries[0] = {
        "response": b64url(modified),
        "responder_certificate": pem(chain["intermediate_cert"]),
    }
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_produced_at_equal_this_update_is_verified(client, root_id, chain):
    now = datetime.now(timezone.utc)
    certificates = _full_chain(chain)
    this_update = now - timedelta(minutes=30)
    leaf_der = make_ocsp_response(
        certificates[0], certificates[1], certificates[1],
        chain["intermediate_key"],
        this_update=this_update,
        next_update=now + timedelta(days=1),
    )
    modified = ocsp_response_with_produced_at(
        leaf_der, this_update, chain["intermediate_key"]
    )
    entries = _good_entries(chain)
    entries[0] = {
        "response": b64url(modified),
        "responder_certificate": pem(chain["intermediate_cert"]),
    }
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "verified"


# ---------------------------------------------------------------------------
# Responder identity, path and signature rejections
# ---------------------------------------------------------------------------


def test_tampered_response_signature_is_rejected(client, root_id, chain):
    certificates = _full_chain(chain)
    leaf_der = bytearray(
        make_ocsp_response(
            certificates[0], certificates[1], certificates[1],
            chain["intermediate_key"],
        )
    )
    leaf_der[-7] ^= 0xFF
    entries = _good_entries(chain)
    entries[0] = {
        "response": b64url(bytes(leaf_der)),
        "responder_certificate": pem(chain["intermediate_cert"]),
    }
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_responder_certificate_not_matching_signer_is_rejected(
    client, root_id, chain
):
    # The response is signed by the intermediate key but the presented
    # responder certificate is the root; neither the direct-CA (DER
    # identity) nor delegated path can hold.
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    entries[0] = {
        "response": make_ocsp_entry(
            certificates[0], certificates[1], certificates[1],
            chain["intermediate_key"],
        )["response"],
        "responder_certificate": pem(chain["root_cert"]),
    }
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_responder_id_naming_another_certificate_is_rejected(
    client, root_id, chain
):
    # ResponderID hashes a delegated cert but the presented responder is
    # the direct CA itself.
    delegate_key, delegate = make_delegated_responder(
        chain["intermediate_cert"], chain["intermediate_key"]
    )
    certificates = _full_chain(chain)
    well_formed = make_ocsp_response(
        certificates[0], certificates[1], delegate, delegate_key
    )
    leaf_der = ocsp_response_resent(
        well_formed, chain["intermediate_key"]
    )
    entries = _good_entries(chain)
    entries[0] = {
        "response": b64url(leaf_der),
        "responder_certificate": pem(chain["intermediate_cert"]),
    }
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_delegate_without_ocsp_signing_eku_is_rejected(
    client, root_id, chain
):
    delegate_key, delegate = make_delegated_responder(
        chain["intermediate_cert"],
        chain["intermediate_key"],
        ocsp_signing=False,
    )
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    entries[0] = make_ocsp_entry(
        certificates[0], certificates[1], delegate, delegate_key
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_delegate_that_is_a_ca_is_rejected(client, root_id, chain):
    delegate_key, delegate = make_delegated_responder(
        chain["intermediate_cert"],
        chain["intermediate_key"],
        ca=True,
    )
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    entries[0] = make_ocsp_entry(
        certificates[0], certificates[1], delegate, delegate_key
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_delegate_issued_by_non_direct_ca_is_rejected(
    client, root_id, chain
):
    # For the leaf, only the intermediate (its direct CA) or a responder
    # delegated by that intermediate may answer; a root-issued delegate
    # is outside the permitted path.
    delegate_key, delegate = make_delegated_responder(
        chain["root_cert"], chain["root_key"], "root-delegate"
    )
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    entries[0] = make_ocsp_entry(
        certificates[0], certificates[1], delegate, delegate_key
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_expired_delegate_is_rejected(client, root_id, chain):
    now = datetime.now(timezone.utc)
    delegate_key, delegate = make_delegated_responder(
        chain["intermediate_cert"],
        chain["intermediate_key"],
        not_before=now - timedelta(days=10),
        not_after=now - timedelta(days=1),
    )
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    entries[0] = make_ocsp_entry(
        certificates[0], certificates[1], delegate, delegate_key
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_delegate_with_forged_certificate_signature_is_rejected(
    client, root_id, chain
):
    # The delegate claims the intermediate as issuer but was signed by
    # another key; its certificate path check must fail.
    delegate_key, delegate = make_delegated_responder(
        chain["intermediate_cert"], generate_key(), "forged-delegate"
    )
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    entries[0] = make_ocsp_entry(
        certificates[0], certificates[1], delegate, delegate_key
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_target_cannot_act_as_its_own_delegate(client, root_id, chain):
    # A certificate may not vouch for its own revocation status: using
    # the target certificate as the presented responder is rejected even
    # though the response signature would otherwise verify.
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    leaf_der = make_ocsp_response(
        certificates[0], certificates[1], certificates[0],
        chain["leaf_key"],
    )
    entries[0] = {
        "response": b64url(leaf_der),
        "responder_certificate": pem(certificates[0]),
    }
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_delegate_ocsp_signature_under_other_key_is_rejected(
    client, root_id, chain
):
    # ResponderID and the presented delegate certificate agree, but the
    # OCSP signature was produced by a different key.
    delegate_key, delegate = make_delegated_responder(
        chain["intermediate_cert"], chain["intermediate_key"]
    )
    certificates = _full_chain(chain)
    well_formed = make_ocsp_response(
        certificates[0], certificates[1], delegate, delegate_key
    )
    leaf_der = ocsp_response_resent(well_formed, generate_key())
    entries = _good_entries(chain)
    entries[0] = {
        "response": b64url(leaf_der),
        "responder_certificate": pem(delegate),
    }
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


# ---------------------------------------------------------------------------
# Ordering with the remaining gates
# ---------------------------------------------------------------------------


def test_good_ocsp_with_bad_leaf_signature_is_rejected(
    client, root_id, chain
):
    created = _challenge(client)
    chain_pems = [pem(c) for c in _full_chain(chain)]
    evidence = evidence_document(
        created["nonce"],
        {},
        chain_pems,
        sign_payload(generate_key(), created["nonce"], {}),
        ocsp_responses=_good_entries(chain),
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_good_ocsp_does_not_override_registered_revocation(
    client, root_id, chain
):
    # The pre-verifier fingerprint revocation merge wins: a revoked
    # certificate settles rejected even with good inline OCSP responses.
    from proof_release.envelopes import b64url_encode

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fingerprint = b64url_encode(
        hashlib.sha256(
            chain["leaf_cert"].public_bytes(Encoding.DER)
        ).digest()
    )
    registration = client.post(
        "/v1/revocations",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "certificate_fingerprint": fingerprint,
            "effective_at": now,
        },
    )
    assert registration.status_code == 201
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"],
        chain["leaf_key"],
        _full_chain(chain),
        _good_entries(chain),
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_good_ocsp_with_non_matching_identity_profile_is_rejected(
    client, root_id, chain
):
    registered = client.post(
        "/v1/workload-identities",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "claims": [
                {
                    "issuer": chain["leaf_cert"].issuer.rfc4514_string(),
                    "subject": chain["leaf_cert"].subject.rfc4514_string(),
                    "uri": "spiffe://example.org/workload/someone-else",
                }
            ],
        },
    )
    assert registered.status_code == 201
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"],
        chain["leaf_key"],
        _full_chain(chain),
        _good_entries(chain),
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_good_ocsp_under_retired_anchor_is_rejected(client, root_id, chain):
    retired = client.post(
        f"/v1/trust-roots/{root_id}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"],
        chain["leaf_key"],
        _full_chain(chain),
        _good_entries(chain),
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


def test_good_ocsp_anchored_to_unconfigured_root_is_rejected(
    client, chain
):
    other_root_key, other_root_cert = make_root("other-root")
    intermediate_key, intermediate_cert = make_intermediate(
        other_root_cert, other_root_key
    )
    leaf_key, leaf_cert = make_leaf(intermediate_cert, intermediate_key)
    certificates = [leaf_cert, intermediate_cert, other_root_cert]
    keys = [leaf_key, intermediate_key, other_root_key]
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"],
        leaf_key,
        certificates,
        ocsp_entries_for_chain(certificates, keys),
    )

    _, response = _settle(client, created, evidence)

    assert response.json()["status"] == "rejected"


# ---------------------------------------------------------------------------
# Settlement, compatibility and non-disclosure
# ---------------------------------------------------------------------------


def test_rejected_ocsp_conclusion_is_settled_and_repeated(
    client, root_id, chain, app
):
    now = datetime.now(timezone.utc)
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    entries[0] = make_ocsp_entry(
        certificates[0], certificates[1], certificates[1],
        chain["intermediate_key"],
        this_update=now - timedelta(days=3),
        next_update=now - timedelta(minutes=1),
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )
    evidence_id, first = _settle(client, created, evidence)
    assert first.json()["status"] == "rejected"

    # Repeating the identical call returns the first conclusion. A later
    # call carrying a different ocsp_responses value also returns that
    # same conclusion: a settled proof is answered before the body is
    # inspected, so the new field is neither read nor re-validated.
    second = _verify(client, evidence_id, created, evidence)
    assert second.status_code == 200
    assert second.json() == first.json()
    altered = json.loads(evidence)
    altered["ocsp_responses"] = []
    third = _verify(client, evidence_id, created, json.dumps(altered))
    assert third.status_code == 200
    assert third.json() == first.json()
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "rejected"
        assert record.verification_result == "rejected"


def test_verified_ocsp_conclusion_is_repeated_without_revalidation(
    client, root_id, chain
):
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"],
        chain["leaf_key"],
        _full_chain(chain),
        _good_entries(chain),
    )
    evidence_id, first = _settle(client, created, evidence)
    assert first.json()["status"] == "verified"

    second = _verify(client, evidence_id, created, evidence)
    assert second.status_code == 200
    assert second.json() == first.json()


def test_ocsp_field_is_ignored_by_hmac_format(
    client, monkeypatch, root_id, chain
):
    # The attested-nonce-json format must not interpret the new field.
    monkeypatch.setenv(
        "PROOF_RELEASE_ATTESTED_NONCE_SECRET", HMAC_SECRET
    )
    created = _challenge(client)
    mac_key = hmac.new(
        HMAC_SECRET.encode(),
        f"{TENANT}:{WORKLOAD}".encode(),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": {}, "nonce": created["nonce"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    mac = hmac.new(mac_key, payload, hashlib.sha256).hexdigest()
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": {},
            "mac": mac,
            # Even a structurally impossible value must be ignored.
            "ocsp_responses": [],
        }
    )

    _, response = _settle(
        client, created, evidence, evidence_format=ATTESTED_NONCE_JSON
    )

    assert response.json()["status"] == "verified"


def test_ocsp_material_is_never_persisted_or_returned(
    client, root_id, chain, app
):
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )
    evidence_id, response = _settle(client, created, evidence)
    assert response.status_code == 200

    assert "BEGIN CERTIFICATE" not in response.text
    for entry in entries:
        assert entry["response"] not in response.text
        assert entry["responder_certificate"] not in response.text
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        columns = {
            column.name: getattr(record, column.name)
            for column in record.__table__.columns
        }
    for name, value in columns.items():
        rendered = str(value)
        assert evidence not in rendered, f"evidence leaked into column {name}"
        assert "BEGIN CERTIFICATE" not in rendered, (
            f"responder certificate leaked into column {name}"
        )
        for entry in entries:
            assert entry["response"] not in rendered, (
                f"OCSP response leaked into column {name}"
            )


def test_revoked_ocsp_material_is_never_returned_or_persisted(
    client, root_id, chain, app
):
    certificates = _full_chain(chain)
    entries = _good_entries(chain)
    entries[0] = make_ocsp_entry(
        certificates[0], certificates[1], certificates[1],
        chain["intermediate_key"],
        status=ocsp.OCSPCertStatus.REVOKED,
    )
    created = _challenge(client)
    evidence = _evidence(
        created["nonce"], chain["leaf_key"], certificates, entries
    )
    evidence_id, response = _settle(client, created, evidence)
    assert response.json()["status"] == "rejected"

    assert "BEGIN CERTIFICATE" not in response.text
    for entry in entries:
        assert entry["response"] not in response.text
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        rendered = str(
            {
                column.name: getattr(record, column.name)
                for column in record.__table__.columns
            }
        )
    assert "BEGIN CERTIFICATE" not in rendered
