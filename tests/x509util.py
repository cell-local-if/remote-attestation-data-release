"""Helpers to build X.509 certificate chains and x509-format evidence."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import NameOID


def generate_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def make_certificate(
    *,
    subject_cn: str,
    subject_key,
    ca: bool = False,
    issuer_cert: x509.Certificate | None = None,
    issuer_key=None,
    not_before: datetime | None = None,
    not_after: datetime | None = None,
) -> x509.Certificate:
    now = datetime.now(timezone.utc)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject_cn)])
    issuer = issuer_cert.subject if issuer_cert is not None else subject
    signing_key = issuer_key if issuer_key is not None else subject_key
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(subject_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before or (now - timedelta(hours=1)))
        .not_valid_after(not_after or (now + timedelta(days=30)))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
        .sign(signing_key, hashes.SHA256())
    )


def make_root_ca(cn: str = "test-root", **overrides):
    key = generate_key()
    return key, make_certificate(subject_cn=cn, subject_key=key, ca=True, **overrides)


def make_leaf(cn: str, issuer_cert, issuer_key, **overrides):
    key = generate_key()
    cert = make_certificate(
        subject_cn=cn,
        subject_key=key,
        issuer_cert=issuer_cert,
        issuer_key=issuer_key,
        **overrides,
    )
    return key, cert


def pem(cert: x509.Certificate) -> str:
    return cert.public_bytes(serialization.Encoding.PEM).decode("ascii")


def make_evidence(nonce: str, claims: dict, chain_pems, leaf_key) -> str:
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = leaf_key.sign(payload, padding.PKCS1v15(), hashes.SHA256())
    return json.dumps(
        {
            "nonce": nonce,
            "claims": claims,
            "certificate_chain": list(chain_pems),
            "signature": base64.b64encode(signature).decode("ascii"),
        }
    )
