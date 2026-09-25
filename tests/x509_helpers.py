"""Shared helpers for building X.509 test certificate chains."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


def generate_key():
    return ec.generate_private_key(ec.SECP256R1())


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def build_certificate(
    common_name,
    public_key,
    issuer_name,
    issuer_key,
    *,
    ca,
    not_before=None,
    not_after=None,
    extra_extensions=(),
):
    now = datetime.now(timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name(common_name))
        .issuer_name(issuer_name)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before or now - timedelta(hours=1))
        .not_valid_after(not_after or now + timedelta(days=30))
        .add_extension(
            x509.BasicConstraints(ca=ca, path_length=None), critical=True
        )
    )
    for extension in extra_extensions:
        builder = builder.add_extension(extension, critical=False)
    return builder.sign(issuer_key, hashes.SHA256())


def make_root(common_name="test-root", **kwargs):
    key = generate_key()
    cert = build_certificate(
        common_name, key.public_key(), _name(common_name), key, ca=True, **kwargs
    )
    return key, cert


def make_intermediate(root_cert, root_key, common_name="test-intermediate", **kwargs):
    key = generate_key()
    kwargs.setdefault("ca", True)
    cert = build_certificate(
        common_name, key.public_key(), root_cert.subject, root_key, **kwargs
    )
    return key, cert


def make_leaf(issuer_cert, issuer_key, common_name="test-leaf", *, uri=None, **kwargs):
    key = generate_key()
    kwargs.setdefault("ca", False)
    extensions = []
    if uri is not None:
        extensions.append(
            x509.SubjectAlternativeName(
                [x509.UniformResourceIdentifier(uri)]
            )
        )
    cert = build_certificate(
        common_name,
        key.public_key(),
        issuer_cert.subject,
        issuer_key,
        extra_extensions=extensions,
        **kwargs,
    )
    return key, cert


def pem(certificate) -> str:
    return certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")


def private_key_pem(key) -> str:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")


def sign_payload(key, nonce: str, claims: dict) -> str:
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = key.sign(payload, ec.ECDSA(hashes.SHA256()))
    return base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")


def evidence_document(nonce, claims, chain_pems, signature) -> str:
    return json.dumps(
        {
            "nonce": nonce,
            "claims": claims,
            "certificate_chain": list(chain_pems),
            "signature": signature,
        }
    )


def make_evidence(nonce, leaf_key, chain_certificates, claims=None) -> str:
    claims = claims if claims is not None else {}
    return evidence_document(
        nonce,
        claims,
        [pem(certificate) for certificate in chain_certificates],
        sign_payload(leaf_key, nonce, claims),
    )
