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


def build_crl(
    issuer_cert,
    issuer_key,
    crl_number,
    revoked,
    *,
    this_update=None,
    next_update=None,
    issuer_name=None,
    signing_key=None,
):
    """Build and sign an X.509 v2 CRL PEM.

    ``revoked`` is an iterable of ``(serial_number, revocation_date)``
    pairs. Times default to the past/future window around now. The issuer
    name and signing key can be overridden to produce a CRL whose DN or
    signature does not match the trust root.
    """
    now = datetime.now(timezone.utc)
    this_update = this_update if this_update is not None else now - timedelta(days=1)
    next_update = next_update if next_update is not None else now + timedelta(days=1)
    builder = (
        x509.CertificateRevocationListBuilder()
        .issuer_name(issuer_name or issuer_cert.subject)
        .last_update(this_update)
        .next_update(next_update)
        .add_extension(x509.CRLNumber(crl_number), critical=False)
    )
    for serial_number, revocation_date in revoked:
        entry = (
            x509.RevokedCertificateBuilder()
            .serial_number(serial_number)
            .revocation_date(revocation_date)
            .build()
        )
        builder = builder.add_revoked_certificate(entry)
    crl = builder.sign(signing_key or issuer_key, hashes.SHA256())
    return crl.public_bytes(serialization.Encoding.PEM).decode("ascii")


def build_crl_without_crl_number(issuer_cert, issuer_key):
    """A v1-style CRL (no CRLNumber extension), which must be rejected."""
    now = datetime.now(timezone.utc)
    crl = (
        x509.CertificateRevocationListBuilder()
        .issuer_name(issuer_cert.subject)
        .last_update(now - timedelta(days=1))
        .next_update(now + timedelta(days=1))
        .sign(issuer_key, hashes.SHA256())
    )
    return crl.public_bytes(serialization.Encoding.PEM).decode("ascii")


# --- Minimal raw-DER CRL construction ---------------------------------------
#
# The high-level cryptography builder refuses the non-conforming CRLs the
# service must reject (no nextUpdate, a CRLNumber of zero, a zero serial),
# so these small DER helpers assemble a signed CRL by hand. Each produced
# CRL is a real, signature-valid ASN.1 object that cryptography parses;
# only the targeted field is malformed.

_ECDSA_SHA256_OID = "1.2.840.10045.4.3.2"
_CRL_NUMBER_OID = "2.5.29.20"


def _der_length(length: int) -> bytes:
    if length < 0x80:
        return bytes([length])
    body = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(body)]) + body


def _der_tag(tag_byte: int, content: bytes) -> bytes:
    return bytes([tag_byte]) + _der_length(len(content)) + content


def _der_sequence(*parts: bytes) -> bytes:
    return _der_tag(0x30, b"".join(parts))


def _der_integer(value: int) -> bytes:
    if value >= 0:
        body = value.to_bytes(max(1, (value.bit_length() + 7) // 8 or 1), "big")
        if body[0] & 0x80:
            body = b"\x00" + body
    else:
        width = (abs(value).bit_length() + 8) // 8
        body = (value + (1 << (8 * width))).to_bytes(width, "big")
    return _der_tag(0x02, body)


def _der_oid(value: str) -> bytes:
    numbers = [int(part) for part in value.split(".")]
    body = bytes([40 * numbers[0] + numbers[1]])
    for number in numbers[2:]:
        if number == 0:
            body += b"\x00"
            continue
        chunks = [number & 0x7F]
        number >>= 7
        while number:
            chunks.append((number & 0x7F) | 0x80)
            number >>= 7
        body += bytes(reversed(chunks))
    return _der_tag(0x06, body)


def _der_algorithm_identifier(oid_value: str) -> bytes:
    return _der_sequence(_der_oid(oid_value))


def _der_crl_time(moment: datetime) -> bytes:
    if moment.year >= 2050:
        return _der_tag(0x18, moment.strftime("%Y%m%d%H%M%SZ").encode())
    return _der_tag(0x17, moment.strftime("%y%m%d%H%M%SZ").encode())


def _der_bit_string(raw: bytes) -> bytes:
    return _der_tag(0x03, b"\x00" + raw)


def _der_octet_string(raw: bytes) -> bytes:
    return _der_tag(0x04, raw)


def _der_explicit(number: int, content: bytes) -> bytes:
    return _der_tag(0xA0 | number, content)


def raw_crl(
    issuer_cert,
    issuer_key,
    crl_number,
    revoked,
    *,
    this_update,
    next_update,
    include_next_update=True,
    include_version=True,
):
    """Assemble a signed DER CRL; returns PEM text.

    ``revoked`` is an iterable of ``(serial, revocation_date)`` and may
    carry serial values the high-level builder rejects. Omitting
    nextUpdate or setting the CRLNumber to zero targets specific
    validation failures while keeping the signature valid.
    """
    algorithm = _der_algorithm_identifier(_ECDSA_SHA256_OID)
    issuer = issuer_cert.subject.public_bytes()
    parts: list[bytes] = []
    if include_version:
        # INTEGER 1 marks a v2 CRL.
        parts.append(_der_integer(1))
    parts += [algorithm, issuer, _der_crl_time(this_update)]
    if include_next_update and next_update is not None:
        parts.append(_der_crl_time(next_update))
    if revoked:
        entries = [
            _der_sequence(_der_integer(serial), _der_crl_time(revocation_date))
            for serial, revocation_date in revoked
        ]
        parts.append(_der_sequence(*entries))
    crl_number_extension = _der_sequence(
        _der_oid(_CRL_NUMBER_OID), _der_octet_string(_der_integer(crl_number))
    )
    parts.append(_der_explicit(0, _der_sequence(crl_number_extension)))
    tbs_cert_list = _der_sequence(*parts)
    signature = issuer_key.sign(tbs_cert_list, ec.ECDSA(hashes.SHA256()))
    der = _der_sequence(
        tbs_cert_list, algorithm, _der_bit_string(signature)
    )
    return (
        b"-----BEGIN X509 CRL-----\n"
        + base64.encodebytes(der)
        + b"-----END X509 CRL-----\n"
    ).decode("ascii")
