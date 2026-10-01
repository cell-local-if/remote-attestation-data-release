"""Shared helpers for building X.509 test certificate chains."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509 import ocsp
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

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


# ---------------------------------------------------------------------------
# Embedded OCSP response helpers
# ---------------------------------------------------------------------------


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def make_delegated_responder(
    issuing_ca_cert,
    issuing_ca_key,
    common_name="test-ocsp-responder",
    **kwargs,
):
    """Build an id-kp-OCSPSigning delegated responder cert/key (cA FALSE)."""
    key = generate_key()
    kwargs.setdefault("ca", False)
    cert = build_certificate(
        common_name,
        key.public_key(),
        issuing_ca_cert.subject,
        issuing_ca_key,
        extra_extensions=[
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.OCSP_SIGNING])
        ],
        **kwargs,
    )
    return key, cert


def make_ocsp_response_der(
    target_cert,
    issuing_ca_cert,
    signer_key,
    *,
    status=ocsp.OCSPCertStatus.GOOD,
    this_update=None,
    next_update="default",
    revocation_time=None,
    responder_cert=None,
    by_key=False,
    hash_algorithm=None,
):
    """Build a single-SingleResponse DER OCSPResponse.

    Times default to a current window (thisUpdate 5 minutes ago,
    nextUpdate an hour ahead). Pass ``next_update=None`` to omit it.
    ``responder_cert`` identifies the signer; when omitted the issuing CA
    itself is the responder.
    """
    now = datetime.now(timezone.utc)
    if this_update is None:
        this_update = now - timedelta(minutes=5)
    if next_update == "default":
        next_update = now + timedelta(hours=1)
    responder = responder_cert or issuing_ca_cert
    digest = hash_algorithm or hashes.SHA1()
    response = (
        ocsp.OCSPResponseBuilder()
        .add_response(
            target_cert,
            issuing_ca_cert,
            digest,
            status,
            this_update,
            next_update,
            revocation_time,
            None,
        )
        .responder_id(
            ocsp.OCSPResponderEncoding.HASH if by_key
            else ocsp.OCSPResponderEncoding.NAME,
            responder,
        )
        .sign(signer_key, hashes.SHA256())
    )
    return response.public_bytes(serialization.Encoding.DER)


def ocsp_entry(
    target_cert,
    issuing_ca_cert,
    signer_key,
    *,
    responder_cert=None,
    **kwargs,
):
    """Build one ocsp_responses array entry (unpadded base64url DER + PEM)."""
    der = make_ocsp_response_der(
        target_cert,
        issuing_ca_cert,
        signer_key,
        responder_cert=responder_cert,
        **kwargs,
    )
    certificate = responder_cert or issuing_ca_cert
    return {"response": _b64url(der), "responder_certificate": pem(certificate)}


def _ocsp_tlv(data: bytes, index: int = 0):
    tag = data[index]
    index += 1
    length = data[index]
    index += 1
    if length & 0x80:
        count = length & 0x7F
        length = int.from_bytes(data[index : index + count], "big")
        index += count
    return tag, data[index : index + length], index + length


def _ocsp_encode(tag: int, value: bytes) -> bytes:
    if len(value) < 0x80:
        length = bytes([len(value)])
    else:
        raw = len(value).to_bytes((len(value).bit_length() + 7) // 8, "big")
        length = bytes([0x80 | len(raw)]) + raw
    return bytes([tag]) + length + value


def _sign_ocsp_tbs(signer_key, tbs: bytes) -> bytes:
    if isinstance(signer_key, ec.EllipticCurvePrivateKey):
        return signer_key.sign(tbs, ec.ECDSA(hashes.SHA256()))
    from cryptography.hazmat.primitives.asymmetric import padding as _padding

    return signer_key.sign(tbs, _padding.PKCS1v15(), hashes.SHA256())


def _rebuild_ocsp_with_tbs(der: bytes, signer_key, tbs_transform) -> bytes:
    """Replace the ResponseData (tbs) of a DER OCSPResponse and re-sign it.

    The BasicOCSPResponse signature algorithm element is preserved; only
    the tbs content changes, so the transform can alter producedAt or the
    responses sequence while keeping every other field.
    """
    _, outer_value, _ = _ocsp_tlv(der)
    _, _, after_status = _ocsp_tlv(outer_value, 0)
    _, a0_value, _ = _ocsp_tlv(outer_value, after_status)
    _, response_bytes, _ = _ocsp_tlv(a0_value)
    _, _, after_oid = _ocsp_tlv(response_bytes, 0)
    _, basic_der, _ = _ocsp_tlv(response_bytes, after_oid)

    _, basic, _ = _ocsp_tlv(basic_der)
    _, _, after_tbs = _ocsp_tlv(basic, 0)
    _, _, after_alg = _ocsp_tlv(basic, after_tbs)
    sig_tag, _, after_sig = _ocsp_tlv(basic, after_alg)
    trailing = basic[after_sig:]

    # Re-parse to capture the raw element byte slices.
    _, tbs_value, _ = _ocsp_tlv(basic, 0)
    alg_tlv = basic[after_tbs:after_alg]

    new_tbs_content = tbs_transform(tbs_value)
    new_tbs = _ocsp_encode(0x30, new_tbs_content)
    # The OCSP signature is over the full ResponseData DER TLV.
    new_signature = _sign_ocsp_tbs(signer_key, new_tbs)
    new_basic = _ocsp_encode(
        0x30,
        new_tbs
        + alg_tlv
        + _ocsp_encode(sig_tag, b"\x00" + new_signature)
        + trailing,
    )
    new_response_bytes = _ocsp_encode(
        0x30,
        response_bytes[:after_oid] + _ocsp_encode(0x04, new_basic),
    )
    new_outer = _ocsp_encode(
        0x30,
        outer_value[:after_status] + _ocsp_encode(0xA0, new_response_bytes),
    )
    return new_outer


def _generalized_time_der(value: datetime) -> bytes:
    return _ocsp_encode(
        0x18, value.astimezone(timezone.utc).strftime("%Y%m%d%H%M%SZ").encode()
    )


def ocsp_with_produced_at(der: bytes, signer_key, produced_at: datetime) -> bytes:
    """Re-sign a DER OCSPResponse with producedAt replaced by ``produced_at``."""

    def transform(tbs: bytes) -> bytes:
        children = []
        pos = 0
        replaced = False
        while pos < len(tbs):
            _, _, next_pos = _ocsp_tlv(tbs, pos)
            tag = tbs[pos]
            if not replaced and tag == 0x18:
                # The first GeneralizedTime in ResponseData is producedAt.
                children.append(_generalized_time_der(produced_at))
                replaced = True
            else:
                children.append(tbs[pos:next_pos])
            pos = next_pos
        return b"".join(children)

    return _rebuild_ocsp_with_tbs(der, signer_key, transform)


def ocsp_with_two_single_responses(der: bytes, signer_key) -> bytes:
    """Re-sign a DER OCSPResponse with its SingleResponse duplicated."""

    def transform(tbs: bytes) -> bytes:
        children = []
        pos = 0
        while pos < len(tbs):
            tag, value, next_pos = _ocsp_tlv(tbs, pos)
            if tag == 0x30:
                # The responses SEQUENCE: duplicate its sole SingleResponse.
                _, _, after_first = _ocsp_tlv(value, 0)
                single_tlv = value[:after_first]
                children.append(_ocsp_encode(0x30, value + single_tlv))
            else:
                children.append(tbs[pos:next_pos])
            pos = next_pos
        return b"".join(children)

    return _rebuild_ocsp_with_tbs(der, signer_key, transform)


def make_crl(
    issuer_cert,
    issuer_key,
    revoked,
    *,
    number,
    this_update=None,
    next_update=None,
    ttl_days=30,
    include_number=True,
    signer_key=None,
    issuer_name=None,
):
    """Build a PEM X.509 v2 CRL.

    ``revoked`` is an iterable of ``(serial_number, revocation_date)``
    pairs (a bare integer means revoked an hour ago). Times default to a
    valid window (thisUpdate in the past, nextUpdate ``ttl_days`` ahead).
    ``include_number=False`` omits the CRLNumber extension, ``signer_key``
    signs with a different key (to forge an invalid signature), and
    ``issuer_name`` overrides the issuer DN (to produce a DN mismatch).
    """
    now = datetime.now(timezone.utc)
    if this_update is None:
        this_update = now - timedelta(hours=1)
    if next_update is None:
        next_update = now + timedelta(days=ttl_days)
    builder = x509.CertificateRevocationListBuilder().issuer_name(
        issuer_name if issuer_name is not None else issuer_cert.subject
    ).last_update(this_update).next_update(next_update)
    if include_number:
        builder = builder.add_extension(x509.CRLNumber(number), critical=False)
    for item in revoked:
        if isinstance(item, tuple):
            serial, revocation_date = item
        else:
            serial, revocation_date = item, now - timedelta(hours=1)
        revoked_builder = (
            x509.RevokedCertificateBuilder()
            .serial_number(serial)
            .revocation_date(revocation_date)
        )
        builder = builder.add_revoked_certificate(revoked_builder.build())
    signing_key = signer_key if signer_key is not None else issuer_key
    crl = builder.sign(signing_key, hashes.SHA256())
    return crl.public_bytes(serialization.Encoding.PEM).decode("ascii")


def _der_tlv(data: bytes, index: int = 0):
    """Parse one DER TLV at index, returning (tag, value_bytes, next_index)."""
    tag = data[index]
    index += 1
    length = data[index]
    index += 1
    if length & 0x80:
        count = length & 0x7F
        length = int.from_bytes(data[index : index + count], "big")
        index += count
    value = data[index : index + length]
    return tag, value, index + length


def _der_encode(tag: int, value: bytes) -> bytes:
    if len(value) < 0x80:
        length = bytes([len(value)])
    else:
        raw = len(value).to_bytes((len(value).bit_length() + 7) // 8, "big")
        length = bytes([0x80 | len(raw)]) + raw
    return bytes([tag]) + length + value


def crl_without_next_update(issuer_cert, issuer_key, number=1, *, revoked=()):
    """Build a v2 CRL carrying CRLNumber but no nextUpdate field.

    The high-level builder refuses to omit nextUpdate, so a valid CRL is
    built and DER-surgery removes the optional Time element from
    tbsCertList; the altered TBS is then re-signed by the issuer.
    """
    from cryptography.hazmat.primitives.asymmetric import ec, rsa

    now = datetime.now(timezone.utc)
    valid = make_crl(
        issuer_cert,
        issuer_key,
        list(revoked),
        number=number,
        this_update=now - timedelta(hours=1),
    )
    der_bytes = x509.load_pem_x509_crl(valid.encode("ascii")).public_bytes(
        serialization.Encoding.DER
    )

    outer_tag, outer_value, _ = _der_tlv(der_bytes)
    # Outer CertificateList ::= SEQUENCE { tbsCertList, signatureAlgorithm,
    # signatureValue BIT STRING }.
    tbs_tag, tbs_value, after_tbs = _der_tlv(outer_value)
    sig_alg_tag, sig_alg_value, after_alg = _der_tlv(outer_value, after_tbs)
    sig_tag, sig_value, _ = _der_tlv(outer_value, after_alg)

    # Walk the TBS children, dropping the *second* Time (nextUpdate):
    # UTCTime (0x17) or GeneralizedTime (0x18). The first is thisUpdate.
    children = []
    pos = 0
    time_seen = 0
    while pos < len(tbs_value):
        tag, value, next_pos = _der_tlv(tbs_value, pos)
        if tag in (0x17, 0x18):
            time_seen += 1
            if time_seen == 2:
                pos = next_pos
                continue
        children.append(tbs_value[pos:next_pos])
        pos = next_pos
    new_tbs = _der_encode(0x30, b"".join(children))

    if isinstance(issuer_key, ec.EllipticCurvePrivateKey):
        signature = issuer_key.sign(new_tbs, ec.ECDSA(hashes.SHA256()))
    elif isinstance(issuer_key, rsa.RSAPrivateKey):
        import hashlib as _hashlib  # noqa: F401
        from cryptography.hazmat.primitives.asymmetric import padding

        signature = issuer_key.sign(
            new_tbs, padding.PKCS1v15(), hashes.SHA256()
        )
    else:
        signature = issuer_key.sign(new_tbs)
    new_sig_value = b"\x00" + signature  # BIT STRING: zero unused bits
    new_outer = _der_encode(
        0x30,
        new_tbs
        + _der_encode(sig_alg_tag, sig_alg_value)
        + _der_encode(sig_tag, new_sig_value),
    )
    import base64 as _base64

    return (
        "-----BEGIN X509 CRL-----\n"
        + _base64.encodebytes(new_outer).decode("ascii")
        + "-----END X509 CRL-----\n"
    )
