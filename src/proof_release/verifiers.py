"""Public verifier plugin interface for attestation evidence formats.

External code registers verifiers keyed by ``evidence_format`` (the value
provided when evidence is submitted). At verification time the service hands
the selected verifier the *raw* evidence together with the challenge, tenant
and workload context. The service never persists, logs or returns the raw
evidence; it is only held transiently for the duration of the call.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.x509 import ocsp
from cryptography.x509.oid import ExtendedKeyUsageOID

__all__ = [
    "ChallengeContext",
    "VerificationContext",
    "VerificationResult",
    "Verifier",
    "VerifierPluginError",
    "VerifierRegistry",
    "AttestedNonceJSONVerifier",
    "AttestedNonceJSONV2Verifier",
    "X509AttestedNonceJSONVerifier",
    "CrlValidationError",
    "ParsedCrl",
    "ValidatedCrl",
    "load_attested_nonce_keys",
    "load_crl",
    "parse_crl",
    "validate_crl_against_root",
    "default_registry",
    "register_verifier",
    "unregister_verifier",
    "get_verifier",
]

#: Built-in format identifier for the reference verifier.
ATTESTED_NONCE_JSON = "attested-nonce-json"

#: Built-in format identifier for the X.509 certificate-chain verifier.
X509_ATTESTED_NONCE_JSON = "x509-attested-nonce-json"

#: Built-in format identifier for the key-rotating shared-secret verifier.
ATTESTED_NONCE_JSON_V2 = "attested-nonce-json-v2"

#: Environment variable holding the MAC secret for the built-in verifier.
ATTESTED_NONCE_SECRET_ENV = "PROOF_RELEASE_ATTESTED_NONCE_SECRET"

#: Environment variable holding the kid -> shared-key JSON object for the
#: v2 verifier.
ATTESTED_NONCE_KEYS_ENV = "PROOF_RELEASE_ATTESTED_NONCE_KEYS"

#: Maximum number of kid entries accepted in the v2 key configuration.
MAX_ATTESTED_NONCE_KEYS = 32

#: Development-only fallback secret. Deployments must set the env var above;
#: this constant exists so the format is usable out of the box in tests and
#: local development and must never be relied on in production.
DEMO_ATTESTED_NONCE_SECRET = "dev-only-attested-nonce-secret"

_BASE64URL_ALPHABET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
)


@dataclass(frozen=True)
class ChallengeContext:
    """Non-sensitive challenge state made available to verifiers.

    The plaintext nonce is intentionally absent: only its digest is stored,
    so it cannot be reconstructed from the context.
    """

    challenge_id: str
    nonce_digest: str
    status: str
    issued_at: datetime
    expires_at: datetime
    consumed_at: datetime | None


@dataclass(frozen=True)
class VerificationContext:
    """Context passed to a verifier for a single verification attempt.

    ``evidence`` is the raw evidence supplied in the verify request. It must
    not be retained by the verifier beyond the call or written to shared
    state; the service discards it as soon as the verifier returns.
    """

    evidence: str
    tenant_id: str
    workload_id: str
    challenge: ChallengeContext
    #: PEM-encoded X.509 trust-root certificates configured for exactly this
    #: tenant and workload (public material only). Empty when none are
    #: configured. Verifiers that do not anchor to trust roots may ignore it.
    trust_roots: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of a verifier invocation.

    ``accepted`` is the only field the service acts on: it alone decides
    the verified/rejected settlement and the fixed, service-defined result
    code that is persisted. ``detail`` is ignored and discarded at the
    service boundary — plugin-supplied text is never persisted, logged, or
    returned, so plugins must not place raw evidence, key material, or
    other private context here (or in raised exceptions) expecting it to
    be retained or propagated.
    """

    accepted: bool
    detail: str | None = None


class Verifier(ABC):
    """Base class for evidence format verifier plugins.

    Subclasses set ``format_name`` and implement :meth:`verify`.
    Implementations must be safe to call from concurrent requests and must
    not raise for ordinary malformed evidence — return
    ``VerificationResult(accepted=False)`` instead. Raise only for genuine
    internal failures (the service treats a raised exception as a 500 and
    leaves no verification state behind).
    """

    #: The ``evidence_format`` this verifier handles.
    format_name: str

    @abstractmethod
    def verify(self, context: VerificationContext) -> VerificationResult:
        """Validate the raw evidence against the bound challenge context."""


class VerifierPluginError(RuntimeError):
    """A verifier cannot run because of its own configuration or internals.

    Distinct from an ordinary rejection: the service maps this to a 500
    (``verifier plugin failure``) and leaves the evidence ``received`` so
    the identical request can be retried once the fault is fixed. Messages
    must be fixed, service-defined category strings — never raw evidence,
    key material, or environment content.
    """


class VerifierRegistry:
    """Maps evidence format names to verifier instances."""

    def __init__(self) -> None:
        self._verifiers: Dict[str, Verifier] = {}
        self._lock = threading.Lock()

    def register(self, verifier: Verifier) -> None:
        """Register (or replace) a verifier for its format name."""
        name = getattr(verifier, "format_name", "")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("verifier format_name must be a non-empty string")
        with self._lock:
            self._verifiers[name] = verifier

    def unregister(self, format_name: str) -> None:
        with self._lock:
            self._verifiers.pop(format_name, None)

    def get(self, format_name: str) -> Verifier | None:
        return self._verifiers.get(format_name)

    def __contains__(self, format_name: object) -> bool:
        return format_name in self._verifiers

    def format_names(self) -> tuple[str, ...]:
        """Snapshot the registered ``evidence_format`` names.

        Returns every currently registered name exactly as it was
        registered, de-duplicated and sorted strictly ascending by Unicode
        code point. The snapshot is taken under the registry lock so a
        concurrent register/replace/unregister can never yield a partial
        or inconsistent view; each call reflects one complete state of the
        registry. Only the names are exposed — never verifier instances,
        classes, module paths, or configuration.
        """
        with self._lock:
            return tuple(sorted(self._verifiers))


def _reject() -> VerificationResult:
    return VerificationResult(accepted=False)


def _is_unpadded_base64url(value: str) -> bool:
    if not value or "=" in value or any(c not in _BASE64URL_ALPHABET for c in value):
        return False
    try:
        base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (binascii.Error, ValueError):
        return False
    return True


class AttestedNonceJSONVerifier(Verifier):
    """Built-in verifier for the ``attested-nonce-json`` format.

    The evidence is a JSON object cryptographically bound to the challenge
    nonce::

        {"nonce": "<unpadded base64url>", "claims": {...}, "mac": "<hex>"}

    ``mac`` is the lowercase hex HMAC-SHA256 over the canonical JSON
    serialization (sorted keys, compact separators) of
    ``{"claims": ..., "nonce": ...}``. The MAC key is derived per workload::

        mac_key = HMAC-SHA256(key_material, tenant_id + ":" + workload_id)

    ``key_material`` is the shared secret of the attestation side. It may be
    supplied explicitly; otherwise the ``PROOF_RELEASE_ATTESTED_NONCE_SECRET``
    environment variable is used, falling back to a clearly labelled
    development-only constant.

    Verification checks, in order: well-formed JSON object, unpadded
    base64url nonce equal to the challenge nonce (compared through stored
    digest), and valid MAC. Any failure yields a plain rejected result;
    no reasons or evidence content are attached, since the service
    discards plugin-supplied text anyway.
    """

    format_name = ATTESTED_NONCE_JSON

    def __init__(
        self,
        key_material: str | bytes | None = None,
        *,
        secret_provider: Callable[[], str | bytes] | None = None,
    ) -> None:
        self._key_material = key_material
        self._secret_provider = secret_provider

    def _current_secret(self) -> str | bytes:
        if self._key_material is not None:
            return self._key_material
        if self._secret_provider is not None:
            return self._secret_provider()
        return os.environ.get(
            ATTESTED_NONCE_SECRET_ENV, DEMO_ATTESTED_NONCE_SECRET
        )

    def _mac_key(self, tenant_id: str, workload_id: str) -> bytes:
        secret = self._current_secret()
        if isinstance(secret, str):
            secret = secret.encode("utf-8")
        return hmac.new(
            secret,
            f"{tenant_id}:{workload_id}".encode("utf-8"),
            hashlib.sha256,
        ).digest()

    def verify(self, context: VerificationContext) -> VerificationResult:
        try:
            document = json.loads(context.evidence)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _reject()
        if not isinstance(document, dict):
            return _reject()

        nonce = document.get("nonce")
        claims = document.get("claims", {})
        mac_hex = document.get("mac")
        if not isinstance(nonce, str) or not nonce:
            return _reject()
        if "claims" in document and not isinstance(claims, dict):
            return _reject()
        if not isinstance(mac_hex, str) or not mac_hex:
            return _reject()

        if not _is_unpadded_base64url(nonce):
            return _reject()

        # The attested nonce must be exactly the nonce of the bound challenge.
        if not hmac.compare_digest(
            hashlib.sha256(nonce.encode("ascii")).hexdigest(),
            context.challenge.nonce_digest,
        ):
            return _reject()

        signed_payload = json.dumps(
            {"claims": claims, "nonce": nonce},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_mac = hmac.new(
            self._mac_key(context.tenant_id, context.workload_id),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()
        try:
            binascii.unhexlify(mac_hex.lower())
        except (binascii.Error, ValueError):
            return _reject()
        if not hmac.compare_digest(expected_mac, mac_hex.lower()):
            return _reject()

        return VerificationResult(accepted=True)


#: Key identifier shape shared by the v2 evidence document and the v2 key
#: configuration.
_KID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")

#: The v2 MAC is exactly 64 lowercase hexadecimal characters.
_MAC_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")

#: Domain-separation prefix of the v2 MAC input: the format name, one NUL
#: byte, then the canonical JSON document.
_V2_MAC_PREFIX = ATTESTED_NONCE_JSON_V2.encode("ascii") + b"\x00"


def load_attested_nonce_keys() -> Dict[str, bytes]:
    """Load and validate the v2 shared-key map from the environment.

    ``PROOF_RELEASE_ATTESTED_NONCE_KEYS`` must be a JSON object of at most
    32 entries mapping a key id (``^[A-Za-z0-9_-]{1,64}$``) to an unpadded
    base64url-encoded 32-byte shared key. Any deviation — variable missing,
    broken JSON, wrong shape, too many entries, an invalid kid, or a key
    that is not exactly 32 bytes — raises :class:`VerifierPluginError`
    with a fixed category message, so the caller fails as a 500 and no
    verification state is settled.
    """
    raw = os.environ.get(ATTESTED_NONCE_KEYS_ENV)
    if raw is None:
        raise VerifierPluginError("attested nonce keys are not configured")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        raise VerifierPluginError("attested nonce keys are not valid JSON")
    if not isinstance(document, dict):
        raise VerifierPluginError("attested nonce keys must be a JSON object")
    if len(document) > MAX_ATTESTED_NONCE_KEYS:
        raise VerifierPluginError("attested nonce keys exceed the entry limit")
    keys: Dict[str, bytes] = {}
    for kid, encoded in document.items():
        if not isinstance(kid, str) or _KID_PATTERN.fullmatch(kid) is None:
            raise VerifierPluginError("attested nonce key id is invalid")
        if not isinstance(encoded, str) or not _is_unpadded_base64url(encoded):
            raise VerifierPluginError("attested nonce key is not base64url")
        key = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        if len(key) != 32:
            raise VerifierPluginError("attested nonce key must be 32 bytes")
        keys[kid] = key
    return keys


class AttestedNonceJSONV2Verifier(Verifier):
    """Built-in verifier for the ``attested-nonce-json-v2`` format.

    Same shape as :class:`AttestedNonceJSONVerifier` plus a key id that
    selects the shared key, supporting shared-secret rotation::

        {"kid": "<key id>", "nonce": "<unpadded base64url>",
         "claims": {...}, "mac": "<64 lowercase hex>"}

    The document must carry exactly these four keys. ``mac`` is the
    lowercase hex HMAC-SHA256 over the domain-separated canonical JSON
    (sorted keys, compact separators)::

        "attested-nonce-json-v2" || NUL || {"claims":...,"kid":...,"nonce":...}

    The MAC key is derived per workload from the selected shared key::

        mac_key = HMAC-SHA256(shared_key, tenant_id + ":" + workload_id)

    Shared keys come from ``PROOF_RELEASE_ATTESTED_NONCE_KEYS`` (see
    :func:`load_attested_nonce_keys`); there is no fallback. The key is
    selected by ``kid`` only — an unknown kid is an ordinary rejection,
    while a missing or invalid configuration raises
    :class:`VerifierPluginError` (the service answers 500 and leaves the
    evidence unsettled for retry). Missing/extra keys, type errors, a
    nonce that is not the bound challenge's nonce, and MAC mismatches all
    yield a plain rejected result with no reasons attached.
    """

    format_name = ATTESTED_NONCE_JSON_V2

    def __init__(
        self,
        keys: Dict[str, bytes] | None = None,
        *,
        keys_provider: Callable[[], Dict[str, bytes]] | None = None,
    ) -> None:
        self._keys = keys
        self._keys_provider = keys_provider

    def _current_keys(self) -> Dict[str, bytes]:
        if self._keys is not None:
            return self._keys
        if self._keys_provider is not None:
            return self._keys_provider()
        return load_attested_nonce_keys()

    def verify(self, context: VerificationContext) -> VerificationResult:
        try:
            document = json.loads(context.evidence)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _reject()
        if not isinstance(document, dict):
            return _reject()
        if set(document) != {"kid", "nonce", "claims", "mac"}:
            return _reject()

        kid = document["kid"]
        nonce = document["nonce"]
        claims = document["claims"]
        mac_hex = document["mac"]
        if not isinstance(kid, str) or _KID_PATTERN.fullmatch(kid) is None:
            return _reject()
        if not isinstance(nonce, str) or not _is_unpadded_base64url(nonce):
            return _reject()
        if not isinstance(claims, dict):
            return _reject()
        if not isinstance(mac_hex, str) or _MAC_HEX_PATTERN.fullmatch(mac_hex) is None:
            return _reject()

        # The attested nonce must be exactly the nonce of the bound challenge.
        if not hmac.compare_digest(
            hashlib.sha256(nonce.encode("ascii")).hexdigest(),
            context.challenge.nonce_digest,
        ):
            return _reject()

        key = self._current_keys().get(kid)
        if key is None:
            # Key selection is by kid only; an unknown kid is an ordinary
            # rejection, never a configuration failure.
            return _reject()
        if isinstance(key, str):
            key = key.encode("utf-8")

        mac_key = hmac.new(
            key,
            f"{context.tenant_id}:{context.workload_id}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
        signed_payload = _V2_MAC_PREFIX + json.dumps(
            {"claims": claims, "kid": kid, "nonce": nonce},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_mac = hmac.new(mac_key, signed_payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_mac, mac_hex):
            return _reject()

        return VerificationResult(accepted=True)


#: Upper bound on the certificate chain length accepted by the X.509
#: verifier, guarding against resource-exhaustion via absurd chains.
MAX_CERTIFICATE_CHAIN_LENGTH = 8


def _decode_unpadded_base64url(value: str) -> bytes | None:
    if not _is_unpadded_base64url(value):
        return None
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _verify_key_signature(public_key, signature: bytes, data: bytes) -> bool:
    """Verify a SHA-256 (or Ed25519) signature; never raises."""
    try:
        if isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(signature, data, padding.PKCS1v15(), hashes.SHA256())
        elif isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
        elif isinstance(public_key, ed25519.Ed25519PublicKey):
            public_key.verify(signature, data)
        else:
            return False
    except Exception:
        # Malformed signatures can raise ValueError and friends in
        # addition to InvalidSignature; any failure is a plain reject.
        return False
    return True


def _verify_certificate_signature(
    certificate: x509.Certificate, issuer_public_key
) -> bool:
    """Verify a certificate's signature against its issuer's key."""
    try:
        if isinstance(issuer_public_key, rsa.RSAPublicKey):
            issuer_public_key.verify(
                certificate.signature,
                certificate.tbs_certificate_bytes,
                padding.PKCS1v15(),
                certificate.signature_hash_algorithm,
            )
        elif isinstance(issuer_public_key, ec.EllipticCurvePublicKey):
            issuer_public_key.verify(
                certificate.signature,
                certificate.tbs_certificate_bytes,
                ec.ECDSA(certificate.signature_hash_algorithm),
            )
        elif isinstance(issuer_public_key, ed25519.Ed25519PublicKey):
            issuer_public_key.verify(
                certificate.signature, certificate.tbs_certificate_bytes
            )
        else:
            return False
    except Exception:
        return False
    return True


def _verify_crl_signature(crl: x509.CertificateRevocationList, issuer_public_key) -> bool:
    """Verify a CRL's signature against its issuer's key; never raises."""
    try:
        if isinstance(issuer_public_key, rsa.RSAPublicKey):
            issuer_public_key.verify(
                crl.signature,
                crl.tbs_certlist_bytes,
                padding.PKCS1v15(),
                crl.signature_hash_algorithm,
            )
        elif isinstance(issuer_public_key, ec.EllipticCurvePublicKey):
            issuer_public_key.verify(
                crl.signature,
                crl.tbs_certlist_bytes,
                ec.ECDSA(crl.signature_hash_algorithm),
            )
        elif isinstance(issuer_public_key, ed25519.Ed25519PublicKey):
            issuer_public_key.verify(crl.signature, crl.tbs_certlist_bytes)
        else:
            return False
    except Exception:
        # Malformed CRL signatures can raise ValueError and friends in
        # addition to InvalidSignature; any failure is a plain failure.
        return False
    return True


def _load_crl(pem: str) -> x509.CertificateRevocationList | None:
    try:
        return x509.load_pem_x509_crl(pem.encode("utf-8"))
    except Exception:
        return None


def _load_certificate(pem: str) -> x509.Certificate | None:
    try:
        return x509.load_pem_x509_certificate(pem.encode("utf-8"))
    except Exception:
        return None


#: Longest accepted span between an embedded OCSP SingleResponse's
#: thisUpdate and its nextUpdate. RFC 6960 §4.2.2.1 recommends a value
#: well under a week; the contract caps it at exactly seven days.
OCSP_MAX_RESPONSE_WINDOW = timedelta(days=7)


def _der_tlv(data: bytes, index: int = 0) -> tuple[int, bytes, int]:
    """Parse one DER TLV at ``index`` -> (tag, content, next_index)."""
    tag = data[index]
    index += 1
    length = data[index]
    index += 1
    if length & 0x80:
        count = length & 0x7F
        length = int.from_bytes(data[index : index + count], "big")
        index += count
    return tag, data[index : index + length], index + length


def _subject_public_key_bits(certificate: x509.Certificate) -> bytes:
    """Return the content of a certificate's subjectPublicKey BIT STRING.

    This is the value OCSP CertID issuerKeyHash and the by-key
    ResponderID are computed over (the BIT STRING payload without its
    tag, length and unused-bits prefix byte).
    """
    spki = certificate.public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )
    _, sequence, _ = _der_tlv(spki)
    # SubjectPublicKeyInfo ::= SEQUENCE { algorithm AlgorithmIdentifier,
    # subjectPublicKey BIT STRING }.
    _, _, after_algorithm = _der_tlv(sequence, 0)
    _, bit_string, _ = _der_tlv(sequence, after_algorithm)
    # BIT STRING's first content byte records the number of unused bits.
    return bit_string[1:]


def _ocsp_time_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _ocsp_this_update(single: ocsp.OCSPSingleResponse) -> datetime:
    aware = getattr(single, "this_update_utc", None)
    return _ocsp_time_utc(aware if aware is not None else single.this_update)


def _ocsp_next_update(single: ocsp.OCSPSingleResponse) -> datetime | None:
    # Newer cryptography exposes timezone-aware accessors (the naive
    # properties are deprecated); a None from the aware accessor means
    # nextUpdate is genuinely absent. Fall back to the naive property
    # only on older libraries that lack the aware accessor.
    if hasattr(single, "next_update_utc"):
        aware = single.next_update_utc
        return None if aware is None else _ocsp_time_utc(aware)
    legacy = single.next_update
    return None if legacy is None else _ocsp_time_utc(legacy)


def _ocsp_produced_at(response: ocsp.OCSPResponse) -> datetime:
    aware = getattr(response, "produced_at_utc", None)
    return _ocsp_time_utc(aware if aware is not None else response.produced_at)


def _ocsp_certid_matches(
    single: ocsp.OCSPSingleResponse,
    target: x509.Certificate,
    issuing_ca: x509.Certificate,
) -> bool:
    """Match a SingleResponse CertID to ``target`` as issued by ``issuing_ca``.

    The serial number must equal the target's serial and both CertID
    hashes must equal, under the CertID's own hash algorithm, the hash
    of the issuer's distinguished name and of its subjectPublicKey BIT
    STRING content. Never raises.
    """
    if single.serial_number != target.serial_number:
        return False
    try:
        name_hasher = hashes.Hash(single.hash_algorithm)
        name_hasher.update(issuing_ca.subject.public_bytes())
        expected_name_hash = name_hasher.finalize()

        key_hasher = hashes.Hash(single.hash_algorithm)
        key_hasher.update(_subject_public_key_bits(issuing_ca))
        expected_key_hash = key_hasher.finalize()
    except Exception:
        # An unsupported/unsatisfiable CertID hash algorithm or an
        # unexpected public-key encoding is a failed match.
        return False
    return hmac.compare_digest(
        expected_name_hash, single.issuer_name_hash
    ) and hmac.compare_digest(expected_key_hash, single.issuer_key_hash)


def _ocsp_responder_matches(
    response: ocsp.OCSPResponse, responder_certificate: x509.Certificate
) -> bool:
    """Match the response's ResponderID against the supplied certificate."""
    if response.responder_name is not None:
        return response.responder_name == responder_certificate.subject
    responder_key_hash = response.responder_key_hash
    if responder_key_hash is None:
        return False
    # RFC 6960: a by-key ResponderID is always the SHA-1 hash of the
    # responder's subjectPublicKey BIT STRING content.
    expected = hashlib.sha1(
        _subject_public_key_bits(responder_certificate)
    ).digest()
    return hmac.compare_digest(expected, responder_key_hash)


def _verify_ocsp_signature(
    responder_public_key, response: ocsp.OCSPResponse
) -> bool:
    """Verify an OCSP response signature; never raises."""
    try:
        if isinstance(responder_public_key, rsa.RSAPublicKey):
            responder_public_key.verify(
                response.signature,
                response.tbs_response_bytes,
                padding.PKCS1v15(),
                response.signature_hash_algorithm,
            )
        elif isinstance(responder_public_key, ec.EllipticCurvePublicKey):
            responder_public_key.verify(
                response.signature,
                response.tbs_response_bytes,
                ec.ECDSA(response.signature_hash_algorithm),
            )
        elif isinstance(responder_public_key, ed25519.Ed25519PublicKey):
            responder_public_key.verify(
                response.signature, response.tbs_response_bytes
            )
        else:
            return False
    except Exception:
        return False
    return True


def _valid_delegated_ocsp_responder(
    responder_certificate: x509.Certificate,
    issuing_ca: x509.Certificate,
    now: datetime,
) -> bool:
    """Validate a delegated OCSP responder certificate against its CA.

    The responder certificate must be within its validity period, carry
    a Basic Constraints extension with cA FALSE, carry the id-kp-OCSPSigning
    extended key usage, and be issued and signed by the target's direct
    issuing CA. Never raises.
    """
    if (
        responder_certificate.not_valid_before_utc > now
        or responder_certificate.not_valid_after_utc < now
    ):
        return False
    try:
        basic = responder_certificate.extensions.get_extension_for_class(
            x509.BasicConstraints
        )
    except x509.ExtensionNotFound:
        return False
    except Exception:
        return False
    if basic.value.ca:
        return False
    try:
        eku = responder_certificate.extensions.get_extension_for_class(
            x509.ExtendedKeyUsage
        )
    except x509.ExtensionNotFound:
        return False
    except Exception:
        return False
    if ExtendedKeyUsageOID.OCSP_SIGNING not in eku.value:
        return False
    if responder_certificate.issuer != issuing_ca.subject:
        return False
    if not _verify_certificate_signature(
        responder_certificate, issuing_ca.public_key()
    ):
        return False
    return True


def _parse_embedded_ocsp_entries(
    raw_entries: list,
) -> list[tuple[bytes, x509.Certificate]] | None:
    """Structurally validate and decode the evidence's ocsp_responses.

    Each entry must be an object with non-empty unpadded-base64url DER in
    ``response`` and a parseable PEM certificate in
    ``responder_certificate``. Returns the decoded pairs, or ``None`` on
    the first malformed entry.
    """
    parsed: list[tuple[bytes, x509.Certificate]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            return None
        response_b64 = entry.get("response")
        responder_pem = entry.get("responder_certificate")
        if not isinstance(response_b64, str) or not response_b64:
            return None
        if not isinstance(responder_pem, str) or not responder_pem:
            return None
        response_der = _decode_unpadded_base64url(response_b64)
        if response_der is None:
            return None
        responder_certificate = _load_certificate(responder_pem)
        if responder_certificate is None:
            return None
        parsed.append((response_der, responder_certificate))
    return parsed


def _validate_one_ocsp_entry(
    response: ocsp.OCSPResponse,
    responder_certificate: x509.Certificate,
    targets: list[x509.Certificate],
    certificates: list[x509.Certificate],
    covered: set[int],
    now: datetime,
) -> bool:
    """Validate a single parsed OCSP response against the chain.

    Returns ``True`` when it addresses a previously uncovered non-root
    certificate and passes every structural, freshness and signature
    rule, recording the covered index. Any failure — including a value
    error raised deep inside the DER parser while reading attacker
    controlled fields — returns ``False`` rather than propagating.
    """
    if response.response_status != ocsp.OCSPResponseStatus.SUCCESSFUL:
        return False
    single_responses = list(response.responses)
    if len(single_responses) != 1:
        return False
    single = single_responses[0]

    # The entry must identify exactly one non-root chain certificate.
    match_index = -1
    for index, target in enumerate(targets):
        if _ocsp_certid_matches(single, target, certificates[index + 1]):
            if match_index != -1:
                return False
            match_index = index
    if match_index == -1 or match_index in covered:
        # Points at no chain certificate, or duplicates another entry.
        return False
    issuing_ca = certificates[match_index + 1]

    # Freshness window relative to the single verification instant.
    this_update = _ocsp_this_update(single)
    next_update = _ocsp_next_update(single)
    produced_at = _ocsp_produced_at(response)
    if next_update is None:
        return False
    if this_update > now:
        return False
    if produced_at < this_update or produced_at > now:
        return False
    if next_update <= now:
        return False
    if next_update > this_update + OCSP_MAX_RESPONSE_WINDOW:
        return False

    if single.certificate_status != ocsp.OCSPCertStatus.GOOD:
        # revoked and unknown both settle the proof as rejected.
        return False

    if not _ocsp_responder_matches(response, responder_certificate):
        return False

    # The supplied signer certificate is either the target's direct
    # issuing CA itself or a delegated responder issued by that CA.
    if responder_certificate.public_bytes(Encoding.DER) != issuing_ca.public_bytes(
        Encoding.DER
    ):
        if not _valid_delegated_ocsp_responder(
            responder_certificate, issuing_ca, now
        ):
            return False

    if not _verify_ocsp_signature(responder_certificate.public_key(), response):
        return False

    covered.add(match_index)
    return True


def _verify_embedded_ocsp_responses(
    parsed_entries: list[tuple[bytes, x509.Certificate]],
    certificates: list[x509.Certificate],
    now: datetime,
) -> bool:
    """Validate decoded embedded OCSP responses against the parsed chain.

    Every non-root chain certificate must be covered by exactly one
    entry; each response must be a successful, single-SingleResponse
    GOOD response whose CertID identifies that target, whose time window
    is current relative to ``now``, and whose signature verifies under
    the direct issuing CA or a valid delegated responder certificate.
    Never raises and never exposes exception detail.
    """
    targets = certificates[:-1]
    covered: set[int] = set()
    for response_der, responder_certificate in parsed_entries:
        try:
            response = ocsp.load_der_ocsp_response(response_der)
        except Exception:
            return False
        try:
            valid = _validate_one_ocsp_entry(
                response,
                responder_certificate,
                targets,
                certificates,
                covered,
                now,
            )
        except Exception:
            # Malformed-but-parseable DER can still raise when individual
            # fields are accessed; every such failure is a plain reject.
            return False
        if not valid:
            return False

    # Exactly one entry per non-root certificate: none missing, none extra.
    return len(covered) == len(targets)


class X509AttestedNonceJSONVerifier(Verifier):
    """Built-in verifier for the ``x509-attested-nonce-json`` format.

    The evidence is a JSON object cryptographically bound to the challenge
    nonce and anchored to a configured trust root::

        {
          "nonce": "<unpadded base64url>",
          "claims": {...},
          "certificate_chain": ["<leaf PEM>", ..., "<root PEM>"],
          "signature": "<unpadded base64url>",
          "ocsp_responses": [
            {"response": "<unpadded base64url DER OCSPResponse>",
             "responder_certificate": "<PEM X.509>"}
          ]
        }

    ``certificate_chain`` is a non-empty list of PEM certificates ordered
    from leaf to root. ``signature`` is produced by the leaf private key
    over the canonical JSON serialization (sorted keys, compact separators)
    of ``{"claims": ..., "nonce": ...}`` — RSA PKCS#1 v1.5 with SHA-256,
    ECDSA with SHA-256, or Ed25519, depending on the leaf key type.

    The optional ``ocsp_responses`` array carries embedded OCSP
    revocation evidence. When absent the chain is accepted without OCSP;
    when present it must be non-empty and stand in one-to-one
    correspondence with the chain's non-root certificates — every
    non-root certificate covered exactly once, with no entry pointing at
    a different certificate. Each response must have status successful
    and exactly one SingleResponse whose CertID (issuer name hash,
    issuer key hash and serial, under the CertID hash algorithm) matches
    the target, whose certificate status is good, whose thisUpdate,
    producedAt and nextUpdate form a current window at the verification
    instant (thisUpdate ≤ now; thisUpdate ≤ producedAt ≤ now;
    now < nextUpdate ≤ thisUpdate + seven days), and whose signature
    verifies under either the target's direct issuing CA or a delegated
    responder certificate that CA has signed. A delegated responder must
    carry cA FALSE basic constraints and the id-kp-OCSPSigning extended
    key usage and pass path and validity checks. Any revoked or unknown
    status or any coverage, parsing, signature, path or time failure
    settles the proof as rejected.

    Verification checks, in order: well-formed document, unpadded base64url
    nonce equal to the challenge nonce (compared through the stored digest),
    parseable chain, the chain root byte-identical to a trust root
    configured for exactly this tenant and workload, every certificate
    within its validity period, every issuer a CA certificate, each
    certificate signed by the next in the chain, the embedded OCSP checks
    above when ``ocsp_responses`` is present, and finally the leaf
    signature over the canonical payload. Any failure yields a plain
    rejected result; certificate material, OCSP material, evidence content
    and exception details are never persisted, logged, or returned.
    """

    format_name = X509_ATTESTED_NONCE_JSON

    def verify(self, context: VerificationContext) -> VerificationResult:
        try:
            document = json.loads(context.evidence)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _reject()
        if not isinstance(document, dict):
            return _reject()

        nonce = document.get("nonce")
        claims = document.get("claims", {})
        chain_pems = document.get("certificate_chain")
        signature_b64 = document.get("signature")
        ocsp_entries = document.get("ocsp_responses")
        if not isinstance(nonce, str) or not nonce:
            return _reject()
        if "claims" in document and not isinstance(claims, dict):
            return _reject()
        if (
            not isinstance(chain_pems, list)
            or not chain_pems
            or len(chain_pems) > MAX_CERTIFICATE_CHAIN_LENGTH
            or any(not isinstance(pem, str) or not pem for pem in chain_pems)
        ):
            return _reject()
        if not isinstance(signature_b64, str) or not signature_b64:
            return _reject()
        if "ocsp_responses" in document:
            # When supplied the array must be a non-empty list and carry
            # exactly one entry per non-root chain certificate. An
            # explicit null or any other type is a format failure.
            if (
                not isinstance(ocsp_entries, list)
                or not ocsp_entries
                or len(ocsp_entries) != len(chain_pems) - 1
            ):
                return _reject()

        if not _is_unpadded_base64url(nonce):
            return _reject()

        # The attested nonce must be exactly the nonce of the bound challenge.
        if not hmac.compare_digest(
            hashlib.sha256(nonce.encode("ascii")).hexdigest(),
            context.challenge.nonce_digest,
        ):
            return _reject()

        signature = _decode_unpadded_base64url(signature_b64)
        if signature is None:
            return _reject()

        certificates = [_load_certificate(pem) for pem in chain_pems]
        if any(certificate is None for certificate in certificates):
            return _reject()

        parsed_ocsp_entries: list[tuple[bytes, x509.Certificate]] | None = None
        if "ocsp_responses" in document:
            parsed_ocsp_entries = _parse_embedded_ocsp_entries(ocsp_entries)
            if parsed_ocsp_entries is None:
                return _reject()

        # The chain root must be byte-identical to a trust root configured
        # for exactly this tenant and workload.
        trusted_ders = set()
        for root_pem in context.trust_roots:
            trusted = _load_certificate(root_pem)
            if trusted is not None:
                trusted_ders.add(trusted.public_bytes(Encoding.DER))
        if not trusted_ders:
            return _reject()
        if certificates[-1].public_bytes(Encoding.DER) not in trusted_ders:
            return _reject()

        now = datetime.now(timezone.utc)
        for index, certificate in enumerate(certificates):
            if (
                certificate.not_valid_before_utc > now
                or certificate.not_valid_after_utc < now
            ):
                return _reject()
            if index > 0:
                # Every issuer in the chain must be a CA certificate.
                try:
                    basic = certificate.extensions.get_extension_for_class(
                        x509.BasicConstraints
                    )
                except x509.ExtensionNotFound:
                    return _reject()
                if not basic.value.ca:
                    return _reject()

        # Each certificate must be issued and signed by the next in the chain.
        for subject, issuer in zip(certificates, certificates[1:]):
            if subject.issuer != issuer.subject:
                return _reject()
            if not _verify_certificate_signature(subject, issuer.public_key()):
                return _reject()

        # Embedded OCSP revocation evidence, when supplied, is checked
        # against the now-validated chain before the leaf signature; only
        # all-good, current, correctly covered and trust-anchored
        # responses let verification continue.
        if parsed_ocsp_entries is not None:
            if not _verify_embedded_ocsp_responses(
                parsed_ocsp_entries, certificates, now
            ):
                return _reject()

        signed_payload = json.dumps(
            {"claims": claims, "nonce": nonce},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if not _verify_key_signature(
            certificates[0].public_key(), signature, signed_payload
        ):
            return _reject()

        return VerificationResult(accepted=True)


#: Process-wide registry used by the service unless one is supplied.
default_registry = VerifierRegistry()
default_registry.register(AttestedNonceJSONVerifier())
default_registry.register(AttestedNonceJSONV2Verifier())
default_registry.register(X509AttestedNonceJSONVerifier())


def register_verifier(verifier: Verifier) -> None:
    """Register a verifier on the process-wide default registry."""
    default_registry.register(verifier)


def unregister_verifier(format_name: str) -> None:
    """Remove a verifier from the process-wide default registry."""
    default_registry.unregister(format_name)


def get_verifier(format_name: str) -> Verifier | None:
    """Look up a verifier on the process-wide default registry."""
    return default_registry.get(format_name)


# ---------------------------------------------------------------------------
# X.509 v2 CRL registration support
# ---------------------------------------------------------------------------


class CrlValidationError(ValueError):
    """A submitted CRL failed a structural or trust validation rule.

    The message is a fixed, service-defined category string — never the CRL
    body, certificate material, or an exception detail — so the API layer
    can safely turn it into a sanitized 422.
    """


@dataclass(frozen=True)
class CrlRevokedEntry:
    """One validated revoked-certificate entry of a parsed CRL."""

    serial_number: int
    revocation_date: datetime


@dataclass(frozen=True)
class ValidatedCrl:
    """The intrinsically valid content of a submitted CRL.

    Produced by :func:`parse_crl` before the trust root is read: every
    check that depends only on the CRL body (PEM/ASN.1, v2, CRLNumber,
    nextUpdate, the time window and the revoked-entry serials) has already
    passed. The trust-dependent checks (issuer DN, signature) run later in
    :func:`validate_crl_against_root`, so the fixed error order
    (intrinsic 422 -> unknown root 404 -> issuer/signature 422 -> 409)
    holds.
    """

    crl: x509.CertificateRevocationList
    crl_number: int
    this_update: datetime
    next_update: datetime
    crl_sha256: str
    entries: tuple[CrlRevokedEntry, ...]
    revoked_count: int


@dataclass(frozen=True)
class ParsedCrl:
    """A CRL that has also passed the trust-root issuer/signature checks."""

    validated: ValidatedCrl
    issuer_dn: str

    @property
    def crl(self) -> x509.CertificateRevocationList:
        return self.validated.crl

    @property
    def crl_number(self) -> int:
        return self.validated.crl_number

    @property
    def this_update(self) -> datetime:
        return self.validated.this_update

    @property
    def next_update(self) -> datetime:
        return self.validated.next_update

    @property
    def crl_sha256(self) -> str:
        return self.validated.crl_sha256

    @property
    def entries(self) -> tuple[CrlRevokedEntry, ...]:
        return self.validated.entries

    @property
    def revoked_count(self) -> int:
        return self.validated.revoked_count


def load_crl(pem_text: str) -> x509.CertificateRevocationList | None:
    """Parse a PEM CRL, returning None on any PEM/ASN.1 failure."""
    if not isinstance(pem_text, str):
        return None
    try:
        return x509.load_pem_x509_crl(pem_text.encode("utf-8"))
    except Exception:
        return None


def verify_crl_signature(
    crl: x509.CertificateRevocationList, issuer_public_key
) -> bool:
    """Verify a CRL signature against its issuer's public key; never raises."""
    return _verify_crl_signature(crl, issuer_public_key)


def _as_utc(value: datetime) -> datetime:
    # Prefer the timezone-aware accessors on newer cryptography; the
    # legacy naive properties are fixed to UTC by RFC 5280, so a naive
    # value is attached to UTC as a fallback for older libraries.
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _crl_this_update(crl: x509.CertificateRevocationList) -> datetime:
    aware = getattr(crl, "last_update_utc", None)
    return _as_utc(aware if aware is not None else crl.last_update)


def _crl_next_update(crl: x509.CertificateRevocationList) -> datetime | None:
    # Newer cryptography exposes timezone-aware accessors (the naive
    # properties are deprecated); fall back to the naive, RFC-5280-UTC
    # value on older libraries.
    if hasattr(crl, "next_update_utc"):
        aware = crl.next_update_utc
        return None if aware is None else _as_utc(aware)
    return None if crl.next_update is None else _as_utc(crl.next_update)


def _revocation_date_utc(revoked) -> datetime:
    aware = getattr(revoked, "revocation_date_utc", None)
    return _as_utc(aware if aware is not None else revoked.revocation_date)


def parse_crl(pem_text: str, *, now: datetime) -> ValidatedCrl:
    """Parse and intrinsically validate a submitted CRL body.

    Enforces, in fixed order: PEM/ASN.1 parse as an X.509 v2 CRL; presence
    of CRLNumber and nextUpdate; thisUpdate not after ``now``; nextUpdate
    strictly later than both thisUpdate and ``now``; and every revoked
    entry carrying a non-negative, non-duplicate serial. Raises
    :class:`CrlValidationError` (sanitized category only) on the first
    failed rule. The trust root is never accessed here.
    """
    crl = load_crl(pem_text)
    if crl is None:
        raise CrlValidationError("crl_pem is not a valid PEM X.509 CRL")
    # X.509 v2 is the only CRL version permitted to carry extensions, and
    # the contract requires the v2-only CRLNumber extension; its presence
    # is therefore the v2 marker (the library exposes no version field on
    # a parsed CRL). A v1 CRL without it is rejected below.
    try:
        number_ext = crl.extensions.get_extension_for_class(x509.CRLNumber)
    except x509.ExtensionNotFound:
        raise CrlValidationError(
            "crl_pem must be an X.509 v2 CRL with a CRLNumber extension"
        )
    crl_number = number_ext.value.crl_number
    # RFC 5280 bounds CRLNumber to a non-negative integer.
    if not isinstance(crl_number, int) or crl_number < 0:
        raise CrlValidationError("invalid CRLNumber")

    next_update = _crl_next_update(crl)
    if next_update is None:
        raise CrlValidationError("crl_pem must include nextUpdate")
    this_update = _crl_this_update(crl)

    if this_update > now:
        raise CrlValidationError("thisUpdate must not be in the future")
    if next_update <= this_update:
        raise CrlValidationError("nextUpdate must be later than thisUpdate")
    if next_update <= now:
        raise CrlValidationError("nextUpdate must be later than the receipt time")

    entries: list[CrlRevokedEntry] = []
    seen_serials: set[int] = set()
    for revoked in crl:
        serial = revoked.serial_number
        # RFC 5280 serials are non-negative integers; a negative value or
        # another non-canonical encoding is a malformed CRL.
        if not isinstance(serial, int) or serial < 0:
            raise CrlValidationError("invalid revoked certificate serial number")
        if serial in seen_serials:
            raise CrlValidationError("duplicate revoked certificate serial number")
        seen_serials.add(serial)
        entries.append(
            CrlRevokedEntry(
                serial_number=serial,
                revocation_date=_revocation_date_utc(revoked),
            )
        )

    der = crl.public_bytes(Encoding.DER)
    revoked_count = sum(1 for entry in entries if entry.revocation_date <= now)
    return ValidatedCrl(
        crl=crl,
        crl_number=crl_number,
        this_update=this_update,
        next_update=next_update,
        crl_sha256=hashlib.sha256(der).hexdigest(),
        entries=tuple(entries),
        revoked_count=revoked_count,
    )


def validate_crl_against_root(
    validated: ValidatedCrl,
    *,
    issuer_certificate: x509.Certificate,
) -> ParsedCrl:
    """Validate the trust-dependent CRL rules against one trust root.

    The CRL issuer DN must equal the trust root's subject and the CRL
    signature must verify under the trust root's public key; either
    mismatch is a sanitized :class:`CrlValidationError` (422). Runs only
    after the trust root has been looked up in the request's exact scope.
    """
    if validated.crl.issuer != issuer_certificate.subject:
        raise CrlValidationError("CRL issuer does not match the trust root subject")
    if not _verify_crl_signature(validated.crl, issuer_certificate.public_key()):
        raise CrlValidationError("CRL signature is not valid for the trust root")
    return ParsedCrl(
        validated=validated,
        issuer_dn=issuer_certificate.subject.rfc4514_string(),
    )
