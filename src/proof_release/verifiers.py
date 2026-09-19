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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.serialization import Encoding

__all__ = [
    "ChallengeContext",
    "VerificationContext",
    "VerificationResult",
    "Verifier",
    "VerifierRegistry",
    "AttestedNonceJSONVerifier",
    "X509AttestedNonceJSONVerifier",
    "default_registry",
    "register_verifier",
    "unregister_verifier",
    "get_verifier",
]

#: Built-in format identifier for the reference verifier.
ATTESTED_NONCE_JSON = "attested-nonce-json"

#: Built-in format identifier for the X.509 certificate-chain verifier.
X509_ATTESTED_NONCE_JSON = "x509-attested-nonce-json"

#: Environment variable holding the MAC secret for the built-in verifier.
ATTESTED_NONCE_SECRET_ENV = "PROOF_RELEASE_ATTESTED_NONCE_SECRET"

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


class VerifierRegistry:
    """Maps evidence format names to verifier instances."""

    def __init__(self) -> None:
        self._verifiers: Dict[str, Verifier] = {}

    def register(self, verifier: Verifier) -> None:
        """Register (or replace) a verifier for its format name."""
        name = getattr(verifier, "format_name", "")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("verifier format_name must be a non-empty string")
        self._verifiers[name] = verifier

    def unregister(self, format_name: str) -> None:
        self._verifiers.pop(format_name, None)

    def get(self, format_name: str) -> Verifier | None:
        return self._verifiers.get(format_name)

    def __contains__(self, format_name: object) -> bool:
        return format_name in self._verifiers


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


def _load_certificate(pem: str) -> x509.Certificate | None:
    try:
        return x509.load_pem_x509_certificate(pem.encode("utf-8"))
    except Exception:
        return None


class X509AttestedNonceJSONVerifier(Verifier):
    """Built-in verifier for the ``x509-attested-nonce-json`` format.

    The evidence is a JSON object cryptographically bound to the challenge
    nonce and anchored to a configured trust root::

        {
          "nonce": "<unpadded base64url>",
          "claims": {...},
          "certificate_chain": ["<leaf PEM>", ..., "<root PEM>"],
          "signature": "<unpadded base64url>"
        }

    ``certificate_chain`` is a non-empty list of PEM certificates ordered
    from leaf to root. ``signature`` is produced by the leaf private key
    over the canonical JSON serialization (sorted keys, compact separators)
    of ``{"claims": ..., "nonce": ...}`` — RSA PKCS#1 v1.5 with SHA-256,
    ECDSA with SHA-256, or Ed25519, depending on the leaf key type.

    Verification checks, in order: well-formed document, unpadded base64url
    nonce equal to the challenge nonce (compared through the stored digest),
    parseable chain, the chain root byte-identical to a trust root
    configured for exactly this tenant and workload, every certificate
    within its validity period, every issuer a CA certificate, each
    certificate signed by the next in the chain, and finally the leaf
    signature over the canonical payload. Any failure yields a plain
    rejected result; certificate material, evidence content and exception
    details are never persisted, logged, or returned.
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
