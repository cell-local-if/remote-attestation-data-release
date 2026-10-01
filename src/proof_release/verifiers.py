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
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509 import ocsp
from cryptography.x509.oid import ExtendedKeyUsageOID

__all__ = [
    "ChallengeContext",
    "VerificationContext",
    "VerificationResult",
    "Verifier",
    "VerifierRegistry",
    "AttestedNonceJSONVerifier",
    "X509AttestedNonceJSONVerifier",
    "CrlValidationError",
    "ParsedCrl",
    "ValidatedCrl",
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


#: Maximum accepted span between an OCSP SingleResponse's thisUpdate and
#: its nextUpdate; a response claiming freshness for longer is rejected.
MAX_OCSP_NEXT_UPDATE_WINDOW = timedelta(days=7)


def _verify_ocsp_signature(response: ocsp.OCSPResponse, public_key) -> bool:
    """Verify an OCSP response signature; never raises."""
    try:
        if isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(
                response.signature,
                response.tbs_response_bytes,
                padding.PKCS1v15(),
                response.signature_hash_algorithm,
            )
        elif isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(
                response.signature,
                response.tbs_response_bytes,
                ec.ECDSA(response.signature_hash_algorithm),
            )
        elif isinstance(public_key, ed25519.Ed25519PublicKey):
            public_key.verify(response.signature, response.tbs_response_bytes)
        else:
            return False
    except Exception:
        # Malformed signatures or an unsupported signature algorithm raise
        # ValueError and friends in addition to InvalidSignature; any
        # failure is a plain rejection.
        return False
    return True


def _ocsp_certid_matches(
    single: ocsp.OCSPSingleResponse,
    target: x509.Certificate,
    issuer: x509.Certificate,
) -> bool:
    """Whether a SingleResponse's CertID names ``target``.

    The issuer name hash, issuer key hash and serial must all equal the
    values computed for ``target`` against its chain issuer, using the
    CertID's own hash algorithm. Never raises.
    """
    try:
        certid = (
            ocsp.OCSPRequestBuilder()
            .add_certificate(target, issuer, single.hash_algorithm)
            .build()
        )
    except Exception:
        return False
    return (
        certid.issuer_name_hash == single.issuer_name_hash
        and certid.issuer_key_hash == single.issuer_key_hash
        and certid.serial_number == single.serial_number
    )


def _ocsp_responder_matches_certificate(
    response: ocsp.OCSPResponse, certificate: x509.Certificate
) -> bool:
    """Whether the response's ResponderID names ``certificate``.

    Accepts either the by-name form (subject DN equality) or the by-key
    form (SHA-1 of the subject public key BIT STRING, equal to the
    certificate's subject key identifier value). Never raises.
    """
    try:
        if response.responder_name is not None:
            return response.responder_name == certificate.subject
        key_hash = response.responder_key_hash
        if key_hash is not None:
            ski = x509.SubjectKeyIdentifier.from_public_key(
                certificate.public_key()
            )
            return ski.digest == key_hash
    except Exception:
        return False
    return False


def _valid_delegated_ocsp_responder(
    responder: x509.Certificate,
    issuer_certificate: x509.Certificate,
    now: datetime,
) -> bool:
    """Validate a non-CA delegated OCSP responder certificate.

    The responder must be within its validity period, must not assert
    ``cA`` in basic constraints (RFC 6960 forbids CA delegated responders
    when the extension is present), must carry the OCSP Signing extended
    key usage, and must be directly issued and signed by the target
    certificate's CA. Never raises.
    """
    try:
        if (
            responder.not_valid_before_utc > now
            or responder.not_valid_after_utc < now
        ):
            return False
        try:
            basic = responder.extensions.get_extension_for_class(
                x509.BasicConstraints
            )
        except x509.ExtensionNotFound:
            pass
        else:
            if basic.value.ca:
                return False
        try:
            eku = responder.extensions.get_extension_for_class(
                x509.ExtendedKeyUsage
            ).value
        except x509.ExtensionNotFound:
            return False
        if ExtendedKeyUsageOID.OCSP_SIGNING not in eku:
            return False
        if responder.issuer != issuer_certificate.subject:
            return False
        if not _verify_certificate_signature(
            responder, issuer_certificate.public_key()
        ):
            return False
    except Exception:
        return False
    return True


def _validate_embedded_ocsp(
    document: dict, certificates: list[x509.Certificate], *, now: datetime
) -> bool:
    """Validate the optional inline ``ocsp_responses`` array.

    Returns ``True`` when the field is absent (existing rules apply) or
    when every non-root chain certificate has exactly one good, fresh,
    correctly signed OCSP response covering it; ``False`` on any
    structural, coverage, parse, status, time, path or signature failure.
    All inputs stay on the stack and no exception text is propagated.
    """
    if "ocsp_responses" not in document:
        return True
    entries = document["ocsp_responses"]
    if not isinstance(entries, list) or not entries:
        return False
    # One response per non-root certificate, no more, no fewer.
    targets = certificates[:-1]
    if len(entries) != len(targets):
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        response_b64 = entry.get("response")
        responder_pem = entry.get("responder_certificate")
        if not isinstance(response_b64, str) or not response_b64:
            return False
        if not isinstance(responder_pem, str) or not responder_pem:
            return False
        if not _is_unpadded_base64url(response_b64):
            return False

    matched_indexes: set[int] = set()
    for entry in entries:
        try:
            response_der = _decode_unpadded_base64url(entry["response"])
            response = ocsp.load_der_ocsp_response(response_der)
            responder = _load_certificate(entry["responder_certificate"])
            if responder is None:
                return False
            if response.response_status != ocsp.OCSPResponseStatus.SUCCESSFUL:
                return False
            try:
                singles = list(response.responses)
            except Exception:
                return False
            if len(singles) != 1:
                return False
            single = singles[0]

            # The CertID must name exactly one of the non-root certificates
            # in this chain; duplicates and references elsewhere are
            # rejected, which together with the count check enforces the
            # one-to-one coverage of every non-root certificate.
            matches = [
                index
                for index, target in enumerate(targets)
                if _ocsp_certid_matches(
                    single, target, certificates[index + 1]
                )
            ]
            if len(matches) != 1:
                return False
            target_index = matches[0]
            if target_index in matched_indexes:
                return False
            issuer_ca = certificates[target_index + 1]

            # Only good continues; revoked and unknown settle the proof as
            # rejected.
            if single.certificate_status != ocsp.OCSPCertStatus.GOOD:
                return False

            this_update = single.this_update_utc
            next_update = single.next_update_utc
            produced_at = response.produced_at_utc
            if next_update is None:
                return False
            if this_update > now:
                return False
            if produced_at < this_update or produced_at > now:
                return False
            if next_update <= now:
                return False
            if next_update > this_update + MAX_OCSP_NEXT_UPDATE_WINDOW:
                return False

            # The signer is either the target's direct CA (the presented
            # responder certificate is byte-identical to the chain issuer)
            # or a delegated responder certificate issued by that CA. The
            # delegate must be a distinct certificate: allowing the target
            # itself to answer would let the certificate holder vouch for
            # its own revocation status.
            responder_der = responder.public_bytes(Encoding.DER)
            if responder_der == issuer_ca.public_bytes(Encoding.DER):
                if not _ocsp_responder_matches_certificate(
                    response, issuer_ca
                ):
                    return False
            else:
                if responder_der == targets[target_index].public_bytes(
                    Encoding.DER
                ):
                    return False
                if not _valid_delegated_ocsp_responder(
                    responder, issuer_ca, now
                ):
                    return False
                if not _ocsp_responder_matches_certificate(
                    response, responder
                ):
                    return False
            if not _verify_ocsp_signature(
                response, responder.public_key()
            ):
                return False
        except Exception:
            # Every malformed-OCSP failure mode is a plain rejection;
            # nothing is raised, logged or propagated.
            return False
        matched_indexes.add(target_index)

    return matched_indexes == set(range(len(targets)))


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
    certificate signed by the next in the chain, the optional inline
    ``ocsp_responses`` array (see below), and finally the leaf signature
    over the canonical payload. Any failure yields a plain rejected
    result; certificate material, evidence content and exception details
    are never persisted, logged, or returned.

    The document may additionally carry an ``ocsp_responses`` array with
    one entry per non-root chain certificate; each entry holds an
    unpadded base64url DER ``OCSPResponse`` (``response``) and the PEM
    ``responder_certificate`` that signed it. When present the array must
    cover every non-root certificate exactly once (no missing, duplicate
    or foreign-cert entries); each response must be ``successful`` with a
    single ``good`` SingleResponse whose CertID (issuer name hash, issuer
    key hash, serial) names its target, must be fresh at verification
    time (thisUpdate not after now; producedAt between thisUpdate and
    now; nextUpdate after now and within seven days of thisUpdate), and
    must be signed either by the target's direct issuing CA or by a
    delegated responder certificate issued by that CA that carries the
    OCSP Signing extended key usage and passes the basic-constraints,
    path and validity checks. A ``revoked``/``unknown`` status or any
    other failure rejects the proof. When the field is absent all other
    rules behave exactly as before.
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

        # Optional inline OCSP revocation evidence. Absent field: existing
        # rules apply unchanged. Present: every non-root certificate must
        # have one good, fresh, correctly signed response or the proof is
        # rejected before the leaf signature is examined.
        if not _validate_embedded_ocsp(document, certificates, now=now):
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
