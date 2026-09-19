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
from datetime import datetime
from typing import Callable, Dict

__all__ = [
    "ChallengeContext",
    "VerificationContext",
    "VerificationResult",
    "Verifier",
    "VerifierRegistry",
    "AttestedNonceJSONVerifier",
    "default_registry",
    "register_verifier",
    "unregister_verifier",
    "get_verifier",
]

#: Built-in format identifier for the reference verifier.
ATTESTED_NONCE_JSON = "attested-nonce-json"

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


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of a verifier invocation.

    ``accepted`` is the *only* information the service reads from a plugin:
    it alone decides whether the record settles as ``verified`` or
    ``rejected``. There is deliberately no message, reason or detail field —
    plugin-supplied text could embed raw evidence, key material or other
    private context, and the service never accepts, truncates, transforms,
    logs, persists or returns such text. Plugins that need richer internal
    diagnostics should keep them in their own (non-shared) state.
    """

    accepted: bool


class Verifier(ABC):
    """Base class for evidence format verifier plugins.

    Subclasses set ``format_name`` and implement :meth:`verify`.
    Implementations must be safe to call from concurrent requests and must
    not raise for ordinary malformed evidence — return
    ``VerificationResult(accepted=False)`` instead. Raise only for genuine
    internal failures (the service treats a raised exception as a 500 and
    leaves no verification state behind). A verifier can only state that
    evidence passed or failed; it cannot attach reasons or any other text
    to the outcome, since nothing a plugin supplies besides the boolean is
    ever recorded or surfaced.
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


#: Singleton rejection outcome. The built-in verifier distinguishes failure
#: modes only for its own readability; the service observes the same boolean
#: rejection regardless of why the evidence failed.
_REJECTED = VerificationResult(accepted=False)


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
    digest), and valid MAC. Any failure yields a plain rejection; the
    verifier reports only the pass/fail boolean, never a reason.
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
            return _REJECTED
        if not isinstance(document, dict):
            return _REJECTED

        nonce = document.get("nonce")
        claims = document.get("claims", {})
        mac_hex = document.get("mac")
        if not isinstance(nonce, str) or not nonce:
            return _REJECTED
        if "claims" in document and not isinstance(claims, dict):
            return _REJECTED
        if not isinstance(mac_hex, str) or not mac_hex:
            return _REJECTED

        if not _is_unpadded_base64url(nonce):
            return _REJECTED

        # The attested nonce must be exactly the nonce of the bound challenge.
        if not hmac.compare_digest(
            hashlib.sha256(nonce.encode("ascii")).hexdigest(),
            context.challenge.nonce_digest,
        ):
            return _REJECTED

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
            return _REJECTED
        if not hmac.compare_digest(expected_mac, mac_hex.lower()):
            return _REJECTED

        return VerificationResult(accepted=True)


#: Process-wide registry used by the service unless one is supplied.
default_registry = VerifierRegistry()
default_registry.register(AttestedNonceJSONVerifier())


def register_verifier(verifier: Verifier) -> None:
    """Register a verifier on the process-wide default registry."""
    default_registry.register(verifier)


def unregister_verifier(format_name: str) -> None:
    """Remove a verifier from the process-wide default registry."""
    default_registry.unregister(format_name)


def get_verifier(format_name: str) -> Verifier | None:
    """Look up a verifier on the process-wide default registry."""
    return default_registry.get(format_name)
