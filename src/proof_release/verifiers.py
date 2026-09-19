"""Public verifier plugin interface for evidence formats.

Verifier plugins are selected by ``evidence_format`` and receive the evidence
plaintext together with non-sensitive challenge/tenant/workload context.
Plugins must treat the evidence plaintext as ephemeral: the service layer
never persists it, and plugins must not persist or log it either.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class VerificationContext:
    """Non-sensitive context handed to verifier plugins.

    Carries identifiers and challenge timestamps only — never the evidence
    plaintext, the nonce, or any persisted secret.
    """

    evidence_id: str
    challenge_id: str
    tenant_id: str
    workload_id: str
    evidence_format: str
    challenge_issued_at: datetime
    challenge_expires_at: datetime


@runtime_checkable
class EvidenceVerifier(Protocol):
    """Plugin interface for verifying one evidence format.

    ``verify`` receives the evidence plaintext for this request only and must
    return True (verified) or False (rejected). Raising an exception aborts
    the verification without persisting any state.
    """

    def verify(self, evidence: str, context: VerificationContext) -> bool: ...


_verifiers: dict[str, EvidenceVerifier] = {}


def register_verifier(evidence_format: str, verifier: EvidenceVerifier) -> None:
    """Register (or replace) the verifier for an evidence format."""
    if not evidence_format or not evidence_format.strip():
        raise ValueError("evidence_format must be a non-empty string")
    if not isinstance(verifier, EvidenceVerifier):
        raise TypeError("verifier must implement the EvidenceVerifier protocol")
    _verifiers[evidence_format] = verifier


def get_verifier(evidence_format: str) -> EvidenceVerifier | None:
    """Return the verifier registered for a format, or None if unsupported."""
    return _verifiers.get(evidence_format)


def registered_formats() -> tuple[str, ...]:
    """Return the sorted evidence formats with a registered verifier."""
    return tuple(sorted(_verifiers))


_PAYLOAD_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class Sha256DigestVerifier:
    """Verifier for the built-in ``sha256-digest`` evidence format.

    Evidence is ``<payload>.<digest>`` where ``payload`` is unpadded base64url
    and ``digest`` is the lowercase hex SHA-256 of the decoded payload bytes.
    Verification recomputes the digest and compares it in constant time.
    """

    def verify(self, evidence: str, context: VerificationContext) -> bool:
        payload_part, sep, digest_part = evidence.rpartition(".")
        if not sep or not _PAYLOAD_RE.fullmatch(payload_part):
            return False
        if not _SHA256_HEX_RE.fullmatch(digest_part):
            return False
        padding = "=" * (-len(payload_part) % 4)
        try:
            payload = base64.urlsafe_b64decode(payload_part + padding)
        except (binascii.Error, ValueError):
            return False
        actual = hashlib.sha256(payload).hexdigest()
        return hmac.compare_digest(actual, digest_part)


register_verifier("sha256-digest", Sha256DigestVerifier())
