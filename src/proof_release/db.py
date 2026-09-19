from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import DateTime, String, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UTCDateTime(TypeDecorator):
    """Store datetimes as UTC and always return timezone-aware UTC values."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class VerificationResultCode(str, Enum):
    """Finite, service-defined set of non-sensitive verification outcomes.

    These are the only result values the service persists for a
    verification. The set is defined here — never by plugins — and each code
    is derived solely from the verifier's controlled pass/fail conclusion
    (``VerificationResult.accepted``). No plugin-supplied text, reason or
    detail is accepted, transformed, truncated or stored, so a compromised or
    buggy verifier cannot smuggle raw evidence, key material or other private
    context into the database through this field.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"

    @classmethod
    def from_accepted(cls, accepted: bool) -> "VerificationResultCode":
        """Map the verifier's boolean verdict onto the finite code set."""
        return cls.ACCEPTED if accepted else cls.REJECTED


class Challenge(Base):
    __tablename__ = "challenges"

    challenge_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    # Only the SHA-256 digest of the nonce is persisted, never the nonce itself.
    nonce_digest: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    issued_at: Mapped[datetime] = mapped_column(UTCDateTime())
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime())
    consumed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class Evidence(Base):
    __tablename__ = "evidence"

    evidence_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # At most one evidence record per challenge.
    challenge_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    evidence_format: Mapped[str] = mapped_column(String(128))
    # received -> verified | rejected, settled atomically on first verify.
    status: Mapped[str] = mapped_column(String(16), default="received")
    received_at: Mapped[datetime] = mapped_column(UTCDateTime())
    # Only the SHA-256 digest of the evidence is persisted, never the evidence itself.
    evidence_sha256: Mapped[str] = mapped_column(String(64))
    verified_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True
    )
    # Service-defined code from the finite VerificationResultCode set, set
    # solely from the verifier's accepted boolean on first settlement. There
    # is no free-text column: plugin-supplied strings are never persisted.
    verification_result_code: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )
