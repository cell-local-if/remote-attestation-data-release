from __future__ import annotations

from datetime import datetime, timezone

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
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    # One challenge can back at most one evidence record; the unique constraint
    # also guarantees concurrent submissions cannot both persist a record.
    challenge_id: Mapped[str] = mapped_column(
        String(36), unique=True, index=True
    )
    evidence_format: Mapped[str] = mapped_column(String(64))
    # Only the SHA-256 digest of the evidence is persisted, never the raw evidence.
    evidence_digest: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="received")
    received_at: Mapped[datetime] = mapped_column(UTCDateTime())
