"""SQLAlchemy persistence layer for proof-of-release challenges."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, String, TypeDecorator, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetime stored in SQLite (which drops tzinfo)."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class Base(DeclarativeBase):
    pass


class Challenge(Base):
    """A one-time random challenge issued to a tenant workload.

    Only the SHA-256 digest of the nonce is persisted; the plaintext nonce
    is returned once at creation and never stored or logged.
    """

    __tablename__ = "challenges"

    challenge_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    workload_id: Mapped[str] = mapped_column(String(256), nullable=False)
    nonce_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    issued_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


def make_engine(database_url: str) -> Engine:
    connect_args: dict[str, Any] = {}
    if database_url.startswith("sqlite"):
        # check_same_thread: connections are shared across request threads.
        # timeout: wait on the SQLite write lock instead of failing fast,
        # so concurrent consumers serialize on the atomic UPDATE.
        connect_args = {"check_same_thread": False, "timeout": 30}
    return create_engine(database_url, connect_args=connect_args)
