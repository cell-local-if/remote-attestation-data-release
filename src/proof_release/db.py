from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


#: Finite, service-defined set of persisted verification outcome codes.
#: The stored code is derived solely from the verifier's accept/reject
#: verdict; plugin-supplied text is never persisted in any form.
VERIFICATION_RESULT_ACCEPTED = "accepted"
VERIFICATION_RESULT_REJECTED = "rejected"
VERIFICATION_RESULT_CODES = frozenset(
    {VERIFICATION_RESULT_ACCEPTED, VERIFICATION_RESULT_REJECTED}
)

#: Finite, service-defined set of persisted release-decision statuses. A
#: decision is allowed exactly when the verified evidence's claims satisfy
#: the requested policy version; otherwise the release is denied.
DECISION_STATUS_ALLOWED = "allowed"
DECISION_STATUS_DENIED = "denied"
DECISION_STATUS_CODES = frozenset({DECISION_STATUS_ALLOWED, DECISION_STATUS_DENIED})

#: Finite, service-defined set of release-grant statuses. A grant is
#: pending until its capability is presented exactly once, after which it
#: is consumed and can never be used again.
GRANT_STATUS_PENDING = "pending"
GRANT_STATUS_CONSUMED = "consumed"
GRANT_STATUS_CODES = frozenset({GRANT_STATUS_PENDING, GRANT_STATUS_CONSUMED})


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
    # Service-defined outcome code (one of VERIFICATION_RESULT_*), derived
    # only from the accept/reject verdict. Plugin-supplied text is never
    # persisted, so no free-form detail column exists here.
    verification_result: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )


class TrustRoot(Base):
    __tablename__ = "trust_roots"
    # A certificate is configured at most once per tenant and workload.
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "workload_id", "cert_sha256", name="uq_trust_root_cert"
        ),
    )

    root_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # PEM of the X.509 CA certificate. This is public material only; private
    # keys are rejected at the API boundary and never stored here.
    root_pem: Mapped[str] = mapped_column(Text)
    # SHA-256 of the DER encoding, used for exact duplicate detection.
    cert_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class Policy(Base):
    """A versioned release policy scoped to a tenant and workload.

    Creating a policy with the same (tenant, workload, name) adds a new,
    higher version; older versions are retained and remain addressable by
    id and version so that decisions already taken against them stay
    explainable.
    """

    __tablename__ = "policies"
    # One version number per policy name within a scope. The unique
    # constraint is what makes concurrent version allocation gap-free:
    # two transactions can never both claim the same version.
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workload_id",
            "name",
            "version",
            name="uq_policy_scope_name_version",
        ),
    )

    policy_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    name: Mapped[str] = mapped_column(String(256))
    version: Mapped[int] = mapped_column(Integer)
    # Canonical JSON serialization (sorted keys, compact separators) of the
    # validated rule tree. Rules contain only claim names and scalar
    # comparison values — never evidence or claims values.
    rule_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class Decision(Base):
    """An auditable release decision for one evidence/policy-version pair.

    Exactly one row may exist per (evidence, policy version); retries and
    concurrent requests return that same row. The row records only
    identifiers, the fixed status code and timestamps — never the raw
    evidence, the nonce, or the evaluated claims.
    """

    __tablename__ = "decisions"
    __table_args__ = (
        UniqueConstraint(
            "evidence_id",
            "policy_id",
            name="uq_decision_evidence_policy",
        ),
    )

    decision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    evidence_id: Mapped[str] = mapped_column(String(36), index=True)
    policy_id: Mapped[str] = mapped_column(String(36), index=True)
    # Snapshot of the evaluated policy version; policy_id already identifies
    # an immutable version, this denormalizes it for audit reads.
    policy_version: Mapped[int] = mapped_column(Integer)
    # One of DECISION_STATUS_*; a fixed, service-defined code only.
    status: Mapped[str] = mapped_column(String(16))
    decided_at: Mapped[datetime] = mapped_column(UTCDateTime())


class ReleaseGrant(Base):
    """A one-time release grant minted from an allowed decision.

    The row records only identifiers (grant, decision, scope, data), the
    fixed status code, timestamps, and the SHA-256 digest of the
    capability. The plaintext capability is returned exactly once, in the
    creation response, and is never persisted, logged, or returned again;
    no evidence, claims, or payload are ever stored here.
    """

    __tablename__ = "release_grants"

    grant_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    decision_id: Mapped[str] = mapped_column(String(36), index=True)
    # Identifier of the protected data this grant authorizes releasing.
    data_id: Mapped[str] = mapped_column(String(256))
    # Only the SHA-256 digest of the capability is persisted, never the
    # capability itself.
    capability_digest: Mapped[str] = mapped_column(String(64))
    # One of GRANT_STATUS_*; pending -> consumed, settled atomically.
    status: Mapped[str] = mapped_column(String(16), default=GRANT_STATUS_PENDING)
    issued_at: Mapped[datetime] = mapped_column(UTCDateTime())
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime())
    consumed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
