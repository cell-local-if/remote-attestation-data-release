from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
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

#: Finite, service-defined set of persisted release-grant statuses. A grant
#: is born pending and settles atomically, exactly once, to either consumed
#: (its first successful presentation or payload release) or revoked (an
#: explicit revocation that found it still pending). Expiry is derived from
#: expires_at rather than stored as a status.
RELEASE_GRANT_STATUS_PENDING = "pending"
RELEASE_GRANT_STATUS_CONSUMED = "consumed"
RELEASE_GRANT_STATUS_REVOKED = "revoked"
RELEASE_GRANT_STATUS_CODES = frozenset(
    {
        RELEASE_GRANT_STATUS_PENDING,
        RELEASE_GRANT_STATUS_CONSUMED,
        RELEASE_GRANT_STATUS_REVOKED,
    }
)

#: Per-envelope outcome codes recorded by a rewrap batch. ``rewrapped``
#: means the wrapping key was replaced by the current master key version;
#: ``skipped`` means the envelope was already wrapped under the current
#: version and no material changed; the remaining codes mark where a page
#: stopped and are never followed by later items in the same batch.
REWRAP_RESULT_REWRAPPED = "rewrapped"
REWRAP_RESULT_SKIPPED = "skipped"
REWRAP_RESULT_KEYRING = "keyring"
REWRAP_RESULT_MISSING_KEY = "missing-key"
REWRAP_RESULT_REWRAP_FAILED = "rewrap"
REWRAP_RESULT_CODES = frozenset(
    {
        REWRAP_RESULT_REWRAPPED,
        REWRAP_RESULT_SKIPPED,
        REWRAP_RESULT_KEYRING,
        REWRAP_RESULT_MISSING_KEY,
        REWRAP_RESULT_REWRAP_FAILED,
    }
)

#: Event types exposed by the read-only compliance audit. A ``grant`` event
#: is the lifecycle record of one release grant (pending at issue, then
#: exactly one consumed/revoked settlement); a ``rewrap`` event is one
#: per-envelope audit row written by a rewrap batch.
AUDIT_EVENT_TYPE_GRANT = "grant"
AUDIT_EVENT_TYPE_REWRAP = "rewrap"
AUDIT_EVENT_TYPES = frozenset({AUDIT_EVENT_TYPE_GRANT, AUDIT_EVENT_TYPE_REWRAP})

#: Finite, service-defined set of compliance event statuses. Grant events
#: track the grant lifecycle (pending/consumed/revoked); rewrap events
#: record the fixed per-envelope batch outcome (rewrapped/skipped).
AUDIT_EVENT_STATUS_PENDING = "pending"
AUDIT_EVENT_STATUS_CONSUMED = "consumed"
AUDIT_EVENT_STATUS_REVOKED = "revoked"
AUDIT_EVENT_STATUS_REWRAPPED = "rewrapped"
AUDIT_EVENT_STATUS_SKIPPED = "skipped"
AUDIT_EVENT_STATUS_CODES = frozenset(
    {
        AUDIT_EVENT_STATUS_PENDING,
        AUDIT_EVENT_STATUS_CONSUMED,
        AUDIT_EVENT_STATUS_REVOKED,
        AUDIT_EVENT_STATUS_REWRAPPED,
        AUDIT_EVENT_STATUS_SKIPPED,
    }
)


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


class DataEnvelope(Base):
    """An envelope-encrypted data item scoped to a tenant and workload.

    The row stores only ciphertext and key material, never the plaintext
    payload or the plaintext data key:

    * ``ciphertext`` — AES-256-GCM ciphertext (tag stored separately),
    * ``iv`` — the 12-byte GCM nonce,
    * ``tag`` — the 16-byte GCM authentication tag,
    * ``wrapped_key`` — the 32-byte data key wrapped with AES-KW under the
      master key version recorded in ``key_version``.

    All four material columns are NOT NULL so a successful insert is a
    single atomic write: there is no observable state in which part of the
    envelope exists. The plaintext is never recoverable from any column.
    """

    __tablename__ = "data_envelopes"
    # A data_id is unique within a tenant/workload scope; the composite
    # primary key is also the lookup key for the GET endpoint.
    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    workload_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    data_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    # Master key version under which wrapped_key was produced.
    key_version: Mapped[int] = mapped_column(Integer)
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    iv: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    tag: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    wrapped_key: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class ReleaseGrant(Base):
    """A one-time, TTL-bounded capability that releases one data item.

    A grant can only be minted for an ``allowed`` decision in the same
    tenant/workload scope. Exactly one settlement may ever succeed: the
    first caller that presents the correct capability while the grant is
    pending and unexpired atomically settles it to ``consumed`` (via the
    consume endpoint or an authorized payload release), or the holder
    revokes a still-pending grant, atomically settling it to ``revoked``.
    Revocation, consumption and release all race on the same guarded
    status transition, so at most one can win.

    The plaintext capability is never stored — only its SHA-256 digest —
    and it is returned exactly once, on the create response. The row stores
    only identifiers, the scope, the digest, the fixed status code and
    timestamps: never the capability, evidence, claims, or any payload.
    """

    __tablename__ = "release_grants"

    grant_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    decision_id: Mapped[str] = mapped_column(String(36), index=True)
    # Caller-supplied identifier of the protected data item this grant
    # authorizes release of. The service never sees the data itself.
    data_id: Mapped[str] = mapped_column(String(256))
    # Only the SHA-256 digest of the capability is persisted.
    capability_digest: Mapped[str] = mapped_column(String(64))
    # pending -> consumed | revoked, settled atomically on the first valid
    # consume/release/revoke transition.
    status: Mapped[str] = mapped_column(String(16), default="pending")
    issued_at: Mapped[datetime] = mapped_column(UTCDateTime())
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime())
    consumed_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True
    )
    # Set exactly once, by the winning pending -> revoked transition; a
    # repeated revoke observes the stored value and never rewrites it.
    revoked_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True
    )


class RewrapBatch(Base):
    """One page of a scoped, cursor-driven envelope rewrap batch.

    Each POST creates one batch bound to exactly one tenant/workload
    scope and the exclusive data_id cursor it started from. The batch is
    persisted before any envelope is touched, so a failed keyring check
    leaves no batch behind while a page that stops mid-way (a missing
    historical key or a failed rewrap) stays queryable after restart with
    all of its per-envelope audit rows.
    """

    __tablename__ = "rewrap_batches"

    batch_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    # Requested page size (1..200); informational for audit reads.
    limit: Mapped[int] = mapped_column(Integer)
    # Exclusive data_id boundary the page started after; empty string for
    # the beginning of the scope. Never NULL.
    cursor: Mapped[str] = mapped_column(String(256))
    # Exclusive boundary to resume from; empty string when the scope has
    # been fully scanned and complete is true.
    next_cursor: Mapped[str] = mapped_column(String(256))
    complete: Mapped[bool] = mapped_column(Boolean)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    rewrapped: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class RewrapBatchItem(Base):
    """One per-envelope audit row within a rewrap batch.

    A row is committed independently for every envelope the page reaches,
    including the envelope that stopped the page. It records only the
    scope, the envelope identifier, the old and new master key versions
    (the new version equals the old one for skips and failures), the
    fixed result code and a UTC timestamp — never the wrapped key, the
    data key, or any payload material.
    """

    __tablename__ = "rewrap_batch_items"

    item_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    batch_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("rewrap_batches.batch_id"), index=True
    )
    # Zero-based position of the item within its batch, giving the audit
    # a stable order even when per-item commits land in the same second.
    seq: Mapped[int] = mapped_column(Integer)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    data_id: Mapped[str] = mapped_column(String(256))
    # Master key version recorded on the envelope before this page touched
    # it; identical to new_key_version for skips/failures.
    old_key_version: Mapped[int] = mapped_column(Integer)
    new_key_version: Mapped[int] = mapped_column(Integer)
    # One of REWRAP_RESULT_*; a fixed, service-defined code only.
    result: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class AuditEvent(Base):
    """One immutable entry in the per-tenant compliance audit trail.

    Rows are append-only: a ``grant`` event is written in the same
    transaction that mints the grant and updated in-place (status and
    occurred_at only) by the single consuming/revoking settlement, while a
    ``rewrap`` event is inserted in the same per-envelope transaction that a
    rewrap batch commits and never changes afterwards. The row records only
    identifiers, fixed type/status codes, timestamps and (for grant events)
    the capability SHA-256 digest — never plaintext capabilities, payloads,
    evidence, data keys, master keys, certificate material or plugin text.

    Ordering on the read path is (occurred_at, event_id); event_id is a
    canonical random UUID, so the tiebreaker is stable even when several
    events share a timestamp.
    """

    __tablename__ = "audit_events"
    __table_args__ = (
        Index(
            "ix_audit_events_scope_occurred",
            "tenant_id",
            "workload_id",
            "occurred_at",
            "event_id",
        ),
    )

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256))
    workload_id: Mapped[str] = mapped_column(String(256))
    # One of AUDIT_EVENT_TYPE_*; a fixed, service-defined code only.
    event_type: Mapped[str] = mapped_column(String(16))
    # Grant events reference the grant and its authorizing decision/data
    # item. Rewrap events carry no grant linkage: these stay NULL.
    grant_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    decision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    data_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # One of AUDIT_EVENT_STATUS_*; a fixed, service-defined code only.
    status: Mapped[str] = mapped_column(String(16))
    # Grant events only: SHA-256 digest of the one-time capability. NULL
    # for rewrap events, whose subject (an envelope) has no capability.
    capability_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    # Timestamp the recorded fact occurred: grant issuance for a pending
    # grant event, the settlement instant afterwards, the per-envelope
    # commit time for a rewrap event.
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime())
