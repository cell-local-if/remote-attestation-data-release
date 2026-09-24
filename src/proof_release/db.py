from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
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
#: is born pending and settles atomically to consumed on its first (and
#: only) successful presentation; expiry is derived from expires_at rather
#: than stored as a status.
RELEASE_GRANT_STATUS_PENDING = "pending"
RELEASE_GRANT_STATUS_CONSUMED = "consumed"
RELEASE_GRANT_STATUS_CODES = frozenset(
    {RELEASE_GRANT_STATUS_PENDING, RELEASE_GRANT_STATUS_CONSUMED}
)

#: Per-envelope outcome codes recorded by a rewrap batch. ``rewrapped`` and
#: ``skipped`` are successful outcomes (a historical-version envelope got a
#: new wrapping key; a current-version envelope needed none); the remaining
#: codes stop the page with the failing envelope left untouched.
REWRAP_RESULT_REWRAPPED = "rewrapped"
REWRAP_RESULT_SKIPPED = "skipped"
REWRAP_RESULT_KEYRING = "keyring"
REWRAP_RESULT_MISSING_KEY = "missing-key"
REWRAP_RESULT_REWRAP = "rewrap"
REWRAP_RESULT_CODES = frozenset(
    {
        REWRAP_RESULT_REWRAPPED,
        REWRAP_RESULT_SKIPPED,
        REWRAP_RESULT_KEYRING,
        REWRAP_RESULT_MISSING_KEY,
        REWRAP_RESULT_REWRAP,
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
    tenant/workload scope. Exactly one consume may ever succeed: the first
    caller that presents the correct capability while the grant is pending
    and unexpired atomically settles it to ``consumed``.

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
    # pending -> consumed, settled atomically on the first valid consume.
    status: Mapped[str] = mapped_column(String(16), default="pending")
    issued_at: Mapped[datetime] = mapped_column(UTCDateTime())
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime())
    consumed_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True
    )


class RewrapCursor(Base):
    """An opaque, scope-bound keyset cursor for rewrap batch pagination.

    The token is an unguessable random string; it maps to the exclusive
    ``data_id`` boundary within one tenant/workload ordering. A NULL
    boundary means "before the smallest data_id" (the start position).
    Cursors are immutable and reusable: retrying a page with the same
    cursor never re-selects an earlier envelope. A forged token (no row)
    or one presented from another scope is rejected.
    """

    __tablename__ = "rewrap_cursors"

    cursor_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    #: Exclusive data_id boundary; NULL represents the start position.
    boundary_data_id: Mapped[str | None] = mapped_column(
        String(256), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class RewrapBatch(Base):
    """One auditable page of a scoped envelope rewrap batch.

    A batch covers at most the requested ``limit`` envelopes of one
    tenant/workload, selected in stable ``data_id`` order after the
    supplied cursor. The summary counters are the per-envelope audit
    outcomes; ``next_cursor`` (an opaque RewrapCursor token) resumes after
    the last handled envelope and is empty when ``done`` is true. Rows are
    written before processing begins, so a batch stays queryable across a
    restart even if a page stops early on a failure.
    """

    __tablename__ = "rewrap_batches"

    batch_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    #: The cursor token presented on the request (NULL for the start
    #: position), retained for traceability.
    request_cursor: Mapped[str | None] = mapped_column(String(32), nullable=True)
    limit_value: Mapped[int] = mapped_column(Integer)
    processed_count: Mapped[int] = mapped_column(Integer, default=0)
    rewrapped_count: Mapped[int] = mapped_column(Integer, default=0)
    skipped_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    #: Issued continuation token (RewrapCursor.cursor_id); NULL together
    #: with done=True marks the end of the range.
    next_cursor: Mapped[str | None] = mapped_column(String(32), nullable=True)
    done: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    completed_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True
    )


class RewrapBatchAudit(Base):
    """The per-envelope audit row of a rewrap batch.

    Exactly one row is written per envelope the batch reaches, committed
    independently of every other envelope. It records only the scope, the
    envelope identifier, the old/new wrapping key versions, a fixed
    service-defined result code (one of REWRAP_RESULT_*) and the UTC time
    — never payload, data key, or wrapped-key material. The target (new)
    version is the keyring's current version established by the batch
    preflight, so it is always known even if the keyring becomes
    unavailable while the page is being processed.
    """

    __tablename__ = "rewrap_batch_audits"
    __table_args__ = (
        UniqueConstraint(
            "batch_id", "data_id", name="uq_rewrap_batch_audit_envelope"
        ),
    )

    audit_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(36), index=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    data_id: Mapped[str] = mapped_column(String(256))
    #: Position within the batch, giving audits a stable order.
    seq: Mapped[int] = mapped_column(Integer)
    old_key_version: Mapped[int] = mapped_column(Integer)
    new_key_version: Mapped[int] = mapped_column(Integer)
    # One of REWRAP_RESULT_*; a fixed service-defined code only.
    result: Mapped[str] = mapped_column(String(16))
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime())
