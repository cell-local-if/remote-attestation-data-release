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


#: Lifecycle statuses of a persistent asynchronous rewrap job. A job is
#: born ``queued``; its background runner (or a post-restart recovery
#: sweep) moves it to ``running`` exactly once, and it settles exactly
#: once to ``succeeded`` (every envelope in scope was reached), ``failed``
#: (the keyring went bad, a historical key was missing, or a single
#: envelope failed), or ``cancelled`` (an explicit cancel won the guarded
#: transition while the job was still queued or running). A terminal
#: status is never changed; a cancelled job is never re-enqueued by the
#: startup recovery sweep.
REWRAP_JOB_STATUS_QUEUED = "queued"
REWRAP_JOB_STATUS_RUNNING = "running"
REWRAP_JOB_STATUS_SUCCEEDED = "succeeded"
REWRAP_JOB_STATUS_FAILED = "failed"
REWRAP_JOB_STATUS_CANCELLED = "cancelled"
REWRAP_JOB_STATUS_CODES = frozenset(
    {
        REWRAP_JOB_STATUS_QUEUED,
        REWRAP_JOB_STATUS_RUNNING,
        REWRAP_JOB_STATUS_SUCCEEDED,
        REWRAP_JOB_STATUS_FAILED,
        REWRAP_JOB_STATUS_CANCELLED,
    }
)


#: Finite, service-defined set of compliance audit-event types. ``grant``
#: events trace the one-time release-grant lifecycle; ``rewrap`` events
#: trace per-envelope key rotation performed by rewrap batches.
AUDIT_EVENT_TYPE_GRANT = "grant"
AUDIT_EVENT_TYPE_REWRAP = "rewrap"
AUDIT_EVENT_TYPE_CODES = frozenset(
    {AUDIT_EVENT_TYPE_GRANT, AUDIT_EVENT_TYPE_REWRAP}
)

#: Finite, service-defined set of compliance audit-event statuses. Grant
#: events move pending -> consumed | revoked (mirroring the grant row);
#: rewrap events are born rewrapped or skipped. Every value is a fixed
#: code derived solely from a committed state transition.
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


#: Finite, service-defined set of workload identity profile lifecycle
#: statuses. A profile is born active and is revoked exactly once; revocation
#: leaves the row in place (still queryable) but removes it from the X.509
#: identity gate. A revoked profile is never made active again.
WORKLOAD_IDENTITY_STATUS_ACTIVE = "active"
WORKLOAD_IDENTITY_STATUS_REVOKED = "revoked"
WORKLOAD_IDENTITY_STATUS_CODES = frozenset(
    {WORKLOAD_IDENTITY_STATUS_ACTIVE, WORKLOAD_IDENTITY_STATUS_REVOKED}
)


#: Finite, service-defined set of trust-root lifecycle statuses. A trust
#: root is born active and is retired exactly once; retirement is a
#: terminal state. A retired root keeps its row (so its id, scope and
#: certificate remain addressable) but no longer anchors X.509 evidence:
#: a chain anchored to it settles as rejected before revocation, identity
#: or signature checks. The same certificate can never be re-created as a
#: new root in the same scope, so retirement cannot be bypassed. A retired
#: root is never made active again.
TRUST_ROOT_STATUS_ACTIVE = "active"
TRUST_ROOT_STATUS_RETIRED = "retired"
TRUST_ROOT_STATUS_CODES = frozenset(
    {TRUST_ROOT_STATUS_ACTIVE, TRUST_ROOT_STATUS_RETIRED}
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
    # A certificate is configured at most once per tenant and workload;
    # retirement leaves the row in place, so the duplicate rule keeps
    # holding for retired roots as well.
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
    # One of TRUST_ROOT_STATUS_*; active -> retired exactly once. A retired
    # root remains stored (its certificate still identifies it for
    # verification short-circuits and duplicate detection) but is a
    # terminal state: it is never made active again.
    status: Mapped[str] = mapped_column(
        String(16), default=TRUST_ROOT_STATUS_ACTIVE
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    # Set exactly once, by the winning active -> retired transition; a
    # repeated retire observes the stored value and never rewrites it.
    retired_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True
    )


class CertificateRevocation(Base):
    """A trust-root-scoped revocation of one X.509 certificate.

    A row revokes exactly one certificate, identified by the unpadded
    base64url SHA-256 of its DER encoding, within exactly one trust root
    (and therefore within that trust root's tenant and workload). The
    revocation becomes effective at ``effective_at`` (UTC): a past instant
    is effective immediately, a future one only once it arrives. Rows are
    never updated or deleted by the service; distinct fingerprints under
    the same trust root are always independent rows — never merged or
    overwritten — while a repeat of an already-registered
    (scope, trust root, fingerprint) tuple is rejected.

    Only identifiers, the scope, the fingerprint and a timestamp are
    stored: no certificate material, evidence, secrets, or free-form text.
    """

    __tablename__ = "certificate_revocations"
    __table_args__ = (
        # At most one registration per (scope, trust root, fingerprint).
        # The constraint is what makes concurrent duplicate registrations
        # settle as exactly one insert plus stable 409s.
        UniqueConstraint(
            "tenant_id",
            "workload_id",
            "trust_root_id",
            "certificate_fingerprint",
            name="uq_cert_revocation_scope_root_fingerprint",
        ),
        Index(
            "ix_cert_revocations_lookup",
            "tenant_id",
            "workload_id",
            "trust_root_id",
            "certificate_fingerprint",
            "effective_at",
        ),
    )

    revocation_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    # The trust root the revocation is anchored to. Revocations never cross
    # trust roots, tenants or workloads.
    trust_root_id: Mapped[str] = mapped_column(String(36), index=True)
    # Unpadded base64url SHA-256 digest (exactly 32 raw bytes, 43 encoded
    # characters) of the certificate's DER encoding, computed identically
    # for the root, intermediate and leaf certificates of an evidence chain.
    certificate_fingerprint: Mapped[str] = mapped_column(String(43))
    # UTC instant at/after which the revocation rejects matching evidence.
    effective_at: Mapped[datetime] = mapped_column(UTCDateTime())
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class WorkloadIdentityProfile(Base):
    """A workload identity profile anchored to one configured trust root.

    A profile belongs to exactly one ``(tenant_id, workload_id,
    trust_root_id)`` scope and carries an ordered set of identity claims.
    During X.509 evidence verification a leaf certificate is admitted when
    at least one of the trust root's profiles has at least one claim whose
    parsed issuer, subject and URI all match the leaf; otherwise — when any
    profile exists for the anchor — the evidence is rejected.

    Two profiles under the same trust root may not carry the identical
    claim *set* while active. The set identity is the SHA-256 of the
    canonical (sorted, compact) JSON of the normalized claims; a unique
    constraint over ``(trust_root_id, claims_fingerprint)`` is what makes
    concurrent duplicate registrations settle as one insert plus stable
    409s, and distinct claim sets are always independent rows that never
    overwrite each other. An update replaces the whole claim set inside
    one atomic transaction (the profile's identity, ownership and
    ``created_at`` never change); when a replacement collides with a
    distinct profile the transaction rolls back and the old set stays in
    force. Revocation sets ``status`` to ``revoked`` once; a revoked
    profile remains queryable but never participates in X.509 gating.

    Only non-sensitive comparison strings are stored — the RFC4514 text of
    a certificate's issuer/subject distinguished names and SAN URI values —
    plus identifiers, scope, lifecycle timestamps and a status. No
    certificate material, evidence, private keys or free-form secrets have
    a column here.
    """

    __tablename__ = "workload_identity_profiles"
    __table_args__ = (
        UniqueConstraint(
            "trust_root_id",
            "claims_fingerprint",
            name="uq_workload_identity_profile_root_claimset",
        ),
        Index(
            "ix_workload_identity_profiles_scope",
            "tenant_id",
            "workload_id",
            "trust_root_id",
        ),
    )

    profile_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    # The configured trust root the profile is anchored to. Profiles never
    # cross trust roots, tenants or workloads.
    trust_root_id: Mapped[str] = mapped_column(String(36), index=True)
    # SHA-256 hex of the canonical JSON of the normalized claim set, used
    # for exact whole-set duplicate detection within one trust root. It is
    # replaced atomically on every successful update.
    claims_fingerprint: Mapped[str] = mapped_column(String(64))
    # One of WORKLOAD_IDENTITY_STATUS_*; active -> revoked exactly once.
    status: Mapped[str] = mapped_column(
        String(16), default=WORKLOAD_IDENTITY_STATUS_ACTIVE
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    # Set by every successful claim-set replacement; NULL for profiles that
    # have never been updated.
    updated_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True
    )
    # Set exactly once, by the active -> revoked transition; a repeated
    # revoke observes the stored value and never rewrites it.
    revoked_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True
    )


class WorkloadIdentityClaim(Base):
    """One identity claim within a workload identity profile.

    A claim matches an X.509 leaf certificate when the certificate's
    parsed issuer distinguished name, subject distinguished name and a
    subjectAlternativeName URI are, as RFC4514/URI strings, byte-for-byte
    equal to ``issuer``, ``subject`` and ``uri`` respectively. All three
    are required (an identity claim is rejected at the API boundary
    otherwise). Rows are written in the same transaction as their parent
    profile; an update deletes the old set and inserts the replacement set
    in the same transaction as the profile change, and a revoked profile
    keeps its (last committed) rows. Only the non-sensitive comparison
    strings are stored.
    """

    __tablename__ = "workload_identity_claims"
    __table_args__ = (
        Index(
            "ix_workload_identity_claims_lookup",
            "tenant_id",
            "workload_id",
            "trust_root_id",
            "issuer",
            "subject",
            "uri",
        ),
        Index(
            "ix_workload_identity_claims_profile_seq",
            "profile_id",
            "seq",
        ),
    )

    claim_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    profile_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workload_identity_profiles.profile_id"),
        index=True,
    )
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    trust_root_id: Mapped[str] = mapped_column(String(36), index=True)
    # RFC4514 string of the leaf certificate issuer DN.
    issuer: Mapped[str] = mapped_column(String(1024))
    # RFC4514 string of the leaf certificate subject DN.
    subject: Mapped[str] = mapped_column(String(1024))
    # SAN URI value the leaf must carry.
    uri: Mapped[str] = mapped_column(String(2048))
    # Zero-based position within the profile's de-duplicated claim list,
    # giving responses a stable first-seen order without an ORDER BY on the
    # long comparison strings.
    seq: Mapped[int] = mapped_column(Integer, default=0)


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


class RewrapJob(Base):
    """One persistent, asynchronously advanced envelope rewrap job.

    Unlike a one-shot rewrap batch, a job is durable state: it is created
    ``queued`` and advanced in the background (and by a post-restart
    recovery sweep) until every envelope of its scope has been reached or
    the first per-envelope failure leaves it ``failed`` with its resume
    cursor parked immediately before the failing envelope. Every page
    commits envelopes independently, so a crash loses only the in-flight
    envelope attempt and the job resumes from its last committed cursor.

    The row stores only identifiers, the scope, the fixed page size,
    opaque cursor strings, counters, status and timestamps — never any
    envelope material, payload or key. Per-envelope outcomes reuse the
    append-only :class:`AuditEvent` rewrap events; this row is only the
    job's progress record.
    """

    __tablename__ = "rewrap_jobs"
    __table_args__ = (
        Index(
            "ix_rewrap_jobs_status",
            "status",
        ),
    )

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    # Requested page size (1..200); fixed for the life of the job.
    limit: Mapped[int] = mapped_column(Integer)
    # Exclusive data_id boundary the job was submitted from; empty string
    # for the beginning of the scope. Never NULL.
    cursor: Mapped[str] = mapped_column(String(256))
    # Resume position: exclusive boundary after the last successfully
    # processed envelope; empty string at the beginning and at the end.
    # On failure it stays immediately before the failing envelope.
    next_cursor: Mapped[str] = mapped_column(String(256))
    # One of REWRAP_JOB_STATUS_*; queued -> running -> succeeded|failed,
    # or queued|running -> cancelled, each transition made at most once
    # with a guarded UPDATE.
    status: Mapped[str] = mapped_column(String(16), default="queued")
    # Cumulative counters across every page the job has advanced.
    processed: Mapped[int] = mapped_column(Integer, default=0)
    rewrapped: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    # True exactly once the scope has been fully scanned (status
    # succeeded); false while queued, running or failed.
    complete: Mapped[bool] = mapped_column(Boolean, default=False)
    # Opaque token held by the runner currently advancing the job; NULL
    # when the job is queued, between pages/processes, or terminal. A
    # guarded UPDATE (claim_token IS NULL) makes advancement mutually
    # exclusive across threads/processes; a startup sweep NULLs the
    # stale claims left by a dead process before recovery begins.
    claim_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    # Bumped on every status/progress commit; equal to created_at until
    # the background runner makes its first transition.
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())
    # Set exactly once, by the winning queued|running -> cancelled
    # transition; NULL in every other state. A repeated cancel observes
    # the stored value and never rewrites it. The cancel marker and the
    # status change commit in the same transaction.
    cancelled_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True
    )


class RewrapJobIdempotencyRecord(Base):
    """The first accepted submission of an idempotency-keyed rewrap job.

    At most one row exists per ``(tenant_id, workload_id,
    idempotency_key)`` triple. The row is inserted in the same
    transaction as the :class:`RewrapJob` it records, so a crash can
    never leave a job without its idempotency record or vice versa; a
    retried request finds the committed row and returns the stored
    response verbatim without creating a second job or touching the
    envelope cursor, regardless of the job's later lifecycle.

    The stored fingerprint covers only non-sensitive request shape:
    the scope, the effective page size and the normalized cursor start
    (omitted and explicit-empty cursors normalize to the same start).
    The stored response is the exact compact JSON body (including its
    trailing newline) first returned to the caller. No payload,
    capability, key or certificate material is ever stored here.
    """

    __tablename__ = "rewrap_job_idempotency_records"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workload_id",
            "idempotency_key",
            name="uq_rewrap_job_idempotency_scope_key",
        ),
    )

    record_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    # Caller-chosen key: 1..64 visible ASCII characters. Unique only
    # within the (tenant, workload) scope; the same key in another scope
    # is an independent row and never conflicts.
    idempotency_key: Mapped[str] = mapped_column(String(64))
    # Job recorded by the first legal submission; every later replay of
    # this key answers with that same job's original acceptance.
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    # Hex SHA-256 over the normalized request shape (scope, effective
    # limit, normalized cursor start); used only to detect a same-key
    # request with different content, which is a 409 that changes
    # nothing.
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    # The exact first 202 response body, stored verbatim (compact JSON
    # plus its single trailing newline) and replayed byte-for-byte.
    response_body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class AuditEvent(Base):
    """Append-only, tenant-scoped compliance audit event.

    A row is inserted in the same committed transaction as the state
    transition it records (grant issuance/consumption/revocation, or a
    per-envelope rewrap-batch outcome), so an event exists if and only if
    the change committed and it survives restarts. The table is never
    updated or deleted by the service.

    Only identifiers, the fixed event type/status codes, timestamps and
    the capability SHA-256 digest are stored. Plaintext capabilities,
    payloads, evidence, data keys, master keys and certificate material
    never have a column here. For ``rewrap`` events the grant/decision
    identifiers and the capability digest are semantically empty and are
    stored as NULL.
    """

    __tablename__ = "audit_events"
    __table_args__ = (
        # Covers the scoped listing ordered by the (occurred_at, event_id)
        # keyset, including its exclusive cursor predicate.
        Index(
            "ix_audit_events_scope_occurred",
            "tenant_id",
            "workload_id",
            "occurred_at",
            "event_id",
        ),
    )

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256), index=True)
    # One of AUDIT_EVENT_TYPE_*; a fixed, service-defined code only.
    event_type: Mapped[str] = mapped_column(String(16))
    # Set for grant events; NULL for rewrap events.
    grant_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    decision_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    data_id: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    # One of AUDIT_EVENT_STATUS_*; a fixed, service-defined code only.
    status: Mapped[str] = mapped_column(String(16))
    # SHA-256 digest of the capability for grant events; NULL otherwise.
    capability_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    # Commit time of the recorded transition; the stable listing key.
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)


class RateLimitCounter(Base):
    """Persistent per-scope admission counter for one UTC natural minute.

    One row exists per ``(tenant_id, workload_id, window_start)`` triple,
    where ``window_start`` is the UTC minute (a datetime with seconds and
    sub-seconds truncated to zero) during which the counted business
    requests arrived. Consume, revoke and payload release share the same
    row, so the budget is shared across all three entry points. The count
    is durable: it is committed before the request enters any business
    judgement and therefore survives restarts and is never refunded when a
    later judgement returns 404/401/409/410/500. A new minute starts a new
    row, which both restores the quota automatically and isolates scopes
    strictly — quotas are never borrowed across scopes or minutes.
    """

    __tablename__ = "rate_limit_counters"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workload_id",
            "window_start",
            name="uq_rate_limit_counter_scope_window",
        ),
    )

    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    workload_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    # UTC minute boundary at which the window starts, truncated to seconds.
    window_start: Mapped[datetime] = mapped_column(UTCDateTime(), primary_key=True)
    # Number of admitted (budget-consuming) requests during this window.
    count: Mapped[int] = mapped_column(Integer, default=0)
