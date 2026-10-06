from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
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

#: Finite, service-defined set of policy-version lifecycle statuses. A
#: policy version is born active and is retired exactly once; retirement
#: is a terminal state. A retired version keeps its row (so its id,
#: version, rule and the decisions already taken against it stay
#: explainable and addressable) but can no longer produce new decisions:
#: a decision request against it fails with 409. Creating the next
#: version of the same name allocates a fresh active row, so retirement
#: of one version cannot be bypassed by deleting or overwriting it —
#: versions are never deleted or updated, and a retired version is never
#: made active again. Versions are strictly isolated: retiring one
#: version changes no other version.
POLICY_STATUS_ACTIVE = "active"
POLICY_STATUS_RETIRED = "retired"
POLICY_STATUS_CODES = frozenset({POLICY_STATUS_ACTIVE, POLICY_STATUS_RETIRED})

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

#: Finite, service-defined set of reasons recorded by a per-grant state
#: migration event. Every event is one committed migration and its reason
#: identifies *which* action committed it: ``issued`` for the empty-state
#: birth into pending, ``consume`` for a winning consume presentation,
#: ``release`` for a winning payload release (both settle the grant to
#: consumed, distinguished only by this reason), and ``revoked`` for the
#: pending -> revoked settlement. Free-form text is never persisted.
RELEASE_GRANT_EVENT_REASON_ISSUED = "issued"
RELEASE_GRANT_EVENT_REASON_CONSUME = "consume"
RELEASE_GRANT_EVENT_REASON_RELEASE = "release"
RELEASE_GRANT_EVENT_REASON_REVOKED = "revoked"
RELEASE_GRANT_EVENT_REASON_CODES = frozenset(
    {
        RELEASE_GRANT_EVENT_REASON_ISSUED,
        RELEASE_GRANT_EVENT_REASON_CONSUME,
        RELEASE_GRANT_EVENT_REASON_RELEASE,
        RELEASE_GRANT_EVENT_REASON_REVOKED,
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


#: Finite, service-defined set of reasons recorded by a rewrap job
#: lifecycle event. Every event is one committed status migration and its
#: reason explains only *which* migration committed: ``submitted`` for the
#: queued birth, ``executed`` for the queued/running -> running claim,
#: ``recovered`` for a post-restart failed -> running retry, ``completed``
#: for running -> succeeded, ``cancelled`` for queued/running ->
#: cancelled, and the three failure classifications for running -> failed
#: (a bad keyring, a missing historical key, or an envelope that could not
#: be rewrapped). Exception text is never persisted: a failed migration is
#: classified into exactly one of the fixed codes below.
REWRAP_JOB_EVENT_REASON_SUBMITTED = "submitted"
REWRAP_JOB_EVENT_REASON_EXECUTED = "executed"
REWRAP_JOB_EVENT_REASON_RECOVERED = "recovered"
REWRAP_JOB_EVENT_REASON_COMPLETED = "completed"
REWRAP_JOB_EVENT_REASON_CANCELLED = "cancelled"
REWRAP_JOB_EVENT_REASON_KEYRING = REWRAP_RESULT_KEYRING
REWRAP_JOB_EVENT_REASON_MISSING_KEY = REWRAP_RESULT_MISSING_KEY
REWRAP_JOB_EVENT_REASON_REWRAP_FAILED = REWRAP_RESULT_REWRAP_FAILED
REWRAP_JOB_EVENT_REASON_CODES = frozenset(
    {
        REWRAP_JOB_EVENT_REASON_SUBMITTED,
        REWRAP_JOB_EVENT_REASON_EXECUTED,
        REWRAP_JOB_EVENT_REASON_RECOVERED,
        REWRAP_JOB_EVENT_REASON_COMPLETED,
        REWRAP_JOB_EVENT_REASON_CANCELLED,
        REWRAP_JOB_EVENT_REASON_KEYRING,
        REWRAP_JOB_EVENT_REASON_MISSING_KEY,
        REWRAP_JOB_EVENT_REASON_REWRAP_FAILED,
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


#: Finite, service-defined set of proof-lifecycle audit-event types. One
#: event type exists per attestation stage traced by the proof timeline:
#: ``proof-received`` when an evidence record is first accepted,
#: ``proof-verified`` when its first verification settles, and
#: ``proof-decision`` when its first policy decision is recorded.
PROOF_EVENT_TYPE_RECEIVED = "proof-received"
PROOF_EVENT_TYPE_VERIFIED = "proof-verified"
PROOF_EVENT_TYPE_DECISION = "proof-decision"
PROOF_EVENT_TYPE_CODES = frozenset(
    {
        PROOF_EVENT_TYPE_RECEIVED,
        PROOF_EVENT_TYPE_VERIFIED,
        PROOF_EVENT_TYPE_DECISION,
    }
)

#: Finite, service-defined set of proof-lifecycle audit-event statuses.
#: Received events are born ``received``; verification settles exactly
#: once to ``verified`` or ``rejected``; a decision settles exactly once
#: to ``allowed`` or ``denied``. Each value is a fixed code derived only
#: from a committed transition, never from plugin-supplied text.
PROOF_EVENT_STATUS_RECEIVED = "received"
PROOF_EVENT_STATUS_VERIFIED = "verified"
PROOF_EVENT_STATUS_REJECTED = "rejected"
PROOF_EVENT_STATUS_ALLOWED = "allowed"
PROOF_EVENT_STATUS_DENIED = "denied"
PROOF_EVENT_STATUS_CODES = frozenset(
    {
        PROOF_EVENT_STATUS_RECEIVED,
        PROOF_EVENT_STATUS_VERIFIED,
        PROOF_EVENT_STATUS_REJECTED,
        PROOF_EVENT_STATUS_ALLOWED,
        PROOF_EVENT_STATUS_DENIED,
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
        # Per-scope commit order: the immutable high-water mark that fixes
        # the lifecycle query's replayable snapshot. Unique so two
        # concurrent allocations can never mint the same sequence; together
        # with the per-scope counter row taken FOR UPDATE (and BEGIN
        # IMMEDIATE on SQLite) this makes the order gap-free on every
        # backend, and the index covers the scoped snapshot predicate.
        Index(
            "ix_trust_roots_scope_commit_seq",
            "tenant_id",
            "workload_id",
            "commit_seq",
            unique=True,
        ),
        # Covers the scoped lifecycle listing ordered by the
        # (created_at, root_id) keyset together with its snapshot cutoff.
        Index(
            "ix_trust_roots_scope_created",
            "tenant_id",
            "workload_id",
            "created_at",
            "root_id",
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
    # Gap-free per-scope sequence allocated in the root's own creation
    # transaction from the shared trust-root lifecycle counter, strictly
    # increasing in business commit order on every backend. NULL only on
    # rows written before the column existed (backfilled on open); every
    # new root carries a positive value. The lifecycle listing bounds its
    # replayable snapshot membership by this marker, never by write timing.
    commit_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # The lifecycle counter sequence allocated inside the winning
    # active -> retired transaction; NULL while the root is active. The
    # lifecycle query reconstructs a root's status *as of* its fixed
    # snapshot from this marker: retired exactly when retired_seq is at or
    # before the snapshot high-water mark. A retirement that commits after
    # a snapshot's first query therefore never changes a replayed page,
    # even though the row itself is updated in place.
    retired_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class TrustRootCommitCounter(Base):
    """Per-scope monotonic allocator for the trust-root lifecycle sequence.

    Exactly one row exists per ``(tenant_id, workload_id)``. It is an
    internal ordering device — never exposed on any response and holding
    no certificate material or secret — whose sole purpose is to make the
    trust-root lifecycle query's replayable snapshot track the business
    commit boundary on every backend:

    * the next value is read ``FOR UPDATE`` (locking backends) so
      concurrent creations and retirements in one scope serialize on the
      counter row itself;
    * on SQLite every write transaction already begins as BEGIN
      IMMEDIATE, serializing all writers process-wide;
    * the scope's first allocation inserts the anchor inside a savepoint,
      so the unique-anchor race never rolls the surrounding write back.

    Both creation (stored as ``TrustRoot.commit_seq``) and the terminal
    retirement (stored as ``TrustRoot.retired_seq``) advance this one
    counter in their own transactions, so the counter's last value is the
    scope's total commit high-water mark: a snapshot fixed at that value
    sees neither a later creation (a new row) nor a later retirement (an
    in-place status change), while a fresh first query observes both.
    """

    __tablename__ = "trust_root_commit_counters"

    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    workload_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    # Last per-scope lifecycle sequence handed out; 1 for the scope's
    # first creation, strictly increasing for every later commit.
    last_seq: Mapped[int] = mapped_column(BigInteger)


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


class CertificateRevocationList(Base):
    """One registered, immutable X.509 v2 CRL snapshot for a trust root.

    A snapshot is the whole CRL as published by its issuer: a strictly
    increasing ``crl_number`` (the CRLNumber extension value) with its own
    validity window (``this_update``/``next_update``) and an independently
    stored set of revoked-certificate entries (see
    :class:`CrlRevokedCertificate`). Snapshots are never updated or
    deleted: a strictly higher CRLNumber for the same trust root inserts a
    new snapshot and becomes the current one; an equal number, an equal
    body or a lower/equal number is rejected. Verification for a trust
    root uses only the highest-numbered snapshot whose ``next_update`` has
    not passed at verification time; when the highest snapshot is already
    expired and no newer snapshot exists, verification fails closed with
    500 rather than trusting stale revocation data.

    Only identifiers, the scope, the parsed metadata, a digest of the CRL
    body and timestamps are stored: no CRL or certificate material and no
    free-form text.
    """

    __tablename__ = "certificate_revocation_lists"
    __table_args__ = (
        # Within one trust root each CRLNumber is registered at most once;
        # the constraint makes concurrent equal-number registrations settle
        # as exactly one insert plus stable 409s. Its scoped prefix index
        # also serves the "highest CRLNumber for this trust root" lookup.
        UniqueConstraint(
            "tenant_id",
            "workload_id",
            "trust_root_id",
            "crl_number",
            name="uq_crl_scope_root_number",
        ),
        # Two registrations of a byte-identical CRL body under one root can
        # never both succeed, so a same-content replay is a stable 409.
        UniqueConstraint(
            "tenant_id",
            "workload_id",
            "trust_root_id",
            "crl_sha256",
            name="uq_crl_scope_root_content",
        ),
    )

    crl_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    # The trust root whose subject and public key certify this CRL. CRLs
    # never cross trust roots, tenants or workloads.
    trust_root_id: Mapped[str] = mapped_column(String(36), index=True)
    # Value of the CRLNumber extension; strictly increases per root.
    crl_number: Mapped[int] = mapped_column(BigInteger)
    # SHA-256 hex of the CRL's DER encoding, used for same-content replay
    # detection.
    crl_sha256: Mapped[str] = mapped_column(String(64))
    this_update: Mapped[datetime] = mapped_column(UTCDateTime())
    next_update: Mapped[datetime] = mapped_column(UTCDateTime())
    # Number of revoked entries whose revocationDate had arrived at
    # registration time; recorded once, never recomputed.
    revoked_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class CrlRevokedCertificate(Base):
    """One revoked-certificate entry within a registered CRL snapshot.

    Rows are inserted together with their parent snapshot in one atomic
    transaction and never updated or deleted independently. A certificate
    is identified within its issuer's scope by the issuer distinguished
    name (RFC4514 text, matching the trust-root subject) plus the
    certificate serial number; ``revocation_date`` gates when the entry
    takes effect. Only these parsed values are persisted — no CRL or
    certificate material.
    """

    __tablename__ = "crl_revoked_certificates"
    __table_args__ = (
        # A CRL may list a serial at most once; duplicate serials in the
        # submitted body are rejected at registration, and the constraint
        # guarantees the invariant on concurrent writers as well. It also
        # serves the verification lookup (crl_id + serial_number IN (...)).
        UniqueConstraint(
            "crl_id", "serial_number", name="uq_crl_entry_crl_serial"
        ),
    )

    entry_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    crl_id: Mapped[str] = mapped_column(String(36), index=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    trust_root_id: Mapped[str] = mapped_column(String(36), index=True)
    # RFC4514 rendering of the CRL issuer distinguished name (the trust-
    # root subject); identical for every entry of one snapshot, stored so
    # each row remains self-describing without a join back to the root.
    issuer_dn: Mapped[str] = mapped_column(Text)
    # Revoked certificate serial number as a non-negative integer in
    # canonical decimal text. DER INTEGERs are unbounded (a parsed client
    # CRL is not bound to the 159-bit builder limit), so this is stored as
    # unlimited text rather than a width-bounded varchar or a 64-bit
    # integer; matching compares it against
    # ``str(certificate.serial_number)``.
    serial_number: Mapped[str] = mapped_column(Text)
    revocation_date: Mapped[datetime] = mapped_column(UTCDateTime())


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


class WorkloadIdentityIdempotencyRecord(Base):
    """The first successful idempotency-keyed workload identity registration.

    At most one row exists per ``(tenant_id, workload_id,
    idempotency_key)`` triple; the same key in a different tenant or
    workload is an independent row and never conflicts. The row is
    inserted in the *same* transaction as the :class:`WorkloadIdentityProfile`
    and its :class:`WorkloadIdentityClaim` rows it records, so a crash can
    never leave a profile without its idempotency record or a record
    pointing at no profile. A failed registration — a 404/409/422/500
    judgement or any rollback — inserts no row, so the key stays free and
    a later recovery re-judges the request normally.

    Once committed, a same-key same-scope replay of the same normalized
    request (same trust root and the same claim set, compared as an
    unordered set) answers with the stored first 201 verbatim: it never
    creates a second profile and never rewrites the claims. A same-key
    request whose trust root or claim set differs is a stable 409 that
    changes nothing.

    Only the scope, the caller-chosen key, the SHA-256 digest of the
    normalized request identity (scope, trust root and the canonical
    claim-set fingerprint), the stored first response body and the
    creation time are persisted — never capabilities, payloads, evidence,
    certificate material, private keys or exception detail.
    """

    __tablename__ = "workload_identity_idempotency_records"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workload_id",
            "idempotency_key",
            name="uq_workload_identity_idempotency_scope_key",
        ),
    )

    record_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    # Caller-chosen key: 1..64 visible ASCII characters. Unique only
    # within the (tenant, workload) scope; the same key in another scope
    # is an independent row and never conflicts.
    idempotency_key: Mapped[str] = mapped_column(String(64))
    # Hex SHA-256 over the normalized request identity (canonical
    # tenant_id, workload_id, trust_root_id and the canonical claim-set
    # fingerprint); used only to detect a same-key request carrying a
    # different trust root or claim set, which is a 409 that changes
    # nothing.
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    # The exact first 201 response body, stored verbatim (compact JSON)
    # and replayed byte-for-byte on every later same-key replay.
    response_body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class Policy(Base):
    """A versioned release policy scoped to a tenant and workload.

    Creating a policy with the same (tenant, workload, name) adds a new,
    higher version; older versions are retained and remain addressable by
    id and version so that decisions already taken against them stay
    explainable.

    A version is born active and retires exactly once. Retirement leaves
    the row and its immutable rule in place but terminates new decisions
    against that version; it never touches the decisions already recorded
    against it, other versions, or any release grant.
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
        # Per-scope commit order: the immutable high-water mark that fixes
        # the lifecycle query's replayable snapshot. Unique so two
        # concurrent allocations can never mint the same sequence; together
        # with the per-scope counter row taken FOR UPDATE (and BEGIN
        # IMMEDIATE on SQLite) this makes the order gap-free on every
        # backend, and the index covers the scoped snapshot predicate.
        Index(
            "ix_policies_scope_commit_seq",
            "tenant_id",
            "workload_id",
            "commit_seq",
            unique=True,
        ),
        # Covers the scoped lifecycle listing ordered by the
        # (created_at, policy_id) keyset together with its snapshot cutoff.
        Index(
            "ix_policies_scope_created",
            "tenant_id",
            "workload_id",
            "created_at",
            "policy_id",
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
    # One of POLICY_STATUS_*; active -> retired exactly once. A retired
    # version remains stored (its id and rule still identify the decisions
    # taken against it) but is a terminal state: it never produces a new
    # decision and is never made active again. Other versions, including
    # newer versions of the same name, are independent rows.
    status: Mapped[str] = mapped_column(
        String(16), default=POLICY_STATUS_ACTIVE
    )
    # Set exactly once, by the winning active -> retired transition; a
    # repeated retire observes the stored value and never rewrites it.
    retired_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True
    )
    # Gap-free per-scope sequence allocated in the version's own creation
    # transaction from the shared policy lifecycle counter, strictly
    # increasing in business commit order on every backend. NULL only on
    # rows written before the column existed (backfilled on open); every
    # new version carries a positive value. The lifecycle listing bounds
    # its replayable snapshot membership by this marker, never by write
    # timing.
    commit_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # The lifecycle counter sequence allocated inside the winning
    # active -> retired transaction; NULL while the version is active. The
    # lifecycle query reconstructs a version's status *as of* its fixed
    # snapshot from this marker: retired exactly when retired_seq is at or
    # before the snapshot high-water mark. A retirement that commits after
    # a snapshot's first query therefore never changes a replayed page,
    # even though the row itself is updated in place.
    retired_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class PolicyCommitCounter(Base):
    """Per-scope monotonic allocator for the policy lifecycle sequence.

    Exactly one row exists per ``(tenant_id, workload_id)``. It is an
    internal ordering device — never exposed on any response and holding
    no rule, claim, material or secret — whose sole purpose is to make the
    policy lifecycle query's replayable snapshot track the business commit
    boundary on every backend:

    * the next value is read ``FOR UPDATE`` (locking backends) so
      concurrent version creations and retirements in one scope serialize
      on the counter row itself;
    * on SQLite every write transaction already begins as BEGIN
      IMMEDIATE, serializing all writers process-wide;
    * the scope's first allocation inserts the anchor inside a savepoint,
      so the unique-anchor race never rolls the surrounding write back.

    Both creation (stored as ``Policy.commit_seq``) and the terminal
    retirement (stored as ``Policy.retired_seq``) advance this one counter
    in their own transactions, so the counter's last value is the scope's
    total commit high-water mark: a snapshot fixed at that value sees
    neither a later creation (a new row) nor a later retirement (an
    in-place status change), while a fresh first query observes both.
    """

    __tablename__ = "policy_commit_counters"

    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    workload_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    # Last per-scope lifecycle sequence handed out; 1 for the scope's
    # first creation, strictly increasing for every later commit.
    last_seq: Mapped[int] = mapped_column(BigInteger)


class PolicyIdempotencyRecord(Base):
    """The first successful idempotency-keyed creation of one policy version.

    At most one row exists per ``(tenant_id, workload_id,
    idempotency_key)`` triple; the same key in a different tenant or
    workload is an independent row and never conflicts. The row is
    inserted in the *same* transaction as the :class:`Policy` version it
    records (and that version's lifecycle commit sequence), so a crash
    can never leave a version without its idempotency record or a record
    pointing at no version. A failed creation — a 422/409/500 judgement,
    a lost version-allocation race, or any rollback — inserts no row, so
    the key stays free and a later recovery re-judges the request
    normally.

    Once committed, a same-key same-scope replay of the same normalized
    request answers with the stored first 201 verbatim: it never creates
    a second version, never recomputes a version number or lifecycle
    commit sequence, and never changes the existing policy. A same-key
    request whose name or normalized rule tree differs is a stable 409
    that changes nothing.

    Only the scope, the caller-chosen key, the SHA-256 digest of the
    normalized request identity (canonical name plus the canonical
    normalized rule tree), the stored first response body and the
    creation time are persisted — never evidence, claims material,
    capabilities, keys or exception text.
    """

    __tablename__ = "policy_idempotency_records"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workload_id",
            "idempotency_key",
            name="uq_policy_idempotency_scope_key",
        ),
    )

    record_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    # Caller-chosen key: 1..64 visible ASCII characters. Unique only
    # within the (tenant, workload) scope; the same key in another scope
    # is an independent row and never conflicts.
    idempotency_key: Mapped[str] = mapped_column(String(64))
    # Hex SHA-256 over the normalized request identity (canonical
    # tenant_id, workload_id, name and the canonical serialization of
    # the validated, normalized rule tree); used only to detect a
    # same-key request carrying a different name or rule, which is a 409
    # that changes nothing. Only rule *shape* (claim names, object
    # paths, comparison operators and expected scalars) participates —
    # never evidence or claim values.
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    # The exact first 201 response body, stored verbatim (compact JSON)
    # and replayed byte-for-byte on every later same-key replay.
    response_body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class Decision(Base):
    """An auditable release decision for one evidence/policy-version pair.

    Exactly one row may exist per (evidence, policy version); retries and
    concurrent requests return that same row. The row records only
    identifiers, the fixed status code and timestamps — never the raw
    evidence, the nonce, or the evaluated claims.

    ``commit_seq`` fixes the compliance decision query's replayable
    snapshot to the business commit boundary rather than to
    ``decided_at``: it is a per-scope gap-free sequence allocated inside
    the same transaction as the decision, strictly increasing in commit
    order on every backend (not just SQLite). Two decisions of one scope
    can therefore never share a sequence and a later commit always takes
    the greater one, even when its business time is older or identical;
    the decision query bounds its replayable snapshot by this marker.
    Rows written by older deployments are backfilled (SQLite) or upgraded
    dialect-neutrally, and fresh deployments always populate it.

    The scope columns denormalize the decision's tenant and workload (its
    scope is otherwise reachable only through the evidence row) so a
    scoped compliance scan never has to join or follow another table.
    """

    __tablename__ = "decisions"
    __table_args__ = (
        UniqueConstraint(
            "evidence_id",
            "policy_id",
            name="uq_decision_evidence_policy",
        ),
        # Covers the scoped compliance listing ordered by the
        # (decided_at, decision_id) keyset together with its fixed-snapshot
        # commit_seq cutoff.
        Index(
            "ix_decisions_scope_decided",
            "tenant_id",
            "workload_id",
            "decided_at",
            "decision_id",
        ),
        # Per-scope commit order: the immutable high-water mark that fixes
        # the compliance snapshot to committed business boundaries. Unique
        # so two concurrent allocations can never mint the same sequence;
        # together with the per-scope counter row taken FOR UPDATE (and
        # BEGIN IMMEDIATE on SQLite) this makes the order gap-free on every
        # backend, and the index covers the scoped snapshot predicate.
        Index(
            "ix_decisions_scope_commit_seq",
            "tenant_id",
            "workload_id",
            "commit_seq",
            unique=True,
        ),
    )

    decision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    evidence_id: Mapped[str] = mapped_column(String(36), index=True)
    policy_id: Mapped[str] = mapped_column(String(36), index=True)
    # Snapshot of the evaluated policy version; policy_id already identifies
    # an immutable version, this denormalizes it for audit reads.
    policy_version: Mapped[int] = mapped_column(Integer)
    # One of DECISION_STATUS_*; a fixed, service-defined code only.
    status: Mapped[str] = mapped_column(String(16))
    decided_at: Mapped[datetime] = mapped_column(UTCDateTime())
    # Gap-free per-scope sequence allocated in the decision's own
    # transaction, strictly increasing in business commit order on every
    # backend. NULL only on rows written before the column existed
    # (backfilled on open); every new decision carries a positive value.
    commit_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class DecisionCommitCounter(Base):
    """Per-scope monotonic allocator for decision ``commit_seq``.

    Exactly one row exists per ``(tenant_id, workload_id)``. It is an
    internal ordering device — never exposed on any response and holding
    no business state, material or secret — whose sole purpose is to make
    the compliance decision query's snapshot cutoff the business commit
    boundary on every backend:

    * allocating the next sequence reads this row ``FOR UPDATE`` (locking
      backends) so concurrent decisions in one scope are serialized on the
      counter row itself, not merely on the latest decision (a lock that
      would not gap-lock a new maximum);
    * on SQLite every write transaction already begins as BEGIN
      IMMEDIATE, serializing all writers process-wide;
    * the scope's first allocation inserts the anchor inside a savepoint,
      so the unique ``(tenant_id, workload_id)`` race costs only that
      savepoint and never rolls the surrounding decision back.

    The counter advances in the same transaction as the decision it
    sequences, so a committed decision's sequence is final and strictly
    greater than every decision that committed before it, independent of
    write timing or equal business timestamps.
    """

    __tablename__ = "decision_commit_counters"

    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    workload_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    # Last per-scope commit sequence handed out; 1 for the scope's first
    # decision, strictly increasing thereafter.
    last_seq: Mapped[int] = mapped_column(BigInteger)


class DecisionEvaluationNode(Base):
    """One node of a decision's depth-first policy-rule evaluation.

    Rows are inserted in the same transaction as the :class:`Decision`
    they explain, in pre-order traversal order, so exactly one immutable
    explanation exists per decision and a committed decision never lacks
    one. Decisions recorded by deployments older than the explanation have
    no rows: reading the explanation for such a decision is a defined 409
    (``evaluation not recorded``) rather than a fabricated reconstruction.

    The table stores only node position, structural type and a boolean:

    * ``node_index`` — zero-based pre-order position, unique and
      gap-free within one decision (0 is the root);
    * ``rule_path`` — canonical JSON array of integer child indexes from
      the root (``[]`` for the root, ``[0]`` for the first child of an
      ``all``/``any`` node or the single child of ``not``);
    * ``node_type`` — one of ``leaf``/``all``/``any``/``not``;
    * ``outcome`` — that node's boolean verdict; the root row's outcome
      equals the decision status.

    It never stores claim names, object paths, comparison operators,
    expected scalars, actual claim values, evidence, nonces, capabilities,
    payloads or keys. The explanation is historical: policy retirement,
    policy updates, identity or revocation changes and key rotation never
    rewrite these rows.
    """

    __tablename__ = "decision_evaluation_nodes"

    # The composite primary key orders and identifies the rows: it is
    # unique per (decision, pre-order position), gap-free inserts never
    # collide within a decision, and an explanation read whole for one
    # decision walks the key prefix in node_index order on every backend.
    # It is also the concurrent-insert backstop: alongside the decision's
    # own (evidence, policy) uniqueness it guarantees one decision with
    # one explanation.
    decision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    node_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Canonical JSON text of the integer path array (e.g. ``[]`` or
    # ``[0,1]``); integers and square brackets only — never claim names.
    rule_path: Mapped[str] = mapped_column(String(256))
    node_type: Mapped[str] = mapped_column(String(8))
    outcome: Mapped[bool] = mapped_column(Boolean)


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

    ``commit_seq`` carries no secret and never leaves the service; it is
    the per-scope gap-free sequence allocated inside the creation
    transaction that lets the read-only directory query fix a replayable
    snapshot to the business commit boundary on every backend.
    """

    __tablename__ = "data_envelopes"
    __table_args__ = (
        # Per-scope commit order: the immutable high-water mark that fixes
        # the directory listing's replayable snapshot. Unique so two
        # concurrent creations can never mint the same sequence; together
        # with the per-scope counter row taken FOR UPDATE (and BEGIN
        # IMMEDIATE on SQLite) this makes the order gap-free on every
        # backend, and the index covers the scoped snapshot predicate. The
        # composite primary key already covers the data_id ordering.
        Index(
            "ix_data_envelopes_scope_commit_seq",
            "tenant_id",
            "workload_id",
            "commit_seq",
            unique=True,
        ),
    )
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
    # Gap-free per-scope sequence allocated in the envelope's own creation
    # transaction, strictly increasing in business commit order on every
    # backend. NULL only on rows written before the column existed
    # (backfilled on open); every new envelope carries a positive value.
    # The read-only directory bounds its replayable snapshot membership by
    # this marker, never by write timing.
    commit_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class DataEnvelopeCommitCounter(Base):
    """Per-scope monotonic allocator for data-envelope ``commit_seq``.

    Exactly one row exists per ``(tenant_id, workload_id)``. It is an
    internal ordering device — never exposed on any response and holding
    no payload, key or envelope material — whose sole purpose is to make
    the read-only directory query's snapshot cutoff the business commit
    boundary on every backend:

    * the next value is read ``FOR UPDATE`` (locking backends) so
      concurrent creations in one scope serialize on the counter row
      itself, not merely on the latest envelope (a lock that would not
      gap-lock a new maximum);
    * on SQLite every write transaction already begins as BEGIN
      IMMEDIATE, serializing all writers process-wide;
    * the scope's first allocation inserts the anchor inside a savepoint,
      so the unique-anchor race never rolls the surrounding creation back.

    The counter advances in the same transaction as the envelope it
    sequences, so a committed envelope's sequence is final and strictly
    greater than every envelope that committed before it in the scope,
    independent of write timing or identical ``created_at`` values.
    Rewrap only updates key material in place and never advances this
    counter, so rotations change neither the ordering nor page membership.
    """

    __tablename__ = "data_envelope_commit_counters"

    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    workload_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    # Last per-scope commit sequence handed out; 1 for the scope's first
    # envelope, strictly increasing for every later creation.
    last_seq: Mapped[int] = mapped_column(BigInteger)


class DataEnvelopeIdempotencyRecord(Base):
    """The first successful idempotency-keyed creation of one envelope.

    At most one row exists per ``(tenant_id, workload_id,
    idempotency_key)`` triple; the same key in a different tenant or
    workload is an independent row and never conflicts. The row is
    inserted in the *same* transaction as the :class:`DataEnvelope` it
    records, so a crash can never leave an envelope without its
    idempotency record or a record pointing at no envelope. A failed
    creation — a 422/409/500 judgement, a lost insert race, or any
    rollback — inserts no row, so the key stays free and a later
    recovery re-judges the request normally.

    Once committed, a same-key same-scope replay of the same normalized
    request answers with the stored first 201 verbatim: it never
    re-encrypts, never creates a second envelope, never advances the
    per-scope directory sequence and never changes a key version. A
    same-key request whose data_id or payload differs is a stable 409
    that changes nothing.

    Only the scope, the caller-chosen key, the data_id, the SHA-256
    digest of the payload, the stored first response body and the
    creation time are persisted — never the plaintext payload, the
    plaintext data key, a master key, any ciphertext material or
    exception text.
    """

    __tablename__ = "data_envelope_idempotency_records"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workload_id",
            "idempotency_key",
            name="uq_data_envelope_idempotency_scope_key",
        ),
    )

    record_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    # Caller-chosen key: 1..64 visible ASCII characters. Unique only
    # within the (tenant, workload) scope; the same key in another scope
    # is an independent row and never conflicts.
    idempotency_key: Mapped[str] = mapped_column(String(64))
    # The data_id of the envelope created by the first legal keyed
    # request; part of the request identity a replay must match.
    data_id: Mapped[str] = mapped_column(String(256))
    # Hex SHA-256 of the payload bytes; irreversible, used only to
    # detect a same-key request carrying a different payload, which is
    # a 409 that changes nothing. The plaintext payload never
    # participates and is never stored.
    payload_sha256: Mapped[str] = mapped_column(String(64))
    # The exact first 201 response body, stored verbatim (compact JSON)
    # and replayed byte-for-byte on every later same-key replay.
    response_body: Mapped[str] = mapped_column(Text)
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


class ReleaseGrantEvent(Base):
    """One immutable state-migration event in one release grant's timeline.

    A row is inserted in the *same* committed transaction as the grant
    state migration it records: the empty-state birth into ``pending`` on
    issuance, the winning ``pending -> consumed`` settlement of a consume
    presentation or a payload release (two distinct reasons for the same
    status change), and the winning ``pending -> revoked`` settlement.
    Exactly one event therefore exists per migration that ever committed:
    a request that fails, rolls back, or loses the shared guarded
    settlement race leaves no event, and a consume, release or repeated
    settlement observed against an already-terminal grant appends nothing.
    The table is never updated or deleted by the service.

    The per-grant ``seq`` is a gap-free, stable ordering key allocated in
    the migration transaction (1 for issuance, increasing by exactly one
    per later committed migration), so the recorded order is the order in
    which migrations committed regardless of identical timestamps. Rows
    reconstructed for grants written by an older deployment are numbered
    into the same gap-free run when the database is opened.

    Only identifiers, the sequence number, the old/new status codes, the
    fixed reason code and a UTC timestamp are stored — never the
    capability (plaintext or digest), a decision payload, an evidence
    text, a payload, a key or exception text.
    """

    __tablename__ = "release_grant_events"
    __table_args__ = (
        # The gap-free per-grant sequence, both for MAX+1 allocation and
        # for the timeline listing ordered by seq.
        Index("ix_release_grant_events_grant_seq", "grant_id", "seq", unique=True),
    )

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    grant_id: Mapped[str] = mapped_column(String(36), index=True)
    # Gap-free, immutable, per-grant sequence number in committed-migration
    # order: 1 for issuance, increasing by exactly one per later committed
    # migration. It is the stable listing key.
    seq: Mapped[int] = mapped_column(Integer)
    # Status the grant moved away from; NULL only for the birth migration,
    # whose old state is the empty "no grant" state.
    old_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # One of RELEASE_GRANT_STATUS_*: the status the migration committed.
    new_status: Mapped[str] = mapped_column(String(16))
    # One of RELEASE_GRANT_EVENT_REASON_*; a fixed, service-defined code
    # only. Exception text is never persisted in any column.
    reason: Mapped[str] = mapped_column(String(16))
    # Commit time of the recorded migration.
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime())


class ReleaseGrantConsumeIdempotencyRecord(Base):
    """The first successful idempotency-keyed consumption of one grant.

    At most one row exists per ``(tenant_id, workload_id,
    idempotency_key)`` triple; the same key in a different tenant or
    workload is an independent row and never conflicts. The row is
    inserted in the *same* transaction as the winning pending -> consumed
    grant settlement, its timeline event and its audit event, so a crash
    can never leave a settled grant without its idempotency record or a
    record pointing at an unsettled grant. A failed consume — a 404/401/
    409/410/500 judgement, a lost settlement race, or any rollback —
    inserts no row, so the key stays free and a later recovery re-judges
    the request normally.

    Once committed, a same-key same-scope replay answers with the stored
    first 200 verbatim without touching the grant: the grant is not read
    for re-judgement, no status or timestamp changes, and no event or
    audit row is appended — even if the grant has since expired or
    undergone a later change. A same-key request whose normalized grant,
    scope or capability differs is a stable 409 that changes nothing.

    The stored fingerprint covers only non-sensitive request identity:
    the normalized grant id, the scope and the capability *digest* —
    never the plaintext capability. The stored response is the exact
    compact JSON body first returned to the caller. No capability
    plaintext, evidence, payload or key material is ever stored here.
    """

    __tablename__ = "release_grant_consume_idempotency_records"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workload_id",
            "idempotency_key",
            name="uq_release_grant_consume_idempotency_scope_key",
        ),
    )

    record_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    # Caller-chosen key: 1..64 visible ASCII characters. Unique only
    # within the (tenant, workload) scope; the same key in another scope
    # is an independent row and never conflicts.
    idempotency_key: Mapped[str] = mapped_column(String(64))
    # Grant settled by the first legal keyed consume; every later replay
    # of this key answers with that same consumption's original 200.
    grant_id: Mapped[str] = mapped_column(String(36), index=True)
    # Hex SHA-256 over the normalized request identity (canonical
    # grant_id, tenant_id, workload_id and the capability digest); used
    # only to detect a same-key request with different content, which is
    # a 409 that changes nothing. The plaintext capability never
    # participates.
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    # The exact first 200 response body, stored verbatim (compact JSON)
    # and replayed byte-for-byte on every later same-key replay.
    response_body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


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


class RewrapJobEvent(Base):
    """One immutable lifecycle event for an asynchronous rewrap job.

    A row is inserted in the *same* committed transaction as the
    committed status migration it records: the queued birth on
    submission, the queued/running -> running claim, the post-restart
    failed -> running recovery, the running -> succeeded completion, the
    queued/running -> cancelled settlement, and the running -> failed
    classification. Exactly one event therefore exists per migration that
    ever committed; migrations that lost a guard race rolled back and left
    no event. The table is never updated or deleted by the service, and
    the per-job ``seq`` is a gap-free, stable ordering key allocated in
    the migration transaction, so the recorded order is the order in
    which migrations committed regardless of identical timestamps.

    Only identifiers, the sequence number, the old/new status codes, the
    fixed reason code and a UTC timestamp are stored — never exception
    text, envelope material, payloads, cursor tokens or keys.
    """

    __tablename__ = "rewrap_job_events"
    __table_args__ = (
        # The gap-free per-job sequence, both for allocation and for the
        # event listing ordered by seq.
        Index("ix_rewrap_job_events_job_seq", "job_id", "seq", unique=True),
    )

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    # Gap-free, immutable, per-job sequence number in committed-migration
    # order: 1 for the submission event, increasing by exactly one per
    # later committed migration. It is the stable listing key.
    seq: Mapped[int] = mapped_column(Integer)
    # Status the job moved away from; NULL only for the birth migration,
    # whose old state is "no prior status".
    old_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # One of REWRAP_JOB_STATUS_*: the status the migration committed.
    new_status: Mapped[str] = mapped_column(String(16))
    # One of REWRAP_JOB_EVENT_REASON_*; a fixed, service-defined code
    # only. Exception text is never persisted in any column.
    reason: Mapped[str] = mapped_column(String(16))
    # Commit time of the recorded migration.
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class RewrapJobItem(Base):
    """One committed per-envelope result row of an asynchronous rewrap job.

    A row is inserted in the *same* committed transaction as the envelope
    material change, the append-only rewrap audit event and the job
    progress update it records, so a committed envelope always has exactly
    one item row and a rolled-back envelope attempt (a cancel, a lost
    claim, or a failure settlement) has none. Only successful outcomes are
    ever recorded: an envelope whose rewrap fails settles the job without
    committing, so ``result`` is only ever ``rewrapped`` or ``skipped``
    and the failing envelope itself never appears here — its failure
    classification remains readable only from the job's lifecycle events.
    The table is append-only (rows are never updated or deleted) and the
    per-job ``seq`` is a gap-free, stable ordering key allocated in the
    commit transaction, so the recorded order is the order in which
    envelopes committed, and a page read can never observe duplicates,
    gaps or rewritten history.

    Jobs created before this table existed simply have no rows; their
    item listing is empty and complete. Only identifiers, the sequence
    number, the old/new master key versions, the fixed result code and a
    UTC timestamp are stored — never envelope material, payloads, cursor
    tokens or keys.
    """

    __tablename__ = "rewrap_job_items"
    __table_args__ = (
        # The gap-free per-job sequence, both for MAX+1 allocation and for
        # the item listing ordered by seq.
        Index("ix_rewrap_job_items_job_seq", "job_id", "seq", unique=True),
    )

    item_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("rewrap_jobs.job_id"), index=True
    )
    # Gap-free, immutable, per-job sequence number in committed-envelope
    # order: 1 for the first committed envelope, increasing by exactly one
    # per later committed envelope. It is the stable listing key.
    seq: Mapped[int] = mapped_column(Integer)
    data_id: Mapped[str] = mapped_column(String(256))
    # Master key version recorded on the envelope before this job touched
    # it; identical to new_key_version for skips.
    old_key_version: Mapped[int] = mapped_column(Integer)
    new_key_version: Mapped[int] = mapped_column(Integer)
    # One of REWRAP_RESULT_REWRAPPED / REWRAP_RESULT_SKIPPED; a fixed,
    # service-defined code only. The failure codes never appear here: a
    # failed envelope attempt is rolled back, never committed.
    result: Mapped[str] = mapped_column(String(16))
    # Commit time of the recorded envelope outcome.
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime())


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
        # Per-scope tamper-evident chain order: the immutable position of
        # each committed event in its scope's SHA-256 hash chain. Unique so
        # two concurrent appends can never mint the same sequence; together
        # with the per-scope chain head row taken FOR UPDATE (and BEGIN
        # IMMEDIATE on SQLite) this makes the chain gap-free on every
        # backend, and the index covers the scoped chain walk.
        Index(
            "ix_audit_events_scope_chain_seq",
            "tenant_id",
            "workload_id",
            "chain_seq",
            unique=True,
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
    # Gap-free per-scope chain position allocated in the event's own commit
    # transaction from the scope's chain head, strictly increasing in
    # business commit order on every backend. NULL only on rows written
    # before the chain existed (backfilled on open); every new event
    # carries a positive value.
    chain_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # SHA-256 hex of the previous event in the scope's chain (the all-zero
    # digest for the first); NULL only on pre-chain legacy rows.
    prev_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # SHA-256 hex over the event's non-sensitive fields, its chain
    # position and prev_hash; NULL only on pre-chain legacy rows.
    event_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AuditChainHead(Base):
    """Per-scope head of the compliance audit-event hash chain.

    Exactly one row exists per ``(tenant_id, workload_id)`` once the scope
    has committed its first audit event (or its legacy events were
    backfilled on open). It is an internal integrity device — never exposed
    as a resource and holding no capability, payload, evidence or key —
    whose purposes are:

    * allocating the next ``chain_seq``: the row is read ``FOR UPDATE``
      inside the appending transaction (locking backends) so concurrent
      appends in one scope serialize on the head row itself, and on SQLite
      every write transaction already begins as BEGIN IMMEDIATE;
    * pinning the chain tip: ``last_seq``/``head_hash`` are the sequence
      and event hash of the last committed event, updated in the same
      transaction as that event, so a committed event always extends a
      committed head and a rolled-back event leaves both untouched;
    * recording ``legacy_count``: how many of the scope's events were
      numbered by the first-open migration of a pre-chain database rather
      than appended by live traffic.

    The integrity query reads this row but never writes it; verification
    failures are reported, never repaired.
    """

    __tablename__ = "audit_chain_heads"

    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    workload_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    # Chain position of the scope's last committed event; equals the
    # scope's committed event count because the chain is gap-free from 1.
    last_seq: Mapped[int] = mapped_column(BigInteger)
    # Event hash of the scope's last committed event (the chain tip).
    head_hash: Mapped[str] = mapped_column(String(64))
    # Number of events chained by the first-open legacy migration rather
    # than by live appends; 0 for scopes born after the chain existed.
    legacy_count: Mapped[int] = mapped_column(BigInteger)


class ProofLifecycleEvent(Base):
    """Append-only, tenant-scoped proof-lifecycle audit event.

    A row is inserted in the same committed transaction as the
    attestation stage it records: evidence reception, the first
    verification settlement, or the first policy decision. At most one
    event therefore exists per ``(evidence_id, event_type)`` pair: a
    failed request, a lost settlement race or any rollback leaves no
    event, and a retry of an already-settled stage inserts nothing. The
    table is never updated or deleted by the service.

    Only identifiers, the fixed event type/status codes, the decided
    policy version (for decisions), the associated evidence format (for
    reception) and UTC timestamps are stored. Raw evidence, nonces,
    claims, capabilities, payloads, keys and exception text never have a
    column here: ``policy_version`` and ``evidence_format`` are fixed
    non-sensitive descriptors, not material. Columns that do not apply to
    an event type (``policy_version`` for non-decisions,
    ``evidence_format`` for non-receptions) are stored as NULL.

    ``commit_seq`` fixes the snapshot the first audit query takes to the
    business commit boundary rather than to ``occurred_at``: it is a
    per-scope gap-free sequence allocated inside the same transaction as
    the event, strictly increasing in commit order on every backend (not
    just SQLite). Two events of one scope can therefore never share a
    sequence and a later commit always takes the greater one, even when
    its business time is older or identical; the audit query bounds its
    replayable snapshot by this marker, never by write timing or equal
    timestamps. Rows written by older deployments are backfilled
    (SQLite) and fresh deployments always populate it; it is nullable
    only for such legacy rows.
    """

    __tablename__ = "proof_lifecycle_events"
    __table_args__ = (
        # At most one lifecycle event per evidence and stage; the
        # constraint is what makes a losing concurrent settlement or a
        # retry unable to create a duplicate record.
        UniqueConstraint(
            "evidence_id",
            "event_type",
            name="uq_proof_lifecycle_event_evidence_type",
        ),
        # Covers the scoped listing ordered by the (occurred_at, event_id)
        # keyset, including its exclusive cursor predicate.
        Index(
            "ix_proof_lifecycle_events_scope_occurred",
            "tenant_id",
            "workload_id",
            "occurred_at",
            "event_id",
        ),
        # Per-scope commit order: the immutable high-water mark that fixes
        # the audit snapshot to committed business boundaries. Unique so
        # two concurrent allocations can never mint the same sequence;
        # together with the per-scope counter row taken FOR UPDATE (and
        # BEGIN IMMEDIATE on SQLite) this makes the order gap-free on
        # every backend, and the index covers the scoped snapshot
        # predicate.
        Index(
            "ix_proof_lifecycle_events_scope_commit_seq",
            "tenant_id",
            "workload_id",
            "commit_seq",
            unique=True,
        ),
    )

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256), index=True)
    # One of PROOF_EVENT_TYPE_*; a fixed, service-defined code only.
    event_type: Mapped[str] = mapped_column(String(16))
    # The evidence whose lifecycle stage this event records; all three
    # stages of one proof share this identifier.
    evidence_id: Mapped[str] = mapped_column(String(36), index=True)
    # Gap-free per-scope sequence allocated in the event's own transaction,
    # strictly increasing in business commit order on every backend. NULL
    # only on rows written before the column existed (backfilled on
    # SQLite); every new event carries a positive value.
    commit_seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # Set for proof-decision events (the version decided against); NULL
    # for reception and verification events.
    policy_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Set for proof-received events, recording the associated evidence
    # format; NULL for verification and decision events. Only the format
    # descriptor is kept, never the evidence itself.
    evidence_format: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    # One of PROOF_EVENT_STATUS_*; a fixed, service-defined code only.
    status: Mapped[str] = mapped_column(String(16))
    # Commit time of the recorded stage; the stable listing key.
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)


class ProofEventCommitCounter(Base):
    """Per-scope monotonic allocator for proof-lifecycle ``commit_seq``.

    Exactly one row exists per ``(tenant_id, workload_id)``. It is an
    internal ordering device — never exposed on any response and holding
    no business state, material or secret — whose sole purpose is to make
    the proof audit's snapshot cutoff the business commit boundary on
    every backend:

    * allocating the next sequence reads this row ``FOR UPDATE`` (locking
      backends) so concurrent proofs in one scope are serialized on the
      counter row itself, not merely on the latest event (a lock that
      would not gap-lock a new maximum);
    * on SQLite every write transaction already begins as BEGIN
      IMMEDIATE, serializing all writers process-wide;
    * the scope's first allocation inserts the anchor inside a savepoint,
      so the unique ``(tenant_id, workload_id)`` race costs only that
      savepoint and never rolls the surrounding reception back.

    The counter advances in the same transaction as the event it
    sequences, so a committed event's sequence is final and strictly
    greater than every event that committed before it, independent of
    write timing or equal business timestamps.
    """

    __tablename__ = "proof_event_commit_counters"

    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    workload_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    # Last per-scope commit sequence handed out; 1 for the scope's first
    # event, strictly increasing thereafter.
    last_seq: Mapped[int] = mapped_column(BigInteger)


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


class ChallengeIssuanceCounter(Base):
    """Persistent per-scope admission counter for challenge issuance.

    One row exists per ``(tenant_id, workload_id, window_start)`` triple,
    where ``window_start`` is the UTC natural minute (seconds and
    sub-seconds truncated) in which a *fully validated* ``POST
    /v1/challenges`` request was admitted. The budget is attributed solely
    to the two request scope fields: the challenge id and nonce are minted
    only after admission and never participate in the window. This counter
    is deliberately separate from :class:`RateLimitCounter`: challenge
    issuance is independent of one-time-grant consumption, revocation and
    payload release, so the two never share rows, quota or lock traffic,
    and different tenants or workloads never share a row.

    A slot is reserved in the same transaction that inserts the challenge,
    so the count is durable (it survives process restarts), an admitted
    request can never leave a counter without its challenge or a challenge
    without its counter, and a rejected (over-budget) request writes
    nothing — no counter, no challenge, and the freshly minted nonce is
    neither returned nor persisted. A new UTC minute starts a new row,
    which restores the quota automatically; quotas are never borrowed
    across scopes or minutes.
    """

    __tablename__ = "challenge_issuance_counters"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workload_id",
            "window_start",
            name="uq_challenge_issuance_counter_scope_window",
        ),
    )

    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    workload_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    # UTC minute boundary at which the window starts, truncated to seconds.
    window_start: Mapped[datetime] = mapped_column(UTCDateTime(), primary_key=True)
    # Number of challenges issued during this window; capped at the
    # per-minute issuance budget by the atomic admission transaction.
    count: Mapped[int] = mapped_column(Integer, default=0)


class VerificationAdmissionCounter(Base):
    """Persistent per-scope admission counter for evidence verification.

    One row exists per ``(tenant_id, workload_id, window_start)`` triple,
    where ``window_start`` is the UTC natural minute (seconds and
    sub-seconds truncated) in which a ``POST
    /v1/evidence/{evidence_id}/verify`` request was admitted into the
    verifier. Only requests whose evidence was still ``received`` and whose
    identity, challenge binding, nonce, digest and format checks all passed
    reserve a slot; requests rejected before that point (422/404/401 and
    idempotent replays of a settled evidence) never touch this table. The
    budget is attributed solely to the two scope fields and is deliberately
    separate from both :class:`ChallengeIssuanceCounter` and
    :class:`RateLimitCounter`: verification, challenge issuance and
    one-time-grant actions never share rows, quota or lock traffic, and
    different tenants or workloads never share a row.

    A slot is reserved inside the verification transaction itself, so on
    the success path the count commits atomically with the settlement it
    admitted; when verification fails after the reservation (a verifier
    plugin failure or an unavailable X.509 revocation registry) the
    consumed slot is kept — committed on its own or replayed against the
    original window — and never refunded, while the evidence stays
    ``received`` for a later retry. A rejected (over-budget) request writes
    nothing. A new UTC minute starts a new row, which restores the quota
    automatically; quotas are never borrowed across scopes or minutes.
    """

    __tablename__ = "verification_admission_counters"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workload_id",
            "window_start",
            name="uq_verification_admission_counter_scope_window",
        ),
    )

    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    workload_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    # UTC minute boundary at which the window starts, truncated to seconds.
    window_start: Mapped[datetime] = mapped_column(UTCDateTime(), primary_key=True)
    # Number of requests admitted into the verifier during this window;
    # capped at the per-minute verification budget by the atomic admission
    # transaction.
    count: Mapped[int] = mapped_column(Integer, default=0)


class RewrapJobAdmissionCounter(Base):
    """Persistent per-scope admission counter for rewrap job creation.

    One row exists per ``(tenant_id, workload_id, window_start)`` triple,
    where ``window_start`` is the UTC natural minute (seconds and
    sub-seconds truncated) in which a ``POST /v1/rewrap-jobs`` request
    created a new persistent job. Only requests that genuinely create a
    job reserve a slot: request-body, cursor and idempotency-key shape
    failures (422), same-key replays and conflicts (202/409) and an
    unusable keyring (500) never touch this table. The budget is
    attributed solely to the two scope fields and is deliberately
    separate from :class:`RateLimitCounter`,
    :class:`ChallengeIssuanceCounter` and
    :class:`VerificationAdmissionCounter`: rewrap job admission shares no
    rows, quota or lock traffic with any other budget, and different
    tenants or workloads never share a row.

    A slot is reserved in the same transaction that inserts the queued
    job, its birth event and (for keyed submissions) its idempotency
    record, so the count is durable (it survives process restarts) and a
    crash or failure can leave neither a counter without its job nor a
    job without its counter. A rejected (over-budget) request writes
    nothing. A new UTC minute starts a new row, which restores the quota
    automatically; quotas are never borrowed across scopes or minutes.
    """

    __tablename__ = "rewrap_job_admission_counters"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workload_id",
            "window_start",
            name="uq_rewrap_job_admission_counter_scope_window",
        ),
    )

    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    workload_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    # UTC minute boundary at which the window starts, truncated to seconds.
    window_start: Mapped[datetime] = mapped_column(UTCDateTime(), primary_key=True)
    # Number of jobs created during this window; capped at the per-minute
    # admission budget by the atomic admission transaction.
    count: Mapped[int] = mapped_column(Integer, default=0)
