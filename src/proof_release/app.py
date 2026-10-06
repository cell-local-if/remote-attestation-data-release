from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import threading
import uuid
import queue
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
)
from sqlalchemy import (
    and_,
    create_engine,
    delete,
    event,
    func,
    inspect,
    insert,
    literal_column,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding

from proof_release.db import (
    DECISION_STATUS_ALLOWED,
    DECISION_STATUS_DENIED,
    DECISION_STATUS_CODES,
    RELEASE_GRANT_STATUS_CODES,
    RELEASE_GRANT_STATUS_CONSUMED,
    RELEASE_GRANT_STATUS_PENDING,
    RELEASE_GRANT_STATUS_REVOKED,
    RELEASE_GRANT_EVENT_REASON_ISSUED,
    RELEASE_GRANT_EVENT_REASON_CONSUME,
    RELEASE_GRANT_EVENT_REASON_RELEASE,
    RELEASE_GRANT_EVENT_REASON_REVOKED,
    REWRAP_RESULT_KEYRING,
    REWRAP_RESULT_MISSING_KEY,
    REWRAP_RESULT_REWRAP_FAILED,
    REWRAP_RESULT_REWRAPPED,
    REWRAP_RESULT_SKIPPED,
    REWRAP_JOB_STATUS_FAILED,
    REWRAP_JOB_STATUS_QUEUED,
    REWRAP_JOB_STATUS_RUNNING,
    REWRAP_JOB_STATUS_SUCCEEDED,
    REWRAP_JOB_STATUS_CANCELLED,
    REWRAP_JOB_STATUS_CODES,
    REWRAP_JOB_EVENT_REASON_SUBMITTED,
    REWRAP_JOB_EVENT_REASON_EXECUTED,
    REWRAP_JOB_EVENT_REASON_RECOVERED,
    REWRAP_JOB_EVENT_REASON_COMPLETED,
    REWRAP_JOB_EVENT_REASON_CANCELLED,
    REWRAP_JOB_EVENT_REASON_KEYRING,
    REWRAP_JOB_EVENT_REASON_MISSING_KEY,
    REWRAP_JOB_EVENT_REASON_REWRAP_FAILED,
    VERIFICATION_RESULT_ACCEPTED,
    VERIFICATION_RESULT_REJECTED,
    WORKLOAD_IDENTITY_STATUS_ACTIVE,
    WORKLOAD_IDENTITY_STATUS_REVOKED,
    POLICY_STATUS_ACTIVE,
    POLICY_STATUS_RETIRED,
    TRUST_ROOT_STATUS_ACTIVE,
    TRUST_ROOT_STATUS_RETIRED,
    TRUST_ROOT_STATUS_CODES,
    AUDIT_EVENT_STATUS_CONSUMED,
    AUDIT_EVENT_STATUS_PENDING,
    AUDIT_EVENT_STATUS_REVOKED,
    AUDIT_EVENT_STATUS_REWRAPPED,
    AUDIT_EVENT_STATUS_SKIPPED,
    AUDIT_EVENT_TYPE_CODES,
    AUDIT_EVENT_TYPE_GRANT,
    AUDIT_EVENT_TYPE_REWRAP,
    AUDIT_EVENT_STATUS_CODES,
    PROOF_EVENT_TYPE_CODES,
    PROOF_EVENT_TYPE_RECEIVED,
    PROOF_EVENT_TYPE_VERIFIED,
    PROOF_EVENT_TYPE_DECISION,
    PROOF_EVENT_STATUS_CODES,
    PROOF_EVENT_STATUS_RECEIVED,
    PROOF_EVENT_STATUS_VERIFIED,
    PROOF_EVENT_STATUS_REJECTED,
    PROOF_EVENT_STATUS_ALLOWED,
    PROOF_EVENT_STATUS_DENIED,
    AuditChainHead,
    AuditEvent,
    Base,
    CertificateRevocation,
    CertificateRevocationList,
    CrlRevokedCertificate,
    Challenge,
    ChallengeIssuanceCounter,
    DataEnvelope,
    DataEnvelopeCommitCounter,
    DataEnvelopeIdempotencyRecord,
    Decision,
    DecisionCommitCounter,
    DecisionEvaluationNode,
    Evidence,
    Policy,
    PolicyCommitCounter,
    PolicyIdempotencyRecord,
    ProofEventCommitCounter,
    ProofLifecycleEvent,
    RateLimitCounter,
    ReleaseGrant,
    ReleaseGrantConsumeIdempotencyRecord,
    ReleaseGrantEvent,
    RewrapBatch,
    RewrapBatchItem,
    RewrapJob,
    RewrapJobAdmissionCounter,
    RewrapJobEvent,
    RewrapJobIdempotencyRecord,
    RewrapJobItem,
    TrustRoot,
    TrustRootCommitCounter,
    VerificationAdmissionCounter,
    WorkloadIdentityClaim,
    WorkloadIdentityIdempotencyRecord,
    WorkloadIdentityProfile,
)
from proof_release.envelopes import (
    MasterKeyError,
    b64url_decode,
    b64url_encode,
    decrypt_payload,
    encrypt_payload,
    load_keyring,
    rewrap_data_key,
)
from proof_release.policies import (
    InvalidRule,
    canonical_rule_json,
    diff_rules,
    evaluate_rule,
    explain_rule,
    rule_structure,
    validate_rule,
)
from proof_release.verifiers import (
    ATTESTED_NONCE_JSON,
    X509_ATTESTED_NONCE_JSON,
    ChallengeContext,
    CrlValidationError,
    VerificationContext,
    VerifierPluginError,
    VerifierRegistry,
    default_registry,
    parse_crl,
    validate_crl_against_root,
)

logger = logging.getLogger("proof_release")

DEFAULT_DATABASE_URL = "sqlite:///./proof_release.db"
DATABASE_URL_ENV = "PROOF_RELEASE_DATABASE_URL"

DEFAULT_TTL_SECONDS = 300
MIN_TTL_SECONDS = 30
MAX_TTL_SECONDS = 900
NONCE_BYTES = 32
CAPABILITY_BYTES = 32

#: Bounds and default for a rewrap batch page size.
REWRAP_BATCH_DEFAULT_LIMIT = 50
REWRAP_BATCH_MIN_LIMIT = 1
REWRAP_BATCH_MAX_LIMIT = 200

#: Name of the optional request header carrying an idempotency key on the
#: asynchronous rewrap job submission. A missing header preserves the
#: original "one job per request" semantics; a present key must be 1..64
#: visible ASCII characters and may appear exactly once.
IDEMPOTENCY_KEY_HEADER = "idempotency-key"
IDEMPOTENCY_KEY_MIN_LENGTH = 1
IDEMPOTENCY_KEY_MAX_LENGTH = 64
#: Visible ASCII only: 0x21..0x7E. Spaces, tabs and other control
#: characters are deliberately excluded, so whitespace can never pad a
#: key into validity.
_IDEMPOTENCY_KEY_RE = re.compile(
    rf"^[\x21-\x7e]{{{IDEMPOTENCY_KEY_MIN_LENGTH},{IDEMPOTENCY_KEY_MAX_LENGTH}}}$"
)

#: Shared per-(tenant, workload) budget for the three one-time-grant
#: entry points — grant consumption, grant revocation and payload release.
#: At most this many requests whose basic fields have validated may enter
#: business judgement during one UTC natural minute. The counter is
#: persistent, the quota is shared across all three paths, and a slot is
#: consumed before the request is judged (so a later 404/401/409/410/500
#: never refunds it). A module-level constant rather than configuration so
#: the contract is fixed; tests that exercise settlement atomicity with
#: larger bursts raise it in-process.
GRANT_BUDGET_PER_MINUTE = 5

#: Per-(tenant, workload) issuance budget for ``POST /v1/challenges``: at
#: most this many fully validated challenge-creation requests may be
#: admitted during one UTC natural minute. Only requests whose fields all
#: validate reach the budget, so the existing 422 responses are unchanged
#: and spend nothing. The budget is attributed solely to the request's
#: tenant_id and workload_id — the challenge id and nonce are minted only
#: after admission and never enter the window — and it is fully
#: independent of the one-time-grant budget (different counter table):
#: grant consumption, revocation and payload release do not draw from it.
CHALLENGE_ISSUANCE_BUDGET_PER_MINUTE = 5

#: Per-(tenant, workload) admission budget for ``POST
#: /v1/evidence/{evidence_id}/verify``: at most this many requests may
#: enter the verifier during one UTC natural minute. Only requests whose
#: evidence is still ``received`` and whose identity, challenge binding,
#: nonce, digest and format checks all passed reserve a slot, so every
#: earlier judgement (422/404/401 and idempotent replays of a settled
#: evidence) is unchanged and spends nothing. A reserved slot is durable
#: and is never refunded — in particular a verifier plugin failure or an
#: unavailable X.509 revocation registry (500) keeps it. The budget uses
#: its own counter table: it shares nothing with challenge issuance or
#: the one-time-grant budget.
VERIFICATION_BUDGET_PER_MINUTE = 5

#: Per-(tenant, workload) admission budget for ``POST /v1/rewrap-jobs``:
#: at most this many new persistent rewrap jobs may be created during one
#: UTC natural minute. Only requests that genuinely create a job reserve a
#: slot: every earlier rejection (422 body/cursor/idempotency-key shapes,
#: a same-key replay or conflict, an unusable keyring) is unchanged and
#: spends nothing. The budget uses its own counter table: it shares
#: nothing with challenge issuance, evidence verification or the
#: one-time-grant budget, and it is not surfaced by the observability
#: summary or metrics, which keep their existing three-budget contract.
REWRAP_JOB_BUDGET_PER_MINUTE = 5

#: Fixed page size for the read-only release-grant audit listing. The
#: listing is cursor-driven; the page size is an internal constant and is
#: never part of the request or response contract.
RELEASE_GRANT_AUDIT_PAGE_SIZE = 100

#: Discriminator embedded in release-grant audit cursors so a rewrap-batch
#: cursor (authenticated with the same secret) can never be replayed here
#: and vice versa.
_GRANT_AUDIT_CURSOR_KIND = "release-grant-audit-v1"

#: Environment variable naming the secret used to authenticate rewrap
#: cursors. Cursors are opaque outside the service: each one carries the
#: scope and exclusive data_id boundary it was issued for plus an HMAC, so
#: a forged or cross-scope cursor cannot move a batch outside its range.
#: Unset in local development only; provision through a secrets manager
#: elsewhere.
CURSOR_SECRET_ENV = "PROOF_RELEASE_CURSOR_SECRET"
_DEMO_CURSOR_SECRET = "dev-only-rewrap-cursor-secret"

_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]+$")
#: Cursors are strict unpadded base64url tokens; empty string is reserved
#: for the beginning-of-scope cursor and never travels through this pattern.
_CURSOR_RE = re.compile(r"^[A-Za-z0-9_-]+$")
#: UUID syntax accepted on the batch lookup path before hitting storage.
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _record_rewrap_job_event(
    session,
    *,
    tenant_id: str,
    workload_id: str,
    job_id: str,
    old_status: str | None,
    new_status: str,
    reason: str,
    now: datetime,
) -> None:
    """Append one lifecycle event inside the caller's open transaction.

    The per-job ``seq`` is the current maximum plus one, allocated in the
    same transaction as the status migration it records, so it is gap-free
    and commits atomically with that migration. Every writer transaction
    on SQLite begins as BEGIN IMMEDIATE (and locking backends serialize the
    status-guarded updates), so the MAX+1 allocation can never race
    another migration of the same job. The caller commits (or rolls back)
    the unit of work; a migration whose guarded status update loses is
    rolled back together with its event.
    """
    max_seq = session.scalar(
        select(func.max(RewrapJobEvent.seq)).where(RewrapJobEvent.job_id == job_id)
    )
    session.add(
        RewrapJobEvent(
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            workload_id=workload_id,
            job_id=job_id,
            seq=(max_seq or 0) + 1,
            old_status=old_status,
            new_status=new_status,
            reason=reason,
            created_at=now,
        )
    )


def _record_rewrap_job_item(
    session,
    *,
    tenant_id: str,
    workload_id: str,
    job_id: str,
    data_id: str,
    old_key_version: int,
    new_key_version: int,
    result: str,
    now: datetime,
) -> None:
    """Append one per-envelope result row inside the caller's open transaction.

    The per-job ``seq`` is the current maximum plus one, allocated in the
    same transaction as the envelope material change, the audit event and
    the job progress update it accompanies, so it is gap-free and commits
    atomically with that envelope's outcome. Only the runner holding the
    job's claim can be inside such a transaction (the claim guard makes
    advancement mutually exclusive, and on SQLite every writer transaction
    already begins as BEGIN IMMEDIATE), so the MAX+1 allocation can never
    race another envelope commit of the same job. The caller commits (or
    rolls back) the unit of work; an envelope operation that is discarded
    — a cancel, a lost claim or a failure settlement — leaves no item.
    """
    max_seq = session.scalar(
        select(func.max(RewrapJobItem.seq)).where(RewrapJobItem.job_id == job_id)
    )
    session.add(
        RewrapJobItem(
            item_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            workload_id=workload_id,
            job_id=job_id,
            seq=(max_seq or 0) + 1,
            data_id=data_id,
            old_key_version=old_key_version,
            new_key_version=new_key_version,
            result=result,
            occurred_at=now,
        )
    )


def _record_release_grant_event(
    session,
    *,
    tenant_id: str,
    workload_id: str,
    grant_id: str,
    old_status: str | None,
    new_status: str,
    reason: str,
    now: datetime,
) -> None:
    """Append one per-grant migration event inside the caller's transaction.

    The per-grant ``seq`` is the current maximum plus one, allocated in the
    same transaction as the status migration it records, so it is gap-free
    and commits atomically with that migration. Every writer transaction on
    SQLite begins as BEGIN IMMEDIATE (and locking backends serialize the
    status-guarded updates), so the MAX+1 allocation can never race another
    migration of the same grant. The caller commits (or rolls back) the
    unit of work; a migration whose guarded status update loses is rolled
    back together with its event.
    """
    max_seq = session.scalar(
        select(func.max(ReleaseGrantEvent.seq)).where(
            ReleaseGrantEvent.grant_id == grant_id
        )
    )
    session.add(
        ReleaseGrantEvent(
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            workload_id=workload_id,
            grant_id=grant_id,
            seq=(max_seq or 0) + 1,
            old_status=old_status,
            new_status=new_status,
            reason=reason,
            occurred_at=now,
        )
    )


def _next_scoped_commit_seq(session, tenant_id: str, workload_id: str, counter_model):
    """Allocate the next per-scope gap-free commit sequence.

    Must be called inside the open write transaction of the row being
    sequenced, before that row is added. Returns 1 for the scope's first
    row and a strictly greater value for every later one. The value is
    fixed by the business commit boundary, not by the business timestamp:

    * the per-scope counter row is read ``FOR UPDATE``, so on locking
      backends concurrent writers in one scope serialize on that anchor
      row and each waiter observes the preceding winner's new maximum —
      this is robust even though a lock on the current last row would not
      gap-lock the next maximum;
    * on SQLite every write transaction already begins as BEGIN
      IMMEDIATE, fully serializing allocations;
    * the scope's first row has no counter yet, so it is inserted inside a
      savepoint: two racing first allocations lose only the savepoint
      (never the surrounding transaction), and the loser re-reads the
      winner's row. The bounded loop also covers the rare case where the
      apparent winner rolled its whole transaction back: the loser simply
      attempts the anchor again instead of failing.

    The counter advances in the same transaction as the row it sequences,
    so a later commit always takes a greater sequence regardless of write
    timing or identical business timestamps — the property the fixed
    snapshot queries bound their replayable membership by.
    """
    for _ in range(10):
        counter = session.scalar(
            select(counter_model)
            .where(
                counter_model.tenant_id == tenant_id,
                counter_model.workload_id == workload_id,
            )
            .with_for_update()
        )
        if counter is not None:
            counter.last_seq = counter.last_seq + 1
            return counter.last_seq
        # First row for this scope: install the anchor in a savepoint so a
        # collision with a concurrent first row rolls back only this
        # insert, leaving the caller's transaction intact.
        try:
            with session.begin_nested():
                session.add(
                    counter_model(
                        tenant_id=tenant_id, workload_id=workload_id, last_seq=1
                    )
                )
            return 1
        except IntegrityError:
            # begin_nested() has already released the rolled-back
            # savepoint; loop to re-read the winner's anchor (or retry the
            # insert if that winner's whole transaction rolled back).
            continue
    # pragma: no cover - bounded backstop for pathological anchor contention
    logger.error("scoped commit sequence could not settle")
    raise HTTPException(status_code=500, detail="commit sequencing unavailable")


def _next_proof_event_commit_seq(
    session, tenant_id: str, workload_id: str
) -> int:
    """Allocate the next per-scope proof-lifecycle commit sequence."""
    return _next_scoped_commit_seq(
        session, tenant_id, workload_id, ProofEventCommitCounter
    )


def _next_decision_commit_seq(session, tenant_id: str, workload_id: str) -> int:
    """Allocate the next per-scope decision commit sequence."""
    return _next_scoped_commit_seq(
        session, tenant_id, workload_id, DecisionCommitCounter
    )


def _next_trust_root_commit_seq(session, tenant_id: str, workload_id: str) -> int:
    """Allocate the next per-scope trust-root lifecycle sequence.

    Both creation (``TrustRoot.commit_seq``) and the terminal retirement
    (``TrustRoot.retired_seq``) draw from this one shared counter inside
    their own transactions, so the counter's last value is the scope's
    total commit high-water mark: the read-only lifecycle query fixes its
    replayable snapshot at that value and reconstructs each root's status
    as-of the snapshot from ``retired_seq``, isolating a replayed page
    from both later creations and later retirements.
    """
    return _next_scoped_commit_seq(
        session, tenant_id, workload_id, TrustRootCommitCounter
    )


def _next_policy_commit_seq(session, tenant_id: str, workload_id: str) -> int:
    """Allocate the next per-scope policy lifecycle sequence.

    Both version creation (``Policy.commit_seq``) and the terminal
    retirement (``Policy.retired_seq``) draw from this one shared counter
    inside their own transactions, so the counter's last value is the
    scope's total commit high-water mark: the read-only lifecycle query
    fixes its replayable snapshot at that value and reconstructs each
    version's status as-of the snapshot from ``retired_seq``, isolating a
    replayed page from both later creations and later retirements.
    """
    return _next_scoped_commit_seq(
        session, tenant_id, workload_id, PolicyCommitCounter
    )


def _next_data_envelope_commit_seq(
    session, tenant_id: str, workload_id: str
) -> int:
    """Allocate the next per-scope data-envelope creation sequence.

    Only envelope creation advances this counter; a rewrap rotates key
    material in place and never draws from it, so rotations change neither
    the directory ordering nor page membership. The read-only directory
    query bounds a replayable snapshot by this commit high-water mark,
    immune to envelopes created after the first page even when their
    ``data_id`` would sort earlier.
    """
    return _next_scoped_commit_seq(
        session, tenant_id, workload_id, DataEnvelopeCommitCounter
    )


def _rfc3339(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


#: Prev-hash carried by a scope's first audit event and the head_hash of a
#: fresh chain anchor: 64 lowercase hex zeros (the SHA-256 hex width).
_AUDIT_CHAIN_GENESIS_HASH = "0" * 64

#: Finite, service-defined set of audit-chain integrity failure codes.
#: ``sequence-gap`` means the per-scope chain sequence is not the
#: contiguous 1..N run (a deleted, renumbered, duplicated or unchained
#: event); ``hash-mismatch`` means an event's stored hash does not
#: recompute from its non-sensitive fields and its predecessor's hash (a
#: modified or reordered event); ``head-mismatch`` means the recorded
#: per-scope range head disagrees with the verified chain tip.
AUDIT_CHAIN_FAILURE_SEQUENCE_GAP = "sequence-gap"
AUDIT_CHAIN_FAILURE_HASH_MISMATCH = "hash-mismatch"
AUDIT_CHAIN_FAILURE_HEAD_MISMATCH = "head-mismatch"


def _audit_event_chain_hash(
    *,
    chain_seq: int,
    prev_hash: str,
    tenant_id: str,
    workload_id: str,
    event_id: str,
    event_type: str,
    grant_id: str | None,
    decision_id: str | None,
    data_id: str | None,
    status: str,
    capability_sha256: str | None,
    occurred_at: datetime,
) -> str:
    """SHA-256 chain hash of one audit event.

    Covers the event's non-sensitive stored fields — identifiers, the
    fixed type/status codes, the capability *digest* (never the capability
    itself), the commit timestamp and the chain position — plus the
    preceding event's hash, so modifying, deleting or reordering any event
    changes every later link. The canonical payload is compact JSON with
    sorted keys and the same UTC RFC3339 timestamp spelling on write,
    migration and verify, so a hash computed at commit time recomputes
    byte-identically from the stored row.
    """
    payload = json.dumps(
        {
            "capability_sha256": capability_sha256,
            "chain_seq": chain_seq,
            "data_id": data_id,
            "decision_id": decision_id,
            "event_id": event_id,
            "event_type": event_type,
            "grant_id": grant_id,
            "occurred_at": _rfc3339(occurred_at),
            "prev_hash": prev_hash,
            "status": status,
            "tenant_id": tenant_id,
            "workload_id": workload_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _audit_chain_head_for_update(
    session, tenant_id: str, workload_id: str
) -> AuditChainHead:
    """Return the scope's chain head row, creating the anchor when absent.

    Must be called inside the open write transaction of the event being
    chained. The head row doubles as the per-scope sequence allocator:

    * an existing head is read ``FOR UPDATE``, so on locking backends
      concurrent appends in one scope serialize on the head row itself and
      each waiter observes the preceding winner's advanced head;
    * on SQLite every write transaction already begins as BEGIN IMMEDIATE,
      fully serializing appends;
    * the scope's first event has no head yet, so the genesis anchor is
      inserted inside a savepoint: two racing first appends lose only the
      savepoint (never the surrounding transition), and the loser re-reads
      the winner's row. The bounded loop also covers the rare case where
      the apparent winner rolled its whole transaction back.
    """
    for _ in range(10):
        head = session.scalar(
            select(AuditChainHead)
            .where(
                AuditChainHead.tenant_id == tenant_id,
                AuditChainHead.workload_id == workload_id,
            )
            .with_for_update()
        )
        if head is not None:
            return head
        # First event for this scope: install the genesis anchor in a
        # savepoint so a collision with a concurrent first event rolls
        # back only this insert, leaving the caller's transaction intact.
        try:
            with session.begin_nested():
                head = AuditChainHead(
                    tenant_id=tenant_id,
                    workload_id=workload_id,
                    last_seq=0,
                    head_hash=_AUDIT_CHAIN_GENESIS_HASH,
                    legacy_count=0,
                )
                session.add(head)
            return head
        except IntegrityError:
            # begin_nested() has already released the rolled-back
            # savepoint; loop to re-read the winner's anchor (or retry the
            # insert if that winner's whole transaction rolled back).
            continue
    # pragma: no cover - bounded backstop for pathological anchor contention
    logger.error("audit chain head could not settle")
    raise HTTPException(status_code=500, detail="audit chain unavailable")


def _append_audit_event(
    session,
    *,
    tenant_id: str,
    workload_id: str,
    event_type: str,
    grant_id: str | None,
    decision_id: str | None,
    data_id: str | None,
    status: str,
    capability_sha256: str | None,
    occurred_at: datetime,
) -> None:
    """Append one compliance audit event and extend the scope's hash chain.

    The chain metadata (per-scope sequence, previous-event hash and this
    event's hash) and the advanced range head are written in the caller's
    open transaction together with the state transition being recorded:
    a rolled-back transition leaves no event, no chain metadata and no
    head movement, and a committed one always leaves all three. Only
    identifiers, the fixed codes, the timestamp and SHA-256 digests enter
    the chain — never a capability, payload, evidence or key.
    """
    head = _audit_chain_head_for_update(session, tenant_id, workload_id)
    chain_seq = head.last_seq + 1
    prev_hash = head.head_hash
    event_id = str(uuid.uuid4())
    event_hash = _audit_event_chain_hash(
        chain_seq=chain_seq,
        prev_hash=prev_hash,
        tenant_id=tenant_id,
        workload_id=workload_id,
        event_id=event_id,
        event_type=event_type,
        grant_id=grant_id,
        decision_id=decision_id,
        data_id=data_id,
        status=status,
        capability_sha256=capability_sha256,
        occurred_at=occurred_at,
    )
    head.last_seq = chain_seq
    head.head_hash = event_hash
    session.add(
        AuditEvent(
            event_id=event_id,
            tenant_id=tenant_id,
            workload_id=workload_id,
            event_type=event_type,
            grant_id=grant_id,
            decision_id=decision_id,
            data_id=data_id,
            status=status,
            capability_sha256=capability_sha256,
            occurred_at=occurred_at,
            chain_seq=chain_seq,
            prev_hash=prev_hash,
            event_hash=event_hash,
        )
    )


def _verify_audit_event_chain(rows, head) -> dict:
    """Verify one scope's audit chain and return the integrity report.

    ``rows`` are the scope's committed audit events (any order) and
    ``head`` its :class:`AuditChainHead` row or ``None``. The walk checks
    the chain sequence (a contiguous 1..N run), every event hash
    (recomputed from the stored non-sensitive fields and the predecessor's
    hash) and the recorded range head, reporting the first problem found
    in chain order. It is purely a function of the snapshot: it writes
    nothing, repairs nothing and never advances the head.
    """
    event_count = len(rows)
    legacy_count = head.legacy_count if head is not None else 0

    by_seq: dict[int, object] = {}
    sequence_broken = False
    for row in rows:
        if row.chain_seq is None or row.chain_seq in by_seq:
            # A committed event always carries a unique chain sequence.
            sequence_broken = True
        else:
            by_seq[row.chain_seq] = row

    first_seq = min(by_seq, default=0)
    last_seq = max(by_seq, default=0)
    head_hash = head.head_hash if head is not None else None

    failure_code: str | None = None
    failure_seq: int | None = None

    if sequence_broken:
        failure_code = AUDIT_CHAIN_FAILURE_SEQUENCE_GAP
        failure_seq = 0
    else:
        prev_hash = _AUDIT_CHAIN_GENESIS_HASH
        tip_hash = _AUDIT_CHAIN_GENESIS_HASH
        expected = 1
        for seq in sorted(by_seq):
            if seq != expected:
                failure_code = AUDIT_CHAIN_FAILURE_SEQUENCE_GAP
                failure_seq = expected
                break
            row = by_seq[seq]
            if row.prev_hash is None or row.event_hash is None:
                # A committed event always carries its chain hashes.
                failure_code = AUDIT_CHAIN_FAILURE_HASH_MISMATCH
                failure_seq = seq
                break
            recomputed = _audit_event_chain_hash(
                chain_seq=seq,
                prev_hash=row.prev_hash,
                tenant_id=row.tenant_id,
                workload_id=row.workload_id,
                event_id=row.event_id,
                event_type=row.event_type,
                grant_id=row.grant_id,
                decision_id=row.decision_id,
                data_id=row.data_id,
                status=row.status,
                capability_sha256=row.capability_sha256,
                occurred_at=row.occurred_at,
            )
            if row.prev_hash != prev_hash or row.event_hash != recomputed:
                failure_code = AUDIT_CHAIN_FAILURE_HASH_MISMATCH
                failure_seq = seq
                break
            prev_hash = row.event_hash
            tip_hash = row.event_hash
            expected += 1
        else:
            # Sequence and hashes verified; the recorded range head must
            # agree with the chain tip (an empty scope only with a
            # never-advanced or absent head).
            if head is None:
                if by_seq:
                    failure_code = AUDIT_CHAIN_FAILURE_HEAD_MISMATCH
                    failure_seq = last_seq
            elif head.last_seq != last_seq or head.head_hash != tip_hash:
                failure_code = AUDIT_CHAIN_FAILURE_HEAD_MISMATCH
                failure_seq = head.last_seq

    return {
        "valid": failure_code is None,
        "event_count": event_count,
        "legacy_count": legacy_count,
        "first_seq": first_seq,
        "last_seq": last_seq,
        "head_hash": head_hash,
        "failure_code": failure_code,
        "failure_seq": failure_seq,
    }



def _rfc3339_z(value: datetime) -> str:
    """UTC RFC3339 with the ``Z`` designator rather than the ``+00:00`` offset."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_minute_window(now: datetime) -> datetime:
    """Return the UTC natural-minute boundary containing ``now``.

    Seconds and sub-second fields are truncated, so every instant within a
    UTC calendar minute maps to the same window start.
    """
    return now.replace(second=0, microsecond=0)


def _seconds_until_next_minute(now: datetime) -> int:
    """Whole seconds from ``now`` to the next UTC minute boundary, ceil, ≥1.

    The value is a positive integer computed at response time: an exact
    boundary reports 60 and any later instant in the same minute reports a
    smaller value, so repeated 429s may carry decreasing values without
    extending the window.
    """
    remaining = (60 - now.second) - (now.microsecond / 1_000_000)
    return max(1, int(math.ceil(remaining)))


def _too_many_requests_response(now: datetime) -> Response:
    """Build the compact 429 body: one positive-int field plus a newline."""
    body = (
        json.dumps(
            {"retry_after_seconds": _seconds_until_next_minute(now)},
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    return Response(
        content=body, status_code=429, media_type="application/json"
    )


def _verification_rate_limited_response() -> Response:
    """Build the verification-admission 429: fixed detail plus Retry-After.

    The body carries only the fixed ``detail`` string and the whole seconds
    until the next UTC minute travel in the ``Retry-After`` header,
    recomputed at response time so repeated rejections may carry decreasing
    values without extending the window.
    """
    return JSONResponse(
        status_code=429,
        content={"detail": "verification rate limit exceeded"},
        headers={"Retry-After": str(_seconds_until_next_minute(_utcnow()))},
    )


def _rewrap_job_rate_limited_response() -> Response:
    """Build the rewrap-job-admission 429: fixed detail plus Retry-After.

    The body carries only the fixed ``detail`` string and the whole seconds
    until the next UTC minute travel in the ``Retry-After`` header,
    recomputed at response time so repeated rejections may carry decreasing
    values without extending the window.
    """
    return JSONResponse(
        status_code=429,
        content={"detail": "rewrap job rate limit exceeded"},
        headers={"Retry-After": str(_seconds_until_next_minute(_utcnow()))},
    )



def _nonce_digest(nonce: str) -> str:
    return hashlib.sha256(nonce.encode("ascii")).hexdigest()


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must be a non-empty string")
    return value


def _nonce_format(value: str) -> str:
    if not _NONCE_RE.fullmatch(value):
        raise ValueError("nonce must be unpadded base64url")
    return value


def _optional_non_blank(value: str | None) -> str | None:
    if value is None:
        return value
    return _require_non_blank(value)


def _cursor_secret() -> bytes:
    return os.environ.get(CURSOR_SECRET_ENV, _DEMO_CURSOR_SECRET).encode("utf-8")


def _encode_cursor(tenant_id: str, workload_id: str, data_id: str) -> str:
    """Build an opaque, scope-bound exclusive cursor for ``data_id``.

    The token is unpadded base64url of a JSON payload naming the scope
    and boundary, authenticated with HMAC-SHA256. Nothing about the
    boundary is secret, but the MAC makes a forged or tampered cursor
    (including one replayed against a different tenant/workload)
    unrecognizable.
    """
    payload = json.dumps(
        {"t": tenant_id, "w": workload_id, "d": data_id},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    mac = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    return b64url_encode(payload + mac)


def _decode_cursor(
    token: str, tenant_id: str, workload_id: str
) -> str | None:
    """Validate a cursor and return its exclusive data_id boundary.

    Returns ``None`` for a malformed/forged token or one minted for any
    other scope. ``""`` (the beginning-of-scope marker) yields ``""``.
    """
    if token == "":
        return ""
    if not _CURSOR_RE.fullmatch(token):
        return None
    try:
        raw = b64url_decode(token)
    except ValueError:
        return None
    # The MAC is a fixed 32-byte suffix; the JSON payload precedes it.
    if len(raw) <= 32:
        return None
    payload, mac = raw[:-32], raw[-32:]
    expected = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected):
        return None
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if not hmac.compare_digest(str(decoded.get("t", "")), tenant_id):
        return None
    if not hmac.compare_digest(str(decoded.get("w", "")), workload_id):
        return None
    boundary = decoded.get("d")
    if not isinstance(boundary, str):
        return None
    return boundary


def _read_idempotency_key(request: Request) -> tuple[bool, str | None]:
    """Extract the optional ``Idempotency-Key`` header for a job submit.

    Returns ``(present, value)``. ``present`` is False when the header is
    absent entirely (the original one-job-per-request semantics apply and
    no validation runs). When present, the header must occur exactly once
    and its value must be 1..64 visible ASCII characters; a duplicated
    header line, an empty value, surrounding whitespace, a non-ASCII or
    control character, or an over-long value raises an indistinguishable
    422 before any state is read or written.
    """
    raw_values = request.headers.getlist(IDEMPOTENCY_KEY_HEADER)
    if not raw_values:
        return False, None
    if len(raw_values) != 1 or not _IDEMPOTENCY_KEY_RE.fullmatch(raw_values[0]):
        raise HTTPException(status_code=422, detail="invalid idempotency key")
    return True, raw_values[0]


def _release_grant_consume_fingerprint(
    grant_id: str, tenant_id: str, workload_id: str, capability_digest: str
) -> str:
    """Hash the request identity that must match for a keyed consume replay.

    Covers exactly the equivalence range fixed by the contract: the
    normalized (canonical lowercase) grant id, the tenant, the workload
    and the capability digest. Only this non-sensitive identity is
    hashed — the plaintext capability never participates and is never
    recoverable from the digest. Canonical JSON with sorted keys makes
    equivalent requests hash identically.
    """
    payload = json.dumps(
        {
            "grant_id": grant_id,
            "tenant_id": tenant_id,
            "workload_id": workload_id,
            "capability_digest": capability_digest,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _release_grant_consumed_body(
    grant_id: str, decision_id: str, data_id: str, consumed_at: datetime
) -> str:
    """Render the exact compact 200 consume body stored for replay.

    This is the wire form persisted verbatim by the first successful
    idempotency-keyed consume and returned byte-for-byte on every later
    same-key replay (including the original ``consumed_at``), so its
    serialization must never depend on response-time clock or state.
    """
    return json.dumps(
        {
            "grant_id": grant_id,
            "decision_id": decision_id,
            "data_id": data_id,
            "consumed": True,
            "consumed_at": _rfc3339(consumed_at),
        },
        separators=(",", ":"),
        allow_nan=False,
    )


def _data_envelope_created_body(
    data_id: str,
    tenant_id: str,
    workload_id: str,
    key_version: int,
    created_at: datetime,
) -> str:
    """Render the exact compact 201 creation body stored for replay.

    This is the wire form persisted verbatim by the first
    idempotency-keyed envelope creation and returned byte-for-byte on
    every later same-key replay (including the original ``created_at``
    and ``key_version``), so its serialization must never depend on
    response-time clock or state. The field order and compact shape
    match the unkeyed response model exactly.
    """
    return json.dumps(
        {
            "data_id": data_id,
            "tenant_id": tenant_id,
            "workload_id": workload_id,
            "key_version": key_version,
            "created_at": _rfc3339(created_at),
        },
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )


def _policy_request_fingerprint(
    tenant_id: str, workload_id: str, name: str, rule_json: str
) -> str:
    """Hash the request identity a keyed policy replay must match.

    Covers exactly the equivalence range fixed by the contract: the
    scope (tenant, workload), the policy name, and the canonical
    serialization of the validated rule tree. Rules that normalize to
    the same canonical JSON (e.g. equivalent ``in`` candidate order is
    not normalized away, but rule key ordering and whitespace are) hash
    identically; a different name or a different normalized rule tree
    hashes differently. Only non-sensitive rule identity (claim names,
    object paths, comparison operators and expected scalars) is hashed
    — never evidence, claim values, capabilities, keys or exception
    text. Canonical JSON with sorted keys makes equivalent requests
    hash identically.
    """
    payload = json.dumps(
        {
            "tenant_id": tenant_id,
            "workload_id": workload_id,
            "name": name,
            "rule": rule_json,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _policy_created_body(
    policy_id: str,
    tenant_id: str,
    workload_id: str,
    name: str,
    version: int,
    rule: dict,
    created_at: datetime,
) -> str:
    """Render the exact compact 201 policy body stored for replay.

    This is the wire form persisted verbatim by the first successful
    idempotency-keyed creation and returned byte-for-byte on every later
    same-key replay (including the original ``policy_id``, ``version``
    and ``created_at``), so its serialization must never depend on
    response-time clock or state. The field order and compact shape
    match the unkeyed response model exactly.
    """
    return json.dumps(
        {
            "policy_id": policy_id,
            "tenant_id": tenant_id,
            "workload_id": workload_id,
            "name": name,
            "version": version,
            "rule": rule,
            "created_at": _rfc3339(created_at),
        },
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )


def _rewrap_job_request_fingerprint(
    tenant_id: str, workload_id: str, limit: int, start_cursor: str
) -> str:
    """Hash the request shape that must match for a keyed replay.

    Covers the scope, the effective page size and the normalized cursor
    start (an omitted cursor and an explicit empty cursor both normalize
    to ``""``; any other start is the verified token verbatim). Only this
    non-sensitive shape is hashed — no payload, capability, key or
    certificate material participates. Canonical JSON with sorted keys
    makes equivalent requests hash identically.
    """
    payload = json.dumps(
        {
            "tenant_id": tenant_id,
            "workload_id": workload_id,
            "limit": limit,
            "cursor": start_cursor,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rewrap_job_acceptance_body(
    job_id: str, limit: int, start_cursor: str, now: datetime
) -> str:
    """Render the exact compact 202 acceptance body, newline included.

    This is the wire form stored verbatim by the first idempotency-keyed
    submission and returned byte-for-byte on every later replay, so its
    formatting must never depend on mutable state.
    """
    return (
        json.dumps(
            {
                "job_id": job_id,
                "status": REWRAP_JOB_STATUS_QUEUED,
                "limit": limit,
                "cursor": start_cursor,
                "created_at": _rfc3339(now),
                "updated_at": _rfc3339(now),
            },
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        )
        + "\n"
    )


def _reserve_rewrap_job_slot(
    session, tenant_id: str, workload_id: str, window_start: datetime
) -> bool:
    """Reserve one slot of the per-scope rewrap-job admission budget.

    The reservation happens inside the caller's open transaction — the
    same one that inserts the queued job, its birth event and (for keyed
    submissions) its idempotency record — so the count commits or rolls
    back together with the job it admitted and no half state is possible.
    Returns ``True`` when one durable slot was reserved for the window,
    ``False`` when the minute's budget is exhausted (nothing is written
    then). A counter read or write that cannot complete propagates to the
    caller, whose rollback leaves neither a half job nor a half count.
    """
    for _ in range(2):
        # Lock the scope's minute row when one exists. SQLite ignores FOR
        # UPDATE but every write transaction already begins as BEGIN
        # IMMEDIATE, serializing concurrent admissions process-wide; on
        # locking backends the row lock orders them so exactly the
        # budgeted number of requests in the minute can be admitted.
        row = session.scalar(
            select(RewrapJobAdmissionCounter)
            .where(
                RewrapJobAdmissionCounter.tenant_id == tenant_id,
                RewrapJobAdmissionCounter.workload_id == workload_id,
                RewrapJobAdmissionCounter.window_start == window_start,
            )
            .with_for_update()
        )
        if row is None:
            # The first admitted job of the minute initializes the counter
            # at one, inside a savepoint so a concurrent initializer on a
            # locking backend costs only the savepoint and is retried as
            # an increment below.
            try:
                with session.begin_nested():
                    session.add(
                        RewrapJobAdmissionCounter(
                            tenant_id=tenant_id,
                            workload_id=workload_id,
                            window_start=window_start,
                            count=1,
                        )
                    )
            except IntegrityError:
                continue
            return True
        if row.count >= REWRAP_JOB_BUDGET_PER_MINUTE:
            # Budget exhausted: no counter write; the caller writes
            # nothing either and recomputes the retry hint at response
            # time.
            return False
        row.count = row.count + 1
        return True
    # Defensive: the unique-insert retry loop failed to settle, which the
    # single retry above makes unreachable.
    logger.error("rewrap job rate-limit reservation could not settle")
    raise HTTPException(status_code=500, detail="rate limit unavailable")


def _parse_utc_rfc3339(value: str) -> datetime:
    """Parse a timestamp that explicitly denotes UTC (zero offset).

    Accepts RFC3339 forms, including a trailing ``Z``. A naive timestamp
    (no offset) or a non-UTC offset is rejected rather than silently
    assumed or converted, so audit-window bounds are always unambiguous.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("timestamp must be UTC RFC3339") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be UTC RFC3339")
    return parsed.astimezone(timezone.utc)


def _certificate_fingerprint_format(value: str) -> str:
    """Validate a certificate fingerprint: unpadded base64url of 32 bytes.

    The digest is re-encoded after decoding so non-canonical spellings of
    the same 32 bytes cannot register as two distinct fingerprints.
    """
    try:
        raw = b64url_decode(value)
    except ValueError as exc:
        raise ValueError(
            "certificate_fingerprint must be unpadded base64url of 32 bytes"
        ) from exc
    if len(raw) != 32 or b64url_encode(raw) != value:
        raise ValueError(
            "certificate_fingerprint must be unpadded base64url of 32 bytes"
        )
    return value


def _canonical_uuid(value: str) -> str:
    """Validate a canonical lowercase UUID string."""
    if not _UUID_RE.fullmatch(value):
        raise ValueError("must be a canonical UUID")
    return value


def _utc_rfc3339_field(value: str) -> str:
    """Validate a UTC RFC3339 timestamp field, returning its normal form."""
    try:
        parsed = _parse_utc_rfc3339(value)
    except ValueError as exc:
        raise ValueError("must be a UTC RFC3339 timestamp") from exc
    return _rfc3339(parsed)


def _ordered_chain_certificates(evidence: str) -> list | None:
    """Parse the certificate chain of an X.509 evidence document.

    Returns the certificates in chain order (leaf first, root last), or
    ``None`` when the document or any certificate cannot be parsed — such
    evidence falls through to the verifier, which rejects it on its own.
    """
    try:
        document = json.loads(evidence)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    chain_pems = document.get("certificate_chain")
    if not isinstance(chain_pems, list) or not chain_pems:
        return None
    certificates = []
    for pem in chain_pems:
        if not isinstance(pem, str):
            return None
        try:
            certificates.append(
                x509.load_pem_x509_certificate(pem.encode("utf-8"))
            )
        except (ValueError, TypeError):
            return None
    return certificates


def _canonical_claim_set(
    claims: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, str]], str]:
    """Normalize an identity claim set and fingerprint it as a set.

    Claims are compared as an *unordered set*: duplicate claims collapse
    and submission order is irrelevant, so registrations that name the
    same claims in a different order or with repeats describe the same
    profile. Returns the de-duplicated claims in first-seen order (for
    storage/response) plus the SHA-256 hex of the canonical JSON of the
    sorted set, which is the stable identity used for duplicate detection.
    """
    seen: set[tuple[str, str, str]] = set()
    ordered: list[tuple[str, str, str]] = []
    for claim in claims:
        if claim not in seen:
            seen.add(claim)
            ordered.append(claim)
    canonical = json.dumps(
        [
            {"issuer": issuer, "subject": subject, "uri": uri}
            for issuer, subject, uri in sorted(ordered)
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ordered, hashlib.sha256(canonical).hexdigest()


def _workload_identity_request_fingerprint(
    tenant_id: str, workload_id: str, trust_root_id: str, claims_fingerprint: str
) -> str:
    """Hash the request identity that must match for a keyed replay.

    Covers exactly the equivalence range fixed by the contract: the
    tenant, the workload, the trust root and the canonical claim-set
    fingerprint (claims already compared as an unordered, de-duplicated
    set). Only this non-sensitive identity is hashed — no certificate
    material or claim value participates beyond its canonical comparison
    string. Canonical JSON with sorted keys makes equivalent requests
    hash identically.
    """
    payload = json.dumps(
        {
            "tenant_id": tenant_id,
            "workload_id": workload_id,
            "trust_root_id": trust_root_id,
            "claims_fingerprint": claims_fingerprint,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _workload_identity_created_body(
    profile_id: str,
    tenant_id: str,
    workload_id: str,
    trust_root_id: str,
    claims: list[tuple[str, str, str]],
    created_at: datetime,
) -> str:
    """Render the exact compact 201 registration body stored for replay.

    This is the wire form persisted verbatim by the first
    idempotency-keyed registration and returned byte-for-byte on every
    later same-key replay (including the original ``profile_id`` and
    ``created_at``), so its serialization must never depend on
    response-time clock or state. The field order and compact shape
    match the unkeyed response exactly.
    """
    return json.dumps(
        {
            "profile_id": profile_id,
            "tenant_id": tenant_id,
            "workload_id": workload_id,
            "trust_root_id": trust_root_id,
            "claims": [
                {"issuer": issuer, "subject": subject, "uri": uri}
                for issuer, subject, uri in claims
            ],
            "created_at": _rfc3339(created_at),
        },
        separators=(",", ":"),
        allow_nan=False,
    )


def _identity_json(payload) -> Response:
    """Compact identity-response JSON with a single terminating newline.

    Every value is a string, list, boolean or UTC timestamp string (plus
    ``null`` for absent timestamps): no floats, ``-0.0`` or non-finite
    values can ever appear (``allow_nan=False`` makes that explicit).
    """
    body = (
        json.dumps(
            payload, separators=(",", ":"), allow_nan=False, ensure_ascii=False
        ).encode("utf-8")
        + b"\n"
    )
    return Response(content=body, media_type="application/json")


def _leaf_identity_strings(
    certificate: x509.Certificate,
) -> tuple[str, str, tuple[str, ...]]:
    """Return a leaf certificate's parsed identity comparison strings.

    The issuer and subject distinguished names are rendered as RFC4514
    strings (the same text stored on registered claims), alongside every
    URI subjectAlternativeName. Comparison is always on these parsed
    strings, never on raw certificate bytes.
    """
    issuer = certificate.issuer.rfc4514_string()
    subject = certificate.subject.rfc4514_string()
    try:
        san = certificate.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
    except x509.ExtensionNotFound:
        uris: tuple[str, ...] = ()
    else:
        uris = tuple(san.get_values_for_type(x509.UniformResourceIdentifier))
    return issuer, subject, uris


def _grant_audit_cursor_payload(
    tenant_id: str,
    workload_id: str,
    boundary: str,
    *,
    grant_id: str,
    decision_id: str,
    data_id: str,
    status: str,
    issued_after: str,
    issued_before: str,
) -> bytes:
    """Canonical byte payload authenticated inside a grant-audit cursor.

    Every active filter is part of the payload, so a cursor minted for one
    filter set cannot be replayed against another. The embedded kind tag
    distinguishes these cursors from rewrap-batch cursors even though both
    are HMAC-authenticated with the same secret.
    """
    return json.dumps(
        {
            "k": _GRANT_AUDIT_CURSOR_KIND,
            "t": tenant_id,
            "w": workload_id,
            "g": boundary,
            "gi": grant_id,
            "di": decision_id,
            "da": data_id,
            "s": status,
            "a": issued_after,
            "b": issued_before,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_grant_audit_cursor(
    tenant_id: str,
    workload_id: str,
    boundary: str,
    *,
    grant_id: str,
    decision_id: str,
    data_id: str,
    status: str,
    issued_after: str,
    issued_before: str,
) -> str:
    """Build an opaque, scope- and filter-bound exclusive grant cursor."""
    payload = _grant_audit_cursor_payload(
        tenant_id,
        workload_id,
        boundary,
        grant_id=grant_id,
        decision_id=decision_id,
        data_id=data_id,
        status=status,
        issued_after=issued_after,
        issued_before=issued_before,
    )
    mac = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    return b64url_encode(payload + mac)


def _decode_grant_audit_cursor(
    token: str,
    tenant_id: str,
    workload_id: str,
    *,
    grant_id: str,
    decision_id: str,
    data_id: str,
    status: str,
    issued_after: str,
    issued_before: str,
) -> str | None:
    """Validate a grant-audit cursor and return its exclusive grant boundary.

    Returns ``None`` for a malformed/forged token, a cursor of another
    kind (e.g. a rewrap-batch cursor), or one minted for any other scope
    or filter combination. The beginning-of-scope marker (``""``) never
    reaches this function: it is handled by the caller.
    """
    if not _CURSOR_RE.fullmatch(token):
        return None
    try:
        raw = b64url_decode(token)
    except ValueError:
        return None
    # The MAC is a fixed 32-byte suffix; the JSON payload precedes it.
    if len(raw) <= 32:
        return None
    payload, mac = raw[:-32], raw[-32:]
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if decoded.get("k") != _GRANT_AUDIT_CURSOR_KIND:
        return None
    boundary = decoded.get("g")
    if not isinstance(boundary, str) or boundary == "":
        # A resume cursor always names the last returned grant; an empty
        # boundary is only valid as the implicit start, which is not encoded.
        return None
    # Recompute the expected MAC over the decoded payload exactly as it
    # was signed, then verify it in constant time.
    expected_mac = hmac.new(
        _cursor_secret(),
        _grant_audit_cursor_payload(
            tenant_id,
            workload_id,
            boundary,
            grant_id=grant_id,
            decision_id=decision_id,
            data_id=data_id,
            status=status,
            issued_after=issued_after,
            issued_before=issued_before,
        ),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(mac, expected_mac):
        return None
    # Defense in depth: the MAC already covers every field, but confirm
    # the scope and each active filter explicitly.
    if not hmac.compare_digest(str(decoded.get("t", "")), tenant_id):
        return None
    if not hmac.compare_digest(str(decoded.get("w", "")), workload_id):
        return None
    for key, expected in (
        ("gi", grant_id),
        ("di", decision_id),
        ("da", data_id),
        ("s", status),
        ("a", issued_after),
        ("b", issued_before),
    ):
        if not hmac.compare_digest(str(decoded.get(key, "")), expected):
            return None
    return boundary


def _audit_event_cursor_payload(
    tenant_id: str,
    workload_id: str,
    boundary_at: str,
    boundary_event: str,
    *,
    event_id: str,
    event_type: str,
    status: str,
    occurred_after: str,
    occurred_before: str,
) -> bytes:
    """Canonical byte payload authenticated inside an audit-event cursor.

    The cursor marks an exclusive ``(occurred_at, event_id)`` position and
    every active filter is part of the signed payload, so a cursor minted
    for one filter set cannot be replayed against another. The kind tag
    distinguishes these cursors from the rewrap and grant-audit families
    even though all share the same HMAC secret.
    """
    return json.dumps(
        {
            "k": _AUDIT_EVENT_CURSOR_KIND,
            "t": tenant_id,
            "w": workload_id,
            "at": boundary_at,
            "e": boundary_event,
            "id": event_id,
            "ty": event_type,
            "s": status,
            "a": occurred_after,
            "b": occurred_before,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_audit_event_cursor(
    tenant_id: str,
    workload_id: str,
    boundary_at: str,
    boundary_event: str,
    *,
    event_id: str,
    event_type: str,
    status: str,
    occurred_after: str,
    occurred_before: str,
) -> str:
    """Build an opaque, scope- and filter-bound exclusive event cursor."""
    payload = _audit_event_cursor_payload(
        tenant_id,
        workload_id,
        boundary_at,
        boundary_event,
        event_id=event_id,
        event_type=event_type,
        status=status,
        occurred_after=occurred_after,
        occurred_before=occurred_before,
    )
    mac = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    return b64url_encode(payload + mac)


def _decode_audit_event_cursor(
    token: str,
    tenant_id: str,
    workload_id: str,
    *,
    event_id: str,
    event_type: str,
    status: str,
    occurred_after: str,
    occurred_before: str,
) -> tuple[str, str] | None:
    """Validate an audit-event cursor and return its exclusive boundary.

    Returns ``(occurred_at, event_id)`` on success or ``None`` for a
    malformed/forged token, a cursor of another kind (rewrap or grant
    audit), or one minted for any other scope or filter combination. The
    beginning-of-scope marker (``""``) never reaches this function.
    """
    if not _CURSOR_RE.fullmatch(token):
        return None
    try:
        raw = b64url_decode(token)
    except ValueError:
        return None
    # The MAC is a fixed 32-byte suffix; the JSON payload precedes it.
    if len(raw) <= 32:
        return None
    payload, mac = raw[:-32], raw[-32:]
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if decoded.get("k") != _AUDIT_EVENT_CURSOR_KIND:
        return None
    boundary_at = decoded.get("at")
    boundary_event = decoded.get("e")
    if not isinstance(boundary_at, str) or boundary_at == "":
        return None
    if not isinstance(boundary_event, str) or not _UUID_RE.fullmatch(boundary_event):
        return None
    expected_mac = hmac.new(
        _cursor_secret(),
        _audit_event_cursor_payload(
            tenant_id,
            workload_id,
            boundary_at,
            boundary_event,
            event_id=event_id,
            event_type=event_type,
            status=status,
            occurred_after=occurred_after,
            occurred_before=occurred_before,
        ),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(mac, expected_mac):
        return None
    # Defense in depth: the MAC already covers every field, but confirm
    # the scope and each active filter explicitly.
    if not hmac.compare_digest(str(decoded.get("t", "")), tenant_id):
        return None
    if not hmac.compare_digest(str(decoded.get("w", "")), workload_id):
        return None
    for key, expected in (
        ("id", event_id),
        ("ty", event_type),
        ("s", status),
        ("a", occurred_after),
        ("b", occurred_before),
    ):
        if not hmac.compare_digest(str(decoded.get(key, "")), expected):
            return None
    return boundary_at, boundary_event


def _revocation_cursor_payload(
    tenant_id: str,
    workload_id: str,
    trust_root_id: str,
    boundary_at: str,
    boundary_revocation: str,
    *,
    revocation_id: str,
    certificate_fingerprint: str,
    effective_after: str,
    effective_before: str,
    snapshot_at: str,
    snapshot_id: str,
) -> bytes:
    """Canonical byte payload authenticated inside a revocation cursor.

    The cursor marks an exclusive ``(effective_at, revocation_id)``
    position and every active filter plus the fixed replayable snapshot
    high-water mark ``(created_at, revocation_id)`` are part of the
    signed payload, so a cursor minted for one filter set or snapshot
    cannot be replayed against another. The kind tag distinguishes these
    cursors from the rewrap, grant-audit and compliance audit-event
    families even though all share the same HMAC secret.
    """
    return json.dumps(
        {
            "k": _REVOCATION_CURSOR_KIND,
            "t": tenant_id,
            "w": workload_id,
            "r": trust_root_id,
            "at": boundary_at,
            "id": boundary_revocation,
            "ri": revocation_id,
            "fp": certificate_fingerprint,
            "a": effective_after,
            "b": effective_before,
            "sa": snapshot_at,
            "si": snapshot_id,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_revocation_cursor(
    tenant_id: str,
    workload_id: str,
    trust_root_id: str,
    boundary_at: str,
    boundary_revocation: str,
    *,
    revocation_id: str,
    certificate_fingerprint: str,
    effective_after: str,
    effective_before: str,
    snapshot_at: str,
    snapshot_id: str,
) -> str:
    """Build an opaque, scope- and filter-bound exclusive revocation cursor."""
    payload = _revocation_cursor_payload(
        tenant_id,
        workload_id,
        trust_root_id,
        boundary_at,
        boundary_revocation,
        revocation_id=revocation_id,
        certificate_fingerprint=certificate_fingerprint,
        effective_after=effective_after,
        effective_before=effective_before,
        snapshot_at=snapshot_at,
        snapshot_id=snapshot_id,
    )
    mac = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    return b64url_encode(payload + mac)


def _decode_revocation_cursor(
    token: str,
    tenant_id: str,
    workload_id: str,
    trust_root_id: str,
    *,
    revocation_id: str,
    certificate_fingerprint: str,
    effective_after: str,
    effective_before: str,
) -> tuple[str, str, str, str] | None:
    """Validate a revocation cursor and return its exclusive boundary.

    Returns ``(effective_at, revocation_id, snapshot_at, snapshot_id)``
    on success or ``None`` for a malformed/forged token, a cursor of
    another kind (rewrap, grant audit or compliance audit events), or one
    minted for any other scope, filter combination or snapshot. The
    beginning-of-range marker (``""``) never reaches this function.
    """
    if not _CURSOR_RE.fullmatch(token):
        return None
    try:
        raw = b64url_decode(token)
    except ValueError:
        return None
    # The MAC is a fixed 32-byte suffix; the JSON payload precedes it.
    if len(raw) <= 32:
        return None
    payload, mac = raw[:-32], raw[-32:]
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if decoded.get("k") != _REVOCATION_CURSOR_KIND:
        return None
    boundary_at = decoded.get("at")
    boundary_revocation = decoded.get("id")
    if not isinstance(boundary_at, str) or boundary_at == "":
        return None
    if not isinstance(boundary_revocation, str) or not _UUID_RE.fullmatch(
        boundary_revocation
    ):
        return None
    snapshot_at = decoded.get("sa")
    snapshot_id = decoded.get("si")
    # A resume cursor always names the snapshot high-water mark
    # established by the range's first query.
    if not isinstance(snapshot_at, str) or snapshot_at == "":
        return None
    if not isinstance(snapshot_id, str) or not _UUID_RE.fullmatch(snapshot_id):
        return None
    expected_mac = hmac.new(
        _cursor_secret(),
        _revocation_cursor_payload(
            tenant_id,
            workload_id,
            trust_root_id,
            boundary_at,
            boundary_revocation,
            revocation_id=revocation_id,
            certificate_fingerprint=certificate_fingerprint,
            effective_after=effective_after,
            effective_before=effective_before,
            snapshot_at=snapshot_at,
            snapshot_id=snapshot_id,
        ),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(mac, expected_mac):
        return None
    # Defense in depth: the MAC already covers every field, but confirm
    # the scope and each active filter explicitly.
    if not hmac.compare_digest(str(decoded.get("t", "")), tenant_id):
        return None
    if not hmac.compare_digest(str(decoded.get("w", "")), workload_id):
        return None
    if not hmac.compare_digest(str(decoded.get("r", "")), trust_root_id):
        return None
    for key, expected in (
        ("ri", revocation_id),
        ("fp", certificate_fingerprint),
        ("a", effective_after),
        ("b", effective_before),
    ):
        if not hmac.compare_digest(str(decoded.get(key, "")), expected):
            return None
    return boundary_at, boundary_revocation, snapshot_at, snapshot_id


def _rewrap_job_history_cursor_payload(
    tenant_id: str,
    workload_id: str,
    boundary_job: str,
    *,
    job_id: str,
    status: str,
    created_after: str,
    created_before: str,
) -> bytes:
    """Canonical byte payload authenticated inside a job-history cursor.

    The cursor marks an exclusive ``job_id`` position and every active
    filter is part of the signed payload, so a cursor minted for one
    scope or filter set cannot be replayed against another. The kind tag
    distinguishes these cursors from the rewrap, grant-audit, compliance
    audit-event and revocation families even though all share the same
    HMAC secret.
    """
    return json.dumps(
        {
            "k": _REWRAP_JOB_HISTORY_CURSOR_KIND,
            "t": tenant_id,
            "w": workload_id,
            "j": boundary_job,
            "id": job_id,
            "s": status,
            "a": created_after,
            "b": created_before,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_rewrap_job_history_cursor(
    tenant_id: str,
    workload_id: str,
    boundary_job: str,
    *,
    job_id: str,
    status: str,
    created_after: str,
    created_before: str,
) -> str:
    """Build an opaque, scope- and filter-bound exclusive history cursor."""
    payload = _rewrap_job_history_cursor_payload(
        tenant_id,
        workload_id,
        boundary_job,
        job_id=job_id,
        status=status,
        created_after=created_after,
        created_before=created_before,
    )
    mac = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    return b64url_encode(payload + mac)


def _decode_rewrap_job_history_cursor(
    token: str,
    tenant_id: str,
    workload_id: str,
    *,
    job_id: str,
    status: str,
    created_after: str,
    created_before: str,
) -> str | None:
    """Validate a job-history cursor and return its exclusive job boundary.

    Returns the canonical boundary job id on success or ``None`` for a
    malformed/forged token, a cursor of another kind (rewrap batch, grant
    audit, compliance audit events or revocations), or one minted for any
    other scope or filter combination. The beginning marker (``""``)
    never reaches this function.
    """
    if not _CURSOR_RE.fullmatch(token):
        return None
    try:
        raw = b64url_decode(token)
    except ValueError:
        return None
    # The MAC is a fixed 32-byte suffix; the JSON payload precedes it.
    if len(raw) <= 32:
        return None
    payload, mac = raw[:-32], raw[-32:]
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if decoded.get("k") != _REWRAP_JOB_HISTORY_CURSOR_KIND:
        return None
    boundary_job = decoded.get("j")
    if not isinstance(boundary_job, str) or not _UUID_RE.fullmatch(boundary_job):
        return None
    expected_mac = hmac.new(
        _cursor_secret(),
        _rewrap_job_history_cursor_payload(
            tenant_id,
            workload_id,
            boundary_job,
            job_id=job_id,
            status=status,
            created_after=created_after,
            created_before=created_before,
        ),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(mac, expected_mac):
        return None
    # Defense in depth: the MAC already covers every field, but confirm
    # the scope and each active filter explicitly.
    if not hmac.compare_digest(str(decoded.get("t", "")), tenant_id):
        return None
    if not hmac.compare_digest(str(decoded.get("w", "")), workload_id):
        return None
    for key, expected in (
        ("id", job_id),
        ("s", status),
        ("a", created_after),
        ("b", created_before),
    ):
        if not hmac.compare_digest(str(decoded.get(key, "")), expected):
            return None
    return boundary_job


def _rewrap_job_event_cursor_payload(
    tenant_id: str,
    workload_id: str,
    job_id: str,
    boundary_seq: int,
    *,
    snapshot_seq: int,
) -> bytes:
    """Canonical byte payload authenticated inside a job-event cursor.

    The cursor marks an exclusive per-job ``seq`` position and carries the
    fixed replayable snapshot high-water mark (the greatest ``seq`` the
    range's first query saw) in the signed payload, so a cursor minted for
    one scope, job or snapshot cannot be replayed against another. The
    kind tag distinguishes these cursors from every other cursor family
    even though all share the same HMAC secret.
    """
    return json.dumps(
        {
            "k": _REWRAP_JOB_EVENT_CURSOR_KIND,
            "t": tenant_id,
            "w": workload_id,
            "j": job_id,
            "q": boundary_seq,
            "h": snapshot_seq,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_rewrap_job_event_cursor(
    tenant_id: str,
    workload_id: str,
    job_id: str,
    boundary_seq: int,
    *,
    snapshot_seq: int,
) -> str:
    """Build an opaque, scope/job/snapshot-bound exclusive event cursor."""
    payload = _rewrap_job_event_cursor_payload(
        tenant_id,
        workload_id,
        job_id,
        boundary_seq,
        snapshot_seq=snapshot_seq,
    )
    mac = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    return b64url_encode(payload + mac)


def _decode_rewrap_job_event_cursor(
    token: str,
    tenant_id: str,
    workload_id: str,
    job_id: str,
) -> tuple[int, int] | None:
    """Validate a job-event cursor and return ``(boundary_seq, snapshot_seq)``.

    Returns the exclusive sequence boundary and the fixed snapshot
    high-water mark on success, or ``None`` for a malformed/forged token,
    a cursor of another kind (rewrap batch, grant audit, compliance audit
    events, revocations or job history), or one minted for any other
    scope or job. The beginning marker (``""``) never reaches this
    function.
    """
    if not _CURSOR_RE.fullmatch(token):
        return None
    try:
        raw = b64url_decode(token)
    except ValueError:
        return None
    # The MAC is a fixed 32-byte suffix; the JSON payload precedes it.
    if len(raw) <= 32:
        return None
    payload, mac = raw[:-32], raw[-32:]
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if decoded.get("k") != _REWRAP_JOB_EVENT_CURSOR_KIND:
        return None
    boundary_seq = decoded.get("q")
    snapshot_seq = decoded.get("h")
    # Sequence positions are positive ints (bools are rejected as ints).
    if not isinstance(boundary_seq, int) or isinstance(boundary_seq, bool):
        return None
    if not isinstance(snapshot_seq, int) or isinstance(snapshot_seq, bool):
        return None
    if boundary_seq < 1 or snapshot_seq < boundary_seq:
        return None
    expected_mac = hmac.new(
        _cursor_secret(),
        _rewrap_job_event_cursor_payload(
            tenant_id,
            workload_id,
            job_id,
            boundary_seq,
            snapshot_seq=snapshot_seq,
        ),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(mac, expected_mac):
        return None
    # Defense in depth: the MAC already covers every field, but confirm
    # the scope and job explicitly.
    if not hmac.compare_digest(str(decoded.get("t", "")), tenant_id):
        return None
    if not hmac.compare_digest(str(decoded.get("w", "")), workload_id):
        return None
    if not hmac.compare_digest(str(decoded.get("j", "")), job_id):
        return None
    return boundary_seq, snapshot_seq


def _rewrap_job_item_cursor_payload(
    tenant_id: str,
    workload_id: str,
    job_id: str,
    boundary_seq: int,
) -> bytes:
    """Canonical byte payload authenticated inside a job-item cursor.

    The cursor marks an exclusive per-job ``seq`` position and the scope
    and job are part of the signed payload, so a cursor minted for one
    job's item listing can never be replayed against another scope, job or
    pagination family. The kind tag distinguishes these cursors from every
    other cursor family even though all share the same HMAC secret.
    """
    return json.dumps(
        {
            "k": _REWRAP_JOB_ITEM_CURSOR_KIND,
            "t": tenant_id,
            "w": workload_id,
            "j": job_id,
            "q": boundary_seq,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_rewrap_job_item_cursor(
    tenant_id: str,
    workload_id: str,
    job_id: str,
    boundary_seq: int,
) -> str:
    """Build an opaque, scope- and job-bound exclusive item cursor."""
    payload = _rewrap_job_item_cursor_payload(
        tenant_id, workload_id, job_id, boundary_seq
    )
    mac = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    return b64url_encode(payload + mac)


def _decode_rewrap_job_item_cursor(
    token: str,
    tenant_id: str,
    workload_id: str,
    job_id: str,
) -> int | None:
    """Validate a job-item cursor and return its exclusive seq boundary.

    Returns the exclusive sequence boundary on success or ``None`` for a
    malformed/forged token, a cursor of another kind (rewrap batch, grant
    audit, compliance audit events, revocations, job history or job
    events), or one minted for any other scope or job. The beginning
    marker (``""``) never reaches this function.
    """
    if not _CURSOR_RE.fullmatch(token):
        return None
    try:
        raw = b64url_decode(token)
    except ValueError:
        return None
    # The MAC is a fixed 32-byte suffix; the JSON payload precedes it.
    if len(raw) <= 32:
        return None
    payload, mac = raw[:-32], raw[-32:]
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if decoded.get("k") != _REWRAP_JOB_ITEM_CURSOR_KIND:
        return None
    boundary_seq = decoded.get("q")
    # Sequence positions are positive ints (bools are rejected as ints).
    if not isinstance(boundary_seq, int) or isinstance(boundary_seq, bool):
        return None
    if boundary_seq < 1:
        return None
    expected_mac = hmac.new(
        _cursor_secret(),
        _rewrap_job_item_cursor_payload(
            tenant_id, workload_id, job_id, boundary_seq
        ),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(mac, expected_mac):
        return None
    # Defense in depth: the MAC already covers every field, but confirm
    # the scope and job explicitly.
    if not hmac.compare_digest(str(decoded.get("t", "")), tenant_id):
        return None
    if not hmac.compare_digest(str(decoded.get("w", "")), workload_id):
        return None
    if not hmac.compare_digest(str(decoded.get("j", "")), job_id):
        return None
    return boundary_seq


def _release_grant_event_cursor_payload(
    tenant_id: str,
    workload_id: str,
    grant_id: str,
    boundary_seq: int,
    *,
    snapshot_seq: int,
) -> bytes:
    """Canonical byte payload authenticated inside a grant-event cursor.

    The cursor marks an exclusive per-grant ``seq`` position and carries
    the fixed replayable snapshot high-water mark (the greatest ``seq``
    the range's first query saw) in the signed payload, so a cursor minted
    for one scope, grant or snapshot cannot be replayed against another.
    The kind tag distinguishes these cursors from every other cursor
    family even though all share the same HMAC secret.
    """
    return json.dumps(
        {
            "k": _RELEASE_GRANT_EVENT_CURSOR_KIND,
            "t": tenant_id,
            "w": workload_id,
            "g": grant_id,
            "q": boundary_seq,
            "h": snapshot_seq,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_release_grant_event_cursor(
    tenant_id: str,
    workload_id: str,
    grant_id: str,
    boundary_seq: int,
    *,
    snapshot_seq: int,
) -> str:
    """Build an opaque, scope/grant/snapshot-bound exclusive event cursor."""
    payload = _release_grant_event_cursor_payload(
        tenant_id,
        workload_id,
        grant_id,
        boundary_seq,
        snapshot_seq=snapshot_seq,
    )
    mac = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    return b64url_encode(payload + mac)


def _decode_release_grant_event_cursor(
    token: str,
    tenant_id: str,
    workload_id: str,
    grant_id: str,
) -> tuple[int, int] | None:
    """Validate a grant-event cursor and return ``(boundary_seq, snapshot_seq)``.

    Returns the exclusive sequence boundary and the fixed snapshot
    high-water mark on success, or ``None`` for a malformed/forged token,
    a cursor of another kind (rewrap batch, grant audit, compliance audit
    events, proof-lifecycle events, decisions, revocations, trust roots,
    policies or rewrap job listings), or one minted for any other scope
    or grant. The beginning marker (``""``) never reaches this function.
    """
    if not _CURSOR_RE.fullmatch(token):
        return None
    try:
        raw = b64url_decode(token)
    except ValueError:
        return None
    # The MAC is a fixed 32-byte suffix; the JSON payload precedes it.
    if len(raw) <= 32:
        return None
    payload, mac = raw[:-32], raw[-32:]
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if decoded.get("k") != _RELEASE_GRANT_EVENT_CURSOR_KIND:
        return None
    boundary_seq = decoded.get("q")
    snapshot_seq = decoded.get("h")
    # Sequence positions are positive ints (bools are rejected as ints).
    if not isinstance(boundary_seq, int) or isinstance(boundary_seq, bool):
        return None
    if not isinstance(snapshot_seq, int) or isinstance(snapshot_seq, bool):
        return None
    if boundary_seq < 1 or snapshot_seq < boundary_seq:
        return None
    expected_mac = hmac.new(
        _cursor_secret(),
        _release_grant_event_cursor_payload(
            tenant_id,
            workload_id,
            grant_id,
            boundary_seq,
            snapshot_seq=snapshot_seq,
        ),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(mac, expected_mac):
        return None
    # Defense in depth: the MAC already covers every field, but confirm
    # the scope and grant explicitly.
    if not hmac.compare_digest(str(decoded.get("t", "")), tenant_id):
        return None
    if not hmac.compare_digest(str(decoded.get("w", "")), workload_id):
        return None
    if not hmac.compare_digest(str(decoded.get("g", "")), grant_id):
        return None
    return boundary_seq, snapshot_seq


def _proof_event_cursor_payload(
    tenant_id: str,
    workload_id: str,
    boundary_at: str,
    boundary_event: str,
    *,
    evidence_id: str,
    event_type: str,
    status: str,
    occurred_after: str,
    occurred_before: str,
    snapshot_seq: int,
) -> bytes:
    """Canonical byte payload authenticated inside a proof-event cursor.

    The cursor marks an exclusive ``(occurred_at, event_id)`` position and
    every active filter plus the fixed replayable snapshot are part of the
    signed payload, so a cursor minted for one filter set or snapshot
    cannot be replayed against another. The snapshot membership cutoff is
    the per-scope ``commit_seq`` high-water mark (``q``): the business
    commit boundary established by the first query. Bounding membership by
    commit order — rather than by ``occurred_at`` — is what keeps an event
    that commits *after* the first query but carries an older or identical
    business time out of the fixed snapshot, on every backend. The kind
    tag distinguishes these cursors from every other cursor family even
    though all share the same HMAC secret.
    """
    return json.dumps(
        {
            "k": _PROOF_EVENT_CURSOR_KIND,
            "t": tenant_id,
            "w": workload_id,
            "at": boundary_at,
            "e": boundary_event,
            "id": evidence_id,
            "ty": event_type,
            "s": status,
            "a": occurred_after,
            "b": occurred_before,
            "q": snapshot_seq,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_proof_event_cursor(
    tenant_id: str,
    workload_id: str,
    boundary_at: str,
    boundary_event: str,
    *,
    evidence_id: str,
    event_type: str,
    status: str,
    occurred_after: str,
    occurred_before: str,
    snapshot_seq: int,
) -> str:
    """Build an opaque, scope/filter/snapshot-bound exclusive proof cursor."""
    payload = _proof_event_cursor_payload(
        tenant_id,
        workload_id,
        boundary_at,
        boundary_event,
        evidence_id=evidence_id,
        event_type=event_type,
        status=status,
        occurred_after=occurred_after,
        occurred_before=occurred_before,
        snapshot_seq=snapshot_seq,
    )
    mac = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    return b64url_encode(payload + mac)


def _decode_proof_event_cursor(
    token: str,
    tenant_id: str,
    workload_id: str,
    *,
    evidence_id: str,
    event_type: str,
    status: str,
    occurred_after: str,
    occurred_before: str,
) -> tuple[str, str, int] | None:
    """Validate a proof-event cursor and return its exclusive boundary.

    Returns ``(occurred_at, event_id, snapshot_seq)`` on success or
    ``None`` for a malformed/forged token, a cursor of another kind
    (rewrap batch, grant audit, compliance audit events, revocations or
    rewrap job listings), or one minted for any other scope, filter
    combination or snapshot. The beginning marker (``""``) never reaches
    this function.
    """
    if not _CURSOR_RE.fullmatch(token):
        return None
    try:
        raw = b64url_decode(token)
    except ValueError:
        return None
    # The MAC is a fixed 32-byte suffix; the JSON payload precedes it.
    if len(raw) <= 32:
        return None
    payload, mac = raw[:-32], raw[-32:]
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if decoded.get("k") != _PROOF_EVENT_CURSOR_KIND:
        return None
    boundary_at = decoded.get("at")
    boundary_event = decoded.get("e")
    if not isinstance(boundary_at, str) or boundary_at == "":
        return None
    if not isinstance(boundary_event, str) or not _UUID_RE.fullmatch(boundary_event):
        return None
    # The commit-order membership cutoff carried by a resume cursor. A
    # resume cursor always names the positive per-scope sequence
    # high-water mark established by the timeline's first query (bools
    # are rejected as ints).
    snapshot_seq = decoded.get("q")
    if not isinstance(snapshot_seq, int) or isinstance(snapshot_seq, bool):
        return None
    if snapshot_seq < 1:
        return None
    expected_mac = hmac.new(
        _cursor_secret(),
        _proof_event_cursor_payload(
            tenant_id,
            workload_id,
            boundary_at,
            boundary_event,
            evidence_id=evidence_id,
            event_type=event_type,
            status=status,
            occurred_after=occurred_after,
            occurred_before=occurred_before,
            snapshot_seq=snapshot_seq,
        ),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(mac, expected_mac):
        return None
    # Defense in depth: the MAC already covers every field, but confirm
    # the scope and each active filter explicitly.
    if not hmac.compare_digest(str(decoded.get("t", "")), tenant_id):
        return None
    if not hmac.compare_digest(str(decoded.get("w", "")), workload_id):
        return None
    for key, expected in (
        ("id", evidence_id),
        ("ty", event_type),
        ("s", status),
        ("a", occurred_after),
        ("b", occurred_before),
    ):
        if not hmac.compare_digest(str(decoded.get(key, "")), expected):
            return None
    return boundary_at, boundary_event, snapshot_seq


def _decision_cursor_payload(
    tenant_id: str,
    workload_id: str,
    boundary_at: str,
    boundary_decision: str,
    *,
    decision_id: str,
    evidence_id: str,
    policy_id: str,
    status: str,
    decided_after: str,
    decided_before: str,
    snapshot_seq: int,
) -> bytes:
    """Canonical byte payload authenticated inside a decision cursor.

    The cursor marks an exclusive ``(decided_at, decision_id)`` position
    and every active filter plus the fixed replayable snapshot are part of
    the signed payload, so a cursor minted for one filter set or snapshot
    cannot be replayed against another. The snapshot membership cutoff is
    the per-scope decision ``commit_seq`` high-water mark (``q``): the
    business commit boundary established by the first query. Bounding
    membership by commit order — rather than by ``decided_at`` — is what
    keeps a decision that commits *after* the first query but carries an
    older or identical business time out of the fixed snapshot, on every
    backend. The kind tag distinguishes these cursors from every other
    cursor family even though all share the same HMAC secret.
    """
    return json.dumps(
        {
            "k": _DECISION_CURSOR_KIND,
            "t": tenant_id,
            "w": workload_id,
            "at": boundary_at,
            "d": boundary_decision,
            "di": decision_id,
            "ev": evidence_id,
            "po": policy_id,
            "s": status,
            "a": decided_after,
            "b": decided_before,
            "q": snapshot_seq,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_decision_cursor(
    tenant_id: str,
    workload_id: str,
    boundary_at: str,
    boundary_decision: str,
    *,
    decision_id: str,
    evidence_id: str,
    policy_id: str,
    status: str,
    decided_after: str,
    decided_before: str,
    snapshot_seq: int,
) -> str:
    """Build an opaque, scope/filter/snapshot-bound exclusive decision cursor."""
    payload = _decision_cursor_payload(
        tenant_id,
        workload_id,
        boundary_at,
        boundary_decision,
        decision_id=decision_id,
        evidence_id=evidence_id,
        policy_id=policy_id,
        status=status,
        decided_after=decided_after,
        decided_before=decided_before,
        snapshot_seq=snapshot_seq,
    )
    mac = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    return b64url_encode(payload + mac)


def _decode_decision_cursor(
    token: str,
    tenant_id: str,
    workload_id: str,
    *,
    decision_id: str,
    evidence_id: str,
    policy_id: str,
    status: str,
    decided_after: str,
    decided_before: str,
) -> tuple[str, str, int] | None:
    """Validate a decision cursor and return its exclusive boundary.

    Returns ``(decided_at, decision_id, snapshot_seq)`` on success or
    ``None`` for a malformed/forged token, a cursor of another kind
    (rewrap batch, grant audit, compliance audit events, proof-lifecycle
    events, revocations or rewrap job listings), or one minted for any
    other scope, filter combination or snapshot. The beginning marker
    (``""``) never reaches this function.
    """
    if not _CURSOR_RE.fullmatch(token):
        return None
    try:
        raw = b64url_decode(token)
    except ValueError:
        return None
    # The MAC is a fixed 32-byte suffix; the JSON payload precedes it.
    if len(raw) <= 32:
        return None
    payload, mac = raw[:-32], raw[-32:]
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if decoded.get("k") != _DECISION_CURSOR_KIND:
        return None
    boundary_at = decoded.get("at")
    boundary_decision = decoded.get("d")
    if not isinstance(boundary_at, str) or boundary_at == "":
        return None
    if not isinstance(boundary_decision, str) or not _UUID_RE.fullmatch(
        boundary_decision
    ):
        return None
    # The commit-order membership cutoff carried by a resume cursor. A
    # resume cursor always names the positive per-scope sequence
    # high-water mark established by the listing's first query (bools are
    # rejected as ints).
    snapshot_seq = decoded.get("q")
    if not isinstance(snapshot_seq, int) or isinstance(snapshot_seq, bool):
        return None
    if snapshot_seq < 1:
        return None
    expected_mac = hmac.new(
        _cursor_secret(),
        _decision_cursor_payload(
            tenant_id,
            workload_id,
            boundary_at,
            boundary_decision,
            decision_id=decision_id,
            evidence_id=evidence_id,
            policy_id=policy_id,
            status=status,
            decided_after=decided_after,
            decided_before=decided_before,
            snapshot_seq=snapshot_seq,
        ),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(mac, expected_mac):
        return None
    # Defense in depth: the MAC already covers every field, but confirm
    # the scope and each active filter explicitly.
    if not hmac.compare_digest(str(decoded.get("t", "")), tenant_id):
        return None
    if not hmac.compare_digest(str(decoded.get("w", "")), workload_id):
        return None
    for key, expected in (
        ("di", decision_id),
        ("ev", evidence_id),
        ("po", policy_id),
        ("s", status),
        ("a", decided_after),
        ("b", decided_before),
    ):
        if not hmac.compare_digest(str(decoded.get(key, "")), expected):
            return None
    return boundary_at, boundary_decision, snapshot_seq


def _trust_root_cursor_payload(
    tenant_id: str,
    workload_id: str,
    boundary_at: str,
    boundary_root: str,
    *,
    root_id: str,
    name: str,
    status: str,
    created_after: str,
    created_before: str,
    snapshot_seq: int,
) -> bytes:
    """Canonical byte payload authenticated inside a trust-root cursor.

    The cursor marks an exclusive ``(created_at, root_id)`` position and
    every active filter plus the fixed replayable snapshot are part of the
    signed payload, so a cursor minted for one filter set or snapshot
    cannot be replayed against another. The snapshot membership cutoff is
    the per-scope trust-root lifecycle ``commit_seq`` high-water mark
    (``q``): the business commit boundary established by the first query,
    shared by creation and retirement. The kind tag distinguishes these
    cursors from every other cursor family even though all share the same
    HMAC secret.
    """
    return json.dumps(
        {
            "k": _TRUST_ROOT_CURSOR_KIND,
            "t": tenant_id,
            "w": workload_id,
            "at": boundary_at,
            "id": boundary_root,
            "ri": root_id,
            "nm": name,
            "s": status,
            "a": created_after,
            "b": created_before,
            "q": snapshot_seq,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_trust_root_cursor(
    tenant_id: str,
    workload_id: str,
    boundary_at: str,
    boundary_root: str,
    *,
    root_id: str,
    name: str,
    status: str,
    created_after: str,
    created_before: str,
    snapshot_seq: int,
) -> str:
    """Build an opaque, scope/filter/snapshot-bound exclusive trust-root cursor."""
    payload = _trust_root_cursor_payload(
        tenant_id,
        workload_id,
        boundary_at,
        boundary_root,
        root_id=root_id,
        name=name,
        status=status,
        created_after=created_after,
        created_before=created_before,
        snapshot_seq=snapshot_seq,
    )
    mac = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    return b64url_encode(payload + mac)


def _decode_trust_root_cursor(
    token: str,
    tenant_id: str,
    workload_id: str,
    *,
    root_id: str,
    name: str,
    status: str,
    created_after: str,
    created_before: str,
) -> tuple[str, str, int] | None:
    """Validate a trust-root cursor and return its exclusive boundary.

    Returns ``(created_at, root_id, snapshot_seq)`` on success or ``None``
    for a malformed/forged token, a cursor of any other kind (rewrap
    batch, grant audit, compliance audit events, proof-lifecycle events,
    decisions, revocations or rewrap job listings), or one minted for any
    other scope, filter combination or snapshot. The beginning marker
    (``""``) never reaches this function.
    """
    if not _CURSOR_RE.fullmatch(token):
        return None
    try:
        raw = b64url_decode(token)
    except ValueError:
        return None
    # The MAC is a fixed 32-byte suffix; the JSON payload precedes it.
    if len(raw) <= 32:
        return None
    payload, mac = raw[:-32], raw[-32:]
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if decoded.get("k") != _TRUST_ROOT_CURSOR_KIND:
        return None
    boundary_at = decoded.get("at")
    boundary_root = decoded.get("id")
    if not isinstance(boundary_at, str) or boundary_at == "":
        return None
    if not isinstance(boundary_root, str) or not _UUID_RE.fullmatch(boundary_root):
        return None
    # The commit-order membership cutoff carried by a resume cursor. A
    # resume cursor always names the positive per-scope lifecycle sequence
    # high-water mark established by the first query (bools are rejected
    # as ints).
    snapshot_seq = decoded.get("q")
    if not isinstance(snapshot_seq, int) or isinstance(snapshot_seq, bool):
        return None
    if snapshot_seq < 1:
        return None
    expected_mac = hmac.new(
        _cursor_secret(),
        _trust_root_cursor_payload(
            tenant_id,
            workload_id,
            boundary_at,
            boundary_root,
            root_id=root_id,
            name=name,
            status=status,
            created_after=created_after,
            created_before=created_before,
            snapshot_seq=snapshot_seq,
        ),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(mac, expected_mac):
        return None
    # Defense in depth: the MAC already covers every field, but confirm
    # the scope and each active filter explicitly.
    if not hmac.compare_digest(str(decoded.get("t", "")), tenant_id):
        return None
    if not hmac.compare_digest(str(decoded.get("w", "")), workload_id):
        return None
    for key, expected in (
        ("ri", root_id),
        ("nm", name),
        ("s", status),
        ("a", created_after),
        ("b", created_before),
    ):
        if not hmac.compare_digest(str(decoded.get(key, "")), expected):
            return None
    return boundary_at, boundary_root, snapshot_seq


def _policy_cursor_payload(
    tenant_id: str,
    workload_id: str,
    boundary_at: str,
    boundary_policy: str,
    *,
    policy_id: str,
    name: str,
    status: str,
    created_after: str,
    created_before: str,
    snapshot_seq: int,
) -> bytes:
    """Canonical byte payload authenticated inside a policy cursor.

    The cursor marks an exclusive ``(created_at, policy_id)`` position and
    every active filter plus the fixed replayable snapshot are part of the
    signed payload, so a cursor minted for one filter set or snapshot
    cannot be replayed against another. The snapshot membership cutoff is
    the per-scope policy lifecycle ``commit_seq`` high-water mark (``q``):
    the business commit boundary established by the first query, shared by
    version creation and retirement. The kind tag distinguishes these
    cursors from every other cursor family even though all share the same
    HMAC secret.
    """
    return json.dumps(
        {
            "k": _POLICY_CURSOR_KIND,
            "t": tenant_id,
            "w": workload_id,
            "at": boundary_at,
            "id": boundary_policy,
            "pi": policy_id,
            "nm": name,
            "s": status,
            "a": created_after,
            "b": created_before,
            "q": snapshot_seq,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_policy_cursor(
    tenant_id: str,
    workload_id: str,
    boundary_at: str,
    boundary_policy: str,
    *,
    policy_id: str,
    name: str,
    status: str,
    created_after: str,
    created_before: str,
    snapshot_seq: int,
) -> str:
    """Build an opaque, scope/filter/snapshot-bound exclusive policy cursor."""
    payload = _policy_cursor_payload(
        tenant_id,
        workload_id,
        boundary_at,
        boundary_policy,
        policy_id=policy_id,
        name=name,
        status=status,
        created_after=created_after,
        created_before=created_before,
        snapshot_seq=snapshot_seq,
    )
    mac = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    return b64url_encode(payload + mac)


def _decode_policy_cursor(
    token: str,
    tenant_id: str,
    workload_id: str,
    *,
    policy_id: str,
    name: str,
    status: str,
    created_after: str,
    created_before: str,
) -> tuple[str, str, int] | None:
    """Validate a policy cursor and return its exclusive boundary.

    Returns ``(created_at, policy_id, snapshot_seq)`` on success or
    ``None`` for a malformed/forged token, a cursor of any other kind
    (rewrap batch, grant audit, compliance audit events, proof-lifecycle
    events, decisions, revocations, trust roots or rewrap job listings),
    or one minted for any other scope, filter combination or snapshot. The
    beginning marker (``""``) never reaches this function.
    """
    if not _CURSOR_RE.fullmatch(token):
        return None
    try:
        raw = b64url_decode(token)
    except ValueError:
        return None
    # The MAC is a fixed 32-byte suffix; the JSON payload precedes it.
    if len(raw) <= 32:
        return None
    payload, mac = raw[:-32], raw[-32:]
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if decoded.get("k") != _POLICY_CURSOR_KIND:
        return None
    boundary_at = decoded.get("at")
    boundary_policy = decoded.get("id")
    if not isinstance(boundary_at, str) or boundary_at == "":
        return None
    if not isinstance(boundary_policy, str) or not _UUID_RE.fullmatch(
        boundary_policy
    ):
        return None
    # The commit-order membership cutoff carried by a resume cursor. A
    # resume cursor always names the positive per-scope lifecycle sequence
    # high-water mark established by the first query (bools are rejected
    # as ints).
    snapshot_seq = decoded.get("q")
    if not isinstance(snapshot_seq, int) or isinstance(snapshot_seq, bool):
        return None
    if snapshot_seq < 1:
        return None
    expected_mac = hmac.new(
        _cursor_secret(),
        _policy_cursor_payload(
            tenant_id,
            workload_id,
            boundary_at,
            boundary_policy,
            policy_id=policy_id,
            name=name,
            status=status,
            created_after=created_after,
            created_before=created_before,
            snapshot_seq=snapshot_seq,
        ),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(mac, expected_mac):
        return None
    # Defense in depth: the MAC already covers every field, but confirm
    # the scope and each active filter explicitly.
    if not hmac.compare_digest(str(decoded.get("t", "")), tenant_id):
        return None
    if not hmac.compare_digest(str(decoded.get("w", "")), workload_id):
        return None
    for key, expected in (
        ("pi", policy_id),
        ("nm", name),
        ("s", status),
        ("a", created_after),
        ("b", created_before),
    ):
        if not hmac.compare_digest(str(decoded.get(key, "")), expected):
            return None
    return boundary_at, boundary_policy, snapshot_seq


#: Fixed page size for the read-only compliance audit-event listing. As
#: with the grant audit, the page size is an internal constant and never
#: part of the request or response contract.
AUDIT_EVENT_PAGE_SIZE = 100

#: Discriminator embedded in compliance audit-event cursors so neither a
#: rewrap-batch cursor nor a release-grant audit cursor (all authenticated
#: with the same secret) can ever be replayed here.
_AUDIT_EVENT_CURSOR_KIND = "compliance-audit-events-v1"

#: Fixed page size for the read-only revocation registry listing. As with
#: the other audit listings, the page size is internal and never part of
#: the request or response contract.
REVOCATION_PAGE_SIZE = 100

#: Discriminator embedded in revocation-registry cursors so a rewrap,
#: release-grant audit or compliance audit-event cursor (all authenticated
#: with the same secret) can never be replayed against the revocation
#: listing, and vice versa.
_REVOCATION_CURSOR_KIND = "certificate-revocations-v1"

#: Fixed page size for the read-only asynchronous rewrap job history.
#: Like the other audit listings it is an internal constant and never
#: part of the request or response contract.
REWRAP_JOB_HISTORY_PAGE_SIZE = 100

#: Discriminator embedded in rewrap-job-history cursors so a rewrap-batch
#: cursor, a release-grant audit cursor, a compliance audit-event cursor
#: and a revocation-registry cursor (all authenticated with the same
#: secret) can never be replayed against the job history, and vice versa.
_REWRAP_JOB_HISTORY_CURSOR_KIND = "rewrap-job-history-v1"

#: Fixed page size for the read-only per-job lifecycle event timeline.
#: Like the other audit listings it is an internal constant and never
#: part of the request or response contract.
REWRAP_JOB_EVENT_PAGE_SIZE = 100

#: Discriminator embedded in rewrap-job-event cursors so a cursor from any
#: other family (rewrap batch, grant audit, compliance audit events,
#: revocations or job history — all authenticated with the same secret)
#: can never be replayed against a job's event timeline, and vice versa.
_REWRAP_JOB_EVENT_CURSOR_KIND = "rewrap-job-events-v1"

#: Discriminator embedded in rewrap-job-item cursors so a cursor from any
#: other family (rewrap batch, grant audit, compliance audit events,
#: revocations, job history or job events — all authenticated with the
#: same secret) can never be replayed against a job's item listing, and
#: vice versa.
_REWRAP_JOB_ITEM_CURSOR_KIND = "rewrap-job-items-v1"

#: Fixed page size for the read-only proof-lifecycle event timeline.
#: Like the other audit listings it is an internal constant and never
#: part of the request or response contract.
PROOF_EVENT_PAGE_SIZE = 100

#: Discriminator embedded in proof-lifecycle event cursors so a cursor
#: from any other family (rewrap batch, grant audit, compliance audit
#: events, revocations or rewrap job listings — all authenticated with
#: the same secret) can never be replayed against the proof timeline, and
#: vice versa.
_PROOF_EVENT_CURSOR_KIND = "compliance-proof-events-v1"

#: Fixed page size for the read-only compliance decision listing. Like
#: the other audit listings it is an internal constant and never part of
#: the request or response contract.
DECISION_PAGE_SIZE = 100

#: Discriminator embedded in compliance decision cursors so a cursor from
#: any other family (rewrap batch, grant audit, compliance audit events,
#: proof-lifecycle events, revocations or rewrap job listings — all
#: authenticated with the same secret) can never be replayed against the
#: decision listing, and vice versa.
_DECISION_CURSOR_KIND = "compliance-decisions-v1"

#: Version of the persisted/returned policy-evaluation explanation shape.
#: Every node carries exactly node_index, rule_path, node_type and
#: outcome (a boolean); bumping this constant is reserved for a future,
#: explicitly versioned explanation format.
EVALUATION_VERSION = 1

#: Fixed page size for the read-only trust-root lifecycle listing. As with
#: the other read-only listings, the page size is an internal constant and
#: never part of the request or response contract.
TRUST_ROOT_PAGE_SIZE = 100

#: Discriminator embedded in trust-root lifecycle cursors so a cursor from
#: any other family (rewrap batch, release-grant audit, compliance audit
#: events, proof-lifecycle events, decisions, revocations or rewrap job
#: listings — all authenticated with the same secret) can never be replayed
#: against the trust-root listing, and vice versa.
_TRUST_ROOT_CURSOR_KIND = "trust-roots-v1"

#: Fixed page size for the read-only policy-version lifecycle listing. As
#: with the other read-only listings, the page size is an internal constant
#: and never part of the request or response contract.
POLICY_PAGE_SIZE = 100

#: Discriminator embedded in policy-version lifecycle cursors so a cursor
#: from any other family (rewrap batch, release-grant audit, compliance
#: audit events, proof-lifecycle events, decisions, revocations,
#: trust roots or rewrap job listings — all authenticated with the same
#: secret) can never be replayed against the policy listing, and vice
#: versa.
_POLICY_CURSOR_KIND = "policies-v1"

#: Fixed page size for the read-only per-grant state-migration event
#: timeline. Like the other audit listings it is an internal constant and
#: never part of the request or response contract.
RELEASE_GRANT_EVENT_PAGE_SIZE = 100

#: Discriminator embedded in release-grant event cursors so a cursor from
#: any other family (rewrap batch, grant audit, compliance audit events,
#: proof-lifecycle events, decisions, revocations, trust roots, policies
#: or rewrap job listings — all authenticated with the same secret) can
#: never be replayed against a grant's event timeline, and vice versa.
_RELEASE_GRANT_EVENT_CURSOR_KIND = "release-grant-events-v1"

#: Fixed page size for the read-only data-envelope directory listing. As
#: with the other read-only listings, the page size is an internal
#: constant and never part of the request or response contract.
DATA_ENVELOPE_PAGE_SIZE = 100

#: Discriminator embedded in data-envelope directory cursors so a cursor
#: from any other family (rewrap batch, grant audit, compliance audit
#: events, proof-lifecycle events, decisions, revocations, trust roots,
#: policies or rewrap job listings — all authenticated with the same
#: secret) can never be replayed against the directory listing, and vice
#: versa.
_DATA_ENVELOPE_CURSOR_KIND = "data-envelopes-v1"


def _data_envelope_cursor_payload(
    tenant_id: str,
    workload_id: str,
    boundary_data_id: str,
    *,
    data_id: str,
    created_after: str,
    created_before: str,
    snapshot_seq: int,
) -> bytes:
    """Canonical byte payload authenticated inside a directory cursor.

    The cursor marks an exclusive ``data_id`` position and every active
    filter plus the fixed replayable snapshot are part of the signed
    payload, so a cursor minted for one filter set or snapshot cannot be
    replayed against another. The snapshot membership cutoff (``q``) is
    the per-scope envelope-creation ``commit_seq`` high-water mark fixed
    by the range's first query. The kind tag distinguishes these cursors
    from every other cursor family even though all share the same HMAC
    secret.
    """
    return json.dumps(
        {
            "k": _DATA_ENVELOPE_CURSOR_KIND,
            "t": tenant_id,
            "w": workload_id,
            "d": boundary_data_id,
            "di": data_id,
            "a": created_after,
            "b": created_before,
            "q": snapshot_seq,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _encode_data_envelope_cursor(
    tenant_id: str,
    workload_id: str,
    boundary_data_id: str,
    *,
    data_id: str,
    created_after: str,
    created_before: str,
    snapshot_seq: int,
) -> str:
    """Build an opaque, scope/filter/snapshot-bound exclusive cursor."""
    payload = _data_envelope_cursor_payload(
        tenant_id,
        workload_id,
        boundary_data_id,
        data_id=data_id,
        created_after=created_after,
        created_before=created_before,
        snapshot_seq=snapshot_seq,
    )
    mac = hmac.new(_cursor_secret(), payload, hashlib.sha256).digest()
    return b64url_encode(payload + mac)


def _decode_data_envelope_cursor(
    token: str,
    tenant_id: str,
    workload_id: str,
    *,
    data_id: str,
    created_after: str,
    created_before: str,
) -> tuple[str, int] | None:
    """Validate a directory cursor and return ``(boundary_data_id, seq)``.

    Returns ``(data_id, snapshot_seq)`` on success or ``None`` for a
    malformed/forged token, a cursor of any other kind, or one minted for
    any other scope, filter combination or snapshot. The beginning marker
    (``""``) never reaches this function.
    """
    if not _CURSOR_RE.fullmatch(token):
        return None
    try:
        raw = b64url_decode(token)
    except ValueError:
        return None
    # The MAC is a fixed 32-byte suffix; the JSON payload precedes it.
    if len(raw) <= 32:
        return None
    payload, mac = raw[:-32], raw[-32:]
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if decoded.get("k") != _DATA_ENVELOPE_CURSOR_KIND:
        return None
    boundary_data_id = decoded.get("d")
    if not isinstance(boundary_data_id, str) or boundary_data_id == "":
        return None
    # The commit-order membership cutoff carried by a resume cursor. A
    # resume cursor is only ever minted for a page with a following page,
    # so its snapshot always names a positive high-water mark (bools are
    # rejected as ints).
    snapshot_seq = decoded.get("q")
    if not isinstance(snapshot_seq, int) or isinstance(snapshot_seq, bool):
        return None
    if snapshot_seq < 1:
        return None
    expected_mac = hmac.new(
        _cursor_secret(),
        _data_envelope_cursor_payload(
            tenant_id,
            workload_id,
            boundary_data_id,
            data_id=data_id,
            created_after=created_after,
            created_before=created_before,
            snapshot_seq=snapshot_seq,
        ),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(mac, expected_mac):
        return None
    # Defense in depth: the MAC already covers every field, but confirm
    # the scope and each active filter explicitly.
    if not hmac.compare_digest(str(decoded.get("t", "")), tenant_id):
        return None
    if not hmac.compare_digest(str(decoded.get("w", "")), workload_id):
        return None
    for key, expected in (
        ("di", data_id),
        ("a", created_after),
        ("b", created_before),
    ):
        if not hmac.compare_digest(str(decoded.get(key, "")), expected):
            return None
    return boundary_data_id, snapshot_seq


class CreateChallengeRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    ttl_seconds: StrictInt = Field(
        default=DEFAULT_TTL_SECONDS, ge=MIN_TTL_SECONDS, le=MAX_TTL_SECONDS
    )

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)


class ConsumeChallengeRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    nonce: StrictStr = Field(min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)
    _nonce_valid = field_validator("nonce")(_nonce_format)


class ChallengeCreatedResponse(BaseModel):
    challenge_id: str
    nonce: str
    issued_at: str
    expires_at: str
    status: str


class ChallengeConsumedResponse(BaseModel):
    challenge_id: str
    status: str
    consumed_at: str


#: Read-only phase names reported by GET /v1/challenges/{challenge_id}.
#: The first two are derived at query time (no expiry write); the
#: remaining four follow the persisted consumption/evidence state.
CHALLENGE_PHASE_PENDING = "pending"
CHALLENGE_PHASE_EXPIRED = "expired"
CHALLENGE_PHASE_CONSUMED = "consumed"
CHALLENGE_PHASE_EVIDENCE_RECEIVED = "evidence_received"
CHALLENGE_PHASE_VERIFIED = "verified"
CHALLENGE_PHASE_REJECTED = "rejected"


class SubmitEvidenceRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    challenge_id: StrictStr = Field(min_length=1)
    nonce: StrictStr = Field(min_length=1)
    evidence_format: StrictStr = Field(min_length=1)
    evidence: StrictStr = Field(min_length=1)

    _non_blank = field_validator(
        "tenant_id", "workload_id", "challenge_id", "evidence_format"
    )(_require_non_blank)
    _nonce_valid = field_validator("nonce")(_nonce_format)


class EvidenceReceivedResponse(BaseModel):
    evidence_id: str
    challenge_id: str
    status: str
    received_at: str


class VerifyEvidenceRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    nonce: StrictStr = Field(min_length=1)
    evidence: StrictStr = Field(min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)
    _nonce_valid = field_validator("nonce")(_nonce_format)


class EvidenceVerifiedResponse(BaseModel):
    evidence_id: str
    challenge_id: str
    status: str
    verified_at: str


class CreateTrustRootRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    root_pem: StrictStr = Field(min_length=1)
    name: StrictStr | None = Field(default=None, min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id", "root_pem")(
        _require_non_blank
    )
    _name_non_blank = field_validator("name")(_optional_non_blank)


class TrustRootCreatedResponse(BaseModel):
    root_id: str
    tenant_id: str
    workload_id: str
    name: str | None
    created_at: str


class RetireTrustRootRequest(BaseModel):
    """Scope naming the trust root to retire.

    The body carries only the two non-blank scope strings; any missing,
    blank, wrong-typed or unknown field is rejected as a client error
    rather than silently ignored, and validation completes before the
    handler touches storage.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)


class CreateRevocationRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    # Canonical lowercase UUID of a trust root in exactly this scope.
    trust_root_id: StrictStr = Field(min_length=1)
    # Unpadded base64url of the 32-byte SHA-256 DER digest.
    certificate_fingerprint: StrictStr = Field(min_length=1)
    # UTC RFC3339 instant at/after which the revocation takes effect.
    effective_at: StrictStr = Field(min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)
    _trust_root_shape = field_validator("trust_root_id")(_canonical_uuid)
    _fingerprint_shape = field_validator("certificate_fingerprint")(
        _certificate_fingerprint_format
    )
    _effective_at_shape = field_validator("effective_at")(_utc_rfc3339_field)


class CreateCrlRequest(BaseModel):
    """Registration request for one X.509 v2 CRL under a trust root.

    The body carries exactly the scope, the canonical trust-root UUID and
    the PEM CRL; every other field is rejected. The CRL's intrinsic
    validity (PEM/ASN.1, v2, CRLNumber, nextUpdate, time window and
    serials) and its trust binding (issuer DN, signature) are checked by
    the handler, since they depend on the receipt time and the stored
    trust root.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    trust_root_id: StrictStr = Field(min_length=1)
    crl_pem: StrictStr = Field(min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id", "crl_pem")(
        _require_non_blank
    )
    _trust_root_shape = field_validator("trust_root_id")(_canonical_uuid)


class IdentityClaimModel(BaseModel):
    """One workload identity claim: the parsed issuer, subject and URI.

    Every field is a required non-blank string; values are stored verbatim
    as the non-sensitive comparison strings used against a leaf
    certificate's RFC4514 issuer/subject DNs and SAN URI. Any additional
    field is rejected as a client error rather than silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    issuer: StrictStr = Field(min_length=1)
    subject: StrictStr = Field(min_length=1)
    uri: StrictStr = Field(min_length=1)

    _non_blank = field_validator("issuer", "subject", "uri")(_require_non_blank)


class RegisterWorkloadIdentityRequest(BaseModel):
    # Unknown fields are rejected rather than dropped, so a client learns
    # immediately that the service did not act on them.
    model_config = ConfigDict(extra="forbid")

    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    # Canonical lowercase UUID of a trust root in exactly this scope.
    trust_root_id: StrictStr = Field(min_length=1)
    # At least one identity claim; an empty list is rejected. Each entry
    # must carry all three non-blank string fields.
    claims: list[IdentityClaimModel]

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)
    _trust_root_shape = field_validator("trust_root_id")(_canonical_uuid)

    @field_validator("claims")
    @classmethod
    def _claims_non_empty(cls, value: list[IdentityClaimModel]) -> list[IdentityClaimModel]:
        if not value:
            raise ValueError("claims must contain at least one identity claim")
        return value


class UpdateWorkloadIdentityRequest(BaseModel):
    """Whole-set replacement of a profile's claims (PUT semantics).

    The body carries the scope, trust root and the complete new claim
    list; every field has the same shape and strictness as on
    registration, and unknown fields are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    trust_root_id: StrictStr = Field(min_length=1)
    claims: list[IdentityClaimModel]

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)
    _trust_root_shape = field_validator("trust_root_id")(_canonical_uuid)

    @field_validator("claims")
    @classmethod
    def _claims_non_empty(cls, value: list[IdentityClaimModel]) -> list[IdentityClaimModel]:
        if not value:
            raise ValueError("claims must contain at least one identity claim")
        return value


class RevokeWorkloadIdentityRequest(BaseModel):
    """Scope and anchor naming the profile to revoke."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    trust_root_id: StrictStr = Field(min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)
    _trust_root_shape = field_validator("trust_root_id")(_canonical_uuid)


class CreatePolicyRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    name: StrictStr = Field(min_length=1)
    # Validated structurally below; pydantic only enforces that it parses
    # as a JSON object.
    rule: dict

    _non_blank = field_validator("tenant_id", "workload_id", "name")(
        _require_non_blank
    )

    @field_validator("rule")
    @classmethod
    def _validate_rule_shape(cls, value: dict) -> dict:
        try:
            return validate_rule(value)
        except InvalidRule as exc:
            raise ValueError(str(exc)) from exc


class PolicyCreatedResponse(BaseModel):
    policy_id: str
    tenant_id: str
    workload_id: str
    name: str
    version: int
    rule: dict
    created_at: str


class RetirePolicyRequest(BaseModel):
    """Scope naming the policy version to retire.

    The body carries only the two non-blank scope strings; any missing,
    blank, wrong-typed, incomplete or unknown field is rejected as a
    client error rather than silently ignored, and validation completes
    before the handler touches storage. The two fields together with the
    path ``policy_id`` name exactly one concrete policy version.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)


class EvaluatePolicyRequest(BaseModel):
    """Scope and claims for a read-only policy trial evaluation.

    The body carries exactly the two non-blank scope strings and the
    ``claims`` JSON object to evaluate; any missing, blank, wrong-typed,
    incomplete or unknown field — and a ``claims`` value that is not a
    JSON object — is a 422 that never reads or writes any state. The
    claims are used only for this evaluation and are never persisted.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    claims: dict

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)


class CreateDecisionRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    nonce: StrictStr = Field(min_length=1)
    evidence: StrictStr = Field(min_length=1)
    policy_id: StrictStr = Field(min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id", "policy_id")(
        _require_non_blank
    )
    _nonce_valid = field_validator("nonce")(_nonce_format)


class DecisionResponse(BaseModel):
    decision_id: str
    evidence_id: str
    policy_id: str
    policy_version: int
    status: str
    decided_at: str


class CreateReleaseGrantRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    decision_id: StrictStr = Field(min_length=1)
    data_id: StrictStr = Field(min_length=1)
    ttl_seconds: StrictInt = Field(
        default=DEFAULT_TTL_SECONDS, ge=MIN_TTL_SECONDS, le=MAX_TTL_SECONDS
    )

    _non_blank = field_validator(
        "tenant_id", "workload_id", "decision_id", "data_id"
    )(_require_non_blank)


class ReleaseGrantCreatedResponse(BaseModel):
    grant_id: str
    decision_id: str
    data_id: str
    #: The plaintext capability. Returned exactly once, here; the database
    #: retains only its SHA-256 digest and every other response omits it.
    capability: str
    pending: bool
    issued_at: str
    expires_at: str


class ConsumeReleaseGrantRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    capability: StrictStr = Field(min_length=1)

    _non_blank = field_validator(
        "tenant_id", "workload_id", "capability"
    )(_require_non_blank)
    # Malformed capability syntax is a field/format error (422); a
    # well-formed value that simply does not match is an authentication
    # failure (401). Capabilities use the same unpadded base64url alphabet
    # as nonces.
    _capability_valid = field_validator("capability")(_nonce_format)


class ReleaseGrantConsumedResponse(BaseModel):
    grant_id: str
    decision_id: str
    data_id: str
    consumed: bool
    consumed_at: str


class RevokeReleaseGrantRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    capability: StrictStr = Field(min_length=1)

    _non_blank = field_validator(
        "tenant_id", "workload_id", "capability"
    )(_require_non_blank)
    # As on consume/release, malformed capability syntax is a field/format
    # error (422); a well-formed value that simply does not match is an
    # authentication failure (401).
    _capability_valid = field_validator("capability")(_nonce_format)


class ReleaseGrantRevokedResponse(BaseModel):
    grant_id: str
    decision_id: str
    data_id: str
    revoked: bool
    revoked_at: str


class ReleasePayloadRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    data_id: StrictStr = Field(min_length=1)
    capability: StrictStr = Field(min_length=1)

    _non_blank = field_validator(
        "tenant_id", "workload_id", "data_id", "capability"
    )(_require_non_blank)
    # As on the consume path, malformed capability syntax is a field/format
    # error (422); a well-formed value that simply does not match is an
    # authentication failure (401).
    _capability_valid = field_validator("capability")(_nonce_format)


class CreateDataEnvelopeRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    data_id: StrictStr = Field(min_length=1)
    # Arbitrary non-empty payload content; whitespace-only is permitted
    # (it is data, not an identifier), so only emptiness is rejected.
    payload: StrictStr = Field(min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id", "data_id")(
        _require_non_blank
    )


class DataEnvelopeCreatedResponse(BaseModel):
    data_id: str
    tenant_id: str
    workload_id: str
    key_version: int
    created_at: str


class DataEnvelopeResponse(BaseModel):
    data_id: str
    tenant_id: str
    workload_id: str
    key_version: int
    created_at: str
    ciphertext: str
    iv: str
    tag: str
    wrapped_key: str


class RewrapDataEnvelopeRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)


class DataEnvelopeRewrappedResponse(BaseModel):
    data_id: str
    tenant_id: str
    workload_id: str
    key_version: int
    rotated_at: str


class CreateRewrapBatchRequest(BaseModel):
    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    # 1..200 inclusive, defaulting to 50. Booleans are rejected by
    # StrictInt even though Python treats them as ints.
    limit: StrictInt = Field(
        default=REWRAP_BATCH_DEFAULT_LIMIT,
        ge=REWRAP_BATCH_MIN_LIMIT,
        le=REWRAP_BATCH_MAX_LIMIT,
    )
    # Opaque scope-bound token from a previous batch; absent or empty
    # means the beginning of the scope.
    cursor: StrictStr | None = None

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)

    @field_validator("cursor")
    @classmethod
    def _cursor_shape(cls, value: str | None) -> str | None:
        # Only the wire shape is checked here; scope binding and the MAC
        # are verified against the request's tenant/workload in the
        # endpoint, where failures are indistinguishable 422s.
        if value is None or value == "":
            return None
        if not value.strip() or not _CURSOR_RE.fullmatch(value):
            raise ValueError("cursor is not a valid rewrap cursor")
        return value


class CreateRewrapJobRequest(BaseModel):
    """Submit a persistent, asynchronously advanced rewrap job.

    Same field contract as the one-shot batch (two required non-blank
    scope strings, an optional 1..200 limit defaulting to 50, an optional
    opaque scope-bound cursor); the difference is that a job is persisted
    and advanced in the background after a ``202`` rather than inline.
    """

    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)
    limit: StrictInt = Field(
        default=REWRAP_BATCH_DEFAULT_LIMIT,
        ge=REWRAP_BATCH_MIN_LIMIT,
        le=REWRAP_BATCH_MAX_LIMIT,
    )
    cursor: StrictStr | None = None

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)

    @field_validator("cursor")
    @classmethod
    def _cursor_shape(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if not value.strip() or not _CURSOR_RE.fullmatch(value):
            raise ValueError("cursor is not a valid rewrap cursor")
        return value


class CancelRewrapJobRequest(BaseModel):
    """Scope naming the asynchronous rewrap job to cancel.

    The body carries only the two non-blank scope strings; any missing,
    blank, wrong-typed or unknown field is rejected as a client error
    rather than silently ignored, and validation completes before the
    handler reads the job.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)


class ResumeRewrapJobRequest(BaseModel):
    """Scope naming the failed asynchronous rewrap job to resume.

    The body carries only the two non-blank scope strings; any missing,
    blank, wrong-typed or unknown field is rejected as a client error
    rather than silently ignored, and validation completes before the
    handler reads the job.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: StrictStr = Field(min_length=1)
    workload_id: StrictStr = Field(min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)


def _migrate_additive(engine) -> None:
    """Apply forward-only additive column additions to pre-existing databases."""
    if engine.dialect.name != "sqlite":
        # Locking backends are created from metadata, so the other
        # historical column additions never apply there. The proof
        # lifecycle commit sequence is different: it shipped after the
        # proof timeline, so a database written by an older deployment
        # lacks the column, the per-scope counters are unseeded and the
        # unique index is missing. Upgrade it dialect-neutrally.
        _migrate_proof_event_commit_sequence(engine)
        # The compliance decision sequence and denormalized scope columns
        # likewise shipped after the decision table first existed.
        _migrate_decision_commit_sequence(engine)
        # The trust-root lifecycle query gained the per-scope commit-order
        # marker (shared by creation and retirement) after creation and
        # retirement first shipped.
        _migrate_trust_root_commit_sequence(engine)
        # The policy lifecycle query gained the analogous per-scope
        # sequence shared by version creation and retirement.
        _migrate_policy_commit_sequence(engine)
        # The read-only data-envelope directory query gained the per-scope
        # commit-order marker that fixes its replayable snapshot.
        _migrate_data_envelope_commit_sequence(engine)
        # The compliance audit events gained their tamper-evident hash
        # chain (per-event link metadata plus the per-scope range head)
        # after the audit table first existed.
        _migrate_audit_event_chain(engine)
        return
    additions = {
        "evidence": (
            ("verified_at", "DATETIME"),
            ("verification_result", "VARCHAR(16)"),
        ),
        "release_grants": (
            ("revoked_at", "DATETIME"),
        ),
        "trust_roots": (
            ("status", "VARCHAR(16)"),
            ("retired_at", "DATETIME"),
            # Per-scope lifecycle sequence shared by creation and the
            # terminal retirement; pre-existing rows are backfilled below
            # in rowid (commit) order.
            ("commit_seq", "BIGINT"),
            ("retired_seq", "BIGINT"),
        ),
        "policies": (
            ("status", "VARCHAR(16)"),
            ("retired_at", "DATETIME"),
            # Per-scope lifecycle sequence shared by version creation and
            # the terminal retirement; pre-existing rows are backfilled
            # below by the dialect-neutral upgrade.
            ("commit_seq", "BIGINT"),
            ("retired_seq", "BIGINT"),
        ),
        "workload_identity_profiles": (
            ("status", "VARCHAR(16)"),
            ("updated_at", "DATETIME"),
            ("revoked_at", "DATETIME"),
        ),
        "workload_identity_claims": (
            ("seq", "INTEGER"),
        ),
        "rewrap_jobs": (
            # Cancellation was added after the job lifecycle first shipped;
            # rows created before then have never been cancelled, so the
            # new column backfills to NULL.
            ("cancelled_at", "DATETIME"),
        ),
        "proof_lifecycle_events": (
            # Per-scope commit-order marker added after the proof timeline
            # first shipped; pre-existing rows are backfilled below in
            # rowid (commit) order so their snapshot order is preserved.
            ("commit_seq", "BIGINT"),
        ),
        "decisions": (
            # The compliance decision query scopes and sequences decisions
            # directly. Older deployments carried neither the denormalized
            # scope (reachable only through the evidence row) nor the
            # per-scope commit-order marker; pre-existing rows get their
            # scope copied from their evidence and are backfilled below in
            # rowid (commit) order.
            ("tenant_id", "VARCHAR(256)"),
            ("workload_id", "VARCHAR(256)"),
            ("commit_seq", "BIGINT"),
        ),
        "data_envelopes": (
            # The read-only directory query fixes its replayable snapshot
            # through a per-scope commit-order marker; pre-existing rows
            # are backfilled below in rowid (commit) order.
            ("commit_seq", "BIGINT"),
        ),
        "audit_events": (
            # The tamper-evident hash chain shipped after the audit table
            # first existed; pre-existing rows are backfilled below per
            # scope in (occurred_at, event_id) order and each scope's
            # range head is built.
            ("chain_seq", "BIGINT"),
            ("prev_hash", "VARCHAR(64)"),
            ("event_hash", "VARCHAR(64)"),
        ),
    }
    with engine.begin() as conn:
        for table, columns in additions.items():
            existing = {
                row[1]
                for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            }
            for column, column_type in columns:
                if column not in existing:
                    conn.execute(
                        text(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
                    )
            # Backfills for columns added to pre-existing rows: profiles
            # created before the lifecycle existed are all active, and
            # legacy claims take a single shared ordering position.
            if table == "workload_identity_profiles":
                conn.execute(
                    text(
                        "UPDATE workload_identity_profiles "
                        "SET status = :active WHERE status IS NULL"
                    ),
                    {"active": WORKLOAD_IDENTITY_STATUS_ACTIVE},
                )
            if table == "trust_roots":
                # Trust roots created before retirement existed are all
                # active; only an explicit retire ever sets retired_at.
                conn.execute(
                    text(
                        "UPDATE trust_roots "
                        "SET status = :active WHERE status IS NULL"
                    ),
                    {"active": TRUST_ROOT_STATUS_ACTIVE},
                )
            if table == "policies":
                # Policy versions created before terminal retirement
                # existed are all active; only an explicit retire ever
                # sets a version's retired_at.
                conn.execute(
                    text(
                        "UPDATE policies "
                        "SET status = :active WHERE status IS NULL"
                    ),
                    {"active": POLICY_STATUS_ACTIVE},
                )
            if table == "workload_identity_claims":
                conn.execute(
                    text("UPDATE workload_identity_claims SET seq = 0 WHERE seq IS NULL")
                )
            # Legacy databases may carry a free-form verification_detail
            # column written by older versions, which can hold arbitrary
            # plugin-supplied text (potentially raw evidence or secrets).
            # It is no longer part of the model; scrub any leftover values
            # so they can never be read back, returned, or logged.
            if "verification_detail" in existing:
                conn.execute(
                    text(f"UPDATE {table} SET verification_detail = NULL")
                )

    # The proof-lifecycle commit sequence is added to pre-existing
    # databases above; the same dialect-neutral upgrade (column,
    # per-scope backfill, seeded counter, unique index) runs for every
    # backend. A fresh database already has all three and the upgrade is
    # a no-op there.
    _migrate_proof_event_commit_sequence(engine)
    # The compliance decision query gained the same per-scope commit-order
    # marker together with denormalized scope columns; legacy databases
    # are upgraded identically on every backend.
    _migrate_decision_commit_sequence(engine)
    # The trust-root lifecycle query gained the per-scope sequence shared
    # by creation and retirement; legacy databases (columns added above on
    # SQLite) are completed identically on every backend.
    _migrate_trust_root_commit_sequence(engine)
    # The policy lifecycle query gained the analogous per-scope sequence
    # shared by version creation and the terminal retirement.
    _migrate_policy_commit_sequence(engine)
    # The read-only data-envelope directory query gained the per-scope
    # commit-order marker that fixes its replayable snapshot.
    _migrate_data_envelope_commit_sequence(engine)
    # The compliance audit events gained their tamper-evident hash chain
    # (per-event link metadata plus the per-scope range head) after the
    # audit table first existed; legacy databases (columns added above on
    # SQLite) are completed identically on every backend.
    _migrate_audit_event_chain(engine)


def _migrate_proof_event_commit_sequence(engine) -> None:
    """Bring a deployment written before ``commit_seq`` up to date.

    The proof-lifecycle audit fixes its replayable snapshot to the
    business commit boundary through a per-scope, gap-free, strictly
    increasing ``commit_seq`` allocated in each event's own write
    transaction. Databases written before the sequence existed must be
    upgraded on open *on every backend*, not only SQLite:

    * the nullable ``commit_seq`` column is added when missing (SQLite's
      ALTER comes from the additive table map above; locking backends get
      it here);
    * legacy rows are backfilled per scope in a stable order — SQLite's
      ``rowid`` (its serialized insert order) or the audit listing key
      ``(occurred_at, event_id)`` numbered client-side on every other
      dialect — so each scope gets a gap-free 1..N run even when business
      times are inverted or identical;
    * each scope's :class:`ProofEventCommitCounter` is seeded at its
      backfilled maximum (only when absent) so the first event written
      after the upgrade allocates N+1 rather than colliding;
    * the unique ``(tenant_id, workload_id, commit_seq)`` index is
      created last, once no NULL remains, so two concurrent allocations
      can never mint the same sequence.

    The whole upgrade is one transaction and changes no business field,
    response shape or secret-handling rule. A database created by the
    current metadata already has the column, counters and index, so each
    step short-circuits.
    """
    is_sqlite = engine.dialect.name == "sqlite"
    events_tbl = ProofLifecycleEvent.__table__
    counters_tbl = ProofEventCommitCounter.__table__

    # A table created by a pre-proof-timeline deployment need not exist at
    # all yet; create_all has just run on the current metadata, so on such
    # a database the table (with every column and index) now exists empty
    # and nothing below has work to do.
    with engine.begin() as conn:
        inspector = inspect(conn)
        if events_tbl.name not in inspector.get_table_names():
            return
        column_names = {col["name"] for col in inspector.get_columns(events_tbl.name)}
        if "commit_seq" not in column_names:
            if is_sqlite:
                conn.execute(
                    text(
                        f"ALTER TABLE {events_tbl.name} ADD COLUMN commit_seq BIGINT"
                    )
                )
            else:
                # Locking backend carrying a pre-sequence table: add the
                # column nullable (legacy rows are populated below before
                # the unique index is built).
                conn.execute(
                    text(
                        f"ALTER TABLE {events_tbl.name} "
                        f"ADD COLUMN commit_seq BIGINT NULL"
                    )
                )

        # Backfill only legacy rows (never touch an already-sequenced row).
        if is_sqlite:
            # rowid is SQLite's serialized insert/commit order; the
            # correlated COUNT assigns 1..N per scope in that order.
            conn.execute(
                text(
                    f"UPDATE {events_tbl.name} "
                    "SET commit_seq = ("
                    "SELECT COUNT(*) FROM proof_lifecycle_events AS prior "
                    "WHERE prior.tenant_id = proof_lifecycle_events.tenant_id "
                    "AND prior.workload_id = proof_lifecycle_events.workload_id "
                    "AND prior.rowid <= proof_lifecycle_events.rowid"
                    ") "
                    "WHERE commit_seq IS NULL"
                )
            )
        else:
            # Historical commit order on a backend with no rowid: the
            # pre-sequence deployment carried no monotonic commit marker
            # (the primary key is a UUID v4 minted in the event's own
            # transaction, whose lexical order is not commit order).
            # Number legacy rows in Python by the audit's own stable
            # listing key — per scope, (occurred_at, event_id) ascending —
            # and write each row's one-based position by primary key.
            # Doing it client-side rather than with a correlated
            # self-update keeps the upgrade identical across locking
            # dialects (several reject or mis-merge a self-referencing
            # UPDATE). The exact historical tie-break is immaterial to
            # snapshot correctness: every legacy row committed before the
            # upgrade and is therefore inside every fresh first query
            # regardless of its position relative to other legacy rows.
            legacy_rows = conn.execute(
                select(
                    events_tbl.c.event_id,
                    events_tbl.c.tenant_id,
                    events_tbl.c.workload_id,
                )
                .where(events_tbl.c.commit_seq.is_(None))
                .order_by(
                    events_tbl.c.tenant_id,
                    events_tbl.c.workload_id,
                    events_tbl.c.occurred_at,
                    events_tbl.c.event_id,
                )
            ).fetchall()
            per_scope: dict[tuple[str, str], int] = {}
            for event_id, tenant_id, workload_id in legacy_rows:
                seq = per_scope.get((tenant_id, workload_id), 0) + 1
                per_scope[(tenant_id, workload_id)] = seq
                conn.execute(
                    events_tbl.update()
                    .where(
                        events_tbl.c.event_id == event_id,
                        events_tbl.c.commit_seq.is_(None),
                    )
                    .values(commit_seq=seq)
                )

        # Seed a per-scope counter at the existing maximum, but never
        # overwrite a counter that is already present (a concurrent
        # upgrader or a scope already writing events). The
        # (tenant_id, workload_id) primary key makes the anti-duplicate
        # portable across dialects.
        conn.execute(
            insert(counters_tbl)
            .from_select(
                ["tenant_id", "workload_id", "last_seq"],
                select(
                    events_tbl.c.tenant_id,
                    events_tbl.c.workload_id,
                    func.max(events_tbl.c.commit_seq),
                )
                .where(events_tbl.c.commit_seq.is_not(None))
                .group_by(
                    events_tbl.c.tenant_id, events_tbl.c.workload_id
                )
                .where(
                    ~select(literal_column("1"))
                    .select_from(counters_tbl.alias("existing_counter"))
                    .where(
                        literal_column("existing_counter.tenant_id")
                        == events_tbl.c.tenant_id,
                        literal_column("existing_counter.workload_id")
                        == events_tbl.c.workload_id,
                    )
                    .exists()
                ),
            )
        )

        # Build the unique ordering index last, once every row is
        # sequenced. Reuse the index object already declared on the
        # model's table (never construct a second Index bound to the same
        # columns — that would register a duplicate on the shared
        # metadata and make every later create_all attempt the index
        # twice); checkfirst makes creation a no-op on current-metadata
        # databases and every other dialect portably.
        index_names = {idx["name"] for idx in inspector.get_indexes(events_tbl.name)}
        if "ix_proof_lifecycle_events_scope_commit_seq" not in index_names:
            model_index = next(
                idx
                for idx in events_tbl.indexes
                if idx.name == "ix_proof_lifecycle_events_scope_commit_seq"
            )
            model_index.create(conn, checkfirst=True)


def _migrate_decision_commit_sequence(engine) -> None:
    """Bring a deployment written before decision scoping/sequencing up to date.

    The read-only compliance decision query scopes decisions directly and
    fixes its replayable snapshot through a per-scope, gap-free, strictly
    increasing ``commit_seq`` allocated in each decision's own write
    transaction. The original decision table carried neither scope columns
    (a decision's scope was reachable only through its evidence row) nor
    the sequence, so databases written by older deployments are upgraded
    on open on every backend:

    * the nullable ``tenant_id``/``workload_id``/``commit_seq`` columns are
      added when missing (SQLite's ALTER comes from the additive table map;
      locking backends get them here);
    * legacy rows inherit their scope from their evidence row (correlated
      subquery UPDATE, portable across the supported dialects);
    * legacy rows are numbered per scope in a stable order — SQLite's
      ``rowid`` (its serialized insert order) or the listing key
      ``(decided_at, decision_id)`` numbered client-side on every other
      dialect — so each scope gets a gap-free 1..N run even when business
      times are inverted or identical;
    * each scope's :class:`DecisionCommitCounter` is seeded at its
      backfilled maximum (only when absent) so the first decision written
      after the upgrade allocates N+1 rather than colliding;
    * the unique ``(tenant_id, workload_id, commit_seq)`` index is created
      last, once no remaining row carries a NULL sequence, and the scoped
      listing index is (re)created checkfirst for parity on legacy files.

    The whole upgrade changes no business field, response shape or
    secret-handling rule. A database created by the current metadata
    already has every column, counter and index, so each step
    short-circuits.
    """
    is_sqlite = engine.dialect.name == "sqlite"
    decisions_tbl = Decision.__table__
    counters_tbl = DecisionCommitCounter.__table__

    # A pre-decisions deployment need not have the table at all yet;
    # create_all has just run on the current metadata, so on such a
    # database the table exists empty with every column and index and the
    # steps below have no work to do.
    with engine.begin() as conn:
        inspector = inspect(conn)
        if decisions_tbl.name not in inspector.get_table_names():
            return
        column_names = {
            col["name"] for col in inspector.get_columns(decisions_tbl.name)
        }
        if not is_sqlite:
            # Locking backend carrying a pre-scoping/pre-sequence table:
            # add the columns nullable; legacy rows are populated below
            # before the unique index is built.
            for column, column_type in (
                ("tenant_id", "VARCHAR(256) NULL"),
                ("workload_id", "VARCHAR(256) NULL"),
                ("commit_seq", "BIGINT NULL"),
            ):
                if column not in column_names:
                    conn.execute(
                        text(
                            f"ALTER TABLE {decisions_tbl.name} "
                            f"ADD COLUMN {column} {column_type}"
                        )
                    )

        # Inherit scope from the evidence row for legacy decisions that
        # predate the denormalized columns. By this point the columns
        # exist on every path (current metadata, SQLite's additive ALTER
        # above, or the ALTER just performed on locking backends). A
        # decision always belonged to exactly one evidence in exactly one
        # scope.
        scope_rows = conn.execute(
            select(func.count())
            .select_from(decisions_tbl)
            .where(decisions_tbl.c.tenant_id.is_(None))
        ).scalar()
        if scope_rows:
            conn.execute(
                decisions_tbl.update()
                .where(decisions_tbl.c.tenant_id.is_(None))
                .values(
                    tenant_id=select(Evidence.tenant_id)
                    .where(Evidence.evidence_id == decisions_tbl.c.evidence_id)
                    .scalar_subquery(),
                    workload_id=select(Evidence.workload_id)
                    .where(Evidence.evidence_id == decisions_tbl.c.evidence_id)
                    .scalar_subquery(),
                )
            )

        # Backfill only legacy rows (never touch an already-sequenced row).
        if is_sqlite:
            # rowid is SQLite's serialized insert/commit order; the
            # correlated COUNT assigns 1..N per scope in that order.
            conn.execute(
                text(
                    f"UPDATE {decisions_tbl.name} "
                    "SET commit_seq = ("
                    "SELECT COUNT(*) FROM decisions AS prior "
                    "WHERE prior.tenant_id = decisions.tenant_id "
                    "AND prior.workload_id = decisions.workload_id "
                    "AND prior.rowid <= decisions.rowid"
                    ") "
                    "WHERE commit_seq IS NULL"
                )
            )
        else:
            # No rowid on a locking backend: number legacy rows in Python
            # by the query's own stable listing key — per scope,
            # (decided_at, decision_id) ascending — and write each row's
            # one-based position by primary key. The exact historical
            # tie-break is immaterial to snapshot correctness: every legacy
            # row committed before the upgrade and is inside every fresh
            # first query regardless of its position relative to other
            # legacy rows.
            legacy_rows = conn.execute(
                select(
                    decisions_tbl.c.decision_id,
                    decisions_tbl.c.tenant_id,
                    decisions_tbl.c.workload_id,
                )
                .where(decisions_tbl.c.commit_seq.is_(None))
                .order_by(
                    decisions_tbl.c.tenant_id,
                    decisions_tbl.c.workload_id,
                    decisions_tbl.c.decided_at,
                    decisions_tbl.c.decision_id,
                )
            ).fetchall()
            per_scope: dict[tuple, int] = {}
            for decision_id, tenant_id, workload_id in legacy_rows:
                seq = per_scope.get((tenant_id, workload_id), 0) + 1
                per_scope[(tenant_id, workload_id)] = seq
                conn.execute(
                    decisions_tbl.update()
                    .where(
                        decisions_tbl.c.decision_id == decision_id,
                        decisions_tbl.c.commit_seq.is_(None),
                    )
                    .values(commit_seq=seq)
                )

        # Seed a per-scope counter at the existing maximum, never
        # overwriting a counter that is already present.
        conn.execute(
            insert(counters_tbl)
            .from_select(
                ["tenant_id", "workload_id", "last_seq"],
                select(
                    decisions_tbl.c.tenant_id,
                    decisions_tbl.c.workload_id,
                    func.max(decisions_tbl.c.commit_seq),
                )
                .where(decisions_tbl.c.commit_seq.is_not(None))
                .group_by(decisions_tbl.c.tenant_id, decisions_tbl.c.workload_id)
                .where(
                    ~select(literal_column("1"))
                    .select_from(counters_tbl.alias("existing_counter"))
                    .where(
                        literal_column("existing_counter.tenant_id")
                        == decisions_tbl.c.tenant_id,
                        literal_column("existing_counter.workload_id")
                        == decisions_tbl.c.workload_id,
                    )
                    .exists()
                ),
            )
        )

        # (Re)create the model-declared indexes checkfirst. A current
        # metadata database already has both; a legacy file receives the
        # unique commit-order index last (every legacy row is sequenced)
        # plus the scoped keyset listing index.
        index_names = {
            idx["name"] for idx in inspector.get_indexes(decisions_tbl.name)
        }
        for index_name in (
            "ix_decisions_scope_commit_seq",
            "ix_decisions_scope_decided",
        ):
            if index_name not in index_names:
                model_index = next(
                    idx for idx in decisions_tbl.indexes if idx.name == index_name
                )
                model_index.create(conn, checkfirst=True)


def _migrate_trust_root_commit_sequence(engine) -> None:
    """Bring a deployment written before the trust-root sequence up to date.

    The read-only trust-root lifecycle query fixes its replayable snapshot
    through a per-scope, gap-free, strictly increasing lifecycle sequence
    drawn from one counter by *both* creation (``commit_seq``) and the
    terminal retirement (``retired_seq``). The original trust-root table
    carried neither marker, so databases written by older deployments are
    upgraded on open on every backend:

    * the nullable ``commit_seq``/``retired_seq`` columns are added when
      missing (SQLite's ALTER comes from the additive table map; locking
      backends get them here);
    * legacy rows are numbered per scope client-side in the listing key
      order — each root's creation 1..N by ``(created_at, root_id)``, then
      each retired root's retirement N+1..N+R by ``(retired_at, root_id)``
      — so every sequence is gap-free, unique and strictly after the
      root's own creation, and an active legacy row keeps
      ``retired_seq NULL``;
    * each scope's :class:`TrustRootCommitCounter` is seeded at the
      backfilled maximum N+R (only when absent) so the next creation or
      retirement allocates N+R+1 rather than colliding;
    * the unique ``(tenant_id, workload_id, commit_seq)`` index and the
      scoped listing index are (re)created checkfirst.

    No snapshot spanning the upgrade could ever have existed (no cursors
    predate this change), so the exact relative order of legacy creations
    and retirements is immaterial: every legacy commit predates the
    upgrade and lies inside every fresh first query, where every legacy
    retired root correctly reconstructs as retired (its retired_seq is at
    or below the seeded maximum) and every legacy active root as active.
    The upgrade changes no business field, response shape or
    secret-handling rule; a current-metadata database short-circuits.
    """
    is_sqlite = engine.dialect.name == "sqlite"
    roots_tbl = TrustRoot.__table__
    counters_tbl = TrustRootCommitCounter.__table__

    with engine.begin() as conn:
        inspector = inspect(conn)
        if roots_tbl.name not in inspector.get_table_names():
            return
        column_names = {col["name"] for col in inspector.get_columns(roots_tbl.name)}
        if not is_sqlite:
            for column, column_type in (
                ("commit_seq", "BIGINT NULL"),
                ("retired_seq", "BIGINT NULL"),
            ):
                if column not in column_names:
                    conn.execute(
                        text(
                            f"ALTER TABLE {roots_tbl.name} "
                            f"ADD COLUMN {column} {column_type}"
                        )
                    )

        # Number legacy rows in Python, identically on every backend.
        # Rows that already carry a sequence (an upgrade partially applied
        # or a current-metadata database) are left untouched.
        legacy_scopes = conn.execute(
            select(
                roots_tbl.c.tenant_id,
                roots_tbl.c.workload_id,
            )
            .where(roots_tbl.c.commit_seq.is_(None))
            .group_by(roots_tbl.c.tenant_id, roots_tbl.c.workload_id)
        ).fetchall()

        per_scope_max: dict[tuple[str, str], int] = {}
        for tenant_id, workload_id in legacy_scopes:
            # Creation order follows the lifecycle listing key.
            ordered = conn.execute(
                select(
                    roots_tbl.c.root_id,
                    roots_tbl.c.status,
                    roots_tbl.c.retired_at,
                )
                .where(
                    roots_tbl.c.tenant_id == tenant_id,
                    roots_tbl.c.workload_id == workload_id,
                    roots_tbl.c.commit_seq.is_(None),
                )
                .order_by(roots_tbl.c.created_at, roots_tbl.c.root_id)
            ).fetchall()
            retired = [
                (root_id, retired_at)
                for root_id, status, retired_at in ordered
                if status == TRUST_ROOT_STATUS_RETIRED
            ]
            retired.sort(key=lambda item: (item[1], item[0]))
            for seq, (root_id, _status, _retired_at) in enumerate(ordered, start=1):
                conn.execute(
                    roots_tbl.update()
                    .where(
                        roots_tbl.c.root_id == root_id,
                        roots_tbl.c.commit_seq.is_(None),
                    )
                    .values(commit_seq=seq)
                )
            n_roots = len(ordered)
            # Retirements follow every legacy creation in the scope, in
            # retirement-time order.
            for offset, (root_id, _retired_at) in enumerate(retired):
                conn.execute(
                    roots_tbl.update()
                    .where(
                        roots_tbl.c.root_id == root_id,
                        roots_tbl.c.retired_seq.is_(None),
                    )
                    .values(retired_seq=n_roots + offset + 1)
                )
            per_scope_max[(tenant_id, workload_id)] = n_roots + len(retired)

        # Seed a per-scope counter at the backfilled maximum, never
        # overwriting a counter already present.
        for (tenant_id, workload_id), last_seq in per_scope_max.items():
            existing = conn.execute(
                select(func.count())
                .select_from(counters_tbl)
                .where(
                    counters_tbl.c.tenant_id == tenant_id,
                    counters_tbl.c.workload_id == workload_id,
                )
            ).scalar()
            if not existing:
                conn.execute(
                    counters_tbl.insert().values(
                        tenant_id=tenant_id,
                        workload_id=workload_id,
                        last_seq=last_seq,
                    )
                )

        # (Re)create the model-declared indexes checkfirst.
        index_names = {idx["name"] for idx in inspector.get_indexes(roots_tbl.name)}
        for index_name in (
            "ix_trust_roots_scope_commit_seq",
            "ix_trust_roots_scope_created",
        ):
            if index_name not in index_names:
                model_index = next(
                    idx for idx in roots_tbl.indexes if idx.name == index_name
                )
                model_index.create(conn, checkfirst=True)


def _migrate_policy_commit_sequence(engine) -> None:
    """Bring a deployment written before the policy sequence up to date.

    The read-only policy lifecycle query fixes its replayable snapshot
    through a per-scope, gap-free, strictly increasing lifecycle sequence
    drawn from one counter by *both* version creation (``commit_seq``) and
    the terminal retirement (``retired_seq``). The original policy table
    carried neither marker, so databases written by older deployments are
    upgraded on open on every backend:

    * the nullable ``commit_seq``/``retired_seq`` columns are added when
      missing (SQLite's ALTER comes from the additive table map; locking
      backends get them here);
    * legacy rows are numbered per scope client-side in the listing key
      order — each version's creation 1..N by ``(created_at, policy_id)``,
      then each retired version's retirement N+1..N+R by
      ``(retired_at, policy_id)`` — so every sequence is gap-free, unique
      and strictly after the version's own creation, and an active legacy
      row keeps ``retired_seq NULL``;
    * each scope's :class:`PolicyCommitCounter` is seeded at the
      backfilled maximum N+R (only when absent) so the next creation or
      retirement allocates N+R+1 rather than colliding;
    * the unique ``(tenant_id, workload_id, commit_seq)`` index and the
      scoped listing index are (re)created checkfirst.

    No snapshot spanning the upgrade could ever have existed (no cursors
    predate this change), so the exact relative order of legacy creations
    and retirements is immaterial: every legacy commit predates the
    upgrade and lies inside every fresh first query, where every legacy
    retired version correctly reconstructs as retired (its retired_seq is
    at or below the seeded maximum) and every legacy active version as
    active. The upgrade changes no business field, response shape or
    secret-handling rule; a current-metadata database short-circuits.
    """
    is_sqlite = engine.dialect.name == "sqlite"
    policies_tbl = Policy.__table__
    counters_tbl = PolicyCommitCounter.__table__

    with engine.begin() as conn:
        inspector = inspect(conn)
        if policies_tbl.name not in inspector.get_table_names():
            return
        column_names = {
            col["name"] for col in inspector.get_columns(policies_tbl.name)
        }
        if not is_sqlite:
            for column, column_type in (
                ("commit_seq", "BIGINT NULL"),
                ("retired_seq", "BIGINT NULL"),
            ):
                if column not in column_names:
                    conn.execute(
                        text(
                            f"ALTER TABLE {policies_tbl.name} "
                            f"ADD COLUMN {column} {column_type}"
                        )
                    )

        # Number legacy rows in Python, identically on every backend.
        # Rows that already carry a sequence (an upgrade partially applied
        # or a current-metadata database) are left untouched.
        legacy_scopes = conn.execute(
            select(
                policies_tbl.c.tenant_id,
                policies_tbl.c.workload_id,
            )
            .where(policies_tbl.c.commit_seq.is_(None))
            .group_by(policies_tbl.c.tenant_id, policies_tbl.c.workload_id)
        ).fetchall()

        per_scope_max: dict[tuple[str, str], int] = {}
        for tenant_id, workload_id in legacy_scopes:
            # Creation order follows the lifecycle listing key.
            ordered = conn.execute(
                select(
                    policies_tbl.c.policy_id,
                    policies_tbl.c.status,
                    policies_tbl.c.retired_at,
                )
                .where(
                    policies_tbl.c.tenant_id == tenant_id,
                    policies_tbl.c.workload_id == workload_id,
                    policies_tbl.c.commit_seq.is_(None),
                )
                .order_by(policies_tbl.c.created_at, policies_tbl.c.policy_id)
            ).fetchall()
            retired = [
                (policy_id, retired_at)
                for policy_id, status, retired_at in ordered
                if status == POLICY_STATUS_RETIRED
            ]
            retired.sort(key=lambda item: (item[1], item[0]))
            for seq, (policy_id, _status, _retired_at) in enumerate(
                ordered, start=1
            ):
                conn.execute(
                    policies_tbl.update()
                    .where(
                        policies_tbl.c.policy_id == policy_id,
                        policies_tbl.c.commit_seq.is_(None),
                    )
                    .values(commit_seq=seq)
                )
            n_versions = len(ordered)
            # Retirements follow every legacy creation in the scope, in
            # retirement-time order.
            for offset, (policy_id, _retired_at) in enumerate(retired):
                conn.execute(
                    policies_tbl.update()
                    .where(
                        policies_tbl.c.policy_id == policy_id,
                        policies_tbl.c.retired_seq.is_(None),
                    )
                    .values(retired_seq=n_versions + offset + 1)
                )
            per_scope_max[(tenant_id, workload_id)] = n_versions + len(retired)

        # Seed a per-scope counter at the backfilled maximum, never
        # overwriting a counter already present.
        for (tenant_id, workload_id), last_seq in per_scope_max.items():
            existing = conn.execute(
                select(func.count())
                .select_from(counters_tbl)
                .where(
                    counters_tbl.c.tenant_id == tenant_id,
                    counters_tbl.c.workload_id == workload_id,
                )
            ).scalar()
            if not existing:
                conn.execute(
                    counters_tbl.insert().values(
                        tenant_id=tenant_id,
                        workload_id=workload_id,
                        last_seq=last_seq,
                    )
                )

        # (Re)create the model-declared indexes checkfirst.
        index_names = {idx["name"] for idx in inspector.get_indexes(policies_tbl.name)}
        for index_name in (
            "ix_policies_scope_commit_seq",
            "ix_policies_scope_created",
        ):
            if index_name not in index_names:
                model_index = next(
                    idx for idx in policies_tbl.indexes if idx.name == index_name
                )
                model_index.create(conn, checkfirst=True)


def _migrate_data_envelope_commit_sequence(engine) -> None:
    """Bring a deployment written before the envelope sequence up to date.

    The read-only data-envelope directory query fixes its replayable
    snapshot to the business commit boundary through a per-scope,
    gap-free, strictly increasing ``commit_seq`` allocated in each
    envelope's own creation transaction. The original ``data_envelopes``
    table already carried the scope columns but not the sequence, so
    databases written by older deployments are upgraded on open on every
    backend:

    * the nullable ``commit_seq`` column is added when missing (SQLite's
      ALTER comes from the additive table map; locking backends get it
      here);
    * legacy rows are numbered per scope in a stable order — SQLite's
      ``rowid`` (its serialized insert order) or the directory listing key
      ``data_id`` numbered client-side on every other dialect — so each
      scope gets a gap-free 1..N run;
    * each scope's :class:`DataEnvelopeCommitCounter` is seeded at its
      backfilled maximum (only when absent) so the first envelope created
      after the upgrade allocates N+1 rather than colliding;
    * the unique ``(tenant_id, workload_id, commit_seq)`` index is created
      last, once no remaining row carries a NULL sequence.

    No snapshot spanning the upgrade could ever have existed (no cursors
    predate this change), so the exact relative order of legacy rows is
    immaterial: every legacy envelope committed before the upgrade and
    lies inside every fresh first query regardless of its assigned
    position. Rewrap updates only key material and never advances the
    sequence. The upgrade changes no business field, response shape or
    secret-handling rule; a current-metadata database short-circuits.
    """
    is_sqlite = engine.dialect.name == "sqlite"
    envelopes_tbl = DataEnvelope.__table__
    counters_tbl = DataEnvelopeCommitCounter.__table__

    with engine.begin() as conn:
        inspector = inspect(conn)
        if envelopes_tbl.name not in inspector.get_table_names():
            return
        column_names = {
            col["name"] for col in inspector.get_columns(envelopes_tbl.name)
        }
        if not is_sqlite and "commit_seq" not in column_names:
            conn.execute(
                text(
                    f"ALTER TABLE {envelopes_tbl.name} "
                    "ADD COLUMN commit_seq BIGINT NULL"
                )
            )

        # Backfill only legacy rows (never touch an already-sequenced row).
        if is_sqlite:
            # rowid is SQLite's serialized insert/commit order; the
            # correlated COUNT assigns 1..N per scope in that order.
            conn.execute(
                text(
                    f"UPDATE {envelopes_tbl.name} "
                    "SET commit_seq = ("
                    "SELECT COUNT(*) FROM data_envelopes AS prior "
                    "WHERE prior.tenant_id = data_envelopes.tenant_id "
                    "AND prior.workload_id = data_envelopes.workload_id "
                    "AND prior.rowid <= data_envelopes.rowid"
                    ") "
                    "WHERE commit_seq IS NULL"
                )
            )
        else:
            # No rowid on a locking backend: number legacy rows in Python
            # per scope by the directory's own listing key, data_id, and
            # write each row's one-based position by primary key. The
            # exact historical tie-break is immaterial to snapshot
            # correctness: every legacy row committed before the upgrade
            # and is inside every fresh first query.
            legacy_rows = conn.execute(
                select(
                    envelopes_tbl.c.tenant_id,
                    envelopes_tbl.c.workload_id,
                    envelopes_tbl.c.data_id,
                )
                .where(envelopes_tbl.c.commit_seq.is_(None))
                .order_by(
                    envelopes_tbl.c.tenant_id,
                    envelopes_tbl.c.workload_id,
                    envelopes_tbl.c.data_id,
                )
            ).fetchall()
            per_scope: dict[tuple, int] = {}
            for tenant_id, workload_id, data_id in legacy_rows:
                seq = per_scope.get((tenant_id, workload_id), 0) + 1
                per_scope[(tenant_id, workload_id)] = seq
                conn.execute(
                    envelopes_tbl.update()
                    .where(
                        envelopes_tbl.c.tenant_id == tenant_id,
                        envelopes_tbl.c.workload_id == workload_id,
                        envelopes_tbl.c.data_id == data_id,
                        envelopes_tbl.c.commit_seq.is_(None),
                    )
                    .values(commit_seq=seq)
                )

        # Seed a per-scope counter at the existing maximum, never
        # overwriting a counter that is already present.
        conn.execute(
            insert(counters_tbl)
            .from_select(
                ["tenant_id", "workload_id", "last_seq"],
                select(
                    envelopes_tbl.c.tenant_id,
                    envelopes_tbl.c.workload_id,
                    func.max(envelopes_tbl.c.commit_seq),
                )
                .where(envelopes_tbl.c.commit_seq.is_not(None))
                .group_by(envelopes_tbl.c.tenant_id, envelopes_tbl.c.workload_id)
                .where(
                    ~select(literal_column("1"))
                    .select_from(counters_tbl.alias("existing_counter"))
                    .where(
                        literal_column("existing_counter.tenant_id")
                        == envelopes_tbl.c.tenant_id,
                        literal_column("existing_counter.workload_id")
                        == envelopes_tbl.c.workload_id,
                    )
                    .exists()
                ),
            )
        )

        # (Re)create the model-declared unique commit-order index
        # checkfirst, once every legacy row is sequenced. A current
        # metadata database already has it.
        index_names = {
            idx["name"] for idx in inspector.get_indexes(envelopes_tbl.name)
        }
        if "ix_data_envelopes_scope_commit_seq" not in index_names:
            model_index = next(
                idx
                for idx in envelopes_tbl.indexes
                if idx.name == "ix_data_envelopes_scope_commit_seq"
            )
            model_index.create(conn, checkfirst=True)


def _migrate_audit_event_chain(engine) -> None:
    """Bring a deployment written before the audit hash chain up to date.

    Every committed audit event carries its chain metadata (per-scope
    ``chain_seq``, ``prev_hash`` and ``event_hash``) and each scope a
    range head row, all allocated in the event's own transaction.
    Databases written before the chain existed must be upgraded on open
    *on every backend*, not only SQLite:

    * the nullable ``chain_seq``/``prev_hash``/``event_hash`` columns are
      added when missing (SQLite's ALTERs come from the additive table map
      above; locking backends get them here);
    * legacy rows are chained per scope in the audit's stable listing
      order — ``(occurred_at, event_id)`` ascending, numbered client-side
      so the upgrade is identical across dialects — giving each scope a
      gap-free 1..N run whose hashes cover the stored fields;
    * each scope's :class:`AuditChainHead` is created (only when absent)
      at the backfilled tip with ``legacy_count`` set to the number of
      backfilled events, so the first event written after the upgrade
      allocates N+1 chained onto the legacy tip rather than colliding;
    * the unique ``(tenant_id, workload_id, chain_seq)`` index is created
      last, once no NULL sequence remains, so two concurrent appends can
      never mint the same sequence.

    The whole upgrade is one transaction and changes no business field,
    response shape or secret-handling rule. A database created by the
    current metadata already has the columns, heads and index, so each
    step short-circuits.
    """
    is_sqlite = engine.dialect.name == "sqlite"
    events_tbl = AuditEvent.__table__
    heads_tbl = AuditChainHead.__table__

    # A table created by a pre-audit deployment need not exist at all yet;
    # create_all has just run on the current metadata, so on such a
    # database the table (with every column and index) now exists empty
    # and nothing below has work to do.
    with engine.begin() as conn:
        inspector = inspect(conn)
        if events_tbl.name not in inspector.get_table_names():
            return
        column_names = {col["name"] for col in inspector.get_columns(events_tbl.name)}
        if not is_sqlite:
            for column, column_type in (
                ("chain_seq", "BIGINT NULL"),
                ("prev_hash", "VARCHAR(64) NULL"),
                ("event_hash", "VARCHAR(64) NULL"),
            ):
                if column not in column_names:
                    conn.execute(
                        text(
                            f"ALTER TABLE {events_tbl.name} "
                            f"ADD COLUMN {column} {column_type}"
                        )
                    )

        # Chain only legacy rows (never touch an already-chained row), per
        # scope in the audit listing order (occurred_at, event_id).
        legacy_rows = conn.execute(
            select(
                events_tbl.c.event_id,
                events_tbl.c.tenant_id,
                events_tbl.c.workload_id,
                events_tbl.c.event_type,
                events_tbl.c.grant_id,
                events_tbl.c.decision_id,
                events_tbl.c.data_id,
                events_tbl.c.status,
                events_tbl.c.capability_sha256,
                events_tbl.c.occurred_at,
            )
            .where(events_tbl.c.chain_seq.is_(None))
            .order_by(
                events_tbl.c.tenant_id,
                events_tbl.c.workload_id,
                events_tbl.c.occurred_at,
                events_tbl.c.event_id,
            )
        ).fetchall()

        # scope -> [last_seq, tip_hash, backfilled_count], continued from
        # any head the scope already has (a fresh scope starts at the
        # genesis anchor).
        scopes: dict[tuple[str, str], list] = {}
        for row in legacy_rows:
            scope = (row.tenant_id, row.workload_id)
            state = scopes.get(scope)
            if state is None:
                existing_head = conn.execute(
                    select(
                        heads_tbl.c.last_seq,
                        heads_tbl.c.head_hash,
                        heads_tbl.c.legacy_count,
                    ).where(
                        heads_tbl.c.tenant_id == row.tenant_id,
                        heads_tbl.c.workload_id == row.workload_id,
                    )
                ).fetchone()
                if existing_head is not None:
                    state = [
                        existing_head.last_seq,
                        existing_head.head_hash,
                        0,
                        existing_head.legacy_count,
                    ]
                else:
                    state = [0, _AUDIT_CHAIN_GENESIS_HASH, 0, 0]
                scopes[scope] = state
            chain_seq = state[0] + 1
            prev_hash = state[1]
            event_hash = _audit_event_chain_hash(
                chain_seq=chain_seq,
                prev_hash=prev_hash,
                tenant_id=row.tenant_id,
                workload_id=row.workload_id,
                event_id=row.event_id,
                event_type=row.event_type,
                grant_id=row.grant_id,
                decision_id=row.decision_id,
                data_id=row.data_id,
                status=row.status,
                capability_sha256=row.capability_sha256,
                occurred_at=row.occurred_at,
            )
            conn.execute(
                events_tbl.update()
                .where(
                    events_tbl.c.event_id == row.event_id,
                    events_tbl.c.chain_seq.is_(None),
                )
                .values(
                    chain_seq=chain_seq,
                    prev_hash=prev_hash,
                    event_hash=event_hash,
                )
            )
            state[0] = chain_seq
            state[1] = event_hash
            state[2] += 1

        # Seal each touched scope's range head at the backfilled tip,
        # recording how many of its events were chained by this upgrade.
        # A head that already exists is advanced, never replaced.
        for (tenant_id, workload_id), state in scopes.items():
            last_seq, tip_hash, backfilled, base_legacy = state
            updated = conn.execute(
                heads_tbl.update()
                .where(
                    heads_tbl.c.tenant_id == tenant_id,
                    heads_tbl.c.workload_id == workload_id,
                )
                .values(
                    last_seq=last_seq,
                    head_hash=tip_hash,
                    legacy_count=base_legacy + backfilled,
                )
            )
            if updated.rowcount == 0:
                conn.execute(
                    heads_tbl.insert().values(
                        tenant_id=tenant_id,
                        workload_id=workload_id,
                        last_seq=last_seq,
                        head_hash=tip_hash,
                        legacy_count=backfilled,
                    )
                )

        # Build the unique chain-order index last, once every row is
        # chained. Reuse the index object already declared on the model's
        # table (never construct a second Index bound to the same columns —
        # that would register a duplicate on the shared metadata and make
        # every later create_all attempt the index twice); checkfirst makes
        # creation a no-op on current-metadata databases and every other
        # dialect portably.
        index_names = {idx["name"] for idx in inspector.get_indexes(events_tbl.name)}
        if "ix_audit_events_scope_chain_seq" not in index_names:
            model_index = next(
                idx
                for idx in events_tbl.indexes
                if idx.name == "ix_audit_events_scope_chain_seq"
            )
            model_index.create(conn, checkfirst=True)


def _rebuild_release_grant_events(engine) -> None:
    """Reconstruct per-grant timeline events for a pre-feature database.

    The per-grant event timeline shipped after release grants themselves,
    so a database written by an older deployment holds grant rows (and the
    existing compliance audit rows) but no ``release_grant_events`` rows.
    ``create_all`` has just created the empty event table, so on open each
    existing grant is reconstructed into its gap-free per-grant sequence:

    * seq 1 is always the birth: no old status -> ``pending`` with reason
      ``issued`` at the grant's ``issued_at``;
    * a ``consumed`` grant gets seq 2 ``pending`` -> ``consumed`` at
      ``consumed_at``;
    * a ``revoked`` grant gets seq 2 ``pending`` -> ``revoked`` at
      ``revoked_at``.

    The pre-feature audit trail records a consume presentation and a
    payload release with the same (``grant``/``consumed``) event pair, so
    the two reasons cannot be told apart for a legacy settlement; the
    reconstructed seq 2 uses ``consume`` (the distinguishing reason exists
    only for migrations written by this and later deployments, which
    always record it faithfully). Event ids are deterministic from
    ``(grant_id, seq)`` so reopening the same database reproduces the same
    rows. The whole reconstruction is one transaction and is a no-op once
    any timeline event exists, so a fresh database and an already-upgraded
    one are both untouched; no grant, audit row, response shape or
    secret-handling rule changes.
    """
    events_tbl = ReleaseGrantEvent.__table__
    with engine.begin() as conn:
        existing = conn.execute(select(func.count()).select_from(events_tbl)).scalar()
        if existing:
            return
        legacy_grants = conn.execute(
            select(
                ReleaseGrant.grant_id,
                ReleaseGrant.tenant_id,
                ReleaseGrant.workload_id,
                ReleaseGrant.status,
                ReleaseGrant.issued_at,
                ReleaseGrant.consumed_at,
                ReleaseGrant.revoked_at,
            ).order_by(ReleaseGrant.grant_id)
        ).all()
        rows: list[dict] = []
        for (
            grant_id,
            tenant_id,
            workload_id,
            status,
            issued_at,
            consumed_at,
            revoked_at,
        ) in legacy_grants:
            rows.append(
                {
                    "event_id": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"release-grant-event:{grant_id}:1",
                        )
                    ),
                    "tenant_id": tenant_id,
                    "workload_id": workload_id,
                    "grant_id": grant_id,
                    "seq": 1,
                    "old_status": None,
                    "new_status": RELEASE_GRANT_STATUS_PENDING,
                    "reason": RELEASE_GRANT_EVENT_REASON_ISSUED,
                    "occurred_at": issued_at,
                }
            )
            if status == RELEASE_GRANT_STATUS_CONSUMED:
                rows.append(
                    {
                        "event_id": str(
                            uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                f"release-grant-event:{grant_id}:2",
                            )
                        ),
                        "tenant_id": tenant_id,
                        "workload_id": workload_id,
                        "grant_id": grant_id,
                        "seq": 2,
                        "old_status": RELEASE_GRANT_STATUS_PENDING,
                        "new_status": RELEASE_GRANT_STATUS_CONSUMED,
                        # The pre-feature audit cannot distinguish a
                        # payload release from a consume presentation.
                        "reason": RELEASE_GRANT_EVENT_REASON_CONSUME,
                        "occurred_at": consumed_at or issued_at,
                    }
                )
            elif status == RELEASE_GRANT_STATUS_REVOKED:
                rows.append(
                    {
                        "event_id": str(
                            uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                f"release-grant-event:{grant_id}:2",
                            )
                        ),
                        "tenant_id": tenant_id,
                        "workload_id": workload_id,
                        "grant_id": grant_id,
                        "seq": 2,
                        "old_status": RELEASE_GRANT_STATUS_PENDING,
                        "new_status": RELEASE_GRANT_STATUS_REVOKED,
                        "reason": RELEASE_GRANT_EVENT_REASON_REVOKED,
                        "occurred_at": revoked_at or issued_at,
                    }
                )
        if rows:
            conn.execute(events_tbl.insert(), rows)


async def _require_empty_query_body(request: Request) -> None:
    """Reject a read-only query carrying any request body (422).

    Read-only listings are ranged entirely by query parameters, so a body
    is never meaningful. The body is read directly (rather than trusting
    Content-Length, which chunked or HTTP/2 clients may omit) and anything
    non-empty — including whitespace or non-JSON — is rejected before any
    state is read.
    """
    if await request.body() != b"":
        raise HTTPException(status_code=422, detail="query body must be empty")


def _prom_label_escape(value: str) -> str:
    """Escape a label value per the Prometheus text exposition format.

    Only backslash, double quote and line feed are escaped (as ``\\\\``,
    ``\\"`` and ``\\n``); every other character is emitted verbatim. The
    escaping is applied to scope label values so a tenant or workload id
    containing any of these characters still produces a syntactically
    valid, unambiguous series line.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _json_safe(value):
    """Replace non-finite floats so a 422 detail can always be rendered.

    The JSON body parser accepts ``NaN``/``Infinity`` literals, so a
    rejected request (for example a rule with a non-finite ordinal bound)
    can carry such a value inside the echoed error input. The strict JSON
    renderer (``allow_nan=False``) cannot serialize it, which would turn a
    client error into a server error; non-finite floats are therefore
    replaced by their literal spelling. Finite values, and every error
    that never contained one, render exactly as before.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def create_app(
    database_url: str | None = None,
    verifier_registry: VerifierRegistry | None = None,
) -> FastAPI:
    url = database_url or os.environ.get(DATABASE_URL_ENV, DEFAULT_DATABASE_URL)
    connect_args = (
        {"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {}
    )
    engine = create_engine(url, connect_args=connect_args)
    if engine.dialect.name == "sqlite":
        # Take a writer lock at the start of every transaction so that
        # concurrent requests against the same evidence serialize rather
        # than racing to settle it. Mirrors the documented pysqlite
        # recipe: disable the driver's implicit BEGIN and emit our own.
        @event.listens_for(engine, "connect")
        def _disable_driver_autobegin(dbapi_connection, connection_record):
            dbapi_connection.isolation_level = None

        @event.listens_for(engine, "begin")
        def _begin_immediate(conn):
            conn.exec_driver_sql("BEGIN IMMEDIATE")

    Base.metadata.create_all(engine)
    _migrate_additive(engine)
    # Reconstruct the per-grant event timeline for grants written before
    # the timeline existed; a no-op on a fresh or already-upgraded database.
    _rebuild_release_grant_events(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    registry = verifier_registry or default_registry

    def _consume_grant_budget(tenant_id: str, workload_id: str) -> Response | None:
        """Reserve one slot of the shared per-scope, per-UTC-minute budget.

        Called only after request fields and the path grant identifier have
        passed basic validation. Returns ``None`` when the request is
        admitted (one durable slot consumed for the current UTC minute) or a
        ready 429 response when the minute's budget is exhausted. The
        reservation is committed in its own transaction *before* any business
        judgement runs, so a later 404/401/409/410/500 never refunds it and
        the quota survives restarts; a rejection writes nothing. A counter
        read or write that cannot be completed is a 500 and the caller never
        enters judgement.
        """
        now = _utcnow()
        window_start = _utc_minute_window(now)
        with session_factory() as session:
            try:
                admitted = False
                for _ in range(2):
                    # Lock the scope's minute row when one exists. SQLite
                    # ignores FOR UPDATE but every write transaction already
                    # begins as BEGIN IMMEDIATE, serializing concurrent
                    # reservations process-wide; on locking backends the row
                    # lock orders them.
                    row = session.scalar(
                        select(RateLimitCounter)
                        .where(
                            RateLimitCounter.tenant_id == tenant_id,
                            RateLimitCounter.workload_id == workload_id,
                            RateLimitCounter.window_start == window_start,
                        )
                        .with_for_update()
                    )
                    if row is None:
                        # The first valid request of the minute initializes
                        # the counter at one. A concurrent initializer on a
                        # locking backend may win the unique constraint; that
                        # race is retried as an increment below.
                        session.add(
                            RateLimitCounter(
                                tenant_id=tenant_id,
                                workload_id=workload_id,
                                window_start=window_start,
                                count=1,
                            )
                        )
                        try:
                            session.flush()
                        except IntegrityError:
                            session.rollback()
                            continue
                        admitted = True
                        break
                    if row.count >= GRANT_BUDGET_PER_MINUTE:
                        # Budget exhausted: no write, no audit, no state
                        # change; recompute the retry hint at response time.
                        session.rollback()
                        return _too_many_requests_response(_utcnow())
                    row.count = row.count + 1
                    admitted = True
                    break
                if not admitted:
                    # Defensive: the unique-insert retry loop failed to
                    # settle, which the single retry above makes unreachable.
                    session.rollback()
                    logger.error("grant rate-limit reservation could not settle")
                    raise HTTPException(
                        status_code=500, detail="rate limit unavailable"
                    )
                session.commit()
            except HTTPException:
                raise
            except Exception:
                session.rollback()
                logger.error("grant rate-limit counter unavailable")
                raise HTTPException(status_code=500, detail="rate limit unavailable")
        return None

    def _keep_verification_slot(
        session, tenant_id: str, workload_id: str, window_start: datetime
    ) -> None:
        """Keep a reserved verification slot when verification then fails.

        Called on every failure path *after* a verification slot was
        reserved (a verifier plugin failure, an unavailable trust-root,
        revocation, CRL or workload-identity registry, or any unexpected
        error): the consumed slot is never refunded, while the evidence
        itself stays ``received``. The reservation lives in the failing
        transaction, where the counter increment is the only pending write,
        so when that transaction is still healthy the increment is simply
        committed on its own; when the transaction is already broken (a
        failed registry query poisons it on locking backends) the increment
        is replayed in a fresh transaction against the original window. A
        failure to retain the slot can only be logged — the 500 response to
        the caller is unchanged either way.
        """
        try:
            session.commit()
            return
        except Exception:
            try:
                session.rollback()
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            with session_factory() as compensation:
                for _ in range(2):
                    row = compensation.scalar(
                        select(VerificationAdmissionCounter)
                        .where(
                            VerificationAdmissionCounter.tenant_id == tenant_id,
                            VerificationAdmissionCounter.workload_id
                            == workload_id,
                            VerificationAdmissionCounter.window_start
                            == window_start,
                        )
                        .with_for_update()
                    )
                    if row is None:
                        # The rolled-back reservation was the minute's
                        # first: re-install the row in a savepoint so a
                        # concurrent first insert costs only the savepoint.
                        try:
                            with compensation.begin_nested():
                                compensation.add(
                                    VerificationAdmissionCounter(
                                        tenant_id=tenant_id,
                                        workload_id=workload_id,
                                        window_start=window_start,
                                        count=1,
                                    )
                                )
                        except IntegrityError:
                            continue
                        break
                    row.count = row.count + 1
                    break
                compensation.commit()
        except Exception:
            logger.error(
                "verification rate-limit slot could not be retained"
            )

    # ------------------------------------------------------------------
    # Persistent asynchronous rewrap jobs
    # ------------------------------------------------------------------

    class RewrapJobRunner:
        """Background machinery that advances persistent rewrap jobs.
        A job is durable progress state. Advancement is page driven and
        every envelope is an independent commit, so a crash can lose at
        most the uncommitted attempt on one envelope; the job resumes from
        its last committed ``next_cursor``. A claim token makes
        advancement mutually exclusive across threads and processes
        (two POSTs, or a live process and a recovery sweep): only the
        runner holding the token for a job runs it, and every status
        transition is additionally guarded so queued/running can never be
        entered twice. The claim is held for the whole run and released
        when the job settles or the runner is shutting down between pages.
        """

        #: Sentinel returned by _process_one when this runner no longer
        #: owns the claim mid-envelope: distinct from a result code so the
        #: page loop does not mistake it for a failure it caused.
        _CLAIM_LOST = object()

        #: Sentinel returned by _process_one when an in-process cancel was
        #: requested before the current envelope committed: the whole
        #: envelope transaction (material change, audit event, job counters)
        #: has been rolled back, leaving the envelope and audit untouched,
        #: so the blocked cancel transaction can settle the job.
        _CANCELLED_BEFORE_COMMIT = object()

        def __init__(self, session_factory, *, max_workers: int = 4) -> None:
            self._session_factory = session_factory
            self._max_workers = max_workers
            # Bounded pool of daemon workers sharing one FIFO: jobs queued
            # while every worker is busy wait in durable ``queued`` state
            # until one frees — submissions are never dropped in memory
            # and survive as rows regardless. Workers are created by
            # start() (application lifespan), not in __init__, so an app
            # built without its lifespan (tests) advances nothing on its
            # own and the durable rows stay exactly as committed.
            self._queue: queue.SimpleQueue[tuple[str, bool] | None] = (
                queue.SimpleQueue()
            )
            self._workers: list[threading.Thread] = []
            self._started = threading.Event()
            self._shutdown = threading.Event()
            # Claim tokens held by this process's runner threads, keyed by
            # job id. A plain dict is sufficient for single-key access; all
            # real mutual exclusion is the guarded UPDATE in the database.
            self._claim_tokens: dict[str, str] = {}
            # Scope of each job this process currently holds a claim on,
            # keyed by job id and populated from the loaded row before any
            # envelope is processed. It lets a cancel endpoint raise its
            # in-process stop signal only for an *in-scope* locally claimed
            # job, so a fraudulent cross-scope (always-404) request can
            # never interrupt another tenant's runner.
            self._claim_scopes: dict[str, tuple[str, str]] = {}
            # In-process cancel requests, keyed by job id. A cancel
            # endpoint sets the event before it opens its write transaction
            # so a worker holding that lock for the current envelope can
            # observe the request and roll the uncommitted envelope
            # operation back (material, audit and progress together)
            # instead of committing it. Cross-process cancellation cannot
            # signal here; there the writer-lock serialization plus the
            # cleared claim token stops the scan at the next envelope.
            self._cancel_events: dict[str, dict[str, threading.Event]] = {}
            self._cancel_lock = threading.Lock()
            # Job ids whose runner thread is currently inside one
            # envelope's independent transaction in _process_one. A cancel
            # endpoint waits for the runner's acknowledgement only while
            # this is set (the worker may have to roll the in-flight
            # envelope back); when the runner is parked outside an
            # envelope transaction (claim hook, page scan, between
            # envelopes) there is nothing to discard and the cancel
            # proceeds immediately, serializing on the writer lock.
            self._envelope_active: set[str] = set()
            # Optional test/observability hook invoked exactly once after a
            # runner has claimed a job (queued/running -> running committed),
            # outside any database transaction. Production leaves it None.
            self.post_claim = None

        def start(self) -> None:
            """Spawn the worker pool (idempotent)."""
            if self._started.is_set() or self._shutdown.is_set():
                return
            self._started.set()
            for index in range(self._max_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"rewrap-job-{index}",
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)

        def _worker_loop(self) -> None:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                job_id, retry = item
                if self._shutdown.is_set():
                    # Shutdown began while this id was queued: leave the
                    # durable row untouched for the next process's sweep
                    # rather than starting a new run.
                    continue
                self._run_guarded(job_id, retry=retry)

        def shutdown(self) -> None:
            # Let in-flight jobs reach their next independent commit
            # boundary; queued-but-unstarted ids die with the process but
            # their durable queued/running/failed rows are resumed by the
            # next process's startup sweep.
            self._shutdown.set()
            for _ in self._workers:
                self._queue.put(None)
            for worker in self._workers:
                worker.join()
            self._workers = []
            self._started.clear()

        #: Statuses resumed by a fresh process at startup. A ``failed``
        #: job is parked, not dead: its cursor sits before the failing
        #: envelope, and a restart re-attempts it from there. Only a
        #: restart makes that retry — a live runner never does — so the
        #: job remains recoverable until its scope is fully scanned.
        _RECOVERABLE_STATUSES = (
            REWRAP_JOB_STATUS_QUEUED,
            REWRAP_JOB_STATUS_RUNNING,
            REWRAP_JOB_STATUS_FAILED,
        )

        def recover_and_start(self) -> None:
            """Start the pool, clear stale claims, then resume every open job.

            Queued/running rows left by a dead process may carry its claim
            token, which can never be satisfied again, so it is cleared
            before the sweep. Failed jobs parked at their failing envelope
            are enqueued as retries: claim moves failed -> running and the
            job resumes from the parked cursor.
            """
            self.start()
            with self._session_factory() as session:
                session.execute(
                    update(RewrapJob)
                    .where(
                        RewrapJob.status.in_(
                            (REWRAP_JOB_STATUS_QUEUED, REWRAP_JOB_STATUS_RUNNING)
                        )
                    )
                    .values(claim_token=None)
                )
                session.commit()
                open_jobs = session.execute(
                    select(RewrapJob.job_id, RewrapJob.status).where(
                        RewrapJob.status.in_(self._RECOVERABLE_STATUSES)
                    )
                ).all()
            for job_id, status in open_jobs:
                self.submit(job_id, retry=status == REWRAP_JOB_STATUS_FAILED)

        def submit(self, job_id: str, *, retry: bool = False) -> None:
            # Enqueue only while the pool is alive. An app built without
            # its lifespan (tests) enqueues nothing: the durable row waits
            # for an explicit runner call or a later process's startup
            # recovery sweep. ``retry`` belongs to that sweep's
            # failed -> running re-attempt; a live submission never retries
            # a failed job.
            if self._started.is_set() and not self._shutdown.is_set():
                self._queue.put((job_id, retry))

        def _run_guarded(self, job_id: str, *, retry: bool = False) -> None:
            try:
                self.run(job_id, retry=retry)
            except Exception:
                # A runner must never die silently: the exception is
                # logged and the job is left claimed/running; the next
                # process's startup recovery resumes it from the last
                # committed cursor.
                logger.exception("rewrap job %s runner failed", job_id)
                with suppress(Exception):
                    self._release_claim(job_id)

        # -- claim / status transitions ---------------------------------

        def _claim(self, job_id: str, *, retry: bool = False) -> bool:
            """Atomically become the exclusive runner of ``job_id``.

            A live submission claims a queued job or takes over an
            unclaimed running job (in-process recovery). A startup sweep
            may additionally move a parked ``failed`` job back to running
            (``retry=True``) to re-attempt it from the parked cursor.
            Returns ``False`` when the job is succeeded or already
            claimed.
            """
            claimable = (
                self._RECOVERABLE_STATUSES
                if retry
                else (REWRAP_JOB_STATUS_QUEUED, REWRAP_JOB_STATUS_RUNNING)
            )
            token = b64url_encode(secrets.token_bytes(CAPABILITY_BYTES))
            now = _utcnow()
            scope: tuple[str, str] | None = None
            prior_status: str | None = None
            with self._session_factory() as session:
                # Snapshot the actual pre-migration status before the
                # guarded update: a non-retry claim can take over an
                # unclaimed running row (running -> running is no status
                # migration and records nothing), and the recovery sweep's
                # retry claim wins the failed -> running retry.
                existing = session.get(RewrapJob, job_id)
                if existing is not None:
                    scope = (existing.tenant_id, existing.workload_id)
                    prior_status = existing.status
                claimed = session.execute(
                    update(RewrapJob)
                    .where(
                        RewrapJob.job_id == job_id,
                        RewrapJob.status.in_(claimable),
                        RewrapJob.claim_token.is_(None),
                    )
                    .values(
                        status=REWRAP_JOB_STATUS_RUNNING,
                        claim_token=token,
                        updated_at=now,
                    )
                    .execution_options(synchronize_session=False)
                ).rowcount
                if claimed == 1 and scope is not None and prior_status is not None:
                    if prior_status != REWRAP_JOB_STATUS_RUNNING:
                        # queued -> running is execution; failed -> running
                        # (only ever taken by the post-restart retry claim)
                        # is recovery. The event and the claim commit
                        # together, so a migration that loses the guard
                        # leaves no event.
                        reason = (
                            REWRAP_JOB_EVENT_REASON_RECOVERED
                            if prior_status == REWRAP_JOB_STATUS_FAILED
                            else REWRAP_JOB_EVENT_REASON_EXECUTED
                        )
                        _record_rewrap_job_event(
                            session,
                            tenant_id=scope[0],
                            workload_id=scope[1],
                            job_id=job_id,
                            old_status=prior_status,
                            new_status=REWRAP_JOB_STATUS_RUNNING,
                            reason=reason,
                            now=now,
                        )
                session.commit()
            if claimed == 1 and scope is not None:
                # Remember the token so this runner releases only its own
                # claim; the guarded UPDATE above is the real exclusion.
                self._claim_tokens[job_id] = token
                self._claim_scopes[job_id] = scope
                return True
            return False

        def _owns(self, session, job_id: str) -> bool:
            token = self._claim_tokens.get(job_id)
            if token is None:
                return False
            row = session.scalar(
                select(RewrapJob.claim_token).where(RewrapJob.job_id == job_id)
            )
            return row is not None and hmac.compare_digest(row, token)

        def _release_claim(self, job_id: str) -> None:
            token = self._claim_tokens.pop(job_id, None)
            self._claim_scopes.pop(job_id, None)
            if token is None:
                return
            with self._session_factory() as session:
                session.execute(
                    update(RewrapJob)
                    .where(
                        RewrapJob.job_id == job_id,
                        RewrapJob.claim_token == token,
                    )
                    .values(claim_token=None, updated_at=_utcnow())
                    .execution_options(synchronize_session=False)
                )
                session.commit()

        # -- cancellation ------------------------------------------------

        def claim_scope(self, job_id: str) -> tuple[str, str] | None:
            """Return the scope of the job this process currently claims.

            ``None`` when no runner in this process holds the job (it is
            queued/unclaimed, claimed by another process, or terminal).
            Tenant and workload are immutable for a job, so the value is
            authoritative for a same-process scope check.
            """
            return self._claim_scopes.get(job_id)

        def request_cancel(
            self,
            job_id: str,
            scope: tuple[str, str],
            *,
            ack_timeout: float = 5.0,
        ) -> None:
            """Ask an in-process runner of an in-scope ``job_id`` to stop.

            The signal is raised *before* the cancel endpoint opens its
            write transaction, but only when this process currently holds
            the job's claim for the exact requesting scope: a cross-scope
            (always-404) request therefore can never interrupt another
            tenant's runner, and an unknown id or a job claimed by another
            process signals nothing. When matched, a worker currently
            inside the current envelope's transaction observes the signal
            and rolls that uncommitted envelope operation back (material,
            audit and progress together), releasing the writer lock the
            endpoint waits on; the endpoint's guarded UPDATE then settles
            the job. A short ack wait only avoids an unnecessary lock wait
            and never blocks cancellation if it elapses.
            """
            if self._claim_scopes.get(job_id) != scope:
                return
            in_envelope = job_id in self._envelope_active
            with self._cancel_lock:
                signal = self._cancel_events.get(job_id)
                if signal is None:
                    signal = {
                        "cancel": threading.Event(),
                        "ack": threading.Event(),
                    }
                    self._cancel_events[job_id] = signal
            signal["cancel"].set()
            if in_envelope:
                # The runner is inside the current envelope's transaction;
                # wait for it to roll that work back and release the writer
                # lock so the cancel settles without discarding a committed
                # envelope. When the runner is between envelopes there is
                # nothing to wait for and the writer lock orders us.
                signal["ack"].wait(timeout=ack_timeout)

        def finish_cancel(self, job_id: str) -> None:
            """Drop the in-process cancel signal once settlement is decided."""
            with self._cancel_lock:
                self._cancel_events.pop(job_id, None)

        def _cancel_requested(self, job_id: str) -> bool:
            signal = self._cancel_events.get(job_id)
            return signal is not None and signal["cancel"].is_set()

        def _ack_cancel(self, job_id: str) -> None:
            signal = self._cancel_events.get(job_id)
            if signal is not None:
                signal["ack"].set()

        # -- advancement ------------------------------------------------

        def run(self, job_id: str, *, retry: bool = False) -> None:
            if not self._claim(job_id, retry=retry):
                # Succeeded, cancelled, already advanced by another
                # runner, or a failed job presented to a non-retry claim.
                return
            if self.post_claim is not None:
                self.post_claim(job_id)
            try:
                while True:
                    terminal = self._advance_page(job_id)
                    if terminal is RewrapJobRunner._CLAIM_LOST:
                        # The claim moved (a cancel cleared it or another
                        # runner took over): stop immediately rather than
                        # spinning, leaving the durable row exactly as the
                        # winning transaction committed it.
                        return
                    if terminal is not None:
                        # succeeded, failed or cancelled: the terminal
                        # transition was committed inside _advance_page and
                        # the claim cleared there.
                        return
                    if self._shutdown.is_set():
                        # Stop between pages at a fully committed boundary.
                        # Release the in-process claim; the row stays
                        # running and the next process's startup sweep
                        # clears any stale claim before resuming it.
                        self._release_claim(job_id)
                        return
            finally:
                # Release any cancel endpoint waiting on this runner. The
                # uncommitted-envelope path acks earlier (right after it
                # rolls back so the cancel can acquire the lock); every
                # other stop path — claim lost between envelopes, a
                # terminal settlement, shutdown — acks here as the run
                # ends. Harmless when no cancel was requested.
                self._ack_cancel(job_id)
                self._claim_tokens.pop(job_id, None)
                self._claim_scopes.pop(job_id, None)

        def _advance_page(self, job_id: str) -> str | object | None:
            """Process up to ``limit`` envelopes from the current cursor.

            Returns the terminal status when the job settles
            (``succeeded``/``failed``/``cancelled``), ``None`` when a full
            page was handled and the runner should page again, or the
            ``_CLAIM_LOST`` sentinel when this runner no longer owns the
            job (a cancel cleared the token or another runner settled it).
            """
            with self._session_factory() as session:
                if not self._owns(session, job_id):
                    return RewrapJobRunner._CLAIM_LOST
                job = session.get(RewrapJob, job_id)
                if job is None:
                    return RewrapJobRunner._CLAIM_LOST
                if job.status not in (
                    REWRAP_JOB_STATUS_QUEUED,
                    REWRAP_JOB_STATUS_RUNNING,
                ):
                    # A terminal status appeared under us (succeeded,
                    # failed or cancelled). It is never ours to advance.
                    return job.status
                tenant_id = job.tenant_id
                workload_id = job.workload_id
                limit = job.limit
                boundary = (
                    _decode_cursor(job.next_cursor, tenant_id, workload_id)
                    if job.next_cursor
                    else ""
                )
                if boundary is None:
                    # A stored cursor must always verify; if it does not
                    # the row is corrupt and the job cannot safely move.
                    # This cannot occur through the API (stored cursors
                    # are always service-minted); classify it under the
                    # generic existing rewrap failure category.
                    logger.error("rewrap job %s has an unverifiable cursor", job_id)
                    self._settle_failed(
                        session,
                        job,
                        job.next_cursor,
                        REWRAP_JOB_EVENT_REASON_REWRAP_FAILED,
                    )
                    return REWRAP_JOB_STATUS_FAILED
                try:
                    rows = list(
                        session.scalars(
                            select(DataEnvelope)
                            .where(
                                DataEnvelope.tenant_id == tenant_id,
                                DataEnvelope.workload_id == workload_id,
                                DataEnvelope.data_id > boundary,
                            )
                            .order_by(DataEnvelope.data_id.asc())
                            .limit(limit + 1)
                        )
                    )
                except Exception:
                    session.rollback()
                    logger.error("rewrap job %s scan failed", job_id)
                    raise

            has_more = len(rows) > limit
            # Snapshot the values the worker needs before this read
            # session closes; later writes use independent sessions.
            page = [
                (
                    row.tenant_id,
                    row.workload_id,
                    row.data_id,
                    row.key_version,
                    row.wrapped_key,
                )
                for row in rows[:limit]
            ]

            for envelope in page:
                processed = self._process_one(job_id, envelope)
                if processed is None:
                    # Committed at an independent boundary: shutdown can
                    # take effect here and resume from next_cursor later.
                    if self._shutdown.is_set():
                        return None
                    continue
                if processed is RewrapJobRunner._CANCELLED_BEFORE_COMMIT:
                    # The current envelope's uncommitted work was
                    # discarded at a cancel's request; the cancel
                    # transaction settles the job. Stop without touching
                    # any later envelope.
                    return RewrapJobRunner._CLAIM_LOST
                if processed is RewrapJobRunner._CLAIM_LOST:
                    # A cancel cleared the claim (or another runner
                    # settled the job): the scan stops immediately after
                    # the last independently committed envelope, with no
                    # further envelope touched.
                    return RewrapJobRunner._CLAIM_LOST
                # This runner settled the job as failed.
                return REWRAP_JOB_STATUS_FAILED
            # Whole page reached without a failure.
            with self._session_factory() as session:
                if not self._owns(session, job_id):
                    return RewrapJobRunner._CLAIM_LOST
                job = session.get(RewrapJob, job_id)
                if job is None or job.status not in (
                    REWRAP_JOB_STATUS_QUEUED,
                    REWRAP_JOB_STATUS_RUNNING,
                ):
                    # A cancel or another settlement landed between the
                    # last envelope commit and this point; leave it.
                    return (
                        job.status
                        if job is not None
                        else RewrapJobRunner._CLAIM_LOST
                    )
                now = _utcnow()
                if not has_more:
                    # The limit+1 probe found nothing beyond this page, so
                    # the scope has been fully scanned: settle exactly once.
                    settled = session.execute(
                        update(RewrapJob)
                        .where(
                            RewrapJob.job_id == job_id,
                            RewrapJob.status == REWRAP_JOB_STATUS_RUNNING,
                            RewrapJob.claim_token == self._claim_tokens.get(job_id),
                        )
                        .values(
                            status=REWRAP_JOB_STATUS_SUCCEEDED,
                            complete=True,
                            next_cursor="",
                            claim_token=None,
                            updated_at=now,
                        )
                        .execution_options(synchronize_session=False)
                    ).rowcount
                    if settled != 1:
                        session.rollback()
                        # A concurrent cancel (or another settlement) won
                        # the row first: observe it and stop rather than
                        # forcing a second terminal state.
                        fresh = session.get(RewrapJob, job_id)
                        if fresh is not None and fresh.status in (
                            REWRAP_JOB_STATUS_CANCELLED,
                            REWRAP_JOB_STATUS_SUCCEEDED,
                            REWRAP_JOB_STATUS_FAILED,
                        ):
                            return fresh.status
                        raise RuntimeError(
                            "rewrap job success settlement did not settle"
                        )
                    # The completion event commits in the same transaction
                    # as the guarded running -> succeeded settlement.
                    _record_rewrap_job_event(
                        session,
                        tenant_id=tenant_id,
                        workload_id=workload_id,
                        job_id=job_id,
                        old_status=REWRAP_JOB_STATUS_RUNNING,
                        new_status=REWRAP_JOB_STATUS_SUCCEEDED,
                        reason=REWRAP_JOB_EVENT_REASON_COMPLETED,
                        now=now,
                    )
                    session.commit()
                    return REWRAP_JOB_STATUS_SUCCEEDED
                # Full page and more may follow: just bump updated_at, and
                # only while the job is still ours and open. A guarded
                # update keeps a concurrent cancel from being overwritten.
                bumped = session.execute(
                    update(RewrapJob)
                    .where(
                        RewrapJob.job_id == job_id,
                        RewrapJob.status.in_(
                            (REWRAP_JOB_STATUS_QUEUED, REWRAP_JOB_STATUS_RUNNING)
                        ),
                        RewrapJob.claim_token == self._claim_tokens.get(job_id),
                    )
                    .values(updated_at=now)
                    .execution_options(synchronize_session=False)
                ).rowcount
                if bumped != 1:
                    session.rollback()
                    fresh = session.get(RewrapJob, job_id)
                    if fresh is not None and fresh.status in (
                        REWRAP_JOB_STATUS_CANCELLED,
                        REWRAP_JOB_STATUS_SUCCEEDED,
                        REWRAP_JOB_STATUS_FAILED,
                    ):
                        return fresh.status
                    return RewrapJobRunner._CLAIM_LOST
                session.commit()
            return None

        def _process_one(self, job_id: str, envelope) -> str | object | None:
            """Independently commit one envelope.

            Returns ``None`` when the envelope committed, a result code
            string when this runner settled the job failed, or one of the
            ``_CLAIM_LOST``/``_CANCELLED_BEFORE_COMMIT`` sentinels when the
            envelope's uncommitted work was discarded and the scan must
            stop without touching a later envelope.

            ``envelope`` is the detached snapshot tuple
            ``(tenant_id, workload_id, data_id, key_version, wrapped_key)``
            produced by the page scan.
            """
            tenant_id, workload_id, data_id, stored_version, wrapped_key = envelope

            # Mark the envelope in flight so a cancel endpoint knows
            # whether to wait for this runner to roll back its current
            # uncommitted envelope. Cleared as soon as the envelope
            # commits or is discarded.
            self._envelope_active.add(job_id)
            try:
                with self._session_factory() as session:
                    if not self._owns(session, job_id):
                        return RewrapJobRunner._CLAIM_LOST
                    job = session.get(RewrapJob, job_id)
                    if job is None or job.status not in (
                        REWRAP_JOB_STATUS_QUEUED,
                        REWRAP_JOB_STATUS_RUNNING,
                    ):
                        return RewrapJobRunner._CLAIM_LOST
                    if self._cancel_requested(job_id):
                        # A cancel arrived before this envelope did any
                        # material work. Do none of it: release the writer
                        # lock the cancel endpoint is waiting on and let
                        # that transaction settle the job at the prior
                        # cursor.
                        session.rollback()
                        self._ack_cancel(job_id)
                        return RewrapJobRunner._CANCELLED_BEFORE_COMMIT

                    # Re-resolve the keyring per envelope: a keyring that
                    # becomes unusable mid-job fails the job on the envelope
                    # that could not be handled.
                    try:
                        keyring = load_keyring()
                    except MasterKeyError:
                        if self._cancel_requested(job_id):
                            # A cancel is waiting on this writer lock:
                            # prefer it over a failure settlement and
                            # discard the uncommitted envelope attempt.
                            session.rollback()
                            self._ack_cancel(job_id)
                            return RewrapJobRunner._CANCELLED_BEFORE_COMMIT
                        logger.error(
                            "master keyring unavailable during rewrap job"
                        )
                        self._settle_failed(
                            session,
                            job,
                            job.next_cursor,
                            REWRAP_JOB_EVENT_REASON_KEYRING,
                        )
                        return REWRAP_RESULT_KEYRING

                    current_version = keyring.current_version
                    audit_old_version = stored_version
                    if stored_version == current_version:
                        result = REWRAP_RESULT_SKIPPED
                        new_version = stored_version
                    else:
                        try:
                            unwrapping_key = keyring.key_for(stored_version)
                            new_wrapped_key = rewrap_data_key(
                                unwrapping_key,
                                keyring.current_key(),
                                wrapped_key,
                            )
                        except MasterKeyError:
                            if self._cancel_requested(job_id):
                                session.rollback()
                                self._ack_cancel(job_id)
                                return RewrapJobRunner._CANCELLED_BEFORE_COMMIT
                            logger.error(
                                "master key version %s unavailable during "
                                "rewrap job",
                                stored_version,
                            )
                            self._settle_failed(
                                session,
                                job,
                                job.next_cursor,
                                REWRAP_JOB_EVENT_REASON_MISSING_KEY,
                            )
                            return REWRAP_RESULT_MISSING_KEY
                        except Exception:
                            if self._cancel_requested(job_id):
                                session.rollback()
                                self._ack_cancel(job_id)
                                return RewrapJobRunner._CANCELLED_BEFORE_COMMIT
                            logger.error(
                                "rewrap job single-envelope rewrap failed"
                            )
                            self._settle_failed(
                                session,
                                job,
                                job.next_cursor,
                                REWRAP_JOB_EVENT_REASON_REWRAP_FAILED,
                            )
                            return REWRAP_RESULT_REWRAP_FAILED

                        # Guarded update, mirroring the batch and
                        # single-envelope paths: only a row still at the
                        # version we unwrapped can rotate. A concurrent
                        # winner leaves current-version material, recorded
                        # as a skip.
                        outcome = session.execute(
                            update(DataEnvelope)
                            .where(
                                DataEnvelope.tenant_id == tenant_id,
                                DataEnvelope.workload_id == workload_id,
                                DataEnvelope.data_id == data_id,
                                DataEnvelope.key_version == stored_version,
                            )
                            .values(
                                key_version=current_version,
                                wrapped_key=new_wrapped_key,
                            )
                            .execution_options(synchronize_session=False)
                        )
                        if outcome.rowcount != 1:
                            session.rollback()
                            fresh = session.scalar(
                                select(DataEnvelope)
                                .where(
                                    DataEnvelope.tenant_id == tenant_id,
                                    DataEnvelope.workload_id == workload_id,
                                    DataEnvelope.data_id == data_id,
                                )
                                .execution_options(populate_existing=True)
                            )
                            if (
                                fresh is not None
                                and fresh.key_version == current_version
                            ):
                                result = REWRAP_RESULT_SKIPPED
                                new_version = current_version
                                audit_old_version = current_version
                                job = session.get(RewrapJob, job_id)
                            else:
                                job = session.get(RewrapJob, job_id)
                                self._settle_failed(
                                    session,
                                    job,
                                    job.next_cursor,
                                    REWRAP_JOB_EVENT_REASON_REWRAP_FAILED,
                                )
                                return REWRAP_RESULT_REWRAP_FAILED
                        else:
                            result = REWRAP_RESULT_REWRAPPED
                            new_version = current_version

                    # Independent per-envelope commit: the material change,
                    # the existing rewrap audit event and the job's
                    # progress land together and durably before the next
                    # envelope.
                    item_occurred_at = _utcnow()
                    event_status = (
                        AUDIT_EVENT_STATUS_REWRAPPED
                        if result == REWRAP_RESULT_REWRAPPED
                        else AUDIT_EVENT_STATUS_SKIPPED
                    )
                    _append_audit_event(
                        session,
                        tenant_id=tenant_id,
                        workload_id=workload_id,
                        event_type=AUDIT_EVENT_TYPE_REWRAP,
                        grant_id=None,
                        decision_id=None,
                        data_id=data_id,
                        status=event_status,
                        capability_sha256=None,
                        occurred_at=item_occurred_at,
                    )
                    # The per-job item row commits in the same transaction
                    # as the material change, the audit event and the
                    # progress update below: a committed envelope always
                    # has exactly one item, and an envelope operation that
                    # rolls back (cancel, lost claim, commit failure)
                    # leaves none. Only rewrapped/skipped outcomes reach
                    # this point; a failed envelope settles the job above
                    # without committing an item.
                    _record_rewrap_job_item(
                        session,
                        tenant_id=tenant_id,
                        workload_id=workload_id,
                        job_id=job_id,
                        data_id=data_id,
                        old_key_version=audit_old_version,
                        new_key_version=new_version,
                        result=result,
                        now=item_occurred_at,
                    )
                    next_boundary = (
                        _encode_cursor(tenant_id, workload_id, data_id)
                        if data_id
                        else ""
                    )
                    if self._cancel_requested(job_id):
                        # A cancel landed while this envelope was being
                        # processed but before its commit. The material
                        # update, the audit event and the job's counters are
                        # all still uncommitted in this one transaction:
                        # roll them back together so the envelope, its
                        # wrapping material and the audit stay exactly as
                        # they were, release the writer lock the cancel is
                        # waiting on, and stop.
                        session.rollback()
                        self._ack_cancel(job_id)
                        return RewrapJobRunner._CANCELLED_BEFORE_COMMIT
                    # Advance the job progress with a guarded UPDATE
                    # rather than an ORM primary-key write: a cancel that
                    # wins the writer lock first moves the row to cancelled
                    # (and clears the claim token), and this guard then
                    # matches zero rows. On a miss the whole envelope
                    # transaction — material, audit and progress — is rolled
                    # back together, so a cancel marker can never be
                    # overwritten back to running and an envelope never
                    # lands without its progress.
                    token = self._claim_tokens.get(job_id)
                    advanced = session.execute(
                        update(RewrapJob)
                        .where(
                            RewrapJob.job_id == job_id,
                            RewrapJob.status.in_(
                                (
                                    REWRAP_JOB_STATUS_QUEUED,
                                    REWRAP_JOB_STATUS_RUNNING,
                                )
                            ),
                            RewrapJob.claim_token == token,
                        )
                        .values(
                            processed=RewrapJob.processed + 1,
                            rewrapped=(
                                RewrapJob.rewrapped + 1
                                if result == REWRAP_RESULT_REWRAPPED
                                else RewrapJob.rewrapped
                            ),
                            skipped=(
                                RewrapJob.skipped
                                if result == REWRAP_RESULT_REWRAPPED
                                else RewrapJob.skipped + 1
                            ),
                            next_cursor=next_boundary,
                            updated_at=item_occurred_at,
                        )
                        .execution_options(synchronize_session=False)
                    ).rowcount
                    if advanced != 1:
                        # A cancel (or another settlement) committed first.
                        # Discard the uncommitted envelope operation
                        # wholesale.
                        session.rollback()
                        self._ack_cancel(job_id)
                        return RewrapJobRunner._CLAIM_LOST
                    try:
                        session.commit()
                    except Exception:
                        session.rollback()
                        logger.error("rewrap job progress commit failed")
                        raise
                    return None
            finally:
                self._envelope_active.discard(job_id)

        def _settle_failed(self, session, job, resume_cursor: str, reason: str) -> None:
            """Move a running job to ``failed`` and park the resume cursor.

            The failed envelope itself is never modified and its failure
            is not counted as processed: ``failed`` increases by one and
            ``next_cursor`` stays immediately before it, so recovery
            re-attempts exactly that envelope first. ``reason`` is the
            fixed failure classification (keyring, missing historical
            key, or a rewrap that failed) — never exception text — and is
            recorded as the running -> failed event in the same
            transaction as the settlement.
            """
            now = _utcnow()
            settled = session.execute(
                update(RewrapJob)
                .where(
                    RewrapJob.job_id == job.job_id,
                    RewrapJob.status == REWRAP_JOB_STATUS_RUNNING,
                )
                .values(
                    status=REWRAP_JOB_STATUS_FAILED,
                    failed=RewrapJob.failed + 1,
                    next_cursor=resume_cursor,
                    complete=False,
                    claim_token=None,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            ).rowcount
            if settled != 1:
                session.rollback()
                # A concurrent terminal transition (e.g. a cancel from
                # another process) won the row first; keep that terminal
                # state instead of forcing a second one.
                fresh = session.get(RewrapJob, job.job_id)
                if fresh is not None and fresh.status in (
                    REWRAP_JOB_STATUS_CANCELLED,
                    REWRAP_JOB_STATUS_SUCCEEDED,
                    REWRAP_JOB_STATUS_FAILED,
                ):
                    return
                raise RuntimeError("rewrap job failure settlement did not settle")
            _record_rewrap_job_event(
                session,
                tenant_id=job.tenant_id,
                workload_id=job.workload_id,
                job_id=job.job_id,
                old_status=REWRAP_JOB_STATUS_RUNNING,
                new_status=REWRAP_JOB_STATUS_FAILED,
                reason=reason,
                now=now,
            )
            session.commit()

    # Per-app runner state is allocated when the app is built, but the
    # background threads start only with the application lifespan (and a
    # recovery sweep runs first, clearing claims left by dead processes).
    runner = RewrapJobRunner(session_factory)

    @asynccontextmanager
    async def _rewrap_jobs_lifespan(app: FastAPI):
        # Tests that build the app without entering the lifespan never run
        # the sweep or the pool; production (uvicorn) always enters it.
        runner.recover_and_start()
        try:
            yield
        finally:
            runner.shutdown()

    app = FastAPI(title="Remote Attestation Data Release", lifespan=_rewrap_jobs_lifespan)
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.verifier_registry = registry
    app.state.rewrap_job_runner = runner

    @app.exception_handler(RequestValidationError)
    async def _request_validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Identical to the default handler, except the echoed error input
        # is first made renderable when the rejected body carried a
        # non-finite float literal.
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(_json_safe(exc.errors()))},
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/health/readiness")
    def health_readiness(
        request: Request,
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> JSONResponse:
        """Report whether the service can currently serve its dependencies.

        A standalone readiness probe for callers deciding whether proof,
        authorization and data-release traffic can be carried. The request
        takes no input: any query parameter is a 422 and any non-empty body
        (including whitespace or non-JSON) is a 422, both rejected before
        any dependency is touched and without changing any state.

        Each call independently re-checks both dependencies. The storage
        check is a single minimal read-only statement confirming a
        connection can be obtained and a session can execute SQL. The
        keyring check loads the currently configured keyring with the
        existing configuration semantics; it never unwraps an envelope,
        rotates a key, writes an audit row or consumes rate-limit budget.
        Both checks run on every call, so one failing dependency never
        suppresses the other's result. When both succeed the response is
        200 with status ``ready``; otherwise it is 503 with status
        ``not_ready`` and each dependency reported as ``ok`` or
        ``unavailable``. The checks are read-only and side-effect free:
        they leave no partial result, temporary table, audit event or
        cached state, so the entry point flips back to 200 as soon as the
        dependencies recover. Failure detail (storage exceptions,
        environment variables, key versions, key bytes, key fingerprints
        and raw configuration) never appears in the response or the logs.
        """
        # --- request shape (all 422, no dependency is touched) ----------
        if request.query_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )

        checks: dict[str, str] = {}

        try:
            with session_factory() as session:
                session.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception:
            # The exception may carry connection or environment detail;
            # only a fixed, detail-free message is ever logged.
            logger.warning("readiness database check failed")
            checks["database"] = "unavailable"

        try:
            load_keyring()
            checks["keyring"] = "ok"
        except Exception:
            # Key material, versions and raw configuration stay out of the
            # logs; only the failure kind is recorded.
            logger.warning("readiness keyring check failed")
            checks["keyring"] = "unavailable"

        ready = all(status == "ok" for status in checks.values())
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "not_ready",
                "checks": checks,
            },
        )

    @app.post("/v1/challenges", status_code=201, response_model=ChallengeCreatedResponse)
    def create_challenge(body: CreateChallengeRequest) -> ChallengeCreatedResponse:
        # Reaching the handler means every request field (tenant_id,
        # workload_id and the ttl_seconds bounds) already passed the same
        # validation as before; malformed requests keep their 422 and never
        # execute this body, so they never touch a counter.
        now = _utcnow()
        window_start = _utc_minute_window(now)
        with session_factory() as session:
            try:
                admitted = False
                for _ in range(2):
                    # Lock the scope's minute row when one exists. SQLite
                    # ignores FOR UPDATE but every write transaction already
                    # begins as BEGIN IMMEDIATE, serializing concurrent
                    # admissions process-wide; on locking backends the row
                    # lock orders them so exactly the budgeted number of
                    # valid requests in the minute can be admitted.
                    row = session.scalar(
                        select(ChallengeIssuanceCounter)
                        .where(
                            ChallengeIssuanceCounter.tenant_id == body.tenant_id,
                            ChallengeIssuanceCounter.workload_id
                            == body.workload_id,
                            ChallengeIssuanceCounter.window_start
                            == window_start,
                        )
                        .with_for_update()
                    )
                    if row is None:
                        # The first valid request of the minute initializes
                        # the counter at one. A concurrent initializer on a
                        # locking backend may win the unique constraint;
                        # that race is retried as an increment below.
                        session.add(
                            ChallengeIssuanceCounter(
                                tenant_id=body.tenant_id,
                                workload_id=body.workload_id,
                                window_start=window_start,
                                count=1,
                            )
                        )
                        try:
                            session.flush()
                        except IntegrityError:
                            session.rollback()
                            continue
                        admitted = True
                        break
                    if row.count >= CHALLENGE_ISSUANCE_BUDGET_PER_MINUTE:
                        # Budget exhausted: no counter write, no challenge,
                        # no audit and no nonce ever minted. The retry hint
                        # is recomputed at response time.
                        session.rollback()
                        return _too_many_requests_response(_utcnow())
                    row.count = row.count + 1
                    admitted = True
                    break
                if not admitted:
                    # Defensive: the unique-insert retry loop failed to
                    # settle, which the single retry above makes
                    # unreachable.
                    session.rollback()
                    logger.error(
                        "challenge issuance rate-limit reservation could not settle"
                    )
                    raise HTTPException(
                        status_code=500, detail="rate limit unavailable"
                    )

                # The slot is reserved inside this still-open transaction.
                # Mint the nonce only now, so an over-budget or failed
                # request never generates, returns or persists one, and
                # insert the challenge in the same commit: a crash or
                # failure can leave neither a counter without its challenge
                # nor a challenge without its reservation.
                nonce_bytes = secrets.token_bytes(NONCE_BYTES)
                nonce = base64.urlsafe_b64encode(nonce_bytes).rstrip(
                    b"="
                ).decode("ascii")
                expires_at = now + timedelta(seconds=body.ttl_seconds)
                challenge = Challenge(
                    challenge_id=str(uuid.uuid4()),
                    tenant_id=body.tenant_id,
                    workload_id=body.workload_id,
                    nonce_digest=_nonce_digest(nonce),
                    status="pending",
                    issued_at=now,
                    expires_at=expires_at,
                    consumed_at=None,
                )
                session.add(challenge)
                session.commit()
            except HTTPException:
                raise
            except Exception:
                # A counter read/write that cannot complete fails closed:
                # the whole reservation transaction rolls back, leaving no
                # half challenge and no half count.
                session.rollback()
                logger.error(
                    "challenge issuance rate-limit counter unavailable"
                )
                raise HTTPException(status_code=500, detail="rate limit unavailable")
        return ChallengeCreatedResponse(
            challenge_id=challenge.challenge_id,
            nonce=nonce,
            issued_at=_rfc3339(now),
            expires_at=_rfc3339(expires_at),
            status="pending",
        )

    @app.post(
        "/v1/challenges/{challenge_id}/consume",
        response_model=ChallengeConsumedResponse,
    )
    def consume_challenge(
        challenge_id: str, body: ConsumeChallengeRequest
    ) -> ChallengeConsumedResponse:
        digest = _nonce_digest(body.nonce)
        now = _utcnow()
        with session_factory() as session:
            challenge = session.get(Challenge, challenge_id)
            if (
                challenge is None
                or challenge.tenant_id != body.tenant_id
                or challenge.workload_id != body.workload_id
            ):
                raise HTTPException(status_code=404, detail="challenge not found")
            if not hmac.compare_digest(challenge.nonce_digest, digest):
                raise HTTPException(status_code=401, detail="invalid nonce")
            if challenge.status == "consumed":
                raise HTTPException(status_code=409, detail="challenge already consumed")
            if challenge.expires_at <= now:
                raise HTTPException(status_code=410, detail="challenge expired")
            # Atomic claim: only one concurrent consumer can flip pending -> consumed.
            result = session.execute(
                update(Challenge)
                .where(
                    Challenge.challenge_id == challenge_id,
                    Challenge.status == "pending",
                    Challenge.expires_at > now,
                )
                .values(status="consumed", consumed_at=now)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                session.rollback()
                fresh = session.get(Challenge, challenge_id)
                if fresh is not None and fresh.status == "consumed":
                    raise HTTPException(
                        status_code=409, detail="challenge already consumed"
                    )
                raise HTTPException(status_code=410, detail="challenge expired")
            session.commit()
        return ChallengeConsumedResponse(
            challenge_id=challenge_id,
            status="consumed",
            consumed_at=_rfc3339(now),
        )

    @app.get("/v1/challenges/{challenge_id}")
    def get_challenge_status(
        request: Request,
        challenge_id: str,
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return the read-only current phase of one random challenge.

        The phase is derived from committed state and the query instant:
        a consumed-less challenge is ``pending`` before ``expires_at`` and
        ``expired`` at/after it; a consumed challenge with no evidence is
        ``consumed``; an associated evidence still ``received`` is
        ``evidence_received``; its first verification settles the phase to
        ``verified`` or ``rejected``. Every shape defect — a path
        identifier that is not a canonical lowercase UUID, a missing,
        blank or repeated ``tenant_id``/``workload_id``, or any other
        query parameter — is one indistinguishable 422 raised before
        storage is touched. An unknown challenge and one outside the two
        scope parameters are one indistinguishable 404.

        The handler issues only SELECTs: it performs no expiry write,
        appends no event, audit or idempotency record and consumes no
        issuance, verification or release budget. The response carries no
        nonce, nonce digest, evidence or evidence digest, plugin text or
        private material. A storage read failure is a sanitized 500 with
        no partial state.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {"tenant_id", "workload_id"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="invalid challenge query"
            )
        # Each scope parameter is a single scalar string; a repeated
        # parameter (a multi-valued/list value) is the wrong shape, not a
        # silently last-wins scalar.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(status_code=422, detail="invalid challenge query")
        tenant_id = request.query_params.get("tenant_id")
        workload_id = request.query_params.get("workload_id")
        if (
            not _UUID_RE.fullmatch(challenge_id)
            or tenant_id is None
            or workload_id is None
            or not tenant_id.strip()
            or not workload_id.strip()
        ):
            raise HTTPException(status_code=422, detail="invalid challenge query")

        # One read-only transaction over the challenge and its at-most-one
        # evidence row against one committed snapshot: a concurrent
        # consume/submit/verify that commits first is the phase this query
        # reports, and repeating the query against the same storage state
        # returns the identical result. Nothing is written or locked for
        # update.
        now = _utcnow()
        try:
            with session_factory() as session:
                challenge = session.get(Challenge, challenge_id)
                if (
                    challenge is None
                    or challenge.tenant_id != tenant_id
                    or challenge.workload_id != workload_id
                ):
                    # Unknown and cross-scope challenges are indistinguishable.
                    raise HTTPException(status_code=404, detail="challenge not found")
                evidence = session.scalar(
                    select(Evidence).where(
                        Evidence.challenge_id == challenge_id
                    )
                )
                if evidence is None:
                    if challenge.status == "consumed":
                        status = CHALLENGE_PHASE_CONSUMED
                        changed_at = challenge.consumed_at
                    elif challenge.expires_at <= now:
                        # Expiry is derived, never persisted: no write occurs.
                        status = CHALLENGE_PHASE_EXPIRED
                        changed_at = challenge.expires_at
                    else:
                        status = CHALLENGE_PHASE_PENDING
                        changed_at = challenge.issued_at
                    evidence_id = None
                    evidence_format = None
                    verification_result = None
                else:
                    evidence_id = evidence.evidence_id
                    evidence_format = evidence.evidence_format
                    if evidence.status == "received":
                        status = CHALLENGE_PHASE_EVIDENCE_RECEIVED
                        changed_at = evidence.received_at
                        verification_result = None
                    else:
                        # The first verification settles the evidence
                        # atomically to exactly one terminal status.
                        status = (
                            CHALLENGE_PHASE_VERIFIED
                            if evidence.status == "verified"
                            else CHALLENGE_PHASE_REJECTED
                        )
                        changed_at = evidence.verified_at
                        verification_result = evidence.verification_result
                payload = {
                    "challenge_id": challenge.challenge_id,
                    "issued_at": _rfc3339(challenge.issued_at),
                    "expires_at": _rfc3339(challenge.expires_at),
                    "status": status,
                    "changed_at": _rfc3339(changed_at),
                    "evidence_id": evidence_id,
                    "evidence_format": evidence_format,
                    "verification_result": verification_result,
                }
        except HTTPException:
            raise
        except Exception:
            logger.error("challenge status query failed")
            raise HTTPException(
                status_code=500, detail="challenge status unavailable"
            )

        return _compact_json(payload)

    @app.post("/v1/evidence", status_code=201, response_model=EvidenceReceivedResponse)
    def submit_evidence(body: SubmitEvidenceRequest) -> EvidenceReceivedResponse:
        digest = _nonce_digest(body.nonce)
        evidence_digest = hashlib.sha256(body.evidence.encode("utf-8")).hexdigest()
        now = _utcnow()
        evidence_id = str(uuid.uuid4())
        with session_factory() as session:
            challenge = session.get(Challenge, body.challenge_id)
            if (
                challenge is None
                or challenge.tenant_id != body.tenant_id
                or challenge.workload_id != body.workload_id
            ):
                raise HTTPException(status_code=404, detail="challenge not found")
            if not hmac.compare_digest(challenge.nonce_digest, digest):
                raise HTTPException(status_code=401, detail="invalid nonce")
            if challenge.status == "consumed":
                raise HTTPException(status_code=409, detail="challenge already consumed")
            if challenge.expires_at <= now:
                raise HTTPException(status_code=410, detail="challenge expired")
            # Atomic claim: only one concurrent submission can flip pending -> consumed.
            result = session.execute(
                update(Challenge)
                .where(
                    Challenge.challenge_id == body.challenge_id,
                    Challenge.status == "pending",
                    Challenge.expires_at > now,
                )
                .values(status="consumed", consumed_at=now)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                session.rollback()
                fresh = session.get(Challenge, body.challenge_id)
                if fresh is not None and fresh.status == "consumed":
                    raise HTTPException(
                        status_code=409, detail="challenge already consumed"
                    )
                raise HTTPException(status_code=410, detail="challenge expired")
            # The evidence itself is never persisted, only its SHA-256 digest.
            session.add(
                Evidence(
                    evidence_id=evidence_id,
                    challenge_id=body.challenge_id,
                    tenant_id=body.tenant_id,
                    workload_id=body.workload_id,
                    evidence_format=body.evidence_format,
                    status="received",
                    received_at=now,
                    evidence_sha256=evidence_digest,
                )
            )
            # The proof-lifecycle reception event commits in the same
            # transaction as the evidence row, so it exists if and only if
            # the reception did. Only identifiers, the fixed received
            # status, the (non-sensitive) format descriptor and a
            # timestamp are recorded — never the evidence, nonce or claims.
            # Its per-scope commit sequence fixes the event's snapshot
            # position to this transaction's commit boundary.
            session.add(
                ProofLifecycleEvent(
                    event_id=str(uuid.uuid4()),
                    tenant_id=body.tenant_id,
                    workload_id=body.workload_id,
                    event_type=PROOF_EVENT_TYPE_RECEIVED,
                    evidence_id=evidence_id,
                    commit_seq=_next_proof_event_commit_seq(
                        session, body.tenant_id, body.workload_id
                    ),
                    policy_version=None,
                    evidence_format=body.evidence_format,
                    status=PROOF_EVENT_STATUS_RECEIVED,
                    occurred_at=now,
                )
            )
            session.commit()
        return EvidenceReceivedResponse(
            evidence_id=evidence_id,
            challenge_id=body.challenge_id,
            status="received",
            received_at=_rfc3339(now),
        )

    @app.post(
        "/v1/evidence/{evidence_id}/verify",
        response_model=EvidenceVerifiedResponse,
    )
    def verify_evidence(
        evidence_id: str, body: VerifyEvidenceRequest
    ) -> EvidenceVerifiedResponse:
        evidence_digest = hashlib.sha256(body.evidence.encode("utf-8")).hexdigest()
        nonce_digest = _nonce_digest(body.nonce)
        with session_factory() as session:
            # Lock the evidence row for the duration of verification. On
            # locking backends a concurrent verifier blocks here until the
            # first transaction settles, then observes the terminal status
            # and never invokes the plugin. SQLite ignores FOR UPDATE but its
            # transactions already begin as BEGIN IMMEDIATE, serializing all
            # writers process- and connection-wide.
            evidence = session.scalar(
                select(Evidence)
                .where(Evidence.evidence_id == evidence_id)
                .with_for_update()
            )
            if (
                evidence is None
                or evidence.tenant_id != body.tenant_id
                or evidence.workload_id != body.workload_id
            ):
                raise HTTPException(status_code=404, detail="evidence not found")

            challenge = session.get(Challenge, evidence.challenge_id)
            # The evidence's challenge is always same-tenant/workload; guard
            # defensively and treat any inconsistency as not-found.
            if (
                challenge is None
                or challenge.tenant_id != body.tenant_id
                or challenge.workload_id != body.workload_id
            ):
                raise HTTPException(status_code=404, detail="evidence not found")

            if not hmac.compare_digest(challenge.nonce_digest, nonce_digest):
                raise HTTPException(status_code=401, detail="invalid nonce")

            # Already settled: subsequent calls return the stored conclusion
            # verbatim; the plugin is never invoked again and the freshly
            # supplied evidence is not inspected.
            if evidence.status in ("verified", "rejected"):
                return EvidenceVerifiedResponse(
                    evidence_id=evidence.evidence_id,
                    challenge_id=evidence.challenge_id,
                    status=evidence.status,
                    verified_at=_rfc3339(evidence.verified_at),
                )

            if not hmac.compare_digest(evidence.evidence_sha256, evidence_digest):
                raise HTTPException(
                    status_code=422, detail="evidence digest mismatch"
                )

            verifier = registry.get(evidence.evidence_format)
            if verifier is None:
                raise HTTPException(
                    status_code=422, detail="unsupported evidence format"
                )

            # --- per-scope verification admission budget -----------------
            # Every pre-verifier judgement has now passed: the evidence
            # exists in this scope, the challenge is bound to it, the nonce
            # and the evidence digest match, the format is registered and
            # the evidence is still received. Reserve exactly one slot of
            # this scope's per-UTC-minute verification budget before the
            # verifier (and the X.509 revocation registries) run. The
            # reservation lives in this transaction — which already holds
            # the evidence row lock, so concurrent verifications of the
            # same evidence can never multiply-reserve — and commits
            # atomically with the settlement on the success path; every
            # later failure keeps the consumed slot (see
            # _keep_verification_slot). An over-budget request writes
            # nothing: no counter, no settlement, no lifecycle event, no
            # audit row.
            window_start = _utc_minute_window(_utcnow())
            try:
                admitted = False
                for _ in range(2):
                    # Lock the scope's minute row when one exists. SQLite
                    # ignores FOR UPDATE but every write transaction already
                    # begins as BEGIN IMMEDIATE, serializing concurrent
                    # admissions process-wide; on locking backends the row
                    # lock orders them so exactly the budgeted number of
                    # requests in the minute can be admitted.
                    counter = session.scalar(
                        select(VerificationAdmissionCounter)
                        .where(
                            VerificationAdmissionCounter.tenant_id
                            == body.tenant_id,
                            VerificationAdmissionCounter.workload_id
                            == body.workload_id,
                            VerificationAdmissionCounter.window_start
                            == window_start,
                        )
                        .with_for_update()
                    )
                    if counter is None:
                        # The first admitted request of the minute
                        # initializes the counter at one. The insert runs in
                        # a savepoint so a concurrent first insert on a
                        # locking backend costs only the savepoint — never
                        # the evidence row lock this transaction holds.
                        try:
                            with session.begin_nested():
                                session.add(
                                    VerificationAdmissionCounter(
                                        tenant_id=body.tenant_id,
                                        workload_id=body.workload_id,
                                        window_start=window_start,
                                        count=1,
                                    )
                                )
                        except IntegrityError:
                            continue
                        admitted = True
                        break
                    if counter.count >= VERIFICATION_BUDGET_PER_MINUTE:
                        # Budget exhausted: no counter write, no settlement,
                        # no lifecycle event and no audit row. The retry
                        # hint is recomputed at response time.
                        session.rollback()
                        return _verification_rate_limited_response()
                    counter.count = counter.count + 1
                    admitted = True
                    break
                if not admitted:
                    # Defensive: the unique-insert retry loop failed to
                    # settle, which the single retry above makes
                    # unreachable.
                    session.rollback()
                    logger.error(
                        "verification rate-limit reservation could not settle"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="verification rate limit unavailable",
                    )
            except HTTPException:
                raise
            except Exception:
                # A counter read/write that cannot complete fails closed:
                # the whole transaction rolls back, leaving the evidence
                # received and no count written, so the identical request
                # can be retried once the counter recovers.
                session.rollback()
                logger.error("verification rate-limit counter unavailable")
                raise HTTPException(
                    status_code=500,
                    detail="verification rate limit unavailable",
                )

            challenge_context = ChallengeContext(
                challenge_id=challenge.challenge_id,
                nonce_digest=challenge.nonce_digest,
                status=challenge.status,
                issued_at=challenge.issued_at,
                expires_at=challenge.expires_at,
                consumed_at=challenge.consumed_at,
            )
            # Trust roots configured for exactly this tenant and workload
            # (public certificate material only) are made available to
            # verifiers that anchor evidence to them.
            trust_roots = tuple(
                session.scalars(
                    select(TrustRoot.root_pem).where(
                        TrustRoot.tenant_id == body.tenant_id,
                        TrustRoot.workload_id == body.workload_id,
                    )
                )
            )
            # Revocation checks, X.509 chain format only. Two independent
            # sources are consulted as a union:
            #
            # * single-certificate fingerprint registrations: the DER
            #   fingerprint of every chain certificate (root,
            #   intermediates, leaf) is matched against in-effect
            #   registrations under the chain's trust root;
            # * the trust root's current CRL snapshot: any chain
            #   certificate whose issuer DN is the CRL issuer and whose
            #   serial is listed with an arrived revocationDate matches.
            #
            # A hit from either settles the evidence as rejected without
            # invoking the verifier — the same outcome as any other
            # rejection. The anchor trust-root row is locked for the rest
            # of this transaction; registration (of fingerprints and CRL
            # snapshots) takes the same lock before writing, so
            # registration and verification on one trust root are
            # strictly ordered: only registrations committed before this
            # settlement are visible (SQLite already serializes every
            # writer via BEGIN IMMEDIATE and ignores FOR UPDATE). A
            # settled conclusion is never rewritten by later registrations.
            # When the root has a CRL but its highest CRLNumber has passed
            # nextUpdate with no newer snapshot, verification fails closed
            # with 500 (evidence stays received) rather than trusting a
            # stale list. Any registry failure is likewise a 500 after
            # rollback, so the request can be retried once it recovers.
            revoked = False
            # Retirement short-circuit, X.509 chain format only: when the
            # chain anchors to a *retired* trust root the evidence settles
            # as rejected before the CRL check, the fingerprint revocation
            # check, the workload identity gate and the verifier run — no
            # certificate signature is ever computed and the verifier
            # plugin is not called. Retirement therefore wins over both
            # revocation sources. Retirement takes the same anchor row
            # lock as registration, so only a retirement committed before
            # this settlement is observed; a later retirement never
            # rewrites a settled conclusion. A failure reading the
            # anchor's status is a 500 before any settlement write: the
            # evidence stays received and can be re-verified after
            # recovery.
            retired_anchor = False
            # Identity gating state, populated only for an X.509 chain
            # that parses and anchors to a configured, *active* trust root.
            # When set, the anchor's workload identity profiles are
            # consulted after the verifier passes; left None when the chain
            # cannot possibly verify (the verifier rejects on its own and
            # no lookup runs) or when its anchor is retired (the rejection
            # above already settles the outcome).
            identity_anchor_id: str | None = None
            identity_leaf: x509.Certificate | None = None
            if evidence.evidence_format == X509_ATTESTED_NONCE_JSON:
                chain_certificates = _ordered_chain_certificates(body.evidence)
                if chain_certificates is not None:
                    anchor_digest = hashlib.sha256(
                        chain_certificates[-1].public_bytes(Encoding.DER)
                    ).hexdigest()
                    # Lock the configured trust root the chain anchors to.
                    # The verifier performs the actual byte-identical
                    # anchoring check; this only maps the anchor to the
                    # trust-root id registrations are scoped under and
                    # serializes with concurrent registrations and
                    # retirements. None means unconfigured: the verifier
                    # rejects on its own and no revocation lookup happens.
                    try:
                        anchor = session.scalar(
                            select(TrustRoot)
                            .where(
                                TrustRoot.tenant_id == body.tenant_id,
                                TrustRoot.workload_id == body.workload_id,
                                TrustRoot.cert_sha256 == anchor_digest,
                            )
                            .with_for_update()
                        )
                    except Exception:
                        # A failure reading the anchor's retirement status
                        # (e.g. an unavailable registry) is a 500 before
                        # any settlement write; nothing about the evidence
                        # changes, so it stays received and re-verifiable.
                        # The reserved verification slot is kept, not
                        # refunded.
                        _keep_verification_slot(
                            session,
                            body.tenant_id,
                            body.workload_id,
                            window_start,
                        )
                        logger.error(
                            "trust root status query failed for evidence %s",
                            evidence.evidence_id,
                        )
                        raise HTTPException(
                            status_code=500,
                            detail="trust root registry unavailable",
                        )
                    if anchor is not None:
                        if anchor.status == TRUST_ROOT_STATUS_RETIRED:
                            # Terminal retirement wins over every other
                            # check: no revocation lookup, no identity gate,
                            # no verifier call and therefore no certificate
                            # signature computation.
                            retired_anchor = True
                        else:
                            now = _utcnow()
                            fingerprints = [
                                b64url_encode(
                                    hashlib.sha256(
                                        certificate.public_bytes(Encoding.DER)
                                    ).digest()
                                )
                                for certificate in chain_certificates
                            ]
                            try:
                                hit = session.scalar(
                                    select(CertificateRevocation.revocation_id)
                                    .where(
                                        CertificateRevocation.tenant_id
                                        == body.tenant_id,
                                        CertificateRevocation.workload_id
                                        == body.workload_id,
                                        CertificateRevocation.trust_root_id
                                        == anchor.root_id,
                                        CertificateRevocation.certificate_fingerprint.in_(
                                            fingerprints
                                        ),
                                        CertificateRevocation.effective_at <= now,
                                    )
                                    .limit(1)
                                )
                            except Exception:
                                # The reserved verification slot is kept,
                                # not refunded.
                                _keep_verification_slot(
                                    session,
                                    body.tenant_id,
                                    body.workload_id,
                                    window_start,
                                )
                                logger.error(
                                    "revocation registry query failed for evidence %s",
                                    evidence.evidence_id,
                                )
                                raise HTTPException(
                                    status_code=500,
                                    detail="revocation registry unavailable",
                                )
                            fingerprint_revoked = hit is not None
                            # CRL revocation. The two sources are a union:
                            # a match from either rejects. For a trust root
                            # that has any CRL snapshot, only its
                            # highest-CRLNumber snapshot is current, and it
                            # is usable only while verification time has
                            # not passed its nextUpdate:
                            #
                            # * the highest snapshot is still within its
                            #   window: any chain certificate whose issuer
                            #   DN equals the CRL issuer (the root subject)
                            #   and whose serial is listed with an arrived
                            #   revocationDate settles the evidence as
                            #   rejected;
                            # * the highest snapshot is already expired and
                            #   no newer snapshot exists: revocation status
                            #   can no longer be trusted, so verification
                            #   fails closed with 500 before any settlement
                            #   write — even when the fingerprint registry
                            #   independently matches — and the evidence
                            #   stays received so the identical request can
                            #   be retried once a fresh CRL is registered;
                            # * no snapshot exists: the CRL mechanism is
                            #   simply not configured for this root and
                            #   behavior is unchanged apart from the
                            #   fingerprint check above.
                            #
                            # A trust root with no CRL registered, the
                            # HMAC format and every other entry point keep
                            # their existing behavior.
                            crl_revoked = False
                            try:
                                current_crl = session.scalar(
                                    select(CertificateRevocationList)
                                    .where(
                                        CertificateRevocationList.tenant_id
                                        == body.tenant_id,
                                        CertificateRevocationList.workload_id
                                        == body.workload_id,
                                        CertificateRevocationList.trust_root_id
                                        == anchor.root_id,
                                    )
                                    .order_by(
                                        CertificateRevocationList.crl_number.desc()
                                    )
                                    .limit(1)
                                )
                            except Exception:
                                # The reserved verification slot is kept,
                                # not refunded.
                                _keep_verification_slot(
                                    session,
                                    body.tenant_id,
                                    body.workload_id,
                                    window_start,
                                )
                                logger.error(
                                    "CRL registry query failed for evidence %s",
                                    evidence.evidence_id,
                                )
                                raise HTTPException(
                                    status_code=500,
                                    detail="CRL registry unavailable",
                                )
                            if current_crl is not None:
                                if current_crl.next_update <= now:
                                    # The highest CRLNumber is past its
                                    # nextUpdate with no newer snapshot:
                                    # fail closed rather than trusting a
                                    # stale revocation list. The reserved
                                    # verification slot is kept, not
                                    # refunded.
                                    _keep_verification_slot(
                                        session,
                                        body.tenant_id,
                                        body.workload_id,
                                        window_start,
                                    )
                                    logger.error(
                                        "current CRL expired for evidence %s",
                                        evidence.evidence_id,
                                    )
                                    raise HTTPException(
                                        status_code=500,
                                        detail="CRL registry unavailable",
                                    )
                                try:
                                    anchor_certificate = (
                                        x509.load_pem_x509_certificate(
                                            anchor.root_pem.encode("utf-8")
                                        )
                                    )
                                except (ValueError, TypeError):
                                    # The reserved verification slot is
                                    # kept, not refunded.
                                    _keep_verification_slot(
                                        session,
                                        body.tenant_id,
                                        body.workload_id,
                                        window_start,
                                    )
                                    logger.error(
                                        "stored trust root certificate is "
                                        "unparseable for evidence %s",
                                        evidence.evidence_id,
                                    )
                                    raise HTTPException(
                                        status_code=500,
                                        detail="CRL registry unavailable",
                                    )
                                # A chain certificate is CRL-revoked when
                                # its issuer DN is the CRL issuer (the root
                                # subject, compared as parsed X.509 names),
                                # its serial is listed in the current
                                # snapshot, and that entry's
                                # revocationDate has arrived.
                                candidate_serials = {
                                    str(certificate.serial_number)
                                    for certificate in chain_certificates
                                    if certificate.issuer
                                    == anchor_certificate.subject
                                }
                                if candidate_serials:
                                    try:
                                        crl_entries = session.scalars(
                                            select(CrlRevokedCertificate)
                                            .where(
                                                CrlRevokedCertificate.crl_id
                                                == current_crl.crl_id,
                                                CrlRevokedCertificate.serial_number.in_(
                                                    candidate_serials
                                                ),
                                            )
                                        ).all()
                                    except Exception:
                                        # The reserved verification slot is
                                        # kept, not refunded.
                                        _keep_verification_slot(
                                            session,
                                            body.tenant_id,
                                            body.workload_id,
                                            window_start,
                                        )
                                        logger.error(
                                            "CRL entry query failed for "
                                            "evidence %s",
                                            evidence.evidence_id,
                                        )
                                        raise HTTPException(
                                            status_code=500,
                                            detail="CRL registry unavailable",
                                        )
                                    crl_revoked = any(
                                        entry.revocation_date <= now
                                        for entry in crl_entries
                                    )
                            revoked = fingerprint_revoked or crl_revoked
                            # The chain parses and anchors to an active,
                            # configured trust root: its identity profiles
                            # gate the verifier's accept verdict. The leaf
                            # supplies the parsed issuer, subject and SAN
                            # URI strings compared against claims.
                            identity_anchor_id = anchor.root_id
                            identity_leaf = chain_certificates[0]
            if revoked or retired_anchor:
                # A revocation hit and a retired anchor are both settled
                # rejections that must never reach the verifier.
                accepted = False
            else:
                verification_context = VerificationContext(
                    evidence=body.evidence,
                    tenant_id=body.tenant_id,
                    workload_id=body.workload_id,
                    challenge=challenge_context,
                    trust_roots=trust_roots,
                )
                # The raw evidence exists only on the stack for this call; it is
                # never logged, persisted, or placed on the response. On plugin
                # failure the exception rolls the transaction back, leaving no
                # half-finished state, and only non-sensitive identifiers are
                # logged.
                try:
                    result = verifier.verify(verification_context)
                    accepted = bool(result.accepted)
                except VerifierPluginError as exc:
                    # A plugin that cannot run (e.g. the v2 shared-key
                    # configuration is missing or invalid) fails closed:
                    # the exception rolls the transaction back, leaving the
                    # evidence received for a later retry. The reserved
                    # verification slot is kept, not refunded. Log only
                    # non-sensitive identifiers and the exception type.
                    _keep_verification_slot(
                        session, body.tenant_id, body.workload_id, window_start
                    )
                    logger.error(
                        "verifier %s for format %r failed on evidence %s: %s",
                        type(verifier).__name__,
                        evidence.evidence_format,
                        evidence.evidence_id,
                        type(exc).__name__,
                    )
                    raise HTTPException(
                        status_code=500, detail="verifier plugin failure"
                    )
                except Exception as exc:
                    # Log only non-sensitive identifiers and the exception type —
                    # never the traceback/message, since a faulty plugin could
                    # embed raw evidence or private context in it. The reserved
                    # verification slot is kept, not refunded.
                    _keep_verification_slot(
                        session, body.tenant_id, body.workload_id, window_start
                    )
                    logger.error(
                        "verifier %s for format %r failed on evidence %s: %s",
                        type(verifier).__name__,
                        evidence.evidence_format,
                        evidence.evidence_id,
                        type(exc).__name__,
                    )
                    raise HTTPException(status_code=500, detail="verification failed")

            # Workload identity gate, X.509 only, consulted only after the
            # chain, signature, validity and revocation checks have passed
            # (the verifier accepted) and only when the chain anchors to a
            # configured trust root. Every committed, *active* identity
            # profile under that trust root is read here; verification
            # observes only committed profiles — registration and update
            # hold the same trust-root lock, so a profile that has not
            # committed before this point can neither affect this
            # settlement nor be applied later to a settled evidence, and a
            # revoked profile no longer participates at all (it remains
            # queryable through the lifecycle endpoints).
            #
            # * No active profile exists for the anchor: the verifier's
            #   verdict is unchanged (backward compatible; this includes
            #   an anchor whose profiles were all revoked).
            # * One or more active profiles exist: the leaf's parsed
            #   issuer DN, subject DN and a SAN URI must all equal, as
            #   parsed strings, the corresponding fields of at least one
            #   claim of at least one profile ("any claim of any profile
            #   hits"); otherwise the evidence is rejected.
            # A query failure is a 500 before any settlement write, so the
            # evidence stays received and can be retried after recovery.
            if accepted and identity_anchor_id is not None:
                try:
                    profile_claims = session.scalars(
                        select(WorkloadIdentityClaim)
                        .join(
                            WorkloadIdentityProfile,
                            WorkloadIdentityProfile.profile_id
                            == WorkloadIdentityClaim.profile_id,
                        )
                        .where(
                            WorkloadIdentityClaim.tenant_id == body.tenant_id,
                            WorkloadIdentityClaim.workload_id == body.workload_id,
                            WorkloadIdentityClaim.trust_root_id == identity_anchor_id,
                            WorkloadIdentityProfile.status
                            == WORKLOAD_IDENTITY_STATUS_ACTIVE,
                        )
                    ).all()
                except Exception:
                    # The reserved verification slot is kept, not refunded.
                    _keep_verification_slot(
                        session, body.tenant_id, body.workload_id, window_start
                    )
                    logger.error(
                        "workload identity profile query failed for evidence %s",
                        evidence.evidence_id,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="workload identity registry unavailable",
                    )
                if profile_claims:
                    leaf_issuer, leaf_subject, leaf_uris = _leaf_identity_strings(
                        identity_leaf
                    )
                    matched_profile: bool = False
                    for claim in profile_claims:
                        if (
                            claim.issuer == leaf_issuer
                            and claim.subject == leaf_subject
                            and claim.uri in leaf_uris
                        ):
                            matched_profile = True
                            break
                    if not matched_profile:
                        # Profiles exist but none describe this leaf: the
                        # only possible outcome is rejection.
                        accepted = False

            # The persisted outcome is a fixed, service-defined result code
            # derived solely from the accept/reject verdict. Any free-form
            # text the plugin attached to its result is discarded here and
            # never persisted, logged, or returned.
            new_status = "verified" if accepted else "rejected"
            result_code = (
                VERIFICATION_RESULT_ACCEPTED
                if accepted
                else VERIFICATION_RESULT_REJECTED
            )
            verified_at = _utcnow()

            # Atomic settlement: exactly one caller can flip
            # received -> a terminal status.
            outcome = session.execute(
                update(Evidence)
                .where(
                    Evidence.evidence_id == evidence_id,
                    Evidence.status == "received",
                )
                .values(
                    status=new_status,
                    verified_at=verified_at,
                    verification_result=result_code,
                )
                .execution_options(synchronize_session=False)
            )
            if outcome.rowcount != 1:
                # A concurrent transaction settled first despite the lock;
                # read and return its conclusion rather than overwriting it.
                session.rollback()
                winner = session.get(Evidence, evidence_id)
                if winner is not None and winner.status in ("verified", "rejected"):
                    return EvidenceVerifiedResponse(
                        evidence_id=winner.evidence_id,
                        challenge_id=winner.challenge_id,
                        status=winner.status,
                        verified_at=_rfc3339(winner.verified_at),
                    )
                raise HTTPException(status_code=409, detail="verification conflict")
            # The proof-lifecycle verification event commits in the same
            # transaction as the winning settlement, recording only the
            # fixed verified/rejected code and the settlement time —
            # never the evidence, nonce, claims or any plugin text. A
            # losing race rolled back above without reaching this insert.
            # The scope already contains this proof's committed reception
            # event, so the next commit-order sequence always has a prior
            # row to lock: concurrent verifications of different proofs in
            # the same scope serialize on that row (BEGIN IMMEDIATE on
            # SQLite) and never need to retry.
            session.add(
                ProofLifecycleEvent(
                    event_id=str(uuid.uuid4()),
                    tenant_id=evidence.tenant_id,
                    workload_id=evidence.workload_id,
                    event_type=PROOF_EVENT_TYPE_VERIFIED,
                    evidence_id=evidence_id,
                    commit_seq=_next_proof_event_commit_seq(
                        session, evidence.tenant_id, evidence.workload_id
                    ),
                    policy_version=None,
                    evidence_format=None,
                    status=(
                        PROOF_EVENT_STATUS_VERIFIED
                        if accepted
                        else PROOF_EVENT_STATUS_REJECTED
                    ),
                    occurred_at=verified_at,
                )
            )
            session.commit()

        return EvidenceVerifiedResponse(
            evidence_id=evidence_id,
            challenge_id=evidence.challenge_id,
            status=new_status,
            verified_at=_rfc3339(verified_at),
        )

    @app.post(
        "/v1/trust-roots", status_code=201, response_model=TrustRootCreatedResponse
    )
    def create_trust_root(body: CreateTrustRootRequest) -> TrustRootCreatedResponse:
        # Only X.509 CA certificates are accepted. A PEM private key (or any
        # other non-certificate material) fails to parse here, so private
        # key material is never persisted — only the public certificate.
        try:
            certificate = x509.load_pem_x509_certificate(
                body.root_pem.encode("utf-8")
            )
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=422, detail="root_pem is not a valid X.509 certificate"
            )
        try:
            basic = certificate.extensions.get_extension_for_class(
                x509.BasicConstraints
            )
        except x509.ExtensionNotFound:
            basic = None
        if basic is None or not basic.value.ca:
            raise HTTPException(
                status_code=422, detail="root_pem must be a CA certificate"
            )
        der = certificate.public_bytes(Encoding.DER)
        cert_digest = hashlib.sha256(der).hexdigest()
        # Persist the normalized PEM serialization of the public certificate.
        pem = certificate.public_bytes(Encoding.PEM).decode("ascii")
        now = _utcnow()
        root_id = str(uuid.uuid4())
        with session_factory() as session:
            duplicate = session.scalar(
                select(TrustRoot).where(
                    TrustRoot.tenant_id == body.tenant_id,
                    TrustRoot.workload_id == body.workload_id,
                    TrustRoot.cert_sha256 == cert_digest,
                )
            )
            if duplicate is not None:
                raise HTTPException(
                    status_code=409, detail="trust root already configured"
                )
            # Sequence the creation inside its own write transaction. The
            # same per-scope counter is advanced by retirement, so the
            # read-only lifecycle query can bound a replayable snapshot by
            # a commit high-water mark immune to both later inserts and
            # later in-place retirements. A rolled-back creation returns
            # the allocation as well.
            commit_seq = _next_trust_root_commit_seq(
                session, body.tenant_id, body.workload_id
            )
            session.add(
                TrustRoot(
                    root_id=root_id,
                    tenant_id=body.tenant_id,
                    workload_id=body.workload_id,
                    name=body.name,
                    root_pem=pem,
                    cert_sha256=cert_digest,
                    created_at=now,
                    commit_seq=commit_seq,
                )
            )
            try:
                session.commit()
            except IntegrityError:
                # A concurrent request configured the same certificate first.
                session.rollback()
                raise HTTPException(
                    status_code=409, detail="trust root already configured"
                )
        return TrustRootCreatedResponse(
            root_id=root_id,
            tenant_id=body.tenant_id,
            workload_id=body.workload_id,
            name=body.name,
            created_at=_rfc3339(now),
        )

    @app.post("/v1/trust-roots//retire")
    def retire_trust_root_identifier_required(
        body: RetireTrustRootRequest,
    ) -> Response:
        # An empty path segment is a missing trust-root identifier: a 422
        # client error rather than a routing-level 404 or 405. It never
        # reads or modifies a trust root.
        raise HTTPException(status_code=422, detail="invalid trust root identifier")

    @app.post("/v1/trust-roots/{root_id}/retire")
    def retire_trust_root(root_id: str, body: RetireTrustRootRequest) -> Response:
        """Retire a configured trust root.

        The path identifier must be a canonical UUID and the body carries
        only the two non-blank scope strings; any missing, blank,
        wrong-typed or unknown field is a 422 that never reads or writes
        the trust root. A path UUID is checked before storage is touched.
        An unknown root or one outside the body's tenant/workload is an
        indistinguishable 404 (existence is never revealed). A repeat
        retire returns 409 and never rewrites the recorded ``retired_at``
        and appends no other record. The active -> retired transition is
        one guarded atomic update under the trust-root row lock — the same
        lock X.509 verification takes while inspecting the anchor — so
        concurrent retires of one root settle as exactly one success and
        stable 409s, and only a retirement committed before an evidence
        settles can affect that verification. A write or commit failure
        returns 500 after a full rollback, leaving status and time
        unchanged.
        """
        # A path identifier that is missing (empty segment), blank,
        # whitespace-padded or not a canonical lowercase UUID is a
        # field/format error; the raw value must match exactly. This runs
        # before any storage access, so a malformed path can neither read
        # nor modify a trust root.
        if not _UUID_RE.fullmatch(root_id):
            raise HTTPException(status_code=422, detail="invalid trust root identifier")

        retired_at = _utcnow()
        with session_factory() as session:
            try:
                # Lock the trust-root row for the whole transition. X.509
                # verification takes the same lock while inspecting the
                # anchor, so retirement and verification are strictly
                # ordered: a retirement either commits before the evidence
                # settles (and the verification rejects) or waits until
                # after it settles (and never retroactively changes the
                # conclusion). SQLite ignores FOR UPDATE but already
                # serializes all writers via BEGIN IMMEDIATE.
                trust_root = session.scalar(
                    select(TrustRoot)
                    .where(
                        TrustRoot.root_id == root_id,
                        TrustRoot.tenant_id == body.tenant_id,
                        TrustRoot.workload_id == body.workload_id,
                    )
                    .with_for_update()
                )
                if trust_root is None:
                    # Do not reveal whether an out-of-scope or unknown root
                    # exists: unknown id and scope mismatch share one 404.
                    raise HTTPException(status_code=404, detail="trust root not found")
                if trust_root.status == TRUST_ROOT_STATUS_RETIRED:
                    # A repeated retire is a conflict. It neither rewrites
                    # the recorded retirement time nor appends any audit or
                    # other state record.
                    raise HTTPException(
                        status_code=409, detail="trust root already retired"
                    )
                # Sequence the retirement from the same per-scope counter
                # creation draws from, before the guarded flip. The root row
                # is already locked, so no concurrent transition can
                # interleave: the lifecycle query reconstructs this root's
                # as-of-snapshot status by comparing retired_seq against its
                # fixed high-water mark. The counter row is the only other
                # lock this path takes and creation never takes the root
                # lock, so the two lock orders cannot cycle.
                retired_seq = _next_trust_root_commit_seq(
                    session, body.tenant_id, body.workload_id
                )
                # Atomic settlement: only one caller can flip
                # active -> retired, and retired_at/retired_seq are written
                # by that same single-row update.
                outcome = session.execute(
                    update(TrustRoot)
                    .where(
                        TrustRoot.root_id == root_id,
                        TrustRoot.status == TRUST_ROOT_STATUS_ACTIVE,
                    )
                    .values(
                        status=TRUST_ROOT_STATUS_RETIRED,
                        retired_at=retired_at,
                        retired_seq=retired_seq,
                    )
                    .execution_options(synchronize_session=False)
                )
                if outcome.rowcount != 1:
                    session.rollback()
                    fresh = session.get(TrustRoot, root_id)
                    if (
                        fresh is not None
                        and fresh.status == TRUST_ROOT_STATUS_RETIRED
                    ):
                        raise HTTPException(
                            status_code=409, detail="trust root already retired"
                        )
                    raise HTTPException(
                        status_code=409, detail="trust root retirement conflict"
                    )
                try:
                    session.commit()
                except Exception:
                    session.rollback()
                    logger.error("trust root retirement failed")
                    raise HTTPException(
                        status_code=500, detail="trust root retirement failed"
                    )
            except HTTPException:
                # 404/409 judgements and the controlled 500 above keep
                # their status; a read-only judgement has written nothing.
                raise
            except Exception:
                # A failure during the locked lookup is a full rollback and
                # a sanitized 500: the status update can never have happened
                # on this path, so the root stays active with no time set.
                session.rollback()
                logger.error("trust root retirement failed")
                raise HTTPException(
                    status_code=500, detail="trust root retirement failed"
                )
        # Compact JSON describing only the retirement result. The keys are
        # emitted in the fixed order root_id, status, retired_at; every
        # value is a string (allow_nan=False makes non-finite numbers
        # impossible), terminated by exactly one newline.
        body_bytes = json.dumps(
            {
                "root_id": root_id,
                "status": TRUST_ROOT_STATUS_RETIRED,
                "retired_at": _rfc3339(retired_at),
            },
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return Response(
            content=body_bytes + b"\n",
            status_code=200,
            media_type="application/json",
        )

    @app.get("/v1/trust-roots")
    def list_trust_roots(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        root_id: str | None = Query(default=None),
        name: str | None = Query(default=None),
        status: str | None = Query(default=None),
        created_after: str | None = Query(default=None),
        created_before: str | None = Query(default=None),
        cursor: str | None = Query(default=None),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return a read-only, cursor-stable page of trust-root registrations.

        The range is fixed by the mandatory tenant and workload and may be
        narrowed by an explicit registration id (a canonical UUID), an
        exact name, a lifecycle status (``active``/``retired``) and an
        inclusive creation-time window. Ordering is stable
        ``(created_at, root_id)`` ascending with an exclusive keyset
        cursor. The cursor carries its own kind tag, is HMAC-authenticated
        and is bound to the scope, every active filter *and* the fixed
        snapshot established by the range's first (cursor-less or explicit
        empty-cursor) query, so it can be neither forged nor replayed
        against a different scope, filter set, snapshot or query family.

        The snapshot is fixed at the per-scope lifecycle commit boundary:
        creation and retirement share one gap-free counter, and a root's
        status as-of the snapshot is reconstructed from the sequence its
        retirement committed at. Replaying a cursor therefore returns the
        identical page even while roots are concurrently retired or new
        roots registered; those changes surface only in a fresh first
        query. The handler issues only SELECTs — it never creates, retires
        or otherwise mutates a trust root — and a storage failure aborts
        the whole request with a 500 rather than returning a half page.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {
            "tenant_id",
            "workload_id",
            "root_id",
            "name",
            "status",
            "created_after",
            "created_before",
            "cursor",
        }
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is rejected rather than silently
        # treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )

        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        root_filter: str | None = None
        if root_id is not None:
            if not root_id.strip() or not _UUID_RE.fullmatch(root_id):
                raise HTTPException(
                    status_code=422, detail="invalid trust root identifier"
                )
            root_filter = root_id

        # Exact-text name filter. Only a blank/whitespace value is a shape
        # error; a non-blank value is matched verbatim (names may contain
        # interior or surrounding visible characters) and a name that
        # matches nothing is an empty range, never a 404.
        name_filter: str | None = None
        if name is not None:
            if not name.strip():
                raise HTTPException(status_code=422, detail="invalid name")
            name_filter = name

        if status is not None:
            if not status.strip() or status not in TRUST_ROOT_STATUS_CODES:
                raise HTTPException(status_code=422, detail="invalid status")
        status_filter = status if status is not None else ""

        def _time_bound(value: str | None, name: str) -> tuple[str, datetime | None]:
            if value is None:
                return "", None
            if not value.strip():
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            try:
                parsed = _parse_utc_rfc3339(value)
            except ValueError:
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            # Normalize the spelling embedded into the cursor so two
            # equivalent UTC spellings cannot mint different cursor domains.
            return _rfc3339(parsed), parsed

        after_raw, after_dt = _time_bound(created_after, "created_after")
        before_raw, before_dt = _time_bound(created_before, "created_before")
        # The window is closed on both ends; equality is a valid
        # single-instant window and the start must not follow the end.
        if after_dt is not None and before_dt is not None and after_dt > before_dt:
            raise HTTPException(
                status_code=422,
                detail="created_after must not be later than created_before",
            )

        # Omitted cursor or an explicit empty string starts before the
        # smallest registration and fixes the range's replayable snapshot.
        # Whitespace, malformed, forged, cross-scope, cross-filter,
        # cross-snapshot or foreign-kind cursors are indistinguishable 422s.
        boundary_dt: datetime | None = None
        boundary_root: str | None = None
        snapshot_seq: int | None = None
        if cursor is not None and cursor != "":
            if not cursor.strip() or not _CURSOR_RE.fullmatch(cursor):
                raise HTTPException(status_code=422, detail="invalid cursor")
            decoded_boundary = _decode_trust_root_cursor(
                cursor,
                tenant_id,
                workload_id,
                root_id=root_filter or "",
                name=name_filter or "",
                status=status_filter,
                created_after=after_raw,
                created_before=before_raw,
            )
            if decoded_boundary is None:
                raise HTTPException(status_code=422, detail="invalid cursor")
            boundary_at_raw, boundary_root, snapshot_seq = decoded_boundary
            try:
                boundary_dt = _parse_utc_rfc3339(boundary_at_raw)
            except ValueError:
                raise HTTPException(status_code=422, detail="invalid cursor")

        # --- read-only scan ---------------------------------------------
        rows: list = []
        try:
            with session_factory() as session:
                # An explicitly named registration must exist in exactly
                # this tenant and workload; an unknown or cross-scope root
                # is an indistinguishable 404 (an explicit resource
                # selector, not a mere range bound). A name-only filter is
                # different: no such resource is named, so a non-match is
                # an empty range rather than a 404.
                if root_filter is not None:
                    named = session.get(TrustRoot, root_filter)
                    if (
                        named is None
                        or named.tenant_id != tenant_id
                        or named.workload_id != workload_id
                    ):
                        raise HTTPException(
                            status_code=404, detail="trust root not found"
                        )

                if snapshot_seq is None:
                    # First query of the range: fix a replayable snapshot
                    # at the scope's current lifecycle commit high-water
                    # mark. Creation and retirement share this one counter,
                    # so a registration or a retirement committed afterwards
                    # takes a greater sequence: the new row lies beyond the
                    # mark and an as-yet-active row keeps reconstructing as
                    # active on replayed pages, regardless of in-place
                    # updates or equal business timestamps.
                    fixed_seq = session.scalar(
                        select(TrustRootCommitCounter.last_seq).where(
                            TrustRootCommitCounter.tenant_id == tenant_id,
                            TrustRootCommitCounter.workload_id == workload_id,
                        )
                    )
                    # No committed lifecycle change in the scope yet.
                    snapshot_seq = int(fixed_seq) if fixed_seq is not None else 0

                if snapshot_seq > 0:
                    stmt = select(TrustRoot).where(
                        TrustRoot.tenant_id == tenant_id,
                        TrustRoot.workload_id == workload_id,
                        # Membership: registrations that had committed by
                        # the snapshot. Roots are never deleted, so the
                        # immutable creation sequence alone bounds membership.
                        TrustRoot.commit_seq <= snapshot_seq,
                    )
                    if root_filter is not None:
                        stmt = stmt.where(TrustRoot.root_id == root_filter)
                    if name_filter is not None:
                        # Name is immutable, so the current column value is
                        # exactly the value the snapshot row carried.
                        stmt = stmt.where(TrustRoot.name == name_filter)
                    if after_dt is not None:
                        stmt = stmt.where(TrustRoot.created_at >= after_dt)
                    if before_dt is not None:
                        stmt = stmt.where(TrustRoot.created_at <= before_dt)

                    # Status is evaluated as-of the snapshot rather than
                    # read from the mutable current column: a root is
                    # retired in the snapshot exactly when its retirement
                    # committed at or before the high-water mark. A
                    # retirement committed afterwards must not flip a
                    # replayed page from active to retired.
                    retired_as_of = and_(
                        TrustRoot.retired_seq.is_not(None),
                        TrustRoot.retired_seq <= snapshot_seq,
                    )
                    if status_filter == TRUST_ROOT_STATUS_RETIRED:
                        stmt = stmt.where(retired_as_of)
                    elif status_filter == TRUST_ROOT_STATUS_ACTIVE:
                        stmt = stmt.where(
                            or_(
                                TrustRoot.retired_seq.is_(None),
                                TrustRoot.retired_seq > snapshot_seq,
                            )
                        )

                    if boundary_dt is not None:
                        # Exclusive (created_at, root_id) keyset. Both
                        # components are immutable, so the boundary walks
                        # the same fixed snapshot set in stable order.
                        stmt = stmt.where(
                            or_(
                                TrustRoot.created_at > boundary_dt,
                                and_(
                                    TrustRoot.created_at == boundary_dt,
                                    TrustRoot.root_id > boundary_root,
                                ),
                            )
                        )
                    stmt = stmt.order_by(
                        TrustRoot.created_at.asc(),
                        TrustRoot.root_id.asc(),
                    ).limit(TRUST_ROOT_PAGE_SIZE + 1)
                    # One extra row is the "more follows" probe. The scan
                    # is a single read-only statement: a storage failure
                    # aborts the whole request with a 500 rather than
                    # returning a partial page.
                    rows = list(session.scalars(stmt))
        except HTTPException:
            raise
        except Exception:
            logger.error("trust root query failed")
            raise HTTPException(
                status_code=500, detail="trust root registry unavailable"
            )

        has_more = len(rows) > TRUST_ROOT_PAGE_SIZE
        page = rows[:TRUST_ROOT_PAGE_SIZE]

        trust_roots = [
            {
                # Creation-response field order, then lifecycle fields.
                "root_id": row.root_id,
                "tenant_id": row.tenant_id,
                "workload_id": row.workload_id,
                "name": row.name,
                "created_at": _rfc3339(row.created_at),
                # Reconstruct the as-of-snapshot status from the retirement
                # commit marker rather than the mutable current column.
                "status": (
                    TRUST_ROOT_STATUS_RETIRED
                    if row.retired_seq is not None
                    and row.retired_seq <= snapshot_seq
                    else TRUST_ROOT_STATUS_ACTIVE
                ),
                # The recorded retirement time is shown only when the
                # retirement had committed by the snapshot; an active
                # (as-of) root reports null even if it was retired later.
                "retired_at": (
                    _rfc3339(row.retired_at)
                    if row.retired_seq is not None
                    and row.retired_seq <= snapshot_seq
                    else None
                ),
            }
            for row in page
        ]

        if has_more:
            last = page[-1]
            next_cursor = _encode_trust_root_cursor(
                tenant_id,
                workload_id,
                _rfc3339(last.created_at),
                last.root_id,
                root_id=root_filter or "",
                name=name_filter or "",
                status=status_filter,
                created_after=after_raw,
                created_before=before_raw,
                snapshot_seq=snapshot_seq,
            )
            complete = False
        else:
            next_cursor = ""
            complete = True

        # Compact container (list, next_cursor, complete) with a single
        # terminating newline. Every entry value is a JSON string or null
        # and complete is a boolean, so no floats, -0.0 or non-finite
        # values can appear. No certificate PEM, key, evidence or exception
        # text is ever included.
        body = (
            json.dumps(
                {
                    "trust_roots": trust_roots,
                    "next_cursor": next_cursor,
                    "complete": complete,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.post("/v1/revocations", status_code=201)
    def register_revocation(body: CreateRevocationRequest) -> Response:
        """Register a trust-root-scoped X.509 certificate revocation.

        Field and format validation is completed by the request model
        before this handler runs (422 with no state written). The named
        trust root must exist in exactly the request's tenant and
        workload; an unknown or cross-scope root is an indistinguishable
        404. Within one trust root the same fingerprint is registered at
        most once (409 on repeat); distinct fingerprints are independent
        rows. The registration (scope, trust root, fingerprint,
        effective_at) is a single atomic row write, so any failure leaves
        no half record. The response is compact JSON carrying only the
        registration identifier, trust root, fingerprint and effective
        time — never certificate material or exception detail.
        """
        effective_at = _parse_utc_rfc3339(body.effective_at)
        revocation_id = str(uuid.uuid4())
        now = _utcnow()
        with session_factory() as session:
            try:
                # Lock the trust-root row for the full registration. X.509
                # verification takes the same lock while checking the registry,
                # so a registration either commits before the evidence settles
                # (and the verification observes it) or waits until after it
                # settles (and never retroactively changes the conclusion).
                # SQLite ignores FOR UPDATE but already serializes all writers
                # via BEGIN IMMEDIATE.
                trust_root = session.scalar(
                    select(TrustRoot)
                    .where(
                        TrustRoot.root_id == body.trust_root_id,
                        TrustRoot.tenant_id == body.tenant_id,
                        TrustRoot.workload_id == body.workload_id,
                    )
                    .with_for_update()
                )
                if trust_root is None:
                    # Do not reveal whether an out-of-scope trust root exists.
                    raise HTTPException(status_code=404, detail="trust root not found")
                existing = session.scalar(
                    select(CertificateRevocation.revocation_id).where(
                        CertificateRevocation.tenant_id == body.tenant_id,
                        CertificateRevocation.workload_id == body.workload_id,
                        CertificateRevocation.trust_root_id == body.trust_root_id,
                        CertificateRevocation.certificate_fingerprint
                        == body.certificate_fingerprint,
                    )
                )
                if existing is not None:
                    # Repeat registrations never merge or overwrite: a second
                    # effective time for the same (root, fingerprint) is
                    # rejected rather than replacing the first.
                    raise HTTPException(
                        status_code=409,
                        detail="certificate already revoked for this trust root",
                    )
                session.add(
                    CertificateRevocation(
                        revocation_id=revocation_id,
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        trust_root_id=body.trust_root_id,
                        certificate_fingerprint=body.certificate_fingerprint,
                        effective_at=effective_at,
                        created_at=now,
                    )
                )
                try:
                    session.commit()
                except IntegrityError:
                    # A concurrent request registered the same fingerprint
                    # first; the unique constraint guarantees at most one row
                    # and the loser reports a stable 409.
                    session.rollback()
                    raise HTTPException(
                        status_code=409,
                        detail="certificate already revoked for this trust root",
                    )
                except Exception:
                    # Any other commit/write failure rolls the (single-row)
                    # transaction back completely, so no half record survives.
                    session.rollback()
                    logger.error("certificate revocation write failed")
                    raise HTTPException(
                        status_code=500, detail="revocation registration failed"
                    )
            except HTTPException:
                # 404/409 judgements and the controlled 500 above keep their
                # status; a read-only judgement has written nothing.
                raise
            except Exception:
                # A failure during the locked lookups (e.g. an unavailable
                # registry) is likewise a full rollback and a sanitized 500:
                # the insert can never have happened on this path.
                session.rollback()
                logger.error("certificate revocation registration failed")
                raise HTTPException(
                    status_code=500, detail="revocation registration failed"
                )
        body_bytes = json.dumps(
            {
                "revocation_id": revocation_id,
                "trust_root_id": body.trust_root_id,
                "certificate_fingerprint": body.certificate_fingerprint,
                "effective_at": _rfc3339(effective_at),
            },
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return Response(
            content=body_bytes, status_code=201, media_type="application/json"
        )

    @app.post("/v1/crls", status_code=201)
    def register_crl(body: CreateCrlRequest) -> Response:
        """Register an X.509 v2 CRL snapshot under a configured trust root.

        Error order is fixed: field/UUID shape and every intrinsic CRL
        defect (PEM/ASN.1, non-v2, missing CRLNumber/nextUpdate, illegal
        time window, illegal or duplicate serials) are 422s produced
        before storage is touched; an unknown or cross-scope trust root is
        an indistinguishable 404; the issuer-DN and signature checks (also
        422) run only once the root has been found; an equal CRLNumber, a
        byte-identical body, or any number not strictly higher than the
        root's current maximum is a 409; storage or service faults are
        sanitized 500s after a full rollback, leaving the previous
        snapshot in place. On success the snapshot and all of its revoked
        entries commit as one atomic write, and the response carries only
        identifiers, parsed metadata and counts — never the CRL body,
        certificate material or exception detail.
        """
        now = _utcnow()
        # Intrinsic validation depends only on the CRL body and the receipt
        # time, so it runs (and can fail 422) before any storage access —
        # in particular before revealing whether the trust root exists.
        try:
            validated = parse_crl(body.crl_pem, now=now)
        except CrlValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        crl_id = str(uuid.uuid4())
        with session_factory() as session:
            try:
                # Lock the trust-root row for the full registration. X.509
                # verification takes the same lock while consulting the
                # CRL, and single-fingerprint registration takes it too,
                # so snapshot replacement and verification are strictly
                # ordered: a snapshot either commits before the evidence
                # settles (and that verification observes it) or waits
                # until after it settles (and never retroactively changes
                # the conclusion). SQLite ignores FOR UPDATE but already
                # serializes all writers via BEGIN IMMEDIATE.
                trust_root = session.scalar(
                    select(TrustRoot)
                    .where(
                        TrustRoot.root_id == body.trust_root_id,
                        TrustRoot.tenant_id == body.tenant_id,
                        TrustRoot.workload_id == body.workload_id,
                    )
                    .with_for_update()
                )
                if trust_root is None:
                    # Do not reveal whether an out-of-scope trust root exists.
                    raise HTTPException(status_code=404, detail="trust root not found")
                try:
                    root_certificate = x509.load_pem_x509_certificate(
                        trust_root.root_pem.encode("utf-8")
                    )
                except (ValueError, TypeError):
                    # The stored root is public material the service itself
                    # accepted and normalized at creation; its failure to
                    # parse is a service-side fault, never a client error.
                    session.rollback()
                    logger.error("stored trust root certificate is unparseable")
                    raise HTTPException(
                        status_code=500, detail="CRL registration failed"
                    )
                # Trust-dependent checks: issuer DN must equal the root's
                # subject and the CRL must be signed by the root's key.
                try:
                    parsed = validate_crl_against_root(
                        validated, issuer_certificate=root_certificate
                    )
                except CrlValidationError as exc:
                    raise HTTPException(status_code=422, detail=str(exc))

                # Conflict rules, all 409:
                # * the same CRLNumber (covered below by the high-water
                #   check, and additionally guarded by the unique index);
                # * a byte-identical CRL body already registered under this
                #   root (checked against every snapshot, not just the
                #   current one);
                # * a number not strictly higher than the root's current
                #   maximum. The row is locked so this high-water mark is
                #   stable for the rest of the transaction.
                content_hit = session.scalar(
                    select(CertificateRevocationList.crl_id).where(
                        CertificateRevocationList.tenant_id == body.tenant_id,
                        CertificateRevocationList.workload_id == body.workload_id,
                        CertificateRevocationList.trust_root_id
                        == body.trust_root_id,
                        CertificateRevocationList.crl_sha256 == parsed.crl_sha256,
                    )
                )
                if content_hit is not None:
                    raise HTTPException(
                        status_code=409,
                        detail="identical CRL already registered for this trust root",
                    )
                current_max = session.scalar(
                    select(func.max(CertificateRevocationList.crl_number)).where(
                        CertificateRevocationList.tenant_id == body.tenant_id,
                        CertificateRevocationList.workload_id == body.workload_id,
                        CertificateRevocationList.trust_root_id
                        == body.trust_root_id,
                    )
                )
                if current_max is not None and parsed.crl_number <= current_max:
                    raise HTTPException(
                        status_code=409,
                        detail="CRLNumber must be higher than the current CRL",
                    )

                session.add(
                    CertificateRevocationList(
                        crl_id=crl_id,
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        trust_root_id=body.trust_root_id,
                        crl_number=parsed.crl_number,
                        crl_sha256=parsed.crl_sha256,
                        this_update=parsed.this_update,
                        next_update=parsed.next_update,
                        revoked_count=parsed.revoked_count,
                        created_at=now,
                    )
                )
                for entry in parsed.entries:
                    session.add(
                        CrlRevokedCertificate(
                            entry_id=str(uuid.uuid4()),
                            crl_id=crl_id,
                            tenant_id=body.tenant_id,
                            workload_id=body.workload_id,
                            trust_root_id=body.trust_root_id,
                            issuer_dn=parsed.issuer_dn,
                            serial_number=str(entry.serial_number),
                            revocation_date=entry.revocation_date,
                        )
                    )
                try:
                    session.commit()
                except IntegrityError:
                    # A concurrent request registered the same number or
                    # the same body first; the unique constraints guarantee
                    # at most one snapshot and the loser reports 409.
                    session.rollback()
                    raise HTTPException(
                        status_code=409,
                        detail="CRL conflicts with the current CRL",
                    )
                except Exception:
                    # Any other commit/write failure rolls the whole
                    # snapshot (parent plus every entry) back, so no
                    # partial registration survives and the previous
                    # snapshot stays current.
                    session.rollback()
                    logger.error("CRL snapshot write failed")
                    raise HTTPException(
                        status_code=500, detail="CRL registration failed"
                    )
            except HTTPException:
                # 404/409 judgements and the controlled 422/500s above keep
                # their status; a read-only judgement has written nothing.
                raise
            except Exception:
                # A failure during the locked lookups (e.g. an unavailable
                # registry) is likewise a full rollback and a sanitized
                # 500: neither the snapshot nor any entry can exist yet.
                session.rollback()
                logger.error("CRL registration failed")
                raise HTTPException(status_code=500, detail="CRL registration failed")
        body_bytes = json.dumps(
            {
                "crl_id": crl_id,
                "trust_root_id": body.trust_root_id,
                "crl_number": parsed.crl_number,
                "this_update": _rfc3339(parsed.this_update),
                "next_update": _rfc3339(parsed.next_update),
                "revoked_count": parsed.revoked_count,
            },
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return Response(
            content=body_bytes, status_code=201, media_type="application/json"
        )

    @app.get("/v1/crls/")
    def crl_identifier_required() -> Response:
        # An empty path segment is a missing CRL identifier: a 422 client
        # error rather than a routing-level 404.
        raise HTTPException(status_code=422, detail="invalid CRL identifier")

    # Registered before /v1/crls/{crl_id} so the literal "status" segment
    # binds here rather than to the snapshot lookup (where it would be a
    # 422 as a non-UUID identifier).
    @app.get("/v1/crls/status")
    def get_crl_status(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        trust_root_id: str = Query(...),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return the current revocation-list freshness of one trust root.

        The range is fixed entirely by the three mandatory query
        parameters, each appearing exactly once; the request body is
        always empty. Every shape failure (a missing, blank or
        whitespace-padded scope parameter, a non-canonical trust root
        UUID, a repeated or unknown parameter, or any non-empty body) is
        a 422 before any state is read. An unknown trust root and one
        belonging to another tenant or workload are the same
        indistinguishable 404. On success the response reports the
        highest-CRLNumber snapshot of the root as of one consistent
        committed read stamped ``observed_at``: ``missing`` when the root
        has no snapshot at all (all four CRL fields null), otherwise
        ``fresh`` while ``observed_at`` is before the snapshot's
        ``next_update`` and ``stale`` once it has been reached. The
        handler issues only SELECTs: it writes no audit record, updates
        or deletes nothing, never parses or returns the CRL body, its
        entries, digests or certificate material, and never changes later
        verification. A concurrent registration is observed either
        completely before or completely after its commit, never as a
        partial snapshot. A storage failure or inconsistent read is a
        sanitized 500 with no partial result.
        """
        # --- request shape (all 422, no state is read) ------------------
        allowed_params = {"tenant_id", "workload_id", "trust_root_id"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is rejected rather than silently
        # treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )
        # Scope identifiers are non-empty and carry no leading or
        # trailing whitespace; the trust root identifier is a canonical
        # lowercase UUID.
        if (
            not tenant_id
            or tenant_id != tenant_id.strip()
            or not workload_id
            or workload_id != workload_id.strip()
        ):
            raise HTTPException(status_code=422, detail="invalid scope parameters")
        if not _UUID_RE.fullmatch(trust_root_id):
            raise HTTPException(
                status_code=422, detail="invalid trust root identifier"
            )

        # One timestamp for the whole read: it stamps the response and
        # decides freshness, so the reported state always agrees with the
        # reported observation time.
        observed_at = _utcnow()
        try:
            with session_factory() as session:
                trust_root = session.scalar(
                    select(TrustRoot.root_id).where(
                        TrustRoot.root_id == trust_root_id,
                        TrustRoot.tenant_id == tenant_id,
                        TrustRoot.workload_id == workload_id,
                    )
                )
                if trust_root is None:
                    # Do not reveal whether an out-of-scope trust root exists.
                    raise HTTPException(
                        status_code=404, detail="trust root not found"
                    )
                # The same highest-CRLNumber selection verification uses.
                # A snapshot commits atomically with its entries, so this
                # single SELECT observes one fully committed snapshot (or
                # none), never a half-registered one.
                current_crl = session.scalar(
                    select(CertificateRevocationList)
                    .where(
                        CertificateRevocationList.tenant_id == tenant_id,
                        CertificateRevocationList.workload_id == workload_id,
                        CertificateRevocationList.trust_root_id == trust_root_id,
                    )
                    .order_by(CertificateRevocationList.crl_number.desc())
                    .limit(1)
                )
        except HTTPException:
            raise
        except Exception:
            logger.error("CRL status query failed")
            raise HTTPException(status_code=500, detail="CRL registry unavailable")

        if current_crl is None:
            state = "missing"
            current_crl_id = None
            crl_number = None
            this_update = None
            next_update = None
        else:
            # next_update reached means stale: the boundary itself is no
            # longer fresh.
            state = "fresh" if observed_at < current_crl.next_update else "stale"
            current_crl_id = current_crl.crl_id
            crl_number = current_crl.crl_number
            this_update = _rfc3339(current_crl.this_update)
            next_update = _rfc3339(current_crl.next_update)

        # Exactly nine fields in a fixed order; compact JSON terminated
        # by a single newline.
        return _compact_json_line(
            {
                "tenant_id": tenant_id,
                "workload_id": workload_id,
                "trust_root_id": trust_root_id,
                "observed_at": _rfc3339(observed_at),
                "state": state,
                "current_crl_id": current_crl_id,
                "crl_number": crl_number,
                "this_update": this_update,
                "next_update": next_update,
            }
        )

    @app.get("/v1/crls/{crl_id}")
    def get_crl(
        request: Request,
        crl_id: str,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        trust_root_id: str = Query(...),
    ) -> Response:
        """Return one registered CRL snapshot with all of its entries.

        The path identifier and the mandatory ``trust_root_id`` query
        parameter must be canonical lowercase UUIDs and the scope
        parameters non-blank; any other query parameter, a repeated
        parameter or a shape defect is a 422 before storage is touched.
        An unknown identifier and one belonging to another tenant,
        workload or trust root are the same indistinguishable 404. The
        handler issues only SELECTs: it writes no audit record, updates
        no snapshot and never parses or returns the CRL body, certificate
        material or key material. ``revoked_count`` is the count recorded
        at registration (entries whose revocationDate had arrived then),
        never recomputed against the current time, while ``entries``
        lists every revoked certificate of the snapshot — including
        entries not yet in effect — ordered by ``(revocation_date,
        entry_id)`` ascending. A storage failure is a sanitized 500.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {"tenant_id", "workload_id", "trust_root_id"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is rejected rather than silently
        # treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )
        if not _UUID_RE.fullmatch(crl_id):
            raise HTTPException(status_code=422, detail="invalid CRL identifier")
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")
        if not trust_root_id.strip() or not _UUID_RE.fullmatch(trust_root_id):
            raise HTTPException(
                status_code=422, detail="invalid trust root identifier"
            )

        try:
            with session_factory() as session:
                crl = session.get(CertificateRevocationList, crl_id)
                if (
                    crl is None
                    or crl.tenant_id != tenant_id
                    or crl.workload_id != workload_id
                    or crl.trust_root_id != trust_root_id
                ):
                    # Unknown and cross-scope snapshots are indistinguishable.
                    raise HTTPException(status_code=404, detail="crl not found")
                entries = session.scalars(
                    select(CrlRevokedCertificate)
                    .where(CrlRevokedCertificate.crl_id == crl.crl_id)
                    .order_by(
                        CrlRevokedCertificate.revocation_date.asc(),
                        CrlRevokedCertificate.entry_id.asc(),
                    )
                ).all()
                payload = {
                    "crl_id": crl.crl_id,
                    "tenant_id": crl.tenant_id,
                    "workload_id": crl.workload_id,
                    "trust_root_id": crl.trust_root_id,
                    "crl_number": crl.crl_number,
                    "crl_sha256": crl.crl_sha256,
                    "this_update": _rfc3339(crl.this_update),
                    "next_update": _rfc3339(crl.next_update),
                    "revoked_count": crl.revoked_count,
                    "created_at": _rfc3339(crl.created_at),
                    "entries": [
                        {
                            "entry_id": entry.entry_id,
                            "issuer_dn": entry.issuer_dn,
                            "serial_number": entry.serial_number,
                            "revocation_date": _rfc3339(entry.revocation_date),
                        }
                        for entry in entries
                    ],
                }
        except HTTPException:
            raise
        except Exception:
            logger.error("CRL lookup failed")
            raise HTTPException(status_code=500, detail="CRL registry unavailable")

        return _compact_json(payload)

    def _register_workload_identity_keyed(
        body: RegisterWorkloadIdentityRequest, idempotency_key: str
    ) -> Response:
        """Register one workload identity profile under an ``Idempotency-Key``.

        The profile, its claims and the idempotency record are one atomic
        commit: the lookup-then-insert runs in a single transaction under
        the trust-root row lock, with the unique (scope, key) constraint
        plus the IntegrityError reread below settling concurrent
        submissions so at most one profile is ever created per key. Every
        judgement failure raises out of the context manager (or an
        explicit rollback), so a failed attempt leaves neither a profile,
        claims nor a record, and the key stays free for a recovered
        retry.
        """
        raw_claims = [
            (claim.issuer, claim.subject, claim.uri) for claim in body.claims
        ]
        # Claims compare as a set: two keyed requests whose claims differ
        # only in order or duplicates carry the same normalized request
        # and replay the first response.
        claims, claims_fingerprint = _canonical_claim_set(raw_claims)
        fingerprint = _workload_identity_request_fingerprint(
            body.tenant_id, body.workload_id, body.trust_root_id, claims_fingerprint
        )

        saved_body: str | None = None
        for _ in range(10):
            with session_factory() as session:
                try:
                    existing = session.scalar(
                        select(WorkloadIdentityIdempotencyRecord).where(
                            WorkloadIdentityIdempotencyRecord.tenant_id
                            == body.tenant_id,
                            WorkloadIdentityIdempotencyRecord.workload_id
                            == body.workload_id,
                            WorkloadIdentityIdempotencyRecord.idempotency_key
                            == idempotency_key,
                        )
                    )
                    if existing is not None:
                        # A replay never creates a profile and never
                        # rewrites claims: the stored first 201 is
                        # returned verbatim, retaining the original
                        # profile_id and created_at. A same-key request
                        # whose trust root or claim set differs is a
                        # stable 409 that changes neither the original
                        # profile nor this record.
                        if not hmac.compare_digest(
                            existing.request_fingerprint, fingerprint
                        ):
                            raise HTTPException(
                                status_code=409,
                                detail="idempotency key conflict",
                            )
                        return Response(
                            content=existing.response_body.encode("utf-8"),
                            status_code=201,
                            media_type="application/json",
                        )

                    # First keyed request for this scope+key. The trust
                    # root must exist in exactly this scope; lock its row
                    # for the whole registration so X.509 verification
                    # either sees the committed profile or settles before
                    # it, exactly as on the unkeyed path.
                    trust_root = session.scalar(
                        select(TrustRoot)
                        .where(
                            TrustRoot.root_id == body.trust_root_id,
                            TrustRoot.tenant_id == body.tenant_id,
                            TrustRoot.workload_id == body.workload_id,
                        )
                        .with_for_update()
                    )
                    if trust_root is None:
                        # Do not reveal whether an out-of-scope trust root
                        # exists; the key is not consumed by the 404.
                        raise HTTPException(
                            status_code=404, detail="trust root not found"
                        )
                    duplicate = session.scalar(
                        select(WorkloadIdentityProfile.profile_id).where(
                            WorkloadIdentityProfile.trust_root_id
                            == body.trust_root_id,
                            WorkloadIdentityProfile.claims_fingerprint
                            == claims_fingerprint,
                        )
                    )
                    if duplicate is not None:
                        # The claim set is already registered under this
                        # trust root (keyed or not): the same 409 as the
                        # unkeyed path, and the key stays free.
                        raise HTTPException(
                            status_code=409,
                            detail="workload identity profile already registered",
                        )
                    profile_id = str(uuid.uuid4())
                    now = _utcnow()
                    session.add(
                        WorkloadIdentityProfile(
                            profile_id=profile_id,
                            tenant_id=body.tenant_id,
                            workload_id=body.workload_id,
                            trust_root_id=body.trust_root_id,
                            claims_fingerprint=claims_fingerprint,
                            status=WORKLOAD_IDENTITY_STATUS_ACTIVE,
                            created_at=now,
                        )
                    )
                    for seq, (issuer, subject, uri) in enumerate(claims):
                        session.add(
                            WorkloadIdentityClaim(
                                claim_id=str(uuid.uuid4()),
                                profile_id=profile_id,
                                tenant_id=body.tenant_id,
                                workload_id=body.workload_id,
                                trust_root_id=body.trust_root_id,
                                issuer=issuer,
                                subject=subject,
                                uri=uri,
                                seq=seq,
                            )
                        )
                    # The exact first 201 body, fixed before commit so the
                    # stored response and the response returned to the
                    # winner are byte-for-byte the same, including the
                    # original profile_id and created_at.
                    saved_body = _workload_identity_created_body(
                        profile_id,
                        body.tenant_id,
                        body.workload_id,
                        body.trust_root_id,
                        claims,
                        now,
                    )
                    session.add(
                        WorkloadIdentityIdempotencyRecord(
                            record_id=str(uuid.uuid4()),
                            tenant_id=body.tenant_id,
                            workload_id=body.workload_id,
                            idempotency_key=idempotency_key,
                            request_fingerprint=fingerprint,
                            response_body=saved_body,
                            created_at=now,
                        )
                    )
                    try:
                        # The profile, its claims and the idempotency
                        # record commit together: a crash can never leave
                        # one without the others.
                        session.commit()
                    except IntegrityError:
                        # A concurrent request committed the same
                        # (scope, key) record or the same
                        # (trust root, claim set) profile first. Reread
                        # and settle as a replay, a key conflict or a
                        # duplicate-set 409 on the next pass.
                        session.rollback()
                        continue
                    break
                except HTTPException:
                    raise
                except Exception:
                    # A storage failure during the lookup or insert is a
                    # server failure: roll back fully, so neither a
                    # profile, claims nor an idempotency record is left
                    # behind, and the key stays free for the same legal
                    # request to retry.
                    session.rollback()
                    logger.error("workload identity registration failed")
                    raise HTTPException(
                        status_code=500,
                        detail="workload identity registration failed",
                    )
        else:
            # Exhausted retries without either a committed insert or a
            # stored record to replay: a storage-level failure that must
            # not look like a successful registration.
            logger.error("workload identity idempotency race did not settle")
            raise HTTPException(
                status_code=500, detail="workload identity registration failed"
            )

        assert saved_body is not None
        return Response(
            content=saved_body.encode("utf-8"),
            status_code=201,
            media_type="application/json",
        )

    @app.post("/v1/workload-identities", status_code=201)
    def register_workload_identity(
        request: Request, body: RegisterWorkloadIdentityRequest
    ) -> Response:
        """Register a workload identity profile under a configured trust root.

        Field and format validation is completed by the request model
        before this handler runs (422 with no state written). The named
        trust root must exist in exactly the request's tenant and
        workload; an unknown or cross-scope root is an indistinguishable
        404. Within one trust root an identical claim set is registered at
        most once (409 on repeat); distinct claim sets are independent
        profiles and never overwrite each other. The profile and its
        claims are a single atomic write under the trust-root row lock, so
        any failure leaves no half profile. The response carries only
        identifiers, the scope, the non-sensitive comparison strings and
        a timestamp — never certificate material or exception detail.

        An optional ``Idempotency-Key`` header makes the registration
        replayable: the first successful keyed registration stores its
        201 body with the profile and claims in the same transaction, and
        a later same-key replay of the same normalized request returns
        those exact bytes without creating anything. A missing header
        preserves the original semantics exactly.
        """
        # The idempotency key is optional and lives only in a header; the
        # body contract is unchanged. An illegal key (duplicated header,
        # empty value, whitespace padding, control/non-ASCII character,
        # or over-long value) is a 422 before any state is read or
        # written.
        idem_present, idempotency_key = _read_idempotency_key(request)
        if idem_present:
            return _register_workload_identity_keyed(body, idempotency_key)

        raw_claims = [
            (claim.issuer, claim.subject, claim.uri) for claim in body.claims
        ]
        # Claims compare as a set: order and duplicate entries do not make
        # a distinct profile.
        claims, claims_fingerprint = _canonical_claim_set(raw_claims)
        profile_id = str(uuid.uuid4())
        now = _utcnow()
        with session_factory() as session:
            try:
                # Lock the trust-root row for the full registration. X.509
                # verification takes the same lock while reading the
                # anchor's profiles, so a profile either commits before an
                # evidence settles (and that verification may match it) or
                # waits until after it settles (and never applies
                # retroactively). SQLite ignores FOR UPDATE but already
                # serializes all writers via BEGIN IMMEDIATE.
                trust_root = session.scalar(
                    select(TrustRoot)
                    .where(
                        TrustRoot.root_id == body.trust_root_id,
                        TrustRoot.tenant_id == body.tenant_id,
                        TrustRoot.workload_id == body.workload_id,
                    )
                    .with_for_update()
                )
                if trust_root is None:
                    # Do not reveal whether an out-of-scope trust root exists.
                    raise HTTPException(status_code=404, detail="trust root not found")
                existing = session.scalar(
                    select(WorkloadIdentityProfile.profile_id).where(
                        WorkloadIdentityProfile.trust_root_id == body.trust_root_id,
                        WorkloadIdentityProfile.claims_fingerprint
                        == claims_fingerprint,
                    )
                )
                if existing is not None:
                    # An identical claim set already exists for this trust
                    # root; distinct sets stay independent and are never
                    # merged or overwritten.
                    raise HTTPException(
                        status_code=409,
                        detail="workload identity profile already registered",
                    )
                session.add(
                    WorkloadIdentityProfile(
                        profile_id=profile_id,
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        trust_root_id=body.trust_root_id,
                        claims_fingerprint=claims_fingerprint,
                        status=WORKLOAD_IDENTITY_STATUS_ACTIVE,
                        created_at=now,
                    )
                )
                for seq, (issuer, subject, uri) in enumerate(claims):
                    session.add(
                        WorkloadIdentityClaim(
                            claim_id=str(uuid.uuid4()),
                            profile_id=profile_id,
                            tenant_id=body.tenant_id,
                            workload_id=body.workload_id,
                            trust_root_id=body.trust_root_id,
                            issuer=issuer,
                            subject=subject,
                            uri=uri,
                            seq=seq,
                        )
                    )
                try:
                    session.commit()
                except IntegrityError:
                    # A concurrent request registered the identical claim
                    # set first; the unique constraint guarantees at most
                    # one profile per set and the loser reports 409.
                    session.rollback()
                    raise HTTPException(
                        status_code=409,
                        detail="workload identity profile already registered",
                    )
                except Exception:
                    # Any other commit/write failure rolls the whole
                    # (profile + claims) transaction back, so no half
                    # profile survives.
                    session.rollback()
                    logger.error("workload identity profile write failed")
                    raise HTTPException(
                        status_code=500,
                        detail="workload identity registration failed",
                    )
            except HTTPException:
                # 404/409 judgements and the controlled 500 above keep
                # their status; a read-only judgement has written nothing.
                raise
            except Exception:
                # A failure during the locked lookups is likewise a full
                # rollback and a sanitized 500: no claim row can exist.
                session.rollback()
                logger.error("workload identity registration failed")
                raise HTTPException(
                    status_code=500, detail="workload identity registration failed"
                )
        body_bytes = _workload_identity_created_body(
            profile_id,
            body.tenant_id,
            body.workload_id,
            body.trust_root_id,
            claims,
            now,
        ).encode("utf-8")
        return Response(
            content=body_bytes, status_code=201, media_type="application/json"
        )

    @app.get("/v1/workload-identities")
    def list_workload_identities(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        trust_root_id: str = Query(...),
        profile_id: str | None = Query(default=None),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return the workload identity profiles in one scope.

        The request body is always empty (enforced by the dependency
        above) and the scope comes entirely from query parameters:
        mandatory non-blank tenant, workload and trust root (a canonical
        UUID), plus an optional canonical profile UUID. Results are
        sorted by creation time and then profile id; an empty range is
        an empty array. An explicitly named profile that is unknown or
        outside the scope is a 404 rather than an empty-looking array.
        The handler issues only reads and observes committed state.
        """
        allowed_params = {"tenant_id", "workload_id", "trust_root_id", "profile_id"}
        if set(request.query_params.keys()) - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")
        if not trust_root_id.strip() or not _UUID_RE.fullmatch(trust_root_id):
            raise HTTPException(status_code=422, detail="invalid trust root identifier")
        profile_filter: str | None = None
        if profile_id is not None:
            if not profile_id.strip() or not _UUID_RE.fullmatch(profile_id):
                raise HTTPException(
                    status_code=422, detail="invalid profile identifier"
                )
            profile_filter = profile_id

        try:
            with session_factory() as session:
                # An explicitly named profile must match the scope and
                # trust root; an unknown or cross-scope profile is a 404.
                # An unknown trust root with no profile filter simply
                # matches an empty range.
                if profile_filter is not None:
                    named = session.get(WorkloadIdentityProfile, profile_filter)
                    if (
                        named is None
                        or named.tenant_id != tenant_id
                        or named.workload_id != workload_id
                        or named.trust_root_id != trust_root_id
                    ):
                        raise HTTPException(
                            status_code=404, detail="workload identity profile not found"
                        )

                stmt = select(WorkloadIdentityProfile).where(
                    WorkloadIdentityProfile.tenant_id == tenant_id,
                    WorkloadIdentityProfile.workload_id == workload_id,
                    WorkloadIdentityProfile.trust_root_id == trust_root_id,
                )
                if profile_filter is not None:
                    stmt = stmt.where(
                        WorkloadIdentityProfile.profile_id == profile_filter
                    )
                stmt = stmt.order_by(
                    WorkloadIdentityProfile.created_at.asc(),
                    WorkloadIdentityProfile.profile_id.asc(),
                )
                profiles = list(session.scalars(stmt))
                claims_by_profile: dict[str, list[WorkloadIdentityClaim]] = {}
                if profiles:
                    claim_rows = session.scalars(
                        select(WorkloadIdentityClaim).where(
                            WorkloadIdentityClaim.profile_id.in_(
                                [profile.profile_id for profile in profiles]
                            )
                        )
                    ).all()
                    for claim in claim_rows:
                        claims_by_profile.setdefault(claim.profile_id, []).append(claim)
        except HTTPException:
            raise
        except Exception:
            logger.error("workload identity profile query failed")
            raise HTTPException(
                status_code=500, detail="workload identity registry unavailable"
            )

        # Query elements keep the creation-response fields and add the
        # lifecycle status; revoked profiles remain listed here even
        # though they no longer gate X.509 verification.
        result = []
        for profile in profiles:
            claims = sorted(
                claims_by_profile.get(profile.profile_id, []),
                key=lambda claim: claim.seq,
            )
            result.append(
                {
                    "profile_id": profile.profile_id,
                    "tenant_id": profile.tenant_id,
                    "workload_id": profile.workload_id,
                    "trust_root_id": profile.trust_root_id,
                    "claims": [
                        {
                            "issuer": claim.issuer,
                            "subject": claim.subject,
                            "uri": claim.uri,
                        }
                        for claim in claims
                    ],
                    "status": profile.status,
                    "created_at": _rfc3339(profile.created_at),
                }
            )
        return _identity_json(result)

    @app.put("/v1/workload-identities/")
    def update_workload_identity_identifier_required() -> Response:
        # An empty path segment is a missing profile identifier: a 422
        # client error rather than a routing-level 404 or 405.
        raise HTTPException(status_code=422, detail="invalid profile identifier")

    @app.put("/v1/workload-identities/{profile_id}")
    def update_workload_identity(
        profile_id: str, body: UpdateWorkloadIdentityRequest
    ) -> Response:
        """Replace a profile's whole claim set (PUT semantics).

        The path identifier must be a canonical UUID and the body has the
        same shape and strictness as registration (scope, trust root and
        a complete, non-empty claim list); any malformed input is a 422
        that writes nothing. The named profile must exist in exactly the
        body's scope and trust root (unknown/cross-scope is 404). A
        request carrying the profile's own current set is an idempotent
        no-op and returns the current result; a set already owned by a
        different profile under the same trust root is a 409 and changes
        nothing. Two updates raced from the same original set settle
        once: the loser's guarded compare-and-swap matches no row and it
        observes the winner — 200 with the current result when both
        requested the same set, 409 for a different set — so only one
        replacement ever takes effect. The profile id, ownership/scope
        and ``created_at`` never change; the replacement (profile row
        plus claim rows) is one atomic commit, so any failure leaves the
        old set, status and timestamps exactly as they were.
        """
        # A path identifier that is missing, blank, whitespace-padded or
        # not a canonical lowercase UUID is a field/format error, never a
        # lookup. The raw value must match exactly — canonical form is
        # never derived by trimming surrounding whitespace.
        if not _UUID_RE.fullmatch(profile_id):
            raise HTTPException(status_code=422, detail="invalid profile identifier")

        raw_claims = [
            (claim.issuer, claim.subject, claim.uri) for claim in body.claims
        ]
        claims, claims_fingerprint = _canonical_claim_set(raw_claims)

        # Capture the set the request is based on *before* opening the
        # write transaction. Two concurrent requests both observe the
        # same original fingerprint; only one compare-and-swap below can
        # then match. Profiles and trust roots are never deleted and
        # profile scope never changes, so this existence/scope judgement
        # stays valid for the write that follows.
        with session_factory() as read_session:
            original = read_session.get(WorkloadIdentityProfile, profile_id)
            if (
                original is None
                or original.tenant_id != body.tenant_id
                or original.workload_id != body.workload_id
                or original.trust_root_id != body.trust_root_id
            ):
                raise HTTPException(
                    status_code=404, detail="workload identity profile not found"
                )
            if original.status == WORKLOAD_IDENTITY_STATUS_REVOKED:
                # A revoked profile is terminal: its last committed claim
                # set is retained for queries but may never be replaced,
                # even with an identical set. Judged after the 404 scope
                # check so an out-of-scope revoked profile stays
                # indistinguishable from an unknown one.
                raise HTTPException(
                    status_code=409,
                    detail="workload identity profile already revoked",
                )
            original_fingerprint = original.claims_fingerprint

        updated_at = _utcnow()
        with session_factory() as session:
            try:
                # Take the same trust-root lock registration and X.509
                # verification take, so an update is ordered against both:
                # only an update committed before an evidence settles can
                # affect it. SQLite serializes all writers via BEGIN
                # IMMEDIATE regardless.
                trust_root = session.scalar(
                    select(TrustRoot)
                    .where(
                        TrustRoot.root_id == body.trust_root_id,
                        TrustRoot.tenant_id == body.tenant_id,
                        TrustRoot.workload_id == body.workload_id,
                    )
                    .with_for_update()
                )
                profile = session.get(WorkloadIdentityProfile, profile_id)
                if trust_root is None or profile is None or (
                    profile.tenant_id != body.tenant_id
                    or profile.workload_id != body.workload_id
                    or profile.trust_root_id != body.trust_root_id
                ):
                    # Do not reveal whether an out-of-scope profile or
                    # trust root exists.
                    raise HTTPException(
                        status_code=404, detail="workload identity profile not found"
                    )
                if profile.status == WORKLOAD_IDENTITY_STATUS_REVOKED:
                    # Re-check under the write lock: a concurrent revoke
                    # may have committed after the scope pre-read. A
                    # revoked profile is terminal and never replaced.
                    raise HTTPException(
                        status_code=409,
                        detail="workload identity profile already revoked",
                    )

                def _current_result(current: WorkloadIdentityProfile) -> Response:
                    current_claims = session.scalars(
                        select(WorkloadIdentityClaim)
                        .where(WorkloadIdentityClaim.profile_id == profile_id)
                    ).all()
                    result = {
                        "profile_id": current.profile_id,
                        "tenant_id": current.tenant_id,
                        "workload_id": current.workload_id,
                        "trust_root_id": current.trust_root_id,
                        "claims": [
                            {
                                "issuer": claim.issuer,
                                "subject": claim.subject,
                                "uri": claim.uri,
                            }
                            for claim in sorted(current_claims, key=lambda c: c.seq)
                        ],
                        "created_at": _rfc3339(current.created_at),
                    }
                    # A no-op never mints a timestamp: updated_at is
                    # present only when an earlier replacement set it.
                    if current.updated_at is not None:
                        result["updated_at"] = _rfc3339(current.updated_at)
                    return _identity_json(result)

                if profile.claims_fingerprint == claims_fingerprint:
                    # Idempotent self-replacement: the current result is
                    # returned verbatim and nothing is written, so
                    # updated_at is never rewritten by a no-op.
                    return _current_result(profile)

                # A different profile under the same trust root (active or
                # revoked) already owns this exact set; sets never merge.
                owner = session.scalar(
                    select(WorkloadIdentityProfile.profile_id).where(
                        WorkloadIdentityProfile.trust_root_id == body.trust_root_id,
                        WorkloadIdentityProfile.claims_fingerprint
                        == claims_fingerprint,
                        WorkloadIdentityProfile.profile_id != profile_id,
                    )
                )
                if owner is not None:
                    raise HTTPException(
                        status_code=409,
                        detail="workload identity profile already registered",
                    )

                # Compare-and-swap on the fingerprint captured before this
                # transaction: the guarded UPDATE plus the delete/insert
                # of claims is the single atomic replacement. A request
                # whose original set has already been replaced by a
                # concurrent winner matches no row.
                outcome = session.execute(
                    update(WorkloadIdentityProfile)
                    .where(
                        WorkloadIdentityProfile.profile_id == profile_id,
                        WorkloadIdentityProfile.claims_fingerprint
                        == original_fingerprint,
                        # Belt and braces alongside the status check above:
                        # a revoked profile may never be replaced even if a
                        # revoke races this transaction.
                        WorkloadIdentityProfile.status
                        == WORKLOAD_IDENTITY_STATUS_ACTIVE,
                    )
                    .values(
                        claims_fingerprint=claims_fingerprint,
                        updated_at=updated_at,
                    )
                    .execution_options(synchronize_session=False)
                )

                def _settlement_conflict() -> HTTPException:
                    # Re-read the winner outside the failed transaction:
                    # a request whose requested set has just been
                    # installed receives that current result; any other
                    # stale write is a conflict and changes nothing.
                    session.rollback()
                    winner = session.get(WorkloadIdentityProfile, profile_id)
                    if (
                        winner is not None
                        and winner.status == WORKLOAD_IDENTITY_STATUS_ACTIVE
                        and winner.claims_fingerprint == claims_fingerprint
                    ):
                        return _current_result(winner)
                    return HTTPException(
                        status_code=409,
                        detail="workload identity profile was modified concurrently",
                    )

                if outcome.rowcount != 1:
                    conflict = _settlement_conflict()
                    if isinstance(conflict, Response):
                        return conflict
                    raise conflict
                session.execute(
                    delete(WorkloadIdentityClaim).where(
                        WorkloadIdentityClaim.profile_id == profile_id
                    )
                )
                for seq, (issuer, subject, uri) in enumerate(claims):
                    session.add(
                        WorkloadIdentityClaim(
                            claim_id=str(uuid.uuid4()),
                            profile_id=profile_id,
                            tenant_id=body.tenant_id,
                            workload_id=body.workload_id,
                            trust_root_id=body.trust_root_id,
                            issuer=issuer,
                            subject=subject,
                            uri=uri,
                            seq=seq,
                        )
                    )
                try:
                    session.commit()
                except IntegrityError:
                    # A concurrent request installed the same (or a
                    # colliding) set first; the unique constraint makes
                    # at most one owner. The winner's current set may
                    # already satisfy this request.
                    conflict = _settlement_conflict()
                    if isinstance(conflict, Response):
                        return conflict
                    raise conflict
                except Exception:
                    session.rollback()
                    logger.error("workload identity profile update failed")
                    raise HTTPException(
                        status_code=500, detail="workload identity update failed"
                    )
            except HTTPException:
                raise
            except Exception:
                session.rollback()
                logger.error("workload identity profile update failed")
                raise HTTPException(
                    status_code=500, detail="workload identity update failed"
                )
        payload = {
            "profile_id": profile_id,
            "tenant_id": body.tenant_id,
            "workload_id": body.workload_id,
            "trust_root_id": body.trust_root_id,
            "claims": [
                {"issuer": issuer, "subject": subject, "uri": uri}
                for issuer, subject, uri in claims
            ],
            "created_at": _rfc3339(profile.created_at),
            "updated_at": _rfc3339(updated_at),
        }
        return _identity_json(payload)

    @app.delete("/v1/workload-identities/")
    def revoke_workload_identity_identifier_required() -> Response:
        # An empty path segment is a missing profile identifier: a 422
        # client error rather than a routing-level 404 or 405.
        raise HTTPException(status_code=422, detail="invalid profile identifier")

    @app.delete("/v1/workload-identities/{profile_id}")
    def revoke_workload_identity(
        profile_id: str, body: RevokeWorkloadIdentityRequest
    ) -> Response:
        """Revoke a workload identity profile.

        The path identifier must be a canonical UUID and the body carries
        only the scope and trust root; any missing, blank, wrong-typed or
        unknown field is a 422 that writes nothing. An unknown profile or
        one outside the body's scope/trust root is an indistinguishable
        404. A repeat revoke returns 409 and never rewrites the stored
        ``revoked_at``. The active -> revoked transition is one guarded
        atomic update under the trust-root lock, so it is ordered against
        registration, update and X.509 verification; a write or commit
        failure returns 500 after a full rollback, leaving the profile,
        its status and its revocation time unchanged.
        """
        # A path identifier that is missing, blank, whitespace-padded or
        # not a canonical lowercase UUID is a field/format error; the raw
        # value must match exactly.
        if not _UUID_RE.fullmatch(profile_id):
            raise HTTPException(status_code=422, detail="invalid profile identifier")

        revoked_at = _utcnow()
        with session_factory() as session:
            try:
                trust_root = session.scalar(
                    select(TrustRoot)
                    .where(
                        TrustRoot.root_id == body.trust_root_id,
                        TrustRoot.tenant_id == body.tenant_id,
                        TrustRoot.workload_id == body.workload_id,
                    )
                    .with_for_update()
                )
                profile = session.get(WorkloadIdentityProfile, profile_id)
                if trust_root is None or profile is None or (
                    profile.tenant_id != body.tenant_id
                    or profile.workload_id != body.workload_id
                    or profile.trust_root_id != body.trust_root_id
                ):
                    raise HTTPException(
                        status_code=404, detail="workload identity profile not found"
                    )
                if profile.status == WORKLOAD_IDENTITY_STATUS_REVOKED:
                    # A repeated revoke is a conflict and never rewrites
                    # the recorded revocation time.
                    raise HTTPException(
                        status_code=409, detail="workload identity profile already revoked"
                    )
                # Atomic settlement: only one caller can flip
                # active -> revoked.
                outcome = session.execute(
                    update(WorkloadIdentityProfile)
                    .where(
                        WorkloadIdentityProfile.profile_id == profile_id,
                        WorkloadIdentityProfile.status
                        == WORKLOAD_IDENTITY_STATUS_ACTIVE,
                    )
                    .values(
                        status=WORKLOAD_IDENTITY_STATUS_REVOKED,
                        revoked_at=revoked_at,
                    )
                    .execution_options(synchronize_session=False)
                )
                if outcome.rowcount != 1:
                    session.rollback()
                    fresh = session.get(WorkloadIdentityProfile, profile_id)
                    if (
                        fresh is not None
                        and fresh.status == WORKLOAD_IDENTITY_STATUS_REVOKED
                    ):
                        raise HTTPException(
                            status_code=409,
                            detail="workload identity profile already revoked",
                        )
                    raise HTTPException(
                        status_code=409,
                        detail="workload identity profile revocation conflict",
                    )
                try:
                    session.commit()
                except Exception:
                    session.rollback()
                    logger.error("workload identity profile revocation failed")
                    raise HTTPException(
                        status_code=500, detail="workload identity revocation failed"
                    )
                final_claims = session.scalars(
                    select(WorkloadIdentityClaim).where(
                        WorkloadIdentityClaim.profile_id == profile_id
                    )
                ).all()
                created_at = profile.created_at
                prior_updated_at = profile.updated_at
            except HTTPException:
                raise
            except Exception:
                session.rollback()
                logger.error("workload identity profile revocation failed")
                raise HTTPException(
                    status_code=500, detail="workload identity revocation failed"
                )
        payload = {
            "profile_id": profile_id,
            "tenant_id": body.tenant_id,
            "workload_id": body.workload_id,
            "trust_root_id": body.trust_root_id,
            "claims": [
                {"issuer": claim.issuer, "subject": claim.subject, "uri": claim.uri}
                for claim in sorted(final_claims, key=lambda c: c.seq)
            ],
            "created_at": _rfc3339(created_at),
            "revoked": True,
            "revoked_at": _rfc3339(revoked_at),
        }
        # A replacement that happened before revocation stays visible.
        if prior_updated_at is not None:
            payload["updated_at"] = _rfc3339(prior_updated_at)
        return _identity_json(payload)

    @app.post("/v1/policies", status_code=201, response_model=PolicyCreatedResponse)
    def create_policy(
        request: Request, body: CreatePolicyRequest
    ) -> PolicyCreatedResponse | Response:
        # The idempotency key is optional and lives only in a header; the
        # body contract is unchanged. A missing key preserves the
        # original one-new-version-per-valid-request semantics exactly.
        # An illegal key (duplicated header, empty value, whitespace
        # padding, control/non-ASCII character, or over-long value) is a
        # 422 before any state is read or written, so it never creates a
        # policy, allocates a version or advances a lifecycle sequence.
        idem_present, idempotency_key = _read_idempotency_key(request)

        if idem_present:
            return _create_policy_keyed(body, idempotency_key)

        policy_id = str(uuid.uuid4())
        now = _utcnow()
        rule_json = canonical_rule_json(body.rule)
        # Allocate the next version inside a write transaction. On SQLite
        # every write transaction begins as BEGIN IMMEDIATE, so competing
        # creators serialize; on locking backends the unique
        # (scope, name, version) constraint plus this retry loop guarantees
        # no two versions ever share a number and no version is skipped.
        with session_factory() as session:
            try:
                for _ in range(10):
                    highest = session.scalar(
                        select(func.max(Policy.version)).where(
                            Policy.tenant_id == body.tenant_id,
                            Policy.workload_id == body.workload_id,
                            Policy.name == body.name,
                        )
                    )
                    version = (highest or 0) + 1
                    # Sequence the creation from the per-scope lifecycle
                    # counter shared with retirement, in the same write
                    # transaction. The read-only lifecycle query bounds a
                    # replayable snapshot by this commit high-water mark,
                    # immune to both later inserts and later in-place
                    # retirements; an IntegrityError below rolls the
                    # allocation back together with the row.
                    commit_seq = _next_policy_commit_seq(
                        session, body.tenant_id, body.workload_id
                    )
                    session.add(
                        Policy(
                            policy_id=policy_id,
                            tenant_id=body.tenant_id,
                            workload_id=body.workload_id,
                            name=body.name,
                            version=version,
                            rule_json=rule_json,
                            created_at=now,
                            commit_seq=commit_seq,
                        )
                    )
                    try:
                        session.commit()
                    except IntegrityError:
                        # A concurrent transaction claimed the same version
                        # first; re-read the high-water mark and retry.
                        session.rollback()
                        continue
                    break
                else:
                    raise HTTPException(
                        status_code=500, detail="policy write failed"
                    )
            except HTTPException:
                raise
            except Exception:
                session.rollback()
                logger.error("policy write failed")
                raise HTTPException(
                    status_code=500, detail="policy write failed"
                )
        return PolicyCreatedResponse(
            policy_id=policy_id,
            tenant_id=body.tenant_id,
            workload_id=body.workload_id,
            name=body.name,
            version=version,
            rule=body.rule,
            created_at=_rfc3339(now),
        )

    def _create_policy_keyed(
        body: CreatePolicyRequest, idempotency_key: str
    ) -> Response:
        """Create one policy version under an ``Idempotency-Key``.

        The policy row, its lifecycle commit sequence and the
        idempotency record are one atomic commit: the lookup-then-insert
        runs in a single transaction, with the unique (scope, key)
        constraint plus the IntegrityError reread below settling
        concurrent submissions so at most one version is ever created
        per key. Every judgement failure raises out of the context
        manager (or an explicit rollback), so a failed attempt leaves
        neither a policy nor a record, consumes no version number or
        commit sequence, and the key stays free for a recovered retry.
        """
        # The rule is already validated by the request model; the
        # canonical serialization is the existing normalization under
        # which two requests must carry the same rule tree to count as a
        # replay.
        rule_json = canonical_rule_json(body.rule)
        fingerprint = _policy_request_fingerprint(
            body.tenant_id, body.workload_id, body.name, rule_json
        )

        saved_body: str | None = None
        for _ in range(10):
            with session_factory() as session:
                try:
                    existing = session.scalar(
                        select(PolicyIdempotencyRecord).where(
                            PolicyIdempotencyRecord.tenant_id == body.tenant_id,
                            PolicyIdempotencyRecord.workload_id == body.workload_id,
                            PolicyIdempotencyRecord.idempotency_key
                            == idempotency_key,
                        )
                    )
                    if existing is not None:
                        # A replay never creates a version, never
                        # recomputes a version number or lifecycle commit
                        # sequence and never changes the existing policy:
                        # the stored first 201 is returned verbatim,
                        # retaining the original policy_id, version and
                        # created_at. A same-key request whose name or
                        # normalized rule tree differs is a stable 409
                        # that changes neither the original policy nor
                        # this record.
                        if not hmac.compare_digest(
                            existing.request_fingerprint, fingerprint
                        ):
                            raise HTTPException(
                                status_code=409,
                                detail="idempotency key reused with a different request",
                            )
                        return Response(
                            content=existing.response_body.encode("utf-8"),
                            status_code=201,
                            media_type="application/json",
                        )

                    # First keyed request for this scope+key. Allocate
                    # the next version, its lifecycle sequence, and the
                    # record together; an IntegrityError on either the
                    # version or the record rolls all three back and
                    # rereads below.
                    now = _utcnow()
                    policy_id = str(uuid.uuid4())
                    highest = session.scalar(
                        select(func.max(Policy.version)).where(
                            Policy.tenant_id == body.tenant_id,
                            Policy.workload_id == body.workload_id,
                            Policy.name == body.name,
                        )
                    )
                    version = (highest or 0) + 1
                    commit_seq = _next_policy_commit_seq(
                        session, body.tenant_id, body.workload_id
                    )
                    session.add(
                        Policy(
                            policy_id=policy_id,
                            tenant_id=body.tenant_id,
                            workload_id=body.workload_id,
                            name=body.name,
                            version=version,
                            rule_json=rule_json,
                            created_at=now,
                            commit_seq=commit_seq,
                        )
                    )
                    # The exact first 201 body, fixed before commit so
                    # the stored response and the response returned to
                    # the winner are byte-for-byte the same, including
                    # the original policy_id, version and created_at.
                    saved_body = _policy_created_body(
                        policy_id,
                        body.tenant_id,
                        body.workload_id,
                        body.name,
                        version,
                        body.rule,
                        now,
                    )
                    session.add(
                        PolicyIdempotencyRecord(
                            record_id=str(uuid.uuid4()),
                            tenant_id=body.tenant_id,
                            workload_id=body.workload_id,
                            idempotency_key=idempotency_key,
                            request_fingerprint=fingerprint,
                            response_body=saved_body,
                            created_at=now,
                        )
                    )
                    try:
                        # The policy, the lifecycle sequence and the
                        # idempotency record commit together: a crash
                        # can never leave one without the others.
                        session.commit()
                    except IntegrityError:
                        # A concurrent request claimed the same
                        # (scope, name, version) or committed the same
                        # (scope, key) first. Reread and settle as a
                        # replay or a key-reuse conflict on the next
                        # pass; the rolled-back version allocation and
                        # commit sequence are not consumed.
                        session.rollback()
                        continue
                    break
                except HTTPException:
                    raise
                except Exception:
                    # A storage failure during the lookup or insert is a
                    # server failure: roll back fully, so neither a
                    # policy version, a lifecycle sequence nor an
                    # idempotency record is left behind, and the key
                    # stays free for the same legal request to retry.
                    session.rollback()
                    logger.error("policy write failed")
                    raise HTTPException(
                        status_code=500, detail="policy write failed"
                    )
        else:
            # Exhausted retries without either a committed insert or a
            # stored record to replay: a storage-level failure that must
            # not look like a successful creation.
            logger.error("policy idempotency race did not settle")
            raise HTTPException(status_code=500, detail="policy write failed")

        assert saved_body is not None
        return Response(
            content=saved_body.encode("utf-8"),
            status_code=201,
            media_type="application/json",
        )

    @app.post("/v1/policies//retire")
    def retire_policy_identifier_required(
        body: RetirePolicyRequest,
    ) -> Response:
        # An empty path segment is a missing policy identifier: a 422
        # client error rather than a routing-level 404 or 405. It never
        # reads or modifies a policy version.
        raise HTTPException(status_code=422, detail="invalid policy identifier")

    @app.post("/v1/policies/{policy_id}/retire")
    def retire_policy(policy_id: str, body: RetirePolicyRequest) -> Response:
        """Retire one concrete versioned policy.

        The path identifier must be a canonical UUID and the body carries
        only the two non-blank scope strings; any missing, blank,
        wrong-typed, incomplete or unknown field is a 422 that never reads
        or writes the policy. A path UUID is checked before storage is
        touched. An unknown policy or one outside the body's
        tenant/workload is an indistinguishable 404 (existence is never
        revealed) — the path id and the body scope name exactly one
        version. A repeat retire returns 409 and never rewrites the
        recorded ``retired_at`` and appends no audit or other state
        record. The active -> retired transition is one guarded atomic
        update under the policy row lock, so concurrent retires of one
        version settle as exactly one success and stable 409s; distinct
        versions (including other versions of the same name) hold
        independent locks and are isolated. A write or commit failure
        returns 500 after a full rollback, leaving the original status and
        time unchanged, and the same request succeeds after recovery.
        Retirement only terminates *new* decisions against this version:
        existing decisions, their policy_version/decided_at and every
        release grant are untouched, and the version cannot be deleted or
        overwritten into bypassing the terminal state.
        """
        # A path identifier that is missing (empty segment), blank,
        # whitespace-padded or not a canonical lowercase UUID is a
        # field/format error; the raw value must match exactly. This runs
        # before any storage access, so a malformed path can neither read
        # nor modify a policy version.
        if not _UUID_RE.fullmatch(policy_id):
            raise HTTPException(status_code=422, detail="invalid policy identifier")

        retired_at = _utcnow()
        with session_factory() as session:
            try:
                # Lock the policy row for the whole transition so two
                # concurrent retires serialize on this exact version; the
                # decision path reads the same row under lock and is
                # therefore strictly ordered against this commit. SQLite
                # ignores FOR UPDATE but already serializes all writers via
                # BEGIN IMMEDIATE.
                policy = session.scalar(
                    select(Policy)
                    .where(
                        Policy.policy_id == policy_id,
                        Policy.tenant_id == body.tenant_id,
                        Policy.workload_id == body.workload_id,
                    )
                    .with_for_update()
                )
                if policy is None:
                    # Do not reveal whether an out-of-scope or unknown
                    # policy exists: unknown id and scope mismatch share
                    # one indistinguishable 404.
                    raise HTTPException(status_code=404, detail="policy not found")
                if policy.status == POLICY_STATUS_RETIRED:
                    # A repeated retire is a conflict. It neither rewrites
                    # the recorded retirement time nor appends any audit or
                    # other state record.
                    raise HTTPException(
                        status_code=409, detail="policy already retired"
                    )
                # Sequence the retirement from the same per-scope counter
                # version creation draws from, before the guarded flip. The
                # policy row is already locked, so no concurrent transition
                # can interleave: the lifecycle query reconstructs this
                # version's as-of-snapshot status by comparing retired_seq
                # against its fixed high-water mark.
                retired_seq = _next_policy_commit_seq(
                    session, body.tenant_id, body.workload_id
                )
                # Atomic settlement: only one caller can flip
                # active -> retired, and retired_at/retired_seq are written
                # by that same single-row update.
                outcome = session.execute(
                    update(Policy)
                    .where(
                        Policy.policy_id == policy_id,
                        Policy.status == POLICY_STATUS_ACTIVE,
                    )
                    .values(
                        status=POLICY_STATUS_RETIRED,
                        retired_at=retired_at,
                        retired_seq=retired_seq,
                    )
                    .execution_options(synchronize_session=False)
                )
                if outcome.rowcount != 1:
                    session.rollback()
                    fresh = session.get(Policy, policy_id)
                    if fresh is not None and fresh.status == POLICY_STATUS_RETIRED:
                        raise HTTPException(
                            status_code=409, detail="policy already retired"
                        )
                    raise HTTPException(
                        status_code=409, detail="policy retirement conflict"
                    )
                try:
                    session.commit()
                except Exception:
                    session.rollback()
                    logger.error("policy retirement failed")
                    raise HTTPException(
                        status_code=500, detail="policy retirement failed"
                    )
            except HTTPException:
                # 404/409 judgements and the controlled 500 above keep
                # their status; a read-only judgement has written nothing.
                raise
            except Exception:
                # A failure during the locked lookup is a full rollback and
                # a sanitized 500: the status update can never have happened
                # on this path, so the version stays active with no time set.
                session.rollback()
                logger.error("policy retirement failed")
                raise HTTPException(
                    status_code=500, detail="policy retirement failed"
                )
        # Compact JSON describing only the retirement result. The keys are
        # emitted in the fixed order policy_id, status, retired_at; every
        # value is a string (allow_nan=False makes non-finite numbers
        # impossible), terminated by exactly one newline. No rule, claim,
        # evidence, nonce, capability, payload, key or exception text is
        # ever present.
        body_bytes = json.dumps(
            {
                "policy_id": policy_id,
                "status": POLICY_STATUS_RETIRED,
                "retired_at": _rfc3339(retired_at),
            },
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return Response(
            content=body_bytes + b"\n",
            status_code=200,
            media_type="application/json",
        )

    @app.get("/v1/policies")
    def list_policies(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        policy_id: str | None = Query(default=None),
        name: str | None = Query(default=None),
        status: str | None = Query(default=None),
        created_after: str | None = Query(default=None),
        created_before: str | None = Query(default=None),
        cursor: str | None = Query(default=None),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return a read-only, cursor-stable page of policy versions.

        The range is fixed by the mandatory tenant and workload and may be
        narrowed by an explicit policy id (a canonical lowercase UUID), an
        exact name, a lifecycle status (``active``/``retired``) and an
        inclusive creation-time window. Ordering is stable
        ``(created_at, policy_id)`` ascending with an exclusive keyset
        cursor. The cursor carries its own kind tag, is HMAC-authenticated
        and is bound to the scope, every active filter *and* the fixed
        snapshot established by the range's first (cursor-less or explicit
        empty-cursor) query, so it can be neither forged nor replayed
        against a different scope, filter set, snapshot or query family.

        The snapshot is fixed at the per-scope lifecycle commit boundary:
        version creation and retirement share one gap-free counter, and a
        version's status as-of the snapshot is reconstructed from the
        sequence its retirement committed at. Replaying a cursor therefore
        returns the identical page even while versions are concurrently
        retired or created; those changes surface only in a fresh first
        query. The handler issues only SELECTs — it never creates, retires
        or otherwise mutates a version, writes no audit record and returns
        no evidence, claim value, capability, key or exception text — and a
        storage failure aborts the whole request with a 500 rather than
        returning a half page.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {
            "tenant_id",
            "workload_id",
            "policy_id",
            "name",
            "status",
            "created_after",
            "created_before",
            "cursor",
        }
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is rejected rather than silently
        # treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )

        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        policy_filter: str | None = None
        if policy_id is not None:
            if not policy_id.strip() or not _UUID_RE.fullmatch(policy_id):
                raise HTTPException(
                    status_code=422, detail="invalid policy identifier"
                )
            policy_filter = policy_id

        # Exact-text name filter. Only a blank/whitespace value is a shape
        # error; a non-blank value is matched verbatim (names may contain
        # interior or surrounding visible characters) and a name that
        # matches nothing is an empty range, never a 404.
        name_filter: str | None = None
        if name is not None:
            if not name.strip():
                raise HTTPException(status_code=422, detail="invalid name")
            name_filter = name

        if status is not None:
            if not status.strip() or status not in (
                POLICY_STATUS_ACTIVE,
                POLICY_STATUS_RETIRED,
            ):
                raise HTTPException(status_code=422, detail="invalid status")
        status_filter = status if status is not None else ""

        def _time_bound(value: str | None, name: str) -> tuple[str, datetime | None]:
            if value is None:
                return "", None
            if not value.strip():
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            try:
                parsed = _parse_utc_rfc3339(value)
            except ValueError:
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            # Normalize the spelling embedded into the cursor so two
            # equivalent UTC spellings cannot mint different cursor domains.
            return _rfc3339(parsed), parsed

        after_raw, after_dt = _time_bound(created_after, "created_after")
        before_raw, before_dt = _time_bound(created_before, "created_before")
        # The window is closed on both ends; equality is a valid
        # single-instant window and the start must not follow the end.
        if after_dt is not None and before_dt is not None and after_dt > before_dt:
            raise HTTPException(
                status_code=422,
                detail="created_after must not be later than created_before",
            )

        # Omitted cursor or an explicit empty string starts before the
        # smallest policy version and fixes the range's replayable
        # snapshot. Whitespace, malformed, forged, cross-scope,
        # cross-filter, cross-snapshot or foreign-kind cursors are
        # indistinguishable 422s.
        boundary_dt: datetime | None = None
        boundary_policy: str | None = None
        snapshot_seq: int | None = None
        if cursor is not None and cursor != "":
            if not cursor.strip() or not _CURSOR_RE.fullmatch(cursor):
                raise HTTPException(status_code=422, detail="invalid cursor")
            decoded_boundary = _decode_policy_cursor(
                cursor,
                tenant_id,
                workload_id,
                policy_id=policy_filter or "",
                name=name_filter or "",
                status=status_filter,
                created_after=after_raw,
                created_before=before_raw,
            )
            if decoded_boundary is None:
                raise HTTPException(status_code=422, detail="invalid cursor")
            boundary_at_raw, boundary_policy, snapshot_seq = decoded_boundary
            try:
                boundary_dt = _parse_utc_rfc3339(boundary_at_raw)
            except ValueError:
                raise HTTPException(status_code=422, detail="invalid cursor")

        # --- read-only scan ---------------------------------------------
        rows: list = []
        try:
            with session_factory() as session:
                # An explicitly named version must exist in exactly this
                # tenant and workload; an unknown or cross-scope id is an
                # indistinguishable 404 (an explicit resource selector, not
                # a mere range bound). A name-only filter is different: no
                # such resource is named, so a non-match is an empty range
                # rather than a 404.
                if policy_filter is not None:
                    named = session.get(Policy, policy_filter)
                    if (
                        named is None
                        or named.tenant_id != tenant_id
                        or named.workload_id != workload_id
                    ):
                        raise HTTPException(
                            status_code=404, detail="policy not found"
                        )

                if snapshot_seq is None:
                    # First query of the range: fix a replayable snapshot
                    # at the scope's current lifecycle commit high-water
                    # mark. Version creation and retirement share this one
                    # counter, so a creation or a retirement committed
                    # afterwards takes a greater sequence: the new row lies
                    # beyond the mark and an as-yet-active row keeps
                    # reconstructing as active on replayed pages, regardless
                    # of in-place updates or equal business timestamps.
                    fixed_seq = session.scalar(
                        select(PolicyCommitCounter.last_seq).where(
                            PolicyCommitCounter.tenant_id == tenant_id,
                            PolicyCommitCounter.workload_id == workload_id,
                        )
                    )
                    # No committed lifecycle change in the scope yet.
                    snapshot_seq = int(fixed_seq) if fixed_seq is not None else 0

                if snapshot_seq > 0:
                    stmt = select(Policy).where(
                        Policy.tenant_id == tenant_id,
                        Policy.workload_id == workload_id,
                        # Membership: versions that had committed by the
                        # snapshot. Versions are never deleted, so the
                        # immutable creation sequence alone bounds membership.
                        Policy.commit_seq <= snapshot_seq,
                    )
                    if policy_filter is not None:
                        stmt = stmt.where(Policy.policy_id == policy_filter)
                    if name_filter is not None:
                        # Name is immutable, so the current column value is
                        # exactly the value the snapshot row carried.
                        stmt = stmt.where(Policy.name == name_filter)
                    if after_dt is not None:
                        stmt = stmt.where(Policy.created_at >= after_dt)
                    if before_dt is not None:
                        stmt = stmt.where(Policy.created_at <= before_dt)

                    # Status is evaluated as-of the snapshot rather than
                    # read from the mutable current column: a version is
                    # retired in the snapshot exactly when its retirement
                    # committed at or before the high-water mark. A
                    # retirement committed afterwards must not flip a
                    # replayed page from active to retired.
                    retired_as_of = and_(
                        Policy.retired_seq.is_not(None),
                        Policy.retired_seq <= snapshot_seq,
                    )
                    if status_filter == POLICY_STATUS_RETIRED:
                        stmt = stmt.where(retired_as_of)
                    elif status_filter == POLICY_STATUS_ACTIVE:
                        stmt = stmt.where(
                            or_(
                                Policy.retired_seq.is_(None),
                                Policy.retired_seq > snapshot_seq,
                            )
                        )

                    if boundary_dt is not None:
                        # Exclusive (created_at, policy_id) keyset. Both
                        # components are immutable, so the boundary walks
                        # the same fixed snapshot set in stable order.
                        stmt = stmt.where(
                            or_(
                                Policy.created_at > boundary_dt,
                                and_(
                                    Policy.created_at == boundary_dt,
                                    Policy.policy_id > boundary_policy,
                                ),
                            )
                        )
                    stmt = stmt.order_by(
                        Policy.created_at.asc(),
                        Policy.policy_id.asc(),
                    ).limit(POLICY_PAGE_SIZE + 1)
                    # One extra row is the "more follows" probe. The scan
                    # is a single read-only statement: a storage failure
                    # aborts the whole request with a 500 rather than
                    # returning a partial page.
                    rows = list(session.scalars(stmt))
        except HTTPException:
            raise
        except Exception:
            logger.error("policy query failed")
            raise HTTPException(
                status_code=500, detail="policy registry unavailable"
            )

        has_more = len(rows) > POLICY_PAGE_SIZE
        page = rows[:POLICY_PAGE_SIZE]

        policies = [
            {
                # Creation-response field order, then lifecycle fields.
                "policy_id": row.policy_id,
                "tenant_id": row.tenant_id,
                "workload_id": row.workload_id,
                "name": row.name,
                "version": row.version,
                # The rule is returned verbatim as created: numbers in the
                # rule are re-serialized from the persisted canonical JSON
                # without float coercion, so decimals and -0.0 round-trip
                # exactly and no other metadata produces a float or a
                # non-finite value.
                "rule": json.loads(row.rule_json),
                "created_at": _rfc3339(row.created_at),
                # Reconstruct the as-of-snapshot status from the retirement
                # commit marker rather than the mutable current column.
                "status": (
                    POLICY_STATUS_RETIRED
                    if row.retired_seq is not None
                    and row.retired_seq <= snapshot_seq
                    else POLICY_STATUS_ACTIVE
                ),
                # The recorded retirement time is shown only when the
                # retirement had committed by the snapshot; an active
                # (as-of) version reports null even if retired later.
                "retired_at": (
                    _rfc3339(row.retired_at)
                    if row.retired_seq is not None
                    and row.retired_seq <= snapshot_seq
                    else None
                ),
            }
            for row in page
        ]

        if has_more:
            last = page[-1]
            next_cursor = _encode_policy_cursor(
                tenant_id,
                workload_id,
                _rfc3339(last.created_at),
                last.policy_id,
                policy_id=policy_filter or "",
                name=name_filter or "",
                status=status_filter,
                created_after=after_raw,
                created_before=before_raw,
                snapshot_seq=snapshot_seq,
            )
            complete = False
        else:
            next_cursor = ""
            complete = True

        # Compact container (policies, next_cursor, complete) with a single
        # terminating newline. The only non-string scalar outside the rule
        # is the integer version and the boolean complete; floats and
        # non-finite values are impossible there, while rule numbers
        # round-trip verbatim (including decimals and -0.0). No evidence,
        # claim value, capability, key or exception text is ever included.
        body = (
            json.dumps(
                {
                    "policies": policies,
                    "next_cursor": next_cursor,
                    "complete": complete,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/policies/compare")
    def compare_policies(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        left_policy_id: str = Query(...),
        right_policy_id: str = Query(...),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Compare the immutable rule trees of two persisted versions.

        The request carries exactly the four required query parameters —
        ``tenant_id``, ``workload_id``, ``left_policy_id`` and
        ``right_policy_id`` — and no body. A missing, blank, wrong-typed,
        duplicated or unknown parameter, a non-canonical (non-lowercase)
        policy UUID, or any non-empty body is a 422 raised before any
        policy is read. An unknown or out-of-scope identifier on either
        side is one indistinguishable 404 (existence is never revealed);
        comparing a version with itself is legal. The handler issues only
        SELECTs: it never creates, retires or otherwise mutates a version,
        appends no audit record and influences no decision, and a storage
        failure aborts the whole request with a 500.

        Only the two immutable rule trees determine ``identical`` and
        ``changes``; name, version, status and timestamps are reported but
        never compared. Comparing a retired version is legal and changes
        nothing: the version keeps its status and never re-enters
        decisioning. The response is compact JSON with a single
        terminating newline and contains no evidence, claim value,
        capability, key or exception text.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {
            "tenant_id",
            "workload_id",
            "left_policy_id",
            "right_policy_id",
        }
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is rejected rather than silently
        # treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )

        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # Both identifiers must be canonical lowercase UUIDs; the raw
        # values must match exactly. This runs before any storage access.
        if not _UUID_RE.fullmatch(left_policy_id) or not _UUID_RE.fullmatch(
            right_policy_id
        ):
            raise HTTPException(
                status_code=422, detail="invalid policy identifier"
            )

        # --- read-only lookup -------------------------------------------
        try:
            with session_factory() as session:
                left_row = session.get(Policy, left_policy_id)
                # A version compared with itself is one row read once.
                right_row = (
                    left_row
                    if right_policy_id == left_policy_id
                    else session.get(Policy, right_policy_id)
                )
                for row in (left_row, right_row):
                    # Do not reveal whether an out-of-scope or unknown
                    # policy exists: unknown id and scope mismatch share
                    # one indistinguishable 404.
                    if (
                        row is None
                        or row.tenant_id != tenant_id
                        or row.workload_id != workload_id
                    ):
                        raise HTTPException(
                            status_code=404, detail="policy not found"
                        )
                # The persisted canonical rule JSON is re-parsed without
                # float coercion, so decimals and -0.0 compare exactly as
                # created.
                left_rule = json.loads(left_row.rule_json)
                right_rule = json.loads(right_row.rule_json)

                def _side(row: Policy) -> dict:
                    return {
                        "policy_id": row.policy_id,
                        "name": row.name,
                        "version": row.version,
                        "status": row.status,
                        "created_at": _rfc3339(row.created_at),
                        "retired_at": (
                            _rfc3339(row.retired_at)
                            if row.retired_at is not None
                            else None
                        ),
                    }

                left_side = _side(left_row)
                right_side = _side(right_row)
        except HTTPException:
            raise
        except Exception:
            logger.error("policy compare failed")
            raise HTTPException(
                status_code=500, detail="policy registry unavailable"
            )

        changes = diff_rules(left_rule, right_rule)
        # Compact container (tenant_id, workload_id, left, right,
        # identical, changes) with a single terminating newline. Rule
        # numbers round-trip verbatim from the persisted canonical JSON;
        # the only other non-string scalars are the integer versions and
        # the boolean identical, so floats and non-finite values are
        # impossible there.
        body = (
            json.dumps(
                {
                    "tenant_id": tenant_id,
                    "workload_id": workload_id,
                    "left": left_side,
                    "right": right_side,
                    "identical": not changes,
                    "changes": changes,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.post("/v1/policies//evaluate")
    def evaluate_policy_identifier_required(
        body: EvaluatePolicyRequest,
    ) -> Response:
        # An empty path segment is a missing policy identifier: a 422
        # client error rather than a routing-level 404 or 405. It never
        # reads or evaluates a policy version.
        raise HTTPException(status_code=422, detail="invalid policy identifier")

    @app.post("/v1/policies/{policy_id}/evaluate")
    def evaluate_policy(policy_id: str, body: EvaluatePolicyRequest) -> Response:
        """Evaluate one immutable policy version against presented claims.

        This is a read-only trial evaluation: it confirms what the exact
        version named in the path would decide for the given claims
        before a real proof is submitted. The path identifier must be a
        canonical lowercase UUID and the body carries only the two
        non-blank scope strings and the ``claims`` JSON object; any
        missing, blank, wrong-typed, incomplete or unknown field, a
        ``claims`` value that is not a JSON object, or a non-canonical
        path identifier is a 422 raised before any storage access. An
        unknown policy or one outside the body's tenant/workload is one
        indistinguishable 404 (existence is never revealed).

        The handler issues only SELECTs against the policy row: it never
        creates a decision, audit event, lifecycle event, idempotency
        record or counter, never mutates the policy, and a storage
        failure aborts the whole request with a 500 rather than
        returning a partial evaluation. A retired version still
        evaluates — the result reflects this exact immutable version,
        never a later one. ``allowed`` is the root outcome of the
        version's rule under the standard evaluation semantics, and
        ``evaluation`` is the same complete depth-first node list a
        formal decision explanation carries: positions, structural types
        and booleans only — never claim names, paths, expected or actual
        values, evidence or key material. The response is compact JSON
        with a single terminating newline; repeated identical inputs
        produce identical bodies except for ``checked_at``.
        """
        # A path identifier that is missing (empty segment), blank,
        # whitespace-padded or not a canonical lowercase UUID is a
        # field/format error; the raw value must match exactly. This
        # runs before any storage access.
        if not _UUID_RE.fullmatch(policy_id):
            raise HTTPException(status_code=422, detail="invalid policy identifier")

        # --- read-only lookup -------------------------------------------
        try:
            with session_factory() as session:
                policy = session.get(Policy, policy_id)
                # Do not reveal whether an out-of-scope or unknown
                # policy exists: unknown id and scope mismatch share one
                # indistinguishable 404.
                if (
                    policy is None
                    or policy.tenant_id != body.tenant_id
                    or policy.workload_id != body.workload_id
                ):
                    raise HTTPException(status_code=404, detail="policy not found")
                # The persisted canonical rule JSON is re-parsed without
                # float coercion; the version's status is irrelevant —
                # a retired version still evaluates as its immutable
                # rule dictates.
                rule = json.loads(policy.rule_json)
                policy_version = policy.version
        except HTTPException:
            raise
        except Exception:
            logger.error("policy evaluate failed")
            raise HTTPException(
                status_code=500, detail="policy registry unavailable"
            )

        # Pure evaluation against the caller-supplied claims; nothing is
        # persisted. The claims never appear in the response.
        allowed = evaluate_rule(rule, body.claims)
        evaluation_nodes = explain_rule(rule, body.claims)
        if not evaluation_nodes or evaluation_nodes[0]["outcome"] != allowed:
            # Defensive: a validated rule always has a root whose outcome
            # is the overall verdict. Treat a mismatch as a server-side
            # integrity failure rather than returning an evaluation that
            # contradicts its own verdict.
            logger.error("policy evaluation explanation mismatch")
            raise HTTPException(
                status_code=500, detail="policy evaluation unavailable"
            )

        checked_at = _utcnow()
        # Compact container (policy_id, policy_version, tenant_id,
        # workload_id, allowed, checked_at, evaluation) with a single
        # terminating newline. The only non-string scalars are the
        # integer version, the booleans and the integer node indices, so
        # floats and non-finite values are impossible.
        body_bytes = (
            json.dumps(
                {
                    "policy_id": policy_id,
                    "policy_version": policy_version,
                    "tenant_id": body.tenant_id,
                    "workload_id": body.workload_id,
                    "allowed": allowed,
                    "checked_at": _rfc3339(checked_at),
                    "evaluation": evaluation_nodes,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body_bytes, media_type="application/json")

    @app.post(
        "/v1/evidence/{evidence_id}/decisions",
        response_model=DecisionResponse,
    )
    def create_decision(
        evidence_id: str, body: CreateDecisionRequest
    ) -> DecisionResponse:
        evidence_digest = hashlib.sha256(body.evidence.encode("utf-8")).hexdigest()
        nonce_digest = _nonce_digest(body.nonce)
        with session_factory() as session:
            # Serialize concurrent decision makers on the evidence row;
            # SQLite writers are already serialized process-wide via
            # BEGIN IMMEDIATE.
            evidence = session.scalar(
                select(Evidence)
                .where(Evidence.evidence_id == evidence_id)
                .with_for_update()
            )
            if (
                evidence is None
                or evidence.tenant_id != body.tenant_id
                or evidence.workload_id != body.workload_id
            ):
                raise HTTPException(status_code=404, detail="evidence not found")

            challenge = session.get(Challenge, evidence.challenge_id)
            if (
                challenge is None
                or challenge.tenant_id != body.tenant_id
                or challenge.workload_id != body.workload_id
            ):
                raise HTTPException(status_code=404, detail="evidence not found")

            # The presented nonce must match the evidence's bound challenge.
            # Only its digest is stored; the plaintext nonce is never kept.
            if not hmac.compare_digest(challenge.nonce_digest, nonce_digest):
                raise HTTPException(status_code=422, detail="invalid nonce")

            policy = session.scalar(
                select(Policy)
                .where(Policy.policy_id == body.policy_id)
                .with_for_update()
            )
            if (
                policy is None
                or policy.tenant_id != body.tenant_id
                or policy.workload_id != body.workload_id
            ):
                raise HTTPException(status_code=404, detail="policy not found")

            # Exactly one auditable decision per evidence/policy version.
            # A retry or concurrent request observes and returns the same
            # row without re-evaluating anything.
            existing = session.scalar(
                select(Decision).where(
                    Decision.evidence_id == evidence_id,
                    Decision.policy_id == body.policy_id,
                )
            )
            if existing is not None:
                return DecisionResponse(
                    decision_id=existing.decision_id,
                    evidence_id=existing.evidence_id,
                    policy_id=existing.policy_id,
                    policy_version=existing.policy_version,
                    status=existing.status,
                    decided_at=_rfc3339(existing.decided_at),
                )

            # A retired policy version is terminal and can never produce a
            # *new* decision. This follows the idempotent replay lookup, so
            # a decision already recorded against the version keeps
            # returning its stored result: retirement is never retroactive
            # and existing decisions keep their status, policy_version and
            # decided_at. The row lock taken above orders this judgement
            # against retirement on the same version — a retire that
            # commits first is observed as 409 and inserts neither a
            # decision nor a proof event, while a retire that commits after
            # this decision never rewrites it. SQLite serializes all
            # writers via BEGIN IMMEDIATE.
            if policy.status == POLICY_STATUS_RETIRED:
                raise HTTPException(status_code=409, detail="policy is retired")

            # Only settled, verified evidence may drive a release decision;
            # received (unverified) and rejected evidence cannot. This gate
            # precedes content checks: an unverified record is never
            # evaluated, regardless of the presented bytes.
            if evidence.status != "verified":
                raise HTTPException(
                    status_code=409, detail="evidence is not verified"
                )

            # The presented evidence must be byte-identical to what was
            # received; compare digests only — never persist the bytes.
            if not hmac.compare_digest(evidence.evidence_sha256, evidence_digest):
                raise HTTPException(
                    status_code=422, detail="evidence digest mismatch"
                )

            # Claims are evaluated only for the built-in JSON evidence
            # formats, whose document shape the service knows. Other
            # formats carry no parseable claims and are rejected as a
            # format error rather than guessed at.
            if evidence.evidence_format not in (
                ATTESTED_NONCE_JSON,
                X509_ATTESTED_NONCE_JSON,
            ):
                raise HTTPException(
                    status_code=422, detail="unsupported evidence format for decision"
                )

            # Parse the presented (digest-matched) evidence just far enough
            # to read its claims. The evidence is verified already; the
            # verifier is not invoked again and neither the document nor
            # the claims are persisted anywhere.
            try:
                document = json.loads(body.evidence)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise HTTPException(
                    status_code=422, detail="evidence is not valid JSON"
                )
            if not isinstance(document, dict):
                raise HTTPException(
                    status_code=422, detail="evidence is not valid JSON"
                )
            claims = document.get("claims", {})
            if not isinstance(claims, dict):
                raise HTTPException(
                    status_code=422, detail="evidence is not valid JSON"
                )

            rule = json.loads(policy.rule_json)
            satisfied = evaluate_rule(rule, claims)
            status = (
                DECISION_STATUS_ALLOWED if satisfied else DECISION_STATUS_DENIED
            )
            # The complete depth-first rule explanation is computed here
            # and persisted in the same transaction as the decision below.
            # It holds only node positions, structural types and booleans
            # — never claim names, comparison values, actual claim values,
            # evidence or nonces — and every node is evaluated against the
            # same verified claims, so the root's outcome is exactly the
            # decision's status. The explanation is historical: it is
            # written once and policy retirement, later policy versions,
            # identity/revocation changes or key rotation never touch it.
            evaluation_nodes = explain_rule(rule, claims)
            if not evaluation_nodes or evaluation_nodes[0]["outcome"] != satisfied:
                # Defensive: a validated rule always has a root whose
                # outcome is the overall verdict. Treat a mismatch as a
                # server-side integrity failure rather than persisting a
                # decision whose explanation contradicts its status.
                logger.error("decision evaluation explanation mismatch")
                raise HTTPException(
                    status_code=500, detail="decision evaluation unavailable"
                )
            decision_id = str(uuid.uuid4())
            decided_at = _utcnow()
            # The proof timeline records the *first* policy decision for an
            # evidence exactly once. Decision transactions for one evidence
            # serialize on its locked row (and SQLite writers serialize
            # process-wide), so when this lookup sees no prior decision of
            # any policy, this commit is necessarily the first and its
            # event wins the (evidence, type) uniqueness backstop; a later
            # decision against another policy inserts no proof event. The
            # lookup runs before the new decision is added/flushed, so it
            # can only observe already-persisted earlier decisions.
            prior_decision = session.scalar(
                select(Decision.decision_id)
                .where(Decision.evidence_id == evidence_id)
                .limit(1)
            )
            decision = Decision(
                decision_id=decision_id,
                tenant_id=body.tenant_id,
                workload_id=body.workload_id,
                evidence_id=evidence_id,
                policy_id=policy.policy_id,
                policy_version=policy.version,
                status=status,
                decided_at=decided_at,
                # Per-scope business commit sequence allocated in this same
                # transaction; it fixes the compliance decision query's
                # replayable snapshot to the commit boundary so a decision
                # committed later (even with an older decided_at) never
                # enters an already-fixed first query.
                commit_seq=_next_decision_commit_seq(
                    session, body.tenant_id, body.workload_id
                ),
            )
            session.add(decision)
            # Persist the full explanation in this same transaction: one
            # row per tree node in pre-order, keyed by (decision_id,
            # node_index). A rollback (including losing the unique-decision
            # race below) removes the decision, proof event and every node
            # row together, so a committed decision never exists without
            # exactly one complete explanation and retries never add a
            # second. Only position, structural type and the boolean are
            # stored; the path is integer-index JSON.
            for node in evaluation_nodes:
                session.add(
                    DecisionEvaluationNode(
                        decision_id=decision_id,
                        node_index=node["node_index"],
                        rule_path=json.dumps(
                            node["rule_path"], separators=(",", ":")
                        ),
                        node_type=node["node_type"],
                        outcome=bool(node["outcome"]),
                    )
                )
            if prior_decision is None:
                # Commits in the same transaction as the first decision
                # row, recording only the fixed allowed/denied code, the
                # decided policy version and the decision time — never the
                # evidence, nonce, claims, capability or any payload. The
                # scope already contains this proof's reception (and
                # verification) events, so the commit-order sequence is
                # allocated off the established per-scope counter.
                session.add(
                    ProofLifecycleEvent(
                        event_id=str(uuid.uuid4()),
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        event_type=PROOF_EVENT_TYPE_DECISION,
                        evidence_id=evidence_id,
                        commit_seq=_next_proof_event_commit_seq(
                            session, body.tenant_id, body.workload_id
                        ),
                        policy_version=policy.version,
                        evidence_format=None,
                        status=(
                            PROOF_EVENT_STATUS_ALLOWED
                            if status == DECISION_STATUS_ALLOWED
                            else PROOF_EVENT_STATUS_DENIED
                        ),
                        occurred_at=decided_at,
                    )
                )
            try:
                session.commit()
            except IntegrityError:
                # A concurrent request created the unique decision first;
                # return its immutable result.
                session.rollback()
                winner = session.scalar(
                    select(Decision).where(
                        Decision.evidence_id == evidence_id,
                        Decision.policy_id == body.policy_id,
                    )
                )
                if winner is None:
                    raise HTTPException(
                        status_code=409, detail="decision conflict"
                    )
                return DecisionResponse(
                    decision_id=winner.decision_id,
                    evidence_id=winner.evidence_id,
                    policy_id=winner.policy_id,
                    policy_version=winner.policy_version,
                    status=winner.status,
                    decided_at=_rfc3339(winner.decided_at),
                )

        return DecisionResponse(
            decision_id=decision_id,
            evidence_id=evidence_id,
            policy_id=policy.policy_id,
            policy_version=policy.version,
            status=status,
            decided_at=_rfc3339(decided_at),
        )

    @app.post(
        "/v1/release-grants",
        status_code=201,
        response_model=ReleaseGrantCreatedResponse,
    )
    def create_release_grant(
        body: CreateReleaseGrantRequest,
    ) -> ReleaseGrantCreatedResponse:
        # Capabilities are 32 bytes from the CSPRNG, rendered unpadded
        # base64url. The plaintext lives only on this stack frame and the
        # create response; only its SHA-256 digest is persisted.
        capability_bytes = secrets.token_bytes(CAPABILITY_BYTES)
        capability = base64.urlsafe_b64encode(capability_bytes).rstrip(
            b"="
        ).decode("ascii")
        now = _utcnow()
        expires_at = now + timedelta(seconds=body.ttl_seconds)
        grant_id = str(uuid.uuid4())
        with session_factory() as session:
            decision = session.get(Decision, body.decision_id)
            # Decisions carry no scope columns of their own; their scope is
            # the scope of the evidence they were taken against.
            evidence = (
                session.get(Evidence, decision.evidence_id)
                if decision is not None
                else None
            )
            if (
                decision is None
                or evidence is None
                or evidence.tenant_id != body.tenant_id
                or evidence.workload_id != body.workload_id
            ):
                # Do not reveal whether an out-of-scope decision exists.
                raise HTTPException(status_code=404, detail="decision not found")
            if decision.status != DECISION_STATUS_ALLOWED:
                # A denied (or any future non-allowed) decision can never
                # authorize data release.
                raise HTTPException(
                    status_code=409, detail="decision is not allowed"
                )
            grant = ReleaseGrant(
                grant_id=grant_id,
                tenant_id=body.tenant_id,
                workload_id=body.workload_id,
                decision_id=body.decision_id,
                data_id=body.data_id,
                capability_digest=_nonce_digest(capability),
                status="pending",
                issued_at=now,
                expires_at=expires_at,
                consumed_at=None,
            )
            session.add(grant)
            # The per-grant migration timeline records exactly one event
            # per committed state transition in the same transaction as the
            # grant row (and the unchanged compliance audit below): the
            # empty-state birth into pending with reason "issued".
            _record_release_grant_event(
                session,
                tenant_id=body.tenant_id,
                workload_id=body.workload_id,
                grant_id=grant_id,
                old_status=None,
                new_status=RELEASE_GRANT_STATUS_PENDING,
                reason=RELEASE_GRANT_EVENT_REASON_ISSUED,
                now=now,
            )
            # The compliance event commits in the same transaction as the
            # grant row, so a pending event exists if and only if the grant
            # did. Only identifiers, the fixed status, the timestamp and
            # the capability digest are recorded — never the capability.
            _append_audit_event(
                session,
                tenant_id=body.tenant_id,
                workload_id=body.workload_id,
                event_type=AUDIT_EVENT_TYPE_GRANT,
                grant_id=grant_id,
                decision_id=body.decision_id,
                data_id=body.data_id,
                status=AUDIT_EVENT_STATUS_PENDING,
                capability_sha256=grant.capability_digest,
                occurred_at=now,
            )
            session.commit()
        return ReleaseGrantCreatedResponse(
            grant_id=grant.grant_id,
            decision_id=body.decision_id,
            data_id=body.data_id,
            capability=capability,
            pending=True,
            issued_at=_rfc3339(now),
            expires_at=_rfc3339(expires_at),
        )

    @app.get("/v1/release-grants")
    def list_release_grants(
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        grant_id: str | None = Query(default=None),
        decision_id: str | None = Query(default=None),
        data_id: str | None = Query(default=None),
        status: str | None = Query(default=None),
        issued_after: str | None = Query(default=None),
        issued_before: str | None = Query(default=None),
        cursor: str | None = Query(default=None),
    ) -> Response:
        """Return a read-only, cursor-stable audit page of release grants.

        The page is scoped by the mandatory tenant/workload and may be
        narrowed by grant/decision/data identifiers, status and an
        inclusive issued-at window. Ordering is grant_id ascending with
        an exclusive keyset cursor; the cursor is HMAC-authenticated and
        bound to the scope *and* every active filter, so it cannot be
        forged or replayed against different filter values. The handler
        only ever issues SELECTs: it never settles, writes or otherwise
        mutates a grant, so concurrent consume/revoke transitions remain
        the sole writers and queries observe only committed state.
        """
        # --- field validation (all 422, no storage touched) -------------
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        def _identifier(value: str | None, *, uuid_shaped: bool) -> str | None:
            if value is None:
                return None
            if not value.strip():
                raise HTTPException(
                    status_code=422, detail="filter must be a non-empty string"
                )
            if uuid_shaped:
                # Canonical lowercase-UUID shape; uppercase is accepted and
                # normalized, but surrounding whitespace is a format error
                # (mirrors the grant/batch path-identifier handling).
                if not _UUID_RE.fullmatch(value.lower()):
                    raise HTTPException(
                        status_code=422,
                        detail="invalid grant or decision identifier",
                    )
                return value.lower()
            return value

        grant_filter = _identifier(grant_id, uuid_shaped=True)
        decision_filter = _identifier(decision_id, uuid_shaped=True)
        # data_id is a caller-chosen identifier: any non-blank string is a
        # legal format; existence in the scope is checked below.
        data_filter = _identifier(data_id, uuid_shaped=False)

        if status is not None:
            if not status.strip() or status not in RELEASE_GRANT_STATUS_CODES:
                raise HTTPException(status_code=422, detail="invalid status")

        # Timestamp filters: absent means unbounded; an explicit empty or
        # whitespace value is an illegal format (422), not "unbounded".
        # Parsed bounds are embedded into the cursor in normalized
        # RFC3339/UTC form so equivalent spellings cannot mint two
        # different cursor domains.
        def _time_bound(value: str | None, name: str) -> tuple[str, datetime | None]:
            if value is None:
                return "", None
            if not value.strip():
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            try:
                parsed = _parse_utc_rfc3339(value)
            except ValueError:
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            return _rfc3339(parsed), parsed

        after_raw, after_dt = _time_bound(issued_after, "issued_after")
        before_raw, before_dt = _time_bound(issued_before, "issued_before")
        # The lower bound must not be later than the upper bound. Equality
        # is a valid (single-instant) window.
        if after_dt is not None and before_dt is not None and after_dt > before_dt:
            raise HTTPException(
                status_code=422, detail="issued_after must not be later than issued_before"
            )

        # An omitted cursor or an explicit empty string means the
        # beginning of the scope (before the smallest grant id). A
        # whitespace-only or otherwise malformed value is a 422, as is a
        # forged, tampered, cross-scope or cross-filter cursor.
        status_filter = status if status is not None else ""
        if cursor is None or cursor == "":
            boundary = ""
        else:
            if not cursor.strip() or not _CURSOR_RE.fullmatch(cursor):
                raise HTTPException(status_code=422, detail="invalid cursor")
            boundary = _decode_grant_audit_cursor(
                cursor,
                tenant_id,
                workload_id,
                grant_id=grant_filter or "",
                decision_id=decision_filter or "",
                data_id=data_filter or "",
                status=status_filter,
                issued_after=after_raw,
                issued_before=before_raw,
            )
            if boundary is None:
                raise HTTPException(status_code=422, detail="invalid cursor")

        # --- read-only audit scan ---------------------------------------
        try:
            with session_factory() as session:
                # Explicitly named identifiers must exist in exactly this
                # scope; an unknown or cross-scope identifier is a 404
                # rather than an empty-looking page. A decision's scope is
                # the scope of its evidence.
                if grant_filter is not None:
                    grant_row = session.get(ReleaseGrant, grant_filter)
                    if (
                        grant_row is None
                        or grant_row.tenant_id != tenant_id
                        or grant_row.workload_id != workload_id
                    ):
                        raise HTTPException(
                            status_code=404, detail="grant not found"
                        )
                if decision_filter is not None:
                    decision_row = session.get(Decision, decision_filter)
                    evidence_row = (
                        session.get(Evidence, decision_row.evidence_id)
                        if decision_row is not None
                        else None
                    )
                    if (
                        decision_row is None
                        or evidence_row is None
                        or evidence_row.tenant_id != tenant_id
                        or evidence_row.workload_id != workload_id
                    ):
                        raise HTTPException(
                            status_code=404, detail="decision not found"
                        )
                if data_filter is not None:
                    # A data identifier is known in a scope when a data
                    # envelope is registered there or an audited grant in
                    # the scope references it (grants may be minted before
                    # the envelope exists). Anything else — including an
                    # identifier belonging to another tenant/workload — is
                    # an indistinguishable 404.
                    envelope_row = session.get(
                        DataEnvelope, (tenant_id, workload_id, data_filter)
                    )
                    if envelope_row is None:
                        referenced = session.scalar(
                            select(ReleaseGrant.grant_id)
                            .where(
                                ReleaseGrant.tenant_id == tenant_id,
                                ReleaseGrant.workload_id == workload_id,
                                ReleaseGrant.data_id == data_filter,
                            )
                            .limit(1)
                        )
                        if referenced is None:
                            raise HTTPException(
                                status_code=404, detail="data envelope not found"
                            )

                stmt = select(ReleaseGrant).where(
                    ReleaseGrant.tenant_id == tenant_id,
                    ReleaseGrant.workload_id == workload_id,
                )
                if grant_filter is not None:
                    stmt = stmt.where(ReleaseGrant.grant_id == grant_filter)
                if decision_filter is not None:
                    stmt = stmt.where(ReleaseGrant.decision_id == decision_filter)
                if data_filter is not None:
                    stmt = stmt.where(ReleaseGrant.data_id == data_filter)
                if status_filter:
                    stmt = stmt.where(ReleaseGrant.status == status_filter)
                if after_dt is not None:
                    stmt = stmt.where(ReleaseGrant.issued_at >= after_dt)
                if before_dt is not None:
                    stmt = stmt.where(ReleaseGrant.issued_at <= before_dt)
                stmt = (
                    stmt.where(ReleaseGrant.grant_id > boundary)
                    .order_by(ReleaseGrant.grant_id.asc())
                    .limit(RELEASE_GRANT_AUDIT_PAGE_SIZE + 1)
                )

                # One extra row is the "more follows" probe. The scan is a
                # single read-only statement: a storage failure aborts the
                # whole request with a 500 rather than returning a partial
                # page.
                rows = list(session.scalars(stmt))
        except HTTPException:
            raise
        except Exception:
            logger.error("release grant audit query failed")
            raise HTTPException(
                status_code=500, detail="release grant audit unavailable"
            )

        has_more = len(rows) > RELEASE_GRANT_AUDIT_PAGE_SIZE
        page = rows[:RELEASE_GRANT_AUDIT_PAGE_SIZE]

        grants = [
            {
                "grant_id": row.grant_id,
                "decision_id": row.decision_id,
                "data_id": row.data_id,
                "status": row.status,
                # The capability is exposed only as its SHA-256 digest;
                # the plaintext capability exists nowhere on this path.
                "capability_sha256": row.capability_digest,
                "issued_at": _rfc3339(row.issued_at),
                "expires_at": _rfc3339(row.expires_at),
                "consumed_at": (
                    _rfc3339(row.consumed_at) if row.consumed_at is not None else None
                ),
                "revoked_at": (
                    _rfc3339(row.revoked_at) if row.revoked_at is not None else None
                ),
            }
            for row in page
        ]

        if has_more:
            next_cursor = _encode_grant_audit_cursor(
                tenant_id,
                workload_id,
                page[-1].grant_id,
                grant_id=grant_filter or "",
                decision_id=decision_filter or "",
                data_id=data_filter or "",
                status=status_filter,
                issued_after=after_raw,
                issued_before=before_raw,
            )
            complete = False
        else:
            next_cursor = ""
            complete = True

        return _compact_json(
            {
                "grants": grants,
                "next_cursor": next_cursor,
                "complete": complete,
            }
        )

    @app.get("/v1/observability/release-summary")
    def get_release_summary(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return a read-only, point-in-time summary of committed release state.

        The range is fixed entirely by the two mandatory, non-blank query
        parameters; the request body is always empty. Every shape failure
        (a missing or blank parameter, a wrong-typed or unknown parameter,
        or any non-empty body) is rejected as a 422 before any state is
        read. The handler then takes a single read-only snapshot and never
        writes: it consumes no rate-limit budget, appends no audit row, and
        returns no capability, payload, evidence or key material.

        The summary reports the three grant status counts, the pending
        grants split into live (still within their validity window) and
        expired, the current UTC minute's shared budget usage, and the
        envelope population split by current versus historical master key
        version. A storage failure or an unusable master keyring is a 500
        with no partial summary; nothing is ever modified.
        """
        # --- request shape (all 422, no state is read) ------------------
        allowed_params = {"tenant_id", "workload_id"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each scope parameter is a single scalar string; a repeated
        # parameter (a multi-valued/list value) is the wrong shape, not a
        # silently last-wins scalar.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # The current master key version classifies the envelope rows; it
        # is key material metadata only (an integer version), never a key.
        # A missing or malformed keyring is a server failure: fail closed
        # with a 500 rather than reporting an unclassified envelope set.
        try:
            current_key_version = load_keyring().current_version
        except MasterKeyError as exc:
            logger.error("master key configuration unavailable: %s", exc)
            raise HTTPException(
                status_code=500, detail="release summary unavailable"
            )

        now = _utcnow()
        window_start = _utc_minute_window(now)

        def _grant_count(*predicates):
            return (
                select(func.count())
                .select_from(ReleaseGrant)
                .where(
                    ReleaseGrant.tenant_id == tenant_id,
                    ReleaseGrant.workload_id == workload_id,
                    *predicates,
                )
                .scalar_subquery()
            )

        def _envelope_count(*predicates):
            return (
                select(func.count())
                .select_from(DataEnvelope)
                .where(
                    DataEnvelope.tenant_id == tenant_id,
                    DataEnvelope.workload_id == workload_id,
                    *predicates,
                )
                .scalar_subquery()
            )

        # Every figure is computed by one statement of independent scalar
        # subqueries. A single statement evaluates against one consistent
        # database snapshot on every backend (and the write lock taken by
        # BEGIN IMMEDIATE on sqlite additionally orders it against
        # concurrent committers), so a grant/envelope group committing
        # concurrently can never appear as a half-applied set. The handler
        # issues no writes: it consumes no rate-limit slot and appends no
        # audit row.
        rate_used_sq = (
            select(RateLimitCounter.count)
            .where(
                RateLimitCounter.tenant_id == tenant_id,
                RateLimitCounter.workload_id == workload_id,
                RateLimitCounter.window_start == window_start,
            )
            .scalar_subquery()
        )
        summary_stmt = select(
            _grant_count(
                ReleaseGrant.status == RELEASE_GRANT_STATUS_PENDING
            ).label("pending"),
            _grant_count(
                ReleaseGrant.status == RELEASE_GRANT_STATUS_CONSUMED
            ).label("consumed"),
            _grant_count(
                ReleaseGrant.status == RELEASE_GRANT_STATUS_REVOKED
            ).label("revoked"),
            # Live pending grants are still within their validity window;
            # every other pending grant is expired. The expired figure is
            # derived as pending minus live so the two parts always sum
            # exactly to pending even at the expiry boundary.
            _grant_count(
                ReleaseGrant.status == RELEASE_GRANT_STATUS_PENDING,
                ReleaseGrant.expires_at > now,
            ).label("live_pending"),
            _envelope_count().label("envelopes"),
            # Envelopes recorded at the keyring's current version are
            # migrated; every other recorded version is historical.
            _envelope_count(
                DataEnvelope.key_version == current_key_version
            ).label("current_key_envelopes"),
            rate_used_sq.label("rate_used"),
        )
        try:
            with session_factory() as session:
                result = session.execute(summary_stmt).one()
        except Exception:
            logger.error("release summary query failed")
            raise HTTPException(
                status_code=500, detail="release summary unavailable"
            )

        pending_count = result.pending
        consumed_count = result.consumed
        revoked_count = result.revoked
        live_pending_count = result.live_pending
        expired_pending_count = pending_count - live_pending_count
        envelopes_count = result.envelopes
        current_key_envelopes = result.current_key_envelopes
        historical_key_envelopes = envelopes_count - current_key_envelopes
        # A scope with no business request this minute has no counter row,
        # so the scalar subquery returns NULL: zero used, the full five
        # remaining.
        rate_used = result.rate_used or 0
        rate_remaining = max(0, GRANT_BUDGET_PER_MINUTE - rate_used)

        # Exactly twelve fields in a fixed order: two JSON strings naming
        # the scope, then ten JSON integers. Every count is a Python int
        # produced by SQL count aggregation (never a float), so no -0.0 or
        # non-finite value is possible; allow_nan=False makes that
        # explicit. Compact JSON terminated by a single newline.
        body = (
            json.dumps(
                {
                    "tenant_id": tenant_id,
                    "workload_id": workload_id,
                    "pending": pending_count,
                    "consumed": consumed_count,
                    "revoked": revoked_count,
                    "live_pending": live_pending_count,
                    "expired_pending": expired_pending_count,
                    "rate_used": rate_used,
                    "rate_remaining": rate_remaining,
                    "envelopes": envelopes_count,
                    "current_key_envelopes": current_key_envelopes,
                    "historical_key_envelopes": historical_key_envelopes,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/observability/rate-limits")
    def get_rate_limits(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return a read-only snapshot of the three per-minute rate budgets.

        The scope is fixed entirely by the two mandatory, non-blank query
        parameters; the request body is always empty. Every shape failure
        (a missing or blank parameter, a wrong-typed or unknown parameter,
        or any non-empty body) is rejected as a 422 before any state is
        read — such a rejection reads no counter, consumes no budget and
        writes no audit.

        On success the handler reads the current UTC natural minute's
        persisted admission counts for the three independent budgets —
        challenge issuance, evidence verification and one-time-grant
        actions — from one consistent read-only snapshot (a single
        statement of scalar subqueries), so all three figures and the two
        window timestamps always belong to the same minute and can never
        observe a half-committed admission. The handler never writes: it
        issues no challenge, verifies no evidence, creates or consumes no
        grant, changes no counter, appends no audit or lifecycle event,
        and returns no nonce, evidence, payload, capability or key
        material — only the scope, the window and the three
        limit/used/remaining triples. A scope with no counter row this
        minute reports zero used. A storage failure is a 500 with no
        partial summary and no state change.
        """
        # --- request shape (all 422, no state is read) ------------------
        allowed_params = {"tenant_id", "workload_id"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each scope parameter is a single scalar string; a repeated
        # parameter (a multi-valued/list value) is the wrong shape, not a
        # silently last-wins scalar.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        now = _utcnow()
        window_start = _utc_minute_window(now)
        reset_at = window_start + timedelta(minutes=1)

        def _used(counter_model):
            return (
                select(counter_model.count)
                .where(
                    counter_model.tenant_id == tenant_id,
                    counter_model.workload_id == workload_id,
                    counter_model.window_start == window_start,
                )
                .scalar_subquery()
            )

        # One statement of independent scalar subqueries evaluates against
        # a single consistent database snapshot on every backend, so the
        # three budgets are always read from the same minute and a
        # concurrently committing admission is never observed half-applied.
        # The three counter tables are deliberately separate: the budgets
        # never share rows, quota or lock traffic, and this read borrows
        # nothing across tenants, workloads or minutes.
        usage_stmt = select(
            _used(ChallengeIssuanceCounter).label("challenge_issuance_used"),
            _used(VerificationAdmissionCounter).label("verification_used"),
            _used(RateLimitCounter).label("grant_actions_used"),
        )
        try:
            with session_factory() as session:
                result = session.execute(usage_stmt).one()
        except Exception:
            logger.error("rate limit summary query failed")
            raise HTTPException(
                status_code=500, detail="rate limit summary unavailable"
            )

        def _budget(limit: int, used: int | None) -> dict:
            # A scope with no admitted request this minute has no counter
            # row, so the scalar subquery returns NULL: zero used, the full
            # budget remaining. remaining is never negative.
            used_value = used or 0
            return {
                "limit": limit,
                "used": used_value,
                "remaining": max(0, limit - used_value),
            }

        # Fixed field order: the two scope strings, the two window
        # timestamps (UTC RFC3339 with the Z designator, both from the same
        # minute), then the three budget objects each carrying limit, used
        # and remaining as JSON integers. Compact JSON terminated by a
        # single newline.
        body = (
            json.dumps(
                {
                    "tenant_id": tenant_id,
                    "workload_id": workload_id,
                    "window_start": _rfc3339_z(window_start),
                    "reset_at": _rfc3339_z(reset_at),
                    "challenge_issuance": _budget(
                        CHALLENGE_ISSUANCE_BUDGET_PER_MINUTE,
                        result.challenge_issuance_used,
                    ),
                    "verification": _budget(
                        VERIFICATION_BUDGET_PER_MINUTE,
                        result.verification_used,
                    ),
                    "grant_actions": _budget(
                        GRANT_BUDGET_PER_MINUTE,
                        result.grant_actions_used,
                    ),
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/observability/metrics")
    def get_observability_metrics(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Export the scope's release state in Prometheus text format.

        A read-only companion to the JSON observability endpoints, rendered
        for scrape-based monitoring. The scope is fixed entirely by the two
        mandatory, non-blank query parameters; the request body is always
        empty. Every shape failure (a missing or blank parameter, a
        wrong-typed or unknown parameter, or any non-empty body) is
        rejected as a 422 before any state is read.

        On success the handler renders one consistent read-only snapshot —
        a single statement of independent scalar subqueries — as a
        Prometheus text exposition (version 0.0.4): the three release-grant
        status totals, the pending grants split into live and expired, the
        envelope population split by current versus historical master key
        version, and the limit/used/remaining triple of each of the three
        per-minute budgets (challenge issuance, verification, grant
        actions). Every series carries the ``tenant_id`` and
        ``workload_id`` labels; status and classification values are fixed
        English tokens; label values are escaped per the exposition rules;
        every sample is a non-negative decimal integer; and series are
        emitted in a deterministic order (by metric name, then by label).
        The body ends with a single newline.

        The handler never writes: it creates no challenge, consumes no
        capability, rotates no key, appends no audit or lifecycle event,
        and reserves no rate-limit budget; it returns no payload,
        capability, nonce, evidence or key material — only counts. A
        storage failure is a 500 (``metrics unavailable``) and an unusable
        master key configuration is a 500 (``master key configuration
        unavailable``); neither returns a partial exposition.
        """
        # --- request shape (all 422, no state is read) ------------------
        allowed_params = {"tenant_id", "workload_id"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each scope parameter is a single scalar string; a repeated
        # parameter (a multi-valued/list value) is the wrong shape, not a
        # silently last-wins scalar.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # The current master key version classifies the envelope rows; it
        # is key material metadata only (an integer version), never a key.
        # A missing or malformed keyring is a server failure: fail closed
        # with a 500 rather than reporting an unclassified envelope set.
        try:
            current_key_version = load_keyring().current_version
        except MasterKeyError as exc:
            logger.error("master key configuration unavailable: %s", exc)
            raise HTTPException(
                status_code=500, detail="master key configuration unavailable"
            )

        now = _utcnow()
        window_start = _utc_minute_window(now)

        def _grant_count(*predicates):
            return (
                select(func.count())
                .select_from(ReleaseGrant)
                .where(
                    ReleaseGrant.tenant_id == tenant_id,
                    ReleaseGrant.workload_id == workload_id,
                    *predicates,
                )
                .scalar_subquery()
            )

        def _envelope_count(*predicates):
            return (
                select(func.count())
                .select_from(DataEnvelope)
                .where(
                    DataEnvelope.tenant_id == tenant_id,
                    DataEnvelope.workload_id == workload_id,
                    *predicates,
                )
                .scalar_subquery()
            )

        def _used(counter_model):
            return (
                select(counter_model.count)
                .where(
                    counter_model.tenant_id == tenant_id,
                    counter_model.workload_id == workload_id,
                    counter_model.window_start == window_start,
                )
                .scalar_subquery()
            )

        # One statement of independent scalar subqueries evaluates against
        # a single consistent database snapshot on every backend, so the
        # grant, envelope and budget figures can never observe a
        # half-committed change. The handler issues no writes: it consumes
        # no rate-limit slot and appends no audit row.
        metrics_stmt = select(
            _grant_count(
                ReleaseGrant.status == RELEASE_GRANT_STATUS_PENDING
            ).label("pending"),
            _grant_count(
                ReleaseGrant.status == RELEASE_GRANT_STATUS_CONSUMED
            ).label("consumed"),
            _grant_count(
                ReleaseGrant.status == RELEASE_GRANT_STATUS_REVOKED
            ).label("revoked"),
            # Live pending grants are still within their validity window;
            # every other pending grant is expired. The expired figure is
            # derived as pending minus live so the two parts always sum
            # exactly to pending even at the expiry boundary.
            _grant_count(
                ReleaseGrant.status == RELEASE_GRANT_STATUS_PENDING,
                ReleaseGrant.expires_at > now,
            ).label("live_pending"),
            _envelope_count().label("envelopes"),
            # Envelopes recorded at the keyring's current version are
            # migrated; every other recorded version is historical.
            _envelope_count(
                DataEnvelope.key_version == current_key_version
            ).label("current_key_envelopes"),
            _used(ChallengeIssuanceCounter).label("challenge_issuance_used"),
            _used(VerificationAdmissionCounter).label("verification_used"),
            _used(RateLimitCounter).label("grant_actions_used"),
        )
        try:
            with session_factory() as session:
                result = session.execute(metrics_stmt).one()
        except Exception:
            logger.error("metrics query failed")
            raise HTTPException(status_code=500, detail="metrics unavailable")

        pending_count = result.pending
        live_pending_count = result.live_pending
        expired_pending_count = pending_count - live_pending_count
        envelopes_count = result.envelopes
        current_key_envelopes = result.current_key_envelopes
        historical_key_envelopes = envelopes_count - current_key_envelopes

        # A scope with no admitted request this minute has no counter row,
        # so the scalar subquery returns NULL: zero used, the full budget
        # remaining. remaining is never negative.
        budgets = {
            "challenge_issuance": (
                CHALLENGE_ISSUANCE_BUDGET_PER_MINUTE,
                result.challenge_issuance_used or 0,
            ),
            "verification": (
                VERIFICATION_BUDGET_PER_MINUTE,
                result.verification_used or 0,
            ),
            "grant_actions": (
                GRANT_BUDGET_PER_MINUTE,
                result.grant_actions_used or 0,
            ),
        }

        # Fixed metric names, help strings and classification label values:
        # the exposition contract is stable, so a scraper can rely on every
        # series below always being present (an empty scope reports zeros,
        # never omitted series).
        metric_help = {
            "proof_release_data_envelopes_total": (
                "Data envelopes stored for the scope, split by master key "
                "version currency."
            ),
            "proof_release_pending_release_grants": (
                "Pending release grants in the scope, split by "
                "validity-window state."
            ),
            "proof_release_rate_limit_budget_limit": (
                "Per-minute admission budget limit for the scope."
            ),
            "proof_release_rate_limit_budget_remaining": (
                "Per-minute admission budget still available to the scope "
                "in the current window."
            ),
            "proof_release_rate_limit_budget_used": (
                "Per-minute admission budget already spent by the scope in "
                "the current window."
            ),
            "proof_release_release_grants_total": (
                "Release grants in the scope, split by status."
            ),
        }
        # (metric name, classifying labels, value); the scope labels are
        # attached to every series at render time.
        series = [
            (
                "proof_release_release_grants_total",
                {"status": "pending"},
                pending_count,
            ),
            (
                "proof_release_release_grants_total",
                {"status": "consumed"},
                result.consumed,
            ),
            (
                "proof_release_release_grants_total",
                {"status": "revoked"},
                result.revoked,
            ),
            (
                "proof_release_pending_release_grants",
                {"state": "live"},
                live_pending_count,
            ),
            (
                "proof_release_pending_release_grants",
                {"state": "expired"},
                expired_pending_count,
            ),
            (
                "proof_release_data_envelopes_total",
                {"key_version": "current"},
                current_key_envelopes,
            ),
            (
                "proof_release_data_envelopes_total",
                {"key_version": "historical"},
                historical_key_envelopes,
            ),
        ]
        for budget_name, (limit, used) in budgets.items():
            series.append(
                (
                    "proof_release_rate_limit_budget_limit",
                    {"budget": budget_name},
                    limit,
                )
            )
            series.append(
                (
                    "proof_release_rate_limit_budget_used",
                    {"budget": budget_name},
                    used,
                )
            )
            series.append(
                (
                    "proof_release_rate_limit_budget_remaining",
                    {"budget": budget_name},
                    max(0, limit - used),
                )
            )

        # Deterministic output: metrics in name order, series within a
        # metric in label order, labels inside a series in name order.
        # Every value is a Python int produced by SQL count aggregation or
        # a fixed budget constant, so each sample renders as a non-negative
        # decimal integer.
        scope_labels = {"tenant_id": tenant_id, "workload_id": workload_id}
        lines = []
        for metric_name in sorted(metric_help):
            lines.append(f"# HELP {metric_name} {metric_help[metric_name]}")
            lines.append(f"# TYPE {metric_name} gauge")
            metric_series = sorted(
                (s for s in series if s[0] == metric_name),
                key=lambda s: sorted(s[1].items()),
            )
            for _, classifying, value in metric_series:
                rendered = ",".join(
                    f'{key}="{_prom_label_escape(val)}"'
                    for key, val in sorted({**scope_labels, **classifying}.items())
                )
                lines.append(f"{metric_name}{{{rendered}}} {value}")
        body = ("\n".join(lines) + "\n").encode("utf-8")
        return Response(
            content=body,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/v1/compliance/audit-events")
    def list_compliance_audit_events(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        event_id: str | None = Query(default=None),
        event_type: str | None = Query(default=None),
        status: str | None = Query(default=None),
        occurred_after: str | None = Query(default=None),
        occurred_before: str | None = Query(default=None),
        cursor: str | None = Query(default=None),
    ) -> Response:
        """Return a read-only, tenant-isolated page of compliance events.

        Events are listed in stable ``(occurred_at, event_id)`` ascending
        order with an exclusive keyset cursor. The cursor is
        HMAC-authenticated, carries its own kind tag and is bound to the
        scope *and* every active filter, so it cannot be forged, tampered
        with, or replayed against a different scope or filter set. The
        handler issues only SELECTs — events are written by the
        grant/consume/revoke/release/rewrap transactions and never here —
        so a query observes only committed state and returns no half page.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {
            "tenant_id",
            "workload_id",
            "event_id",
            "event_type",
            "status",
            "occurred_after",
            "occurred_before",
            "cursor",
        }
        if set(request.query_params.keys()) - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )

        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # event_id is a canonical lowercase UUID; surrounding whitespace
        # and uppercase letters are format errors.
        event_id_filter: str | None = None
        if event_id is not None:
            if not event_id.strip() or not _UUID_RE.fullmatch(event_id.lower()):
                raise HTTPException(status_code=422, detail="invalid event identifier")
            event_id_filter = event_id.lower()

        if event_type is not None:
            if not event_type.strip() or event_type not in AUDIT_EVENT_TYPE_CODES:
                raise HTTPException(status_code=422, detail="invalid event_type")

        if status is not None:
            if not status.strip() or status not in AUDIT_EVENT_STATUS_CODES:
                raise HTTPException(status_code=422, detail="invalid status")

        type_filter = event_type if event_type is not None else ""
        status_filter = status if status is not None else ""

        def _time_bound(value: str | None, name: str) -> tuple[str, datetime | None]:
            if value is None:
                return "", None
            if not value.strip():
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            try:
                parsed = _parse_utc_rfc3339(value)
            except ValueError:
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            # Normalize the spelling embedded into the cursor so two
            # equivalent UTC spellings cannot mint different cursor domains.
            return _rfc3339(parsed), parsed

        after_raw, after_dt = _time_bound(occurred_after, "occurred_after")
        before_raw, before_dt = _time_bound(occurred_before, "occurred_before")
        # Equality is a valid single-instant window; the start must not be
        # later than the end.
        if after_dt is not None and before_dt is not None and after_dt > before_dt:
            raise HTTPException(
                status_code=422,
                detail="occurred_after must not be later than occurred_before",
            )

        # Omitted cursor or an explicit empty string starts before the
        # smallest event. Whitespace, malformed, forged, cross-scope or
        # cross-filter cursors are indistinguishable 422s.
        boundary_dt: datetime | None = None
        boundary_event: str | None = None
        if cursor is not None and cursor != "":
            if not cursor.strip() or not _CURSOR_RE.fullmatch(cursor):
                raise HTTPException(status_code=422, detail="invalid cursor")
            decoded_boundary = _decode_audit_event_cursor(
                cursor,
                tenant_id,
                workload_id,
                event_id=event_id_filter or "",
                event_type=type_filter,
                status=status_filter,
                occurred_after=after_raw,
                occurred_before=before_raw,
            )
            if decoded_boundary is None:
                raise HTTPException(status_code=422, detail="invalid cursor")
            boundary_at_raw, boundary_event = decoded_boundary
            try:
                boundary_dt = _parse_utc_rfc3339(boundary_at_raw)
            except ValueError:
                raise HTTPException(status_code=422, detail="invalid cursor")

        # --- read-only audit scan ---------------------------------------
        try:
            with session_factory() as session:
                # An explicitly named event must exist in exactly this
                # scope; unknown and cross-scope identifiers are an
                # indistinguishable 404 rather than an empty page.
                if event_id_filter is not None:
                    named = session.get(AuditEvent, event_id_filter)
                    if (
                        named is None
                        or named.tenant_id != tenant_id
                        or named.workload_id != workload_id
                    ):
                        raise HTTPException(
                            status_code=404, detail="audit event not found"
                        )

                stmt = select(AuditEvent).where(
                    AuditEvent.tenant_id == tenant_id,
                    AuditEvent.workload_id == workload_id,
                )
                if event_id_filter is not None:
                    stmt = stmt.where(AuditEvent.event_id == event_id_filter)
                if type_filter:
                    stmt = stmt.where(AuditEvent.event_type == type_filter)
                if status_filter:
                    stmt = stmt.where(AuditEvent.status == status_filter)
                if after_dt is not None:
                    stmt = stmt.where(AuditEvent.occurred_at >= after_dt)
                if before_dt is not None:
                    stmt = stmt.where(AuditEvent.occurred_at <= before_dt)
                if boundary_dt is not None:
                    # Exclusive (occurred_at, event_id) keyset position.
                    stmt = stmt.where(
                        or_(
                            AuditEvent.occurred_at > boundary_dt,
                            and_(
                                AuditEvent.occurred_at == boundary_dt,
                                AuditEvent.event_id > boundary_event,
                            ),
                        )
                    )
                stmt = (
                    stmt.order_by(
                        AuditEvent.occurred_at.asc(),
                        AuditEvent.event_id.asc(),
                    )
                    .limit(AUDIT_EVENT_PAGE_SIZE + 1)
                )
                # One extra row is the "more follows" probe. A storage
                # failure aborts the whole request with a 500 rather than
                # returning a partial page.
                rows = list(session.scalars(stmt))
        except HTTPException:
            raise
        except Exception:
            logger.error("compliance audit event query failed")
            raise HTTPException(
                status_code=500, detail="compliance audit unavailable"
            )

        has_more = len(rows) > AUDIT_EVENT_PAGE_SIZE
        page = rows[:AUDIT_EVENT_PAGE_SIZE]

        events = [
            {
                "event_id": row.event_id,
                "event_type": row.event_type,
                "grant_id": row.grant_id,
                "decision_id": row.decision_id,
                "data_id": row.data_id,
                "status": row.status,
                "occurred_at": _rfc3339(row.occurred_at),
                # Grant events expose only the SHA-256 digest; rewrap
                # events carry no grant identity and so expose null.
                "capability_sha256": row.capability_sha256,
            }
            for row in page
        ]

        if has_more:
            last = page[-1]
            next_cursor = _encode_audit_event_cursor(
                tenant_id,
                workload_id,
                _rfc3339(last.occurred_at),
                last.event_id,
                event_id=event_id_filter or "",
                event_type=type_filter,
                status=status_filter,
                occurred_after=after_raw,
                occurred_before=before_raw,
            )
            complete = False
        else:
            next_cursor = ""
            complete = True

        # Compact JSON with a single terminating newline. Every value is a
        # string, null or boolean — no floats, -0.0 or non-finite values.
        body = (
            json.dumps(
                {
                    "events": events,
                    "next_cursor": next_cursor,
                    "complete": complete,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/compliance/audit-events/integrity")
    def verify_compliance_audit_integrity(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Verify the tamper-evident hash chain of one scope's audit events.

        The range is fixed entirely by the two mandatory, non-blank query
        parameters; the request carries no body. Every shape failure (a
        missing, blank or repeated scope parameter, an unknown parameter,
        or a non-empty body) is a 422 raised before any state is read and
        writes no audit. On success the scope's committed events are
        walked in chain order: the per-scope sequence must be the
        contiguous 1..N run, every event hash must recompute from its
        stored non-sensitive fields and its predecessor's hash, and the
        recorded range head must agree with the chain tip. A broken chain
        is still a 200 with ``valid`` false and exactly one of the fixed
        failure codes (``sequence-gap``, ``hash-mismatch``,
        ``head-mismatch``) plus the sequence number at which verification
        failed; the response never carries capabilities, payloads,
        evidence or keys.

        The handler issues only SELECTs against one consistent committed
        snapshot: it never repairs the chain, rewrites an event or
        advances the range head, so with no new committed events a
        repeated request returns byte-identical bytes, and a concurrent
        append is observed either fully before or fully after its commit.
        A storage or verification failure is a 500 with no partial answer.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {"tenant_id", "workload_id"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each scope parameter is a single scalar string; a repeated
        # parameter (a multi-valued/list value) is the wrong shape, not a
        # silently last-wins scalar.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(status_code=422, detail="duplicate query parameter")
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # --- read-only chain snapshot -----------------------------------
        try:
            with session_factory() as session:
                rows = list(
                    session.scalars(
                        select(AuditEvent)
                        .where(
                            AuditEvent.tenant_id == tenant_id,
                            AuditEvent.workload_id == workload_id,
                        )
                        .order_by(
                            AuditEvent.chain_seq.asc(),
                            AuditEvent.event_id.asc(),
                        )
                    )
                )
                head = session.get(AuditChainHead, (tenant_id, workload_id))
        except HTTPException:
            raise
        except Exception:
            logger.error("audit integrity snapshot failed")
            raise HTTPException(
                status_code=500, detail="audit integrity unavailable"
            )

        try:
            report = _verify_audit_event_chain(rows, head)
        except Exception:
            logger.error("audit integrity verification failed")
            raise HTTPException(
                status_code=500, detail="audit integrity unavailable"
            )

        # Compact JSON with a single terminating newline, fields in the
        # fixed contract order. Every value is a string, integer, boolean
        # or null — no floats, -0.0 or non-finite values.
        body = (
            json.dumps(
                {
                    "tenant_id": tenant_id,
                    "workload_id": workload_id,
                    "valid": report["valid"],
                    "event_count": report["event_count"],
                    "legacy_count": report["legacy_count"],
                    "first_seq": report["first_seq"],
                    "last_seq": report["last_seq"],
                    "head_hash": report["head_hash"],
                    "failure_code": report["failure_code"],
                    "failure_seq": report["failure_seq"],
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/compliance/proof-events")
    def list_compliance_proof_events(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        evidence_id: str | None = Query(default=None),
        event_type: str | None = Query(default=None),
        status: str | None = Query(default=None),
        occurred_after: str | None = Query(default=None),
        occurred_before: str | None = Query(default=None),
        cursor: str | None = Query(default=None),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return a read-only, tenant-isolated page of proof-lifecycle events.

        Each traced proof contributes at most one event per stage:
        ``proof-received`` at reception, ``proof-verified`` at its first
        verification settlement (``verified``/``rejected``) and
        ``proof-decision`` at its first policy decision (recording the
        policy version and ``allowed``/``denied``). Events are written in
        the same committed transactions as those stages — never here — and
        retries or lost settlement races leave no duplicate rows.

        Events are listed in stable ``(occurred_at, event_id)`` ascending
        order with an exclusive keyset cursor. The first (cursor-less or
        empty-cursor) query fixes a replayable snapshot at the greatest
        per-scope business commit sequence among the currently committed,
        in-filter events; events committed afterwards surface only in a
        fresh first query, never in later pages of this snapshot. The
        cutoff is a sequence allocated inside each event's own write
        transaction, so it tracks the business commit boundary on every
        backend without relying on write timing or equal
        ``occurred_at`` values — an event committed after the first query
        is excluded even when its business time is older. The cursor is
        HMAC-authenticated, carries its own kind tag and is bound to the
        scope, every active filter *and* the fixed snapshot, so it cannot
        be forged, tampered with, or replayed against another scope,
        filter set, snapshot or cursor family. The handler issues only
        SELECTs, so a query observes only committed state, appends no
        audit and changes no business state; a storage failure aborts the
        whole request with a 500 rather than returning a half page.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {
            "tenant_id",
            "workload_id",
            "evidence_id",
            "event_type",
            "status",
            "occurred_after",
            "occurred_before",
            "cursor",
        }
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is rejected rather than silently
        # treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )

        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # The explicit proof identifier is a canonical lowercase UUID;
        # surrounding whitespace and uppercase letters are format errors.
        evidence_id_filter: str | None = None
        if evidence_id is not None:
            if not evidence_id.strip() or not _UUID_RE.fullmatch(evidence_id):
                raise HTTPException(
                    status_code=422, detail="invalid evidence identifier"
                )
            evidence_id_filter = evidence_id

        if event_type is not None:
            if not event_type.strip() or event_type not in PROOF_EVENT_TYPE_CODES:
                raise HTTPException(status_code=422, detail="invalid event_type")

        if status is not None:
            if not status.strip() or status not in PROOF_EVENT_STATUS_CODES:
                raise HTTPException(status_code=422, detail="invalid status")

        type_filter = event_type if event_type is not None else ""
        status_filter = status if status is not None else ""

        def _time_bound(value: str | None, name: str) -> tuple[str, datetime | None]:
            if value is None:
                return "", None
            if not value.strip():
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            try:
                parsed = _parse_utc_rfc3339(value)
            except ValueError:
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            # Normalize the spelling embedded into the cursor so two
            # equivalent UTC spellings cannot mint different cursor domains.
            return _rfc3339(parsed), parsed

        after_raw, after_dt = _time_bound(occurred_after, "occurred_after")
        before_raw, before_dt = _time_bound(occurred_before, "occurred_before")
        # Equality is a valid single-instant window; the start must not be
        # later than the end.
        if after_dt is not None and before_dt is not None and after_dt > before_dt:
            raise HTTPException(
                status_code=422,
                detail="occurred_after must not be later than occurred_before",
            )

        # Omitted cursor or an explicit empty string starts before the
        # smallest event and fixes the timeline's replayable snapshot.
        # Whitespace, malformed, forged, cross-scope, cross-filter,
        # cross-snapshot or foreign-kind cursors are indistinguishable 422s.
        boundary_dt: datetime | None = None
        boundary_event: str | None = None
        # Per-scope commit-order high-water mark fixed by the first query.
        # None marks the first query of a range; a resume cursor carries the
        # established value bound by its HMAC.
        snapshot_seq: int | None = None
        if cursor is not None and cursor != "":
            if not cursor.strip() or not _CURSOR_RE.fullmatch(cursor):
                raise HTTPException(status_code=422, detail="invalid cursor")
            decoded_boundary = _decode_proof_event_cursor(
                cursor,
                tenant_id,
                workload_id,
                evidence_id=evidence_id_filter or "",
                event_type=type_filter,
                status=status_filter,
                occurred_after=after_raw,
                occurred_before=before_raw,
            )
            if decoded_boundary is None:
                raise HTTPException(status_code=422, detail="invalid cursor")
            boundary_at_raw, boundary_event, snapshot_seq = decoded_boundary
            try:
                boundary_dt = _parse_utc_rfc3339(boundary_at_raw)
            except ValueError:
                raise HTTPException(status_code=422, detail="invalid cursor")

        # --- read-only proof timeline scan ------------------------------
        rows: list = []
        try:
            with session_factory() as session:
                # An explicitly named proof must exist in exactly this
                # tenant/workload; an unknown or cross-scope identifier is
                # an indistinguishable 404 rather than an empty page.
                if evidence_id_filter is not None:
                    named = session.get(Evidence, evidence_id_filter)
                    if (
                        named is None
                        or named.tenant_id != tenant_id
                        or named.workload_id != workload_id
                    ):
                        raise HTTPException(
                            status_code=404, detail="evidence not found"
                        )

                def _filtered(stmt):
                    if evidence_id_filter is not None:
                        stmt = stmt.where(
                            ProofLifecycleEvent.evidence_id
                            == evidence_id_filter
                        )
                    if type_filter:
                        stmt = stmt.where(
                            ProofLifecycleEvent.event_type == type_filter
                        )
                    if status_filter:
                        stmt = stmt.where(
                            ProofLifecycleEvent.status == status_filter
                        )
                    if after_dt is not None:
                        stmt = stmt.where(
                            ProofLifecycleEvent.occurred_at >= after_dt
                        )
                    if before_dt is not None:
                        stmt = stmt.where(
                            ProofLifecycleEvent.occurred_at <= before_dt
                        )
                    return stmt

                if snapshot_seq is None:
                    # First query of the range: fix a replayable snapshot at
                    # the greatest per-scope commit sequence among the
                    # currently committed, in-filter events. ``commit_seq`` is
                    # allocated inside each event's own write transaction and
                    # is strictly increasing in business commit order on
                    # every backend, so an event that commits after this
                    # point takes a greater sequence and can never enter the
                    # snapshot even when its business time is older or equal
                    # to an already-seen event. This is independent of write
                    # timing, equal timestamps and the particular dialect.
                    fixed_seq = session.scalar(
                        _filtered(
                            select(func.max(ProofLifecycleEvent.commit_seq)).where(
                                ProofLifecycleEvent.tenant_id == tenant_id,
                                ProofLifecycleEvent.workload_id
                                == workload_id,
                            )
                        )
                    )
                    if fixed_seq is None:
                        # No in-filter event at snapshot time: the fixed
                        # first page is empty and already complete; later
                        # commits belong to fresh first queries.
                        rows = []
                        snapshot_seq = 0
                    else:
                        snapshot_seq = int(fixed_seq)

                if snapshot_seq > 0:
                    stmt = _filtered(
                        select(ProofLifecycleEvent).where(
                            ProofLifecycleEvent.tenant_id == tenant_id,
                            ProofLifecycleEvent.workload_id == workload_id,
                            # Inclusive fixed-snapshot cutoff by business
                            # commit order.
                            ProofLifecycleEvent.commit_seq <= snapshot_seq,
                        )
                    )
                    if boundary_dt is not None:
                        # Exclusive (occurred_at, event_id) keyset. Snapshot
                        # membership is fixed above; the boundary only walks
                        # the same immutable set in business-time order.
                        stmt = stmt.where(
                            or_(
                                ProofLifecycleEvent.occurred_at > boundary_dt,
                                and_(
                                    ProofLifecycleEvent.occurred_at
                                    == boundary_dt,
                                    ProofLifecycleEvent.event_id
                                    > boundary_event,
                                ),
                            )
                        )
                    stmt = stmt.order_by(
                        ProofLifecycleEvent.occurred_at.asc(),
                        ProofLifecycleEvent.event_id.asc(),
                    ).limit(PROOF_EVENT_PAGE_SIZE + 1)
                    # One extra row is the "more follows" probe. The scan
                    # is a single read-only statement: a storage failure
                    # aborts the whole request with a 500 rather than
                    # returning a partial page.
                    rows = list(session.scalars(stmt))
        except HTTPException:
            raise
        except Exception:
            logger.error("proof lifecycle event query failed")
            raise HTTPException(
                status_code=500, detail="proof event audit unavailable"
            )

        has_more = len(rows) > PROOF_EVENT_PAGE_SIZE
        page = rows[:PROOF_EVENT_PAGE_SIZE]

        events = [
            {
                "event_id": row.event_id,
                "evidence_id": row.evidence_id,
                "event_type": row.event_type,
                "status": row.status,
                # Only the reception event carries the associated format
                # descriptor; later stages expose null.
                "evidence_format": row.evidence_format,
                # Only the first decision carries the policy version; an
                # integer or null — never a float.
                "policy_version": row.policy_version,
                "occurred_at": _rfc3339(row.occurred_at),
            }
            for row in page
        ]

        if has_more:
            last = page[-1]
            next_cursor = _encode_proof_event_cursor(
                tenant_id,
                workload_id,
                _rfc3339(last.occurred_at),
                last.event_id,
                evidence_id=evidence_id_filter or "",
                event_type=type_filter,
                status=status_filter,
                occurred_after=after_raw,
                occurred_before=before_raw,
                snapshot_seq=snapshot_seq,
            )
            complete = False
        else:
            next_cursor = ""
            complete = True

        # Compact JSON with a single terminating newline. Every value is a
        # string, null, boolean or integer — no floats, -0.0 or non-finite
        # values. No evidence, nonce, claims, capability, payload or key
        # material ever appears.
        body = (
            json.dumps(
                {
                    "events": events,
                    "next_cursor": next_cursor,
                    "complete": complete,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/compliance/decisions")
    def list_compliance_decisions(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        decision_id: str | None = Query(default=None),
        evidence_id: str | None = Query(default=None),
        policy_id: str | None = Query(default=None),
        status: str | None = Query(default=None),
        decided_after: str | None = Query(default=None),
        decided_before: str | None = Query(default=None),
        cursor: str | None = Query(default=None),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return a read-only, tenant-isolated page of release decisions.

        Unlike the proof-lifecycle timeline, which records only the first
        decision for an evidence, this listing covers every policy version
        decided against the same evidence: one immutable row per
        (evidence, policy version). Decisions are written in the decision
        transaction — never here — and this handler issues only SELECTs,
        appends no audit and changes no business state.

        Decisions are listed in stable ``(decided_at, decision_id)``
        ascending order with an exclusive keyset cursor. The first
        (cursor-less or empty-cursor) query fixes a replayable snapshot at
        the greatest per-scope business commit sequence among the
        currently committed, in-filter decisions; decisions committed
        afterwards surface only in a fresh first query, never in later
        pages of this snapshot. The cutoff is a sequence allocated inside
        each decision's own write transaction, so it tracks the business
        commit boundary on every backend without relying on write timing
        or equal ``decided_at`` values — a decision committed after the
        first query is excluded even when its business time is older. The
        cursor is HMAC-authenticated, carries its own kind tag and is
        bound to the scope, every active filter *and* the fixed snapshot,
        so it cannot be forged, tampered with, or replayed against another
        scope, filter set, snapshot or cursor family. A storage failure
        aborts the whole request with a 500 rather than returning a half
        page.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {
            "tenant_id",
            "workload_id",
            "decision_id",
            "evidence_id",
            "policy_id",
            "status",
            "decided_after",
            "decided_before",
            "cursor",
        }
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is rejected rather than silently
        # treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )

        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # Every explicit identifier is a canonical lowercase UUID;
        # surrounding whitespace and uppercase letters are format errors.
        def _identifier(value: str | None, name: str) -> str | None:
            if value is None:
                return None
            if not value.strip() or not _UUID_RE.fullmatch(value):
                raise HTTPException(
                    status_code=422, detail=f"invalid {name} identifier"
                )
            return value

        decision_id_filter = _identifier(decision_id, "decision")
        evidence_id_filter = _identifier(evidence_id, "evidence")
        policy_id_filter = _identifier(policy_id, "policy")

        if status is not None:
            if not status.strip() or status not in DECISION_STATUS_CODES:
                raise HTTPException(status_code=422, detail="invalid status")
        status_filter = status if status is not None else ""

        def _time_bound(value: str | None, name: str) -> tuple[str, datetime | None]:
            if value is None:
                return "", None
            if not value.strip():
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            try:
                parsed = _parse_utc_rfc3339(value)
            except ValueError:
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            # Normalize the spelling embedded into the cursor so two
            # equivalent UTC spellings cannot mint different cursor domains.
            return _rfc3339(parsed), parsed

        after_raw, after_dt = _time_bound(decided_after, "decided_after")
        before_raw, before_dt = _time_bound(decided_before, "decided_before")
        # Equality is a valid single-instant window; the start must not be
        # later than the end.
        if after_dt is not None and before_dt is not None and after_dt > before_dt:
            raise HTTPException(
                status_code=422,
                detail="decided_after must not be later than decided_before",
            )

        # Omitted cursor or an explicit empty string starts before the
        # smallest decision and fixes the listing's replayable snapshot.
        # Whitespace, malformed, forged, cross-scope, cross-filter,
        # cross-snapshot or foreign-kind cursors are indistinguishable 422s.
        boundary_dt: datetime | None = None
        boundary_decision: str | None = None
        # Per-scope commit-order high-water mark fixed by the first query.
        # None marks the first query of a range; a resume cursor carries the
        # established value bound by its HMAC.
        snapshot_seq: int | None = None
        if cursor is not None and cursor != "":
            if not cursor.strip() or not _CURSOR_RE.fullmatch(cursor):
                raise HTTPException(status_code=422, detail="invalid cursor")
            decoded_boundary = _decode_decision_cursor(
                cursor,
                tenant_id,
                workload_id,
                decision_id=decision_id_filter or "",
                evidence_id=evidence_id_filter or "",
                policy_id=policy_id_filter or "",
                status=status_filter,
                decided_after=after_raw,
                decided_before=before_raw,
            )
            if decoded_boundary is None:
                raise HTTPException(status_code=422, detail="invalid cursor")
            boundary_at_raw, boundary_decision, snapshot_seq = decoded_boundary
            try:
                boundary_dt = _parse_utc_rfc3339(boundary_at_raw)
            except ValueError:
                raise HTTPException(status_code=422, detail="invalid cursor")

        # --- read-only decision scan ------------------------------------
        rows: list = []
        try:
            with session_factory() as session:
                # Explicitly named identifiers must each exist in exactly
                # this tenant/workload; an unknown or cross-scope decision,
                # evidence or policy identifier is an indistinguishable
                # 404 rather than an empty page.
                if decision_id_filter is not None:
                    named = session.get(Decision, decision_id_filter)
                    if (
                        named is None
                        or named.tenant_id != tenant_id
                        or named.workload_id != workload_id
                    ):
                        raise HTTPException(
                            status_code=404, detail="decision not found"
                        )
                if evidence_id_filter is not None:
                    named = session.get(Evidence, evidence_id_filter)
                    if (
                        named is None
                        or named.tenant_id != tenant_id
                        or named.workload_id != workload_id
                    ):
                        raise HTTPException(
                            status_code=404, detail="evidence not found"
                        )
                if policy_id_filter is not None:
                    named = session.get(Policy, policy_id_filter)
                    if (
                        named is None
                        or named.tenant_id != tenant_id
                        or named.workload_id != workload_id
                    ):
                        raise HTTPException(
                            status_code=404, detail="policy not found"
                        )

                def _filtered(stmt):
                    if decision_id_filter is not None:
                        stmt = stmt.where(Decision.decision_id == decision_id_filter)
                    if evidence_id_filter is not None:
                        stmt = stmt.where(Decision.evidence_id == evidence_id_filter)
                    if policy_id_filter is not None:
                        stmt = stmt.where(Decision.policy_id == policy_id_filter)
                    if status_filter:
                        stmt = stmt.where(Decision.status == status_filter)
                    if after_dt is not None:
                        stmt = stmt.where(Decision.decided_at >= after_dt)
                    if before_dt is not None:
                        stmt = stmt.where(Decision.decided_at <= before_dt)
                    return stmt

                if snapshot_seq is None:
                    # First query of the range: fix a replayable snapshot at
                    # the greatest per-scope commit sequence among the
                    # currently committed, in-filter decisions. commit_seq is
                    # allocated inside each decision's own write transaction
                    # and is strictly increasing in business commit order on
                    # every backend, so a decision that commits after this
                    # point takes a greater sequence and can never enter the
                    # snapshot even when its business time is older or equal
                    # to an already-seen decision.
                    fixed_seq = session.scalar(
                        _filtered(
                            select(func.max(Decision.commit_seq)).where(
                                Decision.tenant_id == tenant_id,
                                Decision.workload_id == workload_id,
                            )
                        )
                    )
                    if fixed_seq is None:
                        # No in-filter decision at snapshot time: the fixed
                        # first page is empty and already complete; later
                        # commits belong to fresh first queries.
                        rows = []
                        snapshot_seq = 0
                    else:
                        snapshot_seq = int(fixed_seq)

                if snapshot_seq > 0:
                    stmt = _filtered(
                        select(Decision).where(
                            Decision.tenant_id == tenant_id,
                            Decision.workload_id == workload_id,
                            # Inclusive fixed-snapshot cutoff by business
                            # commit order.
                            Decision.commit_seq <= snapshot_seq,
                        )
                    )
                    if boundary_dt is not None:
                        # Exclusive (decided_at, decision_id) keyset.
                        # Snapshot membership is fixed above; the boundary
                        # only walks the same immutable set in business-time
                        # order.
                        stmt = stmt.where(
                            or_(
                                Decision.decided_at > boundary_dt,
                                and_(
                                    Decision.decided_at == boundary_dt,
                                    Decision.decision_id > boundary_decision,
                                ),
                            )
                        )
                    stmt = stmt.order_by(
                        Decision.decided_at.asc(),
                        Decision.decision_id.asc(),
                    ).limit(DECISION_PAGE_SIZE + 1)
                    # One extra row is the "more follows" probe. The scan is
                    # a single read-only statement: a storage failure aborts
                    # the whole request with a 500 rather than returning a
                    # partial page.
                    rows = list(session.scalars(stmt))
        except HTTPException:
            raise
        except Exception:
            logger.error("compliance decision query failed")
            raise HTTPException(
                status_code=500, detail="compliance decisions unavailable"
            )

        has_more = len(rows) > DECISION_PAGE_SIZE
        page = rows[:DECISION_PAGE_SIZE]

        decisions = [
            {
                "decision_id": row.decision_id,
                "evidence_id": row.evidence_id,
                "policy_id": row.policy_id,
                # The immutable version this decision was taken against; an
                # integer — never a float.
                "policy_version": row.policy_version,
                "status": row.status,
                "decided_at": _rfc3339(row.decided_at),
            }
            for row in page
        ]

        if has_more:
            last = page[-1]
            next_cursor = _encode_decision_cursor(
                tenant_id,
                workload_id,
                _rfc3339(last.decided_at),
                last.decision_id,
                decision_id=decision_id_filter or "",
                evidence_id=evidence_id_filter or "",
                policy_id=policy_id_filter or "",
                status=status_filter,
                decided_after=after_raw,
                decided_before=before_raw,
                snapshot_seq=snapshot_seq,
            )
            complete = False
        else:
            next_cursor = ""
            complete = True

        # Compact JSON with a single terminating newline. Every value is a
        # string, boolean or integer — no floats, -0.0 or non-finite
        # values. No evidence, nonce, claims, capability, payload, key or
        # exception text ever appears.
        body = (
            json.dumps(
                {
                    "decisions": decisions,
                    "next_cursor": next_cursor,
                    "complete": complete,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/decisions//trace")
    def get_decision_trace_identifier_required() -> Response:
        # An empty path segment is a missing decision identifier: a 422
        # client error rather than a routing-level 404. It never reads a
        # decision or its policy version.
        raise HTTPException(status_code=422, detail="invalid decision identifier")

    @app.get("/v1/decisions/{decision_id}/trace")
    def get_decision_trace(
        request: Request,
        decision_id: str,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Trace one settled decision back to its policy version snapshot.

        The response joins one decision's immutable audit conclusion to the
        exact policy version (and its immutable rule tree) that was used at
        decision time. Every validation failure is a 422 returned before
        any state is read: the body must be missing or zero-length (any
        other bytes, including whitespace, are rejected), exactly one
        non-blank ``tenant_id`` and ``workload_id`` may be supplied as
        query parameters (an unknown or repeated parameter is a 422), and
        the path id must be a canonical lowercase UUID with no surrounding
        whitespace. A decision that is unknown or outside the named scope
        is one indistinguishable 404 — existence in any other tenant or
        workload is never revealed.

        The handler issues only SELECTs of committed state: it never
        rewrites a rule, decided_at or status, appends no audit or event,
        creates no decision, and returns no proof text, nonce, claim
        value, capability, payload or key. A policy version retired after
        the decision still answers with the same stored rule snapshot. A
        storage failure is a 500 with the half-built result discarded
        entirely. Results persist across restarts.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {"tenant_id", "workload_id"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is rejected rather than silently
        # treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )

        # The raw path value must be a canonical lowercase UUID: missing
        # (handled by the dedicated empty-segment route above), blank,
        # whitespace-padded, uppercase or otherwise non-canonical values
        # are format errors rejected before any state is read.
        if not decision_id or not _UUID_RE.fullmatch(decision_id):
            raise HTTPException(
                status_code=422, detail="invalid decision identifier"
            )
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # --- read-only decision + policy-version snapshot ---------------
        try:
            with session_factory() as session:
                decision = session.get(Decision, decision_id)
                if (
                    decision is None
                    or decision.tenant_id != tenant_id
                    or decision.workload_id != workload_id
                ):
                    # Do not reveal whether an unknown or out-of-scope
                    # decision exists under another scope: both share one
                    # indistinguishable 404.
                    raise HTTPException(
                        status_code=404, detail="decision not found"
                    )

                policy = session.get(Policy, decision.policy_id)
                # Policy versions are never deleted, so a missing or
                # out-of-scope version here is an integrity/storage
                # failure, never a client error: discard the half result
                # and answer 500 rather than returning a trace without the
                # rule snapshot or fabricating one.
                if (
                    policy is None
                    or policy.tenant_id != tenant_id
                    or policy.workload_id != workload_id
                ):
                    raise HTTPException(
                        status_code=500,
                        detail="decision trace unavailable",
                    )

                decision_view = {
                    # Original identifiers and audit conclusion only.
                    "decision_id": decision.decision_id,
                    # The associated proof (evidence) this decision was
                    # taken against — an identifier, never the proof
                    # itself.
                    "evidence_id": decision.evidence_id,
                    # The integer policy version decided against.
                    "policy_version": decision.policy_version,
                    # The fixed allowed/denied conclusion code.
                    "status": decision.status,
                    # The original UTC decision time, never rewritten.
                    "decided_at": _rfc3339(decision.decided_at),
                }
                policy_view = {
                    # Version identifier, name, integer version and the
                    # immutable rule tree exactly as persisted when this
                    # version was created — retiring the version never
                    # changes it, so a settled decision keeps tracing to
                    # the conditions actually used.
                    "policy_id": policy.policy_id,
                    "name": policy.name,
                    "version": policy.version,
                    # Re-serialized from the persisted canonical JSON
                    # exactly like the policy lifecycle query: rule
                    # numbers round-trip verbatim (integers stay ints,
                    # decimals and -0.0 keep their submitted form) and no
                    # other metadata can produce a float or non-finite
                    # value, since allow_nan=False guards the dump.
                    "rule": json.loads(policy.rule_json),
                }
        except HTTPException:
            raise
        except Exception:
            # Fixed message only: exception text might carry protected
            # material and is never logged or returned.
            logger.error("decision trace query failed")
            raise HTTPException(
                status_code=500, detail="decision trace unavailable"
            )

        # Compact JSON with a single terminating newline. Outside the
        # rule, every value is a JSON string or an integer — no floats,
        # -0.0 or non-finite values. No evidence text, nonce, claims
        # value, capability, payload, key or exception text appears.
        body = (
            json.dumps(
                {"decision": decision_view, "policy_version": policy_view},
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/decisions//evaluation")
    def get_decision_evaluation_identifier_required() -> Response:
        # An empty path segment is a missing decision identifier: a 422
        # client error rather than a routing-level 404. It never reads a
        # decision or its evaluation.
        raise HTTPException(status_code=422, detail="invalid decision identifier")

    @app.get("/v1/decisions/{decision_id}/evaluation")
    def get_decision_evaluation(
        request: Request,
        decision_id: str,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return one decision's persisted policy-rule evaluation.

        The explanation was written once, in full, in the same transaction
        as the decision and is never rewritten: retiring or updating the
        policy, changing an identity or revocation, or rotating keys leaves
        these stored booleans untouched, so the answer is always the
        evaluation actually used at decision time. Every validation
        failure is a 422 returned before any state is read: the body must
        be missing or zero-length (any other bytes, including whitespace,
        are rejected), exactly one non-blank ``tenant_id`` and
        ``workload_id`` may be supplied as query parameters (an unknown or
        repeated parameter is a 422), and the path id must be a canonical
        lowercase UUID with no surrounding whitespace. A decision that is
        unknown or outside the named scope is one indistinguishable 404 —
        existence in any other tenant or workload is never revealed. A
        decision committed by an older deployment that never recorded an
        explanation answers 409 ``evaluation not recorded``: history is
        not reconstructed, guessed or backfilled on read.

        The handler issues only SELECTs of committed state: it never
        rewrites a node, status or decided_at, appends no audit or event
        and creates no decision. The response carries only identifiers,
        the fixed status code, the decision time, the integer explanation
        version, and node positions/types/booleans — never proof text,
        nonces, claim names or actual claim values, comparison targets,
        capabilities, payloads or keys. A storage or integrity failure is
        a 500 with the half-built result discarded entirely.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {"tenant_id", "workload_id"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is rejected rather than silently
        # treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )

        # The raw path value must be a canonical lowercase UUID: missing
        # (handled by the dedicated empty-segment route above), blank,
        # whitespace-padded, uppercase or otherwise non-canonical values
        # are format errors rejected before any state is read.
        if not decision_id or not _UUID_RE.fullmatch(decision_id):
            raise HTTPException(
                status_code=422, detail="invalid decision identifier"
            )
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # --- read-only decision + persisted explanation -----------------
        try:
            with session_factory() as session:
                decision = session.get(Decision, decision_id)
                if (
                    decision is None
                    or decision.tenant_id != tenant_id
                    or decision.workload_id != workload_id
                ):
                    # Do not reveal whether an unknown or out-of-scope
                    # decision exists under another scope: both share one
                    # indistinguishable 404.
                    raise HTTPException(
                        status_code=404, detail="decision not found"
                    )

                rows = session.scalars(
                    select(DecisionEvaluationNode)
                    .where(DecisionEvaluationNode.decision_id == decision_id)
                    .order_by(DecisionEvaluationNode.node_index.asc())
                ).all()
                if not rows:
                    # A settled decision with no node rows was committed by
                    # a deployment older than the explanation. Its history
                    # is never reconstructed or backfilled on read.
                    raise HTTPException(
                        status_code=409, detail="evaluation not recorded"
                    )

                # Policy versions are immutable and never deleted, so the
                # version decided against is the same tree the explanation
                # was generated from; a missing or out-of-scope version is
                # an integrity/storage failure, never a client error.
                policy = session.get(Policy, decision.policy_id)
                if (
                    policy is None
                    or policy.tenant_id != tenant_id
                    or policy.workload_id != workload_id
                ):
                    raise HTTPException(
                        status_code=500,
                        detail="decision evaluation unavailable",
                    )

                nodes = []
                for position, row in enumerate(rows):
                    # Integrity guards: the explanation was written whole
                    # in node_index order, so any gap, duplicate, bad
                    # path/type, or a root that contradicts the decision is
                    # store damage — discard the half result and answer 500
                    # rather than returning a partial or inconsistent
                    # explanation.
                    if row.node_index != position:
                        raise HTTPException(
                            status_code=500,
                            detail="decision evaluation unavailable",
                        )
                    try:
                        path = json.loads(row.rule_path)
                    except (json.JSONDecodeError, TypeError):
                        raise HTTPException(
                            status_code=500,
                            detail="decision evaluation unavailable",
                        )
                    if (
                        not isinstance(path, list)
                        or any(
                            isinstance(step, bool)
                            or not isinstance(step, int)
                            or step < 0
                            for step in path
                        )
                        or row.node_type not in ("leaf", "all", "any", "not")
                        or not isinstance(row.outcome, bool)
                    ):
                        raise HTTPException(
                            status_code=500,
                            detail="decision evaluation unavailable",
                        )
                    nodes.append(
                        {
                            "node_index": row.node_index,
                            "rule_path": path,
                            "node_type": row.node_type,
                            "outcome": row.outcome,
                        }
                    )

                # Full-shape check against the immutable rule snapshot:
                # same node count and identical pre-order (path, type)
                # sequence detects a truncated or tail-corrupted
                # explanation that a per-row scan alone would miss. The
                # comparison uses positions/types only — no claim name,
                # locator, expected scalar or actual value.
                try:
                    expected_shape = rule_structure(json.loads(policy.rule_json))
                except (json.JSONDecodeError, InvalidRule):
                    raise HTTPException(
                        status_code=500,
                        detail="decision evaluation unavailable",
                    )
                actual_shape = [
                    (tuple(node["rule_path"]), node["node_type"]) for node in nodes
                ]
                if actual_shape != expected_shape:
                    raise HTTPException(
                        status_code=500,
                        detail="decision evaluation unavailable",
                    )

                # The root is node 0 at [] and its outcome must equal the
                # stored conclusion; a mismatch is an integrity failure.
                if nodes[0]["rule_path"] or nodes[0]["outcome"] != (
                    decision.status == DECISION_STATUS_ALLOWED
                ):
                    raise HTTPException(
                        status_code=500,
                        detail="decision evaluation unavailable",
                    )

                evaluation = {
                    "decision_id": decision.decision_id,
                    "policy_version": decision.policy_version,
                    "status": decision.status,
                    "decided_at": _rfc3339(decision.decided_at),
                    "evaluation_version": EVALUATION_VERSION,
                    "nodes": nodes,
                }
        except HTTPException:
            raise
        except Exception:
            # Fixed message only: exception text might carry protected
            # material and is never logged or returned.
            logger.error("decision evaluation query failed")
            raise HTTPException(
                status_code=500, detail="decision evaluation unavailable"
            )

        # Compact JSON with a single terminating newline. Every value is a
        # string, integer, boolean or a (possibly empty) list of integers
        # — no floats, nulls or non-finite values — and no evidence text,
        # nonce, claim name or actual claim value, comparison target,
        # capability, payload or key appears.
        body = (
            json.dumps(
                evaluation,
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/release-grants//trace")
    def get_release_grant_trace_identifier_required() -> Response:
        # An empty path segment is a missing grant identifier: a 422 client
        # error rather than a routing-level 404. It never reads a grant,
        # decision, envelope or outcome.
        raise HTTPException(status_code=422, detail="invalid grant identifier")

    @app.get("/v1/release-grants/{grant_id}/trace")
    def get_release_grant_trace(
        request: Request,
        grant_id: str,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Read-only full-chain trace of one release grant.

        Joins exactly one grant in the named scope to its decision
        (identifier, evidence, policy version, conclusion and decision
        time — never proof text, nonces or claim values), the existing
        envelope the grant was minted against (identifier, scope, key
        version and creation time only — never any cryptographic
        material), and the grant's final outcome (pending, consumed or
        revoked with the corresponding time).

        Every request-shape failure is a 422 returned before any state is
        read: the body must be missing or zero-length (any other bytes,
        including whitespace, are rejected), exactly one non-blank
        ``tenant_id`` and ``workload_id`` may be supplied as query
        parameters (an unknown or repeated parameter is a 422), and the
        path id must be a canonical lowercase UUID with no surrounding
        whitespace. An unknown grant, or one belonging to another tenant
        or workload, is one indistinguishable 404 — existence in any other
        scope is never revealed, and the response never reveals whether
        the referenced decision or envelope exists.

        The handler issues only SELECTs of committed state: it never
        writes a grant, decision, envelope or audit row and it never
        touches the shared grant rate-limit budget. A storage failure at
        any point of the chain is a 500 with the half-built result
        discarded entirely. The capability appears only as its SHA-256
        digest; plaintext capabilities, payloads, data keys, master
        keys, certificate material and exception text never appear.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {"tenant_id", "workload_id"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is rejected rather than silently
        # treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )

        # The raw path value must be a canonical lowercase UUID: missing
        # (handled by the dedicated empty-segment route above), blank,
        # whitespace-padded, uppercase or otherwise non-canonical values
        # are format errors rejected before any state is read.
        if not grant_id or not _UUID_RE.fullmatch(grant_id):
            raise HTTPException(
                status_code=422, detail="invalid grant identifier"
            )
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # --- read-only grant -> decision -> envelope chain --------------
        try:
            with session_factory() as session:
                grant = session.get(ReleaseGrant, grant_id)
                if (
                    grant is None
                    or grant.tenant_id != tenant_id
                    or grant.workload_id != workload_id
                ):
                    # Do not reveal whether an unknown or out-of-scope
                    # grant exists under another scope: both share one
                    # indistinguishable 404, and no later lookup may hint
                    # at whether the decision or envelope exists.
                    raise HTTPException(
                        status_code=404, detail="grant not found"
                    )

                decision = session.get(Decision, grant.decision_id)
                envelope = session.get(
                    DataEnvelope,
                    (grant.tenant_id, grant.workload_id, grant.data_id),
                )
                # A grant always references a settled decision: a missing
                # or out-of-scope row here is an integrity/storage
                # failure, never a client error. The half-built trace is
                # discarded rather than answering without a chain
                # section or fabricating one.
                if (
                    decision is None
                    or decision.tenant_id != tenant_id
                    or decision.workload_id != workload_id
                ):
                    raise HTTPException(
                        status_code=500,
                        detail="grant trace unavailable",
                    )
                # The envelope is read metadata-only and is never
                # unwrapped or decrypted. A grant may be minted before
                # its envelope row exists, so a missing in-scope
                # envelope is not an error and never a 404 (that would
                # reveal whether the data item exists): the section
                # carries the grant's own identifier and scope with the
                # envelope-only fields null until the material exists.

                grant_view = {
                    "grant_id": grant.grant_id,
                    "decision_id": grant.decision_id,
                    "data_id": grant.data_id,
                    "status": grant.status,
                    "issued_at": _rfc3339(grant.issued_at),
                    "expires_at": _rfc3339(grant.expires_at),
                    "consumed_at": (
                        _rfc3339(grant.consumed_at)
                        if grant.consumed_at is not None
                        else None
                    ),
                    "revoked_at": (
                        _rfc3339(grant.revoked_at)
                        if grant.revoked_at is not None
                        else None
                    ),
                    # The capability is exposed only as its SHA-256
                    # digest; the plaintext capability exists nowhere on
                    # this path.
                    "capability_sha256": grant.capability_digest,
                }
                decision_view = {
                    "decision_id": decision.decision_id,
                    # The associated proof (evidence): an identifier,
                    # never the proof itself.
                    "evidence_id": decision.evidence_id,
                    # The integer policy version decided against.
                    "policy_version": decision.policy_version,
                    # The fixed allowed/denied conclusion code.
                    "status": decision.status,
                    # The original UTC decision time, never rewritten.
                    "decided_at": _rfc3339(decision.decided_at),
                }
                envelope_view = {
                    "data_id": grant.data_id,
                    "tenant_id": grant.tenant_id,
                    "workload_id": grant.workload_id,
                    # Integer master key version recorded on the stored
                    # material; null until the envelope exists. The
                    # material itself is never read out.
                    "key_version": (
                        envelope.key_version if envelope is not None else None
                    ),
                    "created_at": (
                        _rfc3339(envelope.created_at)
                        if envelope is not None
                        else None
                    ),
                }
                # The final outcome derives solely from the grant row's
                # committed status: pending while unsettled, otherwise
                # the single terminal state with its first-and-final
                # settlement time. Expiry is not an outcome — a grant
                # expired while pending still reports pending — so
                # repeated reads keep returning the identical result as
                # the one-time state settles at most once.
                if grant.status == RELEASE_GRANT_STATUS_CONSUMED:
                    outcome_view = {
                        "status": RELEASE_GRANT_STATUS_CONSUMED,
                        "at": (
                            _rfc3339(grant.consumed_at)
                            if grant.consumed_at is not None
                            else None
                        ),
                    }
                elif grant.status == RELEASE_GRANT_STATUS_REVOKED:
                    outcome_view = {
                        "status": RELEASE_GRANT_STATUS_REVOKED,
                        "at": (
                            _rfc3339(grant.revoked_at)
                            if grant.revoked_at is not None
                            else None
                        ),
                    }
                else:
                    outcome_view = {
                        "status": RELEASE_GRANT_STATUS_PENDING,
                        "at": None,
                    }
        except HTTPException:
            raise
        except Exception:
            # Fixed message only: exception text might carry protected
            # material and is never logged or returned.
            logger.error("release grant trace query failed")
            raise HTTPException(
                status_code=500, detail="grant trace unavailable"
            )

        # Compact JSON with a single terminating newline. Every value is
        # a JSON string, null or an integer (key_version,
        # policy_version) — no floats, -0.0 or non-finite values. No
        # plaintext capability, evidence text, nonce, claim value,
        # payload, ciphertext, wrapped/raw key, certificate or exception
        # text appears.
        body = (
            json.dumps(
                {
                    "grant": grant_view,
                    "decision": decision_view,
                    "envelope": envelope_view,
                    "outcome": outcome_view,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/release-grants//events")
    def get_release_grant_events_identifier_required() -> Response:
        # An empty path segment is a missing grant identifier: a 422 client
        # error rather than a routing-level 404. It never reads a grant or
        # any event.
        raise HTTPException(status_code=422, detail="invalid grant identifier")

    @app.get("/v1/release-grants/{grant_id}/events")
    def list_release_grant_events(
        request: Request,
        grant_id: str,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        cursor: str | None = Query(default=None),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return the read-only state-migration timeline of one release grant.

        The range is exactly one grant fixed by the mandatory
        tenant/workload scope and the canonical-lowercase-UUID path id;
        only ``tenant_id``, ``workload_id`` and the optional ``cursor``
        are accepted and the body must be missing or zero-length. Events
        are the grant's committed state migrations in stable, immutable
        per-grant sequence order: issuance (no old status -> pending,
        reason ``issued``), the one settlement to consumed by either a
        consume presentation (``consume``) or a payload release
        (``release``), or the settlement to revoked (``revoked``).
        Exactly one event exists per migration that committed; a failed
        or rolled-back request and a request that lost the settlement
        race leave no event, and a consume, release or repeated
        settlement against an already-terminal grant appends nothing.

        The first (cursor-less or explicit-empty-cursor) query, which
        starts before the first event, fixes a replayable snapshot
        high-water mark (the greatest event seq then committed);
        migrations committed afterwards — even with an older business
        time — surface only in a fresh first query, never in a later
        page of this one. The resume cursor is HMAC-authenticated,
        carries its own kind tag and is bound to the scope, grant and
        snapshot, so it cannot be forged, tampered with, or replayed
        against another scope, grant, snapshot or cursor family. The
        handler issues only SELECTs: it never settles a grant, consumes
        budget, appends an audit row or changes any state, and a storage
        failure aborts the whole request with a 500 rather than
        returning half a page. The existing grant audit is untouched; no
        capability (plaintext or digest), payload or key ever appears.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {"tenant_id", "workload_id", "cursor"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is ambiguous and rejected rather than
        # silently treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(status_code=422, detail="query parameter must appear once")

        # The raw path value must be a canonical lowercase UUID; an empty
        # segment is handled by the dedicated route above, and blank,
        # whitespace-padded, uppercase or otherwise non-canonical values
        # are format errors rejected before any state is read.
        if not grant_id or not _UUID_RE.fullmatch(grant_id):
            raise HTTPException(
                status_code=422, detail="invalid grant identifier"
            )
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # Omitted cursor or an explicit empty string starts before the
        # first event and fixes the query family's snapshot. Whitespace,
        # malformed, forged, tampered, cross-scope, cross-grant,
        # cross-snapshot or foreign-kind cursors are indistinguishable
        # 422s and are rejected without reading any state.
        boundary_seq: int = 0
        snapshot_seq: int | None = None
        if cursor is not None and cursor != "":
            if not cursor.strip() or not _CURSOR_RE.fullmatch(cursor):
                raise HTTPException(status_code=422, detail="invalid cursor")
            decoded = _decode_release_grant_event_cursor(
                cursor, tenant_id, workload_id, grant_id
            )
            if decoded is None:
                raise HTTPException(status_code=422, detail="invalid cursor")
            boundary_seq, snapshot_seq = decoded

        # --- read-only timeline scan ------------------------------------
        try:
            with session_factory() as session:
                # The named grant must exist in exactly this scope; an
                # unknown grant and a cross-scope grant are one
                # indistinguishable 404 — existence in any other scope is
                # never revealed.
                grant = session.get(ReleaseGrant, grant_id)
                if (
                    grant is None
                    or grant.tenant_id != tenant_id
                    or grant.workload_id != workload_id
                ):
                    raise HTTPException(status_code=404, detail="grant not found")

                if snapshot_seq is None:
                    # First query of the family: fix the replayable
                    # snapshot at the greatest currently committed seq.
                    # Every grant, legacy or new, always has its issuance
                    # event; a zero here is only possible transiently.
                    fixed_snapshot = session.scalar(
                        select(func.max(ReleaseGrantEvent.seq)).where(
                            ReleaseGrantEvent.grant_id == grant_id
                        )
                    )
                    snapshot_seq = int(fixed_snapshot or 0)

                stmt = (
                    select(ReleaseGrantEvent)
                    .where(
                        ReleaseGrantEvent.grant_id == grant_id,
                        ReleaseGrantEvent.seq > boundary_seq,
                        # Inclusive fixed snapshot: migrations committed
                        # after the first query never enter these pages.
                        ReleaseGrantEvent.seq <= snapshot_seq,
                    )
                    .order_by(ReleaseGrantEvent.seq.asc())
                    .limit(RELEASE_GRANT_EVENT_PAGE_SIZE + 1)
                )
                # One extra row is the "more follows" probe. The scan is a
                # single read-only statement: a storage failure aborts the
                # whole request with a 500 rather than returning a partial
                # page.
                rows = list(session.scalars(stmt))
        except HTTPException:
            raise
        except Exception:
            logger.error("release grant event query failed")
            raise HTTPException(
                status_code=500, detail="release grant events unavailable"
            )

        has_more = len(rows) > RELEASE_GRANT_EVENT_PAGE_SIZE
        page = rows[:RELEASE_GRANT_EVENT_PAGE_SIZE]

        events = [
            {
                "event_id": row.event_id,
                # Stable, immutable per-grant sequence in committed order.
                "seq": row.seq,
                # Null only for the birth migration; a string otherwise.
                "old_status": row.old_status,
                "new_status": row.new_status,
                # A fixed service code (issued/consume/release/revoked),
                # never exception text.
                "reason": row.reason,
                "at": _rfc3339(row.occurred_at),
            }
            for row in page
        ]

        if has_more:
            next_cursor = _encode_release_grant_event_cursor(
                tenant_id,
                workload_id,
                grant_id,
                page[-1].seq,
                snapshot_seq=snapshot_seq,
            )
            complete = False
        else:
            next_cursor = ""
            complete = True

        # Compact container (events, next_cursor, complete) with a single
        # terminating newline. seq is an int, old_status may be null,
        # complete is a bool and every other value is a string — no
        # floats, -0.0 or non-finite values can appear.
        body = (
            json.dumps(
                {
                    "events": events,
                    "next_cursor": next_cursor,
                    "complete": complete,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/revocations")
    def list_certificate_revocations(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        trust_root_id: str = Query(...),
        revocation_id: str | None = Query(default=None),
        certificate_fingerprint: str | None = Query(default=None),
        effective_after: str | None = Query(default=None),
        effective_before: str | None = Query(default=None),
        cursor: str | None = Query(default=None),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return a read-only, cursor-stable page of registrations.

        The range is fixed by the mandatory tenant, workload and trust
        root and may be narrowed by an explicit registration id, a
        certificate fingerprint, and an inclusive effective-at window.
        Ordering is stable ``(effective_at, revocation_id)`` ascending
        with an exclusive keyset cursor. The cursor carries its own kind
        tag, is HMAC-authenticated, and is bound to the scope, every
        active filter *and* the fixed snapshot established by the
        range's first (cursor-less) query, so it can be neither forged
        nor replayed against a different scope/filter/snapshot, and a
        cursor from any other cursor family is rejected. The first
        query's page is a replayable snapshot: registrations committed
        afterwards never enter a replayed page and surface only in a
        fresh first query. The handler issues only SELECTs — it never
        creates an audit row, computes a certificate's current status,
        or mutates a registration.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {
            "tenant_id",
            "workload_id",
            "trust_root_id",
            "revocation_id",
            "certificate_fingerprint",
            "effective_after",
            "effective_before",
            "cursor",
        }
        if set(request.query_params.keys()) - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )

        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")
        if not trust_root_id.strip() or not _UUID_RE.fullmatch(trust_root_id):
            raise HTTPException(status_code=422, detail="invalid trust root identifier")

        revocation_filter: str | None = None
        if revocation_id is not None:
            if not revocation_id.strip() or not _UUID_RE.fullmatch(revocation_id):
                raise HTTPException(
                    status_code=422, detail="invalid revocation identifier"
                )
            revocation_filter = revocation_id

        fingerprint_filter: str | None = None
        if certificate_fingerprint is not None:
            if not certificate_fingerprint.strip():
                raise HTTPException(
                    status_code=422,
                    detail="certificate_fingerprint must be unpadded base64url of 32 bytes",
                )
            try:
                fingerprint_filter = _certificate_fingerprint_format(
                    certificate_fingerprint
                )
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail="certificate_fingerprint must be unpadded base64url of 32 bytes",
                )

        def _time_bound(value: str | None, name: str) -> tuple[str, datetime | None]:
            if value is None:
                return "", None
            if not value.strip():
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            try:
                parsed = _parse_utc_rfc3339(value)
            except ValueError:
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            # Normalize the spelling embedded into the cursor so two
            # equivalent UTC spellings cannot mint different cursor domains.
            return _rfc3339(parsed), parsed

        after_raw, after_dt = _time_bound(effective_after, "effective_after")
        before_raw, before_dt = _time_bound(effective_before, "effective_before")
        # The window is closed on both ends; equality is a valid
        # single-instant window and the start must not follow the end.
        if after_dt is not None and before_dt is not None and after_dt > before_dt:
            raise HTTPException(
                status_code=422,
                detail="effective_after must not be later than effective_before",
            )

        # Omitted cursor or an explicit empty string starts before the
        # smallest registration and fixes the range's replayable snapshot.
        # Whitespace, malformed, forged, cross-scope, cross-filter,
        # cross-snapshot or foreign-kind cursors are indistinguishable 422s.
        boundary_dt: datetime | None = None
        boundary_revocation: str | None = None
        snapshot_dt: datetime | None = None
        if cursor is not None and cursor != "":
            if not cursor.strip() or not _CURSOR_RE.fullmatch(cursor):
                raise HTTPException(status_code=422, detail="invalid cursor")
            decoded_boundary = _decode_revocation_cursor(
                cursor,
                tenant_id,
                workload_id,
                trust_root_id,
                revocation_id=revocation_filter or "",
                certificate_fingerprint=fingerprint_filter or "",
                effective_after=after_raw,
                effective_before=before_raw,
            )
            if decoded_boundary is None:
                raise HTTPException(status_code=422, detail="invalid cursor")
            boundary_at_raw, boundary_revocation, snapshot_at_raw, _ = decoded_boundary
            try:
                boundary_dt = _parse_utc_rfc3339(boundary_at_raw)
                snapshot_dt = _parse_utc_rfc3339(snapshot_at_raw)
            except ValueError:
                raise HTTPException(status_code=422, detail="invalid cursor")

        # --- read-only scan ---------------------------------------------
        try:
            with session_factory() as session:
                # The explicitly named trust root must exist in exactly
                # this tenant and workload; an unknown or cross-scope
                # root is an indistinguishable 404 (it is an explicit
                # resource selector, not a mere range bound).
                root = session.scalar(
                    select(TrustRoot.root_id).where(
                        TrustRoot.root_id == trust_root_id,
                        TrustRoot.tenant_id == tenant_id,
                        TrustRoot.workload_id == workload_id,
                    )
                )
                if root is None:
                    raise HTTPException(status_code=404, detail="trust root not found")

                # An explicitly named registration id or fingerprint must
                # resolve inside exactly this scope and trust root; either
                # way an unknown/cross-scope value is the same 404, so no
                # difference between "unknown" and "belongs elsewhere" is
                # ever revealed.
                if revocation_filter is not None:
                    named = session.get(CertificateRevocation, revocation_filter)
                    if (
                        named is None
                        or named.tenant_id != tenant_id
                        or named.workload_id != workload_id
                        or named.trust_root_id != trust_root_id
                    ):
                        raise HTTPException(
                            status_code=404, detail="revocation not found"
                        )
                if fingerprint_filter is not None:
                    matched = session.scalar(
                        select(CertificateRevocation.revocation_id)
                        .where(
                            CertificateRevocation.tenant_id == tenant_id,
                            CertificateRevocation.workload_id == workload_id,
                            CertificateRevocation.trust_root_id == trust_root_id,
                            CertificateRevocation.certificate_fingerprint
                            == fingerprint_filter,
                        )
                        .limit(1)
                    )
                    if matched is None:
                        # Do not reveal whether the fingerprint is unknown
                        # or registered under another scope/root.
                        raise HTTPException(
                            status_code=404, detail="revocation not found"
                        )

                def _filtered(stmt):
                    if revocation_filter is not None:
                        stmt = stmt.where(
                            CertificateRevocation.revocation_id == revocation_filter
                        )
                    if fingerprint_filter is not None:
                        stmt = stmt.where(
                            CertificateRevocation.certificate_fingerprint
                            == fingerprint_filter
                        )
                    if after_dt is not None:
                        stmt = stmt.where(
                            CertificateRevocation.effective_at >= after_dt
                        )
                    if before_dt is not None:
                        stmt = stmt.where(
                            CertificateRevocation.effective_at <= before_dt
                        )
                    return stmt

                if snapshot_dt is None:
                    # First query of the range: fix a replayable snapshot
                    # as the greatest ``(created_at, revocation_id)`` among
                    # currently committed, in-range rows. Registrations
                    # committed afterwards lie beyond the high-water mark
                    # and never enter this snapshot's pages.
                    hwm_stmt = _filtered(
                        select(
                            CertificateRevocation.created_at,
                            CertificateRevocation.revocation_id,
                        ).where(
                            CertificateRevocation.tenant_id == tenant_id,
                            CertificateRevocation.workload_id == workload_id,
                            CertificateRevocation.trust_root_id == trust_root_id,
                        )
                    )
                    hwm_row = session.execute(
                        hwm_stmt.order_by(
                            CertificateRevocation.created_at.desc(),
                            CertificateRevocation.revocation_id.desc(),
                        ).limit(1)
                    ).first()
                    if hwm_row is None:
                        # No in-range registration at snapshot time: the
                        # fixed first page is empty and already complete;
                        # later commits belong to fresh first queries.
                        rows = []
                        hwm_created_at = None
                        hwm_revocation = ""
                    else:
                        hwm_created_at, hwm_revocation = hwm_row
                        snapshot_dt = hwm_created_at
                else:
                    # The snapshot high-water mark travels inside the
                    # cursor (HMAC-verified above); its id component is
                    # read alongside for the inclusive tie predicate.
                    hwm_created_at = snapshot_dt
                    hwm_revocation = decoded_boundary[3]

                if hwm_created_at is not None:
                    stmt = _filtered(
                        select(CertificateRevocation).where(
                            CertificateRevocation.tenant_id == tenant_id,
                            CertificateRevocation.workload_id == workload_id,
                            CertificateRevocation.trust_root_id == trust_root_id,
                            # Inclusive snapshot high-water mark.
                            or_(
                                CertificateRevocation.created_at < hwm_created_at,
                                and_(
                                    CertificateRevocation.created_at
                                    == hwm_created_at,
                                    CertificateRevocation.revocation_id
                                    <= hwm_revocation,
                                ),
                            ),
                        )
                    )
                    if boundary_dt is not None:
                        # Exclusive (effective_at, revocation_id) keyset.
                        stmt = stmt.where(
                            or_(
                                CertificateRevocation.effective_at > boundary_dt,
                                and_(
                                    CertificateRevocation.effective_at
                                    == boundary_dt,
                                    CertificateRevocation.revocation_id
                                    > boundary_revocation,
                                ),
                            )
                        )
                    stmt = stmt.order_by(
                        CertificateRevocation.effective_at.asc(),
                        CertificateRevocation.revocation_id.asc(),
                    ).limit(REVOCATION_PAGE_SIZE + 1)
                    # One extra row is the "more follows" probe. The scan
                    # is a single read-only statement: a storage failure
                    # aborts the whole request with a 500 rather than
                    # returning a partial page.
                    rows = list(session.scalars(stmt))
        except HTTPException:
            raise
        except Exception:
            logger.error("certificate revocation query failed")
            raise HTTPException(
                status_code=500, detail="revocation registry unavailable"
            )

        if not rows:
            page = []
            has_more = False
        else:
            has_more = len(rows) > REVOCATION_PAGE_SIZE
            page = rows[:REVOCATION_PAGE_SIZE]

        revocations = [
            {
                # Field order follows the registration success response.
                "revocation_id": row.revocation_id,
                "trust_root_id": row.trust_root_id,
                "certificate_fingerprint": row.certificate_fingerprint,
                "effective_at": _rfc3339(row.effective_at),
            }
            for row in page
        ]

        if has_more:
            last = page[-1]
            next_cursor = _encode_revocation_cursor(
                tenant_id,
                workload_id,
                trust_root_id,
                _rfc3339(last.effective_at),
                last.revocation_id,
                revocation_id=revocation_filter or "",
                certificate_fingerprint=fingerprint_filter or "",
                effective_after=after_raw,
                effective_before=before_raw,
                snapshot_at=_rfc3339(hwm_created_at),
                snapshot_id=hwm_revocation,
            )
            complete = False
        else:
            next_cursor = ""
            complete = True

        # Compact container (list, next_cursor, complete — the same shape
        # as the compliance audit query) with a single terminating newline.
        # Every entry value is a string and complete is a boolean, so no
        # floats, -0.0 or non-finite values can appear.
        body = (
            json.dumps(
                {
                    "revocations": revocations,
                    "next_cursor": next_cursor,
                    "complete": complete,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.post("/v1/release-grants//consume")
    def consume_release_grant_identifier_required() -> Response:
        # An empty path segment is a missing grant identifier: a 422 client
        # error rather than a routing-level 404, and it never consumes budget.
        raise HTTPException(status_code=422, detail="invalid grant identifier")

    @app.post(
        "/v1/release-grants/{grant_id}/consume",
        response_model=ReleaseGrantConsumedResponse,
    )
    def consume_release_grant(
        request: Request, grant_id: str, body: ConsumeReleaseGrantRequest
    ) -> Response:
        # Grant ids are canonical lowercase UUIDs; a syntactically illegal
        # path identifier is a 422 field error indistinguishable from any
        # other bad input, never a lookup, and never consumes budget.
        if not grant_id.strip() or not _UUID_RE.fullmatch(grant_id.lower()):
            raise HTTPException(status_code=422, detail="invalid grant identifier")
        grant_id = grant_id.strip().lower()

        # The optional idempotency key lives only in a header; the body
        # contract is unchanged. A missing key preserves the original
        # one-time consume semantics exactly. A present but illegal key
        # (a duplicated header line, an empty value, surrounding
        # whitespace, a control or non-ASCII character, or an over-long
        # value) is an indistinguishable 422 here, after the body and path
        # checks and before the budget reservation or any grant read or
        # write: it never spends quota and never touches storage.
        idem_present, idempotency_key = _read_idempotency_key(request)

        # Basic field/format validation (the request model, the path
        # check above and the header check) has passed: reserve one shared
        # per-scope minute slot before any business judgement, keyed or
        # not. A 429 or a counter failure surfaces here and changes no
        # grant, payload, audit or idempotency state. A keyed replay draws
        # a slot just like any other admitted request; once admitted, the
        # replay never re-judges the grant.
        limited = _consume_grant_budget(body.tenant_id, body.workload_id)
        if limited is not None:
            return limited
        digest = _nonce_digest(body.capability)
        now = _utcnow()

        if not idem_present:
            with session_factory() as session:
                grant = session.get(ReleaseGrant, grant_id)
                if (
                    grant is None
                    or grant.tenant_id != body.tenant_id
                    or grant.workload_id != body.workload_id
                ):
                    raise HTTPException(status_code=404, detail="grant not found")
                if not hmac.compare_digest(grant.capability_digest, digest):
                    raise HTTPException(status_code=401, detail="invalid capability")
                # A settled grant can never be consumed: revocation is a
                # terminal state exactly like consumed, and is judged before
                # expiry so a grant revoked while pending is reported 409 even
                # after it has since expired.
                if grant.status == "consumed":
                    raise HTTPException(
                        status_code=409, detail="grant already consumed"
                    )
                if grant.status == RELEASE_GRANT_STATUS_REVOKED:
                    raise HTTPException(
                        status_code=409, detail="grant already revoked"
                    )
                if grant.expires_at <= now:
                    raise HTTPException(status_code=410, detail="grant expired")
                # Atomic claim: only one concurrent consumer can flip
                # pending -> consumed for an unexpired grant. BEGIN IMMEDIATE
                # (SQLite) / row locks (other backends) plus the guarded UPDATE
                # guarantee exactly one winner across consume, release and
                # revoke, across processes and restarts.
                result = session.execute(
                    update(ReleaseGrant)
                    .where(
                        ReleaseGrant.grant_id == grant_id,
                        ReleaseGrant.status == "pending",
                        ReleaseGrant.expires_at > now,
                    )
                    .values(status="consumed", consumed_at=now)
                    .execution_options(synchronize_session=False)
                )
                if result.rowcount != 1:
                    session.rollback()
                    fresh = session.get(ReleaseGrant, grant_id)
                    if fresh is not None and fresh.status == "consumed":
                        raise HTTPException(
                            status_code=409, detail="grant already consumed"
                        )
                    if (
                        fresh is not None
                        and fresh.status == RELEASE_GRANT_STATUS_REVOKED
                    ):
                        # A concurrent revocation won the shared state; the
                        # loser only observes the terminal revoked status.
                        raise HTTPException(
                            status_code=409, detail="grant already revoked"
                        )
                    raise HTTPException(status_code=410, detail="grant expired")
                # Exactly one immutable timeline event per committed
                # settlement: this consume presentation won pending ->
                # consumed (reason "consume"), in the same transaction as the
                # guarded status flip and the unchanged compliance audit.
                _record_release_grant_event(
                    session,
                    tenant_id=grant.tenant_id,
                    workload_id=grant.workload_id,
                    grant_id=grant_id,
                    old_status=RELEASE_GRANT_STATUS_PENDING,
                    new_status=RELEASE_GRANT_STATUS_CONSUMED,
                    reason=RELEASE_GRANT_EVENT_REASON_CONSUME,
                    now=now,
                )
                _append_audit_event(
                    session,
                    tenant_id=grant.tenant_id,
                    workload_id=grant.workload_id,
                    event_type=AUDIT_EVENT_TYPE_GRANT,
                    grant_id=grant_id,
                    decision_id=grant.decision_id,
                    data_id=grant.data_id,
                    status=AUDIT_EVENT_STATUS_CONSUMED,
                    capability_sha256=grant.capability_digest,
                    occurred_at=now,
                )
                session.commit()
                decision_id = grant.decision_id
                data_id = grant.data_id
            return ReleaseGrantConsumedResponse(
                grant_id=grant_id,
                decision_id=decision_id,
                data_id=data_id,
                consumed=True,
                consumed_at=_rfc3339(now),
            )

        # Idempotency-keyed consume. The equivalence range is the
        # normalized grant id, the scope and the capability digest: a
        # same-key request outside this scope is an independent key (a
        # different (tenant, workload) namespace) and is judged normally;
        # within the scope a same-key request whose grant or capability
        # differs is a stable 409 that changes nothing.
        fingerprint = _release_grant_consume_fingerprint(
            grant_id, body.tenant_id, body.workload_id, digest
        )

        def _stored_record(session):
            return session.scalar(
                select(ReleaseGrantConsumeIdempotencyRecord).where(
                    ReleaseGrantConsumeIdempotencyRecord.tenant_id
                    == body.tenant_id,
                    ReleaseGrantConsumeIdempotencyRecord.workload_id
                    == body.workload_id,
                    ReleaseGrantConsumeIdempotencyRecord.idempotency_key
                    == idempotency_key,
                )
            )

        def _replay(record) -> Response:
            # A replay answers with the stored first 200 verbatim: it
            # never re-judges the grant (so expiry or any later change is
            # irrelevant), appends no event or audit, and changes neither
            # status nor any timestamp. A same-key request outside the
            # fingerprint is a 409 that likewise writes nothing.
            if not hmac.compare_digest(record.request_fingerprint, fingerprint):
                raise HTTPException(
                    status_code=409,
                    detail="idempotency key reused with different request",
                )
            return Response(
                content=record.response_body.encode("utf-8"),
                status_code=200,
                media_type="application/json",
            )

        # The idempotency record, the grant state flip, the timeline event
        # and the audit event are one atomic commit. The lookup-then-insert
        # runs in a single transaction, with the unique (scope, key)
        # constraint plus the IntegrityError reread below settling
        # concurrent identical retries: exactly one request migrates the
        # grant and records one event, and every loser reads the same
        # stored result and returns 200.
        saved_body: str | None = None
        for _ in range(2):
            with session_factory() as session:
                existing = _stored_record(session)
                if existing is not None:
                    return _replay(existing)
                # First keyed request for this scope+key. Judgement runs in
                # its existing order; every failure raises out of the
                # context manager, which rolls back, so no idempotency row
                # is left behind and the key stays free for a recovered
                # retry.
                grant = session.get(ReleaseGrant, grant_id)
                if (
                    grant is None
                    or grant.tenant_id != body.tenant_id
                    or grant.workload_id != body.workload_id
                ):
                    raise HTTPException(status_code=404, detail="grant not found")
                if not hmac.compare_digest(grant.capability_digest, digest):
                    raise HTTPException(status_code=401, detail="invalid capability")
                if grant.status == "consumed":
                    raise HTTPException(
                        status_code=409, detail="grant already consumed"
                    )
                if grant.status == RELEASE_GRANT_STATUS_REVOKED:
                    raise HTTPException(
                        status_code=409, detail="grant already revoked"
                    )
                if grant.expires_at <= now:
                    raise HTTPException(status_code=410, detail="grant expired")
                # Atomic claim shared with the keyless path, payload release
                # and revocation: only one concurrent caller can flip the
                # row, so concurrent same-key retries can never settle it
                # twice.
                result = session.execute(
                    update(ReleaseGrant)
                    .where(
                        ReleaseGrant.grant_id == grant_id,
                        ReleaseGrant.status == "pending",
                        ReleaseGrant.expires_at > now,
                    )
                    .values(status="consumed", consumed_at=now)
                    .execution_options(synchronize_session=False)
                )
                if result.rowcount != 1:
                    # A concurrent consume/release/revoke settled the row
                    # first, or it expired between check and write. On
                    # locking backends a same-key loser reaches here (its
                    # pre-lock snapshot showed no record and a pending
                    # grant); because the winning settlement and its
                    # idempotency record are one commit, the record is now
                    # visible when the winner carried this same key, in
                    # which case this request is a replay (200) or a
                    # same-key mismatch (409), never the state conflict
                    # below. No record means a keyless winner, a release,
                    # a revoke or an expiry: observe final state only.
                    session.rollback()
                    winner_record = _stored_record(session)
                    if winner_record is not None:
                        return _replay(winner_record)
                    fresh = session.get(ReleaseGrant, grant_id)
                    if fresh is not None and fresh.status == "consumed":
                        raise HTTPException(
                            status_code=409, detail="grant already consumed"
                        )
                    if (
                        fresh is not None
                        and fresh.status == RELEASE_GRANT_STATUS_REVOKED
                    ):
                        raise HTTPException(
                            status_code=409, detail="grant already revoked"
                        )
                    raise HTTPException(status_code=410, detail="grant expired")
                # The winning settlement: one timeline event and one audit
                # event, exactly as on the keyless path.
                _record_release_grant_event(
                    session,
                    tenant_id=grant.tenant_id,
                    workload_id=grant.workload_id,
                    grant_id=grant_id,
                    old_status=RELEASE_GRANT_STATUS_PENDING,
                    new_status=RELEASE_GRANT_STATUS_CONSUMED,
                    reason=RELEASE_GRANT_EVENT_REASON_CONSUME,
                    now=now,
                )
                _append_audit_event(
                    session,
                    tenant_id=grant.tenant_id,
                    workload_id=grant.workload_id,
                    event_type=AUDIT_EVENT_TYPE_GRANT,
                    grant_id=grant_id,
                    decision_id=grant.decision_id,
                    data_id=grant.data_id,
                    status=AUDIT_EVENT_STATUS_CONSUMED,
                    capability_sha256=grant.capability_digest,
                    occurred_at=now,
                )
                # The exact first 200 body, fixed before commit so the
                # stored response and the response returned to the winner
                # are byte-for-byte the same, including the original
                # consumed_at.
                saved_body = _release_grant_consumed_body(
                    grant_id, grant.decision_id, grant.data_id, now
                )
                session.add(
                    ReleaseGrantConsumeIdempotencyRecord(
                        record_id=str(uuid.uuid4()),
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        idempotency_key=idempotency_key,
                        grant_id=grant_id,
                        request_fingerprint=fingerprint,
                        response_body=saved_body,
                        created_at=now,
                    )
                )
                try:
                    # State flip, both events and the idempotency record
                    # commit together: a crash can never leave a settled
                    # grant without its record or a record pointing at an
                    # unsettled grant.
                    session.commit()
                except IntegrityError:
                    # A concurrent request for the same scope+key committed
                    # first. Reread its record and answer as a replay
                    # (verbatim 200) or a conflict (409).
                    session.rollback()
                    continue
                break
        else:  # pragma: no cover - defensive: the reread always settles
            logger.error("release grant consume idempotency race did not settle")
            raise HTTPException(status_code=500, detail="grant unavailable")

        assert saved_body is not None
        return Response(
            content=saved_body.encode("utf-8"),
            status_code=200,
            media_type="application/json",
        )

    @app.post("/v1/release-grants//revoke")
    def revoke_release_grant_identifier_required() -> Response:
        # An empty path segment is a missing grant identifier: a 422
        # client error rather than a routing-level 404.
        raise HTTPException(status_code=422, detail="invalid grant identifier")

    @app.post(
        "/v1/release-grants/{grant_id}/revoke",
        response_model=ReleaseGrantRevokedResponse,
    )
    def revoke_release_grant(
        grant_id: str, body: RevokeReleaseGrantRequest
    ) -> ReleaseGrantRevokedResponse:
        # Grant ids are canonical lowercase UUIDs; a syntactically illegal
        # path identifier is a 422 field error indistinguishable from any
        # other bad input, never a lookup.
        if not grant_id.strip() or not _UUID_RE.fullmatch(grant_id.lower()):
            raise HTTPException(status_code=422, detail="invalid grant identifier")
        grant_id = grant_id.strip().lower()

        # Field/format validation has passed (request model plus the path
        # check): reserve one shared per-scope minute slot before any
        # business judgement. A 429 or a counter failure changes no state.
        limited = _consume_grant_budget(body.tenant_id, body.workload_id)
        if limited is not None:
            return limited
        digest = _nonce_digest(body.capability)
        now = _utcnow()
        with session_factory() as session:
            grant = session.get(ReleaseGrant, grant_id)
            # The path grant must belong to exactly the body's tenant and
            # workload; an unknown grant and an out-of-scope one are
            # indistinguishable 404s.
            if (
                grant is None
                or grant.tenant_id != body.tenant_id
                or grant.workload_id != body.workload_id
            ):
                raise HTTPException(status_code=404, detail="grant not found")
            # The capability is used only for this verification; its
            # plaintext is never logged, persisted, or returned. A mismatch
            # is an authentication failure that changes no state: the grant
            # (and its pending/expired status) is left exactly as found.
            if not hmac.compare_digest(grant.capability_digest, digest):
                raise HTTPException(status_code=401, detail="invalid capability")
            # Status is judged completely before any write. A settled grant
            # always reports 409, even if it has since passed its expiry, so
            # a repeated revoke observes the same outcome and never rewrites
            # revoked_at; only a still-pending grant past its expiry reports
            # 410.
            if grant.status == "consumed":
                raise HTTPException(status_code=409, detail="grant already consumed")
            if grant.status == RELEASE_GRANT_STATUS_REVOKED:
                raise HTTPException(status_code=409, detail="grant already revoked")
            if grant.expires_at <= now:
                raise HTTPException(status_code=410, detail="grant expired")
            # Atomic settlement shared with the consume and release
            # endpoints: only one concurrent caller can flip pending ->
            # revoked for an unexpired grant. BEGIN IMMEDIATE (SQLite) /
            # row locks (other backends) plus the guarded UPDATE guarantee
            # that revocation, consumption and release have at most one
            # winner across processes and restarts.
            result = session.execute(
                update(ReleaseGrant)
                .where(
                    ReleaseGrant.grant_id == grant_id,
                    ReleaseGrant.status == "pending",
                    ReleaseGrant.expires_at > now,
                )
                .values(status=RELEASE_GRANT_STATUS_REVOKED, revoked_at=now)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                # A concurrent consume/release/revoke settled the row first,
                # or it expired between the check and the write. Observe the
                # final state only and write nothing: a settled grant is
                # 409 (consumed or revoked), an expired-pending one is 410.
                session.rollback()
                fresh = session.get(ReleaseGrant, grant_id)
                if fresh is None:
                    raise HTTPException(status_code=404, detail="grant not found")
                if fresh.status == "consumed":
                    raise HTTPException(
                        status_code=409, detail="grant already consumed"
                    )
                if fresh.status == RELEASE_GRANT_STATUS_REVOKED:
                    raise HTTPException(
                        status_code=409, detail="grant already revoked"
                    )
                raise HTTPException(status_code=410, detail="grant expired")
            # Exactly one immutable timeline event for the winning pending
            # -> revoked settlement (reason "revoked"), in the same
            # transaction as the guarded status flip and the unchanged
            # compliance audit.
            _record_release_grant_event(
                session,
                tenant_id=grant.tenant_id,
                workload_id=grant.workload_id,
                grant_id=grant_id,
                old_status=RELEASE_GRANT_STATUS_PENDING,
                new_status=RELEASE_GRANT_STATUS_REVOKED,
                reason=RELEASE_GRANT_EVENT_REASON_REVOKED,
                now=now,
            )
            _append_audit_event(
                session,
                tenant_id=grant.tenant_id,
                workload_id=grant.workload_id,
                event_type=AUDIT_EVENT_TYPE_GRANT,
                grant_id=grant_id,
                decision_id=grant.decision_id,
                data_id=grant.data_id,
                status=AUDIT_EVENT_STATUS_REVOKED,
                capability_sha256=grant.capability_digest,
                occurred_at=now,
            )
            session.commit()
            decision_id = grant.decision_id
            data_id = grant.data_id
        return ReleaseGrantRevokedResponse(
            grant_id=grant_id,
            decision_id=decision_id,
            data_id=data_id,
            revoked=True,
            revoked_at=_rfc3339(now),
        )

    @app.post("/v1/release/")
    def release_payload_identifier_required() -> Response:
        # An empty path segment is a missing grant identifier: a 422 client
        # error rather than a routing-level 404, and it never consumes budget.
        raise HTTPException(status_code=422, detail="invalid grant identifier")

    @app.post("/v1/release/{grant_id}")
    def release_payload(grant_id: str, body: ReleasePayloadRequest) -> Response:
        """Release one protected payload against a one-time grant.

        Judgement order is fixed: unknown/cross-scope grant or data item
        (404), capability mismatch (401), expiry (410), already consumed or
        revoked (409). Field/format problems — including a missing, blank or
        syntactically illegal path grant identifier — are rejected with 422
        before this handler's judgement and never consume budget. The shared
        per-scope minute budget is reserved after that basic validation and
        before the 404/401/410/409/500 judgements below. The one-time state
        is the very same release_grants row used by the consume and revoke
        endpoints: the data key is unwrapped and the payload
        authenticated-decrypted *before* the pending -> consumed transition
        is committed, so any keyring or decryption failure leaves the grant
        pending and writes no consumption audit, and a revocation that
        settles first makes this path observe 409 without releasing.
        """
        # A syntactically illegal path identifier is a 422 field error, not
        # a lookup, and never consumes budget or touches storage.
        if not grant_id.strip() or not _UUID_RE.fullmatch(grant_id.lower()):
            raise HTTPException(status_code=422, detail="invalid grant identifier")
        grant_id = grant_id.strip().lower()

        limited = _consume_grant_budget(body.tenant_id, body.workload_id)
        if limited is not None:
            return limited
        digest = _nonce_digest(body.capability)
        now = _utcnow()
        with session_factory() as session:
            grant = session.get(ReleaseGrant, grant_id)
            if (
                grant is None
                or grant.tenant_id != body.tenant_id
                or grant.workload_id != body.workload_id
            ):
                raise HTTPException(status_code=404, detail="grant not found")

            # The envelope must exist in exactly the grant's scope and the
            # request's data identifier must match the one the grant was
            # minted for. An unknown item, a cross-scope item and an
            # identifier mismatch are indistinguishable 404s.
            envelope = session.get(
                DataEnvelope,
                (body.tenant_id, body.workload_id, body.data_id),
            )
            if envelope is None or grant.data_id != body.data_id:
                raise HTTPException(status_code=404, detail="data envelope not found")

            # The capability is used only for this verification; its
            # plaintext is never logged, persisted, or returned.
            if not hmac.compare_digest(grant.capability_digest, digest):
                raise HTTPException(status_code=401, detail="invalid capability")

            # Expiry precedes the settled-status judgement: an expired grant
            # that was also consumed or revoked reports 410.
            if grant.expires_at <= now:
                raise HTTPException(status_code=410, detail="grant expired")
            if grant.status == "consumed":
                raise HTTPException(status_code=409, detail="grant already consumed")
            if grant.status == RELEASE_GRANT_STATUS_REVOKED:
                # Revocation settled the shared one-time state first; no
                # decryption or release happens past this point.
                raise HTTPException(status_code=409, detail="grant already revoked")

            # Keyring problems are server failures that must not consume
            # the grant. Only the failure kind is logged — never key
            # material, the capability, or any payload.
            try:
                keyring = load_keyring()
                unwrapping_key = keyring.key_for(envelope.key_version)
            except MasterKeyError as exc:
                logger.error("master key configuration unavailable: %s", exc)
                raise HTTPException(
                    status_code=500, detail="decryption unavailable"
                )

            # Unwrap the data key with the master key version recorded on
            # the envelope and authenticated-decrypt the payload. Both
            # operations are authenticated: any failure rolls back without
            # touching the grant's one-time state and writes no audit.
            try:
                plaintext_bytes = decrypt_payload(
                    unwrapping_key,
                    envelope.wrapped_key,
                    envelope.iv,
                    envelope.ciphertext,
                    envelope.tag,
                )
                plaintext = plaintext_bytes.decode("utf-8")
            except Exception:
                session.rollback()
                logger.error("payload release decryption failed")
                raise HTTPException(status_code=500, detail="decryption failed")
            # Envelopes are created from non-empty strings, so an empty
            # recovery is unreachable for service-written rows; treat it as
            # a failure rather than releasing an invalid response.
            if not plaintext:
                session.rollback()
                logger.error("released payload was empty")
                raise HTTPException(status_code=500, detail="decryption failed")

            # Only after authenticated decryption succeeds is the one-time
            # state committed. The guarded UPDATE is shared with the
            # consume endpoint: concurrent or duplicate valid requests
            # race here and exactly one winner consumes the grant.
            result = session.execute(
                update(ReleaseGrant)
                .where(
                    ReleaseGrant.grant_id == grant_id,
                    ReleaseGrant.status == "pending",
                    ReleaseGrant.expires_at > now,
                )
                .values(status="consumed", consumed_at=now)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                session.rollback()
                fresh = session.get(ReleaseGrant, grant_id)
                if fresh is not None and fresh.status == "consumed":
                    raise HTTPException(
                        status_code=409, detail="grant already consumed"
                    )
                if fresh is not None and fresh.status == RELEASE_GRANT_STATUS_REVOKED:
                    # A concurrent revocation settled the shared state
                    # before this release: no payload is released and the
                    # loser only observes the terminal revoked status.
                    raise HTTPException(
                        status_code=409, detail="grant already revoked"
                    )
                raise HTTPException(status_code=410, detail="grant expired")
            # Exactly one immutable timeline event for the winning payload
            # release settlement: pending -> consumed with reason
            # "release" (distinct from a plain consume presentation), in
            # the same transaction as the guarded status flip and the
            # unchanged compliance audit.
            _record_release_grant_event(
                session,
                tenant_id=grant.tenant_id,
                workload_id=grant.workload_id,
                grant_id=grant_id,
                old_status=RELEASE_GRANT_STATUS_PENDING,
                new_status=RELEASE_GRANT_STATUS_CONSUMED,
                reason=RELEASE_GRANT_EVENT_REASON_RELEASE,
                now=now,
            )
            _append_audit_event(
                session,
                tenant_id=grant.tenant_id,
                workload_id=grant.workload_id,
                event_type=AUDIT_EVENT_TYPE_GRANT,
                grant_id=grant_id,
                decision_id=grant.decision_id,
                data_id=grant.data_id,
                status=AUDIT_EVENT_STATUS_CONSUMED,
                capability_sha256=grant.capability_digest,
                occurred_at=now,
            )
            session.commit()

        # The plaintext exists only in this local value; it is never
        # logged or persisted. Compact JSON containing exactly one
        # non-empty string field, terminated by a single newline.
        body_bytes = (
            json.dumps(
                {"payload": plaintext},
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body_bytes, media_type="application/json")

    def _create_data_envelope_keyed(
        body: CreateDataEnvelopeRequest, idempotency_key: str
    ) -> Response:
        """Create one envelope under an ``Idempotency-Key``.

        The envelope and its idempotency record are one atomic commit:
        the lookup-then-insert runs in a single transaction, with the
        unique (scope, key) constraint plus the IntegrityError reread
        below settling concurrent identical submissions so at most one
        envelope and one record ever exist per key. Every judgement
        failure raises out of the context manager, which rolls back, so
        a failed attempt leaves neither an envelope nor a record and
        the key stays free for a recovered retry.
        """
        # Irreversible request identity: only the SHA-256 of the payload
        # bytes participates; the plaintext never does and is never
        # persisted.
        payload_digest = hashlib.sha256(body.payload.encode("utf-8")).hexdigest()

        saved_body: str | None = None
        for _ in range(2):
            with session_factory() as session:
                existing = session.scalar(
                    select(DataEnvelopeIdempotencyRecord).where(
                        DataEnvelopeIdempotencyRecord.tenant_id == body.tenant_id,
                        DataEnvelopeIdempotencyRecord.workload_id
                        == body.workload_id,
                        DataEnvelopeIdempotencyRecord.idempotency_key
                        == idempotency_key,
                    )
                )
                if existing is not None:
                    # A replay never re-encrypts, never creates an
                    # envelope, never advances the directory sequence and
                    # never changes a key version: the stored first 201
                    # is returned verbatim. A same-key request whose
                    # data_id or payload differs is a stable 409 that
                    # changes nothing.
                    if not (
                        hmac.compare_digest(existing.data_id, body.data_id)
                        and hmac.compare_digest(
                            existing.payload_sha256, payload_digest
                        )
                    ):
                        raise HTTPException(
                            status_code=409,
                            detail="idempotency key reused with a different request",
                        )
                    return Response(
                        content=existing.response_body.encode("utf-8"),
                        status_code=201,
                        media_type="application/json",
                    )

                # First keyed request for this scope+key. A wholly
                # unusable keyring is a server configuration failure;
                # the check sits inside the transaction so a failure
                # rolls back (no envelope, no record).
                try:
                    keyring = load_keyring()
                except MasterKeyError as exc:
                    session.rollback()
                    logger.error("master key configuration unavailable: %s", exc)
                    raise HTTPException(
                        status_code=500, detail="encryption unavailable"
                    )

                # Same-scope data_id uniqueness, exactly as on the
                # keyless path: a duplicate is a judgement failure that
                # writes nothing and leaves the key free.
                if (
                    session.get(
                        DataEnvelope,
                        (body.tenant_id, body.workload_id, body.data_id),
                    )
                    is not None
                ):
                    raise HTTPException(
                        status_code=409,
                        detail="data_id already exists in this scope",
                    )

                # Encrypt outside any durable state, exactly as on the
                # keyless path: the plaintext payload and plaintext data
                # key live only in local variables and are never logged
                # or placed on a response. A failure here leaves no
                # record at all.
                try:
                    sealed = encrypt_payload(
                        keyring.current_key(), body.payload.encode("utf-8")
                    )
                except Exception:
                    session.rollback()
                    logger.error("payload encryption failed for data envelope")
                    raise HTTPException(
                        status_code=500, detail="encryption failed"
                    )

                now = _utcnow()
                # Sequence the creation from the per-scope envelope
                # counter in this same write transaction, exactly as on
                # the keyless path; a replay never reaches here, so a
                # replay never advances the directory sequence.
                commit_seq = _next_data_envelope_commit_seq(
                    session, body.tenant_id, body.workload_id
                )
                session.add(
                    DataEnvelope(
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        data_id=body.data_id,
                        key_version=keyring.current_version,
                        ciphertext=sealed.ciphertext,
                        iv=sealed.iv,
                        tag=sealed.tag,
                        wrapped_key=sealed.wrapped_key,
                        created_at=now,
                        commit_seq=commit_seq,
                    )
                )
                # The exact first 201 body, fixed before commit so the
                # stored response and the response returned to the
                # winner are byte-for-byte the same, including the
                # original created_at and key_version.
                saved_body = _data_envelope_created_body(
                    body.data_id,
                    body.tenant_id,
                    body.workload_id,
                    keyring.current_version,
                    now,
                )
                session.add(
                    DataEnvelopeIdempotencyRecord(
                        record_id=str(uuid.uuid4()),
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        idempotency_key=idempotency_key,
                        data_id=body.data_id,
                        payload_sha256=payload_digest,
                        response_body=saved_body,
                        created_at=now,
                    )
                )
                try:
                    # The envelope and its idempotency record commit
                    # together: a crash can never leave one without the
                    # other.
                    session.commit()
                except IntegrityError:
                    # A concurrent request committed first — either the
                    # same scope+key (its record is now visible) or the
                    # same (scope, data_id). Reread and settle as a
                    # replay, a key-reuse conflict or a duplicate
                    # data_id on the next pass.
                    session.rollback()
                    continue
                except Exception:
                    session.rollback()
                    logger.error("data envelope write failed")
                    raise HTTPException(
                        status_code=500, detail="encryption failed"
                    )
                break
        else:  # pragma: no cover - defensive: the reread always settles
            logger.error("data envelope idempotency race did not settle")
            raise HTTPException(status_code=500, detail="encryption failed")

        assert saved_body is not None
        return Response(
            content=saved_body.encode("utf-8"),
            status_code=201,
            media_type="application/json",
        )

    @app.post(
        "/v1/data-envelopes",
        status_code=201,
        response_model=DataEnvelopeCreatedResponse,
    )
    def create_data_envelope(
        request: Request, body: CreateDataEnvelopeRequest
    ) -> DataEnvelopeCreatedResponse | Response:
        # The idempotency key is optional and lives only in a header; the
        # body contract is unchanged. A missing key preserves the original
        # one-envelope-per-request semantics exactly. An illegal key
        # (duplicated header, empty, non-visible-ASCII or longer than 64
        # characters) is rejected here, after body validation and before
        # any keyring check, encryption or write, so an invalid key never
        # encrypts, never creates an envelope and never writes a record.
        idem_present, idempotency_key = _read_idempotency_key(request)

        if idem_present:
            return _create_data_envelope_keyed(body, idempotency_key)

        # The master keyring is required for this operation; its absence or
        # malformed value is a server configuration failure, never a client
        # error. Only the failure kind is logged — never the configured
        # values or any key material.
        try:
            keyring = load_keyring()
        except MasterKeyError as exc:
            logger.error("master key configuration unavailable: %s", exc)
            raise HTTPException(status_code=500, detail="encryption unavailable")

        with session_factory() as session:
            # Same-scope data_id uniqueness is enforced by the primary key;
            # the pre-check yields the documented 409 for plain retries, the
            # IntegrityError below covers the concurrent race.
            existing = session.get(
                DataEnvelope, (body.tenant_id, body.workload_id, body.data_id)
            )
            if existing is not None:
                raise HTTPException(
                    status_code=409, detail="data_id already exists in this scope"
                )

            # Encrypt outside any durable state: the plaintext payload and
            # plaintext data key live only in local variables, are encoded
            # into the row, and are never logged or placed on a response.
            # encrypt_payload self-verifies unwrap + authenticated decrypt,
            # so a failure here leaves no record at all.
            try:
                sealed = encrypt_payload(
                    keyring.current_key(), body.payload.encode("utf-8")
                )
            except Exception:
                session.rollback()
                logger.error("payload encryption failed for data envelope")
                raise HTTPException(status_code=500, detail="encryption failed")

            now = _utcnow()
            # Sequence the creation from the per-scope envelope counter,
            # in this same write transaction. The read-only directory
            # bounds a replayable snapshot by this commit high-water mark:
            # an envelope created after a range's first page lies beyond
            # the mark and never enters that range, even when its data_id
            # would sort earlier. Rewrap never advances this sequence.
            commit_seq = _next_data_envelope_commit_seq(
                session, body.tenant_id, body.workload_id
            )
            envelope = DataEnvelope(
                tenant_id=body.tenant_id,
                workload_id=body.workload_id,
                data_id=body.data_id,
                key_version=keyring.current_version,
                ciphertext=sealed.ciphertext,
                iv=sealed.iv,
                tag=sealed.tag,
                wrapped_key=sealed.wrapped_key,
                created_at=now,
                commit_seq=commit_seq,
            )
            session.add(envelope)
            try:
                # All material columns are NOT NULL in one row, so this
                # commit is the single atomic write of the full envelope.
                session.commit()
            except IntegrityError:
                # A concurrent request inserted the same (scope, data_id).
                session.rollback()
                raise HTTPException(
                    status_code=409, detail="data_id already exists in this scope"
                )
            except Exception:
                session.rollback()
                logger.error("data envelope write failed")
                raise HTTPException(status_code=500, detail="encryption failed")

        return DataEnvelopeCreatedResponse(
            data_id=body.data_id,
            tenant_id=body.tenant_id,
            workload_id=body.workload_id,
            key_version=keyring.current_version,
            created_at=_rfc3339(now),
        )

    @app.get(
        "/v1/data-envelopes",
    )
    def list_data_envelopes(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        data_id: str | None = Query(default=None),
        created_after: str | None = Query(default=None),
        created_before: str | None = Query(default=None),
        cursor: str | None = Query(default=None),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return a read-only, cursor-stable page of envelope metadata.

        The range is fixed by the mandatory non-blank tenant and workload
        and may be narrowed by one explicit non-blank ``data_id`` and an
        inclusive UTC creation-time window. Ordering is stable
        ``data_id`` ascending with an exclusive keyset cursor. The cursor
        carries its own kind tag, is HMAC-authenticated and is bound to
        the scope, every active filter *and* the fixed snapshot
        established by the range's first (cursor-less or explicit
        empty-cursor) query, so it can be neither forged nor replayed
        against a different scope, filter set, snapshot or query family.

        The snapshot is fixed at the per-scope envelope-creation commit
        boundary: only creation advances the per-scope gap-free counter,
        so an envelope created after the range began is excluded from
        every replayed page even when its data_id sorts earlier, and a
        concurrent rewrap only changes the reported current
        ``key_version`` without changing ordering or membership. The
        handler issues only SELECTs of metadata columns — never
        ciphertext, iv, tag, wrapped_key, payload or any key — never
        writes, rotates, audits or consumes a rate-limit slot, and a
        storage failure aborts the whole request with a 500 rather than
        returning a half page.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {
            "tenant_id",
            "workload_id",
            "data_id",
            "created_after",
            "created_before",
            "cursor",
        }
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is rejected rather than silently
        # treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )

        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # Explicit data_id selector. A blank/whitespace value is a shape
        # error; a non-blank value is matched verbatim. Unlike the window
        # bounds below, an explicit selector names one resource, so an
        # unknown or cross-scope data_id is a 404 rather than an empty
        # range.
        data_filter: str | None = None
        if data_id is not None:
            if not data_id.strip():
                raise HTTPException(
                    status_code=422, detail="invalid data identifier"
                )
            data_filter = data_id

        def _time_bound(value: str | None, name: str) -> tuple[str, datetime | None]:
            if value is None:
                return "", None
            if not value.strip():
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            # The contract admits only an explicit UTC denoter: a trailing
            # ``Z`` or ``+00:00``. The generic parser also tolerates other
            # zero-offset spellings (e.g. ``+0000``); reject them here so
            # the accepted shape is exactly the documented one.
            if not (value.endswith("Z") or value.endswith("+00:00")):
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            try:
                parsed = _parse_utc_rfc3339(value)
            except ValueError:
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            # Normalize the spelling embedded into the cursor so two
            # equivalent UTC spellings cannot mint different cursor
            # domains.
            return _rfc3339(parsed), parsed

        after_raw, after_dt = _time_bound(created_after, "created_after")
        before_raw, before_dt = _time_bound(created_before, "created_before")
        # The window is closed on both ends; equality is a valid
        # single-instant window and the start must not follow the end.
        if after_dt is not None and before_dt is not None and after_dt > before_dt:
            raise HTTPException(
                status_code=422,
                detail="created_after must not be later than created_before",
            )

        # Omitted cursor or an explicit empty string starts before the
        # smallest data_id and fixes the range's replayable snapshot.
        # Whitespace, malformed, forged, cross-scope, cross-filter,
        # cross-snapshot or foreign-kind cursors are indistinguishable
        # 422s and are rejected before any envelope is read.
        boundary_data_id: str | None = None
        snapshot_seq: int | None = None
        if cursor is not None and cursor != "":
            if not cursor.strip() or not _CURSOR_RE.fullmatch(cursor):
                raise HTTPException(status_code=422, detail="invalid cursor")
            decoded_boundary = _decode_data_envelope_cursor(
                cursor,
                tenant_id,
                workload_id,
                data_id=data_filter or "",
                created_after=after_raw,
                created_before=before_raw,
            )
            if decoded_boundary is None:
                raise HTTPException(status_code=422, detail="invalid cursor")
            boundary_data_id, snapshot_seq = decoded_boundary

        # --- read-only metadata scan ------------------------------------
        rows: list = []
        try:
            with session_factory() as session:
                # An explicitly named envelope must exist in exactly this
                # tenant and workload; an unknown data_id and one owned by
                # another scope are indistinguishable and both return 404.
                # Only a primary-key column is read for the existence
                # probe — never the material columns.
                if data_filter is not None:
                    named = session.scalar(
                        select(DataEnvelope.data_id).where(
                            DataEnvelope.tenant_id == tenant_id,
                            DataEnvelope.workload_id == workload_id,
                            DataEnvelope.data_id == data_filter,
                        )
                    )
                    if named is None:
                        raise HTTPException(
                            status_code=404, detail="data envelope not found"
                        )

                if snapshot_seq is None:
                    # First query of the range: fix a replayable snapshot
                    # at the scope's current creation-commit high-water
                    # mark. Only creation advances this counter, so an
                    # envelope committed afterwards takes a greater
                    # sequence and lies beyond the mark; a rewrap updates
                    # material in place and changes neither the counter
                    # nor the membership.
                    fixed_seq = session.scalar(
                        select(DataEnvelopeCommitCounter.last_seq).where(
                            DataEnvelopeCommitCounter.tenant_id == tenant_id,
                            DataEnvelopeCommitCounter.workload_id == workload_id,
                        )
                    )
                    # No committed envelope in the scope yet.
                    snapshot_seq = int(fixed_seq) if fixed_seq is not None else 0

                # Select only the five metadata fields. The material
                # columns (ciphertext, iv, tag, wrapped_key) are never
                # read, so neither plaintext nor sealed material exists on
                # this path; commit_seq bounds membership but is not
                # returned.
                stmt = select(
                    DataEnvelope.data_id,
                    DataEnvelope.tenant_id,
                    DataEnvelope.workload_id,
                    DataEnvelope.key_version,
                    DataEnvelope.created_at,
                ).where(
                    DataEnvelope.tenant_id == tenant_id,
                    DataEnvelope.workload_id == workload_id,
                    # Membership: envelopes whose creation had committed
                    # by the fixed snapshot. Envelopes are never deleted.
                    DataEnvelope.commit_seq <= snapshot_seq,
                )
                if data_filter is not None:
                    stmt = stmt.where(DataEnvelope.data_id == data_filter)
                if after_dt is not None:
                    stmt = stmt.where(DataEnvelope.created_at >= after_dt)
                if before_dt is not None:
                    stmt = stmt.where(DataEnvelope.created_at <= before_dt)
                if boundary_data_id is not None:
                    # Exclusive data_id keyset; data_id is the immutable
                    # primary key, so the boundary walks the same fixed
                    # snapshot set in stable order.
                    stmt = stmt.where(DataEnvelope.data_id > boundary_data_id)
                stmt = stmt.order_by(DataEnvelope.data_id.asc()).limit(
                    DATA_ENVELOPE_PAGE_SIZE + 1
                )
                # One extra row is the "more follows" probe. The scan is a
                # single read-only statement: a storage failure aborts the
                # whole request with a 500 rather than returning a partial
                # page.
                rows = list(session.execute(stmt).all())
        except HTTPException:
            raise
        except Exception:
            logger.error("data envelope directory query failed")
            raise HTTPException(
                status_code=500, detail="data envelope directory unavailable"
            )

        has_more = len(rows) > DATA_ENVELOPE_PAGE_SIZE
        page = rows[:DATA_ENVELOPE_PAGE_SIZE]

        envelopes = [
            {
                # Exactly the five metadata fields, in fixed order; no
                # ciphertext, iv, tag, wrapped_key, payload or key.
                "data_id": row.data_id,
                "tenant_id": row.tenant_id,
                "workload_id": row.workload_id,
                "key_version": int(row.key_version),
                "created_at": _rfc3339(row.created_at),
            }
            for row in page
        ]

        if has_more:
            last = page[-1]
            next_cursor = _encode_data_envelope_cursor(
                tenant_id,
                workload_id,
                last.data_id,
                data_id=data_filter or "",
                created_after=after_raw,
                created_before=before_raw,
                snapshot_seq=snapshot_seq,
            )
            complete = False
        else:
            next_cursor = ""
            complete = True

        # Compact container (envelopes, next_cursor, complete) with a
        # single terminating newline. key_version is a Python int and
        # complete a boolean, so no floats, -0.0 or non-finite values can
        # appear. No envelope material, payload, key, proof or exception
        # text is ever included.
        body = (
            json.dumps(
                {
                    "envelopes": envelopes,
                    "next_cursor": next_cursor,
                    "complete": complete,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get(
        "/v1/data-envelopes/{data_id}",
        response_model=DataEnvelopeResponse,
    )
    def get_data_envelope(
        data_id: str,
        tenant_id: str = Query(..., min_length=1),
        workload_id: str = Query(..., min_length=1),
    ) -> DataEnvelopeResponse:
        if not data_id.strip() or not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")
        with session_factory() as session:
            # The composite key binds the row to exactly this scope: an
            # unknown data_id and a data_id belonging to another tenant or
            # workload are indistinguishable and both return 404.
            envelope = session.get(
                DataEnvelope, (tenant_id, workload_id, data_id)
            )
            if envelope is None:
                raise HTTPException(status_code=404, detail="data envelope not found")
            # Read-only path: the stored material is returned encoded as
            # received and is never unwrapped, decrypted, or logged, so the
            # plaintext never exists on this path at all.
            return DataEnvelopeResponse(
                data_id=envelope.data_id,
                tenant_id=envelope.tenant_id,
                workload_id=envelope.workload_id,
                key_version=envelope.key_version,
                created_at=_rfc3339(envelope.created_at),
                ciphertext=b64url_encode(envelope.ciphertext),
                iv=b64url_encode(envelope.iv),
                tag=b64url_encode(envelope.tag),
                wrapped_key=b64url_encode(envelope.wrapped_key),
            )

    @app.post(
        "/v1/data-envelopes/{data_id}/rewrap",
        response_model=DataEnvelopeRewrappedResponse,
    )
    def rewrap_data_envelope(
        data_id: str, body: RewrapDataEnvelopeRequest
    ) -> DataEnvelopeRewrappedResponse:
        if not data_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")
        # The keyring must name both the envelope's stored version (to
        # unwrap) and the current version (to re-wrap). Any configuration
        # problem is a server failure; only the failure kind is logged,
        # never configured values or key material.
        try:
            keyring = load_keyring()
        except MasterKeyError as exc:
            logger.error("master key configuration unavailable: %s", exc)
            raise HTTPException(status_code=500, detail="encryption unavailable")

        rotated_at = _utcnow()
        with session_factory() as session:
            # The composite key binds the row to exactly this scope; unknown
            # and cross-scope data_ids are indistinguishable. On SQLite the
            # transaction already holds the write lock (BEGIN IMMEDIATE);
            # on locking backends the guarded update below serializes
            # concurrent rotations of the same row.
            envelope = session.get(
                DataEnvelope, (body.tenant_id, body.workload_id, data_id)
            )
            if envelope is None:
                raise HTTPException(status_code=404, detail="data envelope not found")

            stored_version = envelope.key_version
            if stored_version != keyring.current_version:
                try:
                    unwrapping_key = keyring.key_for(stored_version)
                except MasterKeyError:
                    # The historical key needed to unwrap is gone; the row
                    # must stay exactly as it is.
                    logger.error(
                        "master key version %s unavailable for rewrap",
                        stored_version,
                    )
                    raise HTTPException(
                        status_code=500, detail="encryption unavailable"
                    )
                # Unwrap under the stored version and re-wrap under the
                # current one. The plaintext data key lives only in local
                # variables; ciphertext, iv, tag, created_at and data_id
                # are untouched. On any failure the transaction is rolled
                # back and the stored row is unchanged.
                to_version = keyring.current_version
                try:
                    new_wrapped_key = rewrap_data_key(
                        unwrapping_key, keyring.current_key(), envelope.wrapped_key
                    )
                except Exception:
                    session.rollback()
                    logger.error("data envelope rewrap failed")
                    raise HTTPException(status_code=500, detail="encryption failed")
                # Guarded update: rotate only if the row still holds the
                # version we unwrapped. A concurrent rotation that settled
                # first leaves current-version material, which is retained.
                result = session.execute(
                    update(DataEnvelope)
                    .where(
                        DataEnvelope.tenant_id == body.tenant_id,
                        DataEnvelope.workload_id == body.workload_id,
                        DataEnvelope.data_id == data_id,
                        DataEnvelope.key_version == stored_version,
                    )
                    .values(
                        key_version=to_version, wrapped_key=new_wrapped_key
                    )
                    .execution_options(synchronize_session=False)
                )
                if result.rowcount != 1:
                    session.rollback()
                    fresh = session.get(
                        DataEnvelope, (body.tenant_id, body.workload_id, data_id)
                    )
                    if fresh is None:
                        raise HTTPException(
                            status_code=404, detail="data envelope not found"
                        )
                    return DataEnvelopeRewrappedResponse(
                        data_id=fresh.data_id,
                        tenant_id=fresh.tenant_id,
                        workload_id=fresh.workload_id,
                        key_version=fresh.key_version,
                        rotated_at=_rfc3339(rotated_at),
                    )
                # The compliance event commits in the same transaction as
                # the material rotation; an already-current no-op retry
                # (above) writes no event because it changes no material.
                _append_audit_event(
                    session,
                    tenant_id=body.tenant_id,
                    workload_id=body.workload_id,
                    event_type=AUDIT_EVENT_TYPE_REWRAP,
                    grant_id=None,
                    decision_id=None,
                    data_id=data_id,
                    status=AUDIT_EVENT_STATUS_REWRAPPED,
                    capability_sha256=None,
                    occurred_at=rotated_at,
                )
                try:
                    session.commit()
                except Exception:
                    session.rollback()
                    logger.error("data envelope rewrap write failed")
                    raise HTTPException(status_code=500, detail="encryption failed")
                final_version = to_version
            else:
                # Already wrapped under the current version: a retry changes
                # no material and simply reports the stored state.
                final_version = stored_version

        return DataEnvelopeRewrappedResponse(
            data_id=data_id,
            tenant_id=body.tenant_id,
            workload_id=body.workload_id,
            key_version=final_version,
            rotated_at=_rfc3339(rotated_at),
        )

    def _compact_json(payload: dict) -> Response:
        # Compact JSON, no trailing newline. Counts and key versions are
        # Python ints (never floats, so no -0.0 or non-finite values) and
        # allow_nan=False makes that invariant explicit.
        body = json.dumps(
            payload, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return Response(content=body, media_type="application/json")

    def _compact_json_line(payload: dict, *, status_code: int = 200) -> Response:
        # As _compact_json, but terminated by exactly one newline — the
        # wire form used by the asynchronous rewrap job endpoints.
        body = (
            json.dumps(
                payload, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            + b"\n"
        )
        return Response(
            content=body, status_code=status_code, media_type="application/json"
        )

    @app.get("/v1/verifiers")
    def list_verifiers(
        request: Request,
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return the read-only directory of registered evidence formats.

        The directory is the complete set of ``evidence_format`` names the
        verifier registry currently accepts, de-duplicated and sorted
        strictly ascending by Unicode code point, plus the entry count.
        It admits no filtering or pagination: any query parameter (known,
        unknown, repeated, or blank) and any non-empty request body is a
        422 raised before the registry is read, so a malformed request
        never reveals whether a format exists.

        The handler is purely read-only: it touches no challenge, evidence
        or release state, consumes no rate-limit budget, writes nothing,
        and never invokes a verifier. The response carries only the
        registered names — never plugin instances, class or module names,
        configuration, secrets, certificates, evidence, or exception text.
        A registry read or snapshot failure is a 500 with no partial list;
        the identical request succeeds once the registry is readable again.
        """
        # --- request shape (all 422, registry not yet read) -------------
        if request.query_params.multi_items():
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )

        # One complete, consistent snapshot of the registered names. A
        # concurrent register/replace/unregister may observe the state
        # before or after the change, but never a partial or duplicated
        # view; a registry failure aborts the whole request with a 500.
        try:
            formats = list(registry.format_names())
        except Exception:
            logger.error("verifier registry snapshot failed")
            raise HTTPException(
                status_code=500, detail="verifier registry unavailable"
            )

        # Exactly two fields in a fixed order (formats, then count). The
        # names are emitted verbatim as registered; compact JSON
        # terminated by a single newline.
        body = (
            json.dumps(
                {"formats": formats, "count": len(formats)},
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/key-rotation/status")
    def get_key_rotation_status(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return a read-only key-rotation inventory for one scope.

        The range is fixed entirely by the two mandatory, non-blank query
        parameters; the request body is always empty. Every shape failure
        (a missing or blank parameter, a repeated or unknown parameter, or
        any non-empty body) is rejected as a 422 before any envelope is
        read. The handler then takes a single consistent committed
        snapshot of the scope's envelopes and never writes: it rotates,
        re-wraps, or deletes nothing, appends no audit row, and returns no
        data_id, payload, key, or envelope material — only per-version
        counts. A storage failure or an unusable master keyring is a 500
        with no partial inventory.
        """
        # --- request shape (all 422, no envelope is read) ---------------
        allowed_params = {"tenant_id", "workload_id"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each scope parameter is a single scalar string; a repeated
        # parameter (a multi-valued/list value) is the wrong shape, not a
        # silently last-wins scalar.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(
                status_code=422, detail="query parameter must appear once"
            )
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # The keyring classifies every recorded version: current_version
        # is reported, and a recorded version whose key is no longer
        # configured is unavailable. A missing or malformed keyring —
        # including a current_version with no configured key — is a server
        # configuration failure, never a client error; only the failure
        # kind is logged, never configured values or key material.
        try:
            keyring = load_keyring()
        except MasterKeyError as exc:
            logger.error("master key configuration unavailable: %s", exc)
            raise HTTPException(
                status_code=500, detail="key rotation status unavailable"
            )

        # One GROUP BY statement over the scope. A single statement
        # evaluates against one consistent committed snapshot on every
        # backend, so an envelope created or re-wrapped concurrently is
        # counted exactly once, under exactly one version, and the tally
        # can never be a half-applied mix of two commits.
        tally_stmt = (
            select(DataEnvelope.key_version, func.count())
            .where(
                DataEnvelope.tenant_id == tenant_id,
                DataEnvelope.workload_id == workload_id,
            )
            .group_by(DataEnvelope.key_version)
        )
        try:
            with session_factory() as session:
                rows = session.execute(tally_stmt).all()
        except Exception:
            logger.error("key rotation status query failed")
            raise HTTPException(
                status_code=500, detail="key rotation status unavailable"
            )

        counts = {int(version): int(count) for version, count in rows}
        current_key_version = keyring.current_version
        total = sum(counts.values())
        at_current = counts.get(current_key_version, 0)
        behind = total - at_current
        # Versions are reported as JSON object keys (strings) ordered by
        # their numeric value; only versions actually present in the scope
        # appear, so every count is positive and they sum to total.
        by_key_version = {str(version): counts[version] for version in sorted(counts)}
        # Versions still in use whose unwrapping key is not configured:
        # those envelopes cannot be rotated until the key returns.
        unavailable_key_versions = [
            version for version in sorted(counts) if version not in keyring.keys
        ]
        # Ready to drop historical keys only when nothing is left behind:
        # an empty scope, or every envelope already at the current version
        # with no unavailable version in use. This flag only reports; it
        # never changes the keyring or any envelope.
        scope_ready_for_key_drop = total == 0 or (
            not unavailable_key_versions and behind == 0
        )

        # Exactly ten fields in a fixed order. Every count is a Python int
        # produced by SQL count aggregation (never a float), so no -0.0 or
        # non-finite value is possible; allow_nan=False makes that
        # explicit. Compact JSON terminated by a single newline.
        return _compact_json_line(
            {
                "tenant_id": tenant_id,
                "workload_id": workload_id,
                "current_key_version": current_key_version,
                "total": total,
                "at_current": at_current,
                "behind": behind,
                "by_key_version": by_key_version,
                "unavailable_key_versions": unavailable_key_versions,
                "scope_ready_for_key_drop": scope_ready_for_key_drop,
                "generated_at": _rfc3339(_utcnow()),
            }
        )

    @app.post("/v1/rewrap-batches")
    def create_rewrap_batch(body: CreateRewrapBatchRequest) -> Response:
        # Authenticate the cursor against this exact scope before touching
        # the keyring or storage: a forged, tampered or cross-scope cursor
        # is a client error indistinguishable from any other bad field.
        boundary = (
            _decode_cursor(body.cursor, body.tenant_id, body.workload_id)
            if body.cursor
            else ""
        )
        if boundary is None:
            raise HTTPException(status_code=422, detail="invalid cursor")

        # A wholly unusable keyring is a server configuration failure:
        # no batch row and no envelope may exist as evidence of the call.
        try:
            load_keyring()
        except MasterKeyError as exc:
            logger.error("master key configuration unavailable: %s", exc)
            raise HTTPException(status_code=500, detail="encryption unavailable")

        batch_id = str(uuid.uuid4())
        batch_created_at = _utcnow()
        counts = {"processed": 0, "rewrapped": 0, "skipped": 0, "failed": 0}
        last_processed_data_id = boundary
        failure_code: str | None = None

        with session_factory() as session:
            # Persist the batch before processing any envelope so the page
            # is queryable even if it stops on the first item.
            batch = RewrapBatch(
                batch_id=batch_id,
                tenant_id=body.tenant_id,
                workload_id=body.workload_id,
                limit=body.limit,
                cursor=boundary,
                next_cursor="",
                complete=False,
                processed=0,
                rewrapped=0,
                skipped=0,
                failed=0,
                created_at=batch_created_at,
            )
            session.add(batch)
            try:
                session.commit()
            except Exception:
                session.rollback()
                logger.error("rewrap batch write failed")
                raise HTTPException(status_code=500, detail="batch unavailable")

            # Stable per-page snapshot of the scope, ordered by data_id,
            # starting strictly after the exclusive cursor boundary. One
            # extra row is fetched as a "more follows" probe, so a page
            # that fills the limit exactly at the scope end still reports
            # completion rather than another full-looking page.
            try:
                rows = list(
                    session.scalars(
                        select(DataEnvelope)
                        .where(
                            DataEnvelope.tenant_id == body.tenant_id,
                            DataEnvelope.workload_id == body.workload_id,
                            DataEnvelope.data_id > boundary,
                        )
                        .order_by(DataEnvelope.data_id.asc())
                        .limit(body.limit + 1)
                    )
                )
            except Exception:
                session.rollback()
                logger.error("rewrap batch scan failed")
                raise HTTPException(status_code=500, detail="rewrap batch failed")
            has_more = len(rows) > body.limit
            page = rows[: body.limit]

            def _record_stop(
                code: str, data_id: str, key_version: int
            ) -> None:
                # The failed envelope is left exactly as it was: undo the
                # attempted update, then commit only its audit row. The
                # resume cursor stays before the failed item.
                session.rollback()
                nonlocal failure_code
                failure_code = code
                session.add(
                    RewrapBatchItem(
                        item_id=str(uuid.uuid4()),
                        batch_id=batch_id,
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        data_id=data_id,
                        old_key_version=key_version,
                        new_key_version=key_version,
                        result=code,
                        seq=counts["processed"],
                        created_at=_utcnow(),
                    )
                )
                counts["processed"] += 1
                counts["failed"] += 1
                stored = session.get(RewrapBatch, batch_id)
                stored.processed = counts["processed"]
                stored.failed = counts["failed"]
                stored.next_cursor = (
                    _encode_cursor(
                        body.tenant_id, body.workload_id, last_processed_data_id
                    )
                    if last_processed_data_id
                    else ""
                )
                stored.complete = False
                session.commit()

            for envelope in page:
                data_id = envelope.data_id
                stored_version = envelope.key_version

                # Re-resolve the keyring per envelope: a keyring that
                # becomes unusable mid-page stops the page with a keyring
                # result on the envelope that could not be handled.
                try:
                    keyring = load_keyring()
                except MasterKeyError:
                    logger.error("master keyring became unavailable mid-batch")
                    _record_stop(REWRAP_RESULT_KEYRING, data_id, stored_version)
                    break

                current_version = keyring.current_version
                # Version pair recorded on the audit; the concurrent-loser
                # path below reports both sides as the current version.
                audit_old_version = stored_version
                if stored_version == current_version:
                    result = REWRAP_RESULT_SKIPPED
                    new_version = stored_version
                else:
                    try:
                        unwrapping_key = keyring.key_for(stored_version)
                        new_wrapped_key = rewrap_data_key(
                            unwrapping_key,
                            keyring.current_key(),
                            envelope.wrapped_key,
                        )
                    except MasterKeyError:
                        # The historical key needed to unwrap is gone; the
                        # envelope is not modified.
                        logger.error(
                            "master key version %s unavailable during batch",
                            stored_version,
                        )
                        _record_stop(
                            REWRAP_RESULT_MISSING_KEY, data_id, stored_version
                        )
                        break
                    except Exception:
                        # Any crypto/rewrap failure: roll the material
                        # change back and stop with the row untouched.
                        logger.error("data envelope batch rewrap failed")
                        _record_stop(
                            REWRAP_RESULT_REWRAP_FAILED,
                            data_id,
                            stored_version,
                        )
                        break

                    # Guarded update, mirroring the single-envelope path:
                    # only a row still at the version we unwrapped can be
                    # rotated. A concurrent winner leaves current-version
                    # material, which this page records as a skip.
                    outcome = session.execute(
                        update(DataEnvelope)
                        .where(
                            DataEnvelope.tenant_id == body.tenant_id,
                            DataEnvelope.workload_id == body.workload_id,
                            DataEnvelope.data_id == data_id,
                            DataEnvelope.key_version == stored_version,
                        )
                        .values(
                            key_version=current_version,
                            wrapped_key=new_wrapped_key,
                        )
                        .execution_options(synchronize_session=False)
                    )
                    if outcome.rowcount != 1:
                        session.rollback()
                        fresh = session.scalar(
                            select(DataEnvelope)
                            .where(
                                DataEnvelope.tenant_id == body.tenant_id,
                                DataEnvelope.workload_id == body.workload_id,
                                DataEnvelope.data_id == data_id,
                            )
                            .execution_options(populate_existing=True)
                        )
                        if fresh is not None and fresh.key_version == current_version:
                            # A concurrent rewrap advanced it first; the
                            # envelope is now current and is never
                            # re-wrapped by this page. Record the skip as
                            # observed: already at the current version.
                            result = REWRAP_RESULT_SKIPPED
                            new_version = current_version
                            audit_old_version = current_version
                        else:
                            _record_stop(
                                REWRAP_RESULT_REWRAP_FAILED,
                                data_id,
                                stored_version,
                            )
                            break
                    else:
                        result = REWRAP_RESULT_REWRAPPED
                        new_version = current_version

                # Independent per-envelope commit: one audit row each,
                # durable before the next envelope is attempted.
                item_occurred_at = _utcnow()
                event_status = (
                    AUDIT_EVENT_STATUS_REWRAPPED
                    if result == REWRAP_RESULT_REWRAPPED
                    else AUDIT_EVENT_STATUS_SKIPPED
                )
                session.add(
                    RewrapBatchItem(
                        item_id=str(uuid.uuid4()),
                        batch_id=batch_id,
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        data_id=data_id,
                        old_key_version=audit_old_version,
                        new_key_version=new_version,
                        result=result,
                        seq=counts["processed"],
                        created_at=item_occurred_at,
                    )
                )
                # The compliance event commits in the same transaction as
                # the envelope material and the batch audit row. Rewrap
                # events carry the envelope data identifier and the fixed
                # rewrapped/skipped status; grant/decision identifiers and
                # the capability digest are empty (NULL).
                _append_audit_event(
                    session,
                    tenant_id=body.tenant_id,
                    workload_id=body.workload_id,
                    event_type=AUDIT_EVENT_TYPE_REWRAP,
                    grant_id=None,
                    decision_id=None,
                    data_id=data_id,
                    status=event_status,
                    capability_sha256=None,
                    occurred_at=item_occurred_at,
                )
                counts["processed"] += 1
                if result == REWRAP_RESULT_REWRAPPED:
                    counts["rewrapped"] += 1
                else:
                    counts["skipped"] += 1
                last_processed_data_id = data_id
                stored_batch = session.get(RewrapBatch, batch_id)
                stored_batch.processed = counts["processed"]
                stored_batch.rewrapped = counts["rewrapped"]
                stored_batch.skipped = counts["skipped"]
                stored_batch.failed = counts["failed"]
                try:
                    session.commit()
                except Exception:
                    session.rollback()
                    logger.error("rewrap batch audit write failed")
                    raise HTTPException(status_code=500, detail="rewrap batch failed")
            else:
                # Every envelope in the page was reached without a stop.
                final_batch = session.get(RewrapBatch, batch_id)
                if not has_more:
                    # The limit+1 probe found nothing beyond this page, so
                    # the scope has been fully scanned.
                    final_batch.next_cursor = ""
                    final_batch.complete = True
                    next_cursor = ""
                    complete = True
                else:
                    # A full page: there may be more. The cursor is the
                    # last processed data_id; retrying it skips envelopes
                    # already at the current version instead of rewrapping.
                    next_cursor = _encode_cursor(
                        body.tenant_id,
                        body.workload_id,
                        last_processed_data_id,
                    )
                    final_batch.next_cursor = next_cursor
                    final_batch.complete = False
                    complete = False
                final_batch.processed = counts["processed"]
                final_batch.rewrapped = counts["rewrapped"]
                final_batch.skipped = counts["skipped"]
                final_batch.failed = counts["failed"]
                session.commit()

        if failure_code is not None:
            # Stopped mid-page: the resume cursor was finalized inside
            # _record_stop (it points before the failed envelope).
            with session_factory() as session:
                stored_batch = session.get(RewrapBatch, batch_id)
                next_cursor = stored_batch.next_cursor
                complete = False
                counts = {
                    "processed": stored_batch.processed,
                    "rewrapped": stored_batch.rewrapped,
                    "skipped": stored_batch.skipped,
                    "failed": stored_batch.failed,
                }

        return _compact_json(
            {
                "batch_id": batch_id,
                "processed": counts["processed"],
                "rewrapped": counts["rewrapped"],
                "skipped": counts["skipped"],
                "failed": counts["failed"],
                "next_cursor": next_cursor,
                "complete": complete,
            }
        )

    @app.get("/v1/rewrap-batches/")
    def rewrap_batch_identifier_required() -> Response:
        # An empty path segment is a missing batch identifier: a 422
        # client error rather than a routing-level 404.
        raise HTTPException(status_code=422, detail="invalid batch identifier")

    @app.get("/v1/rewrap-batches/{batch_id}")
    def get_rewrap_batch(
        batch_id: str,
        tenant_id: str = Query(..., min_length=1),
        workload_id: str = Query(..., min_length=1),
    ) -> Response:
        # The path identifier must be a syntactically valid batch id;
        # missing/blank scope query parameters are the same 422 class.
        # Batch ids are canonical lowercase UUIDs.
        if not batch_id.strip() or not _UUID_RE.fullmatch(batch_id.lower()):
            raise HTTPException(status_code=422, detail="invalid batch identifier")
        batch_id = batch_id.strip().lower()
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        try:
            with session_factory() as session:
                batch = session.get(RewrapBatch, batch_id)
                if (
                    batch is None
                    or batch.tenant_id != tenant_id
                    or batch.workload_id != workload_id
                ):
                    # Unknown and cross-scope batches are indistinguishable.
                    raise HTTPException(status_code=404, detail="rewrap batch not found")
                items = session.scalars(
                    select(RewrapBatchItem)
                    .where(RewrapBatchItem.batch_id == batch_id)
                    .order_by(RewrapBatchItem.seq.asc())
                ).all()
                payload = {
                    "batch_id": batch.batch_id,
                    "tenant_id": batch.tenant_id,
                    "workload_id": batch.workload_id,
                    "limit": batch.limit,
                    "processed": batch.processed,
                    "rewrapped": batch.rewrapped,
                    "skipped": batch.skipped,
                    "failed": batch.failed,
                    "next_cursor": batch.next_cursor,
                    "complete": batch.complete,
                    "created_at": _rfc3339(batch.created_at),
                    "audits": [
                        {
                            "data_id": item.data_id,
                            "old_key_version": item.old_key_version,
                            "new_key_version": item.new_key_version,
                            "result": item.result,
                            "audited_at": _rfc3339(item.created_at),
                        }
                        for item in items
                    ],
                }
        except HTTPException:
            raise
        except Exception:
            logger.error("rewrap batch lookup failed")
            raise HTTPException(status_code=500, detail="rewrap batch unavailable")

        return _compact_json(payload)

    # -- persistent asynchronous rewrap jobs ------------------------------

    @app.post("/v1/rewrap-jobs", status_code=202)
    def create_rewrap_job(request: Request, body: CreateRewrapJobRequest) -> Response:
        # Authenticate the cursor against this exact scope before touching
        # the keyring or storage: a forged, tampered or cross-scope cursor
        # is a client error indistinguishable from any other bad field.
        # Body field/limit validation already happened (422) before this
        # handler ran, so it can never be masked by the idempotency-key
        # checks below.
        start_boundary = (
            _decode_cursor(body.cursor, body.tenant_id, body.workload_id)
            if body.cursor
            else ""
        )
        if start_boundary is None:
            raise HTTPException(status_code=422, detail="invalid cursor")

        # The echo/wire cursor: beginning-of-scope normalizes to the empty
        # string; any other value is the verified token verbatim. An
        # omitted cursor and an explicit empty cursor therefore normalize
        # to the same start in both the job row and the fingerprint.
        start_cursor = body.cursor or ""

        # The idempotency key is optional and lives only in a header; the
        # body contract is unchanged. A missing key preserves the original
        # one-job-per-request semantics exactly. An illegal key (duplicated
        # header, empty, non-visible-ASCII or longer than 64 characters) is
        # rejected here, after the body/cursor checks and before any write,
        # audit or envelope advancement.
        idem_present, idempotency_key = _read_idempotency_key(request)

        if not idem_present:
            # A wholly unusable keyring is a server configuration failure:
            # no job row may exist as evidence of the call and no envelope
            # moves.
            try:
                load_keyring()
            except MasterKeyError as exc:
                logger.error("master key configuration unavailable: %s", exc)
                raise HTTPException(status_code=500, detail="encryption unavailable")

            job_id = str(uuid.uuid4())
            now = _utcnow()
            window_start = _utc_minute_window(now)
            try:
                with session_factory() as session:
                    # Only a genuinely new job reserves admission quota,
                    # in the same transaction that commits it: the count
                    # and the job settle or roll back together. An
                    # over-budget request writes nothing — no counter, no
                    # job, no event, no envelope movement.
                    if not _reserve_rewrap_job_slot(
                        session, body.tenant_id, body.workload_id, window_start
                    ):
                        session.rollback()
                        return _rewrap_job_rate_limited_response()
                    session.add(
                        RewrapJob(
                            job_id=job_id,
                            tenant_id=body.tenant_id,
                            workload_id=body.workload_id,
                            limit=body.limit,
                            cursor=start_cursor,
                            next_cursor=start_cursor,
                            status=REWRAP_JOB_STATUS_QUEUED,
                            processed=0,
                            rewrapped=0,
                            skipped=0,
                            failed=0,
                            complete=False,
                            claim_token=None,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    # The birth migration (no old status -> queued) is
                    # event seq 1 and commits in the same transaction as
                    # the job row, so a job always exists together with
                    # its submission event.
                    _record_rewrap_job_event(
                        session,
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        job_id=job_id,
                        old_status=None,
                        new_status=REWRAP_JOB_STATUS_QUEUED,
                        reason=REWRAP_JOB_EVENT_REASON_SUBMITTED,
                        now=now,
                    )
                    session.commit()
            except Exception:
                logger.error("rewrap job write failed")
                raise HTTPException(status_code=500, detail="rewrap job unavailable")

            # Durable acceptance first, background advancement second: even
            # if this process dies before a worker picks it up, the queued
            # row is resumed by the next process's startup sweep.
            app.state.rewrap_job_runner.submit(job_id)

            return _compact_json_line(
                {
                    "job_id": job_id,
                    "status": REWRAP_JOB_STATUS_QUEUED,
                    "limit": body.limit,
                    "cursor": start_cursor,
                    "created_at": _rfc3339(now),
                    "updated_at": _rfc3339(now),
                },
                status_code=202,
            )

        fingerprint = _rewrap_job_request_fingerprint(
            body.tenant_id, body.workload_id, body.limit, start_cursor
        )

        # Keyed submission. The idempotency record and the job row are one
        # atomic commit: the lookup-then-insert runs in a single
        # transaction, with the unique (scope, key) constraint plus the
        # IntegrityError retry below settling concurrent identical
        # submissions so at most one job is ever created per key.
        job_id: str | None = None
        acceptance_body: str | None = None
        for _ in range(2):
            with session_factory() as session:
                try:
                    existing = session.scalar(
                        select(RewrapJobIdempotencyRecord).where(
                            RewrapJobIdempotencyRecord.tenant_id
                            == body.tenant_id,
                            RewrapJobIdempotencyRecord.workload_id
                            == body.workload_id,
                            RewrapJobIdempotencyRecord.idempotency_key
                            == idempotency_key,
                        )
                    )
                    if existing is not None:
                        # A replay never creates a job, never advances an
                        # envelope and never re-checks the keyring: even if
                        # the keyring later becomes unusable, the stored
                        # 202 is returned verbatim. A same-key request with
                        # a different scope-normalized shape is a stable
                        # 409 that changes neither the original job nor its
                        # saved response.
                        if not hmac.compare_digest(
                            existing.request_fingerprint, fingerprint
                        ):
                            raise HTTPException(
                                status_code=409,
                                detail="idempotency key reused with a different request",
                            )
                        return Response(
                            content=existing.response_body.encode("utf-8"),
                            status_code=202,
                            media_type="application/json",
                        )

                    # First submission for this scope+key. A wholly
                    # unusable keyring is a server failure; the check sits
                    # inside the transaction so a failure rolls back (no
                    # job, no idempotency record, no envelope moved).
                    try:
                        load_keyring()
                    except MasterKeyError as exc:
                        session.rollback()
                        logger.error(
                            "master key configuration unavailable: %s", exc
                        )
                        raise HTTPException(
                            status_code=500, detail="encryption unavailable"
                        )

                    job_id = str(uuid.uuid4())
                    now = _utcnow()
                    window_start = _utc_minute_window(now)
                    # Only a genuinely new job reserves admission quota:
                    # replays and conflicts returned above spend nothing,
                    # and the slot commits in this same transaction
                    # together with the job, its birth event and the
                    # idempotency record — or rolls back with them. An
                    # over-budget request writes nothing at all.
                    if not _reserve_rewrap_job_slot(
                        session, body.tenant_id, body.workload_id, window_start
                    ):
                        session.rollback()
                        return _rewrap_job_rate_limited_response()
                    acceptance_body = _rewrap_job_acceptance_body(
                        job_id, body.limit, start_cursor, now
                    )
                    session.add(
                        RewrapJob(
                            job_id=job_id,
                            tenant_id=body.tenant_id,
                            workload_id=body.workload_id,
                            limit=body.limit,
                            cursor=start_cursor,
                            next_cursor=start_cursor,
                            status=REWRAP_JOB_STATUS_QUEUED,
                            processed=0,
                            rewrapped=0,
                            skipped=0,
                            failed=0,
                            complete=False,
                            claim_token=None,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    session.add(
                        RewrapJobIdempotencyRecord(
                            record_id=str(uuid.uuid4()),
                            tenant_id=body.tenant_id,
                            workload_id=body.workload_id,
                            idempotency_key=idempotency_key,
                            job_id=job_id,
                            request_fingerprint=fingerprint,
                            response_body=acceptance_body,
                            created_at=now,
                        )
                    )
                    # The birth event commits together with the job row
                    # and its idempotency record: a crash can never leave
                    # a job without its submission event.
                    _record_rewrap_job_event(
                        session,
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        job_id=job_id,
                        old_status=None,
                        new_status=REWRAP_JOB_STATUS_QUEUED,
                        reason=REWRAP_JOB_EVENT_REASON_SUBMITTED,
                        now=now,
                    )
                    try:
                        # The job and its idempotency record commit
                        # together: a crash can never leave a half job or a
                        # record pointing at nothing.
                        session.commit()
                    except IntegrityError:
                        # A concurrent request for the same scope+key
                        # committed first. Reread its record and answer as
                        # a replay (verbatim 202) or a conflict (409).
                        session.rollback()
                        continue
                    except Exception:
                        session.rollback()
                        logger.error("rewrap job write failed")
                        raise HTTPException(
                            status_code=500, detail="rewrap job unavailable"
                        )
                    break
                except HTTPException:
                    raise
                except Exception:
                    # A storage failure during the lookup or insert is a
                    # server failure: the context manager rolls the
                    # transaction back, so neither a job row nor an
                    # idempotency record is left behind and no envelope
                    # moves.
                    logger.error("rewrap job idempotent submission failed")
                    raise HTTPException(
                        status_code=500, detail="rewrap job unavailable"
                    )
        else:  # pragma: no cover - defensive: the reread always settles
            logger.error("rewrap job idempotency race did not settle")
            raise HTTPException(status_code=500, detail="rewrap job unavailable")
        assert job_id is not None and acceptance_body is not None

        # Durable acceptance first, background advancement second — exactly
        # as on the keyless path. A replay never reaches this point, so a
        # retry never re-enqueues or advances the job.
        app.state.rewrap_job_runner.submit(job_id)

        return Response(
            content=acceptance_body.encode("utf-8"),
            status_code=202,
            media_type="application/json",
        )

    @app.get("/v1/rewrap-jobs")
    def list_rewrap_jobs(
        request: Request,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        job_id: str | None = Query(default=None),
        status: str | None = Query(default=None),
        created_after: str | None = Query(default=None),
        created_before: str | None = Query(default=None),
        cursor: str | None = Query(default=None),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return a read-only page of committed rewrap job history.

        The range is fixed by the mandatory tenant/workload and may be
        narrowed by an explicit job id, a job status and an inclusive
        created-at window. Ordering is ``job_id`` ascending with an
        exclusive keyset cursor; the cursor is HMAC-authenticated, carries
        its own kind tag and is bound to the scope *and* every active
        filter, so it cannot be forged, tampered with, or replayed against
        a different scope, filter set or cursor family. The handler issues
        only SELECTs — it never writes an audit row, changes a job's
        status or advances an envelope cursor — so it observes only
        committed state and a storage failure aborts the whole request
        rather than returning half a page. Concurrent status updates
        change only the values on the rows, never the ``job_id`` ordering,
        so replaying a cursor returns the same page.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {
            "tenant_id",
            "workload_id",
            "job_id",
            "status",
            "created_after",
            "created_before",
            "cursor",
        }
        if set(request.query_params.keys()) - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )

        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # job_id is a canonical lowercase UUID; surrounding whitespace and
        # uppercase letters are format errors, mirroring the single-job
        # path normalization (uppercase normalizes only on the path form).
        job_filter: str | None = None
        if job_id is not None:
            if not job_id.strip() or not _UUID_RE.fullmatch(job_id):
                raise HTTPException(status_code=422, detail="invalid job identifier")
            job_filter = job_id

        if status is not None:
            if not status.strip() or status not in REWRAP_JOB_STATUS_CODES:
                raise HTTPException(status_code=422, detail="invalid status")
        status_filter = status if status is not None else ""

        # Timestamp filters: absent means unbounded; an explicit empty or
        # whitespace value is an illegal format (422), not "unbounded".
        # Bounds are embedded into the cursor in normalized UTC RFC3339 so
        # equivalent spellings cannot mint two different cursor domains.
        def _time_bound(value: str | None, name: str) -> tuple[str, datetime | None]:
            if value is None:
                return "", None
            if not value.strip():
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            try:
                parsed = _parse_utc_rfc3339(value)
            except ValueError:
                raise HTTPException(
                    status_code=422, detail=f"{name} must be UTC RFC3339"
                )
            return _rfc3339(parsed), parsed

        after_raw, after_dt = _time_bound(created_after, "created_after")
        before_raw, before_dt = _time_bound(created_before, "created_before")
        # The window is closed on both ends; equality is a valid
        # single-instant window and the start must not follow the end.
        if after_dt is not None and before_dt is not None and after_dt > before_dt:
            raise HTTPException(
                status_code=422,
                detail="created_after must not be later than created_before",
            )

        # An omitted cursor or an explicit empty string starts before the
        # smallest job id. Whitespace, malformed, forged, cross-scope,
        # cross-filter or foreign-kind cursors are indistinguishable 422s.
        if cursor is None or cursor == "":
            boundary = ""
        else:
            if not cursor.strip() or not _CURSOR_RE.fullmatch(cursor):
                raise HTTPException(status_code=422, detail="invalid cursor")
            boundary = _decode_rewrap_job_history_cursor(
                cursor,
                tenant_id,
                workload_id,
                job_id=job_filter or "",
                status=status_filter,
                created_after=after_raw,
                created_before=before_raw,
            )
            if boundary is None:
                raise HTTPException(status_code=422, detail="invalid cursor")

        # --- read-only history scan -------------------------------------
        try:
            with session_factory() as session:
                # An explicitly named job must exist in exactly this
                # scope; unknown and cross-scope identifiers are an
                # indistinguishable 404 rather than an empty page.
                if job_filter is not None:
                    named = session.get(RewrapJob, job_filter)
                    if (
                        named is None
                        or named.tenant_id != tenant_id
                        or named.workload_id != workload_id
                    ):
                        raise HTTPException(
                            status_code=404, detail="rewrap job not found"
                        )

                stmt = select(RewrapJob).where(
                    RewrapJob.tenant_id == tenant_id,
                    RewrapJob.workload_id == workload_id,
                )
                if job_filter is not None:
                    stmt = stmt.where(RewrapJob.job_id == job_filter)
                if status_filter:
                    stmt = stmt.where(RewrapJob.status == status_filter)
                if after_dt is not None:
                    stmt = stmt.where(RewrapJob.created_at >= after_dt)
                if before_dt is not None:
                    stmt = stmt.where(RewrapJob.created_at <= before_dt)
                stmt = (
                    stmt.where(RewrapJob.job_id > boundary)
                    .order_by(RewrapJob.job_id.asc())
                    .limit(REWRAP_JOB_HISTORY_PAGE_SIZE + 1)
                )
                # One extra row is the "more follows" probe. The scan is a
                # single read-only statement: a storage failure aborts the
                # whole request with a 500 rather than returning a partial
                # page.
                rows = list(session.scalars(stmt))
        except HTTPException:
            raise
        except Exception:
            logger.error("rewrap job history query failed")
            raise HTTPException(
                status_code=500, detail="rewrap job history unavailable"
            )

        has_more = len(rows) > REWRAP_JOB_HISTORY_PAGE_SIZE
        page = rows[:REWRAP_JOB_HISTORY_PAGE_SIZE]

        # Each entry reuses the single-job progress field order and types
        # verbatim; timestamps stay UTC RFC3339. Counters are ints,
        # complete is a bool and every other value is a string — no
        # floats, -0.0 or non-finite values can appear.
        jobs = [
            {
                "job_id": row.job_id,
                "status": row.status,
                "processed": row.processed,
                "rewrapped": row.rewrapped,
                "skipped": row.skipped,
                "failed": row.failed,
                "next_cursor": row.next_cursor,
                "complete": bool(row.complete),
                "created_at": _rfc3339(row.created_at),
                "updated_at": _rfc3339(row.updated_at),
            }
            for row in page
        ]

        if has_more:
            next_cursor = _encode_rewrap_job_history_cursor(
                tenant_id,
                workload_id,
                page[-1].job_id,
                job_id=job_filter or "",
                status=status_filter,
                created_after=after_raw,
                created_before=before_raw,
            )
            complete = False
        else:
            next_cursor = ""
            complete = True

        body = (
            json.dumps(
                {
                    "jobs": jobs,
                    "next_cursor": next_cursor,
                    "complete": complete,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/rewrap-jobs/")
    def rewrap_job_identifier_required() -> Response:
        # An empty path segment is a missing job identifier: a 422 client
        # error rather than a routing-level 404.
        raise HTTPException(status_code=422, detail="invalid job identifier")

    @app.get("/v1/rewrap-jobs/{job_id}")
    def get_rewrap_job(
        job_id: str,
        tenant_id: str = Query(..., min_length=1),
        workload_id: str = Query(..., min_length=1),
    ) -> Response:
        # The path identifier must be a syntactically valid job id;
        # missing/blank scope query parameters are the same 422 class.
        if not job_id.strip() or not _UUID_RE.fullmatch(job_id.lower()):
            raise HTTPException(status_code=422, detail="invalid job identifier")
        job_id = job_id.strip().lower()
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        try:
            with session_factory() as session:
                job = session.get(RewrapJob, job_id)
                if (
                    job is None
                    or job.tenant_id != tenant_id
                    or job.workload_id != workload_id
                ):
                    # Unknown and cross-scope jobs are indistinguishable.
                    raise HTTPException(status_code=404, detail="rewrap job not found")
                payload = {
                    "job_id": job.job_id,
                    "status": job.status,
                    "processed": job.processed,
                    "rewrapped": job.rewrapped,
                    "skipped": job.skipped,
                    "failed": job.failed,
                    "next_cursor": job.next_cursor,
                    "complete": bool(job.complete),
                    "created_at": _rfc3339(job.created_at),
                    "updated_at": _rfc3339(job.updated_at),
                }
        except HTTPException:
            raise
        except Exception:
            logger.error("rewrap job lookup failed")
            raise HTTPException(status_code=500, detail="rewrap job unavailable")

        return _compact_json_line(payload)

    @app.get("/v1/rewrap-jobs//events")
    def rewrap_job_events_identifier_required() -> Response:
        # An empty path segment is a missing job identifier: a 422 client
        # error rather than a routing-level 404. It never reads a job.
        raise HTTPException(status_code=422, detail="invalid job identifier")

    @app.get("/v1/rewrap-jobs/{job_id}/events")
    def list_rewrap_job_events(
        request: Request,
        job_id: str,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        cursor: str | None = Query(default=None),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return the read-only lifecycle event timeline of one rewrap job.

        The range is exactly one job fixed by the mandatory
        tenant/workload scope and the canonical-UUID path id; only
        ``tenant_id``, ``workload_id`` and the optional ``cursor`` are
        accepted. Events are the committed status migrations in their
        stable, immutable per-job sequence order: submission (no old
        status -> queued), execution (queued -> running), recovery
        (failed -> running), completion (running -> succeeded),
        cancellation (queued|running -> cancelled) and the fixed failure
        classifications (running -> failed). Exactly one event exists per
        migration that committed; migrations that lost a guard race left
        no event.

        The first (cursor-less or empty-cursor) query fixes a replayable
        snapshot high-water mark (the greatest event seq then committed);
        migrations committed afterwards surface only in a fresh first
        query, never in a later page of this one. A resume cursor is
        HMAC-authenticated and carries its own kind tag, so it cannot be
        forged, tampered with, or replayed against another scope, job,
        snapshot or cursor family. The handler issues only SELECTs — a
        query changes neither the job nor any envelope — and a storage
        failure aborts the whole request with a 500 rather than returning
        half a page.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {"tenant_id", "workload_id", "cursor"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is ambiguous and rejected rather than
        # silently treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(status_code=422, detail="query parameter must appear once")

        # The path identifier must be a canonical lowercase UUID; an
        # empty segment is handled by the dedicated route above.
        if not job_id or not _UUID_RE.fullmatch(job_id):
            raise HTTPException(status_code=422, detail="invalid job identifier")
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # Omitted cursor or an explicit empty string starts before the
        # first event and fixes the query family's snapshot. Whitespace,
        # malformed, forged, tampered, cross-scope, cross-job,
        # cross-snapshot or foreign-kind cursors are indistinguishable
        # 422s and are rejected without reading any state.
        boundary_seq: int = 0
        snapshot_seq: int | None = None
        if cursor is not None and cursor != "":
            if not cursor.strip() or not _CURSOR_RE.fullmatch(cursor):
                raise HTTPException(status_code=422, detail="invalid cursor")
            decoded = _decode_rewrap_job_event_cursor(
                cursor, tenant_id, workload_id, job_id
            )
            if decoded is None:
                raise HTTPException(status_code=422, detail="invalid cursor")
            boundary_seq, snapshot_seq = decoded

        # --- read-only timeline scan ------------------------------------
        try:
            with session_factory() as session:
                # The named job must exist in exactly this scope; an
                # unknown or cross-scope job is an indistinguishable 404.
                job = session.get(RewrapJob, job_id)
                if (
                    job is None
                    or job.tenant_id != tenant_id
                    or job.workload_id != workload_id
                ):
                    raise HTTPException(status_code=404, detail="rewrap job not found")

                if snapshot_seq is None:
                    # First query of the family: fix the replayable
                    # snapshot at the greatest currently committed seq.
                    # A freshly created job always commits together with
                    # its submission event; a zero here is only possible
                    # for a job row that predates this feature.
                    fixed_snapshot = session.scalar(
                        select(func.max(RewrapJobEvent.seq)).where(
                            RewrapJobEvent.job_id == job_id
                        )
                    )
                    snapshot_seq = int(fixed_snapshot or 0)

                stmt = (
                    select(RewrapJobEvent)
                    .where(
                        RewrapJobEvent.job_id == job_id,
                        RewrapJobEvent.seq > boundary_seq,
                        # Inclusive fixed snapshot: migrations committed
                        # after the first query never enter these pages.
                        RewrapJobEvent.seq <= snapshot_seq,
                    )
                    .order_by(RewrapJobEvent.seq.asc())
                    .limit(REWRAP_JOB_EVENT_PAGE_SIZE + 1)
                )
                # One extra row is the "more follows" probe. The scan is a
                # single read-only statement: a storage failure aborts the
                # whole request with a 500 rather than returning a partial
                # page.
                rows = list(session.scalars(stmt))
        except HTTPException:
            raise
        except Exception:
            logger.error("rewrap job event query failed")
            raise HTTPException(status_code=500, detail="rewrap job events unavailable")

        has_more = len(rows) > REWRAP_JOB_EVENT_PAGE_SIZE
        page = rows[:REWRAP_JOB_EVENT_PAGE_SIZE]

        events = [
            {
                "event_id": row.event_id,
                # Stable, immutable per-job sequence in committed order.
                "job_seq": row.seq,
                # Null only for the birth migration; a string otherwise.
                "old_status": row.old_status,
                "new_status": row.new_status,
                # A fixed service code, never exception text.
                "reason": row.reason,
                "created_at": _rfc3339(row.created_at),
            }
            for row in page
        ]

        if has_more:
            next_cursor = _encode_rewrap_job_event_cursor(
                tenant_id,
                workload_id,
                job_id,
                page[-1].seq,
                snapshot_seq=snapshot_seq,
            )
            complete = False
        else:
            next_cursor = ""
            complete = True

        # Compact container (events, next_cursor, complete) with a single
        # terminating newline. job_seq is an int, old_status may be null,
        # complete is a bool and every other value is a string — no
        # floats, -0.0 or non-finite values can appear.
        body = (
            json.dumps(
                {
                    "events": events,
                    "next_cursor": next_cursor,
                    "complete": complete,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.get("/v1/rewrap-jobs//items")
    def rewrap_job_items_identifier_required() -> Response:
        # An empty path segment is a missing job identifier: a 422 client
        # error rather than a routing-level 404. It never reads a job.
        raise HTTPException(status_code=422, detail="invalid job identifier")

    @app.get("/v1/rewrap-jobs/{job_id}/items")
    def list_rewrap_job_items(
        request: Request,
        job_id: str,
        tenant_id: str = Query(...),
        workload_id: str = Query(...),
        limit: str | None = Query(default=None),
        cursor: str | None = Query(default=None),
        _empty_body: None = Depends(_require_empty_query_body),
    ) -> Response:
        """Return the read-only per-envelope result listing of one rewrap job.

        The range is exactly one job fixed by the mandatory
        tenant/workload scope and the canonical-UUID path id; only
        ``tenant_id``, ``workload_id``, an optional 1..200 ``limit``
        (default 50) and an optional ``cursor`` are accepted. Items are
        the job's committed per-envelope outcomes in their stable,
        immutable per-job ``seq`` order, each carrying only ``seq``,
        ``data_id``, ``old_key_version``, ``new_key_version``, ``result``
        (``rewrapped`` or ``skipped``) and a UTC RFC3339 ``occurred_at``.
        Every item commits in the same transaction as its envelope change
        and the job's progress, so a page only ever reads committed
        results; an envelope whose rewrap failed never appears here (its
        failure classification stays on the job's event timeline), while
        the results committed before it remain queryable.

        An omitted or empty cursor starts at the job's first item;
        otherwise the cursor must be a next-page cursor this job's listing
        previously issued. The cursor is HMAC-authenticated and bound to
        the scope, the job and this pagination family, so it cannot be
        forged, tampered with, or replayed against another job or cursor
        family. Items are append-only and immutable once committed, so
        paging a still-advancing job is stable: each page continues
        strictly after the last returned ``seq`` and later commits only
        ever append. A job created before per-item recording existed has
        no rows and pages as an empty, complete listing. The handler
        issues only SELECTs — a query changes neither the job nor any
        envelope — and a storage failure aborts the whole request with a
        500 rather than returning half a page.
        """
        # --- request shape (all 422, no storage touched) ----------------
        allowed_params = {"tenant_id", "workload_id", "limit", "cursor"}
        supplied = request.query_params.multi_items()
        supplied_keys = {key for key, _ in supplied}
        if supplied_keys - allowed_params:
            raise HTTPException(
                status_code=422, detail="unsupported query parameter"
            )
        # Each parameter is a single scalar string; a repeated parameter
        # (even of an allowed name) is ambiguous and rejected rather than
        # silently treated as last-wins.
        if len(supplied) != len(supplied_keys):
            raise HTTPException(status_code=422, detail="query parameter must appear once")

        # The path identifier must be a canonical lowercase UUID; an
        # empty segment is handled by the dedicated route above.
        if not job_id or not _UUID_RE.fullmatch(job_id):
            raise HTTPException(status_code=422, detail="invalid job identifier")
        if not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid scope parameters")

        # Omitted limit defaults to 50; a supplied limit must be a decimal
        # integer in 1..200. Empty, non-integer and out-of-range values
        # are indistinguishable 422s checked before any state is read.
        if limit is None:
            page_size = REWRAP_BATCH_DEFAULT_LIMIT
        else:
            if not re.fullmatch(r"[0-9]+", limit):
                raise HTTPException(status_code=422, detail="invalid limit")
            page_size = int(limit)
            if not (
                REWRAP_BATCH_MIN_LIMIT <= page_size <= REWRAP_BATCH_MAX_LIMIT
            ):
                raise HTTPException(status_code=422, detail="invalid limit")

        # Omitted cursor or an explicit empty string starts before the
        # first item. Whitespace, malformed, forged, tampered, cross-scope,
        # cross-job or foreign-kind cursors are indistinguishable 422s and
        # are rejected without reading any state.
        boundary_seq = 0
        if cursor is not None and cursor != "":
            if not cursor.strip() or not _CURSOR_RE.fullmatch(cursor):
                raise HTTPException(status_code=422, detail="invalid cursor")
            decoded = _decode_rewrap_job_item_cursor(
                cursor, tenant_id, workload_id, job_id
            )
            if decoded is None:
                raise HTTPException(status_code=422, detail="invalid cursor")
            boundary_seq = decoded

        # --- read-only item scan ----------------------------------------
        try:
            with session_factory() as session:
                # The named job must exist in exactly this scope; an
                # unknown or cross-scope job is an indistinguishable 404.
                job = session.get(RewrapJob, job_id)
                if (
                    job is None
                    or job.tenant_id != tenant_id
                    or job.workload_id != workload_id
                ):
                    raise HTTPException(status_code=404, detail="rewrap job not found")

                stmt = (
                    select(RewrapJobItem)
                    .where(
                        RewrapJobItem.job_id == job_id,
                        RewrapJobItem.seq > boundary_seq,
                    )
                    .order_by(RewrapJobItem.seq.asc())
                    .limit(page_size + 1)
                )
                # One extra row is the "more follows" probe. The scan is a
                # single read-only statement: a storage failure aborts the
                # whole request with a 500 rather than returning a partial
                # page.
                rows = list(session.scalars(stmt))
        except HTTPException:
            raise
        except Exception:
            logger.error("rewrap job items query failed")
            raise HTTPException(
                status_code=500, detail="rewrap job items unavailable"
            )

        has_more = len(rows) > page_size
        page = rows[:page_size]

        items = [
            {
                # Stable, immutable per-job sequence in committed order.
                "seq": row.seq,
                "data_id": row.data_id,
                "old_key_version": row.old_key_version,
                "new_key_version": row.new_key_version,
                # A fixed service code: rewrapped or skipped only.
                "result": row.result,
                "occurred_at": _rfc3339(row.occurred_at),
            }
            for row in page
        ]

        if has_more:
            next_cursor = _encode_rewrap_job_item_cursor(
                tenant_id, workload_id, job_id, page[-1].seq
            )
            complete = False
        else:
            next_cursor = ""
            complete = True

        # Compact container (items, next_cursor, complete) with a single
        # terminating newline. seq and the key versions are ints, complete
        # is a bool and every other value is a string — no floats, -0.0 or
        # non-finite values can appear.
        body = (
            json.dumps(
                {
                    "items": items,
                    "next_cursor": next_cursor,
                    "complete": complete,
                },
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        return Response(content=body, media_type="application/json")

    @app.post("/v1/rewrap-jobs//cancel")
    def cancel_rewrap_job_identifier_required(
        body: CancelRewrapJobRequest,
    ) -> Response:
        # An empty path segment is a missing job identifier: a 422 client
        # error rather than a routing-level 404 or 405. It never reads a
        # job.
        raise HTTPException(status_code=422, detail="invalid job identifier")

    @app.post("/v1/rewrap-jobs/{job_id}/cancel")
    def cancel_rewrap_job(job_id: str, body: CancelRewrapJobRequest) -> Response:
        """Cancel a queued or running asynchronous rewrap job.

        The path identifier must be a canonical lowercase UUID and the
        body carries exactly the two non-blank scope strings; any
        missing, blank, wrong-typed or unknown field is a 422 that never
        reads the job. An unknown job or one outside the body's
        tenant/workload is an indistinguishable 404 (existence is never
        revealed). A succeeded, failed or already-cancelled job returns
        409 and changes neither status, counters nor any timestamp; a
        repeat cancel likewise returns 409 and rewrites nothing.

        The winning cancel commits the status flip to ``cancelled`` and
        ``cancelled_at`` in one guarded transaction (queued|running ->
        cancelled), so concurrent cancels settle as exactly one success
        with stable 409s, and a per-envelope commit racing the flip can
        never overwrite the marker (the runner discards an envelope that
        loses). A running runner stops after the current envelope: an
        envelope still uncommitted when the cancel arrives is rolled back
        wholesale (material, audit, progress), while an envelope that
        already committed keeps its change and audit and the job records
        cancellation at that stopping point. A write or commit failure is
        a 500 after a full rollback.
        """
        # A path identifier that is missing (empty segment), blank,
        # whitespace-padded, uppercase or otherwise non-canonical is a
        # field/format error checked before any state is read.
        if not _UUID_RE.fullmatch(job_id):
            raise HTTPException(status_code=422, detail="invalid job identifier")

        runner = app.state.rewrap_job_runner
        scope = (body.tenant_id, body.workload_id)

        # Raise the in-process stop signal *before* taking the writer
        # lock, but only when this process currently claims the job for
        # exactly the requesting scope. A cross-scope (always-404) request
        # therefore cannot interrupt another tenant's runner, and an
        # unknown id or a job claimed elsewhere signals nothing. The matched
        # worker rolls its current uncommitted envelope back and releases
        # the writer lock so the read below is not blocked; a job claimed
        # by another process is simply cancelled after its current envelope
        # commits (the row lock + cleared claim token stop the scan).
        runner.request_cancel(job_id, scope)

        # Read-only scope/status judgement: an unknown id or a job outside
        # the body's tenant/workload is an indistinguishable 404.
        try:
            with session_factory() as session:
                existing = session.get(RewrapJob, job_id)
                if (
                    existing is None
                    or existing.tenant_id != body.tenant_id
                    or existing.workload_id != body.workload_id
                ):
                    raise HTTPException(
                        status_code=404, detail="rewrap job not found"
                    )
                prior_status = existing.status
        except HTTPException:
            raise
        except Exception:
            logger.error("rewrap job cancel failed")
            raise HTTPException(status_code=500, detail="rewrap job cancel failed")

        if prior_status == REWRAP_JOB_STATUS_CANCELLED:
            # A repeated cancel is a conflict observed without a write: it
            # rewrites neither cancelled_at nor any other state and appends
            # no audit.
            raise HTTPException(status_code=409, detail="rewrap job already cancelled")
        if prior_status in (REWRAP_JOB_STATUS_SUCCEEDED, REWRAP_JOB_STATUS_FAILED):
            raise HTTPException(status_code=409, detail="rewrap job already settled")

        cancelled_at = _utcnow()
        try:
            with session_factory() as session:
                try:
                    # Re-read under the writer lock; the open-status
                    # judgement above is advisory and the guarded UPDATE
                    # below is the real settlement. SQLite ignores FOR
                    # UPDATE but already serializes all writers via BEGIN
                    # IMMEDIATE, so the cancel and the runner's
                    # per-envelope commit are strictly ordered.
                    job = session.scalar(
                        select(RewrapJob)
                        .where(
                            RewrapJob.job_id == job_id,
                            RewrapJob.tenant_id == body.tenant_id,
                            RewrapJob.workload_id == body.workload_id,
                        )
                        .with_for_update()
                    )
                    if job is None:
                        raise HTTPException(
                            status_code=404, detail="rewrap job not found"
                        )
                    if job.status == REWRAP_JOB_STATUS_CANCELLED:
                        raise HTTPException(
                            status_code=409, detail="rewrap job already cancelled"
                        )
                    if job.status in (
                        REWRAP_JOB_STATUS_SUCCEEDED,
                        REWRAP_JOB_STATUS_FAILED,
                    ):
                        raise HTTPException(
                            status_code=409, detail="rewrap job already settled"
                        )
                    # Atomic settlement: only a still-queued/running row
                    # flips, and cancelled_at is written by that same
                    # single-row update together with the status. The
                    # guarded predicate is what makes concurrent cancels
                    # settle with exactly one winner and prevents a racing
                    # runner settlement from being overwritten.
                    outcome = session.execute(
                        update(RewrapJob)
                        .where(
                            RewrapJob.job_id == job_id,
                            RewrapJob.status.in_(
                                (
                                    REWRAP_JOB_STATUS_QUEUED,
                                    REWRAP_JOB_STATUS_RUNNING,
                                )
                            ),
                        )
                        .values(
                            status=REWRAP_JOB_STATUS_CANCELLED,
                            cancelled_at=cancelled_at,
                            claim_token=None,
                            updated_at=cancelled_at,
                        )
                        .execution_options(synchronize_session=False)
                    )
                    if outcome.rowcount != 1:
                        session.rollback()
                        fresh = session.get(RewrapJob, job_id)
                        if fresh is not None:
                            if fresh.status == REWRAP_JOB_STATUS_CANCELLED:
                                raise HTTPException(
                                    status_code=409,
                                    detail="rewrap job already cancelled",
                                )
                            if fresh.status in (
                                REWRAP_JOB_STATUS_SUCCEEDED,
                                REWRAP_JOB_STATUS_FAILED,
                            ):
                                raise HTTPException(
                                    status_code=409,
                                    detail="rewrap job already settled",
                                )
                        raise HTTPException(
                            status_code=500, detail="rewrap job cancel failed"
                        )
                    # The cancellation event records the actual old status
                    # observed under the writer lock (queued or running)
                    # and commits in the same transaction as the guarded
                    # status flip and cancelled_at. The guarded update used
                    # synchronize_session=False, so job.status still holds
                    # the pre-migration value.
                    _record_rewrap_job_event(
                        session,
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        job_id=job_id,
                        old_status=job.status,
                        new_status=REWRAP_JOB_STATUS_CANCELLED,
                        reason=REWRAP_JOB_EVENT_REASON_CANCELLED,
                        now=cancelled_at,
                    )
                    try:
                        session.commit()
                    except Exception:
                        session.rollback()
                        logger.error("rewrap job cancel failed")
                        raise HTTPException(
                            status_code=500, detail="rewrap job cancel failed"
                        )
                except HTTPException:
                    raise
                except Exception:
                    # A failure during the locked lookup or guarded update
                    # is a full rollback and a sanitized 500: the status
                    # change and cancelled_at can never have happened on
                    # this path, so the job keeps its prior state and time.
                    session.rollback()
                    logger.error("rewrap job cancel failed")
                    raise HTTPException(
                        status_code=500, detail="rewrap job cancel failed"
                    )
        finally:
            runner.finish_cancel(job_id)

        # Compact JSON describing only the cancellation result. The keys
        # are emitted in lexicographic (string) order — cancelled_at,
        # job_id, status — every value is a string (allow_nan=False
        # excludes non-finite numbers), terminated by exactly one newline.
        body_bytes = json.dumps(
            {
                "cancelled_at": _rfc3339(cancelled_at),
                "job_id": job_id,
                "status": REWRAP_JOB_STATUS_CANCELLED,
            },
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        ).encode("utf-8")
        return Response(
            content=body_bytes + b"\n",
            status_code=200,
            media_type="application/json",
        )

    @app.post("/v1/rewrap-jobs//resume")
    def resume_rewrap_job_identifier_required(
        body: ResumeRewrapJobRequest,
    ) -> Response:
        # An empty path segment is a missing job identifier: a 422 client
        # error rather than a routing-level 404 or 405. It never reads a
        # job.
        raise HTTPException(status_code=422, detail="invalid job identifier")

    @app.post("/v1/rewrap-jobs/{job_id}/resume")
    def resume_rewrap_job(job_id: str, body: ResumeRewrapJobRequest) -> Response:
        """Resume a failed asynchronous rewrap job from its saved progress.

        The path identifier must be a canonical lowercase UUID and the
        body carries exactly the two non-blank scope strings; any
        missing, blank, wrong-typed or unknown field is a 422 that never
        reads the job. An unknown job or one outside the body's
        tenant/workload is an indistinguishable 404 (existence is never
        revealed). A job in any non-failed state — queued, running,
        succeeded, cancelled, or already flipped back to queued by a
        concurrent resume — returns 409 and changes nothing.

        The winning resume commits exactly one guarded transition
        (failed -> queued) in a single transaction: no second job is
        created, no history row is rewritten, and the failed counter and
        the resume cursor parked immediately before the failing envelope
        are preserved verbatim, so the background runner re-attempts
        exactly that envelope first and converges to the current
        progress. The resume itself appends no audit event; envelopes
        and audits committed before the failure are untouched. A write
        or commit failure is a 500 after a full rollback, leaving the
        status, counters, cursor and timestamps exactly as they were.
        """
        # A path identifier that is missing (empty segment), blank,
        # whitespace-padded, uppercase or otherwise non-canonical is a
        # field/format error checked before any state is read.
        if not _UUID_RE.fullmatch(job_id):
            raise HTTPException(status_code=422, detail="invalid job identifier")

        # Read-only scope/status judgement: an unknown id or a job outside
        # the body's tenant/workload is an indistinguishable 404, and any
        # non-failed state is a 409 observed without a write.
        try:
            with session_factory() as session:
                existing = session.get(RewrapJob, job_id)
                if (
                    existing is None
                    or existing.tenant_id != body.tenant_id
                    or existing.workload_id != body.workload_id
                ):
                    raise HTTPException(
                        status_code=404, detail="rewrap job not found"
                    )
                if existing.status != REWRAP_JOB_STATUS_FAILED:
                    raise HTTPException(
                        status_code=409, detail="rewrap job is not failed"
                    )
        except HTTPException:
            raise
        except Exception:
            logger.error("rewrap job resume failed")
            raise HTTPException(status_code=500, detail="rewrap job resume failed")

        resumed_at = _utcnow()
        try:
            with session_factory() as session:
                try:
                    # Re-read under the writer lock; the open-status
                    # judgement above is advisory and the guarded UPDATE
                    # below is the real settlement. SQLite ignores FOR
                    # UPDATE but already serializes all writers via BEGIN
                    # IMMEDIATE, so concurrent resumes and a racing
                    # startup-sweep claim are strictly ordered.
                    job = session.scalar(
                        select(RewrapJob)
                        .where(
                            RewrapJob.job_id == job_id,
                            RewrapJob.tenant_id == body.tenant_id,
                            RewrapJob.workload_id == body.workload_id,
                        )
                        .with_for_update()
                    )
                    if job is None:
                        raise HTTPException(
                            status_code=404, detail="rewrap job not found"
                        )
                    if job.status != REWRAP_JOB_STATUS_FAILED:
                        raise HTTPException(
                            status_code=409, detail="rewrap job is not failed"
                        )
                    # Atomic settlement: only a still-failed row flips back
                    # to queued, so concurrent resumes settle with exactly
                    # one winner and the rest are stable 409s. Counters
                    # (including failed), the parked resume cursor and
                    # created_at are not part of the update and survive
                    # verbatim; the claim token stays NULL so the next
                    # runner (live pool or startup sweep) claims the job
                    # under the same status guard as any queued row.
                    outcome = session.execute(
                        update(RewrapJob)
                        .where(
                            RewrapJob.job_id == job_id,
                            RewrapJob.status == REWRAP_JOB_STATUS_FAILED,
                        )
                        .values(
                            status=REWRAP_JOB_STATUS_QUEUED,
                            claim_token=None,
                            updated_at=resumed_at,
                        )
                        .execution_options(synchronize_session=False)
                    )
                    if outcome.rowcount != 1:
                        session.rollback()
                        fresh = session.get(RewrapJob, job_id)
                        if fresh is not None:
                            if fresh.status != REWRAP_JOB_STATUS_FAILED:
                                raise HTTPException(
                                    status_code=409,
                                    detail="rewrap job is not failed",
                                )
                        raise HTTPException(
                            status_code=500, detail="rewrap job resume failed"
                        )
                    # Snapshot the response fields from the settled row
                    # before committing: progress, counters and the parked
                    # cursor are exactly what the failure left; only
                    # status and updated_at move.
                    payload = {
                        "job_id": job.job_id,
                        "status": REWRAP_JOB_STATUS_QUEUED,
                        "processed": job.processed,
                        "rewrapped": job.rewrapped,
                        "skipped": job.skipped,
                        "failed": job.failed,
                        "next_cursor": job.next_cursor,
                        "complete": bool(job.complete),
                        "created_at": _rfc3339(job.created_at),
                        "updated_at": _rfc3339(resumed_at),
                    }
                    # The failed -> queued recovery is a committed status
                    # migration and records exactly one recovered event in
                    # the same transaction as the guarded flip. (The
                    # post-restart sweep's failed -> running retry records
                    # the same reason.) The guarded update used
                    # synchronize_session=False, so job.status is still the
                    # pre-migration value.
                    _record_rewrap_job_event(
                        session,
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        job_id=job_id,
                        old_status=REWRAP_JOB_STATUS_FAILED,
                        new_status=REWRAP_JOB_STATUS_QUEUED,
                        reason=REWRAP_JOB_EVENT_REASON_RECOVERED,
                        now=resumed_at,
                    )
                    try:
                        session.commit()
                    except Exception:
                        session.rollback()
                        logger.error("rewrap job resume failed")
                        raise HTTPException(
                            status_code=500, detail="rewrap job resume failed"
                        )
                except HTTPException:
                    raise
                except Exception:
                    # A failure during the locked lookup or guarded update
                    # is a full rollback and a sanitized 500: the status
                    # change can never have happened on this path, so the
                    # job keeps its failed state, counters, cursor and
                    # timestamps.
                    session.rollback()
                    logger.error("rewrap job resume failed")
                    raise HTTPException(
                        status_code=500, detail="rewrap job resume failed"
                    )
        except HTTPException:
            raise
        except Exception:
            logger.error("rewrap job resume failed")
            raise HTTPException(status_code=500, detail="rewrap job resume failed")

        # Durable requeue first, background advancement second — exactly
        # as on submission. The row is queued with its saved cursor, so a
        # live pool claims it under the same status guard, and if this
        # process dies first the next process's startup sweep resumes it.
        app.state.rewrap_job_runner.submit(job_id)

        # Compact JSON in the single-job query's field order and types,
        # terminated by exactly one newline. Counters are ints, complete
        # is a bool and every other value is a string (allow_nan=False
        # excludes non-finite numbers).
        return _compact_json_line(payload)

    return app


app = create_app()
