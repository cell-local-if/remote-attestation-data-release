from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, StrictInt, StrictStr, field_validator
from sqlalchemy import create_engine, event, func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding

from proof_release.db import (
    DECISION_STATUS_ALLOWED,
    DECISION_STATUS_DENIED,
    REWRAP_RESULT_KEYRING,
    REWRAP_RESULT_MISSING_KEY,
    REWRAP_RESULT_REWRAP,
    REWRAP_RESULT_REWRAPPED,
    REWRAP_RESULT_SKIPPED,
    VERIFICATION_RESULT_ACCEPTED,
    VERIFICATION_RESULT_REJECTED,
    Base,
    Challenge,
    DataEnvelope,
    Decision,
    Evidence,
    Policy,
    ReleaseGrant,
    RewrapBatch,
    RewrapBatchAudit,
    RewrapCursor,
    TrustRoot,
)
from proof_release.envelopes import (
    MasterKeyError,
    b64url_encode,
    encrypt_payload,
    load_keyring,
    rewrap_data_key,
)
from proof_release.policies import (
    InvalidRule,
    canonical_rule_json,
    evaluate_rule,
    validate_rule,
)
from proof_release.verifiers import (
    ATTESTED_NONCE_JSON,
    X509_ATTESTED_NONCE_JSON,
    ChallengeContext,
    VerificationContext,
    VerifierRegistry,
    default_registry,
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
MIN_REWRAP_BATCH_LIMIT = 1
MAX_REWRAP_BATCH_LIMIT = 200
DEFAULT_REWRAP_BATCH_LIMIT = 50
#: Opaque cursor tokens carry this many CSPRNG bytes.
CURSOR_BYTES = 24

_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _rfc3339(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


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


def _batch_rewrap_one(
    session_factory,
    *,
    batch_id: str,
    tenant_id: str,
    workload_id: str,
    data_id: str,
    seq: int,
    target_version: int,
) -> tuple[str, int, int]:
    """Process exactly one envelope of a batch and commit one audit row.

    The envelope update (if any) and its audit commit together in a single
    transaction independent of every other envelope, so a failure on one
    envelope leaves every previously handled envelope and audit durable.

    ``target_version`` is the keyring current version established by the
    batch preflight; the audit's new-key version is always that integer
    even if the keyring becomes unavailable while the page is processed.

    Returns ``(result_code, old_key_version, new_key_version)`` where the
    result is one of ``rewrapped``, ``skipped``, ``keyring``,
    ``missing-key`` or ``rewrap``. The plaintext data key exists only in a
    local variable and is never audited or returned.
    """
    occurred_at = _utcnow()
    with session_factory() as session:
        # On SQLite this write transaction already holds BEGIN IMMEDIATE;
        # on locking backends the guarded UPDATE serializes concurrent
        # rewraps of the same row.
        envelope = session.get(DataEnvelope, (tenant_id, workload_id, data_id))
        if envelope is None:
            # The snapshot was taken in this same request, so a missing row
            # here means concurrent deletion through a non-API path; treat
            # it as a service failure rather than silently skipping.
            raise HTTPException(status_code=500, detail="rewrap batch failed")

        result: str
        old_version = envelope.key_version
        new_version: int

        # Reload the keyring for every envelope: the configuration may be
        # rotated (or removed) while a long page is being processed.
        try:
            keyring = load_keyring()
        except MasterKeyError as exc:
            logger.error("master key configuration unavailable: %s", exc)
            result = REWRAP_RESULT_KEYRING
            # The target version was validated before processing started.
            new_version = target_version
        else:
            new_version = keyring.current_version
            if old_version == new_version:
                # Already wrapped under the current version — including an
                # envelope a concurrent batch or request advanced moments
                # ago. No material changes; this batch simply records it.
                result = REWRAP_RESULT_SKIPPED
            else:
                try:
                    unwrapping_key = keyring.key_for(old_version)
                except MasterKeyError:
                    logger.error(
                        "master key version %s unavailable for batch rewrap",
                        old_version,
                    )
                    result = REWRAP_RESULT_MISSING_KEY
                else:
                    try:
                        new_wrapped_key = rewrap_data_key(
                            unwrapping_key,
                            keyring.current_key(),
                            envelope.wrapped_key,
                        )
                    except Exception:
                        logger.error(
                            "batch envelope %s rewrap failed", data_id
                        )
                        result = REWRAP_RESULT_REWRAP
                    else:
                        # Guarded update: advance only if the row still holds
                        # the version we unwrapped. A concurrent winner that
                        # settled first leaves current-version material,
                        # which this request records as an already-current
                        # result — each envelope advances at most once.
                        outcome = session.execute(
                            update(DataEnvelope)
                            .where(
                                DataEnvelope.tenant_id == tenant_id,
                                DataEnvelope.workload_id == workload_id,
                                DataEnvelope.data_id == data_id,
                                DataEnvelope.key_version == old_version,
                            )
                            .values(
                                key_version=new_version,
                                wrapped_key=new_wrapped_key,
                            )
                            .execution_options(synchronize_session=False)
                        )
                        if outcome.rowcount != 1:
                            session.rollback()
                            fresh = session.get(
                                DataEnvelope, (tenant_id, workload_id, data_id)
                            )
                            if fresh is None:
                                raise HTTPException(
                                    status_code=500, detail="rewrap batch failed"
                                )
                            # Lost the race; the envelope is now settled at
                            # another version. Re-enter this helper's
                            # bookkeeping by reporting the stored state.
                            old_version = fresh.key_version
                            new_version = fresh.key_version
                            result = REWRAP_RESULT_SKIPPED
                        else:
                            result = REWRAP_RESULT_REWRAPPED

        session.add(
            RewrapBatchAudit(
                audit_id=str(uuid.uuid4()),
                batch_id=batch_id,
                tenant_id=tenant_id,
                workload_id=workload_id,
                data_id=data_id,
                seq=seq,
                old_key_version=old_version,
                new_key_version=new_version,
                result=result,
                occurred_at=occurred_at,
            )
        )
        try:
            session.commit()
        except Exception:
            # A write failure after the envelope changed must not strand a
            # rotated row without its audit; the rollback undoes both, and
            # the failure surfaces as a service error. Retrying the page
            # (same cursor) reprocesses the envelope exactly once.
            session.rollback()
            logger.error("batch envelope %s audit write failed", data_id)
            raise HTTPException(status_code=500, detail="rewrap batch failed")

    return result, old_version, new_version


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
    # A JSON integer in [1, 200]; booleans, floats and numeric strings are
    # rejected by StrictInt. Absent means 50.
    limit: StrictInt = Field(default=DEFAULT_REWRAP_BATCH_LIMIT, ge=1, le=200)
    # Opaque token from a previous batch for the same scope; None starts at
    # the smallest data_id. Its existence and scope are checked in the
    # handler so forgery/cross-scope use is a 422 like any bad field.
    cursor: StrictStr | None = Field(default=None, min_length=1)

    _non_blank = field_validator("tenant_id", "workload_id")(_require_non_blank)
    _cursor_non_blank = field_validator("cursor")(_optional_non_blank)


class RewrapBatchAuditEntry(BaseModel):
    tenant_id: str
    workload_id: str
    data_id: str
    old_key_version: int
    new_key_version: int
    result: str
    occurred_at: str


class RewrapBatchResponse(BaseModel):
    batch_id: str
    processed: int
    rewrapped: int
    skipped: int
    failed: int
    next_cursor: str
    done: bool


class RewrapBatchDetailResponse(BaseModel):
    batch_id: str
    tenant_id: str
    workload_id: str
    request_cursor: str | None
    limit: int
    processed: int
    rewrapped: int
    skipped: int
    failed: int
    next_cursor: str
    done: bool
    created_at: str
    completed_at: str | None
    audits: list[RewrapBatchAuditEntry]


def _migrate_additive(engine) -> None:
    """Apply forward-only additive column additions to pre-existing databases."""
    if engine.dialect.name != "sqlite":
        # Non-sqlite deployments are created from metadata; nothing to add.
        return
    additions = {
        "evidence": (
            ("verified_at", "DATETIME"),
            ("verification_result", "VARCHAR(16)"),
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
            # Legacy databases may carry a free-form verification_detail
            # column written by older versions, which can hold arbitrary
            # plugin-supplied text (potentially raw evidence or secrets).
            # It is no longer part of the model; scrub any leftover values
            # so they can never be read back, returned, or logged.
            if "verification_detail" in existing:
                conn.execute(
                    text(f"UPDATE {table} SET verification_detail = NULL")
                )


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
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    registry = verifier_registry or default_registry

    app = FastAPI(title="Remote Attestation Data Release")
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.verifier_registry = registry

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/v1/challenges", status_code=201, response_model=ChallengeCreatedResponse)
    def create_challenge(body: CreateChallengeRequest) -> ChallengeCreatedResponse:
        nonce_bytes = secrets.token_bytes(NONCE_BYTES)
        nonce = base64.urlsafe_b64encode(nonce_bytes).rstrip(b"=").decode("ascii")
        now = _utcnow()
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
        with session_factory() as session:
            session.add(challenge)
            session.commit()
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
            except Exception as exc:
                # Log only non-sensitive identifiers and the exception type —
                # never the traceback/message, since a faulty plugin could
                # embed raw evidence or private context in it.
                logger.error(
                    "verifier %s for format %r failed on evidence %s: %s",
                    type(verifier).__name__,
                    evidence.evidence_format,
                    evidence.evidence_id,
                    type(exc).__name__,
                )
                raise HTTPException(status_code=500, detail="verification failed")
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
            session.add(
                TrustRoot(
                    root_id=root_id,
                    tenant_id=body.tenant_id,
                    workload_id=body.workload_id,
                    name=body.name,
                    root_pem=pem,
                    cert_sha256=cert_digest,
                    created_at=now,
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

    @app.post("/v1/policies", status_code=201, response_model=PolicyCreatedResponse)
    def create_policy(body: CreatePolicyRequest) -> PolicyCreatedResponse:
        policy_id = str(uuid.uuid4())
        now = _utcnow()
        rule_json = canonical_rule_json(body.rule)
        # Allocate the next version inside a write transaction. On SQLite
        # every write transaction begins as BEGIN IMMEDIATE, so competing
        # creators serialize; on locking backends the unique
        # (scope, name, version) constraint plus this retry loop guarantees
        # no two versions ever share a number and no version is skipped.
        with session_factory() as session:
            for _ in range(10):
                highest = session.scalar(
                    select(func.max(Policy.version)).where(
                        Policy.tenant_id == body.tenant_id,
                        Policy.workload_id == body.workload_id,
                        Policy.name == body.name,
                    )
                )
                version = (highest or 0) + 1
                session.add(
                    Policy(
                        policy_id=policy_id,
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        name=body.name,
                        version=version,
                        rule_json=rule_json,
                        created_at=now,
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
                raise HTTPException(status_code=500, detail="could not allocate version")
        return PolicyCreatedResponse(
            policy_id=policy_id,
            tenant_id=body.tenant_id,
            workload_id=body.workload_id,
            name=body.name,
            version=version,
            rule=body.rule,
            created_at=_rfc3339(now),
        )

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
                select(Policy).where(Policy.policy_id == body.policy_id)
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
            decision_id = str(uuid.uuid4())
            decided_at = _utcnow()
            decision = Decision(
                decision_id=decision_id,
                evidence_id=evidence_id,
                policy_id=policy.policy_id,
                policy_version=policy.version,
                status=status,
                decided_at=decided_at,
            )
            session.add(decision)
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
                grant_id=str(uuid.uuid4()),
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

    @app.post(
        "/v1/release-grants/{grant_id}/consume",
        response_model=ReleaseGrantConsumedResponse,
    )
    def consume_release_grant(
        grant_id: str, body: ConsumeReleaseGrantRequest
    ) -> ReleaseGrantConsumedResponse:
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
            if not hmac.compare_digest(grant.capability_digest, digest):
                raise HTTPException(status_code=401, detail="invalid capability")
            if grant.status == "consumed":
                raise HTTPException(status_code=409, detail="grant already consumed")
            if grant.expires_at <= now:
                raise HTTPException(status_code=410, detail="grant expired")
            # Atomic claim: only one concurrent consumer can flip
            # pending -> consumed for an unexpired grant. BEGIN IMMEDIATE
            # (SQLite) / row locks (other backends) plus the guarded UPDATE
            # guarantee exactly one winner across processes and restarts.
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
                raise HTTPException(status_code=410, detail="grant expired")
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

    @app.post(
        "/v1/data-envelopes",
        status_code=201,
        response_model=DataEnvelopeCreatedResponse,
    )
    def create_data_envelope(body: CreateDataEnvelopeRequest) -> DataEnvelopeCreatedResponse:
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

    @app.post("/v1/rewrap-batches", response_model=RewrapBatchResponse)
    def create_rewrap_batch(body: CreateRewrapBatchRequest) -> RewrapBatchResponse:
        # Preflight the keyring before anything is persisted. An invalid
        # overall configuration is a server failure: no batch row, no cursor
        # and no audit or envelope change is produced. Only the failure kind
        # is logged, never configured values or key material.
        try:
            keyring = load_keyring()
        except MasterKeyError as exc:
            logger.error("master key configuration unavailable: %s", exc)
            raise HTTPException(status_code=500, detail="encryption unavailable")

        batch_id = str(uuid.uuid4())
        created_at = _utcnow()

        with session_factory() as session:
            # Validate the presented cursor inside the write transaction.
            # A forged token (no row) and one belonging to another scope are
            # both malformed-request failures (422); the token's mere
            # existence is never revealed. The boundary is the exclusive
            # data_id position in this scope's stable ordering; NULL means
            # the start position (before the smallest data_id).
            boundary: str | None = None
            if body.cursor is not None:
                cursor_row = session.get(RewrapCursor, body.cursor)
                if (
                    cursor_row is None
                    or cursor_row.tenant_id != body.tenant_id
                    or cursor_row.workload_id != body.workload_id
                ):
                    raise HTTPException(status_code=422, detail="invalid cursor")
                boundary = cursor_row.boundary_data_id

            # Reserve the batch row before touching any envelope, so the
            # batch and its audits remain queryable if the page stops early
            # or the process restarts mid-page.
            batch = RewrapBatch(
                batch_id=batch_id,
                tenant_id=body.tenant_id,
                workload_id=body.workload_id,
                request_cursor=body.cursor,
                limit_value=body.limit,
                processed_count=0,
                rewrapped_count=0,
                skipped_count=0,
                failed_count=0,
                next_cursor=None,
                done=False,
                created_at=created_at,
                completed_at=None,
            )
            session.add(batch)
            session.commit()

        # Snapshot the page in stable data_id order. Fetching limit + 1 ids
        # reveals whether another envelope exists past the page without a
        # separate count query. The read transaction closes (releasing the
        # SQLite write lock) before per-envelope write transactions begin.
        with session_factory() as session:
            page_stmt = (
                select(DataEnvelope)
                .where(
                    DataEnvelope.tenant_id == body.tenant_id,
                    DataEnvelope.workload_id == body.workload_id,
                )
                .order_by(DataEnvelope.data_id.asc())
                .limit(body.limit + 1)
            )
            if boundary is not None:
                page_stmt = page_stmt.where(DataEnvelope.data_id > boundary)
            selected = list(session.scalars(page_stmt))
        has_more = len(selected) > body.limit
        page = selected[: body.limit]

        rewrapped = 0
        skipped = 0
        failure_code: str | None = None
        handled_ids: list[str] = []
        for seq, item in enumerate(page):
            result, _old_version, _new_version = _batch_rewrap_one(
                session_factory,
                batch_id=batch_id,
                tenant_id=body.tenant_id,
                workload_id=body.workload_id,
                data_id=item.data_id,
                seq=seq,
                target_version=keyring.current_version,
            )
            if result == REWRAP_RESULT_REWRAPPED:
                rewrapped += 1
                handled_ids.append(item.data_id)
            elif result == REWRAP_RESULT_SKIPPED:
                skipped += 1
                handled_ids.append(item.data_id)
            else:
                # keyring / missing-key / rewrap: the failing envelope is
                # unchanged; stop the page here with its audit already
                # independently committed.
                failure_code = result
                break

        processed = rewrapped + skipped
        completed_at = _utcnow()

        with session_factory() as session:
            next_token: str | None = None
            done = False
            if failure_code is None and not has_more:
                # Every envelope of the range has been handled: the end
                # position is reported as an empty cursor with done=true.
                done = True
                next_boundary = None
            else:
                # More work remains, or the page stopped on a failure. The
                # continuation cursor sits on the last handled envelope
                # (exclusive), i.e. immediately before an unhandled/failed
                # one; a start-position token (NULL boundary) is issued when
                # the first envelope of the range failed.
                done = False
                next_boundary = handled_ids[-1] if handled_ids else boundary
                next_token = b64url_encode(secrets.token_bytes(CURSOR_BYTES))
                session.add(
                    RewrapCursor(
                        cursor_id=next_token,
                        tenant_id=body.tenant_id,
                        workload_id=body.workload_id,
                        boundary_data_id=next_boundary,
                        created_at=completed_at,
                    )
                )

            session.execute(
                update(RewrapBatch)
                .where(RewrapBatch.batch_id == batch_id)
                .values(
                    processed_count=processed,
                    rewrapped_count=rewrapped,
                    skipped_count=skipped,
                    failed_count=0 if failure_code is None else 1,
                    next_cursor=next_token,
                    done=done,
                    completed_at=completed_at,
                )
                .execution_options(synchronize_session=False)
            )
            session.commit()

        return RewrapBatchResponse(
            batch_id=batch_id,
            processed=processed,
            rewrapped=rewrapped,
            skipped=skipped,
            failed=0 if failure_code is None else 1,
            next_cursor="" if next_token is None else next_token,
            done=done,
        )

    @app.get("/v1/rewrap-batches/{batch_id}", response_model=RewrapBatchDetailResponse)
    def get_rewrap_batch(
        batch_id: str,
        tenant_id: str = Query(..., min_length=1),
        workload_id: str = Query(..., min_length=1),
    ) -> RewrapBatchDetailResponse:
        # Identifier/scope parameter shape is validated before any lookup:
        # missing, blank or malformed values are 422.
        if not batch_id.strip() or not tenant_id.strip() or not workload_id.strip():
            raise HTTPException(status_code=422, detail="invalid batch parameters")
        try:
            uuid.UUID(batch_id)
        except (ValueError, AttributeError, TypeError):
            raise HTTPException(status_code=422, detail="invalid batch identifier")

        with session_factory() as session:
            # Unknown and cross-scope batches are indistinguishable: 404.
            batch = session.get(RewrapBatch, batch_id)
            if (
                batch is None
                or batch.tenant_id != tenant_id
                or batch.workload_id != workload_id
            ):
                raise HTTPException(status_code=404, detail="rewrap batch not found")
            audits = list(
                session.scalars(
                    select(RewrapBatchAudit)
                    .where(RewrapBatchAudit.batch_id == batch_id)
                    .order_by(
                        RewrapBatchAudit.seq.asc(), RewrapBatchAudit.audit_id.asc()
                    )
                )
            )
            audit_entries = [
                RewrapBatchAuditEntry(
                    tenant_id=audit.tenant_id,
                    workload_id=audit.workload_id,
                    data_id=audit.data_id,
                    old_key_version=audit.old_key_version,
                    new_key_version=audit.new_key_version,
                    result=audit.result,
                    occurred_at=_rfc3339(audit.occurred_at),
                )
                for audit in audits
            ]
            return RewrapBatchDetailResponse(
                batch_id=batch.batch_id,
                tenant_id=batch.tenant_id,
                workload_id=batch.workload_id,
                request_cursor=batch.request_cursor,
                limit=batch.limit_value,
                processed=batch.processed_count,
                rewrapped=batch.rewrapped_count,
                skipped=batch.skipped_count,
                failed=batch.failed_count,
                next_cursor=batch.next_cursor or "",
                done=batch.done,
                created_at=_rfc3339(batch.created_at),
                completed_at=(
                    _rfc3339(batch.completed_at)
                    if batch.completed_at is not None
                    else None
                ),
                audits=audit_entries,
            )

    return app


app = create_app()
