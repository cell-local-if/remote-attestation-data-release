from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictInt, StrictStr, field_validator
from sqlalchemy import create_engine, event, select, text, update
from sqlalchemy.orm import sessionmaker

from proof_release.db import Base, Challenge, Evidence, VerificationResultCode
from proof_release.verifiers import (
    ChallengeContext,
    VerificationContext,
    VerificationResult,
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


def _sqlite_version(conn) -> tuple[int, ...]:
    raw = conn.execute(text("SELECT sqlite_version()")).scalar() or ""
    return tuple(int(part) for part in str(raw).split(".") if part.isdigit())


def _migrate_additive(engine) -> None:
    """Apply forward-only, additive upgrades to pre-existing databases.

    Older releases stored a free-text ``verification_detail`` note supplied
    by the verifier plugin. That column could carry raw evidence or private
    context, so it is retired in favour of ``verification_result_code``: a
    finite, service-defined code derived only from the pass/fail verdict.

    The upgrade is deliberately forward-compatible and never destructive to
    settled conclusions:

    * the new nullable column is added when missing (databases created from
      current metadata already have it);
    * settled rows are backfilled from their ``status`` — verified rows get
      ``accepted``, rejected rows get ``rejected``;
    * any legacy free-text note is scrubbed (and the obsolete column dropped
      when the SQLite version supports it). An old database therefore opens
      cleanly and historical notes can neither fail verification nor leak.
    """
    if engine.dialect.name != "sqlite":
        # Non-sqlite deployments are created from metadata; nothing to add.
        return
    additions = {
        "evidence": (
            ("verified_at", "DATETIME"),
            ("verification_result_code", "VARCHAR(32)"),
        ),
    }
    scrubbed_legacy_text = False
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
            # Re-read the column set in case columns were just added.
            existing = {
                row[1]
                for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            }
            if table == "evidence" and "verification_result_code" in existing:
                # Backfill the controlled code solely from the already
                # settled status; never from the old free-text note.
                conn.execute(
                    text(
                        "UPDATE evidence SET verification_result_code = CASE status "
                        "WHEN 'verified' THEN :accepted "
                        "WHEN 'rejected' THEN :rejected "
                        "ELSE NULL END "
                        "WHERE verification_result_code IS NULL "
                        "AND status IN ('verified', 'rejected')"
                    ).bindparams(
                        accepted=VerificationResultCode.ACCEPTED.value,
                        rejected=VerificationResultCode.REJECTED.value,
                    )
                )
            # Scrub the retired free-text column so nothing a legacy plugin
            # wrote (potentially raw evidence or secrets) remains on disk.
            if table == "evidence" and "verification_detail" in existing:
                conn.execute(text("UPDATE evidence SET verification_detail = NULL"))
                if _sqlite_version(conn) >= (3, 35, 0):
                    # DROP COLUMN is available from SQLite 3.35; on older
                    # versions the column stays behind but is always NULL and
                    # is not mapped by the ORM, so it cannot leak.
                    conn.execute(
                        text("ALTER TABLE evidence DROP COLUMN verification_detail")
                    )
                scrubbed_legacy_text = True

    if scrubbed_legacy_text:
        # Updating/dropping can leave the old note bytes behind in freed
        # pages; VACUUM once to physically erase them. Run it on a raw pooled
        # connection so it executes outside SQLAlchemy's BEGIN IMMEDIATE
        # handling (VACUUM cannot run inside a transaction).
        raw_connection = engine.pool.connect()
        try:
            cursor = raw_connection.cursor()
            cursor.execute("VACUUM")
            cursor.close()
            raw_connection.commit()
        finally:
            raw_connection.close()


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
            verification_context = VerificationContext(
                evidence=body.evidence,
                tenant_id=body.tenant_id,
                workload_id=body.workload_id,
                challenge=challenge_context,
            )
            # The raw evidence exists only on the stack for this call; it is
            # never logged, persisted, or placed on the response. On plugin
            # failure the transaction is rolled back, leaving no
            # half-finished state, and only non-sensitive identifiers are
            # logged.
            #
            # The plugin is trusted with exactly one piece of output: the
            # ``accepted`` boolean. Anything else it might return (a detail
            # string, reason, exception text, extra attributes — any of which
            # could embed the raw evidence, a key or other private context)
            # is never read, truncated, transformed, logged, persisted or
            # returned. A result that is not a VerificationResult carrying a
            # real boolean is treated as a plugin fault, not a verdict.
            try:
                result = verifier.verify(verification_context)
                if not isinstance(result, VerificationResult):
                    # Force the shared fault path below; never render result.
                    raise TypeError("verifier returned a non-result object")
                accepted = result.accepted
                if not isinstance(accepted, bool):
                    raise TypeError("verifier accepted flag is not boolean")
            except Exception as exc:
                # Log only non-sensitive identifiers and the exception type —
                # never the traceback/message, since a faulty plugin could
                # embed raw evidence or private context in it. The session is
                # rolled back as the context exits, so the record stays
                # received and verification can be retried (e.g. after the
                # plugin is replaced).
                logger.error(
                    "verifier %s for format %r failed on evidence %s: %s",
                    type(verifier).__name__,
                    evidence.evidence_format,
                    evidence.evidence_id,
                    type(exc).__name__,
                )
                raise HTTPException(status_code=500, detail="verification failed")

            # The persisted code comes solely from the controlled boolean and
            # belongs to the service's finite result-code set.
            new_status = "verified" if accepted else "rejected"
            result_code = VerificationResultCode.from_accepted(accepted)
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
                    verification_result_code=result_code.value,
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

    return app


app = create_app()
