from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictInt, StrictStr, field_validator
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker

from proof_release.db import Base, Challenge, Evidence
from proof_release.verifiers import VerificationContext, get_verifier

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


def create_app(database_url: str | None = None) -> FastAPI:
    url = database_url or os.environ.get(DATABASE_URL_ENV, DEFAULT_DATABASE_URL)
    connect_args = (
        {"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {}
    )
    engine = create_engine(url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    app = FastAPI(title="Remote Attestation Data Release")
    app.state.engine = engine
    app.state.session_factory = session_factory

    # Per-evidence locks serializing the first (state-changing) verification.
    verify_locks: dict[str, threading.Lock] = {}
    verify_locks_guard = threading.Lock()

    def _verify_lock(evidence_id: str) -> threading.Lock:
        with verify_locks_guard:
            return verify_locks.setdefault(evidence_id, threading.Lock())

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
        nonce_digest = _nonce_digest(body.nonce)
        evidence_digest = hashlib.sha256(body.evidence.encode("utf-8")).hexdigest()
        with session_factory() as session:
            record = session.get(Evidence, evidence_id)
            if (
                record is None
                or record.tenant_id != body.tenant_id
                or record.workload_id != body.workload_id
            ):
                raise HTTPException(status_code=404, detail="evidence not found")
            challenge = session.get(Challenge, record.challenge_id)
            if challenge is None:
                raise HTTPException(status_code=404, detail="evidence not found")
            if not hmac.compare_digest(challenge.nonce_digest, nonce_digest):
                raise HTTPException(status_code=401, detail="invalid nonce")
            if not hmac.compare_digest(record.evidence_sha256, evidence_digest):
                raise HTTPException(
                    status_code=422, detail="evidence does not match its record"
                )
            verifier = get_verifier(record.evidence_format)
            if verifier is None:
                raise HTTPException(
                    status_code=422, detail="unsupported evidence format"
                )
            context = VerificationContext(
                evidence_id=record.evidence_id,
                challenge_id=record.challenge_id,
                tenant_id=record.tenant_id,
                workload_id=record.workload_id,
                evidence_format=record.evidence_format,
                challenge_issued_at=challenge.issued_at,
                challenge_expires_at=challenge.expires_at,
            )
            if record.status in ("verified", "rejected"):
                # Verification is decided exactly once; replays return the
                # persisted verdict without invoking the plugin again.
                return EvidenceVerifiedResponse(
                    evidence_id=record.evidence_id,
                    challenge_id=record.challenge_id,
                    status=record.status,
                    verified_at=_rfc3339(record.verified_at),
                )
        # Serialize first-time verification per evidence so concurrent callers
        # cannot produce contradictory outcomes or invoke the plugin twice.
        with _verify_lock(evidence_id):
            with session_factory() as session:
                record = session.get(Evidence, evidence_id)
                if record.status in ("verified", "rejected"):
                    return EvidenceVerifiedResponse(
                        evidence_id=record.evidence_id,
                        challenge_id=record.challenge_id,
                        status=record.status,
                        verified_at=_rfc3339(record.verified_at),
                    )
            try:
                # The plaintext is passed to the plugin only; it is never
                # persisted or logged by the service layer.
                passed = bool(verifier.verify(body.evidence, context))
            except Exception:
                # No state is written on plugin failure, so no half-completed
                # verification can persist, and no context leaks downstream.
                raise HTTPException(
                    status_code=500, detail="evidence verification failed"
                ) from None
            status = "verified" if passed else "rejected"
            now = _utcnow()
            with session_factory() as session:
                # Atomic settle: only a record still in "received" is updated,
                # so a racing committer can never be overwritten.
                result = session.execute(
                    update(Evidence)
                    .where(
                        Evidence.evidence_id == evidence_id,
                        Evidence.status == "received",
                    )
                    .values(status=status, verified_at=now)
                    .execution_options(synchronize_session=False)
                )
                if result.rowcount != 1:
                    session.rollback()
                    fresh = session.get(Evidence, evidence_id)
                    return EvidenceVerifiedResponse(
                        evidence_id=fresh.evidence_id,
                        challenge_id=fresh.challenge_id,
                        status=fresh.status,
                        verified_at=_rfc3339(fresh.verified_at),
                    )
                session.commit()
            return EvidenceVerifiedResponse(
                evidence_id=evidence_id,
                challenge_id=context.challenge_id,
                status=status,
                verified_at=_rfc3339(now),
            )

    return app


app = create_app()
