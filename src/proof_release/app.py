from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictInt, StrictStr, field_validator
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker

from proof_release.db import Base, Challenge

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

    @field_validator("nonce")
    @classmethod
    def _nonce_format(cls, value: str) -> str:
        if not _NONCE_RE.fullmatch(value):
            raise ValueError("nonce must be unpadded base64url")
        return value


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

    return app


app = create_app()
