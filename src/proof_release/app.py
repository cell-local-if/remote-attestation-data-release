"""FastAPI application: health check plus persistent random challenges."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import update
from sqlalchemy.orm import sessionmaker

from .db import Base, Challenge, make_engine

DEFAULT_DATABASE_URL = "sqlite:///./proof_release.db"
DATABASE_URL_ENV = "PROOF_RELEASE_DATABASE_URL"

NONCE_BYTES = 32
STATUS_PENDING = "pending"
STATUS_CONSUMED = "consumed"


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding, altchars=b"-_", validate=True)


def _nonce_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class ChallengeCreateRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    workload_id: str = Field(min_length=1)
    ttl_seconds: int = Field(default=300, ge=30, le=900)


class ChallengeConsumeRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    workload_id: str = Field(min_length=1)
    nonce: str = Field(min_length=1)

    @field_validator("nonce")
    @classmethod
    def _nonce_must_be_base64url(cls, value: str) -> str:
        try:
            _b64url_decode(value)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("nonce must be base64url encoded") from exc
        return value


def create_app(database_url: str | None = None) -> FastAPI:
    """Build the application bound to a database URL.

    Tables are created at startup; the default SQLite file keeps challenge
    state across process restarts.
    """
    url = database_url or os.environ.get(DATABASE_URL_ENV, DEFAULT_DATABASE_URL)
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)

    app = FastAPI(title="Remote Attestation Data Release")
    app.state.engine = engine
    app.state.session_factory = session_factory

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/v1/challenges", status_code=201)
    def create_challenge(payload: ChallengeCreateRequest) -> dict[str, str]:
        nonce_raw = secrets.token_bytes(NONCE_BYTES)
        now = datetime.now(timezone.utc)
        challenge = Challenge(
            challenge_id=str(uuid.uuid4()),
            tenant_id=payload.tenant_id,
            workload_id=payload.workload_id,
            nonce_hash=_nonce_digest(nonce_raw),
            status=STATUS_PENDING,
            issued_at=now,
            expires_at=now + timedelta(seconds=payload.ttl_seconds),
        )
        with session_factory() as session:
            session.add(challenge)
            session.commit()
        # The plaintext nonce appears only in this response; the database
        # holds its SHA-256 digest and it is never logged.
        return {
            "challenge_id": challenge.challenge_id,
            "nonce": _b64url_encode(nonce_raw),
            "issued_at": _rfc3339(challenge.issued_at),
            "expires_at": _rfc3339(challenge.expires_at),
            "status": challenge.status,
        }

    @app.post("/v1/challenges/{challenge_id}/consume")
    def consume_challenge(challenge_id: str, payload: ChallengeConsumeRequest) -> dict[str, str]:
        nonce_raw = _b64url_decode(payload.nonce)
        digest = _nonce_digest(nonce_raw)
        now = datetime.now(timezone.utc)
        with session_factory() as session:
            challenge = session.get(Challenge, challenge_id)
            if (
                challenge is None
                or challenge.tenant_id != payload.tenant_id
                or challenge.workload_id != payload.workload_id
            ):
                raise HTTPException(status_code=404, detail="challenge not found")
            nonce_hash = challenge.nonce_hash
            status = challenge.status
            expires_at = challenge.expires_at
            # Release the read transaction before the atomic write so
            # concurrent consumers serialize on the UPDATE, not on reads.
            session.rollback()
            if not hmac.compare_digest(nonce_hash, digest):
                raise HTTPException(status_code=401, detail="nonce mismatch")
            if status == STATUS_CONSUMED:
                raise HTTPException(status_code=409, detail="challenge already consumed")
            if expires_at <= now:
                raise HTTPException(status_code=410, detail="challenge expired")
            result = session.execute(
                update(Challenge)
                .where(
                    Challenge.challenge_id == challenge_id,
                    Challenge.status == STATUS_PENDING,
                )
                .values(status=STATUS_CONSUMED, consumed_at=now)
            )
            if result.rowcount != 1:
                session.rollback()
                raise HTTPException(status_code=409, detail="challenge already consumed")
            session.commit()
        return {
            "challenge_id": challenge_id,
            "status": STATUS_CONSUMED,
            "consumed_at": _rfc3339(now),
        }

    return app


app = create_app()
