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
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictInt, StrictStr, field_validator
from sqlalchemy import create_engine, event, func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding

from proof_release.db import (
    DECISION_DENIED,
    DECISION_GRANTED,
    VERIFICATION_RESULT_ACCEPTED,
    VERIFICATION_RESULT_REJECTED,
    Base,
    Challenge,
    Decision,
    Evidence,
    Policy,
    TrustRoot,
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


#: Bound on rule nesting, guarding validation and evaluation against
#: pathologically deep documents.
MAX_RULE_DEPTH = 32

#: Evidence formats whose documents are built-in JSON carrying a ``claims``
#: object; only these can feed claim-based policy evaluation.
_BUILTIN_CLAIMS_FORMATS = frozenset({ATTESTED_NONCE_JSON, X509_ATTESTED_NONCE_JSON})


def _validate_rule_grammar(rule: Any, _depth: int = 0) -> Any:
    """Validate a policy rule against the supported grammar.

    A rule is exactly one of::

        {"claim": "<top-level claim name>", "equals": <scalar>}
        {"all": [<rule>, ...]}
        {"any": [<rule>, ...]}
        {"not": <rule>}

    Raises ValueError (mapped to 422 at the API boundary) for anything else.
    """
    if _depth > MAX_RULE_DEPTH:
        raise ValueError("rule nesting is too deep")
    if not isinstance(rule, dict):
        raise ValueError("rule must be a JSON object")
    keys = set(rule)
    if keys == {"claim", "equals"}:
        claim = rule["claim"]
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("claim must be a non-empty string")
        if isinstance(rule["equals"], (dict, list)):
            raise ValueError("equals must be a scalar (string, number, boolean or null)")
        return rule
    if keys == {"all"} or keys == {"any"}:
        subrules = rule["all"] if "all" in rule else rule["any"]
        if not isinstance(subrules, list) or not subrules:
            raise ValueError("all/any must be a non-empty array of rules")
        for subrule in subrules:
            _validate_rule_grammar(subrule, _depth + 1)
        return rule
    if keys == {"not"}:
        _validate_rule_grammar(rule["not"], _depth + 1)
        return rule
    raise ValueError(
        "rule must be one of "
        '{"claim": ..., "equals": ...}, {"all": [...]}, {"any": [...]} or {"not": ...}'
    )


_MISSING = object()


def _scalar_equals(actual: Any, expected: Any) -> bool:
    """JSON scalar comparison; booleans are never equal to numbers."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return (
            isinstance(actual, bool)
            and isinstance(expected, bool)
            and actual == expected
        )
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected
    if isinstance(actual, str) and isinstance(expected, str):
        return actual == expected
    return False


def _evaluate_rule(rule: dict, claims: dict) -> bool:
    """Evaluate a previously validated rule against top-level claims."""
    if "claim" in rule:
        actual = claims.get(rule["claim"], _MISSING)
        if actual is _MISSING:
            return False
        return _scalar_equals(actual, rule["equals"])
    if "all" in rule:
        return all(_evaluate_rule(subrule, claims) for subrule in rule["all"])
    if "any" in rule:
        return any(_evaluate_rule(subrule, claims) for subrule in rule["any"])
    return not _evaluate_rule(rule["not"], claims)


def _extract_claims(evidence_format: str, evidence: str) -> dict:
    """Extract the claims object from built-in JSON evidence.

    The evidence is used transiently on the stack only; the claims are
    evaluated and discarded, never persisted, logged, or returned.
    """
    if evidence_format not in _BUILTIN_CLAIMS_FORMATS:
        raise HTTPException(
            status_code=422, detail="unsupported evidence format for decisions"
        )
    try:
        document = json.loads(evidence)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="evidence is not valid JSON")
    if not isinstance(document, dict):
        raise HTTPException(status_code=422, detail="evidence must be a JSON object")
    claims = document.get("claims", {})
    if not isinstance(claims, dict):
        raise HTTPException(
            status_code=422, detail="evidence claims must be a JSON object"
        )
    return claims


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
    rule: Any

    _non_blank = field_validator("tenant_id", "workload_id", "name")(
        _require_non_blank
    )
    _rule_valid = field_validator("rule")(_validate_rule_grammar)


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
        # The rule is a validated policy definition; it is stored as
        # canonical JSON so every read returns the same document.
        canonical_rule = json.dumps(body.rule, sort_keys=True, separators=(",", ":"))
        # Allocate the next version within the (tenant, workload, name)
        # scope. SQLite transactions begin as BEGIN IMMEDIATE, serializing
        # concurrent creators; on locking backends the unique constraint
        # guarantees no duplicate numbers, and a loser retries.
        for _attempt in range(3):
            now = _utcnow()
            with session_factory() as session:
                current_max = session.scalar(
                    select(func.max(Policy.version)).where(
                        Policy.tenant_id == body.tenant_id,
                        Policy.workload_id == body.workload_id,
                        Policy.name == body.name,
                    )
                )
                version = (current_max or 0) + 1
                policy = Policy(
                    policy_id=str(uuid.uuid4()),
                    tenant_id=body.tenant_id,
                    workload_id=body.workload_id,
                    name=body.name,
                    version=version,
                    rule=canonical_rule,
                    created_at=now,
                )
                session.add(policy)
                try:
                    session.commit()
                except IntegrityError:
                    # A concurrent creator took this version first; retry
                    # with a fresh number.
                    session.rollback()
                    continue
            return PolicyCreatedResponse(
                policy_id=policy.policy_id,
                tenant_id=policy.tenant_id,
                workload_id=policy.workload_id,
                name=policy.name,
                version=policy.version,
                rule=json.loads(canonical_rule),
                created_at=_rfc3339(now),
            )
        raise HTTPException(status_code=409, detail="policy version conflict")

    @app.post(
        "/v1/evidence/{evidence_id}/decisions",
        response_model=DecisionResponse,
    )
    def decide_evidence(
        evidence_id: str, body: CreateDecisionRequest
    ) -> DecisionResponse:
        evidence_digest = hashlib.sha256(body.evidence.encode("utf-8")).hexdigest()
        nonce_digest = _nonce_digest(body.nonce)
        with session_factory() as session:
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
            policy = session.get(Policy, body.policy_id)
            if (
                policy is None
                or policy.tenant_id != body.tenant_id
                or policy.workload_id != body.workload_id
            ):
                raise HTTPException(status_code=404, detail="policy not found")
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

            # Already decided: retries and concurrent callers observe the
            # stored audit decision verbatim; the freshly supplied evidence
            # is not inspected and no new record is created.
            existing = session.scalar(
                select(Decision).where(
                    Decision.evidence_id == evidence_id,
                    Decision.policy_id == policy.policy_id,
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

            # Only verified evidence can be decided. The verifier is never
            # re-run here: the stored settlement is the sole input, and the
            # nonce and digest checks above bind this call to exactly the
            # evidence that was verified.
            if evidence.status != "verified":
                raise HTTPException(
                    status_code=409, detail="evidence is not verified"
                )
            if not hmac.compare_digest(evidence.evidence_sha256, evidence_digest):
                raise HTTPException(
                    status_code=422, detail="evidence digest mismatch"
                )

            # The claims live only on the stack for this call; the decision
            # persists only the outcome, never the evidence or the claims.
            claims = _extract_claims(evidence.evidence_format, body.evidence)
            rule = json.loads(policy.rule)
            status = (
                DECISION_GRANTED
                if _evaluate_rule(rule, claims)
                else DECISION_DENIED
            )
            decided_at = _utcnow()
            decision = Decision(
                decision_id=str(uuid.uuid4()),
                evidence_id=evidence.evidence_id,
                policy_id=policy.policy_id,
                policy_version=policy.version,
                tenant_id=body.tenant_id,
                workload_id=body.workload_id,
                status=status,
                decided_at=decided_at,
            )
            session.add(decision)
            try:
                session.commit()
            except IntegrityError:
                # A concurrent caller recorded the decision for this
                # evidence and policy first; return its record.
                session.rollback()
                winner = session.scalar(
                    select(Decision).where(
                        Decision.evidence_id == evidence_id,
                        Decision.policy_id == policy.policy_id,
                    )
                )
                if winner is not None:
                    return DecisionResponse(
                        decision_id=winner.decision_id,
                        evidence_id=winner.evidence_id,
                        policy_id=winner.policy_id,
                        policy_version=winner.policy_version,
                        status=winner.status,
                        decided_at=_rfc3339(winner.decided_at),
                    )
                raise HTTPException(status_code=409, detail="decision conflict")
        return DecisionResponse(
            decision_id=decision.decision_id,
            evidence_id=decision.evidence_id,
            policy_id=decision.policy_id,
            policy_version=decision.policy_version,
            status=decision.status,
            decided_at=_rfc3339(decided_at),
        )

    return app


app = create_app()
