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

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field, StrictInt, StrictStr, field_validator
from sqlalchemy import (
    and_,
    create_engine,
    event,
    func,
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
    RELEASE_GRANT_STATUS_CODES,
    RELEASE_GRANT_STATUS_REVOKED,
    REWRAP_RESULT_KEYRING,
    REWRAP_RESULT_MISSING_KEY,
    REWRAP_RESULT_REWRAP_FAILED,
    REWRAP_RESULT_REWRAPPED,
    REWRAP_RESULT_SKIPPED,
    VERIFICATION_RESULT_ACCEPTED,
    VERIFICATION_RESULT_REJECTED,
    AUDIT_EVENT_STATUS_CONSUMED,
    AUDIT_EVENT_STATUS_PENDING,
    AUDIT_EVENT_STATUS_REVOKED,
    AUDIT_EVENT_STATUS_REWRAPPED,
    AUDIT_EVENT_STATUS_SKIPPED,
    AUDIT_EVENT_TYPE_CODES,
    AUDIT_EVENT_TYPE_GRANT,
    AUDIT_EVENT_TYPE_REWRAP,
    AUDIT_EVENT_STATUS_CODES,
    AuditEvent,
    Base,
    Challenge,
    DataEnvelope,
    Decision,
    Evidence,
    Policy,
    ReleaseGrant,
    RewrapBatch,
    RewrapBatchItem,
    TrustRoot,
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
REWRAP_BATCH_DEFAULT_LIMIT = 50
REWRAP_BATCH_MIN_LIMIT = 1
REWRAP_BATCH_MAX_LIMIT = 200

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


def _validate_grant_path_id(grant_id: str) -> str:
    """Validate and normalize a release-grant id taken from a URL path.

    The one-time grant actions (consume, revoke, release) all key off a
    path grant id that must be a canonical-shaped UUID. A missing or empty
    segment, surrounding whitespace, or any other non-UUID spelling is a
    422 client error indistinguishable from any other bad field, and is
    rejected *before* storage is touched — so such a request can never
    create a row, settle a grant, write an audit event or trigger any
    decryption. Uppercase hex is accepted and normalized to lowercase on
    every one of the three actions, matching the batch-identifier path.
    """
    if not grant_id.strip() or not _UUID_RE.fullmatch(grant_id.lower()):
        raise HTTPException(status_code=422, detail="invalid grant identifier")
    return grant_id.strip().lower()


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


#: Fixed page size for the read-only compliance audit-event listing. As
#: with the grant audit, the page size is an internal constant and never
#: part of the request or response contract.
AUDIT_EVENT_PAGE_SIZE = 100

#: Discriminator embedded in compliance audit-event cursors so neither a
#: rewrap-batch cursor nor a release-grant audit cursor (all authenticated
#: with the same secret) can ever be replayed here.
_AUDIT_EVENT_CURSOR_KIND = "compliance-audit-events-v1"


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
        "release_grants": (
            ("revoked_at", "DATETIME"),
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
            # The compliance event commits in the same transaction as the
            # grant row, so a pending event exists if and only if the grant
            # did. Only identifiers, the fixed status, the timestamp and
            # the capability digest are recorded — never the capability.
            session.add(
                AuditEvent(
                    event_id=str(uuid.uuid4()),
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

    @app.post("/v1/release-grants//consume")
    def consume_release_grant_identifier_required() -> Response:
        # An empty path segment is a missing grant identifier: a 422
        # client error rather than a routing-level 404.
        raise HTTPException(status_code=422, detail="invalid grant identifier")

    @app.post(
        "/v1/release-grants/{grant_id}/consume",
        response_model=ReleaseGrantConsumedResponse,
    )
    def consume_release_grant(
        grant_id: str, body: ConsumeReleaseGrantRequest
    ) -> ReleaseGrantConsumedResponse:
        # Grant ids are canonical-shaped UUIDs; a syntactically illegal
        # path identifier is a 422 field error indistinguishable from any
        # other bad input, never a lookup. No storage is touched past it.
        grant_id = _validate_grant_path_id(grant_id)
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
            # A settled grant can never be consumed: revocation is a
            # terminal state exactly like consumed, and is judged before
            # expiry so a grant revoked while pending is reported 409 even
            # after it has since expired.
            if grant.status == "consumed":
                raise HTTPException(status_code=409, detail="grant already consumed")
            if grant.status == RELEASE_GRANT_STATUS_REVOKED:
                raise HTTPException(status_code=409, detail="grant already revoked")
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
                if fresh is not None and fresh.status == RELEASE_GRANT_STATUS_REVOKED:
                    # A concurrent revocation won the shared state; the
                    # loser only observes the terminal revoked status.
                    raise HTTPException(
                        status_code=409, detail="grant already revoked"
                    )
                raise HTTPException(status_code=410, detail="grant expired")
            session.add(
                AuditEvent(
                    event_id=str(uuid.uuid4()),
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
        # The same canonical-UUID path gate as the consume and release
        # actions: a syntactically illegal identifier is a 422, never a
        # lookup, and writes no state.
        grant_id = _validate_grant_path_id(grant_id)

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
            session.add(
                AuditEvent(
                    event_id=str(uuid.uuid4()),
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
        # An empty path segment is a missing grant identifier: a 422
        # client error rather than a routing-level 404.
        raise HTTPException(status_code=422, detail="invalid grant identifier")

    @app.post("/v1/release/{grant_id}")
    def release_payload(grant_id: str, body: ReleasePayloadRequest) -> Response:
        """Release one protected payload against a one-time grant.

        The path grant id is gated to a canonical UUID shape before any
        storage access: an illegal identifier is a 422 that can never
        create a record, settle the grant, write an audit row or trigger
        decryption. Judgement order for a well-formed id is fixed:
        unknown/cross-scope grant or data item (404), capability mismatch
        (401), expiry (410), already consumed or revoked (409). The
        one-time state is the very same release_grants row used by the
        consume and revoke endpoints: the data key is unwrapped and the
        payload authenticated-decrypted *before* the pending -> consumed
        transition is committed, so any keyring or decryption failure
        leaves the grant pending and writes no consumption audit, and a
        revocation that settles first makes this path observe 409 without
        releasing.
        """
        grant_id = _validate_grant_path_id(grant_id)
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
            session.add(
                AuditEvent(
                    event_id=str(uuid.uuid4()),
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
                # The compliance event commits in the same transaction as
                # the material rotation; an already-current no-op retry
                # (above) writes no event because it changes no material.
                session.add(
                    AuditEvent(
                        event_id=str(uuid.uuid4()),
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
                session.add(
                    AuditEvent(
                        event_id=str(uuid.uuid4()),
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

    return app


app = create_app()
