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
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
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
    WORKLOAD_IDENTITY_STATUS_ACTIVE,
    WORKLOAD_IDENTITY_STATUS_REVOKED,
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
    CertificateRevocation,
    Challenge,
    DataEnvelope,
    Decision,
    Evidence,
    Policy,
    RateLimitCounter,
    ReleaseGrant,
    RewrapBatch,
    RewrapBatchItem,
    TrustRoot,
    WorkloadIdentityClaim,
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


def _rfc3339(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


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
        "workload_identity_profiles": (
            ("status", "VARCHAR(16)"),
            ("updated_at", "DATETIME"),
            ("revoked_at", "DATETIME"),
        ),
        "workload_identity_claims": (
            ("seq", "INTEGER"),
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
            # Revocation check, X.509 chain format only: the DER
            # fingerprint of every chain certificate (root, intermediates,
            # leaf) is matched against in-effect registrations under the
            # trust root the chain anchors to. A hit settles the evidence
            # as rejected without invoking the verifier — the same outcome
            # as any other rejection. The anchor trust-root row is locked
            # for the rest of this transaction; registration takes the
            # same lock before inserting, so registration and verification
            # on one trust root are strictly ordered: only registrations
            # committed before this settlement are visible (SQLite already
            # serializes every writer via BEGIN IMMEDIATE and ignores FOR
            # UPDATE). A settled conclusion is never rewritten by later
            # registrations. A registry failure is a 500: the transaction
            # rolls back and the evidence stays received, so it can be
            # re-verified once the registry recovers.
            revoked = False
            # Identity gating state, populated only for an X.509 chain
            # that parses and anchors to a configured trust root. When set,
            # the anchor's workload identity profiles are consulted after
            # the verifier passes; left None when the chain cannot possibly
            # verify (the verifier rejects on its own and no lookup runs).
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
                    # serializes with concurrent registrations. None means
                    # unconfigured: the verifier rejects on its own and no
                    # revocation lookup happens.
                    anchor = session.scalar(
                        select(TrustRoot)
                        .where(
                            TrustRoot.tenant_id == body.tenant_id,
                            TrustRoot.workload_id == body.workload_id,
                            TrustRoot.cert_sha256 == anchor_digest,
                        )
                        .with_for_update()
                    )
                    if anchor is not None:
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
                            logger.error(
                                "revocation registry query failed for evidence %s",
                                evidence.evidence_id,
                            )
                            raise HTTPException(
                                status_code=500,
                                detail="revocation registry unavailable",
                            )
                        revoked = hit is not None
                        # The chain parses and anchors to a configured
                        # trust root: its identity profiles gate the
                        # verifier's accept verdict. The leaf supplies the
                        # parsed issuer, subject and SAN URI strings
                        # compared against claims.
                        identity_anchor_id = anchor.root_id
                        identity_leaf = chain_certificates[0]
            if revoked:
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
                    session.rollback()
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

    @app.post("/v1/workload-identities", status_code=201)
    def register_workload_identity(body: RegisterWorkloadIdentityRequest) -> Response:
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
        """
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
        body_bytes = json.dumps(
            {
                "profile_id": profile_id,
                "tenant_id": body.tenant_id,
                "workload_id": body.workload_id,
                "trust_root_id": body.trust_root_id,
                "claims": [
                    {"issuer": issuer, "subject": subject, "uri": uri}
                    for issuer, subject, uri in claims
                ],
                "created_at": _rfc3339(now),
            },
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return Response(
            content=body_bytes, status_code=201, media_type="application/json"
        )

    async def _require_empty_query_body(request: Request) -> None:
        """Reject a read-only query carrying any request body (422).

        The identity-profile query is ranged entirely by query
        parameters, so a body is never meaningful. The body is read
        directly (rather than trusting Content-Length, which chunked or
        HTTP/2 clients may omit) and anything non-empty — including
        whitespace or non-JSON — is rejected before any state is read.
        """
        if await request.body() != b"":
            raise HTTPException(status_code=422, detail="query body must be empty")

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
        grant_id: str, body: ConsumeReleaseGrantRequest
    ) -> ReleaseGrantConsumedResponse:
        # Grant ids are canonical lowercase UUIDs; a syntactically illegal
        # path identifier is a 422 field error indistinguishable from any
        # other bad input, never a lookup, and never consumes budget.
        if not grant_id.strip() or not _UUID_RE.fullmatch(grant_id.lower()):
            raise HTTPException(status_code=422, detail="invalid grant identifier")
        grant_id = grant_id.strip().lower()

        # Basic field/format validation (the request model and the path
        # check above) has passed: reserve one shared per-scope minute slot
        # before any business judgement. A 429 or a counter failure surfaces
        # here and changes no grant, payload or audit state.
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
