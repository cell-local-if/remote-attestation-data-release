"""Tests for GET /v1/release-grants/{grant_id}/trace.

The trace is the read-only full-chain query that joins one release grant
to its decision, its data envelope and its final outcome in a single
response. It validates every request-shape rule to 422 before any state
is read, hides unknown or cross-scope grants behind one indistinguishable
404, returns only the four fixed sections (grant, decision, envelope,
outcome) as compact JSON with a trailing newline, exposes the capability
only as its SHA-256 digest (never plaintext material, payloads or key
material), writes no state and consumes none of the shared grant
rate-limit budget, survives restarts, and discards the half-built result
with a 500 when storage cannot complete the chain.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    DataEnvelope,
    Decision,
    RateLimitCounter,
    ReleaseGrant,
)
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
MASTER_KEY = b64url_encode(b"0123456789abcdef0123456789abcdef")
PAYLOAD = 'traced secret payload 🔐 with "quote" and \\ backslash'

ZERO_UUID = "00000000-0000-0000-0000-000000000000"
UPPER_UUID = "ABCDEF12-3456-7890-ABCD-EF1234567890"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/grant_traces.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


# --- lifecycle builders ----------------------------------------------------


def _mac(nonce: str, claims: dict, *, tenant=TENANT, workload=WORKLOAD) -> str:
    key = hmac.new(
        SECRET.encode("utf-8"),
        f"{tenant}:{workload}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _evidence_text(nonce: str, claims=None):
    claims = {"m": "x"} if claims is None else claims
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)}
    )


def _decision(client, *, tenant=TENANT, workload=WORKLOAD, claims=None,
              rule=None, name="release"):
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    evidence = _evidence_text(created["nonce"], claims)
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201, submitted.text
    evidence_id = submitted.json()["evidence_id"]
    verified = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.status_code == 200, verified.text
    policy = client.post(
        "/v1/policies",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "name": name,
            "rule": rule if rule is not None else {"claim": "m", "equals": "x"},
        },
    )
    assert policy.status_code == 201, policy.text
    decided = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy.json()["policy_id"],
        },
    )
    assert decided.status_code == 200, decided.text
    return created, evidence, policy.json(), decided.json()


def _envelope(client, data_id=DATA_ID, payload=PAYLOAD, *,
              tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": payload,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _grant(client, decision_id, data_id=DATA_ID, *,
           tenant=TENANT, workload=WORKLOAD, **fields):
    body = {
        "tenant_id": tenant,
        "workload_id": workload,
        "decision_id": decision_id,
        "data_id": data_id,
    }
    body.update(fields)
    response = client.post("/v1/release-grants", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _trace(client, grant_id, *, tenant=TENANT, workload=WORKLOAD, **extra):
    params = {"tenant_id": tenant, "workload_id": workload, **extra}
    return client.get(f"/v1/release-grants/{grant_id}/trace", params=params)


def _full_chain(client, *, data_id=DATA_ID):
    created, evidence, policy, decided = _decision(client)
    envelope = _envelope(client, data_id=data_id)
    grant = _grant(client, decided["decision_id"], data_id=data_id)
    return {
        "created": created,
        "evidence": evidence,
        "policy": policy,
        "decision": decided,
        "envelope": envelope,
        "grant": grant,
    }


# --- request validation ----------------------------------------------------


def test_trace_requires_scope_parameters(client):
    response = client.get(f"/v1/release-grants/{ZERO_UUID}/trace")
    assert response.status_code == 422
    assert (
        client.get(
            f"/v1/release-grants/{ZERO_UUID}/trace",
            params={"workload_id": WORKLOAD},
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/v1/release-grants/{ZERO_UUID}/trace",
            params={"tenant_id": TENANT},
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": "\t"},
        {"workload_id": ""},
        {"workload_id": "  "},
        {"bogus": "value"},
        {"cursor": "abc"},
        {"status": "pending"},
        {"grant_id": ZERO_UUID},
        {"decision_id": ZERO_UUID},
        {"data_id": DATA_ID},
    ],
)
def test_trace_rejects_invalid_or_unknown_parameters(client, params):
    response = client.get(
        f"/v1/release-grants/{ZERO_UUID}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD, **params},
    )
    assert response.status_code == 422, response.text


def test_trace_rejects_repeated_parameter(client):
    response = client.get(
        f"/v1/release-grants/{ZERO_UUID}/trace",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("tenant_id", OTHER_TENANT),
        ],
    )
    assert response.status_code == 422
    response = client.get(
        f"/v1/release-grants/{ZERO_UUID}/trace",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("workload_id", OTHER_WORKLOAD),
        ],
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "grant_id",
    [
        " ",
        "not-a-uuid",
        ZERO_UUID[:-1] + "Z",
        "  " + ZERO_UUID,
        ZERO_UUID + " ",
        UPPER_UUID,
        "1234567890",
        "g0000000-0000-0000-0000-000000000000",
    ],
)
def test_trace_rejects_non_canonical_path_identifier(client, grant_id):
    response = _trace(client, grant_id)
    assert response.status_code == 422, response.text


def test_trace_empty_path_segment_is_422(client):
    response = client.get(
        "/v1/release-grants//trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"{}", b"  ", b"\t", b"not json", b"\x00"])
def test_trace_non_empty_body_is_422(client, body):
    chain = _full_chain(client)
    response = client.request(
        "GET",
        f"/v1/release-grants/{chain['grant']['grant_id']}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=body,
    )
    assert response.status_code == 422


def test_trace_missing_and_zero_length_body_are_accepted(client):
    chain = _full_chain(client)
    no_body = _trace(client, chain["grant"]["grant_id"])
    empty_body = client.request(
        "GET",
        f"/v1/release-grants/{chain['grant']['grant_id']}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"",
    )
    assert no_body.status_code == 200, no_body.text
    assert empty_body.status_code == 200, empty_body.text
    assert no_body.content == empty_body.content


def test_invalid_requests_read_or_write_no_state(app, client):
    chain = _full_chain(client)
    _trace(client, "not-a-uuid")
    _trace(client, ZERO_UUID, tenant=" ")
    client.get(
        f"/v1/release-grants/{ZERO_UUID}/trace",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("workload_id", OTHER_WORKLOAD),
        ],
    )
    client.request(
        "GET",
        "/v1/release-grants//trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"x",
    )
    with app.state.session_factory() as session:
        assert session.query(ReleaseGrant).count() == 1
        assert session.query(Decision).count() == 1
        assert session.query(DataEnvelope).count() == 1
        assert session.query(AuditEvent).count() == 1
        stored = session.get(ReleaseGrant, chain["grant"]["grant_id"])
        assert stored.status == "pending"
        assert stored.consumed_at is None
        assert stored.revoked_at is None
        assert session.query(RateLimitCounter).count() == 0


# --- 404 semantics ---------------------------------------------------------


def test_trace_unknown_grant_is_404(client):
    assert _trace(client, ZERO_UUID).status_code == 404


def test_trace_cross_scope_grant_is_404(client):
    chain = _full_chain(client)
    grant_id = chain["grant"]["grant_id"]
    assert _trace(client, grant_id, tenant=OTHER_TENANT).status_code == 404
    assert _trace(client, grant_id, workload=OTHER_WORKLOAD).status_code == 404
    assert (
        _trace(
            client, grant_id, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
        ).status_code
        == 404
    )


def test_trace_unknown_and_cross_scope_are_indistinguishable(client):
    chain = _full_chain(client)
    unknown = _trace(client, ZERO_UUID)
    cross = _trace(client, chain["grant"]["grant_id"], tenant=OTHER_TENANT)
    assert unknown.status_code == cross.status_code == 404
    assert unknown.content == cross.content


# --- success shape ---------------------------------------------------------


def test_trace_success_shape_and_field_order(client):
    chain = _full_chain(client)
    grant = chain["grant"]
    response = _trace(client, grant["grant_id"])
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    raw = response.content
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw

    parsed = json.loads(raw)
    assert list(parsed) == ["grant", "decision", "envelope", "outcome"]

    result = parsed["grant"]
    assert list(result) == [
        "grant_id",
        "decision_id",
        "data_id",
        "status",
        "issued_at",
        "expires_at",
        "consumed_at",
        "revoked_at",
        "capability_sha256",
    ]
    assert result["grant_id"] == grant["grant_id"]
    assert result["decision_id"] == chain["decision"]["decision_id"]
    assert result["data_id"] == DATA_ID
    assert result["status"] == "pending"
    assert result["issued_at"] == grant["issued_at"]
    assert result["expires_at"] == grant["expires_at"]
    assert result["consumed_at"] is None
    assert result["revoked_at"] is None
    assert result["capability_sha256"] == hashlib.sha256(
        grant["capability"].encode("ascii")
    ).hexdigest()
    for key in (
        "grant_id",
        "decision_id",
        "data_id",
        "status",
        "issued_at",
        "expires_at",
        "capability_sha256",
    ):
        assert isinstance(result[key], str) and result[key]

    decision = parsed["decision"]
    assert list(decision) == [
        "decision_id",
        "evidence_id",
        "policy_version",
        "status",
        "decided_at",
    ]
    assert decision["decision_id"] == chain["decision"]["decision_id"]
    assert decision["evidence_id"] == chain["decision"]["evidence_id"]
    assert decision["policy_version"] == 1
    assert decision["status"] == "allowed"
    assert decision["decided_at"] == chain["decision"]["decided_at"]
    assert isinstance(decision["policy_version"], int)
    assert not isinstance(decision["policy_version"], bool)

    envelope = parsed["envelope"]
    assert list(envelope) == [
        "data_id",
        "tenant_id",
        "workload_id",
        "key_version",
        "created_at",
    ]
    assert envelope["data_id"] == DATA_ID
    assert envelope["tenant_id"] == TENANT
    assert envelope["workload_id"] == WORKLOAD
    assert envelope["key_version"] == chain["envelope"]["key_version"]
    assert envelope["created_at"] == chain["envelope"]["created_at"]
    assert isinstance(envelope["key_version"], int)
    assert not isinstance(envelope["key_version"], bool)
    assert isinstance(envelope["created_at"], str) and envelope["created_at"]

    outcome = parsed["outcome"]
    assert list(outcome) == ["status", "at"]
    assert outcome == {"status": "pending", "at": None}

    for section in ("grant", "decision", "envelope", "outcome"):
        for value in parsed[section].values():
            assert value is None or isinstance(value, (str, int))
            assert not isinstance(value, bool)


def test_trace_timestamps_are_utc_rfc3339_strings(client):
    chain = _full_chain(client)
    parsed = _trace(client, chain["grant"]["grant_id"]).json()
    stamps = [
        parsed["grant"]["issued_at"],
        parsed["grant"]["expires_at"],
        parsed["decision"]["decided_at"],
        parsed["envelope"]["created_at"],
    ]
    for stamp in stamps:
        parsed_at = datetime.fromisoformat(stamp)
        assert parsed_at.utcoffset() == timedelta(0)


def test_trace_pending_outcome_stays_pending_after_expiry(client):
    _, _, _, decided = _decision(client)
    grant = _grant(client, decided["decision_id"], ttl_seconds=30)
    # Force expiry in stored state; no business request settles the grant.
    with client.app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    parsed = _trace(client, grant["grant_id"]).json()
    assert parsed["grant"]["status"] == "pending"
    assert parsed["outcome"] == {"status": "pending", "at": None}
    assert parsed["grant"]["consumed_at"] is None
    assert parsed["grant"]["revoked_at"] is None


def test_trace_grant_without_envelope_uses_null_envelope_fields(client):
    # Grants may be minted before the envelope exists; the trace must
    # still answer 200 (never a 404 that would reveal the missing data
    # item), reporting the grant's own scope and null envelope metadata.
    _, _, _, decided = _decision(client)
    grant = _grant(client, decided["decision_id"])
    response = _trace(client, grant["grant_id"])
    assert response.status_code == 200, response.text
    assert response.json()["envelope"] == {
        "data_id": DATA_ID,
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "key_version": None,
        "created_at": None,
    }
    assert response.json()["outcome"] == {"status": "pending", "at": None}


def test_trace_consumed_grant_outcome(client):
    chain = _full_chain(client)
    consumed = client.post(
        f"/v1/release-grants/{chain['grant']['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": chain["grant"]["capability"],
        },
    )
    assert consumed.status_code == 200, consumed.text
    consumed_at = consumed.json()["consumed_at"]

    parsed = _trace(client, chain["grant"]["grant_id"]).json()
    assert parsed["grant"]["status"] == "consumed"
    assert parsed["grant"]["consumed_at"] == consumed_at
    assert parsed["grant"]["revoked_at"] is None
    assert parsed["outcome"] == {"status": "consumed", "at": consumed_at}
    assert list(parsed["outcome"]) == ["status", "at"]


def test_trace_revoked_grant_outcome(client):
    chain = _full_chain(client)
    revoked = client.post(
        f"/v1/release-grants/{chain['grant']['grant_id']}/revoke",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": chain["grant"]["capability"],
        },
    )
    assert revoked.status_code == 200, revoked.text
    revoked_at = revoked.json()["revoked_at"]

    parsed = _trace(client, chain["grant"]["grant_id"]).json()
    assert parsed["grant"]["status"] == "revoked"
    assert parsed["grant"]["revoked_at"] == revoked_at
    assert parsed["grant"]["consumed_at"] is None
    assert parsed["outcome"] == {"status": "revoked", "at": revoked_at}


def test_trace_consumed_via_payload_release_outcome(client):
    chain = _full_chain(client)
    released = client.post(
        f"/v1/release/{chain['grant']['grant_id']}",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": DATA_ID,
            "capability": chain["grant"]["capability"],
        },
    )
    assert released.status_code == 200, released.text
    assert released.json() == {"payload": PAYLOAD}

    parsed = _trace(client, chain["grant"]["grant_id"]).json()
    assert parsed["grant"]["status"] == "consumed"
    assert parsed["outcome"]["status"] == "consumed"
    assert isinstance(parsed["outcome"]["at"], str)
    assert parsed["grant"]["consumed_at"] == parsed["outcome"]["at"]


def test_trace_repeated_reads_return_identical_result(client):
    chain = _full_chain(client)
    first = _trace(client, chain["grant"]["grant_id"])
    second = _trace(client, chain["grant"]["grant_id"])
    assert first.status_code == second.status_code == 200
    assert first.content == second.content

    consumed = client.post(
        f"/v1/release-grants/{chain['grant']['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": chain["grant"]["capability"],
        },
    )
    assert consumed.status_code == 200
    third = _trace(client, chain["grant"]["grant_id"])
    fourth = _trace(client, chain["grant"]["grant_id"])
    assert third.content == fourth.content
    assert third.content != first.content


# --- no protected material -------------------------------------------------


def test_trace_never_returns_protected_material(client):
    chain = _full_chain(client)
    response = _trace(client, chain["grant"]["grant_id"])
    text = response.text
    # The capability appears only as its digest.
    assert chain["grant"]["capability"] not in text
    assert chain["evidence"] not in text
    assert chain["created"]["nonce"] not in text
    assert PAYLOAD not in text
    for forbidden in (
        '"capability"',
        '"payload"',
        '"nonce"',
        '"claims"',
        '"mac"',
        '"ciphertext"',
        '"iv"',
        '"tag"',
        '"wrapped_key"',
        '"rule"',
        '"root_pem"',
        '"evidence_sha256"',
    ):
        assert forbidden not in text, forbidden
    # The digest is present and is a 64-char lowercase hex string.
    parsed = response.json()
    digest = parsed["grant"]["capability_sha256"]
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_trace_has_no_floats_or_non_finite_values(client):
    chain = _full_chain(client)
    raw = _trace(client, chain["grant"]["grant_id"]).content
    parsed = json.loads(raw)

    def _check(value):
        if isinstance(value, bool):  # pragma: no cover - structural guard
            raise AssertionError("booleans are not part of the trace contract")
        if isinstance(value, float):  # pragma: no cover - structural guard
            raise AssertionError("trace must not contain floats")
        assert value is None or isinstance(value, (str, int))

    for section in ("grant", "decision", "envelope", "outcome"):
        for value in parsed[section].values():
            _check(value)
    assert b"NaN" not in raw and b"Infinity" not in raw


# --- read-only behaviour ---------------------------------------------------


def test_trace_writes_no_state(app, client):
    chain = _full_chain(client)
    with app.state.session_factory() as session:
        grants_before = session.query(ReleaseGrant).count()
        decisions_before = session.query(Decision).count()
        envelopes_before = session.query(DataEnvelope).count()
        audits_before = session.query(AuditEvent).count()
        stored = session.get(ReleaseGrant, chain["grant"]["grant_id"])
        status_before = stored.status
        issued_before = stored.issued_at
        expires_before = stored.expires_at

    for _ in range(3):
        assert _trace(client, chain["grant"]["grant_id"]).status_code == 200

    with app.state.session_factory() as session:
        assert session.query(ReleaseGrant).count() == grants_before
        assert session.query(Decision).count() == decisions_before
        assert session.query(DataEnvelope).count() == envelopes_before
        assert session.query(AuditEvent).count() == audits_before
        assert session.query(RateLimitCounter).count() == 0
        stored = session.get(ReleaseGrant, chain["grant"]["grant_id"])
        assert stored.status == status_before
        assert stored.issued_at == issued_before
        assert stored.expires_at == expires_before
        assert stored.consumed_at is None
        assert stored.revoked_at is None


def test_trace_consumes_no_rate_limit_budget(client):
    chain = _full_chain(client)
    # Far more than the shared budget of five; a read-only trace is
    # explicitly outside that budget.
    for _ in range(8):
        assert _trace(client, chain["grant"]["grant_id"]).status_code == 200
    # The next business judgement is still the first of the minute's
    # budget, not a 429.
    consumed = client.post(
        f"/v1/release-grants/{chain['grant']['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": chain["grant"]["capability"],
        },
    )
    assert consumed.status_code == 200, consumed.text


def test_trace_does_not_decrypt_or_touch_envelope_material(app, client):
    chain = _full_chain(client)
    with app.state.session_factory() as session:
        before = session.get(
            DataEnvelope, (TENANT, WORKLOAD, DATA_ID)
        )
        material_before = (
            before.key_version,
            before.ciphertext,
            before.iv,
            before.tag,
            before.wrapped_key,
        )

    assert _trace(client, chain["grant"]["grant_id"]).status_code == 200

    with app.state.session_factory() as session:
        after = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        assert (
            after.key_version,
            after.ciphertext,
            after.iv,
            after.tag,
            after.wrapped_key,
        ) == material_before


# --- server failure --------------------------------------------------------


def test_trace_storage_failure_returns_500_with_no_half_result(app, client):
    from sqlalchemy import text

    chain = _full_chain(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE release_grants"))
    response = _trace(client, chain["grant"]["grant_id"])
    assert response.status_code == 500
    assert b'"grant"' not in response.content


def test_trace_missing_decision_returns_500(app, client):
    from sqlalchemy import text

    chain = _full_chain(client)
    # Decisions are never deleted by the service; simulate a damaged
    # store so the chain cannot complete. The half result is discarded.
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE decisions"))
    response = _trace(client, chain["grant"]["grant_id"])
    assert response.status_code == 500
    # No half trace leaks: the error envelope carries none of the chain
    # section keys.
    for section in (b'"grant"', b'"decision"', b'"envelope"', b'"outcome"'):
        assert section not in response.content


def test_trace_missing_envelope_returns_500(app, client):
    from sqlalchemy import text

    chain = _full_chain(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE data_envelopes"))
    response = _trace(client, chain["grant"]["grant_id"])
    assert response.status_code == 500


# --- persistence and scope -------------------------------------------------


def test_trace_consistent_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart_trace.db"
    first = create_app(url)
    client1 = TestClient(first)
    chain = _full_chain(client1)
    first_body = client1.get(
        f"/v1/release-grants/{chain['grant']['grant_id']}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).content
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    response = client2.get(
        f"/v1/release-grants/{chain['grant']['grant_id']}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    assert response.content == first_body
    parsed = response.json()
    assert parsed["grant"]["grant_id"] == chain["grant"]["grant_id"]
    assert parsed["decision"]["decision_id"] == chain["decision"]["decision_id"]
    assert parsed["envelope"]["key_version"] == chain["envelope"]["key_version"]
    second.state.engine.dispose()


def test_trace_scope_isolation_is_strict(client):
    chain = _full_chain(client)
    assert (
        _trace(
            client,
            chain["grant"]["grant_id"],
            tenant=OTHER_TENANT,
            workload=OTHER_WORKLOAD,
        ).status_code
        == 404
    )
    assert _trace(client, chain["grant"]["grant_id"]).status_code == 200


def test_trace_envelope_reflects_historical_key_version(tmp_path, monkeypatch):
    # An envelope wrapped under a historical key version is still traced
    # metadata-only: the version integer is reported and no decryption is
    # attempted by the read path.
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    import base64

    key1 = base64.urlsafe_b64encode(b"0123456789abcdef0123456789abcdef").rstrip(b"=").decode()
    key2 = base64.urlsafe_b64encode(b"fedcba9876543210fedcba9876543210").rstrip(b"=").decode()
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING",
        json.dumps({"current_version": 1, "keys": {"1": key1}}),
    )
    application = create_app(f"sqlite:///{tmp_path}/historical.db")
    client = TestClient(application)
    try:
        _, _, _, decided = _decision(client)
        envelope = _envelope(client)
        assert envelope["key_version"] == 1
        grant = _grant(client, decided["decision_id"])

        # Rotate the keyring forward without rewrapping the envelope.
        monkeypatch.setenv(
            "PROOF_RELEASE_KEYRING",
            json.dumps(
                {"current_version": 2, "keys": {"1": key1, "2": key2}}
            ),
        )
        # Recreate nothing in-process; the read path never consults the
        # keyring at all, so even removing the historical key must not
        # change the trace.
        monkeypatch.setenv(
            "PROOF_RELEASE_KEYRING",
            json.dumps({"current_version": 2, "keys": {"2": key2}}),
        )
        response = _trace(client, grant["grant_id"])
        assert response.status_code == 200, response.text
        parsed = response.json()
        assert parsed["envelope"]["key_version"] == 1
    finally:
        application.state.engine.dispose()
