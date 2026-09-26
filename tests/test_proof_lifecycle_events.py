"""Tests for GET /v1/compliance/proof-events.

Tenant-isolated, cursor-stable, snapshot-fixed audit timeline of the
proof lifecycle: at most one event per evidence per stage — reception,
first verification settlement and first policy decision. Events are
written in the receive/verify/decision transactions; the query is
read-only and never appends an event.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import ProofLifecycleEvent
from proof_release.envelopes import b64url_encode
from proof_release.verifiers import (
    VerificationContext,
    VerificationResult,
    Verifier,
    VerifierRegistry,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
SECRET = "unit-test-secret"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"

KEY_V1 = b64url_encode(b"0123456789abcdef0123456789abcdef")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = create_app(f"sqlite:///{tmp_path}/proof_events.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


# --- lifecycle builders ----------------------------------------------------


def _mac_for(nonce: str, claims: dict, *, tenant=TENANT, workload=WORKLOAD) -> str:
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


def _submit(client, *, evidence_format="attested-nonce-json", evidence=None,
            claims=None, tenant=TENANT, workload=WORKLOAD):
    """Create a challenge and submit evidence; return (created, evidence, eid)."""
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    if evidence is None:
        claims = {"m": "x"} if claims is None else claims
        evidence = json.dumps(
            {
                "nonce": created["nonce"],
                "claims": claims,
                "mac": _mac_for(
                    created["nonce"], claims, tenant=tenant, workload=workload
                ),
            }
        )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": evidence_format,
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201, submitted.text
    return created, evidence, submitted.json()["evidence_id"]


def _verify(client, created, evidence, evidence_id, *,
            tenant=TENANT, workload=WORKLOAD):
    return client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )


def _policy(client, rule=None, *, name="r", tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/policies",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "name": name,
            "rule": rule if rule is not None else {"claim": "m", "equals": "x"},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _decide(client, created, evidence, evidence_id, policy_id, *,
            tenant=TENANT, workload=WORKLOAD):
    return client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy_id,
        },
    )


def _full_proof(client, *, rule=None):
    """Receive, verify (accepted) and decide one proof; return its ids."""
    created, evidence, evidence_id = _submit(client)
    verified = _verify(client, created, evidence, evidence_id)
    assert verified.status_code == 200
    assert verified.json()["status"] == "verified"
    policy = _policy(client, rule)
    decided = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    assert decided.status_code == 200
    return created, evidence, evidence_id, policy, decided.json()


def _query(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        "/v1/compliance/proof-events",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


def _walk(client, *, page_size=None, monkeypatch=None, **params):
    token = None
    rows = []
    for _ in range(50):
        query_params = dict(params)
        if token is not None:
            query_params["cursor"] = token
        if page_size is not None:
            monkeypatch.setattr(app_module, "PROOF_EVENT_PAGE_SIZE", page_size)
        page = _query(client, **query_params)
        assert page.status_code == 200, page.text
        data = page.json()
        rows.extend(data["events"])
        if data["complete"]:
            assert data["next_cursor"] == ""
            return rows
        token = data["next_cursor"]
        assert token
    raise AssertionError("pagination never completed")  # pragma: no cover


# --- request validation ----------------------------------------------------


def test_query_requires_scope_parameters(client):
    assert client.get("/v1/compliance/proof-events").status_code == 422
    assert (
        client.get(
            "/v1/compliance/proof-events", params={"workload_id": WORKLOAD}
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/v1/compliance/proof-events", params={"tenant_id": TENANT}
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"workload_id": "\t"},
        {"evidence_id": ""},
        {"evidence_id": "   "},
        {"evidence_id": "not-a-uuid"},
        {"evidence_id": "abc123"},
        {"evidence_id": ZERO_UUID[:-1] + "Z"},
        {"evidence_id": "  " + ZERO_UUID},
        {"evidence_id": "ABCDEF12-3456-7890-ABCD-EF1234567890"},
        {"event_type": ""},
        {"event_type": "   "},
        {"event_type": "received"},
        {"event_type": "grant"},
        {"event_type": "proof"},
        {"event_type": "PROOF-RECEIVED"},
        {"status": ""},
        {"status": "   "},
        {"status": "RECEIVED"},
        {"status": "pending"},
        {"status": "accepted"},
        {"status": "failed"},
        {"occurred_after": "not-a-timestamp"},
        {"occurred_after": "2026-01-01T00:00:00"},
        {"occurred_before": "2026-01-01"},
        {"occurred_after": ""},
        {"occurred_before": "  "},
        {
            "occurred_after": "2026-01-02T00:00:00Z",
            "occurred_before": "2026-01-01T00:00:00Z",
        },
        {
            "occurred_after": "2026-01-01T01:00:00+01:00",
            "occurred_before": "2026-01-01T00:00:00Z",
        },
        {"cursor": "   "},
        {"cursor": "not base64!!"},
        {"cursor": "AAA="},
        {"bogus": "value"},
        {"limit": "10"},
    ],
)
def test_query_rejects_invalid_parameters(client, params):
    response = _query(client, **params)
    assert response.status_code == 422, response.text


def test_repeated_parameter_is_422(client):
    response = client.get(
        "/v1/compliance/proof-events",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("status", "received"),
            ("status", "verified"),
        ],
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"{}", b"  ", b"not json", b"\x00"])
def test_non_empty_body_is_422(client, body):
    response = client.request(
        "GET",
        "/v1/compliance/proof-events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=body,
    )
    assert response.status_code == 422


def test_query_rejects_non_utc_offset_even_if_instant_matches(client):
    response = _query(
        client,
        occurred_after="2026-01-01T02:00:00+02:00",
        occurred_before="2026-01-01T00:00:00Z",
    )
    assert response.status_code == 422


def test_invalid_parameters_write_no_state(app, client):
    _query(client, event_type="nope", status="PENDING")
    _query(client, evidence_id="not-a-uuid")
    with app.state.session_factory() as session:
        assert session.query(ProofLifecycleEvent).count() == 0


# --- 404 semantics ---------------------------------------------------------


def test_unknown_evidence_identifier_returns_404(client):
    response = _query(client, evidence_id=ZERO_UUID)
    assert response.status_code == 404


def test_cross_scope_evidence_identifier_returns_404(client):
    _full_proof(client)
    evidence_id = _query(client).json()["events"][0]["evidence_id"]
    assert _query(client, tenant=OTHER_TENANT, evidence_id=evidence_id).status_code == 404
    assert _query(client, workload=OTHER_WORKLOAD, evidence_id=evidence_id).status_code == 404


# --- empty scope / shape ---------------------------------------------------


def test_empty_scope_returns_empty_completed_page(client):
    response = _query(client)
    assert response.status_code == 200
    assert response.json() == {"events": [], "next_cursor": "", "complete": True}


def test_response_is_compact_json_with_single_trailing_newline(client):
    _full_proof(client)
    response = _query(client)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == ["events", "next_cursor", "complete"]


def test_event_rows_have_exact_shape(client):
    _full_proof(client)
    rows = _query(client).json()["events"]
    expected_fields = {
        "event_id",
        "evidence_id",
        "event_type",
        "status",
        "evidence_format",
        "policy_version",
        "occurred_at",
    }
    for row in rows:
        assert set(row) == expected_fields
        parsed = datetime.fromisoformat(row["occurred_at"])
        assert parsed.utcoffset() == timedelta(0)
        assert isinstance(row["event_id"], str) and row["event_id"]
        assert isinstance(row["evidence_id"], str) and row["evidence_id"]


def test_stage_fields_are_set_only_for_their_stage(client):
    _full_proof(client)
    rows = {row["event_type"]: row for row in _query(client).json()["events"]}

    received = rows["proof-received"]
    assert received["status"] == "received"
    assert received["evidence_format"] == "attested-nonce-json"
    assert received["policy_version"] is None

    verified = rows["proof-verified"]
    assert verified["status"] == "verified"
    assert verified["evidence_format"] is None
    assert verified["policy_version"] is None

    decision = rows["proof-decision"]
    assert decision["status"] == "allowed"
    assert decision["evidence_format"] is None
    assert decision["policy_version"] == 1


def test_no_floats_or_non_finite_values(client):
    _full_proof(client)
    parsed = json.loads(_query(client).content)
    assert isinstance(parsed["complete"], bool)
    assert isinstance(parsed["next_cursor"], str)

    def _check(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            return
        if isinstance(value, float):  # pragma: no cover - structural guard
            raise AssertionError("float present")
        assert value is None or isinstance(value, str)

    for row in parsed["events"]:
        for value in row.values():
            _check(value)


def test_response_never_contains_protected_material(client):
    created, evidence, _, _, _ = _full_proof(client)
    response = _query(client)
    text = response.text
    assert evidence not in text
    assert created["nonce"] not in text
    assert '"claims"' not in text
    assert '"capability"' not in text
    assert '"payload"' not in text


# --- one event per stage / first-settlement semantics ----------------------


def test_one_event_per_stage_in_business_order(client):
    _full_proof(client)
    rows = _query(client).json()["events"]
    assert [row["event_type"] for row in rows] == [
        "proof-received",
        "proof-verified",
        "proof-decision",
    ]
    keys = [(row["occurred_at"], row["event_id"]) for row in rows]
    assert keys == sorted(keys)
    assert len({row["event_id"] for row in rows}) == 3
    assert len({row["evidence_id"] for row in rows}) == 1


def test_received_event_exists_before_verification(client):
    _submit(client)
    rows = _query(client).json()["events"]
    assert [row["event_type"] for row in rows] == ["proof-received"]
    assert rows[0]["status"] == "received"


def test_verify_retry_creates_no_duplicate_event(client):
    created, evidence, evidence_id = _submit(client)
    first = _verify(client, created, evidence, evidence_id)
    assert first.status_code == 200
    second = _verify(client, created, evidence, evidence_id)
    assert second.status_code == 200
    rows = _query(client).json()["events"]
    assert [row["event_type"] for row in rows] == [
        "proof-received",
        "proof-verified",
    ]


def test_rejected_verification_records_rejected_once(client):
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    bad_evidence = json.dumps(
        {"nonce": created["nonce"], "claims": {}, "mac": "0" * 64}
    )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": bad_evidence,
        },
    )
    evidence_id = submitted.json()["evidence_id"]
    rejected = _verify(client, created, bad_evidence, evidence_id)
    assert rejected.json()["status"] == "rejected"
    # Retrying a settled rejection returns the same conclusion, no new event.
    again = _verify(client, created, bad_evidence, evidence_id)
    assert again.json()["status"] == "rejected"
    rows = _query(client).json()["events"]
    assert [(row["event_type"], row["status"]) for row in rows] == [
        ("proof-received", "received"),
        ("proof-verified", "rejected"),
    ]
    # Rejected evidence cannot drive a decision (409) and gets no decision
    # event.
    policy = _policy(client)
    denied = _decide(client, created, bad_evidence, evidence_id, policy["policy_id"])
    assert denied.status_code == 409
    rows = _query(client).json()["events"]
    assert len(rows) == 2


def test_failed_verification_writes_no_event_and_can_settle_later(
    tmp_path, monkeypatch
):
    # A verifier that always raises: the service answers 500 and rolls the
    # settlement and the verification event back, leaving only reception.
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)

    class ExplodingVerifier(Verifier):
        format_name = "exploding-format"

        def verify(self, context: VerificationContext) -> VerificationResult:
            raise RuntimeError("boom containing secrets never logged")

    registry = VerifierRegistry()
    registry.register(ExplodingVerifier())
    application = create_app(
        f"sqlite:///{tmp_path}/exploding.db",
        verifier_registry=registry,
    )
    exploding = TestClient(application)
    try:
        created, evidence, evidence_id = _submit(
            exploding, evidence_format="exploding-format",
            evidence="for-exploding-verifier",
        )
        failed = _verify(exploding, created, evidence, evidence_id)
        assert failed.status_code == 500
        retried = _verify(exploding, created, evidence, evidence_id)
        assert retried.status_code == 500
        rows = _query(exploding).json()["events"]
        assert [(row["event_type"], row["status"]) for row in rows] == [
            ("proof-received", "received")
        ]
    finally:
        application.state.engine.dispose()


def test_denied_decision_records_denied_with_policy_version(client):
    created, evidence, evidence_id = _submit(client, claims={"m": "other"})
    verified = _verify(client, created, evidence, evidence_id)
    assert verified.json()["status"] == "verified"
    policy = _policy(client, {"claim": "m", "equals": "x"}, name="deny")
    decided = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    assert decided.status_code == 200
    assert decided.json()["status"] == "denied"
    rows = _query(client, event_type="proof-decision").json()["events"]
    assert len(rows) == 1
    assert rows[0]["status"] == "denied"
    assert rows[0]["policy_version"] == 1


def test_repeated_decision_same_policy_creates_no_duplicate_event(client):
    created, evidence, evidence_id, policy, _ = _full_proof(client)
    again = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    assert again.status_code == 200
    rows = _query(client).json()["events"]
    assert len(rows) == 3
    assert [row["event_type"] for row in rows].count("proof-decision") == 1


def test_decision_against_second_policy_records_no_additional_event(client):
    created, evidence, evidence_id, _, _ = _full_proof(client)
    second = _policy(client, {"claim": "m", "equals": "nope"}, name="r2")
    other = _decide(client, created, evidence, evidence_id, second["policy_id"])
    assert other.status_code == 200
    assert other.json()["status"] == "denied"
    rows = _query(client).json()["events"]
    # The first decision's audit event is the only proof-decision event;
    # the second decision is business state but not a new lifecycle stage.
    decisions = [row for row in rows if row["event_type"] == "proof-decision"]
    assert len(decisions) == 1
    assert decisions[0]["status"] == "allowed"
    assert decisions[0]["policy_version"] == 1


def test_multiple_proofs_are_independent_timelines(client):
    _full_proof(client)
    _full_proof(client)
    rows = _query(client).json()["events"]
    assert len(rows) == 6
    by_evidence: dict[str, list[str]] = {}
    for row in rows:
        by_evidence.setdefault(row["evidence_id"], []).append(row["event_type"])
    for stages in by_evidence.values():
        assert stages == [
            "proof-received",
            "proof-verified",
            "proof-decision",
        ]
    keys = [(row["occurred_at"], row["event_id"]) for row in rows]
    assert keys == sorted(keys)


# --- filtering -------------------------------------------------------------


@pytest.mark.parametrize(
    "event_type",
    ["proof-received", "proof-verified", "proof-decision"],
)
def test_filter_by_event_type(client, event_type):
    _full_proof(client)
    rows = _query(client, event_type=event_type).json()["events"]
    assert rows and {row["event_type"] for row in rows} == {event_type}


@pytest.mark.parametrize(
    "status",
    ["received", "verified", "rejected", "allowed", "denied"],
)
def test_filter_by_status(client, status):
    _full_proof(client)  # received, verified, allowed
    created, evidence, evidence_id = _submit(client, claims={"m": "other"})
    _verify(client, created, evidence, evidence_id)
    bad = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    rejected_evidence = json.dumps(
        {"nonce": bad["nonce"], "claims": {}, "mac": "0" * 64}
    )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": bad["challenge_id"],
            "nonce": bad["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": rejected_evidence,
        },
    )
    _verify(client, bad, rejected_evidence, submitted.json()["evidence_id"])
    policy = _policy(client, {"claim": "m", "equals": "x"}, name="deny-rule")
    _decide(client, created, evidence, evidence_id, policy["policy_id"])
    rows = _query(client, status=status).json()["events"]
    assert rows and {row["status"] for row in rows} == {status}


def test_filter_by_evidence_id_returns_only_that_timeline(client):
    _, _, first, _, _ = _full_proof(client)
    _full_proof(client)
    rows = _query(client, evidence_id=first).json()["events"]
    assert len(rows) == 3
    assert {row["evidence_id"] for row in rows} == {first}
    assert [row["event_type"] for row in rows] == [
        "proof-received",
        "proof-verified",
        "proof-decision",
    ]


def test_filter_by_time_window_is_inclusive(client):
    _full_proof(client)
    occurred = _query(client).json()["events"][0]["occurred_at"]
    rows = _query(
        client, occurred_after=occurred, occurred_before=occurred
    ).json()["events"]
    assert len(rows) == 1
    past = (datetime.fromisoformat(occurred) - timedelta(seconds=1)).isoformat()
    future = (datetime.fromisoformat(occurred) + timedelta(seconds=1)).isoformat()
    assert _query(client, occurred_before=past).json()["events"] == []
    assert _query(client, occurred_after=future).json()["events"] == []


def test_events_are_scoped_to_tenant_and_workload(client):
    _full_proof(client)
    rows_a = _query(client).json()["events"]
    rows_b = _query(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    ).json()["events"]
    assert len(rows_a) == 3
    assert rows_b == []


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_event_once_in_order(client, monkeypatch):
    count = 4
    for _ in range(count):
        _full_proof(client)
    rows = _walk(client, page_size=3, monkeypatch=monkeypatch)
    assert len(rows) == count * 3
    keys = [(row["occurred_at"], row["event_id"]) for row in rows]
    assert keys == sorted(keys)
    assert len({row["event_id"] for row in rows}) == count * 3


def test_page_carries_cursor_and_complete_flag(client, monkeypatch):
    monkeypatch.setattr(app_module, "PROOF_EVENT_PAGE_SIZE", 2)
    _full_proof(client)

    first = _query(client).json()
    assert len(first["events"]) == 2
    assert first["complete"] is False
    assert first["next_cursor"]

    second = _query(client, cursor=first["next_cursor"]).json()
    assert len(second["events"]) == 1
    assert second["complete"] is True
    assert second["next_cursor"] == ""
    first_keys = [(r["occurred_at"], r["event_id"]) for r in first["events"]]
    second_keys = [(r["occurred_at"], r["event_id"]) for r in second["events"]]
    assert first_keys < second_keys


def test_empty_string_cursor_equals_default(client):
    _full_proof(client)
    assert _query(client).content == _query(client, cursor="").content


def test_tampered_forged_or_cross_scope_cursor_returns_422(client, monkeypatch):
    monkeypatch.setattr(app_module, "PROOF_EVENT_PAGE_SIZE", 1)
    _full_proof(client)
    cursor = _query(client).json()["next_cursor"]
    assert cursor

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _query(client, cursor=tampered).status_code == 422

    forged = b64url_encode(b'{"k":"compliance-proof-events-v1"}' + b"0" * 32)
    assert _query(client, cursor=forged).status_code == 422

    assert _query(client, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert _query(client, workload=OTHER_WORKLOAD, cursor=cursor).status_code == 422


def test_cursor_cannot_cross_filters_or_kinds(client, monkeypatch):
    monkeypatch.setattr(app_module, "PROOF_EVENT_PAGE_SIZE", 1)
    _full_proof(client)
    cursor = _query(client).json()["next_cursor"]

    assert _query(client, event_type="proof-received", cursor=cursor).status_code == 422
    assert _query(client, status="received", cursor=cursor).status_code == 422
    assert _query(client, evidence_id=ZERO_UUID, cursor=cursor).status_code in (
        404,
        422,
    )
    assert (
        _query(
            client,
            occurred_after="2000-01-01T00:00:00Z",
            cursor=cursor,
        ).status_code
        == 422
    )
    assert (
        _query(
            client,
            occurred_before="2100-01-01T00:00:00Z",
            cursor=cursor,
        ).status_code
        == 422
    )

    # Cursors from every other HMAC family are never accepted.
    from proof_release.app import (
        _encode_audit_event_cursor,
        _encode_cursor,
        _encode_grant_audit_cursor,
        _encode_revocation_cursor,
        _encode_rewrap_job_event_cursor,
    )

    foreign_rewrap = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _query(client, cursor=foreign_rewrap).status_code == 422
    foreign_grant = _encode_grant_audit_cursor(
        TENANT, WORKLOAD, ZERO_UUID,
        grant_id="", decision_id="", data_id="", status="",
        issued_after="", issued_before="",
    )
    assert _query(client, cursor=foreign_grant).status_code == 422
    foreign_audit = _encode_audit_event_cursor(
        TENANT, WORKLOAD, "2026-01-01T00:00:00+00:00", ZERO_UUID,
        event_id="", event_type="", status="",
        occurred_after="", occurred_before="",
    )
    assert _query(client, cursor=foreign_audit).status_code == 422
    foreign_revocation = _encode_revocation_cursor(
        TENANT, WORKLOAD, ZERO_UUID,
        "2026-01-01T00:00:00+00:00", ZERO_UUID,
        revocation_id="", certificate_fingerprint="",
        effective_after="", effective_before="",
        snapshot_at="2026-01-01T00:00:00+00:00", snapshot_id=ZERO_UUID,
    )
    assert _query(client, cursor=foreign_revocation).status_code == 422
    foreign_job_event = _encode_rewrap_job_event_cursor(
        TENANT, WORKLOAD, ZERO_UUID, 1, snapshot_seq=1
    )
    assert _query(client, cursor=foreign_job_event).status_code == 422


# --- fixed snapshot semantics ----------------------------------------------


def test_first_query_fixes_snapshot_excluding_later_commits(client, monkeypatch):
    monkeypatch.setattr(app_module, "PROOF_EVENT_PAGE_SIZE", 2)
    _, _, first_id, _, _ = _full_proof(client)  # three events

    first = _query(client).json()
    assert len(first["events"]) == 2
    cursor = first["next_cursor"]

    # A second proof commits after the snapshot was fixed.
    _full_proof(client)

    # The next page of the fixed snapshot contains only the remaining
    # event of the first proof — never the newly committed events.
    second = _query(client, cursor=cursor).json()
    assert len(second["events"]) == 1
    assert second["complete"] is True
    assert second["next_cursor"] == ""
    assert second["events"][0]["evidence_id"] == first_id

    # Replaying the first cursor returns the byte-identical first page.
    replayed = _query(client, cursor=first["next_cursor"]).json()
    assert replayed == second

    # A fresh cursor-less first query sees the whole new committed set.
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert len(fresh) == 6


def test_replay_first_page_is_byte_identical(client):
    _full_proof(client)
    first = _query(client)
    second = _query(client)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_snapshot_stays_empty_until_fresh_first_query(client):
    empty = _query(client).json()
    assert empty == {"events": [], "next_cursor": "", "complete": True}
    _full_proof(client)
    # The fixed empty snapshot has no cursor to replay; a new commit only
    # appears in a fresh first query.
    fresh = _query(client).json()
    assert len(fresh["events"]) == 3
    assert fresh["complete"] is True


def test_filtered_snapshot_excludes_later_in_filter_commits(client, monkeypatch):
    monkeypatch.setattr(app_module, "PROOF_EVENT_PAGE_SIZE", 1)
    _full_proof(client)
    _full_proof(client)  # two receptions committed before the snapshot

    first = _query(client, event_type="proof-received").json()
    assert len(first["events"]) == 1
    assert first["complete"] is False

    # A third reception commits after the snapshot was fixed.
    _submit(client)

    # The fixed snapshot's next page contains the second reception only
    # and is complete — the later commit never enters this snapshot.
    second = _query(
        client, event_type="proof-received", cursor=first["next_cursor"]
    ).json()
    assert len(second["events"]) == 1
    assert second["complete"] is True
    assert second["next_cursor"] == ""
    assert second["events"][0]["event_id"] != first["events"][0]["event_id"]

    # A fresh cursor-less first query (new snapshot) observes all three.
    fresh_rows = _walk(
        client,
        page_size=1,
        monkeypatch=monkeypatch,
        event_type="proof-received",
    )
    assert len(fresh_rows) == 3


# --- read-only behaviour ---------------------------------------------------


def test_query_writes_no_state(app, client):
    _full_proof(client)
    for _ in range(3):
        assert _query(client).status_code == 200
    with app.state.session_factory() as session:
        rows = session.query(ProofLifecycleEvent).all()
        assert len(rows) == 3


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_page(app, client):
    from sqlalchemy import text

    _full_proof(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE proof_lifecycle_events"))
    assert _query(client).status_code == 500


# --- persistence -----------------------------------------------------------


def test_events_available_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    _, _, evidence_id, policy, decided = _full_proof(client1)
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    data = _query(client2).json()
    assert len(data["events"]) == 3
    by_type = {row["event_type"]: row for row in data["events"]}
    assert by_type["proof-received"]["evidence_id"] == evidence_id
    assert by_type["proof-verified"]["status"] == "verified"
    assert by_type["proof-decision"]["status"] == decided["status"]
    assert by_type["proof-decision"]["policy_version"] == policy["version"]
    keys = [(row["occurred_at"], row["event_id"]) for row in data["events"]]
    assert keys == sorted(keys)
    assert data["complete"] is True
    second.state.engine.dispose()
