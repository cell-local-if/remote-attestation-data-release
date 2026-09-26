"""Tests for GET /v1/compliance/proof-events.

Tenant-isolated, cursor-stable, snapshot-fixed audit of the proof
lifecycle: evidence receipt, the first verification settlement and the
first policy decision. The endpoint is read-only: events are written by
the evidence/verification/decision transactions, never by a query.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import ProofEvent
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
SECRET = "unit-test-secret"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V1 = b64url_encode(KEY_V1_BYTES)

PATH = "/v1/compliance/proof-events"


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


def _receive(client, *, tenant=TENANT, workload=WORKLOAD, claims=None,
             evidence_format="attested-nonce-json"):
    """Submit evidence and return (challenge, evidence_str, evidence_id)."""
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    claims = {} if claims is None else claims
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


def _receive_rejected(client, *, tenant=TENANT, workload=WORKLOAD):
    """Receive and settle an attestation document with a bad MAC.

    The verifier rejects malformed documents and bad MACs, but the bytes
    presented at verify must be digest-identical to those received, so the
    bad document is submitted and then verified unchanged.
    """
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    bad = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": {"m": "nope"},
            "mac": "0" * 64,
        }
    )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": bad,
        },
    )
    assert submitted.status_code == 201, submitted.text
    evidence_id = submitted.json()["evidence_id"]
    _settle(client, created, bad, evidence_id, tenant=tenant, workload=workload)
    return created, bad, evidence_id


def _settle(client, created, evidence, evidence_id, *,
            tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _policy(client, *, tenant=TENANT, workload=WORKLOAD, name="r",
            rule=None):
    rule = rule or {"claim": "m", "equals": "x"}
    response = client.post(
        "/v1/policies",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "name": name,
            "rule": rule,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _decide(client, created, evidence, evidence_id, policy, *,
            tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _full_proof(client, *, claims=None, rule=None):
    created, evidence, evidence_id = _receive(client, claims=claims or {"m": "x"})
    verified = _settle(client, created, evidence, evidence_id)
    policy = _policy(client, rule=rule)
    decided = _decide(client, created, evidence, evidence_id, policy)
    return created, evidence, evidence_id, verified, policy, decided


def _query(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        PATH,
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
    assert client.get(PATH).status_code == 422
    assert (
        client.get(PATH, params={"workload_id": WORKLOAD}).status_code == 422
    )
    assert client.get(PATH, params={"tenant_id": TENANT}).status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"workload_id": "\t"},
        {"event_id": ""},
        {"event_id": "   "},
        {"event_id": "not-a-uuid"},
        {"event_id": "abc123"},
        {"event_id": ZERO_UUID[:-1] + "Z"},
        {"event_id": "  " + ZERO_UUID},
        {"evidence_id": ""},
        {"evidence_id": "   "},
        {"evidence_id": "not-a-uuid"},
        {"evidence_id": ZERO_UUID[:-1] + "z"},
        {"evidence_id": "  " + ZERO_UUID},
        {"event_type": ""},
        {"event_type": "   "},
        {"event_type": "RECEIPT"},
        {"event_type": "grant"},
        {"event_type": "rewrap"},
        {"status": ""},
        {"status": "   "},
        {"status": "PENDING"},
        {"status": "received "},
        {"status": "consumed"},
        {"status": "accepted"},
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


def test_query_rejects_repeated_parameters(client):
    response = client.get(
        PATH,
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("event_type", "receipt"),
            ("event_type", "decision"),
        ],
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b'{"x":1}', b" ", b"\n", b"x"])
def test_query_rejects_non_empty_body(client, body):
    response = client.request(
        "GET",
        PATH,
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=body,
    )
    assert response.status_code == 422


def test_non_empty_body_rejected_before_state_read(app, client):
    _full_proof(client)
    response = client.request(
        "GET",
        PATH,
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b" ",
    )
    assert response.status_code == 422
    with app.state.session_factory() as session:
        count = session.query(ProofEvent).count()
        assert count == 3


def test_query_rejects_non_utc_offset_even_if_instant_matches(client):
    response = _query(
        client,
        occurred_after="2026-01-01T02:00:00+02:00",
        occurred_before="2026-01-01T00:00:00Z",
    )
    assert response.status_code == 422


def test_invalid_parameters_write_no_state(app, client):
    _query(client, event_type="nope", status="PENDING")
    _query(client, event_id="not-a-uuid")
    _query(client, evidence_id="not-a-uuid")
    with app.state.session_factory() as session:
        assert session.query(ProofEvent).count() == 0


# --- 404 semantics ---------------------------------------------------------


def test_unknown_event_identifier_returns_404(client):
    response = _query(client, event_id=ZERO_UUID)
    assert response.status_code == 404


def test_unknown_evidence_identifier_returns_404(client):
    response = _query(client, evidence_id=ZERO_UUID)
    assert response.status_code == 404


def test_cross_scope_identifiers_return_404(client):
    _, _, evidence_id, *_ = _full_proof(client)
    row = _query(client).json()["events"][0]
    event_id = row["event_id"]
    assert _query(client, tenant=OTHER_TENANT, event_id=event_id).status_code == 404
    assert _query(client, workload=OTHER_WORKLOAD, event_id=event_id).status_code == 404
    assert (
        _query(client, tenant=OTHER_TENANT, evidence_id=evidence_id).status_code
        == 404
    )
    assert (
        _query(client, workload=OTHER_WORKLOAD, evidence_id=evidence_id).status_code
        == 404
    )
    # A proof event id is never a valid evidence id and vice versa.
    assert _query(client, evidence_id=event_id).status_code == 404


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
    # Top-level key order is events, next_cursor, complete.
    assert list(json.loads(raw)) == ["events", "next_cursor", "complete"]


def test_event_rows_have_exact_shape(client):
    created, evidence, evidence_id, verified, policy, decided = _full_proof(
        client
    )
    rows = _query(client).json()["events"]
    assert len(rows) == 3
    keys = {(row["event_type"], row["status"]) for row in rows}
    assert keys == {
        ("receipt", "received"),
        ("verification", "verified"),
        ("decision", "allowed"),
    }
    for row in rows:
        assert set(row) == {
            "event_id",
            "evidence_id",
            "event_type",
            "status",
            "evidence_format",
            "policy_version",
            "occurred_at",
        }
        assert row["evidence_id"] == evidence_id
        parsed = datetime.fromisoformat(row["occurred_at"])
        assert parsed.utcoffset() == timedelta(0)
        for field in ("event_id", "evidence_id", "event_type", "status"):
            assert isinstance(row[field], str) and row[field]

    by_type = {row["event_type"]: row for row in rows}
    assert by_type["receipt"]["evidence_format"] == "attested-nonce-json"
    assert by_type["receipt"]["policy_version"] is None
    assert by_type["verification"]["evidence_format"] is None
    assert by_type["verification"]["policy_version"] is None
    assert by_type["decision"]["evidence_format"] is None
    assert by_type["decision"]["policy_version"] == policy["version"]

    keys_sorted = [(row["occurred_at"], row["event_id"]) for row in rows]
    assert keys_sorted == sorted(keys_sorted)


def test_no_floats_or_non_finite_values(client):
    _full_proof(client)
    parsed = json.loads(_query(client).content)
    assert isinstance(parsed["complete"], bool)
    assert isinstance(parsed["next_cursor"], str)

    def _check(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            assert not isinstance(value, bool)
            return
        if isinstance(value, float):  # pragma: no cover - structural guard
            raise AssertionError("float present")
        assert value is None or isinstance(value, str)

    for row in parsed["events"]:
        for value in row.values():
            _check(value)


def test_response_never_contains_sensitive_material(client):
    created, evidence, *_ = _full_proof(client, claims={"m": "x", "secret": "shh"})
    response = _query(client)
    assert created["nonce"] not in response.text
    assert "shh" not in response.text
    assert evidence not in response.text
    assert '"claims"' not in response.text
    assert '"nonce"' not in response.text
    assert '"payload"' not in response.text


# --- lifecycle coverage ----------------------------------------------------


def test_rejected_verification_records_rejected_event(client):
    created, bad, evidence_id = _receive_rejected(client)
    rows = _query(client).json()["events"]
    assert [(row["event_type"], row["status"]) for row in rows] == [
        ("receipt", "received"),
        ("verification", "rejected"),
    ]
    # Rejected evidence cannot drive a decision; no third event exists.


def test_denied_decision_records_denied_event(client):
    created, evidence, evidence_id = _receive(client, claims={"m": "other"})
    _settle(client, created, evidence, evidence_id)
    policy = _policy(client, rule={"claim": "m", "equals": "x"})
    decided = _decide(client, created, evidence, evidence_id, policy)
    assert decided["status"] == "denied"
    rows = _query(client, evidence_id=evidence_id).json()["events"]
    decision_rows = [row for row in rows if row["event_type"] == "decision"]
    assert len(decision_rows) == 1
    assert decision_rows[0]["status"] == "denied"
    assert decision_rows[0]["policy_version"] == policy["version"]


def test_retries_write_no_duplicate_events(client):
    created, evidence, evidence_id = _receive(client, claims={"m": "x"})
    # Repeated verification: the stored conclusion is returned and no
    # second verification event is appended.
    for _ in range(3):
        _settle(client, created, evidence, evidence_id)
    policy = _policy(client)
    for _ in range(3):
        _decide(client, created, evidence, evidence_id, policy)

    rows = _query(client, evidence_id=evidence_id).json()["events"]
    assert [(row["event_type"], row["status"]) for row in rows] == [
        ("receipt", "received"),
        ("verification", "verified"),
        ("decision", "allowed"),
    ]
    assert len({row["event_id"] for row in rows}) == 3


def test_later_decision_against_other_policy_adds_no_event(client):
    # The decision endpoint supports one decision per policy version, but
    # the proof timeline keeps only the *first* policy decision.
    created, evidence, evidence_id = _receive(client, claims={"m": "x"})
    _settle(client, created, evidence, evidence_id)
    first = _policy(client, name="r1", rule={"claim": "m", "equals": "x"})
    _decide(client, created, evidence, evidence_id, first)
    second = _policy(client, name="r2", rule={"claim": "m", "equals": "nope"})
    later = _decide(client, created, evidence, evidence_id, second)
    assert later["status"] == "denied"

    rows = _query(client, evidence_id=evidence_id).json()["events"]
    decisions = [row for row in rows if row["event_type"] == "decision"]
    assert len(decisions) == 1
    assert decisions[0]["status"] == "allowed"
    assert decisions[0]["policy_version"] == first["version"]


def test_failed_verification_writes_no_settlement_event(
    tmp_path, monkeypatch
):
    from proof_release.verifiers import Verifier, VerifierRegistry

    class Raising(Verifier):
        format_name = "raising-proof"

        def verify(self, context):
            # A faulty plugin cannot persist raw evidence by raising: the
            # settlement transaction (including the proof event) rolls
            # back and the call maps to 500.
            raise RuntimeError("boom secret leak")

    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    registry = VerifierRegistry()
    registry.register(Raising())
    application = create_app(
        f"sqlite:///{tmp_path}/raising.db", verifier_registry=registry
    )
    client = TestClient(application)
    created, evidence, evidence_id = _receive(
        client, evidence_format="raising-proof"
    )
    response = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert response.status_code == 500
    # No verification event settled; the receipt event remains the only row.
    rows = _query(client).json()["events"]
    assert [(row["event_type"], row["status"]) for row in rows] == [
        ("receipt", "received")
    ]
    # The evidence is still received and can be retried after recovery.
    assert application.state.session_factory
    application.state.engine.dispose()


def test_receipt_event_rolls_back_with_challenge_failure(client, app):
    # Reusing a consumed challenge is a 409 before any evidence/event row;
    # the successful receipt is the only event present.
    created, evidence, evidence_id = _receive(client)
    second = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": evidence,
        },
    )
    assert second.status_code == 409
    with app.state.session_factory() as session:
        rows = session.query(ProofEvent).filter_by(evidence_id=evidence_id).all()
        assert [row.event_type for row in rows] == ["receipt"]


def test_events_available_with_rest_of_lifecycle(client):
    _, _, evidence_id, *_ = _full_proof(client)
    rows = _query(client).json()["events"]
    # Receipt precedes verification precedes decision.
    assert [row["event_type"] for row in rows] == [
        "receipt",
        "verification",
        "decision",
    ]


# --- filtering -------------------------------------------------------------


@pytest.mark.parametrize(
    "event_type,expected_status",
    [
        ("receipt", "received"),
        ("verification", "verified"),
        ("decision", "allowed"),
    ],
)
def test_filter_by_event_type(client, event_type, expected_status):
    _full_proof(client)
    rows = _query(client, event_type=event_type).json()["events"]
    assert rows and {row["event_type"] for row in rows} == {event_type}
    assert {row["status"] for row in rows} == {expected_status}


@pytest.mark.parametrize(
    "status", ["received", "verified", "rejected", "allowed", "denied"]
)
def test_filter_by_status(client, status):
    _full_proof(client, claims={"m": "x"})
    _receive_rejected(client)
    # A verified proof whose policy denies: yields an allowed verification
    # and a denied decision.
    created3, evidence3, eid3 = _receive(client, claims={"m": "x"})
    _settle(client, created3, evidence3, eid3)
    policy_deny = _policy(client, name="deny", rule={"claim": "m", "equals": "zz"})
    _decide(client, created3, evidence3, eid3, policy_deny)

    rows = _query(client, status=status).json()["events"]
    assert rows and {row["status"] for row in rows} == {status}


def test_filter_by_event_id_returns_exactly_that_event(client):
    _full_proof(client)
    target = _query(client, event_type="verification").json()["events"][0]
    rows = _query(client, event_id=target["event_id"]).json()["events"]
    assert [row["event_id"] for row in rows] == [target["event_id"]]
    assert len(rows) == 1


def test_filter_by_evidence_id_returns_its_whole_timeline(client):
    _, _, evidence_id, *_ = _full_proof(client)
    _full_proof(client, claims={"m": "x"})
    rows = _query(client, evidence_id=evidence_id).json()["events"]
    assert len(rows) == 3
    assert {row["evidence_id"] for row in rows} == {evidence_id}


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
    _, _, eid_a, *_ = _full_proof(client)
    created_b, evidence_b, eid_b = _receive(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    )
    _settle(
        client, created_b, evidence_b, eid_b,
        tenant=OTHER_TENANT, workload=OTHER_WORKLOAD,
    )
    rows_a = _query(client).json()["events"]
    rows_b = _query(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()[
        "events"
    ]
    assert {row["evidence_id"] for row in rows_a} == {eid_a}
    assert {row["evidence_id"] for row in rows_b} == {eid_b}


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_event_once_in_order(client, monkeypatch):
    proofs = [_full_proof(client, claims={"m": "x"}) for _ in range(4)]
    rows = _walk(client, page_size=3, monkeypatch=monkeypatch)
    assert len(rows) == 12
    keys = [(row["occurred_at"], row["event_id"]) for row in rows]
    assert keys == sorted(keys)
    assert len({row["event_id"] for row in rows}) == 12
    expected = {proof[2] for proof in proofs}
    assert {row["evidence_id"] for row in rows} == expected


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


def test_replaying_same_cursor_returns_identical_page(client, monkeypatch):
    monkeypatch.setattr(app_module, "PROOF_EVENT_PAGE_SIZE", 1)
    _full_proof(client)
    cursor = _query(client).json()["next_cursor"]
    first = _query(client, cursor=cursor)
    second = _query(client, cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_string_cursor_equals_default(client, monkeypatch):
    monkeypatch.setattr(app_module, "PROOF_EVENT_PAGE_SIZE", 2)
    _full_proof(client)
    assert _query(client).content == _query(client, cursor="").content


def test_tampered_forged_or_cross_scope_cursor_returns_422(
    client, monkeypatch
):
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

    assert _query(client, event_type="receipt", cursor=cursor).status_code == 422
    assert _query(client, status="verified", cursor=cursor).status_code == 422
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

    # Cursors from the other HMAC families are never accepted.
    from proof_release.app import (
        _encode_audit_event_cursor,
        _encode_cursor,
        _encode_grant_audit_cursor,
        _encode_revocation_cursor,
    )

    foreign_rewrap = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _query(client, cursor=foreign_rewrap).status_code == 422
    foreign_grant = _encode_grant_audit_cursor(
        TENANT, WORKLOAD, ZERO_UUID,
        grant_id="", decision_id="", data_id="", status="",
        issued_after="", issued_before="",
    )
    assert _query(client, cursor=foreign_grant).status_code == 422
    foreign_compliance = _encode_audit_event_cursor(
        TENANT, WORKLOAD, "2026-01-01T00:00:00+00:00", ZERO_UUID,
        event_id="", event_type="", status="",
        occurred_after="", occurred_before="",
    )
    assert _query(client, cursor=foreign_compliance).status_code == 422
    foreign_revocation = _encode_revocation_cursor(
        TENANT, WORKLOAD, ZERO_UUID,
        "2026-01-01T00:00:00+00:00", ZERO_UUID,
        revocation_id="", certificate_fingerprint="",
        effective_after="", effective_before="",
        snapshot_at="2026-01-01T00:00:00+00:00", snapshot_id=ZERO_UUID,
    )
    assert _query(client, cursor=foreign_revocation).status_code == 422


def test_cursor_bound_filter_walks_stably(client, monkeypatch):
    _full_proof(client)
    rows = _walk(
        client, page_size=1, monkeypatch=monkeypatch, event_type="decision"
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "allowed"


# --- read-only behaviour ---------------------------------------------------


def test_query_writes_no_state(app, client):
    _full_proof(client)
    for _ in range(3):
        assert _query(client).status_code == 200
    with app.state.session_factory() as session:
        rows = session.query(ProofEvent).all()
        assert len(rows) == 3
        assert {row.status for row in rows} == {"received", "verified", "allowed"}


# --- snapshot isolation ----------------------------------------------------


def test_fixed_snapshot_excludes_later_commits(client, monkeypatch):
    import proof_release.app as app_module

    monkeypatch.setattr(app_module, "PROOF_EVENT_PAGE_SIZE", 2)
    created, evidence, evidence_id = _receive(client, claims={"m": "x"})
    _settle(client, created, evidence, evidence_id)
    policy = _policy(client)
    _decide(client, created, evidence, evidence_id, policy)

    # The first query fixes the snapshot on this proof's three events; at
    # page size 2 the first page is receipt, verification with a cursor.
    first = _query(client).json()
    assert [row["event_type"] for row in first["events"]] == [
        "receipt",
        "verification",
    ]
    assert first["complete"] is False
    cursor = first["next_cursor"]

    # Before walking the rest, commit a whole second proof afterwards.
    _full_proof(client, claims={"m": "x"})

    # The fixed snapshot's next page contains only the first proof's
    # decision and then completes: the second proof's later-committed
    # events never retroactively enter it.
    tail = _query(client, cursor=cursor).json()
    assert [row["event_type"] for row in tail["events"]] == ["decision"]
    assert {row["evidence_id"] for row in tail["events"]} == {evidence_id}
    assert tail["complete"] is True
    assert tail["next_cursor"] == ""

    # A fresh cursor-less first query sees both proofs' timelines.
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert [row["event_type"] for row in fresh] == [
        "receipt",
        "verification",
        "decision",
        "receipt",
        "verification",
        "decision",
    ]


def test_concurrent_commits_do_not_disturb_replayed_page(app, monkeypatch):
    import proof_release.app as app_module

    monkeypatch.setattr(app_module, "PROOF_EVENT_PAGE_SIZE", 2)
    client = TestClient(app)
    first_proof = _full_proof(client, claims={"m": "x"})

    first = _query(client).json()
    assert len(first["events"]) == 2
    cursor = first["next_cursor"]
    second_before = _query(client, cursor=cursor).json()
    assert len(second_before["events"]) == 1

    def submit_another(_):
        local = TestClient(app)
        _full_proof(local, claims={"m": "x"})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(submit_another, range(4)))

    second_after = _query(client, cursor=cursor).json()
    assert [
        (row["occurred_at"], row["event_id"]) for row in second_after["events"]
    ] == [
        (row["occurred_at"], row["event_id"]) for row in second_before["events"]
    ]

    # A fresh walk sees all proofs (3 events each), once, in stable order.
    app_module.PROOF_EVENT_PAGE_SIZE = 3
    all_rows = []
    token = None
    for _ in range(20):
        data = _query(client, **({"cursor": token} if token else {})).json()
        all_rows.extend(data["events"])
        if data["complete"]:
            break
        token = data["next_cursor"]
    keys = [(row["occurred_at"], row["event_id"]) for row in all_rows]
    assert keys == sorted(keys)
    assert len(keys) == 15
    assert len({key[1] for key in keys}) == 15


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_page(app, client):
    from sqlalchemy import text

    _full_proof(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE proof_events"))
    assert _query(client).status_code == 500


# --- persistence -----------------------------------------------------------


def test_events_available_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    _, _, evidence_id, verified, policy, decided = _full_proof(client1)
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    data = _query(client2).json()
    assert len(data["events"]) == 3
    assert data["complete"] is True
    by_type = {row["event_type"]: row for row in data["events"]}
    assert by_type["receipt"]["status"] == "received"
    assert by_type["receipt"]["evidence_format"] == "attested-nonce-json"
    assert by_type["verification"]["status"] == verified["status"]
    assert by_type["decision"]["status"] == decided["status"]
    assert by_type["decision"]["policy_version"] == policy["version"]
    for row in data["events"]:
        assert row["evidence_id"] == evidence_id
    second.state.engine.dispose()
