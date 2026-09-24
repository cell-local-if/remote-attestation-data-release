"""Tests for GET /v1/compliance/audit-events.

Tenant-isolated, cursor-stable compliance audit of grant lifecycle and
rewrap events. The endpoint is read-only: events are written by the
grant/consume/revoke/release/rewrap transactions, never by a query.
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
from proof_release.db import AuditEvent
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
SECRET = "unit-test-secret"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V2_BYTES = b"fedcba9876543210fedcba9876543210"
KEY_V1 = b64url_encode(KEY_V1_BYTES)
KEY_V2 = b64url_encode(KEY_V2_BYTES)


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = create_app(f"sqlite:///{tmp_path}/audit_events.db")
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


def _decision(client, *, tenant=TENANT, workload=WORKLOAD):
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    claims = {"m": "x"}
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
            "evidence_format": "attested-nonce-json",
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201
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
    assert verified.status_code == 200
    policy = client.post(
        "/v1/policies",
        json={"tenant_id": tenant, "workload_id": workload, "name": "r",
              "rule": {"claim": "m", "equals": "x"}},
    ).json()
    decided = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert decided.status_code == 200
    return decided.json()


def _mint(client, decision_id, *, data_id="data-1", tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/release-grants",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "decision_id": decision_id,
            "data_id": data_id,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _create_envelope(client, data_id, *, tenant=TENANT, workload=WORKLOAD,
                     payload="secret-payload"):
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


def _rewrap_batch(client, *, tenant=TENANT, workload=WORKLOAD, **extra):
    return client.post(
        "/v1/rewrap-batches",
        json={"tenant_id": tenant, "workload_id": workload, **extra},
    )


def _query(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        "/v1/compliance/audit-events",
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
            monkeypatch.setattr(app_module, "AUDIT_EVENT_PAGE_SIZE", page_size)
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
    assert client.get("/v1/compliance/audit-events").status_code == 422
    assert (
        client.get(
            "/v1/compliance/audit-events", params={"workload_id": WORKLOAD}
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/v1/compliance/audit-events", params={"tenant_id": TENANT}
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
        {"event_id": ""},
        {"event_id": "   "},
        {"event_id": "not-a-uuid"},
        {"event_id": "abc123"},
        {"event_id": ZERO_UUID[:-1] + "Z"},
        {"event_id": "  " + ZERO_UUID},
        {"event_type": ""},
        {"event_type": "   "},
        {"event_type": "GRANT"},
        {"event_type": "consume"},
        {"event_type": "decrypt"},
        {"status": ""},
        {"status": "   "},
        {"status": "PENDING"},
        {"status": "allowed"},
        {"status": "expired"},
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
    with app.state.session_factory() as session:
        assert session.query(AuditEvent).count() == 0


# --- 404 semantics ---------------------------------------------------------


def test_unknown_event_identifier_returns_404(client):
    response = _query(client, event_id=ZERO_UUID)
    assert response.status_code == 404


def test_cross_scope_event_identifier_returns_404(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
    event_id = _query(client).json()["events"][0]["event_id"]
    assert _query(client, tenant=OTHER_TENANT, event_id=event_id).status_code == 404
    assert _query(client, workload=OTHER_WORKLOAD, event_id=event_id).status_code == 404
    # The grant id space is a different identifier space and never matches.
    assert _query(client, event_id=grant["grant_id"]).status_code == 404


# --- empty scope / shape ---------------------------------------------------


def test_empty_scope_returns_empty_completed_page(client):
    response = _query(client)
    assert response.status_code == 200
    assert response.json() == {"events": [], "next_cursor": "", "complete": True}


def test_response_is_compact_json_with_single_trailing_newline(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    response = _query(client)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    # Top-level key order is events, next_cursor, complete.
    assert list(json.loads(raw)) == ["events", "next_cursor", "complete"]


def test_grant_event_has_exact_shape(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="data-7")
    row = _query(client).json()["events"][0]
    assert set(row) == {
        "event_id",
        "event_type",
        "grant_id",
        "decision_id",
        "data_id",
        "status",
        "occurred_at",
        "capability_sha256",
    }
    assert row["event_type"] == "grant"
    assert row["grant_id"] == grant["grant_id"]
    assert row["decision_id"] == decision["decision_id"]
    assert row["data_id"] == "data-7"
    assert row["status"] == "pending"
    assert row["capability_sha256"] == hashlib.sha256(
        grant["capability"].encode("ascii")
    ).hexdigest()
    parsed = datetime.fromisoformat(row["occurred_at"])
    assert parsed.utcoffset() == timedelta(0)
    for field in (
        "event_id",
        "event_type",
        "grant_id",
        "decision_id",
        "data_id",
        "status",
        "occurred_at",
        "capability_sha256",
    ):
        assert isinstance(row[field], str) and row[field]


def test_no_floats_or_non_finite_values(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
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


def test_audit_never_contains_plaintext_capability(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
    response = _query(client)
    assert grant["capability"] not in response.text


# --- lifecycle events ------------------------------------------------------


def test_consume_revoke_produce_lifecycle_events(client):
    decision = _decision(client)
    consumed = _mint(client, decision["decision_id"], data_id="c")
    revoked = _mint(client, decision["decision_id"], data_id="r")

    cr = client.post(
        f"/v1/release-grants/{consumed['grant_id']}/consume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD,
              "capability": consumed["capability"]},
    )
    assert cr.status_code == 200
    rr = client.post(
        f"/v1/release-grants/{revoked['grant_id']}/revoke",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD,
              "capability": revoked["capability"]},
    )
    assert rr.status_code == 200

    rows = _query(client).json()["events"]
    by_grant: dict[str, list[str]] = {}
    for row in rows:
        assert row["event_type"] == "grant"
        by_grant.setdefault(row["grant_id"], []).append(row["status"])
    assert by_grant[consumed["grant_id"]] == ["pending", "consumed"]
    assert by_grant[revoked["grant_id"]] == ["pending", "revoked"]


def test_settlement_event_identifiers_and_digest_match_grant(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="d")
    client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD,
              "capability": grant["capability"]},
    )
    rows = _query(client, status="consumed").json()["events"]
    assert len(rows) == 1
    row = rows[0]
    assert row["grant_id"] == grant["grant_id"]
    assert row["decision_id"] == decision["decision_id"]
    assert row["data_id"] == "d"
    assert row["capability_sha256"] == hashlib.sha256(
        grant["capability"].encode("ascii")
    ).hexdigest()


def test_failed_consume_writes_no_event(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
    client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD,
              "capability": "A" * 43},
    )
    rows = _query(client).json()["events"]
    assert [row["status"] for row in rows] == ["pending"]


# --- rewrap events ---------------------------------------------------------


def test_rewrap_batch_writes_rewrapped_and_skipped_events(
    client, monkeypatch
):
    _create_envelope(client, "e1")
    _create_envelope(client, "e2")
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(2, {1: KEY_V1, 2: KEY_V2})
    )
    first = _rewrap_batch(client, limit=1)
    assert first.status_code == 200, first.text
    # First run rewraps e1; a replay skips it (already current), then
    # rewraps e2 — exercising both rewrap statuses.
    second = _rewrap_batch(client, cursor=first.json()["next_cursor"], limit=2)
    assert second.status_code == 200, second.text

    rows = _query(client, event_type="rewrap").json()["events"]
    assert len(rows) == 2
    for row in rows:
        assert row["event_type"] == "rewrap"
        assert row["grant_id"] is None
        assert row["decision_id"] is None
        assert row["capability_sha256"] is None
        assert row["status"] in ("rewrapped", "skipped")
        assert isinstance(row["data_id"], str) and row["data_id"]
        parsed = datetime.fromisoformat(row["occurred_at"])
        assert parsed.utcoffset() == timedelta(0)
    assert {row["data_id"]: row["status"] for row in rows} == {
        "e1": "rewrapped",
        "e2": "rewrapped",
    }

    # Replaying the first batch cursor skips the already-current envelope
    # and records a skipped event.
    third = _rewrap_batch(client, cursor=first.json()["next_cursor"], limit=1)
    assert third.status_code == 200
    rows = _query(client, event_type="rewrap", status="skipped").json()["events"]
    assert len(rows) == 1
    assert rows[0]["data_id"] == "e2"


def test_single_envelope_rewrap_writes_rewrapped_event(client, monkeypatch):
    _create_envelope(client, "solo")
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(2, {1: KEY_V1, 2: KEY_V2})
    )
    response = client.post(
        "/v1/data-envelopes/solo/rewrap",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    # A no-op retry at the current version records no new event.
    assert (
        client.post(
            "/v1/data-envelopes/solo/rewrap",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).status_code
        == 200
    )
    rows = _query(client, event_type="rewrap").json()["events"]
    assert len(rows) == 1
    assert rows[0]["data_id"] == "solo"
    assert rows[0]["status"] == "rewrapped"


# --- ordering --------------------------------------------------------------


def test_events_ordered_by_occurred_at_then_event_id(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
    client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD,
              "capability": grant["capability"]},
    )
    rows = _query(client).json()["events"]
    keys = [(row["occurred_at"], row["event_id"]) for row in rows]
    assert keys == sorted(keys)
    assert [row["status"] for row in rows] == ["pending", "consumed"]


def test_mixed_event_types_are_ordered_together(client, monkeypatch):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="d0")
    _create_envelope(client, "e0")
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(2, {1: KEY_V1, 2: KEY_V2})
    )
    _rewrap_batch(client)
    rows = _query(client).json()["events"]
    keys = [(row["occurred_at"], row["event_id"]) for row in rows]
    assert keys == sorted(keys)
    assert {row["event_type"] for row in rows} == {"grant", "rewrap"}


# --- filtering -------------------------------------------------------------


@pytest.mark.parametrize("event_type", ["grant", "rewrap"])
def test_filter_by_event_type(client, monkeypatch, event_type):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    _create_envelope(client, "e")
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(2, {1: KEY_V1, 2: KEY_V2})
    )
    _rewrap_batch(client)
    rows = _query(client, event_type=event_type).json()["events"]
    assert rows and {row["event_type"] for row in rows} == {event_type}


@pytest.mark.parametrize(
    "status", ["pending", "consumed", "revoked", "rewrapped", "skipped"]
)
def test_filter_by_status(client, monkeypatch, status):
    decision = _decision(client)
    consumed = _mint(client, decision["decision_id"], data_id="c")
    revoked = _mint(client, decision["decision_id"], data_id="r")
    _mint(client, decision["decision_id"], data_id="p")
    client.post(
        f"/v1/release-grants/{consumed['grant_id']}/consume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD,
              "capability": consumed["capability"]},
    )
    client.post(
        f"/v1/release-grants/{revoked['grant_id']}/revoke",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD,
              "capability": revoked["capability"]},
    )
    _create_envelope(client, "e")
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(2, {1: KEY_V1, 2: KEY_V2})
    )
    _rewrap_batch(client)
    _rewrap_batch(client)  # second run: envelope already current -> skipped
    rows = _query(client, status=status).json()["events"]
    assert rows and {row["status"] for row in rows} == {status}


def test_filter_by_event_id_returns_exactly_that_event(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    target = _query(client).json()["events"][0]["event_id"]
    rows = _query(client, event_id=target).json()["events"]
    assert [row["event_id"] for row in rows] == [target]
    assert len(rows) == 1


def test_filter_by_time_window_is_inclusive(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
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
    decision_a = _decision(client)
    _mint(client, decision_a["decision_id"], data_id="a")
    decision_b = _decision(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    _mint(client, decision_b["decision_id"], data_id="b",
          tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    rows_a = _query(client).json()["events"]
    rows_b = _query(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()[
        "events"
    ]
    assert len(rows_a) == 1 and rows_a[0]["data_id"] == "a"
    assert len(rows_b) == 1 and rows_b[0]["data_id"] == "b"


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_event_once_in_order(client, monkeypatch):
    decision = _decision(client)
    grants = [
        _mint(client, decision["decision_id"], data_id=f"d{i:03d}")
        for i in range(7)
    ]
    rows = _walk(client, page_size=3, monkeypatch=monkeypatch)
    # Seven pending events, each with a unique id, walked in stable order.
    assert len(rows) == 7
    keys = [(row["occurred_at"], row["event_id"]) for row in rows]
    assert keys == sorted(keys)
    assert len({row["event_id"] for row in rows}) == 7
    minted_grants = {grant["grant_id"] for grant in grants}
    assert {row["grant_id"] for row in rows} == minted_grants


def test_page_carries_cursor_and_complete_flag(client, monkeypatch):
    monkeypatch.setattr(app_module, "AUDIT_EVENT_PAGE_SIZE", 2)
    decision = _decision(client)
    for i in range(5):
        _mint(client, decision["decision_id"], data_id=f"d{i}")

    first = _query(client).json()
    assert len(first["events"]) == 2
    assert first["complete"] is False
    assert first["next_cursor"]

    second = _query(client, cursor=first["next_cursor"]).json()
    assert len(second["events"]) == 2
    assert second["complete"] is False
    first_keys = [(r["occurred_at"], r["event_id"]) for r in first["events"]]
    second_keys = [(r["occurred_at"], r["event_id"]) for r in second["events"]]
    assert first_keys < second_keys

    last = _query(client, cursor=second["next_cursor"]).json()
    assert len(last["events"]) == 1
    assert last["complete"] is True
    assert last["next_cursor"] == ""


def test_replaying_same_cursor_returns_identical_page(client, monkeypatch):
    monkeypatch.setattr(app_module, "AUDIT_EVENT_PAGE_SIZE", 2)
    decision = _decision(client)
    for i in range(5):
        _mint(client, decision["decision_id"], data_id=f"d{i}")
    cursor = _query(client).json()["next_cursor"]
    first = _query(client, cursor=cursor)
    second = _query(client, cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_string_cursor_equals_default(client, monkeypatch):
    monkeypatch.setattr(app_module, "AUDIT_EVENT_PAGE_SIZE", 2)
    decision = _decision(client)
    for i in range(3):
        _mint(client, decision["decision_id"], data_id=f"d{i}")
    assert _query(client).content == _query(client, cursor="").content


def test_tampered_forged_or_cross_scope_cursor_returns_422(
    client, monkeypatch
):
    monkeypatch.setattr(app_module, "AUDIT_EVENT_PAGE_SIZE", 1)
    decision = _decision(client)
    for i in range(3):
        _mint(client, decision["decision_id"], data_id=f"d{i}")
    cursor = _query(client).json()["next_cursor"]
    assert cursor

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _query(client, cursor=tampered).status_code == 422

    forged = b64url_encode(b'{"k":"compliance-audit-events-v1"}' + b"0" * 32)
    assert _query(client, cursor=forged).status_code == 422

    assert _query(client, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert _query(client, workload=OTHER_WORKLOAD, cursor=cursor).status_code == 422


def test_cursor_cannot_cross_filters_or_kinds(client, monkeypatch):
    monkeypatch.setattr(app_module, "AUDIT_EVENT_PAGE_SIZE", 1)
    decision = _decision(client)
    for i in range(3):
        _mint(client, decision["decision_id"], data_id=f"d{i}")
    cursor = _query(client).json()["next_cursor"]

    assert _query(client, event_type="grant", cursor=cursor).status_code == 422
    assert _query(client, status="pending", cursor=cursor).status_code == 422
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

    # Cursors from the other two HMAC families are never accepted.
    from proof_release.app import _encode_cursor, _encode_grant_audit_cursor

    foreign_rewrap = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _query(client, cursor=foreign_rewrap).status_code == 422
    foreign_grant = _encode_grant_audit_cursor(
        TENANT, WORKLOAD, ZERO_UUID,
        grant_id="", decision_id="", data_id="", status="",
        issued_after="", issued_before="",
    )
    assert _query(client, cursor=foreign_grant).status_code == 422


def test_cursor_bound_filter_walks_stably(client, monkeypatch):
    monkeypatch.setattr(app_module, "AUDIT_EVENT_PAGE_SIZE", 2)
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="d")
    client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD,
              "capability": grant["capability"]},
    )
    rows = _walk(
        client, page_size=1, monkeypatch=monkeypatch, event_type="grant",
        status="pending",
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"


# --- read-only behaviour ---------------------------------------------------


def test_query_writes_no_state(app, client):
    decision = _decision(client)
    grants = [_mint(client, decision["decision_id"], data_id=f"d{i}") for i in range(3)]
    for _ in range(3):
        assert _query(client).status_code == 200
    with app.state.session_factory() as session:
        # Exactly the three pending issuance events; no more.
        rows = session.query(AuditEvent).all()
        assert len(rows) == 3
        assert {row.status for row in rows} == {"pending"}
    assert len(grants) == 3


# --- concurrency -----------------------------------------------------------


def test_concurrent_settlement_then_replay_remains_stable(app):
    client = TestClient(app)
    decision = _decision(client)
    grants = [_mint(client, decision["decision_id"], data_id=f"d{i}") for i in range(6)]

    import proof_release.app as app_module

    app_module.AUDIT_EVENT_PAGE_SIZE = 3
    try:
        first = _query(client).json()
        assert first["complete"] is False
        cursor = first["next_cursor"]
        second_before = _query(client, cursor=cursor).json()
        second_keys = [
            (row["occurred_at"], row["event_id"]) for row in second_before["events"]
        ]
        assert len(second_keys) == 3

        def settle(index):
            local = TestClient(app)
            grant = grants[index]
            path = (
                f"/v1/release-grants/{grant['grant_id']}/"
                + ("consume" if index % 2 == 0 else "revoke")
            )
            response = local.post(
                path,
                json={"tenant_id": TENANT, "workload_id": WORKLOAD,
                      "capability": grant["capability"]},
            )
            assert response.status_code == 200

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(settle, range(6)))

        # Replaying the cursor returns the identical page of pre-existing
        # events; newly committed settlement events appear only on later
        # pages, never duplicating or shifting the replayed page.
        second_after = _query(client, cursor=cursor).json()
        assert (
            [(row["occurred_at"], row["event_id"]) for row in second_after["events"]]
            == second_keys
        )

        # A full walk sees every event once in stable order.
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
        assert len(keys) == 12  # 6 pending + 6 settlement events
        assert len({key[1] for key in keys}) == 12
    finally:
        app_module.AUDIT_EVENT_PAGE_SIZE = 100


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_page(app, client):
    from sqlalchemy import text

    decision = _decision(client)
    _mint(client, decision["decision_id"])
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE audit_events"))
    assert _query(client).status_code == 500


# --- persistence -----------------------------------------------------------


def test_events_available_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    decision = _decision(client1)
    grant = _mint(client1, decision["decision_id"], data_id="persist")
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    data = _query(client2).json()
    assert len(data["events"]) == 1
    row = data["events"][0]
    assert row["grant_id"] == grant["grant_id"]
    assert row["status"] == "pending"
    assert data["complete"] is True
    second.state.engine.dispose()
