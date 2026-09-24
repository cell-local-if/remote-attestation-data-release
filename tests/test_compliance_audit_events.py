"""Tests for GET /v1/compliance/audit-events.

Tenant-isolated, read-only compliance evidence query. It merges two
audit families: release-grant lifecycle events (pending at issuance and
exactly one consumed/revoked settlement) and per-envelope rewrap batch
events (rewrapped/skipped). Ordering is (occurred_at, event_id)
ascending with an HMAC-authenticated keyset cursor bound to scope and
every active filter.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

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

KEY_V1 = b64url_encode(b"0123456789abcdef0123456789abcdef")
KEY_V2 = b64url_encode(b"fedcba9876543210fedcba9876543210")


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
    assert (
        client.post(
            f"/v1/evidence/{evidence_id}/verify",
            json={
                "tenant_id": tenant,
                "workload_id": workload,
                "nonce": created["nonce"],
                "evidence": evidence,
            },
        ).status_code
        == 200
    )
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


def _create_envelope(client, data_id, *, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": f"secret-{data_id}",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _run_batch(client, *, tenant=TENANT, workload=WORKLOAD, limit=50, cursor=None):
    body = {"tenant_id": tenant, "workload_id": workload, "limit": limit}
    if cursor is not None:
        body["cursor"] = cursor
    response = client.post("/v1/rewrap-batches", json=body)
    assert response.status_code == 200, response.text
    return response.json()


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
            monkeypatch.setattr(
                app_module, "COMPLIANCE_AUDIT_PAGE_SIZE", page_size
            )
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


def _seq_id(i: int) -> str:
    """Canonical UUID with an 8-hex-digit sortable prefix."""
    return f"{i:08x}-0000-0000-0000-000000000000"


def _insert_event(
    app,
    *,
    event_id,
    occurred_at,
    event_type="grant",
    status="pending",
    tenant=TENANT,
    workload=WORKLOAD,
    grant_id=None,
    decision_id=None,
    data_id="data-x",
    capability_sha256=None,
):
    with app.state.session_factory() as session:
        session.add(
            AuditEvent(
                event_id=event_id,
                tenant_id=tenant,
                workload_id=workload,
                event_type=event_type,
                grant_id=grant_id,
                decision_id=decision_id,
                data_id=data_id,
                status=status,
                capability_sha256=capability_sha256,
                occurred_at=occurred_at,
            )
        )
        session.commit()


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
        {"event_type": "decision"},
        {"event_type": "GRANT"},
        {"status": ""},
        {"status": "   "},
        {"status": "PENDING"},
        {"status": "allowed"},
        {"status": "keyring"},
        {"status": "missing-key"},
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
    ],
)
def test_query_rejects_invalid_parameters(client, params):
    assert _query(client, **params).status_code == 422


def test_query_rejects_unknown_parameter(client):
    assert _query(client, grant_id=ZERO_UUID).status_code == 422
    assert _query(client, limit="10").status_code == 422
    assert _query(client, issued_after="2026-01-01T00:00:00Z").status_code == 422


def test_query_rejects_repeated_parameters(client):
    response = client.get(
        "/v1/compliance/audit-events",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("status", "pending"),
            ("status", "consumed"),
        ],
    )
    assert response.status_code == 422
    response = client.get(
        "/v1/compliance/audit-events",
        params=[
            ("tenant_id", TENANT),
            ("tenant_id", OTHER_TENANT),
            ("workload_id", WORKLOAD),
        ],
    )
    assert response.status_code == 422


def test_query_rejects_non_utc_offset_even_if_instant_matches(client):
    response = _query(
        client,
        occurred_after="2026-01-01T02:00:00+02:00",
        occurred_before="2026-01-01T00:00:00Z",
    )
    assert response.status_code == 422


# --- unknown / cross-scope identifiers -------------------------------------


def test_unknown_event_identifier_returns_404(client):
    assert _query(client, event_id=ZERO_UUID).status_code == 404


def test_cross_scope_event_identifier_returns_404(client, app):
    _insert_event(
        app,
        event_id="11111111-1111-1111-1111-111111111111",
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert (
        _query(client, tenant=OTHER_TENANT,
               event_id="11111111-1111-1111-1111-111111111111").status_code
        == 404
    )
    assert (
        _query(client, workload=OTHER_WORKLOAD,
               event_id="11111111-1111-1111-1111-111111111111").status_code
        == 404
    )


# --- happy path / shape ----------------------------------------------------


def test_empty_scope_returns_empty_completed_page(client):
    response = _query(client)
    assert response.status_code == 200
    assert response.json() == {"events": [], "next_cursor": "", "complete": True}


def test_response_is_compact_json_with_trailing_newline_and_key_order(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    response = _query(client)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[:-1].count(b"\n") == 0
    assert b", " not in raw and b": " not in raw
    # Top-level key order: events, next_cursor, complete.
    assert list(json.loads(raw)) == ["events", "next_cursor", "complete"]
    parsed = json.loads(raw)
    assert isinstance(parsed["complete"], bool)
    assert isinstance(parsed["next_cursor"], str)
    assert isinstance(parsed["events"], list)

    def _check_types(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            assert not isinstance(value, bool)
            return
        if isinstance(value, float):  # pragma: no cover - structural guard
            raise AssertionError("float present in audit response")
        assert value is None or isinstance(value, str)

    for row in parsed["events"]:
        assert list(row) == [
            "event_id",
            "event_type",
            "grant_id",
            "decision_id",
            "data_id",
            "status",
            "occurred_at",
            "capability_sha256",
        ]
        for value in row.values():
            _check_types(value)


def test_grant_pending_event_shape(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="data-7")
    data = _query(client).json()
    assert data["complete"] is True
    assert data["next_cursor"] == ""
    assert len(data["events"]) == 1
    event = data["events"][0]
    assert event["event_type"] == "grant"
    assert event["grant_id"] == grant["grant_id"]
    assert event["decision_id"] == decision["decision_id"]
    assert event["data_id"] == "data-7"
    assert event["status"] == "pending"
    assert event["capability_sha256"] == hashlib.sha256(
        grant["capability"].encode("ascii")
    ).hexdigest()
    assert len(event["capability_sha256"]) == 64
    stamp = datetime.fromisoformat(event["occurred_at"])
    assert stamp.utcoffset() == timedelta(0)
    assert stamp == datetime.fromisoformat(grant["issued_at"])


def test_consume_and_revoke_each_add_one_settlement_event(client):
    decision = _decision(client)
    consumed = _mint(client, decision["decision_id"], data_id="c")
    revoked = _mint(client, decision["decision_id"], data_id="r")
    pending = _mint(client, decision["decision_id"], data_id="p")

    assert (
        client.post(
            f"/v1/release-grants/{consumed['grant_id']}/consume",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "capability": consumed["capability"],
            },
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/v1/release-grants/{revoked['grant_id']}/revoke",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "capability": revoked["capability"],
            },
        ).status_code
        == 200
    )

    events = _query(client).json()["events"]
    # One pending per grant plus two settlement events.
    assert len(events) == 5
    by_grant: dict[str, list[dict]] = {}
    for event in events:
        by_grant.setdefault(event["grant_id"], []).append(event)
    statuses_by_grant = {
        grant_id: [e["status"] for e in evs]
        for grant_id, evs in by_grant.items()
    }
    assert statuses_by_grant[consumed["grant_id"]] == [
        "pending",
        "consumed",
    ]
    assert statuses_by_grant[revoked["grant_id"]] == [
        "pending",
        "revoked",
    ]
    assert statuses_by_grant[pending["grant_id"]] == ["pending"]
    for event in events:
        assert event["event_type"] == "grant"
        assert event["capability_sha256"] is not None
        assert event["decision_id"] == decision["decision_id"]


def test_payload_release_records_consumed_event(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="payload-item")
    _create_envelope(client, "payload-item")
    response = client.post(
        f"/v1/release/{grant['grant_id']}",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": "payload-item",
            "capability": grant["capability"],
        },
    )
    assert response.status_code == 200
    statuses = [
        e["status"]
        for e in _query(client).json()["events"]
        if e["grant_id"] == grant["grant_id"]
    ]
    assert statuses == ["pending", "consumed"]


def test_rewrap_events_have_empty_grant_linkage_and_digest(client, monkeypatch):
    for data_id in ("a", "b"):
        _create_envelope(client, data_id)
    # Envelopes at the current version are skipped; no rotation needed to
    # exercise rewrap event emission.
    _run_batch(client)

    events = _query(client, event_type="rewrap").json()["events"]
    assert {e["data_id"] for e in events} == {"a", "b"}
    assert len(events) == 2
    for event in events:
        assert event["event_type"] == "rewrap"
        assert event["status"] == "skipped"
        assert event["grant_id"] is None
        assert event["decision_id"] is None
        assert event["capability_sha256"] is None
        assert event["data_id"]
        assert datetime.fromisoformat(
            event["occurred_at"]
        ).utcoffset() == timedelta(0)


def test_rewrap_batch_records_rewrapped_and_skipped_events(client, monkeypatch):
    _create_envelope(client, "old")
    _create_envelope(client, "also-old")
    # Rotate the keyring to v2; each batch below starts from the beginning
    # of the scope (no cursor), so already-current envelopes are recorded
    # as skips on the later passes.
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(2, {1: KEY_V1, 2: KEY_V2})
    )
    first = _run_batch(client, limit=1)
    assert first["rewrapped"] == 1
    second = _run_batch(client, limit=5)
    # One envelope now current (skipped), one still rewrapped.
    assert second["rewrapped"] == 1 and second["skipped"] == 1
    third = _run_batch(client, limit=5)
    assert third["processed"] == 2 and third["skipped"] == 2

    events = _query(client, event_type="rewrap").json()["events"]
    assert [e["status"] for e in events].count("rewrapped") == 2
    assert [e["status"] for e in events].count("skipped") == 3
    assert {e["data_id"] for e in events} == {"old", "also-old"}


def test_audit_never_contains_plaintext_capability_or_material(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
    _create_envelope(client, grant["data_id"])
    _run_batch(client)
    response = _query(client)
    assert grant["capability"] not in response.text
    envelope = client.get(
        f"/v1/data-envelopes/{grant['data_id']}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    for secret_field in ("wrapped_key", "ciphertext", "iv", "tag"):
        assert envelope[secret_field] not in response.text


def test_events_are_ordered_by_occurred_at_then_event_id(app):
    client = TestClient(app)
    t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
    # Insert out of order, with three events sharing t2, to exercise the
    # event_id tiebreaker explicitly.
    _insert_event(app, event_id="33333333-3333-3333-3333-333333333333",
                  occurred_at=t2)
    _insert_event(app, event_id="11111111-1111-1111-1111-111111111111",
                  occurred_at=t1)
    _insert_event(app, event_id="22222222-2222-2222-2222-222222222222",
                  occurred_at=t2)
    _insert_event(app, event_id="44444444-4444-4444-4444-444444444444",
                  occurred_at=t2, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)

    rows = _query(client).json()["events"]
    assert [e["event_id"] for e in rows] == [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
        "33333333-3333-3333-3333-333333333333",
    ]


# --- filtering -------------------------------------------------------------


def test_filter_by_event_id_returns_exactly_that_event(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="a")
    _mint(client, decision["decision_id"], data_id="b")
    events = _query(client).json()["events"]
    target = events[0]["event_id"]
    page = _query(client, event_id=target).json()
    assert [e["event_id"] for e in page["events"]] == [target]
    assert page["complete"] is True


def test_filter_by_event_type(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="a")
    _create_envelope(client, "a")
    _run_batch(client)
    grants = _query(client, event_type="grant").json()["events"]
    assert grants and all(e["event_type"] == "grant" for e in grants)
    rewraps = _query(client, event_type="rewrap").json()["events"]
    assert rewraps and all(e["event_type"] == "rewrap" for e in rewraps)


@pytest.mark.parametrize(
    "status",
    ["pending", "consumed", "revoked", "rewrapped", "skipped"],
)
def test_filter_by_status_accepts_every_enum_value(client, status):
    # The filter is valid even when nothing in the scope matches.
    response = _query(client, status=status)
    assert response.status_code == 200
    assert response.json()["events"] == []


def test_filter_by_occurred_time_window_is_inclusive(app):
    client = TestClient(app)
    instant = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
    _insert_event(
        app,
        event_id="11111111-1111-1111-1111-111111111111",
        occurred_at=instant,
    )
    stamp = instant.isoformat()
    rows = _query(
        client, occurred_after=stamp, occurred_before=stamp
    ).json()["events"]
    assert [e["event_id"] for e in rows] == [
        "11111111-1111-1111-1111-111111111111"
    ]
    z_spelling = stamp.replace("+00:00", "Z")
    rows = _query(client, occurred_after=z_spelling).json()["events"]
    assert len(rows) == 1
    past = (instant - timedelta(seconds=1)).isoformat()
    future = (instant + timedelta(seconds=1)).isoformat()
    assert _query(client, occurred_before=past).json()["events"] == []
    assert _query(client, occurred_after=future).json()["events"] == []


def test_query_is_scoped_to_tenant_and_workload(app):
    client = TestClient(app)
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _insert_event(app, event_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                  occurred_at=t)
    _insert_event(
        app,
        event_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        occurred_at=t,
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
    )
    assert [
        e["event_id"] for e in _query(client).json()["events"]
    ] == ["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"]
    assert [
        e["event_id"]
        for e in _query(
            client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
        ).json()["events"]
    ] == ["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"]


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_event_once_in_order(app, monkeypatch):
    client = TestClient(app)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ids = [_seq_id(i) for i in range(7)]
    for i, event_id in enumerate(ids):
        _insert_event(
            app,
            event_id=event_id,
            occurred_at=base + timedelta(seconds=i // 3),
        )
    rows = _walk(client, page_size=3, monkeypatch=monkeypatch)
    assert [e["event_id"] for e in rows] == ids


def test_page_carries_cursor_and_complete_flag(app, monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(app_module, "COMPLIANCE_AUDIT_PAGE_SIZE", 2)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(5):
        _insert_event(
            app,
            event_id=_seq_id(i),
            occurred_at=base,
        )

    first = _query(client).json()
    assert len(first["events"]) == 2
    assert first["complete"] is False and first["next_cursor"]
    second = _query(client, cursor=first["next_cursor"]).json()
    assert len(second["events"]) == 2
    assert second["complete"] is False
    first_keys = [(e["occurred_at"], e["event_id"]) for e in first["events"]]
    second_keys = [(e["occurred_at"], e["event_id"]) for e in second["events"]]
    assert first_keys < second_keys

    last = _query(client, cursor=second["next_cursor"]).json()
    assert len(last["events"]) == 1
    assert last["complete"] is True
    assert last["next_cursor"] == ""


def test_cursor_boundary_is_exclusive_across_timestamp_tie(app, monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(app_module, "COMPLIANCE_AUDIT_PAGE_SIZE", 1)
    instant = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ids = [_seq_id(i) for i in range(4)]
    for event_id in ids:
        _insert_event(app, event_id=event_id, occurred_at=instant)
    seen = _walk(client, page_size=1, monkeypatch=monkeypatch)
    assert [e["event_id"] for e in seen] == ids


def test_replaying_same_cursor_returns_identical_page(app, monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(app_module, "COMPLIANCE_AUDIT_PAGE_SIZE", 2)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(5):
        _insert_event(
            app,
            event_id=_seq_id(i),
            occurred_at=base,
        )
    cursor = _query(client).json()["next_cursor"]
    first = _query(client, cursor=cursor)
    second = _query(client, cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_string_cursor_equals_default(app, monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(app_module, "COMPLIANCE_AUDIT_PAGE_SIZE", 2)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(3):
        _insert_event(
            app,
            event_id=_seq_id(i),
            occurred_at=base,
        )
    assert _query(client).content == _query(client, cursor="").content


def test_tampered_forged_or_cross_scope_cursor_returns_422(app, monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(app_module, "COMPLIANCE_AUDIT_PAGE_SIZE", 1)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(3):
        _insert_event(
            app,
            event_id=_seq_id(i),
            occurred_at=base,
        )
    cursor = _query(client).json()["next_cursor"]
    assert cursor

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _query(client, cursor=tampered).status_code == 422

    forged = b64url_encode(
        b'{"k":"compliance-audit-events-v1"}' + b"0" * 32
    )
    assert _query(client, cursor=forged).status_code == 422

    assert _query(client, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert (
        _query(client, workload=OTHER_WORKLOAD, cursor=cursor).status_code == 422
    )


def test_cursor_cannot_cross_filter_conditions(app, monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(app_module, "COMPLIANCE_AUDIT_PAGE_SIZE", 1)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(3):
        _insert_event(
            app,
            event_id=_seq_id(i),
            occurred_at=base,
            event_type="rewrap" if i % 2 else "grant",
            status="skipped" if i % 2 else "pending",
        )
    cursor = _query(client).json()["next_cursor"]
    assert _query(client, status="pending", cursor=cursor).status_code == 422
    assert (
        _query(client, event_type="grant", cursor=cursor).status_code == 422
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


def test_cursor_minted_under_one_filter_walks_that_filter_stably(
    app, monkeypatch
):
    client = TestClient(app)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(5):
        _insert_event(
            app,
            event_id=_seq_id(i),
            occurred_at=base,
            event_type="rewrap" if i % 2 else "grant",
            status="skipped" if i % 2 else "pending",
        )
    rows = _walk(
        client, page_size=2, monkeypatch=monkeypatch, event_type="rewrap"
    )
    assert len(rows) == 2
    assert {e["event_type"] for e in rows} == {"rewrap"}


def test_other_cursor_kinds_are_rejected(client):
    from proof_release.app import _encode_cursor, _encode_grant_audit_cursor

    # Warm the scope with a real grant.
    decision = _decision(client)
    _mint(client, decision["decision_id"])

    # A rewrap-batch cursor and a grant-audit cursor share the MAC secret
    # but carry a different kind tag, so both are rejected here.
    foreign_batch = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _query(client, cursor=foreign_batch).status_code == 422
    foreign_grant = _encode_grant_audit_cursor(
        TENANT,
        WORKLOAD,
        "11111111-1111-1111-1111-111111111111",
        grant_id="",
        decision_id="",
        data_id="",
        status="",
        issued_after="",
        issued_before="",
    )
    assert _query(client, cursor=foreign_grant).status_code == 422


def test_compliance_cursor_is_not_accepted_on_grant_audit(app, monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(app_module, "COMPLIANCE_AUDIT_PAGE_SIZE", 1)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _insert_event(
        app,
        event_id="11111111-1111-1111-1111-111111111111",
        occurred_at=base,
    )
    _insert_event(
        app,
        event_id="22222222-2222-2222-2222-222222222222",
        occurred_at=base + timedelta(seconds=1),
    )
    cursor = _query(client).json()["next_cursor"]
    assert cursor
    response = client.get(
        "/v1/release-grants",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "cursor": cursor,
        },
    )
    assert response.status_code == 422


# --- read-only behaviour ---------------------------------------------------


def test_query_writes_no_state(app):
    client = TestClient(app)
    decision = _decision(client)
    grants = [
        _mint(client, decision["decision_id"], data_id=f"d{i}") for i in range(3)
    ]
    for _ in range(3):
        assert _query(client).status_code == 200
    with app.state.session_factory() as session:
        rows = session.query(AuditEvent).all()
        # Exactly the three pending issuance events; queries add nothing.
        assert len(rows) == 3
        assert {r.status for r in rows} == {"pending"}
        assert {r.grant_id for r in rows} == {g["grant_id"] for g in grants}


# --- concurrency -----------------------------------------------------------


def test_queries_interleaved_with_settlement_only_see_committed_state(app):
    client = TestClient(app)
    decision = _decision(client)
    grants = [
        _mint(client, decision["decision_id"], data_id=f"d{i}") for i in range(10)
    ]
    settle_targets = grants[:4]

    def settle():
        local = TestClient(app)
        for index, grant in enumerate(settle_targets):
            path = (
                f"/v1/release-grants/{grant['grant_id']}/"
                + ("consume" if index % 2 == 0 else "revoke")
            )
            response = local.post(
                path,
                json={
                    "tenant_id": TENANT,
                    "workload_id": WORKLOAD,
                    "capability": grant["capability"],
                },
            )
            assert response.status_code == 200

    def audit():
        local = TestClient(app)
        for _ in range(20):
            response = local.get(
                "/v1/compliance/audit-events",
                params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            )
            assert response.status_code == 200
            data = response.json()
            for row in data["events"]:
                assert row["status"] in (
                    "pending",
                    "consumed",
                    "revoked",
                    "rewrapped",
                    "skipped",
                )
            keys = [(row["occurred_at"], row["event_id"])
                    for row in data["events"]]
            assert keys == sorted(keys)

    with ThreadPoolExecutor(max_workers=4) as pool:
        pages_future = pool.submit(audit)
        settle_future = pool.submit(settle)
        pages_future.result()
        settle_future.result()

    # Ten pending issuance events plus four settlement events.
    final = _query(client).json()["events"]
    assert len(final) == 14
    assert [e["status"] for e in final].count("pending") == 10


def test_replaying_a_cursor_after_concurrent_commits_keeps_stable_page(
    app, monkeypatch
):
    client = TestClient(app)
    monkeypatch.setattr(app_module, "COMPLIANCE_AUDIT_PAGE_SIZE", 2)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ids = [_seq_id(i) for i in range(5)]
    for i, event_id in enumerate(ids):
        _insert_event(app, event_id=event_id, occurred_at=base)

    cursor = _query(client).json()["next_cursor"]
    before = _query(client, cursor=cursor).json()
    assert [e["event_id"] for e in before["events"]] == ids[2:4]

    # New events commit while the client holds the cursor; the replayed
    # page is unchanged and no boundary entry is duplicated or skipped.
    _insert_event(
        app,
        event_id="ffffffff-0000-0000-0000-000000000000",
        occurred_at=base - timedelta(days=1),
    )
    _insert_event(
        app,
        event_id="ffffffff-0000-0000-0000-000000000001",
        occurred_at=base + timedelta(days=1),
    )
    after = _query(client, cursor=cursor).json()
    assert [e["event_id"] for e in after["events"]] == ids[2:4]

    walked = _walk(client, page_size=2, monkeypatch=monkeypatch)
    all_ids = [e["event_id"] for e in walked]
    assert sorted(all_ids) == sorted(
        ids
        + [
            "ffffffff-0000-0000-0000-000000000000",
            "ffffffff-0000-0000-0000-000000000001",
        ]
    )


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_returns_no_page(app, client):
    from sqlalchemy import text

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE audit_events"))
    assert _query(client).status_code == 500


# --- persistence -----------------------------------------------------------


def test_events_are_available_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    decision = _decision(client1)
    grant = _mint(client1, decision["decision_id"], data_id="persist")
    client1.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    events = _query(client2).json()["events"]
    assert [e["status"] for e in events] == ["pending", "consumed"]
    assert {e["grant_id"] for e in events} == {grant["grant_id"]}
    second.state.engine.dispose()
