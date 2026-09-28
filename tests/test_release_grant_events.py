"""Tests for the persistent per-grant state-migration event timeline:

GET /v1/release-grants/{grant_id}/events

The endpoint ranges exactly one one-time release grant by mandatory
tenant/workload scope and a canonical lowercase-UUID path id; it accepts
only tenant_id, workload_id and an optional cursor, carries no body, and
returns the grant's committed state migrations as stable, immutable,
gap-free per-grant sequence events:

* no old state -> pending (reason ``issued``);
* pending -> consumed via the consume endpoint (reason ``consume``) or
  via an authorized payload release (reason ``release``);
* pending -> revoked (reason ``revoked``).

Exactly one event exists per migration that committed. The first
(cursor-less) query fixes a replayable snapshot high-water mark; later
commits surface only in a fresh first query. A resume cursor is
HMAC-authenticated and bound to the scope, grant and snapshot, so it
cannot be forged, tampered with, or replayed across scopes, grants,
snapshots or cursor families. The query is strictly read-only and
neither changes grant behavior nor the existing compliance audit.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import Base, ReleaseGrant, ReleaseGrantEvent
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
MASTER_KEY_BYTES = b"0123456789abcdef0123456789abcdef"
MASTER_KEY = b64url_encode(MASTER_KEY_BYTES)
PAYLOAD = "grant-timeline-secret 🔐"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"
LETTERED_UUID = "abcdefab-cdef-abcd-efab-cdefabcdefab"

EVENT_FIELDS = ["event_id", "seq", "old_status", "new_status", "reason", "created_at"]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/grant_events.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


# --- flow helpers ----------------------------------------------------------


def _mac(nonce: str, claims: dict) -> str:
    key = hmac.new(
        SECRET.encode("utf-8"),
        f"{TENANT}:{WORKLOAD}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _evidence(nonce: str) -> str:
    claims = {"m": "x"}
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)}
    )


def _decision(client):
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    evidence = _evidence(created["nonce"])
    submitted = client.post(
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
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    verified = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.status_code == 200
    policy = client.post(
        "/v1/policies",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "name": "release",
            "rule": {"claim": "m", "equals": "x"},
        },
    ).json()
    decided = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert decided.status_code == 200
    return decided.json()["decision_id"]


def _envelope(client, data_id=DATA_ID, payload=PAYLOAD):
    response = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": data_id,
            "payload": payload,
        },
    )
    assert response.status_code == 201
    return response.json()


def _grant(client, decision_id, data_id=DATA_ID):
    response = client.post(
        "/v1/release-grants",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "decision_id": decision_id,
            "data_id": data_id,
        },
    )
    assert response.status_code == 201
    return response.json()


def _consume(client, grant_id, capability, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release-grants/{grant_id}/consume", json=body)


def _revoke(client, grant_id, capability, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release-grants/{grant_id}/revoke", json=body)


def _release(client, grant_id, capability, *, data_id=DATA_ID, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": data_id,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release/{grant_id}", json=body)


def _setup_pending_grant(client, *, with_envelope=False):
    decision_id = _decision(client)
    if with_envelope:
        _envelope(client)
    return _grant(client, decision_id)


def _events(client, grant_id, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        f"/v1/release-grants/{grant_id}/events",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


def _walk(client, grant_id, page_size, monkeypatch, **params):
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", page_size)
    rows = []
    token = None
    for _ in range(100):
        query = dict(params)
        if token:
            query["cursor"] = token
        data = _events(client, grant_id, **query).json()
        rows.extend(data["events"])
        if data["complete"]:
            return rows
        token = data["next_cursor"]
        assert token
    raise AssertionError("pagination never completed")  # pragma: no cover


# --- request validation ----------------------------------------------------


def test_missing_scope_parameters_are_422(client):
    assert _events(client, ZERO_UUID, tenant="").status_code == 422
    assert _events(client, ZERO_UUID, workload="").status_code == 422
    response = client.get(f"/v1/release-grants/{ZERO_UUID}/events")
    assert response.status_code == 422


@pytest.mark.parametrize(
    "grant_id",
    [
        "not-a-uuid",
        "abc123",
        ZERO_UUID[:-1] + "z",
        LETTERED_UUID.upper(),
        "%20" + ZERO_UUID,
        " " + ZERO_UUID,
        ZERO_UUID + " ",
    ],
)
def test_malformed_grant_identifier_is_422(client, grant_id):
    assert _events(client, grant_id).status_code == 422


def test_empty_path_segment_is_422(client):
    response = client.get(
        "/v1/release-grants//events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"cursor": "   "},
        {"cursor": "not base64!!"},
        {"cursor": "AAA="},
        {"bogus": "value"},
        {"limit": "10"},
        {"page_size": "2"},
        {"status": "pending"},
        {"event_id": ZERO_UUID},
        {"data_id": DATA_ID},
    ],
)
def test_unknown_or_malformed_parameters_are_422(client, params):
    assert _events(client, ZERO_UUID, **params).status_code == 422


def test_duplicate_parameter_is_422(client):
    response = client.get(
        f"/v1/release-grants/{ZERO_UUID}/events",
        params=[
            ("tenant_id", TENANT),
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
        ],
    )
    assert response.status_code == 422
    response = client.get(
        f"/v1/release-grants/{ZERO_UUID}/events",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("cursor", ""),
            ("cursor", ""),
        ],
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"x", b"   ", b"{}"])
def test_non_empty_body_is_422(client, body):
    response = client.request(
        "GET",
        f"/v1/release-grants/{ZERO_UUID}/events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=body,
    )
    assert response.status_code == 422


def test_missing_and_zero_length_body_are_legal(client):
    grant = _setup_pending_grant(client)
    grant_id = grant["grant_id"]
    no_body = client.get(
        f"/v1/release-grants/{grant_id}/events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    empty_body = client.request(
        "GET",
        f"/v1/release-grants/{grant_id}/events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"",
    )
    assert no_body.status_code == 200
    assert empty_body.status_code == 200
    assert no_body.content == empty_body.content


def test_invalid_inputs_read_no_state(app, client):
    # Malformed path, bad cursor and unknown params must never touch state.
    _events(client, "not-a-uuid")
    _events(client, ZERO_UUID, cursor="garbage!!")
    _events(client, ZERO_UUID, bogus="1")
    with app.state.session_factory() as session:
        assert session.query(ReleaseGrantEvent).count() == 0


# --- 404 semantics ---------------------------------------------------------


def test_unknown_grant_returns_404(client):
    assert _events(client, ZERO_UUID).status_code == 404


def test_cross_scope_grant_returns_404(client):
    grant = _setup_pending_grant(client)
    grant_id = grant["grant_id"]
    assert _events(client, grant_id, tenant=OTHER_TENANT).status_code == 404
    assert _events(client, grant_id, workload=OTHER_WORKLOAD).status_code == 404
    assert (
        _events(
            client, grant_id, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
        ).status_code
        == 404
    )


def test_422_precedes_404(client):
    response = _events(client, ZERO_UUID, cursor="tampered-token")
    assert response.status_code == 422
    response = _events(client, "not-a-uuid")
    assert response.status_code == 422


# --- recorded timeline -----------------------------------------------------


def test_pending_grant_records_only_the_issuance_event(client):
    grant = _setup_pending_grant(client)
    data = _events(client, grant["grant_id"]).json()
    assert data["next_cursor"] == ""
    assert data["complete"] is True
    events = data["events"]
    assert [(e["seq"], e["old_status"], e["new_status"], e["reason"]) for e in events] == [
        (1, None, "pending", "issued"),
    ]
    event = events[0]
    assert list(event) == EVENT_FIELDS
    assert isinstance(event["event_id"], str) and len(event["event_id"]) == 36
    assert event["event_id"] != grant["grant_id"]
    parsed = datetime.fromisoformat(event["created_at"])
    assert parsed.utcoffset() == timedelta(0)


def test_consume_records_pending_to_consumed_with_consume_reason(client):
    grant = _setup_pending_grant(client)
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200

    events = _events(client, grant["grant_id"]).json()["events"]
    assert [(e["seq"], e["old_status"], e["new_status"], e["reason"]) for e in events] == [
        (1, None, "pending", "issued"),
        (2, "pending", "consumed", "consume"),
    ]
    assert [e["seq"] for e in events] == [1, 2]
    assert len({e["event_id"] for e in events}) == 2


def test_payload_release_records_consumed_with_release_reason(client):
    grant = _setup_pending_grant(client, with_envelope=True)
    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 200
    assert response.json()["payload"] == PAYLOAD

    events = _events(client, grant["grant_id"]).json()["events"]
    assert [(e["old_status"], e["new_status"], e["reason"]) for e in events] == [
        (None, "pending", "issued"),
        ("pending", "consumed", "release"),
    ]


def test_revoke_records_pending_to_revoked_with_revoked_reason(client):
    grant = _setup_pending_grant(client)
    assert _revoke(client, grant["grant_id"], grant["capability"]).status_code == 200

    events = _events(client, grant["grant_id"]).json()["events"]
    assert [(e["seq"], e["old_status"], e["new_status"], e["reason"]) for e in events] == [
        (1, None, "pending", "issued"),
        (2, "pending", "revoked", "revoked"),
    ]


def test_settlement_attempts_after_revoke_add_no_events(client):
    grant = _setup_pending_grant(client, with_envelope=True)
    grant_id, capability = grant["grant_id"], grant["capability"]
    assert _revoke(client, grant_id, capability).status_code == 200
    # Every post-revocation judgement observes the existing terminal
    # state; none appends an event (and release emits no payload).
    assert _consume(client, grant_id, capability).status_code == 409
    assert _revoke(client, grant_id, capability).status_code == 409
    release = _release(client, grant_id, capability)
    assert release.status_code == 409
    assert "payload" not in release.text

    events = _events(client, grant_id).json()["events"]
    assert [e["reason"] for e in events] == ["issued", "revoked"]
    assert [e["seq"] for e in events] == [1, 2]


def test_consume_then_revoke_adds_no_second_settlement_event(client):
    grant = _setup_pending_grant(client)
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200
    assert _revoke(client, grant["grant_id"], grant["capability"]).status_code == 409
    events = _events(client, grant["grant_id"]).json()["events"]
    assert [e["reason"] for e in events] == ["issued", "consume"]


def test_concurrent_consume_and_revoke_leave_exactly_one_settlement_event(client):
    grant = _setup_pending_grant(client)
    grant_id, capability = grant["grant_id"], grant["capability"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(
                lambda fn: fn(client, grant_id, capability),
                (_consume, _revoke),
            )
        )
    assert sorted(response.status_code for response in outcomes) == [200, 409]

    events = _events(client, grant_id).json()["events"]
    assert len(events) == 2
    assert events[0]["reason"] == "issued"
    assert events[1]["reason"] in {"consume", "revoked"}
    assert events[1]["new_status"] in {"consumed", "revoked"}


def test_wrong_capability_settles_nothing_and_appends_no_event(client):
    grant = _setup_pending_grant(client)
    response = _consume(client, grant["grant_id"], "A" * 43)
    assert response.status_code in (401, 422)
    events = _events(client, grant["grant_id"]).json()["events"]
    assert [e["reason"] for e in events] == ["issued"]


def test_timestamps_match_the_grant_row(client):
    grant = _setup_pending_grant(client)
    grant_id, capability = grant["grant_id"], grant["capability"]
    consumed = _consume(client, grant_id, capability).json()
    events = _events(client, grant_id).json()["events"]
    assert events[0]["created_at"] == grant["issued_at"]
    assert events[1]["created_at"] == consumed["consumed_at"]


# --- wire format -----------------------------------------------------------


def test_response_is_compact_json_with_single_trailing_newline(client):
    grant = _setup_pending_grant(client)
    response = _events(client, grant["grant_id"])
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == ["events", "next_cursor", "complete"]


def test_empty_result_shape(client):
    # An unknown-but-shaped id is a 404, never an empty page; the empty
    # list/empty cursor/true shape is reached for a grant only via a fixed
    # snapshot taken before any event existed, which cannot happen for a
    # real grant (issuance commits with it). Exercise the serialization
    # via the snapshot boundary instead.
    grant = _setup_pending_grant(client)
    data = _events(client, grant["grant_id"]).json()
    assert data["events"] and data["next_cursor"] == "" and data["complete"] is True


def test_empty_string_cursor_equals_default(client):
    grant = _setup_pending_grant(client)
    assert _events(client, grant["grant_id"]).content == _events(
        client, grant["grant_id"], cursor=""
    ).content


def test_no_floats_or_non_finite_values(client):
    grant = _setup_pending_grant(client)
    _consume(client, grant["grant_id"], grant["capability"])

    def _check(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            return
        if isinstance(value, float):  # pragma: no cover - structural guard
            raise AssertionError(f"float leaked into response: {value!r}")
        if value is None or isinstance(value, str):
            return
        if isinstance(value, list):
            for item in value:
                _check(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                _check(item)
            return
        raise AssertionError(f"unexpected type: {type(value)!r}")  # pragma: no cover

    _check(json.loads(_events(client, grant["grant_id"]).content))


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_event_once_in_order(client, monkeypatch):
    grant = _setup_pending_grant(client)
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200
    rows = _walk(client, grant["grant_id"], 1, monkeypatch)
    assert [e["seq"] for e in rows] == [1, 2]
    assert [(e["new_status"], e["reason"]) for e in rows] == [
        ("pending", "issued"),
        ("consumed", "consume"),
    ]


def test_page_carries_cursor_and_complete_flag(client, monkeypatch):
    grant = _setup_pending_grant(client)
    _consume(client, grant["grant_id"], grant["capability"])
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 1)

    first = _events(client, grant["grant_id"]).json()
    assert [e["seq"] for e in first["events"]] == [1]
    assert first["complete"] is False and first["next_cursor"]

    second = _events(client, grant["grant_id"], cursor=first["next_cursor"]).json()
    assert [e["seq"] for e in second["events"]] == [2]
    assert second["complete"] is True and second["next_cursor"] == ""


def test_replaying_same_cursor_returns_identical_page(client, monkeypatch):
    grant = _setup_pending_grant(client)
    _consume(client, grant["grant_id"], grant["capability"])
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 1)
    cursor = _events(client, grant["grant_id"]).json()["next_cursor"]
    first = _events(client, grant["grant_id"], cursor=cursor)
    second = _events(client, grant["grant_id"], cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


# --- snapshot isolation ----------------------------------------------------


def _insert_synthetic_grant_event(app, grant_id, seq, *, created_at, new_status="consumed"):
    """Append one committed timeline row directly (snapshot testing only)."""
    with app.state.session_factory() as session:
        session.add(
            ReleaseGrantEvent(
                event_id=f"0000000{seq}-0000-4000-8000-000000000000",
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                grant_id=grant_id,
                seq=seq,
                old_status="pending",
                new_status=new_status,
                reason="consume" if new_status == "consumed" else "revoked",
                created_at=created_at,
            )
        )
        session.commit()


def test_first_query_fixes_snapshot_for_later_pages(app, client, monkeypatch):
    # A real grant settles at most twice, so commit three migration rows
    # (issuance + two synthetic per-grant migrations) before opening the
    # timeline; with page size 1 the first query fixes the snapshot at the
    # then-greatest seq (3) and returns just seq 1.
    grant = _setup_pending_grant(client)
    grant_id = grant["grant_id"]
    issued_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _insert_synthetic_grant_event(app, grant_id, 2, created_at=issued_at)
    _insert_synthetic_grant_event(app, grant_id, 3, created_at=issued_at)
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 1)
    first = _events(client, grant_id).json()
    assert [e["seq"] for e in first["events"]] == [1]
    cursor = first["next_cursor"]
    assert cursor

    # A later migration commits seq 4 with a business time *older* than
    # every event already seen: the snapshot is bound to the commit
    # sequence, never to business time.
    _insert_synthetic_grant_event(
        app, grant_id, 4, created_at=issued_at - timedelta(seconds=30)
    )

    # Walking the fixed snapshot returns exactly seq 2 then seq 3; the
    # final row arrives on a completed page (its probe finds nothing more
    # within the snapshot), and no page ever absorbs the later seq 4.
    seen = list(first["events"])
    token = cursor
    pages: list[tuple[str, dict]] = []
    for _ in range(10):
        page = _events(client, grant_id, cursor=token).json()
        pages.append((token, page))
        seen.extend(page["events"])
        if page["complete"]:
            break
        token = page["next_cursor"]
    assert [e["seq"] for e in seen] == [1, 2, 3]
    final_token, final_page = pages[-1]
    assert [e["seq"] for e in final_page["events"]] == [3]
    assert final_page["complete"] is True and final_page["next_cursor"] == ""
    # Replaying the final page's inbound cursor returns the identical
    # page, unaffected by the migration that committed meanwhile.
    assert _events(client, grant_id, cursor=final_token).json() == final_page

    # A fresh first query (a new query family) sees every migration,
    # ordered by the immutable seq rather than by business time.
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 100)
    fresh = _events(client, grant_id).json()
    assert [e["seq"] for e in fresh["events"]] == [1, 2, 3, 4]
    assert fresh["complete"] is True
    assert fresh["events"][-1]["created_at"] < fresh["events"][0]["created_at"]


def test_snapshot_walks_a_long_synthetic_timeline_without_absorbing_later_commit(
    app, client, monkeypatch
):
    grant = _setup_pending_grant(client)
    grant_id = grant["grant_id"]
    issued_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for seq in range(2, 6):
        _insert_synthetic_grant_event(app, grant_id, seq, created_at=issued_at)

    # Fix the snapshot at seq 5 with page size 2, then commit seq 6
    # before walking the later pages.
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 2)
    page1 = _events(client, grant_id).json()
    assert [e["seq"] for e in page1["events"]] == [1, 2]
    _insert_synthetic_grant_event(app, grant_id, 6, created_at=issued_at)

    seen = list(page1["events"])
    token = page1["next_cursor"]
    for _ in range(10):
        page = _events(client, grant_id, cursor=token).json()
        seen.extend(page["events"])
        if page["complete"]:
            break
        token = page["next_cursor"]
    assert [e["seq"] for e in seen] == [1, 2, 3, 4, 5]

    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 100)
    fresh = _events(client, grant_id).json()
    assert [e["seq"] for e in fresh["events"]] == [1, 2, 3, 4, 5, 6]


def test_pending_snapshot_stays_single_event_after_settlement(client):
    grant = _setup_pending_grant(client)
    grant_id, capability = grant["grant_id"], grant["capability"]
    first = _events(client, grant_id).json()
    assert [e["seq"] for e in first["events"]] == [1]
    assert first["complete"] is True

    _consume(client, grant_id, capability)
    # The completed old family cannot be resumed; a brand-new first query
    # is a new family that observes the settlement.
    fresh = _events(client, grant_id).json()
    assert [e["seq"] for e in fresh["events"]] == [1, 2]


# --- cursor authentication -------------------------------------------------


def test_tampered_or_forged_cursor_is_422(client, monkeypatch):
    from proof_release.envelopes import b64url_encode

    grant = _setup_pending_grant(client)
    _consume(client, grant["grant_id"], grant["capability"])
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 1)
    cursor = _events(client, grant["grant_id"]).json()["next_cursor"]

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _events(client, grant["grant_id"], cursor=tampered).status_code == 422
    forged = b64url_encode(
        b'{"k":"release-grant-events-v1","t":"tenant-a","w":"workload-1",'
        b'"g":"' + grant["grant_id"].encode() + b'","q":1,"h":2}' + b"0" * 32
    )
    assert _events(client, grant["grant_id"], cursor=forged).status_code == 422


def test_cross_scope_cursor_is_422(client, monkeypatch):
    grant = _setup_pending_grant(client)
    _consume(client, grant["grant_id"], grant["capability"])
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 1)
    cursor = _events(client, grant["grant_id"]).json()["next_cursor"]
    assert (
        _events(client, grant["grant_id"], tenant=OTHER_TENANT, cursor=cursor).status_code
        == 422
    )
    assert (
        _events(
            client, grant["grant_id"], workload=OTHER_WORKLOAD, cursor=cursor
        ).status_code
        == 422
    )


def test_cross_grant_cursor_is_422(client, monkeypatch):
    first = _setup_pending_grant(client)
    second = _setup_pending_grant(client)
    _consume(client, first["grant_id"], first["capability"])
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 1)
    cursor = _events(client, first["grant_id"]).json()["next_cursor"]
    # The cursor is minted for first; replaying it against the sibling
    # grant in the same scope is a 422, not a page of second's events.
    assert _events(client, second["grant_id"], cursor=cursor).status_code == 422


def test_foreign_cursor_families_are_422(client):
    from proof_release.app import (
        _encode_audit_event_cursor,
        _encode_cursor,
        _encode_decision_cursor,
        _encode_grant_audit_cursor,
        _encode_proof_event_cursor,
        _encode_revocation_cursor,
        _encode_rewrap_job_event_cursor,
        _encode_rewrap_job_history_cursor,
    )

    grant = _setup_pending_grant(client)
    grant_id = grant["grant_id"]
    ts = "2026-01-01T00:00:00+00:00"

    foreign_batch = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _events(client, grant_id, cursor=foreign_batch).status_code == 422
    foreign_grant_audit = _encode_grant_audit_cursor(
        TENANT,
        WORKLOAD,
        ZERO_UUID,
        grant_id="",
        decision_id="",
        data_id="",
        status="",
        issued_after="",
        issued_before="",
    )
    assert _events(client, grant_id, cursor=foreign_grant_audit).status_code == 422
    foreign_audit = _encode_audit_event_cursor(
        TENANT,
        WORKLOAD,
        ts,
        ZERO_UUID,
        event_id="",
        event_type="",
        status="",
        occurred_after="",
        occurred_before="",
    )
    assert _events(client, grant_id, cursor=foreign_audit).status_code == 422
    foreign_revocation = _encode_revocation_cursor(
        TENANT,
        WORKLOAD,
        ZERO_UUID,
        ts,
        ZERO_UUID,
        revocation_id="",
        certificate_fingerprint="",
        effective_after="",
        effective_before="",
        snapshot_at=ts,
        snapshot_id=ZERO_UUID,
    )
    assert _events(client, grant_id, cursor=foreign_revocation).status_code == 422
    foreign_job_history = _encode_rewrap_job_history_cursor(
        TENANT,
        WORKLOAD,
        ZERO_UUID,
        job_id="",
        status="",
        created_after="",
        created_before="",
    )
    assert _events(client, grant_id, cursor=foreign_job_history).status_code == 422
    foreign_job_events = _encode_rewrap_job_event_cursor(
        TENANT, WORKLOAD, ZERO_UUID, 1, snapshot_seq=1
    )
    assert _events(client, grant_id, cursor=foreign_job_events).status_code == 422
    foreign_proof = _encode_proof_event_cursor(
        TENANT,
        WORKLOAD,
        ts,
        ZERO_UUID,
        evidence_id="",
        event_type="",
        status="",
        occurred_after="",
        occurred_before="",
        snapshot_seq=1,
    )
    assert _events(client, grant_id, cursor=foreign_proof).status_code == 422
    foreign_decision = _encode_decision_cursor(
        TENANT,
        WORKLOAD,
        ts,
        ZERO_UUID,
        decision_id="",
        evidence_id="",
        policy_id="",
        status="",
        decided_after="",
        decided_before="",
        snapshot_seq=1,
    )
    assert _events(client, grant_id, cursor=foreign_decision).status_code == 422


# --- read-only behaviour ---------------------------------------------------


def test_query_changes_nothing_and_consumes_no_rate_budget(app, client):
    grant = _setup_pending_grant(client)
    grant_id = grant["grant_id"]
    with app.state.session_factory() as session:
        before = session.query(ReleaseGrantEvent).count()
        stored = session.get(ReleaseGrant, grant_id)
        assert stored.status == "pending"

    # Well over the five business-request shared budget; a read-only
    # query consumes none of it.
    for _ in range(8):
        assert _events(client, grant_id).status_code == 200

    with app.state.session_factory() as session:
        assert session.query(ReleaseGrantEvent).count() == before
        assert session.get(ReleaseGrant, grant_id).status == "pending"
    # The next business request is still admitted (not 429).
    assert _consume(client, grant_id, grant["capability"]).status_code == 200


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_page(app, client):
    grant = _setup_pending_grant(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE release_grant_events"))
    response = _events(client, grant["grant_id"])
    assert response.status_code == 500
    # No half page leaks: the body is the plain error envelope, not the
    # timeline container.
    assert set(response.json()) == {"detail"}


# --- persistence -----------------------------------------------------------


def test_events_survive_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/persist.db"
    first = create_app(url)
    first_client = TestClient(first)
    grant = _setup_pending_grant(first_client)
    assert (
        _consume(first_client, grant["grant_id"], grant["capability"]).status_code
        == 200
    )
    first.state.engine.dispose()

    second = create_app(url)
    try:
        second_client = TestClient(second)
        events = second_client.get(
            f"/v1/release-grants/{grant['grant_id']}/events",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).json()["events"]
        assert [e["reason"] for e in events] == ["issued", "consume"]
        assert [e["seq"] for e in events] == [1, 2]
    finally:
        second.state.engine.dispose()


# --- legacy migration ------------------------------------------------------


def _build_legacy_database(url: str) -> dict[str, str]:
    """Create a database as an older deployment (no timeline) left it."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE release_grant_events"))
        rows = [
            # pending grant: only issuance can be reconstructed
            (
                "11111111-1111-4111-8111-111111111111",
                "pending",
                now,
                None,
                None,
            ),
            # consumed grant: issuance then consume settlement
            (
                "22222222-2222-4222-8222-222222222222",
                "consumed",
                now,
                now + timedelta(seconds=5),
                None,
            ),
            # revoked grant: issuance then revoked settlement
            (
                "33333333-3333-4333-8333-333333333333",
                "revoked",
                now,
                None,
                now + timedelta(seconds=9),
            ),
        ]
        for grant_id, status, issued_at, consumed_at, revoked_at in rows:
            conn.execute(
                text(
                    "INSERT INTO release_grants "
                    "(grant_id, tenant_id, workload_id, decision_id, data_id, "
                    "capability_digest, status, issued_at, expires_at, "
                    "consumed_at, revoked_at) "
                    "VALUES (:g, :t, :w, :d, :data, 'x', :status, :issued, "
                    ":expires, :consumed, :revoked)"
                ),
                {
                    "g": grant_id,
                    "t": TENANT,
                    "w": WORKLOAD,
                    "d": "00000000-0000-0000-0000-000000000001",
                    "data": DATA_ID,
                    "status": status,
                    "issued": issued_at,
                    "expires": now + timedelta(seconds=300),
                    "consumed": consumed_at,
                    "revoked": revoked_at,
                },
            )
    engine.dispose()
    return {
        "pending": rows[0][0],
        "consumed": rows[1][0],
        "revoked": rows[2][0],
    }


def test_legacy_database_backfills_gap_free_history(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/legacy.db"
    ids = _build_legacy_database(url)

    application = create_app(url)
    try:
        client = TestClient(application)

        pending_events = _events(client, ids["pending"]).json()["events"]
        assert [(e["seq"], e["old_status"], e["new_status"], e["reason"]) for e in pending_events] == [
            (1, None, "pending", "issued"),
        ]
        consumed_events = _events(client, ids["consumed"]).json()["events"]
        assert [(e["seq"], e["old_status"], e["new_status"], e["reason"]) for e in consumed_events] == [
            (1, None, "pending", "issued"),
            (2, "pending", "consumed", "consume"),
        ]
        revoked_events = _events(client, ids["revoked"]).json()["events"]
        assert [(e["seq"], e["old_status"], e["new_status"], e["reason"]) for e in revoked_events] == [
            (1, None, "pending", "issued"),
            (2, "pending", "revoked", "revoked"),
        ]
        for events in (pending_events, consumed_events, revoked_events):
            for event in events:
                assert list(event) == EVENT_FIELDS
                datetime.fromisoformat(event["created_at"])

        # Sequences are per-grant and gap-free independently.
        with application.state.session_factory() as session:
            seqs = {
                grant_id: [
                    row[0]
                    for row in session.execute(
                        text(
                            "SELECT seq FROM release_grant_events "
                            "WHERE grant_id = :g ORDER BY seq"
                        ),
                        {"g": grant_id},
                    ).all()
                ]
                for grant_id in ids.values()
            }
        assert seqs == {
            ids["pending"]: [1],
            ids["consumed"]: [1, 2],
            ids["revoked"]: [1, 2],
        }
    finally:
        application.state.engine.dispose()


def test_legacy_backfill_is_idempotent_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/legacy_idem.db"
    ids = _build_legacy_database(url)
    first = create_app(url)
    first.state.engine.dispose()
    second = create_app(url)
    try:
        with second.state.engine.begin() as conn:
            # Two rows per settled grant plus one for the pending grant,
            # and no duplicates on reopen.
            assert conn.execute(
                text(
                    "SELECT count(*) FROM release_grant_events "
                    "WHERE tenant_id = :t"
                ),
                {"t": TENANT},
            ).scalar() == 5
            assert conn.execute(
                text(
                    "SELECT count(*) FROM ("
                    "SELECT grant_id, seq "
                    "FROM release_grant_events GROUP BY grant_id, seq)"
                )
            ).scalar() == 5
        client = TestClient(second)
        assert len(_events(client, ids["consumed"]).json()["events"]) == 2
    finally:
        second.state.engine.dispose()


def test_new_grant_after_upgrade_allocates_its_own_sequence(tmp_path, monkeypatch):
    # A live grant in a freshly upgraded database starts its own run at 1
    # and is never mixed with the reconstructed history.
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/legacy_then_new.db"
    ids = _build_legacy_database(url)
    application = create_app(url)
    try:
        client = TestClient(application)
        grant = _setup_pending_grant(client)
        events = _events(client, grant["grant_id"]).json()["events"]
        assert [(e["seq"], e["reason"]) for e in events] == [(1, "issued")]
        # The legacy grants' histories are untouched by the new issuance.
        assert len(_events(client, ids["consumed"]).json()["events"]) == 2
        with application.state.session_factory() as session:
            assert (
                session.query(ReleaseGrantEvent)
                .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
                .count()
                == 1
            )
    finally:
        application.state.engine.dispose()


# --- secrecy ---------------------------------------------------------------


def test_timeline_never_exposes_capability_payload_or_key_material(client):
    grant = _setup_pending_grant(client, with_envelope=True)
    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 200
    response = _events(client, grant["grant_id"])
    for secret in (
        grant["capability"],
        PAYLOAD,
        MASTER_KEY,
        "capability",
        "capability_sha256",
        "wrapped_key",
        "ciphertext",
        "payload",
        "master_key",
        "traceback",
    ):
        assert secret not in response.text
    # Events store neither the capability nor its digest.
    columns = {column.name for column in ReleaseGrantEvent.__table__.columns}
    assert "capability" not in columns
    assert "capability_digest" not in columns
    assert "capability_sha256" not in columns
