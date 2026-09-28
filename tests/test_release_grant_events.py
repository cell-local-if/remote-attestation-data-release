"""Tests for the persistent per-release-grant state-migration timeline:

GET /v1/release-grants/{grant_id}/events

The endpoint ranges exactly one grant by mandatory tenant/workload scope
and a canonical lowercase UUID path id; it accepts only tenant_id,
workload_id and an optional cursor, carries no body, and returns the
grant's committed state migrations as stable, immutable, gap-free
per-grant sequence events: issuance (no old status -> pending, reason
``issued``), the single consumed settlement reached by either a consume
presentation (``consume``) or a payload release (``release``), or the
revoked settlement (``revoked``). The first (cursor-less) query fixes a
replayable snapshot high-water mark; later commits surface only in a
fresh first query. A resume cursor is HMAC-authenticated and bound to
the scope, grant and snapshot, so it cannot be forged, tampered with or
replayed across scopes, grants, snapshots or cursor families. The query
is strictly read-only and leaves the existing grant audit untouched.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import ReleaseGrant, ReleaseGrantEvent
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
SECRET = "unit-test-secret"
MASTER_KEY_BYTES = b"0123456789abcdef0123456789abcdef"
MASTER_KEY = b64url_encode(MASTER_KEY_BYTES)
PAYLOAD = "grant-timeline-secret 🔐"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"
LETTERED_UUID = "abcdefab-cdef-abcd-efab-cdefabcdefab"

EVENT_FIELDS = ["event_id", "seq", "old_status", "new_status", "reason", "at"]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/grant_events.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


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


def _evidence(nonce: str, claims: dict | None = None) -> str:
    claims = claims if claims is not None else {"m": "x"}
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)}
    )


def _decision(client, *, name="release", data_id=None):
    """Drive challenge -> evidence -> verify -> decision; return decision id."""
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
            "name": name,
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


def _grant(client, decision_id, *, data_id="data-1", **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision_id,
        "data_id": data_id,
    }
    body.update(fields)
    return client.post("/v1/release-grants", json=body)


def _pending_grant(client, *, data_id="data-1"):
    decision_id = _decision(client, name=f"policy-{data_id}")
    created = _grant(client, decision_id, data_id=data_id)
    assert created.status_code == 201
    return created.json()


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


def _envelope(client, data_id="data-1", payload=PAYLOAD):
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


def _release(client, grant_id, capability, *, data_id="data-1"):
    return client.post(
        f"/v1/release/{grant_id}",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": data_id,
            "capability": capability,
        },
    )


def _events(client, grant_id, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        f"/v1/release-grants/{grant_id}/events",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


def _tuples(events):
    return [
        (e["seq"], e["old_status"], e["new_status"], e["reason"])
        for e in events
    ]


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
        ZERO_UUID[:-1] + "Z",
        LETTERED_UUID.upper(),
        "%20" + ZERO_UUID,
        "  " + ZERO_UUID,
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
        {"data_id": "data-1"},
        {"decision_id": ZERO_UUID},
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


def test_invalid_inputs_read_no_state(app, client):
    # Malformed path, bad cursor and unknown params must never touch state.
    _events(client, "not-a-uuid")
    _events(client, ZERO_UUID, cursor="garbage!!")
    _events(client, ZERO_UUID, bogus="1")
    with app.state.session_factory() as session:
        assert session.query(ReleaseGrant).count() == 0
        assert session.query(ReleaseGrantEvent).count() == 0


# --- 404 semantics ---------------------------------------------------------


def test_unknown_grant_returns_404(client):
    assert _events(client, ZERO_UUID).status_code == 404


def test_cross_scope_grant_returns_404(client):
    grant = _pending_grant(client)
    assert _events(client, grant["grant_id"], tenant=OTHER_TENANT).status_code == 404
    assert (
        _events(client, grant["grant_id"], workload=OTHER_WORKLOAD).status_code == 404
    )
    assert (
        _events(
            client,
            grant["grant_id"],
            tenant=OTHER_TENANT,
            workload=OTHER_WORKLOAD,
        ).status_code
        == 404
    )


def test_422_precedes_404(client):
    # An illegal cursor is rejected before the unknown grant is resolved.
    response = _events(client, ZERO_UUID, cursor="tampered-token")
    assert response.status_code == 422
    response = _events(client, "not-a-uuid")
    assert response.status_code == 422


# --- recorded timeline -----------------------------------------------------


def test_new_grant_records_only_the_issuance_event(client):
    grant = _pending_grant(client)
    data = _events(client, grant["grant_id"]).json()
    assert data["next_cursor"] == ""
    assert data["complete"] is True
    events = data["events"]
    assert _tuples(events) == [(1, None, "pending", "issued")]
    event = events[0]
    assert list(event) == EVENT_FIELDS
    assert isinstance(event["event_id"], str) and len(event["event_id"]) == 36
    assert event["seq"] == 1 and isinstance(event["seq"], int)
    parsed = datetime.fromisoformat(event["at"])
    assert parsed.utcoffset() == timedelta(0)


def test_consume_records_consume_reason_pending_to_consumed(client):
    grant = _pending_grant(client)
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200

    events = _events(client, grant["grant_id"]).json()["events"]
    assert _tuples(events) == [
        (1, None, "pending", "issued"),
        (2, "pending", "consumed", "consume"),
    ]
    assert len({e["event_id"] for e in events}) == 2


def test_payload_release_records_release_reason_pending_to_consumed(client):
    _envelope(client)
    grant = _pending_grant(client)
    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 200
    assert response.json()["payload"] == PAYLOAD

    events = _events(client, grant["grant_id"]).json()["events"]
    assert _tuples(events) == [
        (1, None, "pending", "issued"),
        (2, "pending", "consumed", "release"),
    ]


def test_revoke_records_revoked_reason_pending_to_revoked(client):
    grant = _pending_grant(client)
    assert _revoke(client, grant["grant_id"], grant["capability"]).status_code == 200

    events = _events(client, grant["grant_id"]).json()["events"]
    assert _tuples(events) == [
        (1, None, "pending", "issued"),
        (2, "pending", "revoked", "revoked"),
    ]


def test_settlement_after_revoke_appends_no_event(client):
    grant = _pending_grant(client)
    _envelope(client)
    assert _revoke(client, grant["grant_id"], grant["capability"]).status_code == 200

    # Every post-revocation settlement observes the existing terminal
    # state (409) and appends nothing.
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 409
    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 409
    assert _revoke(client, grant["grant_id"], grant["capability"]).status_code == 409

    events = _events(client, grant["grant_id"]).json()["events"]
    assert [e["reason"] for e in events] == ["issued", "revoked"]


def test_repeat_consume_or_release_after_consume_appends_no_event(client):
    _envelope(client)
    grant = _pending_grant(client)
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 409
    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 409
    assert _revoke(client, grant["grant_id"], grant["capability"]).status_code == 409

    events = _events(client, grant["grant_id"]).json()["events"]
    assert [e["reason"] for e in events] == ["issued", "consume"]


def test_failed_release_attempt_rolls_back_with_no_event(client):
    # No envelope exists for the grant's data: release is a 404 that
    # changes no state and appends no settlement event.
    grant = _pending_grant(client)
    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 404
    events = _events(client, grant["grant_id"]).json()["events"]
    assert _tuples(events) == [(1, None, "pending", "issued")]


def test_concurrent_consume_leaves_exactly_one_settlement_event(
    app, client, monkeypatch
):
    # Raise the shared per-minute budget so all bursts are admitted and
    # only the atomic guarded transition decides the winner.
    monkeypatch.setattr(app_module, "GRANT_BUDGET_PER_MINUTE", 64)
    grant = _pending_grant(client)
    url = f"/v1/release-grants/{grant['grant_id']}/consume"
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }

    def consume():
        return TestClient(app).post(url, json=body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: consume(), range(16)))

    assert statuses.count(200) == 1
    assert statuses.count(409) == 15

    with app.state.session_factory() as session:
        events = (
            session.query(ReleaseGrantEvent)
            .order_by(ReleaseGrantEvent.seq.asc())
            .all()
        )
        assert [(e.seq, e.reason) for e in events] == [(1, "issued"), (2, "consume")]


# --- wire format -----------------------------------------------------------


def test_response_is_compact_json_with_single_trailing_newline(client):
    grant = _pending_grant(client)
    response = _events(client, grant["grant_id"])
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == ["events", "next_cursor", "complete"]


def test_empty_string_cursor_equals_default(client):
    grant = _pending_grant(client)
    assert _events(client, grant["grant_id"]).content == _events(
        client, grant["grant_id"], cursor=""
    ).content


def test_grant_without_events_returns_empty_page(app, client):
    # A grant row whose event set is empty (possible only for a
    # reconstructed/corrupted store) answers with an empty list, an empty
    # cursor and complete=true — never an error or a fabricated event.
    grant = _pending_grant(client)
    with app.state.engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM release_grant_events WHERE grant_id = :g"
            ),
            {"g": grant["grant_id"]},
        )
    response = _events(client, grant["grant_id"])
    assert response.status_code == 200
    assert response.json() == {"events": [], "next_cursor": "", "complete": True}


def test_no_floats_or_non_finite_values(client):
    grant = _pending_grant(client)
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


def test_pagination_walks_every_event_once_in_order(client, monkeypatch):
    grant = _pending_grant(client)
    _consume(client, grant["grant_id"], grant["capability"])
    rows = _walk(client, grant["grant_id"], 1, monkeypatch)
    assert [e["seq"] for e in rows] == [1, 2]
    assert [e["reason"] for e in rows] == ["issued", "consume"]


def test_page_carries_cursor_and_complete_flag(client, monkeypatch):
    grant = _pending_grant(client)
    _consume(client, grant["grant_id"], grant["capability"])
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 1)

    first = _events(client, grant["grant_id"]).json()
    assert [e["seq"] for e in first["events"]] == [1]
    assert first["complete"] is False and first["next_cursor"]

    last = _events(client, grant["grant_id"], cursor=first["next_cursor"]).json()
    assert [e["seq"] for e in last["events"]] == [2]
    assert last["complete"] is True and last["next_cursor"] == ""


def test_replaying_same_cursor_returns_identical_page(client, monkeypatch):
    grant = _pending_grant(client)
    _consume(client, grant["grant_id"], grant["capability"])
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 1)
    cursor = _events(client, grant["grant_id"]).json()["next_cursor"]
    first = _events(client, grant["grant_id"], cursor=cursor)
    second = _events(client, grant["grant_id"], cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


# --- snapshot isolation ----------------------------------------------------


def test_first_query_fixes_snapshot_for_later_pages(client, monkeypatch):
    # A grant settles at most once, so the snapshot scenario is driven
    # with both events committed: the first query (page size 1) fixes the
    # high-water mark at seq 2 and returns seq 1 with a resume cursor.
    grant = _pending_grant(client)
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 1)
    first = _events(client, grant["grant_id"]).json()
    assert [e["seq"] for e in first["events"]] == [1]
    cursor = first["next_cursor"]
    assert cursor

    # A later migration commits while the client holds the cursor. A
    # grant's state machine can never produce a third event, so the
    # post-snapshot commit is inserted directly through storage, exactly
    # as a concurrent committer's row would appear.
    with client.app.state.session_factory() as session:
        session.add(
            ReleaseGrantEvent(
                event_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                grant_id=grant["grant_id"],
                seq=3,
                old_status="consumed",
                new_status="consumed",
                reason="consume",
                occurred_at=datetime.fromisoformat(first["events"][0]["at"]),
            )
        )
        session.commit()

    # The old snapshot's next page contains only seq 2 and completes: it
    # neither duplicates nor skips and never absorbs the new migration,
    # even though the event rows share one immutable grant.
    second = _events(client, grant["grant_id"], cursor=cursor).json()
    assert [e["seq"] for e in second["events"]] == [2]
    assert second["complete"] is True and second["next_cursor"] == ""
    # Replaying the held cursor returns the identical page.
    assert _events(client, grant["grant_id"], cursor=cursor).json() == second

    # A fresh first query (new query family) sees all three migrations.
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 100)
    fresh = _events(client, grant["grant_id"]).json()
    assert [e["seq"] for e in fresh["events"]] == [1, 2, 3]
    assert fresh["complete"] is True


def test_completed_snapshot_is_not_extended_by_later_revoke(client):
    grant = _pending_grant(client)
    # The first (completed, cursor-less) query fixes the snapshot at seq 1.
    first = _events(client, grant["grant_id"]).json()
    assert [e["seq"] for e in first["events"]] == [1]
    assert first["complete"] is True

    assert _revoke(client, grant["grant_id"], grant["capability"]).status_code == 200
    # Replaying the old family is impossible (its page was complete); a
    # brand new first query is a new family that observes the revocation.
    fresh = _events(client, grant["grant_id"]).json()
    assert [e["reason"] for e in fresh["events"]] == ["issued", "revoked"]


def test_snapshot_is_bounded_by_seq_not_business_time(app, client, monkeypatch):
    # The snapshot predicate is seq-based: even a later-committed event
    # whose business timestamp predates the snapshot must stay out of the
    # fixed family. Both real events commit first (page size 1) to obtain
    # a resume cursor bound to the snapshot at seq 2, then a
    # post-snapshot commit with an older occurred_at is inserted directly
    # through storage.
    grant = _pending_grant(client)
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 1)
    cursor = _events(client, grant["grant_id"]).json()["next_cursor"]
    assert cursor

    with app.state.session_factory() as session:
        session.add(
            ReleaseGrantEvent(
                event_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                grant_id=grant["grant_id"],
                seq=3,
                old_status="consumed",
                new_status="consumed",
                reason="consume",
                occurred_at=datetime.now(timezone.utc) - timedelta(days=1),
            )
        )
        session.commit()

    second = _events(client, grant["grant_id"], cursor=cursor).json()
    assert [e["seq"] for e in second["events"]] == [2]
    assert second["complete"] is True
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 100)
    fresh = _events(client, grant["grant_id"]).json()
    assert [e["seq"] for e in fresh["events"]] == [1, 2, 3]
    # The out-of-snapshot event has an older business time; the seq
    # ordering still keeps it out of the fixed family and last only in the
    # fresh listing by its immutable sequence position.
    assert fresh["events"][-1]["seq"] == 3


# --- cursor authentication -------------------------------------------------


def test_tampered_or_forged_cursor_is_422(client, monkeypatch):
    from proof_release.envelopes import b64url_encode

    grant = _pending_grant(client)
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
    grant = _pending_grant(client)
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
    first = _pending_grant(client, data_id="data-a")
    second = _pending_grant(client, data_id="data-b")
    _consume(client, first["grant_id"], first["capability"])
    _consume(client, second["grant_id"], second["capability"])
    monkeypatch.setattr(app_module, "RELEASE_GRANT_EVENT_PAGE_SIZE", 1)
    cursor = _events(client, first["grant_id"]).json()["next_cursor"]
    # The cursor is minted for first; replaying it against the sibling
    # grant in the same scope is a 422, not a page of the sibling's events.
    assert _events(client, second["grant_id"], cursor=cursor).status_code == 422


def test_foreign_cursor_families_are_422(client):
    from proof_release.app import (
        _encode_audit_event_cursor,
        _encode_cursor,
        _encode_grant_audit_cursor,
        _encode_revocation_cursor,
        _encode_rewrap_job_event_cursor,
        _encode_rewrap_job_history_cursor,
    )

    grant = _pending_grant(client)
    foreign_batch = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert (
        _events(client, grant["grant_id"], cursor=foreign_batch).status_code == 422
    )
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
    assert (
        _events(client, grant["grant_id"], cursor=foreign_grant_audit).status_code
        == 422
    )
    foreign_audit = _encode_audit_event_cursor(
        TENANT,
        WORKLOAD,
        "2026-01-01T00:00:00+00:00",
        ZERO_UUID,
        event_id="",
        event_type="",
        status="",
        occurred_after="",
        occurred_before="",
    )
    assert _events(client, grant["grant_id"], cursor=foreign_audit).status_code == 422
    foreign_revocation = _encode_revocation_cursor(
        TENANT,
        WORKLOAD,
        ZERO_UUID,
        "2026-01-01T00:00:00+00:00",
        ZERO_UUID,
        revocation_id="",
        certificate_fingerprint="",
        effective_after="",
        effective_before="",
        snapshot_at="2026-01-01T00:00:00+00:00",
        snapshot_id=ZERO_UUID,
    )
    assert (
        _events(client, grant["grant_id"], cursor=foreign_revocation).status_code == 422
    )
    foreign_job_events = _encode_rewrap_job_event_cursor(
        TENANT, WORKLOAD, ZERO_UUID, 1, snapshot_seq=1
    )
    assert (
        _events(client, grant["grant_id"], cursor=foreign_job_events).status_code == 422
    )
    foreign_job_history = _encode_rewrap_job_history_cursor(
        TENANT,
        WORKLOAD,
        ZERO_UUID,
        job_id="",
        status="",
        created_after="",
        created_before="",
    )
    assert (
        _events(client, grant["grant_id"], cursor=foreign_job_history).status_code == 422
    )


# --- read-only behaviour ---------------------------------------------------


def test_query_changes_nothing(app, client):
    grant = _pending_grant(client)
    with app.state.session_factory() as session:
        before_grant = session.get(ReleaseGrant, grant["grant_id"])
        before = (
            before_grant.status,
            before_grant.consumed_at,
            before_grant.revoked_at,
            session.query(ReleaseGrantEvent).count(),
        )

    for _ in range(3):
        assert _events(client, grant["grant_id"]).status_code == 200

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        after = (row.status, row.consumed_at, row.revoked_at,
                 session.query(ReleaseGrantEvent).count())
        assert after == before
        assert row.status == "pending"


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_page(app, client):
    grant = _pending_grant(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE release_grant_events"))
    assert _events(client, grant["grant_id"]).status_code == 500


# --- persistence -----------------------------------------------------------


def test_events_survive_restart_in_original_order(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/persist.db"
    first = create_app(url)
    first_client = TestClient(first)
    grant = _pending_grant(first_client)
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
        assert events[0]["old_status"] is None
    finally:
        second.state.engine.dispose()


# --- legacy database reconstruction ----------------------------------------


def test_legacy_database_is_backfilled_on_open(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/legacy.db"

    first = create_app(url)
    first_client = TestClient(first)
    consumed = _pending_grant(first_client, data_id="data-a")
    revoked = _pending_grant(first_client, data_id="data-b")
    pending = _pending_grant(first_client, data_id="data-c")
    assert (
        _consume(first_client, consumed["grant_id"], consumed["capability"]).status_code
        == 200
    )
    assert (
        _revoke(first_client, revoked["grant_id"], revoked["capability"]).status_code
        == 200
    )
    first.state.engine.dispose()

    # Simulate a database written before the timeline existed.
    from sqlalchemy import create_engine

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE release_grant_events"))
    engine.dispose()

    # Opening the old database reconstructs the gap-free per-grant history.
    second = create_app(url)
    try:
        second_client = TestClient(second)
        consumed_events = second_client.get(
            f"/v1/release-grants/{consumed['grant_id']}/events",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).json()
        assert _tuples(consumed_events["events"]) == [
            (1, None, "pending", "issued"),
            (2, "pending", "consumed", "consume"),
        ]
        assert consumed_events["complete"] is True
        revoked_events = second_client.get(
            f"/v1/release-grants/{revoked['grant_id']}/events",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).json()
        assert _tuples(revoked_events["events"]) == [
            (1, None, "pending", "issued"),
            (2, "pending", "revoked", "revoked"),
        ]
        pending_events = second_client.get(
            f"/v1/release-grants/{pending['grant_id']}/events",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).json()
        assert _tuples(pending_events["events"]) == [
            (1, None, "pending", "issued"),
        ]
        # New migrations continue the reconstructed sequence gap-free.
        assert (
            _consume(
                second_client, pending["grant_id"], pending["capability"]
            ).status_code
            == 200
        )
        continued = second_client.get(
            f"/v1/release-grants/{pending['grant_id']}/events",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).json()
        assert _tuples(continued["events"]) == [
            (1, None, "pending", "issued"),
            (2, "pending", "consumed", "consume"),
        ]
    finally:
        second.state.engine.dispose()

    # Reopening is idempotent: no duplicate reconstructed events.
    third = create_app(url)
    try:
        third_client = TestClient(third)
        for grant in (consumed, revoked, pending):
            data = third_client.get(
                f"/v1/release-grants/{grant['grant_id']}/events",
                params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            ).json()
            assert [e["seq"] for e in data["events"]] == [1, 2]
    finally:
        third.state.engine.dispose()


# --- secrecy ---------------------------------------------------------------


def test_timeline_never_exposes_capability_or_payload(app, client):
    _envelope(client)
    grant = _pending_grant(client)
    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 200
    timeline = _events(client, grant["grant_id"])
    for secret in (
        grant["capability"],
        PAYLOAD,
        MASTER_KEY,
        "capability",
        "wrapped_key",
        "ciphertext",
        "payload",
        "master_key",
    ):
        assert secret not in timeline.text
    with app.state.session_factory() as session:
        for event in session.query(ReleaseGrantEvent).all():
            columns = {
                c.name: getattr(event, c.name) for c in event.__table__.columns
            }
            rendered = str(columns)
            assert grant["capability"] not in rendered
            assert PAYLOAD not in rendered
            assert not hasattr(event, "capability_digest")
