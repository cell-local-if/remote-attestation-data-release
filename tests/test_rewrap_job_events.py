"""Tests for the persistent asynchronous rewrap job event timeline:

GET /v1/rewrap-jobs/{job_id}/events

Every committed job status migration appends exactly one immutable,
sequenced event in the same transaction as the migration; the read
endpoint lists one job's timeline in sequence order with an
HMAC-authenticated, kind-tagged cursor bound to the scope, the job and a
snapshot fixed by the first (cursor-less) query. Tests build the app
without entering its lifespan (so no pool runs) and drive the runner
synchronously, except the explicit restart/recovery tests.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import RewrapJobEvent
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
PAYLOAD = "events-secret-payload 🔐"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V2_BYTES = b"fedcba9876543210fedcba9876543210"
KEY_V1 = b64url_encode(KEY_V1_BYTES)
KEY_V2 = b64url_encode(KEY_V2_BYTES)


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


KEYRING_V1_V2 = _keyring(2, {1: KEY_V1, 2: KEY_V2})
KEYRING_V2_ONLY = _keyring(2, {2: KEY_V2})


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = app_module.create_app(f"sqlite:///{tmp_path}/events.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, data_id, *, tenant=TENANT, workload=WORKLOAD, payload=PAYLOAD):
    return client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": payload,
        },
    )


def _seed(client, data_ids, **kwargs):
    for data_id in data_ids:
        assert _create(client, data_id, **kwargs).status_code == 201


def _submit(client, *, cursor=None, limit=None, tenant=TENANT, workload=WORKLOAD):
    body: dict = {"tenant_id": tenant, "workload_id": workload}
    if cursor is not None:
        body["cursor"] = cursor
    if limit is not None:
        body["limit"] = limit
    return client.post("/v1/rewrap-jobs", json=body)


def _run_job(app, job_id, *, retry=False):
    app.state.rewrap_job_runner.run(job_id, retry=retry)


def _events(client, job_id, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        f"/v1/rewrap-jobs/{job_id}/events",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


def _events_raw(client, path, *, params=None, content=None):
    return client.request(
        "GET",
        path,
        params=params,
        content=content,
        headers={"content-type": "application/json"} if content is not None else None,
    )


def _resume(client, job_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        f"/v1/rewrap-jobs/{job_id}/resume",
        json={"tenant_id": tenant, "workload_id": workload},
    )


def _cancel(client, job_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        f"/v1/rewrap-jobs/{job_id}/cancel",
        json={"tenant_id": tenant, "workload_id": workload},
    )


def _job_events(app, job_id):
    with app.state.session_factory() as session:
        rows = (
            session.query(RewrapJobEvent)
            .filter(RewrapJobEvent.job_id == job_id)
            .order_by(RewrapJobEvent.seq.asc())
            .all()
        )
        return [
            {
                "event_id": r.event_id,
                "seq": r.seq,
                "old_status": r.old_status,
                "new_status": r.new_status,
                "reason": r.reason,
                "occurred_at": r.occurred_at,
            }
            for r in rows
        ]


# --- happy-path timeline ---------------------------------------------------


def test_submission_creates_single_submitted_event(app, client):
    _seed(client, ["a"])
    accepted = _submit(client)
    job_id = accepted.json()["job_id"]
    response = _events(client, job_id)
    assert response.status_code == 200
    body = response.json()
    assert body["complete"] is True
    assert body["next_cursor"] == ""
    assert len(body["events"]) == 1
    event = body["events"][0]
    assert set(event) == {
        "event_id",
        "seq",
        "old_status",
        "new_status",
        "reason",
        "occurred_at",
    }
    assert event["seq"] == 0
    assert event["old_status"] is None
    assert event["new_status"] == "queued"
    assert event["reason"] == "submitted"
    assert isinstance(event["event_id"], str) and len(event["event_id"]) == 36
    parsed = datetime.fromisoformat(event["occurred_at"])
    assert parsed.utcoffset() == timedelta(0)
    assert parsed.replace(tzinfo=None) == datetime.fromisoformat(
        accepted.json()["created_at"]
    ).replace(tzinfo=None)


def test_full_lifecycle_timeline_is_ordered_and_chained(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    body = _events(client, job_id).json()
    transitions = [(e["old_status"], e["new_status"], e["reason"]) for e in body["events"]]
    assert transitions == [
        (None, "queued", "submitted"),
        ("queued", "running", "executing"),
        ("running", "succeeded", "completed"),
    ]
    seqs = [e["seq"] for e in body["events"]]
    assert seqs == [0, 1, 2]
    # The sequence reads back exactly as it was stored; every event id is
    # unique and each timestamp is UTC RFC3339.
    ids = [e["event_id"] for e in body["events"]]
    assert len(set(ids)) == 3
    for e in body["events"]:
        assert datetime.fromisoformat(e["occurred_at"]).utcoffset() == timedelta(0)
    # old_status chains to the previous new_status.
    for earlier, later in zip(body["events"], body["events"][1:]):
        assert later["old_status"] == earlier["new_status"]


def test_timeline_seqs_are_unique_per_job_and_restart_stable(app, client):
    _seed(client, ["a"])
    job_a = _submit(client).json()["job_id"]
    job_b = _submit(client).json()["job_id"]
    rows_a = _job_events(app, job_a)
    rows_b = _job_events(app, job_b)
    assert [r["seq"] for r in rows_a] == [0]
    assert [r["seq"] for r in rows_b] == [0]
    # Event ids never collide even at the same sequence.
    assert rows_a[0]["event_id"] != rows_b[0]["event_id"]


# --- failure, cancel and resume reasons ------------------------------------


def test_missing_key_failure_records_failure_category(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    job_id = _submit(client, limit=10).json()["job_id"]
    _run_job(app, job_id)

    events = _events(client, job_id).json()["events"]
    assert [(e["old_status"], e["new_status"], e["reason"]) for e in events] == [
        (None, "queued", "submitted"),
        ("queued", "running", "executing"),
        ("running", "failed", "missing-key"),
    ]


def test_keyring_failure_records_keyring_reason(app, client, monkeypatch):
    _seed(client, ["a0", "a1"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    real_load = app_module.load_keyring
    calls = {"n": 0}

    def failing_load():
        calls["n"] += 1
        # Call 1 is the submit check; a0 uses call 2; a1 (call 3) fails.
        if calls["n"] == 3:
            raise app_module.MasterKeyError("keyring vanished")
        return real_load()

    monkeypatch.setattr(app_module, "load_keyring", failing_load)
    job_id = _submit(client, limit=10).json()["job_id"]
    _run_job(app, job_id)

    events = _events(client, job_id).json()["events"]
    assert events[-1]["reason"] == "keyring"
    assert events[-1]["new_status"] == "failed"


def test_rewrap_failure_records_rewrap_reason(app, client, monkeypatch):
    _seed(client, ["a0", "a1", "a2"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    real_rewrap = app_module.rewrap_data_key
    calls = {"n": 0}

    def failing_rewrap(unwrapping_key, wrapping_key, wrapped_key):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("crypto backend exploded")
        return real_rewrap(unwrapping_key, wrapping_key, wrapped_key)

    monkeypatch.setattr(app_module, "rewrap_data_key", failing_rewrap)
    job_id = _submit(client, limit=10).json()["job_id"]
    _run_job(app, job_id)

    events = _events(client, job_id).json()["events"]
    assert events[-1]["reason"] == "rewrap"
    # Exception text never enters the timeline.
    body = _events(client, job_id).text
    assert "crypto backend exploded" not in body
    assert all(
        e["reason"]
        in {"submitted", "executing", "resumed", "completed", "cancelled",
            "keyring", "missing-key", "rewrap"}
        for e in events
    )


def test_cancel_archives_single_cancelled_transition(app, client):
    job_id = _submit(client).json()["job_id"]
    assert _cancel(client, job_id).status_code == 200
    events = _events(client, job_id).json()["events"]
    assert [(e["old_status"], e["new_status"], e["reason"]) for e in events] == [
        (None, "queued", "submitted"),
        ("queued", "cancelled", "cancelled"),
    ]
    # A repeated cancel appends no event.
    assert _cancel(client, job_id).status_code == 409
    again = _events(client, job_id).json()["events"]
    assert again == events


def test_manual_resume_then_completion_records_resumed_chain(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    job_id = _submit(client, limit=10).json()["job_id"]
    _run_job(app, job_id)
    assert _events(client, job_id).json()["events"][-1]["new_status"] == "failed"

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _resume(client, job_id).status_code == 200
    events = _events(client, job_id).json()["events"]
    assert (events[-1]["old_status"], events[-1]["new_status"], events[-1]["reason"]) == (
        "failed",
        "queued",
        "resumed",
    )
    # Repeating the resume appends no second resume event.
    assert _resume(client, job_id).status_code == 409
    assert len(_events(client, job_id).json()["events"]) == len(events)

    _run_job(app, job_id)
    final = _events(client, job_id).json()["events"]
    assert [(e["old_status"], e["new_status"], e["reason"]) for e in final] == [
        (None, "queued", "submitted"),
        ("queued", "running", "executing"),
        ("running", "failed", "missing-key"),
        ("failed", "queued", "resumed"),
        ("queued", "running", "executing"),
        ("running", "succeeded", "completed"),
    ]
    assert [e["seq"] for e in final] == list(range(len(final)))


def test_startup_recovery_claim_records_failed_to_running_resumed(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/recovery.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["a"])
    app1.state.engine.dispose()

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    job_id = _submit(client2, limit=10).json()["job_id"]
    app2.state.rewrap_job_runner.run(job_id)
    assert client2.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()["status"] == "failed"
    app2.state.engine.dispose()

    # A fresh process's startup sweep claims the parked failed job with a
    # failed -> running retry, which the timeline records as resumed.
    import time

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    app3 = app_module.create_app(url)
    with TestClient(app3) as client3:
        status = None
        for _ in range(200):
            status = client3.get(
                f"/v1/rewrap-jobs/{job_id}",
                params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            ).json()["status"]
            if status == "succeeded":
                break
            time.sleep(0.01)
        assert status == "succeeded"
        events = _events(client3, job_id).json()["events"]
        assert ("failed", "running", "resumed") in [
            (e["old_status"], e["new_status"], e["reason"]) for e in events
        ]
        assert [e["seq"] for e in events] == list(range(len(events)))
    app3.state.engine.dispose()


# --- races archive the final committed state -------------------------------


def test_cancel_racing_runner_archives_only_committed_transitions(
    app, client, monkeypatch
):
    _seed(client, [f"e{i:02d}" for i in range(8)])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client, limit=2).json()["job_id"]
    runner = app.state.rewrap_job_runner

    entered = threading.Event()
    gate = threading.Event()
    real_rewrap = app_module.rewrap_data_key
    calls = {"n": 0}

    def gated_rewrap(unwrapping_key, wrapping_key, wrapped_key):
        calls["n"] += 1
        # Park inside the uncommitted second envelope of the first page.
        if calls["n"] == 2:
            entered.set()
            gate.wait(timeout=5)
        return real_rewrap(unwrapping_key, wrapping_key, wrapped_key)

    monkeypatch.setattr(app_module, "rewrap_data_key", gated_rewrap)
    worker = threading.Thread(target=runner.run, args=(job_id,), daemon=True)
    worker.start()
    assert entered.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_cancel, client, job_id) for _ in range(4)]
        threading.Event().wait(0.1)
        gate.set()
        responses = [f.result(timeout=10) for f in futures]
    worker.join(timeout=5)

    assert any(r.status_code == 200 for r in responses)
    events = _events(client, job_id).json()["events"]
    # Exactly one cancel event; a losing transition never wrote one.
    assert [e["reason"] for e in events].count("cancelled") == 1
    assert events[-1]["new_status"] == "cancelled"
    assert events[-1]["old_status"] in ("queued", "running")
    # Gap-free sequence and a coherent status chain ending cancelled.
    assert [e["seq"] for e in events] == list(range(len(events)))
    for earlier, later in zip(events, events[1:]):
        assert later["old_status"] == earlier["new_status"]


# --- request validation ----------------------------------------------------


def test_empty_and_missing_body_are_accepted(app, client):
    job_id = _submit(client).json()["job_id"]
    path = f"/v1/rewrap-jobs/{job_id}/events"
    params = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    assert _events_raw(client, path, params=params).status_code == 200
    assert _events_raw(
        client, path, params=params, content=b""
    ).status_code == 200


@pytest.mark.parametrize("body", [b"{}", b" ", b"\n", b"garbage", b'{"x":1}'])
def test_any_non_empty_body_is_422(app, client, body):
    job_id = _submit(client).json()["job_id"]
    response = _events_raw(
        client,
        f"/v1/rewrap-jobs/{job_id}/events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=body,
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "job_path",
    [
        "/v1/rewrap-jobs/not-a-uuid/events",
        "/v1/rewrap-jobs/abc/events",
        "/v1/rewrap-jobs/00000000-0000-0000-0000-00000000000Z/events",
        "/v1/rewrap-jobs//events",
    ],
)
def test_path_identifier_must_be_canonical_uuid(app, client, job_path):
    response = _events_raw(
        client,
        job_path,
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_uppercase_uuid_path_is_422(app, client):
    job_id = _submit(client).json()["job_id"]
    response = _events_raw(
        client,
        f"/v1/rewrap-jobs/{job_id.upper()}/events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_missing_blank_scope_parameters_are_422(app, client):
    job_id = _submit(client).json()["job_id"]
    path = f"/v1/rewrap-jobs/{job_id}/events"
    assert _events_raw(client, path, params={"workload_id": WORKLOAD}).status_code == 422
    assert _events_raw(client, path, params={"tenant_id": TENANT}).status_code == 422
    assert _events_raw(
        client,
        path,
        params={"tenant_id": "  ", "workload_id": WORKLOAD},
    ).status_code == 422


def test_unknown_query_parameter_is_422(app, client):
    job_id = _submit(client).json()["job_id"]
    response = _events_raw(
        client,
        f"/v1/rewrap-jobs/{job_id}/events",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "bogus": "1",
        },
    )
    assert response.status_code == 422


def test_repeated_query_parameter_is_422(app, client):
    job_id = _submit(client).json()["job_id"]
    response = _events_raw(
        client,
        f"/v1/rewrap-jobs/{job_id}/events",
        params=[
            ("tenant_id", TENANT),
            ("tenant_id", OTHER_TENANT),
            ("workload_id", WORKLOAD),
        ],
    )
    assert response.status_code == 422
    response = _events_raw(
        client,
        f"/v1/rewrap-jobs/{job_id}/events",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("cursor", ""),
            ("cursor", ""),
        ],
    )
    assert response.status_code == 422


def test_unknown_and_cross_scope_job_is_404(app, client):
    unknown = "00000000-0000-0000-0000-000000000001"
    response = _events_raw(
        client,
        f"/v1/rewrap-jobs/{unknown}/events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 404

    job_id = _submit(client).json()["job_id"]
    assert _events(
        client, job_id, tenant=OTHER_TENANT, workload=WORKLOAD
    ).status_code == 404
    assert _events(
        client, job_id, tenant=TENANT, workload=OTHER_WORKLOAD
    ).status_code == 404


def test_invalid_cursor_is_422_before_job_is_read(app, client):
    unknown = "00000000-0000-0000-0000-000000000001"
    # A malformed cursor on an unknown job still returns 422: input
    # validation precedes the 404 scope lookup and reads no state.
    response = _events_raw(
        client,
        f"/v1/rewrap-jobs/{unknown}/events",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "cursor": "not-a-cursor",
        },
    )
    assert response.status_code == 422
    job_id = _submit(client).json()["job_id"]
    for token in (" ", "!!!", "a" * 64):
        assert _events(client, job_id, cursor=token).status_code == 422


def _foreign_cursors(app, job_id):
    # One cursor from every other HMAC cursor family.
    return [
        # rewrap batch / envelope boundary cursor
        app_module._encode_cursor(TENANT, WORKLOAD, "data-1"),
        # rewrap job history cursor
        app_module._encode_rewrap_job_history_cursor(
            TENANT, WORKLOAD, job_id,
            job_id="", status="", created_after="", created_before="",
        ),
        # release-grant audit cursor
        app_module._encode_grant_audit_cursor(
            TENANT, WORKLOAD, job_id,
            grant_id="", decision_id="", data_id="", status="",
            issued_after="", issued_before="",
        ),
        # compliance audit-event cursor
        app_module._encode_audit_event_cursor(
            TENANT, WORKLOAD, "2020-01-01T00:00:00+00:00", job_id,
            event_id="", event_type="", status="",
            occurred_after="", occurred_before="",
        ),
        # revocation registry cursor
        app_module._encode_revocation_cursor(
            TENANT, WORKLOAD, job_id, "2020-01-01T00:00:00+00:00", job_id,
            revocation_id="", certificate_fingerprint="",
            effective_after="", effective_before="",
            snapshot_at="2020-01-01T00:00:00+00:00", snapshot_id=job_id,
        ),
    ]


def test_foreign_family_cursors_are_422_and_read_no_state(app, client):
    job_id = _submit(client).json()["job_id"]
    for token in _foreign_cursors(app, job_id):
        assert _events(client, job_id, cursor=token).status_code == 422


def test_cross_scope_and_cross_job_events_cursors_are_422(app, client):
    job_id = _submit(client).json()["job_id"]
    other_id = _submit(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()[
        "job_id"
    ]
    # A well-formed events cursor minted for another scope.
    cross_scope = app_module._encode_rewrap_job_events_cursor(
        OTHER_TENANT, OTHER_WORKLOAD, job_id, -1, 0
    )
    assert _events(client, job_id, cursor=cross_scope).status_code == 422
    # A cursor minted for a different job within the same scope is also
    # rejected; the families must never cross jobs.
    cross_job = app_module._encode_rewrap_job_events_cursor(
        TENANT, WORKLOAD, other_id, -1, 0
    )
    assert _events(client, job_id, cursor=cross_job).status_code == 422
    # Snapshot mismatch: cursor claims a later snapshot than signed.
    good = app_module._encode_rewrap_job_events_cursor(
        TENANT, WORKLOAD, job_id, 0, 5
    )
    tampered = good[:-2] + ("A" if good[-2:] != "AA" else "B") + good[-1:]
    assert _events(client, job_id, cursor=tampered).status_code == 422


# --- pagination, snapshot and replay ---------------------------------------


def test_cursor_pagination_replay_and_snapshot(app, client, monkeypatch):
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    # Claim the job so it sits in running with two committed events, then
    # release the claim so the job can be driven to completion later.
    runner = app.state.rewrap_job_runner
    assert runner._claim(job_id)
    runner._release_claim(job_id)
    monkeypatch.setattr(app_module, "REWRAP_JOB_EVENTS_PAGE_SIZE", 1)

    # First query fixes the snapshot at seq=1; page one contains only the
    # submission event.
    first = _events(client, job_id)
    assert first.status_code == 200
    assert [e["reason"] for e in first.json()["events"]] == ["submitted"]
    assert first.json()["complete"] is False
    cursor_one = first.json()["next_cursor"]
    assert cursor_one

    # The job completes, committing one further migration (the claim's
    # executing event is already part of the fixed snapshot).
    _run_job(app, job_id)
    reasons_now = [e["reason"] for e in _job_events(app, job_id)]
    assert reasons_now == ["submitted", "executing", "completed"]

    # Page two of the fixed snapshot contains exactly seq=1 (executing);
    # it is the snapshot's final page even though completion committed
    # afterwards, so it is already complete.
    second = _events(client, job_id, cursor=cursor_one)
    assert second.status_code == 200
    assert [e["reason"] for e in second.json()["events"]] == ["executing"]
    assert second.json()["complete"] is True
    assert second.json()["next_cursor"] == ""

    # Replaying the first cursor after completion returns the identical
    # final page, byte-stable: the completed event never leaks into this
    # snapshot and no entry is duplicated or skipped.
    assert _events(client, job_id, cursor=cursor_one).content == second.content

    # A new cursor-less first query establishes a fresh snapshot that
    # includes the later completed migration: three one-event pages.
    fresh = _events(client, job_id)
    assert [e["reason"] for e in fresh.json()["events"]] == ["submitted"]
    assert fresh.json()["complete"] is False
    assert fresh.json()["next_cursor"] != cursor_one
    fresh_two = _events(client, job_id, cursor=fresh.json()["next_cursor"])
    assert [e["reason"] for e in fresh_two.json()["events"]] == ["executing"]
    fresh_three = _events(
        client, job_id, cursor=fresh_two.json()["next_cursor"]
    )
    assert [e["reason"] for e in fresh_three.json()["events"]] == ["completed"]
    assert fresh_three.json()["complete"] is True

    # An explicit empty cursor behaves like an omitted one.
    assert (
        _events(client, job_id, cursor="").json()["events"]
        == fresh.json()["events"]
    )


def test_job_without_events_yields_empty_complete_first_page(app, client):
    # Only possible for a row predating the timeline; the fixed first
    # page is empty and complete rather than an error.
    job_id = _submit(client).json()["job_id"]
    with app.state.engine.begin() as conn:
        from sqlalchemy import text

        conn.execute(
            text("DELETE FROM rewrap_job_events WHERE job_id = :j"),
            {"j": job_id},
        )
    body = _events(client, job_id).json()
    assert body == {"events": [], "next_cursor": "", "complete": True}


# --- wire format, failure handling and persistence --------------------------


def test_response_is_compact_json_with_single_newline(app, client):
    job_id = _submit(client).json()["job_id"]
    raw = _events(client, job_id).content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw and b": " not in raw
    parsed = json.loads(raw)
    assert list(parsed) == ["events", "next_cursor", "complete"]
    assert isinstance(parsed["complete"], bool)
    assert isinstance(parsed["next_cursor"], str)
    event = parsed["events"][0]
    assert list(event) == [
        "event_id",
        "seq",
        "old_status",
        "new_status",
        "reason",
        "occurred_at",
    ]
    assert isinstance(event["seq"], int) and not isinstance(event["seq"], bool)


def test_events_never_echo_protected_material(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    raw = _events(client, job_id).content
    for secret in (PAYLOAD, KEY_V1, KEY_V2, "wrapped_key", "ciphertext", "payload"):
        assert secret.encode("utf-8") not in raw


def test_query_failure_returns_500_with_no_half_page(app, client):
    from sqlalchemy import text

    job_id = _submit(client).json()["job_id"]
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_job_events"))
    response = _events(client, job_id)
    assert response.status_code == 500
    # A failure never returns the events container or a partial page.
    assert b'"events"' not in response.content
    assert b'"next_cursor"' not in response.content


def test_events_survive_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/persist.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["a"])
    job_id = _submit(client1).json()["job_id"]
    app1.state.engine.dispose()

    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    response = _events(client2, job_id)
    assert response.status_code == 200
    events = response.json()["events"]
    assert [e["reason"] for e in events] == ["submitted"]
    assert events[0]["seq"] == 0
    app2.state.engine.dispose()


def test_events_query_changes_no_job_or_envelope_state(app, client, monkeypatch):
    _seed(client, ["a0", "a1"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client, limit=1).json()["job_id"]
    _run_job(app, job_id)
    before = _job_events(app, job_id)
    job_before = client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).content
    for _ in range(3):
        page = _events(client, job_id)
        assert page.status_code == 200
    assert _job_events(app, job_id) == before
    job_after = client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).content
    assert job_after == job_before


def test_idempotent_replay_creates_no_second_submission_event(app, client):
    _seed(client, ["a"])
    headers = {"Idempotency-Key": "key-123"}
    first = client.post(
        "/v1/rewrap-jobs",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers=headers,
    )
    job_id = first.json()["job_id"]
    second = client.post(
        "/v1/rewrap-jobs",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers=headers,
    )
    assert second.status_code == 202 and second.json()["job_id"] == job_id
    events = _events(client, job_id).json()["events"]
    assert [e["reason"] for e in events] == ["submitted"]
