"""Tests for the persistent asynchronous rewrap job event timeline:

GET /v1/rewrap-jobs/{job_id}/events

The endpoint ranges exactly one job by mandatory tenant/workload scope and
a canonical-UUID path id; it accepts only tenant_id, workload_id and an
optional cursor, carries no body, and returns the job's committed status
migrations as stable, immutable per-job sequence events. The first
(cursor-less) query fixes a replayable snapshot high-water mark; later
commits surface only in a fresh first query. A resume cursor is
HMAC-authenticated and bound to the scope, job and snapshot, so it cannot
be forged, tampered with, or replayed across scopes, jobs, snapshots or
cursor families. The query is strictly read-only.

Like the other job tests, the app is built without entering its lifespan
so no pool runs; jobs (and the post-restart retry claim) are driven
synchronously.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import RewrapJob, RewrapJobEvent
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"
LETTERED_UUID = "abcdefab-cdef-abcd-efab-cdefabcdefab"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V2_BYTES = b"fedcba9876543210fedcba9876543210"
KEY_V1 = b64url_encode(KEY_V1_BYTES)
KEY_V2 = b64url_encode(KEY_V2_BYTES)

PAYLOAD = "timeline-secret 🔐"


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


KEYRING_V1_V2 = _keyring(2, {1: KEY_V1, 2: KEY_V2})
KEYRING_V2_ONLY = _keyring(2, {2: KEY_V2})

EVENT_FIELDS = [
    "event_id",
    "job_seq",
    "old_status",
    "new_status",
    "reason",
    "created_at",
]


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


def _create(client, data_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": PAYLOAD,
        },
    )


def _seed(client, data_ids, **kwargs):
    for data_id in data_ids:
        assert _create(client, data_id, **kwargs).status_code == 201


def _submit(client, *, tenant=TENANT, workload=WORKLOAD, limit=None):
    body: dict = {"tenant_id": tenant, "workload_id": workload}
    if limit is not None:
        body["limit"] = limit
    return client.post("/v1/rewrap-jobs", json=body)


def _events(client, job_id, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        f"/v1/rewrap-jobs/{job_id}/events",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


def _run_job(app, job_id, *, retry=False):
    app.state.rewrap_job_runner.run(job_id, retry=retry)


def _cancel(client, job_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        f"/v1/rewrap-jobs/{job_id}/cancel",
        json={"tenant_id": tenant, "workload_id": workload},
    )


def _resume(client, job_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        f"/v1/rewrap-jobs/{job_id}/resume",
        json={"tenant_id": tenant, "workload_id": workload},
    )


def _fail_job(app, client, monkeypatch, *, tenant=TENANT, workload=WORKLOAD):
    # Envelopes are sealed under the v1 master key; running with a
    # current-v2 keyring that omits v1 fails the job on the first
    # envelope. The working keyring is restored afterwards.
    _seed(client, ["a"], tenant=tenant, workload=workload)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    job_id = _submit(client, tenant=tenant, workload=workload).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    return job_id


def _walk(client, app, job_id, page_size, monkeypatch, **params):
    monkeypatch.setattr(app_module, "REWRAP_JOB_EVENT_PAGE_SIZE", page_size)
    rows = []
    token = None
    for _ in range(100):
        query = dict(params)
        if token:
            query["cursor"] = token
        data = _events(client, job_id, **query).json()
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
    response = client.get(f"/v1/rewrap-jobs/{ZERO_UUID}/events")
    assert response.status_code == 422


@pytest.mark.parametrize(
    "job_id",
    [
        "not-a-uuid",
        "abc123",
        ZERO_UUID[:-1] + "Z",
        LETTERED_UUID.upper(),
        "%20" + ZERO_UUID,
    ],
)
def test_malformed_job_identifier_is_422(client, job_id):
    assert _events(client, job_id).status_code == 422


def test_empty_path_segment_is_422(client):
    response = client.get(
        "/v1/rewrap-jobs//events",
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
        {"status": "queued"},
        {"event_id": ZERO_UUID},
    ],
)
def test_unknown_or_malformed_parameters_are_422(client, params):
    assert _events(client, ZERO_UUID, **params).status_code == 422


def test_duplicate_parameter_is_422(client):
    response = client.get(
        f"/v1/rewrap-jobs/{ZERO_UUID}/events",
        params=[
            ("tenant_id", TENANT),
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
        ],
    )
    assert response.status_code == 422
    response = client.get(
        f"/v1/rewrap-jobs/{ZERO_UUID}/events",
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
        f"/v1/rewrap-jobs/{ZERO_UUID}/events",
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
        assert session.query(RewrapJob).count() == 0
        assert session.query(RewrapJobEvent).count() == 0


# --- 404 semantics ---------------------------------------------------------


def test_unknown_job_returns_404(client):
    assert _events(client, ZERO_UUID).status_code == 404


def test_cross_scope_job_returns_404(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    assert _events(client, job_id, tenant=OTHER_TENANT).status_code == 404
    assert _events(client, job_id, workload=OTHER_WORKLOAD).status_code == 404
    assert (
        _events(
            client, job_id, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
        ).status_code
        == 404
    )


def test_422_precedes_404(client):
    # An illegal cursor is rejected before the unknown job is resolved.
    response = _events(client, ZERO_UUID, cursor="tampered-token")
    assert response.status_code == 422
    response = _events(client, "not-a-uuid")
    assert response.status_code == 422


# --- recorded timeline -----------------------------------------------------


def test_succeeded_job_records_submitted_executed_completed(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    data = _events(client, job_id).json()
    assert data["next_cursor"] == ""
    assert data["complete"] is True
    events = data["events"]
    assert [(e["job_seq"], e["reason"], e["old_status"], e["new_status"]) for e in events] == [
        (1, "submitted", None, "queued"),
        (2, "executed", "queued", "running"),
        (3, "completed", "running", "succeeded"),
    ]
    # Sequence numbers are gap-free integers from 1, ids unique UUIDs.
    assert [e["job_seq"] for e in events] == [1, 2, 3]
    assert len({e["event_id"] for e in events}) == 3
    for event in events:
        assert list(event) == EVENT_FIELDS
        assert isinstance(event["event_id"], str) and len(event["event_id"]) == 36
        assert isinstance(event["job_seq"], int) and not isinstance(
            event["job_seq"], bool
        )
        parsed = datetime.fromisoformat(event["created_at"])
        assert parsed.utcoffset() == timedelta(0)


def test_queued_job_has_only_the_submission_event(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    events = _events(client, job_id).json()["events"]
    assert [(e["job_seq"], e["reason"], e["old_status"], e["new_status"]) for e in events] == [
        (1, "submitted", None, "queued"),
    ]


def test_cancel_of_queued_job_records_cancelled_from_queued(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    assert _cancel(client, job_id).status_code == 200

    events = _events(client, job_id).json()["events"]
    assert [(e["job_seq"], e["old_status"], e["new_status"], e["reason"]) for e in events] == [
        (1, None, "queued", "submitted"),
        (2, "queued", "cancelled", "cancelled"),
    ]


def test_cancel_of_running_job_records_cancelled_from_running(
    app, client, monkeypatch
):
    import threading

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    claimed = threading.Event()
    release = threading.Event()
    app.state.rewrap_job_runner.post_claim = lambda _: (
        claimed.set(),
        release.wait(timeout=5),
    )
    job_id = _submit(client).json()["job_id"]
    runner = threading.Thread(target=_run_job, args=(app, job_id))
    runner.start()
    try:
        assert claimed.wait(timeout=5)
        assert _cancel(client, job_id).status_code == 200
    finally:
        release.set()
        runner.join(timeout=5)

    events = _events(client, job_id).json()["events"]
    assert [(e["old_status"], e["new_status"], e["reason"]) for e in events] == [
        (None, "queued", "submitted"),
        ("queued", "running", "executed"),
        ("running", "cancelled", "cancelled"),
    ]


def test_missing_key_failure_is_classified_not_exception_text(app, client, monkeypatch):
    # Seed under the v1-only fixture key (no keyring env), then run with a
    # v2-only keyring that has lost the historical v1 key.
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    events = _events(client, job_id).json()["events"]
    terminal = events[-1]
    assert terminal["old_status"] == "running"
    assert terminal["new_status"] == "failed"
    assert terminal["reason"] == "missing-key"
    # Reasons are fixed codes only; no exception text ever appears.
    allowed = {
        "submitted",
        "executed",
        "recovered",
        "completed",
        "cancelled",
        "keyring",
        "missing-key",
        "rewrap",
    }
    assert all(e["reason"] in allowed for e in events)


def test_unusable_keyring_failure_is_classified(app, client, monkeypatch):
    from proof_release.envelopes import MasterKeyError

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]

    real_load = app_module.load_keyring

    def failing_load():
        raise MasterKeyError("keyring vanished with secret detail")

    monkeypatch.setattr(app_module, "load_keyring", failing_load)
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "load_keyring", real_load)

    events = _events(client, job_id).json()["events"]
    assert events[-1]["reason"] == "keyring"
    assert "vanished" not in json.dumps(events)
    assert "secret detail" not in json.dumps(events)


def test_rewrap_failure_is_classified(app, client, monkeypatch):
    # Envelopes are v1 under the fixture key; the run sees a v1/v2
    # keyring and actually attempts the rewrap, which the patched
    # primitive fails.
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]

    def boom(*args, **kwargs):
        raise ValueError("crypto stack trace with internal detail")

    monkeypatch.setattr(app_module, "rewrap_data_key", boom)
    _run_job(app, job_id)

    events = _events(client, job_id).json()["events"]
    assert events[-1]["reason"] == "rewrap"
    assert "crypto stack trace" not in json.dumps(events)


def test_manual_resume_records_recovered_then_executed_completed(
    app, client, monkeypatch
):
    job_id = _fail_job(app, client, monkeypatch=monkeypatch)
    assert _resume(client, job_id).status_code == 200
    _run_job(app, job_id)

    events = _events(client, job_id).json()["events"]
    assert [(e["job_seq"], e["old_status"], e["new_status"], e["reason"]) for e in events] == [
        (1, None, "queued", "submitted"),
        (2, "queued", "running", "executed"),
        (3, "running", "failed", "missing-key"),
        (4, "failed", "queued", "recovered"),
        (5, "queued", "running", "executed"),
        (6, "running", "succeeded", "completed"),
    ]


def test_post_restart_retry_claim_records_recovered_from_failed(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = app_module.create_app(url)
    first_client = TestClient(first)
    _seed(first_client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    job_id = _submit(first_client).json()["job_id"]
    _run_job(first, job_id)
    first.state.engine.dispose()

    # A fresh process starts with both keys present; the startup sweep's
    # retry claim is replayed synchronously as run(retry=True).
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    second = app_module.create_app(url)
    try:
        second_client = TestClient(second)
        second.state.rewrap_job_runner.run(job_id, retry=True)
        events = second_client.get(
            f"/v1/rewrap-jobs/{job_id}/events",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).json()["events"]
        assert [(e["old_status"], e["new_status"], e["reason"]) for e in events[3:]] == [
            ("failed", "running", "recovered"),
            ("running", "succeeded", "completed"),
        ]
        # A failed -> queued resume event is absent on the sweep path:
        # the sweep claims failed -> running directly.
        assert not any(
            e["old_status"] == "failed" and e["new_status"] == "queued" for e in events
        )
    finally:
        second.state.engine.dispose()


def test_losing_guard_race_leaves_no_event(app, client, monkeypatch):
    # A repeated cancel is a 409 and appends no second cancellation event.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    assert _cancel(client, job_id).status_code == 200
    assert _cancel(client, job_id).status_code == 409
    events = _events(client, job_id).json()["events"]
    assert [e["reason"] for e in events] == ["submitted", "cancelled"]


# --- wire format -----------------------------------------------------------


def test_response_is_compact_json_with_single_trailing_newline(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    response = _events(client, job_id)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == ["events", "next_cursor", "complete"]


def test_empty_string_cursor_equals_default(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    assert _events(client, job_id).content == _events(client, job_id, cursor="").content


def test_no_floats_or_non_finite_values(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

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
        if isinstance(value, dict):  # pragma: no cover - structural guard
            for item in value.values():
                _check(item)
            return
        raise AssertionError(f"unexpected type: {type(value)!r}")  # pragma: no cover

    _check(json.loads(_events(client, job_id).content))


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_event_once_in_order(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    rows = _walk(client, app, job_id, 1, monkeypatch)
    assert [e["job_seq"] for e in rows] == [1, 2, 3]
    assert [e["new_status"] for e in rows] == ["queued", "running", "succeeded"]


def test_page_carries_cursor_and_complete_flag(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "REWRAP_JOB_EVENT_PAGE_SIZE", 1)

    first = _events(client, job_id).json()
    assert [e["job_seq"] for e in first["events"]] == [1]
    assert first["complete"] is False and first["next_cursor"]

    second = _events(client, job_id, cursor=first["next_cursor"]).json()
    assert [e["job_seq"] for e in second["events"]] == [2]
    assert second["complete"] is False

    last = _events(client, job_id, cursor=second["next_cursor"]).json()
    assert [e["job_seq"] for e in last["events"]] == [3]
    assert last["complete"] is True and last["next_cursor"] == ""


def test_replaying_same_cursor_returns_identical_page(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "REWRAP_JOB_EVENT_PAGE_SIZE", 1)
    cursor = _events(client, job_id).json()["next_cursor"]
    first = _events(client, job_id, cursor=cursor)
    second = _events(client, job_id, cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


# --- snapshot isolation ----------------------------------------------------


def test_first_query_fixes_snapshot_for_later_pages(app, client, monkeypatch):
    # Fail the job (3 events) and open the timeline with page size 2: the
    # first query fixes the snapshot high-water mark at seq 3.
    job_id = _fail_job(app, client, monkeypatch)
    monkeypatch.setattr(app_module, "REWRAP_JOB_EVENT_PAGE_SIZE", 2)
    first = _events(client, job_id).json()
    assert [e["job_seq"] for e in first["events"]] == [1, 2]
    cursor = first["next_cursor"]
    assert cursor

    # New migrations commit while the client still holds the cursor:
    # resume, re-run to success adds seq 4..6.
    assert _resume(client, job_id).status_code == 200
    _run_job(app, job_id)

    # The old snapshot's next page contains only seq 3 and completes: it
    # neither duplicates nor skips and never sees the new migrations.
    second = _events(client, job_id, cursor=cursor).json()
    assert [e["job_seq"] for e in second["events"]] == [3]
    assert second["complete"] is True and second["next_cursor"] == ""
    # Replaying the held cursor returns the identical page, unaffected by
    # the migrations that committed in the meantime.
    replay = _events(client, job_id, cursor=cursor)
    assert replay.json() == second

    # A fresh first query (new query family). The old page-size patch is
    # lifted so the fresh family returns in one page.
    monkeypatch.setattr(app_module, "REWRAP_JOB_EVENT_PAGE_SIZE", 100)
    fresh = _events(client, job_id).json()
    assert [e["job_seq"] for e in fresh["events"]] == [1, 2, 3, 4, 5, 6]
    assert fresh["complete"] is True


def test_queued_snapshot_stays_a_single_event_after_settlement(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    # The first (completed, cursor-less) query fixes the snapshot at seq 1.
    first = _events(client, job_id).json()
    assert [e["job_seq"] for e in first["events"]] == [1]
    assert first["complete"] is True

    _run_job(app, job_id)
    # The old query family cannot be resumed (its page was complete), and a
    # brand new first query is a new family that observes the new events.
    fresh = _events(client, job_id).json()
    assert [e["job_seq"] for e in fresh["events"]] == [1, 2, 3]


# --- cursor authentication -------------------------------------------------


def test_tampered_or_forged_cursor_is_422(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "REWRAP_JOB_EVENT_PAGE_SIZE", 1)
    cursor = _events(client, job_id).json()["next_cursor"]

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _events(client, job_id, cursor=tampered).status_code == 422
    forged = b64url_encode(
        b'{"k":"rewrap-job-events-v1","t":"tenant-a","w":"workload-1",'
        b'"j":"' + job_id.encode() + b'","q":1,"h":3}' + b"0" * 32
    )
    assert _events(client, job_id, cursor=forged).status_code == 422


def test_cross_scope_cursor_is_422(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "REWRAP_JOB_EVENT_PAGE_SIZE", 1)
    cursor = _events(client, job_id).json()["next_cursor"]
    assert _events(client, job_id, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert (
        _events(client, job_id, workload=OTHER_WORKLOAD, cursor=cursor).status_code
        == 422
    )


def test_cross_job_cursor_is_422(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    first = _submit(client).json()["job_id"]
    second = _submit(client).json()["job_id"]
    _run_job(app, first)
    _run_job(app, second)
    monkeypatch.setattr(app_module, "REWRAP_JOB_EVENT_PAGE_SIZE", 1)
    cursor = _events(client, first).json()["next_cursor"]
    # The cursor is minted for first; replaying it against the sibling job
    # in the same scope is a 422, not a page of the sibling's events.
    assert _events(client, second, cursor=cursor).status_code == 422


def test_foreign_cursor_families_are_422(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    from proof_release.app import (
        _encode_audit_event_cursor,
        _encode_cursor,
        _encode_grant_audit_cursor,
        _encode_revocation_cursor,
        _encode_rewrap_job_history_cursor,
    )

    foreign_batch = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _events(client, job_id, cursor=foreign_batch).status_code == 422
    foreign_history = _encode_rewrap_job_history_cursor(
        TENANT,
        WORKLOAD,
        ZERO_UUID,
        job_id="",
        status="",
        created_after="",
        created_before="",
    )
    assert _events(client, job_id, cursor=foreign_history).status_code == 422
    foreign_grant = _encode_grant_audit_cursor(
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
    assert _events(client, job_id, cursor=foreign_grant).status_code == 422
    foreign_event = _encode_audit_event_cursor(
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
    assert _events(client, job_id, cursor=foreign_event).status_code == 422
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
    assert _events(client, job_id, cursor=foreign_revocation).status_code == 422


# --- read-only behaviour ---------------------------------------------------


def test_query_changes_nothing(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b", "c"])
    job_id = _submit(client, limit=2).json()["job_id"]
    with app.state.session_factory() as session:
        before_job = session.get(RewrapJob, job_id)
        before = (
            before_job.status,
            before_job.processed,
            before_job.next_cursor,
            session.query(RewrapJobEvent).count(),
        )

    for _ in range(3):
        assert _events(client, job_id).status_code == 200

    with app.state.session_factory() as session:
        job = session.get(RewrapJob, job_id)
        after = (job.status, job.processed, job.next_cursor, session.query(RewrapJobEvent).count())
        assert after == before
        assert job.status == "queued"


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_page(app, client, monkeypatch):
    from sqlalchemy import text

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_job_events"))
    assert _events(client, job_id).status_code == 500


# --- persistence -----------------------------------------------------------


def test_events_survive_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/persist.db"
    first = app_module.create_app(url)
    first_client = TestClient(first)
    _seed(first_client, ["a"])
    job_id = _submit(first_client).json()["job_id"]
    first.state.rewrap_job_runner.run(job_id)
    first.state.engine.dispose()

    second = app_module.create_app(url)
    try:
        second_client = TestClient(second)
        events = second_client.get(
            f"/v1/rewrap-jobs/{job_id}/events",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).json()["events"]
        assert [e["reason"] for e in events] == [
            "submitted",
            "executed",
            "completed",
        ]
        assert [e["job_seq"] for e in events] == [1, 2, 3]
    finally:
        second.state.engine.dispose()


# --- secrecy ---------------------------------------------------------------


def test_timeline_never_exposes_payload_or_key_material(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    response = _events(client, job_id)
    for secret in (
        PAYLOAD,
        KEY_V1,
        KEY_V2,
        "wrapped_key",
        "ciphertext",
        "payload",
        "master_key",
        "traceback",
    ):
        assert secret not in response.text
    with app.state.session_factory() as session:
        for event in session.query(RewrapJobEvent).all():
            assert not hasattr(event, "wrapped_key")
            assert not hasattr(event, "ciphertext")
