"""Tests for the per-item results of an asynchronous rewrap job:

GET /v1/rewrap-jobs/{job_id}/items

The endpoint ranges exactly one job by mandatory tenant/workload scope and
a canonical-UUID path id; it accepts only tenant_id, workload_id, an
optional limit (1..200, default 50) and an optional cursor, carries no
body, and returns the job's committed per-envelope results in stable,
immutable per-job seq order. Each committed envelope recorded exactly one
item (result rewrapped or skipped) in the same transaction as the
envelope change, the audit event and the job progress; a failed envelope
has no item and its classification stays on the job's events. A resume
cursor is HMAC-authenticated and bound to the scope, the job and this
paging convention, so it cannot be forged, tampered with, or replayed
across scopes, jobs or cursor families. The query is strictly read-only.

Like the other job tests, the app is built without entering its lifespan
so no pool runs; jobs (and the post-restart retry claim) are driven
synchronously.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import RewrapJob, RewrapJobEvent, RewrapJobItem
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

PAYLOAD = "item-secret 🔐"


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


KEYRING_V1_V2 = _keyring(2, {1: KEY_V1, 2: KEY_V2})
KEYRING_V2_ONLY = _keyring(2, {2: KEY_V2})

ITEM_FIELDS = [
    "seq",
    "data_id",
    "old_key_version",
    "new_key_version",
    "result",
    "occurred_at",
]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = app_module.create_app(f"sqlite:///{tmp_path}/items.db")
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


def _items(client, job_id, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        f"/v1/rewrap-jobs/{job_id}/items",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


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


def _walk(client, job_id, page_size, **params):
    rows = []
    token = None
    for _ in range(100):
        query = dict(params)
        query["limit"] = str(page_size)
        if token:
            query["cursor"] = token
        data = _items(client, job_id, **query).json()
        rows.extend(data["items"])
        if data["complete"]:
            return rows
        token = data["next_cursor"]
        assert token
    raise AssertionError("pagination never completed")  # pragma: no cover


# --- request validation ----------------------------------------------------


def test_missing_scope_parameters_are_422(client):
    assert _items(client, ZERO_UUID, tenant="").status_code == 422
    assert _items(client, ZERO_UUID, workload="").status_code == 422
    response = client.get(f"/v1/rewrap-jobs/{ZERO_UUID}/items")
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
    assert _items(client, job_id).status_code == 422


def test_empty_path_segment_is_422(client):
    response = client.get(
        "/v1/rewrap-jobs//items",
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
        {"page_size": "2"},
        {"status": "queued"},
        {"seq": "1"},
        {"item_id": ZERO_UUID},
    ],
)
def test_unknown_or_malformed_parameters_are_422(client, params):
    assert _items(client, ZERO_UUID, **params).status_code == 422


def test_duplicate_parameter_is_422(client):
    response = client.get(
        f"/v1/rewrap-jobs/{ZERO_UUID}/items",
        params=[
            ("tenant_id", TENANT),
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
        ],
    )
    assert response.status_code == 422
    response = client.get(
        f"/v1/rewrap-jobs/{ZERO_UUID}/items",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("limit", "10"),
            ("limit", "10"),
        ],
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"x", b"   ", b"{}"])
def test_non_empty_body_is_422(client, body):
    response = client.request(
        "GET",
        f"/v1/rewrap-jobs/{ZERO_UUID}/items",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=body,
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "limit",
    ["", "0", "201", "-1", "1.5", "abc", " 1", "1 ", "+1", "1e2", "0x10"],
)
def test_non_integer_or_out_of_range_limit_is_422(client, limit):
    assert _items(client, ZERO_UUID, limit=limit).status_code == 422


@pytest.mark.parametrize("limit", ["1", "50", "200", "007"])
def test_valid_limit_is_accepted(app, client, monkeypatch, limit):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    assert _items(client, job_id, limit=limit).status_code == 200


def test_invalid_inputs_read_no_state(app, client):
    # Malformed path, bad cursor, bad limit and unknown params must never
    # touch state.
    _items(client, "not-a-uuid")
    _items(client, ZERO_UUID, cursor="garbage!!")
    _items(client, ZERO_UUID, limit="abc")
    _items(client, ZERO_UUID, bogus="1")
    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 0
        assert session.query(RewrapJobEvent).count() == 0
        assert session.query(RewrapJobItem).count() == 0


# --- 404 semantics ---------------------------------------------------------


def test_unknown_job_returns_404(client):
    assert _items(client, ZERO_UUID).status_code == 404


def test_cross_scope_job_returns_404(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    assert _items(client, job_id, tenant=OTHER_TENANT).status_code == 404
    assert _items(client, job_id, workload=OTHER_WORKLOAD).status_code == 404
    assert (
        _items(
            client, job_id, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
        ).status_code
        == 404
    )


def test_422_precedes_404(client):
    # An illegal cursor or limit is rejected before the unknown job is
    # resolved.
    assert _items(client, ZERO_UUID, cursor="tampered-token").status_code == 422
    assert _items(client, ZERO_UUID, limit="0").status_code == 422
    assert _items(client, "not-a-uuid").status_code == 422


# --- recorded items --------------------------------------------------------


def test_items_record_rewrapped_and_skipped_in_committed_order(
    app, client, monkeypatch
):
    # a and b are sealed under the v1 fixture key; c is sealed after the
    # keyring rotates, already at the current v2.
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["c"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    data = _items(client, job_id).json()
    assert data["next_cursor"] == ""
    assert data["complete"] is True
    items = data["items"]
    assert [
        (
            i["seq"],
            i["data_id"],
            i["old_key_version"],
            i["new_key_version"],
            i["result"],
        )
        for i in items
    ] == [
        (1, "a", 1, 2, "rewrapped"),
        (2, "b", 1, 2, "rewrapped"),
        (3, "c", 2, 2, "skipped"),
    ]
    # Sequence numbers are gap-free integers from 1; only the two
    # committed-outcome codes ever appear; timestamps are UTC RFC3339.
    assert [i["seq"] for i in items] == [1, 2, 3]
    for item in items:
        assert list(item) == ITEM_FIELDS
        assert isinstance(item["seq"], int) and not isinstance(item["seq"], bool)
        assert isinstance(item["old_key_version"], int)
        assert isinstance(item["new_key_version"], int)
        assert item["result"] in {"rewrapped", "skipped"}
        parsed = datetime.fromisoformat(item["occurred_at"])
        assert parsed.utcoffset() == timedelta(0)


def test_queued_job_has_no_items(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    data = _items(client, job_id).json()
    assert data == {"items": [], "next_cursor": "", "complete": True}


def test_items_are_isolated_per_job(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    first = _submit(client).json()["job_id"]
    second = _submit(client).json()["job_id"]
    _run_job(app, first)
    assert [i["seq"] for i in _items(client, first).json()["items"]] == [1]
    assert _items(client, second).json()["items"] == []


def test_items_match_job_progress_counters(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["c"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    job = client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    items = _items(client, job_id).json()["items"]
    assert job["processed"] == len(items) == 3
    assert job["rewrapped"] == sum(1 for i in items if i["result"] == "rewrapped")
    assert job["skipped"] == sum(1 for i in items if i["result"] == "skipped")


# --- failure / cancel / resume semantics -----------------------------------


def _fail_second_envelope(app, client, monkeypatch):
    """Seed two v1 envelopes and fail the job on the second one."""
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    real = app_module.rewrap_data_key
    calls = {"n": 0}

    def flaky(*args):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("boom with internal detail")
        return real(*args)

    monkeypatch.setattr(app_module, "rewrap_data_key", flaky)
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "rewrap_data_key", real)
    return job_id


def test_failed_envelope_has_no_item_and_committed_items_stay(
    app, client, monkeypatch
):
    job_id = _fail_second_envelope(app, client, monkeypatch)

    data = _items(client, job_id).json()
    assert [(i["seq"], i["data_id"], i["result"]) for i in data["items"]] == [
        (1, "a", "rewrapped")
    ]
    assert data["complete"] is True and data["next_cursor"] == ""
    # The failure itself is only on the job's events, never in the items.
    events = _events(client, job_id).json()["events"]
    assert events[-1]["new_status"] == "failed"
    assert events[-1]["reason"] == "rewrap"
    assert "boom" not in json.dumps(data)
    assert "internal detail" not in json.dumps(data)
    job = client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    assert job["failed"] == 1
    assert job["processed"] == 1


def test_resume_continues_seq_without_gaps_or_duplicates(
    app, client, monkeypatch
):
    job_id = _fail_second_envelope(app, client, monkeypatch)
    assert _resume(client, job_id).status_code == 200
    _run_job(app, job_id)

    data = _items(client, job_id).json()
    assert [(i["seq"], i["data_id"], i["result"]) for i in data["items"]] == [
        (1, "a", "rewrapped"),
        (2, "b", "rewrapped"),
    ]
    assert data["complete"] is True


def test_cancel_discards_the_in_flight_envelope_item(app, client, monkeypatch):
    # Sealed under the v1 fixture key; the run sees the v1/v2 keyring and
    # actually attempts each rewrap.
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    real = app_module.rewrap_data_key
    entered = threading.Event()
    calls = {"n": 0}

    def gated(*args):
        calls["n"] += 1
        if calls["n"] == 2:
            entered.set()
            # Hold the second envelope uncommitted until the cancel
            # signal is raised, then let the runner observe it and roll
            # the whole envelope transaction back.
            for _ in range(500):
                if app.state.rewrap_job_runner._cancel_requested(gated.job_id):
                    break
                time.sleep(0.01)
        return real(*args)

    job_id = _submit(client).json()["job_id"]
    gated.job_id = job_id
    monkeypatch.setattr(app_module, "rewrap_data_key", gated)
    runner = threading.Thread(target=_run_job, args=(app, job_id))
    runner.start()
    try:
        assert entered.wait(timeout=5)
        assert _cancel(client, job_id).status_code == 200
    finally:
        runner.join(timeout=5)

    # Only the first envelope committed; the cancelled-in-flight second
    # envelope left no item, no audit gap and no progress jump.
    data = _items(client, job_id).json()
    assert [(i["seq"], i["data_id"]) for i in data["items"]] == [(1, "a")]
    assert data["complete"] is True
    with app.state.session_factory() as session:
        assert session.query(RewrapJobItem).count() == 1


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_item_once_in_order(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b", "c", "d", "e"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    rows = _walk(client, job_id, 2)
    assert [i["seq"] for i in rows] == [1, 2, 3, 4, 5]
    assert [i["data_id"] for i in rows] == ["a", "b", "c", "d", "e"]


def test_page_carries_cursor_and_complete_flag(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b", "c"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    first = _items(client, job_id, limit="2").json()
    assert [i["seq"] for i in first["items"]] == [1, 2]
    assert first["complete"] is False and first["next_cursor"]

    second = _items(client, job_id, limit="2", cursor=first["next_cursor"]).json()
    assert [i["seq"] for i in second["items"]] == [3]
    assert second["complete"] is True and second["next_cursor"] == ""


def test_default_limit_is_50(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, [f"data-{index:02d}" for index in range(55)])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    first = _items(client, job_id).json()
    assert len(first["items"]) == 50
    assert first["complete"] is False
    second = _items(client, job_id, cursor=first["next_cursor"]).json()
    assert [i["seq"] for i in second["items"]] == [51, 52, 53, 54, 55]
    assert second["complete"] is True


def test_limit_may_change_between_pages(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b", "c"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    first = _items(client, job_id, limit="1").json()
    assert [i["seq"] for i in first["items"]] == [1]
    second = _items(client, job_id, limit="2", cursor=first["next_cursor"]).json()
    assert [i["seq"] for i in second["items"]] == [2, 3]
    assert second["complete"] is True


def test_replaying_same_cursor_returns_identical_page(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b", "c"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    cursor = _items(client, job_id, limit="1").json()["next_cursor"]
    first = _items(client, job_id, limit="1", cursor=cursor)
    second = _items(client, job_id, limit="1", cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_string_cursor_equals_default(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    assert _items(client, job_id).content == _items(client, job_id, cursor="").content


def test_pages_continue_after_last_returned_seq_while_job_advances(
    app, client, monkeypatch
):
    # Fail the job after one committed item and open a page-1 walk: the
    # first page sees only the committed seq 1.
    job_id = _fail_second_envelope(app, client, monkeypatch)
    first = _items(client, job_id, limit="1").json()
    assert [i["seq"] for i in first["items"]] == [1]
    assert first["complete"] is True and first["next_cursor"] == ""

    # The job advances afterwards; a fresh walk continues from the
    # beginning and sees every committed seq exactly once, with no
    # duplicates, gaps or rewritten items.
    assert _resume(client, job_id).status_code == 200
    _run_job(app, job_id)
    rows = _walk(client, job_id, 1)
    assert [(i["seq"], i["data_id"]) for i in rows] == [(1, "a"), (2, "b")]


def test_held_cursor_resumes_after_newly_committed_items(
    app, client, monkeypatch
):
    # Three v1 envelopes; the run fails on the third, leaving two
    # committed items (seq 1, 2) and the job failed.
    _seed(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    real = app_module.rewrap_data_key
    calls = {"n": 0}

    def flaky(*args):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ValueError("third envelope boom")
        return real(*args)

    monkeypatch.setattr(app_module, "rewrap_data_key", flaky)
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "rewrap_data_key", real)

    # A limit-1 first page holds a cursor minted at seq 1.
    first = _items(client, job_id, limit="1").json()
    assert [i["seq"] for i in first["items"]] == [1]
    cursor = first["next_cursor"]
    assert cursor

    # The job advances afterwards: the third item commits.
    assert _resume(client, job_id).status_code == 200
    _run_job(app, job_id)

    # The held cursor continues strictly after seq 1, observing the items
    # committed since, each exactly once and in order.
    second = _items(client, job_id, limit="1", cursor=cursor).json()
    assert [i["seq"] for i in second["items"]] == [2]
    third = _items(client, job_id, limit="1", cursor=second["next_cursor"]).json()
    assert [i["seq"] for i in third["items"]] == [3]
    assert third["complete"] is True


# --- cursor authentication -------------------------------------------------


def test_tampered_or_forged_cursor_is_422(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    cursor = _items(client, job_id, limit="1").json()["next_cursor"]

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _items(client, job_id, cursor=tampered).status_code == 422
    forged = b64url_encode(
        b'{"k":"rewrap-job-items-v1","t":"tenant-a","w":"workload-1",'
        b'"j":"' + job_id.encode() + b'","q":1}' + b"0" * 32
    )
    assert _items(client, job_id, cursor=forged).status_code == 422


def test_cross_scope_cursor_is_422(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    cursor = _items(client, job_id, limit="1").json()["next_cursor"]
    assert _items(client, job_id, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert (
        _items(client, job_id, workload=OTHER_WORKLOAD, cursor=cursor).status_code
        == 422
    )


def test_cross_job_cursor_is_422(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    first = _submit(client).json()["job_id"]
    second = _submit(client).json()["job_id"]
    _run_job(app, first)
    _run_job(app, second)
    cursor = _items(client, first, limit="1").json()["next_cursor"]
    # The cursor is minted for first; replaying it against the sibling job
    # in the same scope is a 422, not a page of the sibling's items.
    assert _items(client, second, cursor=cursor).status_code == 422


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
        _encode_rewrap_job_event_cursor,
        _encode_rewrap_job_history_cursor,
    )

    foreign_batch = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _items(client, job_id, cursor=foreign_batch).status_code == 422
    foreign_history = _encode_rewrap_job_history_cursor(
        TENANT,
        WORKLOAD,
        ZERO_UUID,
        job_id="",
        status="",
        created_after="",
        created_before="",
    )
    assert _items(client, job_id, cursor=foreign_history).status_code == 422
    foreign_event = _encode_rewrap_job_event_cursor(
        TENANT, WORKLOAD, job_id, 1, snapshot_seq=1
    )
    assert _items(client, job_id, cursor=foreign_event).status_code == 422
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
    assert _items(client, job_id, cursor=foreign_grant).status_code == 422
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
    assert _items(client, job_id, cursor=foreign_audit).status_code == 422
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
    assert _items(client, job_id, cursor=foreign_revocation).status_code == 422


# --- wire format -----------------------------------------------------------


def test_response_is_compact_json_with_single_trailing_newline(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    response = _items(client, job_id)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == ["items", "next_cursor", "complete"]


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

    _check(json.loads(_items(client, job_id).content))


# --- read-only behaviour ---------------------------------------------------


def test_query_changes_nothing(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    with app.state.session_factory() as session:
        before_job = session.get(RewrapJob, job_id)
        before = (
            before_job.status,
            before_job.processed,
            before_job.next_cursor,
            before_job.updated_at,
            session.query(RewrapJobEvent).count(),
            session.query(RewrapJobItem).count(),
        )

    for _ in range(3):
        assert _items(client, job_id).status_code == 200
        assert _items(client, job_id, limit="1").status_code == 200

    with app.state.session_factory() as session:
        job = session.get(RewrapJob, job_id)
        after = (
            job.status,
            job.processed,
            job.next_cursor,
            job.updated_at,
            session.query(RewrapJobEvent).count(),
            session.query(RewrapJobItem).count(),
        )
        assert after == before


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_page(app, client, monkeypatch):
    from sqlalchemy import text

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_job_items"))
    response = _items(client, job_id)
    assert response.status_code == 500
    assert response.json()["detail"] == "rewrap job items unavailable"


# --- persistence and migration ---------------------------------------------


def test_items_survive_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/persist.db"
    first = app_module.create_app(url)
    first_client = TestClient(first)
    _seed(first_client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(first_client).json()["job_id"]
    first.state.rewrap_job_runner.run(job_id)
    first.state.engine.dispose()

    second = app_module.create_app(url)
    try:
        second_client = TestClient(second)
        data = second_client.get(
            f"/v1/rewrap-jobs/{job_id}/items",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).json()
        assert [(i["seq"], i["data_id"], i["result"]) for i in data["items"]] == [
            (1, "a", "rewrapped"),
            (2, "b", "rewrapped"),
        ]
        assert data["complete"] is True
    finally:
        second.state.engine.dispose()


def test_pre_upgrade_job_without_items_returns_empty_complete_page(
    app, client, monkeypatch
):
    # A job created and advanced before the items table existed has no
    # per-item rows: simulate that upgrade state by removing them.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    with app.state.session_factory() as session:
        session.query(RewrapJobItem).delete()
        session.commit()

    data = _items(client, job_id).json()
    assert data == {"items": [], "next_cursor": "", "complete": True}
    # The job's own progress is untouched and still queryable.
    job = client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    assert job["status"] == "succeeded"
    assert job["processed"] == 1


def test_items_table_is_created_by_additive_migration(tmp_path, monkeypatch):
    # A database written by a pre-upgrade deployment lacks the items
    # table; opening it with the upgraded service recreates it.
    from sqlalchemy import text

    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/upgrade.db"
    first = app_module.create_app(url)
    first_client = TestClient(first)
    _seed(first_client, ["a"])
    job_id = _submit(first_client).json()["job_id"]
    with first.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_job_items"))
    first.state.engine.dispose()

    second = app_module.create_app(url)
    try:
        second_client = TestClient(second)
        response = second_client.get(
            f"/v1/rewrap-jobs/{job_id}/items",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
        assert response.status_code == 200
        assert response.json() == {
            "items": [],
            "next_cursor": "",
            "complete": True,
        }
    finally:
        second.state.engine.dispose()


# --- secrecy ---------------------------------------------------------------


def test_items_never_expose_payload_or_key_material(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    response = _items(client, job_id)
    for secret in (
        PAYLOAD,
        KEY_V1,
        KEY_V2,
        "wrapped_key",
        "ciphertext",
        "payload",
        "master_key",
        "item_id",
        "traceback",
    ):
        assert secret not in response.text
    with app.state.session_factory() as session:
        for item in session.query(RewrapJobItem).all():
            assert not hasattr(item, "wrapped_key")
            assert not hasattr(item, "ciphertext")
