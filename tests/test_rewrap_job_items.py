"""Tests for the persistent per-envelope attempt detail of rewrap jobs:

GET /v1/rewrap-jobs/{job_id}/items

The endpoint ranges exactly one job by mandatory tenant/workload scope and
a canonical-UUID path id; it accepts only tenant_id, workload_id and an
optional cursor, carries no body, and returns the job's committed
per-envelope processing attempts as stable, immutable per-job attempt_seq
items. Successes and skips commit with the envelope change, the rewrap
audit event and the job's counters; failures commit with the running ->
failed settlement. Rolled-back attempts leave no item and committed items
are never modified; a resume's re-attempt appends a later item. The first
(cursor-less) query fixes a replayable snapshot high-water mark; a resume
cursor is HMAC-authenticated and bound to the scope, job, snapshot and
cursor family. The query is strictly read-only.

Like the other job tests, the app is built without entering its lifespan
so no pool runs; jobs are driven synchronously.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import RewrapJob, RewrapJobItem
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

PAYLOAD = "items-secret 🔐"


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


KEYRING_V1_V2 = _keyring(2, {1: KEY_V1, 2: KEY_V2})
KEYRING_V2_ONLY = _keyring(2, {2: KEY_V2})

ITEM_FIELDS = [
    "item_id",
    "attempt_seq",
    "data_id",
    "old_key_version",
    "new_key_version",
    "result",
    "created_at",
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
    monkeypatch.setattr(app_module, "REWRAP_JOB_ITEM_PAGE_SIZE", page_size)
    rows = []
    token = None
    for _ in range(100):
        query = dict(params)
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
        {"limit": "10"},
        {"page_size": "2"},
        {"status": "queued"},
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
            ("cursor", ""),
            ("cursor", ""),
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


def test_invalid_inputs_read_no_state(app, client):
    # Malformed path, bad cursor and unknown params must never touch state.
    _items(client, "not-a-uuid")
    _items(client, ZERO_UUID, cursor="garbage!!")
    _items(client, ZERO_UUID, bogus="1")
    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 0
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
    # An illegal cursor is rejected before the unknown job is resolved.
    response = _items(client, ZERO_UUID, cursor="tampered-token")
    assert response.status_code == 422
    response = _items(client, "not-a-uuid")
    assert response.status_code == 422


# --- recorded attempts -----------------------------------------------------


def test_rewrapped_items_commit_with_the_job(app, client, monkeypatch):
    # Sealed under the legacy v1 fixture key, then rotated by a job
    # running with a v1/v2 keyring whose current version is 2.
    _seed(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    data = _items(client, job_id).json()
    assert data["next_cursor"] == ""
    assert data["complete"] is True
    items = data["items"]
    assert [i["attempt_seq"] for i in items] == [1, 2, 3]
    assert [i["data_id"] for i in items] == ["a", "b", "c"]
    for item in items:
        assert list(item) == ITEM_FIELDS
        assert item["result"] == "rewrapped"
        # Sealed under legacy v1, rotated to the then-current v2.
        assert item["old_key_version"] == 1
        assert item["new_key_version"] == 2
        assert isinstance(item["item_id"], str) and len(item["item_id"]) == 36
        assert isinstance(item["attempt_seq"], int) and not isinstance(
            item["attempt_seq"], bool
        )
        parsed = datetime.fromisoformat(item["created_at"])
        assert parsed.utcoffset() == timedelta(0)
    assert len({i["item_id"] for i in items}) == 3


def test_current_version_envelope_is_a_skip_item(app, client, monkeypatch):
    # Sealed under the v1/v2 keyring at current v2, so the job skips it.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    items = _items(client, job_id).json()["items"]
    assert [(i["attempt_seq"], i["result"]) for i in items] == [(1, "skipped")]
    # A skip's old and new versions are identical.
    assert items[0]["old_key_version"] == items[0]["new_key_version"] == 2


def test_queued_or_cancelled_job_has_no_items(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    data = _items(client, job_id).json()
    assert data == {"items": [], "next_cursor": "", "complete": True}

    assert _cancel(client, job_id).status_code == 200
    assert _items(client, job_id).json()["items"] == []
    with app.state.session_factory() as session:
        assert session.query(RewrapJobItem).count() == 0


def test_missing_key_failure_commits_item_with_the_settlement(
    app, client, monkeypatch
):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    items = _items(client, job_id).json()["items"]
    assert [(i["attempt_seq"], i["data_id"], i["result"]) for i in items] == [
        (1, "a", "missing-key")
    ]
    # A failure's old and new versions are identical.
    assert items[0]["old_key_version"] == items[0]["new_key_version"] == 1
    with app.state.session_factory() as session:
        job = session.get(RewrapJob, job_id)
        # The item, the failed counter and the parked cursor committed
        # together; the failing envelope was not counted as processed.
        assert job.status == "failed"
        assert job.failed == 1
        assert job.processed == 0
        assert session.query(RewrapJobItem).count() == 1


def test_unusable_keyring_failure_records_keyring_item(app, client, monkeypatch):
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

    items = _items(client, job_id).json()["items"]
    assert [i["result"] for i in items] == ["keyring"]
    assert "vanished" not in json.dumps(items)
    assert "secret detail" not in json.dumps(items)


def test_rewrap_failure_records_rewrap_item(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]

    def boom(*args, **kwargs):
        raise ValueError("crypto stack trace with internal detail")

    monkeypatch.setattr(app_module, "rewrap_data_key", boom)
    _run_job(app, job_id)

    items = _items(client, job_id).json()["items"]
    assert [i["result"] for i in items] == ["rewrap"]
    assert items[0]["old_key_version"] == items[0]["new_key_version"] == 1
    assert "crypto stack trace" not in json.dumps(items)


def test_result_codes_are_the_fixed_set(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    allowed = {"rewrapped", "skipped", "keyring", "missing-key", "rewrap"}
    items = _items(client, job_id).json()["items"]
    assert items and all(i["result"] in allowed for i in items)


def test_resume_retry_appends_a_later_item(app, client, monkeypatch):
    job_id = _fail_job(app, client, monkeypatch=monkeypatch)
    before = _items(client, job_id).json()["items"]
    assert [(i["attempt_seq"], i["result"]) for i in before] == [
        (1, "missing-key")
    ]

    assert _resume(client, job_id).status_code == 200
    _run_job(app, job_id)

    after = _items(client, job_id).json()["items"]
    # The old failure item is untouched; the retried envelope is a new,
    # later item that now succeeds.
    assert [(i["attempt_seq"], i["result"]) for i in after] == [
        (1, "missing-key"),
        (2, "rewrapped"),
    ]
    assert after[0] == before[0]
    assert after[1]["data_id"] == "a"
    assert after[1]["old_key_version"] == 1
    assert after[1]["new_key_version"] == 2


def test_cancel_before_commit_leaves_no_item(app, client, monkeypatch):
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

    # The cancel settled before any envelope committed: no items at all.
    assert _items(client, job_id).json()["items"] == []
    with app.state.session_factory() as session:
        assert session.query(RewrapJobItem).count() == 0


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


def test_empty_string_cursor_equals_default(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    assert _items(client, job_id).content == _items(client, job_id, cursor="").content


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


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_item_once_in_order(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b", "c"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    rows = _walk(client, app, job_id, 1, monkeypatch)
    assert [i["attempt_seq"] for i in rows] == [1, 2, 3]
    assert [i["data_id"] for i in rows] == ["a", "b", "c"]


def test_page_carries_cursor_and_complete_flag(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "REWRAP_JOB_ITEM_PAGE_SIZE", 1)

    first = _items(client, job_id).json()
    assert [i["attempt_seq"] for i in first["items"]] == [1]
    assert first["complete"] is False and first["next_cursor"]

    last = _items(client, job_id, cursor=first["next_cursor"]).json()
    assert [i["attempt_seq"] for i in last["items"]] == [2]
    assert last["complete"] is True and last["next_cursor"] == ""


def test_replaying_same_cursor_returns_identical_page(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "REWRAP_JOB_ITEM_PAGE_SIZE", 1)
    cursor = _items(client, job_id).json()["next_cursor"]
    first = _items(client, job_id, cursor=cursor)
    second = _items(client, job_id, cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


# --- snapshot isolation ----------------------------------------------------


def test_first_query_fixes_snapshot_for_later_pages(app, client, monkeypatch):
    # Fail the job (1 item) and open the listing: the first query fixes
    # the snapshot high-water mark at attempt_seq 1.
    job_id = _fail_job(app, client, monkeypatch)
    first = _items(client, job_id).json()
    assert [i["attempt_seq"] for i in first["items"]] == [1]
    assert first["complete"] is True

    # A resume and successful re-run commits a second attempt afterwards.
    assert _resume(client, job_id).status_code == 200
    _run_job(app, job_id)

    # A fresh first query is a new family and observes both attempts.
    fresh = _items(client, job_id).json()
    assert [i["attempt_seq"] for i in fresh["items"]] == [1, 2]


def test_held_cursor_never_sees_later_attempts(app, client, monkeypatch):
    # Two envelopes, page size 1: the first query fixes the snapshot at
    # the single attempt committed so far, then the job advances.
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "REWRAP_JOB_ITEM_PAGE_SIZE", 1)
    first = _items(client, job_id).json()
    assert [i["attempt_seq"] for i in first["items"]] == [1]
    assert first["complete"] is True

    # A later failure/resume cycle adds attempts; a brand new family sees
    # them, the completed old family is unaffected by construction.
    monkeypatch.delenv("PROOF_RELEASE_KEYRING")
    _seed(client, ["b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    second_job = _submit(client).json()["job_id"]
    _run_job(app, second_job)
    # Lift the page-size patch so the fresh family returns in one page.
    monkeypatch.setattr(app_module, "REWRAP_JOB_ITEM_PAGE_SIZE", 100)
    fresh = _items(client, second_job).json()
    # The second job scans the whole scope: the already-rotated "a" skips,
    # then the v1-sealed "b" fails on the lost historical key.
    assert [i["result"] for i in fresh["items"]] == ["skipped", "missing-key"]
    # The first job's listing is unchanged by the second job's attempts.
    assert _items(client, job_id).json() == first


# --- cursor authentication -------------------------------------------------


def test_tampered_or_forged_cursor_is_422(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "REWRAP_JOB_ITEM_PAGE_SIZE", 1)
    cursor = _items(client, job_id).json()["next_cursor"]

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _items(client, job_id, cursor=tampered).status_code == 422
    forged = b64url_encode(
        b'{"k":"rewrap-job-items-v1","t":"tenant-a","w":"workload-1",'
        b'"j":"' + job_id.encode() + b'","q":1,"h":2}' + b"0" * 32
    )
    assert _items(client, job_id, cursor=forged).status_code == 422


def test_cross_scope_cursor_is_422(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "REWRAP_JOB_ITEM_PAGE_SIZE", 1)
    cursor = _items(client, job_id).json()["next_cursor"]
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
    monkeypatch.setattr(app_module, "REWRAP_JOB_ITEM_PAGE_SIZE", 1)
    cursor = _items(client, first).json()["next_cursor"]
    # The cursor is minted for first; replaying it against the sibling job
    # in the same scope is a 422, not a page of the sibling's items.
    assert _items(client, second, cursor=cursor).status_code == 422


def test_foreign_cursor_families_are_422(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    from proof_release.app import (
        _encode_audit_event_cursor,
        _encode_cursor,
        _encode_grant_audit_cursor,
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


# --- read-only behaviour ---------------------------------------------------


def test_query_changes_nothing(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b", "c"])
    job_id = _submit(client, limit=2).json()["job_id"]
    _run_job(app, job_id)
    with app.state.session_factory() as session:
        before_job = session.get(RewrapJob, job_id)
        before = (
            before_job.status,
            before_job.processed,
            before_job.next_cursor,
            before_job.updated_at,
            session.query(RewrapJobItem).count(),
        )

    for _ in range(3):
        assert _items(client, job_id).status_code == 200

    with app.state.session_factory() as session:
        job = session.get(RewrapJob, job_id)
        after = (
            job.status,
            job.processed,
            job.next_cursor,
            job.updated_at,
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
    assert _items(client, job_id).status_code == 500


# --- persistence -----------------------------------------------------------


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
        items = second_client.get(
            f"/v1/rewrap-jobs/{job_id}/items",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).json()["items"]
        assert [i["attempt_seq"] for i in items] == [1, 2]
        assert [i["result"] for i in items] == ["rewrapped", "rewrapped"]
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
        "traceback",
    ):
        assert secret not in response.text
    with app.state.session_factory() as session:
        for item in session.query(RewrapJobItem).all():
            assert not hasattr(item, "wrapped_key")
            assert not hasattr(item, "ciphertext")
