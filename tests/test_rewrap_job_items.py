"""Tests for the per-envelope result listing of asynchronous rewrap jobs:

GET /v1/rewrap-jobs/{job_id}/items

The endpoint ranges exactly one job by mandatory tenant/workload scope and
a canonical-UUID path id; it accepts only tenant_id, workload_id, an
optional 1..200 limit (default 50) and an optional cursor, carries no
body, and returns the job's committed per-envelope outcomes in stable,
immutable per-job seq order. Every item commits in the same transaction
as its envelope change and the job's progress, so pages only ever read
committed results; a failed envelope never appears, while results
committed before a failure remain queryable. A resume cursor is
HMAC-authenticated and bound to the scope, the job and this cursor
family, so it cannot be forged, tampered with, or replayed across scopes,
jobs or cursor families. The query is strictly read-only.

Like the other job tests, the app is built without entering its lifespan
so no pool runs; jobs are driven synchronously.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

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

ITEM_FIELDS = [
    "seq",
    "data_id",
    "old_key_version",
    "new_key_version",
    "result",
    "occurred_at",
]

RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|\+00:00)$"
)


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
    application = app_module.create_app(f"sqlite:///{tmp_path}/items.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    # Deliberately NOT used as a context manager: no pool runs, so jobs
    # are advanced synchronously and deterministically.
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


def _advance_one_page(app, job_id):
    runner = app.state.rewrap_job_runner
    assert runner._claim(job_id)
    assert runner._advance_page(job_id) is None
    runner._release_claim(job_id)


def _get_job(client, job_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _rotate_keyring(client, monkeypatch, keyring=KEYRING_V1_V2):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", keyring)


# --- request validation ----------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"workload_id": WORKLOAD},
        {"tenant_id": TENANT},
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "   ", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": "\t"},
    ],
)
def test_items_reject_missing_or_blank_scope(client, params):
    response = client.get(f"/v1/rewrap-jobs/{ZERO_UUID}/items", params=params)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "limit",
    ["", " ", "0", "201", "-1", "abc", "1.5", "10.0", "+5", "5 ", "0x10", "1e2"],
)
def test_items_reject_invalid_limit(client, limit):
    response = _items(client, ZERO_UUID, limit=limit)
    assert response.status_code == 422


@pytest.mark.parametrize("limit", ["1", "50", "200", "007"])
def test_items_accept_valid_limit_shapes(app, client, monkeypatch, limit):
    _rotate_keyring(client, monkeypatch)
    job_id = _submit(client).json()["job_id"]
    response = _items(client, job_id, limit=limit)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "job_id",
    ["", " ", "not-a-uuid", LETTERED_UUID.upper(), "0" * 32],
)
def test_items_reject_non_canonical_job_id(client, job_id):
    response = client.get(
        f"/v1/rewrap-jobs/{job_id}/items",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    # An empty segment is the dedicated 422 route; the rest are 422 too.
    assert response.status_code == 422


def test_items_reject_unsupported_and_repeated_params(client):
    response = _items(client, ZERO_UUID, bogus="1")
    assert response.status_code == 422
    response = client.get(
        f"/v1/rewrap-jobs/{ZERO_UUID}/items"
        f"?tenant_id={TENANT}&workload_id={WORKLOAD}&limit=1&limit=2"
    )
    assert response.status_code == 422
    response = client.get(
        f"/v1/rewrap-jobs/{ZERO_UUID}/items"
        f"?tenant_id={TENANT}&tenant_id={TENANT}&workload_id={WORKLOAD}"
    )
    assert response.status_code == 422


def test_items_reject_non_empty_body(client):
    response = client.request(
        "GET",
        f"/v1/rewrap-jobs/{ZERO_UUID}/items",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content="{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "cursor",
    [" ", "bad token!!", "AAA=", "!!!!", "a" * 5],
)
def test_items_reject_malformed_cursor(client, cursor):
    assert _items(client, ZERO_UUID, cursor=cursor).status_code == 422


# --- existence / scoping ---------------------------------------------------


def test_items_unknown_job_is_404(client):
    response = _items(client, ZERO_UUID)
    assert response.status_code == 404
    assert response.json()["detail"] == "rewrap job not found"


def test_items_cross_scope_job_is_indistinguishable_404(app, client, monkeypatch):
    _rotate_keyring(client, monkeypatch)
    job_id = _submit(client).json()["job_id"]
    assert _items(client, job_id, tenant=OTHER_TENANT).status_code == 404
    assert _items(client, job_id, workload=OTHER_WORKLOAD).status_code == 404
    assert (
        _items(client, job_id, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).status_code
        == 404
    )


# --- empty listings --------------------------------------------------------


def test_fresh_job_has_empty_complete_listing(app, client, monkeypatch):
    _rotate_keyring(client, monkeypatch)
    job_id = _submit(client).json()["job_id"]
    response = _items(client, job_id)
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": "", "complete": True}


def test_job_created_before_items_existed_pages_empty(tmp_path, monkeypatch):
    # A database written by an older deployment has jobs but no item
    # rows; opening it with the new build creates the table additively
    # and the legacy job pages as an empty, complete listing.
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    db_url = f"sqlite:///{tmp_path}/legacy.db"
    app1 = app_module.create_app(db_url)
    legacy_client = TestClient(app1)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(legacy_client).json()["job_id"]
    with app1.state.engine.begin() as conn:
        conn.execute(app_module.text("DROP TABLE rewrap_job_items"))
    app1.state.engine.dispose()

    app2 = app_module.create_app(db_url)
    try:
        upgraded = TestClient(app2)
        response = _items(upgraded, job_id)
        assert response.status_code == 200
        assert response.json() == {"items": [], "next_cursor": "", "complete": True}
        # The rest of the schema is untouched and the job itself reads back.
        assert _get_job(upgraded, job_id).status_code == 200
    finally:
        app2.state.engine.dispose()


# --- recorded content ------------------------------------------------------


def test_full_run_records_one_item_per_envelope(app, client, monkeypatch):
    data_ids = ["a0", "a1", "a2", "a3"]
    _seed(client, data_ids)
    _rotate_keyring(client, monkeypatch)
    job_id = _submit(client, limit=10).json()["job_id"]
    _run_job(app, job_id)
    assert _get_job(client, job_id).json()["status"] == "succeeded"

    response = _items(client, job_id)
    assert response.status_code == 200
    body = response.json()
    assert body["complete"] is True
    assert body["next_cursor"] == ""
    items = body["items"]
    assert [item["data_id"] for item in items] == data_ids
    assert [item["seq"] for item in items] == [1, 2, 3, 4]
    for item in items:
        assert list(item.keys()) == ITEM_FIELDS
        assert item["result"] == "rewrapped"
        assert item["old_key_version"] == 1
        assert item["new_key_version"] == 2
        assert RFC3339_RE.match(item["occurred_at"])
        # Parses as a real UTC timestamp.
        datetime.fromisoformat(item["occurred_at"])


def test_skipped_envelopes_are_recorded_as_skips(app, client, monkeypatch):
    _seed(client, ["a0", "a1"])
    _rotate_keyring(client, monkeypatch)
    first = _submit(client, limit=10).json()["job_id"]
    _run_job(app, first)
    assert _get_job(client, first).json()["status"] == "succeeded"

    # A second job over the same scope finds every envelope already
    # current: all items are skips with old == new == current version.
    second = _submit(client, limit=10).json()["job_id"]
    _run_job(app, second)
    assert _get_job(client, second).json()["status"] == "succeeded"
    items = _items(client, second).json()["items"]
    assert [item["result"] for item in items] == ["skipped", "skipped"]
    for item in items:
        assert item["old_key_version"] == 2
        assert item["new_key_version"] == 2
    # The first job's items are untouched by the second job.
    assert [i["result"] for i in _items(client, first).json()["items"]] == [
        "rewrapped",
        "rewrapped",
    ]


def test_items_do_not_leak_into_other_jobs_or_scopes(app, client, monkeypatch):
    _seed(client, ["a0", "a1"])
    _seed(client, ["b0"], tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    _rotate_keyring(client, monkeypatch)
    job_a = _submit(client, limit=10).json()["job_id"]
    job_b = _submit(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD, limit=10
    ).json()["job_id"]
    _run_job(app, job_a)
    _run_job(app, job_b)

    items_a = _items(client, job_a).json()["items"]
    assert [item["data_id"] for item in items_a] == ["a0", "a1"]
    items_b = _items(
        client, job_b, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    ).json()["items"]
    assert [item["data_id"] for item in items_b] == ["b0"]


# --- pagination ------------------------------------------------------------


def test_pagination_walks_all_items_without_gaps_or_duplicates(
    app, client, monkeypatch
):
    data_ids = [f"d{i}" for i in range(7)]
    _seed(client, data_ids)
    _rotate_keyring(client, monkeypatch)
    job_id = _submit(client, limit=10).json()["job_id"]
    _run_job(app, job_id)

    seen: list[dict] = []
    cursor = None
    pages = 0
    while True:
        params = {"limit": "3"}
        if cursor:
            params["cursor"] = cursor
        body = _items(client, job_id, **params).json()
        seen.extend(body["items"])
        pages += 1
        if body["complete"]:
            assert body["next_cursor"] == ""
            break
        assert body["next_cursor"]
        cursor = body["next_cursor"]
    assert pages == 3  # 3 + 3 + 1
    assert [item["data_id"] for item in seen] == data_ids
    assert [item["seq"] for item in seen] == list(range(1, 8))


def test_default_limit_is_50(app, client, monkeypatch):
    _seed(client, [f"d{i}" for i in range(3)])
    _rotate_keyring(client, monkeypatch)
    job_id = _submit(client, limit=10).json()["job_id"]
    _run_job(app, job_id)
    body = _items(client, job_id).json()
    assert len(body["items"]) == 3
    assert body["complete"] is True


def test_cursor_is_bound_to_job_scope_and_family(app, client, monkeypatch):
    _seed(client, ["a0", "a1", "a2"])
    _seed(client, ["b0", "b1", "b2"], tenant=OTHER_TENANT)
    _rotate_keyring(client, monkeypatch)
    job_a = _submit(client, limit=10).json()["job_id"]
    job_b = _submit(client, tenant=OTHER_TENANT, limit=10).json()["job_id"]
    _run_job(app, job_a)
    _run_job(app, job_b)

    first_page = _items(client, job_a, limit="1").json()
    cursor = first_page["next_cursor"]
    assert cursor

    # Same job, same scope: accepted.
    assert _items(client, job_a, cursor=cursor).status_code == 200
    # Another job, another tenant, another workload, a foreign cursor
    # family and a tampered token are all indistinguishable 422s.
    assert _items(client, job_b, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert _items(client, job_a, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert (
        _items(client, job_a, workload=OTHER_WORKLOAD, cursor=cursor).status_code
        == 422
    )
    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _items(client, job_a, cursor=tampered).status_code == 422
    events_cursor = client.get(
        f"/v1/rewrap-jobs/{job_a}/events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()["next_cursor"]
    if events_cursor:
        assert _items(client, job_a, cursor=events_cursor).status_code == 422


def test_pagination_is_stable_while_job_advances(app, client, monkeypatch):
    data_ids = ["a0", "a1", "a2", "a3"]
    _seed(client, data_ids)
    _rotate_keyring(client, monkeypatch)
    job_id = _submit(client, limit=1).json()["job_id"]

    # Commit exactly one envelope, then read the first page.
    _advance_one_page(app, job_id)
    first = _items(client, job_id, limit="2").json()
    assert [item["data_id"] for item in first["items"]] == ["a0"]
    assert first["complete"] is True

    # Advance the job further; a fresh first query sees the new commits
    # appended after the old ones, and the previously returned item is
    # byte-identical (immutable history).
    _advance_one_page(app, job_id)
    second = _items(client, job_id, limit="1").json()
    assert second["items"] == first["items"]
    cursor = second["next_cursor"]
    assert cursor
    rest: list[dict] = []
    while True:
        page = _items(client, job_id, cursor=cursor, limit="1").json()
        rest.extend(page["items"])
        if page["complete"]:
            break
        cursor = page["next_cursor"]
    assert [item["data_id"] for item in rest] == ["a1"]
    assert [item["seq"] for item in rest] == [2]

    # Finish the job; the full listing is gap-free and duplicate-free.
    _run_job(app, job_id)
    assert _get_job(client, job_id).json()["status"] == "succeeded"
    final = _items(client, job_id).json()["items"]
    assert [item["data_id"] for item in final] == data_ids
    assert [item["seq"] for item in final] == [1, 2, 3, 4]


# --- failure and cancellation semantics ------------------------------------


def test_failed_job_keeps_committed_items_and_omits_the_failed_envelope(
    app, client, monkeypatch
):
    data_ids = ["a0", "a1", "a2", "a3"]
    _seed(client, data_ids)
    _rotate_keyring(client, monkeypatch)

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
    monkeypatch.setattr(app_module, "rewrap_data_key", real_rewrap)

    job = _get_job(client, job_id).json()
    assert job["status"] == "failed"
    assert job["processed"] == 2

    # The two committed envelopes are listed; the failed envelope a2 is
    # not an item, and its failure reason stays on the event timeline.
    body = _items(client, job_id).json()
    assert [item["data_id"] for item in body["items"]] == ["a0", "a1"]
    assert [item["seq"] for item in body["items"]] == [1, 2]
    assert body["complete"] is True
    assert body["next_cursor"] == ""
    events = client.get(
        f"/v1/rewrap-jobs/{job_id}/events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()["events"]
    assert events[-1]["new_status"] == "failed"
    assert events[-1]["reason"] == "rewrap"


def test_recovered_job_continues_item_sequence_without_gaps(
    app, client, monkeypatch
):
    data_ids = ["a0", "a1", "a2"]
    _seed(client, data_ids)
    _rotate_keyring(client, monkeypatch)

    real_rewrap = app_module.rewrap_data_key
    calls = {"n": 0}

    def failing_rewrap(unwrapping_key, wrapping_key, wrapped_key):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("crypto backend exploded")
        return real_rewrap(unwrapping_key, wrapping_key, wrapped_key)

    monkeypatch.setattr(app_module, "rewrap_data_key", failing_rewrap)
    job_id = _submit(client, limit=10).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "rewrap_data_key", real_rewrap)
    assert _get_job(client, job_id).json()["status"] == "failed"

    # A post-restart retry resumes from the parked cursor: the retried
    # envelope commits exactly once and the sequence continues gap-free.
    _run_job(app, job_id, retry=True)
    assert _get_job(client, job_id).json()["status"] == "succeeded"
    items = _items(client, job_id).json()["items"]
    assert [item["data_id"] for item in items] == data_ids
    assert [item["seq"] for item in items] == [1, 2, 3]


def test_cancelled_job_keeps_only_committed_items(app, client, monkeypatch):
    _seed(client, ["a0", "a1", "a2", "a3"])
    _rotate_keyring(client, monkeypatch)
    job_id = _submit(client, limit=1).json()["job_id"]
    _advance_one_page(app, job_id)

    response = client.post(
        f"/v1/rewrap-jobs/{job_id}/cancel",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    body = _items(client, job_id).json()
    assert [item["data_id"] for item in body["items"]] == ["a0"]
    assert body["complete"] is True


# --- read-only behavior and storage failure --------------------------------


def test_items_query_is_read_only(app, client, monkeypatch):
    _seed(client, ["a0"])
    _rotate_keyring(client, monkeypatch)
    job_id = _submit(client, limit=10).json()["job_id"]
    _run_job(app, job_id)
    before = _get_job(client, job_id).json()
    assert _items(client, job_id).status_code == 200
    assert _get_job(client, job_id).json() == before


def test_items_storage_failure_returns_500_without_half_page(
    app, client, monkeypatch
):
    _seed(client, ["a0", "a1"])
    _rotate_keyring(client, monkeypatch)
    job_id = _submit(client, limit=10).json()["job_id"]
    _run_job(app, job_id)
    with app.state.engine.begin() as conn:
        conn.execute(app_module.text("DROP TABLE rewrap_job_items"))

    response = _items(client, job_id)
    assert response.status_code == 500
    assert response.json()["detail"] == "rewrap job items unavailable"


def test_items_are_persisted_and_survive_reopen(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    db_url = f"sqlite:///{tmp_path}/reopen.db"
    app1 = app_module.create_app(db_url)
    client1 = TestClient(app1)
    _seed(client1, ["a0", "a1"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client1, limit=10).json()["job_id"]
    _run_job(app1, job_id)
    first = _items(client1, job_id).json()
    app1.state.engine.dispose()

    app2 = app_module.create_app(db_url)
    try:
        client2 = TestClient(app2)
        assert _items(client2, job_id).json() == first
    finally:
        app2.state.engine.dispose()


def test_item_rows_match_job_progress_counters(app, client, monkeypatch):
    _seed(client, ["a0", "a1", "a2"])
    _rotate_keyring(client, monkeypatch)
    job_id = _submit(client, limit=10).json()["job_id"]
    _run_job(app, job_id)
    job = _get_job(client, job_id).json()
    with app.state.session_factory() as session:
        rows = (
            session.query(RewrapJobItem)
            .filter(RewrapJobItem.job_id == job_id)
            .order_by(RewrapJobItem.seq)
            .all()
        )
    assert len(rows) == job["processed"] == 3
    assert job["rewrapped"] == 3
    assert all(row.result == "rewrapped" for row in rows)
