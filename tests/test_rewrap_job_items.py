"""Tests for the per-envelope attempt detail of an asynchronous rewrap job:

GET /v1/rewrap-jobs/{job_id}/items

The endpoint ranges exactly one job by mandatory tenant/workload scope and
a canonical-UUID path id; it accepts only tenant_id, workload_id and an
optional cursor, carries no body, and returns the job's committed
per-envelope processing attempts as stable, immutable per-job
``attempt_seq`` items. The first (cursor-less) query fixes a replayable
snapshot high-water mark; attempts committed afterwards surface only in a
fresh first query. A resume cursor is HMAC-authenticated and bound to the
scope, job and snapshot, so it cannot be forged, tampered with, or
replayed across scopes, jobs, snapshots or cursor families. The query is
strictly read-only.

Like the other job tests, the app is built without entering its lifespan
so no pool runs; jobs (and the post-restart retry claim) are driven
synchronously.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import RewrapJob, RewrapJobEvent, RewrapJobItem
from proof_release.envelopes import b64url_decode, b64url_encode

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


def _fail_job(app, client, monkeypatch, *, tenant=TENANT, workload=WORKLOAD):
    # Envelopes are sealed under the v1 master key; running with a
    # current-v2 keyring that omits v1 fails the job on the first
    # unrotated envelope. The working keyring is restored afterwards.
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
            assert data["next_cursor"] == ""
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
    # Unknown and cross-scope jobs are indistinguishable.
    assert _items(client, job_id, tenant=OTHER_TENANT).status_code == 404
    assert _items(client, job_id, workload=OTHER_WORKLOAD).status_code == 404
    assert _items(client, ZERO_UUID).status_code == 404


# --- item content ----------------------------------------------------------


def test_rewrapped_items_record_versions_and_fields(app, client, monkeypatch):
    # Sealed under the fixture's v1 master key, then rotated to v2.
    _seed(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    response = _items(client, job_id)
    assert response.status_code == 200
    # Compact JSON with exactly one trailing newline.
    assert response.content.endswith(b"\n")
    assert b"\n" not in response.content[:-1]
    body = response.json()
    assert body["complete"] is True
    assert body["next_cursor"] == ""
    items = body["items"]
    assert [item["data_id"] for item in items] == ["a", "b", "c"]
    assert [item["attempt_seq"] for item in items] == [1, 2, 3]
    for item in items:
        assert list(item.keys()) == ITEM_FIELDS
        assert item["result"] == "rewrapped"
        assert item["old_key_version"] == 1
        assert item["new_key_version"] == 2
        # UTC RFC3339 and parseable.
        created = datetime.fromisoformat(item["created_at"])
        assert created.tzinfo is not None
        assert item["created_at"].endswith("+00:00")


def test_skipped_items_have_equal_versions(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    first = _submit(client).json()["job_id"]
    _run_job(app, first)
    # A second job over the same scope finds every envelope already
    # current: all skips, old == new == current version.
    second = _submit(client).json()["job_id"]
    _run_job(app, second)
    items = _items(client, second).json()["items"]
    assert [item["result"] for item in items] == ["skipped", "skipped"]
    for item in items:
        assert item["old_key_version"] == 2
        assert item["new_key_version"] == 2


def test_empty_job_pages_empty_and_complete(app, client):
    job_id = _submit(client).json()["job_id"]
    body = _items(client, job_id).json()
    assert body == {"items": [], "next_cursor": "", "complete": True}


def test_items_never_expose_material_or_exception_text(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    raw = _items(client, job_id).content.decode()
    assert PAYLOAD not in raw
    assert KEY_V1 not in raw and KEY_V2 not in raw


# --- failure and resume ----------------------------------------------------


def test_failed_attempt_commits_item_with_the_settlement(
    app, client, monkeypatch
):
    job_id = _fail_job(app, client, monkeypatch)
    job = client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    assert job["status"] == "failed"
    assert job["failed"] == 1
    items = _items(client, job_id).json()["items"]
    # The failing envelope's attempt is recorded exactly once, with
    # identical old/new versions and the fixed missing-key code.
    assert len(items) == 1
    assert items[0]["data_id"] == "a"
    assert items[0]["result"] == "missing-key"
    assert items[0]["old_key_version"] == 1
    assert items[0]["new_key_version"] == 1


def test_resume_appends_later_items_without_touching_old_ones(
    app, client, monkeypatch
):
    job_id = _fail_job(app, client, monkeypatch)
    before = _items(client, job_id).json()["items"]
    _run_job(app, job_id, retry=True)
    job = client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    assert job["status"] == "succeeded"
    after = _items(client, job_id).json()["items"]
    # The failed attempt's item is kept verbatim; the retry's attempt is
    # appended with a later sequence number.
    assert after[: len(before)] == before
    assert [item["attempt_seq"] for item in after] == [1, 2]
    assert after[-1]["data_id"] == "a"
    assert after[-1]["result"] == "rewrapped"
    assert after[-1]["old_key_version"] == 1
    assert after[-1]["new_key_version"] == 2


def test_cancelled_job_keeps_committed_items_only(app, client, monkeypatch):
    _seed(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client, limit=1).json()["job_id"]
    # Process exactly one envelope, then cancel before the next page.
    runner = app.state.rewrap_job_runner
    assert runner._claim(job_id)
    runner._advance_page(job_id)
    response = client.post(
        f"/v1/rewrap-jobs/{job_id}/cancel",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    items = _items(client, job_id).json()["items"]
    # Only the committed envelope has an item; the cancelled remainder
    # never appears.
    assert [item["data_id"] for item in items] == ["a"]
    assert items[0]["result"] == "rewrapped"


# --- pagination and snapshot -----------------------------------------------


def test_pagination_is_gap_free_and_strictly_ascending(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, [f"d{i}" for i in range(7)])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    items = _walk(client, app, job_id, 3, monkeypatch)
    assert [item["attempt_seq"] for item in items] == [1, 2, 3, 4, 5, 6, 7]
    assert [item["data_id"] for item in items] == [f"d{i}" for i in range(7)]


def test_first_query_fixes_the_snapshot(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    job_id = _submit(client, limit=1).json()["job_id"]
    runner = app.state.rewrap_job_runner
    assert runner._claim(job_id)
    runner._advance_page(job_id)  # commits "a" only

    monkeypatch.setattr(app_module, "REWRAP_JOB_ITEM_PAGE_SIZE", 100)
    first = _items(client, job_id).json()
    assert [item["data_id"] for item in first["items"]] == ["a"]
    assert first["complete"] is True

    # More attempts commit after the snapshot was fixed.
    runner._advance_page(job_id)  # commits "b" and settles succeeded
    runner._release_claim(job_id)

    # A fresh first query observes the new attempt; the old snapshot's
    # page (replayed through its own family) never changes.
    fresh = _items(client, job_id).json()
    assert [item["data_id"] for item in fresh["items"]] == ["a", "b"]


def test_snapshot_cursor_excludes_later_commits(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b", "c"])
    job_id = _submit(client, limit=2).json()["job_id"]
    runner = app.state.rewrap_job_runner
    assert runner._claim(job_id)
    runner._advance_page(job_id)  # commits "a", "b"

    monkeypatch.setattr(app_module, "REWRAP_JOB_ITEM_PAGE_SIZE", 1)
    page1 = _items(client, job_id).json()
    assert [item["data_id"] for item in page1["items"]] == ["a"]
    assert page1["complete"] is False

    # A third attempt commits after the snapshot was fixed.
    runner._advance_page(job_id)  # commits "c", settles succeeded
    runner._release_claim(job_id)

    # The in-flight family still pages only the fixed snapshot.
    page2 = _items(client, job_id, cursor=page1["next_cursor"]).json()
    assert [item["data_id"] for item in page2["items"]] == ["b"]
    assert page2["complete"] is True
    assert page2["next_cursor"] == ""


# --- cursor security -------------------------------------------------------


def _minted_cursor(client, app, monkeypatch, job_id):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "REWRAP_JOB_ITEM_PAGE_SIZE", 1)
    page = _items(client, job_id).json()
    assert page["next_cursor"]
    return page["next_cursor"]


def test_forged_and_tampered_cursors_are_422(app, client, monkeypatch):
    job_id = _submit(client).json()["job_id"]
    token = _minted_cursor(client, app, monkeypatch, job_id)
    # Flip one MAC bit: the payload still parses but no longer verifies.
    raw = bytearray(b64url_decode(token))
    raw[-1] ^= 0x01
    tampered = b64url_encode(bytes(raw))
    assert _items(client, job_id, cursor=tampered).status_code == 422
    assert _items(client, job_id, cursor=token + "x").status_code == 422


def test_cursor_is_bound_to_scope_and_job(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    job_id = _submit(client).json()["job_id"]
    other_job = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    _run_job(app, other_job)
    monkeypatch.setattr(app_module, "REWRAP_JOB_ITEM_PAGE_SIZE", 1)
    token = _items(client, job_id).json()["next_cursor"]
    assert token
    # Cross-scope and cross-job replays are indistinguishable 422s.
    assert _items(client, job_id, tenant=OTHER_TENANT, cursor=token).status_code == 422
    assert _items(client, job_id, workload=OTHER_WORKLOAD, cursor=token).status_code == 422
    assert _items(client, other_job, cursor=token).status_code == 422


def test_event_cursor_cannot_be_replayed_against_items(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    monkeypatch.setattr(app_module, "REWRAP_JOB_EVENT_PAGE_SIZE", 1)
    events_page = client.get(
        f"/v1/rewrap-jobs/{job_id}/events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    assert events_page["next_cursor"]
    # A cursor of another family (authenticated with the same secret) is
    # rejected here.
    assert (
        _items(client, job_id, cursor=events_page["next_cursor"]).status_code
        == 422
    )


# --- read-only guarantee ---------------------------------------------------


def test_query_does_not_change_job_or_items(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    first = _items(client, job_id).json()
    job_before = client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    _items(client, job_id)
    _items(client, job_id, cursor="")
    assert _items(client, job_id).json() == first
    job_after = client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    assert job_after == job_before
    with app.state.session_factory() as session:
        assert session.query(RewrapJobItem).count() == 1
        # No lifecycle event is appended by a read.
        assert (
            session.query(RewrapJobEvent)
            .filter(RewrapJobEvent.job_id == job_id)
            .count()
            == 3  # submitted, executed, completed
        )


# --- persistence upgrade ---------------------------------------------------


def test_reopened_database_keeps_items(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/upgrade.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client1, ["a"])
    job_id = _submit(client1).json()["job_id"]
    _run_job(app1, job_id)
    expected = _items(client1, job_id).json()
    app1.state.engine.dispose()

    # Reopening the same database upgrades in place and serves the
    # committed items unchanged.
    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    try:
        assert _items(client2, job_id).json() == expected
    finally:
        app2.state.engine.dispose()
