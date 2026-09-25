"""Tests for the read-only asynchronous rewrap job history:

GET /v1/rewrap-jobs

The endpoint ranges committed RewrapJob rows by mandatory tenant/workload,
optionally narrowed by job id, status and an inclusive created-at window,
with stable job_id keyset pagination. It is strictly read-only: jobs are
written and advanced only by POST /v1/rewrap-jobs and its background
runner; a history query never writes an audit row, changes a status or
advances a cursor. Like the rest of the job tests, the app is built
without entering its lifespan so no pool runs; jobs are driven
synchronously where a terminal status is needed.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import AuditEvent, RewrapJob
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"
#: UUID with hex letters, used to prove uppercase spellings are rejected.
LETTERED_UUID = "abcdefab-cdef-abcd-efab-cdefabcdefab"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V2_BYTES = b"fedcba9876543210fedcba9876543210"
KEY_V1 = b64url_encode(KEY_V1_BYTES)
KEY_V2 = b64url_encode(KEY_V2_BYTES)

PAYLOAD = "history-secret 🔐"


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


KEYRING_V1_V2 = _keyring(2, {1: KEY_V1, 2: KEY_V2})
KEYRING_V2_ONLY = _keyring(2, {2: KEY_V2})

JOB_FIELDS = [
    "job_id",
    "status",
    "processed",
    "rewrapped",
    "skipped",
    "failed",
    "next_cursor",
    "complete",
    "created_at",
    "updated_at",
]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = app_module.create_app(f"sqlite:///{tmp_path}/history.db")
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


def _history(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        "/v1/rewrap-jobs",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


def _run_job(app, job_id):
    app.state.rewrap_job_runner.run(job_id)


def _get_job(client, job_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _walk(client, page_size, monkeypatch, **params):
    monkeypatch.setattr(app_module, "REWRAP_JOB_HISTORY_PAGE_SIZE", page_size)
    rows = []
    token = None
    for _ in range(100):
        query = dict(params)
        if token:
            query["cursor"] = token
        data = _history(client, **query).json()
        rows.extend(data["jobs"])
        if data["complete"]:
            return rows
        token = data["next_cursor"]
        assert token
    raise AssertionError("pagination never completed")  # pragma: no cover


# --- request validation ----------------------------------------------------


def test_query_requires_scope_parameters(client):
    assert client.get("/v1/rewrap-jobs").status_code == 422
    assert (
        client.get("/v1/rewrap-jobs", params={"workload_id": WORKLOAD}).status_code
        == 422
    )
    assert (
        client.get("/v1/rewrap-jobs", params={"tenant_id": TENANT}).status_code == 422
    )


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"workload_id": "\t"},
        # job_id must be a canonical UUID: missing value, whitespace and
        # any non-canonical spelling are all 422.
        {"job_id": ""},
        {"job_id": "   "},
        {"job_id": "not-a-uuid"},
        {"job_id": "abc123"},
        {"job_id": ZERO_UUID[:-1] + "Z"},
        {"job_id": "  " + ZERO_UUID},
        {"job_id": LETTERED_UUID.upper()},
        # status is a fixed enum.
        {"status": ""},
        {"status": " QUEUED "},
        {"status": "QUEUED"},
        {"status": "pending"},
        {"status": "complete"},
        {"status": "done"},
        # Time filters must be explicit-UTC RFC3339 strings.
        {"created_after": "not-a-timestamp"},
        {"created_after": "2026-01-01"},
        {"created_after": "2026-01-01T00:00:00"},
        {"created_before": "2026-01-01T00:00:00"},
        {"created_after": "2026-01-01T01:00:00+01:00"},
        {"created_after": ""},
        {"created_before": "  "},
        # A reversed window is rejected.
        {
            "created_after": "2026-01-02T00:00:00Z",
            "created_before": "2026-01-01T00:00:00Z",
        },
        # Malformed cursors and unknown parameters.
        {"cursor": "   "},
        {"cursor": "not base64!!"},
        {"cursor": "AAA="},
        {"bogus": "value"},
        {"limit": "10"},
        {"page_size": "2"},
    ],
)
def test_query_rejects_invalid_parameters(client, params):
    response = _history(client, **params)
    assert response.status_code == 422, response.text


def test_query_rejects_non_utc_offset_even_if_instant_matches(client):
    response = _history(
        client,
        created_after="2026-01-01T02:00:00+02:00",
        created_before="2026-01-01T00:00:00Z",
    )
    assert response.status_code == 422


def test_equal_time_window_is_accepted(client):
    response = _history(
        client,
        created_after="2026-01-01T00:00:00Z",
        created_before="2026-01-01T00:00:00Z",
    )
    assert response.status_code == 200
    assert response.json() == {"jobs": [], "next_cursor": "", "complete": True}


def test_non_empty_body_is_rejected(client):
    for body in (b"x", b"   ", b"{}"):
        response = client.request(
            "GET",
            "/v1/rewrap-jobs",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            content=body,
        )
        assert response.status_code == 422


def test_invalid_parameters_read_no_state(app, client):
    _history(client, status="nope")
    _history(client, job_id="not-a-uuid")
    _history(client, created_after="2026-01-02T00:00:00Z",
             created_before="2026-01-01T00:00:00Z")
    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 0
        assert session.query(AuditEvent).count() == 0


# --- 404 semantics ---------------------------------------------------------


def test_unknown_job_identifier_returns_404(client):
    assert _history(client, job_id=ZERO_UUID).status_code == 404


def test_cross_scope_job_identifier_returns_404(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    assert _history(client, tenant=OTHER_TENANT, job_id=job_id).status_code == 404
    assert _history(client, workload=OTHER_WORKLOAD, job_id=job_id).status_code == 404
    assert (
        _history(
            client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD, job_id=job_id
        ).status_code
        == 404
    )


# --- empty scope / wire format ---------------------------------------------


def test_empty_scope_returns_empty_completed_page(client):
    response = _history(client)
    assert response.status_code == 200
    assert response.json() == {"jobs": [], "next_cursor": "", "complete": True}


def test_response_is_compact_json_with_single_trailing_newline(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    _submit(client)
    response = _history(client)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == ["jobs", "next_cursor", "complete"]


def test_each_entry_matches_single_job_progress_fields(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, [f"e{i}" for i in range(3)])
    for _ in range(3):
        job_id = _submit(client).json()["job_id"]
        _run_job(app, job_id)

    rows = _history(client).json()["jobs"]
    assert len(rows) == 3
    for row in rows:
        assert list(row) == JOB_FIELDS
        single = _get_job(client, row["job_id"]).json()
        assert list(single) == JOB_FIELDS
        assert row == single
        assert isinstance(row["job_id"], str) and len(row["job_id"]) == 36
        assert row["status"] in ("queued", "running", "succeeded", "failed")
        for key in ("processed", "rewrapped", "skipped", "failed"):
            assert isinstance(row[key], int) and not isinstance(row[key], bool)
        assert isinstance(row["next_cursor"], str)
        assert isinstance(row["complete"], bool)
        for key in ("created_at", "updated_at"):
            parsed = datetime.fromisoformat(row[key])
            assert parsed.utcoffset() == timedelta(0)


def test_no_floats_or_non_finite_values(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    _submit(client)

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

    _check(json.loads(_history(client).content))


# --- scoping, ordering and filters -----------------------------------------


def test_history_is_scoped_to_tenant_and_workload(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a1", "a2"], tenant=TENANT, workload=WORKLOAD)
    _seed(client, ["b1"], tenant=OTHER_TENANT, workload=WORKLOAD)
    _seed(client, ["c1"], tenant=TENANT, workload=OTHER_WORKLOAD)
    _submit(client)
    _submit(client)
    _submit(client, tenant=OTHER_TENANT)
    _submit(client, workload=OTHER_WORKLOAD)

    rows = _history(client).json()["jobs"]
    assert len(rows) == 2
    for row in rows:
        single = _get_job(client, row["job_id"]).json()
        assert single["status"] == "queued"
    assert len(_history(client, tenant=OTHER_TENANT).json()["jobs"]) == 1
    assert len(_history(client, workload=OTHER_WORKLOAD).json()["jobs"]) == 1


def test_history_is_ordered_by_job_id_ascending(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    for _ in range(9):
        assert _submit(client).status_code == 202
    ids = [row["job_id"] for row in _history(client).json()["jobs"]]
    assert ids == sorted(ids)
    assert len(set(ids)) == 9


def test_filter_by_status(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, [f"e{i}" for i in range(4)])
    queued_id = _submit(client).json()["job_id"]
    succeeded_id = _submit(client).json()["job_id"]
    _run_job(app, succeeded_id)

    assert [r["job_id"] for r in _history(client, status="queued").json()["jobs"]] == [
        queued_id
    ]
    assert [
        r["job_id"] for r in _history(client, status="succeeded").json()["jobs"]
    ] == [succeeded_id]
    assert _history(client, status="running").json()["jobs"] == []
    assert _history(client, status="failed").json()["jobs"] == []


def test_failed_jobs_appear_in_history(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    rows = _history(client, status="failed").json()["jobs"]
    assert [r["job_id"] for r in rows] == [job_id]
    assert rows[0]["status"] == "failed"
    assert rows[0]["complete"] is False


def test_filter_by_explicit_job_id_returns_exactly_that_job(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    for _ in range(3):
        _submit(client)
    target = _submit(client).json()["job_id"]
    rows = _history(client, job_id=target).json()["jobs"]
    assert len(rows) == 1
    assert rows[0]["job_id"] == target


def test_time_window_is_inclusive_and_utc(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    first_id = _submit(client).json()["job_id"]
    first_at = _get_job(client, first_id).json()["created_at"]
    second_id = _submit(client).json()["job_id"]
    second_at = _get_job(client, second_id).json()["created_at"]

    # Both bounds inclusive: the exact created_at instant keeps the row.
    rows = _history(
        client, created_after=first_at, created_before=second_at
    ).json()["jobs"]
    assert {r["job_id"] for r in rows} == {first_id, second_id}

    after_rows = _history(client, created_after=second_at).json()["jobs"]
    assert [r["job_id"] for r in after_rows] == [second_id]
    before_rows = _history(client, created_before=first_at).json()["jobs"]
    assert [r["job_id"] for r in before_rows] == [first_id]

    # A window in the distant future matches nothing but is still a
    # successful empty page.
    assert (
        _history(
            client, created_after="2100-01-01T00:00:00Z"
        ).json()
        == {"jobs": [], "next_cursor": "", "complete": True}
    )


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_job_once_in_order(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    for _ in range(7):
        _submit(client)
    rows = _walk(client, 3, monkeypatch)
    ids = [r["job_id"] for r in rows]
    assert len(rows) == 7
    assert ids == sorted(ids)
    assert len(set(ids)) == 7


def test_page_carries_cursor_and_complete_flag(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    for _ in range(5):
        _submit(client)
    monkeypatch.setattr(app_module, "REWRAP_JOB_HISTORY_PAGE_SIZE", 2)

    first = _history(client).json()
    assert len(first["jobs"]) == 2
    assert first["complete"] is False
    assert first["next_cursor"]

    second = _history(client, cursor=first["next_cursor"]).json()
    assert len(second["jobs"]) == 2
    assert second["complete"] is False
    first_ids = [r["job_id"] for r in first["jobs"]]
    second_ids = [r["job_id"] for r in second["jobs"]]
    assert first_ids < second_ids

    last = _history(client, cursor=second["next_cursor"]).json()
    assert len(last["jobs"]) == 1
    assert last["complete"] is True
    assert last["next_cursor"] == ""


def test_replaying_same_cursor_returns_identical_page(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    for _ in range(5):
        _submit(client)
    monkeypatch.setattr(app_module, "REWRAP_JOB_HISTORY_PAGE_SIZE", 2)
    cursor = _history(client).json()["next_cursor"]
    first = _history(client, cursor=cursor)
    second = _history(client, cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_string_cursor_equals_default(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    for _ in range(3):
        _submit(client)
    monkeypatch.setattr(app_module, "REWRAP_JOB_HISTORY_PAGE_SIZE", 2)
    assert _history(client).content == _history(client, cursor="").content


def test_status_updates_do_not_change_page_membership(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, [f"e{i}" for i in range(6)])
    job_ids = [_submit(client).json()["job_id"] for _ in range(6)]
    monkeypatch.setattr(app_module, "REWRAP_JOB_HISTORY_PAGE_SIZE", 3)

    cursor = _history(client).json()["next_cursor"]
    second_before = [r["job_id"] for r in _history(client, cursor=cursor).json()["jobs"]]
    assert len(second_before) == 3

    # Every job settles while the client still holds the cursor. The
    # status (and counters/timestamps) changes, but job ids are the
    # immutable ordering key, so the replayed page neither duplicates nor
    # skips.
    for job_id in job_ids:
        _run_job(app, job_id)

    second_after = [r["job_id"] for r in _history(client, cursor=cursor).json()["jobs"]]
    assert second_after == second_before

    walked = _walk(client, 3, monkeypatch)
    assert [r["job_id"] for r in walked] == sorted(job_ids)
    assert {r["status"] for r in walked} == {"succeeded"}


# --- cursor authentication -------------------------------------------------


def test_tampered_forged_or_cross_scope_cursor_returns_422(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    for _ in range(3):
        _submit(client)
    monkeypatch.setattr(app_module, "REWRAP_JOB_HISTORY_PAGE_SIZE", 1)
    cursor = _history(client).json()["next_cursor"]
    assert cursor

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _history(client, cursor=tampered).status_code == 422

    forged = b64url_encode(
        b'{"k":"rewrap-job-history-v1","t":"tenant-a","w":"workload-1"}'
        + b"0" * 32
    )
    assert _history(client, cursor=forged).status_code == 422

    assert _history(client, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert _history(client, workload=OTHER_WORKLOAD, cursor=cursor).status_code == 422


def test_cursor_cannot_cross_filters_or_kinds(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    for _ in range(3):
        _submit(client)
    monkeypatch.setattr(app_module, "REWRAP_JOB_HISTORY_PAGE_SIZE", 1)
    cursor = _history(client).json()["next_cursor"]
    target = _history(client).json()["jobs"][0]["job_id"]

    assert _history(client, status="queued", cursor=cursor).status_code == 422
    assert _history(client, job_id=target, cursor=cursor).status_code == 422
    assert (
        _history(
            client, created_after="2000-01-01T00:00:00Z", cursor=cursor
        ).status_code
        == 422
    )
    assert (
        _history(
            client, created_before="2100-01-01T00:00:00Z", cursor=cursor
        ).status_code
        == 422
    )

    # Cursors from every other HMAC family are rejected, including the
    # structurally similar rewrap-batch cursor and the other listings.
    from proof_release.app import (
        _encode_audit_event_cursor,
        _encode_cursor,
        _encode_grant_audit_cursor,
        _encode_revocation_cursor,
    )

    foreign_rewrap = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _history(client, cursor=foreign_rewrap).status_code == 422
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
    assert _history(client, cursor=foreign_grant).status_code == 422
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
    assert _history(client, cursor=foreign_event).status_code == 422
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
    assert _history(client, cursor=foreign_revocation).status_code == 422


def test_cursor_bound_filter_walks_stably(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    succeeded_id = _submit(client).json()["job_id"]
    _run_job(app, succeeded_id)
    for _ in range(3):
        _submit(client)

    rows = _walk(client, 1, monkeypatch, status="succeeded")
    assert [r["job_id"] for r in rows] == [succeeded_id]
    assert all(r["status"] == "succeeded" for r in rows)


# --- read-only behaviour ---------------------------------------------------


def test_query_writes_no_state_and_does_not_advance_jobs(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a", "b", "c"])
    job_id = _submit(client, limit=2).json()["job_id"]
    before = _get_job(client, job_id).json()

    for _ in range(4):
        assert _history(client).status_code == 200
        assert _history(client, status="queued").status_code == 200
        assert _history(client, job_id=job_id).status_code == 200

    after = _get_job(client, job_id).json()
    assert after == before
    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 1
        # History queries never append compliance events.
        assert session.query(AuditEvent).count() == 0
        job = session.get(RewrapJob, job_id)
        assert job.status == "queued"
        assert job.processed == 0
        assert job.next_cursor == ""


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_page(app, client, monkeypatch):
    from sqlalchemy import text

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    _submit(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_jobs"))
    assert _history(client).status_code == 500


# --- persistence -----------------------------------------------------------


def test_history_available_after_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = app_module.create_app(url)
    first_client = TestClient(first)
    _seed(first_client, ["a"])
    job_id = first_client.post(
        "/v1/rewrap-jobs",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()["job_id"]
    first.state.engine.dispose()

    second = app_module.create_app(url)
    try:
        second_client = TestClient(second)
        rows = second_client.get(
            "/v1/rewrap-jobs",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).json()["jobs"]
        assert [r["job_id"] for r in rows] == [job_id]
        assert rows[0]["status"] == "queued"
    finally:
        second.state.engine.dispose()


# --- secrecy ---------------------------------------------------------------


def test_history_never_exposes_payload_or_key_material(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    _seed(client, ["a"])
    accepted = _submit(client)
    job_id = accepted.json()["job_id"]
    _run_job(app, job_id)
    response = _history(client)
    for secret in (
        PAYLOAD,
        KEY_V1,
        KEY_V2,
        "wrapped_key",
        "ciphertext",
        "payload",
        "master_key",
        "certificate",
    ):
        assert secret not in response.text
    with app.state.session_factory() as session:
        job = session.get(RewrapJob, job_id)
        for forbidden in ("wrapped_key", "ciphertext", "iv", "tag"):
            assert not hasattr(job, forbidden)
