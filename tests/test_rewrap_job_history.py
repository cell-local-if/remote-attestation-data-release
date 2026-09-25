"""Tests for the read-only rewrap job history listing:

GET /v1/rewrap-jobs.

The listing is scoped by mandatory tenant/workload query parameters and
may be narrowed by an exact job identifier, a job status and an
inclusive created-at UTC window. Results are ordered by job_id ascending
with an HMAC-authenticated, scope- and filter-bound keyset cursor. The
handler is read-only: it never writes audit events, never mutates a job
and never advances an envelope.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.app import (
    _encode_grant_audit_cursor,
    _encode_rewrap_job_history_cursor,
    _encode_cursor,
)
from proof_release.db import RewrapJob
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
PAYLOAD = "history-secret-payload 🔐"

KEY_V1 = b64url_encode(b"0123456789abcdef0123456789abcdef")

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
    # No lifespan: submission persists the queued row synchronously and
    # the listing only ever reads committed state.
    return TestClient(app)


def _submit(client, *, tenant=TENANT, workload=WORKLOAD, limit=None):
    body: dict = {"tenant_id": tenant, "workload_id": workload}
    if limit is not None:
        body["limit"] = limit
    response = client.post("/v1/rewrap-jobs", json=body)
    assert response.status_code == 202
    return response.json()["job_id"]


def _history(client, **params):
    return client.get("/v1/rewrap-jobs", params=params)


def _scope(**extra):
    return {"tenant_id": TENANT, "workload_id": WORKLOAD, **extra}


def _run_job(app, job_id):
    app.state.rewrap_job_runner.run(job_id)


def _set_created_at(app, job_id, value):
    with app.state.session_factory() as session:
        job = session.get(RewrapJob, job_id)
        job.created_at = value
        session.commit()


# --- scope, shape and field contract ----------------------------------------


def test_empty_scope_returns_empty_page(client):
    response = _history(client, **_scope())
    assert response.status_code == 200
    assert response.json() == {"jobs": [], "next_cursor": "", "complete": True}


def test_jobs_are_listed_in_job_id_order_with_progress_fields(app, client):
    job_ids = [_submit(client) for _ in range(3)]
    _run_job(app, job_ids[0])

    response = _history(client, **_scope())
    assert response.status_code == 200
    body = response.json()
    assert [job["job_id"] for job in body["jobs"]] == sorted(job_ids)
    assert body["complete"] is True
    assert body["next_cursor"] == ""

    first = body["jobs"][0]
    # Same field order and types as the single-job progress query.
    assert list(first.keys()) == JOB_FIELDS
    single = client.get(
        f"/v1/rewrap-jobs/{first['job_id']}", params=_scope()
    ).json()
    assert first == single
    for job in body["jobs"]:
        for key in ("processed", "rewrapped", "skipped", "failed"):
            assert isinstance(job[key], int) and not isinstance(job[key], bool)
        assert isinstance(job["complete"], bool)
        assert isinstance(job["next_cursor"], str)
        for key in ("created_at", "updated_at"):
            parsed = datetime.fromisoformat(job[key])
            assert parsed.utcoffset() == timedelta(0)
    statuses = {job["job_id"]: job["status"] for job in body["jobs"]}
    assert statuses[job_ids[0]] == "succeeded"
    assert statuses[job_ids[1]] == statuses[job_ids[2]] == "queued"


def test_history_is_scoped_to_tenant_and_workload(client):
    mine = _submit(client)
    _submit(client, tenant="tenant-b")
    _submit(client, workload="workload-2")

    body = _history(client, **_scope()).json()
    assert [job["job_id"] for job in body["jobs"]] == [mine]
    assert _history(client, tenant_id="tenant-b", workload_id=WORKLOAD).json()[
        "jobs"
    ][0]["job_id"] != mine


def test_history_survives_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    db_url = f"sqlite:///{tmp_path}/restart.db"
    app_one = app_module.create_app(db_url)
    client_one = TestClient(app_one)
    job_id = _submit(client_one)
    _run_job(app_one, job_id)
    app_one.state.engine.dispose()

    app_two = app_module.create_app(db_url)
    client_two = TestClient(app_two)
    body = _history(client_two, **_scope()).json()
    assert [job["job_id"] for job in body["jobs"]] == [job_id]
    assert body["jobs"][0]["status"] == "succeeded"
    app_two.state.engine.dispose()


# --- request validation (422, no state read) ---------------------------------


def test_missing_or_blank_scope_returns_422(client):
    assert client.get("/v1/rewrap-jobs").status_code == 422
    assert client.get("/v1/rewrap-jobs", params={"tenant_id": TENANT}).status_code == 422
    assert _history(client, tenant_id="  ", workload_id=WORKLOAD).status_code == 422
    assert _history(client, tenant_id=TENANT, workload_id="").status_code == 422


def test_unknown_query_parameter_returns_422(client):
    assert _history(client, **_scope(limit="10")).status_code == 422
    assert _history(client, **_scope(next_cursor="")).status_code == 422


def test_non_empty_body_returns_422_without_reading_state(app, client):
    from sqlalchemy import text

    _submit(client)
    # Dropping the table proves the rejection happens before any state
    # is read: a read would surface a 500 instead.
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_jobs"))
    for body in (b"{}", b"  ", b"not-json"):
        response = client.request(
            "GET", "/v1/rewrap-jobs", params=_scope(), content=body
        )
        assert response.status_code == 422


@pytest.mark.parametrize(
    "job_id",
    [
        "",
        "   ",
        "not-a-uuid",
        "00000000-0000-0000-0000-00000000000",
        "00000000-0000-0000-0000-000000000000 ",  # surrounding whitespace
        "AAAAAAAA-0000-0000-0000-000000000000",  # uppercase is non-canonical
    ],
)
def test_non_canonical_job_id_returns_422(client, job_id):
    assert _history(client, **_scope(job_id=job_id)).status_code == 422


def test_unknown_or_cross_scope_job_id_returns_404(client):
    mine = _submit(client)
    assert (
        _history(
            client,
            **_scope(job_id="00000000-0000-0000-0000-000000000000"),
        ).status_code
        == 404
    )
    assert (
        _history(client, tenant_id="tenant-b", workload_id=WORKLOAD, job_id=mine)
        .status_code
        == 404
    )
    assert (
        _history(client, tenant_id=TENANT, workload_id="workload-2", job_id=mine)
        .status_code
        == 404
    )


def test_job_id_filter_returns_exactly_that_job(client):
    mine = _submit(client)
    _submit(client)
    body = _history(client, **_scope(job_id=mine)).json()
    assert [job["job_id"] for job in body["jobs"]] == [mine]
    assert body["complete"] is True


@pytest.mark.parametrize("status", ["queued", "running", "succeeded", "failed"])
def test_status_filter_accepts_each_job_status(app, client, status):
    response = _history(client, **_scope(status=status))
    assert response.status_code == 200


@pytest.mark.parametrize("status", ["", "  ", "pending", "SUCCEEDED", "done"])
def test_invalid_status_returns_422(client, status):
    assert _history(client, **_scope(status=status)).status_code == 422


def test_status_filter_narrows_results(app, client):
    succeeded = _submit(client)
    _run_job(app, succeeded)
    queued = _submit(client)

    body = _history(client, **_scope(status="succeeded")).json()
    assert [job["job_id"] for job in body["jobs"]] == [succeeded]
    body = _history(client, **_scope(status="queued")).json()
    assert [job["job_id"] for job in body["jobs"]] == [queued]
    body = _history(client, **_scope(status="failed")).json()
    assert body["jobs"] == []


# --- created-at window --------------------------------------------------------


def test_created_at_window_filters_inclusively(app, client):
    early = _submit(client)
    middle = _submit(client)
    late = _submit(client)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _set_created_at(app, early, base)
    _set_created_at(app, middle, base + timedelta(hours=1))
    _set_created_at(app, late, base + timedelta(hours=2))

    after = (base + timedelta(minutes=30)).isoformat()
    before = (base + timedelta(hours=2)).isoformat()
    body = _history(
        client, **_scope(created_after=after, created_before=before)
    ).json()
    # Both bounds are inclusive; results stay in job_id order.
    assert [job["job_id"] for job in body["jobs"]] == sorted([middle, late])

    # Equality is a valid single-instant window.
    instant = (base + timedelta(hours=1)).isoformat()
    body = _history(
        client, **_scope(created_after=instant, created_before=instant)
    ).json()
    assert [job["job_id"] for job in body["jobs"]] == [middle]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "not-a-timestamp",
        "2026-01-01 00:00:00",  # not RFC3339
        "2026-01-01T00:00:00",  # naive: no explicit UTC offset
        "2026-01-01T00:00:00+01:00",  # non-UTC offset
    ],
)
def test_illegal_created_bound_returns_422(client, value):
    assert _history(client, **_scope(created_after=value)).status_code == 422
    assert _history(client, **_scope(created_before=value)).status_code == 422


def test_reversed_created_window_returns_422(client):
    assert (
        _history(
            client,
            **_scope(
                created_after="2026-01-02T00:00:00Z",
                created_before="2026-01-01T00:00:00Z",
            ),
        ).status_code
        == 422
    )


def test_equivalent_utc_spellings_share_one_cursor_domain(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "REWRAP_JOB_HISTORY_PAGE_SIZE", 1)
    _submit(client)
    _submit(client)
    first = _history(
        client, **_scope(created_after="2020-01-01T00:00:00Z")
    ).json()
    cursor = first["next_cursor"]
    assert cursor
    # "+00:00" and "Z" spell the same instant: the cursor stays valid.
    replay = _history(
        client, **_scope(created_after="2020-01-01T00:00:00+00:00", cursor=cursor)
    )
    assert replay.status_code == 200


# --- pagination and cursor authentication -------------------------------------


def test_pagination_walks_all_jobs_without_duplicates_or_gaps(
    app, client, monkeypatch
):
    monkeypatch.setattr(app_module, "REWRAP_JOB_HISTORY_PAGE_SIZE", 2)
    job_ids = sorted(_submit(client) for _ in range(5))

    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        params = _scope()
        if cursor is not None:
            params["cursor"] = cursor
        body = _history(client, **params).json()
        seen.extend(job["job_id"] for job in body["jobs"])
        pages += 1
        if body["complete"]:
            assert body["next_cursor"] == ""
            break
        cursor = body["next_cursor"]
        assert cursor
    assert pages == 3
    assert seen == job_ids  # exact order, no duplicates, no gaps


def test_default_and_explicit_empty_cursor_start_at_the_beginning(client):
    job_ids = sorted(_submit(client) for _ in range(2))
    default = _history(client, **_scope()).json()
    explicit = _history(client, **_scope(cursor="")).json()
    assert default == explicit
    assert [job["job_id"] for job in default["jobs"]] == job_ids


def test_replaying_a_cursor_returns_the_same_page(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "REWRAP_JOB_HISTORY_PAGE_SIZE", 2)
    job_ids = [_submit(client) for _ in range(4)]
    first = _history(client, **_scope()).json()
    cursor = first["next_cursor"]

    # Concurrent progress elsewhere does not move the keyset boundary.
    _run_job(app, job_ids[0])
    _run_job(app, job_ids[2])

    replay_one = _history(client, **_scope(cursor=cursor)).json()
    replay_two = _history(client, **_scope(cursor=cursor)).json()
    assert replay_one == replay_two
    expected_tail = sorted(job_ids)[2:]
    assert [job["job_id"] for job in replay_one["jobs"]] == expected_tail
    # No overlap with the first page and nothing skipped.
    assert not set(job["job_id"] for job in first["jobs"]) & set(expected_tail)


def test_cursor_is_bound_to_scope_and_filters(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "REWRAP_JOB_HISTORY_PAGE_SIZE", 1)
    _submit(client)
    _submit(client)
    cursor = _history(client, **_scope()).json()["next_cursor"]
    assert cursor

    # Cross-scope and cross-filter replays are indistinguishable 422s.
    assert (
        _history(client, tenant_id="tenant-b", workload_id=WORKLOAD, cursor=cursor)
        .status_code
        == 422
    )
    assert (
        _history(client, tenant_id=TENANT, workload_id="workload-2", cursor=cursor)
        .status_code
        == 422
    )
    assert _history(client, **_scope(cursor=cursor, status="queued")).status_code == 422
    assert (
        _history(
            client, **_scope(cursor=cursor, created_after="2020-01-01T00:00:00Z")
        ).status_code
        == 422
    )


@pytest.mark.parametrize("cursor", [" ", "!!!", "not a cursor", "a" * 40])
def test_malformed_cursor_returns_422(client, cursor):
    assert _history(client, **_scope(cursor=cursor)).status_code == 422


def test_forged_and_tampered_cursors_return_422(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "REWRAP_JOB_HISTORY_PAGE_SIZE", 1)
    _submit(client)
    _submit(client)
    genuine = _history(client, **_scope()).json()["next_cursor"]

    # A cursor minted under a different secret is a forgery.
    monkeypatch.setenv("PROOF_RELEASE_CURSOR_SECRET", "another-secret")
    assert _history(client, **_scope(cursor=genuine)).status_code == 422
    monkeypatch.delenv("PROOF_RELEASE_CURSOR_SECRET")

    # Flipping any authenticated bit invalidates the MAC.
    tampered = genuine[:-2] + ("A" if genuine[-2] != "A" else "B") + genuine[-1]
    assert _history(client, **_scope(cursor=tampered)).status_code == 422


def test_foreign_family_cursors_return_422(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "REWRAP_JOB_HISTORY_PAGE_SIZE", 1)
    _submit(client)
    _submit(client)

    # A rewrap-batch cursor and a grant-audit cursor share the HMAC
    # secret but belong to other query families.
    batch_cursor = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _history(client, **_scope(cursor=batch_cursor)).status_code == 422
    grant_cursor = _encode_grant_audit_cursor(
        TENANT,
        WORKLOAD,
        "00000000-0000-0000-0000-000000000000",
        grant_id="",
        decision_id="",
        data_id="",
        status="",
        issued_after="",
        issued_before="",
    )
    assert _history(client, **_scope(cursor=grant_cursor)).status_code == 422

    # A history cursor minted for a different filter set is bound out.
    other_filter = _encode_rewrap_job_history_cursor(
        TENANT,
        WORKLOAD,
        "00000000-0000-0000-0000-000000000000",
        job_id="",
        status="failed",
        created_after="",
        created_before="",
    )
    assert _history(client, **_scope(cursor=other_filter)).status_code == 422


# --- failure atomicity, wire format and secrecy -------------------------------


def test_query_failure_returns_500_with_no_partial_page(app, client):
    from sqlalchemy import text

    _submit(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_jobs"))
    response = _history(client, **_scope())
    assert response.status_code == 500
    assert b'"jobs"' not in response.content


def test_history_is_read_only(app, client):
    job_id = _submit(client)
    before = _history(client, **_scope()).json()
    with app.state.session_factory() as session:
        job = session.get(RewrapJob, job_id)
        snapshot = (job.status, job.processed, job.next_cursor, job.updated_at)
    assert _history(client, **_scope()).json() == before
    with app.state.session_factory() as session:
        job = session.get(RewrapJob, job_id)
        assert (job.status, job.processed, job.next_cursor, job.updated_at) == snapshot


def test_response_is_compact_json_with_single_newline_and_no_floats(client):
    _submit(client)
    raw = _history(client, **_scope()).content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw and b": " not in raw
    parsed = json.loads(raw)
    assert list(parsed.keys()) == ["jobs", "next_cursor", "complete"]
    assert list(parsed["jobs"][0].keys()) == JOB_FIELDS
    assert b"-0.0" not in raw
    for token in (b"NaN", b"Infinity"):
        assert token not in raw


def test_history_never_exposes_payload_or_key_material(app, client):
    assert (
        client.post(
            "/v1/data-envelopes",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "data_id": "secret-item",
                "payload": PAYLOAD,
            },
        ).status_code
        == 201
    )
    job_id = _submit(client)
    _run_job(app, job_id)
    raw = _history(client, **_scope()).text
    for secret in (PAYLOAD, KEY_V1, "wrapped_key", "ciphertext", "payload"):
        assert secret not in raw
