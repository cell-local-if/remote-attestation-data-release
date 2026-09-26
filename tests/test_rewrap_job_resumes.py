"""Tests for asynchronous rewrap job manual resume:

POST /v1/rewrap-jobs/{job_id}/resume

A failed job (parked with its cursor immediately before the failing
envelope) can be manually resumed exactly once. The winning resume
commits status=queued in one guarded transaction, preserving the failed
counter and the parked cursor, creating no second job and appending no
audit; the background runner then re-attempts the failing envelope first
and converges to the terminal state under the existing advancement
rules. Every non-failed state (queued, running, succeeded, cancelled or
just-resumed) is a stable 409, unknown or cross-scope jobs an
indistinguishable 404, and concurrent resumes settle with one winner.

Tests build the app without entering its lifespan (no pool) except the
explicit restart test; a worker is driven synchronously through the
runner where deterministic page/failure timing is needed.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import AuditEvent, RewrapJob
from proof_release.envelopes import b64url_decode, b64url_encode

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
PAYLOAD = "resume-secret-payload 🔐"

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
    application = app_module.create_app(f"sqlite:///{tmp_path}/resume.db")
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


def _submit(client, *, tenant=TENANT, workload=WORKLOAD, limit=None, headers=None):
    body: dict = {"tenant_id": tenant, "workload_id": workload}
    if limit is not None:
        body["limit"] = limit
    return client.post("/v1/rewrap-jobs", json=body, headers=headers)


def _resume(client, job_id, *, tenant=TENANT, workload=WORKLOAD, **extra):
    body = {"tenant_id": tenant, "workload_id": workload, **extra}
    return client.post(f"/v1/rewrap-jobs/{job_id}/resume", json=body)


def _cancel(client, job_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        f"/v1/rewrap-jobs/{job_id}/cancel",
        json={"tenant_id": tenant, "workload_id": workload},
    )


def _get_job(client, job_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _get_envelope(client, data_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/data-envelopes/{data_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _history(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        "/v1/rewrap-jobs",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


def _unwrap_and_decrypt(master_key: bytes, fields: dict) -> bytes:
    data_key = aes_key_unwrap(master_key, b64url_decode(fields["wrapped_key"]))
    iv = b64url_decode(fields["iv"])
    sealed = b64url_decode(fields["ciphertext"]) + b64url_decode(fields["tag"])
    return AESGCM(data_key).decrypt(iv, sealed, None)


def _job_row(app, job_id):
    with app.state.session_factory() as session:
        row = session.get(RewrapJob, job_id)
        return (
            None
            if row is None
            else {
                "status": row.status,
                "processed": row.processed,
                "rewrapped": row.rewrapped,
                "skipped": row.skipped,
                "failed": row.failed,
                "next_cursor": row.next_cursor,
                "complete": bool(row.complete),
                "cancelled_at": row.cancelled_at,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "claim_token": row.claim_token,
            }
        )


def _rewrap_audits(app):
    with app.state.session_factory() as session:
        events = (
            session.query(AuditEvent)
            .filter(AuditEvent.event_type == "rewrap")
            .order_by(AuditEvent.data_id.asc())
            .all()
        )
        return [(e.data_id, e.status) for e in events]


def _fail_job(app, client, monkeypatch, *, tenant=TENANT, workload=WORKLOAD,
              limit=None):
    """Submit a job whose scope holds historical-v1 envelopes and fail it.

    The current keyring is v2-only (the historical v1 key is absent), so
    the first envelope the runner touches cannot be unwrapped and the job
    settles ``failed`` with its cursor parked before that envelope.
    """
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    job_id = _submit(client, tenant=tenant, workload=workload, limit=limit).json()[
        "job_id"
    ]
    app.state.rewrap_job_runner.run(job_id)
    assert _job_row(app, job_id)["status"] == "failed"
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    return job_id


# --- request validation ----------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/v1/rewrap-jobs/not-a-uuid/resume",
        "/v1/rewrap-jobs/abc123/resume",
        "/v1/rewrap-jobs/00000000-0000-0000-0000-00000000000Z/resume",
        # Uppercase spellings are not canonical; resume never normalizes.
        "/v1/rewrap-jobs/AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA/resume",
        # Surrounding whitespace is part of the path segment and invalid.
        "/v1/rewrap-jobs/%200000000-0000-0000-0000-000000000000/resume",
    ],
)
def test_resume_rejects_malformed_job_identifier(client, path):
    response = client.post(
        path, json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert response.status_code == 422, response.text


def test_resume_without_identifier_returns_422(client):
    response = client.post(
        "/v1/rewrap-jobs//resume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"workload_id": WORKLOAD},
        {"tenant_id": TENANT},
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "   ", "workload_id": WORKLOAD},
        {"tenant_id": 7, "workload_id": WORKLOAD},
        {"tenant_id": True, "workload_id": WORKLOAD},
        {"tenant_id": None, "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": "\t"},
        {"tenant_id": TENANT, "workload_id": 3},
        {"tenant_id": TENANT, "workload_id": None},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "extra": 1},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "status": "queued"},
    ],
)
def test_resume_rejects_invalid_bodies(client, body):
    job_id = "00000000-0000-0000-0000-000000000000"
    response = client.post(f"/v1/rewrap-jobs/{job_id}/resume", json=body)
    assert response.status_code == 422, response.text


def test_resume_rejects_non_json_body(client):
    job_id = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        f"/v1/rewrap-jobs/{job_id}/resume",
        content="not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_resume_422_reads_and_changes_no_job(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _fail_job(app, client, monkeypatch)
    before = _job_row(app, job_id)
    # A malformed body must not change or read the job; it stays failed.
    response = client.post(
        f"/v1/rewrap-jobs/{job_id}/resume", json={"tenant_id": TENANT}
    )
    assert response.status_code == 422
    assert _job_row(app, job_id) == before


# --- 404 indistinguishability ---------------------------------------------


def test_resume_unknown_or_cross_scope_job_returns_404(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _fail_job(app, client, monkeypatch)
    before = _job_row(app, job_id)

    unknown = "00000000-0000-0000-0000-000000000000"
    assert _resume(client, unknown).status_code == 404
    assert _resume(client, job_id, tenant=OTHER_TENANT).status_code == 404
    assert _resume(client, job_id, workload=OTHER_WORKLOAD).status_code == 404
    # None of the failing resumes touched the real job.
    assert _job_row(app, job_id) == before


# --- successful resume -----------------------------------------------------


def test_resume_failed_job_is_atomic_compact_and_appends_no_audit(
    app, client, monkeypatch
):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _fail_job(app, client, monkeypatch)
    failed_row = _job_row(app, job_id)
    assert failed_row["failed"] == 1
    assert failed_row["processed"] == 0
    assert failed_row["next_cursor"] == ""
    assert _rewrap_audits(app) == []

    response = _resume(client, job_id)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    raw = response.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw and b": " not in raw
    # The single-job progress query's exact field order is reused.
    assert list(json.loads(raw)) == [
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
    data = json.loads(raw)
    assert data["job_id"] == job_id
    assert data["status"] == "queued"
    # Counters keep their integer types and their pre-resume values.
    assert data["processed"] == 0 and isinstance(data["processed"], int)
    assert data["rewrapped"] == 0 and isinstance(data["rewrapped"], int)
    assert data["skipped"] == 0 and isinstance(data["skipped"], int)
    # The failed count and the parked cursor are preserved verbatim.
    assert data["failed"] == 1 and isinstance(data["failed"], int)
    assert data["next_cursor"] == ""
    assert data["complete"] is False
    assert datetime.fromisoformat(data["created_at"]) == failed_row["created_at"]
    updated = datetime.fromisoformat(data["updated_at"])
    assert updated.utcoffset() == timedelta(0)
    assert updated >= failed_row["updated_at"]

    row = _job_row(app, job_id)
    assert row["status"] == "queued"
    assert row["claim_token"] is None
    assert row["failed"] == 1
    assert row["processed"] == row["rewrapped"] == row["skipped"] == 0
    assert row["next_cursor"] == ""
    assert row["complete"] is False
    assert row["cancelled_at"] is None
    assert row["created_at"] == failed_row["created_at"]

    # The resume itself appended no compliance event and created no
    # second job; the envelope is untouched until the runner advances.
    assert _rewrap_audits(app) == []
    assert _get_envelope(client, "a").json()["key_version"] == 1
    history = _history(client).json()
    assert [j["job_id"] for j in history["jobs"]] == [job_id]
    assert history["jobs"][0]["status"] == "queued"

    # The single-job query observes the same resumed state.
    progress = _get_job(client, job_id).json()
    assert progress["status"] == "queued"
    assert progress["failed"] == 1
    assert progress["updated_at"] == data["updated_at"]


def test_resumed_job_recovers_from_parked_cursor_and_converges(
    app, client, monkeypatch
):
    _seed(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client, limit=1).json()["job_id"]
    runner = app.state.rewrap_job_runner

    # Commit envelope a, then park the job (claim released, still running).
    assert runner._claim(job_id)
    assert runner._advance_page(job_id) is None
    runner._release_claim(job_id)
    mid = _job_row(app, job_id)
    assert mid["status"] == "running" and mid["processed"] == 1

    # The historical v1 key disappears: the job fails on envelope b with
    # its cursor parked immediately before it (after a).
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    runner.run(job_id)
    failed_row = _job_row(app, job_id)
    assert failed_row["status"] == "failed"
    assert failed_row["failed"] == 1
    assert failed_row["processed"] == 1
    assert failed_row["rewrapped"] == 1
    parked_cursor = failed_row["next_cursor"]
    assert parked_cursor == mid["next_cursor"]
    assert parked_cursor != ""
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _rewrap_audits(app) == [("a", "rewrapped")]

    response = _resume(client, job_id)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert data["processed"] == 1
    assert data["rewrapped"] == 1
    assert data["failed"] == 1
    assert data["next_cursor"] == parked_cursor
    # The resume appended nothing to the audit trail.
    assert _rewrap_audits(app) == [("a", "rewrapped")]

    # The background resumes from the parked boundary: b is re-attempted
    # first, a is never re-advanced, and the job converges to succeeded.
    runner.run(job_id)
    final = _job_row(app, job_id)
    assert final["status"] == "succeeded"
    assert final["complete"] is True
    assert final["processed"] == 3
    assert final["rewrapped"] == 3
    assert final["failed"] == 1
    assert final["next_cursor"] == ""
    # Exactly one audit event per envelope: no duplicate for the
    # pre-failure envelope a, none emitted by the resume itself.
    assert _rewrap_audits(app) == [
        ("a", "rewrapped"),
        ("b", "rewrapped"),
        ("c", "rewrapped"),
    ]
    for d in ("a", "b", "c"):
        envelope = _get_envelope(client, d).json()
        assert envelope["key_version"] == 2
        assert _unwrap_and_decrypt(KEY_V2_BYTES, envelope) == PAYLOAD.encode("utf-8")


# --- non-failed conflicts --------------------------------------------------


def test_resume_non_failed_jobs_returns_409_and_rewrites_nothing(
    app, client, monkeypatch
):
    # Seed both scopes while the fixture keyring is v1-only, so every
    # envelope is sealed under the historical v1 key.
    _seed(client, ["a", "b", "c", "d", "e"])
    _seed(client, ["f"], tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    runner = app.state.rewrap_job_runner

    # Queued.
    queued = _submit(client).json()["job_id"]
    queued_row = _job_row(app, queued)

    # Running (claimed by this thread, parked before its first page).
    running = _submit(client).json()["job_id"]
    assert runner._claim(running)
    running_row = _job_row(app, running)
    assert running_row["status"] == "running"

    # Succeeded.
    succeeded = _submit(client).json()["job_id"]
    runner.run(succeeded)
    succeeded_row = _job_row(app, succeeded)
    assert succeeded_row["status"] == "succeeded"

    # Cancelled.
    cancelled = _submit(client).json()["job_id"]
    assert _cancel(client, cancelled).status_code == 200
    cancelled_row = _job_row(app, cancelled)

    assert _resume(client, queued).status_code == 409
    assert _resume(client, running).status_code == 409
    assert _resume(client, succeeded).status_code == 409
    assert _resume(client, cancelled).status_code == 409
    # Stable on repetition.
    assert _resume(client, queued).status_code == 409
    assert _resume(client, cancelled).status_code == 409

    assert _job_row(app, queued) == queued_row
    assert _job_row(app, running) == running_row
    assert _job_row(app, succeeded) == succeeded_row
    assert _job_row(app, cancelled) == cancelled_row
    runner._release_claim(running)

    # A just-resumed job is equally a 409 for any further resume. Use the
    # other scope, whose envelope is still on the historical v1 key (the
    # succeeded job above rotated every envelope in the main scope).
    failed = _fail_job(
        app, client, monkeypatch, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    )
    assert (
        _resume(client, failed, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).status_code
        == 200
    )
    resumed_row = _job_row(app, failed)
    assert (
        _resume(client, failed, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).status_code
        == 409
    )
    assert _job_row(app, failed) == resumed_row


# --- concurrency -----------------------------------------------------------


def test_concurrent_resumes_have_exactly_one_winner(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _fail_job(app, client, monkeypatch)
    failed_row = _job_row(app, job_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: _resume(client, job_id), range(8)))
    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(200) == 1
    assert statuses.count(409) == 7

    row = _job_row(app, job_id)
    assert row["status"] == "queued"
    assert row["claim_token"] is None
    # The loser 409s changed nothing; the winner changed only the status
    # and updated_at.
    assert row["failed"] == failed_row["failed"]
    assert row["processed"] == failed_row["processed"]
    assert row["next_cursor"] == failed_row["next_cursor"]
    assert row["created_at"] == failed_row["created_at"]
    # A later resume is still a stable 409.
    assert _resume(client, job_id).status_code == 409
    # No audit was appended by any of the resumes.
    assert _rewrap_audits(app) == []


def test_resume_racing_startup_sweep_claim_settles_exactly_once(
    app, client, monkeypatch
):
    # The startup sweep's retry claim (failed -> running) and a manual
    # resume (failed -> queued) are guarded by the same status predicate:
    # whichever commits first wins and the other observes a non-failed
    # job. Drive the sweep's claim directly to make the race
    # deterministic.
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _fail_job(app, client, monkeypatch)
    runner = app.state.rewrap_job_runner

    # The sweep wins: the job is claimed failed -> running, so the manual
    # resume is a 409 that changes nothing.
    assert runner._claim(job_id, retry=True)
    claimed_row = _job_row(app, job_id)
    assert claimed_row["status"] == "running"
    assert _resume(client, job_id).status_code == 409
    assert _job_row(app, job_id) == claimed_row
    runner._release_claim(job_id)

    # Back to failed for the mirror case: the resume wins, so a late
    # sweep claim cannot flip the row again (it is queued, and a retry
    # claim on the same runner would need the claim token free — the
    # status guard alone never resurrects a second transition).
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    runner.run(job_id, retry=True)
    assert _job_row(app, job_id)["status"] == "failed"
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _resume(client, job_id).status_code == 200
    assert _job_row(app, job_id)["status"] == "queued"


# --- restart persistence ---------------------------------------------------


def test_resumed_job_survives_restart_and_sweep_advances_it(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/resume-restart.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _fail_job(app1, client1, monkeypatch)
    failed_row = _job_row(app1, job_id)

    response = _resume(client1, job_id)
    assert response.status_code == 200
    resumed = response.json()
    app1.state.engine.dispose()

    # Restart with the live pool and recovery sweep running: the queued
    # row left by the resume is picked up and advanced to succeeded,
    # preserving the failed counter and never re-processing committed
    # envelopes.
    app2 = app_module.create_app(url)
    with TestClient(app2) as client2:
        deadline = time.time() + 5
        while time.time() < deadline:
            if _job_row(app2, job_id)["status"] == "succeeded":
                break
            time.sleep(0.05)
        row = _job_row(app2, job_id)
        assert row["status"] == "succeeded"
        assert row["failed"] == failed_row["failed"] == 1
        assert row["processed"] == 2
        assert row["rewrapped"] == 2
        assert row["created_at"] == failed_row["created_at"]
        assert datetime.fromisoformat(resumed["updated_at"]) <= row["updated_at"]
        for d in ("a", "b"):
            assert _get_envelope(client2, d).json()["key_version"] == 2
        assert _rewrap_audits(app2) == [("a", "rewrapped"), ("b", "rewrapped")]
        # The resumed job still answers the single-job query after the
        # restart, with the same identifier and terminal progress.
        progress = _get_job(client2, job_id).json()
        assert progress["status"] == "succeeded"
        assert progress["failed"] == 1
    app2.state.engine.dispose()


# --- server failure --------------------------------------------------------


def test_resume_storage_failure_is_500_and_changes_nothing(app, client, monkeypatch):
    from sqlalchemy import text

    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _fail_job(app, client, monkeypatch)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_jobs"))
    response = _resume(client, job_id)
    assert response.status_code == 500
    assert _get_envelope(client, "a").status_code == 200


# --- idempotency record stays intact --------------------------------------


def test_idempotent_submission_replay_still_returns_original_202_after_resume(
    app, client, monkeypatch
):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    headers = {"idempotency-key": "key-1"}
    first = _submit(client, headers=headers)
    assert first.status_code == 202
    job_id = first.json()["job_id"]

    # Fail the job, resume it, then replay the keyed submission.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    app.state.rewrap_job_runner.run(job_id)
    assert _job_row(app, job_id)["status"] == "failed"
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _resume(client, job_id).status_code == 200

    replay = _submit(client, headers=headers)
    assert replay.status_code == 202
    assert replay.content == first.content
    # The replay created no second job; the resumed original is the only
    # one and it still owns the idempotency record.
    history = _history(client).json()
    assert [j["job_id"] for j in history["jobs"]] == [job_id]
