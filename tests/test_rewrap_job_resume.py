"""Tests for asynchronous rewrap job manual resume:

POST /v1/rewrap-jobs/{job_id}/resume

A failed job can be flipped back to ``queued`` exactly once per failure.
The winning resume commits the guarded failed -> queued transition in one
transaction, preserving the failed counter and the resume cursor parked
immediately before the failing envelope; the background runner then
re-attempts exactly that envelope first and converges to the current
progress. The resume itself creates no second job, rewrites no history
and appends no audit event. Non-failed jobs (queued, running, succeeded,
cancelled, or just-resumed) are stable 409s, unknown or cross-scope jobs
are indistinguishable 404s, and a storage failure is a 500 after a full
rollback.

Tests build the app without entering its lifespan (no pool) except the
explicit restart test; a worker is driven synchronously or on its own
thread where mid-envelope timing has to be deterministic.
"""

from __future__ import annotations

import json
import threading
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
#: Current v2 keyring with the historical v1 key absent: any v1 envelope
#: fails its rewrap attempt.
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


def _submit(client, *, tenant=TENANT, workload=WORKLOAD, limit=None):
    body: dict = {"tenant_id": tenant, "workload_id": workload}
    if limit is not None:
        body["limit"] = limit
    return client.post("/v1/rewrap-jobs", json=body)


def _resume(client, job_id, *, tenant=TENANT, workload=WORKLOAD, **extra):
    body = {"tenant_id": tenant, "workload_id": workload, **extra}
    return client.post(f"/v1/rewrap-jobs/{job_id}/resume", json=body)


def _resume_raw(client, path, body, **kwargs):
    return client.post(path, json=body, **kwargs)


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


def _audit_events(app):
    with app.state.session_factory() as session:
        return (
            session.query(AuditEvent)
            .filter(AuditEvent.event_type == "rewrap")
            .order_by(AuditEvent.data_id.asc())
            .all()
        )


def _fail_job(app, client, monkeypatch, *, tenant=TENANT, workload=WORKLOAD):
    """Submit a job whose first envelope cannot be unwrapped and run it.

    The seeded envelopes are wrapped under the historical v1 key; running
    with a current-v2 keyring that no longer holds v1 fails the job on
    the first envelope, parking the cursor at the submission boundary.
    The broken keyring is left in place; callers restore a working one.
    """
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    job_id = _submit(client, tenant=tenant, workload=workload).json()["job_id"]
    app.state.rewrap_job_runner.run(job_id)
    row = _job_row(app, job_id)
    assert row["status"] == "failed"
    assert row["failed"] == 1
    return job_id


def _fail_job_mid_scope(app, client, monkeypatch, data_ids):
    """Run a job that commits its first envelope then fails on the second.

    Returns the job id. The failure is induced by a patched rewrap
    primitive raising on its second invocation, so envelope ``data_ids[0]``
    commits (processed=1) and the cursor parks immediately before
    ``data_ids[1]``.
    """
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    runner = app.state.rewrap_job_runner
    real_rewrap = app_module.rewrap_data_key
    calls = {"n": 0}

    def failing_rewrap(unwrapping_key, wrapping_key, wrapped_key):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected rewrap failure")
        return real_rewrap(unwrapping_key, wrapping_key, wrapped_key)

    monkeypatch.setattr(app_module, "rewrap_data_key", failing_rewrap)
    runner.run(job_id)
    monkeypatch.setattr(app_module, "rewrap_data_key", real_rewrap)
    row = _job_row(app, job_id)
    assert row["status"] == "failed"
    assert row["processed"] == 1
    assert row["failed"] == 1
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
    response = _resume_raw(
        client, path, {"tenant_id": TENANT, "workload_id": WORKLOAD}
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


def test_resume_422_reads_no_job(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    # A malformed body must not change or read the job; it stays queued.
    response = client.post(
        f"/v1/rewrap-jobs/{job_id}/resume", json={"tenant_id": TENANT}
    )
    assert response.status_code == 422
    assert _job_row(app, job_id)["status"] == "queued"


# --- 404 indistinguishability ---------------------------------------------


def test_resume_unknown_or_cross_scope_job_returns_404(app, client, monkeypatch):
    _seed(client, ["a"])
    job_id = _fail_job(app, client, monkeypatch)

    unknown = "00000000-0000-0000-0000-000000000000"
    assert _resume(client, unknown).status_code == 404
    assert _resume(client, job_id, tenant=OTHER_TENANT).status_code == 404
    assert _resume(client, job_id, workload=OTHER_WORKLOAD).status_code == 404
    # None of the failing resumes touched the real job.
    assert _job_row(app, job_id)["status"] == "failed"


# --- non-failed conflicts ---------------------------------------------------


def test_resume_non_failed_jobs_return_409_and_change_nothing(
    app, client, monkeypatch
):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    # Queued job.
    queued = _submit(client).json()["job_id"]
    queued_row = _job_row(app, queued)

    # Running job (claimed by a runner, parked before its first page).
    running = _submit(client).json()["job_id"]
    runner = app.state.rewrap_job_runner
    assert runner._claim(running)
    running_row = _job_row(app, running)

    # Succeeded job.
    succeeded = _submit(client).json()["job_id"]
    runner.run(succeeded)
    assert _job_row(app, succeeded)["status"] == "succeeded"
    succeeded_row = _job_row(app, succeeded)

    # Cancelled job.
    cancelled = _submit(client).json()["job_id"]
    assert _cancel(client, cancelled).status_code == 200
    cancelled_row = _job_row(app, cancelled)

    assert _resume(client, queued).status_code == 409
    assert _resume(client, running).status_code == 409
    assert _resume(client, succeeded).status_code == 409
    assert _resume(client, cancelled).status_code == 409

    runner._release_claim(running)
    # updated_at moves only for the released claim's own write; every
    # other field of every job is exactly as before the 409s.
    assert _job_row(app, queued) == queued_row
    assert _job_row(app, succeeded) == succeeded_row
    assert _job_row(app, cancelled) == cancelled_row
    after_release = _job_row(app, running)
    assert after_release["status"] == "running"
    assert after_release["processed"] == running_row["processed"]


def test_resume_just_resumed_job_returns_409(app, client, monkeypatch):
    _seed(client, ["a"])
    job_id = _fail_job(app, client, monkeypatch)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _resume(client, job_id).status_code == 200
    # The job is queued now: a repeat resume is a stable 409.
    assert _resume(client, job_id).status_code == 409
    assert _resume(client, job_id).status_code == 409
    assert _job_row(app, job_id)["status"] == "queued"


# --- successful resume ------------------------------------------------------


def test_resume_failed_job_response_and_state(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    job_id = _fail_job(app, client, monkeypatch)
    failed_row = _job_row(app, job_id)
    assert failed_row["next_cursor"] == ""
    assert failed_row["processed"] == 0

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _resume(client, job_id)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    raw = response.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw and b": " not in raw
    # Same keys, order and types as the single-job query.
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
    # Counters and the parked cursor are preserved verbatim, as ints.
    assert data["processed"] == failed_row["processed"] == 0
    assert data["rewrapped"] == failed_row["rewrapped"] == 0
    assert data["skipped"] == failed_row["skipped"] == 0
    assert data["failed"] == failed_row["failed"] == 1
    assert isinstance(data["failed"], int)
    assert data["next_cursor"] == failed_row["next_cursor"] == ""
    assert data["complete"] is False
    created = datetime.fromisoformat(data["created_at"])
    updated = datetime.fromisoformat(data["updated_at"])
    assert created.utcoffset() == timedelta(0)
    assert updated.utcoffset() == timedelta(0)
    assert created == failed_row["created_at"]
    assert updated > failed_row["updated_at"]

    row = _job_row(app, job_id)
    assert row["status"] == "queued"
    assert row["claim_token"] is None
    assert row["failed"] == 1
    assert row["next_cursor"] == failed_row["next_cursor"]
    assert row["complete"] is False
    assert row["cancelled_at"] is None

    # The resume appended no audit event and the GET view agrees.
    assert _audit_events(app) == []
    progress = _get_job(client, job_id).json()
    assert progress == data


def test_resume_creates_no_second_job_and_rewrites_no_history(
    app, client, monkeypatch
):
    _seed(client, ["a"])
    job_id = _fail_job(app, client, monkeypatch)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    before = _history(client).json()
    assert [j["job_id"] for j in before["jobs"]] == [job_id]

    assert _resume(client, job_id).status_code == 200

    after = _history(client).json()
    assert [j["job_id"] for j in after["jobs"]] == [job_id]
    entry = after["jobs"][0]
    assert entry["status"] == "queued"
    assert entry["failed"] == 1
    # History filtering sees the resumed job under its new status only.
    assert _history(client, status="failed").json()["jobs"] == []
    assert [j["job_id"] for j in _history(client, status="queued").json()["jobs"]] == [
        job_id
    ]


def test_resumed_job_converges_from_parked_cursor(app, client, monkeypatch):
    data_ids = ["a", "b", "c"]
    _seed(client, data_ids)
    job_id = _fail_job_mid_scope(app, client, monkeypatch, data_ids)
    failed_row = _job_row(app, job_id)
    parked_cursor = failed_row["next_cursor"]
    assert parked_cursor != ""
    # Envelope a committed before the failure; b and c are untouched.
    assert _get_envelope(client, "a").json()["key_version"] == 2
    for d in ("b", "c"):
        assert _get_envelope(client, d).json()["key_version"] == 1
    audits_before = [(e.data_id, e.status) for e in _audit_events(app)]
    assert audits_before == [("a", "rewrapped")]

    response = _resume(client, job_id)
    assert response.status_code == 200
    data = response.json()
    # The failed counter and the cursor parked before envelope b survive.
    assert data["failed"] == 1
    assert data["processed"] == 1
    assert data["next_cursor"] == parked_cursor

    # The background runner resumes from the parked boundary: b is
    # re-attempted first, a is never re-processed, and the job converges.
    app.state.rewrap_job_runner.run(job_id)
    row = _job_row(app, job_id)
    assert row["status"] == "succeeded"
    assert row["processed"] == 3
    assert row["rewrapped"] == 3
    assert row["skipped"] == 0
    assert row["failed"] == 1
    assert row["complete"] is True
    for d in data_ids:
        envelope = _get_envelope(client, d).json()
        assert envelope["key_version"] == 2
        assert _unwrap_and_decrypt(KEY_V2_BYTES, envelope) == PAYLOAD.encode("utf-8")
    # Exactly one audit event per envelope; the resume added none and no
    # envelope was advanced twice.
    assert [(e.data_id, e.status) for e in _audit_events(app)] == [
        ("a", "rewrapped"),
        ("b", "rewrapped"),
        ("c", "rewrapped"),
    ]


def test_resumed_job_can_be_cancelled_before_it_runs(app, client, monkeypatch):
    _seed(client, ["a"])
    job_id = _fail_job(app, client, monkeypatch)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _resume(client, job_id).status_code == 200
    response = _cancel(client, job_id)
    assert response.status_code == 200
    row = _job_row(app, job_id)
    assert row["status"] == "cancelled"
    assert row["failed"] == 1
    # A cancelled job is terminal for resume as well.
    assert _resume(client, job_id).status_code == 409


# --- concurrency ------------------------------------------------------------


def test_concurrent_resumes_have_one_winner(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    job_id = _fail_job(app, client, monkeypatch)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: _resume(client, job_id), range(8)))
    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(200) == 1
    assert statuses.count(409) == 7
    row = _job_row(app, job_id)
    assert row["status"] == "queued"
    assert row["failed"] == 1
    # A later resume is still a stable 409.
    assert _resume(client, job_id).status_code == 409
    # The single winner queued the job exactly once for the runner.
    app.state.rewrap_job_runner.run(job_id)
    assert _job_row(app, job_id)["status"] == "succeeded"


def test_resume_racing_startup_sweep_settles_once(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    job_id = _fail_job(app, client, monkeypatch)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    runner = app.state.rewrap_job_runner
    results: dict = {}

    def do_resume():
        results["response"] = _resume(client, job_id)

    resumer = threading.Thread(target=do_resume, daemon=True)
    resumer.start()
    # A recovery sweep may claim the failed job (failed -> running) while
    # the resume is in flight; exactly one terminal observation is legal.
    runner.run(job_id, retry=True)
    resumer.join(timeout=5)
    assert not resumer.is_alive()
    assert results["response"].status_code in (200, 409)
    row = _job_row(app, job_id)
    # Either the resume won and the sweep then ran the queued job, or the
    # sweep claimed first and the resume conflicted; both converge.
    assert row["status"] == "succeeded"
    assert row["failed"] == 1
    assert row["processed"] == 2
    assert _resume(client, job_id).status_code == 409


# --- restart persistence ----------------------------------------------------


def test_resumed_job_survives_restart_and_recovery_completes_it(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/resume-restart.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["a", "b"])
    job_id = _fail_job(app1, client1, monkeypatch)
    failed_row = _job_row(app1, job_id)

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _resume(client1, job_id)
    assert response.status_code == 200
    resumed = response.json()
    app1.state.engine.dispose()

    # Restart with the live pool + recovery sweep running: the queued row
    # left by the resume is claimed under the same status guard and the
    # job converges with its failed counter and history intact.
    app2 = app_module.create_app(url)
    with TestClient(app2) as client2:
        deadline = time.time() + 5
        while time.time() < deadline:
            if _job_row(app2, job_id)["status"] == "succeeded":
                break
            time.sleep(0.05)
        row = _job_row(app2, job_id)
        assert row["status"] == "succeeded"
        assert row["processed"] == 2
        assert row["rewrapped"] == 2
        assert row["failed"] == 1
        assert row["complete"] is True
        assert row["created_at"] == failed_row["created_at"]
        for d in ("a", "b"):
            assert _get_envelope(client2, d).json()["key_version"] == 2
        # The resume itself added no audit event across the restart.
        assert [(e.data_id, e.status) for e in _audit_events(app2)] == [
            ("a", "rewrapped"),
            ("b", "rewrapped"),
        ]
        # The resumed state observed before the restart matches the row.
        assert resumed["failed"] == 1
        assert resumed["next_cursor"] == failed_row["next_cursor"]
    app2.state.engine.dispose()


def test_failed_job_stays_failed_across_restart_until_resumed(
    tmp_path, monkeypatch
):
    # Without a manual resume the recovery sweep re-attempts a failed job
    # and, with a still-broken keyring, fails it again on the same
    # envelope: the job stays failed with its cursor parked and no
    # envelope processed, and remains resumable. This pins that the
    # restart never loses the saved progress the resume entry point
    # depends on.
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/resume-park.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["a"])
    job_id = _fail_job(app1, client1, monkeypatch)
    failed_row = _job_row(app1, job_id)
    app1.state.engine.dispose()

    # Restart with the keyring still missing v1: the sweep re-attempts and
    # fails again on the same envelope; the cursor stays parked and no
    # envelope is processed.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    app2 = app_module.create_app(url)
    with TestClient(app2):
        deadline = time.time() + 5
        while time.time() < deadline:
            row = _job_row(app2, job_id)
            if row["status"] == "failed" and row["claim_token"] is None:
                break
            time.sleep(0.05)
        row = _job_row(app2, job_id)
        assert row["status"] == "failed"
        assert row["processed"] == failed_row["processed"]
        assert row["next_cursor"] == failed_row["next_cursor"]
        assert row["complete"] is False
    app2.state.engine.dispose()


# --- server failure ---------------------------------------------------------


def test_resume_storage_failure_is_500_and_changes_nothing(app, client, monkeypatch):
    from sqlalchemy import text

    _seed(client, ["a"])
    job_id = _fail_job(app, client, monkeypatch)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_jobs"))
    response = _resume(client, job_id)
    assert response.status_code == 500
    assert _get_envelope(client, "a").status_code == 200


def test_resume_write_failure_rolls_back_and_preserves_failed_state(
    app, client, monkeypatch
):
    from sqlalchemy import text

    _seed(client, ["a"])
    job_id = _fail_job(app, client, monkeypatch)
    failed_row = _job_row(app, job_id)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    # Abort every UPDATE on the job row: the guarded transition fails
    # mid-transaction, the handler rolls back and the job keeps its
    # failed state, counters, cursor and timestamps.
    with app.state.engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TRIGGER fail_rewrap_job_update BEFORE UPDATE "
                "ON rewrap_jobs BEGIN SELECT RAISE(ABORT, 'injected'); END"
            )
        )
    try:
        response = _resume(client, job_id)
        assert response.status_code == 500
    finally:
        with app.state.engine.begin() as conn:
            conn.execute(text("DROP TRIGGER fail_rewrap_job_update"))
    assert _job_row(app, job_id) == failed_row
    # Once the storage fault clears the same resume succeeds.
    assert _resume(client, job_id).status_code == 200
    assert _job_row(app, job_id)["status"] == "queued"


# --- idempotency record stays intact ---------------------------------------


def test_idempotent_submission_replay_still_returns_original_202_after_resume(
    app, client, monkeypatch
):
    _seed(client, ["a"])
    headers = {"idempotency-key": "key-1"}
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    first = client.post(
        "/v1/rewrap-jobs",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers=headers,
    )
    assert first.status_code == 202
    job_id = first.json()["job_id"]
    app.state.rewrap_job_runner.run(job_id)
    assert _job_row(app, job_id)["status"] == "failed"

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _resume(client, job_id).status_code == 200

    # A replay of the original submission returns the stored 202 verbatim
    # and never creates a second job for the resumed work.
    replay = client.post(
        "/v1/rewrap-jobs",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers=headers,
    )
    assert replay.status_code == 202
    assert replay.content == first.content
    assert [j["job_id"] for j in _history(client).json()["jobs"]] == [job_id]

    # The resumed job still converges under the original job id.
    app.state.rewrap_job_runner.run(job_id)
    assert _job_row(app, job_id)["status"] == "succeeded"
