"""Tests for asynchronous rewrap job cancellation:

POST /v1/rewrap-jobs/{job_id}/cancel

A queued or running job can be cancelled exactly once. The winning
cancel commits status=cancelled and cancelled_at in one guarded
transaction; a running worker finishes the envelope already committed at
the cancel instant and discards any envelope still uncommitted (material,
wrapping material and audit all unchanged), then stops. Cancellation
survives restarts, is never re-queued by recovery, is a stable filter in
the read-only history, and settles concurrent cancels with exactly one
winner.

Tests build the app without entering its lifespan (no pool) except the
explicit restart test; a worker is driven on its own thread where
mid-envelope timing has to be deterministic, gated through a patched
rewrap primitive.
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
PAYLOAD = "cancel-secret-payload 🔐"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V2_BYTES = b"fedcba9876543210fedcba9876543210"
KEY_V1 = b64url_encode(KEY_V1_BYTES)
KEY_V2 = b64url_encode(KEY_V2_BYTES)


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


KEYRING_V1_V2 = _keyring(2, {1: KEY_V1, 2: KEY_V2})


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = app_module.create_app(f"sqlite:///{tmp_path}/cancel.db")
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


def _cancel(client, job_id, *, tenant=TENANT, workload=WORKLOAD, **extra):
    body = {"tenant_id": tenant, "workload_id": workload, **extra}
    return client.post(f"/v1/rewrap-jobs/{job_id}/cancel", json=body)


def _cancel_raw(client, path, body, **kwargs):
    return client.post(path, json=body, **kwargs)


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
                "updated_at": row.updated_at,
                "claim_token": row.claim_token,
            }
        )


def _raw_job_status(app, job_id):
    """Read committed job status without opening a write transaction.

    Used while a worker thread holds SQLite's writer lock mid-envelope:
    the ORM session factory begins every transaction IMMEDIATE and would
    block, but a plain deferred read observes the last committed state.
    """
    import sqlite3

    db_path = app.state.engine.url.database
    with sqlite3.connect(db_path, timeout=1) as ro:
        return ro.execute(
            "SELECT status, processed FROM rewrap_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()


# --- request validation ----------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/v1/rewrap-jobs/not-a-uuid/cancel",
        "/v1/rewrap-jobs/abc123/cancel",
        "/v1/rewrap-jobs/00000000-0000-0000-0000-00000000000Z/cancel",
        # Uppercase spellings are not canonical; cancel never normalizes.
        "/v1/rewrap-jobs/AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA/cancel",
        # Surrounding whitespace is part of the path segment and invalid.
        "/v1/rewrap-jobs/%200000000-0000-0000-0000-000000000000/cancel",
    ],
)
def test_cancel_rejects_malformed_job_identifier(client, path):
    response = _cancel_raw(
        client, path, {"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert response.status_code == 422, response.text


def test_cancel_without_identifier_returns_422(client):
    response = client.post(
        "/v1/rewrap-jobs//cancel",
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
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "status": "cancelled"},
    ],
)
def test_cancel_rejects_invalid_bodies(client, body):
    job_id = "00000000-0000-0000-0000-000000000000"
    response = client.post(f"/v1/rewrap-jobs/{job_id}/cancel", json=body)
    assert response.status_code == 422, response.text


def test_cancel_rejects_non_json_body(client):
    job_id = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        f"/v1/rewrap-jobs/{job_id}/cancel",
        content="not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_cancel_422_reads_no_job(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    # A malformed body must not change or read the job; it stays queued.
    response = client.post(
        f"/v1/rewrap-jobs/{job_id}/cancel", json={"tenant_id": TENANT}
    )
    assert response.status_code == 422
    assert _job_row(app, job_id)["status"] == "queued"


# --- 404 indistinguishability ---------------------------------------------


def test_cancel_unknown_or_cross_scope_job_returns_404(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]

    unknown = "00000000-0000-0000-0000-000000000000"
    assert _cancel(client, unknown).status_code == 404
    assert _cancel(client, job_id, tenant=OTHER_TENANT).status_code == 404
    assert _cancel(client, job_id, workload=OTHER_WORKLOAD).status_code == 404
    # None of the failing cancels touched the real job.
    assert _job_row(app, job_id)["status"] == "queued"


def test_cross_scope_cancel_never_signals_in_scope_runner(app, client, monkeypatch):
    _seed(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client, limit=1).json()["job_id"]
    runner = app.state.rewrap_job_runner
    # Simulate a live claim held for the real scope (as a worker has it).
    assert runner._claim(job_id)
    try:
        runner.request_cancel(job_id, (OTHER_TENANT, WORKLOAD))
        # No in-process cancel state may be created for a wrong scope.
        assert job_id not in runner._cancel_events
        runner.request_cancel(job_id, (TENANT, OTHER_WORKLOAD))
        assert job_id not in runner._cancel_events
    finally:
        runner._release_claim(job_id)


# --- successful queued cancellation ---------------------------------------


def test_cancel_queued_job_is_atomic_and_compact(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    accepted = _submit(client)
    job_id = accepted.json()["job_id"]

    response = _cancel(client, job_id)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    raw = response.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw and b": " not in raw
    # Keys appear in lexicographic (string) order.
    assert list(json.loads(raw)) == ["cancelled_at", "job_id", "status"]
    data = json.loads(raw)
    assert data["job_id"] == job_id
    assert data["status"] == "cancelled"
    parsed = datetime.fromisoformat(data["cancelled_at"])
    assert parsed.utcoffset() == timedelta(0)

    row = _job_row(app, job_id)
    assert row["status"] == "cancelled"
    assert row["cancelled_at"] == parsed
    assert row["complete"] is False
    assert row["claim_token"] is None
    assert row["processed"] == row["rewrapped"] == 0
    # The envelope is untouched by a queued cancellation.
    assert _get_envelope(client, "a").json()["key_version"] == 1
    with app.state.session_factory() as session:
        assert session.query(AuditEvent).count() == 0

    progress = _get_job(client, job_id).json()
    assert progress["status"] == "cancelled"
    assert "cancelled_at" not in progress
    assert progress["complete"] is False


def test_cancel_running_job_before_any_envelope_commits(monkeypatch, tmp_path):
    # A worker that has claimed the job (queued -> running) but is parked
    # before scanning its first page holds no envelope transaction; a
    # cancel settles cleanly and no envelope ever moves.
    import sqlite3

    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    db_path = tmp_path / "live-running.db"
    application = app_module.create_app(f"sqlite:///{db_path}")
    claimed = threading.Event()
    release = threading.Event()
    application.state.rewrap_job_runner.post_claim = lambda _: (
        claimed.set(), release.wait(timeout=5)
    )
    with TestClient(application) as client:
        _seed(client, ["a", "b"])
        monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
        job_id = _submit(client).json()["job_id"]
        assert claimed.wait(timeout=5)
        # Read via a separate deferred sqlite connection: the live worker
        # holds no writer lock while parked in post_claim.
        with sqlite3.connect(db_path) as ro:
            status = ro.execute(
                "SELECT status FROM rewrap_jobs WHERE job_id=?", (job_id,)
            ).fetchone()[0]
        assert status == "running"

        assert _cancel(client, job_id).status_code == 200
        release.set()
        time.sleep(0.1)
        row = _job_row(application, job_id)
        assert row["status"] == "cancelled"
        assert row["processed"] == 0
        for d in ("a", "b"):
            assert _get_envelope(client, d).json()["key_version"] == 1
    application.state.engine.dispose()


# --- terminal conflicts ----------------------------------------------------


def test_cancel_succeeded_failed_or_cancelled_returns_409_and_rewrites_nothing(
    app, client, monkeypatch
):
    # Seed both scopes while the fixture keyring is v1-only, so the
    # envelopes are sealed under the historical v1 key.
    _seed(client, ["a"])
    _seed(client, ["f"], tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    # Succeeded and cancelled jobs in the main scope.
    succeeded = _submit(client).json()["job_id"]
    app.state.rewrap_job_runner.run(succeeded)
    assert _get_job(client, succeeded).json()["status"] == "succeeded"
    done = _job_row(app, succeeded)

    cancelled = _submit(client).json()["job_id"]
    first_cancel = _cancel(client, cancelled)
    assert first_cancel.status_code == 200
    cancelled_row = _job_row(app, cancelled)

    # A failed job: v2-only current keyring with the historical v1 key
    # absent, in a separate scope whose envelope is still v1.
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(2, {2: KEY_V2})
    )
    failed = _submit(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    ).json()["job_id"]
    app.state.rewrap_job_runner.run(failed)
    assert (
        _get_job(client, failed, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
        .json()["status"]
        == "failed"
    )
    failed_row = _job_row(app, failed)
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    assert _cancel(client, succeeded).status_code == 409
    assert (
        _cancel(client, failed, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).status_code
        == 409
    )
    assert _cancel(client, cancelled).status_code == 409
    # A repeat cancel keeps returning 409 forever.
    assert _cancel(client, cancelled).status_code == 409

    assert _job_row(app, succeeded) == done
    assert _job_row(app, failed) == failed_row
    assert _job_row(app, cancelled) == cancelled_row


def test_repeat_cancel_does_not_rewrite_cancelled_at(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    first = _cancel(client, job_id)
    stamp = first.json()["cancelled_at"]
    time.sleep(0.01)
    assert _cancel(client, job_id).status_code == 409
    assert _job_row(app, job_id)["cancelled_at"] == datetime.fromisoformat(stamp)


# --- running cancellation: committed vs uncommitted envelope --------------


def test_cancel_running_job_keeps_committed_envelope_discards_uncommitted(
    app, client, monkeypatch
):
    data_ids = ["a", "b", "c"]
    _seed(client, data_ids)
    before = {d: _get_envelope(client, d).json() for d in data_ids}
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    job_id = _submit(client, limit=1).json()["job_id"]
    runner = app.state.rewrap_job_runner

    entered = threading.Event()
    gate = threading.Event()
    real_rewrap = app_module.rewrap_data_key
    calls = {"n": 0}

    def gated_rewrap(unwrapping_key, wrapping_key, wrapped_key):
        calls["n"] += 1
        if calls["n"] == 2:
            # Envelope "b": signal we are inside its uncommitted
            # transaction and park until the cancel is waiting.
            entered.set()
            gate.wait(timeout=5)
        return real_rewrap(unwrapping_key, wrapping_key, wrapped_key)

    monkeypatch.setattr(app_module, "rewrap_data_key", gated_rewrap)
    worker = threading.Thread(target=runner.run, args=(job_id,), daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        # Envelope a committed; b is uncommitted and parked in the worker
        # (which holds the writer lock). Read committed state without
        # opening a competing write transaction.
        mid_status, mid_processed = _raw_job_status(app, job_id)
        assert mid_status == "running" and mid_processed == 1
        result: dict = {}

        def do_cancel():
            result["response"] = _cancel(client, job_id)

        canceller = threading.Thread(target=do_cancel, daemon=True)
        canceller.start()
        # Give the cancel thread time to raise its signal and wait on the
        # worker's acknowledgement (it must not be holding a DB lock while
        # it waits).
        time.sleep(0.1)
        gate.set()
        canceller.join(timeout=5)
        assert not canceller.is_alive()
    finally:
        gate.set()
        worker.join(timeout=5)

    response = result["response"]
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "cancelled"

    row = _job_row(app, job_id)
    assert row["status"] == "cancelled"
    assert row["cancelled_at"] is not None
    assert row["claim_token"] is None
    # Only envelope a committed; b was discarded, c never reached.
    assert row["processed"] == 1
    assert row["rewrapped"] == 1
    assert row["skipped"] == row["failed"] == 0
    assert _get_envelope(client, "a").json()["key_version"] == 2
    assert _unwrap_and_decrypt(KEY_V2_BYTES, _get_envelope(client, "a").json()) == (
        PAYLOAD.encode("utf-8")
    )
    for d in ("b", "c"):
        after = _get_envelope(client, d).json()
        assert after == before[d]
        assert after["key_version"] == 1

    # Exactly one rewrap audit event, for the committed envelope a only.
    with app.state.session_factory() as session:
        events = (
            session.query(AuditEvent)
            .filter(AuditEvent.event_type == "rewrap")
            .order_by(AuditEvent.data_id.asc())
            .all()
        )
        assert [(e.data_id, e.status) for e in events] == [("a", "rewrapped")]

    # The job does not resume in a live runner after cancellation.
    runner2 = app.state.rewrap_job_runner
    runner2.run(job_id)
    assert _job_row(app, job_id)["status"] == "cancelled"
    assert _get_envelope(client, "b").json()["key_version"] == 1


def test_cancel_between_pages_stops_scan_and_keeps_committed_page(
    app, client, monkeypatch
):
    _seed(client, ["a", "b", "c", "d"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client, limit=2).json()["job_id"]
    runner = app.state.rewrap_job_runner
    # Process one page (a, b) synchronously but keep the job claimed and
    # running, as a worker parked between pages.
    assert runner._claim(job_id)
    assert runner._advance_page(job_id) is None
    row = _job_row(app, job_id)
    assert row["status"] == "running" and row["processed"] == 2
    # The cancel wins from another thread even though this test thread is
    # notionally the claimant; the guarded update clears the stale token.
    response = _cancel(client, job_id)
    assert response.status_code == 200
    final = _job_row(app, job_id)
    assert final["status"] == "cancelled"
    assert final["processed"] == 2
    assert final["claim_token"] is None
    for d in ("a", "b"):
        assert _get_envelope(client, d).json()["key_version"] == 2
    for d in ("c", "d"):
        assert _get_envelope(client, d).json()["key_version"] == 1
    runner._claim_tokens.pop(job_id, None)
    runner._claim_scopes.pop(job_id, None)


# --- concurrency -----------------------------------------------------------


def test_concurrent_cancels_have_one_winner(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: _cancel(client, job_id), range(8)))
    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(200) == 1
    assert statuses.count(409) == 7
    stamps = {r.json().get("cancelled_at") for r in responses if r.status_code == 200}
    assert len(stamps) == 1
    row = _job_row(app, job_id)
    assert row["status"] == "cancelled"
    assert row["cancelled_at"] is not None
    # A later cancel is still a stable 409.
    assert _cancel(client, job_id).status_code == 409


def test_cancel_racing_runner_leaves_one_terminal_and_consistent_audit(
    app, client, monkeypatch
):
    count = 12
    _seed(client, [f"e{i:02d}" for i in range(count)])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client, limit=3).json()["job_id"]
    runner = app.state.rewrap_job_runner

    entered = threading.Event()
    gate = threading.Event()
    real_rewrap = app_module.rewrap_data_key
    calls = {"n": 0}

    def gated_rewrap(unwrapping_key, wrapping_key, wrapped_key):
        calls["n"] += 1
        # Park inside the uncommitted transaction of the second envelope
        # of the first page (e01): e00 has committed, e01 has not.
        if calls["n"] == 2:
            entered.set()
            gate.wait(timeout=5)
        return real_rewrap(unwrapping_key, wrapping_key, wrapped_key)

    monkeypatch.setattr(app_module, "rewrap_data_key", gated_rewrap)
    worker = threading.Thread(target=runner.run, args=(job_id,), daemon=True)
    worker.start()
    assert entered.wait(timeout=5)
    assert _raw_job_status(app, job_id) == ("running", 1)

    # Several cancels race while the worker is parked in the uncommitted
    # envelope. Releasing the gate lets the worker roll e01 back and the
    # cancel transactions serialize; exactly one may win.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_cancel, client, job_id) for _ in range(4)]
        time.sleep(0.1)
        gate.set()
        responses = [f.result(timeout=10) for f in futures]
    worker.join(timeout=5)

    assert sorted(r.status_code for r in responses).count(200) == 1
    assert all(r.status_code in (200, 409) for r in responses)
    row = _job_row(app, job_id)
    assert row["status"] == "cancelled"
    assert row["claim_token"] is None
    assert row["processed"] == 1

    # The one committed envelope (e00) stays rewrapped; the discarded e01
    # and every later envelope stay on v1. processed equals the committed
    # audit count; no half envelope exists.
    with app.state.session_factory() as session:
        audits = (
            session.query(AuditEvent)
            .filter(AuditEvent.event_type == "rewrap")
            .all()
        )
        rewrapped = {e.data_id for e in audits if e.status == "rewrapped"}
        assert row["processed"] == len(audits)
    stored_rewrapped = {
        d
        for d in [f"e{i:02d}" for i in range(count)]
        if _get_envelope(client, d).json()["key_version"] == 2
    }
    assert stored_rewrapped == rewrapped == {"e00"}


# --- restart persistence ---------------------------------------------------


def test_cancelled_job_survives_restart_and_is_never_requeued(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/cancel-restart.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["a", "b", "c"])
    job_id = _submit(client1, limit=1).json()["job_id"]
    response = _cancel(client1, job_id)
    stamp = response.json()["cancelled_at"]
    assert response.status_code == 200
    app1.state.engine.dispose()

    # Restart with rotation configured and the live pool + recovery sweep
    # running: a cancelled job must not be re-enqueued or advanced.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    app2 = app_module.create_app(url)
    with TestClient(app2) as client2:
        time.sleep(0.2)
        row = _job_row(app2, job_id)
        assert row["status"] == "cancelled"
        assert row["cancelled_at"] == datetime.fromisoformat(stamp)
        assert row["processed"] == 0
        assert row["complete"] is False
        for d in ("a", "b", "c"):
            assert _get_envelope(client2, d).json()["key_version"] == 1
        with app2.state.session_factory() as session:
            assert session.query(AuditEvent).count() == 0
    app2.state.engine.dispose()


def test_running_job_cancelled_mid_scope_stays_cancelled_after_restart(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/cancel-running-restart.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["a", "b", "c", "d"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client1, limit=1).json()["job_id"]
    runner = app1.state.rewrap_job_runner
    assert runner._claim(job_id)
    assert runner._advance_page(job_id) is None  # commits envelope a
    assert _cancel(client1, job_id).status_code == 200
    runner._claim_tokens.pop(job_id, None)
    runner._claim_scopes.pop(job_id, None)
    stopped = _job_row(app1, job_id)
    assert stopped["processed"] == 1
    app1.state.engine.dispose()

    app2 = app_module.create_app(url)
    with TestClient(app2) as client2:
        time.sleep(0.2)
        row = _job_row(app2, job_id)
        assert row["status"] == "cancelled"
        assert row["processed"] == 1
        assert _get_envelope(client2, "a").json()["key_version"] == 2
        for d in ("b", "c", "d"):
            assert _get_envelope(client2, d).json()["key_version"] == 1
    app2.state.engine.dispose()


# --- history ---------------------------------------------------------------


def test_history_can_filter_cancelled_and_pagination_stays_stable(
    app, client, monkeypatch
):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    cancelled_id = _submit(client).json()["job_id"]
    queued_id = _submit(client).json()["job_id"]
    assert _cancel(client, cancelled_id).status_code == 200

    cancelled_page = _history(client, status="cancelled").json()
    assert [j["job_id"] for j in cancelled_page["jobs"]] == [cancelled_id]
    assert cancelled_page["jobs"][0]["status"] == "cancelled"
    assert cancelled_page["complete"] is True

    queued_page = _history(client, status="queued").json()
    assert [j["job_id"] for j in queued_page["jobs"]] == [queued_id]

    both = _history(client).json()
    assert {j["job_id"] for j in both["jobs"]} == {cancelled_id, queued_id}

    # An explicit cancelled job_id resolves in history and via GET.
    named = _history(client, job_id=cancelled_id).json()
    assert [j["job_id"] for j in named["jobs"]] == [cancelled_id]


def test_cancelled_status_rejected_filter_value_is_422_only_for_garbage(client):
    # "cancelled" is accepted; a neighboring unknown status is not.
    assert _history(client, status="cancelled").status_code == 200
    assert _history(client, status="canceled").status_code == 422


# --- server failure --------------------------------------------------------


def test_cancel_storage_failure_is_500_and_changes_nothing(app, client):
    from sqlalchemy import text

    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_jobs"))
    response = _cancel(client, job_id)
    assert response.status_code == 500
    assert _get_envelope(client, "a").status_code == 200


# --- idempotency record stays intact --------------------------------------


def test_idempotent_submission_replay_still_returns_original_202_after_cancel(
    app, client, monkeypatch
):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    headers = {"idempotency-key": "key-1"}
    first = client.post(
        "/v1/rewrap-jobs",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers=headers,
    )
    assert first.status_code == 202
    job_id = first.json()["job_id"]
    assert _cancel(client, job_id).status_code == 200
    replay = client.post(
        "/v1/rewrap-jobs",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers=headers,
    )
    assert replay.status_code == 202
    assert replay.content == first.content
