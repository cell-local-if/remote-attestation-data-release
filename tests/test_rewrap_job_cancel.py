"""Ad-hoc verification of POST /v1/rewrap-jobs/{job_id}/cancel."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import AuditEvent, DataEnvelope, RewrapJob
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V2_BYTES = b"fedcba9876543210fedcba9876543210"
KEY_V1 = b64url_encode(KEY_V1_BYTES)
KEY_V2 = b64url_encode(KEY_V2_BYTES)
KEYRING_V1_V2 = json.dumps(
    {"current_version": 2, "keys": {"1": KEY_V1, "2": KEY_V2}}
)


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


def _seed(client, data_ids, tenant=TENANT, workload=WORKLOAD):
    for data_id in data_ids:
        assert client.post(
            "/v1/data-envelopes",
            json={
                "tenant_id": tenant,
                "workload_id": workload,
                "data_id": data_id,
                "payload": "secret",
            },
        ).status_code == 201


def _submit(client, tenant=TENANT, workload=WORKLOAD, **extra):
    body = {"tenant_id": tenant, "workload_id": workload, **extra}
    return client.post("/v1/rewrap-jobs", json=body)


def _cancel(client, job_id, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        f"/v1/rewrap-jobs/{job_id}/cancel",
        json={"tenant_id": tenant, "workload_id": workload},
    )


def _get_job(client, job_id, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def test_cancel_queued_job(app, client):
    job_id = _submit(client).json()["job_id"]
    resp = _cancel(client, job_id)
    assert resp.status_code == 200, resp.text
    raw = resp.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    body = json.loads(raw)
    assert list(body.keys()) == ["cancelled_at", "job_id", "status"]
    assert body["job_id"] == job_id
    assert body["status"] == "cancelled"
    parsed = datetime.fromisoformat(body["cancelled_at"])
    assert parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0

    shown = _get_job(client, job_id).json()
    assert shown["status"] == "cancelled"

    # repeat cancel -> 409, marker unchanged
    again = _cancel(client, job_id)
    assert again.status_code == 409
    with app.state.session_factory() as session:
        row = session.get(RewrapJob, job_id)
        assert row.status == "cancelled"
        assert row.cancelled_at is not None
        assert json.loads(raw)["cancelled_at"] == row.cancelled_at.isoformat()


def test_cancel_terminal_jobs_409(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    ok_job = _submit(client).json()["job_id"]
    app.state.rewrap_job_runner.run(ok_job)
    assert _get_job(client, ok_job).json()["status"] == "succeeded"
    assert _cancel(client, ok_job).status_code == 409

    # failed job
    monkeypatch.delenv("PROOF_RELEASE_KEYRING")
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V2)
    bad_job = _submit(client).json()["job_id"]
    app.state.rewrap_job_runner.run(bad_job)
    assert _get_job(client, bad_job).json()["status"] == "failed"
    assert _cancel(client, bad_job).status_code == 409


def test_cancel_unknown_and_cross_scope_404(client):
    job_id = _submit(client).json()["job_id"]
    zero = "00000000-0000-0000-0000-000000000000"
    assert _cancel(client, zero).status_code == 404
    assert _cancel(client, job_id, tenant="other").status_code == 404
    assert _cancel(client, job_id, workload="other").status_code == 404
    # still queued afterwards (no state change)
    assert _get_job(client, job_id).json()["status"] == "queued"


def test_cancel_validation_422(client):
    job_id = _submit(client).json()["job_id"]
    for bad_id in ["", " ", "not-a-uuid", "AAA=", "0" * 36]:
        resp = client.post(
            f"/v1/rewrap-jobs/{bad_id}/cancel",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
        assert resp.status_code == 422, (bad_id, resp.status_code)
    bad_bodies = [
        {},
        {"tenant_id": TENANT},
        {"workload_id": WORKLOAD},
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "  ", "workload_id": WORKLOAD},
        {"tenant_id": 7, "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": None},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "extra": 1},
    ]
    for body in bad_bodies:
        resp = client.post(f"/v1/rewrap-jobs/{job_id}/cancel", json=body)
        assert resp.status_code == 422, body
    # job untouched
    assert _get_job(client, job_id).json()["status"] == "queued"


def test_cancel_running_job_stops_runner(app, client, monkeypatch):
    _seed(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client, limit=1).json()["job_id"]
    runner = app.state.rewrap_job_runner
    assert runner._claim(job_id)
    resp = _cancel(client, job_id)
    assert resp.status_code == 200
    # runner stops at the next boundary; nothing is processed
    runner._advance_page(job_id)
    runner.run(job_id)
    shown = _get_job(client, job_id).json()
    assert shown["status"] == "cancelled"
    assert shown["processed"] == 0
    with app.state.session_factory() as session:
        assert session.query(AuditEvent).count() == 0
        for env in session.query(DataEnvelope).all():
            assert env.key_version == 1


def test_cancel_mid_run_keeps_committed_and_discards_rest(
    app, client, monkeypatch
):
    _seed(client, [f"d{i:02d}" for i in range(50)])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client, limit=1).json()["job_id"]
    runner = app.state.rewrap_job_runner

    # Widen the lock-free gap between envelope commits so the cancel
    # lands while the job is still running (on SQLite every transaction
    # holds the writer lock, so a pause inside the transaction would
    # starve the cancel instead).
    real_process = runner._process_one

    def slow_process(job_id, envelope):
        outcome = real_process(job_id, envelope)
        time.sleep(0.02)
        return outcome

    runner._process_one = slow_process
    worker = threading.Thread(target=runner.run, args=(job_id,))
    worker.start()
    # Wait until at least one envelope has committed, then cancel.
    for _ in range(500):
        with app.state.session_factory() as session:
            if session.get(RewrapJob, job_id).processed > 0:
                break
        time.sleep(0.01)
    resp = _cancel(client, job_id)
    assert resp.status_code == 200, resp.text
    worker.join(timeout=10)
    assert not worker.is_alive()

    shown = _get_job(client, job_id).json()
    assert shown["status"] == "cancelled"
    assert shown["processed"] == shown["rewrapped"]
    assert 0 < shown["processed"] < 50
    with app.state.session_factory() as session:
        # Every committed envelope is fully rotated with its audit row;
        # everything after the stop point is byte-identical to before.
        events = session.query(AuditEvent).all()
        assert len(events) == shown["processed"]
        rotated = sum(
            1 for e in session.query(DataEnvelope).all() if e.key_version == 2
        )
        assert rotated == shown["processed"]
        row = session.get(RewrapJob, job_id)
        assert row.cancelled_at is not None


def test_concurrent_cancels_single_winner(app, client):
    job_id = _submit(client).json()["job_id"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _cancel(client, job_id), range(16)))
    codes = sorted(r.status_code for r in results)
    assert codes.count(200) == 1
    assert codes.count(409) == 15
    bodies = {r.content for r in results if r.status_code == 200}
    assert len(bodies) == 1


def test_restart_does_not_requeue_cancelled(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart.db"
    # No lifespan on the first app: the job stays queued until cancelled.
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["a", "b"])
    job_id = _submit(client1).json()["job_id"]
    assert _cancel(client1, job_id).status_code == 200
    app1.state.engine.dispose()

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    app2 = app_module.create_app(url)
    with TestClient(app2) as client2:
        time.sleep(0.3)  # allow the recovery sweep to run
        shown = _get_job(client2, job_id).json()
        assert shown["status"] == "cancelled"
        assert shown["processed"] == 0
        with app2.state.session_factory() as session:
            assert session.query(AuditEvent).count() == 0
            assert all(
                e.key_version == 1 for e in session.query(DataEnvelope).all()
            )
    app2.state.engine.dispose()


def test_history_filters_cancelled(app, client):
    keep = _submit(client).json()["job_id"]
    gone = _submit(client).json()["job_id"]
    assert _cancel(client, gone).status_code == 200
    resp = client.get(
        "/v1/rewrap-jobs",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD, "status": "cancelled"},
    )
    assert resp.status_code == 200
    rows = resp.json()["jobs"]
    assert [r["job_id"] for r in rows] == [gone]
    assert rows[0]["status"] == "cancelled"
    # unfiltered history still lists both
    both = client.get(
        "/v1/rewrap-jobs",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()["jobs"]
    assert {r["job_id"] for r in both} == {keep, gone}
