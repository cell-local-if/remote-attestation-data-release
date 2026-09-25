"""Tests for persistent, recoverable asynchronous envelope rewrap jobs:

POST /v1/rewrap-jobs and GET /v1/rewrap-jobs/{job_id}.

Unlike the one-shot rewrap batch, a job is durable state advanced by a
background runner. Tests build the app without entering its lifespan
(so no pool runs) and drive the runner synchronously for deterministic
single-process checks; restart/recovery behavior is exercised by
disposing one app and entering the lifespan of a fresh app over the same
database file.
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
from proof_release.db import (
    AuditEvent,
    DataEnvelope,
    RewrapBatch,
    RewrapBatchItem,
    RewrapJob,
)
from proof_release.envelopes import (
    MasterKeyError,
    b64url_decode,
    b64url_encode,
)

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap

TENANT = "tenant-a"
WORKLOAD = "workload-1"
PAYLOAD = "batch-secret-payload 🔐"

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
    application = app_module.create_app(f"sqlite:///{tmp_path}/jobs.db")
    yield application
    # No lifespan was entered for the default fixture, so there is no pool
    # to shut down; just drop the engine.
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    # Deliberately NOT used as a context manager: the lifespan only runs
    # in tests that opt in (restart/recovery), keeping the default tests
    # synchronous and deterministic.
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


def _submit(client, *, cursor=None, limit=None, tenant=TENANT, workload=WORKLOAD):
    body: dict = {"tenant_id": tenant, "workload_id": workload}
    if cursor is not None:
        body["cursor"] = cursor
    if limit is not None:
        body["limit"] = limit
    return client.post("/v1/rewrap-jobs", json=body)


def _run_job(app, job_id, *, retry=False):
    """Drive a job's runner synchronously in the calling thread."""
    app.state.rewrap_job_runner.run(job_id, retry=retry)


def _advance_one_page(app, job_id):
    """Claim a job and advance exactly one page, leaving it running.

    The claim is released afterwards so the same thread can later resume
    it with a normal run; tests that simulate a crashed process instead
    overwrite the row's claim token directly.
    """
    runner = app.state.rewrap_job_runner
    assert runner._claim(job_id)
    assert runner._advance_page(job_id) is None
    runner._release_claim(job_id)


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


def _wait_terminal(client, job_id, *, tries=200, delay=0.01):
    last = None
    for _ in range(tries):
        last = _get_job(client, job_id).json()
        if last["status"] in ("succeeded", "failed"):
            return last
        time.sleep(delay)
    raise AssertionError(f"job did not settle: {last}")


def _unwrap_and_decrypt(master_key: bytes, fields: dict) -> bytes:
    data_key = aes_key_unwrap(master_key, b64url_decode(fields["wrapped_key"]))
    iv = b64url_decode(fields["iv"])
    sealed = b64url_decode(fields["ciphertext"]) + b64url_decode(fields["tag"])
    return AESGCM(data_key).decrypt(iv, sealed, None)


# --- request validation ----------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"workload_id": WORKLOAD},
        {"tenant_id": TENANT},
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "   ", "workload_id": WORKLOAD},
        {"tenant_id": 7, "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": "\t"},
        {"tenant_id": TENANT, "workload_id": 3},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": 0},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": 201},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": -1},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": "10"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": 1.5},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": True},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": None},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "cursor": 9},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "cursor": "   "},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "cursor": "bad token!!"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "cursor": "AAA="},
        {},
    ],
)
def test_submit_rejects_invalid_requests(client, body):
    response = client.post("/v1/rewrap-jobs", json=body)
    assert response.status_code == 422, response.text


def test_submit_rejects_non_json_body(client):
    response = client.post(
        "/v1/rewrap-jobs",
        content="not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_forged_or_cross_scope_cursor_returns_422(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    accepted = _submit(client, limit=1)
    assert accepted.status_code == 202
    assert accepted.json()["cursor"] == ""
    # Advance exactly one page so the job is parked mid-scope with a real
    # resume cursor (a full run would complete and reset the cursor to "").
    job_id = accepted.json()["job_id"]
    _advance_one_page(app, job_id)
    with app.state.session_factory() as session:
        advanced_cursor = session.get(RewrapJob, job_id).next_cursor
    assert advanced_cursor

    tampered = advanced_cursor[:-2] + (
        "AA" if advanced_cursor[-2:] != "AA" else "BB"
    )
    assert _submit(client, cursor=tampered).status_code == 422
    assert _submit(client, cursor=advanced_cursor, tenant="tenant-b").status_code == 422
    assert (
        _submit(client, cursor=advanced_cursor, workload="workload-2").status_code
        == 422
    )
    forged = b64url_encode(b'{"t":"tenant-a","w":"workload-1","d":"x"}' + b"0" * 32)
    assert _submit(client, cursor=forged).status_code == 422
    # The empty/missing beginning cursor is accepted in both spellings.
    assert _submit(client, cursor="").status_code == 202


def test_invalid_keyring_returns_500_creates_nothing_and_changes_nothing(
    app, client, monkeypatch
):
    _seed(client, ["a", "b"])
    before = {d: _get_envelope(client, d).json() for d in ("a", "b")}
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")

    response = _submit(client)
    assert response.status_code == 500
    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 0
        assert session.query(DataEnvelope).count() == 2
    for d in ("a", "b"):
        assert _get_envelope(client, d).json() == before[d]


def test_job_write_failure_returns_500_and_changes_no_envelope(app, client):
    from sqlalchemy import text

    _seed(client, ["a", "b"])
    before = {d: _get_envelope(client, d).json() for d in ("a", "b")}
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_jobs"))
    response = _submit(client)
    assert response.status_code == 500
    for d in ("a", "b"):
        assert _get_envelope(client, d).json() == before[d]


# --- acceptance wire format ------------------------------------------------


def test_submit_returns_202_queued_with_expected_fields(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _submit(client, limit=7)
    assert response.status_code == 202
    assert response.headers["content-type"] == "application/json"
    raw = response.content
    # Compact JSON terminated by exactly one newline.
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw and b": " not in raw
    assert list(json.loads(raw)) == [
        "job_id",
        "status",
        "limit",
        "cursor",
        "created_at",
        "updated_at",
    ]
    data = json.loads(raw)
    assert isinstance(data["job_id"], str) and len(data["job_id"]) == 36
    assert data["status"] == "queued"
    assert data["limit"] == 7 and isinstance(data["limit"], int)
    assert data["cursor"] == "" and isinstance(data["cursor"], str)
    for key in ("created_at", "updated_at"):
        parsed = datetime.fromisoformat(data[key])
        assert parsed.utcoffset() == timedelta(0)
    assert data["created_at"] == data["updated_at"]


def test_submit_echoes_verified_cursor(app, client, monkeypatch):
    _seed(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _submit(client, limit=1)
    job_id = first.json()["job_id"]
    assert first.json()["cursor"] == ""
    _advance_one_page(app, job_id)
    with app.state.session_factory() as session:
        cursor = session.get(RewrapJob, job_id).next_cursor
    assert cursor
    second = _submit(client, cursor=cursor, limit=1)
    assert second.status_code == 202
    assert second.json()["cursor"] == cursor


def test_default_limit_is_50(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    accepted = _submit(client)
    assert accepted.json()["limit"] == 50
    job_id = accepted.json()["job_id"]
    with app.state.session_factory() as session:
        assert session.get(RewrapJob, job_id).limit == 50


# --- background advancement ------------------------------------------------


def test_empty_scope_succeeds_immediately(app, client):
    accepted = _submit(client)
    job_id = accepted.json()["job_id"]
    _run_job(app, job_id)
    data = _get_job(client, job_id).json()
    assert data["status"] == "succeeded"
    assert data["processed"] == 0
    assert data["rewrapped"] == 0
    assert data["skipped"] == 0
    assert data["failed"] == 0
    assert data["next_cursor"] == ""
    assert data["complete"] is True


def test_job_rewraps_in_data_id_order_and_only_changes_wrapping(
    app, client, monkeypatch
):
    data_ids = ["zeta", "alpha", "mid", "001", "beta"]
    _seed(client, data_ids)
    before = {d: _get_envelope(client, d).json() for d in data_ids}
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    data = _get_job(client, job_id).json()
    assert list(data) == [
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
    assert data["status"] == "succeeded"
    assert data["processed"] == 5
    assert data["rewrapped"] == 5
    assert data["skipped"] == 0
    assert data["failed"] == 0
    assert data["next_cursor"] == ""
    assert data["complete"] is True

    for d in data_ids:
        after = _get_envelope(client, d).json()
        assert after["key_version"] == 2
        assert after["wrapped_key"] != before[d]["wrapped_key"]
        for field in ("ciphertext", "iv", "tag", "created_at", "data_id"):
            assert after[field] == before[d][field]
        assert _unwrap_and_decrypt(KEY_V2_BYTES, after) == PAYLOAD.encode("utf-8")


def test_job_pages_through_large_scope_with_fixed_limit(app, client, monkeypatch):
    _seed(client, [f"item-{i:03d}" for i in range(125)])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client, limit=50).json()["job_id"]
    _run_job(app, job_id)
    data = _get_job(client, job_id).json()
    assert data["status"] == "succeeded"
    assert data["processed"] == 125
    assert data["rewrapped"] == 125
    assert data["complete"] is True
    assert data["next_cursor"] == ""


def test_job_is_scoped_to_tenant_and_workload(app, client, monkeypatch):
    _seed(client, ["a1", "a2"], tenant=TENANT, workload=WORKLOAD)
    _seed(client, ["b1"], tenant="tenant-b", workload=WORKLOAD)
    _seed(client, ["c1"], tenant=TENANT, workload="workload-9")
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    data = _get_job(client, job_id).json()
    assert data["processed"] == 2 and data["rewrapped"] == 2
    assert _get_envelope(client, "b1", tenant="tenant-b").json()["key_version"] == 1
    assert (
        _get_envelope(
            client, "c1", tenant=TENANT, workload="workload-9"
        ).json()["key_version"]
        == 1
    )


def test_current_version_envelopes_are_skipped(app, client, monkeypatch):
    _seed(client, ["old-1", "new-1", "old-2"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert (
        client.post(
            "/v1/data-envelopes/new-1/rewrap",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).status_code
        == 200
    )
    current_before = _get_envelope(client, "new-1").json()

    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    data = _get_job(client, job_id).json()
    assert data["processed"] == 3
    assert data["rewrapped"] == 2
    assert data["skipped"] == 1
    assert data["failed"] == 0
    assert _get_envelope(client, "new-1").json() == current_before


def test_job_writes_existing_rewrap_audit_events_but_no_batch_rows(
    app, client, monkeypatch
):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    with app.state.session_factory() as session:
        events = (
            session.query(AuditEvent)
            .filter(AuditEvent.event_type == "rewrap")
            .order_by(AuditEvent.data_id.asc())
            .all()
        )
        assert [(e.data_id, e.status) for e in events] == [
            ("a", "rewrapped"),
            ("b", "rewrapped"),
        ]
        for event in events:
            assert event.grant_id is None
            assert event.decision_id is None
            assert event.capability_sha256 is None
        assert session.query(RewrapBatch).count() == 0
        assert session.query(RewrapBatchItem).count() == 0


# --- status lifecycle ------------------------------------------------------


def test_status_transitions_through_running_under_a_live_pool(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/running.db"
    application = app_module.create_app(url)

    seen_running = threading.Event()
    release = threading.Event()

    def post_claim(job_id):
        # Fires in the worker after queued -> running is committed and the
        # claim transaction closed, so the request thread can read the
        # running row while the worker parks here, lock-free.
        seen_running.set()
        release.wait(timeout=5)

    application.state.rewrap_job_runner.post_claim = post_claim
    try:
        with TestClient(application) as client:
            _seed(client, ["a", "b", "c"])  # sealed under key version 1
            monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
            job_id = _submit(client).json()["job_id"]
            assert seen_running.wait(timeout=5)
            mid = _get_job(client, job_id).json()
            assert mid["status"] == "running"
            assert mid["complete"] is False
            release.set()
            final = _wait_terminal(client, job_id)
            assert final["status"] == "succeeded"
            assert final["processed"] == 3
            assert final["rewrapped"] == 3
    finally:
        release.set()
    application.state.engine.dispose()


# --- failure handling ------------------------------------------------------


def test_missing_historical_key_fails_job_parks_cursor_and_keeps_envelope(
    app, client, monkeypatch
):
    data_ids = ["a0", "a1", "a2", "a3"]
    _seed(client, data_ids)
    before = {d: _get_envelope(client, d).json() for d in data_ids}
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)

    job_id = _submit(client, limit=10).json()["job_id"]
    _run_job(app, job_id)
    data = _get_job(client, job_id).json()
    assert data["status"] == "failed"
    assert data["processed"] == 0
    assert data["rewrapped"] == 0
    assert data["skipped"] == 0
    assert data["failed"] == 1
    assert data["complete"] is False
    # Cursor parks before the first (failing) envelope.
    assert data["next_cursor"] == ""
    for d in data_ids:
        assert _get_envelope(client, d).json() == before[d]


def test_keyring_becoming_bad_mid_job_fails_and_parks_cursor(
    app, client, monkeypatch
):
    _seed(client, ["a0", "a1", "a2"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    real_load = app_module.load_keyring
    calls = {"n": 0}

    def failing_load():
        calls["n"] += 1
        # Call 1 is the submit check; a0 uses call 2; a1 (call 3) fails.
        if calls["n"] == 3:
            raise MasterKeyError("keyring vanished")
        return real_load()

    monkeypatch.setattr(app_module, "load_keyring", failing_load)
    job_id = _submit(client, limit=10).json()["job_id"]
    _run_job(app, job_id)

    data = _get_job(client, job_id).json()
    assert data["status"] == "failed"
    assert data["processed"] == 1
    assert data["rewrapped"] == 1
    assert data["failed"] == 1
    assert data["complete"] is False
    assert _get_envelope(client, "a0").json()["key_version"] == 2
    assert _get_envelope(client, "a1").json()["key_version"] == 1
    assert _get_envelope(client, "a2").json()["key_version"] == 1


def test_single_rewrap_failure_commits_prior_progress_and_parks(
    app, client, monkeypatch
):
    data_ids = ["a0", "a1", "a2", "a3"]
    _seed(client, data_ids)
    before = {d: _get_envelope(client, d).json() for d in data_ids}
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

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

    data = _get_job(client, job_id).json()
    assert data["status"] == "failed"
    assert data["processed"] == 2
    assert data["rewrapped"] == 2
    assert data["failed"] == 1
    assert data["complete"] is False
    assert _get_envelope(client, "a0").json()["key_version"] == 2
    assert _get_envelope(client, "a1").json()["key_version"] == 2
    for d in ("a2", "a3"):
        assert _get_envelope(client, d).json() == before[d]

    # A live runner never re-attempts a failed job; it stays failed.
    _run_job(app, job_id)
    assert _get_job(client, job_id).json()["status"] == "failed"


def test_failed_job_is_not_retried_until_restart_then_succeeds(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart-failed.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["a0", "a1", "a2", "a3"])
    app1.state.engine.dispose()

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    job_id = _submit(client2, limit=10).json()["job_id"]
    app2.state.rewrap_job_runner.run(job_id)
    parked = _get_job(client2, job_id).json()
    assert parked["status"] == "failed" and parked["failed"] == 1
    assert parked["next_cursor"] == ""
    app2.state.engine.dispose()

    # Restart with the historical key restored; the lifespan sweep
    # retries the parked failed job from its cursor.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    app3 = app_module.create_app(url)
    with TestClient(app3) as client3:
        final = _wait_terminal(client3, job_id)
        assert final["status"] == "succeeded"
        assert final["processed"] == 4
        assert final["rewrapped"] == 4
        # The historical failure counter is retained.
        assert final["failed"] == 1
        assert final["complete"] is True
        for d in ("a0", "a1", "a2", "a3"):
            assert _get_envelope(client3, d).json()["key_version"] == 2
    app3.state.engine.dispose()


# --- crash recovery --------------------------------------------------------


def test_queued_job_survives_process_death_and_is_resumed_on_restart(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/queued-restart.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["r0", "r1", "r2", "r3"])
    # Submit without a running pool: the row is durably queued and nothing
    # advances it before the simulated crash.
    job_id = _submit(client1, limit=2).json()["job_id"]
    assert _get_job(client1, job_id).json()["status"] == "queued"
    app1.state.engine.dispose()

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    app2 = app_module.create_app(url)
    with TestClient(app2) as client2:
        final = _wait_terminal(client2, job_id)
        assert final["status"] == "succeeded"
        assert final["processed"] == 4
        assert final["rewrapped"] == 4
        assert final["complete"] is True
    app2.state.engine.dispose()


def test_running_job_with_stale_claim_is_resumed_from_its_cursor(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/stale-claim.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["e0", "e1", "e2", "e3"])  # sealed under version 1
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client1, limit=2).json()["job_id"]
    # Advance exactly one page (2 envelopes); the job is left running and
    # claimed as if the process died between pages.
    _advance_one_page(app1, job_id)
    mid = _get_job(client1, job_id).json()
    assert mid["status"] == "running" and mid["processed"] == 2
    with app1.state.session_factory() as session:
        cursor = session.get(RewrapJob, job_id).next_cursor
        session.execute(
            RewrapJob.__table__.update()
            .where(RewrapJob.job_id == job_id)
            .values(claim_token="dead-process-token")
        )
        session.commit()
    assert cursor
    app1.state.rewrap_job_runner._claim_tokens.pop(job_id, None)
    app1.state.engine.dispose()

    app2 = app_module.create_app(url)
    with TestClient(app2) as client2:
        final = _wait_terminal(client2, job_id)
        assert final["status"] == "succeeded"
        assert final["processed"] == 4
        assert final["rewrapped"] == 4
        assert final["complete"] is True
        for d in ("e0", "e1", "e2", "e3"):
            assert _get_envelope(client2, d).json()["key_version"] == 2
    app2.state.engine.dispose()


def test_restart_does_not_touch_succeeded_jobs(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/done-restart.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["x"])  # sealed under version 1
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client1).json()["job_id"]
    app1.state.rewrap_job_runner.run(job_id)
    done = _get_job(client1, job_id).json()
    assert done["status"] == "succeeded"
    stamp = done["updated_at"]
    app1.state.engine.dispose()

    app2 = app_module.create_app(url)
    with TestClient(app2) as client2:
        time.sleep(0.02)
        again = _get_job(client2, job_id).json()
        assert again["status"] == "succeeded"
        assert again["updated_at"] == stamp
        assert again["processed"] == 1
    app2.state.engine.dispose()


# --- concurrency -----------------------------------------------------------


def test_concurrent_jobs_advance_each_envelope_at_most_once(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    count = 8
    url = f"sqlite:///{tmp_path}/concurrent.db"
    application = app_module.create_app(url)
    job_ids = []
    with TestClient(application) as client:
        _seed(client, [f"e{n}" for n in range(count)])  # sealed under version 1
        monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
        with ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(
                pool.map(lambda _: _submit(client, limit=50), range(6))
            )
        assert all(r.status_code == 202 for r in responses)
        job_ids = [r.json()["job_id"] for r in responses]
        finals = [_wait_terminal(client, jid) for jid in job_ids]

        assert all(f["status"] == "succeeded" for f in finals)
        totals = {
            key: sum(f[key] for f in finals)
            for key in ("processed", "rewrapped", "skipped", "failed")
        }
        assert totals["rewrapped"] == count
        assert totals["failed"] == 0
        assert totals["processed"] == count * 6
        assert totals["skipped"] == count * 6 - count
        for n in range(count):
            stored = _get_envelope(client, f"e{n}").json()
            assert stored["key_version"] == 2
            assert _unwrap_and_decrypt(KEY_V2_BYTES, stored) == PAYLOAD.encode(
                "utf-8"
            )
        # Exactly one rewrapped compliance event per envelope, ever.
        with application.state.session_factory() as session:
            rewrapped_events = (
                session.query(AuditEvent)
                .filter(
                    AuditEvent.event_type == "rewrap",
                    AuditEvent.status == "rewrapped",
                )
                .count()
            )
            assert rewrapped_events == count
    application.state.engine.dispose()


# --- job lookup ------------------------------------------------------------


def test_get_unknown_or_cross_scope_job_returns_404(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)

    unknown = "00000000-0000-0000-0000-000000000000"
    assert _get_job(client, unknown).status_code == 404
    assert _get_job(client, job_id, tenant="tenant-b").status_code == 404
    assert _get_job(client, job_id, workload="workload-2").status_code == 404
    # Uppercase spelling is normalized and resolves.
    assert _get_job(client, job_id.upper()).status_code == 200


@pytest.mark.parametrize(
    "job_id",
    [
        "not-a-uuid",
        "abc123",
        "00000000-0000-0000-0000-00000000000Z",
        "  00000000-0000-0000-0000-000000000000",
    ],
)
def test_get_job_rejects_malformed_identifier(client, job_id):
    response = client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_get_job_rejects_blank_or_missing_scope_query(client):
    job_id = "00000000-0000-0000-0000-000000000000"
    assert (
        client.get(
            f"/v1/rewrap-jobs/{job_id}",
            params={"tenant_id": "  ", "workload_id": WORKLOAD},
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/v1/rewrap-jobs/{job_id}",
            params={"tenant_id": TENANT, "workload_id": ""},
        ).status_code
        == 422
    )
    assert client.get(f"/v1/rewrap-jobs/{job_id}").status_code == 422


def test_get_job_without_identifier_returns_422(client):
    response = client.get(
        "/v1/rewrap-jobs/",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_get_job_failure_returns_500_with_no_partial_progress(app, client):
    from sqlalchemy import text

    _seed(client, ["a"])
    job_id = _submit(client).json()["job_id"]
    _run_job(app, job_id)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_jobs"))
    assert _get_job(client, job_id).status_code == 500


# --- wire format, timestamps, secrecy --------------------------------------


def test_responses_are_compact_json_with_single_newline_and_no_floats(
    app, client, monkeypatch
):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    accepted = _submit(client)
    raw = accepted.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw and b": " not in raw
    parsed = json.loads(raw)
    assert isinstance(parsed["limit"], int) and not isinstance(parsed["limit"], bool)
    assert isinstance(parsed["cursor"], str)

    job_id = accepted.json()["job_id"]
    _run_job(app, job_id)
    summary_raw = _get_job(client, job_id).content
    assert summary_raw.endswith(b"\n") and not summary_raw.endswith(b"\n\n")
    assert b", " not in summary_raw and b": " not in summary_raw
    summary = json.loads(summary_raw)
    assert isinstance(summary["complete"], bool)
    assert isinstance(summary["next_cursor"], str)
    for key in ("processed", "rewrapped", "skipped", "failed"):
        assert isinstance(summary[key], int) and not isinstance(summary[key], bool)
    for key in ("created_at", "updated_at"):
        parsed_ts = datetime.fromisoformat(summary[key])
        assert parsed_ts.utcoffset() == timedelta(0)


def test_created_at_is_stable_and_updated_at_advances(app, client, monkeypatch):
    _seed(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client, limit=1).json()["job_id"]
    accepted = _get_job(client, job_id).json()
    _advance_one_page(app, job_id)
    after_first_page = _get_job(client, job_id).json()
    _run_job(app, job_id)
    final = _get_job(client, job_id).json()

    assert final["created_at"] == accepted["created_at"] == after_first_page["created_at"]
    assert final["updated_at"] >= after_first_page["updated_at"] >= accepted["updated_at"]
    assert final["status"] == "succeeded"


def test_job_never_exposes_payload_or_key_material(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    accepted = _submit(client)
    job_id = accepted.json()["job_id"]
    _run_job(app, job_id)
    for response in (accepted, _get_job(client, job_id)):
        for secret in (
            PAYLOAD,
            KEY_V1,
            KEY_V2,
            "wrapped_key",
            "ciphertext",
            "payload",
        ):
            assert secret not in response.text
    with app.state.session_factory() as session:
        job = session.get(RewrapJob, job_id)
        for forbidden in ("wrapped_key", "ciphertext", "iv", "tag"):
            assert not hasattr(job, forbidden)
