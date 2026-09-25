"""Tests for persistent, asynchronous, resumable rewrap jobs:

POST /v1/rewrap-jobs and GET /v1/rewrap-jobs/{job_id}.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import DataEnvelope, RewrapJob
from proof_release.envelopes import (
    MasterKeyError,
    b64url_decode,
    b64url_encode,
)

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap

TENANT = "tenant-a"
WORKLOAD = "workload-1"
PAYLOAD = "job-secret-payload 🔐"

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


def _submit(client, *, cursor=..., limit=None, tenant=TENANT, workload=WORKLOAD):
    body: dict = {"tenant_id": tenant, "workload_id": workload}
    if cursor is not ...:
        body["cursor"] = cursor
    if limit is not None:
        body["limit"] = limit
    return client.post("/v1/rewrap-jobs", json=body)


def _get_job(client, job_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _wait_for_terminal(client, job_id, *, timeout=15.0, **scope):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = _get_job(client, job_id, **scope)
        assert response.status_code == 200, response.text
        data = response.json()
        if data["status"] in ("succeeded", "failed"):
            return data
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not settle: {data}")


def _get_envelope(client, data_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/data-envelopes/{data_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


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
def test_submit_rejects_invalid_requests(app, client, body):
    response = client.post("/v1/rewrap-jobs", json=body)
    assert response.status_code == 422, response.text
    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 0


def test_submit_rejects_non_json_body(client):
    response = client.post(
        "/v1/rewrap-jobs",
        content="not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_forged_or_cross_scope_cursor_returns_422_and_creates_nothing(
    app, client, monkeypatch
):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    good_cursor = app_module._encode_cursor(TENANT, WORKLOAD, "a")

    # Tampered token.
    tampered = good_cursor[:-2] + ("AA" if good_cursor[-2:] != "AA" else "BB")
    assert _submit(client, cursor=tampered).status_code == 422
    # Token minted for another tenant/workload.
    other_tenant = app_module._encode_cursor("tenant-b", WORKLOAD, "a")
    other_workload = app_module._encode_cursor(TENANT, "workload-2", "a")
    assert _submit(client, cursor=other_tenant).status_code == 422
    assert _submit(client, cursor=other_workload).status_code == 422
    # Opaque garbage that still parses as base64url.
    forged = b64url_encode(b'{"t":"tenant-a","w":"workload-1","d":"x"}' + b"0" * 32)
    assert _submit(client, cursor=forged).status_code == 422
    with app.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 0


# --- submission contract ---------------------------------------------------


def test_submit_returns_202_with_queued_job_and_compact_newline_json(client):
    response = _submit(client)
    assert response.status_code == 202
    raw = response.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw and b": " not in raw
    data = json.loads(raw)
    assert list(data) == [
        "job_id",
        "status",
        "limit",
        "cursor",
        "created_at",
        "updated_at",
    ]
    assert isinstance(data["job_id"], str)
    assert data["status"] == "queued"
    assert data["limit"] == 50 and isinstance(data["limit"], int)
    assert data["cursor"] == "" and isinstance(data["cursor"], str)
    for key in ("created_at", "updated_at"):
        stamped = datetime.fromisoformat(data[key])
        assert stamped.utcoffset() == timedelta(0)


def test_submit_echoes_limit_and_cursor(client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _submit(client, limit=1)
    assert first.status_code == 202
    assert first.json()["limit"] == 1
    cursor = app_module._encode_cursor(TENANT, WORKLOAD, "a")
    second = _submit(client, cursor=cursor, limit=7)
    assert second.status_code == 202
    assert second.json()["limit"] == 7
    assert second.json()["cursor"] == cursor
    # An explicit empty cursor is the same as omitting it.
    third = _submit(client, cursor="")
    assert third.status_code == 202
    assert third.json()["cursor"] == ""


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


# --- background execution --------------------------------------------------


def test_job_rewraps_scope_and_reports_terminal_progress(
    app, client, monkeypatch
):
    data_ids = ["zeta", "alpha", "mid", "001", "beta"]
    _seed(client, data_ids)
    before = {d: _get_envelope(client, d).json() for d in data_ids}
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    job_id = _submit(client).json()["job_id"]
    data = _wait_for_terminal(client, job_id)
    assert data["status"] == "succeeded"
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
    assert data["job_id"] == job_id
    assert data["processed"] == 5
    assert data["rewrapped"] == 5
    assert data["skipped"] == 0
    assert data["failed"] == 0
    assert data["next_cursor"] == ""
    assert data["complete"] is True
    for key in ("processed", "rewrapped", "skipped", "failed"):
        assert isinstance(data[key], int) and not isinstance(data[key], bool)
    assert isinstance(data["next_cursor"], str)
    assert isinstance(data["complete"], bool)
    for key in ("created_at", "updated_at"):
        assert datetime.fromisoformat(data[key]).utcoffset() == timedelta(0)
    assert datetime.fromisoformat(data["updated_at"]) >= datetime.fromisoformat(
        data["created_at"]
    )

    for d in data_ids:
        after = _get_envelope(client, d).json()
        assert after["key_version"] == 2
        assert after["wrapped_key"] != before[d]["wrapped_key"]
        # Nothing but the wrapping key changes.
        for field in ("ciphertext", "iv", "tag", "created_at", "data_id"):
            assert after[field] == before[d][field]
        assert _unwrap_and_decrypt(KEY_V2_BYTES, after) == PAYLOAD.encode("utf-8")

    # Every envelope produced exactly one rewrap compliance audit event.
    events = client.get(
        "/v1/compliance/audit-events",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "event_type": "rewrap",
        },
    ).json()
    assert sorted(e["data_id"] for e in events["events"]) == sorted(data_ids)
    assert all(e["status"] == "rewrapped" for e in events["events"])


def test_empty_scope_job_succeeds_immediately(client):
    job_id = _submit(client).json()["job_id"]
    data = _wait_for_terminal(client, job_id)
    assert data["status"] == "succeeded"
    assert data["processed"] == 0
    assert data["complete"] is True
    assert data["next_cursor"] == ""


def test_current_version_envelopes_are_skipped(app, client, monkeypatch):
    _seed(client, ["old-1", "new-1", "old-2", "new-2"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    for data_id in ("new-1", "new-2"):
        assert (
            client.post(
                f"/v1/data-envelopes/{data_id}/rewrap",
                json={"tenant_id": TENANT, "workload_id": WORKLOAD},
            ).status_code
            == 200
        )
    current_before = {
        d: _get_envelope(client, d).json() for d in ("new-1", "new-2")
    }

    job_id = _submit(client).json()["job_id"]
    data = _wait_for_terminal(client, job_id)
    assert data["status"] == "succeeded"
    assert data["processed"] == 4
    assert data["rewrapped"] == 2
    assert data["skipped"] == 2
    assert data["failed"] == 0
    # Skipped envelopes are byte-identical afterwards.
    for d in ("new-1", "new-2"):
        assert _get_envelope(client, d).json() == current_before[d]


def test_job_is_scoped_to_tenant_and_workload(client, monkeypatch):
    _seed(client, ["a1", "a2"], tenant=TENANT, workload=WORKLOAD)
    _seed(client, ["b1"], tenant="tenant-b", workload=WORKLOAD)
    _seed(client, ["c1"], tenant=TENANT, workload="workload-9")
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    job_id = _submit(client).json()["job_id"]
    data = _wait_for_terminal(client, job_id)
    assert data["status"] == "succeeded"
    assert data["processed"] == 2
    assert _get_envelope(client, "b1", tenant="tenant-b").json()["key_version"] == 1
    assert (
        _get_envelope(client, "c1", tenant=TENANT, workload="workload-9").json()[
            "key_version"
        ]
        == 1
    )


def test_job_resumes_from_cursor_without_rewrapping_twice(app, client, monkeypatch):
    _seed(client, [f"i{n}" for n in range(6)])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _submit(client, limit=6).json()["job_id"]
    assert _wait_for_terminal(client, first)["rewrapped"] == 6
    after_first = {f"i{n}": _get_envelope(client, f"i{n}").json() for n in range(6)}

    # Replaying from the beginning only observes current-version results.
    second = _submit(client, limit=6).json()["job_id"]
    data = _wait_for_terminal(client, second)
    assert data["status"] == "succeeded"
    assert data["rewrapped"] == 0
    assert data["skipped"] == 6
    for n in range(6):
        assert _get_envelope(client, f"i{n}").json() == after_first[f"i{n}"]


# --- failure handling ------------------------------------------------------


def test_missing_historical_key_fails_job_and_keeps_envelopes(
    app, client, monkeypatch
):
    data_ids = ["a0", "a1", "a2", "a3"]
    _seed(client, data_ids)
    before = {d: _get_envelope(client, d).json() for d in data_ids}
    # Keyring without the historical version 1: the first envelope fails.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)

    job_id = _submit(client, limit=10).json()["job_id"]
    data = _wait_for_terminal(client, job_id)
    assert data["status"] == "failed"
    assert data["processed"] == 0
    assert data["rewrapped"] == 0
    assert data["failed"] == 1
    assert data["complete"] is False
    # Failure on the first item leaves the cursor at the scope beginning.
    assert data["next_cursor"] == ""
    for d in data_ids:
        assert _get_envelope(client, d).json() == before[d]

    # A failed job is terminal; restoring the key and submitting a new job
    # from the parked cursor advances the scope to completion.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    again = _submit(client, cursor=data["next_cursor"], limit=10).json()["job_id"]
    resumed = _wait_for_terminal(client, again)
    assert resumed["status"] == "succeeded"
    assert resumed["rewrapped"] == 4
    assert resumed["failed"] == 0


def test_rewrap_failure_mid_run_commits_prior_progress_and_parks_cursor(
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
    data = _wait_for_terminal(client, job_id)
    monkeypatch.setattr(app_module, "rewrap_data_key", real_rewrap)

    assert data["status"] == "failed"
    assert data["processed"] == 2
    assert data["rewrapped"] == 2
    assert data["failed"] == 1
    assert data["complete"] is False
    # The cursor is parked before the failed envelope (after a1).
    assert data["next_cursor"] != ""
    # The failed envelope and everything after it are untouched.
    for d in ("a2", "a3"):
        assert _get_envelope(client, d).json() == before[d]

    # A new job resumes exactly where the failed one stopped.
    resumed_id = _submit(client, cursor=data["next_cursor"], limit=10).json()["job_id"]
    resumed = _wait_for_terminal(client, resumed_id)
    assert resumed["status"] == "succeeded"
    assert resumed["rewrapped"] == 2
    assert _get_envelope(client, "a2").json()["key_version"] == 2


def test_keyring_becoming_invalid_mid_run_fails_job(app, client, monkeypatch):
    _seed(client, ["a0", "a1", "a2"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    real_load = app_module.load_keyring
    calls = {"n": 0}

    def failing_load():
        calls["n"] += 1
        # Call 1 is the up-front submit check; envelope 1 uses call 2;
        # envelope 2 (call 3) observes the keyring going bad mid-run.
        if calls["n"] == 3:
            raise MasterKeyError("keyring vanished")
        return real_load()

    monkeypatch.setattr(app_module, "load_keyring", failing_load)
    job_id = _submit(client, limit=10).json()["job_id"]
    data = _wait_for_terminal(client, job_id)

    assert data["status"] == "failed"
    assert data["processed"] == 1
    assert data["rewrapped"] == 1
    assert data["failed"] == 1
    assert data["complete"] is False
    # The stopped envelope and everything after it are unchanged.
    assert _get_envelope(client, "a1").json()["key_version"] == 1
    assert _get_envelope(client, "a2").json()["key_version"] == 1


# --- concurrency -----------------------------------------------------------


def test_concurrent_jobs_advance_each_envelope_at_most_once(
    app, client, monkeypatch
):
    count = 8
    _seed(client, [f"e{n}" for n in range(count)])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: _submit(client, limit=50), range(8)))
    assert all(response.status_code == 202 for response in responses)
    settled = [
        _wait_for_terminal(client, response.json()["job_id"])
        for response in responses
    ]
    assert all(data["status"] == "succeeded" for data in settled)
    totals = {
        key: sum(data[key] for data in settled)
        for key in ("rewrapped", "skipped", "failed")
    }
    assert totals["rewrapped"] == count
    assert totals["failed"] == 0
    # Every envelope processed after a winning job observes the
    # already-current result: 8 jobs x 8 envelopes - 8 wins.
    assert totals["skipped"] == count * 8 - count

    for n in range(count):
        stored = _get_envelope(client, f"e{n}").json()
        assert stored["key_version"] == 2
        assert _unwrap_and_decrypt(KEY_V2_BYTES, stored) == PAYLOAD.encode("utf-8")


# --- job lookup ------------------------------------------------------------


def test_get_unknown_or_cross_scope_job_returns_404(client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client).json()["job_id"]
    _wait_for_terminal(client, job_id)

    unknown = "00000000-0000-0000-0000-000000000000"
    assert _get_job(client, unknown).status_code == 404
    # Syntactically valid uppercase UUID is normalized, then not found.
    assert (
        _get_job(client, "11111111-1111-1111-1111-111111111111").status_code == 404
    )
    # The actual id works even if the client uppercases it.
    assert _get_job(client, job_id.upper()).status_code == 200
    assert _get_job(client, job_id, tenant="tenant-b").status_code == 404
    assert _get_job(client, job_id, workload="workload-2").status_code == 404


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


def test_get_job_failure_returns_500_without_partial_progress(app, client):
    job_id = _submit(client).json()["job_id"]
    _wait_for_terminal(client, job_id)

    # Simulate a storage/service failure on the job read.
    from sqlalchemy import text

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_jobs"))
    response = _get_job(client, job_id)
    assert response.status_code == 500
    assert b"processed" not in response.content


# --- persistence, recovery, wire format ------------------------------------


def test_interrupted_job_resumes_after_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart-jobs.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, [f"r{n}" for n in range(4)])

    # Simulate a process interruption: a job left running with its resume
    # cursor parked at the scope beginning.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    now = datetime.now(timezone.utc)
    with app1.state.session_factory() as session:
        session.add(
            RewrapJob(
                job_id="11111111-1111-1111-1111-111111111111",
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                limit=2,
                cursor="",
                next_cursor="",
                status="running",
                complete=False,
                processed=0,
                rewrapped=0,
                skipped=0,
                failed=0,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    app1.state.engine.dispose()

    # The next process picks the interrupted job up and drives it to
    # completion from the parked cursor.
    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    with app2.state.session_factory() as session:
        job_id = session.query(RewrapJob).one().job_id
    data = _wait_for_terminal(client2, job_id)
    assert data["status"] == "succeeded"
    assert data["processed"] == 4
    assert data["rewrapped"] == 4
    assert data["failed"] == 0
    assert data["complete"] is True
    for n in range(4):
        stored = _get_envelope(client2, f"r{n}").json()
        assert stored["key_version"] == 2
        assert _unwrap_and_decrypt(KEY_V2_BYTES, stored) == PAYLOAD.encode("utf-8")
    app2.state.engine.dispose()


def test_jobs_survive_restart_and_stay_queryable(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart-jobs-2.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["x", "y"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    job_id = _submit(client1).json()["job_id"]
    settled = _wait_for_terminal(client1, job_id)
    assert settled["status"] == "succeeded"
    app1.state.engine.dispose()

    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    data = _get_job(client2, job_id).json()
    assert data["status"] == "succeeded"
    assert data["processed"] == 2
    assert data["rewrapped"] == 2
    assert data["complete"] is True
    app2.state.engine.dispose()


def test_job_responses_are_compact_json_with_single_newline_and_no_floats(
    client, monkeypatch
):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    submit = _submit(client)
    job_id = submit.json()["job_id"]
    settled = _wait_for_terminal(client, job_id)
    assert settled["status"] == "succeeded"

    response = _get_job(client, job_id)
    raw = response.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw
    assert b": " not in raw
    assert response.headers["content-type"] == "application/json"
    parsed = json.loads(raw)
    assert isinstance(parsed["job_id"], str)
    assert isinstance(parsed["status"], str)
    assert isinstance(parsed["next_cursor"], str)
    assert isinstance(parsed["complete"], bool)
    for key in ("processed", "rewrapped", "skipped", "failed"):
        assert isinstance(parsed[key], int) and not isinstance(parsed[key], bool)


def test_job_never_exposes_payload_or_key_material(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    submit = _submit(client)
    job_id = submit.json()["job_id"]
    _wait_for_terminal(client, job_id)
    for text in (submit.text, _get_job(client, job_id).text):
        for secret in (PAYLOAD, KEY_V1, KEY_V2, "wrapped_key", "ciphertext", "payload"):
            assert secret not in text
    with app.state.session_factory() as session:
        job = session.query(RewrapJob).one()
        assert not hasattr(job, "wrapped_key")
