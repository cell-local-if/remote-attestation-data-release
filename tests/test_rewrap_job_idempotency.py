"""Tests for the optional Idempotency-Key header on asynchronous rewrap
job submission (POST /v1/rewrap-jobs).

A keyed submission creates one persistent job per (tenant, workload,
key) and replays the exact stored 202 afterwards; a missing key keeps
the legacy one-job-per-request behavior. As in test_rewrap_jobs.py the
default app is built without entering its lifespan so no pool runs;
restart and live-concurrency checks create dedicated apps.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import AuditEvent, RewrapJob, RewrapJobIdempotencyRecord
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
PAYLOAD = "idempotent-secret-payload"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V2_BYTES = b"fedcba9876543210fedcba9876543210"
KEY_V1 = b64url_encode(KEY_V1_BYTES)
KEY_V2 = b64url_encode(KEY_V2_BYTES)

IDEMPOTENCY_HEADER = "Idempotency-Key"


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


KEYRING_V1_V2 = _keyring(2, {1: KEY_V1, 2: KEY_V2})


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = app_module.create_app(f"sqlite:///{tmp_path}/idem.db")
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


def _submit(
    client,
    *,
    cursor=None,
    limit=None,
    tenant=TENANT,
    workload=WORKLOAD,
    key=None,
):
    body: dict = {"tenant_id": tenant, "workload_id": workload}
    if cursor is not None:
        body["cursor"] = cursor
    if limit is not None:
        body["limit"] = limit
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else None
    return client.post("/v1/rewrap-jobs", json=body, headers=headers)


def _run_job(app, job_id):
    app.state.rewrap_job_runner.run(job_id)


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


async def _asgi_call(application, path, headers, body: bytes):
    """Invoke the ASGI app directly with arbitrary raw header bytes.

    The HTTP test client refuses some header values (control characters,
    # non-ASCII bytes) before they reach the app; driving the ASGI app
    # lets the service's own header validation see them.
    """
    chunks: list[bytes] = []
    status: dict[str, int] = {}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            status["code"] = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }
    await application(scope, receive, send)
    return status["code"], b"".join(chunks)


def _raw_submit(app, body_obj, *, key_raw=b"key-1", duplicate_key=False):
    payload = json.dumps(body_obj).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"idempotency-key", key_raw),
    ]
    if duplicate_key:
        headers.append((b"idempotency-key", key_raw))
    return asyncio.run(_asgi_call(app, "/v1/rewrap-jobs", headers, payload))


def _count_rows(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


# --- header validation -----------------------------------------------------


@pytest.mark.parametrize(
    "raw_key",
    [
        b"",  # empty value
        b" ",  # bare whitespace
        b" key-1",  # leading space
        b"key-1 ",  # trailing space
        b"key 1",  # embedded space
        b"key\t1",  # tab
        b"key-1\n",  # control character
        b"k\xc3\xa9y",  # non-ASCII (UTF-8 e-acute)
        b"k" * 65,  # over-long
    ],
)
def test_invalid_idempotency_key_is_422_before_any_write(app, raw_key):
    status, raw = _raw_submit(
        app, {"tenant_id": TENANT, "workload_id": WORKLOAD}, key_raw=raw_key
    )
    assert status == 422, raw
    assert _count_rows(app, RewrapJob) == 0
    assert _count_rows(app, RewrapJobIdempotencyRecord) == 0
    assert _count_rows(app, AuditEvent) == 0


def test_duplicate_idempotency_header_is_422(app):
    status, raw = _raw_submit(
        app,
        {"tenant_id": TENANT, "workload_id": WORKLOAD},
        key_raw=b"key-1",
        duplicate_key=True,
    )
    assert status == 422, raw
    assert _count_rows(app, RewrapJob) == 0
    assert _count_rows(app, RewrapJobIdempotencyRecord) == 0


def test_duplicate_idempotency_header_through_client_is_422(client):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    response = client.post(
        "/v1/rewrap-jobs",
        json=body,
        headers=[
            (IDEMPOTENCY_HEADER, "key-1"),
            (IDEMPOTENCY_HEADER, "key-1"),
        ],
    )
    assert response.status_code == 422
    assert _count_rows(client.app, RewrapJob) == 0


def test_empty_idempotency_header_through_client_is_422(client):
    response = _submit(client, key="")
    assert response.status_code == 422
    assert _count_rows(app=client.app, model=RewrapJob) == 0


@pytest.mark.parametrize("length", [1, 64])
def test_boundary_length_keys_accepted(app, client, monkeypatch, length):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    key = "A" * length
    response = _submit(client, key=key)
    assert response.status_code == 202, response.text
    assert _count_rows(app, RewrapJobIdempotencyRecord) == 1


def test_visible_punctuation_keys_accepted(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _submit(client, key="~!@#$%^&*()_+-=[]{}|;:',.<>/?`")
    assert response.status_code == 202, response.text


def test_invalid_key_is_not_consumed_and_later_valid_use_succeeds(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _submit(client, key="").status_code == 422
    assert _raw_submit(
        app, {"tenant_id": TENANT, "workload_id": WORKLOAD}, key_raw=b"bad key"
    )[0] == 422
    accepted = _submit(client, key="key-1")
    assert accepted.status_code == 202
    assert _count_rows(app, RewrapJob) == 1


def test_body_validation_still_fails_with_valid_idempotency_key(client):
    bad_bodies = [
        {"workload_id": WORKLOAD, "limit": 50},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": 0},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": 201},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": "10"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": True},
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": 7, "workload_id": WORKLOAD},
        {},
    ]
    for body in bad_bodies:
        response = client.post(
            "/v1/rewrap-jobs",
            json=body,
            headers={IDEMPOTENCY_HEADER: "key-1"},
        )
        assert response.status_code == 422, body
    assert _count_rows(client.app, RewrapJob) == 0
    assert _count_rows(client.app, RewrapJobIdempotencyRecord) == 0


def test_body_validation_cannot_be_masked_by_duplicate_header(app, client):
    # Both faults are 422-class; the bad limit must surface as 422 and
    # create nothing even though the header is itself illegal.
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": 999}
    payload = json.dumps(body).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"idempotency-key", b"one"),
        (b"idempotency-key", b"two"),
    ]
    status, _ = asyncio.run(
        _asgi_call(app, "/v1/rewrap-jobs", headers, payload)
    )
    assert status == 422
    assert _count_rows(app, RewrapJob) == 0
    assert _count_rows(app, RewrapJobIdempotencyRecord) == 0


def test_invalid_cursor_is_422_even_with_key_and_bad_keyring(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    response = client.post(
        "/v1/rewrap-jobs",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "cursor": "!!!not-a-cursor",
        },
        headers={IDEMPOTENCY_HEADER: "key-1"},
    )
    assert response.status_code == 422
    assert _count_rows(app, RewrapJob) == 0


def test_invalid_key_rejected_before_keyring_check(app, client, monkeypatch):
    # An illegal header must be a 422 even when the keyring is unusable:
    # header validation precedes the keyring check and writes nothing.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    response = _submit(client, key="")
    assert response.status_code == 422
    assert _raw_submit(
        app, {"tenant_id": TENANT, "workload_id": WORKLOAD}, key_raw=b"x" * 65
    )[0] == 422
    assert _count_rows(app, RewrapJob) == 0
    assert _count_rows(app, RewrapJobIdempotencyRecord) == 0


# --- core replay semantics -------------------------------------------------


def test_first_keyed_submit_creates_one_job_and_persists_record(
    app, client, monkeypatch
):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _submit(client, key="job-key-1", limit=7)
    assert response.status_code == 202
    data = response.json()
    assert list(data) == [
        "job_id",
        "status",
        "limit",
        "cursor",
        "created_at",
        "updated_at",
    ]
    assert data["status"] == "queued"
    assert data["limit"] == 7
    assert data["cursor"] == ""
    assert response.content.endswith(b"\n") and not response.content.endswith(b"\n\n")
    assert _count_rows(app, RewrapJob) == 1
    with app.state.session_factory() as session:
        record = session.query(RewrapJobIdempotencyRecord).one()
        assert record.tenant_id == TENANT
        assert record.workload_id == WORKLOAD
        assert record.idempotency_key == "job-key-1"
        assert record.job_id == data["job_id"]
        assert len(record.request_fingerprint) == 64
        assert record.response_body.encode() == response.content


def test_replay_returns_saved_202_verbatim_and_creates_nothing(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _submit(client, key="job-key-1", limit=10)
    assert first.status_code == 202

    second = _submit(client, key="job-key-1", limit=10)
    third = _submit(client, key="job-key-1", limit=10)
    assert second.status_code == 202 and third.status_code == 202
    # Byte-for-byte identical: same job id, cursor and original times.
    assert second.content == first.content
    assert third.content == first.content
    assert _count_rows(app, RewrapJob) == 1
    assert _count_rows(app, RewrapJobIdempotencyRecord) == 1


def test_replay_after_job_succeeded_still_returns_original_queued_202(
    app, client, monkeypatch
):
    _seed(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    accepted = _submit(client, key="job-key-1")
    original = accepted.content
    job_id = accepted.json()["job_id"]
    _run_job(app, job_id)
    final = _get_job(client, job_id).json()
    assert final["status"] == "succeeded"
    assert final["processed"] == 3

    replay = _submit(client, key="job-key-1")
    assert replay.status_code == 202
    # The saved acceptance predates advancement: queued, original cursor,
    # original timestamps — even though the job is now finished.
    assert replay.content == original
    assert replay.json()["job_id"] == job_id
    assert replay.json()["status"] == "queued"
    assert replay.json()["created_at"] == final["created_at"]
    # The replay neither created a job nor moved the finished job again.
    assert _count_rows(app, RewrapJob) == 1
    assert _get_job(client, job_id).json()["processed"] == 3


def test_replay_after_job_failed_still_returns_original_202(
    app, client, monkeypatch
):
    _seed(client, ["a0", "a1"])
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(2, {2: KEY_V2})
    )  # historical v1 missing
    accepted = _submit(client, key="job-key-1", limit=10)
    original = accepted.content
    job_id = accepted.json()["job_id"]
    _run_job(app, job_id)
    assert _get_job(client, job_id).json()["status"] == "failed"

    replay = _submit(client, key="job-key-1", limit=10)
    assert replay.status_code == 202
    assert replay.content == original
    assert replay.json()["job_id"] == job_id
    assert _count_rows(app, RewrapJob) == 1


def test_replay_does_not_re_enqueue_or_advance(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    accepted = _submit(client, key="job-key-1")
    job_id = accepted.json()["job_id"]
    _run_job(app, job_id)
    calls: list[str] = []
    real_submit = app.state.rewrap_job_runner.submit
    app.state.rewrap_job_runner.submit = lambda jid, **kw: calls.append(jid)
    try:
        replay = _submit(client, key="job-key-1")
    finally:
        app.state.rewrap_job_runner.submit = real_submit
    assert replay.status_code == 202
    assert calls == []
    assert _get_job(client, job_id).json()["status"] == "succeeded"


def test_same_key_different_limit_is_409_and_changes_nothing(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _submit(client, key="dup", limit=50)
    assert first.status_code == 202

    conflict = _submit(client, key="dup", limit=51)
    assert conflict.status_code == 409
    # Original job and saved response are untouched; exact replay works.
    assert _submit(client, key="dup", limit=50).content == first.content
    assert _count_rows(app, RewrapJob) == 1
    assert _count_rows(app, RewrapJobIdempotencyRecord) == 1


def test_same_key_different_cursor_is_409(app, client, monkeypatch):
    _seed(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _submit(client, key="dup", limit=1)
    job_id = first.json()["job_id"]
    _advance_one_page(app, job_id)
    with app.state.session_factory() as session:
        advanced_cursor = session.get(RewrapJob, job_id).next_cursor
    assert advanced_cursor

    assert _submit(client, key="dup", limit=1, cursor=advanced_cursor).status_code == 409
    # Replaying the original start remains a verbatim 202.
    assert _submit(client, key="dup", limit=1).content == first.content


def test_409_takes_precedence_over_later_keyring_failure(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _submit(client, key="dup", limit=50).status_code == 202
    # Keyring later becomes unusable; a mismatched replay is still a 409
    # (a stored-record judgement never re-checks the keyring)...
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    conflict = _submit(client, key="dup", limit=77)
    assert conflict.status_code == 409
    # ...and an exact replay is still the verbatim 202, never a 500.
    replay = _submit(client, key="dup", limit=50)
    assert replay.status_code == 202


# --- scope isolation -------------------------------------------------------


def test_same_key_in_other_tenant_or_workload_is_independent(
    app, client, monkeypatch
):
    _seed(client, ["a"], tenant=TENANT, workload=WORKLOAD)
    _seed(client, ["a"], tenant="tenant-b", workload=WORKLOAD)
    _seed(client, ["a"], tenant=TENANT, workload="workload-9")
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    r1 = _submit(client, key="shared-key")
    r2 = _submit(client, key="shared-key", tenant="tenant-b")
    r3 = _submit(client, key="shared-key", workload="workload-9")
    assert {r.status_code for r in (r1, r2, r3)} == {202}
    job_ids = {r.json()["job_id"] for r in (r1, r2, r3)}
    assert len(job_ids) == 3
    assert _count_rows(app, RewrapJob) == 3
    assert _count_rows(app, RewrapJobIdempotencyRecord) == 3
    # Replay in each scope resolves to that scope's own job.
    assert _submit(client, key="shared-key").json()["job_id"] == r1.json()["job_id"]
    assert (
        _submit(client, key="shared-key", tenant="tenant-b").json()["job_id"]
        == r2.json()["job_id"]
    )
    assert (
        _submit(client, key="shared-key", workload="workload-9").json()["job_id"]
        == r3.json()["job_id"]
    )


# --- cursor normalization --------------------------------------------------


def test_omitted_and_explicit_empty_cursor_share_one_job(app, client, monkeypatch):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _submit(client, key="start-key", limit=10)
    # No cursor field on the first request; an explicit empty string on
    # the replay is the same normalized start.
    replay = _submit(client, key="start-key", limit=10, cursor="")
    assert replay.status_code == 202
    assert replay.content == first.content
    assert _count_rows(app, RewrapJob) == 1

    # The reverse order normalizes identically as well.
    other = _submit(client, key="start-key-2", limit=10, cursor="")
    assert (
        _submit(client, key="start-key-2", limit=10).content == other.content
    )
    assert _count_rows(app, RewrapJob) == 2


# --- missing key preserves legacy semantics --------------------------------


def test_missing_key_creates_one_job_per_request(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _submit(client)
    second = _submit(client)
    assert first.status_code == 202 and second.status_code == 202
    assert first.json()["job_id"] != second.json()["job_id"]
    assert _count_rows(app, RewrapJob) == 2
    assert _count_rows(app, RewrapJobIdempotencyRecord) == 0


# --- failure rollback ------------------------------------------------------


def test_invalid_keyring_with_valid_key_returns_500_and_rolls_back(
    app, client, monkeypatch
):
    _seed(client, ["a", "b"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    response = _submit(client, key="job-key-1")
    assert response.status_code == 500
    assert _count_rows(app, RewrapJob) == 0
    assert _count_rows(app, RewrapJobIdempotencyRecord) == 0

    # The failed attempt consumed no key: once the keyring is fixed the
    # same request accepts normally.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    accepted = _submit(client, key="job-key-1")
    assert accepted.status_code == 202
    assert _count_rows(app, RewrapJob) == 1
    assert _count_rows(app, RewrapJobIdempotencyRecord) == 1


def test_job_write_failure_rolls_back_idempotency_record(app, client):
    from sqlalchemy import text

    _seed(client, ["a", "b"])
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_jobs"))
    response = _submit(client, key="job-key-1")
    assert response.status_code == 500
    # No half state: a record must never exist without its job.
    assert _count_rows(app, RewrapJobIdempotencyRecord) == 0


def test_idempotency_table_write_failure_returns_500_with_no_job(app, client):
    from sqlalchemy import text

    _seed(client, ["a", "b"])
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_job_idempotency_records"))
    response = _submit(client, key="job-key-1")
    assert response.status_code == 500
    assert _count_rows(app, RewrapJob) == 0
    # A keyless request is unaffected by the missing table.
    assert _submit(client).status_code == 202


# --- concurrency -----------------------------------------------------------


def test_concurrent_same_key_submissions_create_one_job(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/concurrent-idem.db"
    application = app_module.create_app(url)
    with TestClient(application) as client:
        _seed(client, [f"e{n}" for n in range(6)])
        monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(
                pool.map(lambda _: _submit(client, key="race-key"), range(8))
            )
        assert all(r.status_code == 202 for r in responses), [
            r.status_code for r in responses
        ]
        bodies = {r.content for r in responses}
        assert len(bodies) == 1
        job_id = responses[0].json()["job_id"]

        def _terminal():
            last = None
            import time

            for _ in range(200):
                last = _get_job(client, job_id).json()
                if last["status"] in ("succeeded", "failed"):
                    return last
                time.sleep(0.01)
            raise AssertionError(last)

        final = _terminal()
        assert final["status"] == "succeeded"
        assert final["rewrapped"] == 6
        assert final["processed"] == 6
        with application.state.session_factory() as session:
            assert session.query(RewrapJob).count() == 1
            assert session.query(RewrapJobIdempotencyRecord).count() == 1
            # Exactly one rewrapped audit event per envelope.
            assert (
                session.query(AuditEvent)
                .filter(
                    AuditEvent.event_type == "rewrap",
                    AuditEvent.status == "rewrapped",
                )
                .count()
                == 6
            )
    application.state.engine.dispose()


# --- restart persistence ---------------------------------------------------


def test_idempotency_record_survives_restart_and_replays_original(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart-idem.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["a", "b"])
    app1.state.engine.dispose()

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    accepted = _submit(client2, key="durable-key", limit=3)
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    app2.state.rewrap_job_runner.run(job_id)
    assert _get_job(client2, job_id).json()["status"] == "succeeded"
    app2.state.engine.dispose()

    # A fresh process (even with the keyring later altered) still has the
    # record and answers replays with the original 202.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    app3 = app_module.create_app(url)
    client3 = TestClient(app3)
    replay = _submit(client3, key="durable-key", limit=3)
    assert replay.status_code == 202
    assert replay.content == accepted.content
    assert replay.json()["job_id"] == job_id
    assert replay.json()["status"] == "queued"
    with app3.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 1
        assert session.query(RewrapJobIdempotencyRecord).count() == 1
    app3.state.engine.dispose()


# --- secrecy ---------------------------------------------------------------


def test_keyed_path_never_exposes_material(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    accepted = _submit(client, key="key-1")
    replay = _submit(client, key="key-1")
    for response in (accepted, replay):
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
        record = session.query(RewrapJobIdempotencyRecord).one()
        for secret in (PAYLOAD, KEY_V1, KEY_V2):
            assert secret not in record.response_body
            assert secret not in record.request_fingerprint
        # The fingerprint is a digest: the cursor/limit shape cannot be
        # read back out of it, and no raw header value beyond the key is
        # stored on the record.
        assert not hasattr(record, "payload")
