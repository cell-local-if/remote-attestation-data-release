"""Tests for the optional Idempotency-Key header on POST /v1/rewrap-jobs.

A key is scoped to the request's tenant/workload. The first valid
submission under a key creates one durable job and saves its verbatim
202 response; identical retries replay that response byte-for-byte,
same-key requests with different content get a stable 409, and keys in
different scopes are independent. Like the job tests, the app is built
without its lifespan unless restart or concurrency behavior is under
test.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import RewrapJob, RewrapJobIdempotencyRecord

from proof_release.envelopes import b64url_encode
from tests.test_rewrap_jobs import (
    KEYRING_V1_V2,
    KEYRING_V2_ONLY,
    KEY_V1,
    PAYLOAD,
    TENANT,
    WORKLOAD,
)


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


def _seed(client, data_ids, *, tenant=TENANT, workload=WORKLOAD):
    for data_id in data_ids:
        response = client.post(
            "/v1/data-envelopes",
            json={
                "tenant_id": tenant,
                "workload_id": workload,
                "data_id": data_id,
                "payload": PAYLOAD,
            },
        )
        assert response.status_code == 201


def _submit(
    client,
    *,
    key=None,
    cursor=None,
    limit=None,
    tenant=TENANT,
    workload=WORKLOAD,
):
    body: dict = {"tenant_id": tenant, "workload_id": workload}
    if cursor is not None:
        body["cursor"] = cursor
    if limit is not None:
        body["limit"] = limit
    headers = {"Idempotency-Key": key} if key is not None else None
    return client.post("/v1/rewrap-jobs", json=body, headers=headers)


def _get_job(client, job_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/rewrap-jobs/{job_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _job_count(app) -> int:
    with app.state.session_factory() as session:
        return session.query(RewrapJob).count()


def _record_count(app) -> int:
    with app.state.session_factory() as session:
        return session.query(RewrapJobIdempotencyRecord).count()


# --- no key: legacy semantics ----------------------------------------------


def test_missing_key_creates_a_job_per_request(app, client):
    _seed(client, ["a", "b"])
    first = _submit(client)
    second = _submit(client)
    assert first.status_code == second.status_code == 202
    assert first.json()["job_id"] != second.json()["job_id"]
    assert _job_count(app) == 2
    assert _record_count(app) == 0


# --- replay: verbatim saved 202 --------------------------------------------


def test_replay_returns_saved_202_verbatim_and_creates_one_job(app, client):
    _seed(client, ["a", "b"])
    first = _submit(client, key="key-1", limit=7)
    assert first.status_code == 202
    job_id = first.json()["job_id"]

    replay = _submit(client, key="key-1", limit=7)
    assert replay.status_code == 202
    # Byte-for-byte identical: same job id, cursor and timestamps.
    assert replay.content == first.content
    assert replay.json()["job_id"] == job_id
    assert _job_count(app) == 1
    assert _record_count(app) == 1


def test_replay_after_job_runs_fails_and_succeeds_still_returns_original(
    app, client, monkeypatch
):
    # Seed both scopes while the keyring still has only the v1 master key,
    # so every envelope is sealed under version 1.
    _seed(client, ["a"])
    _seed(client, ["v1-only"], workload="workload-fail")
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _submit(client, key="run-key")
    assert first.status_code == 202
    original = first.content
    job_id = first.json()["job_id"]

    app.state.rewrap_job_runner.run(job_id)
    assert _get_job(client, job_id).json()["status"] == "succeeded"
    assert _submit(client, key="run-key").content == original

    # A failed job behaves the same: in its own scope the historical v1
    # key is missing, so the job fails parked at the first envelope.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    failed_first = _submit(
        client, key="fail-key", limit=10, workload="workload-fail"
    )
    failed_job = failed_first.json()["job_id"]
    app.state.rewrap_job_runner.run(failed_job)
    failed_view = _get_job(client, failed_job, workload="workload-fail")
    assert failed_view.json()["status"] == "failed"
    replay = _submit(client, key="fail-key", limit=10, workload="workload-fail")
    assert replay.status_code == 202
    assert replay.content == failed_first.content
    assert replay.json()["job_id"] == failed_job
    # Retries never advance the envelope cursor or counters.
    assert failed_view.json()["processed"] == 0
    assert (
        _get_job(client, failed_job, workload="workload-fail").json()["processed"]
        == 0
    )


def test_replay_does_not_recheck_keyring(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _submit(client, key="ring-key")
    assert first.status_code == 202
    # The keyring has since become wholly unusable; a replay must still
    # return the saved 202 rather than a 500.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    replay = _submit(client, key="ring-key")
    assert replay.status_code == 202
    assert replay.content == first.content


def test_replay_after_restart_returns_same_job(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart-idem.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    _seed(client1, ["a", "b"])
    first = _submit(client1, key="persist-key", limit=3)
    assert first.status_code == 202
    app1.state.engine.dispose()

    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    replay = _submit(client2, key="persist-key", limit=3)
    assert replay.status_code == 202
    assert replay.content == first.content
    with app2.state.session_factory() as session:
        assert session.query(RewrapJob).count() == 1
        assert session.query(RewrapJobIdempotencyRecord).count() == 1
    app2.state.engine.dispose()


# --- conflict: same key, different content ---------------------------------


def test_same_key_different_limit_returns_409_and_keeps_original(app, client):
    _seed(client, ["a", "b"])
    first = _submit(client, key="dup", limit=10)
    assert first.status_code == 202

    conflict = _submit(client, key="dup", limit=20)
    assert conflict.status_code == 409
    assert _job_count(app) == 1

    # The original request still replays, unchanged.
    assert _submit(client, key="dup", limit=10).content == first.content
    assert _job_count(app) == 1


def test_same_key_different_cursor_returns_409(app, client, monkeypatch):
    _seed(client, ["a", "b", "c"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _submit(client, key="cursor-key", limit=1)
    job_id = first.json()["job_id"]
    # Produce a real verified cursor by advancing one page.
    runner = app.state.rewrap_job_runner
    assert runner._claim(job_id)
    assert runner._advance_page(job_id) is None
    runner._release_claim(job_id)
    with app.state.session_factory() as session:
        advanced = session.get(RewrapJob, job_id).next_cursor
    assert advanced

    assert _submit(client, key="cursor-key", limit=1, cursor=advanced).status_code == 409
    # Original start (no cursor) still replays.
    assert _submit(client, key="cursor-key", limit=1).content == first.content


def test_absent_and_explicit_empty_cursor_are_the_same_request(app, client):
    _seed(client, ["a"])
    first = _submit(client, key="start-key")
    # An explicit empty string is the same normalized start position.
    replay = _submit(client, key="start-key", cursor="")
    assert replay.status_code == 202
    assert replay.content == first.content
    assert _job_count(app) == 1


def test_conflict_409_does_not_write_or_advance(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    first = _submit(client, key="k", limit=10)
    job_id = first.json()["job_id"]
    app.state.rewrap_job_runner.run(job_id)
    done = _get_job(client, job_id).json()

    # A conflicting replay is a 409 and leaves the job exactly where it was.
    assert _submit(client, key="k", limit=11).status_code == 409
    assert _get_job(client, job_id).json() == done
    assert _record_count(app) == 1


# --- scope independence -----------------------------------------------------


def test_same_key_in_other_scope_is_independent(app, client):
    _seed(client, ["a"], tenant=TENANT, workload=WORKLOAD)
    _seed(client, ["b"], tenant="tenant-b", workload=WORKLOAD)
    _seed(client, ["c"], tenant=TENANT, workload="workload-9")

    one = _submit(client, key="shared", tenant=TENANT, workload=WORKLOAD)
    two = _submit(client, key="shared", tenant="tenant-b", workload=WORKLOAD)
    three = _submit(client, key="shared", tenant=TENANT, workload="workload-9")
    assert one.status_code == two.status_code == three.status_code == 202
    ids = {one.json()["job_id"], two.json()["job_id"], three.json()["job_id"]}
    assert len(ids) == 3
    assert _job_count(app) == 3

    # Each scope replays its own job.
    assert (
        _submit(client, key="shared", tenant=TENANT, workload=WORKLOAD).json()["job_id"]
        == one.json()["job_id"]
    )
    assert (
        _submit(
            client, key="shared", tenant="tenant-b", workload=WORKLOAD
        ).json()["job_id"]
        == two.json()["job_id"]
    )


# --- header validation ------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "a b",
        "a\tb",
        "a\nb",
        "x" * 65,
        "key\x00",
        " leading",
        "trailing ",
    ],
)
def test_invalid_idempotency_key_returns_422(app, client, value):
    _seed(client, ["a"])
    response = _submit(client, key=value)
    assert response.status_code == 422
    assert _job_count(app) == 0
    assert _record_count(app) == 0


def test_non_ascii_idempotency_key_returns_422(app, client):
    # A non-ASCII byte must be rejected even though the HTTP client would
    # otherwise refuse to encode it: send the raw UTF-8 header bytes.
    _seed(client, ["a"])
    response = client.post(
        "/v1/rewrap-jobs",
        content=json.dumps({"tenant_id": TENANT, "workload_id": WORKLOAD}),
        headers=[
            (b"content-type", b"application/json"),
            (b"Idempotency-Key", "é-key".encode("utf-8")),
        ],
    )
    assert response.status_code == 422
    assert _job_count(app) == 0
    assert _record_count(app) == 0


def test_duplicate_idempotency_header_returns_422(app, client):
    _seed(client, ["a"])
    response = client.post(
        "/v1/rewrap-jobs",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers=[
            ("Idempotency-Key", "one"),
            ("Idempotency-Key", "two"),
        ],
    )
    assert response.status_code == 422
    assert _job_count(app) == 0


def test_idempotency_header_is_case_insensitive(app, client):
    _seed(client, ["a"])
    response = client.post(
        "/v1/rewrap-jobs",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        headers={"idempotency-key": "lowercase-key"},
    )
    assert response.status_code == 202
    replay = _submit(client, key="lowercase-key")
    assert replay.status_code == 202
    assert replay.json()["job_id"] == response.json()["job_id"]


def test_boundary_length_keys_are_accepted(app, client):
    _seed(client, ["a"])
    assert _submit(client, key="x").status_code == 202
    assert _submit(client, key="y" * 64).status_code == 202
    # Visible ASCII punctuation is accepted.
    assert _submit(client, key="~!@#$%^&*()_+-=[]{}|;:',.<>?/`").status_code == 202


def test_invalid_key_is_rejected_before_keyring_and_storage(app, client, monkeypatch):
    _seed(client, ["a"])
    # Even a broken keyring must not matter: the key is rejected first.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    response = _submit(client, key="bad key")
    assert response.status_code == 422
    assert _job_count(app) == 0
    assert _record_count(app) == 0


def test_valid_key_does_not_mask_an_invalid_body_field(app, client):
    _seed(client, ["a"])
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": 201}
    response = client.post(
        "/v1/rewrap-jobs", json=body, headers={"Idempotency-Key": "k"}
    )
    assert response.status_code == 422
    assert _job_count(app) == 0
    assert _record_count(app) == 0

    # A forged cursor with a valid key is still the cursor's 422 and
    # leaves no record, so a corrected retry may still use the key.
    forged = b64url_encode(b'{"t":"tenant-a","w":"workload-1","d":"x"}' + b"0" * 32)
    bad_cursor = client.post(
        "/v1/rewrap-jobs",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD, "cursor": forged},
        headers={"Idempotency-Key": "k"},
    )
    assert bad_cursor.status_code == 422
    assert _record_count(app) == 0


# --- server failure rollback ------------------------------------------------


def test_invalid_keyring_on_first_submit_returns_500_and_persists_nothing(
    app, client, monkeypatch
):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    response = _submit(client, key="doomed")
    assert response.status_code == 500
    assert _job_count(app) == 0
    assert _record_count(app) == 0

    # The key was never consumed: with a healthy keyring it now creates.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    retry = _submit(client, key="doomed")
    assert retry.status_code == 202
    assert _job_count(app) == 1


def test_job_write_failure_rolls_back_the_idempotency_record(app, client):
    from sqlalchemy import text

    _seed(client, ["a"])
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE rewrap_jobs"))
    response = _submit(client, key="write-fail")
    assert response.status_code == 500
    # The record is in the same transaction, so it is gone too.
    assert _record_count(app) == 0

    # Recover the table and reuse the key: it behaves as a first submit.
    from proof_release.db import Base

    Base.metadata.create_all(app.state.engine)
    retry = _submit(client, key="write-fail")
    assert retry.status_code == 202
    assert _job_count(app) == 1
    assert _record_count(app) == 1


# --- concurrency ------------------------------------------------------------


def test_concurrent_identical_keyed_submits_create_one_job(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/concurrent-idem.db"
    application = app_module.create_app(url)
    try:
        with TestClient(application) as client:
            _seed(client, [f"e{n}" for n in range(6)])
            monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
            with ThreadPoolExecutor(max_workers=8) as pool:
                responses = list(
                    pool.map(lambda _: _submit(client, key="race", limit=50), range(8))
                )
            assert all(r.status_code == 202 for r in responses), [
                r.status_code for r in responses
            ]
            job_ids = {r.json()["job_id"] for r in responses}
            assert job_ids == {responses[0].json()["job_id"]}
            assert len(job_ids) == 1
            with application.state.session_factory() as session:
                assert session.query(RewrapJob).count() == 1
                assert session.query(RewrapJobIdempotencyRecord).count() == 1
    finally:
        application.state.engine.dispose()


# --- secrecy ----------------------------------------------------------------


def test_saved_response_and_record_never_store_material(app, client, monkeypatch):
    _seed(client, ["a"])
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    accepted = _submit(client, key="secret-key")
    assert accepted.status_code == 202
    for secret in (PAYLOAD, KEY_V1, "wrapped_key", "ciphertext", "payload"):
        assert secret not in accepted.text
    with app.state.session_factory() as session:
        record = session.get(
            RewrapJobIdempotencyRecord, (TENANT, WORKLOAD, "secret-key")
        )
        saved = record.saved_response.decode("utf-8")
        assert set(json.loads(saved)) == {
            "job_id",
            "status",
            "limit",
            "cursor",
            "created_at",
            "updated_at",
        }
        for secret in (PAYLOAD, KEY_V1, "wrapped_key", "ciphertext"):
            assert secret not in saved
        assert record.request_fingerprint != "secret-key"
