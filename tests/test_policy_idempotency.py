"""Tests for the optional Idempotency-Key header on POST /v1/policies.

A keyed creation writes the policy version, its per-scope lifecycle
commit sequence and its idempotency record in one transaction and
replays the exact stored 201 afterwards; a missing key keeps the legacy
one-version-per-request behavior. The record persists only the scope,
the key, the normalized request fingerprint, the first response body
and the creation time — never evidence, claims, capabilities, keys or
exception text.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import Policy, PolicyCommitCounter, PolicyIdempotencyRecord

TENANT = "tenant-a"
WORKLOAD = "workload-1"

LEAF = {"claim": "measurement", "equals": "abc"}
OTHER_LEAF = {"claim": "tier", "equals": 2}

IDEMPOTENCY_HEADER = "Idempotency-Key"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/idem-policies.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, rule=LEAF, name="release", key=None, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "name": name,
        "rule": rule,
    }
    body.update(overrides)
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else None
    return client.post("/v1/policies", json=body, headers=headers)


def _count_rows(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


def _records(app):
    with app.state.session_factory() as session:
        return session.query(PolicyIdempotencyRecord).all()


def _scope_seq(app, tenant=TENANT, workload=WORKLOAD):
    with app.state.session_factory() as session:
        counter = session.get(PolicyCommitCounter, (tenant, workload))
        return None if counter is None else counter.last_seq


async def _asgi_call(application, path, headers, body: bytes):
    """Invoke the ASGI app directly with arbitrary raw header bytes.

    The HTTP test client refuses some header values (control characters,
    non-ASCII bytes) before they reach the app; driving the ASGI app
    lets the service's own header validation see them.
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


def _raw_create(app, body_obj, *, key_raw=b"key-1", duplicate_key=False):
    payload = json.dumps(body_obj).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"idempotency-key", key_raw),
    ]
    if duplicate_key:
        headers.append((b"idempotency-key", key_raw))
    return asyncio.run(_asgi_call(app, "/v1/policies", headers, payload))


def _body(**overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "name": "release",
        "rule": LEAF,
    }
    body.update(overrides)
    return body


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
    status, raw = _raw_create(app, _body(), key_raw=raw_key)
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    assert _count_rows(app, Policy) == 0
    assert _count_rows(app, PolicyIdempotencyRecord) == 0
    # No version number or commit sequence was consumed.
    assert _scope_seq(app) is None


def test_duplicate_idempotency_header_is_422(app):
    status, raw = _raw_create(app, _body(), duplicate_key=True)
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    assert _count_rows(app, Policy) == 0
    assert _count_rows(app, PolicyIdempotencyRecord) == 0


def test_duplicate_idempotency_header_through_client_is_422(client):
    response = client.post(
        "/v1/policies",
        json=_body(),
        headers=[
            (IDEMPOTENCY_HEADER, "key-1"),
            (IDEMPOTENCY_HEADER, "key-1"),
        ],
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid idempotency key"}
    assert _count_rows(client.app, Policy) == 0


@pytest.mark.parametrize("length", [1, 64])
def test_boundary_length_keys_accepted(app, client, length):
    response = _create(client, key="A" * length, name=f"policy-{length}")
    assert response.status_code == 201, response.text
    assert _count_rows(app, PolicyIdempotencyRecord) == 1


def test_visible_punctuation_key_accepted(client):
    response = _create(client, key="~!@#$%^&*()_+-=[]{}|;:',.<>/?`")
    assert response.status_code == 201, response.text


def test_invalid_key_is_not_consumed_and_later_valid_use_succeeds(app, client):
    assert _create(client, key="").status_code == 422
    assert _raw_create(app, _body(), key_raw=b"bad key")[0] == 422
    accepted = _create(client, key="key-1")
    assert accepted.status_code == 201
    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1


def test_invalid_rule_with_valid_key_is_422_and_key_stays_free(app, client):
    bad_bodies = [
        _body(rule={}),
        _body(rule={"claim": "measurement"}),
        _body(rule={"unknown": [LEAF]}),
        _body(rule={"all": []}),
        _body(tenant_id=""),
        _body(name="   "),
        {},
    ]
    for body in bad_bodies:
        response = client.post(
            "/v1/policies",
            json=body,
            headers={IDEMPOTENCY_HEADER: "key-1"},
        )
        assert response.status_code == 422, body
    assert _count_rows(app, Policy) == 0
    assert _count_rows(app, PolicyIdempotencyRecord) == 0
    assert _scope_seq(app) is None

    # The failed judgements wrote nothing: the same key succeeds later.
    created = _create(client, key="key-1")
    assert created.status_code == 201
    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1


# --- core replay semantics -------------------------------------------------


def test_first_keyed_create_writes_policy_and_record_atomically(app, client):
    response = _create(client, key="key-1")

    assert response.status_code == 201
    data = response.json()
    assert set(data) == {
        "policy_id",
        "tenant_id",
        "workload_id",
        "name",
        "version",
        "rule",
        "created_at",
    }
    assert data["version"] == 1
    assert data["rule"] == LEAF

    assert _count_rows(app, Policy) == 1
    records = _records(app)
    assert len(records) == 1
    record = records[0]
    assert record.tenant_id == TENANT
    assert record.workload_id == WORKLOAD
    assert record.idempotency_key == "key-1"
    assert len(record.request_fingerprint) == 64
    assert json.loads(record.response_body) == data
    # The stored body is the exact wire form of the first response.
    assert record.response_body.encode("utf-8") == response.content
    # The lifecycle sequence advanced exactly once, for this version.
    assert _scope_seq(app) == 1


def test_replay_returns_first_response_byte_for_byte(app, client):
    first = _create(client, key="key-1")
    assert first.status_code == 201

    replay = _create(client, key="key-1")
    assert replay.status_code == 201
    assert replay.content == first.content
    assert replay.json() == first.json()

    # No second version, no second record, no sequence advancement.
    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1
    assert _scope_seq(app) == 1


def test_replay_with_normalized_equivalent_rule_matches(app, client):
    rule = {"all": [LEAF, OTHER_LEAF]}
    first = _create(client, rule=rule, key="key-1")
    assert first.status_code == 201

    # Same tree with object keys in a different order: identical after
    # canonical normalization, so it is a replay, not a new version.
    reordered = {"all": [{"equals": "abc", "claim": "measurement"},
                         {"equals": 2, "claim": "tier"}]}
    replay = _create(client, rule=reordered, key="key-1")
    assert replay.status_code == 201
    assert replay.content == first.content

    assert _count_rows(app, Policy) == 1
    assert _scope_seq(app) == 1


def test_same_key_different_name_is_409_and_changes_nothing(app, client):
    first = _create(client, key="key-1")
    assert first.status_code == 201

    conflict = _create(client, key="key-1", name="other")
    assert conflict.status_code == 409
    assert conflict.json() == {
        "detail": "idempotency key reused with a different request"
    }

    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1
    # The original record still replays the original response.
    assert _create(client, key="key-1").content == first.content


def test_same_key_different_rule_is_409_and_changes_nothing(app, client):
    first = _create(client, key="key-1")
    assert first.status_code == 201

    conflict = _create(client, key="key-1", rule=OTHER_LEAF)
    assert conflict.status_code == 409
    assert conflict.json() == {
        "detail": "idempotency key reused with a different request"
    }

    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1
    assert _create(client, key="key-1").content == first.content


def test_same_key_is_independent_across_scopes(app, client):
    first = _create(client, key="key-1")
    assert first.status_code == 201

    other_tenant = _create(client, key="key-1", tenant_id="tenant-b")
    assert other_tenant.status_code == 201
    other_workload = _create(client, key="key-1", workload_id="workload-2")
    assert other_workload.status_code == 201

    # Three independent scopes, three versions, three records.
    assert _count_rows(app, Policy) == 3
    assert _count_rows(app, PolicyIdempotencyRecord) == 3

    # Within each scope the key replays its own first response only.
    assert _create(client, key="key-1").content == first.content
    assert _create(client, key="key-1", tenant_id="tenant-b").content == (
        other_tenant.content
    )
    # Cross-scope replay never leaks the other scope's response.
    assert other_tenant.content != first.content


def test_keyed_create_then_keyless_create_allocates_next_version(app, client):
    keyed = _create(client, key="key-1")
    assert keyed.status_code == 201
    assert keyed.json()["version"] == 1

    # A keyless creation of the same name still allocates a fresh version.
    keyless = _create(client, rule=OTHER_LEAF)
    assert keyless.status_code == 201
    assert keyless.json()["version"] == 2

    # The keyed replay still returns the stored first response and
    # creates nothing.
    assert _create(client, key="key-1").content == keyed.content
    assert _count_rows(app, Policy) == 2
    assert _count_rows(app, PolicyIdempotencyRecord) == 1
    assert _scope_seq(app) == 2


def test_concurrent_same_key_requests_create_exactly_one_version(app):
    body = _body()

    def submit(_):
        with TestClient(app) as thread_client:
            return thread_client.post(
                "/v1/policies",
                json=body,
                headers={IDEMPOTENCY_HEADER: "key-1"},
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(submit, range(8)))

    statuses = {response.status_code for response in responses}
    assert statuses == {201}
    bodies = {response.content for response in responses}
    assert len(bodies) == 1
    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1
    assert _scope_seq(app) == 1


def test_concurrent_same_key_different_requests_settle_one_success(app):
    def submit(name):
        with TestClient(app) as thread_client:
            return thread_client.post(
                "/v1/policies",
                json=_body(name=name),
                headers={IDEMPOTENCY_HEADER: "key-1"},
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(submit, [f"policy-{i}" for i in range(8)]))

    successes = [r for r in responses if r.status_code == 201]
    conflicts = [r for r in responses if r.status_code == 409]
    # At most one request wins the key; every loser gets the stable 409.
    assert len(successes) == 1
    assert len(conflicts) == len(responses) - 1
    for conflict in conflicts:
        assert conflict.json() == {
            "detail": "idempotency key reused with a different request"
        }
    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1


# --- failure atomicity -----------------------------------------------------


def test_persistence_failure_rolls_back_and_key_stays_free(
    app, client, monkeypatch
):
    def _failing_seq(session, tenant_id, workload_id):
        raise RuntimeError("simulated storage failure")

    monkeypatch.setattr(app_module, "_next_policy_commit_seq", _failing_seq)
    failed = _create(client, key="key-1")
    assert failed.status_code == 500
    assert failed.json() == {"detail": "policy write failed"}
    # No half policy, no half record, no consumed sequence.
    assert _count_rows(app, Policy) == 0
    assert _count_rows(app, PolicyIdempotencyRecord) == 0
    assert _scope_seq(app) is None

    # After recovery the same key is judged fresh and succeeds.
    monkeypatch.undo()
    created = _create(client, key="key-1")
    assert created.status_code == 201
    assert created.json()["version"] == 1
    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1


# --- durability and read-side isolation ------------------------------------


def test_record_survives_restart_and_replays(tmp_path):
    db_url = f"sqlite:///{tmp_path}/idem-policies-restart.db"
    app_one = create_app(db_url)
    first = _create(TestClient(app_one), key="key-1")
    assert first.status_code == 201
    app_one.state.engine.dispose()

    app_two = create_app(db_url)
    try:
        replay = _create(TestClient(app_two), key="key-1")
        assert replay.status_code == 201
        assert replay.content == first.content
        assert _count_rows(app_two, Policy) == 1
        assert _count_rows(app_two, PolicyIdempotencyRecord) == 1

        conflict = _create(TestClient(app_two), key="key-1", name="other")
        assert conflict.status_code == 409
    finally:
        app_two.state.engine.dispose()


def test_policy_listing_unaffected_by_keyed_create(app, client):
    created = _create(client, key="key-1")
    assert created.status_code == 201

    listing = client.get(
        "/v1/policies",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert listing.status_code == 200
    policies = listing.json()["policies"]
    assert len(policies) == 1
    # Only the established public fields are exposed; the idempotency
    # record never appears.
    assert set(policies[0]) == {
        "policy_id",
        "tenant_id",
        "workload_id",
        "name",
        "version",
        "rule",
        "status",
        "created_at",
        "retired_at",
    }

    # A replay does not add a listing entry.
    assert _create(client, key="key-1").status_code == 201
    listing = client.get(
        "/v1/policies",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert len(listing.json()["policies"]) == 1


def test_record_stores_only_normalized_request_identity(app, client):
    response = _create(client, key="key-1")
    assert response.status_code == 201

    (record,) = _records(app)
    # The record carries exactly its own identity, the scope, the key,
    # the fingerprint, the first response and the creation time.
    assert json.loads(record.response_body) == response.json()
    assert record.response_body.encode("utf-8") == response.content
    # The fingerprint is an irreversible digest, not the request itself.
    assert record.request_fingerprint != ""
    assert "measurement" not in record.request_fingerprint
    assert "abc" not in record.request_fingerprint


# --- keyless behavior is unchanged -----------------------------------------


def test_missing_header_keeps_legacy_semantics(app, client):
    first = _create(client)
    assert first.status_code == 201
    assert first.json()["version"] == 1

    # Every keyless legal request creates a new version of the name.
    second = _create(client)
    assert second.status_code == 201
    assert second.json()["version"] == 2
    assert second.json()["policy_id"] != first.json()["policy_id"]

    # No idempotency record is ever written for keyless requests.
    assert _count_rows(app, Policy) == 2
    assert _count_rows(app, PolicyIdempotencyRecord) == 0
    assert _scope_seq(app) == 2
