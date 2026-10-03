"""Tests for the optional Idempotency-Key header on POST /v1/policies.

A keyed creation writes the policy version, its lifecycle commit
sequence and the idempotency record in one transaction and replays the
exact stored 201 afterwards; a missing key keeps the legacy
one-new-version-per-request behavior. The record persists only the
scope, the key, the normalized request fingerprint (canonical name
plus canonical normalized rule tree), the first response body and the
creation time — never evidence, claim values, capabilities, keys or
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
from proof_release.db import (
    Policy,
    PolicyCommitCounter,
    PolicyIdempotencyRecord,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
NAME = "release"

LEAF = {"claim": "measurement", "equals": "abc"}
LEAF_REORDERED = {"equals": "abc", "claim": "measurement"}
OTHER_LEAF = {"claim": "measurement", "equals": "xyz"}

IDEMPOTENCY_HEADER = "Idempotency-Key"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/idem-policies.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(
    client,
    *,
    tenant=TENANT,
    workload=WORKLOAD,
    name=NAME,
    rule=LEAF,
    key=None,
):
    body = {
        "tenant_id": tenant,
        "workload_id": workload,
        "name": name,
        "rule": rule,
    }
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else None
    return client.post("/v1/policies", json=body, headers=headers)


def _body(**overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "name": NAME,
        "rule": LEAF,
    }
    body.update(overrides)
    return body


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
    response = _create(client, key="A" * length)
    assert response.status_code == 201, response.text
    assert _count_rows(app, PolicyIdempotencyRecord) == 1


def test_visible_punctuation_key_accepted(client):
    response = _create(client, key="~!@#$%^&*()_+-=[]{}|;:',.<>/?`")
    assert response.status_code == 201, response.text


def test_invalid_key_is_not_consumed_and_later_valid_use_succeeds(app, client):
    status, _raw = _raw_create(app, _body(), key_raw=b"")
    assert status == 422
    assert _raw_create(app, _body(), key_raw=b"bad key")[0] == 422
    accepted = _create(client, key="key-1")
    assert accepted.status_code == 201
    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1


def test_invalid_rule_with_valid_key_is_422_and_writes_nothing(app, client):
    response = client.post(
        "/v1/policies",
        json=_body(rule={"claim": "measurement", "equals": []}),
        headers={IDEMPOTENCY_HEADER: "key-1"},
    )
    assert response.status_code == 422
    assert _count_rows(app, Policy) == 0
    assert _count_rows(app, PolicyIdempotencyRecord) == 0
    # The key was never consumed: the same key now succeeds.
    accepted = _create(client, key="key-1")
    assert accepted.status_code == 201
    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1


def test_body_validation_still_fails_with_valid_idempotency_key(client):
    bad_bodies = [
        {"workload_id": WORKLOAD, "name": NAME, "rule": LEAF},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "name": NAME},
        _body(tenant_id=""),
        _body(tenant_id="   "),
        _body(workload_id=""),
        _body(name=""),
        _body(tenant_id=7),
        {},
    ]
    for body in bad_bodies:
        response = client.post(
            "/v1/policies",
            json=body,
            headers={IDEMPOTENCY_HEADER: "key-1"},
        )
        assert response.status_code == 422, body
    assert _count_rows(client.app, Policy) == 0
    assert _count_rows(client.app, PolicyIdempotencyRecord) == 0


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
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["name"] == NAME
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


def test_replay_returns_first_response_byte_for_byte(app, client):
    first = _create(client, key="key-1")
    assert first.status_code == 201

    replay = _create(client, key="key-1")
    assert replay.status_code == 201
    assert replay.content == first.content
    assert replay.json() == first.json()

    # No second policy, no second record, no version or sequence change.
    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1
    assert _scope_seq(app) == 1


def test_replay_preserves_original_policy_id_version_and_created_at(
    app, client, monkeypatch
):
    first = _create(client, key="key-1")
    assert first.status_code == 201
    first_data = first.json()

    # A later, unkeyed creation of the same name advances to version 2 and
    # the lifecycle sequence; the keyed replay must still answer v1.
    later = _create(client)
    assert later.status_code == 201
    assert later.json()["version"] == 2

    replay = _create(client, key="key-1")
    assert replay.status_code == 201
    replayed = replay.json()
    assert replayed["policy_id"] == first_data["policy_id"]
    assert replayed["version"] == 1
    assert replayed["created_at"] == first_data["created_at"]
    assert replay.content == first.content

    assert _count_rows(app, Policy) == 2
    assert _count_rows(app, PolicyIdempotencyRecord) == 1
    assert _scope_seq(app) == 2


def test_canonically_equivalent_rule_replays_first_response(app, client):
    # Key ordering inside a rule node is normalized away by the existing
    # canonical serialization, so the requests carry the same normalized
    # rule tree; the first response bytes (with the first rule spelling)
    # are what comes back.
    first = _create(client, rule=LEAF, key="key-1")
    assert first.status_code == 201

    replay = _create(client, rule=LEAF_REORDERED, key="key-1")
    assert replay.status_code == 201
    assert replay.content == first.content
    assert replay.json()["rule"] == LEAF

    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1


def test_same_key_different_rule_is_409_and_changes_nothing(app, client):
    first = _create(client, rule=LEAF, key="key-1")
    assert first.status_code == 201

    conflict = _create(client, rule=OTHER_LEAF, key="key-1")
    assert conflict.status_code == 409
    assert conflict.json() == {
        "detail": "idempotency key reused with a different request"
    }

    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1
    # The failed conflict consumed no version or lifecycle sequence.
    assert _scope_seq(app) == 1
    # The original record still replays the original response.
    assert _create(client, rule=LEAF, key="key-1").content == first.content


def test_same_key_different_name_is_409_and_changes_nothing(app, client):
    first = _create(client, name="release", key="key-1")
    assert first.status_code == 201

    conflict = _create(client, name="different", key="key-1")
    assert conflict.status_code == 409
    assert conflict.json() == {
        "detail": "idempotency key reused with a different request"
    }

    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1
    with app.state.session_factory() as session:
        rows = session.query(Policy).all()
        assert [(r.name, r.version) for r in rows] == [("release", 1)]
    # The original record still replays.
    assert (
        _create(client, name="release", key="key-1").content == first.content
    )


def test_same_key_in_other_scopes_is_independent(app, client):
    first = _create(client, key="key-1")
    assert first.status_code == 201

    other_tenant = _create(client, key="key-1", tenant="tenant-b")
    assert other_tenant.status_code == 201
    other_workload = _create(client, key="key-1", workload="workload-2")
    assert other_workload.status_code == 201

    # Three independent scopes, three policies, three records.
    assert _count_rows(app, Policy) == 3
    assert _count_rows(app, PolicyIdempotencyRecord) == 3

    # Within each scope the key replays its own first response only.
    assert _create(client, key="key-1").content == first.content
    assert _create(client, key="key-1", tenant="tenant-b").content == (
        other_tenant.content
    )
    # Cross-scope replay never leaks the other scope's response.
    assert other_tenant.content != first.content


def test_different_keys_same_name_create_independent_versions(app, client):
    first = _create(client, key="key-1")
    assert first.json()["version"] == 1
    second = _create(client, key="key-2")
    assert second.status_code == 201
    assert second.json()["version"] == 2
    assert second.json()["policy_id"] != first.json()["policy_id"]

    # Each key keeps replaying its own original version.
    assert _create(client, key="key-1").content == first.content
    assert _create(client, key="key-2").content == second.content
    assert _count_rows(app, Policy) == 2
    assert _count_rows(app, PolicyIdempotencyRecord) == 2


def test_failed_conflict_consumes_no_version_for_later_key(app, client):
    assert _create(client, key="key-1").status_code == 201
    # A different request reusing the key is a 409 and allocates nothing.
    assert _create(client, rule=OTHER_LEAF, key="key-1").status_code == 409

    # A fresh business intent with a new key on the same name is version 2,
    # never version 3: the conflict consumed no version number.
    second = _create(client, key="key-2")
    assert second.status_code == 201
    assert second.json()["version"] == 2
    assert _scope_seq(app) == 2


# --- concurrency -----------------------------------------------------------


def test_concurrent_same_key_same_request_create_exactly_one_version(app):
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

    assert {r.status_code for r in responses} == {201}
    bodies = {r.content for r in responses}
    assert len(bodies) == 1
    assert {r.json()["version"] for r in responses} == {1}
    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1
    assert _scope_seq(app) == 1


def test_concurrent_different_requests_same_key_one_wins_rest_409(app):
    # Eight genuinely different requests (distinct rule trees) competing
    # for one key: at most one can create, the rest settle as a stable
    # 409. Same-request concurrency (all 201, one version) is covered
    # separately.
    distinct_rules = [
        {"claim": f"trait-{index}", "equals": index} for index in range(8)
    ]

    def submit(index):
        with TestClient(app) as thread_client:
            return thread_client.post(
                "/v1/policies",
                json=_body(rule=distinct_rules[index]),
                headers={IDEMPOTENCY_HEADER: "race-key"},
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(submit, range(8)))

    successes = [r for r in responses if r.status_code == 201]
    conflicts = [r for r in responses if r.status_code == 409]
    assert len(successes) == 1
    assert len(conflicts) == 7
    assert all(
        r.json() == {"detail": "idempotency key reused with a different request"}
        for r in conflicts
    )
    assert _count_rows(app, Policy) == 1
    assert _count_rows(app, PolicyIdempotencyRecord) == 1
    assert successes[0].json()["version"] == 1
    # A sequential replay of the winning request returns the same bytes.
    with TestClient(app) as thread_client:
        winning_rule = successes[0].json()["rule"]
        replay = thread_client.post(
            "/v1/policies",
            json=_body(rule=winning_rule),
            headers={IDEMPOTENCY_HEADER: "race-key"},
        )
    assert replay.status_code == 201
    assert replay.content == successes[0].content
    # A later attempt with any other rule still gets the stable 409.
    with TestClient(app) as thread_client:
        again = thread_client.post(
            "/v1/policies",
            json=_body(rule=distinct_rules[0] if winning_rule != distinct_rules[0] else distinct_rules[1]),
            headers={IDEMPOTENCY_HEADER: "race-key"},
        )
    assert again.status_code == 409


# --- keyless behavior is unchanged -----------------------------------------


def test_missing_header_keeps_legacy_incrementing_versions(app, client):
    first = _create(client)
    second = _create(client)
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["version"] == 1
    assert second.json()["version"] == 2

    # No idempotency record is ever written for keyless requests.
    assert _count_rows(app, Policy) == 2
    assert _count_rows(app, PolicyIdempotencyRecord) == 0


def test_keyed_and_keyless_creates_share_version_sequence(app, client):
    keyed = _create(client, key="key-1")
    assert keyed.json()["version"] == 1

    # A keyless request of the same name allocates the next version.
    keyless = _create(client)
    assert keyless.status_code == 201
    assert keyless.json()["version"] == 2

    # The keyed replay still returns the stored v1 response.
    replay = _create(client, key="key-1")
    assert replay.content == keyed.content
    assert replay.json()["version"] == 1
    assert _count_rows(app, Policy) == 2
    assert _count_rows(app, PolicyIdempotencyRecord) == 1


def test_policy_listing_unaffected_by_keyed_create(app, client):
    created = _create(client, key="key-1")
    assert created.status_code == 201

    listed = client.get(
        "/v1/policies",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert listed.status_code == 200
    page = listed.json()["policies"]
    assert len(page) == 1
    assert page[0]["policy_id"] == created.json()["policy_id"]
    # Idempotency storage is not public data: no key field leaks onto the
    # public policy shape.
    assert "idempotency_key" not in page[0]
    assert "request_fingerprint" not in page[0]

    # A replay adds no listing entry.
    assert _create(client, key="key-1").status_code == 201
    listed = client.get(
        "/v1/policies",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert len(listed.json()["policies"]) == 1


# --- storage failures and atomicity ----------------------------------------


def test_idempotency_table_write_failure_returns_500_with_no_policy(app, client):
    from sqlalchemy import text

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policy_idempotency_records"))

    response = _create(client, key="key-1")
    assert response.status_code == 500
    assert response.json() == {"detail": "policy write failed"}
    # No half state: a policy must never exist without its record.
    assert _count_rows(app, Policy) == 0

    # The keyless path does not touch idempotency storage.
    assert _create(client).status_code == 201


def test_policy_table_write_failure_returns_500_with_no_record(app, client):
    from sqlalchemy import text

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policies"))

    response = _create(client, key="key-1")
    assert response.status_code == 500
    assert response.json() == {"detail": "policy write failed"}
    assert _count_rows(app, PolicyIdempotencyRecord) == 0


def test_failed_write_leaves_key_free_and_recovers_on_retry(tmp_path):
    from sqlalchemy import text

    url = f"sqlite:///{tmp_path}/recover-idem.db"
    application = create_app(url)
    client = TestClient(application)

    with application.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policy_idempotency_records"))
    failed = _create(client, key="key-1")
    assert failed.status_code == 500
    application.state.engine.dispose()

    # A fresh process re-creates the missing table additively and the same
    # legal request now succeeds under the same key.
    application = create_app(url)
    client = TestClient(application)
    created = _create(client, key="key-1")
    assert created.status_code == 201
    assert created.json()["version"] == 1
    with application.state.session_factory() as session:
        assert session.query(Policy).count() == 1
        assert session.query(PolicyIdempotencyRecord).count() == 1
    application.state.engine.dispose()


# --- restart persistence ----------------------------------------------------


def test_idempotency_record_survives_restart_and_replays_original(tmp_path):
    url = f"sqlite:///{tmp_path}/restart-idem.db"

    app1 = create_app(url)
    client1 = TestClient(app1)
    accepted = _create(client1, key="durable-key")
    assert accepted.status_code == 201
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    replay = _create(client2, key="durable-key")
    assert replay.status_code == 201
    assert replay.content == accepted.content
    assert replay.json()["policy_id"] == accepted.json()["policy_id"]
    assert replay.json()["version"] == 1
    with app2.state.session_factory() as session:
        assert session.query(Policy).count() == 1
        assert session.query(PolicyIdempotencyRecord).count() == 1
    app2.state.engine.dispose()


# --- additive migration on an old database ----------------------------------


def test_old_database_creates_idempotency_table_on_first_open(tmp_path):
    from sqlalchemy import create_engine, inspect, text

    from proof_release.db import Base

    url = f"sqlite:///{tmp_path}/legacy.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    # Simulate a deployment older than policy idempotency: its database
    # has every other table but not the idempotency records table.
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE policy_idempotency_records"))
    assert "policy_idempotency_records" not in inspect(engine).get_table_names()
    engine.dispose()

    # First open of the old database establishes the table additively;
    # existing policies and keyless creation keep working.
    application = create_app(url)
    assert "policy_idempotency_records" in inspect(
        application.state.engine
    ).get_table_names()
    client = TestClient(application)

    keyed = _create(client, key="legacy-key")
    assert keyed.status_code == 201
    assert _create(client, key="legacy-key").content == keyed.content
    assert _create(client).status_code == 201
    with application.state.session_factory() as session:
        assert session.query(PolicyIdempotencyRecord).count() == 1
        assert session.query(Policy).count() == 2

    # Migration is idempotent: a second open on the migrated file works.
    application.state.engine.dispose()
    application = create_app(url)
    client = TestClient(application)
    assert _create(client, key="legacy-key").status_code == 201
    application.state.engine.dispose()


# --- secrecy: only the documented columns exist -----------------------------


def test_record_has_no_evidence_or_material_columns(app, client):
    response = _create(client, key="key-1")
    assert response.status_code == 201

    (record,) = _records(app)
    table = type(record).metadata.tables["policy_idempotency_records"]
    columns = {column.name for column in table.columns}
    assert columns == {
        "record_id",
        "tenant_id",
        "workload_id",
        "idempotency_key",
        "request_fingerprint",
        "response_body",
        "created_at",
    }
    # The fingerprint is an irreversible digest, not the rule itself.
    assert "measurement" not in record.request_fingerprint
