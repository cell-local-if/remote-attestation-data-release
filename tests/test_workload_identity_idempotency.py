"""Tests for the optional Idempotency-Key header on
``POST /v1/workload-identities``."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select

from proof_release.app import create_app
from proof_release.db import (
    WorkloadIdentityClaim,
    WorkloadIdentityIdempotencyRecord,
    WorkloadIdentityProfile,
)

from x509_helpers import make_root, pem

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

CLAIM_A = {"issuer": "CN=ca", "subject": "CN=leaf", "uri": "spiffe://example.org/a"}
CLAIM_B = {"issuer": "CN=ca", "subject": "CN=leaf", "uri": "spiffe://example.org/b"}


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/test.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def root_id(client):
    root_key, root_cert = make_root()
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(root_cert),
        },
    )
    assert response.status_code == 201
    return response.json()["root_id"]


@pytest.fixture()
def other_root_id(client):
    root_key, root_cert = make_root("other-root")
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(root_cert),
        },
    )
    assert response.status_code == 201
    return response.json()["root_id"]


def _register(client, root, claims, *, tenant=TENANT, workload=WORKLOAD, key=None):
    headers = {} if key is None else {"Idempotency-Key": key}
    return client.post(
        "/v1/workload-identities",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "trust_root_id": root,
            "claims": claims,
        },
        headers=headers,
    )


def _state(client):
    with client.app.state.session_factory() as session:
        profiles = session.scalars(select(WorkloadIdentityProfile)).all()
        claims = session.scalars(select(WorkloadIdentityClaim)).all()
        records = session.scalars(select(WorkloadIdentityIdempotencyRecord)).all()
    return profiles, claims, records


# --- replay -----------------------------------------------------------------


def test_keyed_replay_returns_first_201_verbatim(client, root_id):
    first = _register(client, root_id, [CLAIM_A], key="key-1")
    assert first.status_code == 201

    replay = _register(client, root_id, [dict(CLAIM_A)], key="key-1")
    assert replay.status_code == 201
    assert replay.content == first.content

    profiles, claims, records = _state(client)
    assert len(profiles) == 1
    assert len(claims) == 1
    assert len(records) == 1


def test_keyed_replay_normalizes_claim_order_and_duplicates(client, root_id):
    first = _register(client, root_id, [CLAIM_A, CLAIM_B, CLAIM_A], key="key-1")
    assert first.status_code == 201
    assert first.json()["claims"] == [CLAIM_A, CLAIM_B]

    # Same set, different order and no duplicate: still the same request.
    replay = _register(client, root_id, [CLAIM_B, CLAIM_A], key="key-1")
    assert replay.status_code == 201
    assert replay.content == first.content

    profiles, claims, records = _state(client)
    assert len(profiles) == 1
    assert len(claims) == 2
    assert len(records) == 1


def test_keyed_registration_persists_across_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    root_key, root_cert = make_root()
    root = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(root_cert),
        },
    ).json()["root_id"]
    first = _register(client1, root, [CLAIM_A], key="key-1")
    assert first.status_code == 201
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    replay = _register(client2, root, [CLAIM_A], key="key-1")
    assert replay.status_code == 201
    assert replay.content == first.content
    profiles, claims, records = _state(client2)
    assert len(profiles) == 1
    assert len(claims) == 1
    assert len(records) == 1
    app2.state.engine.dispose()


def test_same_key_conflict_on_different_claim_set(client, root_id):
    first = _register(client, root_id, [CLAIM_A], key="key-1")
    assert first.status_code == 201

    conflict = _register(client, root_id, [CLAIM_B], key="key-1")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency key conflict"

    # Stable: the conflict repeats and nothing changes.
    again = _register(client, root_id, [CLAIM_B], key="key-1")
    assert again.status_code == 409
    profiles, claims, records = _state(client)
    assert len(profiles) == 1
    assert len(claims) == 1
    assert len(records) == 1


def test_same_key_conflict_on_different_trust_root(client, root_id, other_root_id):
    first = _register(client, root_id, [CLAIM_A], key="key-1")
    assert first.status_code == 201

    conflict = _register(client, other_root_id, [CLAIM_A], key="key-1")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency key conflict"

    profiles, claims, records = _state(client)
    assert len(profiles) == 1
    assert len(records) == 1


def test_same_key_is_independent_in_other_scope(client, root_id):
    first = _register(client, root_id, [CLAIM_A], key="key-1")
    assert first.status_code == 201

    other_tenant = _register(
        client, root_id, [CLAIM_A], tenant=OTHER_TENANT, key="key-1"
    )
    # The root belongs to TENANT, so a cross-tenant request is a 404 and
    # consumes nothing; register a root in the other scope instead.
    assert other_tenant.status_code == 404

    root_key, root_cert = make_root("scope-root")
    created = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": OTHER_TENANT,
            "workload_id": OTHER_WORKLOAD,
            "root_pem": pem(root_cert),
        },
    )
    other_root = created.json()["root_id"]
    independent = _register(
        client,
        other_root,
        [CLAIM_A],
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
        key="key-1",
    )
    assert independent.status_code == 201
    assert independent.json()["profile_id"] != first.json()["profile_id"]

    profiles, _, records = _state(client)
    assert len(profiles) == 2
    assert len(records) == 2


# --- interaction with the unkeyed semantics ---------------------------------


def test_keyed_request_against_existing_claim_set_is_409_and_key_stays_free(
    client, root_id
):
    unkeyed = _register(client, root_id, [CLAIM_A])
    assert unkeyed.status_code == 201

    duplicate = _register(client, root_id, [CLAIM_A], key="key-1")
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "workload identity profile already registered"

    # The failed attempt did not consume the key: a distinct claim set
    # under the same key succeeds.
    recovered = _register(client, root_id, [CLAIM_B], key="key-1")
    assert recovered.status_code == 201
    replay = _register(client, root_id, [CLAIM_B], key="key-1")
    assert replay.status_code == 201
    assert replay.content == recovered.content

    profiles, _, records = _state(client)
    assert len(profiles) == 2
    assert len(records) == 1


def test_unkeyed_semantics_unchanged(client, root_id):
    first = _register(client, root_id, [CLAIM_A])
    assert first.status_code == 201
    duplicate = _register(client, root_id, [CLAIM_A])
    assert duplicate.status_code == 409
    assert (
        duplicate.json()["detail"] == "workload identity profile already registered"
    )
    distinct = _register(client, root_id, [CLAIM_B])
    assert distinct.status_code == 201

    # Unkeyed duplicates of a keyed profile are still plain 409s.
    keyed = _register(client, root_id, [{"issuer": "i", "subject": "s", "uri": "u"}], key="k")
    assert keyed.status_code == 201
    unkeyed_repeat = _register(
        client, root_id, [{"issuer": "i", "subject": "s", "uri": "u"}]
    )
    assert unkeyed_repeat.status_code == 409

    _, _, records = _state(client)
    assert len(records) == 1


def test_keyed_unknown_trust_root_is_404_and_key_stays_free(client, root_id):
    missing = _register(
        client, "11111111-1111-1111-1111-111111111111", [CLAIM_A], key="key-1"
    )
    assert missing.status_code == 404
    assert missing.json()["detail"] == "trust root not found"

    recovered = _register(client, root_id, [CLAIM_A], key="key-1")
    assert recovered.status_code == 201
    _, _, records = _state(client)
    assert len(records) == 1


# --- header validation --------------------------------------------------------


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


def _raw_register(app, root, claims, *, key_raw=b"key-1", duplicate_key=False):
    payload = json.dumps(
        {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root,
            "claims": claims,
        }
    ).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"idempotency-key", key_raw),
    ]
    if duplicate_key:
        headers.append((b"idempotency-key", key_raw))
    return asyncio.run(_asgi_call(app, "/v1/workload-identities", headers, payload))


@pytest.mark.parametrize(
    "raw_key",
    [
        b"",  # empty value
        b" key-1",  # leading space
        b"key-1 ",  # trailing space
        b"key 1",  # embedded space
        b"key\t1",  # tab
        b"key-1\n",  # control character
        b"k\xc3\xa9y",  # non-ASCII (UTF-8 e-acute)
        b"k" * 65,  # over-long
    ],
)
def test_invalid_idempotency_key_returns_422(app, root_id, raw_key):
    status, raw = _raw_register(app, root_id, [CLAIM_A], key_raw=raw_key)
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    with app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentityProfile)).all() == []
        assert session.scalars(select(WorkloadIdentityClaim)).all() == []
        assert session.scalars(select(WorkloadIdentityIdempotencyRecord)).all() == []


def test_duplicated_idempotency_key_header_returns_422(client, root_id, app):
    status, raw = _raw_register(app, root_id, [CLAIM_A], duplicate_key=True)
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    response = client.post(
        "/v1/workload-identities",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "claims": [CLAIM_A],
        },
        headers=[("Idempotency-Key", "key-1"), ("Idempotency-Key", "key-1")],
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid idempotency key"


def test_boundary_key_lengths_are_accepted(client, root_id, other_root_id):
    one = _register(client, root_id, [CLAIM_A], key="x")
    assert one.status_code == 201
    long_key = "k" * 64
    full = _register(client, other_root_id, [CLAIM_A], key=long_key)
    assert full.status_code == 201
    replay = _register(client, other_root_id, [CLAIM_A], key=long_key)
    assert replay.status_code == 201
    assert replay.content == full.content


# --- concurrency --------------------------------------------------------------


def test_concurrent_identical_keyed_requests_settle_one_profile(app, root_id):
    def register():
        return TestClient(app).post(
            "/v1/workload-identities",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "trust_root_id": root_id,
                "claims": [CLAIM_A],
            },
            headers={"Idempotency-Key": "key-1"},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: register(), range(20)))

    assert all(r.status_code == 201 for r in responses)
    bodies = {r.content for r in responses}
    assert len(bodies) == 1
    with app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentityProfile)).all().__len__() == 1
        assert session.scalars(select(WorkloadIdentityClaim)).all().__len__() == 1
        assert (
            session.scalars(select(WorkloadIdentityIdempotencyRecord)).all().__len__()
            == 1
        )


def test_concurrent_same_key_different_requests_one_wins(app, root_id):
    def register(claim):
        return TestClient(app).post(
            "/v1/workload-identities",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "trust_root_id": root_id,
                "claims": [claim],
            },
            headers={"Idempotency-Key": "key-1"},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(register, [CLAIM_A, CLAIM_B] * 10))

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(201) >= 1
    assert statuses.count(409) == 20 - statuses.count(201)
    # Every 201 replays the single winning body; every 409 is the conflict.
    winners = {r.content for r in responses if r.status_code == 201}
    assert len(winners) == 1
    for r in responses:
        if r.status_code == 409:
            assert r.json()["detail"] == "idempotency key conflict"
    with app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentityProfile)).all().__len__() == 1
        assert (
            session.scalars(select(WorkloadIdentityIdempotencyRecord)).all().__len__()
            == 1
        )


# --- failure semantics --------------------------------------------------------


def test_failed_keyed_write_rolls_back_and_key_stays_free(app, client, root_id):
    engine = app.state.engine

    def fail_insert(conn, cursor, statement, parameters, context, executemany):
        if "INTO workload_identity_idempotency_records" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_insert)
    try:
        response = _register(client, root_id, [CLAIM_A], key="key-1")
        assert response.status_code == 500
        assert response.json()["detail"] == "workload identity registration failed"
    finally:
        event.remove(engine, "before_cursor_execute", fail_insert)

    profiles, claims, records = _state(client)
    assert profiles == []
    assert claims == []
    assert records == []

    # Once storage recovers the same keyed request succeeds and replays.
    recovered = _register(client, root_id, [CLAIM_A], key="key-1")
    assert recovered.status_code == 201
    replay = _register(client, root_id, [CLAIM_A], key="key-1")
    assert replay.status_code == 201
    assert replay.content == recovered.content


def test_record_stores_no_sensitive_material(client, root_id):
    registered = _register(client, root_id, [CLAIM_A], key="key-1")
    assert registered.status_code == 201
    _, _, records = _state(client)
    assert len(records) == 1
    dumped = str(
        {c.name: getattr(records[0], c.name) for c in records[0].__table__.columns}
    )
    assert "BEGIN CERTIFICATE" not in dumped
    assert "PRIVATE KEY" not in dumped
