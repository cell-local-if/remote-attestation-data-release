"""Tests for the optional Idempotency-Key header on POST /v1/workload-identities.

A keyed registration writes the profile, its claims and the idempotency
record in one transaction and replays the exact stored 201 afterwards; a
missing key keeps the legacy duplicate-claim-set 409 behavior. The record
persists only the scope, the key, the normalized request fingerprint
(scope, trust root and the canonical claim-set fingerprint), the first
response body and the creation time — never capabilities, payloads,
evidence, certificate material, private keys or exception detail.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

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

IDEMPOTENCY_HEADER = "Idempotency-Key"

CLAIM_A = {
    "issuer": "CN=issuer-a",
    "subject": "CN=subject-a",
    "uri": "spiffe://example.org/workload/a",
}
CLAIM_B = {
    "issuer": "CN=issuer-b",
    "subject": "CN=subject-b",
    "uri": "spiffe://example.org/workload/b",
}
CLAIM_C = {
    "issuer": "CN=issuer-c",
    "subject": "CN=subject-c",
    "uri": "spiffe://example.org/workload/c",
}


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/idem-workload-identities.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _trust_root(client, tenant=TENANT, workload=WORKLOAD):
    _, root_cert = make_root()
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "root_pem": pem(root_cert),
        },
    )
    assert response.status_code == 201
    return response.json()["root_id"]


@pytest.fixture()
def root_id(client):
    return _trust_root(client)


def _register(
    client,
    root,
    claims,
    *,
    tenant=TENANT,
    workload=WORKLOAD,
    key=None,
):
    body = {
        "tenant_id": tenant,
        "workload_id": workload,
        "trust_root_id": root,
        "claims": claims,
    }
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else None
    return client.post("/v1/workload-identities", json=body, headers=headers)


def _count_rows(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


def _records(app):
    with app.state.session_factory() as session:
        return session.query(WorkloadIdentityIdempotencyRecord).all()


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


def _raw_register(app, body_obj, *, key_raw=b"key-1", duplicate_key=False):
    payload = json.dumps(body_obj).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
        (b"idempotency-key", key_raw),
    ]
    if duplicate_key:
        headers.append((b"idempotency-key", key_raw))
    return asyncio.run(_asgi_call(app, "/v1/workload-identities", headers, payload))


def _body(root, *, tenant=TENANT, workload=WORKLOAD, claims=None):
    return {
        "tenant_id": tenant,
        "workload_id": workload,
        "trust_root_id": root,
        "claims": claims if claims is not None else [CLAIM_A],
    }


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
def test_invalid_idempotency_key_is_422_before_any_write(app, root_id, raw_key):
    status, raw = _raw_register(app, _body(root_id), key_raw=raw_key)
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    assert _count_rows(app, WorkloadIdentityProfile) == 0
    assert _count_rows(app, WorkloadIdentityClaim) == 0
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 0


def test_duplicate_idempotency_header_is_422(app, root_id):
    status, raw = _raw_register(app, _body(root_id), duplicate_key=True)
    assert status == 422, raw
    assert json.loads(raw) == {"detail": "invalid idempotency key"}
    assert _count_rows(app, WorkloadIdentityProfile) == 0
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 0


def test_duplicate_idempotency_header_through_client_is_422(client, root_id):
    response = client.post(
        "/v1/workload-identities",
        json=_body(root_id),
        headers=[
            (IDEMPOTENCY_HEADER, "key-1"),
            (IDEMPOTENCY_HEADER, "key-1"),
        ],
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid idempotency key"}
    assert _count_rows(client.app, WorkloadIdentityProfile) == 0


@pytest.mark.parametrize("length", [1, 64])
def test_boundary_length_keys_accepted(app, client, root_id, length):
    response = _register(client, root_id, [CLAIM_A], key="A" * length)
    assert response.status_code == 201, response.text
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 1


def test_visible_punctuation_key_accepted(client, root_id):
    response = _register(
        client, root_id, [CLAIM_A], key="~!@#$%^&*()_+-=[]{}|;:',.<>/?`"
    )
    assert response.status_code == 201, response.text


def test_invalid_key_is_not_consumed_and_later_valid_use_succeeds(
    app, client, root_id
):
    status, _raw = _raw_register(app, _body(root_id), key_raw=b"")
    assert status == 422
    assert _raw_register(app, _body(root_id), key_raw=b"bad key")[0] == 422
    accepted = _register(client, root_id, [CLAIM_A], key="key-1")
    assert accepted.status_code == 201
    assert _count_rows(app, WorkloadIdentityProfile) == 1
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 1


def test_invalid_body_with_valid_key_is_422_and_writes_nothing(
    app, client, root_id
):
    bad_bodies = [
        _body(root_id, claims=[]),  # empty claim list
        _body(root_id, claims=[{"issuer": "CN=i", "subject": "CN=s"}]),
        _body(root_id, tenant=""),
        _body(root_id, tenant="   "),
        _body(root_id, workload=""),
        _body(root_id, claims=[{"issuer": " ", "subject": "s", "uri": "u"}]),
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": [CLAIM_A]},
    ]
    for body in bad_bodies:
        response = client.post(
            "/v1/workload-identities",
            json=body,
            headers={IDEMPOTENCY_HEADER: "key-1"},
        )
        assert response.status_code == 422, body
    assert _count_rows(app, WorkloadIdentityProfile) == 0
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 0
    # The key was never consumed: the same key now succeeds.
    accepted = _register(client, root_id, [CLAIM_A], key="key-1")
    assert accepted.status_code == 201
    assert _count_rows(app, WorkloadIdentityProfile) == 1
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 1


# --- core replay semantics -------------------------------------------------


def test_first_keyed_register_writes_profile_claims_and_record_atomically(
    app, client, root_id
):
    response = _register(client, root_id, [CLAIM_A, CLAIM_B], key="key-1")

    assert response.status_code == 201
    data = response.json()
    assert set(data) == {
        "profile_id",
        "tenant_id",
        "workload_id",
        "trust_root_id",
        "claims",
        "created_at",
    }
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["trust_root_id"] == root_id
    assert data["claims"] == [CLAIM_A, CLAIM_B]

    assert _count_rows(app, WorkloadIdentityProfile) == 1
    assert _count_rows(app, WorkloadIdentityClaim) == 2
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


def test_replay_returns_first_response_byte_for_byte(app, client, root_id):
    first = _register(client, root_id, [CLAIM_A], key="key-1")
    assert first.status_code == 201

    replay = _register(client, root_id, [CLAIM_A], key="key-1")
    assert replay.status_code == 201
    assert replay.content == first.content
    assert replay.json() == first.json()

    # No second profile, no rewritten claims, no second record.
    assert _count_rows(app, WorkloadIdentityProfile) == 1
    assert _count_rows(app, WorkloadIdentityClaim) == 1
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 1


def test_canonically_equivalent_claims_replay_first_response(
    app, client, root_id
):
    # Claims compare as an unordered, de-duplicated set: the replay names
    # the same claims reordered and repeated, and the first response bytes
    # (with the first request's first-seen order) are what comes back.
    first = _register(client, root_id, [CLAIM_A, CLAIM_B], key="key-1")
    assert first.status_code == 201
    assert first.json()["claims"] == [CLAIM_A, CLAIM_B]

    replay = _register(
        client, root_id, [CLAIM_B, CLAIM_A, CLAIM_B], key="key-1"
    )
    assert replay.status_code == 201
    assert replay.content == first.content
    assert replay.json()["claims"] == [CLAIM_A, CLAIM_B]

    assert _count_rows(app, WorkloadIdentityProfile) == 1
    assert _count_rows(app, WorkloadIdentityClaim) == 2
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 1


def test_same_key_different_claim_set_is_409_and_changes_nothing(
    app, client, root_id
):
    first = _register(client, root_id, [CLAIM_A], key="key-1")
    assert first.status_code == 201

    conflict = _register(client, root_id, [CLAIM_C], key="key-1")
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "idempotency key conflict"}

    assert _count_rows(app, WorkloadIdentityProfile) == 1
    assert _count_rows(app, WorkloadIdentityClaim) == 1
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 1
    # The original record still replays the original response.
    assert (
        _register(client, root_id, [CLAIM_A], key="key-1").content
        == first.content
    )


def test_same_key_different_trust_root_is_409_and_changes_nothing(
    app, client, root_id
):
    other_root = _trust_root(client)
    first = _register(client, root_id, [CLAIM_A], key="key-1")
    assert first.status_code == 201

    conflict = _register(client, other_root, [CLAIM_A], key="key-1")
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "idempotency key conflict"}

    assert _count_rows(app, WorkloadIdentityProfile) == 1
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 1
    # The conflicting key use consumed nothing: the other trust root can
    # still register the same claim set under a different key.
    assert (
        _register(client, other_root, [CLAIM_A], key="key-2").status_code
        == 201
    )
    assert (
        _register(client, root_id, [CLAIM_A], key="key-1").content
        == first.content
    )


def test_same_key_in_other_scopes_is_independent(app, client, root_id):
    other_tenant_root = _trust_root(client, tenant=OTHER_TENANT)
    other_workload_root = _trust_root(client, workload=OTHER_WORKLOAD)

    first = _register(client, root_id, [CLAIM_A], key="key-1")
    assert first.status_code == 201

    other_tenant = _register(
        client, other_tenant_root, [CLAIM_A], tenant=OTHER_TENANT, key="key-1"
    )
    assert other_tenant.status_code == 201
    other_workload = _register(
        client,
        other_workload_root,
        [CLAIM_A],
        workload=OTHER_WORKLOAD,
        key="key-1",
    )
    assert other_workload.status_code == 201

    # Three independent scopes, three profiles, three records.
    assert _count_rows(app, WorkloadIdentityProfile) == 3
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 3

    # Within each scope the key replays its own first response only.
    assert (
        _register(client, root_id, [CLAIM_A], key="key-1").content
        == first.content
    )
    assert (
        _register(
            client, other_tenant_root, [CLAIM_A], tenant=OTHER_TENANT,
            key="key-1",
        ).content
        == other_tenant.content
    )
    # Cross-scope replay never leaks the other scope's response.
    assert other_tenant.content != first.content


def test_keyed_duplicate_claim_set_is_409_and_key_stays_free(
    app, client, root_id
):
    # A keyless registration of the claim set already exists.
    keyless = _register(client, root_id, [CLAIM_A])
    assert keyless.status_code == 201

    # A keyed request for the same set is the duplicate-set 409, not a
    # replay, and it does not consume the key.
    duplicate = _register(client, root_id, [CLAIM_A], key="key-1")
    assert duplicate.status_code == 409
    assert duplicate.json() == {
        "detail": "workload identity profile already registered"
    }
    assert _count_rows(app, WorkloadIdentityProfile) == 1
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 0

    # The same key remains usable for a genuinely new registration.
    created = _register(client, root_id, [CLAIM_B], key="key-1")
    assert created.status_code == 201
    assert _count_rows(app, WorkloadIdentityProfile) == 2
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 1


def test_unknown_trust_root_with_key_is_404_and_key_stays_free(
    app, client, root_id
):
    missing = "00000000-0000-0000-0000-000000000000"
    response = _register(client, missing, [CLAIM_A], key="key-1")
    assert response.status_code == 404
    assert response.json() == {"detail": "trust root not found"}
    assert _count_rows(app, WorkloadIdentityProfile) == 0
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 0

    # The key was not consumed by the 404.
    assert _register(client, root_id, [CLAIM_A], key="key-1").status_code == 201


def test_cross_scope_trust_root_with_key_is_404(app, client, root_id):
    # A trust root that exists only in another tenant is
    # indistinguishable from an unknown one.
    response = _register(client, root_id, [CLAIM_A], tenant=OTHER_TENANT, key="key-1")
    assert response.status_code == 404
    assert response.json() == {"detail": "trust root not found"}
    assert _count_rows(app, WorkloadIdentityProfile) == 0
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 0


# --- concurrency -----------------------------------------------------------


def test_concurrent_same_key_same_request_create_exactly_one_profile(
    app, root_id
):
    body = _body(root_id, claims=[CLAIM_A, CLAIM_B])

    def submit(_):
        with TestClient(app) as thread_client:
            return thread_client.post(
                "/v1/workload-identities",
                json=body,
                headers={IDEMPOTENCY_HEADER: "key-1"},
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(submit, range(8)))

    assert {r.status_code for r in responses} == {201}
    bodies = {r.content for r in responses}
    assert len(bodies) == 1
    assert _count_rows(app, WorkloadIdentityProfile) == 1
    assert _count_rows(app, WorkloadIdentityClaim) == 2
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 1


def test_concurrent_different_requests_same_key_one_wins_rest_409(
    app, root_id
):
    # Eight genuinely different requests (distinct claim sets) competing
    # for one key: at most one can register, the rest settle as a stable
    # 409. Same-request concurrency (all 201, one profile) is covered
    # separately.
    distinct_claims = [
        [
            {
                "issuer": f"CN=issuer-{index}",
                "subject": f"CN=subject-{index}",
                "uri": f"spiffe://example.org/workload/{index}",
            }
        ]
        for index in range(8)
    ]

    def submit(index):
        with TestClient(app) as thread_client:
            return thread_client.post(
                "/v1/workload-identities",
                json=_body(root_id, claims=distinct_claims[index]),
                headers={IDEMPOTENCY_HEADER: "race-key"},
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(submit, range(8)))

    successes = [r for r in responses if r.status_code == 201]
    conflicts = [r for r in responses if r.status_code == 409]
    assert len(successes) == 1
    assert len(conflicts) == 7
    assert all(
        r.json() == {"detail": "idempotency key conflict"} for r in conflicts
    )
    assert _count_rows(app, WorkloadIdentityProfile) == 1
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 1
    # A sequential replay of the winning request returns the same bytes.
    winning_claims = successes[0].json()["claims"]
    with TestClient(app) as thread_client:
        replay = thread_client.post(
            "/v1/workload-identities",
            json=_body(root_id, claims=winning_claims),
            headers={IDEMPOTENCY_HEADER: "race-key"},
        )
    assert replay.status_code == 201
    assert replay.content == successes[0].content
    # A later attempt with any other claim set still gets the stable 409.
    other = next(c for c in distinct_claims if c != winning_claims)
    with TestClient(app) as thread_client:
        again = thread_client.post(
            "/v1/workload-identities",
            json=_body(root_id, claims=other),
            headers={IDEMPOTENCY_HEADER: "race-key"},
        )
    assert again.status_code == 409
    assert again.json() == {"detail": "idempotency key conflict"}


# --- keyless behavior is unchanged -----------------------------------------


def test_missing_header_keeps_legacy_duplicate_set_semantics(
    app, client, root_id
):
    first = _register(client, root_id, [CLAIM_A, CLAIM_B])
    assert first.status_code == 201

    # The same set (reordered, duplicated) under the same trust root is
    # the legacy 409.
    duplicate = _register(client, root_id, [CLAIM_B, CLAIM_A, CLAIM_B])
    assert duplicate.status_code == 409
    assert duplicate.json() == {
        "detail": "workload identity profile already registered"
    }

    # A distinct claim set still registers independently.
    second = _register(client, root_id, [CLAIM_C])
    assert second.status_code == 201
    assert second.json()["profile_id"] != first.json()["profile_id"]

    # No idempotency record is ever written for keyless requests.
    assert _count_rows(app, WorkloadIdentityProfile) == 2
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 0


def test_keyed_and_keyless_registrations_coexist(app, client, root_id):
    keyed = _register(client, root_id, [CLAIM_A], key="key-1")
    assert keyed.status_code == 201

    # A keyless request with a distinct claim set registers normally.
    keyless = _register(client, root_id, [CLAIM_B])
    assert keyless.status_code == 201

    # The keyed replay still returns the stored first response.
    replay = _register(client, root_id, [CLAIM_A], key="key-1")
    assert replay.content == keyed.content
    assert _count_rows(app, WorkloadIdentityProfile) == 2
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 1


def test_profile_queries_unaffected_by_keyed_registration(
    app, client, root_id
):
    created = _register(client, root_id, [CLAIM_A], key="key-1")
    assert created.status_code == 201
    profile_id = created.json()["profile_id"]

    listed = client.get(
        "/v1/workload-identities",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
        },
    )
    assert listed.status_code == 200
    page = listed.json()
    assert len(page) == 1
    assert page[0]["profile_id"] == profile_id
    # Idempotency storage is not public data: no key field leaks onto the
    # public profile shape.
    assert "idempotency_key" not in page[0]
    assert "request_fingerprint" not in page[0]

    # A replay adds no listing entry.
    assert _register(client, root_id, [CLAIM_A], key="key-1").status_code == 201
    listed = client.get(
        "/v1/workload-identities",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
        },
    )
    assert len(listed.json()) == 1

    # PUT and DELETE on the keyed-created profile behave as before.
    updated = client.put(
        f"/v1/workload-identities/{profile_id}",
        json=_body(root_id, claims=[CLAIM_B]),
    )
    assert updated.status_code == 200
    # The installed httpx TestClient.delete does not accept json=.
    revoked = client.request(
        "DELETE",
        f"/v1/workload-identities/{profile_id}",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
        },
    )
    assert revoked.status_code == 200

    # The stored replay is untouched by the profile's later lifecycle.
    replay = _register(client, root_id, [CLAIM_A], key="key-1")
    assert replay.status_code == 201
    assert replay.content == created.content


# --- storage failures and atomicity ----------------------------------------


def test_idempotency_table_write_failure_returns_500_with_no_profile(
    app, client, root_id
):
    from sqlalchemy import text

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE workload_identity_idempotency_records"))

    response = _register(client, root_id, [CLAIM_A], key="key-1")
    assert response.status_code == 500
    assert response.json() == {"detail": "workload identity registration failed"}
    # No half state: a profile must never exist without its record.
    assert _count_rows(app, WorkloadIdentityProfile) == 0
    assert _count_rows(app, WorkloadIdentityClaim) == 0

    # The keyless path does not touch idempotency storage.
    assert _register(client, root_id, [CLAIM_A]).status_code == 201


def test_profile_table_write_failure_returns_500_with_no_record(
    app, client, root_id
):
    from sqlalchemy import text

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE workload_identity_profiles"))

    response = _register(client, root_id, [CLAIM_A], key="key-1")
    assert response.status_code == 500
    assert response.json() == {"detail": "workload identity registration failed"}
    assert _count_rows(app, WorkloadIdentityIdempotencyRecord) == 0


def test_failed_write_leaves_key_free_and_recovers_on_retry(tmp_path):
    from sqlalchemy import text

    url = f"sqlite:///{tmp_path}/recover-idem.db"
    application = create_app(url)
    client = TestClient(application)
    root = _trust_root(client)

    with application.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE workload_identity_idempotency_records"))
    failed = _register(client, root, [CLAIM_A], key="key-1")
    assert failed.status_code == 500
    application.state.engine.dispose()

    # A fresh process re-creates the missing table additively and the same
    # legal request now succeeds under the same key.
    application = create_app(url)
    client = TestClient(application)
    created = _register(client, root, [CLAIM_A], key="key-1")
    assert created.status_code == 201
    with application.state.session_factory() as session:
        assert session.query(WorkloadIdentityProfile).count() == 1
        assert session.query(WorkloadIdentityIdempotencyRecord).count() == 1
    application.state.engine.dispose()


# --- restart persistence ----------------------------------------------------


def test_idempotency_record_survives_restart_and_replays_original(tmp_path):
    url = f"sqlite:///{tmp_path}/restart-idem.db"

    app1 = create_app(url)
    client1 = TestClient(app1)
    root = _trust_root(client1)
    accepted = _register(client1, root, [CLAIM_A, CLAIM_B], key="durable-key")
    assert accepted.status_code == 201
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    replay = _register(client2, root, [CLAIM_B, CLAIM_A], key="durable-key")
    assert replay.status_code == 201
    assert replay.content == accepted.content
    assert replay.json()["profile_id"] == accepted.json()["profile_id"]
    with app2.state.session_factory() as session:
        assert session.query(WorkloadIdentityProfile).count() == 1
        assert session.query(WorkloadIdentityIdempotencyRecord).count() == 1
    app2.state.engine.dispose()


# --- additive migration on an old database ----------------------------------


def test_old_database_creates_idempotency_table_on_first_open(tmp_path):
    from sqlalchemy import create_engine, inspect, text

    from proof_release.db import Base

    url = f"sqlite:///{tmp_path}/legacy.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    # Simulate a deployment older than workload-identity idempotency: its
    # database has every other table but not the idempotency records table.
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE workload_identity_idempotency_records"))
    assert "workload_identity_idempotency_records" not in inspect(
        engine
    ).get_table_names()
    engine.dispose()

    # First open of the old database establishes the table additively;
    # existing profiles and keyless registration keep working.
    application = create_app(url)
    assert "workload_identity_idempotency_records" in inspect(
        application.state.engine
    ).get_table_names()
    client = TestClient(application)
    root = _trust_root(client)

    keyed = _register(client, root, [CLAIM_A], key="legacy-key")
    assert keyed.status_code == 201
    assert _register(client, root, [CLAIM_A], key="legacy-key").content == (
        keyed.content
    )
    assert _register(client, root, [CLAIM_B]).status_code == 201
    with application.state.session_factory() as session:
        assert session.query(WorkloadIdentityIdempotencyRecord).count() == 1
        assert session.query(WorkloadIdentityProfile).count() == 2

    # Migration is idempotent: a second open on the migrated file works.
    application.state.engine.dispose()
    application = create_app(url)
    client = TestClient(application)
    assert (
        _register(client, root, [CLAIM_A], key="legacy-key").status_code == 201
    )
    application.state.engine.dispose()


# --- secrecy: only the documented columns exist -----------------------------


def test_record_has_no_material_or_detail_columns(app, client, root_id):
    response = _register(client, root_id, [CLAIM_A], key="key-1")
    assert response.status_code == 201

    (record,) = _records(app)
    table = type(record).metadata.tables["workload_identity_idempotency_records"]
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
    # The fingerprint is an irreversible digest, not the claim material.
    assert "issuer" not in record.request_fingerprint
    assert "spiffe" not in record.request_fingerprint
