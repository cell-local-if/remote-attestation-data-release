"""Tests for the read-only GET /v1/challenges/{challenge_id} status query."""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from proof_release.app import create_app
from proof_release.db import Base, Challenge, Evidence

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"

EXPECTED_KEYS = [
    "challenge_id",
    "issued_at",
    "expires_at",
    "status",
    "changed_at",
    "evidence_id",
    "evidence_format",
    "verification_result",
]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    return create_app(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, **overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    body.update(overrides)
    return client.post("/v1/challenges", json=body)


def _status(client, challenge_id, *, tenant_id=TENANT, workload_id=WORKLOAD, **params):
    query = {"tenant_id": tenant_id, "workload_id": workload_id}
    query.update(params)
    return client.get(f"/v1/challenges/{challenge_id}", params=query)


def _consume(client, created, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
    }
    body.update(overrides)
    return client.post(
        f"/v1/challenges/{created['challenge_id']}/consume", json=body
    )


def _mac(nonce: str, claims: dict) -> str:
    key = hmac.new(
        SECRET.encode("utf-8"),
        f"{TENANT}:{WORKLOAD}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _submit(client, created, evidence, evidence_format="attested-nonce-json"):
    return client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": evidence_format,
            "evidence": evidence,
        },
    )


def _attested_evidence(nonce: str, claims: dict | None = None, *, mac_override=None):
    mac = mac_override or _mac(nonce, claims or {})
    return json.dumps({"nonce": nonce, "claims": claims or {}, "mac": mac})


def _verify(client, evidence_id, created, evidence):
    return client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )


def _all_table_counts(app) -> dict[str, int]:
    counts = {}
    with app.state.engine.connect() as conn:
        for table_name in sorted(Base.metadata.tables):
            counts[table_name] = conn.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            ).scalar_one()
    return counts


# --- happy paths: the six derived phases -------------------------------------


def test_pending_challenge_status(client):
    created = _create(client).json()

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    data = response.json()
    assert list(data.keys()) == EXPECTED_KEYS
    assert data["challenge_id"] == created["challenge_id"]
    assert data["issued_at"] == created["issued_at"]
    assert data["expires_at"] == created["expires_at"]
    assert data["status"] == "pending"
    # The deciding timestamp for a pending challenge is its issuance time.
    assert data["changed_at"] == created["issued_at"]
    assert data["evidence_id"] is None
    assert data["evidence_format"] is None
    assert data["verification_result"] is None
    for field in ("issued_at", "expires_at", "changed_at"):
        assert datetime.fromisoformat(data[field]).utcoffset() == timedelta(0)


def test_pending_status_does_not_expire_or_mutate_row(client, app):
    created = _create(client).json()

    assert _status(client, created["challenge_id"]).json()["status"] == "pending"
    assert _status(client, created["challenge_id"]).json()["status"] == "pending"

    # No expiry writeback: the stored row is still pending with its original
    # timestamps.
    with app.state.session_factory() as session:
        row = session.get(Challenge, created["challenge_id"])
        assert row.status == "pending"
        assert row.consumed_at is None
        assert _rfc(row.issued_at) == created["issued_at"]
        assert _rfc(row.expires_at) == created["expires_at"]


def test_expired_status_is_derived_without_write(client, app):
    created = _create(client, ttl_seconds=30).json()
    with app.state.session_factory() as session:
        row = session.get(Challenge, created["challenge_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "expired"
    # changed_at is the expiry instant for the expired phase.
    assert data["changed_at"] == data["expires_at"]
    assert data["evidence_id"] is None
    assert data["evidence_format"] is None
    assert data["verification_result"] is None

    # Expiry is derived at read time: the persisted status is unchanged.
    with app.state.session_factory() as session:
        row = session.get(Challenge, created["challenge_id"])
        assert row.status == "pending"
        assert row.consumed_at is None


def test_status_at_or_after_expiry_instant_is_expired(client, app):
    created = _create(client, ttl_seconds=30).json()

    # expires_at exactly equal to the query instant: "已到" expires_at is
    # expired (the stored timestamp is necessarily no later than the
    # subsequent query instant).
    with app.state.session_factory() as session:
        row = session.get(Challenge, created["challenge_id"])
        row.expires_at = datetime.now(timezone.utc)
        session.commit()
    assert _status(client, created["challenge_id"]).json()["status"] == "expired"

    # One second past it is expired as well.
    with app.state.session_factory() as session:
        row = session.get(Challenge, created["challenge_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
    assert _status(client, created["challenge_id"]).json()["status"] == "expired"


def test_consumed_status(client):
    created = _create(client).json()
    consumed = _consume(client, created).json()

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "consumed"
    assert data["changed_at"] == consumed["consumed_at"]
    assert data["evidence_id"] is None
    assert data["evidence_format"] is None
    assert data["verification_result"] is None


def test_evidence_received_status(client):
    created = _create(client).json()
    evidence = _attested_evidence(created["nonce"], {"measurement": "abc"})
    submitted = _submit(client, created, evidence).json()

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "evidence_received"
    assert data["evidence_id"] == submitted["evidence_id"]
    assert data["evidence_format"] == "attested-nonce-json"
    assert data["changed_at"] == submitted["received_at"]
    # Not settled yet: no result code.
    assert data["verification_result"] is None


def test_verified_status(client):
    created = _create(client).json()
    evidence = _attested_evidence(created["nonce"], {"measurement": "abc"})
    evidence_id = _submit(client, created, evidence).json()["evidence_id"]
    verified = _verify(client, evidence_id, created, evidence).json()

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "verified"
    assert data["evidence_id"] == evidence_id
    assert data["evidence_format"] == "attested-nonce-json"
    assert data["changed_at"] == verified["verified_at"]
    assert data["verification_result"] == "accepted"


def test_rejected_status(client):
    created = _create(client).json()
    evidence = _attested_evidence(
        created["nonce"], {}, mac_override="0" * 64
    )
    evidence_id = _submit(client, created, evidence).json()["evidence_id"]
    rejected = _verify(client, evidence_id, created, evidence).json()
    assert rejected["status"] == "rejected"

    response = _status(client, created["challenge_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["evidence_id"] == evidence_id
    assert data["changed_at"] == rejected["verified_at"]
    assert data["verification_result"] == "rejected"


def test_response_never_carries_secret_material(client):
    created = _create(client).json()
    evidence_text = _attested_evidence(created["nonce"], {"measurement": "abc"})
    _submit(client, created, evidence_text)
    nonce_digest = hashlib.sha256(created["nonce"].encode("ascii")).hexdigest()
    evidence_digest = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()

    raw = _status(client, created["challenge_id"]).content.decode("utf-8")

    assert created["nonce"] not in raw
    assert nonce_digest not in raw
    assert evidence_text not in raw
    assert evidence_digest not in raw


# --- request shape: 422 ------------------------------------------------------


@pytest.mark.parametrize(
    "path_id",
    [
        "not-a-uuid",
        "00000000-0000-0000-0000-00000000000",  # one char short
        "00000000-0000-0000-0000-0000000000000",  # one char long
        "0000000000000000000000000000000z",  # non-hex
        "{00000000-0000-0000-0000-000000000000}",  # braces
        "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",  # uppercase hex
        "00000000-0000-0000-0000-00000000000Z",
    ],
)
def test_invalid_path_identifier_returns_422(client, path_id):
    response = _status(client, path_id)

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid challenge query"}


def test_missing_scope_params_return_422(client):
    challenge_id = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/v1/challenges/{challenge_id}").status_code == 422
    assert (
        client.get(
            f"/v1/challenges/{challenge_id}", params={"tenant_id": TENANT}
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/v1/challenges/{challenge_id}", params={"workload_id": WORKLOAD}
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "   ", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": "\t"},
    ],
)
def test_blank_scope_params_return_422(client, params):
    response = client.get(
        "/v1/challenges/00000000-0000-0000-0000-000000000000", params=params
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid challenge query"}


def test_unknown_query_parameter_returns_422(client):
    response = _status(
        client,
        "00000000-0000-0000-0000-000000000000",
        extra="1",
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid challenge query"}


def test_repeated_scope_parameter_returns_422(client):
    challenge_id = "00000000-0000-0000-0000-000000000000"
    response = client.get(
        f"/v1/challenges/{challenge_id}",
        params=[("tenant_id", TENANT), ("tenant_id", TENANT), ("workload_id", WORKLOAD)],
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid challenge query"}


def test_shape_errors_are_422_even_when_challenge_exists(client):
    created = _create(client).json()

    response = client.get(
        f"/v1/challenges/{created['challenge_id']}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD, "extra": "x"},
    )
    assert response.status_code == 422

    response = client.get(
        f"/v1/challenges/AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_non_empty_body_returns_422(client):
    created = _create(client).json()

    response = client.request(
        "GET",
        f"/v1/challenges/{created['challenge_id']}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"{}",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


# --- not found: 404 ----------------------------------------------------------


def test_unknown_challenge_returns_404(client):
    response = _status(client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json() == {"detail": "challenge not found"}


def test_cross_scope_challenge_returns_404(client):
    created = _create(client).json()

    other_tenant = _status(client, created["challenge_id"], tenant_id="tenant-b")
    other_workload = _status(
        client, created["challenge_id"], workload_id="workload-2"
    )

    assert other_tenant.status_code == 404
    assert other_tenant.json() == {"detail": "challenge not found"}
    assert other_workload.status_code == 404
    assert other_workload.json() == {"detail": "challenge not found"}


# --- server failure: 500 -----------------------------------------------------


def test_database_unreadable_returns_500(client, app):
    created = _create(client).json()
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE challenges"))

    response = _status(client, created["challenge_id"])

    assert response.status_code == 500
    assert response.content == b'{"detail":"challenge status unavailable"}'


# --- pure-read guarantees ----------------------------------------------------


def test_query_writes_nothing_in_any_table(client, app):
    created = _create(client).json()
    evidence = _attested_evidence(created["nonce"], {"measurement": "abc"})
    evidence_id = _submit(client, created, evidence).json()["evidence_id"]
    _verify(client, evidence_id, created, evidence)

    before = _all_table_counts(app)
    # Repeated queries across every settled phase create nothing.
    for _ in range(5):
        _status(client, created["challenge_id"])
    after = _all_table_counts(app)

    assert after == before


def test_query_consumes_no_issuance_budget(client):
    # Four of five issuance slots for this minute.
    created = [_create(client).json() for _ in range(4)]
    for _ in range(10):
        assert _status(client, created[0]["challenge_id"]).status_code == 200

    # The fifth issuance must still be admitted: queries spent no slots.
    fifth = _create(client)
    assert fifth.status_code == 201
    # And the budget is genuinely untouched: the sixth is now the one 429.
    assert _create(client).status_code == 429


def test_repeated_queries_against_same_state_are_byte_stable(client):
    created = _create(client).json()
    evidence = _attested_evidence(created["nonce"], {"measurement": "abc"})
    evidence_id = _submit(client, created, evidence).json()["evidence_id"]
    _verify(client, evidence_id, created, evidence)

    first = _status(client, created["challenge_id"]).content
    for _ in range(5):
        assert _status(client, created["challenge_id"]).content == first


def test_query_does_not_audit_or_emit_events(client, app):
    created = _create(client).json()
    _consume(client, created)

    before_events = _all_table_counts(app)
    _status(client, created["challenge_id"])
    _status(client, created["challenge_id"])
    after_events = _all_table_counts(app)

    assert after_events == before_events


# --- concurrency: first committer defines the read phase ---------------------


def test_concurrent_queries_observe_only_committed_phases(app):
    client = TestClient(app)
    created = _create(client).json()
    url = f"/v1/challenges/{created['challenge_id']}"
    params = {"tenant_id": TENANT, "workload_id": WORKLOAD}

    def query():
        response = TestClient(app).get(url, params=params)
        assert response.status_code == 200
        data = response.json()
        # Every observed body is internally consistent with one phase.
        if data["status"] == "pending":
            assert data["changed_at"] == data["issued_at"]
            assert data["evidence_id"] is None
        elif data["status"] == "consumed":
            assert data["evidence_id"] is None
        elif data["status"] == "evidence_received":
            assert data["evidence_id"] is not None
            assert data["verification_result"] is None
        else:
            pytest.fail(f"unexpected status under consume race: {data['status']}")
        return data["status"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for _ in range(8):
            futures.append(pool.submit(lambda: query()))
        # The single consumer commits at some point while reads are running.
        consume = pool.submit(lambda: _consume(TestClient(app), created))
        statuses = [future.result() for future in futures]
        assert consume.result().status_code == 200

    assert set(statuses) <= {"pending", "consumed"}
    # The settled state is now stable for every later query.
    final = _status(client, created["challenge_id"])
    assert final.json()["status"] == "consumed"


def test_concurrent_queries_observe_receive_then_verify_phases(app):
    client = TestClient(app)
    created = _create(client).json()
    evidence = _attested_evidence(created["nonce"], {"measurement": "abc"})
    submitted = _submit(client, created, evidence)
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]

    observed = set()

    def query():
        response = TestClient(app).get(
            f"/v1/challenges/{created['challenge_id']}",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
        observed.add(response.json()["status"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(query) for _ in range(8)]
        verify = pool.submit(lambda: _verify(TestClient(app), evidence_id, created, evidence))
        for future in futures:
            future.result()
        assert verify.result().status_code == 200

    assert observed <= {"evidence_received", "verified"}
    assert _status(client, created["challenge_id"]).json()["status"] == "verified"


def _rfc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()
