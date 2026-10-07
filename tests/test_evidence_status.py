"""Tests for the read-only GET /v1/evidence/{evidence_id} status query."""

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
from proof_release.db import Base, Evidence

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"

EXPECTED_KEYS = [
    "evidence_id",
    "challenge_id",
    "evidence_format",
    "status",
    "received_at",
    "verified_at",
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


def _query(client, evidence_id, *, tenant_id=TENANT, workload_id=WORKLOAD, **params):
    query = {"tenant_id": tenant_id, "workload_id": workload_id}
    query.update(params)
    return client.get(f"/v1/evidence/{evidence_id}", params=query)


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


def _attested_evidence(nonce: str, claims: dict | None = None, *, mac_override=None):
    mac = mac_override or _mac(nonce, claims or {})
    return json.dumps({"nonce": nonce, "claims": claims or {}, "mac": mac})


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


def _submitted(client, evidence=None, **create_overrides):
    """Create a challenge and submit evidence for it; return (created, submitted, evidence)."""
    created = _create(client, **create_overrides).json()
    if evidence is None:
        evidence = _attested_evidence(created["nonce"], {"measurement": "abc"})
    submitted = _submit(client, created, evidence).json()
    return created, submitted, evidence


def _all_table_counts(app) -> dict[str, int]:
    counts = {}
    with app.state.engine.connect() as conn:
        for table_name in sorted(Base.metadata.tables):
            counts[table_name] = conn.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            ).scalar_one()
    return counts


# --- happy paths: received / verified / rejected ------------------------------


def test_received_evidence_status(client):
    created, submitted, _ = _submitted(client)

    response = _query(client, submitted["evidence_id"])

    assert response.status_code == 200
    data = response.json()
    assert list(data.keys()) == EXPECTED_KEYS
    assert data["evidence_id"] == submitted["evidence_id"]
    assert data["challenge_id"] == created["challenge_id"]
    assert data["evidence_format"] == "attested-nonce-json"
    assert data["status"] == "received"
    assert data["received_at"] == submitted["received_at"]
    # Not yet settled: both settlement fields are null.
    assert data["verified_at"] is None
    assert data["verification_result"] is None
    assert datetime.fromisoformat(data["received_at"]).utcoffset() == timedelta(0)


def test_verified_evidence_status(client):
    created, submitted, evidence = _submitted(client)
    verified = _verify(client, submitted["evidence_id"], created, evidence).json()
    assert verified["status"] == "verified"

    response = _query(client, submitted["evidence_id"])

    assert response.status_code == 200
    data = response.json()
    assert list(data.keys()) == EXPECTED_KEYS
    assert data["status"] == "verified"
    assert data["verified_at"] == verified["verified_at"]
    assert data["verification_result"] == "accepted"
    assert datetime.fromisoformat(data["verified_at"]).utcoffset() == timedelta(0)


def test_rejected_evidence_status(client):
    created = _create(client).json()
    evidence = _attested_evidence(created["nonce"], {}, mac_override="0" * 64)
    submitted = _submit(client, created, evidence).json()
    rejected = _verify(client, submitted["evidence_id"], created, evidence).json()
    assert rejected["status"] == "rejected"

    response = _query(client, submitted["evidence_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["verified_at"] == rejected["verified_at"]
    assert data["verification_result"] == "rejected"


def test_query_reports_first_settlement_after_repeat_verify(client):
    created, submitted, evidence = _submitted(client)
    first = _verify(client, submitted["evidence_id"], created, evidence).json()
    # A repeated verification returns the stored conclusion; the query must
    # report that same first settlement.
    _verify(client, submitted["evidence_id"], created, evidence)

    data = _query(client, submitted["evidence_id"]).json()

    assert data["status"] == "verified"
    assert data["verified_at"] == first["verified_at"]
    assert data["verification_result"] == "accepted"


def test_response_never_carries_secret_material(client):
    created, submitted, evidence_text = _submitted(client)
    nonce_digest = hashlib.sha256(created["nonce"].encode("ascii")).hexdigest()
    evidence_digest = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()

    raw = _query(client, submitted["evidence_id"]).content.decode("utf-8")

    assert created["nonce"] not in raw
    assert nonce_digest not in raw
    assert evidence_text not in raw
    assert evidence_digest not in raw


# --- request shape: 422 --------------------------------------------------------


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
    response = _query(client, path_id)

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid evidence query"}


def test_missing_scope_params_return_422(client):
    evidence_id = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/v1/evidence/{evidence_id}").status_code == 422
    assert (
        client.get(f"/v1/evidence/{evidence_id}", params={"tenant_id": TENANT}).status_code
        == 422
    )
    assert (
        client.get(
            f"/v1/evidence/{evidence_id}", params={"workload_id": WORKLOAD}
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
        "/v1/evidence/00000000-0000-0000-0000-000000000000", params=params
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid evidence query"}


def test_unknown_query_parameter_returns_422(client):
    response = _query(
        client,
        "00000000-0000-0000-0000-000000000000",
        extra="1",
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid evidence query"}


def test_repeated_scope_parameter_returns_422(client):
    evidence_id = "00000000-0000-0000-0000-000000000000"
    response = client.get(
        f"/v1/evidence/{evidence_id}",
        params=[("tenant_id", TENANT), ("tenant_id", TENANT), ("workload_id", WORKLOAD)],
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid evidence query"}


def test_shape_errors_are_422_even_when_evidence_exists(client):
    _, submitted, _ = _submitted(client)

    response = client.get(
        f"/v1/evidence/{submitted['evidence_id']}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD, "extra": "x"},
    )
    assert response.status_code == 422

    response = client.get(
        f"/v1/evidence/{submitted['evidence_id'].upper()}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_non_empty_body_returns_422(client):
    _, submitted, _ = _submitted(client)

    response = client.request(
        "GET",
        f"/v1/evidence/{submitted['evidence_id']}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"{}",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


# --- not found: 404 ------------------------------------------------------------


def test_unknown_evidence_returns_404(client):
    response = _query(client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json() == {"detail": "evidence not found"}


def test_cross_scope_evidence_returns_404(client):
    _, submitted, _ = _submitted(client)

    other_tenant = _query(client, submitted["evidence_id"], tenant_id="tenant-b")
    other_workload = _query(client, submitted["evidence_id"], workload_id="workload-2")

    assert other_tenant.status_code == 404
    assert other_tenant.json() == {"detail": "evidence not found"}
    assert other_workload.status_code == 404
    assert other_workload.json() == {"detail": "evidence not found"}


def test_cross_scope_404_is_indistinguishable_from_unknown(client):
    _, submitted, _ = _submitted(client)

    cross_scope = _query(client, submitted["evidence_id"], tenant_id="tenant-b")
    unknown = _query(
        client, "00000000-0000-0000-0000-000000000000", tenant_id="tenant-b"
    )

    assert cross_scope.status_code == unknown.status_code == 404
    assert cross_scope.content == unknown.content


# --- server failure: 500 -------------------------------------------------------


def test_database_unreadable_returns_500(client, app):
    _, submitted, _ = _submitted(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE evidence"))

    response = _query(client, submitted["evidence_id"])

    assert response.status_code == 500
    assert response.content == b'{"detail":"evidence status unavailable"}'


# --- pure-read guarantees ------------------------------------------------------


def test_query_writes_nothing_in_any_table(client, app):
    created, submitted, evidence = _submitted(client)
    _verify(client, submitted["evidence_id"], created, evidence)

    before = _all_table_counts(app)
    # Repeated queries across the settled state create nothing.
    for _ in range(5):
        _query(client, submitted["evidence_id"])
    after = _all_table_counts(app)

    assert after == before


def test_query_does_not_mutate_evidence_row(client, app):
    _, submitted, _ = _submitted(client)

    assert _query(client, submitted["evidence_id"]).status_code == 200
    assert _query(client, submitted["evidence_id"]).status_code == 200

    with app.state.session_factory() as session:
        row = session.get(Evidence, submitted["evidence_id"])
        assert row.status == "received"
        assert row.verified_at is None
        assert row.verification_result is None


def test_query_consumes_no_verification_budget(client):
    created, submitted, evidence = _submitted(client)
    for _ in range(10):
        assert _query(client, submitted["evidence_id"]).status_code == 200

    # The first verification must still be admitted: queries spent no slots.
    response = _verify(client, submitted["evidence_id"], created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_repeated_queries_against_same_state_are_byte_stable(client):
    created, submitted, evidence = _submitted(client)
    _verify(client, submitted["evidence_id"], created, evidence)

    first = _query(client, submitted["evidence_id"]).content
    for _ in range(5):
        assert _query(client, submitted["evidence_id"]).content == first


# --- concurrency: first committer defines the read state -----------------------


def test_concurrent_queries_observe_only_committed_states(app):
    client = TestClient(app)
    created = _create(client).json()
    evidence = _attested_evidence(created["nonce"], {"measurement": "abc"})
    submitted = _submit(client, created, evidence)
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]

    observed = set()

    def query():
        response = TestClient(app).get(
            f"/v1/evidence/{evidence_id}",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
        assert response.status_code == 200
        data = response.json()
        # Every observed body is internally consistent with one state.
        if data["status"] == "received":
            assert data["verified_at"] is None
            assert data["verification_result"] is None
        elif data["status"] == "verified":
            assert data["verified_at"] is not None
            assert data["verification_result"] == "accepted"
        else:
            pytest.fail(f"unexpected status under verify race: {data['status']}")
        observed.add(data["status"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(query) for _ in range(8)]
        verify = pool.submit(
            lambda: _verify(TestClient(app), evidence_id, created, evidence)
        )
        for future in futures:
            future.result()
        assert verify.result().status_code == 200

    assert observed <= {"received", "verified"}
    # The settled state is now stable for every later query.
    final = _query(client, evidence_id)
    assert final.json()["status"] == "verified"
