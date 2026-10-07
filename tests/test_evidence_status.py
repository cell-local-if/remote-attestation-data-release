"""Tests for the read-only GET /v1/evidence/{evidence_id} status query."""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from proof_release.app import VERIFICATION_BUDGET_PER_MINUTE, create_app
from proof_release.db import Base, Challenge, Evidence

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


def _status(client, evidence_id, *, tenant_id=TENANT, workload_id=WORKLOAD, **params):
    query = {"tenant_id": tenant_id, "workload_id": workload_id}
    query.update(params)
    return client.get(f"/v1/evidence/{evidence_id}", params=query)


def _all_table_counts(app) -> dict[str, int]:
    counts = {}
    with app.state.engine.connect() as conn:
        for table_name in sorted(Base.metadata.tables):
            counts[table_name] = conn.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            ).scalar_one()
    return counts


@pytest.fixture()
def received(client):
    created = _create(client).json()
    evidence = _attested_evidence(created["nonce"], {"measurement": "abc"})
    submitted = _submit(client, created, evidence)
    assert submitted.status_code == 201
    return created, evidence, submitted.json()


# --- happy paths -------------------------------------------------------------


def test_received_evidence_status(received, client):
    created, evidence, submitted = received

    response = _status(client, submitted["evidence_id"])

    assert response.status_code == 200
    data = response.json()
    assert list(data.keys()) == EXPECTED_KEYS
    assert data["evidence_id"] == submitted["evidence_id"]
    assert data["challenge_id"] == created["challenge_id"]
    assert data["evidence_format"] == "attested-nonce-json"
    assert data["status"] == "received"
    assert data["received_at"] == submitted["received_at"]
    # Before the first verification the settlement fields are null.
    assert data["verified_at"] is None
    assert data["verification_result"] is None
    for field in ("received_at",):
        assert datetime.fromisoformat(data[field]).utcoffset() == timedelta(0)


def test_verified_evidence_status(client, received):
    created, evidence, submitted = received
    evidence_id = submitted["evidence_id"]
    verified = _verify(client, evidence_id, created, evidence).json()

    response = _status(client, evidence_id)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "verified"
    assert data["received_at"] == submitted["received_at"]
    assert data["verified_at"] == verified["verified_at"]
    assert data["verification_result"] == "accepted"
    assert (
        datetime.fromisoformat(data["verified_at"]).utcoffset() == timedelta(0)
    )


def test_rejected_evidence_status(client):
    created = _create(client).json()
    evidence = _attested_evidence(
        created["nonce"], {}, mac_override="0" * 64
    )
    evidence_id = _submit(client, created, evidence).json()["evidence_id"]
    rejected = _verify(client, evidence_id, created, evidence).json()
    assert rejected["status"] == "rejected"

    response = _status(client, evidence_id)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["verified_at"] == rejected["verified_at"]
    assert data["verification_result"] == "rejected"


def test_status_query_does_not_settle_received_evidence(client, received, app):
    _, _, submitted = received
    evidence_id = submitted["evidence_id"]

    assert _status(client, evidence_id).json()["status"] == "received"
    assert _status(client, evidence_id).json()["status"] == "received"

    # The query never writes a settlement back.
    with app.state.session_factory() as session:
        row = session.get(Evidence, evidence_id)
        assert row.status == "received"
        assert row.verified_at is None
        assert row.verification_result is None


def test_response_never_carries_secret_material(client, received):
    created, evidence, submitted = received
    _verify(client, submitted["evidence_id"], created, evidence)
    nonce_digest = hashlib.sha256(created["nonce"].encode("ascii")).hexdigest()
    evidence_digest = hashlib.sha256(evidence.encode("utf-8")).hexdigest()

    raw = _status(client, submitted["evidence_id"]).content.decode("utf-8")

    assert created["nonce"] not in raw
    assert nonce_digest not in raw
    assert evidence not in raw
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
    assert response.json() == {"detail": "invalid evidence query"}


def test_missing_scope_params_return_422(client):
    evidence_id = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/v1/evidence/{evidence_id}").status_code == 422
    assert (
        client.get(
            f"/v1/evidence/{evidence_id}", params={"tenant_id": TENANT}
        ).status_code
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
    response = _status(
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
        params=[
            ("tenant_id", TENANT),
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
        ],
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid evidence query"}

    response = client.get(
        f"/v1/evidence/{evidence_id}",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("workload_id", WORKLOAD),
        ],
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid evidence query"}


def test_shape_errors_are_422_even_when_evidence_exists(client, received):
    _, _, submitted = received

    response = client.get(
        f"/v1/evidence/{submitted['evidence_id']}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD, "extra": "x"},
    )
    assert response.status_code == 422

    response = client.get(
        "/v1/evidence/AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_non_empty_body_returns_422(client, received):
    _, _, submitted = received

    response = client.request(
        "GET",
        f"/v1/evidence/{submitted['evidence_id']}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"{}",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


# --- not found: 404 ----------------------------------------------------------


def test_unknown_evidence_returns_404(client):
    response = _status(client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json() == {"detail": "evidence not found"}


def test_cross_scope_evidence_returns_404(client, received):
    _, _, submitted = received

    other_tenant = _status(
        client, submitted["evidence_id"], tenant_id="tenant-b"
    )
    other_workload = _status(
        client, submitted["evidence_id"], workload_id="workload-2"
    )

    assert other_tenant.status_code == 404
    assert other_tenant.json() == {"detail": "evidence not found"}
    assert other_workload.status_code == 404
    assert other_workload.json() == {"detail": "evidence not found"}


def test_other_tenants_evidence_is_indistinguishable_from_unknown(client, received):
    # An existing evidence id queried cross-scope produces exactly the same
    # response as a never-used identifier.
    _, _, submitted = received
    existing = _status(
        client, submitted["evidence_id"], tenant_id="tenant-b"
    )
    missing = _status(client, "11111111-1111-1111-1111-111111111111")

    assert existing.status_code == missing.status_code == 404
    assert existing.content == missing.content


# --- server failure: 500 -----------------------------------------------------


def test_database_unreadable_returns_500(client, app, received):
    _, _, submitted = received
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE evidence"))

    response = _status(client, submitted["evidence_id"])

    assert response.status_code == 500
    assert response.content == b'{"detail":"evidence status unavailable"}'


# --- pure-read guarantees ----------------------------------------------------


def test_query_writes_nothing_in_any_table(client, app, received):
    created, evidence, submitted = received
    _verify(client, submitted["evidence_id"], created, evidence)

    before = _all_table_counts(app)
    for _ in range(5):
        _status(client, submitted["evidence_id"])
    after = _all_table_counts(app)

    assert after == before


def test_query_consumes_no_verification_budget(tmp_path):
    # A dedicated app with an always-accept verifier lets evidence be
    # seeded directly, so the independent challenge-issuance budget can
    # never interfere with the verification-budget assertion.
    from proof_release.verifiers import (
        VerificationContext,
        VerificationResult,
        Verifier,
        VerifierRegistry,
    )

    class AcceptVerifier(Verifier):
        format_name = "accept"

        def verify(self, context: VerificationContext) -> VerificationResult:
            return VerificationResult(accepted=True)

    registry = VerifierRegistry()
    registry.register(AcceptVerifier())
    application = create_app(
        f"sqlite:///{tmp_path}/budget.db",
        verifier_registry=registry,
    )
    client = TestClient(application)
    nonce = "test-nonce"
    evidence = "opaque-blob"

    def seed():
        # Bypasses the API: no issuance budget is spent.
        now = datetime.now(timezone.utc)
        challenge_id = str(uuid.uuid4())
        evidence_id = str(uuid.uuid4())
        with application.state.session_factory() as session:
            session.add(
                Challenge(
                    challenge_id=challenge_id,
                    tenant_id=TENANT,
                    workload_id=WORKLOAD,
                    nonce_digest=hashlib.sha256(nonce.encode("ascii")).hexdigest(),
                    status="consumed",
                    issued_at=now,
                    expires_at=now + timedelta(seconds=300),
                    consumed_at=now,
                )
            )
            session.add(
                Evidence(
                    evidence_id=evidence_id,
                    challenge_id=challenge_id,
                    tenant_id=TENANT,
                    workload_id=WORKLOAD,
                    evidence_format="accept",
                    status="received",
                    received_at=now,
                    evidence_sha256=hashlib.sha256(
                        evidence.encode("utf-8")
                    ).hexdigest(),
                )
            )
            session.commit()
        return evidence_id

    def verify(evidence_id):
        return client.post(
            f"/v1/evidence/{evidence_id}/verify",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "nonce": nonce,
                "evidence": evidence,
            },
        )

    # Fill budget-1 slots with real verifications.
    partial = [seed() for _ in range(VERIFICATION_BUDGET_PER_MINUTE - 1)]
    for evidence_id in partial:
        assert verify(evidence_id).status_code == 200

    # Status queries, against both received and settled evidence, spend no
    # budget at all.
    received_id = seed()
    for _ in range(10):
        assert _status(client, received_id).status_code == 200
        assert _status(client, partial[0]).status_code == 200

    # The final real verification must still be admitted: queries spent
    # nothing.
    assert verify(received_id).status_code == 200

    # The budget is genuinely exhausted now: the next real verify is 429,
    # while the read-only status query remains free.
    over = seed()
    assert verify(over).status_code == 429
    assert _status(client, over).status_code == 200
    application.state.engine.dispose()


def test_repeated_queries_against_same_state_are_byte_stable(client, received):
    created, evidence, submitted = received

    first = _status(client, submitted["evidence_id"]).content
    for _ in range(5):
        assert _status(client, submitted["evidence_id"]).content == first

    _verify(client, submitted["evidence_id"], created, evidence)
    settled = _status(client, submitted["evidence_id"]).content
    for _ in range(5):
        assert _status(client, submitted["evidence_id"]).content == settled


def test_query_does_not_audit_or_emit_events(client, app, received):
    before = _all_table_counts(app)
    _status(client, received[2]["evidence_id"])
    _status(client, received[2]["evidence_id"])
    after = _all_table_counts(app)

    assert after == before


# --- concurrency: first committed conclusion wins the read -------------------


def test_concurrent_queries_observe_only_committed_states(app, received):
    created, evidence, submitted = received
    evidence_id = submitted["evidence_id"]
    params = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    observed = set()

    def query():
        response = TestClient(app).get(
            f"/v1/evidence/{evidence_id}", params=params
        )
        assert response.status_code == 200
        data = response.json()
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
    final = TestClient(app).get(
        f"/v1/evidence/{evidence_id}", params=params
    )
    assert final.json()["status"] == "verified"
