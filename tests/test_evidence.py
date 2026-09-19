import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from proof_release.app import create_app
from proof_release.db import Challenge, Evidence


@pytest.fixture()
def app(tmp_path):
    return create_app(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, **overrides):
    body = {"tenant_id": "tenant-a", "workload_id": "workload-1"}
    body.update(overrides)
    return client.post("/v1/challenges", json=body)


def _submit(client, created, **overrides):
    body = {
        "tenant_id": "tenant-a",
        "workload_id": "workload-1",
        "challenge_id": created["challenge_id"],
        "nonce": created["nonce"],
        "evidence_format": "tpm-quote",
        "evidence": "cXVvdGU=",
    }
    body.update(overrides)
    return client.post("/v1/evidence", json=body)


def test_submit_evidence_success(client):
    created = _create(client).json()

    response = _submit(client, created)

    assert response.status_code == 201
    data = response.json()
    assert data["evidence_id"]
    assert data["challenge_id"] == created["challenge_id"]
    assert data["status"] == "received"
    received_at = datetime.fromisoformat(data["received_at"])
    assert received_at.tzinfo is not None
    assert received_at.utcoffset() == timedelta(0)


def test_submit_evidence_persists_digest_only(client, app):
    created = _create(client).json()
    evidence = "cXVvdGU="

    response = _submit(client, created, evidence=evidence)
    assert response.status_code == 201

    with app.state.session_factory() as session:
        record = session.get(Evidence, response.json()["evidence_id"])
        assert record is not None
        assert record.tenant_id == "tenant-a"
        assert record.workload_id == "workload-1"
        assert record.challenge_id == created["challenge_id"]
        assert record.evidence_format == "tpm-quote"
        assert record.status == "received"
        assert record.received_at.tzinfo is not None
        assert record.evidence_sha256 == hashlib.sha256(
            evidence.encode("utf-8")
        ).hexdigest()
        assert not hasattr(record, "evidence")
        # The challenge is consumed by the evidence submission.
        challenge = session.get(Challenge, created["challenge_id"])
        assert challenge.status == "consumed"
        assert challenge.consumed_at is not None


def test_submit_evidence_unknown_challenge_returns_404(client):
    created = {"challenge_id": "00000000-0000-0000-0000-000000000000", "nonce": "x"}

    response = _submit(client, created)

    assert response.status_code == 404


def test_submit_evidence_tenant_and_workload_mismatch_return_404(client):
    created = _create(client).json()

    assert _submit(client, created, tenant_id="tenant-b").status_code == 404
    assert _submit(client, created, workload_id="workload-2").status_code == 404


def test_submit_evidence_wrong_nonce_returns_401(client):
    created = _create(client).json()
    wrong = "A" + created["nonce"][1:]

    response = _submit(client, created, nonce=wrong)

    assert response.status_code == 401


def test_submit_evidence_twice_returns_409(client, app):
    created = _create(client).json()

    assert _submit(client, created).status_code == 201
    assert _submit(client, created).status_code == 409

    with app.state.session_factory() as session:
        count = len(session.scalars(select(Evidence)).all())
    assert count == 1


def test_submit_evidence_after_consume_returns_409(client):
    created = _create(client).json()
    consumed = client.post(
        f"/v1/challenges/{created['challenge_id']}/consume",
        json={
            "tenant_id": "tenant-a",
            "workload_id": "workload-1",
            "nonce": created["nonce"],
        },
    )
    assert consumed.status_code == 200

    assert _submit(client, created).status_code == 409


def test_consume_after_submit_evidence_returns_409(client):
    created = _create(client).json()
    assert _submit(client, created).status_code == 201

    response = client.post(
        f"/v1/challenges/{created['challenge_id']}/consume",
        json={
            "tenant_id": "tenant-a",
            "workload_id": "workload-1",
            "nonce": created["nonce"],
        },
    )

    assert response.status_code == 409


def test_submit_evidence_expired_challenge_returns_410(client, app):
    created = _create(client, ttl_seconds=30).json()
    with app.state.session_factory() as session:
        challenge = session.get(Challenge, created["challenge_id"])
        challenge.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = _submit(client, created)

    assert response.status_code == 410


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"workload_id": ""},
        {"challenge_id": ""},
        {"challenge_id": None},
        {"nonce": ""},
        {"nonce": "not base64!!!"},
        {"nonce": "with=padding"},
        {"evidence_format": ""},
        {"evidence_format": "  "},
        {"evidence": ""},
        {"evidence": 42},
    ],
)
def test_submit_evidence_rejects_invalid_fields(client, overrides):
    created = _create(client).json()

    response = _submit(client, created, **overrides)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "missing",
    [
        "tenant_id",
        "workload_id",
        "challenge_id",
        "nonce",
        "evidence_format",
        "evidence",
    ],
)
def test_submit_evidence_requires_all_fields(client, missing):
    created = _create(client).json()
    body = {
        "tenant_id": "tenant-a",
        "workload_id": "workload-1",
        "challenge_id": created["challenge_id"],
        "nonce": created["nonce"],
        "evidence_format": "tpm-quote",
        "evidence": "cXVvdGU=",
    }
    del body[missing]

    response = client.post("/v1/evidence", json=body)

    assert response.status_code == 422


def test_evidence_survives_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    created = _create(client1).json()
    submitted = _submit(client1, created)
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record is not None
        assert record.status == "received"
        assert record.challenge_id == created["challenge_id"]
        challenge = session.get(Challenge, created["challenge_id"])
        assert challenge.status == "consumed"


def test_concurrent_submit_only_one_creates_evidence(app):
    client = TestClient(app)
    created = _create(client).json()

    def submit():
        return TestClient(app).post(
            "/v1/evidence",
            json={
                "tenant_id": "tenant-a",
                "workload_id": "workload-1",
                "challenge_id": created["challenge_id"],
                "nonce": created["nonce"],
                "evidence_format": "tpm-quote",
                "evidence": "cXVvdGU=",
            },
        ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: submit(), range(16)))

    assert statuses.count(201) == 1
    assert statuses.count(409) == 15
    with app.state.session_factory() as session:
        count = len(session.scalars(select(Evidence)).all())
    assert count == 1
