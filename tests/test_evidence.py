import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

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


def _submit(client, challenge_id, nonce, **overrides):
    body = {
        "tenant_id": "tenant-a",
        "workload_id": "workload-1",
        "challenge_id": challenge_id,
        "nonce": nonce,
        "evidence_format": "tpm-quote",
        "evidence": "quote-binary-blob",
    }
    body.update(overrides)
    return client.post("/v1/evidence", json=body)


def test_submit_evidence_success(client):
    created = _create(client).json()

    response = _submit(client, created["challenge_id"], created["nonce"])

    assert response.status_code == 201
    data = response.json()
    assert data["challenge_id"] == created["challenge_id"]
    assert data["evidence_id"]
    assert data["status"] == "received"
    received = datetime.fromisoformat(data["received_at"])
    assert received.tzinfo is not None
    assert received.utcoffset() == timedelta(0)


def test_evidence_persisted_without_raw_payload(client, app):
    created = _create(client).json()
    raw = "secret-attestation-quote"

    response = _submit(client, created["challenge_id"], created["nonce"], evidence=raw)
    evidence_id = response.json()["evidence_id"]

    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record is not None
        assert record.tenant_id == "tenant-a"
        assert record.workload_id == "workload-1"
        assert record.challenge_id == created["challenge_id"]
        assert record.evidence_format == "tpm-quote"
        assert record.status == "received"
        assert record.evidence_digest == hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert record.received_at.tzinfo is not None
        # Raw evidence must never be persisted.
        columns = {
            c.name: getattr(record, c.name) for c in record.__table__.columns
        }
        assert raw not in {str(v) for v in columns.values() if v is not None}
        challenge = session.get(Challenge, created["challenge_id"])
        assert challenge.status == "consumed"
        assert challenge.consumed_at is not None


def test_submit_unknown_challenge_returns_404(client):
    response = _submit(
        client, "00000000-0000-0000-0000-000000000000", "A" * 43
    )
    assert response.status_code == 404


def test_submit_tenant_and_workload_mismatch_return_404(client):
    created = _create(client).json()

    other_tenant = _submit(
        client, created["challenge_id"], created["nonce"], tenant_id="tenant-b"
    )
    assert other_tenant.status_code == 404

    other_workload = _submit(
        client, created["challenge_id"], created["nonce"], workload_id="workload-2"
    )
    assert other_workload.status_code == 404


def test_submit_wrong_nonce_returns_401(client):
    created = _create(client).json()
    wrong = "A" + created["nonce"][1:]

    response = _submit(client, created["challenge_id"], wrong)

    assert response.status_code == 401


def test_submit_twice_second_is_409(client):
    created = _create(client).json()

    first = _submit(client, created["challenge_id"], created["nonce"])
    assert first.status_code == 201

    second = _submit(client, created["challenge_id"], created["nonce"])
    assert second.status_code == 409


def test_consume_blocks_evidence_and_vice_versa(client):
    # Consumed first -> evidence rejected.
    consumed_first = _create(client).json()
    consume = client.post(
        f"/v1/challenges/{consumed_first['challenge_id']}/consume",
        json={
            "tenant_id": "tenant-a",
            "workload_id": "workload-1",
            "nonce": consumed_first["nonce"],
        },
    )
    assert consume.status_code == 200
    after_consume = _submit(
        client, consumed_first["challenge_id"], consumed_first["nonce"]
    )
    assert after_consume.status_code == 409

    # Evidence first -> consume rejected.
    evidence_first = _create(client).json()
    submitted = _submit(
        client, evidence_first["challenge_id"], evidence_first["nonce"]
    )
    assert submitted.status_code == 201
    after_evidence = client.post(
        f"/v1/challenges/{evidence_first['challenge_id']}/consume",
        json={
            "tenant_id": "tenant-a",
            "workload_id": "workload-1",
            "nonce": evidence_first["nonce"],
        },
    )
    assert after_evidence.status_code == 409


def test_submit_expired_challenge_returns_410(client, app):
    created = _create(client, ttl_seconds=30).json()
    with app.state.session_factory() as session:
        challenge = session.get(Challenge, created["challenge_id"])
        challenge.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = _submit(client, created["challenge_id"], created["nonce"])

    assert response.status_code == 410


@pytest.mark.parametrize(
    "field",
    ["tenant_id", "workload_id", "challenge_id", "nonce", "evidence_format", "evidence"],
)
def test_submit_missing_fields_return_422(client, field):
    created = _create(client).json()
    body = {
        "tenant_id": "tenant-a",
        "workload_id": "workload-1",
        "challenge_id": created["challenge_id"],
        "nonce": created["nonce"],
        "evidence_format": "tpm-quote",
        "evidence": "blob",
    }
    del body[field]

    response = client.post("/v1/evidence", json=body)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_id", "   "),
        ("workload_id", ""),
        ("challenge_id", " "),
        ("nonce", "not base64!!!"),
        ("nonce", "with=padding"),
        ("evidence_format", "   "),
        ("evidence", ""),
        ("evidence", "   "),
        ("tenant_id", 1),
        ("evidence", 123),
    ],
)
def test_submit_invalid_fields_return_422(client, field, value):
    created = _create(client).json()
    body = {
        "tenant_id": "tenant-a",
        "workload_id": "workload-1",
        "challenge_id": created["challenge_id"],
        "nonce": created["nonce"],
        "evidence_format": "tpm-quote",
        "evidence": "blob",
    }
    body[field] = value

    response = client.post("/v1/evidence", json=body)

    # A malformed nonce must be rejected before the challenge lookup.
    assert response.status_code == 422


def test_evidence_survives_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    created = client1.post(
        "/v1/challenges", json={"tenant_id": "t", "workload_id": "w"}
    ).json()
    submitted = client1.post(
        "/v1/evidence",
        json={
            "tenant_id": "t",
            "workload_id": "w",
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "fmt",
            "evidence": "blob",
        },
    )
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record is not None
        assert record.challenge_id == created["challenge_id"]
        assert record.status == "received"
        assert record.evidence_digest == hashlib.sha256(b"blob").hexdigest()
        challenge = session.get(Challenge, created["challenge_id"])
        assert challenge.status == "consumed"


def test_concurrent_evidence_only_one_succeeds(app):
    client = TestClient(app)
    created = _create(client).json()
    body = {
        "tenant_id": "tenant-a",
        "workload_id": "workload-1",
        "challenge_id": created["challenge_id"],
        "nonce": created["nonce"],
        "evidence_format": "tpm-quote",
        "evidence": "blob",
    }

    def submit():
        return TestClient(app).post("/v1/evidence", json=body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: submit(), range(16)))

    assert statuses.count(201) == 1
    assert statuses.count(409) == 15
    with app.state.session_factory() as session:
        assert session.query(Evidence).count() == 1
