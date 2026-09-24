import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Challenge


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


def _consume(client, challenge_id, **overrides):
    body = {
        "tenant_id": "tenant-a",
        "workload_id": "workload-1",
        "nonce": "placeholder",
    }
    body.update(overrides)
    return client.post(f"/v1/challenges/{challenge_id}/consume", json=body)


def test_create_challenge_returns_pending_with_nonce(client):
    response = _create(client)

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "pending"
    assert data["challenge_id"]
    # 32 random bytes -> 43 chars of unpadded base64url.
    assert len(data["nonce"]) == 43
    assert "=" not in data["nonce"]
    issued = datetime.fromisoformat(data["issued_at"])
    expires = datetime.fromisoformat(data["expires_at"])
    assert issued.tzinfo is not None
    assert expires.tzinfo is not None
    assert (expires - issued) == timedelta(seconds=300)


def test_create_challenge_custom_ttl(client):
    response = _create(client, ttl_seconds=60)

    assert response.status_code == 201
    data = response.json()
    delta = datetime.fromisoformat(data["expires_at"]) - datetime.fromisoformat(
        data["issued_at"]
    )
    assert delta == timedelta(seconds=60)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"tenant_id": "", "workload_id": "w"},
        {"tenant_id": "   ", "workload_id": "w"},
        {"tenant_id": "t"},
        {"tenant_id": "t", "workload_id": ""},
        {"tenant_id": 1, "workload_id": "w"},
        {"tenant_id": "t", "workload_id": "w", "ttl_seconds": 29},
        {"tenant_id": "t", "workload_id": "w", "ttl_seconds": 901},
        {"tenant_id": "t", "workload_id": "w", "ttl_seconds": "300"},
        {"tenant_id": "t", "workload_id": "w", "ttl_seconds": 300.5},
        {"tenant_id": "t", "workload_id": "w", "ttl_seconds": True},
    ],
)
def test_create_challenge_rejects_invalid_fields(client, body):
    response = client.post("/v1/challenges", json=body)

    assert response.status_code == 422


def test_nonce_is_not_stored_in_plaintext(client, app):
    response = _create(client)
    nonce = response.json()["nonce"]

    with app.state.session_factory() as session:
        challenge = session.get(Challenge, response.json()["challenge_id"])
        assert challenge.nonce_digest == hashlib.sha256(nonce.encode("ascii")).hexdigest()
        assert nonce not in (
            challenge.challenge_id,
            challenge.tenant_id,
            challenge.workload_id,
            challenge.nonce_digest,
            challenge.status,
        )


def test_consume_success_then_conflict(client):
    created = _create(client).json()

    response = _consume(client, created["challenge_id"], nonce=created["nonce"])

    assert response.status_code == 200
    data = response.json()
    assert data["challenge_id"] == created["challenge_id"]
    assert data["status"] == "consumed"
    assert datetime.fromisoformat(data["consumed_at"]).tzinfo is not None

    second = _consume(client, created["challenge_id"], nonce=created["nonce"])
    assert second.status_code == 409


def test_consume_wrong_nonce_returns_401(client):
    created = _create(client).json()
    wrong = ("B" if created["nonce"][0] != "B" else "C") + created["nonce"][1:]

    response = _consume(client, created["challenge_id"], nonce=wrong)

    assert response.status_code == 401


def test_consume_unknown_challenge_returns_404(client):
    response = _consume(client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404


def test_consume_tenant_and_workload_mismatch_return_404(client):
    created = _create(client).json()

    other_tenant = _consume(
        client, created["challenge_id"], tenant_id="tenant-b", nonce=created["nonce"]
    )
    assert other_tenant.status_code == 404

    other_workload = _consume(
        client, created["challenge_id"], workload_id="workload-2", nonce=created["nonce"]
    )
    assert other_workload.status_code == 404


@pytest.mark.parametrize("nonce", ["", "not base64!!!", "with=padding"])
def test_consume_rejects_malformed_nonce(client, nonce):
    created = _create(client).json()

    response = _consume(client, created["challenge_id"], nonce=nonce)

    assert response.status_code == 422


def test_consume_expired_challenge_returns_410(client, app):
    created = _create(client, ttl_seconds=30).json()
    with app.state.session_factory() as session:
        challenge = session.get(Challenge, created["challenge_id"])
        challenge.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = _consume(client, created["challenge_id"], nonce=created["nonce"])

    assert response.status_code == 410


def test_data_survives_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    created = TestClient(app1).post(
        "/v1/challenges", json={"tenant_id": "t", "workload_id": "w"}
    ).json()
    app1.state.engine.dispose()

    app2 = create_app(url)
    response = TestClient(app2).post(
        f"/v1/challenges/{created['challenge_id']}/consume",
        json={"tenant_id": "t", "workload_id": "w", "nonce": created["nonce"]},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "consumed"


def test_concurrent_consume_only_one_succeeds(app):
    client = TestClient(app)
    created = _create(client).json()
    url = f"/v1/challenges/{created['challenge_id']}/consume"
    body = {
        "tenant_id": "tenant-a",
        "workload_id": "workload-1",
        "nonce": created["nonce"],
    }

    def consume():
        return TestClient(app).post(url, json=body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: consume(), range(16)))

    assert statuses.count(200) == 1
    assert statuses.count(409) == 15
