"""Tests for the persistent challenge endpoints."""

from __future__ import annotations

import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Challenge


@pytest.fixture()
def client(tmp_path):
    app = create_app(f"sqlite:///{tmp_path}/test.db")
    return TestClient(app)


def _create(client, tenant="tenant-a", workload="workload-1", **extra):
    body = {"tenant_id": tenant, "workload_id": workload, **extra}
    return client.post("/v1/challenges", json=body)


def _consume(client, challenge_id, tenant="tenant-a", workload="workload-1", nonce=""):
    body = {"tenant_id": tenant, "workload_id": workload, "nonce": nonce}
    return client.post(f"/v1/challenges/{challenge_id}/consume", json=body)


def test_create_challenge_returns_pending_with_nonce(client):
    response = _create(client)

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "pending"
    assert set(data) == {"challenge_id", "nonce", "issued_at", "expires_at", "status"}

    # Nonce is unpadded base64url of at least 32 random bytes.
    raw = base64.urlsafe_b64decode(data["nonce"] + "=" * (-len(data["nonce"]) % 4))
    assert len(raw) >= 32
    assert "=" not in data["nonce"]

    issued = datetime.fromisoformat(data["issued_at"].replace("Z", "+00:00"))
    expires = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
    assert (expires - issued) == timedelta(seconds=300)


def test_create_challenge_custom_ttl(client):
    response = _create(client, ttl_seconds=60)

    assert response.status_code == 201
    data = response.json()
    issued = datetime.fromisoformat(data["issued_at"].replace("Z", "+00:00"))
    expires = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
    assert (expires - issued) == timedelta(seconds=60)


@pytest.mark.parametrize(
    "body",
    [
        {"workload_id": "w"},  # missing tenant_id
        {"tenant_id": "t"},  # missing workload_id
        {"tenant_id": "", "workload_id": "w"},  # empty tenant_id
        {"tenant_id": "t", "workload_id": ""},  # empty workload_id
        {"tenant_id": "t", "workload_id": "w", "ttl_seconds": 29},
        {"tenant_id": "t", "workload_id": "w", "ttl_seconds": 901},
        {"tenant_id": "t", "workload_id": "w", "ttl_seconds": "abc"},
    ],
)
def test_create_challenge_validation_errors(client, body):
    assert client.post("/v1/challenges", json=body).status_code == 422


def test_plaintext_nonce_is_not_persisted(client):
    response = _create(client)
    data = response.json()

    session_factory = client.app.state.session_factory
    with session_factory() as session:
        challenge = session.get(Challenge, data["challenge_id"])
    assert challenge is not None
    assert challenge.nonce_hash != data["nonce"]
    raw = base64.urlsafe_b64decode(data["nonce"] + "=" * (-len(data["nonce"]) % 4))
    assert challenge.nonce_hash == hashlib.sha256(raw).hexdigest()


def test_consume_success(client):
    created = _create(client).json()

    response = _consume(client, created["challenge_id"], nonce=created["nonce"])

    assert response.status_code == 200
    data = response.json()
    assert data["challenge_id"] == created["challenge_id"]
    assert data["status"] == "consumed"
    datetime.fromisoformat(data["consumed_at"].replace("Z", "+00:00"))


def test_consume_twice_returns_409(client):
    created = _create(client).json()
    first = _consume(client, created["challenge_id"], nonce=created["nonce"])
    assert first.status_code == 200

    second = _consume(client, created["challenge_id"], nonce=created["nonce"])
    assert second.status_code == 409


def test_consume_unknown_challenge_returns_404(client):
    response = _consume(client, "no-such-challenge", nonce=_b64url(b"nonce"))
    assert response.status_code == 404


def test_consume_wrong_tenant_or_workload_returns_404(client):
    created = _create(client).json()

    wrong_tenant = _consume(client, created["challenge_id"], tenant="tenant-b", nonce=created["nonce"])
    assert wrong_tenant.status_code == 404

    wrong_workload = _consume(client, created["challenge_id"], workload="workload-2", nonce=created["nonce"])
    assert wrong_workload.status_code == 404


def test_consume_wrong_nonce_returns_401(client):
    created = _create(client).json()

    response = _consume(client, created["challenge_id"], nonce=_b64url(b"wrong-nonce-value-0000000000000000"))
    assert response.status_code == 401


def test_consume_malformed_nonce_returns_422(client):
    created = _create(client).json()

    response = _consume(client, created["challenge_id"], nonce="not!base64url!")
    assert response.status_code == 422


def test_consume_expired_challenge_returns_410(client):
    created = _create(client, ttl_seconds=30).json()

    session_factory = client.app.state.session_factory
    with session_factory() as session:
        challenge = session.get(Challenge, created["challenge_id"])
        challenge.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = _consume(client, created["challenge_id"], nonce=created["nonce"])
    assert response.status_code == 410


def test_challenge_survives_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"

    first_app = create_app(url)
    created = _create(TestClient(first_app)).json()

    # Simulate a process restart: a brand-new app instance on the same file.
    second_app = create_app(url)
    client = TestClient(second_app)
    response = _consume(client, created["challenge_id"], nonce=created["nonce"])

    assert response.status_code == 200
    assert response.json()["status"] == "consumed"


def test_concurrent_consume_only_one_wins(client):
    created = _create(client).json()

    def attempt(_):
        # One client per thread; all share the same app and database.
        with TestClient(client.app) as thread_client:
            return _consume(thread_client, created["challenge_id"], nonce=created["nonce"]).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(16)))

    assert results.count(200) == 1
    assert results.count(409) == 15


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
