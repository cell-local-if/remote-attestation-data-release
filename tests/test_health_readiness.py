"""Tests for GET /health/readiness.

A standalone readiness probe: every call independently checks storage
(one minimal read-only statement) and the configured master keyring
(loaded with the usual configuration semantics, never used to unwrap,
rotate, audit or rate-limit). Both ok -> 200 ``ready``; any failure ->
503 ``not_ready`` with each dependency still reported on its own. Shape
rejections (any query parameter, any body) are 422 and never trigger a
dependency check. The probe is read-only and stateless: a recovered
dependency flips the next response back to ready.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    ChallengeIssuanceCounter,
    ProofLifecycleEvent,
    RateLimitCounter,
    ReleaseGrantEvent,
    RewrapJobEvent,
)
from proof_release.envelopes import b64url_encode

SECRET = "unit-test-secret"
MASTER_KEY = b64url_encode(b"0123456789abcdef0123456789abcdef")
READINESS_PATH = "/health/readiness"

READY_BODY = {"status": "ready", "checks": {"database": "ok", "keyring": "ok"}}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/readiness.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _break_database(app):
    """Force every new engine connection to fail until the patch exits."""
    return patch.object(
        app.state.engine,
        "connect",
        side_effect=OperationalError("forced outage", {}, None),
    )


def test_readiness_ready(client) -> None:
    response = client.get(READINESS_PATH)

    assert response.status_code == 200
    assert response.json() == READY_BODY


def test_readiness_repeated_calls_return_same_structure(client) -> None:
    first = client.get(READINESS_PATH)
    second = client.get(READINESS_PATH)

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == READY_BODY
    assert first.content == second.content


def test_readiness_rejects_any_query_parameter(client, monkeypatch) -> None:
    # Even with a broken keyring the shape rejection wins: no dependency
    # check is ever triggered by a rejected request.
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY")

    for params in (
        {"probe": "1"},
        {"tenant_id": "t"},
        {"probe": ""},
    ):
        response = client.get(READINESS_PATH, params=params)
        assert response.status_code == 422
        assert response.json() == {"detail": "unsupported query parameter"}


def test_readiness_rejects_any_body(client, monkeypatch) -> None:
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY")

    for kwargs in (
        {"json": {"anything": "goes"}},
        {"content": "   \n\t "},
        {"content": "not json at all"},
    ):
        response = client.request("GET", READINESS_PATH, **kwargs)
        assert response.status_code == 422
        assert response.json() == {"detail": "query body must be empty"}


def test_readiness_keyring_missing(client, monkeypatch) -> None:
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY")

    response = client.get(READINESS_PATH)

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "ok", "keyring": "unavailable"},
    }


def test_readiness_keyring_malformed(client, monkeypatch) -> None:
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")

    response = client.get(READINESS_PATH)

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "ok", "keyring": "unavailable"},
    }


def test_readiness_keyring_recovers(client, monkeypatch) -> None:
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY")
    assert client.get(READINESS_PATH).status_code == 503

    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    response = client.get(READINESS_PATH)

    assert response.status_code == 200
    assert response.json() == READY_BODY


def test_readiness_database_unavailable(app, client) -> None:
    with _break_database(app):
        response = client.get(READINESS_PATH)

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "unavailable", "keyring": "ok"},
    }


def test_readiness_database_recovers(app, client) -> None:
    with _break_database(app):
        assert client.get(READINESS_PATH).status_code == 503

    response = client.get(READINESS_PATH)

    assert response.status_code == 200
    assert response.json() == READY_BODY


def test_readiness_both_dependencies_unavailable(app, client, monkeypatch) -> None:
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY")
    with _break_database(app):
        response = client.get(READINESS_PATH)

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "unavailable", "keyring": "unavailable"},
    }


def test_readiness_response_never_carries_details(app, client, monkeypatch) -> None:
    # Neither the key material nor any configuration value may appear in
    # a failure response.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", json.dumps({"bogus": "shape"}))
    with _break_database(app):
        response = client.get(READINESS_PATH)

    assert response.status_code == 503
    assert set(response.json()) == {"status", "checks"}
    assert set(response.json()["checks"]) == {"database", "keyring"}
    assert MASTER_KEY not in response.text
    assert "bogus" not in response.text


def test_readiness_leaves_no_state_behind(app, client, monkeypatch) -> None:
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY")
    with _break_database(app):
        client.get(READINESS_PATH)
    client.get(READINESS_PATH)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    client.get(READINESS_PATH)

    # Successful and failed probes alike write nothing: no audit or
    # lifecycle event and no rate-limit counter of any kind.
    with app.state.session_factory() as session:
        for model in (
            AuditEvent,
            ProofLifecycleEvent,
            ReleaseGrantEvent,
            RewrapJobEvent,
            RateLimitCounter,
            ChallengeIssuanceCounter,
        ):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_health_liveness_unchanged(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
