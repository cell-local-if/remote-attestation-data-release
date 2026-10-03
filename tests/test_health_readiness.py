"""Tests for GET /health/readiness.

A standalone readiness probe reporting whether the service can currently
carry proof, authorization and data-release traffic. It takes no input
(any query parameter or non-empty body is a 422 before any dependency is
touched), independently checks storage (one minimal read-only statement)
and the configured keyring (loaded with the existing configuration
semantics, never unwrapped, rotated, audited or rate-limited), and
reports 200/``ready`` only when both are ``ok``. The checks are read-only
and side-effect free, and failure detail never leaks into the response.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from proof_release.app import create_app
from proof_release.db import AuditEvent, RateLimitCounter
from proof_release.envelopes import b64url_encode

SECRET = "unit-test-secret"
MASTER_KEY = b64url_encode(b"0123456789abcdef0123456789abcdef")
READINESS_PATH = "/health/readiness"

READY_BODY = {"status": "ready", "checks": {"database": "ok", "keyring": "ok"}}


@pytest.fixture()
def db_dir(tmp_path):
    directory = tmp_path / "readiness-db"
    directory.mkdir()
    return directory


@pytest.fixture()
def app(db_dir, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{db_dir}/readiness.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _row_counts(app) -> tuple[int, int]:
    with app.state.session_factory() as session:
        audits = session.scalar(select(func.count()).select_from(AuditEvent))
        rate_rows = session.scalar(
            select(func.count()).select_from(RateLimitCounter)
        )
    return audits, rate_rows


def test_readiness_reports_ready(client) -> None:
    response = client.get(READINESS_PATH)

    assert response.status_code == 200
    assert response.json() == READY_BODY
    # The response carries exactly these two fields, in this order.
    assert list(response.json()) == ["status", "checks"]
    assert list(response.json()["checks"]) == ["database", "keyring"]


def test_readiness_repeated_calls_return_the_same_structure(client) -> None:
    first = client.get(READINESS_PATH)
    second = client.get(READINESS_PATH)

    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_readiness_rejects_any_query_parameter(client) -> None:
    response = client.get(READINESS_PATH, params={"tenant_id": "tenant-a"})

    assert response.status_code == 422
    assert response.json() == {"detail": "unsupported query parameter"}


def test_readiness_rejects_unknown_query_parameter_even_when_ready(client) -> None:
    response = client.get(READINESS_PATH, params={"verbose": "true"})

    assert response.status_code == 422
    assert response.json() == {"detail": "unsupported query parameter"}


@pytest.mark.parametrize("body", [b"{}", b"   ", b"not json"])
def test_readiness_rejects_any_non_empty_body(client, body) -> None:
    response = client.request("GET", READINESS_PATH, content=body)

    assert response.status_code == 422
    assert response.json() == {"detail": "query body must be empty"}


def test_readiness_shape_failures_never_touch_dependencies(
    client, monkeypatch
) -> None:
    # With the keyring configuration removed, a malformed request is still
    # a 422 — the dependency checks never run for a rejected shape.
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY")

    with_param = client.get(READINESS_PATH, params={"x": "1"})
    with_body = client.request("GET", READINESS_PATH, content=b"{}")

    assert with_param.status_code == 422
    assert with_param.json() == {"detail": "unsupported query parameter"}
    assert with_body.status_code == 422
    assert with_body.json() == {"detail": "query body must be empty"}


def test_readiness_keyring_unavailable(client, monkeypatch) -> None:
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY")

    response = client.get(READINESS_PATH)

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "ok", "keyring": "unavailable"},
    }


def test_readiness_keyring_malformed(client, monkeypatch) -> None:
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "{not json")

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


def test_readiness_database_unavailable(client, app, db_dir) -> None:
    # Make the database path unopenable (its directory is replaced by a
    # plain file) and drop pooled connections so the next check reconnects.
    app.state.engine.dispose()
    for entry in db_dir.iterdir():
        entry.unlink()
    db_dir.rmdir()
    db_dir.write_text("not a directory")

    response = client.get(READINESS_PATH)

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "unavailable", "keyring": "ok"},
    }


def test_readiness_database_recovers(client, app, db_dir) -> None:
    app.state.engine.dispose()
    for entry in db_dir.iterdir():
        entry.unlink()
    db_dir.rmdir()
    db_dir.write_text("not a directory")
    assert client.get(READINESS_PATH).status_code == 503

    # Restore the directory; the very next call is ready again and the
    # failed checks left nothing behind.
    db_dir.unlink()
    db_dir.mkdir()
    app.state.engine.dispose()

    response = client.get(READINESS_PATH)

    assert response.status_code == 200
    assert response.json() == READY_BODY


def test_readiness_both_dependencies_unavailable(client, app, db_dir, monkeypatch) -> None:
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY")
    app.state.engine.dispose()
    for entry in db_dir.iterdir():
        entry.unlink()
    db_dir.rmdir()
    db_dir.write_text("not a directory")

    response = client.get(READINESS_PATH)

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "unavailable", "keyring": "unavailable"},
    }


def test_readiness_leaves_no_side_effects(client, app, monkeypatch) -> None:
    assert _row_counts(app) == (0, 0)

    client.get(READINESS_PATH)
    client.get(READINESS_PATH, params={"x": "1"})
    client.request("GET", READINESS_PATH, content=b"{}")
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY")
    client.get(READINESS_PATH)

    # No audit event, no rate-limit reservation, no other state: the
    # checks are read-only even when dependencies fail.
    assert _row_counts(app) == (0, 0)


def test_readiness_response_never_carries_configuration_detail(
    client, monkeypatch
) -> None:
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING",
        json.dumps({"current_version": 7, "keys": {"7": "too-short"}}),
    )

    response = client.get(READINESS_PATH)

    assert response.status_code == 503
    body = response.text
    assert "7" not in body.replace('"checks"', "")
    assert "too-short" not in body
    assert "PROOF_RELEASE" not in body
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "ok", "keyring": "unavailable"},
    }
