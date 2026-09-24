"""Tests for POST /v1/release-grants/{grant_id}/revoke."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import ReleaseGrant
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
MASTER_KEY_BYTES = b"0123456789abcdef0123456789abcdef"
MASTER_KEY = b64url_encode(MASTER_KEY_BYTES)
PAYLOAD = "revocation-test payload"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/revoke.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


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


def _evidence(nonce: str) -> str:
    claims = {"m": "x"}
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)}
    )


def _decision(client):
    """Drive the full challenge -> evidence -> verify -> decision flow."""
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    evidence = _evidence(created["nonce"])
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    verified = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.status_code == 200
    policy = client.post(
        "/v1/policies",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "name": "release",
            "rule": {"claim": "m", "equals": "x"},
        },
    ).json()
    decided = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert decided.status_code == 200
    return decided.json()["decision_id"]


def _envelope(client, data_id=DATA_ID, payload=PAYLOAD):
    response = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": data_id,
            "payload": payload,
        },
    )
    assert response.status_code == 201
    return response.json()


def _grant(client, decision_id, data_id=DATA_ID, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision_id,
        "data_id": data_id,
    }
    body.update(fields)
    response = client.post("/v1/release-grants", json=body)
    assert response.status_code == 201
    return response.json()


def _setup(client, data_id=DATA_ID):
    decision_id = _decision(client)
    _envelope(client, data_id=data_id)
    return _grant(client, decision_id, data_id=data_id)


def _revoke(client, grant_id, capability, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release-grants/{grant_id}/revoke", json=body)


def _consume(client, grant_id, capability):
    return client.post(
        f"/v1/release-grants/{grant_id}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": capability,
        },
    )


def _release(client, grant_id, capability, data_id=DATA_ID):
    return client.post(
        f"/v1/release/{grant_id}",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": data_id,
            "capability": capability,
        },
    )


def test_revoke_success_returns_exact_fields(client, app):
    grant = _setup(client)

    response = _revoke(client, grant["grant_id"], grant["capability"])

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "grant_id",
        "decision_id",
        "data_id",
        "revoked",
        "revoked_at",
    }
    assert data["grant_id"] == grant["grant_id"]
    assert data["data_id"] == DATA_ID
    assert data["revoked"] is True
    revoked_at = datetime.fromisoformat(data["revoked_at"])
    assert revoked_at.utcoffset() == timedelta(0)
    # The plaintext capability never appears on any response after create.
    assert grant["capability"] not in response.text

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "revoked"
        assert row.revoked_at is not None
        assert row.consumed_at is None
        # The audit row carries only identifiers, scope, status, timestamps
        # and the capability digest.
        assert row.capability_digest == hashlib.sha256(
            grant["capability"].encode("ascii")
        ).hexdigest()
        assert session.query(ReleaseGrant).count() == 1


def test_revoke_unknown_grant_returns_404(client):
    _decision(client)
    response = _revoke(
        client, "00000000-0000-0000-0000-000000000000", "A" * 43
    )
    assert response.status_code == 404


def test_revoke_cross_scope_returns_404(client):
    grant = _setup(client)
    assert (
        _revoke(
            client, grant["grant_id"], grant["capability"], tenant_id="tenant-b"
        ).status_code
        == 404
    )
    assert (
        _revoke(
            client,
            grant["grant_id"],
            grant["capability"],
            workload_id="workload-2",
        ).status_code
        == 404
    )
    # The grant is untouched and still revocable in its own scope.
    assert _revoke(client, grant["grant_id"], grant["capability"]).status_code == 200


@pytest.mark.parametrize(
    "bad_id",
    [
        "not-a-uuid",
        "12345",
        "00000000-0000-0000-0000-00000000000g",
        " 00000000-0000-0000-0000-000000000000 ",
    ],
)
def test_revoke_invalid_path_identifier_returns_422(client, bad_id):
    grant = _setup(client)
    response = _revoke(client, bad_id, grant["capability"])
    assert response.status_code == 422
    # No state is written for a malformed identifier.
    assert _revoke(client, grant["grant_id"], grant["capability"]).status_code == 200


def test_revoke_wrong_capability_returns_401_and_grant_stays_pending(client, app):
    grant = _setup(client)
    wrong = "B" + grant["capability"][1:]

    response = _revoke(client, grant["grant_id"], wrong)
    assert response.status_code == 401

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.revoked_at is None
        assert row.consumed_at is None

    # The wrong capability changed nothing: the correct one still revokes.
    follow_up = _revoke(client, grant["grant_id"], grant["capability"])
    assert follow_up.status_code == 200


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "capability"])
def test_revoke_requires_all_fields(client, missing):
    grant = _setup(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }
    del body[missing]
    response = client.post(
        f"/v1/release-grants/{grant['grant_id']}/revoke", json=body
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"workload_id": "   "},
        {"capability": ""},
        {"capability": "   "},
        {"capability": "not base64!!!"},
        {"capability": "with=padding"},
        {"tenant_id": 1},
        {"workload_id": 7},
        {"capability": 9},
        {"capability": True},
    ],
)
def test_revoke_rejects_invalid_fields(client, overrides):
    grant = _setup(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }
    body.update(overrides)
    assert (
        client.post(
            f"/v1/release-grants/{grant['grant_id']}/revoke", json=body
        ).status_code
        == 422
    )
    # A rejected request writes no state.
    assert _revoke(client, grant["grant_id"], grant["capability"]).status_code == 200


def test_revoke_rejects_non_object_body(client):
    grant = _setup(client)
    assert (
        client.post(
            f"/v1/release-grants/{grant['grant_id']}/revoke",
            json=["not", "an", "object"],
        ).status_code
        == 422
    )


def test_revoke_consumed_grant_returns_409(client):
    grant = _setup(client)
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200

    response = _revoke(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 409


def test_revoke_released_grant_returns_409(client):
    grant = _setup(client)
    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 200

    response = _revoke(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 409


def test_repeat_revoke_returns_409_and_keeps_original_time(client, app):
    grant = _setup(client)
    first = _revoke(client, grant["grant_id"], grant["capability"])
    assert first.status_code == 200
    first_at = first.json()["revoked_at"]

    second = _revoke(client, grant["grant_id"], grant["capability"])
    assert second.status_code == 409

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "revoked"
        # The repeated revocation changed neither the timestamp nor the row.
        assert row.revoked_at == datetime.fromisoformat(first_at)
        assert session.query(ReleaseGrant).count() == 1


def test_revoke_expired_grant_returns_410_and_stays_pending(client, app):
    grant = _setup(client)
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = _revoke(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 410

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.revoked_at is None


def test_revoke_wins_then_consume_and_release_observe_409(client, app):
    grant = _setup(client)
    assert _revoke(client, grant["grant_id"], grant["capability"]).status_code == 200

    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 409
    release = _release(client, grant["grant_id"], grant["capability"])
    assert release.status_code == 409
    assert "payload" not in release.text

    # The losers only observed the terminal state: still exactly one row,
    # still revoked, no consumption audit.
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "revoked"
        assert row.consumed_at is None
        assert session.query(ReleaseGrant).count() == 1


def test_consume_wins_then_revoke_observes_409(client):
    grant = _setup(client)
    assert _consume(client, grant["grant_id"], grant["capability"]).status_code == 200
    assert _revoke(client, grant["grant_id"], grant["capability"]).status_code == 409


def test_concurrent_revokes_only_one_succeeds(app):
    client = TestClient(app)
    grant = _setup(client)
    url = f"/v1/release-grants/{grant['grant_id']}/revoke"
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }

    def revoke():
        return TestClient(app).post(url, json=body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: revoke(), range(16)))

    assert statuses.count(200) == 1
    assert statuses.count(409) == 15

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "revoked"
        assert row.revoked_at is not None
        assert session.query(ReleaseGrant).count() == 1


def test_concurrent_revoke_and_consume_share_single_win(app):
    client = TestClient(app)
    grant = _setup(client)
    shared = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }

    def call(i):
        c = TestClient(app)
        if i % 2 == 0:
            return c.post(
                f"/v1/release-grants/{grant['grant_id']}/revoke", json=shared
            ).status_code
        return c.post(
            f"/v1/release-grants/{grant['grant_id']}/consume", json=shared
        ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(call, range(16)))

    assert statuses.count(200) == 1
    assert all(code in (200, 409) for code in statuses)
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status in ("consumed", "revoked")
        assert session.query(ReleaseGrant).count() == 1


def test_concurrent_revoke_and_release_share_single_win(app):
    client = TestClient(app)
    grant = _setup(client)
    revoke_body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }
    release_body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "capability": grant["capability"],
    }

    def call(i):
        c = TestClient(app)
        if i % 2 == 0:
            return c.post(
                f"/v1/release-grants/{grant['grant_id']}/revoke", json=revoke_body
            ).status_code
        return c.post(f"/v1/release/{grant['grant_id']}", json=release_body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(call, range(16)))

    assert statuses.count(200) == 1
    assert all(code in (200, 409) for code in statuses)
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status in ("consumed", "revoked")
        assert session.query(ReleaseGrant).count() == 1


def test_revoked_state_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    grant = _setup(client1)
    first = _revoke(client1, grant["grant_id"], grant["capability"])
    assert first.status_code == 200
    revoked_at = first.json()["revoked_at"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    # The revoked state is durable: consume, release and re-revoke all
    # observe the terminal state, and the timestamp is unchanged.
    assert _consume(client2, grant["grant_id"], grant["capability"]).status_code == 409
    assert _release(client2, grant["grant_id"], grant["capability"]).status_code == 409
    assert _revoke(client2, grant["grant_id"], grant["capability"]).status_code == 409
    with app2.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "revoked"
        assert row.revoked_at == datetime.fromisoformat(revoked_at)
    app2.state.engine.dispose()


def test_capability_plaintext_never_persisted_or_logged(app, client, caplog):
    grant = _setup(client)
    caplog.set_level(logging.DEBUG, logger="proof_release")
    response = _revoke(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 200

    assert grant["capability"] not in caplog.text
    assert PAYLOAD not in caplog.text
    assert MASTER_KEY not in caplog.text

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        for column in row.__table__.columns:
            value = getattr(row, column.name)
            assert grant["capability"] not in str(value), column.name
            assert PAYLOAD not in str(value), column.name
