"""Tests for POST /v1/release-grants and /v1/release-grants/{id}/consume."""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import ReleaseGrant

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"
DATA_ID = "data-123"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    return create_app(f"sqlite:///{tmp_path}/grants.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


def _mac(nonce: str, claims: dict, tenant: str, workload: str) -> str:
    key = hmac.new(
        SECRET.encode("utf-8"),
        f"{tenant}:{workload}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _evidence(nonce: str, claims: dict, tenant: str, workload: str) -> str:
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims, tenant, workload)}
    )


def _allowed_decision(client, claims=None, scope=None):
    """Drive the full flow to an allowed decision; return its decision_id."""
    scope = scope or {}
    tenant = scope.get("tenant_id", TENANT)
    workload = scope.get("workload_id", WORKLOAD)
    claims = claims if claims is not None else {"measurement": "abc"}
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    evidence = _evidence(created["nonce"], claims, tenant, workload)
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
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
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.json()["status"] == "verified"
    policy = client.post(
        "/v1/policies",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "name": "release",
            "rule": {"claim": "measurement", "equals": "abc"},
        },
    )
    assert policy.status_code == 201
    decided = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy.json()["policy_id"],
        },
    )
    assert decided.status_code == 200
    return decided.json()


def _denied_decision(client):
    return _allowed_decision(client, claims={"measurement": "other"})


def _create_grant(client, decision_id, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision_id,
        "data_id": DATA_ID,
    }
    body.update(overrides)
    return client.post("/v1/release-grants", json=body)


def _consume(client, grant_id, capability, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": capability,
    }
    body.update(overrides)
    return client.post(f"/v1/release-grants/{grant_id}/consume", json=body)


def test_create_grant_returns_capability_once(client, app):
    decision = _allowed_decision(client)

    response = _create_grant(client, decision["decision_id"])

    assert response.status_code == 201
    data = response.json()
    assert data["grant_id"]
    assert data["decision_id"] == decision["decision_id"]
    assert data["data_id"] == DATA_ID
    assert data["capability"]
    assert data["status"] == "pending"
    issued_at = datetime.fromisoformat(data["issued_at"])
    expires_at = datetime.fromisoformat(data["expires_at"])
    assert issued_at.utcoffset() == timedelta(0)
    assert expires_at.utcoffset() == timedelta(0)
    assert (expires_at - issued_at).total_seconds() == 300

    # Only the digest is persisted; the plaintext capability appears
    # nowhere in storage.
    with app.state.session_factory() as session:
        grant = session.query(ReleaseGrant).one()
        assert grant.capability_digest == hashlib.sha256(
            data["capability"].encode("ascii")
        ).hexdigest()
        for column in grant.__table__.columns:
            assert data["capability"] not in str(
                getattr(grant, column.name)
            ), f"capability leaked in {column.name}"


def test_create_grant_honours_ttl_bounds(client):
    decision = _allowed_decision(client)

    response = _create_grant(client, decision["decision_id"], ttl_seconds=30)
    assert response.status_code == 201
    data = response.json()
    delta = datetime.fromisoformat(data["expires_at"]) - datetime.fromisoformat(
        data["issued_at"]
    )
    assert delta.total_seconds() == 30

    assert (
        _create_grant(client, decision["decision_id"], ttl_seconds=900).status_code
        == 201
    )
    for bad in (0, 29, 901, -5, "300", 300.5, True):
        assert (
            _create_grant(client, decision["decision_id"], ttl_seconds=bad).status_code
            == 422
        ), bad


def test_unknown_decision_returns_404(client):
    response = _create_grant(client, "00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_cross_scope_decision_returns_404(client):
    foreign = _allowed_decision(
        client, scope={"tenant_id": "tenant-b", "workload_id": "workload-2"}
    )
    assert _create_grant(client, foreign["decision_id"]).status_code == 404

    own = _allowed_decision(client)
    assert (
        _create_grant(client, own["decision_id"], tenant_id="tenant-b").status_code
        == 404
    )
    assert (
        _create_grant(client, own["decision_id"], workload_id="workload-2").status_code
        == 404
    )


def test_denied_decision_returns_409(client):
    decision = _denied_decision(client)
    assert _create_grant(client, decision["decision_id"]).status_code == 409


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "decision_id", "data_id"])
def test_create_grant_requires_all_fields(client, missing):
    decision = _allowed_decision(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision["decision_id"],
        "data_id": DATA_ID,
    }
    del body[missing]
    assert client.post("/v1/release-grants", json=body).status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"decision_id": ""},
        {"decision_id": "   "},
        {"data_id": ""},
        {"data_id": "   "},
        {"tenant_id": 1},
    ],
)
def test_create_grant_rejects_invalid_fields(client, overrides):
    decision = _allowed_decision(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision["decision_id"],
        "data_id": DATA_ID,
    }
    body.update(overrides)
    assert client.post("/v1/release-grants", json=body).status_code == 422


def test_consume_grant_succeeds_once(client):
    decision = _allowed_decision(client)
    created = _create_grant(client, decision["decision_id"]).json()

    response = _consume(client, created["grant_id"], created["capability"])

    assert response.status_code == 200
    data = response.json()
    assert data == {
        "grant_id": created["grant_id"],
        "decision_id": decision["decision_id"],
        "data_id": DATA_ID,
        "status": "consumed",
        "consumed_at": data["consumed_at"],
    }
    assert datetime.fromisoformat(data["consumed_at"]).utcoffset() == timedelta(0)
    # The capability is never returned again.
    assert created["capability"] not in response.text


def test_consume_unknown_and_cross_scope_return_404(client):
    decision = _allowed_decision(client)
    created = _create_grant(client, decision["decision_id"]).json()

    assert (
        _consume(
            client, "00000000-0000-0000-0000-000000000000", created["capability"]
        ).status_code
        == 404
    )
    assert (
        _consume(
            client, created["grant_id"], created["capability"], tenant_id="tenant-b"
        ).status_code
        == 404
    )
    assert (
        _consume(
            client, created["grant_id"], created["capability"], workload_id="workload-2"
        ).status_code
        == 404
    )


def test_consume_wrong_capability_returns_401(client):
    decision = _allowed_decision(client)
    created = _create_grant(client, decision["decision_id"]).json()
    wrong = "B" + created["capability"][1:]

    assert _consume(client, created["grant_id"], wrong).status_code == 401
    # The grant is still pending and consumable with the right capability.
    assert (
        _consume(client, created["grant_id"], created["capability"]).status_code
        == 200
    )


def test_consume_twice_returns_409(client):
    decision = _allowed_decision(client)
    created = _create_grant(client, decision["decision_id"]).json()

    assert (
        _consume(client, created["grant_id"], created["capability"]).status_code
        == 200
    )
    assert (
        _consume(client, created["grant_id"], created["capability"]).status_code
        == 409
    )


def test_consume_expired_returns_410(client, app):
    decision = _allowed_decision(client)
    created = _create_grant(client, decision["decision_id"]).json()
    with app.state.session_factory() as session:
        grant = session.get(ReleaseGrant, created["grant_id"])
        grant.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    assert (
        _consume(client, created["grant_id"], created["capability"]).status_code
        == 410
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"capability": ""},
        {"capability": "not base64!!!"},
        {"capability": "with=padding"},
        {"tenant_id": ""},
        {"workload_id": "   "},
        {"capability": 1},
    ],
)
def test_consume_rejects_invalid_fields(client, overrides):
    decision = _allowed_decision(client)
    created = _create_grant(client, decision["decision_id"]).json()
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": created["capability"],
    }
    body.update(overrides)
    response = client.post(f"/v1/release-grants/{created['grant_id']}/consume", json=body)
    assert response.status_code == 422


def test_concurrent_consumes_exactly_one_success(app):
    client = TestClient(app)
    decision = _allowed_decision(client)
    created = _create_grant(client, decision["decision_id"]).json()
    payload = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": created["capability"],
    }

    def consume():
        return TestClient(app).post(
            f"/v1/release-grants/{created['grant_id']}/consume", json=payload
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: consume(), range(16)))

    assert [r.status_code for r in responses].count(200) == 1
    assert {r.status_code for r in responses} == {200, 409}
    with app.state.session_factory() as session:
        grant = session.query(ReleaseGrant).one()
        assert grant.status == "consumed"


def test_grants_survive_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = TestClient(create_app(url))
    decision = _allowed_decision(first)
    created = _create_grant(first, decision["decision_id"]).json()

    # A brand-new app instance over the same database can still consume.
    second = TestClient(create_app(url))
    response = _consume(second, created["grant_id"], created["capability"])
    assert response.status_code == 200
    assert response.json()["decision_id"] == decision["decision_id"]
