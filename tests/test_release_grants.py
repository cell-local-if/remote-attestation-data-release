"""Tests for POST /v1/release-grants and .../{grant_id}/consume."""

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


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    return create_app(f"sqlite:///{tmp_path}/grants.db")


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


def _evidence(nonce: str, claims: dict | None = None) -> str:
    claims = claims if claims is not None else {}
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)}
    )


def _decision(client, claims=None, satisfied=True):
    """Drive the full challenge -> evidence -> verify -> decision flow."""
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    evidence = _evidence(created["nonce"], claims if claims is not None else {"m": "x"})
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
    rule = (
        {"claim": "m", "equals": "x"}
        if satisfied
        else {"claim": "m", "equals": "other"}
    )
    policy = client.post(
        "/v1/policies",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "name": "release",
            "rule": rule,
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
    return decided.json(), evidence


def _grant(client, _decision_id, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": _decision_id,
        "data_id": "data-1",
    }
    body.update(fields)
    return client.post("/v1/release-grants", json=body)


def _consume(client, _grant_id, _capability, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": _capability,
    }
    body.update(fields)
    return client.post(f"/v1/release-grants/{_grant_id}/consume", json=body)


def test_create_grant_on_allowed_decision_returns_201(client):
    decision, _ = _decision(client)

    response = _grant(client, decision["decision_id"])

    assert response.status_code == 201
    data = response.json()
    assert set(data) == {
        "grant_id",
        "decision_id",
        "data_id",
        "capability",
        "pending",
        "issued_at",
        "expires_at",
    }
    assert data["decision_id"] == decision["decision_id"]
    assert data["data_id"] == "data-1"
    assert data["pending"] is True
    # 32 random bytes -> 43 chars of unpadded base64url.
    capability = data["capability"]
    assert len(capability) == 43
    assert "=" not in capability
    assert all(c.isalnum() or c in "_-" for c in capability)
    issued = datetime.fromisoformat(data["issued_at"])
    expires = datetime.fromisoformat(data["expires_at"])
    assert issued.utcoffset() == timedelta(0)
    assert expires.utcoffset() == timedelta(0)
    assert (expires - issued) == timedelta(seconds=300)


@pytest.mark.parametrize("ttl", [30, 900])
def test_create_grant_honours_ttl_bounds(client, ttl):
    decision, _ = _decision(client)
    response = _grant(client, decision["decision_id"], ttl_seconds=ttl)
    assert response.status_code == 201
    data = response.json()
    assert datetime.fromisoformat(data["expires_at"]) - datetime.fromisoformat(
        data["issued_at"]
    ) == timedelta(seconds=ttl)


def test_denied_decision_returns_409(client):
    decision, _ = _decision(client, satisfied=False)
    assert decision["status"] == "denied"

    response = _grant(client, decision["decision_id"])
    assert response.status_code == 409
    assert "capability" not in response.text


def test_unknown_decision_returns_404(client):
    response = _grant(
        client, "00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 404


def test_cross_scope_decision_returns_404(client):
    decision, _ = _decision(client)

    assert (
        _grant(client, decision["decision_id"], tenant_id="tenant-b").status_code
        == 404
    )
    assert (
        _grant(client, decision["decision_id"], workload_id="workload-2").status_code
        == 404
    )


@pytest.mark.parametrize(
    "missing", ["tenant_id", "workload_id", "decision_id", "data_id"]
)
def test_create_grant_requires_all_fields(client, missing):
    decision, _ = _decision(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision["decision_id"],
        "data_id": "data-1",
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
        {"ttl_seconds": 29},
        {"ttl_seconds": 901},
        {"ttl_seconds": "300"},
        {"ttl_seconds": 300.5},
        {"ttl_seconds": True},
    ],
)
def test_create_grant_rejects_invalid_fields(client, overrides):
    decision, _ = _decision(client)
    assert _grant(client, decision["decision_id"], **overrides).status_code == 422


def test_capabilities_are_unique_and_unpredictable(client):
    decision, _ = _decision(client)
    first = _grant(client, decision["decision_id"]).json()
    second = _grant(client, decision["decision_id"]).json()
    assert first["grant_id"] != second["grant_id"]
    assert first["capability"] != second["capability"]


def test_capability_plaintext_is_never_persisted(client, app):
    # The grant row may hold only identifiers/scope/status/timestamps and
    # the capability digest — never the plaintext capability, evidence or
    # claims. (data_id is itself the audited data identifier, so it is
    # expected to be stored.)
    decision, evidence = _decision(client, claims={"m": "x", "secret": "z" * 32})
    created = _grant(client, decision["decision_id"], data_id="data-1")
    capability = created.json()["capability"]

    with app.state.session_factory() as session:
        grant = session.query(ReleaseGrant).one()
        assert grant.capability_digest == hashlib.sha256(
            capability.encode("ascii")
        ).hexdigest()
        columns = {
            c.name: getattr(grant, c.name) for c in grant.__table__.columns
        }
        for name, value in columns.items():
            assert capability not in str(value), f"capability leaked in {name}"
            # Evidence/claims material must not end up on the audit row.
            assert evidence not in str(value), f"evidence leaked in {name}"
            assert "z" * 32 not in str(value), f"claims leaked in {name}"


def test_consume_grant_success(client):
    decision, _ = _decision(client)
    grant = _grant(client, decision["decision_id"], data_id="data-7").json()

    response = _consume(client, grant["grant_id"], grant["capability"])

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "grant_id",
        "decision_id",
        "data_id",
        "consumed",
        "consumed_at",
    }
    assert data["grant_id"] == grant["grant_id"]
    assert data["decision_id"] == decision["decision_id"]
    assert data["data_id"] == "data-7"
    assert data["consumed"] is True
    consumed_at = datetime.fromisoformat(data["consumed_at"])
    assert consumed_at.utcoffset() == timedelta(0)
    assert "capability" not in response.text


def test_consume_unknown_grant_returns_404(client):
    decision, _ = _decision(client)
    response = _consume(
        client,
        "00000000-0000-0000-0000-000000000000",
        "A" * 43,
    )
    assert response.status_code == 404


def test_consume_cross_scope_returns_404(client):
    decision, _ = _decision(client)
    grant = _grant(client, decision["decision_id"]).json()

    assert (
        _consume(
            client, grant["grant_id"], grant["capability"], tenant_id="tenant-b"
        ).status_code
        == 404
    )
    assert (
        _consume(
            client, grant["grant_id"], grant["capability"], workload_id="workload-2"
        ).status_code
        == 404
    )


def test_consume_wrong_capability_returns_401(client):
    decision, _ = _decision(client)
    grant = _grant(client, decision["decision_id"]).json()
    wrong = ("B" if grant["capability"][0] != "B" else "C") + grant["capability"][1:]

    response = _consume(client, grant["grant_id"], wrong)
    assert response.status_code == 401
    # A failed capability check must not settle the grant.
    follow_up = _consume(client, grant["grant_id"], grant["capability"])
    assert follow_up.status_code == 200


def test_consume_twice_second_returns_409(client):
    decision, _ = _decision(client)
    grant = _grant(client, decision["decision_id"]).json()

    first = _consume(client, grant["grant_id"], grant["capability"])
    assert first.status_code == 200
    second = _consume(client, grant["grant_id"], grant["capability"])
    assert second.status_code == 409


def test_consume_expired_grant_returns_410(client, app):
    decision, _ = _decision(client)
    grant = _grant(client, decision["decision_id"]).json()
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = _consume(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 410


@pytest.mark.parametrize(
    "missing", ["tenant_id", "workload_id", "capability"]
)
def test_consume_requires_all_fields(client, missing):
    decision, _ = _decision(client)
    grant = _grant(client, decision["decision_id"]).json()
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }
    del body[missing]
    response = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume", json=body
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"capability": ""},
        {"capability": "   "},
        {"capability": "not base64!!!"},
        {"capability": "with=padding"},
        {"tenant_id": 1},
    ],
)
def test_consume_rejects_invalid_fields(client, overrides):
    decision, _ = _decision(client)
    grant = _grant(client, decision["decision_id"]).json()
    assert (
        _consume(client, grant["grant_id"], grant["capability"], **overrides).status_code
        == 422
    )


def test_grant_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    decision, _ = _decision(client1)
    grant = _grant(client1, decision["decision_id"]).json()
    app1.state.engine.dispose()

    app2 = create_app(url)
    response = _consume(
        TestClient(app2), grant["grant_id"], grant["capability"]
    )
    assert response.status_code == 200
    assert response.json()["consumed"] is True


def test_concurrent_consume_only_one_succeeds(app):
    client = TestClient(app)
    decision, _ = _decision(client)
    grant = _grant(client, decision["decision_id"]).json()
    url = f"/v1/release-grants/{grant['grant_id']}/consume"
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }

    def consume():
        return TestClient(app).post(url, json=body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: consume(), range(16)))

    # The shared per-scope minute budget admits exactly five of the
    # sixteen valid requests; the rest are rate-limited. Of the admitted
    # five, exactly one settles the grant and the others observe 409.
    assert statuses.count(200) == 1
    assert statuses.count(409) == 4
    assert statuses.count(429) == 11

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"
        assert row.consumed_at is not None
        assert session.query(ReleaseGrant).count() == 1
