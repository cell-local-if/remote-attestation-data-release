"""Non-SQLite coverage for the release-grant event timeline rebuild.

The per-grant timeline shipped after release grants themselves, so a
database written by an older deployment holds grant rows but no
``release_grant_events`` rows. On open the service reconstructs each
grant's gap-free per-grant history (the birth event plus the single
settlement event) from the committed grant state. These tests run only
against a real locking backend (PostgreSQL) because the SQLite rebuild
path is already covered by ``test_release_grant_events.py``.

They are skipped unless ``PROOF_RELEASE_TEST_PG_URL`` names a reachable
PostgreSQL database (a SQLAlchemy psycopg2 URL, e.g.
``postgresql+psycopg2://user:pass@localhost/proofrel_test``), so the
default, dependency-free test run stays SQLite-only. The named database
is reset (all tables dropped) per test.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from proof_release.app import create_app
from proof_release.db import Base

PG_URL = os.environ.get("PROOF_RELEASE_TEST_PG_URL")
pytestmark = pytest.mark.skipif(
    PG_URL is None,
    reason="set PROOF_RELEASE_TEST_PG_URL to run non-SQLite migration tests",
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"


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


def _pending_grant(client, *, data_id):
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
    client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    policy = client.post(
        "/v1/policies",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "name": f"policy-{data_id}",
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
    ).json()
    created_grant = client.post(
        "/v1/release-grants",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "decision_id": decided["decision_id"],
            "data_id": data_id,
        },
    )
    assert created_grant.status_code == 201
    return created_grant.json()


def _events(client, grant_id):
    return client.get(
        f"/v1/release-grants/{grant_id}/events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv(
        "PROOF_RELEASE_MASTER_KEY",
        "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY",
    )
    # Start every test from an empty database.
    engine = create_engine(PG_URL)
    Base.metadata.drop_all(engine)
    engine.dispose()
    application = create_app(PG_URL)
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def test_open_reconstructs_legacy_grant_timelines(client):
    consumed = _pending_grant(client, data_id="data-a")
    revoked = _pending_grant(client, data_id="data-b")
    pending = _pending_grant(client, data_id="data-c")
    assert (
        client.post(
            f"/v1/release-grants/{consumed['grant_id']}/consume",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "capability": consumed["capability"],
            },
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/v1/release-grants/{revoked['grant_id']}/revoke",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "capability": revoked["capability"],
            },
        ).status_code
        == 200
    )
    client.app.state.engine.dispose()

    # Simulate a pre-feature database.
    engine = create_engine(PG_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE release_grant_events"))
    engine.dispose()

    second = create_app(PG_URL)
    try:
        second_client = TestClient(second)

        def _tuples(grant_id):
            events = _events(second_client, grant_id).json()["events"]
            return [
                (e["seq"], e["old_status"], e["new_status"], e["reason"])
                for e in events
            ]

        assert _tuples(consumed["grant_id"]) == [
            (1, None, "pending", "issued"),
            (2, "pending", "consumed", "consume"),
        ]
        assert _tuples(revoked["grant_id"]) == [
            (1, None, "pending", "issued"),
            (2, "pending", "revoked", "revoked"),
        ]
        assert _tuples(pending["grant_id"]) == [
            (1, None, "pending", "issued"),
        ]
    finally:
        second.state.engine.dispose()

    # Reopening must not duplicate reconstructed rows.
    third = create_app(PG_URL)
    try:
        third_client = TestClient(third)
        for grant in (consumed, revoked, pending):
            events = _events(third_client, grant["grant_id"]).json()["events"]
            assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    finally:
        third.state.engine.dispose()
