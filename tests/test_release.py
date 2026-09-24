"""Tests for POST /v1/release/{grant_id} (payload release)."""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import DataEnvelope, ReleaseGrant
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"
DATA_ID = "data-1"
PAYLOAD = "super-secret-payload 🔐"

# 32-byte fixed master key for tests, rendered as unpadded base64url.
MASTER_KEY_BYTES = b"0123456789abcdef0123456789abcdef"
MASTER_KEY = b64url_encode(MASTER_KEY_BYTES)


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/release.db")
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


def _evidence(nonce: str, claims: dict) -> str:
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)}
    )


def _decision(client):
    """Drive the full challenge -> evidence -> verify -> decision flow."""
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    evidence = _evidence(created["nonce"], {"m": "x"})
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
    return decided.json()


def _envelope(client, data_id=DATA_ID, payload=PAYLOAD, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": data_id,
        "payload": payload,
    }
    body.update(overrides)
    response = client.post("/v1/data-envelopes", json=body)
    assert response.status_code == 201
    return response.json()


def _grant(client, decision_id, data_id=DATA_ID, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision_id,
        "data_id": data_id,
    }
    body.update(overrides)
    response = client.post("/v1/release-grants", json=body)
    assert response.status_code == 201
    return response.json()


def _release(client, grant_id, _capability, data_id=DATA_ID, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": data_id,
        "capability": _capability,
    }
    body.update(overrides)
    return client.post(f"/v1/release/{grant_id}", json=body)


def _grant_row(app, grant_id):
    with app.state.session_factory() as session:
        return session.get(ReleaseGrant, grant_id)


def test_release_success_returns_payload_only(client):
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])

    response = _release(client, grant["grant_id"], grant["capability"])

    assert response.status_code == 200
    # Compact JSON with exactly one key and a single trailing newline.
    assert response.content == b'{"payload":"' + PAYLOAD.encode("utf-8") + b'"}\n'
    assert response.text.endswith("\n") and not response.text.endswith("\n\n")
    data = response.json()
    assert set(data) == {"payload"}
    assert data["payload"] == PAYLOAD
    assert isinstance(data["payload"], str) and data["payload"]
    assert grant["capability"] not in response.text


def test_release_success_consumes_grant_once(client, app):
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])

    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 200

    row = _grant_row(app, grant["grant_id"])
    assert row.status == "consumed"
    assert row.consumed_at is not None
    # The single audit row records scope, data id, status, timestamps and
    # only the capability digest — never the plaintext capability.
    assert row.tenant_id == TENANT
    assert row.workload_id == WORKLOAD
    assert row.data_id == DATA_ID
    assert row.capability_digest == hashlib.sha256(
        grant["capability"].encode("ascii")
    ).hexdigest()
    assert grant["capability"] not in str(
        {c.name: getattr(row, c.name) for c in row.__table__.columns}
    )

    # The shared one-time state: both entry points now see it consumed.
    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 409
    consumed = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )
    assert consumed.status_code == 409


def test_release_after_plain_consume_returns_409(client):
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])
    consumed = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )
    assert consumed.status_code == 200

    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 409


def test_release_unknown_grant_returns_404(client):
    _envelope(client)
    response = _release(
        client, "00000000-0000-0000-0000-000000000000", "A" * 43
    )
    assert response.status_code == 404


def test_release_cross_scope_returns_404(client):
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])

    assert (
        _release(client, grant["grant_id"], grant["capability"], tenant_id="tenant-b")
        .status_code
        == 404
    )
    assert (
        _release(
            client, grant["grant_id"], grant["capability"], workload_id="workload-2"
        ).status_code
        == 404
    )


def test_release_data_id_mismatch_returns_404(client, app):
    _envelope(client)
    _envelope(client, data_id="data-2")
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"], data_id="data-2")

    response = _release(client, grant["grant_id"], grant["capability"], data_id=DATA_ID)
    assert response.status_code == 404
    # No consumption: the grant stays pending.
    assert _grant_row(app, grant["grant_id"]).status == "pending"


def test_release_missing_envelope_returns_404(client, app):
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 404
    assert _grant_row(app, grant["grant_id"]).status == "pending"


def test_release_wrong_capability_returns_401_and_stays_pending(client, app):
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])
    wrong = "B" + grant["capability"][1:]

    response = _release(client, grant["grant_id"], wrong)
    assert response.status_code == 401
    assert _grant_row(app, grant["grant_id"]).status == "pending"

    # The grant is still consumable afterwards.
    follow_up = _release(client, grant["grant_id"], grant["capability"])
    assert follow_up.status_code == 200


def test_release_expired_grant_returns_410(client, app):
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 410
    assert _grant_row(app, grant["grant_id"]).status == "pending"


def test_release_consumed_and_expired_reports_expired_first(client, app):
    # Consumed is judged last: an expired grant reports 410 even when it
    # has already been consumed.
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.status = "consumed"
        row.consumed_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 410


@pytest.mark.parametrize(
    "missing", ["tenant_id", "workload_id", "data_id", "capability"]
)
def test_release_requires_all_fields(client, missing):
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "capability": grant["capability"],
    }
    del body[missing]
    response = client.post(f"/v1/release/{grant['grant_id']}", json=body)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"data_id": ""},
        {"data_id": "   "},
        {"capability": ""},
        {"capability": "   "},
        {"capability": "not base64!!!"},
        {"capability": "with=padding"},
        {"tenant_id": 1},
        {"data_id": 1},
        {"capability": 1},
    ],
)
def test_release_rejects_invalid_fields(client, app, overrides):
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])
    assert (
        _release(client, grant["grant_id"], grant["capability"], **overrides).status_code
        == 422
    )
    assert _grant_row(app, grant["grant_id"]).status == "pending"


def test_release_without_keyring_returns_500_and_stays_pending(
    client, app, monkeypatch
):
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY")

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 500
    assert _grant_row(app, grant["grant_id"]).status == "pending"


def test_release_missing_historical_key_returns_500_and_stays_pending(
    client, app, monkeypatch
):
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])
    # Rotate to a keyring that no longer holds version 1.
    other_key = b64url_encode(b"z" * 32)
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING",
        json.dumps({"current_version": 2, "keys": {"2": other_key}}),
    )

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 500
    assert _grant_row(app, grant["grant_id"]).status == "pending"


def test_release_corrupt_wrapped_key_returns_500_and_stays_pending(client, app):
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])
    with app.state.session_factory() as session:
        row = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        row.wrapped_key = b"\x00" * 40
        session.commit()

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 500
    assert _grant_row(app, grant["grant_id"]).status == "pending"


def test_release_tampered_ciphertext_returns_500_and_stays_pending(client, app):
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])
    with app.state.session_factory() as session:
        row = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        tampered = bytearray(row.ciphertext)
        tampered[0] ^= 0x01
        row.ciphertext = bytes(tampered)
        session.commit()

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 500
    assert _grant_row(app, grant["grant_id"]).status == "pending"


def test_release_concurrent_only_one_succeeds(app):
    client = TestClient(app)
    _envelope(client)
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])
    url = f"/v1/release/{grant['grant_id']}"
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "capability": grant["capability"],
    }

    def release():
        return TestClient(app).post(url, json=body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: release(), range(16)))

    assert statuses.count(200) == 1
    assert statuses.count(409) == 15

    row = _grant_row(app, grant["grant_id"])
    assert row.status == "consumed"
    assert row.consumed_at is not None


def test_release_consumed_state_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    _envelope(client1)
    decision = _decision(client1)
    grant = _grant(client1, decision["decision_id"])
    response = _release(client1, grant["grant_id"], grant["capability"])
    assert response.status_code == 200
    app1.state.engine.dispose()

    app2 = create_app(url)
    try:
        # The consumption audit persisted: a replay reports 409.
        replay = _release(TestClient(app2), grant["grant_id"], grant["capability"])
        assert replay.status_code == 409
        row = _grant_row(app2, grant["grant_id"])
        assert row.status == "consumed"
        assert row.consumed_at is not None
    finally:
        app2.state.engine.dispose()
