"""Tests for POST /v1/release/{grant_id} (authorized payload release)."""

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
from proof_release.db import DataEnvelope, ReleaseGrant
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
MASTER_KEY_BYTES = b"0123456789abcdef0123456789abcdef"
MASTER_KEY = b64url_encode(MASTER_KEY_BYTES)
PAYLOAD = 'super-secret payload 🔐 with "quote" and \\ backslash'


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


def _evidence(nonce: str) -> str:
    claims = {"m": "x"}
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)}
    )


def _decision(client):
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
    assert decided.json()["status"] == "allowed"
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


def _release(client, grant_id, capability, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release/{grant_id}", json=body)

def _setup(client, data_id=DATA_ID, payload=PAYLOAD):
    decision_id = _decision(client)
    _envelope(client, data_id=data_id, payload=payload)
    grant = _grant(client, decision_id, data_id=data_id)
    return grant


def test_release_success_returns_compact_payload_json_with_newline(client):
    grant = _setup(client)

    response = _release(client, grant["grant_id"], grant["capability"])

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    raw = response.content
    # Exactly one trailing newline, compact separators, single field.
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    assert raw == json.dumps(
        {"payload": PAYLOAD}, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8") + b"\n"
    data = json.loads(raw)
    assert set(data) == {"payload"}
    assert isinstance(data["payload"], str)
    assert data["payload"] == PAYLOAD
    # No metadata fields, no echoed capability.
    assert "capability" not in response.text
    assert grant["grant_id"] not in response.text


def test_release_consumes_grant_exactly_once(client, app):
    grant = _setup(client)

    first = _release(client, grant["grant_id"], grant["capability"])
    assert first.status_code == 200
    second = _release(client, grant["grant_id"], grant["capability"])
    assert second.status_code == 409

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"
        assert row.consumed_at is not None
        # One grant row is the sole consumption audit; it carries scope,
        # data id, status, time and the capability digest.
        assert session.query(ReleaseGrant).count() == 1
        assert row.tenant_id == TENANT
        assert row.workload_id == WORKLOAD
        assert row.data_id == DATA_ID
        assert row.capability_digest == hashlib.sha256(
            grant["capability"].encode("ascii")
        ).hexdigest()


def test_release_shares_atomic_state_with_consume_endpoint(client, app):
    grant = _setup(client)

    # The existing consume endpoint settles the same one-time state first.
    consumed = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )
    assert consumed.status_code == 200
    # Release then observes the consumed state and never decrypts/releases.
    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 409
    assert "payload" not in response.text


def test_consume_loses_when_release_wins_first(client):
    grant = _setup(client)

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 200
    follow_up = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )
    assert follow_up.status_code == 409


def test_unknown_grant_returns_404(client):
    _decision(client)
    response = _release(
        client, "00000000-0000-0000-0000-000000000000", "A" * 43
    )
    assert response.status_code == 404
    assert "payload" not in response.text


def test_cross_scope_grant_returns_404(client):
    grant = _setup(client)
    assert (
        _release(
            client, grant["grant_id"], grant["capability"], tenant_id="tenant-b"
        ).status_code
        == 404
    )
    assert (
        _release(
            client, grant["grant_id"], grant["capability"], workload_id="workload-2"
        ).status_code
        == 404
    )


def test_missing_envelope_returns_404(client):
    decision_id = _decision(client)
    # Grant names an item that was never sealed.
    grant = _grant(client, decision_id, data_id="ghost")
    response = _release(
        client, grant["grant_id"], grant["capability"], data_id="ghost"
    )
    assert response.status_code == 404


def test_data_id_mismatch_returns_404(client, app):
    decision_id = _decision(client)
    _envelope(client, data_id="data-1")
    _envelope(client, data_id="data-2")
    grant = _grant(client, decision_id, data_id="data-1")

    # Requesting another existing same-scope item than the grant names.
    response = _release(
        client, grant["grant_id"], grant["capability"], data_id="data-2"
    )
    assert response.status_code == 404
    # The mismatch 404 takes precedence over a wrong capability.
    wrong = ("B" if grant["capability"][0] != "B" else "C") + grant["capability"][1:]
    response = _release(
        client, grant["grant_id"], wrong, data_id="data-2"
    )
    assert response.status_code == 404
    # Grant stays pending and consumable for the authorized item.
    ok = _release(client, grant["grant_id"], grant["capability"], data_id="data-1")
    assert ok.status_code == 200


def test_cross_scope_envelope_returns_404(client):
    decision_id = _decision(client)
    _envelope(client, data_id=DATA_ID)
    # Same data_id sealed independently in another scope.
    other = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": "tenant-b",
            "workload_id": WORKLOAD,
            "data_id": DATA_ID,
            "payload": PAYLOAD,
        },
    )
    assert other.status_code == 201
    grant = _grant(client, decision_id, data_id=DATA_ID)
    assert (
        _release(
            client, grant["grant_id"], grant["capability"], tenant_id="tenant-b"
        ).status_code
        == 404
    )


def test_404_takes_precedence_over_401(client):
    _decision(client)
    # Unknown grant with a malformed-in-substance capability: still 404.
    response = client.post(
        "/v1/release/00000000-0000-0000-0000-000000000000",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": DATA_ID,
            "capability": "A" * 43,
        },
    )
    assert response.status_code == 404


def test_wrong_capability_returns_401_and_grant_stays_pending(client, app):
    grant = _setup(client)
    wrong = ("B" if grant["capability"][0] != "B" else "C") + grant["capability"][1:]

    response = _release(client, grant["grant_id"], wrong)
    assert response.status_code == 401
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.consumed_at is None

    # The correct capability still releases afterwards.
    follow_up = _release(client, grant["grant_id"], grant["capability"])
    assert follow_up.status_code == 200
    assert follow_up.json()["payload"] == PAYLOAD


def test_capability_checked_before_expiry(client, app):
    grant = _setup(client)
    wrong = ("B" if grant["capability"][0] != "B" else "C") + grant["capability"][1:]
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    # Wrong capability on an expired grant is still an authentication
    # failure, not an expiry.
    assert _release(client, grant["grant_id"], wrong).status_code == 401


def test_expired_grant_returns_410_and_stays_pending(client, app):
    grant = _setup(client)
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 410
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.consumed_at is None


def test_expiry_takes_precedence_over_consumed(client, app):
    grant = _setup(client)
    # Consume first, then mark the settled grant expired: the release path
    # must report 410 because expiry is judged before consumed.
    assert (
        _release(client, grant["grant_id"], grant["capability"]).status_code == 200
    )
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 410


@pytest.mark.parametrize(
    "missing", ["tenant_id", "workload_id", "data_id", "capability"]
)
def test_release_requires_all_fields(client, missing):
    grant = _setup(client)
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
        {"workload_id": "   "},
        {"data_id": ""},
        {"data_id": "   "},
        {"capability": ""},
        {"capability": "   "},
        {"capability": "not base64!!!"},
        {"capability": "with=padding"},
        {"tenant_id": 1},
        {"data_id": 7},
        {"capability": 9},
        {"capability": True},
    ],
)
def test_release_rejects_invalid_fields(client, overrides):
    grant = _setup(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "capability": grant["capability"],
    }
    body.update(overrides)
    assert (
        client.post(f"/v1/release/{grant['grant_id']}", json=body).status_code
        == 422
    )


def test_release_rejects_non_object_body(client):
    grant = _setup(client)
    assert client.post(
        f"/v1/release/{grant['grant_id']}", json=["not", "an", "object"]
    ).status_code == 422


def test_concurrent_releases_only_one_succeeds(app):
    client = TestClient(app)
    grant = _setup(client)
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


def test_concurrent_release_and_consume_share_single_win(app):
    client = TestClient(app)
    grant = _setup(client)
    release_body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "capability": grant["capability"],
    }
    consume_body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }

    def call(i):
        c = TestClient(app)
        if i % 2 == 0:
            return c.post(
                f"/v1/release/{grant['grant_id']}", json=release_body
            ).status_code
        return c.post(
            f"/v1/release-grants/{grant['grant_id']}/consume",
            json=consume_body,
        ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(call, range(16)))

    assert statuses.count(200) == 1
    assert all(code in (200, 409) for code in statuses)
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"
        assert session.query(ReleaseGrant).count() == 1


def test_tampered_ciphertext_returns_500_and_grant_stays_pending(
    app, client
):
    grant = _setup(client)
    with app.state.session_factory() as session:
        envelope = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        tampered = bytearray(envelope.ciphertext)
        tampered[0] ^= 0x01
        envelope.ciphertext = bytes(tampered)
        session.commit()

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 500
    assert PAYLOAD not in response.text

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.consumed_at is None

    # The grant is still consumable through the existing endpoint.
    follow_up = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )
    assert follow_up.status_code == 200


def test_tampered_wrapped_key_returns_500_and_grant_stays_pending(app, client):
    grant = _setup(client)
    with app.state.session_factory() as session:
        envelope = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        tampered = bytearray(envelope.wrapped_key)
        tampered[-1] ^= 0xFF
        envelope.wrapped_key = bytes(tampered)
        session.commit()

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 500
    with app.state.session_factory() as session:
        assert session.get(ReleaseGrant, grant["grant_id"]).status == "pending"


def test_missing_historical_key_returns_500_and_grant_stays_pending(
    tmp_path, monkeypatch
):
    # Seal the envelope under version 1 with the legacy single key.
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/missing-key.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    grant = _setup(client1)
    app1.state.engine.dispose()

    # Restart with a keyring whose current version is 2 and which no
    # longer carries the historical version 1 needed to unwrap.
    new_key = b64url_encode(b"9876543210fedcba9876543210fedcba")
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING",
        json.dumps({"current_version": 2, "keys": {"2": new_key}}),
    )
    app2 = create_app(url)
    client2 = TestClient(app2)
    response = _release(client2, grant["grant_id"], grant["capability"])
    assert response.status_code == 500
    with app2.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.consumed_at is None
    app2.state.engine.dispose()


def test_invalid_keyring_returns_500_and_grant_stays_pending(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/bad-keyring.db"
    app1 = create_app(url)
    grant = _setup(TestClient(app1))
    app1.state.engine.dispose()

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "{not valid json")
    app2 = create_app(url)
    response = _release(TestClient(app2), grant["grant_id"], grant["capability"])
    assert response.status_code == 500
    with app2.state.session_factory() as session:
        assert session.get(ReleaseGrant, grant["grant_id"]).status == "pending"
    app2.state.engine.dispose()


def test_release_unwraps_with_envelope_recorded_key_version(tmp_path, monkeypatch):
    # Seal under version 1, then rotate the current version to 2 without
    # rewrapping: release must select version 1 from the envelope row.
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/rotated.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    grant = _setup(client1)
    app1.state.engine.dispose()

    new_key_bytes = b"9876543210fedcba9876543210fedcba"
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING",
        json.dumps(
            {
                "current_version": 2,
                "keys": {"1": MASTER_KEY, "2": b64url_encode(new_key_bytes)},
            }
        ),
    )
    app2 = create_app(url)
    client2 = TestClient(app2)
    with app2.state.session_factory() as session:
        envelope = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        assert envelope.key_version == 1

    response = _release(client2, grant["grant_id"], grant["capability"])
    assert response.status_code == 200
    assert response.json()["payload"] == PAYLOAD
    app2.state.engine.dispose()


def test_release_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    grant = _setup(client1)
    app1.state.engine.dispose()

    app2 = create_app(url)
    response = _release(TestClient(app2), grant["grant_id"], grant["capability"])
    assert response.status_code == 200
    assert response.json()["payload"] == PAYLOAD

    # The consumed state also survives.
    app2.state.engine.dispose()
    app3 = create_app(url)
    again = _release(TestClient(app3), grant["grant_id"], grant["capability"])
    assert again.status_code == 409
    app3.state.engine.dispose()


def test_plaintext_capability_payload_and_keys_never_logged(app, client, caplog):
    grant = _setup(client, payload=PAYLOAD)
    # Tamper so the release path logs a failure line that must not carry
    # the payload, capability or any key material.
    with app.state.session_factory() as session:
        envelope = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        tampered = bytearray(envelope.ciphertext)
        tampered[0] ^= 0x01
        envelope.ciphertext = bytes(tampered)
        session.commit()

    caplog.set_level(logging.ERROR, logger="proof_release")
    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 500

    assert PAYLOAD not in caplog.text
    assert grant["capability"] not in caplog.text
    assert MASTER_KEY not in caplog.text


def test_plaintext_payload_and_capability_never_persisted(app, client):
    grant = _setup(client, payload=PAYLOAD)
    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 200

    with app.state.session_factory() as session:
        grant_row = session.get(ReleaseGrant, grant["grant_id"])
        for column in grant_row.__table__.columns:
            value = getattr(grant_row, column.name)
            assert PAYLOAD not in str(value)
            assert grant["capability"] not in str(value)
        envelope = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        payload_bytes = PAYLOAD.encode("utf-8")
        for column in envelope.__table__.columns:
            value = getattr(envelope, column.name)
            blob = value if isinstance(value, bytes) else str(value).encode("utf-8")
            assert payload_bytes not in blob


def test_successful_release_produces_no_extra_audit_rows(app, client):
    grant = _setup(client)
    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 200
    with app.state.session_factory() as session:
        # Exactly the single grant row; one consumption, one audit record.
        rows = session.query(ReleaseGrant).all()
        assert len(rows) == 1
        assert rows[0].status == "consumed"
        assert rows[0].consumed_at is not None


def test_release_after_failed_decryption_can_still_succeed(app, client):
    grant = _setup(client)
    with app.state.session_factory() as session:
        envelope = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        tampered = bytearray(envelope.tag)
        tampered[0] ^= 0x01
        envelope.tag = bytes(tampered)
        session.commit()

    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 500

    # Restore the authentic tag: the still-pending grant now releases once.
    with app.state.session_factory() as session:
        envelope = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        # Re-seal the same payload to recover a valid tag.
        from proof_release.app import encrypt_payload
        from proof_release.envelopes import load_keyring

        sealed = encrypt_payload(
            load_keyring().current_key(), PAYLOAD.encode("utf-8")
        )
        envelope.ciphertext = sealed.ciphertext
        envelope.iv = sealed.iv
        envelope.tag = sealed.tag
        envelope.wrapped_key = sealed.wrapped_key
        session.commit()

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 200
    assert response.json()["payload"] == PAYLOAD
    assert _release(client, grant["grant_id"], grant["capability"]).status_code == 409
