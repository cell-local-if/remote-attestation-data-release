"""Tests for POST /v1/release-grants/{grant_id}/revoke.

Covers revocation of a pending one-time grant and the shared atomic state
race across revocation, grant consumption and payload release.
"""

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
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
MASTER_KEY = b64url_encode(b"0123456789abcdef0123456789abcdef")
PAYLOAD = 'revocable secret payload'


def _rfc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _wrong(token: str) -> str:
    """A well-formed token guaranteed to differ from ``token``.

    Flipping the first character keeps the base64url alphabet and length
    intact (so it is an authentication failure, not a 422 format error)
    while never colliding with the real value.
    """
    return ("B" if token[0] != "B" else "C") + token[1:]


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
    return decided.json()


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


def _setup(client, data_id=DATA_ID, payload=PAYLOAD):
    decision = _decision(client)
    _envelope(client, data_id=data_id, payload=payload)
    return _grant(client, decision["decision_id"], data_id=data_id)


def _revoke(client, grant_id, capability, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release-grants/{grant_id}/revoke", json=body)


def _consume(client, grant_id, capability, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release-grants/{grant_id}/consume", json=body)


def _release(client, grant_id, capability, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": DATA_ID,
        "capability": capability,
    }
    body.update(fields)
    return client.post(f"/v1/release/{grant_id}", json=body)


# ---------------------------------------------------------------------------
# Success shape
# ---------------------------------------------------------------------------


def test_revoke_pending_grant_returns_200(client):
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
    assert data["decision_id"] == grant["decision_id"]
    assert data["data_id"] == DATA_ID
    assert data["revoked"] is True
    revoked_at = datetime.fromisoformat(data["revoked_at"])
    assert revoked_at.utcoffset() == timedelta(0)
    # The plaintext capability is only ever on the create response.
    assert "capability" not in response.text
    assert grant["capability"] not in response.text


def test_revoke_sets_revoked_state_exactly_once(client, app):
    grant = _setup(client)

    response = _revoke(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 200
    revoked_at = response.json()["revoked_at"]

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "revoked"
        assert row.revoked_at is not None
        assert row.consumed_at is None
        assert _rfc(row.revoked_at) == revoked_at
        # The grant row is the sole audit record; no extra table is
        # involved in revocation.
        assert session.query(ReleaseGrant).count() == 1


def test_revoke_accepts_uppercase_path_uuid(client):
    grant = _setup(client)

    response = _revoke(
        client, grant["grant_id"].upper(), grant["capability"]
    )
    assert response.status_code == 200
    assert response.json()["grant_id"] == grant["grant_id"]


# ---------------------------------------------------------------------------
# 404: unknown or cross-scope
# ---------------------------------------------------------------------------


def test_revoke_unknown_grant_returns_404(client):
    _decision(client)
    response = _revoke(
        client, "00000000-0000-0000-0000-000000000000", "A" * 43
    )
    assert response.status_code == 404
    assert "capability" not in response.text


def test_revoke_cross_scope_returns_404(client, app):
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
    # Cross-scope failures change no state.
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.revoked_at is None


def test_404_takes_precedence_over_401(client):
    _decision(client)
    response = client.post(
        "/v1/release-grants/00000000-0000-0000-0000-000000000000/revoke",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": "A" * 43,
        },
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 401: capability mismatch leaves all state untouched
# ---------------------------------------------------------------------------


def test_revoke_wrong_capability_returns_401_and_grant_stays_pending(client, app):
    grant = _setup(client)
    wrong = _wrong(grant["capability"])

    response = _revoke(client, grant["grant_id"], wrong)
    assert response.status_code == 401
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.revoked_at is None
        assert row.consumed_at is None

    # The correct capability still revokes afterwards.
    follow_up = _revoke(client, grant["grant_id"], grant["capability"])
    assert follow_up.status_code == 200


def test_wrong_capability_on_expired_grant_is_still_401(client, app):
    grant = _setup(client)
    wrong = _wrong(grant["capability"])
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    assert _revoke(client, grant["grant_id"], wrong).status_code == 401


# ---------------------------------------------------------------------------
# 409: consumed or already revoked
# ---------------------------------------------------------------------------


def test_revoke_consumed_grant_returns_409(client, app):
    grant = _setup(client)
    assert (
        _consume(client, grant["grant_id"], grant["capability"]).status_code
        == 200
    )

    response = _revoke(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 409
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"
        assert row.revoked_at is None


def test_revoke_after_release_returns_409(client, app):
    grant = _setup(client)
    assert (
        _release(client, grant["grant_id"], grant["capability"]).status_code
        == 200
    )

    response = _revoke(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 409
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"
        assert row.revoked_at is None


def test_revoke_twice_second_returns_409_and_keeps_timestamp(client, app):
    grant = _setup(client)

    first = _revoke(client, grant["grant_id"], grant["capability"])
    assert first.status_code == 200
    first_at = first.json()["revoked_at"]

    with app.state.session_factory() as session:
        before = session.get(ReleaseGrant, grant["grant_id"]).revoked_at

    second = _revoke(client, grant["grant_id"], grant["capability"])
    assert second.status_code == 409

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "revoked"
        assert row.revoked_at == before
        assert _rfc(row.revoked_at) == first_at
        # Exactly one audit row still.
        assert session.query(ReleaseGrant).count() == 1


# ---------------------------------------------------------------------------
# 410: expiry is judged before the write
# ---------------------------------------------------------------------------


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


def test_revoked_grant_reports_409_on_revoke_and_consume_even_after_expiry(
    client, app
):
    grant = _setup(client)
    assert (
        _revoke(client, grant["grant_id"], grant["capability"]).status_code
        == 200
    )
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    # On the revoke and consume paths the settled status is observed before
    # expiry; a repeated revoke or a consume is 409, not 410, and rewrites
    # nothing.
    assert (
        _revoke(client, grant["grant_id"], grant["capability"]).status_code
        == 409
    )
    assert (
        _consume(client, grant["grant_id"], grant["capability"]).status_code
        == 409
    )
    # The release endpoint keeps its long-standing judgement order — expiry
    # is reported before the settled status (the same precedence an expired
    # consumed grant has always had) — and still releases nothing.
    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 410
    assert PAYLOAD not in response.text
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "revoked"
        assert row.consumed_at is None


# ---------------------------------------------------------------------------
# Revoked grants are unusable by consume and release
# ---------------------------------------------------------------------------


def test_consume_revoked_grant_returns_409(client, app):
    grant = _setup(client)
    assert (
        _revoke(client, grant["grant_id"], grant["capability"]).status_code
        == 200
    )

    response = _consume(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 409
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "revoked"
        assert row.consumed_at is None


def test_release_revoked_grant_returns_409_without_payload(client, app):
    grant = _setup(client)
    assert (
        _revoke(client, grant["grant_id"], grant["capability"]).status_code
        == 200
    )

    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 409
    assert "payload" not in response.text
    assert PAYLOAD not in response.text
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "revoked"
        assert row.consumed_at is None
        assert session.query(ReleaseGrant).count() == 1


def test_revoke_after_consume_loses_and_writes_nothing(client, app):
    grant = _setup(client)
    consumed_at = _consume(
        client, grant["grant_id"], grant["capability"]
    ).json()["consumed_at"]

    assert (
        _revoke(client, grant["grant_id"], grant["capability"]).status_code
        == 409
    )
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "consumed"
        assert row.revoked_at is None
        assert _rfc(row.consumed_at) == consumed_at


# ---------------------------------------------------------------------------
# 422: fields and path identifier
# ---------------------------------------------------------------------------


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
        {"capability": 9},
        {"capability": True},
    ],
)
def test_revoke_rejects_invalid_fields(client, app, overrides):
    grant = _setup(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }
    body.update(overrides)
    assert client.post(
        f"/v1/release-grants/{grant['grant_id']}/revoke", json=body
    ).status_code == 422
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.revoked_at is None


def test_revoke_rejects_non_object_body(client):
    grant = _setup(client)
    assert client.post(
        f"/v1/release-grants/{grant['grant_id']}/revoke",
        json=["not", "an", "object"],
    ).status_code == 422


@pytest.mark.parametrize(
    "grant_id",
    [
        "not-a-uuid",
        "12345",
        "00000000000000000000000000000000",
        "gggggggg-gggg-gggg-gggg-gggggggggggg",
        "00000000-0000-0000-0000-00000000000",
        "000000000-0000-0000-0000-000000000000",
    ],
)
def test_revoke_invalid_path_identifier_returns_422(client, grant_id):
    _decision(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": "A" * 43,
    }
    assert client.post(
        f"/v1/release-grants/{grant_id}/revoke", json=body
    ).status_code == 422


def test_revoke_empty_path_segment_returns_422(client):
    _decision(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": "A" * 43,
    }
    assert client.post(
        "/v1/release-grants//revoke", json=body
    ).status_code == 422


def test_revoke_valid_uuid_path_unknown_row_returns_404(client):
    _decision(client)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": "A" * 43,
    }
    response = client.post(
        "/v1/release-grants/00000000-0000-0000-0000-000000000000/revoke",
        json=body,
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Concurrency: revoke / consume / release share one atomic transition
# ---------------------------------------------------------------------------


def test_concurrent_revokes_only_one_succeeds(app):
    client = TestClient(app)
    grant = _setup(client)
    url = f"/v1/release-grants/{grant['grant_id']}/revoke"
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }

    def call():
        return TestClient(app).post(url, json=body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: call(), range(16)))

    assert statuses.count(200) == 1
    assert statuses.count(409) == 15
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "revoked"
        assert row.revoked_at is not None
        assert row.consumed_at is None
        assert session.query(ReleaseGrant).count() == 1


def test_concurrent_revoke_and_consume_single_win(app):
    client = TestClient(app)
    grant = _setup(client)
    revoke_body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "capability": grant["capability"],
    }
    consume_body = dict(revoke_body)

    def call(i):
        c = TestClient(app)
        if i % 2 == 0:
            return c.post(
                f"/v1/release-grants/{grant['grant_id']}/revoke",
                json=revoke_body,
            ).status_code
        return c.post(
            f"/v1/release-grants/{grant['grant_id']}/consume",
            json=consume_body,
        ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(call, range(16)))

    # Exactly one legal settlement across both operations; every loser
    # observes the final state as 409, nothing else.
    assert statuses.count(200) == 1
    assert all(code in (200, 409) for code in statuses)
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status in ("revoked", "consumed")
        assert session.query(ReleaseGrant).count() == 1


def test_concurrent_revoke_and_release_single_win(app):
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

    payloads: list[str] = []

    def call(i):
        c = TestClient(app)
        if i % 2 == 0:
            return c.post(
                f"/v1/release-grants/{grant['grant_id']}/revoke",
                json=revoke_body,
            ).status_code
        response = c.post(
            f"/v1/release/{grant['grant_id']}", json=release_body
        )
        if response.status_code == 200:
            payloads.append(response.text)
        return response.status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(call, range(16)))

    assert statuses.count(200) == 1
    assert all(code in (200, 409) for code in statuses)
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status in ("revoked", "consumed")
        # One audit row regardless of which operation won.
        assert session.query(ReleaseGrant).count() == 1
    # The payload is released at most once, and never when revoke won.
    assert len(payloads) <= 1
    if row.status == "revoked":
        assert payloads == []


def test_release_that_loses_revocation_race_does_not_release(app, monkeypatch):
    """Deterministic ordering: revoke settles first, release observes 409."""
    client = TestClient(app)
    grant = _setup(client)

    revoked = _revoke(client, grant["grant_id"], grant["capability"])
    assert revoked.status_code == 200
    response = _release(client, grant["grant_id"], grant["capability"])
    assert response.status_code == 409
    assert PAYLOAD not in response.text


# ---------------------------------------------------------------------------
# Persistence / secrecy
# ---------------------------------------------------------------------------


def test_revocation_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart-revoke.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    grant = _setup(client1)
    revoked_at = _revoke(
        client1, grant["grant_id"], grant["capability"]
    ).json()["revoked_at"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    # Repeated revoke, consume and release all observe the terminal state.
    assert (
        _revoke(client2, grant["grant_id"], grant["capability"]).status_code
        == 409
    )
    assert (
        _consume(client2, grant["grant_id"], grant["capability"]).status_code
        == 409
    )
    assert (
        _release(client2, grant["grant_id"], grant["capability"]).status_code
        == 409
    )
    with app2.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "revoked"
        assert _rfc(row.revoked_at) == revoked_at
    app2.state.engine.dispose()


def test_plaintext_capability_never_persisted_or_logged_after_revoke(
    app, client, caplog
):
    grant = _setup(client, payload=PAYLOAD)
    # A failed capability attempt exercises the 401 logging-free path too.
    wrong = _wrong(grant["capability"])
    assert _revoke(client, grant["grant_id"], wrong).status_code == 401
    assert (
        _revoke(client, grant["grant_id"], grant["capability"]).status_code
        == 200
    )

    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.capability_digest == hashlib.sha256(
            grant["capability"].encode("ascii")
        ).hexdigest()
        for column in row.__table__.columns:
            value = getattr(row, column.name)
            assert grant["capability"] not in str(value)
            assert PAYLOAD not in str(value)
    assert grant["capability"] not in caplog.text
    assert MASTER_KEY not in caplog.text


def test_revocation_does_not_touch_envelope(app, client):
    grant = _setup(client)
    assert (
        _revoke(client, grant["grant_id"], grant["capability"]).status_code
        == 200
    )

    # A different, non-revoked grant for the same data item can still be
    # minted and releases; revocation is scoped to the single grant.
    decision = _decision(client)
    second = _grant(client, decision["decision_id"])
    response = _release(client, second["grant_id"], second["capability"])
    assert response.status_code == 200
    assert response.json()["payload"] == PAYLOAD
