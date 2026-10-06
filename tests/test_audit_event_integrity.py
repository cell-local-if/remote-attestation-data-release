"""Tests for GET /v1/compliance/audit-events/integrity.

The committed audit events of one ``(tenant_id, workload_id)`` scope form
a tamper-evident SHA-256 hash chain: grant lifecycle and rewrap
transactions append each event together with its chain metadata
(sequence, predecessor hash, event hash) and advance the per-scope range
head in the same commit. The integrity endpoint replays the chain
read-only and reports the first problem; it never repairs, rewrites or
advances anything.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from proof_release.app import create_app
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
SECRET = "unit-test-secret"

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V2_BYTES = b"fedcba9876543210fedcba9876543210"
KEY_V1 = b64url_encode(KEY_V1_BYTES)
KEY_V2 = b64url_encode(KEY_V2_BYTES)

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = create_app(f"sqlite:///{tmp_path}/audit_integrity.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _mac_for(nonce: str, claims: dict, *, tenant=TENANT, workload=WORKLOAD) -> str:
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


def _decision(client, *, tenant=TENANT, workload=WORKLOAD):
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    claims = {"m": "x"}
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": claims,
            "mac": _mac_for(
                created["nonce"], claims, tenant=tenant, workload=workload
            ),
        }
    )
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
    assert verified.status_code == 200
    policy = client.post(
        "/v1/policies",
        json={"tenant_id": tenant, "workload_id": workload, "name": "r",
              "rule": {"claim": "m", "equals": "x"}},
    ).json()
    decided = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert decided.status_code == 200
    return decided.json()


def _mint(client, decision_id, *, data_id="data-1", tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/release-grants",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "decision_id": decision_id,
            "data_id": data_id,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _consume(client, grant, *, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "capability": grant["capability"],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_envelope(client, data_id, *, tenant=TENANT, workload=WORKLOAD,
                     payload="secret-payload"):
    response = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": payload,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _integrity(client, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        "/v1/compliance/audit-events/integrity",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _valid_body(tenant=TENANT, workload=WORKLOAD, *, event_count, legacy_count,
                head_hash):
    return {
        "tenant_id": tenant,
        "workload_id": workload,
        "valid": True,
        "event_count": event_count,
        "legacy_count": legacy_count,
        "first_seq": 1 if event_count else 0,
        "last_seq": event_count,
        "head_hash": head_hash,
        "failure_code": None,
        "failure_seq": None,
    }


# --- request validation (all 422, no audit written) --------------------------


def test_requires_both_scope_parameters(client):
    assert client.get("/v1/compliance/audit-events/integrity").status_code == 422
    assert (
        client.get(
            "/v1/compliance/audit-events/integrity",
            params={"workload_id": WORKLOAD},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/v1/compliance/audit-events/integrity",
            params={"tenant_id": TENANT},
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "   ", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": "\t"},
    ],
)
def test_blank_scope_parameters_rejected(client, params):
    assert (
        client.get("/v1/compliance/audit-events/integrity", params=params).status_code
        == 422
    )


def test_duplicate_scope_parameters_rejected(client):
    response = client.get(
        "/v1/compliance/audit-events/integrity",
        params=[
            ("tenant_id", TENANT),
            ("tenant_id", OTHER_TENANT),
            ("workload_id", WORKLOAD),
        ],
    )
    assert response.status_code == 422
    response = client.get(
        "/v1/compliance/audit-events/integrity",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("workload_id", OTHER_WORKLOAD),
        ],
    )
    assert response.status_code == 422


def test_unknown_query_parameter_rejected(client):
    assert (
        client.get(
            "/v1/compliance/audit-events/integrity",
            params={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "cursor": "abc",
            },
        ).status_code
        == 422
    )


def test_non_empty_body_rejected(client):
    response = client.request(
        "GET",
        "/v1/compliance/audit-events/integrity",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"{}",
    )
    assert response.status_code == 422


# --- valid chains ------------------------------------------------------------


def test_empty_scope_is_valid_with_null_head(client):
    response = _integrity(client)
    assert response.status_code == 200, response.text
    assert response.content == (
        b'{"tenant_id":"tenant-a","workload_id":"workload-1","valid":true,'
        b'"event_count":0,"legacy_count":0,"first_seq":0,"last_seq":0,'
        b'"head_hash":null,"failure_code":null,"failure_seq":null}\n'
    )


def test_grant_lifecycle_chain_is_valid(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
    _consume(client, grant)

    response = _integrity(client)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["valid"] is True
    assert data["event_count"] == 2
    assert data["legacy_count"] == 0
    assert data["first_seq"] == 1
    assert data["last_seq"] == 2
    assert HEX64_RE.fullmatch(data["head_hash"])
    assert data["failure_code"] is None
    assert data["failure_seq"] is None
    # The response carries no capability, payload, evidence or key material.
    assert grant["capability"] not in response.text


def test_rewrap_events_continue_the_same_chain(client, monkeypatch):
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(2, {1: KEY_V1, 2: KEY_V2})
    )
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    _create_envelope(client, "e1")
    rewrapped = client.post(
        "/v1/rewrap-batches", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert rewrapped.status_code == 200, rewrapped.text

    data = _integrity(client).json()
    assert data["valid"] is True
    # One grant event plus one rewrap event in a single gap-free chain.
    assert data["event_count"] == 2
    assert data["first_seq"] == 1
    assert data["last_seq"] == 2
    assert HEX64_RE.fullmatch(data["head_hash"])


def test_scopes_are_isolated(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])

    other = _integrity(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    assert other.status_code == 200
    assert other.json() == _valid_body(
        OTHER_TENANT, OTHER_WORKLOAD, event_count=0, legacy_count=0, head_hash=None
    )
    mine = _integrity(client).json()
    assert mine["valid"] is True
    assert mine["event_count"] == 1


def test_repeated_response_is_byte_stable(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    first = _integrity(client)
    second = _integrity(client)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.content == second.content


def test_new_events_extend_the_chain(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="d1")
    first = _integrity(client).json()
    assert first["event_count"] == 1
    assert first["last_seq"] == 1

    _mint(client, decision["decision_id"], data_id="d2")
    second = _integrity(client).json()
    assert second["valid"] is True
    assert second["event_count"] == 2
    assert second["first_seq"] == 1
    assert second["last_seq"] == 2
    assert second["head_hash"] != first["head_hash"]


def test_concurrent_appends_keep_one_gap_free_chain(client):
    decision = _decision(client)

    def _mint_one(index):
        return _mint(client, decision["decision_id"], data_id=f"d-{index}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_mint_one, range(16)))
    assert len({grant["grant_id"] for grant in results}) == 16

    data = _integrity(client).json()
    assert data["valid"] is True
    assert data["event_count"] == 16
    assert data["first_seq"] == 1
    assert data["last_seq"] == 16
    assert HEX64_RE.fullmatch(data["head_hash"])


# --- tamper detection ---------------------------------------------------------


def _three_events(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="d1")
    grant = _mint(client, decision["decision_id"], data_id="d2")
    _consume(client, grant)


def test_modified_event_settles_hash_mismatch(client, app):
    _three_events(client)
    with app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE audit_events SET status = 'revoked' WHERE chain_seq = 1")
        )
    data = _integrity(client).json()
    assert data["valid"] is False
    assert data["failure_code"] == "hash-mismatch"
    assert data["failure_seq"] == 1
    assert data["event_count"] == 3


def test_deleted_middle_event_settles_sequence_gap(client, app):
    _three_events(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DELETE FROM audit_events WHERE chain_seq = 2"))
    data = _integrity(client).json()
    assert data["valid"] is False
    assert data["failure_code"] == "sequence-gap"
    assert data["failure_seq"] == 2
    assert data["event_count"] == 2


def test_deleted_tail_event_settles_head_mismatch(client, app):
    _three_events(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DELETE FROM audit_events WHERE chain_seq = 3"))
    data = _integrity(client).json()
    assert data["valid"] is False
    assert data["failure_code"] == "head-mismatch"
    assert data["failure_seq"] == 3
    assert data["event_count"] == 2
    assert data["last_seq"] == 2


def test_reordered_events_settle_hash_mismatch(client, app):
    _three_events(client)
    with app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE audit_events SET chain_seq = -1 WHERE chain_seq = 2")
        )
        conn.execute(text("UPDATE audit_events SET chain_seq = 2 WHERE chain_seq = 3"))
        conn.execute(text("UPDATE audit_events SET chain_seq = 3 WHERE chain_seq = -1"))
    data = _integrity(client).json()
    assert data["valid"] is False
    assert data["failure_code"] == "hash-mismatch"
    assert data["failure_seq"] == 2


def test_rewritten_head_settles_head_mismatch(client, app):
    _three_events(client)
    with app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE audit_chain_heads SET head_hash = :forged "
                "WHERE tenant_id = :tenant AND workload_id = :workload"
            ),
            {
                "forged": "f" * 64,
                "tenant": TENANT,
                "workload": WORKLOAD,
            },
        )
    data = _integrity(client).json()
    assert data["valid"] is False
    assert data["failure_code"] == "head-mismatch"
    assert data["failure_seq"] == 3


def test_tamper_in_other_scope_does_not_move_this_chain(client, app):
    _three_events(client)
    other_decision = _decision(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    _mint(client, other_decision["decision_id"], tenant=OTHER_TENANT,
          workload=OTHER_WORKLOAD)
    with app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE audit_events SET status = 'revoked' "
                "WHERE tenant_id = :tenant AND workload_id = :workload"
            ),
            {"tenant": OTHER_TENANT, "workload": OTHER_WORKLOAD},
        )
    assert _integrity(client).json()["valid"] is True
    tampered = _integrity(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()
    assert tampered["valid"] is False
    assert tampered["failure_code"] == "hash-mismatch"


# --- legacy migration ----------------------------------------------------------


def _strip_chain_metadata(app):
    """Rewind the database to a pre-chain deployment's audit table."""
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP INDEX ix_audit_events_scope_chain_seq"))
        conn.execute(
            text(
                "UPDATE audit_events "
                "SET chain_seq = NULL, prev_hash = NULL, event_hash = NULL"
            )
        )
        conn.execute(text("DELETE FROM audit_chain_heads"))


def test_legacy_database_is_chained_on_first_open(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/legacy.db"

    first = create_app(url)
    client = TestClient(first)
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
    _consume(client, grant)
    _strip_chain_metadata(first)
    first.state.engine.dispose()

    # Reopening chains the pre-existing events in (occurred_at, event_id)
    # order and builds the scope's head; the events are reported as legacy.
    second = create_app(url)
    try:
        client = TestClient(second)
        data = _integrity(client).json()
        assert data == _valid_body(
            event_count=2, legacy_count=2, head_hash=data["head_hash"]
        )
        assert HEX64_RE.fullmatch(data["head_hash"])

        # New events continue the migrated chain without touching the
        # legacy count.
        _mint(client, decision["decision_id"], data_id="d2")
        continued = _integrity(client).json()
        assert continued["valid"] is True
        assert continued["event_count"] == 3
        assert continued["legacy_count"] == 2
        assert continued["first_seq"] == 1
        assert continued["last_seq"] == 3
    finally:
        second.state.engine.dispose()


def test_legacy_migration_preserves_listing_order(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/legacy_order.db"

    first = create_app(url)
    client = TestClient(first)
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="d1")
    _mint(client, decision["decision_id"], data_id="d2")
    listed_before = client.get(
        "/v1/compliance/audit-events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()["events"]
    _strip_chain_metadata(first)
    first.state.engine.dispose()

    second = create_app(url)
    try:
        client = TestClient(second)
        # The migrated chain follows the audit listing's stable
        # (occurred_at, event_id) order.
        with second.state.engine.begin() as conn:
            rows = conn.execute(
                text(
                    "SELECT event_id, chain_seq FROM audit_events "
                    "ORDER BY chain_seq"
                )
            ).fetchall()
        assert [row[0] for row in rows] == [row["event_id"] for row in listed_before]
        assert [row[1] for row in rows] == [1, 2]
        assert _integrity(client).json()["valid"] is True
    finally:
        second.state.engine.dispose()


# --- availability --------------------------------------------------------------


def test_database_failure_is_500(client, app):
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE audit_events"))
    response = _integrity(client)
    assert response.status_code == 500
    assert response.json()["detail"] == "audit integrity unavailable"
