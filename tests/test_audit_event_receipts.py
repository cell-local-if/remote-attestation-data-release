"""Tests for the audit-event tamper-evident chain receipts.

GET /v1/compliance/audit-events/{event_id}/receipt and
GET /v1/compliance/audit-events/integrity expose the hash chain that
every compliance audit event is anchored to. Chain links are written in
the same transaction as the event they receipt; both endpoints are
read-only.
"""

from __future__ import annotations

import hashlib
import hmac
import json

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

ZERO_UUID = "00000000-0000-0000-0000-000000000000"
GENESIS = "0" * 64

KEY_V1_BYTES = b"0123456789abcdef0123456789abcdef"
KEY_V1 = b64url_encode(KEY_V1_BYTES)


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = create_app(f"sqlite:///{tmp_path}/receipts.db")
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
    return response


def _events(client, *, tenant=TENANT, workload=WORKLOAD):
    response = client.get(
        "/v1/compliance/audit-events",
        params={"tenant_id": tenant, "workload_id": workload},
    )
    assert response.status_code == 200, response.text
    return response.json()["events"]


def _receipt(client, event_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/compliance/audit-events/{event_id}/receipt",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _integrity(client, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        "/v1/compliance/audit-events/integrity",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _receipts(client, *, tenant=TENANT, workload=WORKLOAD):
    return [
        _receipt(client, row["event_id"], tenant=tenant, workload=workload).json()
        for row in _events(client, tenant=tenant, workload=workload)
    ]


# --- chain construction ----------------------------------------------------


def test_first_event_receipt_has_genesis_predecessor(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    (event,) = _events(client)
    receipt = _receipt(client, event["event_id"])
    assert receipt.status_code == 200, receipt.text
    data = receipt.json()
    assert data["event_id"] == event["event_id"]
    assert data["sequence"] == 1
    assert data["previous_sha256"] == GENESIS
    assert len(data["event_sha256"]) == 64
    assert data["verified"] is True
    # The receipt carries only chain fields.
    assert set(data) == {
        "event_id",
        "sequence",
        "previous_sha256",
        "event_sha256",
        "verified",
    }


def test_chain_links_sequential_events(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="c")
    _consume(client, grant)
    receipts = _receipts(client)
    assert [row["sequence"] for row in receipts] == [1, 2]
    assert receipts[0]["previous_sha256"] == GENESIS
    assert receipts[1]["previous_sha256"] == receipts[0]["event_sha256"]
    assert receipts[0]["event_sha256"] != receipts[1]["event_sha256"]
    assert all(row["verified"] is True for row in receipts)


def test_rewrap_events_join_the_same_chain(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    envelope = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": "d1",
            "payload": "secret",
        },
    )
    assert envelope.status_code == 201
    rotated = client.post(
        "/v1/data-envelopes/d1/rewrap",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    # Already current: a no-op retry writes no event.
    assert rotated.status_code == 200
    receipts = _receipts(client)
    assert [row["sequence"] for row in receipts] == [1]
    assert receipts[0]["verified"] is True


def test_scopes_have_independent_chains(client):
    decision = _decision(client)
    other_decision = _decision(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    _mint(client, decision["decision_id"], data_id="a")
    _mint(client, other_decision["decision_id"], data_id="b",
          tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    own = _receipts(client)
    other = _receipts(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    assert [row["sequence"] for row in own] == [1]
    assert [row["sequence"] for row in other] == [1]
    assert own[0]["previous_sha256"] == GENESIS
    assert other[0]["previous_sha256"] == GENESIS
    assert own[0]["event_sha256"] != other[0]["event_sha256"]


# --- integrity endpoint ----------------------------------------------------


def test_integrity_of_valid_chain(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="c")
    _consume(client, grant)
    response = _integrity(client)
    assert response.status_code == 200, response.text
    data = response.json()
    receipts = _receipts(client)
    assert data == {
        "verified": True,
        "checked_count": 2,
        "last_sequence": 2,
        "head_event_sha256": receipts[-1]["event_sha256"],
        "failure_event_id": None,
        "failure_reason": None,
    }


def test_integrity_of_empty_scope(client):
    data = _integrity(client).json()
    assert data == {
        "verified": True,
        "checked_count": 0,
        "last_sequence": 0,
        "head_event_sha256": None,
        "failure_event_id": None,
        "failure_reason": None,
    }


def test_integrity_is_scoped(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    assert _integrity(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()[
        "checked_count"
    ] == 0


# --- tamper detection ------------------------------------------------------


def test_tampered_event_field_is_event_mismatch(app, client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    (event,) = _events(client)
    with app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE audit_events SET status = 'revoked' WHERE event_id = :eid"),
            {"eid": event["event_id"]},
        )
    receipt = _receipt(client, event["event_id"]).json()
    assert receipt["verified"] is False
    integrity = _integrity(client).json()
    assert integrity["verified"] is False
    assert integrity["failure_event_id"] == event["event_id"]
    assert integrity["failure_reason"] == "event-mismatch"


def test_tampered_previous_digest_to_unknown_is_missing_predecessor(app, client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="c")
    _consume(client, grant)
    events = _events(client)
    with app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE audit_event_chain_links SET previous_sha256 = :digest "
                "WHERE event_id = :eid"
            ),
            {"digest": "f" * 64, "eid": events[1]["event_id"]},
        )
    assert _receipt(client, events[1]["event_id"]).json()["verified"] is False
    integrity = _integrity(client).json()
    assert integrity["verified"] is False
    assert integrity["failure_event_id"] == events[1]["event_id"]
    assert integrity["failure_reason"] == "missing-predecessor"


def test_tampered_link_to_existing_digest_is_chain_mismatch(app, client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="c")
    _consume(client, grant)
    events = _events(client)
    receipts = _receipts(client)
    with app.state.engine.begin() as conn:
        # Point the second link at its own digest: the referenced digest
        # exists, but it is not the predecessor's.
        conn.execute(
            text(
                "UPDATE audit_event_chain_links SET previous_sha256 = :digest "
                "WHERE event_id = :eid"
            ),
            {
                "digest": receipts[1]["event_sha256"],
                "eid": events[1]["event_id"],
            },
        )
    assert _receipt(client, events[1]["event_id"]).json()["verified"] is False
    integrity = _integrity(client).json()
    assert integrity["verified"] is False
    assert integrity["failure_event_id"] == events[1]["event_id"]
    assert integrity["failure_reason"] == "chain-mismatch"


def test_deleted_link_is_sequence_mismatch(app, client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="c")
    _consume(client, grant)
    _mint(client, decision["decision_id"], data_id="d2")
    events = _events(client)
    with app.state.engine.begin() as conn:
        conn.execute(
            text("DELETE FROM audit_event_chain_links WHERE event_id = :eid"),
            {"eid": events[1]["event_id"]},
        )
    assert _receipt(client, events[2]["event_id"]).json()["verified"] is False
    integrity = _integrity(client).json()
    assert integrity["verified"] is False
    assert integrity["failure_event_id"] == events[2]["event_id"]
    assert integrity["failure_reason"] == "sequence-mismatch"


def test_tamper_in_one_scope_leaves_other_scope_verified(app, client):
    decision = _decision(client)
    other_decision = _decision(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    _mint(client, decision["decision_id"], data_id="a")
    _mint(client, other_decision["decision_id"], data_id="b",
          tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    (event,) = _events(client)
    with app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE audit_events SET status = 'revoked' WHERE event_id = :eid"),
            {"eid": event["event_id"]},
        )
    assert _integrity(client).json()["verified"] is False
    assert _integrity(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()[
        "verified"
    ] is True


# --- backfill and restart ---------------------------------------------------


def test_legacy_events_are_chained_on_open(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/legacy.db"
    first = create_app(url)
    client1 = TestClient(first)
    decision = _decision(client1)
    grant = _mint(client1, decision["decision_id"], data_id="c")
    _consume(client1, grant)
    before = _receipts(client1)
    # Simulate a pre-chain database: drop every link and counter.
    with first.state.engine.begin() as conn:
        conn.execute(text("DELETE FROM audit_event_chain_links"))
        conn.execute(text("DELETE FROM audit_event_chain_counters"))
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    after = _receipts(client2)
    # Backfilled in (occurred_at, event_id) order with identical receipts.
    assert after == before
    assert _integrity(client2).json()["verified"] is True
    # New events continue the backfilled chain at N+1.
    _mint(client2, decision["decision_id"], data_id="d2")
    receipts = _receipts(client2)
    assert [row["sequence"] for row in receipts] == [1, 2, 3]
    assert receipts[2]["previous_sha256"] == receipts[1]["event_sha256"]
    assert _integrity(client2).json()["verified"] is True
    second.state.engine.dispose()


def test_receipt_is_stable_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    decision = _decision(client1)
    _mint(client1, decision["decision_id"])
    before = _receipts(client1)
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    assert _receipts(client2) == before
    assert _integrity(client2).json()["verified"] is True
    second.state.engine.dispose()


# --- request validation ----------------------------------------------------


def test_receipt_requires_scope_parameters(client):
    event_id = ZERO_UUID
    assert client.get(f"/v1/compliance/audit-events/{event_id}/receipt").status_code == 422
    assert (
        client.get(
            f"/v1/compliance/audit-events/{event_id}/receipt",
            params={"workload_id": WORKLOAD},
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/v1/compliance/audit-events/{event_id}/receipt",
            params={"tenant_id": TENANT},
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"workload_id": "\t"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "cursor": "x"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "sequence": "1"},
    ],
)
def test_receipt_rejects_bad_query_parameters(client, params):
    response = client.get(
        f"/v1/compliance/audit-events/{ZERO_UUID}/receipt", params=params
    )
    assert response.status_code == 422


def test_receipt_rejects_duplicate_query_parameters(client):
    response = client.get(
        f"/v1/compliance/audit-events/{ZERO_UUID}/receipt"
        f"?tenant_id={TENANT}&tenant_id={TENANT}&workload_id={WORKLOAD}"
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "event_id",
    ["", " ", "not-a-uuid", "abc123", ZERO_UUID[:-1] + "Z", "  " + ZERO_UUID],
)
def test_receipt_rejects_malformed_event_identifier(client, event_id):
    response = client.get(
        f"/v1/compliance/audit-events/{event_id}/receipt",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_receipt_requires_empty_body(client):
    response = client.request(
        "GET",
        f"/v1/compliance/audit-events/{ZERO_UUID}/receipt",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"{}",
    )
    assert response.status_code == 422


def test_integrity_rejects_bad_query_parameters(client):
    assert client.get("/v1/compliance/audit-events/integrity").status_code == 422
    assert (
        client.get(
            "/v1/compliance/audit-events/integrity",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD, "event_id": ZERO_UUID},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/v1/compliance/audit-events/integrity"
            f"?tenant_id={TENANT}&workload_id={WORKLOAD}&workload_id={WORKLOAD}"
        ).status_code
        == 422
    )
    assert (
        client.request(
            "GET",
            "/v1/compliance/audit-events/integrity",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            content=b" ",
        ).status_code
        == 422
    )


def test_receipt_unknown_event_is_404(client):
    response = _receipt(client, ZERO_UUID)
    assert response.status_code == 404


def test_receipt_cross_scope_event_is_404(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    (event,) = _events(client)
    assert _receipt(client, event["event_id"], tenant=OTHER_TENANT).status_code == 404
    assert (
        _receipt(client, event["event_id"], workload=OTHER_WORKLOAD).status_code == 404
    )


# --- failure and non-regression --------------------------------------------


def test_storage_failure_returns_500(app, client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    (event,) = _events(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE audit_event_chain_links"))
    assert _receipt(client, event["event_id"]).status_code == 500
    assert _integrity(client).status_code == 500


def test_audit_listing_is_unchanged(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    (event,) = _events(client)
    assert set(event) == {
        "event_id",
        "event_type",
        "grant_id",
        "decision_id",
        "data_id",
        "status",
        "occurred_at",
        "capability_sha256",
    }
