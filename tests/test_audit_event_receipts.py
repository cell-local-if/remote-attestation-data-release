"""Tests for the tamper-evident audit-event receipts:

GET /v1/compliance/audit-events/{event_id}/receipt
GET /v1/compliance/audit-events/integrity

Every compliance audit event is chained in its own write transaction to a
per-``(tenant_id, workload_id)`` receipt: a gap-free ``sequence`` from 1,
a ``previous_sha256`` (64 zeros for the scope's first event) and an
``event_sha256`` over the event's public fields and the previous digest.
The receipt stores only those chain fields — never a capability,
payload, key or evidence — and both endpoints are strictly read-only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from proof_release.app import create_app
from proof_release.db import AuditEvent, AuditEventChainLink
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

RECEIPT_FIELDS = [
    "event_id",
    "sequence",
    "previous_sha256",
    "event_sha256",
    "verified",
]
INTEGRITY_FIELDS = [
    "verified",
    "checked_count",
    "last_sequence",
    "head_event_sha256",
    "failure_event_id",
    "failure_reason",
]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = create_app(f"sqlite:///{tmp_path}/audit_receipts.db")
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


def _mint(client, decision_id, *, data_id="data-1", tenant=TENANT,
          workload=WORKLOAD):
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


def _events(client, *, tenant=TENANT, workload=WORKLOAD):
    response = client.get(
        "/v1/compliance/audit-events",
        params={"tenant_id": tenant, "workload_id": workload},
    )
    assert response.status_code == 200, response.text
    return response.json()["events"]


def _receipt(client, event_id, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        f"/v1/compliance/audit-events/{event_id}/receipt",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


def _integrity(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        "/v1/compliance/audit-events/integrity",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


def _expected_digest(event, sequence, previous_sha256):
    """Independent recomputation of a link's event digest."""
    canonical = json.dumps(
        {
            "tenant_id": event["tenant_id"],
            "workload_id": event["workload_id"],
            "event_id": event["event_id"],
            "event_type": event["event_type"],
            "grant_id": event["grant_id"],
            "decision_id": event["decision_id"],
            "data_id": event["data_id"],
            "status": event["status"],
            "occurred_at": event["occurred_at"],
            "capability_sha256": event["capability_sha256"],
            "sequence": sequence,
            "previous_sha256": previous_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _scoped_events(app, tenant=TENANT, workload=WORKLOAD):
    """Raw audit rows of one scope as plain dicts, in chain order."""
    with app.state.session_factory() as session:
        rows = (
            session.query(AuditEvent)
            .filter_by(tenant_id=tenant, workload_id=workload)
            .all()
        )
        return [
            {
                "tenant_id": row.tenant_id,
                "workload_id": row.workload_id,
                "event_id": row.event_id,
                "event_type": row.event_type,
                "grant_id": row.grant_id,
                "decision_id": row.decision_id,
                "data_id": row.data_id,
                "status": row.status,
                "occurred_at": row.occurred_at.astimezone(timezone.utc).isoformat(),
                "capability_sha256": row.capability_sha256,
            }
            for row in rows
        ]


# --- receipt request validation -------------------------------------------


def test_receipt_requires_scope_parameters(client):
    event_id = ZERO_UUID
    assert (
        client.get(f"/v1/compliance/audit-events/{event_id}/receipt").status_code
        == 422
    )
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
        {"bogus": "value"},
        {"cursor": "abc"},
        {"event_id": ZERO_UUID},
    ],
)
def test_receipt_rejects_invalid_parameters(client, params):
    response = client.get(
        f"/v1/compliance/audit-events/{ZERO_UUID}/receipt",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD, **params},
    )
    assert response.status_code == 422, response.text


def test_receipt_rejects_duplicate_parameters(client):
    url = (
        f"/v1/compliance/audit-events/{ZERO_UUID}/receipt"
        f"?tenant_id={TENANT}&tenant_id={TENANT}&workload_id={WORKLOAD}"
    )
    assert client.get(url).status_code == 422
    url = (
        f"/v1/compliance/audit-events/{ZERO_UUID}/receipt"
        f"?tenant_id={TENANT}&workload_id={WORKLOAD}&workload_id={WORKLOAD}"
    )
    assert client.get(url).status_code == 422


@pytest.mark.parametrize(
    "event_id",
    [
        "not-a-uuid",
        "abc123",
        ZERO_UUID[:-1] + "Z",
        "  " + ZERO_UUID,
        ZERO_UUID + " ",
    ],
)
def test_receipt_rejects_malformed_event_identifier(client, event_id):
    assert _receipt(client, event_id).status_code == 422


def test_integrity_rejects_invalid_parameters(client):
    assert client.get("/v1/compliance/audit-events/integrity").status_code == 422
    assert _integrity(client, tenant="").status_code == 422
    assert _integrity(client, workload="  ").status_code == 422
    assert _integrity(client, bogus="value").status_code == 422
    assert _integrity(client, event_id=ZERO_UUID).status_code == 422
    url = (
        "/v1/compliance/audit-events/integrity"
        f"?tenant_id={TENANT}&tenant_id={TENANT}&workload_id={WORKLOAD}"
    )
    assert client.get(url).status_code == 422


def test_invalid_parameters_write_no_state(app, client):
    _receipt(client, "not-a-uuid")
    _integrity(client, bogus="x")
    with app.state.session_factory() as session:
        assert session.query(AuditEventChainLink).count() == 0


# --- receipt 404 semantics ---------------------------------------------------


def test_receipt_unknown_event_returns_404(client):
    assert _receipt(client, ZERO_UUID).status_code == 404


def test_receipt_cross_scope_event_returns_404(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
    event_id = _events(client)[0]["event_id"]
    assert _receipt(client, event_id, tenant=OTHER_TENANT).status_code == 404
    assert _receipt(client, event_id, workload=OTHER_WORKLOAD).status_code == 404
    # The grant id space is a different identifier space and never matches.
    assert _receipt(client, grant["grant_id"]).status_code == 404


# --- receipt shape and chain -------------------------------------------------


def test_first_receipt_chains_to_genesis(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    event_id = _events(client)[0]["event_id"]
    response = _receipt(client, event_id)
    assert response.status_code == 200
    body = response.json()
    assert list(body) == RECEIPT_FIELDS
    assert body["event_id"] == event_id
    assert body["sequence"] == 1
    assert body["previous_sha256"] == GENESIS
    assert isinstance(body["event_sha256"], str)
    assert len(body["event_sha256"]) == 64
    assert body["verified"] is True


def test_chain_links_sequentially_across_events(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="d1")
    _mint(client, decision["decision_id"], data_id="d2")
    client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD,
              "capability": grant["capability"]},
    )
    events = _events(client)
    assert len(events) == 3
    receipts = [
        _receipt(client, row["event_id"]).json() for row in events
    ]
    assert [r["sequence"] for r in receipts] == [1, 2, 3]
    assert receipts[0]["previous_sha256"] == GENESIS
    for earlier, later in zip(receipts, receipts[1:]):
        assert later["previous_sha256"] == earlier["event_sha256"]
    assert all(r["verified"] is True for r in receipts)


def test_receipt_digest_recomputes_from_public_fields(app, client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="d1")
    _mint(client, decision["decision_id"], data_id="d2")
    rows = _scoped_events(app)
    previous = GENESIS
    for sequence, row in enumerate(rows, start=1):
        receipt = _receipt(client, row["event_id"]).json()
        assert receipt["sequence"] == sequence
        assert receipt["previous_sha256"] == previous
        assert receipt["event_sha256"] == _expected_digest(
            row, sequence, previous
        )
        previous = receipt["event_sha256"]


def test_receipt_response_is_compact_json_with_trailing_newline(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    event_id = _events(client)[0]["event_id"]
    response = _receipt(client, event_id)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == RECEIPT_FIELDS


def test_receipt_never_contains_plaintext_capability(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
    event_id = _events(client)[0]["event_id"]
    response = _receipt(client, event_id)
    assert grant["capability"] not in response.text
    assert _integrity(client).text.find(grant["capability"]) == -1


def test_chains_are_independent_per_scope(client):
    decision_a = _decision(client)
    _mint(client, decision_a["decision_id"], data_id="a")
    decision_b = _decision(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    _mint(client, decision_b["decision_id"], data_id="b",
          tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    event_a = _events(client)[0]["event_id"]
    event_b = _events(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    )[0]["event_id"]
    receipt_a = _receipt(client, event_a).json()
    receipt_b = _receipt(
        client, event_b, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    ).json()
    # Each scope starts its own chain at sequence 1 from the genesis
    # previous digest; the digests differ because the scope fields are
    # covered by the event digest.
    assert receipt_a["sequence"] == receipt_b["sequence"] == 1
    assert receipt_a["previous_sha256"] == receipt_b["previous_sha256"] == GENESIS
    assert receipt_a["event_sha256"] != receipt_b["event_sha256"]


# --- integrity endpoint -------------------------------------------------------


def test_integrity_of_empty_scope(client):
    response = _integrity(client)
    assert response.status_code == 200
    assert response.json() == {
        "verified": True,
        "checked_count": 0,
        "last_sequence": 0,
        "head_event_sha256": GENESIS,
        "failure_event_id": None,
        "failure_reason": None,
    }


def test_integrity_of_legal_chain(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="d1")
    _mint(client, decision["decision_id"], data_id="d2")
    client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD,
              "capability": grant["capability"]},
    )
    response = _integrity(client)
    assert response.status_code == 200
    body = response.json()
    assert list(body) == INTEGRITY_FIELDS
    assert body["verified"] is True
    assert body["checked_count"] == 3
    assert body["last_sequence"] == 3
    last_event = _events(client)[-1]["event_id"]
    assert body["head_event_sha256"] == _receipt(client, last_event).json()[
        "event_sha256"
    ]
    assert body["failure_event_id"] is None
    assert body["failure_reason"] is None


def test_integrity_response_is_compact_json_with_trailing_newline(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    response = _integrity(client)
    raw = response.content
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == INTEGRITY_FIELDS


def test_integrity_is_scoped_to_tenant_and_workload(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="a")
    assert _integrity(client, tenant=OTHER_TENANT).json()["checked_count"] == 0
    assert _integrity(client, workload=OTHER_WORKLOAD).json()["checked_count"] == 0


# --- read-only behaviour ------------------------------------------------------


def test_queries_write_no_state(app, client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    event_id = _events(client)[0]["event_id"]
    for _ in range(3):
        assert _receipt(client, event_id).status_code == 200
        assert _integrity(client).status_code == 200
    with app.state.session_factory() as session:
        assert session.query(AuditEvent).count() == 1
        assert session.query(AuditEventChainLink).count() == 1


# --- tamper evidence -----------------------------------------------------------


def _rewrite(app, statement, **params):
    with app.state.engine.begin() as conn:
        conn.execute(text(statement), params)


def test_tampered_event_field_fails_receipt_and_integrity(app, client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="d1")
    _mint(client, decision["decision_id"], data_id="d2")
    target = _events(client)[0]["event_id"]
    _rewrite(
        app,
        "UPDATE audit_events SET data_id = 'forged' WHERE event_id = :e",
        e=target,
    )
    receipt = _receipt(client, target).json()
    assert receipt["verified"] is False
    body = _integrity(client).json()
    assert body["verified"] is False
    assert body["checked_count"] == 0
    assert body["last_sequence"] == 0
    assert body["head_event_sha256"] == GENESIS
    assert body["failure_event_id"] == target
    assert body["failure_reason"] == "event-mismatch"


def test_tampered_link_digest_fails_closed(app, client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="d1")
    _mint(client, decision["decision_id"], data_id="d2")
    target = _events(client)[1]["event_id"]
    _rewrite(
        app,
        "UPDATE audit_event_chain_links SET event_sha256 = :h "
        "WHERE event_id = :e",
        h="ab" * 32,
        e=target,
    )
    assert _receipt(client, target).json()["verified"] is False
    body = _integrity(client).json()
    assert body["verified"] is False
    assert body["failure_event_id"] == target
    assert body["failure_reason"] == "event-mismatch"
    # The verified prefix covers exactly the first link.
    assert body["checked_count"] == 1
    assert body["last_sequence"] == 1


def test_unknown_previous_digest_is_missing_predecessor(app, client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="d1")
    _mint(client, decision["decision_id"], data_id="d2")
    target = _events(client)[1]["event_id"]
    _rewrite(
        app,
        "UPDATE audit_event_chain_links SET previous_sha256 = :h "
        "WHERE event_id = :e",
        h="cd" * 32,
        e=target,
    )
    assert _receipt(client, target).json()["verified"] is False
    body = _integrity(client).json()
    assert body["verified"] is False
    assert body["failure_event_id"] == target
    assert body["failure_reason"] == "missing-predecessor"


def test_cross_scope_previous_digest_is_chain_mismatch(app, client):
    decision_a = _decision(client)
    _mint(client, decision_a["decision_id"], data_id="a1")
    _mint(client, decision_a["decision_id"], data_id="a2")
    decision_b = _decision(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    _mint(client, decision_b["decision_id"], data_id="b1",
          tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    foreign = _receipt(
        client,
        _events(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)[0][
            "event_id"
        ],
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
    ).json()
    target = _events(client)[1]["event_id"]
    # Point the second link of scope A at scope B's head digest: the
    # referenced predecessor exists, but across the scope boundary.
    _rewrite(
        app,
        "UPDATE audit_event_chain_links SET previous_sha256 = :h "
        "WHERE event_id = :e",
        h=foreign["event_sha256"],
        e=target,
    )
    assert _receipt(client, target).json()["verified"] is False
    body = _integrity(client).json()
    assert body["verified"] is False
    assert body["failure_event_id"] == target
    assert body["failure_reason"] == "chain-mismatch"
    # Scope B's own chain is untouched.
    assert _integrity(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    ).json()["verified"] is True


def test_sequence_skip_is_sequence_mismatch(app, client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="d1")
    _mint(client, decision["decision_id"], data_id="d2")
    target = _events(client)[1]["event_id"]
    _rewrite(
        app,
        "UPDATE audit_event_chain_links SET sequence = 7 WHERE event_id = :e",
        e=target,
    )
    assert _receipt(client, target).json()["verified"] is False
    body = _integrity(client).json()
    assert body["verified"] is False
    assert body["failure_event_id"] == target
    assert body["failure_reason"] == "sequence-mismatch"


def test_deleted_link_breaks_the_chain(app, client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="d1")
    _mint(client, decision["decision_id"], data_id="d2")
    target = _events(client)[1]["event_id"]
    _rewrite(
        app,
        "DELETE FROM audit_event_chain_links WHERE event_id = :e",
        e=target,
    )
    # The event lost its receipt: the target is inconsistent.
    receipt = _receipt(client, target).json()
    assert receipt["verified"] is False
    body = _integrity(client).json()
    assert body["verified"] is False
    assert body["failure_event_id"] == target
    assert body["failure_reason"] == "event-mismatch"


# --- backfill and persistence ---------------------------------------------------


def test_pre_feature_events_are_backfilled_in_listing_order(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/backfill.db"
    first = create_app(url)
    client1 = TestClient(first)
    decision = _decision(client1)
    _mint(client1, decision["decision_id"], data_id="d1")
    _mint(client1, decision["decision_id"], data_id="d2")
    head_before = _integrity(client1).json()["head_event_sha256"]
    # Simulate a database written before receipts existed: no links, no
    # counters. The next open must re-chain the existing events.
    with first.state.engine.begin() as conn:
        conn.execute(text("DELETE FROM audit_event_chain_links"))
        conn.execute(text("DELETE FROM audit_event_chain_counters"))
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    body = _integrity(client2).json()
    assert body["verified"] is True
    assert body["checked_count"] == 2
    assert body["last_sequence"] == 2
    # The backfilled chain is identical to the original one.
    assert body["head_event_sha256"] == head_before
    # A new event written after the upgrade continues the chain at N+1.
    _mint(client2, decision["decision_id"], data_id="d3")
    body = _integrity(client2).json()
    assert body["verified"] is True
    assert body["checked_count"] == 3
    assert body["last_sequence"] == 3
    second.state.engine.dispose()


def test_receipts_are_stable_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    decision = _decision(client1)
    _mint(client1, decision["decision_id"], data_id="persist")
    event_id = _events(client1)[0]["event_id"]
    before = _receipt(client1, event_id).content
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    assert _receipt(client2, event_id).content == before
    assert _integrity(client2).json()["verified"] is True
    second.state.engine.dispose()


# --- concurrency -----------------------------------------------------------------


def test_concurrent_writes_take_unique_consecutive_sequences(app, monkeypatch):
    import proof_release.app as app_module

    monkeypatch.setattr(app_module, "GRANT_BUDGET_PER_MINUTE", 64)
    client = TestClient(app)
    decision = _decision(client)

    def mint(index):
        local = TestClient(app)
        response = local.post(
            "/v1/release-grants",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "decision_id": decision["decision_id"],
                "data_id": f"d{index:03d}",
            },
        )
        assert response.status_code == 201, response.text

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(mint, range(8)))

    with app.state.session_factory() as session:
        sequences = sorted(
            row.sequence
            for row in session.query(AuditEventChainLink)
            .filter_by(tenant_id=TENANT, workload_id=WORKLOAD)
            .all()
        )
    assert sequences == list(range(1, 9))
    body = _integrity(client).json()
    assert body["verified"] is True
    assert body["checked_count"] == 8
    assert body["last_sequence"] == 8


# --- server failure -----------------------------------------------------------------


def test_storage_failure_returns_500_and_no_receipt(app, client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    event_id = _events(client)[0]["event_id"]
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE audit_event_chain_links"))
    assert _receipt(client, event_id).status_code == 500
    assert _integrity(client).status_code == 500
