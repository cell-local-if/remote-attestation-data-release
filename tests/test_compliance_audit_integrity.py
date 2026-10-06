"""Tests for GET /v1/compliance/audit-events/integrity.

Tamper-evident SHA-256 hash chain over one scope's committed compliance
audit events. Grant lifecycle and rewrap appends carry their chain
metadata (per-scope sequence, previous hash, event hash) and advance the
per-scope range head in the same transaction as the state transition;
legacy databases are chained on first open. The endpoint is read-only:
it never repairs the chain, rewrites an event or advances the head.
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
from proof_release.db import AuditChainHead, AuditEvent
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

PATH = "/v1/compliance/audit-events/integrity"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

FIELD_ORDER = [
    "tenant_id",
    "workload_id",
    "valid",
    "event_count",
    "legacy_count",
    "first_seq",
    "last_seq",
    "head_hash",
    "failure_code",
    "failure_seq",
]


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


def _create_envelope(client, data_id, *, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": "secret-payload",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _integrity(client, *, tenant=TENANT, workload=WORKLOAD, **kwargs):
    params = kwargs.pop(
        "params", {"tenant_id": tenant, "workload_id": workload}
    )
    return client.get(PATH, params=params, **kwargs)


def _chain_rows(app, *, tenant=TENANT, workload=WORKLOAD):
    with app.state.session_factory() as session:
        return (
            session.query(AuditEvent)
            .filter_by(tenant_id=tenant, workload_id=workload)
            .order_by(AuditEvent.chain_seq)
            .all()
        )


def _head(app, *, tenant=TENANT, workload=WORKLOAD):
    with app.state.session_factory() as session:
        return session.get(AuditChainHead, (tenant, workload))


# --- request validation ----------------------------------------------------


def test_requires_both_scope_parameters(client):
    assert client.get(PATH).status_code == 422
    assert client.get(PATH, params={"tenant_id": TENANT}).status_code == 422
    assert client.get(PATH, params={"workload_id": WORKLOAD}).status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "   ", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": " \t"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "bogus": "1"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "event_id": "x"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "cursor": ""},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": "10"},
    ],
)
def test_rejects_blank_and_unknown_parameters(client, params):
    assert client.get(PATH, params=params).status_code == 422


@pytest.mark.parametrize(
    "pairs",
    [
        [("tenant_id", TENANT), ("tenant_id", TENANT),
         ("workload_id", WORKLOAD)],
        [("tenant_id", TENANT), ("workload_id", WORKLOAD),
         ("workload_id", WORKLOAD)],
        [("tenant_id", TENANT), ("workload_id", WORKLOAD),
         ("tenant_id", "other")],
    ],
)
def test_rejects_repeated_parameters(client, pairs):
    assert client.get(PATH, params=pairs).status_code == 422


@pytest.mark.parametrize("content", [b"{}", b" ", b"\n", b"not-json"])
def test_rejects_non_empty_body(client, content):
    response = client.request(
        "GET",
        PATH,
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=content,
    )
    assert response.status_code == 422


def test_invalid_requests_write_no_state(app, client):
    client.get(PATH)
    client.get(PATH, params={"tenant_id": TENANT, "workload_id": WORKLOAD,
                             "bogus": "1"})
    client.request(
        "GET",
        PATH,
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"{}",
    )
    with app.state.session_factory() as session:
        assert session.query(AuditEvent).count() == 0
        assert session.query(AuditChainHead).count() == 0


# --- empty scope / response shape ------------------------------------------


def test_empty_scope_is_valid_with_zeroed_range(client):
    response = _integrity(client)
    assert response.status_code == 200
    assert response.json() == {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "valid": True,
        "event_count": 0,
        "legacy_count": 0,
        "first_seq": 0,
        "last_seq": 0,
        "head_hash": None,
        "failure_code": None,
        "failure_seq": None,
    }


def test_response_is_compact_json_with_fixed_key_order(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    response = _integrity(client)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == FIELD_ORDER


def test_response_never_contains_capability(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
    response = _integrity(client)
    assert grant["capability"] not in response.text


# --- valid chains ------------------------------------------------------------


def test_grant_lifecycle_forms_valid_chain(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="d1")

    first = _integrity(client).json()
    assert first["valid"] is True
    assert first["event_count"] == 1
    assert first["legacy_count"] == 0
    assert first["first_seq"] == 1
    assert first["last_seq"] == 1
    assert HEX64_RE.fullmatch(first["head_hash"])
    assert first["failure_code"] is None
    assert first["failure_seq"] is None

    consumed = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD,
              "capability": grant["capability"]},
    )
    assert consumed.status_code == 200

    second = _integrity(client).json()
    assert second["valid"] is True
    assert second["event_count"] == 2
    assert second["first_seq"] == 1
    assert second["last_seq"] == 2
    assert HEX64_RE.fullmatch(second["head_hash"])
    assert second["head_hash"] != first["head_hash"]


def test_rewrap_events_extend_the_same_chain(client, monkeypatch):
    _create_envelope(client, "e1")
    _create_envelope(client, "e2")
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(2, {1: KEY_V1, 2: KEY_V2})
    )
    response = client.post(
        "/v1/rewrap-batches", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert response.status_code == 200, response.text

    report = _integrity(client).json()
    assert report["valid"] is True
    assert report["event_count"] == 2
    assert report["first_seq"] == 1
    assert report["last_seq"] == 2


def test_grant_and_rewrap_events_share_one_chain(client, monkeypatch):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="d")
    _create_envelope(client, "e")
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING", _keyring(2, {1: KEY_V1, 2: KEY_V2})
    )
    client.post(
        "/v1/rewrap-batches", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    report = _integrity(client).json()
    assert report["valid"] is True
    assert report["event_count"] == 2
    assert report["last_seq"] == 2


def test_repeated_response_is_byte_stable(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    first = _integrity(client)
    second = _integrity(client)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_scopes_have_independent_chains(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="a")
    other = _integrity(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()
    assert other["valid"] is True
    assert other["event_count"] == 0
    assert other["head_hash"] is None
    mine = _integrity(client).json()
    assert mine["valid"] is True
    assert mine["event_count"] == 1


# --- tamper detection --------------------------------------------------------


def _mint_three(client):
    decision = _decision(client)
    for i in range(3):
        _mint(client, decision["decision_id"], data_id=f"d{i}")
    assert _integrity(client).json()["valid"] is True


def test_modified_event_field_is_hash_mismatch(app, client):
    _mint_three(client)
    with app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE audit_events SET status = 'revoked' WHERE chain_seq = 2")
        )
    report = _integrity(client).json()
    assert report["valid"] is False
    assert report["failure_code"] == "hash-mismatch"
    assert report["failure_seq"] == 2


def test_modified_prev_hash_is_hash_mismatch(app, client):
    _mint_three(client)
    with app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE audit_events SET prev_hash = :bogus WHERE chain_seq = 3"),
            {"bogus": "a" * 64},
        )
    report = _integrity(client).json()
    assert report["valid"] is False
    assert report["failure_code"] == "hash-mismatch"
    assert report["failure_seq"] == 3


def test_deleted_middle_event_is_sequence_gap(app, client):
    _mint_three(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DELETE FROM audit_events WHERE chain_seq = 2"))
    report = _integrity(client).json()
    assert report["valid"] is False
    assert report["failure_code"] == "sequence-gap"
    assert report["failure_seq"] == 2
    assert report["event_count"] == 2


def test_deleted_first_event_is_sequence_gap(app, client):
    _mint_three(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DELETE FROM audit_events WHERE chain_seq = 1"))
    report = _integrity(client).json()
    assert report["valid"] is False
    assert report["failure_code"] == "sequence-gap"
    assert report["failure_seq"] == 1


def test_deleted_last_event_is_head_mismatch(app, client):
    _mint_three(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DELETE FROM audit_events WHERE chain_seq = 3"))
    report = _integrity(client).json()
    assert report["valid"] is False
    assert report["failure_code"] == "head-mismatch"
    assert report["failure_seq"] == 3
    assert report["last_seq"] == 2


def test_reordered_events_are_hash_mismatch(app, client):
    _mint_three(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("UPDATE audit_events SET chain_seq = -1 WHERE chain_seq = 1"))
        conn.execute(text("UPDATE audit_events SET chain_seq = 1 WHERE chain_seq = 2"))
        conn.execute(text("UPDATE audit_events SET chain_seq = 2 WHERE chain_seq = -1"))
    report = _integrity(client).json()
    assert report["valid"] is False
    assert report["failure_code"] == "hash-mismatch"
    assert report["failure_seq"] == 1


def test_tampered_head_hash_is_head_mismatch(app, client):
    _mint_three(client)
    with app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE audit_chain_heads SET head_hash = :bogus"),
            {"bogus": "f" * 64},
        )
    report = _integrity(client).json()
    assert report["valid"] is False
    assert report["failure_code"] == "head-mismatch"
    assert report["failure_seq"] == 3
    assert report["head_hash"] == "f" * 64


def test_tampered_head_sequence_is_head_mismatch(app, client):
    _mint_three(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("UPDATE audit_chain_heads SET last_seq = last_seq + 5"))
    report = _integrity(client).json()
    assert report["valid"] is False
    assert report["failure_code"] == "head-mismatch"
    assert report["failure_seq"] == 8


def test_missing_head_is_head_mismatch(app, client):
    _mint_three(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DELETE FROM audit_chain_heads"))
    report = _integrity(client).json()
    assert report["valid"] is False
    assert report["failure_code"] == "head-mismatch"
    assert report["failure_seq"] == 3
    assert report["head_hash"] is None


def test_tampering_in_one_scope_leaves_other_scopes_valid(app, client):
    _mint_three(client)
    decision = _decision(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    _mint(client, decision["decision_id"], data_id="b",
          tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    with app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE audit_events SET status = 'revoked' WHERE chain_seq = 1 "
                 "AND tenant_id = :t"),
            {"t": TENANT},
        )
    report = _integrity(client).json()
    assert report["valid"] is False
    assert report["failure_code"] == "hash-mismatch"
    other = _integrity(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()
    assert other["valid"] is True
    assert other["event_count"] == 1


# --- read-only behaviour -----------------------------------------------------


def test_verification_writes_no_state(app, client):
    _mint_three(client)
    before_events = [
        (row.event_id, row.chain_seq, row.prev_hash, row.event_hash)
        for row in _chain_rows(app)
    ]
    head = _head(app)
    before_head = (head.last_seq, head.head_hash, head.legacy_count)
    for _ in range(3):
        assert _integrity(client).status_code == 200
    after_events = [
        (row.event_id, row.chain_seq, row.prev_hash, row.event_hash)
        for row in _chain_rows(app)
    ]
    head = _head(app)
    after_head = (head.last_seq, head.head_hash, head.legacy_count)
    assert before_events == after_events
    assert before_head == after_head


def test_failed_verification_does_not_repair_chain(app, client):
    _mint_three(client)
    with app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE audit_events SET status = 'revoked' WHERE chain_seq = 2")
        )
    first = _integrity(client).json()
    second = _integrity(client).json()
    assert first == second
    assert first["valid"] is False
    # The tampered row is still there, unrepaired.
    with app.state.session_factory() as session:
        row = (
            session.query(AuditEvent)
            .filter_by(tenant_id=TENANT, workload_id=WORKLOAD, chain_seq=2)
            .one()
        )
        assert row.status == "revoked"


# --- concurrency -------------------------------------------------------------


def test_concurrent_appends_keep_chain_valid(app, monkeypatch):
    import proof_release.app as app_module

    monkeypatch.setattr(app_module, "GRANT_BUDGET_PER_MINUTE", 64)
    client = TestClient(app)
    decision = _decision(client)
    grants = [
        _mint(client, decision["decision_id"], data_id=f"d{i}") for i in range(6)
    ]

    def settle(index):
        local = TestClient(app)
        grant = grants[index]
        path = (
            f"/v1/release-grants/{grant['grant_id']}/"
            + ("consume" if index % 2 == 0 else "revoke")
        )
        response = local.post(
            path,
            json={"tenant_id": TENANT, "workload_id": WORKLOAD,
                  "capability": grant["capability"]},
        )
        assert response.status_code == 200

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(settle, range(6)))

    report = _integrity(client).json()
    assert report["valid"] is True
    assert report["event_count"] == 12
    assert report["first_seq"] == 1
    assert report["last_seq"] == 12

    rows = _chain_rows(app)
    assert [row.chain_seq for row in rows] == list(range(1, 13))


def test_integrity_observes_consistent_chain_under_concurrent_appends(
    app, monkeypatch
):
    import proof_release.app as app_module

    monkeypatch.setattr(app_module, "GRANT_BUDGET_PER_MINUTE", 64)
    client = TestClient(app)
    decision = _decision(client)
    grants = [
        _mint(client, decision["decision_id"], data_id=f"d{i}") for i in range(6)
    ]

    def settle(index):
        local = TestClient(app)
        grant = grants[index]
        return local.post(
            f"/v1/release-grants/{grant['grant_id']}/consume",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD,
                  "capability": grant["capability"]},
        ).status_code

    def verify(_):
        local = TestClient(app)
        response = local.get(
            PATH, params={"tenant_id": TENANT, "workload_id": WORKLOAD}
        )
        assert response.status_code == 200
        # Whatever prefix of the appends was committed, the observed chain
        # is internally consistent.
        assert response.json()["valid"] is True
        return response.json()["event_count"]

    def task(pair):
        kind, index = pair
        if kind == "settle":
            return settle(index)
        return verify(index)

    tasks = []
    for i in range(6):
        tasks.append(("settle", i))
        tasks.append(("verify", i))
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(task, tasks))
    assert all(code == 200 for code in results[0::2])

    final = _integrity(client).json()
    assert final["valid"] is True
    assert final["event_count"] == 12


# --- server failure ----------------------------------------------------------


def test_dropped_events_table_returns_500(app, client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE audit_events"))
    response = _integrity(client)
    assert response.status_code == 500
    assert response.json() == {"detail": "audit integrity unavailable"}


def test_dropped_heads_table_returns_500(app, client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE audit_chain_heads"))
    response = _integrity(client)
    assert response.status_code == 500
    assert response.json() == {"detail": "audit integrity unavailable"}


# --- legacy migration --------------------------------------------------------


def _strip_chain_from_database(url):
    """Rewrite a current database to its pre-chain shape."""
    from sqlalchemy import create_engine

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX IF EXISTS ix_audit_events_scope_chain_seq"))
        conn.execute(text("ALTER TABLE audit_events DROP COLUMN chain_seq"))
        conn.execute(text("ALTER TABLE audit_events DROP COLUMN prev_hash"))
        conn.execute(text("ALTER TABLE audit_events DROP COLUMN event_hash"))
        conn.execute(text("DROP TABLE audit_chain_heads"))
    engine.dispose()


def test_legacy_database_is_chained_on_first_open(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/legacy.db"

    first = create_app(url)
    client1 = TestClient(first)
    decision = _decision(client1)
    grant = _mint(client1, decision["decision_id"], data_id="legacy")
    consumed = client1.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD,
              "capability": grant["capability"]},
    )
    assert consumed.status_code == 200
    first.state.engine.dispose()
    _strip_chain_from_database(url)

    second = create_app(url)
    client2 = TestClient(second)
    report = client2.get(
        PATH, params={"tenant_id": TENANT, "workload_id": WORKLOAD}
    ).json()
    assert report["valid"] is True
    assert report["event_count"] == 2
    assert report["legacy_count"] == 2
    assert report["first_seq"] == 1
    assert report["last_seq"] == 2
    assert HEX64_RE.fullmatch(report["head_hash"])

    # New events continue the backfilled chain; the legacy count is fixed.
    grant2 = _mint(client2, decision["decision_id"], data_id="fresh")
    assert grant2["grant_id"]
    continued = client2.get(
        PATH, params={"tenant_id": TENANT, "workload_id": WORKLOAD}
    ).json()
    assert continued["valid"] is True
    assert continued["event_count"] == 3
    assert continued["legacy_count"] == 2
    assert continued["last_seq"] == 3
    second.state.engine.dispose()


def test_reopening_current_database_keeps_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    url = f"sqlite:///{tmp_path}/restart.db"

    first = create_app(url)
    client1 = TestClient(first)
    decision = _decision(client1)
    _mint(client1, decision["decision_id"], data_id="persist")
    before = client1.get(
        PATH, params={"tenant_id": TENANT, "workload_id": WORKLOAD}
    ).json()
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    after = client2.get(
        PATH, params={"tenant_id": TENANT, "workload_id": WORKLOAD}
    ).json()
    assert after == before
    assert after["valid"] is True
    assert after["legacy_count"] == 0
    second.state.engine.dispose()
