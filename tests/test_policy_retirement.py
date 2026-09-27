"""Tests for POST /v1/policies/{policy_id}/retire.

Covers the terminal per-version retirement of a versioned release
policy: path/body validation, the indistinguishable 404, the compact
200 response, repeat/concurrency settlement, write failure rollback,
restart persistence, the additive upgrade of a pre-retirement database,
version isolation, and the compatibility guarantees — a retired
version refuses new decisions with 409 and writes no business or audit
state, while existing decisions keep their status/policy_version/
decided_at and existing release grants keep their full lifecycle.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy import event as sqla_event

from proof_release.app import create_app
from proof_release.db import Decision, Policy

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
SECRET = "unit-test-secret"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    # A fixed 32-byte master key for the envelope-backed grant tests.
    monkeypatch.setenv(
        "PROOF_RELEASE_MASTER_KEY",
        base64.urlsafe_b64encode(b"0" * 32).rstrip(b"=").decode("ascii"),
    )
    application = create_app(f"sqlite:///{tmp_path}/policies.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


# --- helpers ---------------------------------------------------------------


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


def _challenge(client, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    )
    assert response.status_code == 201
    return response.json()


def _receive_and_verify(client, claims=None, *, tenant=TENANT, workload=WORKLOAD):
    created = _challenge(client, tenant, workload)
    evidence = _evidence(created["nonce"], claims)
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
    assert verified.json()["status"] == "verified"
    return created, evidence, evidence_id


def _receive_only(client, claims=None):
    created = _challenge(client)
    evidence = _evidence(created["nonce"], claims)
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
    return created, evidence, submitted.json()["evidence_id"]


def _policy(client, rule, name="release", *, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/policies",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "name": name,
            "rule": rule,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _retire(client, policy, *, tenant=TENANT, workload=WORKLOAD, **extra):
    body = {"tenant_id": tenant, "workload_id": workload}
    body.update(extra)
    return client.post(f"/v1/policies/{policy}/retire", json=body)


def _decide(client, created, evidence, evidence_id, policy_id):
    return client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy_id,
        },
    )


def _iso(value):
    return value.astimezone(timezone.utc).isoformat()


def _table_counts(application, *tables):
    with application.state.engine.connect() as conn:
        return {
            table: conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            for table in tables
        }


@pytest.fixture()
def policy_id(client):
    return _policy(client, {"claim": "region", "equals": "eu"})["policy_id"]


# --- happy path and response shape ----------------------------------------


def test_retire_returns_200_compact_ordered_json(client, policy_id):
    response = _retire(client, policy_id)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    # Exactly one terminating newline; no surrounding whitespace.
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert response.content == response.content.strip() + b"\n"
    # Field order and field set are fixed, and every value is a string.
    data = response.json()
    assert list(data.keys()) == ["policy_id", "status", "retired_at"]
    assert data["policy_id"] == policy_id
    assert data["status"] == "retired"
    retired_at = datetime.fromisoformat(data["retired_at"])
    assert retired_at.utcoffset() == timedelta(0)
    assert all(isinstance(value, str) for value in data.values())
    # Byte-exact body: retirement result only, no rule, floats or metadata.
    assert response.content == (
        b'{"policy_id":"'
        + policy_id.encode("ascii")
        + b'","status":"retired","retired_at":"'
        + data["retired_at"].encode("ascii")
        + b'"}\n'
    )


def test_retire_persists_status_and_time(app, client, policy_id):
    response = _retire(client, policy_id)
    retired_at = response.json()["retired_at"]

    with app.state.session_factory() as session:
        policy = session.get(Policy, policy_id)
        assert policy.status == "retired"
        assert policy.retired_at is not None
        assert _iso(policy.retired_at) == retired_at


def test_retire_writes_no_audit_or_other_record(app, client, policy_id):
    before = _table_counts(
        app, "decisions", "proof_lifecycle_events", "audit_events"
    )
    assert _retire(client, policy_id).status_code == 200
    after = _table_counts(
        app, "decisions", "proof_lifecycle_events", "audit_events"
    )
    assert after == before


def test_retire_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    created = _policy(client1, {"claim": "a", "equals": 1})
    policy = created["policy_id"]
    retired = _retire(client1, policy)
    assert retired.status_code == 200
    retired_at = retired.json()["retired_at"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        record = session.get(Policy, policy)
        assert record.status == "retired"
        assert _iso(record.retired_at) == retired_at
    # A repeat retire after restart is still a stable 409.
    client2 = TestClient(app2)
    assert _retire(client2, policy).status_code == 409
    app2.state.engine.dispose()


# --- path validation -------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [
        "not-a-uuid",
        "ABCDEFAB-ABCD-ABCD-ABCD-ABCDEFABCDEF",  # uppercase hex rejected
        "deadbeefdead-dead-dead-dead-deadbeefdead",  # malformed grouping
        "deadbeef-dead-dead-dead",  # truncated
        "%20",
        "%09",
    ],
)
def test_retire_invalid_path_identifier_returns_422(client, policy_id, policy):
    response = client.request(
        "POST",
        f"/v1/policies/{policy}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_retire_whitespace_padded_path_identifier_returns_422(client, policy_id):
    response = client.request(
        "POST",
        f"/v1/policies/%09{policy_id}%09/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422
    # The version must still be active: a malformed path never retires.
    with client.app.state.session_factory() as session:
        assert session.get(Policy, policy_id).status == "active"


def test_retire_empty_path_segment_returns_422(client):
    response = client.post(
        "/v1/policies//retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_retire_invalid_path_does_not_touch_storage(app, client, policy_id):
    statements: list[str] = []

    def spy(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    sqla_event.listen(app.state.engine, "before_cursor_execute", spy)
    try:
        response = client.post(
            "/v1/policies/not-a-uuid/retire",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
    finally:
        sqla_event.remove(app.state.engine, "before_cursor_execute", spy)
    assert response.status_code == 422
    # No policy read or write occurred for a malformed path id.
    assert not any("policies" in s for s in statements)
    with app.state.session_factory() as session:
        assert session.get(Policy, policy_id).status == "active"


# --- body validation --------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        None,
        b"",
        b"   ",
        b"not json",
        {"workload_id": WORKLOAD},
        {"tenant_id": TENANT},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "extra": 1},
        {"tenant_id": 1, "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": 2},
        {"tenant_id": None, "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": None},
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "   ", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": "\t"},
        [],
        "a string",
        42,
    ],
)
def test_retire_invalid_bodies_return_422(client, policy_id, payload):
    if payload is None:
        response = client.post(
            f"/v1/policies/{policy_id}/retire",
            headers={"content-type": "application/json"},
            content=b"",
        )
    elif isinstance(payload, (bytes, bytearray)):
        response = client.post(
            f"/v1/policies/{policy_id}/retire",
            headers={"content-type": "application/json"},
            content=bytes(payload),
        )
    else:
        response = client.post(
            f"/v1/policies/{policy_id}/retire", json=payload
        )
    assert response.status_code == 422
    with client.app.state.session_factory() as session:
        policy = session.get(Policy, policy_id)
        assert policy.status == "active"
        assert policy.retired_at is None


def test_retire_unknown_field_rejects_without_writing(client, policy_id):
    response = _retire(client, policy_id, unexpected="x")
    assert response.status_code == 422
    with client.app.state.session_factory() as session:
        assert session.get(Policy, policy_id).status == "active"


# --- 404 semantics ----------------------------------------------------------


def test_retire_unknown_policy_returns_404(client, policy_id):
    response = _retire(client, "66666666-6666-6666-6666-666666666666")
    assert response.status_code == 404


def test_retire_wrong_tenant_or_workload_returns_404(client, policy_id):
    assert _retire(client, policy_id, tenant=OTHER_TENANT).status_code == 404
    assert _retire(client, policy_id, workload=OTHER_WORKLOAD).status_code == 404
    # Still active after the cross-scope attempts.
    with client.app.state.session_factory() as session:
        assert session.get(Policy, policy_id).status == "active"


def test_retire_404_is_indistinguishable_for_unknown_and_cross_scope(client):
    other = _policy(
        client,
        {"claim": "a", "equals": 1},
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
    )["policy_id"]

    unknown = _retire(client, "77777777-7777-7777-7777-777777777777")
    cross_scope = _retire(client, other)
    assert unknown.status_code == cross_scope.status_code == 404
    assert unknown.text == cross_scope.text


# --- repeat and concurrency -------------------------------------------------


def test_repeat_retire_returns_409_and_keeps_first_time(client, policy_id):
    first = _retire(client, policy_id)
    assert first.status_code == 200
    first_time = first.json()["retired_at"]

    second = _retire(client, policy_id)
    assert second.status_code == 409
    with client.app.state.session_factory() as session:
        policy = session.get(Policy, policy_id)
        assert policy.status == "retired"
        assert _iso(policy.retired_at) == first_time


def test_repeat_retire_appends_no_other_state_record(app, client, policy_id):
    assert _retire(client, policy_id).status_code == 200
    before = _table_counts(
        app, "decisions", "proof_lifecycle_events", "audit_events"
    )

    assert _retire(client, policy_id).status_code == 409

    assert _table_counts(
        app, "decisions", "proof_lifecycle_events", "audit_events"
    ) == before


def test_concurrent_retires_settle_once(app, policy_id):
    def call():
        return TestClient(app).post(
            f"/v1/policies/{policy_id}/retire",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )

    # A barrier maximizes contention despite SQLite serialization.
    barrier = threading.Barrier(8)

    def gated_call():
        barrier.wait()
        return call()

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: gated_call(), range(8)))

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(200) == 1
    assert statuses.count(409) == 7
    bodies = [r.json() for r in responses if r.status_code == 200]
    assert len(bodies) == 1
    with app.state.session_factory() as session:
        policy = session.get(Policy, policy_id)
        assert policy.status == "retired"
        assert policy.retired_at is not None


# --- write/read failure -----------------------------------------------------


def test_retire_write_failure_rolls_back(app, client, policy_id):
    def fail_update(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE policies"):
            raise RuntimeError("simulated storage failure")

    sqla_event.listen(app.state.engine, "before_cursor_execute", fail_update)
    try:
        response = _retire(client, policy_id)
        assert response.status_code == 500
    finally:
        sqla_event.remove(app.state.engine, "before_cursor_execute", fail_update)

    with app.state.session_factory() as session:
        policy = session.get(Policy, policy_id)
        assert policy.status == "active"
        assert policy.retired_at is None

    # After recovery the retire succeeds; no exception text is echoed.
    recovered = _retire(client, policy_id)
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "retired"
    assert "simulated storage failure" not in recovered.text


def test_retire_read_failure_returns_500_and_writes_nothing(app, client, policy_id):
    def fail_read(conn, cursor, statement, parameters, context, executemany):
        # The retire handler loads the full policy row by id.
        if statement.lstrip().startswith("SELECT policies.policy_id,"):
            raise RuntimeError("simulated storage failure")

    sqla_event.listen(app.state.engine, "before_cursor_execute", fail_read)
    try:
        response = _retire(client, policy_id)
        assert response.status_code == 500
    finally:
        sqla_event.remove(app.state.engine, "before_cursor_execute", fail_read)

    with app.state.session_factory() as session:
        policy = session.get(Policy, policy_id)
        assert policy.status == "active"
        assert policy.retired_at is None


# --- version isolation ------------------------------------------------------


def test_retirement_is_scoped_to_one_version(client):
    v1 = _policy(client, {"claim": "a", "equals": 1}, name="p")
    v2 = _policy(client, {"claim": "a", "equals": 1}, name="p")
    assert (v1["version"], v2["version"]) == (1, 2)

    assert _retire(client, v1["policy_id"]).status_code == 200

    with client.app.state.session_factory() as session:
        assert session.get(Policy, v1["policy_id"]).status == "retired"
        assert session.get(Policy, v2["policy_id"]).status == "active"

    # Retirement cannot be bypassed by allocating another version: the
    # old row stays retired, the new version is independently active.
    v3 = _policy(client, {"claim": "a", "equals": 1}, name="p")
    assert v3["version"] == 3
    with client.app.state.session_factory() as session:
        assert session.get(Policy, v1["policy_id"]).status == "retired"
        assert session.get(Policy, v3["policy_id"]).status == "active"
    assert _retire(client, v1["policy_id"]).status_code == 409
    assert _retire(client, v2["policy_id"]).status_code == 200
    with client.app.state.session_factory() as session:
        assert session.get(Policy, v3["policy_id"]).status == "active"


def test_retirement_does_not_leak_rule_content(app, client):
    created = _policy(client, {"claim": "super-secret-claim", "equals": "x"})
    response = _retire(client, created["policy_id"])
    assert response.status_code == 200
    assert b"super-secret-claim" not in response.content


# --- effect on decisions ----------------------------------------------------


def test_retired_policy_refuses_new_decisions_with_409(app, client, policy_id):
    created, evidence, evidence_id = _receive_and_verify(client, {"region": "eu"})
    assert _retire(client, policy_id).status_code == 200

    before = _table_counts(
        app, "decisions", "proof_lifecycle_events", "audit_events"
    )
    response = _decide(client, created, evidence, evidence_id, policy_id)
    assert response.status_code == 409
    assert response.json()["detail"] == "policy is retired"
    # No decision, no proof event, no audit row is written.
    assert _table_counts(
        app, "decisions", "proof_lifecycle_events", "audit_events"
    ) == before


def test_retired_policy_refuses_unverified_evidence_too(client, policy_id):
    # The retirement gate precedes the verified-state gate: even a
    # received (never verified) evidence cannot produce a decision
    # against a retired version.
    created, evidence, evidence_id = _receive_only(client, {"region": "eu"})
    assert _retire(client, policy_id).status_code == 200
    response = _decide(client, created, evidence, evidence_id, policy_id)
    assert response.status_code == 409
    assert response.json()["detail"] == "policy is retired"


def test_existing_decision_survives_retirement_unchanged(client, policy_id):
    created, evidence, evidence_id = _receive_and_verify(client, {"region": "eu"})
    first = _decide(client, created, evidence, evidence_id, policy_id)
    assert first.status_code == 200
    assert first.json()["status"] == "allowed"
    first_body = first.json()

    assert _retire(client, policy_id).status_code == 200

    # Retries after retirement return the identical stored result — the
    # status, snapshot policy_version and decided_at are never rewritten.
    repeated = _decide(client, created, evidence, evidence_id, policy_id)
    assert repeated.status_code == 200
    assert repeated.json() == first_body
    with client.app.state.session_factory() as session:
        row = session.get(Decision, first_body["decision_id"])
        assert row.status == "allowed"
        assert row.policy_version == first_body["policy_version"]
        assert _iso(row.decided_at) == first_body["decided_at"]


def test_other_version_still_decides_after_retirement(client):
    v1 = _policy(client, {"claim": "region", "equals": "eu"}, name="p")["policy_id"]
    v2 = _policy(client, {"claim": "region", "equals": "us"}, name="p")["policy_id"]

    created, evidence, evidence_id = _receive_and_verify(client, {"region": "us"})
    assert _decide(client, created, evidence, evidence_id, v1).json()["status"] == "denied"
    assert _retire(client, v1).status_code == 200

    # The retired v1 cannot take a new decision on fresh evidence; v2 can.
    created2, evidence2, evidence_id2 = _receive_and_verify(client, {"region": "us"})
    assert _decide(client, created2, evidence2, evidence_id2, v1).status_code == 409
    second = _decide(client, created2, evidence2, evidence_id2, v2)
    assert second.status_code == 200
    assert second.json()["status"] == "allowed"
    assert second.json()["policy_version"] == 2


def test_retirement_adds_no_proof_event(app, client, policy_id):
    # One decision (with its first-decision proof event) exists first.
    created, evidence, evidence_id = _receive_and_verify(client, {"region": "eu"})
    decision = _decide(client, created, evidence, evidence_id, policy_id)
    assert decision.status_code == 200

    assert _retire(client, policy_id).status_code == 200
    # Fresh evidence contributes its own receive/verify events; those are
    # the pre-decision baseline — the refused decision must add nothing.
    created2, evidence2, evidence_id2 = _receive_and_verify(client, {"region": "eu"})
    before = _table_counts(app, "decisions", "proof_lifecycle_events")
    assert _decide(
        client, created2, evidence2, evidence_id2, policy_id
    ).status_code == 409

    assert _table_counts(app, "decisions", "proof_lifecycle_events") == before


# --- existing release grants keep their lifecycle --------------------------


def test_retirement_does_not_revoke_or_change_existing_grant(client, policy_id):
    # A pre-retirement allowed decision authorizes the full grant flow,
    # which still completes after the policy version is retired.
    created, evidence, evidence_id = _receive_and_verify(client, {"region": "eu"})
    decision = _decide(client, created, evidence, evidence_id, policy_id)
    assert decision.json()["status"] == "allowed"
    decision_id = decision.json()["decision_id"]

    data_id = "item-1"
    envelope = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": data_id,
            "payload": "protected-payload",
        },
    )
    assert envelope.status_code == 201

    grant = client.post(
        "/v1/release-grants",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "decision_id": decision_id,
            "data_id": data_id,
        },
    )
    assert grant.status_code == 201
    grant_id = grant.json()["grant_id"]
    capability = grant.json()["capability"]

    assert _retire(client, policy_id).status_code == 200

    # Consumption after retirement still settles the pending grant.
    consumed = client.post(
        f"/v1/release-grants/{grant_id}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": capability,
        },
    )
    assert consumed.status_code == 200
    assert consumed.json()["consumed"] is True
    # The one-time transition is already settled; a repeat is 409,
    # exactly as before retirement existed.
    repeat = client.post(
        f"/v1/release-grants/{grant_id}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": capability,
        },
    )
    assert repeat.status_code == 409


def test_grant_release_after_retirement_releases_payload(client, policy_id):
    created, evidence, evidence_id = _receive_and_verify(client, {"region": "eu"})
    decision_id = _decide(
        client, created, evidence, evidence_id, policy_id
    ).json()["decision_id"]
    data_id = "item-2"
    assert client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": data_id,
            "payload": "the-plaintext",
        },
    ).status_code == 201
    grant = client.post(
        "/v1/release-grants",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "decision_id": decision_id,
            "data_id": data_id,
        },
    ).json()
    assert _retire(client, policy_id).status_code == 200

    released = client.post(
        f"/v1/release/{grant['grant_id']}",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": data_id,
            "capability": grant["capability"],
        },
    )
    assert released.status_code == 200
    assert released.json() == {"payload": "the-plaintext"}


# --- upgrade of a database written before retirement existed ---------------


def test_pre_retirement_database_is_upgraded_and_policy_starts_active(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/legacy.db"

    # Build a database whose policies table predates the lifecycle
    # columns (no status / retired_at).
    legacy_engine = create_engine(url)
    with legacy_engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE policies ("
                "policy_id VARCHAR(36) NOT NULL PRIMARY KEY, "
                "tenant_id VARCHAR(256) NOT NULL, "
                "workload_id VARCHAR(256) NOT NULL, "
                "name VARCHAR(256) NOT NULL, "
                "version INTEGER NOT NULL, "
                "rule_json TEXT NOT NULL, "
                "created_at DATETIME NOT NULL)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO policies "
                "(policy_id, tenant_id, workload_id, name, version, "
                "rule_json, created_at) VALUES "
                "(:pid, :t, :w, 'n', 1, '{}', :now)"
            ),
            {
                "pid": "55555555-5555-5555-5555-555555555555",
                "t": TENANT,
                "w": WORKLOAD,
                "now": "2026-01-01T00:00:00+00:00",
            },
        )
    legacy_engine.dispose()

    application = create_app(url)
    with application.state.session_factory() as session:
        policy = session.get(Policy, "55555555-5555-5555-5555-555555555555")
        assert policy.status == "active"
        assert policy.retired_at is None

    client = TestClient(application)
    response = _retire(client, "55555555-5555-5555-5555-555555555555")
    assert response.status_code == 200
    assert response.json()["status"] == "retired"
    application.state.engine.dispose()
