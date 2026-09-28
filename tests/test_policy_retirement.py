"""Tests for POST /v1/policies/{policy_id}/retire — terminal retirement of
one concrete versioned policy — and for its compatibility guarantees:

* retired policy versions produce no new decisions (409, no state change);
* existing decisions keep status/policy_version/decided_at;
* existing release grants (capability, consumption, revocation, payload
  release) are entirely unaffected;
* different policy versions are isolated, state survives restart, and the
  terminal state cannot be bypassed by overwriting a version.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, text

from proof_release.app import create_app
from proof_release.db import Decision, Evidence, Policy, ProofLifecycleEvent

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
SECRET = "unit-test-secret"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    # A fixed master key so envelope creation/release and the release
    # summary work in-process without external provisioning.
    monkeypatch.setenv(
        "PROOF_RELEASE_MASTER_KEY",
        "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY",
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


def _policy(client, rule=None, *, name="release", tenant=TENANT,
            workload=WORKLOAD):
    response = client.post(
        "/v1/policies",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "name": name,
            "rule": rule or {"claim": "measurement", "equals": "abc"},
        },
    )
    assert response.status_code == 201
    return response.json()


def _retire(client, policy, *, tenant=TENANT, workload=WORKLOAD, **extra):
    body = {"tenant_id": tenant, "workload_id": workload}
    body.update(extra)
    return client.post(f"/v1/policies/{policy}/retire", json=body)


def _challenge(client, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/challenges", json={"tenant_id": tenant, "workload_id": workload}
    )
    assert response.status_code == 201
    return response.json()


def _receive_and_verify(client, claims=None, *, tenant=TENANT,
                        workload=WORKLOAD):
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


def _decide(client, evidence_id, created, evidence, policy_id, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
        "policy_id": policy_id,
    }
    body.update(overrides)
    return client.post(f"/v1/evidence/{evidence_id}/decisions", json=body)


def _iso(value):
    return value.astimezone(timezone.utc).isoformat()


def _audit_rowcounts(app):
    with app.state.engine.connect() as conn:
        return {
            table: conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            for table in (
                "decisions",
                "audit_events",
                "proof_lifecycle_events",
            )
        }


# --- happy path and response shape -----------------------------------------


def test_retire_returns_200_compact_ordered_json(client):
    policy = _policy(client)

    response = _retire(client, policy["policy_id"])

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    # Exactly one terminating newline; no surrounding whitespace.
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert response.content == response.content.strip() + b"\n"
    # Field order and field set are fixed, and every value is a string.
    data = response.json()
    assert list(data.keys()) == ["policy_id", "status", "retired_at"]
    assert data["policy_id"] == policy["policy_id"]
    assert data["status"] == "retired"
    retired_at = datetime.fromisoformat(data["retired_at"])
    assert retired_at.utcoffset() == timedelta(0)
    assert all(isinstance(value, str) for value in data.values())
    # The body describes only the retirement result: no extra metadata,
    # no rule/version/name, no floats.
    assert response.content == (
        b'{"policy_id":"'
        + policy["policy_id"].encode("ascii")
        + b'","status":"retired","retired_at":"'
        + data["retired_at"].encode("ascii")
        + b'"}\n'
    )


def test_retire_persists_status_and_time(app, client):
    policy = _policy(client)
    response = _retire(client, policy["policy_id"])
    retired_at = response.json()["retired_at"]

    with app.state.session_factory() as session:
        row = session.get(Policy, policy["policy_id"])
        assert row.status == "retired"
        assert row.retired_at is not None
        assert _iso(row.retired_at) == retired_at
        # The immutable identifying/rule fields are untouched.
        assert row.version == 1
        assert row.tenant_id == TENANT
        assert row.workload_id == WORKLOAD
        assert row.name == "release"


def test_retire_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    db_path = str(tmp_path / "restart.db")
    url = f"sqlite:///{db_path}"
    app1 = create_app(url)
    client1 = TestClient(app1)
    policy = _policy(client1)
    retired = _retire(client1, policy["policy_id"])
    assert retired.status_code == 200
    retired_at = retired.json()["retired_at"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        row = session.get(Policy, policy["policy_id"])
        assert row.status == "retired"
        assert _iso(row.retired_at) == retired_at
    # A repeat retire after restart is still a stable 409.
    client2 = TestClient(app2)
    assert _retire(client2, policy["policy_id"]).status_code == 409
    app2.state.engine.dispose()


# --- path validation --------------------------------------------------------


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
def test_retire_invalid_path_identifier_returns_422(client, policy):
    response = client.request(
        "POST",
        f"/v1/policies/{policy}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_retire_whitespace_padded_path_identifier_returns_422(client):
    created = _policy(client)
    response = client.request(
        "POST",
        f"/v1/policies/%09{created['policy_id']}%09/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422
    with client.app.state.session_factory() as session:
        assert session.get(Policy, created["policy_id"]).status == "active"


def test_retire_empty_path_segment_returns_422(client):
    response = client.post(
        "/v1/policies//retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_retire_invalid_path_does_not_touch_storage(app, client):
    created = _policy(client)
    statements: list[str] = []

    def spy(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(app.state.engine, "before_cursor_execute", spy)
    try:
        response = client.post(
            "/v1/policies/not-a-uuid/retire",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
    finally:
        event.remove(app.state.engine, "before_cursor_execute", spy)
    assert response.status_code == 422
    # No policy read or write occurred for a malformed path id.
    assert not any("policies" in s for s in statements)
    with app.state.session_factory() as session:
        assert session.get(Policy, created["policy_id"]).status == "active"


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
def test_retire_invalid_bodies_return_422(client, payload):
    created = _policy(client)
    if payload is None:
        response = client.post(
            f"/v1/policies/{created['policy_id']}/retire",
            headers={"content-type": "application/json"},
            content=b"",
        )
    elif isinstance(payload, (bytes, bytearray)):
        response = client.post(
            f"/v1/policies/{created['policy_id']}/retire",
            headers={"content-type": "application/json"},
            content=bytes(payload),
        )
    else:
        response = client.post(
            f"/v1/policies/{created['policy_id']}/retire", json=payload
        )
    assert response.status_code == 422
    with client.app.state.session_factory() as session:
        row = session.get(Policy, created["policy_id"])
        assert row.status == "active"
        assert row.retired_at is None


def test_retire_unknown_field_rejects_without_writing(client):
    created = _policy(client)
    response = _retire(client, created["policy_id"], unexpected="x")
    assert response.status_code == 422
    with client.app.state.session_factory() as session:
        assert session.get(Policy, created["policy_id"]).status == "active"


# --- 404 semantics ----------------------------------------------------------


def test_retire_unknown_policy_returns_404(client):
    response = _retire(client, "66666666-6666-6666-6666-666666666666")
    assert response.status_code == 404


def test_retire_wrong_tenant_or_workload_returns_404(client):
    created = _policy(client)
    assert _retire(client, created["policy_id"], tenant=OTHER_TENANT) \
        .status_code == 404
    assert _retire(client, created["policy_id"], workload=OTHER_WORKLOAD) \
        .status_code == 404
    with client.app.state.session_factory() as session:
        assert session.get(Policy, created["policy_id"]).status == "active"


def test_retire_404_is_indistinguishable_for_unknown_and_cross_scope(client):
    other = _policy(
        client,
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
    )

    unknown = _retire(client, "77777777-7777-7777-7777-777777777777")
    cross_scope = _retire(client, other["policy_id"])
    assert unknown.status_code == cross_scope.status_code == 404
    assert unknown.text == cross_scope.text


# --- repeat and concurrency -------------------------------------------------


def test_repeat_retire_returns_409_and_keeps_first_time(client):
    created = _policy(client)
    first = _retire(client, created["policy_id"])
    assert first.status_code == 200
    first_time = first.json()["retired_at"]

    second = _retire(client, created["policy_id"])
    assert second.status_code == 409
    with client.app.state.session_factory() as session:
        row = session.get(Policy, created["policy_id"])
        assert row.status == "retired"
        assert _iso(row.retired_at) == first_time


def test_repeat_retire_appends_no_other_state_record(app, client):
    created = _policy(client)
    assert _retire(client, created["policy_id"]).status_code == 200
    counts_before = _audit_rowcounts(app)

    assert _retire(client, created["policy_id"]).status_code == 409

    assert _audit_rowcounts(app) == counts_before


def test_concurrent_retires_settle_once(app):
    client = TestClient(app)
    created = _policy(client)
    policy_id = created["policy_id"]
    barrier = threading.Barrier(8)

    def call():
        barrier.wait()
        return TestClient(app).post(
            f"/v1/policies/{policy_id}/retire",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: call(), range(8)))

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(200) == 1
    assert statuses.count(409) == 7
    bodies = [r.json() for r in responses if r.status_code == 200]
    assert len(bodies) == 1
    with app.state.session_factory() as session:
        row = session.get(Policy, policy_id)
        assert row.status == "retired"
        assert row.retired_at is not None


# --- write/read failure -----------------------------------------------------


def test_retire_write_failure_rolls_back(app, client):
    created = _policy(client)

    def fail_update(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE policies"):
            raise RuntimeError("simulated storage failure")

    event.listen(app.state.engine, "before_cursor_execute", fail_update)
    try:
        response = _retire(client, created["policy_id"])
        assert response.status_code == 500
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_update)

    with app.state.session_factory() as session:
        row = session.get(Policy, created["policy_id"])
        assert row.status == "active"
        assert row.retired_at is None

    # After recovery the retire succeeds.
    recovered = _retire(client, created["policy_id"])
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "retired"


def test_retire_read_failure_returns_500_and_writes_nothing(app, client):
    created = _policy(client)

    def fail_read(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().startswith("SELECT policies.policy_id,"):
            raise RuntimeError("simulated storage failure")

    event.listen(app.state.engine, "before_cursor_execute", fail_read)
    try:
        response = _retire(client, created["policy_id"])
        assert response.status_code == 500
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_read)

    with app.state.session_factory() as session:
        row = session.get(Policy, created["policy_id"])
        assert row.status == "active"
        assert row.retired_at is None


def test_retire_failure_response_has_no_exception_text(app, client):
    created = _policy(client)

    def fail_update(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE policies"):
            raise RuntimeError("super-secret-failure-detail")

    event.listen(app.state.engine, "before_cursor_execute", fail_update)
    try:
        response = _retire(client, created["policy_id"])
        assert response.status_code == 500
        assert b"super-secret-failure-detail" not in response.content
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_update)


# --- new decisions blocked --------------------------------------------------


def test_retired_policy_rejects_new_decision_with_409(client, app):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    policy = _policy(client)
    assert _retire(client, policy["policy_id"]).status_code == 200

    counts_before = _audit_rowcounts(app)
    response = _decide(
        client, evidence_id, created, evidence, policy["policy_id"]
    )
    assert response.status_code == 409
    # No decision and no proof event were written, and business/audit
    # state is otherwise unchanged.
    assert _audit_rowcounts(app) == counts_before
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 0
        assert session.query(ProofLifecycleEvent).filter(
            ProofLifecycleEvent.event_type == "proof-decision"
        ).count() == 0
        # The evidence is untouched and still verified.
        assert session.get(Evidence, evidence_id).status == "verified"


def test_new_decision_against_retired_policy_is_blocked_even_when_denied(
    client, app
):
    # A retired version must never re-evaluate, regardless of claim match.
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "different"}
    )
    policy = _policy(client, {"claim": "measurement", "equals": "abc"})
    assert _retire(client, policy["policy_id"]).status_code == 200

    response = _decide(
        client, evidence_id, created, evidence, policy["policy_id"]
    )
    assert response.status_code == 409
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 0


def test_retired_decision_repeat_after_retirement_keeps_stored_result(
    client, app
):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    policy = _policy(client)
    first = _decide(client, evidence_id, created, evidence, policy["policy_id"])
    assert first.status_code == 200
    first_body = first.json()

    assert _retire(client, policy["policy_id"]).status_code == 200

    # Replaying the already-recorded decision returns the identical
    # stored result — retirement is not retroactive.
    repeated = _decide(
        client, evidence_id, created, evidence, policy["policy_id"]
    )
    assert repeated.status_code == 200
    assert repeated.json() == first_body

    with app.state.session_factory() as session:
        row = session.query(Decision).one()
        assert row.status == first_body["status"]
        assert row.policy_version == first_body["policy_version"] == 1
        assert _iso(row.decided_at) == first_body["decided_at"]


def test_retiring_one_version_does_not_block_other_versions(client, app):
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    v1 = _policy(client, name="release")
    v2 = _policy(
        client,
        {"claim": "measurement", "equals": "abc"},
        name="release",
    )
    assert v1["version"] == 1 and v2["version"] == 2

    assert _retire(client, v1["policy_id"]).status_code == 200

    # v1 is blocked...
    assert _decide(
        client, evidence_id, created, evidence, v1["policy_id"]
    ).status_code == 409
    # ...but v2 is an independent version and still decides.
    response = _decide(
        client, evidence_id, created, evidence, v2["policy_id"]
    )
    assert response.status_code == 200
    assert response.json()["policy_version"] == 2
    assert response.json()["status"] == "allowed"

    with app.state.session_factory() as session:
        row = session.get(Policy, v2["policy_id"])
        assert row.status == "active"
        assert row.retired_at is None
        row1 = session.get(Policy, v1["policy_id"])
        assert row1.status == "retired"


def test_retire_is_isolated_across_names_and_scopes(client):
    release = _policy(client, name="release")
    other_name = _policy(client, name="other-name")
    other_scope = _policy(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    )

    assert _retire(client, release["policy_id"]).status_code == 200

    with client.app.state.session_factory() as session:
        assert session.get(Policy, release["policy_id"]).status == "retired"
        assert session.get(Policy, other_name["policy_id"]).status == "active"
        assert session.get(Policy, other_scope["policy_id"]).status == "active"

    # The cross-scope policy retires independently under its own scope.
    assert (
        _retire(
            client,
            other_scope["policy_id"],
            tenant=OTHER_TENANT,
            workload=OTHER_WORKLOAD,
        ).status_code
        == 200
    )


def test_terminal_state_cannot_be_bypassed_by_creating_a_new_version(client):
    # Creating the next version allocates a new row; it never revives,
    # deletes or overwrites the retired version.
    v1 = _policy(client, name="release")
    assert _retire(client, v1["policy_id"]).status_code == 200

    v2 = _policy(client, name="release")
    assert v2["version"] == 2
    assert v2["policy_id"] != v1["policy_id"]

    assert _retire(client, v1["policy_id"]).status_code == 409
    with client.app.state.session_factory() as session:
        assert session.get(Policy, v1["policy_id"]).status == "retired"
        assert session.get(Policy, v2["policy_id"]).status == "active"


# --- release-grant compatibility --------------------------------------------


def _allowed_decision_and_grant(client, data_id="data-1"):
    """Drive evidence -> verified -> allowed decision -> pending grant."""
    created, evidence, evidence_id = _receive_and_verify(
        client, {"measurement": "abc"}
    )
    policy = _policy(client)
    decided = _decide(
        client, evidence_id, created, evidence, policy["policy_id"]
    )
    assert decided.status_code == 200
    decision = decided.json()
    minted = client.post(
        "/v1/release-grants",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "decision_id": decision["decision_id"],
            "data_id": data_id,
        },
    )
    assert minted.status_code == 201
    return policy, decision, minted.json(), evidence_id


def test_retirement_does_not_revoke_or_change_existing_pending_grant(
    client, app
):
    policy, decision, grant, _ = _allowed_decision_and_grant(client)
    assert grant["pending"] is True

    assert _retire(client, policy["policy_id"]).status_code == 200

    # The grant is still present, pending, with unchanged timestamps.
    listing = client.get(
        "/v1/release-grants",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert listing.status_code == 200
    entry = listing.json()["grants"][0]
    assert entry["grant_id"] == grant["grant_id"]
    assert entry["status"] == "pending"
    assert entry["decision_id"] == decision["decision_id"]
    assert entry["issued_at"] == grant["issued_at"]
    assert entry["expires_at"] == grant["expires_at"]
    assert entry["consumed_at"] is None
    assert entry["revoked_at"] is None


def test_pending_grant_still_consumable_after_policy_retirement(client):
    policy, decision, grant, _ = _allowed_decision_and_grant(client)
    assert _retire(client, policy["policy_id"]).status_code == 200

    consumed = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )
    assert consumed.status_code == 200
    assert consumed.json()["consumed"] is True
    assert consumed.json()["grant_id"] == grant["grant_id"]


def test_pending_grant_still_releasable_after_policy_retirement(client):
    data_id = "payload-data"
    envelope = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": data_id,
            "payload": "the protected payload",
        },
    )
    assert envelope.status_code == 201

    policy, decision, grant, _ = _allowed_decision_and_grant(client, data_id)
    assert _retire(client, policy["policy_id"]).status_code == 200

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
    assert released.json()["payload"] == "the protected payload"
    assert released.content.endswith(b"\n")
    assert not released.content.endswith(b"\n\n")


def test_pending_grant_still_revocable_after_policy_retirement(client):
    policy, decision, grant, _ = _allowed_decision_and_grant(client)
    assert _retire(client, policy["policy_id"]).status_code == 200

    revoked = client.post(
        f"/v1/release-grants/{grant['grant_id']}/revoke",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )
    assert revoked.status_code == 200
    assert revoked.json()["revoked"] is True


def test_existing_grant_capability_audit_unchanged_after_retirement(
    client, app
):
    policy, decision, grant, _ = _allowed_decision_and_grant(client)
    before = client.get(
        "/v1/compliance/audit-events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert before.status_code == 200
    before_events = before.json()["events"]

    assert _retire(client, policy["policy_id"]).status_code == 200

    after = client.get(
        "/v1/compliance/audit-events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    # Retirement appends no compliance audit event.
    assert after.json()["events"] == before_events

    # Consuming the grant still appends exactly the consumed event.
    consumed = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )
    assert consumed.status_code == 200
    final = client.get(
        "/v1/compliance/audit-events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    statuses = {e["status"] for e in final.json()["events"]}
    assert {"pending", "consumed"} <= statuses
    # The capability digest is still the only capability material present.
    assert grant["capability"] not in final.text


# --- no sensitive material in retirement artifacts --------------------------


def test_retire_never_echoes_rule_or_sensitive_material(client, app):
    secret_claim = "super-secret-claim-value-123456"
    rule = {"claim": "secret", "equals": secret_claim}
    policy = _policy(client, rule)

    response = _retire(client, policy["policy_id"])
    assert response.status_code == 200
    assert secret_claim.encode() not in response.content
    assert b"rule" not in response.content
    assert policy["policy_id"].encode() in response.content

    # The retirement record itself carries only status and time.
    with app.state.session_factory() as session:
        row = session.get(Policy, policy["policy_id"])
        columns = {
            c.name: getattr(row, c.name) for c in row.__table__.columns
        }
        assert columns["status"] == "retired"
        assert columns["retired_at"] is not None
        # The stored rule is the pre-existing rule tree (unchanged), but
        # retirement added no new column holding claim/evidence material.
        assert set(columns) == {
            "policy_id",
            "tenant_id",
            "workload_id",
            "name",
            "version",
            "rule_json",
            "created_at",
            "status",
            "retired_at",
            "commit_seq",
            "retired_seq",
        }


# --- legacy migration -------------------------------------------------------


def test_preexisting_database_backfills_active_status(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    from sqlalchemy import create_engine, text as sql_text

    db_path = tmp_path / "legacy.db"
    url = f"sqlite:///{db_path}"

    # Build the current schema, then drop the lifecycle columns so the
    # database looks exactly like one written before policy retirement
    # existed (the additive migration re-adds them nullable and backfills
    # pre-existing rows to active).
    bootstrap = create_app(url)
    client = TestClient(bootstrap)
    policy = _policy(client)
    bootstrap.state.engine.dispose()

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(sql_text("ALTER TABLE policies DROP COLUMN retired_at"))
        conn.execute(sql_text("ALTER TABLE policies DROP COLUMN status"))
        names = {
            r[0]
            for r in conn.execute(
                sql_text("SELECT name FROM pragma_table_info('policies')")
            ).fetchall()
        }
        assert "status" not in names and "retired_at" not in names
    engine.dispose()

    upgraded = create_app(url)
    with upgraded.state.session_factory() as session:
        row = session.get(Policy, policy["policy_id"])
        assert row.status == "active"
        assert row.retired_at is None
    upgraded.state.engine.dispose()


# --- other endpoints unaffected ---------------------------------------------


def test_policy_creation_and_other_entries_unchanged(client):
    # Creating policies still works and new versions start active even
    # while a sibling version is retired.
    v1 = _policy(client, name="release")
    assert _retire(client, v1["policy_id"]).status_code == 200
    v2 = _policy(client, name="release")
    assert v2["version"] == 2
    # Health and unrelated read endpoints remain available.
    assert client.get("/health").status_code == 200
    summary = client.get(
        "/v1/observability/release-summary",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert summary.status_code == 200
