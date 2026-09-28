"""Tests for GET /v1/decisions/{decision_id}/trace.

The trace is the read-only link from one recorded compliance decision to
the concrete policy version it was taken against and the rule snapshot in
force at decision time. Validation is strict and precedes every state
read; an unknown or cross-scope decision is one indistinguishable 404; a
successful response carries only decision audit metadata and the saved
rule structure — never evidence, nonce, claims, capabilities, payloads or
keys — and a retired or superseded version keeps serving the original
immutable snapshot.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, text

from proof_release.app import create_app
from proof_release.db import AuditEvent, Decision, Policy, ProofLifecycleEvent

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
SECRET = "unit-test-secret"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    application = create_app(f"sqlite:///{tmp_path}/decision_trace.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


# --- lifecycle builders ----------------------------------------------------


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


def _submit_verified(client, *, claims=None, tenant=TENANT, workload=WORKLOAD):
    """Create a challenge, receive and verify evidence; retain its nonce."""
    claims = {"m": "x", "n": {"v": 1}} if claims is None else claims
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
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
    assert submitted.status_code == 201, submitted.text
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
    assert verified.status_code == 200, verified.text
    return created, evidence, evidence_id


def _policy(client, rule=None, *, name="r", tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/policies",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "name": name,
            "rule": rule if rule is not None else {"claim": "m", "equals": "x"},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _decide(client, created, evidence, evidence_id, policy_id, *,
            tenant=TENANT, workload=WORKLOAD):
    return client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy_id,
        },
    )


def _one_decision(client, *, claims=None, rule=None, name="r"):
    created, evidence, evidence_id = _submit_verified(client, claims=claims)
    policy = _policy(client, rule, name=name)
    decided = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    assert decided.status_code == 200, decided.text
    return created, evidence, evidence_id, policy, decided.json()


def _trace(client, did, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        f"/v1/decisions/{did}/trace",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


# --- request validation ----------------------------------------------------


def test_trace_requires_scope_parameters(client):
    _, _, _, _, decided = _one_decision(client)
    decision_id = decided["decision_id"]
    assert client.get(f"/v1/decisions/{decision_id}/trace").status_code == 422
    assert (
        client.get(
            f"/v1/decisions/{decision_id}/trace",
            params={"workload_id": WORKLOAD},
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/v1/decisions/{decision_id}/trace",
            params={"tenant_id": TENANT},
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": "\t"},
        {"workload_id": ""},
        {"workload_id": "  "},
        {"bogus": "value"},
        {"cursor": "abc"},
        {"decision_id": ZERO_UUID},
        {"evidence_id": ZERO_UUID},
        {"policy_id": ZERO_UUID},
        {"status": "allowed"},
        {"limit": "1"},
    ],
)
def test_trace_rejects_invalid_or_unknown_parameters(client, params):
    _, _, _, _, decided = _one_decision(client)
    response = _trace(client, decided["decision_id"], **params)
    assert response.status_code == 422, response.text


def test_trace_rejects_repeated_parameter(client):
    _, _, _, _, decided = _one_decision(client)
    response = client.get(
        f"/v1/decisions/{decided['decision_id']}/trace",
        params=[
            ("tenant_id", TENANT),
            ("tenant_id", OTHER_TENANT),
            ("workload_id", WORKLOAD),
        ],
    )
    assert response.status_code == 422
    response = client.get(
        f"/v1/decisions/{decided['decision_id']}/trace",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("workload_id", OTHER_WORKLOAD),
        ],
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "decision_id",
    [
        "",
        "not-a-uuid",
        "abc123",
        "deadbeef-dead-dead-dead",  # truncated
        "ABCDEFAB-ABCD-4ABCD-8ABCD-ABCDEFABCDEF",  # uppercase
        "%09",
        "%20",
    ],
)
def test_trace_rejects_invalid_path_identifier(client, decision_id):
    response = client.get(
        f"/v1/decisions/{decision_id}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_trace_empty_path_segment_is_422(client):
    response = client.get(
        "/v1/decisions//trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_trace_whitespace_padded_path_identifier_is_422(client):
    _, _, _, _, decided = _one_decision(client)
    decision_id = decided["decision_id"]
    for padded in (f"%20{decision_id}", f"%09{decision_id}",
                   f"{decision_id}%20", f"{decision_id}%0A"):
        response = client.get(
            f"/v1/decisions/{padded}/trace",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
        assert response.status_code == 422


def test_trace_validation_runs_before_any_state_read(app, client):
    _, _, _, _, decided = _one_decision(client)
    statements: list[str] = []

    def spy(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(app.state.engine, "before_cursor_execute", spy)
    try:
        response = client.get(
            "/v1/decisions/not-a-uuid/trace",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
        assert response.status_code == 422
        response = client.get(
            f"/v1/decisions/{decided['decision_id']}/trace",
            params=[("tenant_id", TENANT), ("tenant_id", OTHER_TENANT),
                    ("workload_id", WORKLOAD)],
        )
        assert response.status_code == 422
    finally:
        event.remove(app.state.engine, "before_cursor_execute", spy)
    # No decision or policy read for a rejected request.
    assert not any("decisions" in s or "policies" in s for s in statements)


@pytest.mark.parametrize("body", [b"{}", b"  ", b"not json", b"\x00"])
def test_trace_non_empty_body_is_422(client, body):
    _, _, _, _, decided = _one_decision(client)
    response = client.request(
        "GET",
        f"/v1/decisions/{decided['decision_id']}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=body,
    )
    assert response.status_code == 422


def test_trace_missing_and_zero_length_body_are_accepted(client):
    _, _, _, _, decided = _one_decision(client)
    no_body = _trace(client, decided["decision_id"])
    empty_body = client.request(
        "GET",
        f"/v1/decisions/{decided['decision_id']}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"",
    )
    assert no_body.status_code == 200
    assert empty_body.status_code == 200
    assert no_body.content == empty_body.content


def test_trace_invalid_requests_write_no_state(app, client):
    _, _, _, _, decided = _one_decision(client)
    with app.state.session_factory() as session:
        decisions_before = session.query(Decision).count()
        policies_before = session.query(Policy).count()
        events_before = session.query(ProofLifecycleEvent).count()
        audits_before = session.query(AuditEvent).count()
    _trace(client, "not-a-uuid")
    _trace(client, decided["decision_id"], bogus="x")
    client.get(
        f"/v1/decisions/{decided['decision_id']}/trace",
        params=[("tenant_id", TENANT), ("tenant_id", OTHER_TENANT),
                ("workload_id", WORKLOAD)],
    )
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == decisions_before
        assert session.query(Policy).count() == policies_before
        assert session.query(ProofLifecycleEvent).count() == events_before
        assert session.query(AuditEvent).count() == audits_before


# --- 404 semantics ---------------------------------------------------------


def test_trace_unknown_decision_is_404(client):
    response = _trace(client, ZERO_UUID)
    assert response.status_code == 404


def test_trace_cross_scope_decision_is_404(client):
    _, _, _, _, decided = _one_decision(client)
    assert (
        _trace(client, decided["decision_id"], tenant=OTHER_TENANT).status_code
        == 404
    )
    assert (
        _trace(client, decided["decision_id"], workload=OTHER_WORKLOAD).status_code
        == 404
    )
    assert (
        _trace(
            client,
            decided["decision_id"],
            tenant=OTHER_TENANT,
            workload=OTHER_WORKLOAD,
        ).status_code
        == 404
    )


def test_trace_unknown_and_cross_scope_are_indistinguishable(client):
    _, _, _, _, decided = _one_decision(client)
    unknown = _trace(client, ZERO_UUID)
    cross = _trace(client, decided["decision_id"], tenant=OTHER_TENANT)
    assert unknown.status_code == cross.status_code == 404
    assert unknown.content == cross.content


# --- successful trace ------------------------------------------------------


def test_trace_success_exact_shape_and_field_order(client):
    _, _, _, _, decided = _one_decision(client)
    response = _trace(client, decided["decision_id"])
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    raw = response.content
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw

    payload = json.loads(raw)
    assert list(payload) == ["decision", "policy_version"]
    assert list(payload["decision"]) == [
        "decision_id",
        "evidence_id",
        "policy_version",
        "status",
        "decided_at",
    ]
    assert list(payload["policy_version"]) == [
        "policy_id",
        "name",
        "version",
        "rule",
    ]


def test_trace_decision_section_is_audit_metadata(client):
    _, _, evidence_id, policy, decided = _one_decision(client)
    payload = _trace(client, decided["decision_id"]).json()
    decision = payload["decision"]
    assert decision["decision_id"] == decided["decision_id"]
    assert decision["evidence_id"] == evidence_id == decided["evidence_id"]
    assert decision["policy_version"] == 1 == decided["policy_version"]
    assert decision["status"] == "allowed" == decided["status"]
    assert decision["decided_at"] == decided["decided_at"]
    parsed = datetime.fromisoformat(decision["decided_at"])
    assert parsed.utcoffset() == timedelta(0)

    for key in ("decision_id", "evidence_id", "status", "decided_at"):
        assert isinstance(decision[key], str) and decision[key]
    assert isinstance(decision["policy_version"], int) and not isinstance(
        decision["policy_version"], bool
    )
    # The decision section carries exactly the five enumerated audit
    # fields — the version identifier belongs to policy_version.
    assert set(decision) == {
        "decision_id",
        "evidence_id",
        "policy_version",
        "status",
        "decided_at",
    }
    assert payload["policy_version"]["policy_id"] == policy["policy_id"]


def test_trace_policy_version_section_is_the_saved_snapshot(client):
    rule = {
        "all": [
            {"claim": "m", "equals": "x"},
            {"path": ["n", "v"], "equals": 1},
            {"not": {"claim": "missing", "equals": None}},
        ]
    }
    _, _, _, policy, decided = _one_decision(client, rule=rule, name="snap")
    payload = _trace(client, decided["decision_id"]).json()
    version = payload["policy_version"]
    assert version["policy_id"] == policy["policy_id"]
    assert version["name"] == "snap"
    assert version["version"] == 1
    assert version["rule"] == rule


def test_trace_denied_decision_keeps_its_conclusion(client):
    _, _, _, policy, decided = _one_decision(
        client,
        claims={"m": "other"},
        rule={"claim": "m", "equals": "x"},
        name="deny",
    )
    payload = _trace(client, decided["decision_id"]).json()
    assert payload["decision"]["status"] == "denied"
    assert payload["policy_version"]["policy_id"] == policy["policy_id"]


def test_trace_rule_numbers_round_trip_verbatim_including_negative_zero(client):
    rule = {
        "all": [
            {"claim": "a", "equals": -0.0},
            {"path": ["b"], "equals": 1.250},
            {"claim": "c", "equals": -1.5},
            {"claim": "d", "equals": 3},
        ]
    }
    _, _, _, _, decided = _one_decision(client, rule=rule, name="numbers")
    response = _trace(client, decided["decision_id"])
    payload = response.json()
    assert payload["policy_version"]["rule"] == rule

    raw = response.content
    # Numbers are re-serialized from the saved canonical rule text: the
    # negative zero keeps its sign and decimals survive, while no other
    # metadata field produces a float.
    assert b"-0.0" in raw
    assert b"1.25" in raw
    assert b"-1.5" in raw

    parsed = json.loads(raw, parse_float=lambda token: ("float", token))
    floats: list[tuple] = []

    def _walk(value):
        if isinstance(value, list):
            for item in value:
                _walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                _walk(item)
        elif isinstance(value, tuple) and value and value[0] == "float":
            floats.append(value)

    _walk(parsed)
    assert {token for _, token in floats} <= {"-0.0", "1.25", "-1.5"}

    signed_zero = payload["policy_version"]["rule"]["all"][0]["equals"]
    assert isinstance(signed_zero, float)
    assert math.copysign(1.0, signed_zero) < 0
    integer_value = payload["policy_version"]["rule"]["all"][3]["equals"]
    assert integer_value == 3 and not isinstance(integer_value, bool)
    assert payload["decision"]["policy_version"] == 1
    assert payload["policy_version"]["version"] == 1


def test_trace_never_contains_protected_material(client):
    created, evidence, _, _, decided = _one_decision(
        client, claims={"m": "x", "secret-claim": "super-secret-value"}
    )
    response = _trace(client, decided["decision_id"])
    text = response.text
    assert evidence not in text
    assert created["nonce"] not in text
    assert "super-secret-value" not in text
    assert '"claims"' not in text
    assert '"nonce"' not in text
    assert '"mac"' not in text
    assert '"capability"' not in text
    assert '"payload"' not in text
    assert '"evidence"' not in text  # evidence bytes are not a field value
    # The associated proof identifier is present, but never its content.
    assert decided["evidence_id"] in text
    # No lifecycle bookkeeping leaks into the trace.
    assert "commit_seq" not in text
    assert "retired" not in text


# --- immutability across retirement / newer versions -----------------------


def test_trace_keeps_original_snapshot_after_version_retirement(client):
    created, evidence, evidence_id, v1, first = _one_decision(
        client, rule={"claim": "m", "equals": "x"}, name="r"
    )
    # A newer version of the same name exists independently.
    v2 = _policy(client, {"claim": "m", "equals": "new"}, name="r")
    assert v2["version"] == 2

    retired = client.post(
        f"/v1/policies/{v1['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200, retired.text

    # A *new* decision against the retired version is refused...
    created_fresh, evidence_fresh, eid_fresh = _submit_verified(client)
    refused = _decide(
        client, created_fresh, evidence_fresh, eid_fresh, v1["policy_id"]
    )
    assert refused.status_code == 409

    # ...while replaying the original evidence/version pair stays the
    # idempotent stored result (retirement is never retroactive).
    replay = _decide(client, created, evidence, evidence_id, v1["policy_id"])
    assert replay.status_code == 200
    assert replay.json()["decision_id"] == first["decision_id"]

    # ...but the settled decision's trace still resolves to exactly the
    # original version and its rule, with no retirement metadata.
    payload = _trace(client, first["decision_id"]).json()
    assert payload["policy_version"]["policy_id"] == v1["policy_id"]
    assert payload["decision"]["policy_version"] == 1
    assert payload["decision"]["status"] == "allowed"
    assert payload["policy_version"] == {
        "policy_id": v1["policy_id"],
        "name": "r",
        "version": 1,
        "rule": {"claim": "m", "equals": "x"},
    }

    # A fresh evidence decided against v2 traces to v2 independently.
    created2, evidence2, eid2 = _submit_verified(client, claims={"m": "new"})
    second = _decide(client, created2, evidence2, eid2, v2["policy_id"])
    assert second.status_code == 200
    second_payload = _trace(client, second.json()["decision_id"]).json()
    assert second_payload["policy_version"]["policy_id"] == v2["policy_id"]
    assert second_payload["policy_version"]["version"] == 2
    assert second_payload["policy_version"]["rule"] == {
        "claim": "m",
        "equals": "new",
    }


def test_trace_is_stable_across_repeated_reads(client):
    _, _, _, _, decided = _one_decision(client)
    first = _trace(client, decided["decision_id"])
    second = _trace(client, decided["decision_id"])
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


# --- read-only behaviour ---------------------------------------------------


def test_trace_writes_no_state(app, client):
    _, _, _, _, decided = _one_decision(client)
    with app.state.session_factory() as session:
        decisions_before = session.query(Decision).count()
        policies_before = session.query(Policy).count()
        proof_before = session.query(ProofLifecycleEvent).count()
        audit_before = session.query(AuditEvent).count()
    for _ in range(3):
        assert _trace(client, decided["decision_id"]).status_code == 200
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == decisions_before
        assert session.query(Policy).count() == policies_before
        assert session.query(ProofLifecycleEvent).count() == proof_before
        assert session.query(AuditEvent).count() == audit_before
        # The trace must not rewrite the decision time, status or rule.
        row = session.get(Decision, decided["decision_id"])
        assert row.status == decided["status"]
        assert row.decided_at.isoformat()  # unchanged, still parseable


# --- server failure --------------------------------------------------------


def test_trace_storage_failure_returns_500(app, client):
    _, _, _, _, decided = _one_decision(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE decisions"))
    response = _trace(client, decided["decision_id"])
    assert response.status_code == 500


def test_trace_missing_policy_version_returns_500_not_404(app, client):
    _, _, _, policy, decided = _one_decision(client)
    # Removing the referenced version is storage corruption: the decision
    # exists and is in scope, so this must surface as a 500 rather than a
    # scope-style 404 or a half result.
    with app.state.engine.begin() as conn:
        conn.execute(
            text("DELETE FROM policies WHERE policy_id = :pid"),
            {"pid": policy["policy_id"]},
        )
    response = _trace(client, decided["decision_id"])
    assert response.status_code == 500


# --- persistence -----------------------------------------------------------


def test_trace_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/trace_restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    _, _, _, policy, decided = _one_decision(
        client1, rule={"all": [{"claim": "m", "equals": "x"}]}, name="persist"
    )
    expected = client1.get(
        f"/v1/decisions/{decided['decision_id']}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert expected.status_code == 200
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    response = client2.get(
        f"/v1/decisions/{decided['decision_id']}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    assert response.content == expected.content
    payload = response.json()
    assert payload["decision"]["decision_id"] == decided["decision_id"]
    assert payload["policy_version"]["policy_id"] == policy["policy_id"]
    assert payload["policy_version"]["rule"] == {
        "all": [{"claim": "m", "equals": "x"}]
    }
    second.state.engine.dispose()


# --- scope isolation -------------------------------------------------------


def test_trace_is_strictly_scope_isolated(client):
    _, _, _, _, decided = _one_decision(client)
    # The same decision id must never resolve from another tenant/workload
    # even when the other scope has its own unrelated decision.
    created2, evidence2, eid2 = _submit_verified(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    )
    policy2 = _policy(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    )
    other = _decide(
        client,
        created2,
        evidence2,
        eid2,
        policy2["policy_id"],
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
    )
    assert other.status_code == 200

    assert (
        _trace(client, decided["decision_id"], tenant=OTHER_TENANT).status_code
        == 404
    )
    assert (
        _trace(
            client,
            other.json()["decision_id"],
            tenant=TENANT,
            workload=WORKLOAD,
        ).status_code
        == 404
    )
    own = _trace(
        client,
        other.json()["decision_id"],
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
    )
    assert own.status_code == 200
    assert own.json()["decision"]["decision_id"] == other.json()["decision_id"]
