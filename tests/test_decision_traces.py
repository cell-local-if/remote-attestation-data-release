"""Tests for GET /v1/decisions/{decision_id}/trace.

The trace completes the audit chain the decision listing starts: it reads
exactly one settled decision and joins its immutable audit conclusion to
the concrete policy version and the immutable rule snapshot used at
decision time. It is strictly read-only (no audit, event or decision is
created and no rule, decided_at or status is rewritten), validates every
request-shape rule to 422 before any state is read, hides unknown or
cross-scope decisions behind one indistinguishable 404, returns only
decision audit metadata plus the saved rule structure (never proof text,
nonces, claim values, capabilities, payloads or keys), survives restarts
and keeps returning the original snapshot even after the policy version
is retired.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    Decision,
    DecisionCommitCounter,
    Policy,
    ProofLifecycleEvent,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
SECRET = "unit-test-secret"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"
UPPER_UUID = "ABCDEF12-3456-7890-ABCD-EF1234567890"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    application = create_app(f"sqlite:///{tmp_path}/decision_traces.db")
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
    claims = {"m": "x"} if claims is None else claims
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


def _trace(client, decision_id, *, tenant=TENANT, workload=WORKLOAD, **extra):
    params = {"tenant_id": tenant, "workload_id": workload, **extra}
    return client.get(f"/v1/decisions/{decision_id}/trace", params=params)


# --- request validation ----------------------------------------------------


def test_trace_requires_scope_parameters(client):
    response = client.get(f"/v1/decisions/{ZERO_UUID}/trace")
    assert response.status_code == 422
    assert (
        client.get(
            f"/v1/decisions/{ZERO_UUID}/trace",
            params={"workload_id": WORKLOAD},
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/v1/decisions/{ZERO_UUID}/trace",
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
        {"status": "allowed"},
        {"policy_id": ZERO_UUID},
    ],
)
def test_trace_rejects_invalid_or_unknown_parameters(client, params):
    response = client.get(
        f"/v1/decisions/{ZERO_UUID}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD, **params},
    )
    assert response.status_code == 422, response.text


def test_trace_rejects_repeated_parameter(client):
    response = client.get(
        f"/v1/decisions/{ZERO_UUID}/trace",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("tenant_id", OTHER_TENANT),
        ],
    )
    assert response.status_code == 422
    response = client.get(
        f"/v1/decisions/{ZERO_UUID}/trace",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("workload_id", OTHER_WORKLOAD),
        ],
    )
    assert response.status_code == 422


@pytest.mark.parametrize("decision_id", [
    " ",
    "not-a-uuid",
    ZERO_UUID[:-1] + "Z",
    "  " + ZERO_UUID,
    ZERO_UUID + " ",
    UPPER_UUID,
    "1234567890",
    "g0000000-0000-0000-0000-000000000000",
])
def test_trace_rejects_non_canonical_path_identifier(client, decision_id):
    response = _trace(client, decision_id)
    assert response.status_code == 422, response.text


def test_trace_empty_path_segment_is_422(client):
    response = client.get(
        "/v1/decisions//trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"{}", b"  ", b"\t", b"not json", b"\x00"])
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
    assert no_body.status_code == 200, no_body.text
    assert empty_body.status_code == 200, empty_body.text
    assert no_body.content == empty_body.content


def test_invalid_requests_read_or_write_no_state(app, client):
    _one_decision(client)
    _trace(client, "not-a-uuid")
    _trace(client, ZERO_UUID, tenant=" ")
    client.get(
        f"/v1/decisions/{ZERO_UUID}/trace",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("workload_id", OTHER_WORKLOAD),
        ],
    )
    client.request(
        "GET",
        "/v1/decisions//trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"x",
    )
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 1
        assert session.query(Policy).count() == 1
        assert session.query(DecisionCommitCounter).count() == 1


# --- 404 semantics ---------------------------------------------------------


def test_trace_unknown_decision_is_404(client):
    assert _trace(client, ZERO_UUID).status_code == 404


def test_trace_cross_scope_decision_is_404(client):
    _, _, _, _, decided = _one_decision(client)
    decision_id = decided["decision_id"]
    assert _trace(client, decision_id, tenant=OTHER_TENANT).status_code == 404
    assert _trace(client, decision_id, workload=OTHER_WORKLOAD).status_code == 404
    assert (
        _trace(
            client, decision_id, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
        ).status_code
        == 404
    )


def test_trace_unknown_and_cross_scope_are_indistinguishable(client):
    _, _, _, _, decided = _one_decision(client)
    unknown = _trace(client, ZERO_UUID)
    cross = _trace(client, decided["decision_id"], tenant=OTHER_TENANT)
    assert unknown.status_code == cross.status_code == 404
    assert unknown.content == cross.content


# --- success shape ---------------------------------------------------------


def test_trace_success_shape_and_field_order(client):
    _, _, evidence_id, policy, decided = _one_decision(client)
    response = _trace(client, decided["decision_id"])
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    raw = response.content
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw

    parsed = json.loads(raw)
    assert list(parsed) == ["decision", "policy_version"]

    result = parsed["decision"]
    assert list(result) == [
        "decision_id",
        "evidence_id",
        "policy_version",
        "status",
        "decided_at",
    ]
    assert result == {
        "decision_id": decided["decision_id"],
        "evidence_id": evidence_id,
        "policy_version": decided["policy_version"],
        "status": decided["status"],
        "decided_at": decided["decided_at"],
    }
    for key in ("decision_id", "evidence_id", "status", "decided_at"):
        assert isinstance(result[key], str) and result[key]
    assert isinstance(result["policy_version"], int)
    assert not isinstance(result["policy_version"], bool)
    parsed_at = datetime.fromisoformat(result["decided_at"])
    assert parsed_at.utcoffset() == timedelta(0)

    version = parsed["policy_version"]
    assert list(version) == ["policy_id", "name", "version", "rule"]
    assert version["policy_id"] == policy["policy_id"]
    assert version["name"] == policy["name"]
    assert version["version"] == policy["version"]
    assert isinstance(version["version"], int)
    assert not isinstance(version["version"], bool)
    assert version["rule"] == policy["rule"]


def test_trace_denied_decision_keeps_its_conclusion(client):
    _, _, _, policy, decided = _one_decision(
        client,
        claims={"m": "different"},
        rule={"claim": "m", "equals": "x"},
        name="deny",
    )
    assert decided["status"] == "denied"
    response = _trace(client, decided["decision_id"])
    assert response.status_code == 200
    parsed = response.json()
    assert parsed["decision"]["status"] == "denied"
    assert parsed["policy_version"]["policy_id"] == policy["policy_id"]


def test_trace_targets_exact_version_when_newer_versions_exist(client):
    created, evidence, evidence_id = _submit_verified(client)
    v1 = _policy(client, {"claim": "m", "equals": "x"}, name="r")
    decided = _decide(client, created, evidence, evidence_id, v1["policy_id"])
    assert decided.status_code == 200
    v2 = _policy(client, {"claim": "m", "equals": "nope"}, name="r")
    assert v2["version"] == 2

    parsed = _trace(client, decided.json()["decision_id"]).json()
    snapshot = parsed["policy_version"]
    assert snapshot["policy_id"] == v1["policy_id"]
    assert snapshot["version"] == 1
    assert snapshot["rule"] == {"claim": "m", "equals": "x"}


# --- rule snapshot numbers -------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        {"claim": "x", "equals": -0.0},
        {"claim": "x", "equals": 0.0},
        {"path": ["t", "v"], "equals": 1.25},
        {"claim": "x", "equals": -1.5},
        {"claim": "x", "equals": 3},
        {"claim": "x", "equals": True},
        {"claim": "x", "equals": None},
        {"claim": "x", "equals": "str"},
        {"all": [
            {"claim": "a", "equals": 0.1},
            {"path": ["b", "c"], "equals": -2.25},
            {"not": {"any": [{"claim": "d", "equals": -7}]}},
        ]},
    ],
)
def test_trace_rule_numbers_round_trip_verbatim(client, rule):
    # Decide with a claim that satisfies the rule when it is a plain claim
    # leaf so the decision is allowed; the trace is identical regardless.
    claims = {"x": rule.get("equals")} if set(rule) == {"claim", "equals"} else None
    _, _, _, policy, decided = _one_decision(
        client, claims=claims, rule=rule, name=json.dumps(rule, sort_keys=True)
    )
    response = _trace(client, decided["decision_id"])
    assert response.status_code == 200, response.text
    snapshot = response.json()["policy_version"]["rule"]
    assert snapshot == rule

    original = rule.get("equals")
    if (
        isinstance(original, float)
        and original == 0.0
        and math.copysign(1.0, original) < 0
    ):
        # The sign of -0.0 survives both on the wire and after parsing.
        assert b"-0.0" in response.content
        assert math.copysign(1.0, snapshot["equals"]) < 0
    if "all" in rule:
        assert b"0.1" in response.content
        assert b"-2.25" in response.content
        assert b"-7" in response.content


def test_trace_metadata_has_no_floats_or_non_finite_values(client):
    _, _, _, _, decided = _one_decision(
        client, rule={"claim": "m", "equals": 1.25}, name="f"
    )
    parsed = json.loads(_trace(client, decided["decision_id"]).content)

    def _check(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            return
        if isinstance(value, float):  # pragma: no cover - structural guard
            raise AssertionError("metadata must not introduce floats")
        assert value is None or isinstance(value, str)

    decision = parsed["decision"]
    for value in decision.values():
        _check(value)
    version = parsed["policy_version"]
    for key, value in version.items():
        if key == "rule":
            # Numbers inside the rule are the one legal float source; the
            # other rule metadata obeys the structural guard.
            def _walk(node):
                if isinstance(node, dict):
                    for child in node.values():
                        _walk(child)
                elif isinstance(node, list):
                    for child in node:
                        _walk(child)
                else:
                    assert not (
                        isinstance(node, float) and not math.isfinite(node)
                    )
            _walk(value)
        else:
            _check(value)


# --- no protected material -------------------------------------------------


def test_trace_never_returns_protected_material(client):
    created, evidence, _, _, decided = _one_decision(client)
    text = _trace(client, decided["decision_id"]).text
    assert evidence not in text
    assert created["nonce"] not in text
    assert '"claims"' not in text
    assert '"mac"' not in text
    assert '"capability"' not in text
    assert '"payload"' not in text
    assert '"nonce"' not in text
    # The rule snapshot is present, but no policy lifecycle metadata.
    assert '"retired_at"' not in text
    assert '"status":"retired"' not in text
    # The decision conclusion status is present, but the evidence's
    # verification/audit fields are not.
    assert '"verified_at"' not in text
    assert '"evidence_sha256"' not in text


# --- retired policy keeps the snapshot -------------------------------------


def test_trace_returns_original_snapshot_after_policy_retirement(client):
    created, evidence, evidence_id = _submit_verified(client)
    rule = {"all": [{"claim": "m", "equals": "x"}, {"path": ["m"], "equals": "x"}]}
    v1 = _policy(client, rule, name="r")
    decided = _decide(client, created, evidence, evidence_id, v1["policy_id"])
    assert decided.status_code == 200
    decision_id = decided.json()["decision_id"]

    before = _trace(client, decision_id)
    assert before.status_code == 200

    retired = client.post(
        f"/v1/policies/{v1['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200, retired.text

    after = _trace(client, decision_id)
    assert after.status_code == 200
    # The trace is byte-identical before and after retirement: settled
    # decisions keep the original snapshot, status and decision time.
    assert after.content == before.content
    parsed = after.json()
    assert parsed["policy_version"]["rule"] == rule
    assert parsed["policy_version"]["version"] == 1
    assert parsed["decision"]["status"] == "allowed"


def test_trace_is_stable_across_repeated_reads(client):
    _, _, _, _, decided = _one_decision(client)
    first = _trace(client, decided["decision_id"])
    second = _trace(client, decided["decision_id"])
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


# --- read-only behaviour ---------------------------------------------------


def test_trace_writes_no_audit_event_or_decision(app, client):
    _, _, _, _, decided = _one_decision(client)
    with app.state.session_factory() as session:
        decisions_before = session.query(Decision).count()
        audits_before = session.query(AuditEvent).count()
        proof_before = session.query(ProofLifecycleEvent).count()
        counter_before = session.get(
            DecisionCommitCounter, (TENANT, WORKLOAD)
        ).last_seq
        stored = session.get(Decision, decided["decision_id"])
        status_before = stored.status
        decided_at_before = stored.decided_at
        rule_before = session.get(Policy, stored.policy_id).rule_json

    for _ in range(3):
        assert _trace(client, decided["decision_id"]).status_code == 200

    with app.state.session_factory() as session:
        assert session.query(Decision).count() == decisions_before
        assert session.query(AuditEvent).count() == audits_before
        assert session.query(ProofLifecycleEvent).count() == proof_before
        assert (
            session.get(DecisionCommitCounter, (TENANT, WORKLOAD)).last_seq
            == counter_before
        )
        stored = session.get(Decision, decided["decision_id"])
        assert stored.status == status_before
        assert stored.decided_at == decided_at_before
        assert session.get(Policy, stored.policy_id).rule_json == rule_before


# --- server failure --------------------------------------------------------


def test_trace_storage_failure_returns_500_with_no_half_result(app, client):
    from sqlalchemy import text

    _, _, _, _, decided = _one_decision(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE decisions"))
    assert _trace(client, decided["decision_id"]).status_code == 500


def test_trace_missing_policy_snapshot_returns_500(app, client):
    from sqlalchemy import text

    _, _, _, _, decided = _one_decision(client)
    # Policy versions are never deleted by the service; simulate a damaged
    # store so the join cannot complete. The half result is discarded.
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policies"))
    assert _trace(client, decided["decision_id"]).status_code == 500


# --- persistence -----------------------------------------------------------


def test_trace_consistent_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart_trace.db"
    first = create_app(url)
    client1 = TestClient(first)
    _, _, evidence_id, policy, decided = _one_decision(client1)
    first_body = client1.get(
        f"/v1/decisions/{decided['decision_id']}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).content
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    response = client2.get(
        f"/v1/decisions/{decided['decision_id']}/trace",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    assert response.content == first_body
    parsed = response.json()
    assert parsed["decision"]["decision_id"] == decided["decision_id"]
    assert parsed["decision"]["evidence_id"] == evidence_id
    assert parsed["policy_version"]["policy_id"] == policy["policy_id"]
    assert parsed["policy_version"]["version"] == policy["version"]
    second.state.engine.dispose()


def test_trace_scope_isolation_is_strict(client):
    _, _, _, _, decided = _one_decision(client)
    # A different scope sees neither the decision nor the snapshot.
    assert (
        _trace(
            client,
            decided["decision_id"],
            tenant=OTHER_TENANT,
            workload=OTHER_WORKLOAD,
        ).status_code
        == 404
    )
    # The owning scope still reads it.
    assert _trace(client, decided["decision_id"]).status_code == 200
