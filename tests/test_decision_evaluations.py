"""Tests for GET /v1/decisions/{decision_id}/evaluation.

The evaluation endpoint returns the complete policy-rule explanation
persisted in the same transaction as POST
/v1/evidence/{evidence_id}/decisions: a depth-first node list covering
leaf/all/any/not, each node carrying exactly node_index, rule_path,
node_type and a boolean outcome, the root outcome equal to the decision
status. The explanation stores and returns no claim names, locators,
expected or actual values, evidence, nonces, capabilities, payloads or
keys.

Like the trace endpoint it validates every request-shape rule to 422
before any state is read, hides unknown or cross-scope decisions behind
one indistinguishable 404, and turns any storage or integrity failure
into a 500 with a fixed detail and no half result. A decision written by
an older deployment (no explanation) is a defined 409. The explanation is
historical: retirement, newer policy versions and key rotation never
rewrite it, and retries/concurrent creates share one decision and one
explanation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    Decision,
    DecisionEvaluationNode,
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

SECRET_CLAIM = "super-secret-claim-value-987654"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    application = create_app(f"sqlite:///{tmp_path}/decision_evaluations.db")
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


def _evaluation(client, decision_id, *, tenant=TENANT, workload=WORKLOAD, **extra):
    params = {"tenant_id": tenant, "workload_id": workload, **extra}
    return client.get(
        f"/v1/decisions/{decision_id}/evaluation", params=params
    )


# --- request validation ----------------------------------------------------


def test_evaluation_requires_scope_parameters(client):
    response = client.get(f"/v1/decisions/{ZERO_UUID}/evaluation")
    assert response.status_code == 422
    assert (
        client.get(
            f"/v1/decisions/{ZERO_UUID}/evaluation",
            params={"workload_id": WORKLOAD},
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/v1/decisions/{ZERO_UUID}/evaluation",
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
        {"evaluation_version": "2"},
    ],
)
def test_evaluation_rejects_invalid_or_unknown_parameters(client, params):
    response = client.get(
        f"/v1/decisions/{ZERO_UUID}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD, **params},
    )
    assert response.status_code == 422, response.text


def test_evaluation_rejects_repeated_parameter(client):
    response = client.get(
        f"/v1/decisions/{ZERO_UUID}/evaluation",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("tenant_id", OTHER_TENANT),
        ],
    )
    assert response.status_code == 422
    response = client.get(
        f"/v1/decisions/{ZERO_UUID}/evaluation",
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
def test_evaluation_rejects_non_canonical_path_identifier(client, decision_id):
    response = _evaluation(client, decision_id)
    assert response.status_code == 422, response.text


def test_evaluation_empty_path_segment_is_422(client):
    response = client.get(
        "/v1/decisions//evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"{}", b"  ", b"\t", b"not json", b"\x00"])
def test_evaluation_non_empty_body_is_422(client, body):
    _, _, _, _, decided = _one_decision(client)
    response = client.request(
        "GET",
        f"/v1/decisions/{decided['decision_id']}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=body,
    )
    assert response.status_code == 422


def test_evaluation_missing_and_zero_length_body_are_accepted(client):
    _, _, _, _, decided = _one_decision(client)
    no_body = _evaluation(client, decided["decision_id"])
    empty_body = client.request(
        "GET",
        f"/v1/decisions/{decided['decision_id']}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"",
    )
    assert no_body.status_code == 200, no_body.text
    assert empty_body.status_code == 200, empty_body.text
    assert no_body.content == empty_body.content


def test_invalid_requests_read_or_write_no_state(app, client):
    _, _, _, _, decided = _one_decision(client)
    _evaluation(client, "not-a-uuid")
    _evaluation(client, ZERO_UUID, tenant=" ")
    client.get(
        f"/v1/decisions/{ZERO_UUID}/evaluation",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("workload_id", OTHER_WORKLOAD),
        ],
    )
    client.request(
        "GET",
        "/v1/decisions//evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"x",
    )
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 1
        assert session.query(Policy).count() == 1
        assert session.query(DecisionEvaluationNode).count() > 0


# --- 404 semantics ---------------------------------------------------------


def test_evaluation_unknown_decision_is_404(client):
    assert _evaluation(client, ZERO_UUID).status_code == 404


def test_evaluation_cross_scope_decision_is_404(client):
    _, _, _, _, decided = _one_decision(client)
    decision_id = decided["decision_id"]
    assert _evaluation(client, decision_id, tenant=OTHER_TENANT).status_code == 404
    assert _evaluation(client, decision_id, workload=OTHER_WORKLOAD).status_code == 404
    assert (
        _evaluation(
            client, decision_id, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
        ).status_code
        == 404
    )


def test_evaluation_unknown_and_cross_scope_are_indistinguishable(client):
    _, _, _, _, decided = _one_decision(client)
    unknown = _evaluation(client, ZERO_UUID)
    cross = _evaluation(client, decided["decision_id"], tenant=OTHER_TENANT)
    assert unknown.status_code == cross.status_code == 404
    assert unknown.content == cross.content


# --- success shape ---------------------------------------------------------


def test_evaluation_success_shape_and_field_order(client):
    _, _, _, policy, decided = _one_decision(
        client, rule={"claim": "m", "equals": "x"}
    )
    response = _evaluation(client, decided["decision_id"])
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    raw = response.content
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw

    parsed = json.loads(raw)
    assert list(parsed) == [
        "decision_id",
        "policy_version",
        "status",
        "decided_at",
        "evaluation_version",
        "nodes",
    ]
    assert parsed["decision_id"] == decided["decision_id"]
    assert parsed["policy_version"] == policy["version"] == 1
    assert isinstance(parsed["policy_version"], int)
    assert not isinstance(parsed["policy_version"], bool)
    assert parsed["status"] == "allowed"
    assert parsed["decided_at"] == decided["decided_at"]
    parsed_at = datetime.fromisoformat(parsed["decided_at"])
    assert parsed_at.utcoffset() == timedelta(0)
    assert parsed["evaluation_version"] == 1
    assert isinstance(parsed["evaluation_version"], int)
    assert not isinstance(parsed["evaluation_version"], bool)

    nodes = parsed["nodes"]
    assert len(nodes) == 1
    assert list(nodes[0]) == ["node_index", "rule_path", "node_type", "outcome"]
    assert nodes[0] == {
        "node_index": 0,
        "rule_path": [],
        "node_type": "leaf",
        "outcome": True,
    }


def test_evaluation_root_outcome_equals_denied_status(client):
    _, _, _, _, decided = _one_decision(
        client,
        claims={"m": "different"},
        rule={"claim": "m", "equals": "x"},
        name="deny",
    )
    assert decided["status"] == "denied"
    parsed = _evaluation(client, decided["decision_id"]).json()
    assert parsed["status"] == "denied"
    assert parsed["nodes"][0]["outcome"] is False


@pytest.mark.parametrize(
    "rule,claims",
    [
        (
            {"all": [
                {"claim": "a", "equals": 1},
                {"any": [
                    {"path": ["b", "c"], "equals": "z"},
                    {"not": {"claim": "blocked", "exists": True}},
                ]},
                {"not": {"all": [
                    {"claim": "n", "gte": 10},
                    {"claim": "n", "lt": 20},
                ]}},
            ]},
            {"a": 1, "b": {"c": "z"}, "n": 15},
        ),
        (
            {"not": {"any": [
                {"claim": "x", "in": [1, 2, 3]},
                {"not": {"not": {"claim": "y", "lte": 0}}},
            ]}},
            {"x": 9, "y": 5},
        ),
        (
            {"any": [
                {"all": [
                    {"claim": "p", "exists": True},
                    {"claim": "q", "equals": None},
                ]},
                {"claim": "r", "gt": 100},
            ]},
            {"p": "present", "q": None, "r": 1},
        ),
    ],
)
def test_evaluation_nodes_are_depth_first_complete_and_consistent(
    client, rule, claims
):
    _, _, _, _, decided = _one_decision(
        client, claims=claims, rule=rule, name="deep"
    )
    parsed = _evaluation(client, decided["decision_id"]).json()
    nodes = parsed["nodes"]

    # node_index is continuous from zero and equals list position.
    assert [node["node_index"] for node in nodes] == list(range(len(nodes)))

    # Every node carries exactly the four fixed keys, with the fixed value
    # domains; rule_path is a list of non-negative integers.
    for node in nodes:
        assert set(node) == {"node_index", "rule_path", "node_type", "outcome"}
        assert node["node_type"] in ("leaf", "all", "any", "not")
        assert isinstance(node["outcome"], bool)
        assert isinstance(node["rule_path"], list)
        assert all(
            isinstance(step, int) and not isinstance(step, bool) and step >= 0
            for step in node["rule_path"]
        )

    by_path = {tuple(node["rule_path"]): node for node in nodes}

    # Root and unique pre-order paths.
    assert () in by_path
    assert len(by_path) == len(nodes)
    root = by_path[()]
    assert root["node_index"] == 0
    assert root["node_type"] in ("all", "any", "not", "leaf")
    assert root["outcome"] == (decided["status"] == "allowed")

    def check(node_rule, path):
        entry = by_path[tuple(path)]
        keys = set(node_rule)
        if "all" in keys:
            assert entry["node_type"] == "all"
            child_outcomes = [
                check(child, [*path, i])
                for i, child in enumerate(node_rule["all"])
            ]
            assert entry["outcome"] == all(child_outcomes)
        elif "any" in keys:
            assert entry["node_type"] == "any"
            child_outcomes = [
                check(child, [*path, i])
                for i, child in enumerate(node_rule["any"])
            ]
            assert entry["outcome"] == any(child_outcomes)
        elif "not" in keys:
            assert entry["node_type"] == "not"
            child_outcome = check(node_rule["not"], [*path, 0])
            assert entry["outcome"] == (not child_outcome)
        else:
            assert entry["node_type"] == "leaf"
        return entry["outcome"]

    check(rule, [])

    # Depth-first pre-order: compare against an independently generated
    # pre-order path list.
    paths = [tuple(node["rule_path"]) for node in nodes]
    expected = []

    def index_order(node_rule, path):
        expected.append(tuple(path))
        keys = set(node_rule)
        if "all" in keys:
            for i, child in enumerate(node_rule["all"]):
                index_order(child, [*path, i])
        elif "any" in keys:
            for i, child in enumerate(node_rule["any"]):
                index_order(child, [*path, i])
        elif "not" in keys:
            index_order(node_rule["not"], [*path, 0])

    index_order(rule, [])
    assert paths == expected

    # Every node is visited, including subtrees short-circuiting away from
    # the overall verdict (complete coverage, not a pruned trace).
    def count(node_rule):
        keys = set(node_rule)
        if "all" in keys:
            return 1 + sum(count(c) for c in node_rule["all"])
        if "any" in keys:
            return 1 + sum(count(c) for c in node_rule["any"])
        if "not" in keys:
            return 1 + count(node_rule["not"])
        return 1

    assert len(nodes) == count(rule)


def test_evaluation_never_returns_claim_names_or_values(client):
    # Claim names, expected scalars and actual values must not appear; the
    # explanation knows only positions, types and booleans.
    rule = {"all": [
        {"claim": "secret-measurement", "equals": SECRET_CLAIM},
        {"not": {"path": ["nested", "token"], "in": ["alpha", "beta"]}},
    ]}
    _, evidence, _, _, decided = _one_decision(
        client,
        claims={"secret-measurement": SECRET_CLAIM, "nested": {"token": "gamma"}},
        rule=rule,
        name="sensitive",
    )
    text = _evaluation(client, decided["decision_id"]).text
    assert SECRET_CLAIM not in text
    assert "secret-measurement" not in text
    assert "nested" not in text
    assert "token" not in text
    assert "alpha" not in text and "beta" not in text and "gamma" not in text
    assert evidence not in text
    assert '"claims"' not in text
    assert '"equals"' not in text
    assert '"in"' not in text
    assert '"path"' not in text
    assert '"claim"' not in text
    assert '"nonce"' not in text
    assert '"mac"' not in text
    assert '"capability"' not in text
    assert '"payload"' not in text


def test_evaluation_table_stores_no_protected_material(app, client):
    rule = {"all": [
        {"claim": "secret-measurement", "equals": SECRET_CLAIM},
        {"not": {"path": ["nested", "token"], "equals": SECRET_CLAIM}},
    ]}
    _, evidence, _, _, decided = _one_decision(
        client,
        claims={
            "secret-measurement": SECRET_CLAIM,
            "nested": {"token": SECRET_CLAIM},
        },
        rule=rule,
        name="sensitive-row",
    )
    with app.state.session_factory() as session:
        rows = (
            session.query(DecisionEvaluationNode)
            .filter(
                DecisionEvaluationNode.decision_id == decided["decision_id"]
            )
            .all()
        )
        assert rows
        for row in rows:
            columns = {
                c.name: getattr(row, c.name) for c in row.__table__.columns
            }
            for name, value in columns.items():
                rendered = str(value)
                assert evidence not in rendered, f"evidence leaked in {name}"
                assert SECRET_CLAIM not in rendered, f"claim leaked in {name}"
                assert "secret-measurement" not in rendered
                assert "nested" not in rendered and "token" not in rendered
        # Only the four documented columns exist: position, structural
        # type, boolean outcome and the integer-path JSON.
        assert set(rows[0].__table__.columns.keys()) == {
            "decision_id",
            "node_index",
            "rule_path",
            "node_type",
            "outcome",
        }


# --- idempotency, concurrency and immutability -----------------------------


def test_decision_retry_returns_the_same_explanation(app, client):
    created, evidence, evidence_id = _submit_verified(client, claims={"m": "x"})
    policy = _policy(client, {"not": {"claim": "m", "equals": "x"}})
    first = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    assert first.status_code == 200
    second = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    assert second.status_code == 200
    assert second.json() == first.json()

    decision_id = first.json()["decision_id"]
    body1 = _evaluation(client, decision_id).content
    body2 = _evaluation(client, decision_id).content
    assert body1 == body2
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 1
        assert (
            session.query(DecisionEvaluationNode)
            .filter(DecisionEvaluationNode.decision_id == decision_id)
            .count()
            == 2
        )


def test_concurrent_decisions_leave_one_decision_and_one_explanation(app):
    client = TestClient(app)
    created, evidence, evidence_id = _submit_verified(
        client, claims={"m": "x", "n": 3}
    )
    rule = {"all": [
        {"claim": "m", "equals": "x"},
        {"not": {"any": [
            {"claim": "n", "gt": 10},
            {"claim": "missing", "exists": True},
        ]}},
    ]}
    policy = _policy(client, rule)
    payload = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
        "policy_id": policy["policy_id"],
    }

    def decide():
        return TestClient(app).post(
            f"/v1/evidence/{evidence_id}/decisions", json=payload
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: decide(), range(16)))

    assert all(r.status_code == 200 for r in responses)
    assert len({r.text for r in responses}) == 1
    decision_id = responses[0].json()["decision_id"]
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 1
        rows = (
            session.query(DecisionEvaluationNode)
            .filter(DecisionEvaluationNode.decision_id == decision_id)
            .order_by(DecisionEvaluationNode.node_index)
            .all()
        )
        assert [r.node_index for r in rows] == [0, 1, 2, 3, 4, 5]

    # Every concurrent winner observes that same stored explanation.
    bodies = {
        client.get(
            f"/v1/decisions/{decision_id}/evaluation",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        ).content
        for _ in range(4)
    }
    assert len(bodies) == 1


def test_evaluation_is_immutable_after_retirement_and_new_version(client):
    created, evidence, evidence_id = _submit_verified(client, claims={"m": "x"})
    rule = {"all": [{"claim": "m", "equals": "x"}, {"not": {"claim": "z",
                                                              "exists": True}}]}
    v1 = _policy(client, rule, name="r")
    decided = _decide(client, created, evidence, evidence_id, v1["policy_id"])
    assert decided.status_code == 200
    decision_id = decided.json()["decision_id"]
    before = _evaluation(client, decision_id)
    assert before.status_code == 200

    # A newer, contradicting version and retirement of v1 must not rewrite
    # history.
    _policy(client, {"claim": "m", "equals": "other"}, name="r")
    retired = client.post(
        f"/v1/policies/{v1['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200, retired.text

    after = _evaluation(client, decision_id)
    assert after.status_code == 200
    assert after.content == before.content


def test_evaluation_is_stable_across_repeated_reads(client):
    _, _, _, _, decided = _one_decision(client)
    first = _evaluation(client, decided["decision_id"])
    second = _evaluation(client, decided["decision_id"])
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_evaluation_read_writes_nothing(app, client):
    _, _, _, _, decided = _one_decision(client)
    with app.state.session_factory() as session:
        decisions_before = session.query(Decision).count()
        audits_before = session.query(AuditEvent).count()
        proof_before = session.query(ProofLifecycleEvent).count()
        nodes_before = session.query(DecisionEvaluationNode).count()

    for _ in range(3):
        assert _evaluation(client, decided["decision_id"]).status_code == 200

    with app.state.session_factory() as session:
        assert session.query(Decision).count() == decisions_before
        assert session.query(AuditEvent).count() == audits_before
        assert session.query(ProofLifecycleEvent).count() == proof_before
        assert session.query(DecisionEvaluationNode).count() == nodes_before


# --- persistence across restart --------------------------------------------


def test_evaluation_consistent_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart_evaluation.db"
    first = create_app(url)
    client1 = TestClient(first)
    _, _, _, policy, decided = _one_decision(
        client1,
        claims={"m": "x"},
        rule={"not": {"all": [
            {"claim": "m", "equals": "x"},
            {"claim": "m", "exists": True},
        ]}},
    )
    first_body = client1.get(
        f"/v1/decisions/{decided['decision_id']}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).content
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    response = client2.get(
        f"/v1/decisions/{decided['decision_id']}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    assert response.content == first_body
    parsed = response.json()
    assert parsed["decision_id"] == decided["decision_id"]
    assert parsed["policy_version"] == policy["version"]
    assert [n["node_type"] for n in parsed["nodes"]] == ["not", "all", "leaf", "leaf"]
    second.state.engine.dispose()


# --- legacy decisions (pre-upgrade) ----------------------------------------


def test_decision_without_evaluation_is_409(app, client):
    _, _, _, _, decided = _one_decision(client)
    decision_id = decided["decision_id"]
    with app.state.session_factory() as session:
        deleted = (
            session.query(DecisionEvaluationNode)
            .filter(DecisionEvaluationNode.decision_id == decision_id)
            .delete()
        )
        assert deleted >= 1
        session.commit()

    response = _evaluation(client, decision_id)
    assert response.status_code == 409
    assert response.json() == {"detail": "evaluation not recorded"}
    # Repeated reads stay 409; nothing is backfilled or reconstructed.
    assert _evaluation(client, decision_id).status_code == 409
    with app.state.session_factory() as session:
        assert (
            session.query(DecisionEvaluationNode)
            .filter(DecisionEvaluationNode.decision_id == decision_id)
            .count()
            == 0
        )


def test_evaluation_unchanged_across_key_rotation(tmp_path, monkeypatch):
    from proof_release.envelopes import b64url_encode

    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/rotation_evaluation.db"
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv(
        "PROOF_RELEASE_MASTER_KEY", b64url_encode(b"0123456789abcdef0123456789abcdef")
    )
    first = create_app(url)
    client1 = TestClient(first)
    _, _, _, _, decided = _one_decision(
        client1,
        claims={"m": "x"},
        rule={"not": {"claim": "m", "equals": "x"}},
    )
    first_body = client1.get(
        f"/v1/decisions/{decided['decision_id']}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).content
    first.state.engine.dispose()

    # Rotate to a new current master key and reopen the same database.
    new_key = b64url_encode(b"fedcba9876543210fedcba9876543210")
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING",
        json.dumps({"current_version": 2, "keys": {"2": new_key}}),
    )
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    second = create_app(url)
    client2 = TestClient(second)
    response = client2.get(
        f"/v1/decisions/{decided['decision_id']}/evaluation",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    assert response.content == first_body
    second.state.engine.dispose()


def test_legacy_409_is_distinct_from_unknown_404(client):
    _, _, _, _, decided = _one_decision(client)
    decision_id = decided["decision_id"]
    # An existing, in-scope decision with no explanation is 409; an
    # unknown id is still 404.
    with client.app.state.session_factory() as session:
        session.query(DecisionEvaluationNode).filter(
            DecisionEvaluationNode.decision_id == decision_id
        ).delete()
        session.commit()
    assert _evaluation(client, decision_id).status_code == 409
    assert _evaluation(client, ZERO_UUID).status_code == 404


# --- server failure and integrity ------------------------------------------


def test_evaluation_storage_failure_returns_500(app, client):
    from sqlalchemy import text

    _, _, _, _, decided = _one_decision(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE decision_evaluation_nodes"))
    response = _evaluation(client, decided["decision_id"])
    assert response.status_code == 500
    assert response.json() == {"detail": "decision evaluation unavailable"}


def test_evaluation_missing_policy_snapshot_returns_500(app, client):
    from sqlalchemy import text

    _, _, _, _, decided = _one_decision(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policies"))
    response = _evaluation(client, decided["decision_id"])
    assert response.status_code == 500
    assert response.json() == {"detail": "decision evaluation unavailable"}


def test_evaluation_corrupted_node_is_500_with_no_half_result(app, client):
    _, _, _, _, decided = _one_decision(
        client, rule={"all": [
            {"claim": "m", "equals": "x"},
            {"not": {"claim": "n", "exists": True}},
        ]},
    )
    decision_id = decided["decision_id"]
    with app.state.session_factory() as session:
        tail = (
            session.query(DecisionEvaluationNode)
            .filter(
                DecisionEvaluationNode.decision_id == decision_id,
                DecisionEvaluationNode.node_index == 2,
            )
            .one()
        )
        tail.node_type = "bogus"
        session.commit()

    response = _evaluation(client, decision_id)
    assert response.status_code == 500
    assert response.json() == {"detail": "decision evaluation unavailable"}


def test_evaluation_truncated_nodes_are_500(app, client):
    _, _, _, _, decided = _one_decision(
        client, rule={"all": [
            {"claim": "m", "equals": "x"},
            {"not": {"claim": "n", "exists": True}},
        ]},
    )
    decision_id = decided["decision_id"]
    with app.state.session_factory() as session:
        deleted = (
            session.query(DecisionEvaluationNode)
            .filter(
                DecisionEvaluationNode.decision_id == decision_id,
                DecisionEvaluationNode.node_index >= 2,
            )
            .delete()
        )
        assert deleted == 2
        session.commit()

    response = _evaluation(client, decision_id)
    assert response.status_code == 500
    assert response.json() == {"detail": "decision evaluation unavailable"}


def test_evaluation_root_status_contradiction_is_500(app, client):
    _, _, _, _, decided = _one_decision(client)
    decision_id = decided["decision_id"]
    with app.state.session_factory() as session:
        root = (
            session.query(DecisionEvaluationNode)
            .filter(
                DecisionEvaluationNode.decision_id == decision_id,
                DecisionEvaluationNode.node_index == 0,
            )
            .one()
        )
        root.outcome = not root.outcome
        session.commit()

    response = _evaluation(client, decision_id)
    assert response.status_code == 500
    assert response.json() == {"detail": "decision evaluation unavailable"}


def test_evaluation_node_index_gap_is_500(app, client):
    _, _, _, _, decided = _one_decision(
        client, rule={"all": [
            {"claim": "m", "equals": "x"},
            {"not": {"claim": "n", "exists": True}},
        ]},
    )
    decision_id = decided["decision_id"]
    with app.state.session_factory() as session:
        deleted = (
            session.query(DecisionEvaluationNode)
            .filter(
                DecisionEvaluationNode.decision_id == decision_id,
                DecisionEvaluationNode.node_index == 1,
            )
            .delete()
        )
        assert deleted == 1
        session.commit()

    response = _evaluation(client, decision_id)
    assert response.status_code == 500
    assert response.json() == {"detail": "decision evaluation unavailable"}
