"""Tests for POST /v1/policies/{policy_id}/simulate — the read-only
policy simulation entry point.

A simulation evaluates one fixed policy version's immutable rule against
caller-supplied claims and returns the verdict plus a depth-first
explanation of the complete rule tree. It validates every request-shape
rule to 422 before any state is read or written, hides unknown or
cross-scope policies behind one indistinguishable 404, simulates active
and retired versions alike, and never creates a decision, release grant,
proof event or audit event, never persists claims and never consumes the
shared business rate-limit budget. The response carries identifiers, the
version's name/version/status, the boolean verdict and node
positions/types/booleans only — never claims, matched or compared
values, floats or protected material.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from proof_release.app import create_app

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
SECRET = "unit-test-secret"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"
UPPER_UUID = "ABCDEF12-3456-7890-ABCD-EF1234567890"

SECRET_CLAIM = "super-secret-claim-value-987654"

RESPONSE_KEYS = [
    "tenant_id",
    "workload_id",
    "policy_id",
    "policy_name",
    "policy_version",
    "policy_status",
    "allowed",
    "evaluation",
]

#: Tables a simulation must never write to, directly or indirectly.
READ_ONLY_TABLES = (
    "decisions",
    "decision_evaluation_nodes",
    "release_grants",
    "release_grant_events",
    "audit_events",
    "proof_lifecycle_events",
    "rate_limit_counters",
)


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    application = create_app(f"sqlite:///{tmp_path}/policy_simulations.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


# --- helpers ---------------------------------------------------------------


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


def _simulate(client, policy_id, *, tenant=TENANT, workload=WORKLOAD,
              claims=None):
    return client.post(
        f"/v1/policies/{policy_id}/simulate",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "claims": {} if claims is None else claims,
        },
    )


def _table_counts(app):
    counts = {}
    with app.state.engine.connect() as conn:
        for table in READ_ONLY_TABLES:
            counts[table] = conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar()
    return counts


# --- happy path ------------------------------------------------------------


def test_simulate_allowed_leaf(client):
    policy = _policy(client)
    response = _simulate(client, policy["policy_id"],
                         claims={"measurement": "abc"})
    assert response.status_code == 200
    body = response.content
    assert body.endswith(b"\n") and not body.endswith(b"\n\n")
    # Compact separators: no whitespace anywhere outside string values.
    assert b", " not in body and b": " not in body
    payload = json.loads(body)
    assert list(payload.keys()) == RESPONSE_KEYS
    assert payload["tenant_id"] == TENANT
    assert payload["workload_id"] == WORKLOAD
    assert payload["policy_id"] == policy["policy_id"]
    assert payload["policy_name"] == "release"
    assert payload["policy_version"] == 1
    assert payload["policy_status"] == "active"
    assert payload["allowed"] is True
    assert payload["evaluation"] == [
        {"node_index": 0, "rule_path": [], "node_type": "leaf",
         "outcome": True}
    ]


def test_simulate_denied_when_claim_differs_or_missing(client):
    policy = _policy(client)
    mismatch = _simulate(client, policy["policy_id"],
                         claims={"measurement": "xyz"})
    assert mismatch.status_code == 200
    assert mismatch.json()["allowed"] is False
    missing = _simulate(client, policy["policy_id"], claims={})
    assert missing.status_code == 200
    assert missing.json()["allowed"] is False


def test_simulate_compound_tree_depth_first(client):
    rule = {
        "all": [
            {"claim": "measurement", "equals": "abc"},
            {"any": [
                {"path": ["device", "tier"], "in": ["gold", "silver"]},
                {"not": {"claim": "revoked", "exists": True}},
            ]},
            {"claim": "version", "gte": 2},
        ]
    }
    policy = _policy(client, rule)
    claims = {
        "measurement": "abc",
        "device": {"tier": "gold"},
        "version": 5,
    }
    response = _simulate(client, policy["policy_id"], claims=claims)
    assert response.status_code == 200
    payload = response.json()
    assert payload["allowed"] is True
    assert payload["evaluation"] == [
        {"node_index": 0, "rule_path": [], "node_type": "all",
         "outcome": True},
        {"node_index": 1, "rule_path": [0], "node_type": "leaf",
         "outcome": True},
        {"node_index": 2, "rule_path": [1], "node_type": "any",
         "outcome": True},
        {"node_index": 3, "rule_path": [1, 0], "node_type": "leaf",
         "outcome": True},
        {"node_index": 4, "rule_path": [1, 1], "node_type": "not",
         "outcome": True},
        {"node_index": 5, "rule_path": [1, 1, 0], "node_type": "leaf",
         "outcome": False},
        {"node_index": 6, "rule_path": [2], "node_type": "leaf",
         "outcome": True},
    ]
    # The root outcome equals the verdict and every compound node agrees
    # with its children.
    nodes = payload["evaluation"]
    assert nodes[0]["outcome"] is payload["allowed"]


def test_simulate_compound_outcomes_follow_all_any_not(client):
    rule = {
        "any": [
            {"all": [
                {"claim": "a", "equals": 1},
                {"claim": "b", "equals": 2},
            ]},
            {"not": {"claim": "c", "exists": True}},
        ]
    }
    policy = _policy(client, rule)
    # First branch fails (b != 2), second holds (c absent).
    response = _simulate(client, policy["policy_id"],
                         claims={"a": 1, "b": 3})
    payload = response.json()
    assert payload["allowed"] is True
    nodes = payload["evaluation"]
    by_path = {tuple(n["rule_path"]): n for n in nodes}
    assert by_path[(0,)]["outcome"] is False  # all: a ok, b fails
    assert by_path[(0, 0)]["outcome"] is True
    assert by_path[(0, 1)]["outcome"] is False
    assert by_path[(1,)]["outcome"] is True  # not: c absent
    # Both branches present in the explanation even though `any` could
    # short-circuit: the tree is covered completely.
    assert len(nodes) == 6


@pytest.mark.parametrize(
    "rule, claims, expected",
    [
        ({"claim": "x", "equals": None}, {"x": None}, True),
        ({"claim": "x", "equals": 1}, {"x": True}, False),
        ({"claim": "x", "equals": 1}, {"x": 1.0}, True),
        ({"claim": "x", "in": ["a", "b"]}, {"x": "b"}, True),
        ({"claim": "x", "in": ["a", "b"]}, {"x": "c"}, False),
        ({"claim": "x", "exists": True}, {"x": None}, True),
        ({"claim": "x", "exists": True}, {}, False),
        ({"claim": "x", "lt": 3}, {"x": 2}, True),
        ({"claim": "x", "lt": 3}, {"x": 3}, False),
        ({"claim": "x", "lte": 3}, {"x": 3}, True),
        ({"claim": "x", "gt": 3}, {"x": 4}, True),
        ({"claim": "x", "gte": 3}, {"x": 3}, True),
        ({"claim": "x", "gt": 3}, {"x": True}, False),
        ({"claim": "x", "lt": 3}, {"x": "2"}, False),
    ],
)
def test_simulate_leaf_comparison_semantics(client, rule, claims, expected):
    policy = _policy(client, rule)
    response = _simulate(client, policy["policy_id"], claims=claims)
    assert response.status_code == 200
    assert response.json()["allowed"] is expected


def test_simulate_path_descends_objects_only(client):
    rule = {"path": ["device", "board", "rev"], "gte": 3}
    policy = _policy(client, rule)
    # Object descent works.
    ok = _simulate(client, policy["policy_id"],
                   claims={"device": {"board": {"rev": 4}}})
    assert ok.json()["allowed"] is True
    # A missing intermediate field is unsatisfied, not an error.
    missing = _simulate(client, policy["policy_id"],
                        claims={"device": {}})
    assert missing.json()["allowed"] is False
    # Arrays never expand and are never indexed: a list under a segment
    # blocks further descent.
    array = _simulate(client, policy["policy_id"],
                      claims={"device": {"board": [{"rev": 4}]}})
    assert array.json()["allowed"] is False
    indexed = _simulate(client, policy["policy_id"],
                        claims={"device": [{"board": {"rev": 4}}]})
    assert indexed.json()["allowed"] is False


def test_simulate_retired_policy(client):
    policy = _policy(client)
    retired = client.post(
        f"/v1/policies/{policy['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200
    response = _simulate(client, policy["policy_id"],
                         claims={"measurement": "abc"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["policy_status"] == "retired"
    assert payload["allowed"] is True


def test_simulate_is_repeatable(client):
    policy = _policy(client)
    first = _simulate(client, policy["policy_id"],
                      claims={"measurement": "abc"})
    second = _simulate(client, policy["policy_id"],
                       claims={"measurement": "abc"})
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_simulate_response_hides_claim_material(client):
    policy = _policy(client)
    response = _simulate(client, policy["policy_id"],
                         claims={"measurement": "abc",
                                 "secret": SECRET_CLAIM})
    assert response.status_code == 200
    assert SECRET_CLAIM not in response.text
    assert "secret" not in response.text
    # The claims themselves are never echoed back.
    assert "claims" not in response.json()


def test_simulate_writes_no_state(client, app):
    policy = _policy(client)
    before = _table_counts(app)
    for _ in range(3):
        response = _simulate(client, policy["policy_id"],
                             claims={"measurement": "abc"})
        assert response.status_code == 200
    assert _table_counts(app) == before
    # The policy row itself is untouched (still active, no retirement).
    unchanged = client.get(
        "/v1/policies",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert unchanged.status_code == 200
    (listed,) = unchanged.json()["policies"]
    assert listed["status"] == "active"
    assert listed["retired_at"] is None


def test_simulate_consumes_no_grant_budget(client, app):
    # Far more simulations than the shared per-minute business budget
    # would admit; every one succeeds and no counter row ever appears.
    policy = _policy(client)
    for _ in range(12):
        response = _simulate(client, policy["policy_id"],
                             claims={"measurement": "abc"})
        assert response.status_code == 200
    assert _table_counts(app)["rate_limit_counters"] == 0


# --- 404 semantics ---------------------------------------------------------


def test_simulate_unknown_policy_is_404(client):
    response = _simulate(client, ZERO_UUID)
    assert response.status_code == 404


def test_simulate_cross_scope_is_one_indistinguishable_404(client):
    policy = _policy(client)
    unknown = _simulate(client, ZERO_UUID)
    other_tenant = _simulate(client, policy["policy_id"],
                             tenant=OTHER_TENANT)
    other_workload = _simulate(client, policy["policy_id"],
                               workload=OTHER_WORKLOAD)
    assert unknown.status_code == 404
    assert other_tenant.status_code == 404
    assert other_workload.status_code == 404
    assert unknown.content == other_tenant.content == other_workload.content


# --- 422 request shape -----------------------------------------------------


def test_simulate_rejects_non_canonical_path_ids(client):
    for bad in (
        UPPER_UUID,  # uppercase is not the canonical form
        f" {ZERO_UUID}",  # whitespace-padded
        f"{ZERO_UUID} ",
        ZERO_UUID.replace("-", ""),  # dashes missing
        "not-a-uuid",
        " ",
    ):
        response = _simulate(client, bad)
        assert response.status_code == 422, bad


def test_simulate_rejects_empty_path_segment(client):
    response = client.post(
        "/v1/policies//simulate",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": {}},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {},  # everything missing
        {"tenant_id": TENANT, "workload_id": WORKLOAD},  # claims missing
        {"tenant_id": TENANT, "claims": {}},  # workload_id missing
        {"workload_id": WORKLOAD, "claims": {}},  # tenant_id missing
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": {},
         "policy_id": ZERO_UUID},  # unknown field
        {"tenant_id": "", "workload_id": WORKLOAD, "claims": {}},
        {"tenant_id": "   ", "workload_id": WORKLOAD, "claims": {}},
        {"tenant_id": TENANT, "workload_id": "", "claims": {}},
        {"tenant_id": TENANT, "workload_id": "\t", "claims": {}},
        {"tenant_id": 7, "workload_id": WORKLOAD, "claims": {}},
        {"tenant_id": TENANT, "workload_id": None, "claims": {}},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": []},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": "x"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": None},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": 1},
    ],
)
def test_simulate_rejects_invalid_bodies(client, body):
    policy = _policy(client)
    response = client.post(
        f"/v1/policies/{policy['policy_id']}/simulate", json=body
    )
    assert response.status_code == 422


def test_simulate_rejects_non_object_and_malformed_bodies(client):
    policy = _policy(client)
    for raw in (b"[]", b'"text"', b"1", b"null", b"", b"{not json"):
        response = client.post(
            f"/v1/policies/{policy['policy_id']}/simulate",
            content=raw,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422, raw


def test_simulate_rejects_duplicate_fields(client):
    policy = _policy(client)
    raw = (
        b'{"tenant_id":"tenant-a","workload_id":"workload-1",'
        b'"claims":{},"tenant_id":"tenant-a"}'
    )
    response = client.post(
        f"/v1/policies/{policy['policy_id']}/simulate",
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_simulate_rejects_duplicate_fields_nested_in_claims(client):
    policy = _policy(client)
    raw = (
        b'{"tenant_id":"tenant-a","workload_id":"workload-1",'
        b'"claims":{"x":1,"x":2}}'
    )
    response = client.post(
        f"/v1/policies/{policy['policy_id']}/simulate",
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "raw",
    [
        b'{"tenant_id":"tenant-a","workload_id":"workload-1","claims":NaN}',
        b'{"tenant_id":"tenant-a","workload_id":"workload-1",'
        b'"claims":{"x":Infinity}}',
        b'{"tenant_id":"tenant-a","workload_id":"workload-1",'
        b'"claims":{"x":-Infinity}}',
        # Overflow spelling: parses to inf without a NaN/Infinity literal.
        b'{"tenant_id":"tenant-a","workload_id":"workload-1",'
        b'"claims":{"x":1e999}}',
        b'{"tenant_id":NaN,"workload_id":"workload-1","claims":{}}',
    ],
)
def test_simulate_rejects_non_finite_numbers(client, raw):
    policy = _policy(client)
    response = client.post(
        f"/v1/policies/{policy['policy_id']}/simulate",
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_simulate_422_reads_and_writes_no_state(client, app):
    policy = _policy(client)
    before = _table_counts(app)
    bad_bodies = [
        {"tenant_id": TENANT, "workload_id": WORKLOAD},  # missing claims
        {"tenant_id": "", "workload_id": WORKLOAD, "claims": {}},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": {},
         "extra": 1},
    ]
    for body in bad_bodies:
        response = client.post(
            f"/v1/policies/{policy['policy_id']}/simulate", json=body
        )
        assert response.status_code == 422
    for bad_id in (UPPER_UUID, "not-a-uuid"):
        response = _simulate(client, bad_id)
        assert response.status_code == 422
    assert _table_counts(app) == before
    # The policy is still simulatable afterwards: nothing was touched.
    ok = _simulate(client, policy["policy_id"],
                   claims={"measurement": "abc"})
    assert ok.status_code == 200


# --- failure semantics -----------------------------------------------------


def test_simulate_query_failure_is_500(client, app, monkeypatch):
    policy = _policy(client)
    from sqlalchemy import event as sa_event

    def fail_read(dbapi_connection, cursor, statement, parameters, context,
                  executemany):
        if "FROM policies" in statement:
            raise RuntimeError("injected read failure")

    sa_event.listen(app.state.engine, "before_cursor_execute", fail_read)
    try:
        response = _simulate(client, policy["policy_id"],
                             claims={"measurement": "abc"})
    finally:
        sa_event.remove(app.state.engine, "before_cursor_execute", fail_read)
    assert response.status_code == 500
    # Fixed sanitized detail; no exception text leaks.
    assert response.json() == {"detail": "policy simulation unavailable"}


def test_simulate_corrupt_stored_rule_is_500(client, app):
    policy = _policy(client)
    with app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE policies SET rule_json = :junk WHERE policy_id = :pid"),
            {"junk": "{not-json", "pid": policy["policy_id"]},
        )
    response = _simulate(client, policy["policy_id"],
                         claims={"measurement": "abc"})
    assert response.status_code == 500
    assert response.json() == {"detail": "policy simulation unavailable"}


# --- compatibility: neighbouring entry points are unchanged ----------------


def test_policy_lifecycle_entry_points_unchanged(client):
    policy = _policy(client)
    assert policy["name"] == "release"
    assert policy["version"] == 1
    listed = client.get(
        "/v1/policies",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert listed.status_code == 200
    assert len(listed.json()["policies"]) == 1
    retired = client.post(
        f"/v1/policies/{policy['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200
    again = client.post(
        f"/v1/policies/{policy['policy_id']}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert again.status_code == 409
