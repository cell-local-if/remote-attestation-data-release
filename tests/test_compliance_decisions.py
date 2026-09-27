"""Tests for GET /v1/compliance/decisions.

Tenant-isolated, cursor-stable, snapshot-fixed compliance listing of
release decisions. Unlike the proof-lifecycle timeline — which records
only the *first* decision for an evidence — this listing covers every
policy version decided against the same evidence: one immutable row per
(evidence, policy version). Decisions are written in the decision
transaction; the query is read-only, never appends an event or audit, and
fixes its replayable snapshot to the per-scope business commit boundary.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import Decision, DecisionCommitCounter
from proof_release.envelopes import b64url_encode

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
    application = create_app(f"sqlite:///{tmp_path}/compliance_decisions.db")
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
    """Receive, verify and decide exactly once; return all generated ids."""
    created, evidence, evidence_id = _submit_verified(client, claims=claims)
    policy = _policy(client, rule, name=name)
    decided = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    assert decided.status_code == 200, decided.text
    return created, evidence, evidence_id, policy, decided.json()


def _query(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        "/v1/compliance/decisions",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


def _walk(client, *, page_size=None, monkeypatch=None, **params):
    token = None
    rows = []
    for _ in range(50):
        query_params = dict(params)
        if token is not None:
            query_params["cursor"] = token
        if page_size is not None:
            monkeypatch.setattr(app_module, "DECISION_PAGE_SIZE", page_size)
        page = _query(client, **query_params)
        assert page.status_code == 200, page.text
        data = page.json()
        rows.extend(data["decisions"])
        if data["complete"]:
            assert data["next_cursor"] == ""
            return rows
        token = data["next_cursor"]
        assert token
    raise AssertionError("pagination never completed")  # pragma: no cover


# --- request validation ----------------------------------------------------


def test_query_requires_scope_parameters(client):
    assert client.get("/v1/compliance/decisions").status_code == 422
    assert (
        client.get(
            "/v1/compliance/decisions", params={"workload_id": WORKLOAD}
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/v1/compliance/decisions", params={"tenant_id": TENANT}
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
        {"decision_id": ""},
        {"decision_id": "   "},
        {"decision_id": "not-a-uuid"},
        {"decision_id": ZERO_UUID[:-1] + "Z"},
        {"decision_id": "  " + ZERO_UUID},
        {"decision_id": "ABCDEF12-3456-7890-ABCD-EF1234567890"},
        {"evidence_id": "abc123"},
        {"evidence_id": "ABCDEF12-3456-7890-ABCD-EF1234567890"},
        {"policy_id": ""},
        {"policy_id": "   "},
        {"policy_id": "not-a-uuid"},
        {"policy_id": "ABCDEF12-3456-7890-ABCD-EF1234567890"},
        {"status": ""},
        {"status": "   "},
        {"status": "ALLOWED"},
        {"status": "pending"},
        {"status": "accepted"},
        {"decided_after": "not-a-timestamp"},
        {"decided_after": "2026-01-01T00:00:00"},
        {"decided_before": "2026-01-01"},
        {"decided_after": ""},
        {"decided_before": "  "},
        {
            "decided_after": "2026-01-02T00:00:00Z",
            "decided_before": "2026-01-01T00:00:00Z",
        },
        {
            "decided_after": "2026-01-01T01:00:00+01:00",
            "decided_before": "2026-01-01T00:00:00Z",
        },
        {"cursor": "   "},
        {"cursor": "not base64!!"},
        {"cursor": "AAA="},
        {"bogus": "value"},
        {"limit": "10"},
        {"event_type": "proof-decision"},
    ],
)
def test_query_rejects_invalid_parameters(client, params):
    response = _query(client, **params)
    assert response.status_code == 422, response.text


def test_repeated_parameter_is_422(client):
    response = client.get(
        "/v1/compliance/decisions",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("status", "allowed"),
            ("status", "denied"),
        ],
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"{}", b"  ", b"not json", b"\x00"])
def test_non_empty_body_is_422(client, body):
    response = client.request(
        "GET",
        "/v1/compliance/decisions",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=body,
    )
    assert response.status_code == 422


def test_missing_and_zero_length_body_are_accepted(client):
    no_body = client.get(
        "/v1/compliance/decisions",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    empty_body = client.request(
        "GET",
        "/v1/compliance/decisions",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"",
    )
    assert no_body.status_code == 200
    assert empty_body.status_code == 200
    assert no_body.content == empty_body.content


def test_query_rejects_non_utc_offset_even_if_instant_matches(client):
    response = _query(
        client,
        decided_after="2026-01-01T02:00:00+02:00",
        decided_before="2026-01-01T00:00:00Z",
    )
    assert response.status_code == 422


def test_invalid_parameters_write_no_state(app, client):
    _query(client, status="nope")
    _query(client, decision_id="not-a-uuid", policy_id="also-bad")
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 0
        assert session.query(DecisionCommitCounter).count() == 0


# --- 404 semantics ---------------------------------------------------------


def test_unknown_identifiers_return_404(client):
    assert _query(client, decision_id=ZERO_UUID).status_code == 404
    assert _query(client, evidence_id=ZERO_UUID).status_code == 404
    assert _query(client, policy_id=ZERO_UUID).status_code == 404


def test_cross_scope_identifiers_return_404(client):
    created, evidence, evidence_id, policy, decided = _one_decision(client)
    decision_id = decided["decision_id"]

    assert (
        _query(client, tenant=OTHER_TENANT, decision_id=decision_id).status_code
        == 404
    )
    assert (
        _query(client, workload=OTHER_WORKLOAD, decision_id=decision_id).status_code
        == 404
    )
    assert (
        _query(client, tenant=OTHER_TENANT, evidence_id=evidence_id).status_code
        == 404
    )
    assert (
        _query(client, workload=OTHER_WORKLOAD, evidence_id=evidence_id).status_code
        == 404
    )
    assert (
        _query(client, tenant=OTHER_TENANT, policy_id=policy["policy_id"]).status_code
        == 404
    )
    assert (
        _query(client, workload=OTHER_WORKLOAD, policy_id=policy["policy_id"]).status_code
        == 404
    )


def test_unknown_identifier_never_distinguishes_existence(client):
    # Unknown and cross-scope identifiers return the same status and body
    # shape; nothing leaks about whether the id exists elsewhere.
    created, evidence, evidence_id, policy, decided = _one_decision(client)
    unknown = _query(client, decision_id=ZERO_UUID)
    cross = _query(client, tenant=OTHER_TENANT, decision_id=decided["decision_id"])
    assert unknown.status_code == cross.status_code == 404


# --- empty scope / response shape ------------------------------------------


def test_empty_scope_returns_empty_completed_page(client):
    response = _query(client)
    assert response.status_code == 200
    assert response.json() == {"decisions": [], "next_cursor": "", "complete": True}


def test_response_is_compact_json_with_single_trailing_newline(client):
    _one_decision(client)
    response = _query(client)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == ["decisions", "next_cursor", "complete"]


def test_decision_rows_have_exact_shape_and_ordered_fields(client):
    _one_decision(client)
    rows = _query(client).json()["decisions"]
    assert len(rows) == 1
    row = rows[0]
    assert list(row) == [
        "decision_id",
        "evidence_id",
        "policy_id",
        "policy_version",
        "status",
        "decided_at",
    ]
    assert set(row) == {
        "decision_id",
        "evidence_id",
        "policy_id",
        "policy_version",
        "status",
        "decided_at",
    }
    for key in ("decision_id", "evidence_id", "policy_id", "status", "decided_at"):
        assert isinstance(row[key], str) and row[key]
    assert isinstance(row["policy_version"], int) and not isinstance(
        row["policy_version"], bool
    )
    parsed = datetime.fromisoformat(row["decided_at"])
    assert parsed.utcoffset() == timedelta(0)
    assert row["status"] in {"allowed", "denied"}


def test_no_floats_or_non_finite_values(client):
    _one_decision(client)
    parsed = json.loads(_query(client).content)
    assert isinstance(parsed["complete"], bool)
    assert isinstance(parsed["next_cursor"], str)

    def _check(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            return
        if isinstance(value, float):  # pragma: no cover - structural guard
            raise AssertionError("float present")
        assert value is None or isinstance(value, str)

    for row in parsed["decisions"]:
        for value in row.values():
            _check(value)


def test_response_never_contains_protected_material(client):
    created, evidence, _, _, _ = _one_decision(client)
    text = _query(client).text
    assert evidence not in text
    assert created["nonce"] not in text
    assert '"claims"' not in text
    assert '"capability"' not in text
    assert '"payload"' not in text
    assert '"rule"' not in text


# --- every policy version decision is covered ------------------------------


def test_every_policy_version_decision_for_one_evidence_is_listed(client):
    # The baseline proof audit records only the FIRST decision; this query
    # must cover every policy version decided against the same evidence.
    created, evidence, evidence_id, policy_one, first = _one_decision(client)
    policy_two = _policy(
        client, {"claim": "m", "equals": "nope"}, name="r2"
    )
    second = _decide(
        client, created, evidence, evidence_id, policy_two["policy_id"]
    )
    assert second.status_code == 200, second.text
    second_json = second.json()
    assert second_json["decision_id"] != first["decision_id"]
    assert second_json["evidence_id"] == evidence_id

    rows = _walk(client, evidence_id=evidence_id)
    assert len(rows) == 2
    assert {row["decision_id"] for row in rows} == {
        first["decision_id"],
        second_json["decision_id"],
    }
    assert {row["evidence_id"] for row in rows} == {evidence_id}
    assert {row["policy_id"] for row in rows} == {
        policy_one["policy_id"],
        policy_two["policy_id"],
    }
    by_policy = {row["policy_id"]: row for row in rows}
    assert by_policy[policy_one["policy_id"]]["policy_version"] == 1
    assert by_policy[policy_one["policy_id"]]["status"] == "allowed"
    assert by_policy[policy_two["policy_id"]]["policy_version"] == 1
    assert by_policy[policy_two["policy_id"]]["status"] == "denied"


def test_same_policy_replay_returns_one_row(client):
    created, evidence, evidence_id, policy, first = _one_decision(client)
    again = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    assert again.json()["decision_id"] == first["decision_id"]
    rows = _walk(client)
    assert len(rows) == 1


def test_multiple_versions_of_same_named_policy_are_distinct_rows(client):
    created, evidence, evidence_id, v1, first = _one_decision(client)
    v2 = _policy(client, {"claim": "m", "equals": "x"}, name="r")
    assert v2["version"] == 2
    # v2 can never produce a decision against the first evidence because
    # the first evidence already has a v1 decision; a *fresh* evidence
    # decided against v2 yields a second row carrying version 2.
    created2, evidence2, eid2 = _submit_verified(client)
    decided2 = _decide(client, created2, evidence2, eid2, v2["policy_id"])
    assert decided2.status_code == 200
    rows = _walk(client)
    versions = sorted(row["policy_version"] for row in rows)
    assert versions == [1, 2]


def test_rows_are_ordered_by_decided_at_then_decision_id(client):
    _one_decision(client, name="a")
    _one_decision(client, name="b")
    _one_decision(client, name="c")
    rows = _walk(client)
    keys = [(row["decided_at"], row["decision_id"]) for row in rows]
    assert keys == sorted(keys)
    assert len({row["decision_id"] for row in rows}) == 3


# --- filtering -------------------------------------------------------------


def test_filter_by_decision_id(client):
    _, _, _, _, wanted = _one_decision(client)
    _one_decision(client, name="other")
    rows = _query(client, decision_id=wanted["decision_id"]).json()["decisions"]
    assert [row["decision_id"] for row in rows] == [wanted["decision_id"]]


def test_filter_by_evidence_id(client):
    _, _, first_eid, _, _ = _one_decision(client)
    _, _, other_eid, _, _ = _one_decision(client, name="other")
    rows = _query(client, evidence_id=first_eid).json()["decisions"]
    assert len(rows) == 1
    assert {row["evidence_id"] for row in rows} == {first_eid}
    rows = _query(client, evidence_id=other_eid).json()["decisions"]
    assert {row["evidence_id"] for row in rows} == {other_eid}


def test_filter_by_policy_id(client):
    _, _, _, policy, _ = _one_decision(client)
    _one_decision(client, name="other")
    rows = _walk(client, policy_id=policy["policy_id"])
    assert rows and {row["policy_id"] for row in rows} == {policy["policy_id"]}


@pytest.mark.parametrize("status", ["allowed", "denied"])
def test_filter_by_status(client, status):
    _one_decision(client, rule={"claim": "m", "equals": "x"}, name="allow")
    _one_decision(
        client,
        claims={"m": "other"},
        rule={"claim": "m", "equals": "x"},
        name="deny",
    )
    rows = _walk(client, status=status)
    assert rows and {row["status"] for row in rows} == {status}


def test_filter_by_time_window_is_inclusive(client):
    _one_decision(client)
    decided_at = _query(client).json()["decisions"][0]["decided_at"]
    rows = _query(
        client, decided_after=decided_at, decided_before=decided_at
    ).json()["decisions"]
    assert len(rows) == 1
    past = (datetime.fromisoformat(decided_at) - timedelta(seconds=1)).isoformat()
    future = (datetime.fromisoformat(decided_at) + timedelta(seconds=1)).isoformat()
    assert _query(client, decided_before=past).json()["decisions"] == []
    assert _query(client, decided_after=future).json()["decisions"] == []


def test_decisions_are_scoped_to_tenant_and_workload(client):
    _one_decision(client)
    rows_a = _query(client).json()["decisions"]
    rows_b = _query(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    ).json()["decisions"]
    assert len(rows_a) == 1
    assert rows_b == []


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_decision_once_in_order(client, monkeypatch):
    count = 5
    for i in range(count):
        _one_decision(client, name=f"r{i}")
    rows = _walk(client, page_size=3, monkeypatch=monkeypatch)
    assert len(rows) == count
    keys = [(row["decided_at"], row["decision_id"]) for row in rows]
    assert keys == sorted(keys)
    assert len({row["decision_id"] for row in rows}) == count


def test_page_carries_cursor_and_complete_flag(client, monkeypatch):
    monkeypatch.setattr(app_module, "DECISION_PAGE_SIZE", 1)
    for i in range(2):
        _one_decision(client, name=f"r{i}")

    first = _query(client).json()
    assert len(first["decisions"]) == 1
    assert first["complete"] is False
    assert first["next_cursor"]

    second = _query(client, cursor=first["next_cursor"]).json()
    assert len(second["decisions"]) == 1
    assert second["complete"] is True
    assert second["next_cursor"] == ""
    first_keys = [(r["decided_at"], r["decision_id"]) for r in first["decisions"]]
    second_keys = [(r["decided_at"], r["decision_id"]) for r in second["decisions"]]
    assert first_keys < second_keys


def test_empty_string_cursor_equals_default(client):
    _one_decision(client)
    assert _query(client).content == _query(client, cursor="").content


def test_tampered_forged_or_cross_scope_cursor_returns_422(client, monkeypatch):
    monkeypatch.setattr(app_module, "DECISION_PAGE_SIZE", 1)
    _one_decision(client)
    _one_decision(client, name="second")
    cursor = _query(client).json()["next_cursor"]
    assert cursor

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _query(client, cursor=tampered).status_code == 422

    forged = b64url_encode(b'{"k":"compliance-decisions-v1"}' + b"0" * 32)
    assert _query(client, cursor=forged).status_code == 422

    assert _query(client, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert _query(client, workload=OTHER_WORKLOAD, cursor=cursor).status_code == 422


def test_cursor_cannot_cross_filters_or_kinds(client, monkeypatch):
    monkeypatch.setattr(app_module, "DECISION_PAGE_SIZE", 1)
    _, _, _, _, decided = _one_decision(client)
    _one_decision(client, name="second")
    cursor = _query(client).json()["next_cursor"]
    assert cursor

    assert _query(client, status="allowed", cursor=cursor).status_code == 422
    assert (
        _query(client, evidence_id=ZERO_UUID, cursor=cursor).status_code in (404, 422)
    )
    assert (
        _query(client, decision_id=ZERO_UUID, cursor=cursor).status_code in (404, 422)
    )
    assert (
        _query(
            client,
            decided_after="2000-01-01T00:00:00Z",
            cursor=cursor,
        ).status_code
        == 422
    )
    assert (
        _query(
            client,
            decided_before="2100-01-01T00:00:00Z",
            cursor=cursor,
        ).status_code
        == 422
    )

    # Cursors from every other HMAC family are never accepted.
    from proof_release.app import (
        _encode_audit_event_cursor,
        _encode_cursor,
        _encode_grant_audit_cursor,
        _encode_proof_event_cursor,
        _encode_revocation_cursor,
        _encode_rewrap_job_event_cursor,
    )

    foreign_rewrap = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _query(client, cursor=foreign_rewrap).status_code == 422
    foreign_grant = _encode_grant_audit_cursor(
        TENANT, WORKLOAD, ZERO_UUID,
        grant_id="", decision_id="", data_id="", status="",
        issued_after="", issued_before="",
    )
    assert _query(client, cursor=foreign_grant).status_code == 422
    foreign_audit = _encode_audit_event_cursor(
        TENANT, WORKLOAD, "2026-01-01T00:00:00+00:00", ZERO_UUID,
        event_id="", event_type="", status="",
        occurred_after="", occurred_before="",
    )
    assert _query(client, cursor=foreign_audit).status_code == 422
    foreign_proof = _encode_proof_event_cursor(
        TENANT, WORKLOAD, "2026-01-01T00:00:00+00:00", ZERO_UUID,
        evidence_id="", event_type="", status="",
        occurred_after="", occurred_before="", snapshot_seq=1,
    )
    assert _query(client, cursor=foreign_proof).status_code == 422
    foreign_revocation = _encode_revocation_cursor(
        TENANT, WORKLOAD, ZERO_UUID,
        "2026-01-01T00:00:00+00:00", ZERO_UUID,
        revocation_id="", certificate_fingerprint="",
        effective_after="", effective_before="",
        snapshot_at="2026-01-01T00:00:00+00:00", snapshot_id=ZERO_UUID,
    )
    assert _query(client, cursor=foreign_revocation).status_code == 422
    foreign_job_event = _encode_rewrap_job_event_cursor(
        TENANT, WORKLOAD, ZERO_UUID, 1, snapshot_seq=1
    )
    assert _query(client, cursor=foreign_job_event).status_code == 422


def test_decision_cursor_is_not_accepted_by_other_listings(client):
    from proof_release.app import _encode_decision_cursor

    _one_decision(client)
    decision_cursor = _encode_decision_cursor(
        TENANT, WORKLOAD, "2026-01-01T00:00:00+00:00", ZERO_UUID,
        decision_id="", evidence_id="", policy_id="", status="",
        decided_after="", decided_before="", snapshot_seq=1,
    )
    response = client.get(
        "/v1/compliance/proof-events",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "cursor": decision_cursor,
        },
    )
    assert response.status_code == 422


# --- fixed snapshot semantics ----------------------------------------------


def test_first_query_fixes_snapshot_excluding_later_commits(client, monkeypatch):
    monkeypatch.setattr(app_module, "DECISION_PAGE_SIZE", 2)
    for i in range(3):
        _one_decision(client, name=f"before-{i}")

    first = _query(client).json()
    assert len(first["decisions"]) == 2
    cursor = first["next_cursor"]

    # A fourth decision commits after the snapshot was fixed.
    _, _, later_eid, _, later = _one_decision(client, name="after")

    second = _query(client, cursor=cursor).json()
    assert len(second["decisions"]) == 1
    assert second["complete"] is True
    assert second["next_cursor"] == ""
    assert second["decisions"][0]["decision_id"] != later["decision_id"]
    assert later_eid not in {
        row["evidence_id"] for row in first["decisions"] + second["decisions"]
    }

    # Replaying the cursor returns the identical page.
    assert _query(client, cursor=cursor).json() == second

    # A fresh cursor-less first query sees the whole committed set.
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert len(fresh) == 4


def test_replay_first_page_is_byte_identical(client):
    _one_decision(client)
    first = _query(client)
    second = _query(client)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_snapshot_stays_empty_until_fresh_first_query(client):
    empty = _query(client).json()
    assert empty == {"decisions": [], "next_cursor": "", "complete": True}
    _one_decision(client)
    fresh = _query(client).json()
    assert len(fresh["decisions"]) == 1
    assert fresh["complete"] is True


def test_filtered_snapshot_excludes_later_in_filter_commits(client, monkeypatch):
    monkeypatch.setattr(app_module, "DECISION_PAGE_SIZE", 1)
    _one_decision(client, rule={"claim": "m", "equals": "x"}, name="a1")
    _one_decision(client, rule={"claim": "m", "equals": "x"}, name="a2")
    _one_decision(
        client,
        claims={"m": "no"},
        rule={"claim": "m", "equals": "x"},
        name="d1",
    )

    first = _query(client, status="allowed").json()
    assert len(first["decisions"]) == 1
    assert first["complete"] is False

    # A third allowed decision commits after the snapshot.
    _one_decision(client, name="a3")

    second = _query(
        client, status="allowed", cursor=first["next_cursor"]
    ).json()
    assert len(second["decisions"]) == 1
    assert second["complete"] is True
    assert second["next_cursor"] == ""

    fresh_rows = _walk(client, page_size=1, monkeypatch=monkeypatch, status="allowed")
    assert len(fresh_rows) == 3


def test_fixed_snapshot_excludes_later_commit_with_older_business_time(
    app, client, monkeypatch
):
    # A writer that stamps its business time before committing can land a
    # row *after* the first query yet carry an older decided_at. Snapshot
    # membership is bounded by commit order, so such a row must never enter
    # the already-fixed snapshot even though it sorts inside its
    # business-time window.
    monkeypatch.setattr(app_module, "DECISION_PAGE_SIZE", 2)
    _, _, first_eid, _, first = _one_decision(client, name="a")
    _, _, second_eid, _, second = _one_decision(client, name="b")
    _, _, third_eid, _, third = _one_decision(client, name="c")

    paged = _query(client).json()
    assert len(paged["decisions"]) == 2
    assert paged["complete"] is False
    cursor = paged["next_cursor"]

    t0 = datetime.fromisoformat(paged["decisions"][0]["decided_at"])
    t1 = datetime.fromisoformat(paged["decisions"][1]["decided_at"])
    backdated = t0 + (t1 - t0) / 2

    # Commits strictly after the first query, with a business time that
    # sorts between the two rows already returned. It takes the next
    # business commit boundary for the scope (sequence 4).
    late_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    with app.state.session_factory() as session:
        from proof_release.app import _next_decision_commit_seq

        late_seq = _next_decision_commit_seq(session, TENANT, WORKLOAD)
        assert late_seq == 4
        session.add(
            Decision(
                decision_id=late_id,
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                evidence_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                policy_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                policy_version=9,
                status="allowed",
                decided_at=backdated,
                commit_seq=late_seq,
            )
        )
        session.commit()

    # The fixed snapshot's next page holds only the original third
    # decision, already complete — the late, backdated commit is excluded.
    next_page = _query(client, cursor=cursor).json()
    assert len(next_page["decisions"]) == 1
    assert next_page["complete"] is True
    assert next_page["next_cursor"] == ""
    assert next_page["decisions"][0]["decision_id"] == third["decision_id"]

    # Replaying the cursor is byte-identical.
    assert _query(client, cursor=cursor).json() == next_page

    # A fresh snapshot observes the late commit exactly once, in stable
    # business-time order (between the first two business times).
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert [row["decision_id"] for row in fresh] == [
        first["decision_id"],
        late_id,
        second["decision_id"],
        third["decision_id"],
    ]
    keys = [(row["decided_at"], row["decision_id"]) for row in fresh]
    assert keys == sorted(keys)


# --- read-only behaviour ---------------------------------------------------


def test_query_writes_no_state(app, client):
    _one_decision(client)
    for _ in range(3):
        assert _query(client).status_code == 200
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 1
        counter = session.get(DecisionCommitCounter, (TENANT, WORKLOAD))
        assert counter is not None and counter.last_seq == 1


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_page(app, client):
    from sqlalchemy import text

    _one_decision(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE decisions"))
    assert _query(client).status_code == 500


# --- persistence -----------------------------------------------------------


def test_decisions_available_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    _, _, evidence_id, policy, decided = _one_decision(client1)
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    data = _query(client2).json()
    assert len(data["decisions"]) == 1
    row = data["decisions"][0]
    assert row["decision_id"] == decided["decision_id"]
    assert row["evidence_id"] == evidence_id
    assert row["policy_id"] == policy["policy_id"]
    assert row["policy_version"] == policy["version"]
    assert row["status"] == decided["status"]
    assert data["complete"] is True
    second.state.engine.dispose()


# --- per-scope business commit sequence ------------------------------------


def test_commit_sequence_is_gap_free_in_commit_order(app, client):
    _one_decision(client, name="a")
    _one_decision(client, name="b")
    with app.state.session_factory() as session:
        seqs = [
            row.commit_seq
            for row in session.query(Decision).order_by(Decision.commit_seq).all()
        ]
        assert seqs == [1, 2]


def test_commit_sequence_counter_is_per_scope(app, client):
    _one_decision(client)
    created, evidence, eid = _submit_verified(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    )
    policy = _policy(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    )
    decided = _decide(
        client,
        created,
        evidence,
        eid,
        policy["policy_id"],
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
    )
    assert decided.status_code == 200
    with app.state.session_factory() as session:
        first = session.get(DecisionCommitCounter, (TENANT, WORKLOAD))
        assert first is not None and first.last_seq == 1
        other = session.get(DecisionCommitCounter, (OTHER_TENANT, OTHER_WORKLOAD))
        assert other is not None and other.last_seq == 1
        assert first.last_seq == 1  # untouched by the other scope


def test_commit_sequence_continues_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/seq_restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    _one_decision(client1)
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    _one_decision(client2)
    with second.state.session_factory() as session:
        seqs = sorted(row.commit_seq for row in session.query(Decision).all())
        assert seqs == [1, 2]
    second.state.engine.dispose()


# --- legacy SQLite migration ----------------------------------------------


def test_legacy_sqlite_database_backfills_scope_sequence_and_indexes(
    tmp_path, monkeypatch
):
    # A database written by a deployment without decision scoping or
    # commit_seq: open it with the current build and confirm the scope is
    # copied from evidence, a rowid-order (commit-order) per-scope sequence
    # is backfilled, the per-scope counter is seeded and both indexes
    # exist.
    from sqlalchemy import create_engine, text

    from proof_release.db import Base

    url = f"sqlite:///{tmp_path}/legacy.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX ix_decisions_scope_commit_seq"))
        conn.execute(text("DROP INDEX ix_decisions_scope_decided"))
        conn.execute(text("DROP INDEX ix_decisions_tenant_id"))
        conn.execute(text("ALTER TABLE decisions DROP COLUMN commit_seq"))
        conn.execute(text("ALTER TABLE decisions DROP COLUMN tenant_id"))
        conn.execute(text("ALTER TABLE decisions DROP COLUMN workload_id"))
        conn.execute(text("DROP TABLE decision_commit_counters"))
        # Evidence rows supply the scope the backfill copies from. Decided
        # timestamps are deliberately inverted relative to insert order.
        conn.execute(
            text(
                """
                INSERT INTO evidence
                (evidence_id, challenge_id, tenant_id, workload_id,
                 evidence_format, status, received_at, evidence_sha256)
                VALUES
                ('11111111-1111-4111-8111-111111111111','c1','tA','w1',
                 'attested-nonce-json','verified','2026-01-01T00:00:00+00:00',
                 'aa'),
                ('22222222-2222-4222-8222-222222222222','c2','tA','w1',
                 'attested-nonce-json','verified','2026-01-01T00:00:00+00:00',
                 'bb'),
                ('33333333-3333-4333-8333-333333333333','c3','tB','w9',
                 'attested-nonce-json','verified','2026-01-01T00:00:00+00:00',
                 'cc')
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO decisions
                (decision_id, evidence_id, policy_id, policy_version, status,
                 decided_at)
                VALUES
                ('44444444-4444-4444-8444-444444444444',
                 '11111111-1111-4111-8111-111111111111',
                 '77777777-7777-4777-8777-777777777777', 1, 'allowed',
                 '2026-01-01T00:00:03+00:00'),
                ('55555555-5555-4555-8555-555555555555',
                 '22222222-2222-4222-8222-222222222222',
                 '77777777-7777-4777-8777-777777777778', 1, 'denied',
                 '2026-01-01T00:00:01+00:00'),
                ('66666666-6666-4666-8666-666666666666',
                 '33333333-3333-4333-8333-333333333333',
                 '77777777-7777-4777-8777-777777777779', 1, 'allowed',
                 '2026-01-01T00:00:02+00:00')
                """
            )
        )
    engine.dispose()

    application = create_app(url)
    try:
        with application.state.session_factory() as session:
            backfilled = {
                decision_id: (tenant, workload, seq)
                for decision_id, tenant, workload, seq in session.execute(
                    text(
                        "SELECT decision_id, tenant_id, workload_id, commit_seq "
                        "FROM decisions ORDER BY decision_id"
                    )
                ).fetchall()
            }
            # Insert/commit order within tA is preserved even though the
            # second row's business time is older; the other scope starts 1.
            assert backfilled["44444444-4444-4444-8444-444444444444"] == (
                "tA",
                "w1",
                1,
            )
            assert backfilled["55555555-5555-4555-8555-555555555555"] == (
                "tA",
                "w1",
                2,
            )
            assert backfilled["66666666-6666-4666-8666-666666666666"] == (
                "tB",
                "w9",
                1,
            )

            counters = dict(
                session.execute(
                    text(
                        "SELECT tenant_id || '/' || workload_id, last_seq "
                        "FROM decision_commit_counters"
                    )
                ).fetchall()
            )
            assert counters == {"tA/w1": 2, "tB/w9": 1}

            for index_name in (
                "ix_decisions_scope_commit_seq",
                "ix_decisions_scope_decided",
            ):
                indexed = session.execute(
                    text(
                        "SELECT 1 FROM sqlite_master WHERE type = 'index' "
                        "AND name = :name"
                    ),
                    {"name": index_name},
                ).fetchall()
                assert indexed

        # The legacy rows are listed in business-time order, and a new
        # decision in tA/w1 continues the sequence at 3.
        client = TestClient(application)
        response = client.get(
            "/v1/compliance/decisions",
            params={"tenant_id": "tA", "workload_id": "w1"},
        )
        assert response.status_code == 200
        statuses = [
            (row["decision_id"], row["status"])
            for row in response.json()["decisions"]
        ]
        assert statuses == [
            ("55555555-5555-4555-8555-555555555555", "denied"),
            ("44444444-4444-4444-8444-444444444444", "allowed"),
        ]
    finally:
        application.state.engine.dispose()
