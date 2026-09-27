"""Tests for GET /v1/compliance/decisions.

The read-only compliance decision query lists every recorded release
decision — one per (evidence, policy version) pair, unlike the proof
timeline which records only the first decision per evidence — in stable
(decided_at, decision_id) order with snapshot-fixed keyset pagination.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import proof_release.app as app_module
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
    return create_app(f"sqlite:///{tmp_path}/compliance_decisions.db")


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


def _submit(client, *, claims=None, tenant=TENANT, workload=WORKLOAD):
    claims = claims if claims is not None else {"m": "x"}
    created = client.post(
        "/v1/challenges", json={"tenant_id": tenant, "workload_id": workload}
    ).json()
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": claims,
            "mac": _mac_for(created["nonce"], claims, tenant=tenant, workload=workload),
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
    return created, evidence, submitted.json()["evidence_id"]


def _verify(client, created, evidence, evidence_id, *, tenant=TENANT, workload=WORKLOAD):
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
    assert verified.json()["status"] == "verified"
    return verified


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


def _full_proof(client, *, rule=None, claims=None, name="r"):
    """Receive, verify (accepted) and decide one proof; return its ids."""
    created, evidence, evidence_id = _submit(client, claims=claims)
    _verify(client, created, evidence, evidence_id)
    policy = _policy(client, rule, name=name)
    decided = _decide(client, created, evidence, evidence_id, policy["policy_id"])
    assert decided.status_code == 200, decided.text
    return created, evidence, evidence_id, policy, decided.json()


def _multi_version_decisions(client):
    """One evidence decided against two versions of the same policy name."""
    created, evidence, evidence_id = _submit(client)
    _verify(client, created, evidence, evidence_id)
    v1 = _policy(client, {"claim": "m", "equals": "x"}, name="multi")
    d1 = _decide(client, created, evidence, evidence_id, v1["policy_id"])
    assert d1.status_code == 200, d1.text
    v2 = _policy(client, {"claim": "m", "equals": "y"}, name="multi")
    d2 = _decide(client, created, evidence, evidence_id, v2["policy_id"])
    assert d2.status_code == 200, d2.text
    return evidence_id, v1, v2, d1.json(), d2.json()


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
        {"decision_id": "  " + ZERO_UUID},
        {"decision_id": "ABCDEF12-3456-7890-ABCD-EF1234567890"},
        {"evidence_id": ""},
        {"evidence_id": "abc123"},
        {"evidence_id": ZERO_UUID[:-1] + "Z"},
        {"policy_id": ""},
        {"policy_id": "not-a-uuid"},
        {"policy_id": "ABCDEF12-3456-7890-ABCD-EF1234567890"},
        {"status": ""},
        {"status": "   "},
        {"status": "ALLOWED"},
        {"status": "pending"},
        {"status": "accepted"},
        {"status": " allowed"},
        {"decided_after": "not-a-timestamp"},
        {"decided_after": "2026-01-01T00:00:00"},
        {"decided_before": "2026-01-01"},
        {"decided_after": ""},
        {"decided_before": "  "},
        {"decided_after": "2026-01-01T01:00:00+01:00"},
        {
            "decided_after": "2026-01-02T00:00:00Z",
            "decided_before": "2026-01-01T00:00:00Z",
        },
        {"cursor": "   "},
        {"cursor": "not base64!!"},
        {"cursor": "AAA="},
        {"bogus": "value"},
    ],
)
def test_query_rejects_invalid_parameters(client, params):
    assert _query(client, **params).status_code == 422


def test_repeated_parameter_is_422(client):
    response = client.get(
        "/v1/compliance/decisions"
        f"?tenant_id={TENANT}&tenant_id={TENANT}&workload_id={WORKLOAD}"
    )
    assert response.status_code == 422
    response = client.get(
        "/v1/compliance/decisions"
        f"?tenant_id={TENANT}&workload_id={WORKLOAD}&status=allowed&status=denied"
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b" ", b"{}", b"\n", b"null", b"x"])
def test_non_empty_body_is_422(client, body):
    response = client.request(
        "GET",
        "/v1/compliance/decisions",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=body,
    )
    assert response.status_code == 422


def test_zero_length_body_is_legal(client):
    response = client.request(
        "GET",
        "/v1/compliance/decisions",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"",
    )
    assert response.status_code == 200


def test_invalid_parameters_write_no_state(app, client):
    _full_proof(client)
    assert _query(client, status="bogus").status_code == 422
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 1
        assert session.query(DecisionCommitCounter).count() == 1


# --- explicit identifier existence -----------------------------------------


def test_unknown_identifiers_return_404(client):
    _full_proof(client)
    assert _query(client, decision_id=ZERO_UUID).status_code == 404
    assert _query(client, evidence_id=ZERO_UUID).status_code == 404
    assert _query(client, policy_id=ZERO_UUID).status_code == 404


def test_cross_scope_identifiers_return_404(client):
    _, _, evidence_id, policy, decided = _full_proof(client)
    # The same identifiers under another tenant or workload are an
    # indistinguishable 404 — existence is never revealed.
    for tenant, workload in ((OTHER_TENANT, WORKLOAD), (TENANT, OTHER_WORKLOAD)):
        assert (
            _query(client, tenant=tenant, workload=workload,
                   decision_id=decided["decision_id"]).status_code == 404
        )
        assert (
            _query(client, tenant=tenant, workload=workload,
                   evidence_id=evidence_id).status_code == 404
        )
        assert (
            _query(client, tenant=tenant, workload=workload,
                   policy_id=policy["policy_id"]).status_code == 404
        )


def test_empty_scope_returns_empty_completed_page(client):
    response = _query(client)
    assert response.status_code == 200
    assert response.json() == {
        "decisions": [],
        "next_cursor": "",
        "complete": True,
    }


# --- response shape ----------------------------------------------------------


def test_response_is_compact_json_with_single_trailing_newline(client):
    _full_proof(client)
    response = _query(client)
    assert response.status_code == 200
    raw = response.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b"\n" not in raw[:-1]
    assert b": " not in raw and b", " not in raw
    assert list(response.json()) == ["decisions", "next_cursor", "complete"]


def test_decision_entries_have_exact_shape_and_order(client):
    _, _, evidence_id, policy, decided = _full_proof(client)
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
    assert row["decision_id"] == decided["decision_id"]
    assert row["evidence_id"] == evidence_id
    assert row["policy_id"] == policy["policy_id"]
    assert row["policy_version"] == policy["version"] == 1
    assert isinstance(row["policy_version"], int)
    assert not isinstance(row["policy_version"], bool)
    assert row["status"] == "allowed"
    assert isinstance(row["decided_at"], str)
    # The listing echoes the stored business time exactly.
    assert row["decided_at"] == decided["decided_at"]


def test_no_floats_or_non_finite_values(client):
    _full_proof(client)
    raw = _query(client).content.decode()

    def _check(value):
        if isinstance(value, dict):
            for item in value.values():
                _check(item)
        elif isinstance(value, list):
            for item in value:
                _check(item)
        else:
            assert not isinstance(value, float)

    _check(json.loads(raw))
    assert "NaN" not in raw and "Infinity" not in raw


# --- listing semantics -------------------------------------------------------


def test_every_policy_version_decision_is_listed(client):
    # The proof timeline records only the first decision per evidence;
    # this query covers every policy-version decision of that evidence.
    evidence_id, v1, v2, d1, d2 = _multi_version_decisions(client)
    rows = _query(client).json()["decisions"]
    assert [row["decision_id"] for row in rows] == [
        d1["decision_id"],
        d2["decision_id"],
    ]
    assert [row["policy_version"] for row in rows] == [1, 2]
    assert [row["status"] for row in rows] == ["allowed", "denied"]
    assert all(row["evidence_id"] == evidence_id for row in rows)


def test_filter_by_decision_id(client):
    _, _, _, _, d2 = _multi_version_decisions(client)
    rows = _query(client, decision_id=d2["decision_id"]).json()["decisions"]
    assert [row["decision_id"] for row in rows] == [d2["decision_id"]]
    assert rows[0]["policy_version"] == 2


def test_filter_by_evidence_id(client):
    evidence_id, _, _, d1, d2 = _multi_version_decisions(client)
    _full_proof(client, name="other")
    rows = _query(client, evidence_id=evidence_id).json()["decisions"]
    assert {row["decision_id"] for row in rows} == {
        d1["decision_id"],
        d2["decision_id"],
    }


def test_filter_by_policy_id(client):
    _, v1, v2, d1, d2 = _multi_version_decisions(client)
    rows = _query(client, policy_id=v1["policy_id"]).json()["decisions"]
    assert [row["decision_id"] for row in rows] == [d1["decision_id"]]
    rows = _query(client, policy_id=v2["policy_id"]).json()["decisions"]
    assert [row["decision_id"] for row in rows] == [d2["decision_id"]]


def test_filter_by_status(client):
    _multi_version_decisions(client)
    allowed = _query(client, status="allowed").json()["decisions"]
    denied = _query(client, status="denied").json()["decisions"]
    assert len(allowed) == 1 and allowed[0]["status"] == "allowed"
    assert len(denied) == 1 and denied[0]["status"] == "denied"


def test_filter_by_time_window_is_inclusive(client):
    _full_proof(client, name="p1")
    _full_proof(client, name="p2")
    rows = _query(client).json()["decisions"]
    first_at = rows[0]["decided_at"]
    last_at = rows[-1]["decided_at"]
    windowed = _query(
        client, decided_after=first_at, decided_before=last_at
    ).json()["decisions"]
    assert [row["decision_id"] for row in windowed] == [
        row["decision_id"] for row in rows
    ]
    # A single-instant window is valid and inclusive on both ends.
    single = _query(
        client, decided_after=first_at, decided_before=first_at
    ).json()["decisions"]
    assert [row["decision_id"] for row in single] == [rows[0]["decision_id"]]
    empty = _query(client, decided_after="2100-01-01T00:00:00Z").json()
    assert empty == {"decisions": [], "next_cursor": "", "complete": True}


def test_decisions_are_scoped_to_tenant_and_workload(client):
    _full_proof(client, name="mine")
    created, evidence, evidence_id = _submit(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    )
    _verify(client, created, evidence, evidence_id,
            tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    policy = _policy(client, name="theirs",
                     tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    decided = _decide(client, created, evidence, evidence_id,
                      policy["policy_id"],
                      tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    assert decided.status_code == 200

    mine = _query(client).json()["decisions"]
    assert len(mine) == 1
    theirs = _query(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()
    assert len(theirs["decisions"]) == 1
    assert (
        mine[0]["decision_id"] != theirs["decisions"][0]["decision_id"]
    )


# --- pagination --------------------------------------------------------------


def test_pagination_walks_every_decision_once_in_order(client, monkeypatch):
    for index in range(5):
        _full_proof(client, name=f"p{index}")
    rows = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert len(rows) == 5
    keys = [(row["decided_at"], row["decision_id"]) for row in rows]
    assert keys == sorted(keys)
    assert len({row["decision_id"] for row in rows}) == 5


def test_page_carries_cursor_and_complete_flag(client, monkeypatch):
    monkeypatch.setattr(app_module, "DECISION_PAGE_SIZE", 2)
    for index in range(3):
        _full_proof(client, name=f"p{index}")
    first = _query(client).json()
    assert len(first["decisions"]) == 2
    assert first["complete"] is False
    assert first["next_cursor"]
    second = _query(client, cursor=first["next_cursor"]).json()
    assert len(second["decisions"]) == 1
    assert second["complete"] is True
    assert second["next_cursor"] == ""


def test_empty_string_cursor_equals_default(client):
    _full_proof(client)
    assert _query(client).content == _query(client, cursor="").content


def test_replay_first_page_is_byte_identical(client):
    _full_proof(client)
    first = _query(client)
    assert _query(client).content == first.content


# --- cursor security ---------------------------------------------------------


def test_tampered_forged_or_cross_scope_cursor_returns_422(client, monkeypatch):
    monkeypatch.setattr(app_module, "DECISION_PAGE_SIZE", 1)
    _full_proof(client, name="p1")
    _full_proof(client, name="p2")
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
    _full_proof(client, name="p1")
    _full_proof(client, name="p2")
    cursor = _query(client).json()["next_cursor"]

    assert _query(client, status="allowed", cursor=cursor).status_code == 422
    assert _query(client, decision_id=ZERO_UUID, cursor=cursor).status_code in (
        404,
        422,
    )
    assert _query(client, evidence_id=ZERO_UUID, cursor=cursor).status_code in (
        404,
        422,
    )
    assert _query(client, policy_id=ZERO_UUID, cursor=cursor).status_code in (
        404,
        422,
    )
    assert (
        _query(client, decided_after="2000-01-01T00:00:00Z",
               cursor=cursor).status_code
        == 422
    )
    assert (
        _query(client, decided_before="2100-01-01T00:00:00Z",
               cursor=cursor).status_code
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
    foreign_proof = _encode_proof_event_cursor(
        TENANT, WORKLOAD, "2026-01-01T00:00:00+00:00", ZERO_UUID,
        evidence_id="", event_type="", status="",
        occurred_after="", occurred_before="", snapshot_seq=1,
    )
    assert _query(client, cursor=foreign_proof).status_code == 422


# --- fixed snapshot semantics ------------------------------------------------


def test_first_query_fixes_snapshot_excluding_later_commits(client, monkeypatch):
    monkeypatch.setattr(app_module, "DECISION_PAGE_SIZE", 1)
    _, _, _, _, d1 = _full_proof(client, name="p1")
    _full_proof(client, name="p2")

    first = _query(client).json()
    assert len(first["decisions"]) == 1
    assert first["decisions"][0]["decision_id"] == d1["decision_id"]
    cursor = first["next_cursor"]
    assert cursor

    # A third decision commits after the snapshot was fixed.
    _full_proof(client, name="p3")

    # The next page of the fixed snapshot contains only the second
    # decision — the newly committed one never enters the old snapshot.
    second = _query(client, cursor=cursor).json()
    assert len(second["decisions"]) == 1
    assert second["complete"] is True
    assert second["next_cursor"] == ""
    # Replaying the cursor is stable.
    assert _query(client, cursor=cursor).json() == second

    # A fresh cursor-less first query sees all three decisions.
    fresh = _walk(client, page_size=1, monkeypatch=monkeypatch)
    assert len(fresh) == 3
    assert fresh[0]["decision_id"] == d1["decision_id"]


def test_empty_snapshot_stays_empty_until_fresh_first_query(client):
    empty = _query(client).json()
    assert empty == {"decisions": [], "next_cursor": "", "complete": True}
    _full_proof(client)
    # The old (empty) snapshot had no cursor; a fresh first query is the
    # only way to observe the new decision.
    fresh = _query(client).json()
    assert len(fresh["decisions"]) == 1


def test_filtered_snapshot_excludes_later_in_filter_commits(client, monkeypatch):
    monkeypatch.setattr(app_module, "DECISION_PAGE_SIZE", 1)
    _full_proof(client, name="p1")
    _full_proof(client, name="p2")
    first = _query(client, status="allowed").json()
    assert len(first["decisions"]) == 1
    cursor = first["next_cursor"]
    assert cursor

    # A denied decision commits after the filtered snapshot was fixed;
    # an allowed one follows it.
    _full_proof(client, name="p3", rule={"claim": "m", "equals": "z"})
    _full_proof(client, name="p4")

    # The fixed snapshot's remaining page holds only the second allowed
    # decision; the later allowed commit is excluded.
    second = _query(client, status="allowed", cursor=cursor).json()
    assert len(second["decisions"]) == 1
    assert second["complete"] is True
    fresh = _walk(client, page_size=1, monkeypatch=monkeypatch, status="allowed")
    assert len(fresh) == 3


def test_fixed_snapshot_excludes_later_commit_with_older_business_time(
    app, client, monkeypatch
):
    # A decision that commits after the first query but carries an older
    # decided_at must not enter the fixed snapshot's later pages.
    monkeypatch.setattr(app_module, "DECISION_PAGE_SIZE", 1)
    _full_proof(client, name="p1")
    _full_proof(client, name="p2")

    first = _query(client).json()
    assert len(first["decisions"]) == 1
    cursor = first["next_cursor"]

    t0 = datetime.fromisoformat(first["decisions"][0]["decided_at"])
    backdated = t0.replace(year=max(t0.year - 1, 1))
    late_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    with app.state.session_factory() as session:
        late_seq = app_module._next_decision_commit_seq(session, TENANT, WORKLOAD)
        assert late_seq == 3
        session.add(
            Decision(
                decision_id=late_id,
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                evidence_id=ZERO_UUID,
                policy_id=ZERO_UUID,
                policy_version=1,
                status="allowed",
                decided_at=backdated,
                commit_seq=late_seq,
            )
        )
        session.commit()

    # The fixed snapshot's remaining pages exclude the late commit even
    # though its business time sorts before the already-returned page.
    seen = {first["decisions"][0]["decision_id"]}
    token = cursor
    page = None
    for _ in range(10):
        page = _query(client, cursor=token).json()
        seen.update(row["decision_id"] for row in page["decisions"])
        if page["complete"]:
            break
        token = page["next_cursor"]
    assert page["complete"] is True
    assert len(seen) == 2
    assert late_id not in seen

    # A fresh first query observes it exactly once, in business-time order.
    monkeypatch.setattr(app_module, "DECISION_PAGE_SIZE", 100)
    fresh = _query(client).json()
    assert fresh["complete"] is True and len(fresh["decisions"]) == 3
    ids = [row["decision_id"] for row in fresh["decisions"]]
    assert ids.count(late_id) == 1
    keys = [(row["decided_at"], row["decision_id"]) for row in fresh["decisions"]]
    assert keys == sorted(keys)


# --- read-only behaviour -----------------------------------------------------


def test_query_writes_no_state(app, client):
    _full_proof(client)
    for _ in range(3):
        assert _query(client).status_code == 200
    with app.state.session_factory() as session:
        assert session.query(Decision).count() == 1
        counters = session.query(DecisionCommitCounter).all()
        assert len(counters) == 1 and counters[0].last_seq == 1


def test_storage_failure_returns_500_and_no_page(app, client):
    from sqlalchemy import text

    _full_proof(client)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE decisions"))
    assert _query(client).status_code == 500


# --- persistence and commit sequence -----------------------------------------


def test_decisions_available_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    _, _, evidence_id, policy, decided = _full_proof(client1)
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    data = _query(client2).json()
    assert data["complete"] is True
    assert len(data["decisions"]) == 1
    row = data["decisions"][0]
    assert row["decision_id"] == decided["decision_id"]
    assert row["evidence_id"] == evidence_id
    assert row["policy_id"] == policy["policy_id"]
    assert row["policy_version"] == 1
    assert row["decided_at"] == decided["decided_at"]
    second.state.engine.dispose()


def test_commit_sequence_is_gap_free_in_commit_order(app, client):
    _multi_version_decisions(client)
    _full_proof(client, name="p3")
    with app.state.session_factory() as session:
        seqs = [
            row.commit_seq
            for row in session.query(Decision).order_by(Decision.commit_seq)
        ]
        assert seqs == [1, 2, 3]
        counter = session.query(DecisionCommitCounter).one()
        assert counter.last_seq == 3


def test_commit_sequence_counter_is_per_scope(app, client):
    _full_proof(client, name="mine")
    created, evidence, evidence_id = _submit(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    )
    _verify(client, created, evidence, evidence_id,
            tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    policy = _policy(client, name="theirs",
                     tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    decided = _decide(client, created, evidence, evidence_id,
                      policy["policy_id"],
                      tenant=OTHER_TENANT, workload=OTHER_WORKLOAD)
    assert decided.status_code == 200
    with app.state.session_factory() as session:
        counters = {
            (row.tenant_id, row.workload_id): row.last_seq
            for row in session.query(DecisionCommitCounter)
        }
        assert counters == {
            (TENANT, WORKLOAD): 1,
            (OTHER_TENANT, OTHER_WORKLOAD): 1,
        }
        seqs = sorted(row.commit_seq for row in session.query(Decision))
        assert seqs == [1, 1]


def test_legacy_sqlite_database_backfills_scope_and_commit_sequence(
    tmp_path, monkeypatch
):
    # A database written by a deployment without the decision scope and
    # commit_seq columns: open it with the current build and confirm the
    # scope backfill (from the referenced evidence), a rowid-order
    # (commit-order) per-scope sequence backfill, a seeded per-scope
    # counter and the indexes.
    from sqlalchemy import create_engine, text

    from proof_release.db import Base

    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
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
        conn.execute(
            text(
                """
                INSERT INTO evidence
                (evidence_id, challenge_id, tenant_id, workload_id,
                 evidence_format, status, received_at, evidence_sha256)
                VALUES
                ('11111111-1111-4111-8111-111111111111','c1','tA','w1',
                 'f','verified','2026-01-01T00:00:00+00:00','d1'),
                ('33333333-3333-4333-8333-333333333333','c3','tB','w9',
                 'f','verified','2026-01-01T00:00:00+00:00','d3')
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO decisions
                (decision_id, evidence_id, policy_id, policy_version,
                 status, decided_at)
                VALUES
                ('11111111-1111-4111-8111-111111111111',
                 '11111111-1111-4111-8111-111111111111',
                 '11111111-1111-4111-8111-111111111111',
                 1,'allowed','2026-01-01T00:00:03+00:00'),
                ('22222222-2222-4222-8222-222222222222',
                 '11111111-1111-4111-8111-111111111111',
                 '22222222-2222-4222-8222-222222222222',
                 2,'denied','2026-01-01T00:00:01+00:00'),
                ('33333333-3333-4333-8333-333333333333',
                 '33333333-3333-4333-8333-333333333333',
                 '33333333-3333-4333-8333-333333333333',
                 1,'allowed','2026-01-01T00:00:02+00:00')
                """
            )
        )
    engine.dispose()

    application = create_app(url)
    try:
        with application.state.session_factory() as session:
            rows = session.execute(
                text(
                    "SELECT decision_id, tenant_id, workload_id, commit_seq "
                    "FROM decisions ORDER BY rowid"
                )
            ).fetchall()
            # Scope backfilled from the referenced evidence; commit order
            # within tA is preserved even though the second row's business
            # time is older; the other scope restarts at 1.
            assert rows == [
                ("11111111-1111-4111-8111-111111111111", "tA", "w1", 1),
                ("22222222-2222-4222-8222-222222222222", "tA", "w1", 2),
                ("33333333-3333-4333-8333-333333333333", "tB", "w9", 1),
            ]
            counters = dict(
                session.execute(
                    text(
                        "SELECT tenant_id || '/' || workload_id, last_seq "
                        "FROM decision_commit_counters"
                    )
                ).fetchall()
            )
            assert counters == {"tA/w1": 2, "tB/w9": 1}

        client = TestClient(application)
        response = client.get(
            "/v1/compliance/decisions",
            params={"tenant_id": "tA", "workload_id": "w1"},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["complete"] is True
        # Listed in business-time order, not commit order.
        assert [row["decision_id"] for row in data["decisions"]] == [
            "22222222-2222-4222-8222-222222222222",
            "11111111-1111-4111-8111-111111111111",
        ]
        assert [row["policy_version"] for row in data["decisions"]] == [2, 1]
    finally:
        application.state.engine.dispose()
