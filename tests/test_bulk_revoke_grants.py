"""Tests for POST /v1/release-grants/bulk-revoke.

Contract under test:

* the request JSON carries exactly ``tenant_id``, ``workload_id`` and
  ``grants``, the latter a list of one to one hundred objects each
  carrying exactly one unique canonical ``grant_id`` (UUID) and one
  ``capability`` in the existing unpadded-base64url format; any other
  shape is a single indistinguishable 422 with detail
  ``invalid bulk revoke request``, raised before any grant read and
  without reserving budget;
* an admitted request draws exactly one slot of the shared per-scope
  five-per-minute grant budget, regardless of batch size;
* grants are judged in ascending ``grant_id`` order, within a grant in
  the existing precedence unknown/cross-scope (404), capability mismatch
  (401), consumed (409), already revoked (409), pending-expired (410);
  the first such result fails the whole batch and leaves every grant,
  timeline, audit and idempotency record exactly as found;
* a batch that passes settles every grant to revoked in one transaction
  under one shared UTC RFC3339 ``revoked_at``, with one reason-``revoked``
  timeline event and one revoked audit per grant;
* the 200 is compact JSON with fields in the fixed order
  tenant_id, workload_id, revoked_count, revoked_at, revoked_grants and
  each entry ordered by grant_id with grant_id, decision_id, data_id
  only;
* an optional single Idempotency-Key stores the exact first 200 for a
  byte-for-byte replay (even after later state change), a same-key
  different-content request is 409 ``idempotency conflict``, and only
  success occupies the key;
* a persistence failure is a 500 ``grant unavailable`` with the whole
  batch rolled back.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    RateLimitCounter,
    ReleaseGrant,
    ReleaseGrantBulkRevokeIdempotencyRecord,
    ReleaseGrantEvent,
)
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"
MASTER_KEY = b64url_encode(b"0123456789abcdef0123456789abcdef")
IDEMPOTENCY_HEADER = "Idempotency-Key"
ZERO_UUID = "00000000-0000-0000-0000-000000000000"
PATH = "/v1/release-grants/bulk-revoke"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/bulk-revoke.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


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


def _evidence(nonce: str) -> str:
    claims = {"m": "x"}
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims)}
    )


def _allowed_decision(client):
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    evidence = _evidence(created["nonce"])
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
    evidence_id = submitted.json()["evidence_id"]
    verified = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.status_code == 200
    policy = client.post(
        "/v1/policies",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "name": "release",
            "rule": {"claim": "m", "equals": "x"},
        },
    ).json()
    decided = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy["policy_id"],
        },
    )
    assert decided.status_code == 200
    return decided.json()["decision_id"]


def _grant(client, decision_id, data_id, **fields):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision_id,
        "data_id": data_id,
    }
    body.update(fields)
    response = client.post("/v1/release-grants", json=body)
    assert response.status_code == 201
    return response.json()


def _grants(client, decision_id, count, *, prefix="data"):
    created = [
        _grant(client, decision_id, f"{prefix}-{i}") for i in range(count)
    ]
    return sorted(created, key=lambda g: g["grant_id"])


def _entry(grant, capability=None):
    return {
        "grant_id": grant["grant_id"],
        "capability": grant["capability"] if capability is None else capability,
    }


def _bulk(client, grants_or_entries, *, tenant=TENANT, workload=WORKLOAD,
          key=..., order=None):
    entries = [
        _entry(g) if isinstance(g, dict) and "capability" in g else g
        for g in grants_or_entries
    ]
    if order == "shuffle" and len(entries) > 1:
        entries = entries[::-1]
    body = {"tenant_id": tenant, "workload_id": workload, "grants": entries}
    headers = (
        None
        if key is ...
        else ({IDEMPOTENCY_HEADER: key} if key else None)
    )
    return client.post(PATH, json=body, headers=headers)


def _set_expired(app, grant_id):
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant_id)
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()


def _count(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


# ---------------------------------------------------------------------------
# Success shape and persistence
# ---------------------------------------------------------------------------


def test_bulk_revoke_three_pending_grants_returns_compact_200(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 3)
    # Submit deliberately out of order; the response must be sorted.
    response = _bulk(client, grants, order="shuffle")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert list(data) == [
        "tenant_id",
        "workload_id",
        "revoked_count",
        "revoked_at",
        "revoked_grants",
    ]
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["revoked_count"] == 3
    revoked_at = datetime.fromisoformat(data["revoked_at"])
    assert revoked_at.utcoffset() == timedelta(0)
    assert [g["grant_id"] for g in data["revoked_grants"]] == [
        g["grant_id"] for g in grants
    ]
    for entry, grant in zip(data["revoked_grants"], grants):
        assert list(entry) == ["grant_id", "decision_id", "data_id"]
        assert entry["grant_id"] == grant["grant_id"]
        assert entry["decision_id"] == decision_id
        assert entry["data_id"] == grant["data_id"]

    # Exact compact wire form: fixed field order, no whitespace.
    expected = json.dumps(
        {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "revoked_count": 3,
            "revoked_at": data["revoked_at"],
            "revoked_grants": [
                {
                    "grant_id": g["grant_id"],
                    "decision_id": decision_id,
                    "data_id": g["data_id"],
                }
                for g in grants
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    assert response.content == expected
    assert b" " not in response.content
    # Never echoed: capability or anything beyond the three identifiers.
    for grant in grants:
        assert grant["capability"] not in response.text
    assert "payload" not in response.text
    assert "evidence" not in response.text
    assert "claims" not in response.text

    with app.state.session_factory() as session:
        rows = [session.get(ReleaseGrant, g["grant_id"]) for g in grants]
        assert all(row.status == "revoked" for row in rows)
        assert len({row.revoked_at for row in rows}) == 1
        assert all(row.consumed_at is None for row in rows)


def test_bulk_revoke_writes_one_event_and_one_audit_per_grant(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 4)
    response = _bulk(client, grants[::-1])
    assert response.status_code == 200
    shared_at = response.json()["revoked_at"]

    with app.state.session_factory() as session:
        for grant in grants:
            events = (
                session.query(ReleaseGrantEvent)
                .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
                .order_by(ReleaseGrantEvent.seq)
                .all()
            )
            assert [(e.seq, e.reason, e.new_status) for e in events] == [
                (1, "issued", "pending"),
                (2, "revoked", "revoked"),
            ]
            revoked_audits = (
                session.query(AuditEvent)
                .filter(
                    AuditEvent.grant_id == grant["grant_id"],
                    AuditEvent.status == "revoked",
                )
                .all()
            )
            assert len(revoked_audits) == 1
        # The batch leaves one contiguous audit chain run in the scope.
        audits = (
            session.query(AuditEvent)
            .order_by(AuditEvent.chain_seq)
            .all()
        )
        assert [a.chain_seq for a in audits] == list(range(1, len(audits) + 1))
        for earlier, later in zip(audits, audits[1:]):
            assert later.prev_hash == earlier.event_hash
        revoked_rows = session.query(ReleaseGrant).all()
        assert {row.status for row in revoked_rows} == {"revoked"}
        assert len({row.revoked_at for row in revoked_rows}) == 1
        assert (
            shared_at
            == revoked_rows[0].revoked_at.astimezone(timezone.utc).isoformat()
        )


def test_bulk_revoke_single_grant_batch(client, app):
    decision_id = _allowed_decision(client)
    grant = _grant(client, decision_id, "only")
    response = _bulk(client, [grant])
    assert response.status_code == 200
    assert response.json()["revoked_count"] == 1
    with app.state.session_factory() as session:
        assert session.get(ReleaseGrant, grant["grant_id"]).status == "revoked"


def test_bulk_revoke_normalizes_uppercase_grant_uuids(client, app):
    decision_id = _allowed_decision(client)
    grant = _grant(client, decision_id, "upper")
    entry = _entry(grant)
    entry["grant_id"] = grant["grant_id"].upper()
    response = client.post(
        PATH,
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "grants": [entry],
        },
    )
    assert response.status_code == 200
    assert response.json()["revoked_grants"][0]["grant_id"] == grant["grant_id"]


def test_bulk_revoke_boundary_of_one_hundred_grants(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 100)
    response = _bulk(client, grants)
    assert response.status_code == 200
    assert response.json()["revoked_count"] == 100
    with app.state.session_factory() as session:
        assert session.query(ReleaseGrant).filter(
            ReleaseGrant.status == "revoked"
        ).count() == 100


# ---------------------------------------------------------------------------
# 422: request shape — indistinguishable, no grant read, no budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"",
        b"[]",
        b'"x"',
        b"123",
        b"null",
        json.dumps({}).encode(),
        json.dumps(
            {"tenant_id": TENANT, "workload_id": WORKLOAD}
        ).encode(),
        json.dumps(
            {
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "grants": [],
                "extra": 1,
            }
        ).encode(),
        json.dumps(
            {"tenant_id": TENANT, "workload_id": WORKLOAD, "grants": None}
        ).encode(),
        json.dumps(
            {"tenant_id": TENANT, "workload_id": WORKLOAD, "grants": {}}
        ).encode(),
        json.dumps(
            {"tenant_id": "", "workload_id": WORKLOAD, "grants": []}
        ).encode(),
        json.dumps(
            {"tenant_id": "   ", "workload_id": WORKLOAD, "grants": []}
        ).encode(),
        json.dumps(
            {"tenant_id": 1, "workload_id": WORKLOAD, "grants": []}
        ).encode(),
        json.dumps(
            {"tenant_id": TENANT, "workload_id": True, "grants": []}
        ).encode(),
        json.dumps(
            {"tenant_id": TENANT, "workload_id": "", "grants": []}
        ).encode(),
    ],
)
def test_invalid_bulk_request_is_422_before_budget(client, app, body):
    for _ in range(7):
        response = client.post(
            PATH, content=body, headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "invalid bulk revoke request"
    assert _count(app, RateLimitCounter) == 0


def test_grants_list_empty_and_too_long_are_422(client, app):
    empty = json.dumps(
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "grants": []}
    ).encode()
    too_many = json.dumps(
        {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "grants": [
                {"grant_id": ZERO_UUID, "capability": "A" * 43}
                for _ in range(101)
            ],
        }
    ).encode()
    for body in (empty, too_many):
        response = client.post(
            PATH, content=body, headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "invalid bulk revoke request"
    assert _count(app, RateLimitCounter) == 0


@pytest.mark.parametrize(
    "entry",
    [
        [],
        "x",
        1,
        None,
        {},
        {"grant_id": ZERO_UUID},
        {"capability": "A" * 43},
        {"grant_id": ZERO_UUID, "capability": "A" * 43, "extra": 1},
        {"grant_id": "not-a-uuid", "capability": "A" * 43},
        {"grant_id": ZERO_UUID[:-1], "capability": "A" * 43},
        {"grant_id": 123, "capability": "A" * 43},
        {"grant_id": None, "capability": "A" * 43},
        {"grant_id": "  ", "capability": "A" * 43},
        {"grant_id": "gggggggg-gggg-gggg-gggg-gggggggggggg",
         "capability": "A" * 43},
        {"grant_id": ZERO_UUID, "capability": ""},
        {"grant_id": ZERO_UUID, "capability": "   "},
        {"grant_id": ZERO_UUID, "capability": "not base64!!!"},
        {"grant_id": ZERO_UUID, "capability": "with=padding"},
        {"grant_id": ZERO_UUID, "capability": 9},
        {"grant_id": ZERO_UUID, "capability": True},
    ],
)
def test_invalid_grant_entry_is_422(client, app, entry):
    body = json.dumps(
        {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "grants": [entry],
        }
    ).encode()
    response = client.post(
        PATH, content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid bulk revoke request"
    assert _count(app, RateLimitCounter) == 0


def test_duplicate_grant_ids_are_422_even_across_case(client, app):
    for ids in ([ZERO_UUID, ZERO_UUID], [ZERO_UUID, ZERO_UUID.upper()]):
        body = json.dumps(
            {
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "grants": [
                    {"grant_id": gid, "capability": "A" * 43} for gid in ids
                ],
            }
        ).encode()
        response = client.post(
            PATH, content=body, headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "invalid bulk revoke request"
    assert _count(app, RateLimitCounter) == 0


def test_422_does_not_read_or_change_grants(client, app):
    decision_id = _allowed_decision(client)
    grant = _grant(client, decision_id, "untouched")
    # A well-formed grant entry plus a malformed sibling: the request
    # never reaches storage.
    body = json.dumps(
        {
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "grants": [
                {"grant_id": grant["grant_id"], "capability": grant["capability"]},
                {"grant_id": "bad", "capability": "A" * 43},
            ],
        }
    ).encode()
    response = client.post(
        PATH, content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.revoked_at is None
        assert (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
            .count()
            == 1
        )


# ---------------------------------------------------------------------------
# Judgement order and per-grant error precedence
# ---------------------------------------------------------------------------


def test_unknown_grant_is_404_and_changes_nothing(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 2)
    response = _bulk(
        client,
        [
            {"grant_id": ZERO_UUID, "capability": "A" * 43},
            _entry(grants[0]),
        ],
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "grant not found"
    assert "A" * 43 not in response.text
    _assert_all_pending(app, grants)
    assert _count(app, ReleaseGrantBulkRevokeIdempotencyRecord) == 0


def test_cross_scope_grant_is_404(client, app):
    decision_id = _allowed_decision(client)
    grant = _grant(client, decision_id, "scoped")
    response = _bulk(client, [grant], tenant="tenant-b")
    assert response.status_code == 404
    assert response.json()["detail"] == "grant not found"
    response = _bulk(client, [grant], workload="workload-2")
    assert response.status_code == 404
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"
        assert row.revoked_at is None


def test_capability_mismatch_is_401_and_batch_stays_pending(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 3)
    wrong = _wrong(grants[1]["capability"])
    entries = [
        _entry(grants[0]),
        _entry(grants[1], capability=wrong),
        _entry(grants[2]),
    ]
    response = client.post(
        PATH,
        json={"tenant_id": TENANT, "workload_id": WORKLOAD, "grants": entries},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "invalid capability"
    _assert_all_pending(app, grants)
    # The correct capabilities still settle the batch afterwards.
    recovered = _bulk(client, grants)
    assert recovered.status_code == 200


def test_wrong_capability_on_expired_grant_is_still_401(client, app):
    decision_id = _allowed_decision(client)
    grant = _grant(client, decision_id, "expired-cap")
    _set_expired(app, grant["grant_id"])
    response = _bulk(
        client, [_entry(grant, capability=_wrong(grant["capability"]))]
    )
    assert response.status_code == 401
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant["grant_id"])
        assert row.status == "pending"


def test_consumed_grant_is_409_and_fails_whole_batch(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 3)
    consumed = client.post(
        f"/v1/release-grants/{grants[1]['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grants[1]["capability"],
        },
    )
    assert consumed.status_code == 200
    # The sorted-middle grant being consumed beats the later expired one.
    _set_expired(app, grants[2]["grant_id"])
    response = _bulk(client, grants)
    assert response.status_code == 409
    assert response.json()["detail"] == "grant already consumed"
    _assert_all_pending(app, [grants[0], grants[2]])
    with app.state.session_factory() as session:
        assert session.get(
            ReleaseGrant, grants[1]["grant_id"]
        ).status == "consumed"
    _assert_no_revoke_writes(app, [grants[0], grants[2]])


def test_already_revoked_grant_is_409_before_later_expiry(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 3)
    first = client.post(
        f"/v1/release-grants/{grants[1]['grant_id']}/revoke",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grants[1]["capability"],
        },
    )
    assert first.status_code == 200
    _set_expired(app, grants[2]["grant_id"])
    response = _bulk(client, grants)
    assert response.status_code == 409
    assert response.json()["detail"] == "grant already revoked"
    _assert_all_pending(app, [grants[0], grants[2]])
    _assert_no_revoke_writes(app, [grants[0], grants[2]])


def test_expired_pending_grant_is_410_and_changes_nothing(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 2)
    _set_expired(app, grants[0]["grant_id"])
    response = _bulk(client, grants)
    assert response.status_code == 410
    assert response.json()["detail"] == "grant expired"
    _assert_all_pending(app, grants)
    _assert_no_revoke_writes(app, grants)


def test_judgement_runs_in_grant_id_order(client, app):
    # Four grants sorted by id: wrong-capability on the second must be the
    # reported failure even though the third is consumed and the fourth
    # expired — and a fully valid first grant is settled by nothing.
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 4)
    wrong = _wrong(grants[1]["capability"])
    client.post(
        f"/v1/release-grants/{grants[2]['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grants[2]["capability"],
        },
    )
    _set_expired(app, grants[3]["grant_id"])
    entries = [
        _entry(grants[0]),
        _entry(grants[1], capability=wrong),
        _entry(grants[2]),
        _entry(grants[3]),
    ]
    response = client.post(
        PATH,
        json={"tenant_id": TENANT, "workload_id": WORKLOAD, "grants": entries},
    )
    assert response.status_code == 401
    _assert_all_pending(app, [grants[0], grants[1], grants[3]])


def test_failed_batch_writes_no_timeline_audit_or_idempotency_row(
    client, app
):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 3)
    response = _bulk(
        client,
        [
            _entry(grants[0]),
            {"grant_id": ZERO_UUID, "capability": "A" * 43},
            _entry(grants[2]),
        ],
        key="failed-key",
    )
    assert response.status_code == 404
    _assert_no_revoke_writes(app, grants)
    assert _count(app, ReleaseGrantBulkRevokeIdempotencyRecord) == 0
    # The key stayed free: the recovered batch succeeds under it.
    recovered = _bulk(client, grants, key="failed-key")
    assert recovered.status_code == 200


# ---------------------------------------------------------------------------
# Shared budget
# ---------------------------------------------------------------------------


def test_bulk_revoke_spends_exactly_one_budget_slot(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 5)
    response = _bulk(client, grants)
    assert response.status_code == 200
    with app.state.session_factory() as session:
        row = session.query(RateLimitCounter).one()
        assert row.count == 1


def test_four_replays_after_bulk_fill_the_budget(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 2)
    first = _bulk(client, grants, key="budget-key")
    assert first.status_code == 200
    for _ in range(4):
        assert _bulk(client, grants, key="budget-key").status_code == 200
    exhausted = _bulk(client, grants, key="budget-key")
    assert exhausted.status_code == 429
    assert "retry_after_seconds" in exhausted.json()


def test_business_errors_still_consume_the_slot(client, app):
    decision_id = _allowed_decision(client)
    grant = _grant(client, decision_id, "x")
    wrong = _wrong(grant["capability"])
    for _ in range(5):
        assert _bulk(client, [_entry(grant, capability=wrong)]).status_code == 401
    sixth = _bulk(client, [grant])
    assert sixth.status_code == 429
    with app.state.session_factory() as session:
        assert session.get(ReleaseGrant, grant["grant_id"]).status == "pending"


def test_429_changes_nothing(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 3)
    with app.state.session_factory() as session:
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        session.add(
            RateLimitCounter(
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                window_start=now,
                count=app_module.GRANT_BUDGET_PER_MINUTE,
            )
        )
        session.commit()
    response = _bulk(client, grants)
    assert response.status_code == 429
    _assert_all_pending(app, grants)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_keyed_replay_returns_first_200_verbatim_and_appends_nothing(
    client, app
):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 3)
    first = _bulk(client, grants[::-1], key="bulk-key")
    assert first.status_code == 200
    second = _bulk(client, grants, key="bulk-key")
    third = _bulk(client, grants[::-1], key="bulk-key")
    assert second.content == first.content
    assert third.content == first.content
    assert _count(app, ReleaseGrantBulkRevokeIdempotencyRecord) == 1
    with app.state.session_factory() as session:
        assert (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.reason == "revoked")
            .count()
            == 3
        )
        assert (
            session.query(AuditEvent)
            .filter(AuditEvent.status == "revoked")
            .count()
            == 3
        )


def test_replay_returns_original_200_after_grants_expired(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 2)
    first = _bulk(client, grants, key="replay-key")
    assert first.status_code == 200
    with app.state.session_factory() as session:
        for grant in grants:
            session.get(
                ReleaseGrant, grant["grant_id"]
            ).expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
    replay = _bulk(client, grants, key="replay-key")
    assert replay.status_code == 200
    assert replay.content == first.content


def test_same_key_different_batch_is_409_and_changes_nothing(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 3)
    first = _bulk(client, grants[:2], key="shared")
    assert first.status_code == 200
    conflict = _bulk(client, grants[1:], key="shared")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency conflict"
    # Grant 3 stays pending; an exact replay still yields the first body.
    with app.state.session_factory() as session:
        assert session.get(
            ReleaseGrant, grants[2]["grant_id"]
        ).status == "pending"
    assert _bulk(client, grants[:2], key="shared").content == first.content


def test_same_key_different_capability_is_409(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 2)
    assert _bulk(client, grants, key="shared").status_code == 200
    # The grants are now revoked; present the same batch with a wrong
    # capability: the stored fingerprint mismatch is judged before the
    # grant state, so it is the idempotency 409 rather than 401/409.
    entries = [
        _entry(g, capability=_wrong(g["capability"])) for g in grants
    ]
    conflict = client.post(
        PATH,
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "grants": entries,
        },
        headers={IDEMPOTENCY_HEADER: "shared"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency conflict"
    assert _count(app, ReleaseGrantBulkRevokeIdempotencyRecord) == 1


def test_same_key_in_other_scope_is_independent(client, app):
    decision_a = _allowed_decision(client)
    grants_a = _grants(client, decision_a, 1, prefix="a")
    # A grant for another tenant needs its own allowed decision chain.
    other = client.post(
        "/v1/challenges",
        json={"tenant_id": "tenant-b", "workload_id": WORKLOAD},
    ).json()
    key_b = hmac.new(
        SECRET.encode(),
        b"tenant-b:" + WORKLOAD.encode(),
        hashlib.sha256,
    ).digest()
    claims = {"m": "x"}
    ev_b = json.dumps(
        {
            "nonce": other["nonce"],
            "claims": claims,
            "mac": hmac.new(
                key_b,
                json.dumps(
                    {"claims": claims, "nonce": other["nonce"]},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
                hashlib.sha256,
            ).hexdigest(),
        }
    )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": "tenant-b",
            "workload_id": WORKLOAD,
            "challenge_id": other["challenge_id"],
            "nonce": other["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": ev_b,
        },
    )
    assert submitted.status_code == 201
    ev_b_id = submitted.json()["evidence_id"]
    client.post(
        f"/v1/evidence/{ev_b_id}/verify",
        json={
            "tenant_id": "tenant-b",
            "workload_id": WORKLOAD,
            "nonce": other["nonce"],
            "evidence": ev_b,
        },
    )
    policy_b = client.post(
        "/v1/policies",
        json={
            "tenant_id": "tenant-b",
            "workload_id": WORKLOAD,
            "name": "release",
            "rule": {"claim": "m", "equals": "x"},
        },
    ).json()
    decided_b = client.post(
        f"/v1/evidence/{ev_b_id}/decisions",
        json={
            "tenant_id": "tenant-b",
            "workload_id": WORKLOAD,
            "nonce": other["nonce"],
            "evidence": ev_b,
            "policy_id": policy_b["policy_id"],
        },
    )
    assert decided_b.status_code == 200
    grants_b = [
        _grant(
            client,
            decided_b.json()["decision_id"],
            "b-0",
            tenant_id="tenant-b",
        )
    ]
    r1 = _bulk(client, grants_a, key="shared-key")
    r2 = _bulk(client, grants_b, key="shared-key", tenant="tenant-b")
    assert r1.status_code == r2.status_code == 200
    assert r1.content != r2.content
    assert _count(app, ReleaseGrantBulkRevokeIdempotencyRecord) == 2


def test_keyless_success_does_not_occupy_key(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 1)
    assert _bulk(client, grants, key=None).status_code == 200
    second = _bulk(client, grants, key="late-key")
    assert second.status_code == 409
    assert second.json()["detail"] == "grant already revoked"
    assert _count(app, ReleaseGrantBulkRevokeIdempotencyRecord) == 0


def test_invalid_idempotency_header_is_422_before_budget(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 1)
    response = client.post(
        PATH,
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "grants": [_entry(grants[0])],
        },
        headers=[
            (IDEMPOTENCY_HEADER, "k"),
            (IDEMPOTENCY_HEADER, "k"),
        ],
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid idempotency key"
    assert _count(app, RateLimitCounter) == 0
    assert _count(app, ReleaseGrantBulkRevokeIdempotencyRecord) == 0


def test_invalid_header_is_422_even_when_budget_exhausted(client, app):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 1)
    with app.state.session_factory() as session:
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        session.add(
            RateLimitCounter(
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                window_start=now,
                count=app_module.GRANT_BUDGET_PER_MINUTE,
            )
        )
        session.commit()
    response = client.post(
        PATH,
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "grants": [_entry(grants[0])],
        },
        headers=[(IDEMPOTENCY_HEADER, "k"), (IDEMPOTENCY_HEADER, "k")],
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid idempotency key"


# ---------------------------------------------------------------------------
# Concurrency, persistence failure, restart and secrecy
# ---------------------------------------------------------------------------


def test_concurrent_identical_bulks_settle_once(app, monkeypatch):
    monkeypatch.setattr(app_module, "GRANT_BUDGET_PER_MINUTE", 64)
    client = TestClient(app)
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 3)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "grants": [_entry(g) for g in grants],
    }

    def call():
        return TestClient(app).post(PATH, json=body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: call(), range(8)))

    assert statuses.count(200) == 1
    assert statuses.count(409) == 7
    with app.state.session_factory() as session:
        assert session.query(ReleaseGrant).filter(
            ReleaseGrant.status == "revoked"
        ).count() == 3
        assert (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.reason == "revoked")
            .count()
            == 3
        )
        assert (
            session.query(AuditEvent)
            .filter(AuditEvent.status == "revoked")
            .count()
            == 3
        )


def test_concurrent_same_key_bulks_settle_once(app, monkeypatch):
    monkeypatch.setattr(app_module, "GRANT_BUDGET_PER_MINUTE", 64)
    client = TestClient(app)
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 3)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "grants": [_entry(g) for g in grants],
    }
    headers = {IDEMPOTENCY_HEADER: "race-key"}

    def call():
        return TestClient(app).post(PATH, json=body, headers=headers)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: call(), range(8)))

    assert [r.status_code for r in responses].count(200) == 8
    assert len({r.content for r in responses}) == 1
    with app.state.session_factory() as session:
        assert session.query(
            ReleaseGrantBulkRevokeIdempotencyRecord
        ).count() == 1
        assert (
            session.query(ReleaseGrantEvent)
            .filter(ReleaseGrantEvent.reason == "revoked")
            .count()
            == 3
        )


def test_persistence_failure_keyless_returns_500_and_rolls_back(
    client, app, monkeypatch
):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 3)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(app_module, "_record_release_grant_event", boom)
    response = _bulk(client, grants)
    assert response.status_code == 500
    assert response.json()["detail"] == "grant unavailable"

    monkeypatch.undo()
    _assert_all_pending(app, grants)
    _assert_no_revoke_writes(app, grants)
    recovered = _bulk(client, grants)
    assert recovered.status_code == 200


def test_persistence_failure_keyed_returns_500_and_leaves_nothing(
    client, app, monkeypatch
):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 3)
    real_commit = Session.commit

    def raise_commit(self):
        if any(
            isinstance(obj, ReleaseGrantBulkRevokeIdempotencyRecord)
            for obj in self.new
        ):
            raise OSError("disk full")
        return real_commit(self)

    monkeypatch.setattr(Session, "commit", raise_commit)
    failed = _bulk(client, grants, key="durable-key")
    assert failed.status_code == 500
    assert failed.json()["detail"] == "grant unavailable"
    monkeypatch.undo()

    _assert_all_pending(app, grants)
    _assert_no_revoke_writes(app, grants)
    assert _count(app, ReleaseGrantBulkRevokeIdempotencyRecord) == 0
    recovered = _bulk(client, grants, key="durable-key")
    assert recovered.status_code == 200
    assert _bulk(client, grants, key="durable-key").content == recovered.content


def test_bulk_revoke_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart-bulk.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    decision_id = _allowed_decision(client1)
    grants = _grants(client1, decision_id, 2)
    first = _bulk(client1, grants, key="durable-key")
    assert first.status_code == 200
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    replay = _bulk(client2, grants, key="durable-key")
    assert replay.status_code == 200
    assert replay.content == first.content
    with app2.state.session_factory() as session:
        for grant in grants:
            assert session.get(
                ReleaseGrant, grant["grant_id"]
            ).status == "revoked"
    app2.state.engine.dispose()


def test_idempotency_table_created_additively_on_old_database(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/legacy-bulk.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    decision_id = _allowed_decision(client1)
    grants = _grants(client1, decision_id, 1)
    app1.state.engine.dispose()

    from sqlalchemy import text

    app2 = create_app(url)
    with app2.state.engine.begin() as conn:
        conn.execute(
            text(
                "DROP TABLE release_grant_bulk_revoke_idempotency_records"
            )
        )
    app2.state.engine.dispose()

    app3 = create_app(url)
    client3 = TestClient(app3)
    response = _bulk(client3, grants, key="after-upgrade")
    assert response.status_code == 200
    with app3.state.session_factory() as session:
        assert session.query(
            ReleaseGrantBulkRevokeIdempotencyRecord
        ).count() == 1
    app3.state.engine.dispose()


def test_capabilities_never_persisted_or_logged(client, app, caplog):
    decision_id = _allowed_decision(client)
    grants = _grants(client, decision_id, 2)
    response = _bulk(client, grants, key="secret-key")
    assert response.status_code == 200
    for grant in grants:
        assert grant["capability"] not in response.text
    with app.state.session_factory() as session:
        record = session.query(
            ReleaseGrantBulkRevokeIdempotencyRecord
        ).one()
        values = {
            c.name: getattr(record, c.name)
            for c in record.__table__.columns
        }
        for grant in grants:
            for name, value in values.items():
                assert grant["capability"] not in str(value), name
        assert record.request_fingerprint != hashlib.sha256(
            grants[0]["capability"].encode()
        ).hexdigest()
        assert not hasattr(record, "capability")
    for grant in grants:
        assert grant["capability"] not in caplog.text
    assert MASTER_KEY not in caplog.text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrong(token: str) -> str:
    return ("B" if token[0] != "B" else "C") + token[1:]


def _assert_all_pending(app, grants):
    with app.state.session_factory() as session:
        for grant in grants:
            row = session.get(ReleaseGrant, grant["grant_id"])
            assert row.status == "pending", grant["grant_id"]
            assert row.revoked_at is None
            assert row.consumed_at is None


def _assert_no_revoke_writes(app, grants):
    with app.state.session_factory() as session:
        for grant in grants:
            assert (
                session.query(ReleaseGrantEvent)
                .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
                .count()
                == 1  # birth event only
            )
            assert (
                session.query(AuditEvent)
                .filter(
                    AuditEvent.grant_id == grant["grant_id"],
                    AuditEvent.status == "revoked",
                )
                .count()
                == 0
            )
