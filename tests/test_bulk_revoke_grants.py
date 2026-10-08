"""Tests for POST /v1/release-grants/bulk-revoke.

Covers the atomic batch revocation of pending one-time grants in one
tenant/workload scope: the fixed 422 request contract, the sorted
judgement order and its fail-atomic semantics, the single shared budget
slot per request, the per-grant timeline and audit writes, and the
optional Idempotency-Key replay/conflict/persistence-failure behaviour.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.app import create_app
from proof_release.db import (
    AuditEvent,
    RateLimitCounter,
    ReleaseGrant,
    ReleaseGrantBulkRevokeIdempotencyRecord,
    ReleaseGrantEvent,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"
IDEMPOTENCY_HEADER = "Idempotency-Key"
URL = "/v1/release-grants/bulk-revoke"


def _rfc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _wrong(token: str) -> str:
    """A well-formed token guaranteed to differ from ``token``."""
    return ("B" if token[0] != "B" else "C") + token[1:]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    application = create_app(f"sqlite:///{tmp_path}/bulk-revoke.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _mac(nonce: str, claims: dict, tenant: str, workload: str) -> str:
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


def _decision(client, tenant=TENANT, workload=WORKLOAD):
    claims = {"m": "x"}
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": claims,
            "mac": _mac(created["nonce"], claims, tenant, workload),
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
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "name": "release",
            "rule": {"claim": "m", "equals": "x"},
        },
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


def _grant(client, decision_id, data_id, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/release-grants",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "decision_id": decision_id,
            "data_id": data_id,
        },
    )
    assert response.status_code == 201
    return response.json()


def _setup_grants(client, count, tenant=TENANT, workload=WORKLOAD):
    """Mint ``count`` pending grants in one scope off a single decision."""
    decision = _decision(client, tenant=tenant, workload=workload)
    return [
        _grant(
            client,
            decision["decision_id"],
            data_id=f"data-{i}",
            tenant=tenant,
            workload=workload,
        )
        for i in range(count)
    ]


def _pairs(grants):
    return [
        {"grant_id": g["grant_id"], "capability": g["capability"]}
        for g in grants
    ]


def _bulk(client, pairs, tenant=TENANT, workload=WORKLOAD, key=None, **overrides):
    body = {"tenant_id": tenant, "workload_id": workload, "grants": pairs}
    body.update(overrides)
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else {}
    return client.post(URL, json=body, headers=headers)


def _grant_row(app, grant_id):
    with app.state.session_factory() as session:
        return session.get(ReleaseGrant, grant_id)


def _assert_all_pending(app, grants):
    with app.state.session_factory() as session:
        for grant in grants:
            row = session.get(ReleaseGrant, grant["grant_id"])
            assert row.status == "pending"
            assert row.revoked_at is None
            assert row.consumed_at is None


def _revoke_events(app, grant_id):
    with app.state.session_factory() as session:
        return (
            session.query(ReleaseGrantEvent)
            .filter(
                ReleaseGrantEvent.grant_id == grant_id,
                ReleaseGrantEvent.new_status == "revoked",
            )
            .all()
        )


def _revoke_audits(app, grant_id):
    with app.state.session_factory() as session:
        return (
            session.query(AuditEvent)
            .filter(
                AuditEvent.grant_id == grant_id,
                AuditEvent.status == "revoked",
            )
            .all()
        )


# ---------------------------------------------------------------------------
# Success shape
# ---------------------------------------------------------------------------


def test_bulk_revoke_all_pending_returns_200(client, app):
    grants = _setup_grants(client, 3)

    response = _bulk(client, _pairs(grants))

    assert response.status_code == 200
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
    by_id = {g["grant_id"]: g for g in grants}
    assert [item["grant_id"] for item in data["revoked_grants"]] == sorted(by_id)
    for item in data["revoked_grants"]:
        assert list(item) == ["grant_id", "decision_id", "data_id"]
        assert item["decision_id"] == by_id[item["grant_id"]]["decision_id"]
        assert item["data_id"] == by_id[item["grant_id"]]["data_id"]
    # The compact wire form carries no capability, payload, evidence,
    # claims or key material.
    assert response.text == json.dumps(data, separators=(",", ":"))
    for grant in grants:
        assert grant["capability"] not in response.text
    assert "capability" not in response.text


def test_bulk_revoke_sets_state_once_with_shared_timestamp(client, app):
    grants = _setup_grants(client, 3)

    revoked_at = _bulk(client, _pairs(grants)).json()["revoked_at"]

    with app.state.session_factory() as session:
        for grant in grants:
            row = session.get(ReleaseGrant, grant["grant_id"])
            assert row.status == "revoked"
            assert row.consumed_at is None
            assert row.revoked_at is not None
            # One shared UTC revoked_at for the whole batch.
            assert _rfc(row.revoked_at) == revoked_at


def test_bulk_revoke_sorts_response_regardless_of_request_order(client):
    grants = _setup_grants(client, 4)
    pairs = list(reversed(sorted(_pairs(grants), key=lambda p: p["grant_id"])))

    response = _bulk(client, pairs)

    assert response.status_code == 200
    ids = [item["grant_id"] for item in response.json()["revoked_grants"]]
    assert ids == sorted(ids)


def test_bulk_revoke_single_grant_batch(client, app):
    grants = _setup_grants(client, 1)

    response = _bulk(client, _pairs(grants))

    assert response.status_code == 200
    data = response.json()
    assert data["revoked_count"] == 1
    assert data["revoked_grants"][0]["grant_id"] == grants[0]["grant_id"]
    assert _grant_row(app, grants[0]["grant_id"]).status == "revoked"


def test_bulk_revoke_accepts_uppercase_uuids(client):
    grants = _setup_grants(client, 2)
    pairs = [
        {"grant_id": g["grant_id"].upper(), "capability": g["capability"]}
        for g in grants
    ]

    response = _bulk(client, pairs)

    assert response.status_code == 200
    ids = [item["grant_id"] for item in response.json()["revoked_grants"]]
    assert ids == sorted(g["grant_id"] for g in grants)


def test_bulk_revoke_max_batch_size(client, app):
    grants = _setup_grants(client, 100)

    response = _bulk(client, _pairs(grants))

    assert response.status_code == 200
    assert response.json()["revoked_count"] == 100
    with app.state.session_factory() as session:
        assert (
            session.query(ReleaseGrant)
            .filter(ReleaseGrant.status == "revoked")
            .count()
            == 100
        )


def test_bulk_revoke_writes_one_timeline_event_and_audit_per_grant(client, app):
    grants = _setup_grants(client, 3)

    revoked_at = _bulk(client, _pairs(grants)).json()["revoked_at"]

    with app.state.session_factory() as session:
        for grant in grants:
            events = (
                session.query(ReleaseGrantEvent)
                .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
                .order_by(ReleaseGrantEvent.seq)
                .all()
            )
            # The birth event plus exactly one revoked settlement event.
            assert [e.new_status for e in events] == ["pending", "revoked"]
            settlement = events[1]
            assert settlement.seq == 2
            assert settlement.old_status == "pending"
            assert settlement.reason == "revoked"
            assert settlement.tenant_id == TENANT
            assert settlement.workload_id == WORKLOAD
            assert _rfc(settlement.occurred_at) == revoked_at
            audits = (
                session.query(AuditEvent)
                .filter(
                    AuditEvent.grant_id == grant["grant_id"],
                    AuditEvent.status == "revoked",
                )
                .all()
            )
            assert len(audits) == 1
            audit = audits[0]
            assert audit.event_type == "grant"
            assert audit.decision_id == grant["decision_id"]
            assert _rfc(audit.occurred_at) == revoked_at


# ---------------------------------------------------------------------------
# 422: the fixed request contract, enforced before any read or budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"workload_id": ""},
        {"workload_id": "   "},
        {"grants": None},
        {"grants": "not-a-list"},
        {"grants": []},
        {"grants": [{"grant_id": "not-a-uuid", "capability": "A" * 43}]},
        {
            "grants": [
                {
                    "grant_id": " 00000000-0000-0000-0000-000000000000",
                    "capability": "A" * 43,
                }
            ]
        },
        {"grants": [{"grant_id": "00000000-0000-0000-0000-000000000000"}]},
        {
            "grants": [
                {"grant_id": "00000000-0000-0000-0000-000000000000",
                 "capability": ""}
            ]
        },
        {
            "grants": [
                {"grant_id": "00000000-0000-0000-0000-000000000000",
                 "capability": "with=padding"}
            ]
        },
        {
            "grants": [
                {"grant_id": "00000000-0000-0000-0000-000000000000",
                 "capability": "not base64!!!"}
            ]
        },
        {
            "grants": [
                {"grant_id": "00000000-0000-0000-0000-000000000000",
                 "capability": "A" * 43, "extra": "nope"}
            ]
        },
        {"grants": ["not-an-object"]},
        {"unexpected": "field"},
    ],
)
def test_bulk_revoke_rejects_invalid_requests(client, app, overrides):
    grants = _setup_grants(client, 1)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "grants": _pairs(grants),
    }
    body.update(overrides)

    response = client.post(URL, json=body)

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid bulk revoke request"
    _assert_all_pending(app, grants)


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "grants"])
def test_bulk_revoke_requires_all_fields(client, missing):
    grants = _setup_grants(client, 1)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "grants": _pairs(grants),
    }
    del body[missing]

    response = client.post(URL, json=body)

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid bulk revoke request"


def test_bulk_revoke_rejects_non_object_body(client):
    _setup_grants(client, 1)
    response = client.post(URL, json=["not", "an", "object"])
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid bulk revoke request"


def test_bulk_revoke_rejects_malformed_json(client):
    _setup_grants(client, 1)
    response = client.post(
        URL, content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid bulk revoke request"


def test_bulk_revoke_rejects_overlarge_batch(client):
    grants = _setup_grants(client, 1)
    pair = _pairs(grants)[0]
    pairs = [
        {"grant_id": f"00000000-0000-0000-0000-{i:012d}", "capability": "A" * 43}
        for i in range(100)
    ]
    pairs.append(pair)

    response = _bulk(client, pairs)

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid bulk revoke request"


def test_bulk_revoke_rejects_duplicate_grant_ids(client, app):
    grants = _setup_grants(client, 2)
    pairs = _pairs(grants)
    pairs.append(dict(pairs[0]))

    response = _bulk(client, pairs)

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid bulk revoke request"
    _assert_all_pending(app, grants)


def test_bulk_revoke_rejects_case_insensitive_duplicate_grant_ids(client):
    grants = _setup_grants(client, 1)
    pair = _pairs(grants)[0]
    pairs = [pair, {"grant_id": pair["grant_id"].upper(),
                    "capability": pair["capability"]}]

    response = _bulk(client, pairs)

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid bulk revoke request"


def test_422_reads_no_grant_and_draws_no_budget(client, app):
    grants = _setup_grants(client, 1)

    response = client.post(URL, json={"tenant_id": TENANT})

    assert response.status_code == 422
    with app.state.session_factory() as session:
        # No budget slot was reserved for the scope.
        assert session.query(RateLimitCounter).count() == 0
    _assert_all_pending(app, grants)


# ---------------------------------------------------------------------------
# Judgement: sorted order, fail-atomic
# ---------------------------------------------------------------------------


def test_unknown_grant_fails_whole_batch(client, app):
    grants = _setup_grants(client, 2)
    pairs = _pairs(grants)
    pairs.append(
        {"grant_id": "00000000-0000-0000-0000-000000000000",
         "capability": "A" * 43}
    )

    response = _bulk(client, pairs)

    assert response.status_code == 404
    assert response.json()["detail"] == "grant not found"
    _assert_all_pending(app, grants)
    for grant in grants:
        assert _revoke_events(app, grant["grant_id"]) == []
        assert _revoke_audits(app, grant["grant_id"]) == []


def test_cross_scope_grant_fails_whole_batch(client, app):
    grants = _setup_grants(client, 1)
    other = _setup_grants(client, 1, tenant="tenant-b", workload="workload-2")
    pairs = _pairs(grants) + _pairs(other)

    for tenant, workload in [
        (TENANT, WORKLOAD),
        ("tenant-b", "workload-2"),
    ]:
        # From either scope the batch contains a foreign grant: 404 and
        # nothing changes anywhere.
        response = _bulk(client, pairs, tenant=tenant, workload=workload)
        assert response.status_code == 404
        assert response.json()["detail"] == "grant not found"
    _assert_all_pending(app, grants)
    _assert_all_pending(app, other)


def test_wrong_capability_returns_401_and_batch_unchanged(client, app):
    grants = _setup_grants(client, 2)
    pairs = _pairs(grants)
    pairs[1] = dict(pairs[1], capability=_wrong(pairs[1]["capability"]))

    response = _bulk(client, pairs)

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid capability"
    _assert_all_pending(app, grants)
    for grant in grants:
        assert _revoke_events(app, grant["grant_id"]) == []
        assert _revoke_audits(app, grant["grant_id"]) == []


def test_consumed_grant_returns_409_and_batch_unchanged(client, app):
    grants = _setup_grants(client, 2)
    victim = grants[0]
    consumed = client.post(
        f"/v1/release-grants/{victim['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": victim["capability"],
        },
    )
    assert consumed.status_code == 200

    response = _bulk(client, _pairs(grants))

    assert response.status_code == 409
    assert response.json()["detail"] == "grant already consumed"
    assert _grant_row(app, grants[1]["grant_id"]).status == "pending"
    assert _revoke_events(app, grants[1]["grant_id"]) == []


def test_already_revoked_grant_returns_409_and_batch_unchanged(client, app):
    grants = _setup_grants(client, 2)
    victim = grants[0]
    revoked = client.post(
        f"/v1/release-grants/{victim['grant_id']}/revoke",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": victim["capability"],
        },
    )
    assert revoked.status_code == 200

    response = _bulk(client, _pairs(grants))

    assert response.status_code == 409
    assert response.json()["detail"] == "grant already revoked"
    assert _grant_row(app, grants[1]["grant_id"]).status == "pending"
    assert _revoke_events(app, grants[1]["grant_id"]) == []


def test_expired_pending_grant_returns_410_and_batch_unchanged(client, app):
    grants = _setup_grants(client, 2)
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grants[0]["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = _bulk(client, _pairs(grants))

    assert response.status_code == 410
    assert response.json()["detail"] == "grant expired"
    assert _grant_row(app, grants[0]["grant_id"]).status == "pending"
    assert _grant_row(app, grants[1]["grant_id"]).status == "pending"


def test_judgement_follows_lexicographic_grant_id_order(client, app):
    grants = _setup_grants(client, 2)
    first, second = sorted(grants, key=lambda g: g["grant_id"])
    # The lexicographically smaller grant is expired (410); the larger one
    # carries a wrong capability (401). The smaller id judges first.
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, first["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
    pairs = [
        {"grant_id": first["grant_id"], "capability": first["capability"]},
        {"grant_id": second["grant_id"],
         "capability": _wrong(second["capability"])},
    ]

    response = _bulk(client, pairs)

    assert response.status_code == 410
    assert response.json()["detail"] == "grant expired"
    _assert_all_pending(app, grants)


def test_capability_mismatch_judged_before_expiry(client, app):
    grants = _setup_grants(client, 1)
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grants[0]["grant_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
    pairs = [
        {"grant_id": grants[0]["grant_id"],
         "capability": _wrong(grants[0]["capability"])}
    ]

    response = _bulk(client, pairs)

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid capability"


# ---------------------------------------------------------------------------
# Budget: exactly one shared slot per request
# ---------------------------------------------------------------------------


def test_bulk_revoke_draws_exactly_one_slot_per_request(client, app):
    grants = _setup_grants(client, 3)

    response = _bulk(client, _pairs(grants))

    assert response.status_code == 200
    with app.state.session_factory() as session:
        row = session.query(RateLimitCounter).one()
        assert row.count == 1


def test_bulk_revoke_shares_the_five_per_minute_budget(client, app):
    grants = _setup_grants(client, 6)

    # Five admitted requests (one grant each) exhaust the shared minute
    # budget; the sixth is rejected before any judgement.
    for grant in grants[:5]:
        response = _bulk(client, _pairs([grant]))
        assert response.status_code == 200
    sixth = _bulk(client, _pairs([grants[5]]))
    assert sixth.status_code == 429
    assert _grant_row(app, grants[5]["grant_id"]).status == "pending"


def test_failed_judgement_keeps_its_slot_but_changes_nothing(client, app):
    grants = _setup_grants(client, 2)
    pairs = _pairs(grants)
    pairs.append(
        {"grant_id": "00000000-0000-0000-0000-000000000000",
         "capability": "A" * 43}
    )

    assert _bulk(client, pairs).status_code == 404

    with app.state.session_factory() as session:
        row = session.query(RateLimitCounter).one()
        assert row.count == 1
    _assert_all_pending(app, grants)


# ---------------------------------------------------------------------------
# Idempotency-Key
# ---------------------------------------------------------------------------


def test_keyed_replay_returns_first_response_and_changes_nothing(client, app):
    grants = _setup_grants(client, 2)

    first = _bulk(client, _pairs(grants), key="bulk-key")
    assert first.status_code == 200

    replay = _bulk(client, _pairs(grants), key="bulk-key")
    assert replay.status_code == 200
    assert replay.content == first.content

    with app.state.session_factory() as session:
        assert (
            session.query(ReleaseGrantBulkRevokeIdempotencyRecord).count() == 1
        )
        for grant in grants:
            # Still exactly the birth and the one revoked settlement event.
            assert (
                session.query(ReleaseGrantEvent)
                .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
                .count()
                == 2
            )
            assert (
                session.query(AuditEvent)
                .filter(
                    AuditEvent.grant_id == grant["grant_id"],
                    AuditEvent.status == "revoked",
                )
                .count()
                == 1
            )


def test_keyed_replay_ignores_grant_expiry_and_request_order(client, app):
    grants = _setup_grants(client, 2)
    first = _bulk(client, _pairs(grants), key="bulk-key")
    assert first.status_code == 200
    with app.state.session_factory() as session:
        for grant in grants:
            row = session.get(ReleaseGrant, grant["grant_id"])
            row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    # The same pairs in the opposite wire order are an equivalent replay:
    # the stored first 200 comes back verbatim even though every grant has
    # since expired.
    replay = _bulk(client, list(reversed(_pairs(grants))), key="bulk-key")
    assert replay.status_code == 200
    assert replay.content == first.content


def test_keyed_conflict_returns_409_and_changes_nothing(client, app):
    grants = _setup_grants(client, 2)
    first = _bulk(client, _pairs(grants[:1]), key="bulk-key")
    assert first.status_code == 200

    # Same key, different grant set.
    conflict = _bulk(client, _pairs(grants), key="bulk-key")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency conflict"
    # Same key, same grant, different capability.
    altered = [
        {"grant_id": grants[0]["grant_id"],
         "capability": _wrong(grants[0]["capability"])}
    ]
    conflict2 = _bulk(client, altered, key="bulk-key")
    assert conflict2.status_code == 409
    assert conflict2.json()["detail"] == "idempotency conflict"
    # The untouched second grant is still pending and revocable.
    assert _grant_row(app, grants[1]["grant_id"]).status == "pending"
    assert _bulk(client, _pairs(grants[1:])).status_code == 200


def test_keyed_conflict_scoped_to_tenant_and_workload(client, app):
    first_grants = _setup_grants(client, 1)
    other_grants = _setup_grants(client, 1, tenant="tenant-b",
                                 workload="workload-2")
    assert (
        _bulk(client, _pairs(first_grants), key="bulk-key").status_code == 200
    )

    # The same key in another scope is an independent key.
    other = _bulk(
        client, _pairs(other_grants),
        tenant="tenant-b", workload="workload-2", key="bulk-key",
    )
    assert other.status_code == 200
    with app.state.session_factory() as session:
        assert (
            session.query(ReleaseGrantBulkRevokeIdempotencyRecord).count() == 2
        )


def test_failed_keyed_request_saves_no_record(client, app):
    grants = _setup_grants(client, 1)
    pairs = _pairs(grants)
    pairs.append(
        {"grant_id": "00000000-0000-0000-0000-000000000000",
         "capability": "A" * 43}
    )

    failed = _bulk(client, pairs, key="bulk-key")
    assert failed.status_code == 404
    with app.state.session_factory() as session:
        assert (
            session.query(ReleaseGrantBulkRevokeIdempotencyRecord).count() == 0
        )

    # The key stayed free: the recovered request is judged normally.
    recovered = _bulk(client, _pairs(grants), key="bulk-key")
    assert recovered.status_code == 200
    replay = _bulk(client, _pairs(grants), key="bulk-key")
    assert replay.status_code == 200
    assert replay.content == recovered.content


def test_duplicate_idempotency_header_returns_422(client, app):
    grants = _setup_grants(client, 1)
    response = client.post(
        URL,
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "grants": _pairs(grants),
        },
        headers=[(IDEMPOTENCY_HEADER, "a"), (IDEMPOTENCY_HEADER, "b")],
    )
    assert response.status_code == 422
    _assert_all_pending(app, grants)
    with app.state.session_factory() as session:
        assert session.query(RateLimitCounter).count() == 0


@pytest.mark.parametrize("key", ["", " key", "key ", "a" * 65, "bad key"])
def test_invalid_idempotency_key_returns_422(client, app, key):
    grants = _setup_grants(client, 1)
    response = _bulk(client, _pairs(grants), key=key)
    assert response.status_code == 422
    _assert_all_pending(app, grants)


def test_idempotency_record_never_stores_capability(client, app):
    grants = _setup_grants(client, 2)
    assert _bulk(client, _pairs(grants), key="bulk-key").status_code == 200

    with app.state.session_factory() as session:
        record = session.query(ReleaseGrantBulkRevokeIdempotencyRecord).one()
        for column in record.__table__.columns:
            value = str(getattr(record, column.name))
            for grant in grants:
                assert grant["capability"] not in value


def test_concurrent_same_key_bulk_revokes_settle_once(client, app, monkeypatch):
    # Settlement-race test, not the shared per-minute budget: admit all.
    monkeypatch.setattr(app_module, "GRANT_BUDGET_PER_MINUTE", 64)
    grants = _setup_grants(client, 2)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "grants": _pairs(grants),
    }

    def call():
        return TestClient(app).post(
            URL, json=body, headers={IDEMPOTENCY_HEADER: "bulk-key"}
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: call(), range(16)))

    assert [r.status_code for r in responses].count(200) == 16
    assert len({r.content for r in responses}) == 1
    with app.state.session_factory() as session:
        assert (
            session.query(ReleaseGrantBulkRevokeIdempotencyRecord).count() == 1
        )
        for grant in grants:
            assert (
                session.query(ReleaseGrantEvent)
                .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
                .count()
                == 2
            )


def test_concurrent_unkeyed_bulk_revokes_single_winner(client, app, monkeypatch):
    # Settlement-race test, not the shared per-minute budget: admit all.
    monkeypatch.setattr(app_module, "GRANT_BUDGET_PER_MINUTE", 64)
    grants = _setup_grants(client, 2)
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "grants": _pairs(grants),
    }

    def call():
        return TestClient(app).post(URL, json=body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: call(), range(16)))

    assert statuses.count(200) == 1
    assert statuses.count(409) == 15
    with app.state.session_factory() as session:
        for grant in grants:
            row = session.get(ReleaseGrant, grant["grant_id"])
            assert row.status == "revoked"
            assert (
                session.query(ReleaseGrantEvent)
                .filter(ReleaseGrantEvent.grant_id == grant["grant_id"])
                .count()
                == 2
            )


# ---------------------------------------------------------------------------
# Persistence failure and restart
# ---------------------------------------------------------------------------


def test_keyed_persistence_failure_returns_500_and_rolls_back(
    client, app, monkeypatch
):
    from sqlalchemy.orm import Session

    grants = _setup_grants(client, 2)
    real_commit = Session.commit

    def raise_commit(self):
        # The budget reservation commits its own counter row first; only
        # the bulk business transaction carries the idempotency record
        # (plus the events and audits), so fail precisely that commit.
        if any(
            isinstance(obj, ReleaseGrantBulkRevokeIdempotencyRecord)
            for obj in self.new
        ):
            raise OSError("disk full")
        return real_commit(self)

    monkeypatch.setattr(Session, "commit", raise_commit)

    failed = _bulk(client, _pairs(grants), key="bulk-key")
    assert failed.status_code == 500
    assert failed.json()["detail"] == "grant unavailable"

    monkeypatch.undo()
    _assert_all_pending(app, grants)
    for grant in grants:
        assert _revoke_events(app, grant["grant_id"]) == []
        assert _revoke_audits(app, grant["grant_id"]) == []
    with app.state.session_factory() as session:
        assert (
            session.query(ReleaseGrantBulkRevokeIdempotencyRecord).count() == 0
        )

    # The failed attempt occupied neither the grants nor the key.
    recovered = _bulk(client, _pairs(grants), key="bulk-key")
    assert recovered.status_code == 200


def test_keyless_persistence_failure_returns_500_and_rolls_back(
    client, app, monkeypatch
):
    from sqlalchemy.orm import Session

    grants = _setup_grants(client, 2)
    real_flush = Session.flush

    def raise_flush(self):
        # Fail the business transaction (it carries the revoked timeline
        # events); the budget reservation commits separately before it.
        # The per-grant guarded UPDATEs autoflush earlier events, so the
        # failure is triggered on flush rather than on commit.
        if any(
            isinstance(obj, ReleaseGrantEvent) and obj.new_status == "revoked"
            for obj in self.new
        ):
            raise OSError("disk full")
        return real_flush(self)

    monkeypatch.setattr(Session, "flush", raise_flush)

    failed = _bulk(client, _pairs(grants))
    assert failed.status_code == 500
    assert failed.json()["detail"] == "grant unavailable"

    monkeypatch.undo()
    _assert_all_pending(app, grants)
    for grant in grants:
        assert _revoke_events(app, grant["grant_id"]) == []
        assert _revoke_audits(app, grant["grant_id"]) == []

    assert _bulk(client, _pairs(grants)).status_code == 200


def test_bulk_revoke_and_idempotency_record_survive_restart(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart-bulk-revoke.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    grants = _setup_grants(client1, 2)
    first = _bulk(client1, _pairs(grants), key="bulk-key")
    assert first.status_code == 200
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    replay = _bulk(client2, _pairs(grants), key="bulk-key")
    assert replay.status_code == 200
    assert replay.content == first.content
    # An unkeyed repeat observes the terminal state as 409.
    assert _bulk(client2, _pairs(grants)).status_code == 409
    with app2.state.session_factory() as session:
        for grant in grants:
            assert session.get(ReleaseGrant, grant["grant_id"]).status == "revoked"
    app2.state.engine.dispose()
