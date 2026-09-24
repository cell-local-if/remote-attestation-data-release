"""Tests for the per-scope rate limit on the authorization actions.

POST /v1/release-grants/{grant_id}/consume, POST /v1/release-grants/
{grant_id}/revoke and POST /v1/release/{grant_id} share one budget of
five field-valid requests per tenant/workload per UTC minute. Requests
rejected for field/format errors (422) never consume budget; every other
syntactically valid, clearly scoped request consumes exactly one unit
regardless of its business outcome (404/401/409/410/500). The sixth
request in a window is rejected with 429 and a compact JSON body naming
the whole seconds until the window ends; it writes no state, no audit
and releases no payload. The budget persists across process restarts and
is strictly per scope.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import AuditEvent, RateLimitWindow, ReleaseGrant
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "data-1"
SECRET = "unit-test-secret"
MASTER_KEY = b64url_encode(b"0123456789abcdef0123456789abcdef")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = create_app(f"sqlite:///{tmp_path}/rate_limit.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _mac(nonce: str, claims: dict, *, tenant: str, workload: str) -> str:
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


def _decision(client, *, tenant=TENANT, workload=WORKLOAD):
    created = client.post(
        "/v1/challenges", json={"tenant_id": tenant, "workload_id": workload}
    ).json()
    claims = {"m": "x"}
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": claims,
            "mac": _mac(created["nonce"], claims, tenant=tenant, workload=workload),
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
    evidence_id = submitted.json()["evidence_id"]
    client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
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
    return decided.json()


def _grant(client, decision_id, *, tenant=TENANT, workload=WORKLOAD, data_id=DATA_ID):
    return client.post(
        "/v1/release-grants",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "decision_id": decision_id,
            "data_id": data_id,
        },
    ).json()


def _consume_body(grant, *, tenant=TENANT, workload=WORKLOAD):
    return {
        "tenant_id": tenant,
        "workload_id": workload,
        "capability": grant["capability"],
    }


def _expected_retry_after() -> int:
    return max(1, 60 - datetime.now(timezone.utc).second)


def _assert_429(response):
    assert response.status_code == 429
    # Compact JSON, exactly one positive-integer field, one trailing
    # newline, no floats.
    assert re.fullmatch(rb'\{"retry_after_seconds":[0-9]+\}\n', response.content)
    value = response.json()["retry_after_seconds"]
    assert isinstance(value, int) and value >= 1
    assert value in (_expected_retry_after(), _expected_retry_after() + 1, 60)


# --- budget accounting -----------------------------------------------------


def test_five_valid_requests_admitted_sixth_is_429(client):
    decision = _decision(client)
    grants = [_grant(client, decision["decision_id"], data_id=f"d{i}") for i in range(6)]
    statuses = [
        client.post(
            f"/v1/release-grants/{g['grant_id']}/consume", json=_consume_body(g)
        ).status_code
        for g in grants
    ]
    assert statuses[:5] == [200] * 5
    assert statuses[5] == 429


def test_budget_shared_across_consume_revoke_and_release(client):
    decision = _decision(client)
    client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": DATA_ID,
            "payload": "shared-budget payload",
        },
    )
    grants = [_grant(client, decision["decision_id"], data_id=DATA_ID) for _ in range(6)]
    # Two consumes, two revokes, one release: five admitted in total.
    outcomes = []
    for g in grants[:2]:
        outcomes.append(
            client.post(
                f"/v1/release-grants/{g['grant_id']}/consume", json=_consume_body(g)
            ).status_code
        )
    for g in grants[2:4]:
        outcomes.append(
            client.post(
                f"/v1/release-grants/{g['grant_id']}/revoke", json=_consume_body(g)
            ).status_code
        )
    outcomes.append(
        client.post(
            f"/v1/release/{grants[4]['grant_id']}",
            json={**_consume_body(grants[4]), "data_id": DATA_ID},
        ).status_code
    )
    assert outcomes == [200, 200, 200, 200, 200]
    # The sixth valid request, on any of the three endpoints, is 429.
    sixth = client.post(
        f"/v1/release-grants/{grants[5]['grant_id']}/consume",
        json=_consume_body(grants[5]),
    )
    _assert_429(sixth)


def test_429_body_is_compact_json_with_positive_integer(client):
    decision = _decision(client)
    grants = [_grant(client, decision["decision_id"], data_id=f"d{i}") for i in range(6)]
    for g in grants[:5]:
        client.post(f"/v1/release-grants/{g['grant_id']}/consume", json=_consume_body(g))
    response = client.post(
        f"/v1/release-grants/{grants[5]['grant_id']}/consume",
        json=_consume_body(grants[5]),
    )
    _assert_429(response)
    # A repeated 429 in the same window uses the same computation and
    # does not extend the window or consume further budget.
    again = client.post(
        f"/v1/release-grants/{grants[5]['grant_id']}/consume",
        json=_consume_body(grants[5]),
    )
    _assert_429(again)
    assert again.json()["retry_after_seconds"] <= response.json()["retry_after_seconds"]


def test_business_failures_still_consume_budget(client):
    # 404 (unknown grant), 401 (wrong capability) and 409 (already
    # settled) each consume one unit: five such requests exhaust the
    # window and the sixth is 429.
    decision = _decision(client)
    grant = _grant(client, decision["decision_id"])
    unknown = "00000000-0000-0000-0000-000000000000"
    wrong = ("B" if grant["capability"][0] != "B" else "C") + grant["capability"][1:]

    assert client.post(
        f"/v1/release-grants/{unknown}/consume", json=_consume_body(grant)
    ).status_code == 404  # 1
    assert client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json=_consume_body({"capability": wrong}),
    ).status_code == 401  # 2
    assert client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume", json=_consume_body(grant)
    ).status_code == 200  # 3
    assert client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume", json=_consume_body(grant)
    ).status_code == 409  # 4
    assert client.post(
        f"/v1/release-grants/{grant['grant_id']}/revoke", json=_consume_body(grant)
    ).status_code == 409  # 5
    sixth = client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume", json=_consume_body(grant)
    )
    _assert_429(sixth)


def test_field_validation_errors_do_not_consume_budget(client):
    decision = _decision(client)
    grants = [_grant(client, decision["decision_id"], data_id=f"d{i}") for i in range(5)]
    target = grants[0]["grant_id"]
    # Missing, blank, wrong-typed and malformed fields, plus an illegal
    # path identifier on revoke: all 422, none consuming budget.
    assert client.post(
        f"/v1/release-grants/{target}/consume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).status_code == 422
    assert client.post(
        f"/v1/release-grants/{target}/consume",
        json={"tenant_id": "  ", "workload_id": WORKLOAD, "capability": "abc"},
    ).status_code == 422
    assert client.post(
        f"/v1/release-grants/{target}/consume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD, "capability": 42},
    ).status_code == 422
    assert client.post(
        f"/v1/release-grants/{target}/consume",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD, "capability": "not*b64url"},
    ).status_code == 422
    assert client.post(
        "/v1/release-grants/not-a-uuid/revoke", json=_consume_body(grants[0])
    ).status_code == 422
    # The full budget of five is still available afterwards.
    for g in grants:
        assert client.post(
            f"/v1/release-grants/{g['grant_id']}/consume", json=_consume_body(g)
        ).status_code == 200


def test_budget_is_strictly_per_scope(client):
    decision = _decision(client)
    other = _decision(client, tenant="tenant-b", workload="workload-2")
    grants_a = [_grant(client, decision["decision_id"], data_id=f"a{i}") for i in range(6)]
    grant_b = _grant(
        client, other["decision_id"], tenant="tenant-b", workload="workload-2"
    )
    for g in grants_a[:5]:
        assert client.post(
            f"/v1/release-grants/{g['grant_id']}/consume", json=_consume_body(g)
        ).status_code == 200
    # Scope A is exhausted...
    _assert_429(
        client.post(
            f"/v1/release-grants/{grants_a[5]['grant_id']}/consume",
            json=_consume_body(grants_a[5]),
        )
    )
    # ...but scope B is untouched.
    assert client.post(
        f"/v1/release-grants/{grant_b['grant_id']}/consume",
        json=_consume_body(grant_b, tenant="tenant-b", workload="workload-2"),
    ).status_code == 200


def test_429_writes_no_state_no_audit_and_releases_no_payload(app, client):
    decision = _decision(client)
    client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": DATA_ID,
            "payload": "never released",
        },
    )
    grants = [_grant(client, decision["decision_id"], data_id=DATA_ID) for _ in range(6)]
    for g in grants[:5]:
        assert client.post(
            f"/v1/release-grants/{g['grant_id']}/consume", json=_consume_body(g)
        ).status_code == 200
    with app.state.session_factory() as session:
        events_before = session.query(AuditEvent).count()
    # The sixth request targets a still-pending grant via release: it must
    # not decrypt, release, settle or audit anything.
    blocked = client.post(
        f"/v1/release/{grants[5]['grant_id']}",
        json={**_consume_body(grants[5]), "data_id": DATA_ID},
    )
    _assert_429(blocked)
    assert "payload" not in blocked.text
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grants[5]["grant_id"])
        assert row.status == "pending"
        assert row.consumed_at is None
        assert session.query(AuditEvent).count() == events_before


def test_concurrent_requests_admit_exactly_five(app):
    client = TestClient(app)
    decision = _decision(client)
    grants = [_grant(client, decision["decision_id"], data_id=f"d{i}") for i in range(16)]

    def consume(g):
        return TestClient(app).post(
            f"/v1/release-grants/{g['grant_id']}/consume", json=_consume_body(g)
        ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(consume, grants))
    assert statuses.count(200) == 5
    assert statuses.count(429) == 11
    with app.state.session_factory() as session:
        window = session.query(RateLimitWindow).one()
        assert window.used == 5


def test_budget_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/rate_limit_restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    decision = _decision(client1)
    grants = [_grant(client1, decision["decision_id"], data_id=f"d{i}") for i in range(6)]
    for g in grants[:5]:
        assert client1.post(
            f"/v1/release-grants/{g['grant_id']}/consume", json=_consume_body(g)
        ).status_code == 200
    app1.state.engine.dispose()

    # A fresh process over the same database still sees the spent window.
    app2 = create_app(url)
    try:
        client2 = TestClient(app2)
        _assert_429(
            client2.post(
                f"/v1/release-grants/{grants[5]['grant_id']}/consume",
                json=_consume_body(grants[5]),
            )
        )
    finally:
        app2.state.engine.dispose()
