"""Tests for GET /v1/release-grants: scoped, cursor-stable grant audit.

The endpoint is read-only: it lists the release-grant audit rows for one
tenant/workload, optionally narrowed by grant/decision/data identifier,
status and an issued-at window, with stable grant_id-ascending keyset
pagination. It never returns the plaintext capability (only its
SHA-256 digest) and never writes state.
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
from proof_release.db import ReleaseGrant
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
    application = create_app(f"sqlite:///{tmp_path}/grant_audits.db")
    yield application
    application.state.engine.dispose()


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


def _decision(client, *, tenant=TENANT, workload=WORKLOAD, satisfied=True):
    """Drive challenge -> evidence -> verify -> decision; return decision."""
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    claims = {"m": "x"}
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
    rule = (
        {"claim": "m", "equals": "x"}
        if satisfied
        else {"claim": "m", "equals": "other"}
    )
    policy = client.post(
        "/v1/policies",
        json={"tenant_id": tenant, "workload_id": workload, "name": "r", "rule": rule},
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


def _mint(client, decision_id, *, data_id="data-1", tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/release-grants",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "decision_id": decision_id,
            "data_id": data_id,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _query(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    return client.get(
        "/v1/release-grants",
        params={"tenant_id": tenant, "workload_id": workload, **params},
    )


def _walk(client, *, page_size=None, monkeypatch=None, **params):
    """Walk every page and return the ordered list of grant rows seen."""
    token = None
    rows = []
    for _ in range(50):
        query_params = dict(params)
        if token is not None:
            query_params["cursor"] = token
        if page_size is not None:
            monkeypatch.setattr(
                app_module, "RELEASE_GRANT_AUDIT_PAGE_SIZE", page_size
            )
        page = _query(client, **query_params)
        assert page.status_code == 200, page.text
        data = page.json()
        rows.extend(data["grants"])
        if data["complete"]:
            assert data["next_cursor"] == ""
            return rows
        token = data["next_cursor"]
        assert token
    raise AssertionError("pagination never completed")  # pragma: no cover


# --- request validation ----------------------------------------------------


def test_query_requires_scope_parameters(client):
    assert client.get("/v1/release-grants").status_code == 422
    assert (
        client.get("/v1/release-grants", params={"workload_id": WORKLOAD}).status_code
        == 422
    )
    assert (
        client.get("/v1/release-grants", params={"tenant_id": TENANT}).status_code
        == 422
    )


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"workload_id": "\t"},
        {"grant_id": ""},
        {"grant_id": "   "},
        {"decision_id": ""},
        {"decision_id": "  "},
        {"data_id": ""},
        {"data_id": "   "},
        {"grant_id": "not-a-uuid"},
        {"grant_id": "abc123"},
        {"grant_id": ZERO_UUID[:-1] + "Z"},
        {"decision_id": "nope"},
        {"decision_id": "  " + ZERO_UUID},
        {"status": ""},
        {"status": "   "},
        {"status": "PENDING"},
        {"status": "allowed"},
        {"status": "expired"},
        {"issued_after": "not-a-timestamp"},
        {"issued_after": "2026-01-01T00:00:00"},
        {"issued_before": "2026-01-01"},
        {"issued_after": ""},
        {"issued_before": "  "},
        {
            "issued_after": "2026-01-02T00:00:00Z",
            "issued_before": "2026-01-01T00:00:00Z",
        },
        {
            "issued_after": "2026-01-01T01:00:00+01:00",
            "issued_before": "2026-01-01T00:00:00Z",
        },
        {"cursor": "   "},
        {"cursor": "not base64!!"},
        {"cursor": "AAA="},
    ],
)
def test_query_rejects_invalid_parameters(client, params):
    response = _query(client, **params)
    assert response.status_code == 422, response.text


def test_query_rejects_non_utc_offset_even_if_instant_matches(client):
    # Both spellings name the same UTC instant, but the contract requires
    # an explicit UTC offset on every bound.
    response = _query(
        client,
        issued_after="2026-01-01T02:00:00+02:00",
        issued_before="2026-01-01T00:00:00Z",
    )
    assert response.status_code == 422


# --- unknown / cross-scope identifiers -------------------------------------


def test_unknown_grant_identifier_returns_404(client):
    response = _query(client, grant_id=ZERO_UUID)
    assert response.status_code == 404


def test_unknown_decision_identifier_returns_404(client):
    response = _query(client, decision_id=ZERO_UUID)
    assert response.status_code == 404


def test_unknown_data_identifier_returns_404(client):
    response = _query(client, data_id="never-used")
    assert response.status_code == 404


def test_cross_scope_grant_identifier_returns_404(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
    assert (
        _query(client, tenant=OTHER_TENANT, grant_id=grant["grant_id"]).status_code
        == 404
    )
    assert (
        _query(client, workload=OTHER_WORKLOAD, grant_id=grant["grant_id"]).status_code
        == 404
    )


def test_cross_scope_decision_identifier_returns_404(client):
    decision = _decision(client)
    assert (
        _query(client, tenant=OTHER_TENANT, decision_id=decision["decision_id"]).status_code
        == 404
    )
    assert (
        _query(
            client, workload=OTHER_WORKLOAD, decision_id=decision["decision_id"]
        ).status_code
        == 404
    )


def test_data_identifier_only_known_in_another_scope_returns_404(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="scoped-data")
    # The identifier exists for TENANT/WORKLOAD but not for other scopes.
    assert _query(client, tenant=OTHER_TENANT, data_id="scoped-data").status_code == 404
    assert (
        _query(client, workload=OTHER_WORKLOAD, data_id="scoped-data").status_code
        == 404
    )


# --- happy path / shape ----------------------------------------------------


def test_empty_scope_returns_empty_completed_page(client):
    response = _query(client)
    assert response.status_code == 200
    data = response.json()
    assert data == {"grants": [], "next_cursor": "", "complete": True}


def test_grant_rows_have_exact_audit_shape(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"], data_id="data-7")

    data = _query(client).json()
    assert data["complete"] is True
    assert data["next_cursor"] == ""
    assert len(data["grants"]) == 1
    row = data["grants"][0]
    assert set(row) == {
        "grant_id",
        "decision_id",
        "data_id",
        "status",
        "capability_sha256",
        "issued_at",
        "expires_at",
        "consumed_at",
        "revoked_at",
    }
    assert row["grant_id"] == grant["grant_id"]
    assert row["decision_id"] == decision["decision_id"]
    assert row["data_id"] == "data-7"
    assert row["status"] == "pending"
    # Only the SHA-256 digest is exposed; it matches the one-time capability.
    assert row["capability_sha256"] == hashlib.sha256(
        grant["capability"].encode("ascii")
    ).hexdigest()
    assert len(row["capability_sha256"]) == 64
    for field in (
        "grant_id",
        "decision_id",
        "data_id",
        "status",
        "capability_sha256",
        "issued_at",
        "expires_at",
    ):
        assert isinstance(row[field], str) and row[field]
    for stamp in ("issued_at", "expires_at"):
        parsed = datetime.fromisoformat(row[stamp])
        assert parsed.utcoffset() == timedelta(0)
    assert row["consumed_at"] is None
    assert row["revoked_at"] is None


def test_rows_are_ordered_by_grant_id_ascending(client):
    decision = _decision(client)
    minted = [_mint(client, decision["decision_id"], data_id=f"d{i}") for i in range(6)]
    rows = _query(client).json()["grants"]
    returned = [row["grant_id"] for row in rows]
    assert returned == sorted(g["grant_id"] for g in minted)


def test_consumed_and_revoked_rows_report_fixed_status_and_timestamps(client):
    decision = _decision(client)
    consumed = _mint(client, decision["decision_id"], data_id="c")
    revoked = _mint(client, decision["decision_id"], data_id="r")
    pending = _mint(client, decision["decision_id"], data_id="p")

    cr = client.post(
        f"/v1/release-grants/{consumed['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": consumed["capability"],
        },
    )
    assert cr.status_code == 200
    rr = client.post(
        f"/v1/release-grants/{revoked['grant_id']}/revoke",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": revoked["capability"],
        },
    )
    assert rr.status_code == 200

    rows = {
        row["grant_id"]: row for row in _query(client).json()["grants"]
    }
    by_id = {
        consumed["grant_id"]: ("consumed", "consumed_at"),
        revoked["grant_id"]: ("revoked", "revoked_at"),
        pending["grant_id"]: ("pending", None),
    }
    for grant_id, (status, stamp_field) in by_id.items():
        row = rows[grant_id]
        assert row["status"] == status
        if stamp_field is None:
            assert row["consumed_at"] is None
            assert row["revoked_at"] is None
        else:
            stamp = row[stamp_field]
            assert stamp is not None
            assert datetime.fromisoformat(stamp).utcoffset() == timedelta(0)
            other = "revoked_at" if stamp_field == "consumed_at" else "consumed_at"
            assert row[other] is None


def test_audit_never_contains_plaintext_capability(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
    response = _query(client)
    assert grant["capability"] not in response.text
    # Only a 64-char lowercase-hex digest appears.
    row = response.json()["grants"][0]
    assert all(c in "0123456789abcdef" for c in row["capability_sha256"])


def test_response_is_compact_json_without_floats(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"])
    response = _query(client)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] != b"\n"
    assert b", " not in raw
    assert b": " not in raw
    parsed = json.loads(raw)
    assert isinstance(parsed["complete"], bool)
    assert isinstance(parsed["next_cursor"], str)
    assert isinstance(parsed["grants"], list)

    def _check_types(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            assert not isinstance(value, bool)
            return
        if isinstance(value, float):  # pragma: no cover - structural guard
            raise AssertionError("float present in audit response")
        assert value is None or isinstance(value, str)

    for row in parsed["grants"]:
        for value in row.values():
            _check_types(value)


# --- filtering -------------------------------------------------------------


def test_filter_by_grant_id_returns_exactly_that_grant(client):
    decision = _decision(client)
    grants = [_mint(client, decision["decision_id"], data_id=f"d{i}") for i in range(3)]
    target = sorted(g["grant_id"] for g in grants)[1]
    data = _query(client, grant_id=target).json()
    assert [row["grant_id"] for row in data["grants"]] == [target]


def test_filter_by_decision_id(client):
    first = _decision(client)
    second = _decision(client)
    _mint(client, first["decision_id"], data_id="a")
    _mint(client, first["decision_id"], data_id="b")
    _mint(client, second["decision_id"], data_id="c")
    rows = _query(client, decision_id=first["decision_id"]).json()["grants"]
    assert {row["decision_id"] for row in rows} == {first["decision_id"]}
    assert len(rows) == 2


def test_filter_by_data_id(client):
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="same")
    _mint(client, decision["decision_id"], data_id="same")
    _mint(client, decision["decision_id"], data_id="other")
    rows = _query(client, data_id="same").json()["grants"]
    assert len(rows) == 2
    assert {row["data_id"] for row in rows} == {"same"}


def test_data_identifier_known_only_via_a_grant_is_accepted(client):
    # A grant may reference a data item for which no envelope exists; the
    # identifier is still known in this scope through the audit row.
    decision = _decision(client)
    _mint(client, decision["decision_id"], data_id="grant-only-item")
    response = _query(client, data_id="grant-only-item")
    assert response.status_code == 200
    assert len(response.json()["grants"]) == 1


@pytest.mark.parametrize("status", ["pending", "consumed", "revoked"])
def test_filter_by_status(client, status):
    decision = _decision(client)
    consumed = _mint(client, decision["decision_id"], data_id="c")
    revoked = _mint(client, decision["decision_id"], data_id="r")
    _mint(client, decision["decision_id"], data_id="p")
    client.post(
        f"/v1/release-grants/{consumed['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": consumed["capability"],
        },
    )
    client.post(
        f"/v1/release-grants/{revoked['grant_id']}/revoke",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": revoked["capability"],
        },
    )
    rows = _query(client, status=status).json()["grants"]
    assert rows and {row["status"] for row in rows} == {status}


def test_filter_by_issued_time_window_is_inclusive(client):
    decision = _decision(client)
    grant = _mint(client, decision["decision_id"])
    issued_at = grant["issued_at"]
    issued_dt = datetime.fromisoformat(issued_at)

    # Both bounds equal to the issuance instant include the row.
    rows = _query(
        client, issued_after=issued_at, issued_before=issued_at
    ).json()["grants"]
    assert [row["grant_id"] for row in rows] == [grant["grant_id"]]

    # +00:00 and Z spellings of the same UTC instant are equivalent.
    plus_zero = issued_dt.isoformat().replace("+00:00", "Z")
    rows = _query(client, issued_after=plus_zero).json()["grants"]
    assert [row["grant_id"] for row in rows] == [grant["grant_id"]]

    # An upper bound strictly in the past excludes every row.
    past = (issued_dt - timedelta(seconds=1)).isoformat()
    assert _query(client, issued_before=past).json()["grants"] == []
    # A lower bound strictly in the future excludes every row.
    future = (issued_dt + timedelta(seconds=1)).isoformat()
    assert _query(client, issued_after=future).json()["grants"] == []


def test_query_is_scoped_to_tenant_and_workload(client):
    decision_a = _decision(client)
    _mint(client, decision_a["decision_id"], data_id="a")
    decision_b = _decision(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    )
    _mint(
        client,
        decision_b["decision_id"],
        data_id="b",
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
    )
    rows = _query(client).json()["grants"]
    assert len(rows) == 1
    assert rows[0]["data_id"] == "a"
    rows = _query(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()[
        "grants"
    ]
    assert len(rows) == 1
    assert rows[0]["data_id"] == "b"


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_grant_once_in_order(client, monkeypatch):
    decision = _decision(client)
    minted = [
        _mint(client, decision["decision_id"], data_id=f"d{i:03d}")
        for i in range(7)
    ]
    expected = sorted(g["grant_id"] for g in minted)
    rows = _walk(client, page_size=3, monkeypatch=monkeypatch)
    assert [row["grant_id"] for row in rows] == expected


def test_page_carries_cursor_and_complete_flag(client, monkeypatch):
    monkeypatch.setattr(app_module, "RELEASE_GRANT_AUDIT_PAGE_SIZE", 2)
    decision = _decision(client)
    for i in range(5):
        _mint(client, decision["decision_id"], data_id=f"d{i}")

    first = _query(client).json()
    assert len(first["grants"]) == 2
    assert first["complete"] is False
    assert first["next_cursor"]

    second = _query(client, cursor=first["next_cursor"]).json()
    assert len(second["grants"]) == 2
    assert second["complete"] is False
    assert [a["grant_id"] for a in first["grants"]] < [
        b["grant_id"] for b in second["grants"]
    ]

    # A scope that ends exactly at a page boundary: the limit+1 probe
    # distinguishes a full final page from a page with more to come.
    last = _query(client, cursor=second["next_cursor"]).json()
    assert len(last["grants"]) == 1
    assert last["complete"] is True
    assert last["next_cursor"] == ""


def test_replaying_same_cursor_returns_identical_page(client, monkeypatch):
    monkeypatch.setattr(app_module, "RELEASE_GRANT_AUDIT_PAGE_SIZE", 2)
    decision = _decision(client)
    for i in range(5):
        _mint(client, decision["decision_id"], data_id=f"d{i}")
    cursor = _query(client).json()["next_cursor"]
    first = _query(client, cursor=cursor)
    second = _query(client, cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_string_cursor_equals_default(client, monkeypatch):
    monkeypatch.setattr(app_module, "RELEASE_GRANT_AUDIT_PAGE_SIZE", 2)
    decision = _decision(client)
    for i in range(3):
        _mint(client, decision["decision_id"], data_id=f"d{i}")
    default = _query(client).content
    explicit = _query(client, cursor="").content
    assert default == explicit


def test_tampered_forged_or_cross_scope_cursor_returns_422(
    client, monkeypatch
):
    monkeypatch.setattr(app_module, "RELEASE_GRANT_AUDIT_PAGE_SIZE", 1)
    decision = _decision(client)
    for i in range(3):
        _mint(client, decision["decision_id"], data_id=f"d{i}")
    cursor = _query(client).json()["next_cursor"]
    assert cursor

    # Tamper with the last characters while staying within base64url.
    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _query(client, cursor=tampered).status_code == 422

    # Opaque garbage that still decodes as base64url, with a 32-byte
    # suffix shaped like a MAC.
    forged = b64url_encode(b'{"k":"release-grant-audit-v1"}' + b"0" * 32)
    assert _query(client, cursor=forged).status_code == 422

    # Cross-scope replay.
    assert _query(client, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert _query(client, workload=OTHER_WORKLOAD, cursor=cursor).status_code == 422


def test_cursor_cannot_cross_filter_conditions(client, monkeypatch):
    monkeypatch.setattr(app_module, "RELEASE_GRANT_AUDIT_PAGE_SIZE", 1)
    decision = _decision(client)
    for i in range(3):
        _mint(client, decision["decision_id"], data_id=f"d{i}")
    cursor = _query(client).json()["next_cursor"]

    # The cursor authenticates the full filter set, not only the scope.
    assert _query(client, status="pending", cursor=cursor).status_code == 422
    assert (
        _query(client, decision_id=decision["decision_id"], cursor=cursor).status_code
        == 422
    )
    assert _query(client, data_id="d0", cursor=cursor).status_code == 422
    assert (
        _query(
            client,
            issued_after="2000-01-01T00:00:00Z",
            cursor=cursor,
        ).status_code
        == 422
    )
    assert (
        _query(
            client,
            issued_before="2100-01-01T00:00:00Z",
            cursor=cursor,
        ).status_code
        == 422
    )


def test_cursor_minted_under_one_filter_walks_that_filter_stably(
    client, monkeypatch
):
    monkeypatch.setattr(app_module, "RELEASE_GRANT_AUDIT_PAGE_SIZE", 2)
    decision = _decision(client)
    for data_id in ("x", "x", "y"):
        _mint(client, decision["decision_id"], data_id=data_id)
    rows = _walk(
        client, page_size=2, monkeypatch=monkeypatch, data_id="x"
    )
    assert len(rows) == 2
    assert {row["data_id"] for row in rows} == {"x"}


def test_rewrap_batch_cursor_is_not_accepted_as_grant_cursor(client):
    # Both cursor families share the HMAC secret but carry a kind tag; a
    # rewrap-batch cursor replayed on the audit endpoint is invalid.
    from proof_release.app import _encode_cursor

    foreign = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _query(client, cursor=foreign).status_code == 422


# --- read-only behaviour ---------------------------------------------------


def test_query_writes_no_state(client, app):
    decision = _decision(client)
    grants = [_mint(client, decision["decision_id"], data_id=f"d{i}") for i in range(3)]
    for _ in range(3):
        assert _query(client).status_code == 200
    # Walking pages also leaves every row exactly as minted.
    with app.state.session_factory() as session:
        rows = session.query(ReleaseGrant).order_by(ReleaseGrant.grant_id).all()
        assert len(rows) == 3
        for row in rows:
            assert row.status == "pending"
            assert row.consumed_at is None
            assert row.revoked_at is None


# --- concurrency -----------------------------------------------------------


def test_queries_interleaved_with_settlement_only_see_committed_state(app):
    client = TestClient(app)
    decision = _decision(client)
    grants = [_mint(client, decision["decision_id"], data_id=f"d{i}") for i in range(10)]

    settle_targets = grants[:4]

    def settle():
        local = TestClient(app)
        for index, grant in enumerate(settle_targets):
            path = (
                f"/v1/release-grants/{grant['grant_id']}/"
                + ("consume" if index % 2 == 0 else "revoke")
            )
            response = local.post(
                path,
                json={
                    "tenant_id": TENANT,
                    "workload_id": WORKLOAD,
                    "capability": grant["capability"],
                },
            )
            assert response.status_code == 200

    def audit():
        local = TestClient(app)
        captured = []
        for _ in range(20):
            response = local.get(
                "/v1/release-grants",
                params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            )
            assert response.status_code == 200
            data = response.json()
            # Every observed status is one of the fixed terminal/pending
            # codes; there is never a half-settled row.
            for row in data["grants"]:
                assert row["status"] in ("pending", "consumed", "revoked")
                if row["status"] == "pending":
                    assert row["consumed_at"] is None
                    assert row["revoked_at"] is None
            captured.append(data)
        return captured

    with ThreadPoolExecutor(max_workers=4) as pool:
        pages_future = pool.submit(audit)
        settle_future = pool.submit(settle)
        pages = pages_future.result()
        settle_future.result()

    # All ten grants remain visible exactly once after settlement.
    final_ids = [row["grant_id"] for row in _query(client).json()["grants"]]
    assert final_ids == sorted(g["grant_id"] for g in grants)
    # During the interleaving, pages were always internally ordered.
    for data in pages:
        ids = [row["grant_id"] for row in data["grants"]]
        assert ids == sorted(ids)


def test_replaying_a_cursor_after_settlement_keeps_stable_order(app):
    client = TestClient(app)
    decision = _decision(client)
    grants = [_mint(client, decision["decision_id"], data_id=f"d{i}") for i in range(6)]
    expected_order = sorted(g["grant_id"] for g in grants)

    import proof_release.app as app_module

    app_module.RELEASE_GRANT_AUDIT_PAGE_SIZE = 3
    try:
        first = _query(client).json()
        assert first["complete"] is False
        # The cursor is the exclusive start of the second page.
        cursor = first["next_cursor"]
        second_before = _query(client, cursor=cursor).json()
        assert len(second_before["grants"]) == 3
        second_ids = [row["grant_id"] for row in second_before["grants"]]

        # Settle grants: two on the first page and one on the second.
        sorted_grants = sorted(grants, key=lambda g: g["grant_id"])
        to_consume = [sorted_grants[0], sorted_grants[2], sorted_grants[3]]
        for grant in to_consume:
            consumed = client.post(
                f"/v1/release-grants/{grant['grant_id']}/consume",
                json={
                    "tenant_id": TENANT,
                    "workload_id": WORKLOAD,
                    "capability": grant["capability"],
                },
            )
            assert consumed.status_code == 200

        # Replaying the same cursor returns the same second-page grant ids
        # in the same order; only the committed status of one row differs.
        second_after = _query(client, cursor=cursor).json()
        assert [row["grant_id"] for row in second_after["grants"]] == second_ids
        statuses = {row["grant_id"]: row["status"] for row in second_after["grants"]}
        assert statuses[sorted_grants[3]["grant_id"]] == "consumed"

        # Walking from the beginning still visits every grant once with
        # no duplication or gap.
        rows = []
        token = None
        for _ in range(10):
            data = _query(client, **({"cursor": token} if token else {})).json()
            rows.extend(row["grant_id"] for row in data["grants"])
            if data["complete"]:
                break
            token = data["next_cursor"]
        assert rows == expected_order
    finally:
        app_module.RELEASE_GRANT_AUDIT_PAGE_SIZE = 100


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_returns_no_page(app, client):
    from sqlalchemy import text

    decision = _decision(client)
    _mint(client, decision["decision_id"])
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE release_grants"))
    response = _query(client)
    assert response.status_code == 500


# --- persistence -----------------------------------------------------------


def test_audit_is_available_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    decision = _decision(client1)
    grant = _mint(client1, decision["decision_id"], data_id="persist")
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    data = _query(client2).json()
    assert [row["grant_id"] for row in data["grants"]] == [grant["grant_id"]]
    assert data["grants"][0]["status"] == "pending"
    assert data["complete"] is True
    second.state.engine.dispose()
