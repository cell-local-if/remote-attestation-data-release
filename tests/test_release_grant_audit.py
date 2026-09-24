"""Tests for GET /v1/release-grants: scoped, filterable, stable-key-set
paged auditing of release grants.

The endpoint is read-only: every test here runs against grants minted
through the normal challenge -> evidence -> verify -> decision -> grant
flow, and the audit query must never create, consume, revoke or
otherwise settle anything.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from proof_release.app import create_app
from proof_release.db import ReleaseGrant
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
SECRET = "unit-test-secret"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    application = create_app(f"sqlite:///{tmp_path}/grant_audit.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


# --- flow helpers ----------------------------------------------------------


def _mac(nonce: str, claims: dict, *, tenant=TENANT, workload=WORKLOAD) -> str:
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


def _evidence(nonce: str, claims: dict, **scope) -> str:
    return json.dumps(
        {"nonce": nonce, "claims": claims, "mac": _mac(nonce, claims, **scope)}
    )


def _decision(
    client,
    *,
    tenant=TENANT,
    workload=WORKLOAD,
    claims=None,
    satisfied=True,
):
    """Drive the full challenge -> evidence -> verify -> decision flow."""
    claims = claims if claims is not None else {"m": "x"}
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": tenant, "workload_id": workload},
    ).json()
    evidence = _evidence(
        created["nonce"], claims, tenant=tenant, workload=workload
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
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "name": "release",
            "rule": rule,
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


def _grant(client, decision_id, *, data_id="data-1", tenant=TENANT, workload=WORKLOAD):
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


def _seed_grants(client, count, *, data_ids=None, start=0):
    decision_id = _decision(client)["decision_id"]
    grants = []
    for i in range(start, start + count):
        data_id = data_ids[i - start] if data_ids is not None else f"data-{i}"
        grants.append(_grant(client, decision_id, data_id=data_id))
    return grants


def _list(client, **params):
    params = {"tenant_id": TENANT, "workload_id": WORKLOAD, **params}
    return client.get("/v1/release-grants", params=params)


def _walk(client, **params):
    """Page through the whole filtered scope, returning all records."""
    cursor = None
    records = []
    pages = []
    for _ in range(50):
        page_params = dict(params)
        if cursor is not None:
            page_params["cursor"] = cursor
        response = _list(client, **page_params)
        assert response.status_code == 200, response.text
        data = response.json()
        pages.append(data)
        records.extend(data["grants"])
        if data["complete"]:
            return records, pages
        cursor = data["next_cursor"]
    raise AssertionError("pagination did not complete")  # pragma: no cover


def _consume(client, grant):
    return client.post(
        f"/v1/release-grants/{grant['grant_id']}/consume",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )


def _revoke(client, grant):
    return client.post(
        f"/v1/release-grants/{grant['grant_id']}/revoke",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "capability": grant["capability"],
        },
    )


def _set_issued_at(app, grant_id, value):
    with app.state.session_factory() as session:
        row = session.get(ReleaseGrant, grant_id)
        row.issued_at = value
        session.commit()


# --- request validation ----------------------------------------------------


def test_missing_scope_parameters_return_422(client):
    assert _list(client, tenant_id=None).status_code == 422
    assert _list(client, workload_id=None).status_code == 422
    assert client.get("/v1/release-grants").status_code == 422


@pytest.mark.parametrize("scope_field", ["tenant_id", "workload_id"])
def test_blank_scope_parameters_return_422(client, scope_field):
    params = {"tenant_id": TENANT, "workload_id": WORKLOAD, scope_field: "   "}
    assert client.get("/v1/release-grants", params=params).status_code == 422


def test_blank_optional_identifiers_return_422(client):
    base = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    for field in ("grant_id", "decision_id", "data_id"):
        assert (
            client.get(
                "/v1/release-grants", params={**base, field: "   "}
            ).status_code
            == 422
        )


@pytest.mark.parametrize("grant_id", ["not-a-uuid", "abc123", "00000000-0000-0000-0000-00000000000Z"])
def test_malformed_grant_identifier_returns_422(client, grant_id):
    assert _list(client, grant_id=grant_id).status_code == 422


@pytest.mark.parametrize(
    "decision_id",
    ["not-a-uuid", "abc", "00000000-0000-0000-0000-00000000000Z"],
)
def test_malformed_decision_identifier_returns_422(client, decision_id):
    assert _list(client, decision_id=decision_id).status_code == 422


@pytest.mark.parametrize("status", ["", "   ", "expired", "PENDING", "allowed", "pending "])
def test_illegal_status_returns_422(client, status):
    assert _list(client, status=status).status_code == 422


@pytest.mark.parametrize("limit", [0, -1, 201, 202])
def test_limit_out_of_bounds_returns_422(client, limit):
    assert _list(client, limit=limit).status_code == 422


@pytest.mark.parametrize("limit", [True, False, 1.5, "abc"])
def test_limit_wrong_type_returns_422(client, limit):
    assert _list(client, limit=limit).status_code == 422


@pytest.mark.parametrize(
    "stamp",
    [
        "not a timestamp",
        "2024-01-01T00:00:00",  # naive: offset is mandatory
        "2024-01-01",
        "2024-13-01T00:00:00Z",
    ],
)
def test_malformed_issued_bounds_return_422(client, stamp):
    assert _list(client, issued_after=stamp).status_code == 422
    assert _list(client, issued_before=stamp).status_code == 422


def test_reversed_issued_bounds_return_422(client):
    response = _list(
        client,
        issued_after="2024-06-01T00:00:00Z",
        issued_before="2024-01-01T00:00:00Z",
    )
    assert response.status_code == 422


def test_equal_issued_bounds_are_accepted(client):
    response = _list(
        client,
        issued_after="2024-01-01T00:00:00Z",
        issued_before="2024-01-01T00:00:00Z",
    )
    assert response.status_code == 200


@pytest.mark.parametrize("cursor", ["   ", "bad token!!", "AAA=", "12 34"])
def test_malformed_cursor_returns_422(client, cursor):
    assert _list(client, cursor=cursor).status_code == 422


def test_validation_errors_write_no_state(app, client):
    _seed_grants(client, 2)
    for kwargs in (
        {"status": "bogus"},
        {"grant_id": "not-a-uuid"},
        {"issued_after": "2024-01-01T00:00:00"},
        {"cursor": "!!!"},
        {
            "issued_after": "2024-06-01T00:00:00Z",
            "issued_before": "2024-01-01T00:00:00Z",
        },
    ):
        assert _list(client, **kwargs).status_code == 422
    with app.state.session_factory() as session:
        assert session.query(ReleaseGrant).count() == 2


# --- 404 semantics ---------------------------------------------------------


def test_unknown_grant_returns_404(client):
    response = _list(
        client, grant_id="00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 404


def test_cross_scope_grant_returns_404(client):
    grants = _seed_grants(client, 1)
    grant_id = grants[0]["grant_id"]
    assert (
        client.get(
            "/v1/release-grants",
            params={
                "tenant_id": OTHER_TENANT,
                "workload_id": WORKLOAD,
                "grant_id": grant_id,
            },
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/v1/release-grants",
            params={
                "tenant_id": TENANT,
                "workload_id": OTHER_WORKLOAD,
                "grant_id": grant_id,
            },
        ).status_code
        == 404
    )


def test_unknown_decision_returns_404(client):
    assert (
        _list(
            client, decision_id="00000000-0000-0000-0000-000000000000"
        ).status_code
        == 404
    )


def test_cross_scope_decision_returns_404(client):
    decision_id = _decision(client)["decision_id"]
    other_scope = {
        "tenant_id": OTHER_TENANT,
        "workload_id": WORKLOAD,
        "decision_id": decision_id,
    }
    assert client.get("/v1/release-grants", params=other_scope).status_code == 404


def test_unknown_data_identifier_returns_404(client):
    assert _list(client, data_id="never-used").status_code == 404


def test_data_identifier_only_known_in_another_scope_returns_404(client):
    # A grant in the other scope names the data id; it must not be
    # discoverable as "known" from this scope.
    other_decision = _decision(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    )
    _grant(
        client,
        other_decision["decision_id"],
        data_id="shared-name",
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
    )
    assert _list(client, data_id="shared-name").status_code == 404


def test_known_grant_filtered_out_by_other_constraints_is_empty_not_404(client):
    grant = _seed_grants(client, 1)[0]
    # The grant exists in scope, but an impossible time window matches
    # nothing: existence and filtering are separate outcomes.
    response = _list(
        client,
        grant_id=grant["grant_id"],
        issued_before="2000-01-01T00:00:00Z",
    )
    assert response.status_code == 200
    data = response.json()
    assert data["grants"] == []
    assert data["complete"] is True
    assert data["next_cursor"] == ""


# --- empty scope and response shape ----------------------------------------


def test_empty_scope_completes_immediately_with_compact_shape(client):
    response = _list(client)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"grants", "next_cursor", "complete"}
    assert data == {"grants": [], "next_cursor": "", "complete": True}


def test_get_with_empty_body_is_accepted(client):
    # The request body is always empty; an explicit empty body changes
    # nothing.
    response = client.request(
        "GET",
        "/v1/release-grants",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"",
    )
    assert response.status_code == 200


def test_records_have_exact_fields_and_types(client):
    capability = _seed_grants(client, 1)[0]["capability"]
    data = _list(client).json()
    (record,) = data["grants"]
    assert set(record) == {
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
    for key in (
        "grant_id",
        "decision_id",
        "data_id",
        "status",
        "capability_sha256",
        "issued_at",
        "expires_at",
    ):
        assert isinstance(record[key], str), key
    assert record["status"] == "pending"
    assert record["consumed_at"] is None
    assert record["revoked_at"] is None
    assert record["capability_sha256"] == hashlib.sha256(
        capability.encode("ascii")
    ).hexdigest()
    for stamp_key in ("issued_at", "expires_at"):
        parsed = datetime.fromisoformat(record[stamp_key])
        assert parsed.utcoffset() == timedelta(0)


def test_records_reflect_consumed_and_revoked_states(client):
    grants = _seed_grants(client, 2)
    assert _consume(client, grants[0]).status_code == 200
    assert _revoke(client, grants[1]).status_code == 200

    by_id = {r["grant_id"]: r for r in _list(client).json()["grants"]}
    consumed = by_id[grants[0]["grant_id"]]
    revoked = by_id[grants[1]["grant_id"]]
    assert consumed["status"] == "consumed"
    assert consumed["consumed_at"] is not None
    assert consumed["revoked_at"] is None
    assert revoked["status"] == "revoked"
    assert revoked["revoked_at"] is not None
    assert revoked["consumed_at"] is None
    for record in (consumed, revoked):
        datetime.fromisoformat(record["consumed_at"] or record["revoked_at"])


def test_capability_plaintext_never_appears(client):
    grants = _seed_grants(client, 2)
    assert _consume(client, grants[0]).status_code == 200
    response = _list(client, limit=1)
    body = response.text
    for grant in grants:
        assert grant["capability"] not in body


# --- ordering and pagination -----------------------------------------------


def test_default_page_size_is_50(client):
    _seed_grants(client, 51)
    data = _list(client).json()
    assert len(data["grants"]) == 50
    assert data["complete"] is False
    assert data["next_cursor"] != ""


@pytest.mark.parametrize("limit", [1, 2, 200])
def test_limit_bounds_accepted(client, limit):
    _seed_grants(client, 3)
    data = _list(client, limit=limit).json()
    assert len(data["grants"]) == min(limit, 3)


def test_pages_are_ordered_by_grant_id_ascending(client):
    created = _seed_grants(client, 7)
    expected = sorted(g["grant_id"] for g in created)

    records, pages = _walk(client, limit=3)
    assert [r["grant_id"] for r in records] == expected
    # Every page is internally sorted as well.
    for page in pages:
        ids = [r["grant_id"] for r in page["grants"]]
        assert ids == sorted(ids)
    assert [len(p["grants"]) for p in pages] == [3, 3, 1]
    assert pages[0]["complete"] is False
    assert pages[-1]["complete"] is True
    assert pages[-1]["next_cursor"] == ""


def test_full_final_page_reports_complete(client):
    _seed_grants(client, 4)
    records, pages = _walk(client, limit=2)
    assert len(records) == 4
    assert [len(p["grants"]) for p in pages] == [2, 2]
    assert pages[-1]["complete"] is True
    assert pages[-1]["next_cursor"] == ""


def test_explicit_empty_cursor_equals_default(client):
    _seed_grants(client, 3)
    absent = _list(client, limit=2).json()
    empty = _list(client, cursor="", limit=2).json()
    assert absent == empty


def test_replaying_same_cursor_returns_identical_page(client):
    _seed_grants(client, 5)
    first = _list(client, limit=2).json()
    cursor = first["next_cursor"]
    second = _list(client, limit=2, cursor=cursor)
    third = _list(client, limit=2, cursor=cursor)
    assert second.status_code == 200
    assert second.content == third.content
    assert [r["grant_id"] for r in second.json()["grants"]] == [
        r["grant_id"] for r in third.json()["grants"]
    ]
    # The page continues strictly after the first page.
    first_ids = {r["grant_id"] for r in first["grants"]}
    second_ids = {r["grant_id"] for r in second.json()["grants"]}
    assert not first_ids & second_ids


def test_uppercase_grant_identifier_is_normalized(client):
    grant = _seed_grants(client, 1)[0]
    response = _list(client, grant_id=grant["grant_id"].upper())
    assert response.status_code == 200
    assert [r["grant_id"] for r in response.json()["grants"]] == [
        grant["grant_id"]
    ]


# --- filtering -------------------------------------------------------------


def test_filter_by_grant_id(client):
    grants = _seed_grants(client, 3)
    target = grants[1]
    data = _list(client, grant_id=target["grant_id"]).json()
    assert [r["grant_id"] for r in data["grants"]] == [target["grant_id"]]
    assert data["complete"] is True


def test_filter_by_decision_id(client):
    decision_a = _decision(client)["decision_id"]
    decision_b = _decision(client)["decision_id"]
    a_grants = [_grant(client, decision_a, data_id=f"a-{i}") for i in range(2)]
    _grant(client, decision_b, data_id="b-0")

    data = _list(client, decision_id=decision_a, limit=50).json()
    assert sorted(r["grant_id"] for r in data["grants"]) == sorted(
        g["grant_id"] for g in a_grants
    )
    assert all(r["decision_id"] == decision_a for r in data["grants"])


def test_filter_by_data_id(client):
    decision_id = _decision(client)["decision_id"]
    matching = [
        _grant(client, decision_id, data_id="picked"),
        _grant(client, decision_id, data_id="picked"),
    ]
    _grant(client, decision_id, data_id="other")

    data = _list(client, data_id="picked", limit=50).json()
    assert sorted(r["grant_id"] for r in data["grants"]) == sorted(
        g["grant_id"] for g in matching
    )
    assert all(r["data_id"] == "picked" for r in data["grants"])


@pytest.mark.parametrize(
    "status_value",
    ["pending", "consumed", "revoked"],
)
def test_filter_by_status(client, status_value):
    grants = _seed_grants(client, 6)
    assert _consume(client, grants[0]).status_code == 200
    assert _consume(client, grants[1]).status_code == 200
    assert _revoke(client, grants[2]).status_code == 200

    data = _list(client, status=status_value, limit=50).json()
    returned = {r["grant_id"]: r for r in data["grants"]}
    expected = {
        "pending": [grants[3]["grant_id"], grants[4]["grant_id"], grants[5]["grant_id"]],
        "consumed": [grants[0]["grant_id"], grants[1]["grant_id"]],
        "revoked": [grants[2]["grant_id"]],
    }[status_value]
    assert sorted(returned) == sorted(expected)
    assert all(r["status"] == status_value for r in data["grants"])


def test_issued_time_bounds_are_inclusive_and_normalized(app, client):
    grants = _seed_grants(client, 3)
    early = datetime(2024, 1, 1, tzinfo=timezone.utc)
    middle = datetime(2024, 6, 1, 12, tzinfo=timezone.utc)
    late = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for grant, stamp in zip(grants, (early, middle, late)):
        _set_issued_at(app, grant["grant_id"], stamp)

    after = _list(
        client, issued_after="2024-06-01T12:00:00Z", limit=50
    ).json()
    assert sorted(r["grant_id"] for r in after["grants"]) == sorted(
        g["grant_id"] for g in grants[1:]
    )

    before = _list(
        client, issued_before="2024-06-01T12:00:00Z", limit=50
    ).json()
    assert sorted(r["grant_id"] for r in before["grants"]) == sorted(
        g["grant_id"] for g in grants[:2]
    )

    window = _list(
        client,
        issued_after="2024-06-01T13:00:00+01:00",
        issued_before="2024-06-01T07:00:00-05:00",
        limit=50,
    ).json()
    assert [r["grant_id"] for r in window["grants"]] == [grants[1]["grant_id"]]

    future = _list(client, issued_after="2030-01-01T00:00:00Z").json()
    assert future["grants"] == []
    assert future["complete"] is True


def test_filters_combine_with_pagination(client):
    decision_id = _decision(client)["decision_id"]
    kept = [_grant(client, decision_id, data_id="keep") for _ in range(5)]
    for i in range(3):
        _grant(client, decision_id, data_id="drop")

    records, pages = _walk(client, data_id="keep", limit=2)
    assert sorted(r["grant_id"] for r in records) == sorted(
        g["grant_id"] for g in kept
    )
    assert all(r["data_id"] == "keep" for r in records)
    assert len(pages) == 3


# --- cursor authentication -------------------------------------------------


def test_forged_tampered_and_cross_scope_cursors_return_422(client):
    _seed_grants(client, 2)
    good_cursor = _list(client, limit=1).json()["next_cursor"]
    assert good_cursor

    tampered = good_cursor[:-2] + ("AA" if good_cursor[-2:] != "AA" else "BB")
    assert _list(client, cursor=tampered).status_code == 422

    forged = b64url_encode(b'{"t":"tenant-a","w":"workload-1","id":"x"}' + b"0" * 32)
    assert _list(client, cursor=forged).status_code == 422

    assert (
        _list(client, cursor=good_cursor, tenant_id=OTHER_TENANT).status_code == 422
    )
    assert (
        _list(client, cursor=good_cursor, workload_id=OTHER_WORKLOAD).status_code
        == 422
    )


def test_cursor_is_bound_to_every_filter(client):
    grants = _seed_grants(client, 4)
    assert _consume(client, grants[0]).status_code == 200
    # A second grant for data-0 guarantees the filtered page is multi-row.
    _grant(client, grants[0]["decision_id"], data_id="data-0")

    # Cursor minted while filtering on data_id at a small page size...
    cursor = _list(client, data_id="data-0", limit=1).json()["next_cursor"]
    assert cursor
    # ...is unrecognizable once that filter changes or is dropped.
    assert _list(client, cursor=cursor).status_code == 422
    assert _list(client, cursor=cursor, data_id="data-1").status_code == 422

    status_cursor = _list(client, status="pending", limit=1).json()["next_cursor"]
    assert _list(client, cursor=status_cursor, status="consumed").status_code == 422
    assert _list(client, cursor=status_cursor).status_code == 422

    # Cursors minted under different decision filters are not
    # interchangeable either.
    second_decision = _decision(client)["decision_id"]
    # Add a second page's worth of grants for that decision so it mints a
    # cursor at limit 1.
    for i in range(2):
        _grant(client, second_decision, data_id=f"second-{i}")
    decision_cursor = _list(
        client, decision_id=second_decision, limit=1
    ).json()["next_cursor"]
    assert decision_cursor
    assert (
        _list(client, cursor=decision_cursor, decision_id=grants[0]["decision_id"]).status_code
        == 422
    )
    assert _list(client, cursor=decision_cursor).status_code == 422


def test_cross_filter_cursor_from_real_page_is_rejected(client):
    grants = _seed_grants(client, 4)
    # Two pages over decision-less scope.
    cursor = _list(client, limit=2).json()["next_cursor"]
    # Replaying it with any added filter changes the bound filter set.
    assert (
        _list(client, cursor=cursor, status="pending").status_code == 422
    )
    assert (
        _list(client, cursor=cursor, data_id=grants[0]["data_id"]).status_code == 422
    )
    # A cursor minted on a status-filtered page cannot be replayed broadly.
    status_cursor = _list(client, status="pending", limit=2).json()["next_cursor"]
    assert _list(client, cursor=status_cursor).status_code == 422


# --- isolation -------------------------------------------------------------


def test_query_is_scoped_to_tenant_and_workload(client):
    own = _seed_grants(client, 2)
    other_decision = _decision(
        client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD
    )
    _grant(
        client,
        other_decision["decision_id"],
        data_id="x",
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
    )

    data = _list(client, limit=50).json()
    assert sorted(r["grant_id"] for r in data["grants"]) == sorted(
        g["grant_id"] for g in own
    )
    other = client.get(
        "/v1/release-grants",
        params={"tenant_id": OTHER_TENANT, "workload_id": OTHER_WORKLOAD},
    ).json()
    assert len(other["grants"]) == 1


# --- read-only / concurrency ----------------------------------------------


def test_query_never_writes_state(app, client):
    grants = _seed_grants(client, 3)
    before = {}
    with app.state.session_factory() as session:
        for grant in grants:
            row = session.get(ReleaseGrant, grant["grant_id"])
            before[grant["grant_id"]] = (
                row.status,
                row.issued_at,
                row.expires_at,
                row.consumed_at,
                row.revoked_at,
                row.capability_digest,
            )

    for _ in range(5):
        _list(client, limit=2)
        _list(client, status="pending", limit=1)
        _list(client, data_id="data-0")

    with app.state.session_factory() as session:
        assert session.query(ReleaseGrant).count() == 3
        for grant in grants:
            row = session.get(ReleaseGrant, grant["grant_id"])
            after = (
                row.status,
                row.issued_at,
                row.expires_at,
                row.consumed_at,
                row.revoked_at,
                row.capability_digest,
            )
            assert after == before[grant["grant_id"]]


def test_concurrent_consume_and_pagination_observe_committed_state(app):
    seeder = TestClient(app)
    grants = _seed_grants(seeder, 12)
    ordered_ids = sorted(g["grant_id"] for g in grants)

    def consume_all():
        local = TestClient(app)
        for grant in grants:
            local.post(
                f"/v1/release-grants/{grant['grant_id']}/consume",
                json={
                    "tenant_id": TENANT,
                    "workload_id": WORKLOAD,
                    "capability": grant["capability"],
                },
            )

    def walk_pages():
        local = TestClient(app)
        seen = []
        cursor = None
        for _ in range(20):
            params = {
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "limit": 3,
            }
            if cursor is not None:
                params["cursor"] = cursor
            response = local.get("/v1/release-grants", params=params)
            assert response.status_code == 200, response.text
            data = response.json()
            ids = [r["grant_id"] for r in data["grants"]]
            # Each page is internally sorted and has no repeated entry.
            assert ids == sorted(ids)
            assert len(ids) == len(set(ids))
            seen.extend(ids)
            if data["complete"]:
                return seen
            cursor = data["next_cursor"]
        raise AssertionError("pagination did not complete")  # pragma: no cover

    with ThreadPoolExecutor(max_workers=4) as pool:
        consumers = [pool.submit(consume_all) for _ in range(3)]
        readers = [pool.submit(walk_pages) for _ in range(6)]
        for future in consumers:
            future.result()
        reader_seen = [future.result() for future in readers]

    # Readers never skip or duplicate within their own walk.
    for seen in reader_seen:
        assert len(seen) == len(set(seen))
        assert set(seen).issubset(set(ordered_ids))

    # After settlement every grant is consumed exactly once and the
    # settled set pages through in stable grant_id order.
    records, _ = _walk(TestClient(app), status="consumed", limit=4)
    assert [r["grant_id"] for r in records] == ordered_ids
    pending = _list(TestClient(app), status="pending", limit=50).json()
    assert pending["grants"] == []
    assert pending["complete"] is True


def test_replayed_cursor_stays_stable_after_settlement(client):
    grants = _seed_grants(client, 4)
    first = _list(client, limit=2).json()
    cursor = first["next_cursor"]
    second_before = _list(client, limit=2, cursor=cursor).json()
    # Settle a grant that appears on the second page; the unfiltered
    # cursor still returns the same identities in the same order.
    target_id = second_before["grants"][0]["grant_id"]
    target = next(g for g in grants if g["grant_id"] == target_id)
    assert _consume(client, target).status_code == 200
    second_after = _list(client, limit=2, cursor=cursor).json()
    assert [r["grant_id"] for r in second_after["grants"]] == [
        r["grant_id"] for r in second_before["grants"]
    ]


# --- wire format -----------------------------------------------------------


def test_response_is_compact_json_without_floats_or_newline(client):
    _seed_grants(client, 2)
    response = _list(client, limit=1)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] != b"\n"
    assert b", " not in raw
    assert b": " not in raw
    parsed = json.loads(raw)
    assert isinstance(parsed["complete"], bool)
    assert isinstance(parsed["next_cursor"], str)
    # The wire format carries no numeric fields at all, so floats,
    # -0.0 and non-finite values cannot appear.
    def assert_no_numbers(value):
        assert not isinstance(value, float)
        if isinstance(value, dict):
            for child in value.values():
                assert_no_numbers(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_numbers(child)

    assert_no_numbers(parsed)


# --- persistence -----------------------------------------------------------


def test_cursor_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    grants = _seed_grants(client1, 4)
    first = client1.get(
        "/v1/release-grants",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "limit": 2,
        },
    ).json()
    cursor = first["next_cursor"]
    assert first["complete"] is False
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    second = client2.get(
        "/v1/release-grants",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "limit": 2,
            "cursor": cursor,
        },
    )
    assert second.status_code == 200
    all_ids = sorted(g["grant_id"] for g in grants)
    walked = [r["grant_id"] for r in first["grants"]] + [
        r["grant_id"] for r in second.json()["grants"]
    ]
    assert sorted(walked) == all_ids
    assert second.json()["complete"] is True
    app2.state.engine.dispose()


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_half_page(app, client):
    _seed_grants(client, 3)
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE release_grants"))
    response = _list(client)
    assert response.status_code == 500
