"""Tests for GET /v1/policies.

Read-only lifecycle query of versioned release policies. The query is
ranged by tenant/workload, narrowed by an optional policy id, exact name,
lifecycle status and an inclusive creation-time window, and paginated with
opaque, scope/filter/snapshot-bound cursors. The snapshot is fixed at a
per-scope commit boundary shared by version creation and retirement, so a
replayed page is stable even under concurrent retirements and new
versions. The endpoint never writes state or audit and never returns
evidence, claim values, capabilities, keys or exception text.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.app import (
    create_app,
    _encode_audit_event_cursor,
    _encode_cursor,
    _encode_decision_cursor,
    _encode_grant_audit_cursor,
    _encode_revocation_cursor,
    _encode_trust_root_cursor,
)
from proof_release.db import Policy, PolicyCommitCounter

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

ZERO_UUID = "00000000-0000-0000-0000-000000000000"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/policies_query.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, rule=None, *, name="release", tenant=TENANT,
            workload=WORKLOAD):
    body = {
        "tenant_id": tenant,
        "workload_id": workload,
        "name": name,
        "rule": rule or {"claim": "measurement", "equals": "abc"},
    }
    response = client.post("/v1/policies", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _retire(client, policy_id_value, *, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        f"/v1/policies/{policy_id_value}/retire",
        json={"tenant_id": tenant, "workload_id": workload},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _query(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    query = {"tenant_id": tenant, "workload_id": workload}
    query.update(params)
    return client.get("/v1/policies", params=query)


def _walk(client, *, page_size=None, monkeypatch=None, **params):
    token = None
    rows = []
    for _ in range(50):
        query_params = dict(params)
        if token is not None:
            query_params["cursor"] = token
        if page_size is not None:
            monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", page_size)
        page = _query(client, **query_params)
        assert page.status_code == 200, page.text
        data = page.json()
        rows.extend(data["policies"])
        if data["complete"]:
            assert data["next_cursor"] == ""
            return rows
        token = data["next_cursor"]
        assert token
    raise AssertionError("pagination never completed")  # pragma: no cover


# --- request validation ----------------------------------------------------


def test_query_requires_scope_parameters(client):
    assert client.get("/v1/policies").status_code == 422
    assert (
        client.get("/v1/policies", params={"workload_id": WORKLOAD}).status_code
        == 422
    )
    assert (
        client.get("/v1/policies", params={"tenant_id": TENANT}).status_code == 422
    )


@pytest.mark.parametrize(
    "params",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"workload_id": "\t"},
        {"policy_id": ""},
        {"policy_id": "   "},
        {"policy_id": "not-a-uuid"},
        {"policy_id": ZERO_UUID[:-1] + "z"},
        {"policy_id": "  " + ZERO_UUID},
        {"policy_id": "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"},  # uppercase
        {"name": ""},
        {"name": "   "},
        {"name": "\t"},
        {"status": ""},
        {"status": "   "},
        {"status": "pending"},
        {"status": "RETIRED"},
        {"status": "retired "},
        {"created_after": ""},
        {"created_after": "   "},
        {"created_after": "not-a-timestamp"},
        {"created_after": "2026-01-01T00:00:00"},  # naive
        {"created_before": "2026-01-01"},
        {"created_after": "2026-01-01T00:00:00+02:00"},  # non-UTC offset
        {
            "created_after": "2026-01-02T00:00:00Z",
            "created_before": "2026-01-01T00:00:00Z",
        },
        {"cursor": "   "},
        {"cursor": "not base64!!"},
        {"cursor": "AAA="},
        {"bogus": "value"},
        {"limit": "10"},
        {"root_id": ZERO_UUID},
    ],
)
def test_query_rejects_invalid_parameters(client, params):
    response = _query(client, **params)
    assert response.status_code == 422, response.text


def test_repeated_parameter_is_422(client):
    response = client.get(
        "/v1/policies",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("status", "active"),
            ("status", "retired"),
        ],
    )
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"{}", b"  ", b"not json", b"\x00"])
def test_non_empty_body_is_422_before_state_read(app, client, body):
    def fail_read(conn, cursor, statement, parameters, context, executemany):
        if "FROM policies" in statement:
            raise AssertionError("storage must not be read for a bodied query")

    from sqlalchemy import event

    event.listen(app.state.engine, "before_cursor_execute", fail_read)
    try:
        response = client.request(
            "GET",
            "/v1/policies",
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            content=body,
        )
        assert response.status_code == 422
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_read)


def test_invalid_parameters_write_no_state(app, client):
    _create(client, name="one")
    _query(client, policy_id="nope")
    _query(client, created_after="2020-01-01T00:00:00+03:00")
    _query(client, unknown="x")
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == 1


# --- 404 semantics ---------------------------------------------------------


def test_unknown_policy_identifier_returns_404(client):
    assert _query(client, policy_id=ZERO_UUID).status_code == 404


def test_cross_scope_policy_identifier_is_indistinguishable_404(client):
    created = _create(client, name="release")
    assert (
        _query(client, tenant=OTHER_TENANT, policy_id=created["policy_id"]).status_code
        == 404
    )
    assert (
        _query(client, workload=OTHER_WORKLOAD, policy_id=created["policy_id"]).status_code
        == 404
    )


def test_explicit_unknown_identifier_404_even_with_other_matching_rows(client):
    _create(client, name="one")
    assert _query(client, policy_id=ZERO_UUID).status_code == 404


# --- empty range / response shape ------------------------------------------


def test_empty_range_returns_empty_completed_page(client):
    response = _query(client)
    assert response.status_code == 200
    assert response.json() == {"policies": [], "next_cursor": "", "complete": True}


def test_name_with_no_match_returns_empty_range_not_404(client):
    _create(client, name="one")
    response = _query(client, name="no-such-name")
    assert response.status_code == 200
    assert response.json() == {"policies": [], "next_cursor": "", "complete": True}


def test_window_with_no_matches_returns_empty_completed_page(client):
    _create(client, name="one")
    response = _query(client, created_after="2030-01-01T00:00:00Z")
    assert response.status_code == 200
    assert response.json() == {"policies": [], "next_cursor": "", "complete": True}


def test_response_is_compact_json_with_single_trailing_newline(client):
    _create(client, name="one")
    response = _query(client)
    raw = response.content
    assert response.headers["content-type"] == "application/json"
    assert raw[-1:] == b"\n"
    assert raw[-2:] != b"\n\n"
    assert b", " not in raw
    assert b": " not in raw
    assert list(json.loads(raw)) == ["policies", "next_cursor", "complete"]


def test_entry_has_exact_shape_and_field_order(client):
    rule = {"all": [{"claim": "measurement", "equals": "abc"}]}
    created = _create(client, rule, name="primary")
    raw = _query(client).content
    entry = json.loads(raw)["policies"][0]
    assert list(entry) == [
        "policy_id",
        "tenant_id",
        "workload_id",
        "name",
        "version",
        "rule",
        "created_at",
        "status",
        "retired_at",
    ]
    assert entry["policy_id"] == created["policy_id"]
    assert entry["tenant_id"] == TENANT
    assert entry["workload_id"] == WORKLOAD
    assert entry["name"] == "primary"
    assert entry["version"] == 1
    assert entry["rule"] == rule
    assert entry["created_at"] == created["created_at"]
    assert entry["status"] == "active"
    assert entry["retired_at"] is None


def test_entry_reuses_creation_fields_including_version(client):
    v1 = _create(client, name="release")
    v2 = _create(client, {"claim": "tier", "equals": 2}, name="release")
    rows = _query(client, name="release").json()["policies"]
    assert [r["policy_id"] for r in rows] == [v1["policy_id"], v2["policy_id"]]
    assert [r["version"] for r in rows] == [1, 2]
    assert rows[1]["rule"] == {"claim": "tier", "equals": 2}


def test_retired_entry_carries_status_and_retirement_time(client):
    created = _create(client, name="primary")
    retired = _retire(client, created["policy_id"])
    entry = _query(client).json()["policies"][0]
    assert entry["status"] == "retired"
    assert entry["retired_at"] == retired["retired_at"]
    assert entry["created_at"] == created["created_at"]


def test_no_metadata_floats_or_non_finite_values(client):
    _create(client, name="one")
    parsed = json.loads(_query(client).content)
    assert isinstance(parsed["complete"], bool)
    assert isinstance(parsed["next_cursor"], str)

    def _check(value):
        if isinstance(value, bool):
            return
        if isinstance(value, int):
            return
        if isinstance(value, float):  # pragma: no cover - structural guard
            raise AssertionError("metadata float present")
        assert value is None or isinstance(value, str) or isinstance(value, dict)

    for row in parsed["policies"]:
        for key, value in row.items():
            if key == "rule":
                # The rule is the one place a number may live; its
                # non-number metadata still obeys the guard.
                _walk_rule(value, _check)
            else:
                _check(value)


def _walk_rule(node, check):
    if isinstance(node, dict):
        for value in node.values():
            _walk_rule(value, check)
    elif isinstance(node, list):
        for value in node:
            _walk_rule(value, check)
    else:
        check(node)


@pytest.mark.parametrize(
    "rule",
    [
        {"claim": "x", "equals": -0.0},
        {"claim": "x", "equals": 0.0},
        {"path": ["t", "v"], "equals": 1.25},
        {"claim": "x", "equals": -1.5},
        {"claim": "x", "equals": 3},
        {"all": [{"claim": "a", "equals": 0.1}, {"path": ["b"], "equals": -2.25}]},
    ],
)
def test_rule_numbers_round_trip_verbatim_including_negative_zero(client, rule):
    created = _create(client, rule, name=json.dumps(rule, sort_keys=True))
    response = _query(client, name=created["name"])
    entry = response.json()["policies"][0]
    assert entry["rule"] == rule
    if "equals" in rule:
        original = rule["equals"]
        parsed_number = entry["rule"]["equals"]
        if isinstance(original, float):
            assert isinstance(parsed_number, float)
        if math.copysign(1.0, original) < 0 and original == 0.0:
            # -0.0 keeps its sign on the wire and after parsing.
            assert math.copysign(1.0, parsed_number) < 0
            assert b"-0.0" in response.content
    else:
        # A compound rule keeps every nested number verbatim, including
        # the negative decimal, with no float added elsewhere.
        assert b"-2.25" in response.content
        assert b"0.1" in response.content


# --- ordering and scoping --------------------------------------------------


def test_results_ordered_by_created_at_then_policy_id(client):
    created = [_create(client, name=f"release-{i}") for i in range(5)]
    rows = _query(client).json()["policies"]
    keys = [(row["created_at"], row["policy_id"]) for row in rows]
    assert keys == sorted(keys)
    assert [row["policy_id"] for row in rows] == [c["policy_id"] for c in created]


def test_results_are_scoped_to_tenant_and_workload(client):
    a = _create(client, name="a")
    b = _create(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD, name="b")
    rows_a = _query(client).json()["policies"]
    rows_b = _query(client, tenant=OTHER_TENANT, workload=OTHER_WORKLOAD).json()[
        "policies"
    ]
    assert [r["policy_id"] for r in rows_a] == [a["policy_id"]]
    assert [r["policy_id"] for r in rows_b] == [b["policy_id"]]


# --- filtering -------------------------------------------------------------


def test_filter_by_policy_id_returns_exactly_that_version(client):
    bodies = [_create(client, name=f"release-{i}") for i in range(3)]
    target = bodies[1]
    rows = _query(client, policy_id=target["policy_id"]).json()["policies"]
    assert [r["policy_id"] for r in rows] == [target["policy_id"]]


def test_filter_by_name_returns_every_version_of_that_name(client):
    v1 = _create(client, name="release")
    v2 = _create(client, name="release")
    other = _create(client, name="other")
    rows = _query(client, name="release").json()["policies"]
    assert [r["policy_id"] for r in rows] == [v1["policy_id"], v2["policy_id"]]
    assert all(r["version"] == i for i, r in enumerate(rows, start=1))
    assert other["policy_id"] not in [r["policy_id"] for r in rows]


def test_filter_by_name_is_exact_text(client):
    _create(client, name="alpha")
    _create(client, name="alpha-two")
    rows = _query(client, name="alpha").json()["policies"]
    assert len(rows) == 1
    assert rows[0]["name"] == "alpha"


def test_name_matches_verbatim_including_surrounding_spaces(client):
    # A non-blank name with surrounding visible whitespace is a distinct
    # exact value; only an all-whitespace filter is a 422.
    _create(client, name="pad")
    _create(client, name=" pad ")
    assert _query(client, name="pad").json()["policies"][0]["name"] == "pad"
    spaced = _query(client, name=" pad ").json()["policies"]
    assert len(spaced) == 1 and spaced[0]["name"] == " pad "


def test_filter_by_status(client):
    active_a = _create(client, name="a")
    retiring = _create(client, name="b")
    _retire(client, retiring["policy_id"])
    active_c = _create(client, name="c")

    active_rows = _query(client, status="active").json()["policies"]
    assert {r["policy_id"] for r in active_rows} == {
        active_a["policy_id"],
        active_c["policy_id"],
    }
    retired_rows = _query(client, status="retired").json()["policies"]
    assert [r["policy_id"] for r in retired_rows] == [retiring["policy_id"]]
    assert all(r["status"] == "active" for r in active_rows)
    assert all(r["status"] == "retired" for r in retired_rows)


def _get_id(client, name):
    rows = _query(client, name=name).json()["policies"]
    assert len(rows) == 1
    return rows[0]["policy_id"]


def test_creation_window_is_inclusive_on_both_ends(client):
    first = _create(client, name="first")
    _create(client, name="middle")
    last = _create(client, name="last")
    ta = first["created_at"]
    tb = last["created_at"]
    if ta == tb:
        # Degenerate identical timestamps: the single-instant window must
        # include every version created at that instant.
        rows = _query(
            client, created_after=ta, created_before=tb
        ).json()["policies"]
        assert {r["policy_id"] for r in rows} == {
            first["policy_id"],
            _get_id(client, "middle"),
            last["policy_id"],
        }
        return
    middle_id = _get_id(client, "middle")
    rows = _query(client, created_after=ta, created_before=tb).json()["policies"]
    assert {r["policy_id"] for r in rows} == {
        first["policy_id"],
        middle_id,
        last["policy_id"],
    }
    only_first = _query(client, created_after=ta, created_before=ta).json()[
        "policies"
    ]
    assert [r["policy_id"] for r in only_first] == [first["policy_id"]]


def test_equivalent_utc_spellings_share_one_cursor_domain(client, monkeypatch):
    for i in range(3):
        _create(client, name=f"release-{i}")
    monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", 1)
    zed = _query(client, created_after="2000-01-01T00:00:00Z")
    cursor = zed.json()["next_cursor"]
    offset = _query(
        client, created_after="2000-01-01T00:00:00+00:00", cursor=cursor
    )
    assert offset.status_code == 200
    replayed = _query(client, created_after="2000-01-01T00:00:00Z", cursor=cursor)
    assert offset.content == replayed.content


# --- pagination ------------------------------------------------------------


def test_pagination_walks_every_version_once_in_order(client, monkeypatch):
    bodies = [_create(client, name=f"release-{i}") for i in range(7)]
    rows = _walk(client, page_size=3, monkeypatch=monkeypatch)
    assert len(rows) == 7
    keys = [(r["created_at"], r["policy_id"]) for r in rows]
    assert keys == sorted(keys)
    assert {r["policy_id"] for r in rows} == {b["policy_id"] for b in bodies}


def test_page_carries_cursor_and_complete_flag(client, monkeypatch):
    monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", 2)
    bodies = [_create(client, name=f"release-{i}") for i in range(5)]

    first = _query(client).json()
    assert len(first["policies"]) == 2
    assert first["complete"] is False
    assert first["next_cursor"]

    second = _query(client, cursor=first["next_cursor"]).json()
    assert len(second["policies"]) == 2
    assert second["complete"] is False
    first_keys = [(r["created_at"], r["policy_id"]) for r in first["policies"]]
    second_keys = [(r["created_at"], r["policy_id"]) for r in second["policies"]]
    assert first_keys < second_keys

    last = _query(client, cursor=second["next_cursor"]).json()
    assert len(last["policies"]) == 1
    assert last["complete"] is True
    assert last["next_cursor"] == ""
    walked = [r["policy_id"] for r in first["policies"]]
    walked += [r["policy_id"] for r in second["policies"]]
    walked += [r["policy_id"] for r in last["policies"]]
    assert walked == [b["policy_id"] for b in bodies]


def test_replaying_same_cursor_returns_identical_page(client, monkeypatch):
    monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", 2)
    for i in range(5):
        _create(client, name=f"release-{i}")
    cursor = _query(client).json()["next_cursor"]
    first = _query(client, cursor=cursor)
    second = _query(client, cursor=cursor)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_empty_string_cursor_equals_default(client, monkeypatch):
    monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", 2)
    for i in range(3):
        _create(client, name=f"release-{i}")
    assert _query(client).content == _query(client, cursor="").content


def test_tampered_or_forged_cursor_returns_422(client, monkeypatch):
    monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", 1)
    for i in range(3):
        _create(client, name=f"release-{i}")
    cursor = _query(client).json()["next_cursor"]
    assert cursor

    tampered = cursor[:-2] + ("AA" if cursor[-2:] != "AA" else "BB")
    assert _query(client, cursor=tampered).status_code == 422

    forged = app_module.b64url_encode(
        b'{"k":"policies-v1"}' + b"0" * 32
    )
    assert _query(client, cursor=forged).status_code == 422


def test_cross_scope_cursor_returns_422(client, monkeypatch):
    monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", 1)
    for i in range(3):
        _create(client, name=f"release-{i}")
    cursor = _query(client).json()["next_cursor"]
    assert _query(client, tenant=OTHER_TENANT, cursor=cursor).status_code == 422
    assert _query(client, workload=OTHER_WORKLOAD, cursor=cursor).status_code == 422


def test_cursor_cannot_cross_filters(client, monkeypatch):
    monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", 1)
    bodies = [_create(client, name=f"release-{i}") for i in range(3)]
    cursor = _query(client).json()["next_cursor"]

    assert _query(client, policy_id=bodies[0]["policy_id"], cursor=cursor).status_code == 422
    assert _query(client, name="release-0", cursor=cursor).status_code == 422
    assert _query(client, status="active", cursor=cursor).status_code == 422
    assert (
        _query(client, created_after="2000-01-01T00:00:00Z", cursor=cursor).status_code
        == 422
    )
    assert (
        _query(client, created_before="2100-01-01T00:00:00Z", cursor=cursor).status_code
        == 422
    )


def test_cursor_rejects_other_cursor_families(client):
    foreign_rewrap = _encode_cursor(TENANT, WORKLOAD, "some-data-id")
    assert _query(client, cursor=foreign_rewrap).status_code == 422

    foreign_grant = _encode_grant_audit_cursor(
        TENANT, WORKLOAD, ZERO_UUID,
        grant_id="", decision_id="", data_id="", status="",
        issued_after="", issued_before="",
    )
    assert _query(client, cursor=foreign_grant).status_code == 422

    foreign_audit = _encode_audit_event_cursor(
        TENANT, WORKLOAD, "2020-01-01T00:00:00+00:00", ZERO_UUID,
        event_id="", event_type="", status="",
        occurred_after="", occurred_before="",
    )
    assert _query(client, cursor=foreign_audit).status_code == 422

    foreign_revocation = _encode_revocation_cursor(
        TENANT, WORKLOAD, ZERO_UUID,
        "2020-01-01T00:00:00+00:00", ZERO_UUID,
        revocation_id="", certificate_fingerprint="",
        effective_after="", effective_before="",
        snapshot_at="2020-01-01T00:00:00+00:00", snapshot_id=ZERO_UUID,
    )
    assert _query(client, cursor=foreign_revocation).status_code == 422

    foreign_decision = _encode_decision_cursor(
        TENANT, WORKLOAD, "2020-01-01T00:00:00+00:00", ZERO_UUID,
        decision_id="", evidence_id="", policy_id="", status="",
        decided_after="", decided_before="", snapshot_seq=1,
    )
    assert _query(client, cursor=foreign_decision).status_code == 422

    foreign_trust_root = _encode_trust_root_cursor(
        TENANT, WORKLOAD, "2020-01-01T00:00:00+00:00", ZERO_UUID,
        root_id="", name="", status="",
        created_after="", created_before="", snapshot_seq=1,
    )
    assert _query(client, cursor=foreign_trust_root).status_code == 422


# --- snapshot stability ----------------------------------------------------


def test_first_query_snapshot_excludes_later_versions(client, monkeypatch):
    monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", 2)
    bodies = [_create(client, name=f"release-{i}") for i in range(3)]

    first = _query(client).json()
    assert [r["policy_id"] for r in first["policies"]] == [
        b["policy_id"] for b in bodies[:2]
    ]
    cursor = first["next_cursor"]

    # A version committed after the first query never enters the
    # snapshot's later pages.
    late = _create(client, name="late")
    second = _query(client, cursor=cursor).json()
    assert [r["policy_id"] for r in second["policies"]] == [bodies[2]["policy_id"]]
    assert second["complete"] is True
    assert second["next_cursor"] == ""

    # Replaying the cursor stays byte-for-byte stable.
    assert _query(client, cursor=cursor).json() == second

    # A fresh first query includes the later version.
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert {r["policy_id"] for r in fresh} == {
        b["policy_id"] for b in bodies
    } | {late["policy_id"]}


def test_snapshot_is_stable_under_concurrent_retirement(app, client, monkeypatch):
    monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", 2)
    bodies = [_create(client, name=f"release-{i}") for i in range(6)]
    first = _query(client).json()
    assert len(first["policies"]) == 2
    cursor = first["next_cursor"]

    # Retire every version after the snapshot was fixed. The replayed walk
    # must keep presenting all six as active with null retired_at and
    # never duplicate or skip one.
    def retire_later(policy_id_value):
        response = TestClient(app).post(
            f"/v1/policies/{policy_id_value}/retire",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
        assert response.status_code == 200

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(retire_later, [b["policy_id"] for b in bodies]))

    snapshot_rows = []
    token = cursor
    for _ in range(10):
        data = _query(client, cursor=token).json()
        snapshot_rows.extend(data["policies"])
        if data["complete"]:
            break
        token = data["next_cursor"]

    assert [r["policy_id"] for r in first["policies"]] == [
        b["policy_id"] for b in bodies[:2]
    ]
    snapshot_ids = [r["policy_id"] for r in snapshot_rows]
    assert snapshot_ids == [b["policy_id"] for b in bodies[2:]]
    for row in first["policies"] + snapshot_rows:
        assert row["status"] == "active"
        assert row["retired_at"] is None

    # A fresh first query sees every version retired.
    fresh = _walk(client, page_size=2, monkeypatch=monkeypatch)
    assert len(fresh) == 6
    assert all(r["status"] == "retired" for r in fresh)
    assert all(r["retired_at"] for r in fresh)


def test_snapshot_status_filter_stable_under_concurrent_retirement(client):
    bodies = [_create(client, name=f"release-{i}") for i in range(3)]
    # Fix the snapshot while all three are active, filtered to active.
    first = _query(client, status="active").json()
    assert {r["policy_id"] for r in first["policies"]} == {
        b["policy_id"] for b in bodies
    }

    for created in bodies:
        _retire(client, created["policy_id"])

    # The snapshot-less first page is already complete: the same filter
    # parameters from a fresh query now return an empty active range and a
    # complete retired range, while nothing about the old first page changes.
    fresh_active = _query(client, status="active").json()
    assert fresh_active["policies"] == []
    fresh_retired = _query(client, status="retired").json()
    assert {r["policy_id"] for r in fresh_retired["policies"]} == {
        b["policy_id"] for b in bodies
    }
    assert first["policies"][0]["status"] == "active"


def test_retirement_between_pages_does_not_change_snapshot(client, monkeypatch):
    monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", 1)
    bodies = [_create(client, name=f"release-{i}") for i in range(3)]
    first = _query(client).json()
    cursor = first["next_cursor"]

    # Retire the very version the next page will display.
    _retire(client, bodies[1]["policy_id"])
    second = _query(client, cursor=cursor).json()
    assert [r["policy_id"] for r in second["policies"]] == [bodies[1]["policy_id"]]
    # The snapshot still presents it as active with no retirement time.
    assert second["policies"][0]["status"] == "active"
    assert second["policies"][0]["retired_at"] is None

    last = _query(client, cursor=second["next_cursor"]).json()
    assert [r["policy_id"] for r in last["policies"]] == [bodies[2]["policy_id"]]
    assert last["complete"] is True

    # A fresh query shows the middle version retired.
    fresh = _walk(client, page_size=1, monkeypatch=monkeypatch)
    middle = next(r for r in fresh if r["policy_id"] == bodies[1]["policy_id"])
    assert middle["status"] == "retired"
    assert middle["retired_at"]


# --- read-only behaviour ---------------------------------------------------


def test_query_changes_no_state_and_creates_no_counters_or_retirement(
    app, client
):
    created = _create(client, name="one")
    before = _query(client).content
    for _ in range(3):
        assert _query(client).status_code == 200
        assert _query(client, policy_id=created["policy_id"]).status_code == 200
    assert _query(client).content == before
    with app.state.session_factory() as session:
        policy = session.get(Policy, created["policy_id"])
        assert policy.status == "active"
        assert policy.retired_at is None
        assert policy.retired_seq is None
        # Reading never mints or advances the lifecycle counter beyond the
        # one sequence consumed by creation.
        counter = session.get(PolicyCommitCounter, (TENANT, WORKLOAD))
        assert counter.last_seq == 1


# --- server failure --------------------------------------------------------


def test_storage_failure_returns_500_and_no_half_page(app, client):
    for i in range(3):
        _create(client, name=f"release-{i}")

    def fail_scan(conn, cursor, statement, parameters, context, executemany):
        if "FROM policies" in statement:
            raise RuntimeError("simulated storage failure")

    from sqlalchemy import event

    event.listen(app.state.engine, "before_cursor_execute", fail_scan)
    try:
        response = _query(client)
        # A storage failure is a 500 with no partial list.
        assert response.status_code == 500
        assert b'"policies"' not in response.content
        assert b"simulated storage failure" not in response.content
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_scan)

    recovered = _query(client)
    assert recovered.status_code == 200
    assert len(recovered.json()["policies"]) == 3
    with app.state.session_factory() as session:
        assert session.query(Policy).count() == 3


# --- persistence -----------------------------------------------------------


def test_policies_queryable_after_restart_and_cursor_replays(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", 1)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = create_app(url)
    client1 = TestClient(first)
    bodies = []
    for i in range(3):
        response = client1.post(
            "/v1/policies",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "name": f"release-{i}",
                "rule": {"claim": "i", "equals": i},
            },
        )
        assert response.status_code == 201
        bodies.append(response.json())
    cursor = client1.get(
        "/v1/policies",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()["next_cursor"]
    first.state.engine.dispose()

    second = create_app(url)
    client2 = TestClient(second)
    page = client2.get(
        "/v1/policies",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "cursor": cursor,
        },
    )
    assert page.status_code == 200
    assert [r["policy_id"] for r in page.json()["policies"]] == [
        bodies[1]["policy_id"]
    ]

    monkeypatch.setattr(app_module, "POLICY_PAGE_SIZE", 100)
    all_rows = client2.get(
        "/v1/policies",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()["policies"]
    assert [r["policy_id"] for r in all_rows] == [b["policy_id"] for b in bodies]
    second.state.engine.dispose()


# --- legacy migration ------------------------------------------------------


def test_legacy_sqlite_database_backfills_sequences_and_counters(tmp_path):
    from sqlalchemy import create_engine, text

    from proof_release.db import Base

    url = f"sqlite:///{tmp_path}/legacy.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # Build a pre-lifecycle-sequence database: drop the new columns,
        # the new indexes and the counter table.
        conn.execute(text("DROP INDEX IF EXISTS ix_policies_scope_commit_seq"))
        conn.execute(text("DROP INDEX IF EXISTS ix_policies_scope_created"))
        conn.execute(text("ALTER TABLE policies DROP COLUMN retired_seq"))
        conn.execute(text("ALTER TABLE policies DROP COLUMN commit_seq"))
        conn.execute(text("DROP TABLE policy_commit_counters"))
        conn.execute(
            text(
                "INSERT INTO policies "
                "(policy_id, tenant_id, workload_id, name, version, rule_json, "
                "created_at, status, retired_at) VALUES "
                "('11111111-1111-4111-8111-111111111111', :t, :w, 'a', 1, '{}', "
                "'2026-01-01T00:00:01+00:00', 'retired', '2026-01-03T00:00:00+00:00'),"
                "('22222222-2222-4222-8222-222222222222', :t, :w, 'b', 1, '{}', "
                "'2026-01-02T00:00:00+00:00', 'active', NULL)"
            ),
            {"t": TENANT, "w": WORKLOAD},
        )
    engine.dispose()

    application = create_app(url)
    client = TestClient(application)
    data = client.get(
        "/v1/policies",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    ids = [r["policy_id"] for r in data["policies"]]
    assert ids == [
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    ]
    by_id = {r["policy_id"]: r for r in data["policies"]}
    assert by_id["11111111-1111-4111-8111-111111111111"]["status"] == "retired"
    assert by_id["22222222-2222-4222-8222-222222222222"]["status"] == "active"

    # The next creation/retirement allocates past the seeded maximum (2
    # creations + 1 retirement = 3) without colliding.
    with application.state.session_factory() as session:
        counter = session.get(PolicyCommitCounter, (TENANT, WORKLOAD))
        assert counter.last_seq == 3
    application.state.engine.dispose()
