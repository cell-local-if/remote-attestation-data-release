"""Tests for the workload identity profile lifecycle: scoped query
(GET), whole claim-set update (PUT) and revocation (DELETE), plus the way
revocation and settled updates interact with the X.509 identity gate."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select, text

from proof_release.app import create_app
from proof_release.db import (
    WorkloadIdentityClaim,
    WorkloadIdentityProfile,
)
from proof_release.verifiers import (
    VerificationContext,
    VerifierRegistry,
    X509_ATTESTED_NONCE_JSON,
    X509AttestedNonceJSONVerifier,
)

from x509_helpers import (
    make_evidence,
    make_intermediate,
    make_leaf,
    make_root,
    pem,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"

LEAF_URI = "spiffe://example.org/workload/payments"
OTHER_URI = "spiffe://example.org/workload/billing"
THIRD_URI = "spiffe://example.org/workload/ledger"

#: RFC4514 DNs of the default x509_helpers chain: the leaf is issued by
#: "test-intermediate" and its own CN is "test-leaf".
LEAF_ISSUER = "CN=test-intermediate"
LEAF_SUBJECT = "CN=test-leaf"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/test.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def chain():
    root_key, root_cert = make_root()
    intermediate_key, intermediate_cert = make_intermediate(root_cert, root_key)
    leaf_key, leaf_cert = make_leaf(
        intermediate_cert, intermediate_key, uri=LEAF_URI
    )
    return {
        "root_key": root_key,
        "root_cert": root_cert,
        "intermediate_key": intermediate_key,
        "intermediate_cert": intermediate_cert,
        "leaf_key": leaf_key,
        "leaf_cert": leaf_cert,
    }


@pytest.fixture()
def root_id(client, chain):
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    assert response.status_code == 201
    return response.json()["root_id"]


def _claim(uri):
    return {"issuer": LEAF_ISSUER, "subject": LEAF_SUBJECT, "uri": uri}


def _register(client, root, uris, *, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        "/v1/workload-identities",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "trust_root_id": root,
            "claims": [_claim(uri) for uri in uris],
        },
    )


def _list(client, root, *, profile_id=None, tenant=TENANT, workload=WORKLOAD, **extra):
    params = {"tenant_id": tenant, "workload_id": workload, "trust_root_id": root}
    if profile_id is not None:
        params["profile_id"] = profile_id
    params.update(extra)
    return client.get("/v1/workload-identities", params=params)


def _update(client, profile_id, root, uris, *, tenant=TENANT, workload=WORKLOAD):
    return client.put(
        f"/v1/workload-identities/{profile_id}",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "trust_root_id": root,
            "claims": [_claim(uri) for uri in uris],
        },
    )


def _revoke(client, profile_id, *, tenant=TENANT, workload=WORKLOAD, **body_extra):
    body = {"tenant_id": tenant, "workload_id": workload}
    body.update(body_extra)
    return client.request(
        "DELETE", f"/v1/workload-identities/{profile_id}", json=body
    )


def _challenge(client):
    response = client.post(
        "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert response.status_code == 201
    return response.json()


def _settle_verification(client, created, evidence):
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": X509_ATTESTED_NONCE_JSON,
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    response = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    return evidence_id, response


def _leaf_evidence(created, chain):
    return make_evidence(
        created["nonce"],
        chain["leaf_key"],
        [chain["leaf_cert"], chain["intermediate_cert"], chain["root_cert"]],
        {"m": "abc"},
    )


# --- query ------------------------------------------------------------------


def test_list_empty_range_returns_empty_array(client, root_id):
    response = _list(client, root_id)
    assert response.status_code == 200
    assert response.json() == {"profiles": []}
    # Compact JSON with exactly one trailing newline.
    assert response.content == b'{"profiles":[]}\n'


def test_list_returns_profile_with_status_and_creation_fields(
    client, root_id, chain
):
    claim = _claim(LEAF_URI)
    registered = _register(client, root_id, [LEAF_URI])
    assert registered.status_code == 201

    response = _list(client, root_id)
    assert response.status_code == 200
    profiles = response.json()["profiles"]
    assert len(profiles) == 1
    profile = profiles[0]
    assert set(profile.keys()) == {
        "profile_id",
        "tenant_id",
        "workload_id",
        "trust_root_id",
        "claims",
        "created_at",
        "status",
    }
    assert profile["profile_id"] == registered.json()["profile_id"]
    assert profile["tenant_id"] == TENANT
    assert profile["workload_id"] == WORKLOAD
    assert profile["trust_root_id"] == root_id
    assert profile["claims"] == [claim]
    assert profile["created_at"] == registered.json()["created_at"]
    assert profile["status"] == "active"
    # Compact JSON: no insignificant whitespace, single trailing newline.
    assert b", " not in response.content
    assert b": " not in response.content
    assert response.content.endswith(b"\n")
    assert response.content.count(b"\n") == 1
    assert "BEGIN CERTIFICATE" not in response.text


def test_list_orders_by_created_at_then_profile_id(app, client, root_id, chain):
    # Two profiles created at controlled, distinct times via direct storage
    # so the created_at ordering is deterministic regardless of clock
    # resolution; profile ids are chosen deliberately to test the
    # tie-breaker direction as well.
    from proof_release.db import WorkloadIdentityProfile as Profile

    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    later = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    earlier = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    # ``later`` is created first (older), ``earlier`` second (newer); the
    # created_at ordering must dominate the id ordering.
    with app.state.session_factory() as session:
        session.add(
            Profile(
                profile_id=later,
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                trust_root_id=root_id,
                claims_fingerprint="older",
                status="active",
                created_at=base,
            )
        )
        session.add(
            Profile(
                profile_id=earlier,
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                trust_root_id=root_id,
                claims_fingerprint="newer",
                status="active",
                created_at=base + timedelta(seconds=1),
            )
        )
        session.commit()

    response = _list(client, root_id)
    ids = [p["profile_id"] for p in response.json()["profiles"]]
    assert ids == [later, earlier]


def test_list_tie_breaks_on_profile_id_for_equal_created_at(
    app, client, root_id
):
    from datetime import datetime, timezone

    from proof_release.db import WorkloadIdentityProfile as Profile

    moment = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ids = [
        "33333333-3333-3333-3333-333333333333",
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
    ]
    with app.state.session_factory() as session:
        for index, profile_id in enumerate(ids):
            session.add(
                Profile(
                    profile_id=profile_id,
                    tenant_id=TENANT,
                    workload_id=WORKLOAD,
                    trust_root_id=root_id,
                    claims_fingerprint=f"fp-{index}",
                    status="active",
                    created_at=moment,
                )
            )
        session.commit()

    response = _list(client, root_id)
    assert [p["profile_id"] for p in response.json()["profiles"]] == sorted(ids)


def test_list_narrows_to_one_profile_id(client, root_id, chain):
    first = _register(client, root_id, [LEAF_URI])
    _register(client, root_id, [OTHER_URI])
    response = _list(client, root_id, profile_id=first.json()["profile_id"])
    assert response.status_code == 200
    profiles = response.json()["profiles"]
    assert len(profiles) == 1
    assert profiles[0]["profile_id"] == first.json()["profile_id"]


def test_list_includes_revoked_profiles_with_revoked_status(client, root_id, chain):
    registered = _register(client, root_id, [LEAF_URI])
    profile_id = registered.json()["profile_id"]
    revoked = _revoke(client, profile_id)
    assert revoked.status_code == 200

    response = _list(client, root_id)
    profile = response.json()["profiles"][0]
    assert profile["status"] == "revoked"

    narrowed = _list(client, root_id, profile_id=profile_id)
    assert narrowed.status_code == 200
    assert narrowed.json()["profiles"][0]["status"] == "revoked"


def test_list_is_scoped_to_tenant_workload_and_root(client, chain):
    # Same certificate configured as a trust root in two scopes.
    first_root = client.post(
        "/v1/trust-roots",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD, "root_pem": pem(chain["root_cert"])},
    ).json()["root_id"]
    other_root = client.post(
        "/v1/trust-roots",
        json={"tenant_id": OTHER_TENANT, "workload_id": WORKLOAD, "root_pem": pem(chain["root_cert"])},
    ).json()["root_id"]
    _register(client, first_root, [LEAF_URI])
    _register(client, other_root, [LEAF_URI], tenant=OTHER_TENANT)

    own = _list(client, first_root)
    assert len(own.json()["profiles"]) == 1
    other = _list(client, other_root, tenant=OTHER_TENANT)
    assert len(other.json()["profiles"]) == 1
    assert other.json()["profiles"][0]["tenant_id"] == OTHER_TENANT
    # A scope with no profiles is an empty array, not an error.
    assert (
        _list(client, first_root, tenant=TENANT, workload=OTHER_WORKLOAD).json()
        == {"profiles": []}
    )


@pytest.mark.parametrize(
    "params",
    [
        {"workload_id": WORKLOAD, "trust_root_id": "x"},               # missing tenant
        {"tenant_id": TENANT, "trust_root_id": "x"},                   # missing workload
        {"tenant_id": TENANT, "workload_id": WORKLOAD},                # missing root
        {"tenant_id": "  ", "workload_id": WORKLOAD, "trust_root_id": "x"},
        {"tenant_id": TENANT, "workload_id": "\t", "trust_root_id": "x"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "trust_root_id": "not-a-uuid"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "trust_root_id": "x",
         "profile_id": "not-a-uuid"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "trust_root_id": "x",
         "profile_id": "  "},
    ],
)
def test_list_invalid_parameters_return_422(client, root_id, params):
    response = client.get("/v1/workload-identities", params=params)
    assert response.status_code == 422


def test_list_unknown_query_parameter_returns_422(client, root_id):
    response = _list(client, root_id, unexpected="1")
    assert response.status_code == 422


def test_list_explicit_unknown_profile_returns_404(client, root_id):
    response = _list(
        client, root_id, profile_id="11111111-1111-1111-1111-111111111111"
    )
    assert response.status_code == 404


def test_list_cross_scope_profile_returns_404(client, chain):
    first_root = client.post(
        "/v1/trust-roots",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD, "root_pem": pem(chain["root_cert"])},
    ).json()["root_id"]
    other_root = client.post(
        "/v1/trust-roots",
        json={"tenant_id": OTHER_TENANT, "workload_id": WORKLOAD, "root_pem": pem(chain["root_cert"])},
    ).json()["root_id"]
    registered = _register(client, first_root, [LEAF_URI])
    profile_id = registered.json()["profile_id"]

    # The id exists but each query names a different scope/root.
    assert _list(client, first_root, profile_id=profile_id,
                 tenant=OTHER_TENANT).status_code == 404
    assert _list(client, first_root, profile_id=profile_id,
                 workload=OTHER_WORKLOAD).status_code == 404
    assert _list(client, other_root, profile_id=profile_id,
                 tenant=OTHER_TENANT).status_code == 404


def test_list_query_failure_returns_500(app, client, root_id, chain):
    _register(client, root_id, [LEAF_URI])
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE workload_identity_claims"))
    response = _list(client, root_id)
    assert response.status_code == 500
    assert "BEGIN CERTIFICATE" not in response.text


# --- update -----------------------------------------------------------------


def test_update_replaces_claims_keeps_created_at_and_sets_updated_at(
    client, root_id, chain
):
    registered = _register(client, root_id, [LEAF_URI])
    profile_id = registered.json()["profile_id"]
    created_at = registered.json()["created_at"]

    response = _update(client, profile_id, root_id, [OTHER_URI, THIRD_URI])
    assert response.status_code == 200
    body = response.json()
    assert body["profile_id"] == profile_id
    assert body["tenant_id"] == TENANT
    assert body["workload_id"] == WORKLOAD
    assert body["trust_root_id"] == root_id
    assert body["created_at"] == created_at
    assert [c["uri"] for c in body["claims"]] == [OTHER_URI, THIRD_URI]
    assert body["updated_at"] is not None
    assert body["updated_at"].endswith("+00:00")
    assert response.content.endswith(b"\n")
    assert response.content.count(b"\n") == 1
    assert set(body.keys()) == {
        "profile_id",
        "tenant_id",
        "workload_id",
        "trust_root_id",
        "claims",
        "created_at",
        "updated_at",
    }

    # The old claims are gone; the stored set is exactly the new one.
    with client.app.state.session_factory() as session:
        rows = session.scalars(
            select(WorkloadIdentityClaim).where(
                WorkloadIdentityClaim.profile_id == profile_id
            )
        ).all()
        assert sorted(c.uri for c in rows) == [OTHER_URI, THIRD_URI]


def test_update_normalizes_duplicates_and_order(client, root_id, chain):
    profile_id = _register(client, root_id, [LEAF_URI]).json()["profile_id"]
    # Duplicate entries collapse in first-seen order, exactly like
    # registration.
    response = _update(client, profile_id, root_id, [THIRD_URI, OTHER_URI, THIRD_URI])
    assert response.status_code == 200
    assert [c["uri"] for c in response.json()["claims"]] == [THIRD_URI, OTHER_URI]


def test_update_with_own_current_set_is_idempotent(client, root_id, chain):
    profile_id = _register(client, root_id, [LEAF_URI, OTHER_URI]).json()["profile_id"]

    # Same set (reordered, with a duplicate) on a never-updated profile.
    first = _update(client, profile_id, root_id, [OTHER_URI, LEAF_URI, OTHER_URI])
    assert first.status_code == 200
    assert [c["uri"] for c in first.json()["claims"]] == [LEAF_URI, OTHER_URI]
    assert first.json()["updated_at"] is None

    # After a real change, resubmitting the then-current set keeps the
    # earlier updated_at rather than minting a new one.
    changed = _update(client, profile_id, root_id, [THIRD_URI])
    assert changed.status_code == 200
    updated_at = changed.json()["updated_at"]
    again = _update(client, profile_id, root_id, [THIRD_URI])
    assert again.status_code == 200
    assert again.json()["updated_at"] == updated_at
    assert [c["uri"] for c in again.json()["claims"]] == [THIRD_URI]


def test_update_to_another_profiles_set_returns_409(client, root_id, chain):
    first = _register(client, root_id, [LEAF_URI])
    second = _register(client, root_id, [OTHER_URI])
    first_id = first.json()["profile_id"]

    response = _update(client, first_id, root_id, [OTHER_URI])
    assert response.status_code == 409
    # Neither profile changed.
    listed = {p["profile_id"]: p for p in _list(client, root_id).json()["profiles"]}
    assert [c["uri"] for c in listed[first_id]["claims"]] == [LEAF_URI]
    assert [c["uri"] for c in listed[second.json()["profile_id"]]["claims"]] == [OTHER_URI]


def test_update_to_revoked_profiles_set_returns_409(client, root_id, chain):
    first = _register(client, root_id, [LEAF_URI])
    second = _register(client, root_id, [OTHER_URI])
    assert _revoke(client, second.json()["profile_id"]).status_code == 200
    # Revoked profiles still occupy their claim set.
    assert _update(client, first.json()["profile_id"], root_id, [OTHER_URI]).status_code == 409


@pytest.mark.parametrize(
    "payload",
    [
        {"workload_id": WORKLOAD, "trust_root_id": "0" * 36, "claims": []},
        {"tenant_id": TENANT, "trust_root_id": "0" * 36, "claims": []},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": []},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": []},
        {"tenant_id": "  ", "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": [{"issuer": "a", "subject": "b", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "not-a-uuid",
         "claims": [{"issuer": "a", "subject": "b", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36,
         "claims": [{"subject": "b", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36,
         "claims": [{"issuer": "a", "subject": "   ", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": ["nope"]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": {"issuer": "a"}},
        # Unknown top-level and claim-level fields.
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36,
         "claims": [{"issuer": "a", "subject": "b", "uri": "c"}],
         "unexpected": True},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36,
         "claims": [{"issuer": "a", "subject": "b", "uri": "c", "extra": 1}]},
    ],
)
def test_update_invalid_bodies_return_422(client, root_id, chain, payload):
    profile_id = _register(client, root_id, [LEAF_URI]).json()["profile_id"]
    before = _list(client, root_id, profile_id=profile_id).json()
    response = client.put(
        f"/v1/workload-identities/{profile_id}", json=payload
    )
    assert response.status_code == 422
    after = _list(client, root_id, profile_id=profile_id).json()
    assert after == before


@pytest.mark.parametrize(
    "path_id",
    [
        "not-a-uuid",
        "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
        "  ",
        "",
    ],
)
def test_update_invalid_path_identifier_returns_422(client, root_id, chain, path_id):
    response = client.put(
        f"/v1/workload-identities/{path_id}",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "claims": [_claim(LEAF_URI)],
        },
    )
    assert response.status_code == 422


def test_update_empty_path_segment_returns_422(client, root_id, chain):
    response = client.put(
        "/v1/workload-identities/",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "claims": [_claim(LEAF_URI)],
        },
    )
    assert response.status_code == 422


def test_update_wrong_types_return_422(client, root_id, chain):
    profile_id = _register(client, root_id, [LEAF_URI]).json()["profile_id"]
    base = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "claims": [_claim(LEAF_URI)],
    }
    for field, value in [
        ("tenant_id", 123),
        ("workload_id", None),
        ("trust_root_id", ["x"]),
        ("claims", "nope"),
    ]:
        payload = dict(base)
        payload[field] = value
        assert client.put(
            f"/v1/workload-identities/{profile_id}", json=payload
        ).status_code == 422, field


def test_update_unknown_or_cross_scope_profile_returns_404(client, root_id, chain):
    profile_id = _register(client, root_id, [LEAF_URI]).json()["profile_id"]

    # Unknown id in an existing scope.
    assert _update(
        client, "11111111-1111-1111-1111-111111111111", root_id, [OTHER_URI]
    ).status_code == 404
    # Cross-tenant / cross-workload.
    assert _update(
        client, profile_id, root_id, [OTHER_URI], tenant=OTHER_TENANT
    ).status_code == 404
    assert _update(
        client, profile_id, root_id, [OTHER_URI], workload=OTHER_WORKLOAD
    ).status_code == 404
    # Unknown trust root in the scope.
    assert _update(
        client,
        profile_id,
        "22222222-2222-2222-2222-222222222222",
        [OTHER_URI],
    ).status_code == 404


def test_update_does_not_reactivate_a_revoked_profile(client, app, root_id, chain):
    profile_id = _register(client, root_id, [LEAF_URI]).json()["profile_id"]
    revoked = _revoke(client, profile_id)
    assert revoked.status_code == 200
    revoked_at = revoked.json()["revoked_at"]

    response = _update(client, profile_id, root_id, [OTHER_URI])
    assert response.status_code == 200
    assert [c["uri"] for c in response.json()["claims"]] == [OTHER_URI]
    # Revocation is terminal and its timestamp immutable: the new claims do
    # not bring the profile back into the gate or move revoked_at.
    listed = _list(client, root_id, profile_id=profile_id).json()["profiles"][0]
    assert listed["status"] == "revoked"
    with app.state.session_factory() as session:
        from proof_release.app import _rfc3339

        stored = session.get(WorkloadIdentityProfile, profile_id)
        assert stored.status == "revoked"
        assert _rfc3339(stored.revoked_at) == revoked_at


def test_concurrent_identical_updates_settle_at_most_once(app, root_id, chain):
    profile_id = TestClient(app).post(
        "/v1/workload-identities",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "claims": [_claim(LEAF_URI)],
        },
    ).json()["profile_id"]
    target = [_claim(OTHER_URI)]

    def update():
        return TestClient(app).put(
            f"/v1/workload-identities/{profile_id}",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "trust_root_id": root_id,
                "claims": target,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: update(), range(12)))

    assert all(r.status_code == 200 for r in responses)
    # Every response reports the same post-update result; the old claim set
    # was replaced exactly once (one target claim, no duplicates).
    with app.state.session_factory() as session:
        rows = session.scalars(
            select(WorkloadIdentityClaim).where(
                WorkloadIdentityClaim.profile_id == profile_id
            )
        ).all()
        assert len(rows) == 1
        assert rows[0].uri == OTHER_URI
        profile = session.get(WorkloadIdentityProfile, profile_id)
        assert profile.claims_fingerprint is not None
    # All callers observed the same single updated_at.
    updated_ats = {r.json()["updated_at"] for r in responses}
    assert len(updated_ats) == 1


def test_failed_update_claim_write_leaves_old_profile(app, client, root_id, chain):
    profile_id = _register(client, root_id, [LEAF_URI]).json()["profile_id"]
    engine = app.state.engine

    def fail_insert(conn, cursor, statement, parameters, context, executemany):
        if "INSERT INTO workload_identity_claims" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_insert)
    try:
        response = _update(client, profile_id, root_id, [OTHER_URI])
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_insert)

    with app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, profile_id)
        claims = session.scalars(
            select(WorkloadIdentityClaim).where(
                WorkloadIdentityClaim.profile_id == profile_id
            )
        ).all()
        assert profile.status == "active"
        assert profile.updated_at is None
        assert profile.revoked_at is None
        assert [c.uri for c in claims] == [LEAF_URI]

    # Recovery: the same update then succeeds.
    recovered = _update(client, profile_id, root_id, [OTHER_URI])
    assert recovered.status_code == 200


def test_failed_update_commit_leaves_old_profile(app, client, root_id, chain):
    profile_id = _register(client, root_id, [LEAF_URI]).json()["profile_id"]
    engine = app.state.engine

    def fail_update(conn, cursor, statement, parameters, context, executemany):
        # The profile fingerprint/updated_at move is an UPDATE; fail it.
        if "UPDATE workload_identity_profiles" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_update)
    try:
        response = _update(client, profile_id, root_id, [OTHER_URI])
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_update)

    with app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, profile_id)
        uris = [
            c.uri
            for c in session.scalars(
                select(WorkloadIdentityClaim).where(
                    WorkloadIdentityClaim.profile_id == profile_id
                )
            )
        ]
        assert profile.updated_at is None
        assert uris == [LEAF_URI]


# --- revocation -------------------------------------------------------------


def test_revoke_returns_200_with_revoked_flag_and_time(client, root_id, chain):
    registered = _register(client, root_id, [LEAF_URI])
    profile_id = registered.json()["profile_id"]
    created_at = registered.json()["created_at"]

    response = _revoke(client, profile_id)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "profile_id",
        "tenant_id",
        "workload_id",
        "trust_root_id",
        "claims",
        "created_at",
        "revoked",
        "revoked_at",
    }
    assert body["profile_id"] == profile_id
    assert body["claims"] == [_claim(LEAF_URI)]
    assert body["created_at"] == created_at
    assert body["revoked"] is True
    assert body["revoked_at"].endswith("+00:00")
    assert response.content.endswith(b"\n")
    assert response.content.count(b"\n") == 1
    assert b", " not in response.content
    assert "BEGIN CERTIFICATE" not in response.text


def test_revoke_changes_status_and_excludes_profile_from_gate(
    client, root_id, chain
):
    # A strict (non-matching) profile rejects evidence while active.
    profile_id = _register(client, root_id, [OTHER_URI]).json()["profile_id"]
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    _, before = _settle_verification(client, created, evidence)
    assert before.json()["status"] == "rejected"

    assert _revoke(client, profile_id).status_code == 200
    with client.app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, profile_id)
        assert profile.status == "revoked"
        assert profile.revoked_at is not None

    # Revoked profile remains queryable ...
    listed = _list(client, root_id, profile_id=profile_id)
    assert listed.status_code == 200
    assert listed.json()["profiles"][0]["status"] == "revoked"

    # ... but no longer participates in the gate: with no active profile
    # for the anchor, the ordinary verifier verdict stands (verified).
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    _, after = _settle_verification(client, created, evidence)
    assert after.json()["status"] == "verified"


def test_repeat_revoke_returns_409_and_keeps_revoked_at(client, root_id):
    profile_id = _register(client, root_id, [LEAF_URI]).json()["profile_id"]
    first = _revoke(client, profile_id)
    assert first.status_code == 200
    revoked_at = first.json()["revoked_at"]

    second = _revoke(client, profile_id)
    assert second.status_code == 409

    with client.app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, profile_id)
        assert profile.status == "revoked"
        from proof_release.app import _rfc3339

        assert _rfc3339(profile.revoked_at) == revoked_at


@pytest.mark.parametrize(
    "path_id",
    ["not-a-uuid", "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA", " "],
)
def test_revoke_invalid_path_returns_422(client, root_id, path_id):
    response = client.request(
        "DELETE",
        f"/v1/workload-identities/{path_id}",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_revoke_empty_path_segment_returns_422(client):
    response = client.request(
        "DELETE",
        "/v1/workload-identities/",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"workload_id": WORKLOAD},
        {"tenant_id": TENANT},
        {"tenant_id": "  ", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": "\t"},
        {"tenant_id": 1, "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "unexpected": 1},
    ],
)
def test_revoke_invalid_body_returns_422(client, root_id, body):
    profile_id = _register(client, root_id, [LEAF_URI]).json()["profile_id"]
    response = client.request(
        "DELETE", f"/v1/workload-identities/{profile_id}", json=body
    )
    assert response.status_code == 422
    # A 422 never revokes.
    with client.app.state.session_factory() as session:
        assert session.get(WorkloadIdentityProfile, profile_id).status == "active"


def test_revoke_unknown_or_cross_scope_profile_returns_404(client, root_id):
    profile_id = _register(client, root_id, [LEAF_URI]).json()["profile_id"]
    assert _revoke(
        client, "11111111-1111-1111-1111-111111111111"
    ).status_code == 404
    assert _revoke(client, profile_id, tenant=OTHER_TENANT).status_code == 404
    assert _revoke(client, profile_id, workload=OTHER_WORKLOAD).status_code == 404
    # The cross-scope 404 left the profile active.
    with client.app.state.session_factory() as session:
        assert session.get(WorkloadIdentityProfile, profile_id).status == "active"


def test_concurrent_revokes_settle_at_most_once(app, root_id, chain):
    profile_id = TestClient(app).post(
        "/v1/workload-identities",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "claims": [_claim(LEAF_URI)],
        },
    ).json()["profile_id"]

    def revoke():
        return TestClient(app).request(
            "DELETE",
            f"/v1/workload-identities/{profile_id}",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: revoke(), range(12)))

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(200) == 1
    assert statuses.count(409) == 11
    with app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, profile_id)
        assert profile.status == "revoked"


def test_failed_revoke_commit_leaves_active_profile(app, client, root_id):
    profile_id = _register(client, root_id, [LEAF_URI]).json()["profile_id"]
    engine = app.state.engine

    def fail_update(conn, cursor, statement, parameters, context, executemany):
        if "UPDATE workload_identity_profiles" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_update)
    try:
        response = _revoke(client, profile_id)
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_update)

    with app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, profile_id)
        assert profile.status == "active"
        assert profile.revoked_at is None

    recovered = _revoke(client, profile_id)
    assert recovered.status_code == 200


def test_revoke_claim_read_failure_leaves_active_profile(app, client, root_id):
    profile_id = _register(client, root_id, [LEAF_URI]).json()["profile_id"]
    engine = app.state.engine

    def fail_claim_select(conn, cursor, statement, parameters, context, executemany):
        normalized = statement.strip().upper()
        if normalized.startswith("SELECT") and "WORKLOAD_IDENTITY_CLAIMS" in normalized:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_claim_select)
    try:
        response = _revoke(client, profile_id)
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_claim_select)

    # The read failed before the status write: still active, no revoked_at.
    with app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, profile_id)
        assert profile.status == "active"
        assert profile.revoked_at is None

    assert _revoke(client, profile_id).status_code == 200


# --- non-retroactivity ------------------------------------------------------


def test_settled_rejection_is_not_changed_by_later_revoke(client, root_id, chain):
    profile_id = _register(client, root_id, [OTHER_URI]).json()["profile_id"]
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    evidence_id, first = _settle_verification(client, created, evidence)
    assert first.json()["status"] == "rejected"

    assert _revoke(client, profile_id).status_code == 200

    repeated = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert repeated.status_code == 200
    assert repeated.json() == first.json()


def test_settled_verification_is_not_changed_by_later_update(
    client, root_id, chain
):
    # A matching profile verifies; a later update that makes the profile
    # non-matching must not rewrite the settled conclusion.
    _register(client, root_id, [LEAF_URI])
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    evidence_id, first = _settle_verification(client, created, evidence)
    assert first.json()["status"] == "verified"

    profile_id = _list(client, root_id).json()["profiles"][0]["profile_id"]
    updated = _update(client, profile_id, root_id, [OTHER_URI])
    assert updated.status_code == 200

    repeated = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert repeated.json()["status"] == "verified"


def test_revocation_committed_after_verification_is_not_retroactive(
    tmp_path, chain
):
    """A revocation that can only commit while a verification is mid-flight
    cannot affect that verification; only later evidence observes it."""

    entered = threading.Event()
    release = threading.Event()

    class GatedX509Verifier(X509AttestedNonceJSONVerifier):
        def verify(self, context: VerificationContext):
            entered.set()
            release.wait(timeout=10)
            return super().verify(context)

    registry = VerifierRegistry()
    registry.register(GatedX509Verifier())
    application = create_app(f"sqlite:///{tmp_path}/gated.db", registry)
    client = TestClient(application)
    root = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    ).json()["root_id"]
    # Non-matching active profile: the in-flight evidence must reject.
    profile_id = client.post(
        "/v1/workload-identities",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root,
            "claims": [_claim(OTHER_URI)],
        },
    ).json()["profile_id"]

    challenge = _challenge(client)
    evidence = _leaf_evidence(challenge, chain)
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": challenge["challenge_id"],
            "nonce": challenge["nonce"],
            "evidence_format": X509_ATTESTED_NONCE_JSON,
            "evidence": evidence,
        },
    )
    evidence_id = submitted.json()["evidence_id"]

    def verify():
        return TestClient(application).post(
            f"/v1/evidence/{evidence_id}/verify",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "nonce": challenge["nonce"],
                "evidence": evidence,
            },
        )

    def revoke():
        return TestClient(application).request(
            "DELETE",
            f"/v1/workload-identities/{profile_id}",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        verify_future = pool.submit(verify)
        assert entered.wait(timeout=5)
        revoke_future = pool.submit(revoke)
        threading.Event().wait(0.5)
        release.set()
        verify_response = verify_future.result(timeout=15)
        revoke_response = revoke_future.result(timeout=15)

    # The in-flight verification observed the still-active (non-matching)
    # profile and settled rejected; the revocation committed afterwards.
    assert verify_response.status_code == 200
    assert verify_response.json()["status"] == "rejected"
    assert revoke_response.status_code == 200

    # The settled conclusion is never rewritten by the now-committed
    # revocation.
    repeated = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": challenge["nonce"],
            "evidence": evidence,
        },
    )
    assert repeated.json()["status"] == "rejected"

    # New evidence after the revocation sees no active profile and takes
    # the ordinary verifier verdict.
    fresh_challenge = _challenge(client)
    fresh_evidence = _leaf_evidence(fresh_challenge, chain)
    _, fresh = _settle_verification(client, fresh_challenge, fresh_evidence)
    assert fresh.json()["status"] == "verified"
    application.state.engine.dispose()


# --- registration compatibility ---------------------------------------------


def test_registration_unknown_fields_now_rejected(client, root_id, chain):
    payload = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "claims": [_claim(LEAF_URI)],
        "unexpected": "value",
    }
    response = client.post("/v1/workload-identities", json=payload)
    assert response.status_code == 422
    with client.app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentityProfile)).all() == []


def test_legacy_database_is_migrated_with_active_status_and_positions(
    tmp_path, chain
):
    import sqlalchemy as sa

    url = f"sqlite:///{tmp_path}/legacy.db"
    engine = sa.create_engine(url)
    # Hand-build the pre-change schema (no status/updated_at/revoked_at on
    # profiles; no position on claims).
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "CREATE TABLE workload_identity_profiles ("
                "profile_id VARCHAR(36) PRIMARY KEY, "
                "tenant_id VARCHAR(256), workload_id VARCHAR(256), "
                "trust_root_id VARCHAR(36), claims_fingerprint VARCHAR(64), "
                "created_at DATETIME)"
            )
        )
        conn.execute(
            sa.text(
                "CREATE TABLE workload_identity_claims ("
                "claim_id VARCHAR(36) PRIMARY KEY, profile_id VARCHAR(36), "
                "tenant_id VARCHAR(256), workload_id VARCHAR(256), "
                "trust_root_id VARCHAR(36), issuer VARCHAR(1024), "
                "subject VARCHAR(1024), uri VARCHAR(2048))"
            )
        )
        conn.execute(
            sa.text(
                "INSERT INTO workload_identity_profiles VALUES ("
                "'p1','t','w','r','fp','2026-01-01 00:00:00.000000')"
            )
        )
        conn.execute(
            sa.text(
                "INSERT INTO workload_identity_claims (claim_id, profile_id, "
                "tenant_id, workload_id, trust_root_id, issuer, subject, uri) "
                "VALUES ('c1','p1','t','w','r','i','s','u1')"
            )
        )
        conn.execute(
            sa.text(
                "INSERT INTO workload_identity_claims (claim_id, profile_id, "
                "tenant_id, workload_id, trust_root_id, issuer, subject, uri) "
                "VALUES ('c2','p1','t','w','r','i','s','u2')"
            )
        )
    engine.dispose()

    application = create_app(url)
    with application.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, "p1")
        assert profile.status == "active"
        assert profile.updated_at is None
        assert profile.revoked_at is None
        claims = session.scalars(
            select(WorkloadIdentityClaim)
            .where(WorkloadIdentityClaim.profile_id == "p1")
            .order_by(WorkloadIdentityClaim.position)
        ).all()
        assert [(c.uri, c.position) for c in claims] == [("u1", 0), ("u2", 1)]
    application.state.engine.dispose()
