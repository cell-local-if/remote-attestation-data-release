"""Tests for the workload identity profile query/update/revoke lifecycle
and for the interaction of revocation and replacement with the X.509
identity gate."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select, text

from proof_release.app import create_app
from proof_release.db import (
    WorkloadIdentityClaim,
    WorkloadIdentityProfile,
)
from proof_release.verifiers import X509_ATTESTED_NONCE_JSON

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
THIRD_URI = "spiffe://example.org/workload/risk"


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


def _claim(leaf_cert, *, uri=LEAF_URI):
    return {
        "issuer": leaf_cert.issuer.rfc4514_string(),
        "subject": leaf_cert.subject.rfc4514_string(),
        "uri": uri,
    }


def _register(client, root, claims, *, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        "/v1/workload-identities",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "trust_root_id": root,
            "claims": claims,
        },
    )


def _query(client, root, *, profile=None, tenant=TENANT, workload=WORKLOAD, **extra):
    params = {"tenant_id": tenant, "workload_id": workload, "trust_root_id": root}
    if profile is not None:
        params["profile_id"] = profile
    params.update(extra)
    return client.get("/v1/workload-identities", params=params)


def _update(client, profile, root, claims, *, tenant=TENANT, workload=WORKLOAD):
    return client.put(
        f"/v1/workload-identities/{profile}",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "trust_root_id": root,
            "claims": claims,
        },
    )


def _revoke(client, profile, root, *, tenant=TENANT, workload=WORKLOAD):
    # The installed httpx TestClient.delete does not accept json=; the
    # generic request call serializes the body identically.
    return client.request(
        "DELETE",
        f"/v1/workload-identities/{profile}",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "trust_root_id": root,
        },
    )


def _challenge(client, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/challenges", json={"tenant_id": tenant, "workload_id": workload}
    )
    assert response.status_code == 201
    return response.json()


def _verify_leaf(client, chain, *, tenant=TENANT, workload=WORKLOAD):
    created = _challenge(client, tenant, workload)
    evidence = make_evidence(
        created["nonce"],
        chain["leaf_key"],
        [chain["leaf_cert"], chain["intermediate_cert"], chain["root_cert"]],
        {"m": "abc"},
    )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
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
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    return evidence_id, evidence, created, response


# --- query ------------------------------------------------------------------


def test_query_empty_range_returns_empty_array(client, root_id):
    response = _query(client, root_id)
    assert response.status_code == 200
    assert response.content == b"[]\n"
    assert response.json() == []


def test_query_unknown_trust_root_is_empty_range(client):
    response = _query(client, "11111111-1111-1111-1111-111111111111")
    assert response.status_code == 200
    assert response.content == b"[]\n"


def test_query_returns_profiles_with_status_and_creation_fields(
    client, root_id, chain
):
    claim = _claim(chain["leaf_cert"])
    registered = _register(client, root_id, [claim])
    assert registered.status_code == 201
    profile_id = registered.json()["profile_id"]

    response = _query(client, root_id)
    assert response.status_code == 200
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert b", " not in response.content
    assert b": " not in response.content
    body = response.json()
    assert len(body) == 1
    assert set(body[0].keys()) == {
        "profile_id",
        "tenant_id",
        "workload_id",
        "trust_root_id",
        "claims",
        "status",
        "created_at",
    }
    assert body[0] == {
        "profile_id": profile_id,
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "claims": [claim],
        "status": "active",
        "created_at": registered.json()["created_at"],
    }
    assert "BEGIN CERTIFICATE" not in response.text


def test_query_orders_by_created_at_then_profile_id(app, client, root_id):
    now = datetime.now(timezone.utc)
    with app.state.session_factory() as session:
        for index, created in enumerate(
            (now - timedelta(seconds=30), now, now - timedelta(seconds=10))
        ):
            session.add(
                WorkloadIdentityProfile(
                    profile_id=f"00000000-0000-0000-0000-{index:012d}",
                    tenant_id=TENANT,
                    workload_id=WORKLOAD,
                    trust_root_id=root_id,
                    claims_fingerprint=f"fp-{index}",
                    status="active",
                    created_at=created,
                )
            )
        session.commit()
    # Two profiles share one creation instant; they tie-break on id.
    with app.state.session_factory() as session:
        session.add(
            WorkloadIdentityProfile(
                profile_id="00000000-0000-0000-0000-000000000009",
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                trust_root_id=root_id,
                claims_fingerprint="fp-tie",
                status="active",
                created_at=now,
            )
        )
        session.commit()

    response = _query(client, root_id)
    ids = [p["profile_id"] for p in response.json()]
    assert ids == [
        "00000000-0000-0000-0000-000000000000",
        "00000000-0000-0000-0000-000000000002",
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000009",
    ]


def test_query_scopes_by_tenant_workload_and_root(client, chain):
    roots = {}
    for tenant in (TENANT, OTHER_TENANT):
        roots[tenant] = client.post(
            "/v1/trust-roots",
            json={
                "tenant_id": tenant,
                "workload_id": WORKLOAD,
                "root_pem": pem(chain["root_cert"]),
            },
        ).json()["root_id"]
    registered_a = _register(client, roots[TENANT], [_claim(chain["leaf_cert"])])
    _register(
        client,
        roots[OTHER_TENANT],
        [_claim(chain["leaf_cert"])],
        tenant=OTHER_TENANT,
    )

    response = _query(client, roots[TENANT])
    assert [p["profile_id"] for p in response.json()] == [
        registered_a.json()["profile_id"]
    ]
    assert _query(
        client, roots[OTHER_TENANT], tenant=OTHER_TENANT
    ).json().__len__() == 1
    # A profile from another tenant is invisible even when named.
    cross = _query(
        client,
        roots[TENANT],
        profile=registered_a.json()["profile_id"],
        tenant=OTHER_TENANT,
    )
    assert cross.status_code == 404


def test_query_explicit_profile_filter(client, root_id, chain):
    first = _register(client, root_id, [_claim(chain["leaf_cert"], uri=LEAF_URI)])
    _register(client, root_id, [_claim(chain["leaf_cert"], uri=OTHER_URI)])

    one = _query(client, root_id, profile=first.json()["profile_id"])
    assert one.status_code == 200
    assert [p["profile_id"] for p in one.json()] == [first.json()["profile_id"]]

    # Naming the profile while scoping the query to an unknown trust root
    # is outside the requested range: an indistinguishable 404.
    missing = _query(
        client,
        "22222222-2222-2222-2222-222222222222",
        profile=first.json()["profile_id"],
    )
    assert missing.status_code == 404


def test_query_unknown_explicit_profile_returns_404(client, root_id):
    response = _query(
        client, root_id, profile="33333333-3333-3333-3333-333333333333"
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    "params",
    [
        {"workload_id": WORKLOAD, "trust_root_id": "0" * 36},
        {"tenant_id": TENANT, "trust_root_id": "0" * 36},
        {"tenant_id": TENANT, "workload_id": WORKLOAD},
        {"tenant_id": "  ", "workload_id": WORKLOAD, "trust_root_id": "0" * 36},
        {"tenant_id": TENANT, "workload_id": "\t", "trust_root_id": "0" * 36},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "trust_root_id": "not-a-uuid"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "profile_id": "nope"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "profile_id": "  "},
    ],
)
def test_query_invalid_params_return_422(client, params):
    response = client.get("/v1/workload-identities", params=params)
    assert response.status_code == 422


def test_query_unknown_query_parameter_returns_422(client, root_id):
    response = _query(client, root_id, unexpected="x")
    assert response.status_code == 422


@pytest.mark.parametrize("body", [b'{"unexpected": 1}', b"   ", b"null", b"[]"])
def test_query_with_non_empty_body_returns_422(client, root_id, body):
    # The range comes entirely from query parameters; any body is a
    # malformed query rejected before any state is read.
    response = client.request(
        "GET",
        "/v1/workload-identities",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
        },
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_query_empty_body_is_accepted(client, root_id, chain):
    _register(client, root_id, [_claim(chain["leaf_cert"])])
    response = client.request(
        "GET",
        "/v1/workload-identities",
        params={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
        },
        content=b"",
    )
    assert response.status_code == 200
    assert len(response.json()) == 1


# --- update -----------------------------------------------------------------


def test_update_replaces_claims_keeps_identity_and_creation(client, root_id, chain):
    original_claim = _claim(chain["leaf_cert"], uri=LEAF_URI)
    registered = _register(client, root_id, [original_claim])
    body = registered.json()
    created_at = body["created_at"]

    new_claims = [
        _claim(chain["leaf_cert"], uri=OTHER_URI),
        _claim(chain["leaf_cert"], uri=THIRD_URI),
    ]
    response = _update(client, body["profile_id"], root_id, new_claims)
    assert response.status_code == 200
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    updated = response.json()
    assert set(updated.keys()) == {
        "profile_id",
        "tenant_id",
        "workload_id",
        "trust_root_id",
        "claims",
        "created_at",
        "updated_at",
    }
    assert updated["profile_id"] == body["profile_id"]
    assert updated["tenant_id"] == TENANT
    assert updated["workload_id"] == WORKLOAD
    assert updated["trust_root_id"] == root_id
    assert updated["claims"] == new_claims
    assert updated["created_at"] == created_at
    assert updated["updated_at"].endswith("+00:00")
    assert updated["updated_at"] >= created_at

    # The query shows the replacement; old claim rows are gone.
    queried = _query(client, root_id, profile=body["profile_id"]).json()[0]
    assert queried["claims"] == new_claims
    with client.app.state.session_factory() as session:
        rows = session.scalars(select(WorkloadIdentityClaim)).all()
        assert sorted((c.uri, c.seq) for c in rows) == sorted(
            [(OTHER_URI, 0), (THIRD_URI, 1)]
        )


def test_update_dedupes_claims_like_registration(client, root_id, chain):
    claim_a = _claim(chain["leaf_cert"], uri=LEAF_URI)
    claim_b = _claim(chain["leaf_cert"], uri=OTHER_URI)
    registered = _register(client, root_id, [claim_a]).json()

    response = _update(
        client, registered["profile_id"], root_id, [claim_b, claim_a, claim_b]
    )
    assert response.status_code == 200
    assert response.json()["claims"] == [claim_b, claim_a]


def test_update_to_own_set_is_idempotent(client, root_id, chain):
    claim = _claim(chain["leaf_cert"])
    registered = _register(client, root_id, [claim]).json()

    first = _update(client, registered["profile_id"], root_id, [dict(claim)])
    assert first.status_code == 200
    # Never updated before: no updated_at is minted by a no-op.
    assert "updated_at" not in first.json()
    assert first.json()["claims"] == [claim]

    # A genuine replacement sets updated_at; repeating the same (now
    # current) set does not rewrite it.
    replaced = _update(
        client, registered["profile_id"], root_id,
        [_claim(chain["leaf_cert"], uri=OTHER_URI)],
    )
    assert replaced.status_code == 200
    updated_at = replaced.json()["updated_at"]
    repeated = _update(
        client,
        registered["profile_id"],
        root_id,
        [_claim(chain["leaf_cert"], uri=OTHER_URI)],
    )
    assert repeated.status_code == 200
    assert repeated.json()["updated_at"] == updated_at
    assert repeated.json()["claims"] == replaced.json()["claims"]


def test_update_to_another_profiles_set_returns_409(client, root_id, chain):
    first = _register(client, root_id, [_claim(chain["leaf_cert"], uri=LEAF_URI)])
    second = _register(client, root_id, [_claim(chain["leaf_cert"], uri=OTHER_URI)])
    assert first.status_code == 201 and second.status_code == 201
    second_set = second.json()["claims"]

    conflict = _update(
        client, first.json()["profile_id"], root_id, second_set
    )
    assert conflict.status_code == 409

    # Neither profile moved.
    first_now = _query(client, root_id, profile=first.json()["profile_id"]).json()[0]
    second_now = _query(client, root_id, profile=second.json()["profile_id"]).json()[0]
    assert first_now["claims"] == first.json()["claims"]
    assert second_now["claims"] == second.json()["claims"]


def test_update_unknown_or_cross_scope_profile_returns_404(client, root_id, chain):
    claim = [_claim(chain["leaf_cert"])]
    unknown = _update(
        client, "44444444-4444-4444-4444-444444444444", root_id, claim
    )
    assert unknown.status_code == 404

    registered = _register(client, root_id, claim).json()
    assert _update(
        client, registered["profile_id"], root_id, claim, tenant=OTHER_TENANT
    ).status_code == 404
    assert _update(
        client, registered["profile_id"], root_id, claim, workload=OTHER_WORKLOAD
    ).status_code == 404
    other_root = "55555555-5555-5555-5555-555555555555"
    cross_root = _update(client, registered["profile_id"], other_root, claim)
    assert cross_root.status_code == 404


def test_update_revoked_profile_returns_409_and_changes_nothing(
    client, root_id, chain
):
    claim = [_claim(chain["leaf_cert"])]
    registered = _register(client, root_id, claim).json()
    # Give the profile an updated_at before revoking, so the test also
    # proves a failed update never rewrites it.
    updated = _update(
        client,
        registered["profile_id"],
        root_id,
        [_claim(chain["leaf_cert"], uri=OTHER_URI)],
    ).json()
    assert _revoke(client, registered["profile_id"], root_id).status_code == 200

    # Neither a new set nor the revoked profile's own current set may be
    # written: the terminal set is retained verbatim.
    for target in (
        [_claim(chain["leaf_cert"], uri=THIRD_URI)],
        [_claim(chain["leaf_cert"], uri=OTHER_URI)],
    ):
        response = _update(
            client, registered["profile_id"], root_id, target
        )
        assert response.status_code == 409
        queried = _query(
            client, root_id, profile=registered["profile_id"]
        ).json()[0]
        assert queried["status"] == "revoked"
        assert queried["claims"] == [
            _claim(chain["leaf_cert"], uri=OTHER_URI)
        ]
    with client.app.state.session_factory() as session:
        profile = session.get(
            WorkloadIdentityProfile, registered["profile_id"]
        )
        assert profile.status == "revoked"
        assert profile.updated_at.isoformat() == updated["updated_at"]
        assert profile.revoked_at is not None


def test_update_revoked_profile_cross_scope_is_404_not_409(client, root_id, chain):
    # The scope judgement precedes the terminal-state judgement, so an
    # out-of-scope revoked profile never reveals its existence.
    registered = _register(
        client, root_id, [_claim(chain["leaf_cert"])]
    ).json()
    assert _revoke(client, registered["profile_id"], root_id).status_code == 200
    assert _update(
        client,
        registered["profile_id"],
        root_id,
        [_claim(chain["leaf_cert"])],
        tenant=OTHER_TENANT,
    ).status_code == 404


@pytest.mark.parametrize(
    "profile_id",
    [
        "not-a-uuid",
        "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
        "",
        "%20",
        "%09",
    ],
)
def test_update_invalid_path_identifier_returns_422(
    client, root_id, chain, profile_id
):
    response = client.put(
        f"/v1/workload-identities/{profile_id}",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "claims": [_claim(chain["leaf_cert"])],
        },
    )
    assert response.status_code == 422


def test_update_whitespace_padded_path_identifier_returns_422(
    client, root_id, chain
):
    registered = _register(client, root_id, [_claim(chain["leaf_cert"])]).json()
    response = client.put(
        f"/v1/workload-identities/%20{registered['profile_id']}%20",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "claims": [_claim(chain["leaf_cert"], uri=OTHER_URI)],
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
            "claims": [_claim(chain["leaf_cert"])],
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"workload_id": WORKLOAD, "trust_root_id": "0" * 36,
         "claims": [{"issuer": "a", "subject": "b", "uri": "c"}]},
        {"tenant_id": TENANT, "trust_root_id": "0" * 36,
         "claims": [{"issuer": "a", "subject": "b", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "claims": [{"issuer": "a", "subject": "b", "uri": "c"}]},
        {"tenant_id": " ", "workload_id": WORKLOAD, "trust_root_id": "0" * 36,
         "claims": [{"issuer": "a", "subject": "b", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "not-a-uuid",
         "claims": [{"issuer": "a", "subject": "b", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": []},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36,
         "claims": [{"issuer": "a", "subject": "b"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36,
         "claims": [{"issuer": " ", "subject": "b", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36,
         "claims": [{"issuer": "a", "subject": "b", "uri": "c"}],
         "unexpected": True},
    ],
)
def test_update_invalid_bodies_return_422(client, root_id, chain, payload):
    registered = _register(
        client, root_id, [_claim(chain["leaf_cert"])]
    ).json()
    response = client.put(
        f"/v1/workload-identities/{registered['profile_id']}", json=payload
    )
    assert response.status_code == 422


def test_update_wrong_types_return_422(client, root_id, chain):
    registered = _register(client, root_id, [_claim(chain["leaf_cert"])]).json()
    base = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "claims": [_claim(chain["leaf_cert"])],
    }
    for field, value in [
        ("tenant_id", 1),
        ("workload_id", None),
        ("trust_root_id", 7),
        ("claims", {}),
    ]:
        payload = dict(base)
        payload[field] = value
        response = client.put(
            f"/v1/workload-identities/{registered['profile_id']}", json=payload
        )
        assert response.status_code == 422, field


def test_register_unknown_field_returns_422(client, root_id, chain):
    response = client.post(
        "/v1/workload-identities",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "claims": [_claim(chain["leaf_cert"])],
            "extra": "ignored?",
        },
    )
    assert response.status_code == 422
    with client.app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentityProfile)).all() == []


def test_update_write_failure_leaves_old_set(app, client, root_id, chain):
    registered = _register(client, root_id, [_claim(chain["leaf_cert"])]).json()

    def fail_update(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE workload_identity_profiles"):
            raise RuntimeError("simulated storage failure")

    event.listen(app.state.engine, "before_cursor_execute", fail_update)
    try:
        response = _update(
            client,
            registered["profile_id"],
            root_id,
            [_claim(chain["leaf_cert"], uri=OTHER_URI)],
        )
        assert response.status_code == 500
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_update)

    with app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, registered["profile_id"])
        assert profile.updated_at is None
        claims = session.scalars(select(WorkloadIdentityClaim)).all()
        assert len(claims) == 1
        assert claims[0].uri == LEAF_URI


def test_update_claim_write_failure_rolls_profile_back(app, client, root_id, chain):
    registered = _register(client, root_id, [_claim(chain["leaf_cert"])]).json()

    def fail_insert(conn, cursor, statement, parameters, context, executemany):
        if "INTO workload_identity_claims" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(app.state.engine, "before_cursor_execute", fail_insert)
    try:
        response = _update(
            client,
            registered["profile_id"],
            root_id,
            [_claim(chain["leaf_cert"], uri=OTHER_URI)],
        )
        assert response.status_code == 500
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_insert)

    # Profile fingerprint, updated_at and the claim set are one atomic
    # commit: the failed replacement leaves the original set exactly.
    with app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, registered["profile_id"])
        assert profile.updated_at is None
        claims = session.scalars(select(WorkloadIdentityClaim)).all()
        assert len(claims) == 1
        assert claims[0].uri == LEAF_URI
    recovered = _update(
        client,
        registered["profile_id"],
        root_id,
        [_claim(chain["leaf_cert"], uri=OTHER_URI)],
    )
    assert recovered.status_code == 200


def test_concurrent_identical_updates_settle_once(app, root_id, chain):
    client = TestClient(app)
    profile_id = _register(
        client, root_id, [_claim(chain["leaf_cert"], uri=LEAF_URI)]
    ).json()["profile_id"]
    target = [_claim(chain["leaf_cert"], uri=OTHER_URI)]

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
    updated_ats = {r.json()["updated_at"] for r in responses}
    # Only the winner wrote; every other request observed its result.
    assert len(updated_ats) == 1
    with app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, profile_id)
        claims = session.scalars(select(WorkloadIdentityClaim)).all()
        assert profile.updated_at is not None
        assert len(claims) == 1
        assert claims[0].uri == OTHER_URI


def test_concurrent_distinct_updates_only_one_takes_effect(app, root_id, chain):
    client = TestClient(app)
    profile_id = _register(
        client, root_id, [_claim(chain["leaf_cert"], uri=LEAF_URI)]
    ).json()["profile_id"]
    targets = {
        "b": [_claim(chain["leaf_cert"], uri=OTHER_URI)],
        "c": [_claim(chain["leaf_cert"], uri=THIRD_URI)],
    }

    def update(kind):
        return TestClient(app).put(
            f"/v1/workload-identities/{profile_id}",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "trust_root_id": root_id,
                "claims": targets[kind],
            },
        )

    kinds = ["b", "c"] * 10
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(update, kinds))

    statuses = sorted(r.status_code for r in responses)
    assert set(statuses) <= {200, 409}
    assert statuses.count(200) >= 1
    with app.state.session_factory() as session:
        claims = session.scalars(
            select(WorkloadIdentityClaim).where(
                WorkloadIdentityClaim.profile_id == profile_id
            )
        ).all()
        assert len(claims) == 1
        assert claims[0].uri in {OTHER_URI, THIRD_URI}
        # No duplicate profiles were created by the races.
        assert session.scalars(select(WorkloadIdentityProfile)).all().__len__() == 1


def test_concurrent_update_and_revoke_leave_revoked_set_untouched(
    app, root_id, chain
):
    client = TestClient(app)
    profile_id = _register(
        client, root_id, [_claim(chain["leaf_cert"], uri=LEAF_URI)]
    ).json()["profile_id"]
    target = [_claim(chain["leaf_cert"], uri=OTHER_URI)]

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

    def revoke():
        return TestClient(app).request(
            "DELETE",
            f"/v1/workload-identities/{profile_id}",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "trust_root_id": root_id,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        updaters = [pool.submit(update) for _ in range(10)]
        revokers = [pool.submit(revoke) for _ in range(4)]
        results = [f.result() for f in updaters + revokers]

    # The profile ends revoked. If a revoke wins, every update is a 409
    # and the original set is retained; if an update wins first, exactly
    # one update returns 200 and the revoke still wins the terminal
    # state with the updated set. In no case is a revoked profile later
    # replaced by another update.
    with app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, profile_id)
        assert profile.status == "revoked"
        assert profile.revoked_at is not None
        claims = session.scalars(
            select(WorkloadIdentityClaim).where(
                WorkloadIdentityClaim.profile_id == profile_id
            )
        ).all()
        assert len(claims) == 1
    assert all(r.status_code in (200, 409) for r in results)


# --- revoke -----------------------------------------------------------------


def test_revoke_returns_200_with_revoked_fields(client, root_id, chain):
    claim = _claim(chain["leaf_cert"])
    registered = _register(client, root_id, [claim]).json()

    response = _revoke(client, registered["profile_id"], root_id)
    assert response.status_code == 200
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
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
    assert body["profile_id"] == registered["profile_id"]
    assert body["tenant_id"] == TENANT
    assert body["workload_id"] == WORKLOAD
    assert body["trust_root_id"] == root_id
    assert body["claims"] == [claim]
    assert body["created_at"] == registered["created_at"]
    assert body["revoked"] is True
    assert body["revoked_at"].endswith("+00:00")

    queried = _query(client, root_id, profile=registered["profile_id"]).json()[0]
    assert queried["status"] == "revoked"
    assert queried["claims"] == [claim]


def test_revoke_after_update_preserves_updated_at(client, root_id, chain):
    registered = _register(client, root_id, [_claim(chain["leaf_cert"])]).json()
    updated = _update(
        client,
        registered["profile_id"],
        root_id,
        [_claim(chain["leaf_cert"], uri=OTHER_URI)],
    ).json()

    revoked = _revoke(client, registered["profile_id"], root_id)
    assert revoked.status_code == 200
    body = revoked.json()
    assert body["claims"] == [_claim(chain["leaf_cert"], uri=OTHER_URI)]
    assert body["updated_at"] == updated["updated_at"]
    assert body["revoked_at"] >= body["updated_at"]


def test_repeated_revoke_returns_409_and_keeps_revoked_at(client, root_id, chain):
    registered = _register(client, root_id, [_claim(chain["leaf_cert"])]).json()
    first = _revoke(client, registered["profile_id"], root_id)
    assert first.status_code == 200
    revoked_at = first.json()["revoked_at"]

    second = _revoke(client, registered["profile_id"], root_id)
    assert second.status_code == 409
    with client.app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, registered["profile_id"])
        assert profile.status == "revoked"
        assert profile.revoked_at.isoformat() == revoked_at
        claims = session.scalars(select(WorkloadIdentityClaim)).all()
        assert len(claims) == 1


def test_revoke_unknown_or_cross_scope_returns_404(client, root_id, chain):
    claim = [_claim(chain["leaf_cert"])]
    assert _revoke(
        client, "66666666-6666-6666-6666-666666666666", root_id
    ).status_code == 404

    registered = _register(client, root_id, claim).json()
    assert _revoke(
        client, registered["profile_id"], root_id, tenant=OTHER_TENANT
    ).status_code == 404
    assert _revoke(
        client, registered["profile_id"], root_id, workload=OTHER_WORKLOAD
    ).status_code == 404
    assert _revoke(
        client, registered["profile_id"], "77777777-7777-7777-7777-777777777777"
    ).status_code == 404


@pytest.mark.parametrize(
    "profile_id",
    [
        "not-a-uuid",
        "ABCDEFAB-ABCD-ABCD-ABCD-ABCDEFABCDEF",
        "%20",
        "%09",
    ],
)
def test_revoke_invalid_path_identifier_returns_422(
    client, root_id, chain, profile_id
):
    response = client.request(
        "DELETE",
        f"/v1/workload-identities/{profile_id}",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
        },
    )
    assert response.status_code == 422


def test_revoke_whitespace_padded_path_identifier_returns_422(
    client, root_id, chain
):
    registered = _register(client, root_id, [_claim(chain["leaf_cert"])]).json()
    response = client.request(
        "DELETE",
        f"/v1/workload-identities/%09{registered['profile_id']}%09",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
        },
    )
    assert response.status_code == 422
    # The profile must still be active: a malformed path never revokes.
    assert (
        _query(client, root_id, profile=registered["profile_id"]).json()[0][
            "status"
        ]
        == "active"
    )


def test_revoke_empty_path_segment_returns_422(client, root_id):
    response = client.request(
        "DELETE",
        "/v1/workload-identities/",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"workload_id": WORKLOAD, "trust_root_id": "0" * 36},
        {"tenant_id": TENANT, "trust_root_id": "0" * 36},
        {"tenant_id": TENANT, "workload_id": WORKLOAD},
        {"tenant_id": "\t", "workload_id": WORKLOAD, "trust_root_id": "0" * 36},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "not-a-uuid"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "surprise": 1},
    ],
)
def test_revoke_invalid_bodies_return_422(client, root_id, chain, payload):
    registered = _register(client, root_id, [_claim(chain["leaf_cert"])]).json()
    response = client.request(
        "DELETE",
        f"/v1/workload-identities/{registered['profile_id']}", json=payload
    )
    assert response.status_code == 422


def test_revoke_write_failure_leaves_profile_active(app, client, root_id, chain):
    registered = _register(client, root_id, [_claim(chain["leaf_cert"])]).json()

    def fail_update(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE workload_identity_profiles"):
            raise RuntimeError("simulated storage failure")

    event.listen(app.state.engine, "before_cursor_execute", fail_update)
    try:
        response = _revoke(client, registered["profile_id"], root_id)
        assert response.status_code == 500
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_update)

    with app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, registered["profile_id"])
        assert profile.status == "active"
        assert profile.revoked_at is None

    recovered = _revoke(client, registered["profile_id"], root_id)
    assert recovered.status_code == 200


def test_concurrent_revokes_settle_once(app, root_id, chain):
    client = TestClient(app)
    profile_id = _register(
        client, root_id, [_claim(chain["leaf_cert"])]
    ).json()["profile_id"]

    def revoke():
        return TestClient(app).request(
            "DELETE",
            f"/v1/workload-identities/{profile_id}",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "trust_root_id": root_id,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: revoke(), range(12)))

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(200) == 1
    assert statuses.count(409) == 11
    revoked_ats = {
        r.json()["revoked_at"] for r in responses if r.status_code == 200
    }
    assert len(revoked_ats) == 1
    with app.state.session_factory() as session:
        profile = session.get(WorkloadIdentityProfile, profile_id)
        assert profile.status == "revoked"
        assert profile.revoked_at is not None


# --- gate interaction -------------------------------------------------------


def test_revoked_profile_no_longer_gates_verification(client, root_id, chain):
    # A strict (non-matching) profile rejects while it is active.
    strict = _register(client, root_id, [_claim(chain["leaf_cert"], uri=OTHER_URI)])
    _, _, _, rejected = _verify_leaf(client, chain)
    assert rejected.json()["status"] == "rejected"

    # Revoking the only profile makes the anchor behave as if it had none:
    # the ordinary verdict is unchanged (backward compatible), and the
    # revoked profile remains queryable with status revoked.
    assert _revoke(client, strict.json()["profile_id"], root_id).status_code == 200
    _, _, _, admitted = _verify_leaf(client, chain)
    assert admitted.json()["status"] == "verified"
    queried = _query(client, root_id, profile=strict.json()["profile_id"]).json()[0]
    assert queried["status"] == "revoked"


def test_revoked_matching_profile_does_not_admit_with_only_strict_active(
    client, root_id, chain
):
    # Two profiles: one matching, one strict. The leaf is admitted because
    # any active profile hits; revoking the matching one leaves only the
    # strict profile, so subsequent evidence is rejected.
    matching = _register(client, root_id, [_claim(chain["leaf_cert"])])
    strict = _register(client, root_id, [_claim(chain["leaf_cert"], uri=OTHER_URI)])
    _, _, _, admitted = _verify_leaf(client, chain)
    assert admitted.json()["status"] == "verified"

    assert _revoke(client, matching.json()["profile_id"], root_id).status_code == 200
    _, _, _, rejected = _verify_leaf(client, chain)
    assert rejected.json()["status"] == "rejected"
    del strict


def test_committed_update_is_seen_by_later_verification(client, root_id, chain):
    # Non-matching profile rejects ...
    registered = _register(
        client, root_id, [_claim(chain["leaf_cert"], uri=OTHER_URI)]
    ).json()
    _, _, _, rejected = _verify_leaf(client, chain)
    assert rejected.json()["status"] == "rejected"

    # ... replacing it with a matching profile admits the next evidence.
    updated = _update(
        client, registered["profile_id"], root_id, [_claim(chain["leaf_cert"])]
    )
    assert updated.status_code == 200
    _, _, _, admitted = _verify_leaf(client, chain)
    assert admitted.json()["status"] == "verified"

    # Replacing it back to non-matching rejects again.
    _update(
        client,
        registered["profile_id"],
        root_id,
        [_claim(chain["leaf_cert"], uri=OTHER_URI)],
    )
    _, _, _, rejected_again = _verify_leaf(client, chain)
    assert rejected_again.json()["status"] == "rejected"


def test_settled_conclusions_are_not_changed_by_update_or_revoke(
    client, root_id, chain
):
    # Verified while no profile exists for the anchor.
    verified_id, verified_evidence, verified_challenge, verified = _verify_leaf(
        client, chain
    )
    assert verified.json()["status"] == "verified"

    # Registering, replacing and then revoking a (now non-matching)
    # profile must not rewrite the settled conclusion.
    matching = _register(client, root_id, [_claim(chain["leaf_cert"])]).json()
    _update(
        client,
        matching["profile_id"],
        root_id,
        [_claim(chain["leaf_cert"], uri=OTHER_URI)],
    )
    _revoke(client, matching["profile_id"], root_id)
    repeated = client.post(
        f"/v1/evidence/{verified_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": verified_challenge["nonce"],
            "evidence": verified_evidence,
        },
    )
    assert repeated.status_code == 200
    assert repeated.json() == verified.json()

    # A rejection settled while a strict profile is active stays rejected
    # after that profile is revoked. (A fresh URI is used: the revoked
    # profile above still owns its claim-set fingerprint, and registration
    # compatibility is unchanged.)
    strict = _register(
        client, root_id, [_claim(chain["leaf_cert"], uri=THIRD_URI)]
    ).json()
    rejected_id, rejected_evidence, rejected_challenge, rejected = _verify_leaf(
        client, chain
    )
    assert rejected.json()["status"] == "rejected"
    _revoke(client, strict["profile_id"], root_id)
    repeated_rejected = client.post(
        f"/v1/evidence/{rejected_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": rejected_challenge["nonce"],
            "evidence": rejected_evidence,
        },
    )
    assert repeated_rejected.status_code == 200
    assert repeated_rejected.json() == rejected.json()


def test_revoke_then_reverify_is_retryable_after_registry_recovery(
    app, client, root_id, chain
):
    # While the gate registry is unavailable, verification fails with 500
    # and leaves the evidence received, even after the profile was
    # revoked; recovery settles it normally (and observes the revocation).
    strict = _register(
        client, root_id, [_claim(chain["leaf_cert"], uri=OTHER_URI)]
    ).json()
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"],
        chain["leaf_key"],
        [chain["leaf_cert"], chain["intermediate_cert"], chain["root_cert"]],
        {"m": "abc"},
    )
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
    evidence_id = submitted.json()["evidence_id"]
    verify_body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
    }

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE workload_identity_profiles"))
    failed = client.post(f"/v1/evidence/{evidence_id}/verify", json=verify_body)
    assert failed.status_code == 500
    with app.state.session_factory() as session:
        from proof_release.db import Evidence

        assert session.get(Evidence, evidence_id).status == "received"

    # Recreating the table leaves the anchor without profiles, so after
    # recovery the same evidence settles on the ordinary verdict.
    WorkloadIdentityProfile.__table__.create(app.state.engine)
    recovered = client.post(
        f"/v1/evidence/{evidence_id}/verify", json=verify_body
    )
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "verified"
    # The old strict profile is gone with the dropped table; nothing to
    # revoke here, but the registry-failure path is exercised.
    del strict


# --- legacy database migration ----------------------------------------------


def test_legacy_database_is_migrated_to_lifecycle_columns(tmp_path, chain):
    from sqlalchemy import create_engine

    url = f"sqlite:///{tmp_path}/legacy.db"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE workload_identity_profiles ("
                "profile_id VARCHAR(36) NOT NULL PRIMARY KEY, "
                "tenant_id VARCHAR(256), workload_id VARCHAR(256), "
                "trust_root_id VARCHAR(36), claims_fingerprint VARCHAR(64), "
                "created_at DATETIME)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE workload_identity_claims ("
                "claim_id VARCHAR(36) NOT NULL PRIMARY KEY, "
                "profile_id VARCHAR(36), tenant_id VARCHAR(256), "
                "workload_id VARCHAR(256), trust_root_id VARCHAR(36), "
                "issuer VARCHAR(1024), subject VARCHAR(1024), "
                "uri VARCHAR(2048))"
            )
        )
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            text(
                "INSERT INTO workload_identity_profiles VALUES "
                "(:p, :t, :w, :r, :fp, :c)"
            ),
            {
                "p": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "t": TENANT,
                "w": WORKLOAD,
                "r": "12121212-1212-1212-1212-121212121212",
                "fp": "legacy-fp",
                "c": now,
            },
        )
        conn.execute(
            text(
                "INSERT INTO workload_identity_claims VALUES "
                "(:c, :p, :t, :w, :r, :i, :s, :u)"
            ),
            {
                "c": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "p": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "t": TENANT,
                "w": WORKLOAD,
                "r": "12121212-1212-1212-1212-121212121212",
                "i": "CN=legacy-issuer",
                "s": "CN=legacy-subject",
                "u": LEAF_URI,
            },
        )
    engine.dispose()

    application = create_app(url)
    with application.state.session_factory() as session:
        profile = session.get(
            WorkloadIdentityProfile, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        )
        assert profile.status == "active"
        assert profile.updated_at is None
        assert profile.revoked_at is None
        claim = session.scalars(select(WorkloadIdentityClaim)).one()
        assert claim.seq == 0
    application.state.engine.dispose()
