"""Tests for workload identity profile registration and the X.509
workload-identity gate during evidence verification."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select

from proof_release.app import create_app
from proof_release.db import (
    Evidence,
    WorkloadIdentityClaim,
    WorkloadIdentityProfile,
)
from proof_release.verifiers import (
    ATTESTED_NONCE_JSON,
    X509_ATTESTED_NONCE_JSON,
    VerificationContext,
    VerifierRegistry,
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


def _challenge(client, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/challenges", json={"tenant_id": tenant, "workload_id": workload}
    )
    assert response.status_code == 201
    return response.json()


def _submit_and_verify(client, created, evidence, tenant=TENANT, workload=WORKLOAD):
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
    return evidence_id, response


def _full_chain(chain):
    return [chain["leaf_cert"], chain["intermediate_cert"], chain["root_cert"]]


def _leaf_evidence(created, chain):
    return make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain), {"m": "abc"}
    )


# --- registration -----------------------------------------------------------


def test_register_returns_201_compact_json(client, root_id, chain):
    claim = _claim(chain["leaf_cert"])
    response = _register(client, root_id, [claim])

    assert response.status_code == 201
    body = response.json()
    assert set(body.keys()) == {
        "profile_id",
        "tenant_id",
        "workload_id",
        "trust_root_id",
        "claims",
        "created_at",
    }
    assert body["tenant_id"] == TENANT
    assert body["workload_id"] == WORKLOAD
    assert body["trust_root_id"] == root_id
    assert body["claims"] == [claim]
    assert body["created_at"].endswith("+00:00")
    # Compact JSON: no insignificant whitespace.
    assert b", " not in response.content
    assert b": " not in response.content
    # No certificate or evidence material on the response.
    assert "BEGIN CERTIFICATE" not in response.text


def test_registration_persists_across_restart(tmp_path, chain):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    root = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    ).json()["root_id"]
    claim = _claim(chain["leaf_cert"])
    registered = _register(client1, root, [claim])
    assert registered.status_code == 201
    profile_id = registered.json()["profile_id"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        profiles = session.scalars(select(WorkloadIdentityProfile)).all()
        claims = session.scalars(select(WorkloadIdentityClaim)).all()
    assert len(profiles) == 1
    assert profiles[0].profile_id == profile_id
    assert len(claims) == 1
    assert (claims[0].issuer, claims[0].subject, claims[0].uri) == (
        claim["issuer"],
        claim["subject"],
        claim["uri"],
    )
    app2.state.engine.dispose()


def test_duplicate_claim_set_returns_409_and_keeps_original(client, root_id, chain):
    claim = _claim(chain["leaf_cert"])
    first = _register(client, root_id, [claim])
    assert first.status_code == 201
    first_body = first.json()

    second = _register(client, root_id, [dict(claim)])
    assert second.status_code == 409

    with client.app.state.session_factory() as session:
        profiles = session.scalars(select(WorkloadIdentityProfile)).all()
        claims = session.scalars(select(WorkloadIdentityClaim)).all()
    assert len(profiles) == 1
    assert profiles[0].profile_id == first_body["profile_id"]
    assert len(claims) == 1


def test_claim_set_order_and_duplicates_do_not_make_new_profile(
    client, root_id, chain
):
    claim_a = _claim(chain["leaf_cert"], uri=LEAF_URI)
    claim_b = _claim(chain["leaf_cert"], uri=OTHER_URI)
    # First profile carries two distinct claims (a duplicate entry is
    # submitted too and must collapse).
    first = _register(client, root_id, [claim_b, claim_a, claim_b])
    assert first.status_code == 201
    assert first.json()["claims"] == [claim_b, claim_a]

    # Same set in a different order is the identical profile.
    second = _register(client, root_id, [claim_a, claim_b])
    assert second.status_code == 409

    with client.app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentityProfile)).all().__len__() == 1
        assert session.scalars(select(WorkloadIdentityClaim)).all().__len__() == 2


def test_distinct_claim_sets_are_registered_independently(client, root_id, chain):
    first = _register(client, root_id, [_claim(chain["leaf_cert"], uri=LEAF_URI)])
    assert first.status_code == 201
    second = _register(client, root_id, [_claim(chain["leaf_cert"], uri=OTHER_URI)])
    assert second.status_code == 201
    assert second.json()["profile_id"] != first.json()["profile_id"]

    with client.app.state.session_factory() as session:
        profiles = session.scalars(select(WorkloadIdentityProfile)).all()
        claims = session.scalars(select(WorkloadIdentityClaim)).all()
    assert len(profiles) == 2
    assert len(claims) == 2
    # The first profile is untouched by the second registration.
    original = next(
        p for p in profiles if p.profile_id == first.json()["profile_id"]
    )
    original_claims = [
        c for c in claims if c.profile_id == original.profile_id
    ]
    assert len(original_claims) == 1
    assert original_claims[0].uri == LEAF_URI


def test_concurrent_identical_registrations_settle_at_most_once(
    app, root_id, chain
):
    claim = _claim(chain["leaf_cert"])

    def register():
        return TestClient(app).post(
            "/v1/workload-identities",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "trust_root_id": root_id,
                "claims": [claim],
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: register(), range(20)))

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(201) == 1
    assert statuses.count(409) == 19
    with app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentityProfile)).all().__len__() == 1
        assert session.scalars(select(WorkloadIdentityClaim)).all().__len__() == 1


def test_concurrent_distinct_claim_sets_all_succeed(app, root_id, chain):
    uris = [f"spiffe://example.org/workload/{i}" for i in range(10)]

    def register(uri):
        claim = _claim(chain["leaf_cert"], uri=uri)
        return TestClient(app).post(
            "/v1/workload-identities",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "trust_root_id": root_id,
                "claims": [claim],
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(register, uris))

    assert all(r.status_code == 201 for r in responses)
    with app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentityProfile)).all().__len__() == 10


# --- field validation -------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"workload_id": WORKLOAD, "trust_root_id": "0" * 36, "claims": []},
        {"tenant_id": TENANT, "trust_root_id": "0" * 36, "claims": []},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "claims": []},
        {"tenant_id": "  ", "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": [{"issuer": "a", "subject": "b", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": "\t",
         "trust_root_id": "0" * 36, "claims": [{"issuer": "a", "subject": "b", "uri": "c"}]},
        # Non-canonical trust root UUID.
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
         "claims": [{"issuer": "a", "subject": "b", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "not-a-uuid",
         "claims": [{"issuer": "a", "subject": "b", "uri": "c"}]},
        # Empty claim list.
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": []},
        # Claim missing one of the fields.
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": [{"subject": "b", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": [{"issuer": "a", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": [{"issuer": "a", "subject": "b"}]},
        # Blank claim fields.
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": [{"issuer": "   ", "subject": "b", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": [{"issuer": "a", "subject": "", "uri": "c"}]},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": [{"issuer": "a", "subject": "b", "uri": "  "}]},
        # Claims not a list / claim not an object.
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": {"issuer": "a"}},
        {"tenant_id": TENANT, "workload_id": WORKLOAD,
         "trust_root_id": "0" * 36, "claims": ["nope"]},
    ],
)
def test_invalid_requests_return_422_without_writing_state(
    client, root_id, payload
):
    response = client.post("/v1/workload-identities", json=payload)
    assert response.status_code == 422
    with client.app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentityProfile)).all() == []
        assert session.scalars(select(WorkloadIdentityClaim)).all() == []


def test_wrong_types_return_422(client, root_id, chain):
    claim = _claim(chain["leaf_cert"])
    base = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "claims": [claim],
    }
    for field, value in [
        ("tenant_id", 123),
        ("workload_id", None),
        ("trust_root_id", ["x"]),
        ("claims", "not-a-list"),
    ]:
        payload = dict(base)
        payload[field] = value
        response = client.post("/v1/workload-identities", json=payload)
        assert response.status_code == 422, field
    for field in ("issuer", "subject", "uri"):
        bad_claim = dict(claim)
        bad_claim[field] = 42
        payload = dict(base)
        payload["claims"] = [bad_claim]
        response = client.post("/v1/workload-identities", json=payload)
        assert response.status_code == 422, field


# --- trust root isolation ---------------------------------------------------


def test_unknown_trust_root_returns_404(client, chain):
    response = _register(
        client,
        "11111111-1111-1111-1111-111111111111",
        [_claim(chain["leaf_cert"])],
    )
    assert response.status_code == 404


def test_trust_root_of_other_tenant_returns_404(client, root_id, chain):
    response = _register(
        client, root_id, [_claim(chain["leaf_cert"])], tenant=OTHER_TENANT
    )
    assert response.status_code == 404


def test_trust_root_of_other_workload_returns_404(client, root_id, chain):
    response = _register(
        client, root_id, [_claim(chain["leaf_cert"])], workload=OTHER_WORKLOAD
    )
    assert response.status_code == 404


# --- storage failure semantics ----------------------------------------------


def test_failed_profile_write_leaves_no_profile(app, client, root_id, chain):
    engine = app.state.engine

    def fail_insert(conn, cursor, statement, parameters, context, executemany):
        if "INTO workload_identity_profiles" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_insert)
    try:
        response = _register(client, root_id, [_claim(chain["leaf_cert"])])
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_insert)

    with app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentityProfile)).all() == []
        assert session.scalars(select(WorkloadIdentityClaim)).all() == []


def test_failed_claim_write_leaves_no_half_profile(app, client, root_id, chain):
    engine = app.state.engine

    def fail_insert(conn, cursor, statement, parameters, context, executemany):
        if "INTO workload_identity_claims" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_insert)
    try:
        response = _register(client, root_id, [_claim(chain["leaf_cert"])])
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_insert)

    # The parent profile and the claim are one atomic write: a failed
    # claim insert rolls the profile back too.
    with app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentityProfile)).all() == []
        assert session.scalars(select(WorkloadIdentityClaim)).all() == []

    # Once storage recovers the same registration succeeds exactly once.
    recovered = _register(client, root_id, [_claim(chain["leaf_cert"])])
    assert recovered.status_code == 201


# --- verification gating ----------------------------------------------------


def test_no_profile_keeps_current_verification_result(client, root_id, chain):
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_matching_profile_allows_verification(client, root_id, chain):
    registered = _register(client, root_id, [_claim(chain["leaf_cert"])])
    assert registered.status_code == 201
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_profile_with_no_matching_claim_rejects(client, root_id, chain):
    # The profile names a different URI than the leaf carries.
    _register(client, root_id, [_claim(chain["leaf_cert"], uri=OTHER_URI)])
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    evidence_id, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    with client.app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "rejected"
        assert record.verification_result == "rejected"


def test_issuer_must_match(client, root_id, chain):
    bad = dict(_claim(chain["leaf_cert"]))
    bad["issuer"] = "CN=some-other-issuer"
    _register(client, root_id, [bad])
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    _, response = _submit_and_verify(client, created, evidence)
    assert response.json()["status"] == "rejected"


def test_subject_must_match(client, root_id, chain):
    bad = dict(_claim(chain["leaf_cert"]))
    bad["subject"] = "CN=some-other-leaf"
    _register(client, root_id, [bad])
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    _, response = _submit_and_verify(client, created, evidence)
    assert response.json()["status"] == "rejected"


def test_any_claim_hit_is_sufficient(client, root_id, chain):
    claims = [
        _claim(chain["leaf_cert"], uri=OTHER_URI),  # does not hit
        _claim(chain["leaf_cert"], uri=LEAF_URI),   # hits
    ]
    _register(client, root_id, claims)
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    _, response = _submit_and_verify(client, created, evidence)
    assert response.json()["status"] == "verified"


def test_any_profile_hit_is_sufficient(client, root_id, chain):
    # First profile cannot match; second can.
    _register(client, root_id, [_claim(chain["leaf_cert"], uri=OTHER_URI)])
    _register(client, root_id, [_claim(chain["leaf_cert"], uri=LEAF_URI)])
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    _, response = _submit_and_verify(client, created, evidence)
    assert response.json()["status"] == "verified"


def test_leaf_without_uri_san_does_not_match_required_uri(client, root_id, chain):
    leaf_key, leaf_cert = make_leaf(
        chain["intermediate_cert"], chain["intermediate_key"]
    )
    # A profile exists (anchored root has profiles) and demands a URI the
    # SAN-less leaf cannot present.
    _register(
        client,
        root_id,
        [
            {
                "issuer": leaf_cert.issuer.rfc4514_string(),
                "subject": leaf_cert.subject.rfc4514_string(),
                "uri": LEAF_URI,
            }
        ],
    )
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"],
        leaf_key,
        [leaf_cert, chain["intermediate_cert"], chain["root_cert"]],
    )
    _, response = _submit_and_verify(client, created, evidence)
    assert response.json()["status"] == "rejected"


def test_single_certificate_chain_against_profiles_is_rejected(client, root_id, chain):
    # The leaf is the root itself; no SAN URI exists, so an anchored
    # profile that requires one cannot match.
    _register(
        client,
        root_id,
        [
            {
                "issuer": chain["root_cert"].issuer.rfc4514_string(),
                "subject": chain["root_cert"].subject.rfc4514_string(),
                "uri": LEAF_URI,
            }
        ],
    )
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["root_key"], [chain["root_cert"]]
    )
    _, response = _submit_and_verify(client, created, evidence)
    assert response.json()["status"] == "rejected"


def test_identity_rejection_is_settled_and_repeated(client, root_id, chain):
    _register(client, root_id, [_claim(chain["leaf_cert"], uri=OTHER_URI)])
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    evidence_id, first = _submit_and_verify(client, created, evidence)
    assert first.json()["status"] == "rejected"

    second = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert second.status_code == 200
    assert second.json() == first.json()


def test_profile_registered_after_settlement_is_not_retroactive(
    client, root_id, chain
):
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    evidence_id, first = _submit_and_verify(client, created, evidence)
    assert first.json()["status"] == "verified"

    # A non-matching profile registered after settlement must not rewrite
    # the already-settled conclusion.
    registered = _register(
        client, root_id, [_claim(chain["leaf_cert"], uri=OTHER_URI)]
    )
    assert registered.status_code == 201

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


def test_registration_committed_after_verification_does_not_change_winner(
    tmp_path, chain
):
    """A profile registration that commits while a verification is
    mid-flight cannot affect that verification (it observes only
    committed profiles), and the settled result is never rewritten."""

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

    def register(uri):
        claim = _claim(chain["leaf_cert"], uri=uri)
        return TestClient(application).post(
            "/v1/workload-identities",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "trust_root_id": root,
                "claims": [claim],
            },
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        verify_future = pool.submit(verify)
        assert entered.wait(timeout=5)
        # The non-matching registration blocks on the writer lock held by
        # verification; it can only commit once verification settles.
        register_future = pool.submit(register, OTHER_URI)
        threading.Event().wait(0.5)
        release.set()
        verify_response = verify_future.result(timeout=15)
        register_response = register_future.result(timeout=15)

    assert verify_response.status_code == 200
    assert verify_response.json()["status"] == "verified"
    assert register_response.status_code == 201

    repeated = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": challenge["nonce"],
            "evidence": evidence,
        },
    )
    assert repeated.json()["status"] == "verified"
    application.state.engine.dispose()


# --- gate failure semantics -------------------------------------------------


def test_profile_query_unavailable_returns_500_and_keeps_received(
    app, client, root_id, chain
):
    _register(client, root_id, [_claim(chain["leaf_cert"], uri=OTHER_URI)])
    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
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

    from sqlalchemy import text

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE workload_identity_claims"))

    verify_body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
    }
    failed = client.post(f"/v1/evidence/{evidence_id}/verify", json=verify_body)
    assert failed.status_code == 500
    assert "BEGIN CERTIFICATE" not in failed.text
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"

    # After recovery (table recreated and profile re-registered) the same
    # evidence settles normally and is retryable.
    WorkloadIdentityClaim.__table__.create(app.state.engine)
    recovered_registration = _register(
        client, root_id, [_claim(chain["leaf_cert"])]
    )
    assert recovered_registration.status_code == 201
    recovered = client.post(
        f"/v1/evidence/{evidence_id}/verify", json=verify_body
    )
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "verified"


# --- scope of behavior ------------------------------------------------------


def test_profiles_under_one_scope_do_not_affect_other(client, chain):
    # The same root certificate is a trust root for two distinct tenants.
    roots = {}
    for tenant in (TENANT, OTHER_TENANT):
        response = client.post(
            "/v1/trust-roots",
            json={
                "tenant_id": tenant,
                "workload_id": WORKLOAD,
                "root_pem": pem(chain["root_cert"]),
            },
        )
        assert response.status_code == 201
        roots[tenant] = response.json()["root_id"]

    # A strict profile exists only under tenant A.
    registered = _register(
        client, roots[TENANT], [_claim(chain["leaf_cert"], uri=OTHER_URI)]
    )
    assert registered.status_code == 201

    # Tenant B has no profiles: its evidence against the same chain keeps
    # the ordinary verdict (verified).
    created_b = _challenge(client, tenant=OTHER_TENANT)
    evidence_b = make_evidence(
        created_b["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, response_b = _submit_and_verify(
        client, created_b, evidence_b, tenant=OTHER_TENANT
    )
    assert response_b.json()["status"] == "verified"

    # Tenant A has the non-matching profile: rejected.
    created_a = _challenge(client)
    evidence_a = _leaf_evidence(created_a, chain)
    _, response_a = _submit_and_verify(client, created_a, evidence_a)
    assert response_a.json()["status"] == "rejected"


def test_identity_gate_does_not_change_hmac_format(app, client, root_id, chain):
    # A non-matching X.509 profile exists; HMAC evidence must be unaffected.
    _register(client, root_id, [_claim(chain["leaf_cert"], uri=OTHER_URI)])

    import hashlib
    import hmac
    import json
    import os

    secret = os.environ.get(
        "PROOF_RELEASE_ATTESTED_NONCE_SECRET", "dev-only-attested-nonce-secret"
    ).encode("utf-8")
    mac_key = hmac.new(
        secret, f"{TENANT}:{WORKLOAD}".encode("utf-8"), hashlib.sha256
    ).digest()
    created = _challenge(client)
    nonce = created["nonce"]
    claims = {}
    signed = json.dumps(
        {"claims": claims, "nonce": nonce}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    mac = hmac.new(mac_key, signed, hashlib.sha256).hexdigest()
    evidence = json.dumps(
        {"nonce": nonce, "claims": claims, "mac": mac},
        sort_keys=True,
        separators=(",", ":"),
    )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": nonce,
            "evidence_format": ATTESTED_NONCE_JSON,
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
            "nonce": nonce,
            "evidence": evidence,
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_gate_does_not_rescue_a_bad_signature(client, root_id, chain):
    # Even when a matching profile exists, a chain/signature failure still
    # rejects — the gate only narrows an otherwise-accepting verdict.
    other_root_key, other_root_cert = make_root("other-root")
    leaf_key, leaf_cert = make_leaf(other_root_cert, other_root_key)
    _register(
        client,
        root_id,
        [
            {
                "issuer": leaf_cert.issuer.rfc4514_string(),
                "subject": leaf_cert.subject.rfc4514_string(),
                "uri": LEAF_URI,
            }
        ],
    )
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], leaf_key, [leaf_cert, other_root_cert]
    )
    _, response = _submit_and_verify(client, created, evidence)
    assert response.json()["status"] == "rejected"


def test_certificate_and_evidence_never_persisted_or_returned(
    app, client, root_id, chain
):
    claim = _claim(chain["leaf_cert"])
    registered = _register(client, root_id, [claim])
    assert pem(chain["leaf_cert"]) not in registered.text

    created = _challenge(client)
    evidence = _leaf_evidence(created, chain)
    evidence_id, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert evidence not in response.text
    assert "BEGIN CERTIFICATE" not in response.text
    with app.state.session_factory() as session:
        for model in (WorkloadIdentityProfile, WorkloadIdentityClaim):
            for row in session.scalars(select(model)).all():
                dumped = str(
                    {c.name: getattr(row, c.name) for c in row.__table__.columns}
                )
                assert "BEGIN CERTIFICATE" not in dumped
                assert evidence not in dumped
        record = session.get(Evidence, evidence_id)
        columns = str(
            {c.name: getattr(record, c.name) for c in record.__table__.columns}
        )
        assert "BEGIN CERTIFICATE" not in columns
