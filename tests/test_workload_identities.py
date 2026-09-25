"""Tests for trust-root-scoped workload identity profiles: registration
and leaf-identity gating during X.509 evidence verification."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select

from proof_release.app import create_app
from proof_release.db import Evidence, TrustRoot, WorkloadIdentity
from proof_release.verifiers import (
    ATTESTED_NONCE_JSON,
    X509_ATTESTED_NONCE_JSON,
)

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.x509.oid import NameOID

from x509_helpers import (
    generate_key,
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

LEAF_CN = "test-leaf"
INTERMEDIATE_CN = "test-intermediate"
LEAF_SUBJECT = f"CN={LEAF_CN}"
LEAF_ISSUER = f"CN={INTERMEDIATE_CN}"
LEAF_URI = "spiffe://example.test/workload-1"
OTHER_URI = "spiffe://example.test/other"


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
        intermediate_cert, intermediate_key, common_name=LEAF_CN
    )
    return {
        "root_key": root_key,
        "root_cert": root_cert,
        "intermediate_key": intermediate_key,
        "intermediate_cert": intermediate_cert,
        "leaf_key": leaf_key,
        "leaf_cert": leaf_cert,
    }


def make_leaf_with_uris(issuer_cert, issuer_key, common_name, uris):
    key = generate_key()
    now = datetime.now(timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
        )
        .issuer_name(issuer_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(hours=1))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.UniformResourceIdentifier(uri) for uri in uris]
            ),
            critical=False,
        )
    )
    return key, builder.sign(issuer_key, hashes.SHA256())


@pytest.fixture()
def uri_chain(chain):
    leaf_key, leaf_cert = make_leaf_with_uris(
        chain["intermediate_cert"],
        chain["intermediate_key"],
        LEAF_CN,
        [LEAF_URI, "spiffe://example.test/secondary"],
    )
    return {**chain, "leaf_key": leaf_key, "leaf_cert": leaf_cert}


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


def _register(client, root_id_value, claims, **scope):
    payload = {
        "tenant_id": scope.get("tenant_id", TENANT),
        "workload_id": scope.get("workload_id", WORKLOAD),
        "trust_root_id": root_id_value,
        "claims": claims,
    }
    return client.post("/v1/workload-identities", json=payload)


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


def _verify_fresh(client, chain):
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))
    return _submit_and_verify(client, created, evidence)


# --- registration -----------------------------------------------------------


def test_register_identity_returns_201_with_profile_fields(client, root_id):
    claims = [
        {
            "issuer": LEAF_ISSUER,
            "subject": LEAF_SUBJECT,
            "uri": LEAF_URI,
        }
    ]
    response = _register(client, root_id, claims)

    assert response.status_code == 201
    body = response.json()
    assert set(body.keys()) == {
        "identity_id",
        "tenant_id",
        "workload_id",
        "trust_root_id",
        "claims",
        "created_at",
    }
    assert body["tenant_id"] == TENANT
    assert body["workload_id"] == WORKLOAD
    assert body["trust_root_id"] == root_id
    assert body["claims"] == claims
    parsed = datetime.fromisoformat(body["created_at"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


def test_registration_persists_across_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    root_key, root_cert = make_root()
    created = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(root_cert),
        },
    )
    root = created.json()["root_id"]
    claims = [{"subject": LEAF_SUBJECT}]
    response = _register(client1, root, claims)
    assert response.status_code == 201
    identity_id = response.json()["identity_id"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        rows = session.scalars(select(WorkloadIdentity)).all()
    assert len(rows) == 1
    assert rows[0].identity_id == identity_id
    assert rows[0].trust_root_id == root
    assert json.loads(rows[0].claims_json) == claims
    app2.state.engine.dispose()


def test_duplicate_claim_set_returns_409_and_keeps_original(client, root_id):
    claims = [{"issuer": LEAF_ISSUER}, {"subject": LEAF_SUBJECT}]
    first = _register(client, root_id, claims)
    assert first.status_code == 201
    first_id = first.json()["identity_id"]

    # The identical set, and the same set in a different order, both
    # conflict; the first profile is never merged or overwritten.
    assert _register(client, root_id, claims).status_code == 409
    assert _register(client, root_id, list(reversed(claims))).status_code == 409

    with client.app.state.session_factory() as session:
        rows = session.scalars(select(WorkloadIdentity)).all()
    assert len(rows) == 1
    assert rows[0].identity_id == first_id


def test_distinct_claim_sets_are_registered_independently(client, root_id):
    first = _register(client, root_id, [{"subject": LEAF_SUBJECT}])
    second = _register(client, root_id, [{"subject": "CN=other-leaf"}])
    third = _register(client, root_id, [{"issuer": LEAF_ISSUER, "uri": LEAF_URI}])
    assert first.status_code == 201
    assert second.status_code == 201
    assert third.status_code == 201
    assert len({first.json()["identity_id"], second.json()["identity_id"],
               third.json()["identity_id"]}) == 3
    with client.app.state.session_factory() as session:
        rows = session.scalars(select(WorkloadIdentity)).all()
    assert len(rows) == 3


def test_same_claim_set_under_different_trust_roots_is_independent(
    client, chain, root_id
):
    other_root_key, other_root_cert = make_root("other-root")
    created = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(other_root_cert),
        },
    )
    assert created.status_code == 201
    other_root_id = created.json()["root_id"]

    claims = [{"subject": LEAF_SUBJECT}]
    assert _register(client, root_id, claims).status_code == 201
    assert _register(client, other_root_id, claims).status_code == 201


def test_concurrent_duplicate_registrations_settle_at_most_once(app, root_id):
    claims = [{"issuer": LEAF_ISSUER, "subject": LEAF_SUBJECT}]

    def register():
        return TestClient(app).post(
            "/v1/workload-identities",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "trust_root_id": root_id,
                "claims": claims,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: register(), range(20)))

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(201) == 1
    assert statuses.count(409) == 19
    with app.state.session_factory() as session:
        rows = session.scalars(select(WorkloadIdentity)).all()
    assert len(rows) == 1


# --- field validation -------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_id", ""),
        ("tenant_id", "   "),
        ("workload_id", ""),
        ("workload_id", "\t"),
        ("trust_root_id", ""),
        ("trust_root_id", "not-a-uuid"),
        ("trust_root_id", "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"),
        ("trust_root_id", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaZ"),
        ("claims", []),
        ("claims", "not-a-list"),
        ("claims", {}),
        ("claims", None),
        ("claims", [{}]),
        ("claims", [{"issuer": ""}]),
        ("claims", [{"issuer": "   "}]),
        ("claims", [{"subject": 123}]),
        ("claims", [{"uri": None}]),
        ("claims", [{"fingerprint": "x"}]),
        ("claims", [{"issuer": LEAF_ISSUER, "extra": "x"}]),
        ("claims", ["not-an-object"]),
        ("claims", [None]),
    ],
)
def test_invalid_fields_return_422_without_writing_state(
    client, root_id, field, value
):
    payload = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "claims": [{"subject": LEAF_SUBJECT}],
    }
    payload[field] = value
    response = client.post("/v1/workload-identities", json=payload)
    assert response.status_code == 422
    with client.app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentity)).all() == []


def test_missing_fields_return_422(client, root_id):
    base = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "claims": [{"subject": LEAF_SUBJECT}],
    }
    for field in ("tenant_id", "workload_id", "trust_root_id", "claims"):
        payload = dict(base)
        del payload[field]
        response = client.post("/v1/workload-identities", json=payload)
        assert response.status_code == 422, field


def test_wrong_types_return_422(client, root_id):
    base = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "claims": [{"subject": LEAF_SUBJECT}],
    }
    for field, value in [
        ("tenant_id", 123),
        ("workload_id", None),
        ("trust_root_id", ["x"]),
        ("claims", 42),
    ]:
        payload = dict(base)
        payload[field] = value
        response = client.post("/v1/workload-identities", json=payload)
        assert response.status_code == 422, field


# --- trust root isolation ---------------------------------------------------


def test_unknown_trust_root_returns_404(client):
    response = _register(
        client,
        "11111111-1111-1111-1111-111111111111",
        [{"subject": LEAF_SUBJECT}],
    )
    assert response.status_code == 404


def test_trust_root_of_other_tenant_returns_404(client, root_id):
    response = _register(
        client, root_id, [{"subject": LEAF_SUBJECT}], tenant_id=OTHER_TENANT
    )
    assert response.status_code == 404


def test_trust_root_of_other_workload_returns_404(client, root_id):
    response = _register(
        client, root_id, [{"subject": LEAF_SUBJECT}], workload_id=OTHER_WORKLOAD
    )
    assert response.status_code == 404


# --- failure semantics ------------------------------------------------------


def test_failed_registration_write_leaves_no_record(app, client, root_id):
    engine = app.state.engine

    def fail_insert(conn, cursor, statement, parameters, context, executemany):
        if "INTO workload_identities" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_insert)
    try:
        response = _register(client, root_id, [{"subject": LEAF_SUBJECT}])
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_insert)

    with app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentity)).all() == []
    # Once storage recovers the same registration succeeds exactly once.
    recovered = _register(client, root_id, [{"subject": LEAF_SUBJECT}])
    assert recovered.status_code == 201


def test_failed_registration_read_returns_500_and_leaves_no_record(
    app, client, root_id
):
    engine = app.state.engine

    def fail_lookup(conn, cursor, statement, parameters, context, executemany):
        if "FROM trust_roots" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_lookup)
    try:
        response = _register(client, root_id, [{"subject": LEAF_SUBJECT}])
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_lookup)

    with app.state.session_factory() as session:
        assert session.scalars(select(WorkloadIdentity)).all() == []


# --- verification gating ----------------------------------------------------


def test_matching_profile_allows_verification(client, root_id, uri_chain):
    registered = _register(
        client,
        root_id,
        [{"issuer": LEAF_ISSUER, "subject": LEAF_SUBJECT, "uri": LEAF_URI}],
    )
    assert registered.status_code == 201

    evidence_id, response = _verify_fresh(client, uri_chain)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"
    with client.app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "verified"
        assert record.verification_result == "accepted"


def test_single_field_claim_matches(client, root_id, chain):
    _register(client, root_id, [{"subject": LEAF_SUBJECT}])
    _, response = _verify_fresh(client, chain)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_any_claim_hit_suffices(client, root_id, chain):
    _register(
        client,
        root_id,
        [{"subject": "CN=somebody-else"}, {"issuer": LEAF_ISSUER}],
    )
    _, response = _verify_fresh(client, chain)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_any_profile_hit_suffices(client, root_id, chain):
    _register(client, root_id, [{"subject": "CN=somebody-else"}])
    _register(client, root_id, [{"issuer": LEAF_ISSUER}])
    _, response = _verify_fresh(client, chain)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_uri_claim_matches_one_of_several_leaf_uris(client, root_id, uri_chain):
    _register(client, root_id, [{"uri": LEAF_URI}])
    _, response = _verify_fresh(client, uri_chain)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_profiles_exist_but_none_match_settles_rejected(client, root_id, chain):
    _register(client, root_id, [{"subject": "CN=somebody-else"}])
    _register(client, root_id, [{"issuer": "CN=another-issuer", "uri": OTHER_URI}])

    evidence_id, response = _verify_fresh(client, chain)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    with client.app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "rejected"
        assert record.verification_result == "rejected"


def test_partial_claim_mismatch_rejects(client, root_id, chain):
    # Issuer matches but the claimed subject does not: the claim fails.
    _register(
        client, root_id, [{"issuer": LEAF_ISSUER, "subject": "CN=other"}]
    )
    _, response = _verify_fresh(client, chain)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_uri_claim_without_matching_san_rejects(client, root_id, uri_chain):
    _register(client, root_id, [{"uri": OTHER_URI}])
    _, response = _verify_fresh(client, uri_chain)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_no_profiles_keeps_existing_verification_result(client, root_id, chain):
    _, response = _verify_fresh(client, chain)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_profile_under_other_trust_root_does_not_apply(client, chain, root_id):
    # A second trust root in the same scope carries a non-matching profile;
    # evidence anchored to the first root is unaffected by it.
    other_root_key, other_root_cert = make_root("other-root")
    created = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(other_root_cert),
        },
    )
    other_root_id = created.json()["root_id"]
    _register(client, other_root_id, [{"subject": "CN=somebody-else"}])

    _, response = _verify_fresh(client, chain)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_profile_of_other_tenant_does_not_apply(client, chain, root_id):
    # The same root certificate is a trust root for another tenant, with a
    # non-matching profile; this tenant's evidence is unaffected.
    created = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": OTHER_TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    assert created.status_code == 201
    other_root_id = created.json()["root_id"]
    registered = _register(
        client,
        other_root_id,
        [{"subject": "CN=somebody-else"}],
        tenant_id=OTHER_TENANT,
    )
    assert registered.status_code == 201

    _, response = _verify_fresh(client, chain)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_registration_after_settlement_is_not_retroactive(client, root_id, chain):
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))
    evidence_id, first = _submit_and_verify(client, created, evidence)
    assert first.json()["status"] == "verified"

    # Settled first; the later non-matching profile must not rewrite the
    # conclusion.
    registered = _register(client, root_id, [{"subject": "CN=somebody-else"}])
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
    assert repeated.status_code == 200
    assert repeated.json()["status"] == "verified"


def test_registry_unavailable_returns_500_and_keeps_evidence_received(
    app, client, root_id, chain
):
    _register(client, root_id, [{"subject": LEAF_SUBJECT}])
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))
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

    # Take the registry away while the evidence is still received.
    from sqlalchemy import text

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE workload_identities"))

    response = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert response.status_code == 500
    assert "BEGIN CERTIFICATE" not in response.text
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"

    # Once the registry is back, the same evidence verifies normally.
    WorkloadIdentity.__table__.create(app.state.engine)
    recovered = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "verified"


def test_identity_profile_does_not_change_hmac_format_verification(
    client, root_id, chain
):
    # A non-matching profile exists under the trust root; the HMAC
    # evidence format's verdict must be unaffected.
    _register(client, root_id, [{"subject": "CN=somebody-else"}])

    import hmac
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


def test_rejection_response_never_contains_certificate_material(
    client, root_id, chain
):
    _register(client, root_id, [{"subject": "CN=somebody-else"}])
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))
    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert "BEGIN CERTIFICATE" not in response.text
    assert pem(chain["leaf_cert"]) not in response.text
