"""Tests for trust-root-scoped X.509 certificate revocation registration
and real-time rejection during evidence verification."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select

from proof_release.app import create_app
from proof_release.db import CertificateRevocation, Evidence, TrustRoot
from proof_release.envelopes import b64url_encode
from proof_release.verifiers import (
    ATTESTED_NONCE_JSON,
    X509_ATTESTED_NONCE_JSON,
    VerificationContext,
    VerifierRegistry,
    X509AttestedNonceJSONVerifier,
)

from cryptography.hazmat.primitives.serialization import Encoding

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


def fingerprint(certificate) -> str:
    return b64url_encode(
        hashlib.sha256(certificate.public_bytes(Encoding.DER)).digest()
    )


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
    leaf_key, leaf_cert = make_leaf(intermediate_cert, intermediate_key)
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


def _register(client, root_id_value, fp, effective_at, **scope):
    payload = {
        "tenant_id": scope.get("tenant_id", TENANT),
        "workload_id": scope.get("workload_id", WORKLOAD),
        "trust_root_id": root_id_value,
        "certificate_fingerprint": fp,
        "effective_at": effective_at,
    }
    return client.post("/v1/revocations", json=payload)


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


# --- registration -----------------------------------------------------------


def test_register_revocation_returns_201_compact_json(client, root_id, chain):
    fp = fingerprint(chain["leaf_cert"])
    effective = "2020-01-01T00:00:00+00:00"
    response = _register(client, root_id, fp, effective)

    assert response.status_code == 201
    body = response.json()
    assert set(body.keys()) == {
        "revocation_id",
        "trust_root_id",
        "certificate_fingerprint",
        "effective_at",
    }
    assert body["trust_root_id"] == root_id
    assert body["certificate_fingerprint"] == fp
    assert body["effective_at"] == "2020-01-01T00:00:00+00:00"
    # Compact JSON: no whitespace between tokens.
    assert b", " not in response.content
    assert b": " not in response.content


def test_registration_persists_across_restart(tmp_path, chain):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    created = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    root = created.json()["root_id"]
    response = _register(
        client1, root, fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z"
    )
    assert response.status_code == 201
    revocation_id = response.json()["revocation_id"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        rows = session.scalars(select(CertificateRevocation)).all()
    assert len(rows) == 1
    assert rows[0].revocation_id == revocation_id
    app2.state.engine.dispose()


def test_duplicate_registration_returns_409_and_keeps_original(
    client, root_id, chain
):
    fp = fingerprint(chain["leaf_cert"])
    first = _register(client, root_id, fp, "2020-01-01T00:00:00Z")
    assert first.status_code == 201
    first_body = first.json()

    # A repeat with a different effective time still conflicts and never
    # merges or overwrites the first registration.
    second = _register(client, root_id, fp, "2030-01-01T00:00:00Z")
    assert second.status_code == 409
    assert first_body["effective_at"] == "2020-01-01T00:00:00+00:00"

    with client.app.state.session_factory() as session:
        rows = session.scalars(select(CertificateRevocation)).all()
    assert len(rows) == 1
    assert rows[0].revocation_id == first_body["revocation_id"]
    assert rows[0].effective_at.isoformat() == "2020-01-01T00:00:00+00:00"


def test_distinct_fingerprints_are_registered_independently(client, root_id, chain):
    leaf = _register(client, root_id, fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z")
    intermediate = _register(
        client, root_id, fingerprint(chain["intermediate_cert"]), "2020-01-01T00:00:00Z"
    )
    root = _register(
        client, root_id, fingerprint(chain["root_cert"]), "2020-01-01T00:00:00Z"
    )
    assert leaf.status_code == 201
    assert intermediate.status_code == 201
    assert root.status_code == 201
    with client.app.state.session_factory() as session:
        rows = session.scalars(select(CertificateRevocation)).all()
    assert len(rows) == 3


def test_concurrent_duplicate_registrations_settle_at_most_once(
    app, root_id, chain
):
    fp = fingerprint(chain["leaf_cert"])

    def register():
        return TestClient(app).post(
            "/v1/revocations",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "trust_root_id": root_id,
                "certificate_fingerprint": fp,
                "effective_at": "2020-01-01T00:00:00Z",
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: register(), range(20)))

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(201) == 1
    assert statuses.count(409) == 19
    with app.state.session_factory() as session:
        rows = session.scalars(select(CertificateRevocation)).all()
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
        ("trust_root_id", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaZ"),
        ("certificate_fingerprint", ""),
        ("certificate_fingerprint", "abcd"),  # too short
        ("certificate_fingerprint", "A" * 44),  # padded/wrong length shape
        ("certificate_fingerprint", "?" * 43),  # bad alphabet
        ("effective_at", ""),
        ("effective_at", "2020-01-01T00:00:00"),  # naive
        ("effective_at", "2020-01-01T00:00:00+02:00"),  # non-UTC offset
        ("effective_at", "not a timestamp"),
    ],
)
def test_invalid_fields_return_422_without_writing_state(
    client, root_id, chain, field, value
):
    fp = fingerprint(chain["leaf_cert"])
    payload = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "certificate_fingerprint": fp,
        "effective_at": "2020-01-01T00:00:00Z",
    }
    payload[field] = value
    response = client.post("/v1/revocations", json=payload)
    assert response.status_code == 422
    with client.app.state.session_factory() as session:
        assert session.scalars(select(CertificateRevocation)).all() == []


def test_padded_fingerprint_is_422(client, root_id, chain):
    import base64

    raw = hashlib.sha256(chain["leaf_cert"].public_bytes(Encoding.DER)).digest()
    padded = base64.urlsafe_b64encode(raw).decode("ascii")
    response = _register(client, root_id, padded, "2020-01-01T00:00:00Z")
    assert response.status_code == 422


def test_wrong_types_return_422(client, root_id, chain):
    fp = fingerprint(chain["leaf_cert"])
    base = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "certificate_fingerprint": fp,
        "effective_at": "2020-01-01T00:00:00Z",
    }
    for field, value in [
        ("tenant_id", 123),
        ("workload_id", None),
        ("trust_root_id", ["x"]),
        ("certificate_fingerprint", {"x": 1}),
        ("effective_at", 42),
    ]:
        payload = dict(base)
        payload[field] = value
        response = client.post("/v1/revocations", json=payload)
        assert response.status_code == 422, field


# --- trust root isolation ---------------------------------------------------


def test_unknown_trust_root_returns_404(client, chain):
    response = _register(
        client,
        "11111111-1111-1111-1111-111111111111",
        fingerprint(chain["leaf_cert"]),
        "2020-01-01T00:00:00Z",
    )
    assert response.status_code == 404


def test_trust_root_of_other_tenant_returns_404(client, root_id, chain):
    response = _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        "2020-01-01T00:00:00Z",
        tenant_id=OTHER_TENANT,
    )
    assert response.status_code == 404


def test_trust_root_of_other_workload_returns_404(client, root_id, chain):
    response = _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        "2020-01-01T00:00:00Z",
        workload_id=OTHER_WORKLOAD,
    )
    assert response.status_code == 404


def test_revocation_under_one_tenant_does_not_affect_other(client, chain):
    # The same root certificate is a trust root for two distinct tenants;
    # a revocation registered for one must not reject the other's evidence.
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

    roots = {}
    with client.app.state.session_factory() as session:
        for tenant in (TENANT, OTHER_TENANT):
            roots[tenant] = session.scalar(
                select(TrustRoot.root_id).where(
                    TrustRoot.tenant_id == tenant,
                    TrustRoot.workload_id == WORKLOAD,
                )
            )

    registered = _register(
        client,
        roots[TENANT],
        fingerprint(chain["leaf_cert"]),
        "2020-01-01T00:00:00Z",
    )
    assert registered.status_code == 201

    # Tenant A's evidence is rejected.
    created_a = _challenge(client, tenant=TENANT)
    evidence_a = make_evidence(
        created_a["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, response_a = _submit_and_verify(
        client, created_a, evidence_a, tenant=TENANT
    )
    assert response_a.status_code == 200
    assert response_a.json()["status"] == "rejected"

    # Tenant B's evidence against the same certificate still verifies.
    created_b = _challenge(client, tenant=OTHER_TENANT)
    evidence_b = make_evidence(
        created_b["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, response_b = _submit_and_verify(
        client, created_b, evidence_b, tenant=OTHER_TENANT
    )
    assert response_b.status_code == 200
    assert response_b.json()["status"] == "verified"


# --- real-time rejection during verification --------------------------------


def test_revoked_leaf_rejects_at_verification(client, root_id, chain):
    _register(client, root_id, fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z")
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))

    evidence_id, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    with client.app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "rejected"
        assert record.verification_result == "rejected"


def test_revoked_intermediate_rejects_at_verification(client, root_id, chain):
    _register(
        client, root_id, fingerprint(chain["intermediate_cert"]), "2020-01-01T00:00:00Z"
    )
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))

    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_revoked_root_rejects_at_verification(client, root_id, chain):
    _register(client, root_id, fingerprint(chain["root_cert"]), "2020-01-01T00:00:00Z")
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["root_key"], [chain["root_cert"]]
    )

    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_future_effective_registration_does_not_reject_yet(client, root_id, chain):
    _register(client, root_id, fingerprint(chain["leaf_cert"]), "2099-01-01T00:00:00Z")
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))

    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_future_registration_takes_effect_for_fresh_verification(
    client, root_id, chain
):
    from datetime import datetime, timedelta, timezone

    soon = datetime.now(timezone.utc) + timedelta(seconds=2)
    _register(client, root_id, fingerprint(chain["leaf_cert"]), soon.isoformat())

    # Before it takes effect: verified.
    created_first = _challenge(client)
    evidence_first = make_evidence(
        created_first["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, first = _submit_and_verify(client, created_first, evidence_first)
    assert first.json()["status"] == "verified"

    # After the effective instant, a fresh evidence submission is rejected.
    threading.Event().wait(2.5)
    created_second = _challenge(client)
    evidence_second = make_evidence(
        created_second["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, second = _submit_and_verify(client, created_second, evidence_second)
    assert second.json()["status"] == "rejected"


def test_unrelated_fingerprint_does_not_reject(client, root_id, chain):
    other_root_key, other_root_cert = make_root("other")
    _register(
        client, root_id, fingerprint(other_root_cert), "2020-01-01T00:00:00Z"
    )
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))

    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_registration_after_settlement_is_not_retroactive(client, root_id, chain):
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))
    evidence_id, first = _submit_and_verify(client, created, evidence)
    assert first.json()["status"] == "verified"

    # Settled first; the later revocation must not rewrite the conclusion.
    registered = _register(
        client, root_id, fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z"
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
    assert repeated.status_code == 200
    assert repeated.json()["status"] == "verified"


def test_registration_committed_during_verification_does_not_change_winner(
    tmp_path, chain
):
    """A revocation that commits while verification is mid-flight cannot
    change that verification's settlement; after settlement it is not
    retroactively rewritten either."""

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
    created_root = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    root = created_root.json()["root_id"]

    challenge = client.post(
        "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    ).json()
    evidence = make_evidence(challenge["nonce"], chain["leaf_key"], _full_chain(chain))
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

    def register():
        return TestClient(application).post(
            "/v1/revocations",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "trust_root_id": root,
                "certificate_fingerprint": fingerprint(chain["leaf_cert"]),
                "effective_at": "2020-01-01T00:00:00Z",
            },
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        verify_future = pool.submit(verify)
        assert entered.wait(timeout=5)
        register_future = pool.submit(register)
        # Give the registration time to start and block on the writer lock.
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


# --- failure semantics ------------------------------------------------------


def test_registry_unavailable_returns_500_and_keeps_evidence_received(
    app, client, root_id, chain
):
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
    evidence_id = submitted.json()["evidence_id"]

    # Take the registry away while the evidence is still received.
    from sqlalchemy import text

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE certificate_revocations"))

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
    CertificateRevocation.__table__.create(app.state.engine)
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


def test_failed_registration_write_leaves_no_record(app, client, root_id, chain):
    engine = app.state.engine

    def fail_insert(conn, cursor, statement, parameters, context, executemany):
        if "INTO certificate_revocations" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_insert)
    try:
        response = _register(
            client,
            root_id,
            fingerprint(chain["leaf_cert"]),
            "2020-01-01T00:00:00Z",
        )
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_insert)

    with app.state.session_factory() as session:
        assert session.scalars(select(CertificateRevocation)).all() == []


def test_failed_registration_read_returns_500_and_leaves_no_record(
    app, client, root_id, chain
):
    engine = app.state.engine

    def fail_lookup(conn, cursor, statement, parameters, context, executemany):
        if "FROM trust_roots" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_lookup)
    try:
        response = _register(
            client,
            root_id,
            fingerprint(chain["leaf_cert"]),
            "2020-01-01T00:00:00Z",
        )
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_lookup)

    # Full rollback: no half record, and once storage recovers the same
    # registration succeeds exactly once.
    with app.state.session_factory() as session:
        assert session.scalars(select(CertificateRevocation)).all() == []
    recovered = _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        "2020-01-01T00:00:00Z",
    )
    assert recovered.status_code == 201


# --- scope of behavior: X.509 only ------------------------------------------


def test_revocation_does_not_change_hmac_format_verification(
    app, client, root_id, chain
):
    # A revocation exists for the X.509 leaf fingerprint; the HMAC evidence
    # format's verdict must be unaffected.
    _register(client, root_id, fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z")

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
    _register(client, root_id, fingerprint(chain["leaf_cert"]), "2020-01-01T00:00:00Z")
    created = _challenge(client)
    evidence = make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))
    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert "BEGIN CERTIFICATE" not in response.text
    assert pem(chain["leaf_cert"]) not in response.text
