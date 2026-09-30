"""Tests for CRL-based revocation: POST /v1/crls registration and the
current-snapshot rejection check during X.509 evidence verification."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select, text

from cryptography.hazmat.primitives.serialization import Encoding

from proof_release.app import create_app
from proof_release.db import (
    CRLRevokedEntry,
    CRLSnapshot,
    CertificateRevocation,
    Evidence,
    TrustRoot,
)
from proof_release.envelopes import b64url_encode
from proof_release.verifiers import (
    ATTESTED_NONCE_JSON,
    X509_ATTESTED_NONCE_JSON,
)

from x509_helpers import (
    build_crl,
    build_crl_without_crl_number,
    make_evidence,
    make_intermediate,
    make_leaf,
    make_root,
    pem,
    raw_crl,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"


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
    # A leaf issued directly by the root: a root-signed CRL entry naming
    # its serial has the chain cert's issuer DN as the CRL issuer.
    direct_key, direct_cert = make_leaf(root_cert, root_key, "direct-leaf")
    return {
        "root_key": root_key,
        "root_cert": root_cert,
        "intermediate_key": intermediate_key,
        "intermediate_cert": intermediate_cert,
        "leaf_key": leaf_key,
        "leaf_cert": leaf_cert,
        "direct_key": direct_key,
        "direct_cert": direct_cert,
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


def _register_crl(client, root_id_value, crl_pem, **scope):
    payload = {
        "tenant_id": scope.get("tenant_id", TENANT),
        "workload_id": scope.get("workload_id", WORKLOAD),
        "trust_root_id": root_id_value,
        "crl_pem": crl_pem,
    }
    return client.post("/v1/crls", json=payload)


def _register_fingerprint(client, root_id_value, certificate, effective_at):
    fp = b64url_encode(
        hashlib.sha256(certificate.public_bytes(Encoding.DER)).digest()
    )
    return client.post(
        "/v1/revocations",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id_value,
            "certificate_fingerprint": fp,
            "effective_at": effective_at,
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


def _past(hours=1):
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def _future(days=1):
    return datetime.now(timezone.utc) + timedelta(days=days)


# --- registration success ---------------------------------------------------


def test_register_crl_returns_201_compact_json(client, root_id, chain):
    crl_pem = build_crl(
        chain["root_cert"], chain["root_key"], 1, [(1234, _past())]
    )
    response = _register_crl(client, root_id, crl_pem)

    assert response.status_code == 201
    body = response.json()
    assert set(body.keys()) == {
        "crl_id",
        "trust_root_id",
        "crl_number",
        "this_update",
        "next_update",
        "revoked_count",
    }
    assert body["trust_root_id"] == root_id
    assert body["crl_number"] == 1
    assert body["revoked_count"] == 1
    # Compact JSON: no whitespace between tokens, no material echoed.
    assert b", " not in response.content
    assert b": " not in response.content
    assert b"BEGIN X509 CRL" not in response.content
    assert b"BEGIN CERTIFICATE" not in response.content
    # RFC3339 UTC timestamps.
    assert body["this_update"].endswith("+00:00")
    assert body["next_update"].endswith("+00:00")


def test_revoked_count_only_counts_arrived_revocation_dates(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = build_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [
            (100, now - timedelta(days=2)),  # arrived
            (101, now - timedelta(seconds=1)),  # arrived
            (102, now + timedelta(days=2)),  # still in the future
        ],
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 201
    assert response.json()["revoked_count"] == 2
    # The future entry is still part of the enrolled snapshot.
    with client.app.state.session_factory() as session:
        assert session.scalars(select(CRLRevokedEntry)).all().__len__() == 3


def test_empty_crl_enrolls_with_zero_count(client, root_id, chain):
    response = _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 1, []),
    )
    assert response.status_code == 201
    assert response.json()["revoked_count"] == 0


def test_crl_registration_persists_across_restart(tmp_path, chain):
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
    response = _register_crl(
        client1,
        root,
        build_crl(chain["root_cert"], chain["root_key"], 1, [(7, _past())]),
    )
    assert response.status_code == 201
    crl_id = response.json()["crl_id"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        snapshots = session.scalars(select(CRLSnapshot)).all()
        entries = session.scalars(select(CRLRevokedEntry)).all()
    assert len(snapshots) == 1
    assert snapshots[0].crl_id == crl_id
    assert len(entries) == 1
    app2.state.engine.dispose()


def test_snapshot_and_entries_are_atomic(app, client, root_id, chain):
    engine = app.state.engine

    def fail_entry_insert(conn, cursor, statement, parameters, context, executemany):
        if "INTO crl_revoked_entries" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_entry_insert)
    try:
        response = _register_crl(
            client,
            root_id,
            build_crl(chain["root_cert"], chain["root_key"], 1, [(1, _past())]),
        )
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_entry_insert)

    with app.state.session_factory() as session:
        assert session.scalars(select(CRLSnapshot)).all() == []
        assert session.scalars(select(CRLRevokedEntry)).all() == []


# --- CRLNumber/content conflicts -------------------------------------------


def test_same_crl_number_returns_409_and_keeps_original(client, root_id, chain):
    now = datetime.now(timezone.utc)
    first_pem = build_crl(
        chain["root_cert"], chain["root_key"], 5, [(1, now - timedelta(days=1))]
    )
    first = _register_crl(client, root_id, first_pem)
    assert first.status_code == 201
    first_body = first.json()

    # A distinct CRL that nevertheless reuses CRLNumber 5.
    second_pem = build_crl(
        chain["root_cert"],
        chain["root_key"],
        5,
        [(1, now - timedelta(days=1)), (2, now - timedelta(days=1))],
    )
    second = _register_crl(client, root_id, second_pem)
    assert second.status_code == 409
    with client.app.state.session_factory() as session:
        rows = session.scalars(select(CRLSnapshot)).all()
    assert len(rows) == 1
    assert rows[0].crl_id == first_body["crl_id"]
    assert rows[0].revoked_count == 1


def test_identical_content_resubmitted_returns_409(client, root_id, chain):
    crl_pem = build_crl(chain["root_cert"], chain["root_key"], 1, [(1, _past())])
    assert _register_crl(client, root_id, crl_pem).status_code == 201
    repeat = _register_crl(client, root_id, crl_pem)
    assert repeat.status_code == 409
    with client.app.state.session_factory() as session:
        assert len(session.scalars(select(CRLSnapshot)).all()) == 1


def test_lower_crl_number_returns_409(client, root_id, chain):
    assert (
        _register_crl(
            client,
            root_id,
            build_crl(chain["root_cert"], chain["root_key"], 10, []),
        ).status_code
        == 201
    )
    lower = _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 9, []),
    )
    assert lower.status_code == 409


def test_higher_crl_number_replaces_current_snapshot(client, root_id, chain):
    first = _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 1, [(1, _past())]),
    )
    assert first.status_code == 201
    second = _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 2, [(1, _past())]),
    )
    assert second.status_code == 201
    assert second.json()["crl_number"] == 2
    # Both snapshots are retained immutably; the highest is current.
    with client.app.state.session_factory() as session:
        numbers = sorted(s.crl_number for s in session.scalars(select(CRLSnapshot)).all())
    assert numbers == [1, 2]


def test_concurrent_same_number_registrations_settle_at_most_once(app, root_id, chain):
    crl_pem = build_crl(chain["root_cert"], chain["root_key"], 1, [(1, _past())])

    def register():
        return TestClient(app).post(
            "/v1/crls",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "trust_root_id": root_id,
                "crl_pem": crl_pem,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: register(), range(20)))

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(201) == 1
    assert statuses.count(409) == 19
    with app.state.session_factory() as session:
        assert len(session.scalars(select(CRLSnapshot)).all()) == 1


# --- content validation (all 422, no state) ---------------------------------


def _assert_no_crl_state(client):
    with client.app.state.session_factory() as session:
        assert session.scalars(select(CRLSnapshot)).all() == []
        assert session.scalars(select(CRLRevokedEntry)).all() == []


def test_garbage_pem_is_422(client, root_id, chain):
    response = _register_crl(client, root_id, "not a crl at all")
    assert response.status_code == 422
    _assert_no_crl_state(client)


def test_certificate_pem_is_422(client, root_id, chain):
    response = _register_crl(client, root_id, pem(chain["root_cert"]))
    assert response.status_code == 422
    _assert_no_crl_state(client)


def test_garbage_inside_crl_pem_block_is_422(client, root_id, chain):
    response = _register_crl(
        client,
        root_id,
        "-----BEGIN X509 CRL-----\nAAAA\n-----END X509 CRL-----\n",
    )
    assert response.status_code == 422
    _assert_no_crl_state(client)


def test_v1_crl_without_crl_number_is_422(client, root_id, chain):
    response = _register_crl(
        client, root_id, build_crl_without_crl_number(chain["root_cert"], chain["root_key"])
    )
    assert response.status_code == 422
    _assert_no_crl_state(client)


def test_crl_number_zero_is_422(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = raw_crl(
        chain["root_cert"],
        chain["root_key"],
        0,
        [],
        this_update=now - timedelta(days=1),
        next_update=now + timedelta(days=1),
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422
    _assert_no_crl_state(client)


def test_missing_next_update_is_422(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = raw_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [],
        this_update=now - timedelta(days=1),
        next_update=None,
        include_next_update=False,
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422
    _assert_no_crl_state(client)


def test_future_this_update_is_422(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = build_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [],
        this_update=now + timedelta(days=1),
        next_update=now + timedelta(days=2),
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422
    _assert_no_crl_state(client)


def test_next_update_not_after_this_update_is_422(client, root_id, chain):
    now = datetime.now(timezone.utc)
    # The high-level builder refuses this ordering, so assemble a signed
    # CRL by hand: nextUpdate before thisUpdate but otherwise well-formed.
    crl_pem = raw_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [],
        this_update=now - timedelta(days=1),
        next_update=now - timedelta(days=2),
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422
    _assert_no_crl_state(client)


def test_already_expired_crl_is_422(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = build_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [],
        this_update=now - timedelta(days=10),
        next_update=now - timedelta(seconds=1),
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422
    _assert_no_crl_state(client)


def test_issuer_dn_mismatch_is_422(client, root_id, chain):
    _other_key, other_cert = make_root("some-other-root")
    crl_pem = build_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [],
        issuer_name=other_cert.subject,
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422
    assert "some-other-root" not in response.text
    _assert_no_crl_state(client)


def test_signature_mismatch_is_422(client, root_id, chain):
    other_key, _other_cert = make_root("signer")
    crl_pem = build_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [],
        signing_key=other_key,
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422
    _assert_no_crl_state(client)


def test_zero_serial_is_422(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = raw_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [(0, now - timedelta(hours=1))],
        this_update=now - timedelta(days=1),
        next_update=now + timedelta(days=1),
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422
    _assert_no_crl_state(client)


def test_negative_serial_is_422(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = raw_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [(-1, now - timedelta(hours=1))],
        this_update=now - timedelta(days=1),
        next_update=now + timedelta(days=1),
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422
    _assert_no_crl_state(client)


def test_duplicate_serial_within_crl_is_422(client, root_id, chain):
    crl_pem = build_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [(1234, _past()), (1234, _past(2))],
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422
    _assert_no_crl_state(client)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_id", ""),
        ("tenant_id", "   "),
        ("workload_id", ""),
        ("trust_root_id", ""),
        ("trust_root_id", "not-a-uuid"),
        ("trust_root_id", "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"),
        ("crl_pem", ""),
        ("crl_pem", "   "),
    ],
)
def test_invalid_fields_return_422_without_writing_state(
    client, root_id, chain, field, value
):
    payload = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "crl_pem": build_crl(chain["root_cert"], chain["root_key"], 1, []),
    }
    payload[field] = value
    response = client.post("/v1/crls", json=payload)
    assert response.status_code == 422
    _assert_no_crl_state(client)


def test_extra_and_wrong_typed_fields_return_422(client, root_id, chain):
    base = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "crl_pem": build_crl(chain["root_cert"], chain["root_key"], 1, []),
    }
    extra = dict(base)
    extra["unexpected"] = "x"
    assert client.post("/v1/crls", json=extra).status_code == 422
    for field, value in [
        ("tenant_id", 123),
        ("workload_id", None),
        ("trust_root_id", ["x"]),
        ("crl_pem", {"x": 1}),
    ]:
        payload = dict(base)
        payload[field] = value
        assert client.post("/v1/crls", json=payload).status_code == 422, field
    _assert_no_crl_state(client)


# --- trust root isolation ---------------------------------------------------


def test_unknown_trust_root_returns_404(client, chain):
    response = _register_crl(
        client,
        "11111111-1111-1111-1111-111111111111",
        build_crl(chain["root_cert"], chain["root_key"], 1, []),
    )
    assert response.status_code == 404


def test_trust_root_of_other_tenant_returns_404(client, root_id, chain):
    response = _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 1, []),
        tenant_id=OTHER_TENANT,
    )
    assert response.status_code == 404


def test_trust_root_of_other_workload_returns_404(client, root_id, chain):
    response = _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 1, []),
        workload_id=OTHER_WORKLOAD,
    )
    assert response.status_code == 404


def test_crl_under_one_tenant_does_not_affect_other(client, chain):
    # The same root certificate anchors two distinct tenants; a CRL
    # enrolled under one must not reject the other's evidence.
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

    registered = _register_crl(
        client,
        roots[TENANT],
        build_crl(
            chain["root_cert"],
            chain["root_key"],
            1,
            [(chain["intermediate_cert"].serial_number, _past())],
        ),
    )
    assert registered.status_code == 201

    created_a = _challenge(client, tenant=TENANT)
    _, response_a = _submit_and_verify(
        client,
        created_a,
        make_evidence(created_a["nonce"], chain["leaf_key"], _full_chain(chain)),
        tenant=TENANT,
    )
    assert response_a.status_code == 200
    assert response_a.json()["status"] == "rejected"

    created_b = _challenge(client, tenant=OTHER_TENANT)
    _, response_b = _submit_and_verify(
        client,
        created_b,
        make_evidence(created_b["nonce"], chain["leaf_key"], _full_chain(chain)),
        tenant=OTHER_TENANT,
    )
    assert response_b.status_code == 200
    assert response_b.json()["status"] == "verified"


# --- real-time rejection during verification --------------------------------


def test_crl_revoked_direct_leaf_rejects(client, root_id, chain):
    crl_pem = build_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [(chain["direct_cert"].serial_number, _past())],
    )
    assert _register_crl(client, root_id, crl_pem).status_code == 201

    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"],
        chain["direct_key"],
        [chain["direct_cert"], chain["root_cert"]],
    )
    evidence_id, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    with client.app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "rejected"
        assert record.verification_result == "rejected"


def test_crl_revoked_intermediate_rejects(client, root_id, chain):
    crl_pem = build_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [(chain["intermediate_cert"].serial_number, _past())],
    )
    _register_crl(client, root_id, crl_pem)
    created = _challenge(client)
    _, response = _submit_and_verify(
        client, created, make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))
    )
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_crl_revoked_root_rejects(client, root_id, chain):
    crl_pem = build_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [(chain["root_cert"].serial_number, _past())],
    )
    _register_crl(client, root_id, crl_pem)
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["root_key"], [chain["root_cert"]]
    )
    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_serial_match_with_wrong_issuer_dn_does_not_reject(client, root_id, chain):
    # A root-signed CRL naming the LEAF serial cannot revoke the leaf of a
    # two-tier chain: the leaf's issuer DN is the intermediate, not the
    # root. Both issuer DN and serial must match.
    crl_pem = build_crl(
        chain["root_cert"],
        chain["root_key"],
        1,
        [(chain["leaf_cert"].serial_number, _past())],
    )
    _register_crl(client, root_id, crl_pem)
    created = _challenge(client)
    _, response = _submit_and_verify(
        client, created, make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))
    )
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_unrelated_serial_does_not_reject(client, root_id, chain):
    _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 1, [(999999, _past())]),
    )
    created = _challenge(client)
    _, response = _submit_and_verify(
        client,
        created,
        make_evidence(
            created["nonce"],
            chain["direct_key"],
            [chain["direct_cert"], chain["root_cert"]],
        ),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_future_revocation_date_does_not_reject_then_takes_effect(
    client, root_id, chain
):
    soon = datetime.now(timezone.utc) + timedelta(seconds=2)
    enrolled = _register_crl(
        client,
        root_id,
        build_crl(
            chain["root_cert"],
            chain["root_key"],
            1,
            [(chain["direct_cert"].serial_number, soon)],
        ),
    )
    # The entry is not effective at registration time.
    assert enrolled.json()["revoked_count"] == 0

    created_first = _challenge(client)
    _, first = _submit_and_verify(
        client,
        created_first,
        make_evidence(
            created_first["nonce"],
            chain["direct_key"],
            [chain["direct_cert"], chain["root_cert"]],
        ),
    )
    assert first.json()["status"] == "verified"

    import threading

    threading.Event().wait(2.5)
    created_second = _challenge(client)
    _, second = _submit_and_verify(
        client,
        created_second,
        make_evidence(
            created_second["nonce"],
            chain["direct_key"],
            [chain["direct_cert"], chain["root_cert"]],
        ),
    )
    assert second.json()["status"] == "rejected"


def test_higher_crl_replaces_current_snapshot_at_verification(client, root_id, chain):
    # Snapshot 1 lists an unrelated serial: the direct leaf verifies.
    _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 1, [(424242, _past())]),
    )
    created_first = _challenge(client)
    _, first = _submit_and_verify(
        client,
        created_first,
        make_evidence(
            created_first["nonce"],
            chain["direct_key"],
            [chain["direct_cert"], chain["root_cert"]],
        ),
    )
    assert first.json()["status"] == "verified"

    # Snapshot 2 (higher CRLNumber) lists the direct leaf: now rejected.
    _register_crl(
        client,
        root_id,
        build_crl(
            chain["root_cert"],
            chain["root_key"],
            2,
            [(chain["direct_cert"].serial_number, _past())],
        ),
    )
    created_second = _challenge(client)
    _, second = _submit_and_verify(
        client,
        created_second,
        make_evidence(
            created_second["nonce"],
            chain["direct_key"],
            [chain["direct_cert"], chain["root_cert"]],
        ),
    )
    assert second.json()["status"] == "rejected"


def test_expired_highest_crl_fails_closed_and_retry_succeeds(app, client, root_id, chain):
    _register_crl(
        client,
        root_id,
        build_crl(
            chain["root_cert"],
            chain["root_key"],
            1,
            [(chain["direct_cert"].serial_number, _past())],
        ),
    )
    # Simulate the highest snapshot's nextUpdate passing without an update.
    with app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE crl_snapshots SET next_update = :past"),
            {"past": datetime(2000, 1, 1, tzinfo=timezone.utc)},
        )

    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"],
        chain["direct_key"],
        [chain["direct_cert"], chain["root_cert"]],
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
    expired = client.post(f"/v1/evidence/{evidence_id}/verify", json=verify_body)
    assert expired.status_code == 500
    assert "BEGIN CERTIFICATE" not in expired.text
    # The evidence is left received and the same request can be retried.
    with app.state.session_factory() as session:
        assert session.get(Evidence, evidence_id).status == "received"

    # A fresh, higher-numbered CRL restores a valid current snapshot.
    _register_crl(
        client,
        root_id,
        build_crl(
            chain["root_cert"],
            chain["root_key"],
            2,
            [(chain["direct_cert"].serial_number, _past())],
        ),
    )
    retried = client.post(f"/v1/evidence/{evidence_id}/verify", json=verify_body)
    assert retried.status_code == 200
    assert retried.json()["status"] == "rejected"


def test_lower_valid_snapshot_does_not_shadow_expired_highest(app, client, root_id, chain):
    _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 1, []),
    )
    _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 2, []),
    )
    # Expire only the highest snapshot; the still-valid number 1 must not
    # be used in its place.
    with app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE crl_snapshots SET next_update = :past WHERE crl_number = 2"
            ),
            {"past": datetime(2000, 1, 1, tzinfo=timezone.utc)},
        )
    created = _challenge(client)
    _, response = _submit_and_verify(
        client,
        created,
        make_evidence(
            created["nonce"],
            chain["direct_key"],
            [chain["direct_cert"], chain["root_cert"]],
        ),
    )
    assert response.status_code == 500


def test_no_enrolled_crl_keeps_existing_behavior(client, root_id, chain):
    created = _challenge(client)
    _, response = _submit_and_verify(
        client,
        created,
        make_evidence(
            created["nonce"],
            chain["direct_key"],
            [chain["direct_cert"], chain["root_cert"]],
        ),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_fingerprint_revocation_and_crl_are_union(client, root_id, chain):
    # The fingerprint registry revokes the intermediate and the CRL names
    # the direct leaf; with both sources present the chain anchored to the
    # root is rejected. Each source is also sufficient on its own.
    _register_fingerprint(
        client, root_id, chain["intermediate_cert"], "2020-01-01T00:00:00Z"
    )
    _register_crl(
        client,
        root_id,
        build_crl(
            chain["root_cert"],
            chain["root_key"],
            1,
            [(chain["intermediate_cert"].serial_number, _past())],
        ),
    )
    created = _challenge(client)
    _, response = _submit_and_verify(
        client, created, make_evidence(created["nonce"], chain["leaf_key"], _full_chain(chain))
    )
    assert response.json()["status"] == "rejected"


def test_retired_root_rejects_even_with_crl_present(client, root_id, chain):
    _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 1, []),
    )
    retired = client.post(
        f"/v1/trust-roots/{root_id}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert retired.status_code == 200
    created = _challenge(client)
    _, response = _submit_and_verify(
        client,
        created,
        make_evidence(
            created["nonce"],
            chain["direct_key"],
            [chain["direct_cert"], chain["root_cert"]],
        ),
    )
    assert response.json()["status"] == "rejected"


def test_crl_enrollment_after_settlement_is_not_retroactive(client, root_id, chain):
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"],
        chain["direct_key"],
        [chain["direct_cert"], chain["root_cert"]],
    )
    evidence_id, first = _submit_and_verify(client, created, evidence)
    assert first.json()["status"] == "verified"

    enrolled = _register_crl(
        client,
        root_id,
        build_crl(
            chain["root_cert"],
            chain["root_key"],
            1,
            [(chain["direct_cert"].serial_number, _past())],
        ),
    )
    assert enrolled.status_code == 201

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


def test_crl_does_not_change_hmac_format_verification(client, root_id, chain):
    _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 1, [(1, _past())]),
    )
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
    signed = json.dumps(
        {"claims": {}, "nonce": nonce}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    mac = hmac.new(mac_key, signed, hashlib.sha256).hexdigest()
    evidence = json.dumps(
        {"nonce": nonce, "claims": {}, "mac": mac},
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


def test_crl_rejection_response_never_contains_material(client, root_id, chain):
    _register_crl(
        client,
        root_id,
        build_crl(
            chain["root_cert"],
            chain["root_key"],
            1,
            [(chain["direct_cert"].serial_number, _past())],
        ),
    )
    created = _challenge(client)
    _, response = _submit_and_verify(
        client,
        created,
        make_evidence(
            created["nonce"],
            chain["direct_key"],
            [chain["direct_cert"], chain["root_cert"]],
        ),
    )
    assert response.status_code == 200
    assert "BEGIN X509 CRL" not in response.text
    assert "BEGIN CERTIFICATE" not in response.text


# --- failure semantics ------------------------------------------------------


def test_crl_registry_unavailable_returns_500_and_keeps_evidence_received(
    app, client, root_id, chain
):
    _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 1, []),
    )
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"],
        chain["direct_key"],
        [chain["direct_cert"], chain["root_cert"]],
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

    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE crl_snapshots"))

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
    with app.state.session_factory() as session:
        assert session.get(Evidence, evidence_id).status == "received"

    CRLSnapshot.__table__.create(app.state.engine)
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


def test_failed_registration_read_returns_500_and_leaves_no_record(
    app, client, root_id, chain
):
    engine = app.state.engine

    def fail_lookup(conn, cursor, statement, parameters, context, executemany):
        if "FROM trust_roots" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_lookup)
    try:
        response = _register_crl(
            client,
            root_id,
            build_crl(chain["root_cert"], chain["root_key"], 1, []),
        )
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_lookup)

    with app.state.session_factory() as session:
        assert session.scalars(select(CRLSnapshot)).all() == []
    recovered = _register_crl(
        client,
        root_id,
        build_crl(chain["root_cert"], chain["root_key"], 1, []),
    )
    assert recovered.status_code == 201
