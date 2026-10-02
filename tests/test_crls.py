"""Tests for trust-root-scoped X.509 v2 CRL registration and enforcement
during x509-attested-nonce-json evidence verification."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select

from proof_release.app import create_app
from proof_release.db import (
    CertificateRevocationList,
    CrlRevokedCertificate,
    Evidence,
)
from proof_release.verifiers import X509_ATTESTED_NONCE_JSON

from cryptography import x509
from cryptography.x509.oid import NameOID

from x509_helpers import (
    crl_without_next_update,
    make_crl,
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


def _full_chain(chain):
    return [chain["leaf_cert"], chain["intermediate_cert"], chain["root_cert"]]


def _register_crl(client, root_id_value, crl_pem, **scope):
    payload = {
        "tenant_id": scope.get("tenant_id", TENANT),
        "workload_id": scope.get("workload_id", WORKLOAD),
        "trust_root_id": root_id_value,
        "crl_pem": crl_pem,
    }
    return client.post("/v1/crls", json=payload)


def _challenge(client, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/challenges", json={"tenant_id": tenant, "workload_id": workload}
    )
    assert response.status_code == 201
    return response.json()


def _verify(client, evidence, leaf_key, chain_certificates, tenant=TENANT,
            workload=WORKLOAD):
    created = _challenge(client, tenant=tenant, workload=workload)
    evidence_doc = make_evidence(created["nonce"], leaf_key, chain_certificates)
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": X509_ATTESTED_NONCE_JSON,
            "evidence": evidence_doc,
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
            "evidence": evidence_doc,
        },
    )
    return evidence_id, response


# --- registration -----------------------------------------------------------


def test_register_crl_returns_201_compact_json(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [(chain["intermediate_cert"].serial_number, now - timedelta(hours=1))],
        number=7,
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
    assert body["crl_number"] == 7
    assert body["revoked_count"] == 1
    assert body["this_update"].endswith("+00:00")
    assert body["next_update"].endswith("+00:00")
    # The CRL body and any certificate material are never echoed.
    assert "BEGIN X509 CRL" not in response.text
    assert "BEGIN CERTIFICATE" not in response.text
    # Compact JSON: no whitespace between tokens.
    assert b", " not in response.content
    assert b": " not in response.content


def test_revoked_count_only_counts_arrived_revocation_dates(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [
            (1001, now - timedelta(hours=2)),
            (1002, now - timedelta(seconds=1)),
            (1003, now + timedelta(days=3)),
        ],
        number=1,
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 201
    assert response.json()["revoked_count"] == 2


def test_registration_persists_snapshot_and_entries_across_restart(
    tmp_path, chain
):
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
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [(chain["intermediate_cert"].serial_number, now - timedelta(hours=1))],
        number=1,
    )
    created = _register_crl(client1, root, crl_pem)
    assert created.status_code == 201
    crl_id = created.json()["crl_id"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        snapshots = session.scalars(select(CertificateRevocationList)).all()
        entries = session.scalars(select(CrlRevokedCertificate)).all()
    assert len(snapshots) == 1
    assert snapshots[0].crl_id == crl_id
    assert snapshots[0].crl_number == 1
    assert snapshots[0].revoked_count == 1
    assert len(entries) == 1
    assert entries[0].crl_id == crl_id
    assert entries[0].serial_number == str(
        chain["intermediate_cert"].serial_number
    )
    app2.state.engine.dispose()


def test_higher_number_replaces_current_but_keeps_old_snapshots(
    client, app, root_id, chain
):
    now = datetime.now(timezone.utc)
    serial = chain["intermediate_cert"].serial_number
    first = make_crl(
        chain["root_cert"], chain["root_key"],
        [(serial, now - timedelta(hours=1))], number=1,
    )
    assert _register_crl(client, root_id, first).status_code == 201
    # A strictly higher, empty CRL becomes the current snapshot.
    second = make_crl(
        chain["root_cert"], chain["root_key"], [], number=2,
        this_update=now - timedelta(minutes=5),
    )
    response = _register_crl(client, root_id, second)
    assert response.status_code == 201
    assert response.json()["revoked_count"] == 0

    with app.state.session_factory() as session:
        snapshots = session.scalars(
            select(CertificateRevocationList).order_by(
                CertificateRevocationList.crl_number
            )
        ).all()
    # Snapshots are immutable: the replaced one is retained.
    assert [s.crl_number for s in snapshots] == [1, 2]


@pytest.mark.parametrize("second_number", [1])
def test_same_crl_number_returns_409_and_keeps_original(
    client, app, root_id, chain, second_number
):
    now = datetime.now(timezone.utc)
    first_pem = make_crl(
        chain["root_cert"], chain["root_key"],
        [(chain["intermediate_cert"].serial_number, now - timedelta(hours=1))],
        number=1,
    )
    first = _register_crl(client, root_id, first_pem)
    assert first.status_code == 201
    first_id = first.json()["crl_id"]

    # Re-posting the byte-identical body hits both the equal-number and
    # equal-content rules and must be a stable 409.
    repeat = _register_crl(client, root_id, first_pem)
    assert repeat.status_code == 409
    with app.state.session_factory() as session:
        rows = session.scalars(select(CertificateRevocationList)).all()
    assert len(rows) == 1
    assert rows[0].crl_id == first_id


def test_lower_number_returns_409(client, app, root_id, chain):
    now = datetime.now(timezone.utc)
    second = make_crl(
        chain["root_cert"], chain["root_key"], [], number=2,
        this_update=now - timedelta(minutes=10),
    )
    assert _register_crl(client, root_id, second).status_code == 201
    first = make_crl(
        chain["root_cert"], chain["root_key"], [], number=1,
        this_update=now - timedelta(hours=2),
    )
    response = _register_crl(client, root_id, first)
    assert response.status_code == 409
    with app.state.session_factory() as session:
        rows = session.scalars(select(CertificateRevocationList)).all()
    assert [r.crl_number for r in rows] == [2]


def test_concurrent_equal_number_registrations_settle_at_most_once(
    app, root_id, chain
):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"],
        [(chain["intermediate_cert"].serial_number, now - timedelta(hours=1))],
        number=1,
    )

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
        responses = list(pool.map(lambda _: register(), range(12)))

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(201) == 1
    assert statuses.count(409) == 11
    with app.state.session_factory() as session:
        assert len(session.scalars(select(CertificateRevocationList)).all()) == 1
        assert len(session.scalars(select(CrlRevokedCertificate)).all()) == 1


# --- field / format validation (422 before storage) -------------------------


def test_unknown_field_is_422(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    response = client.post(
        "/v1/crls",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "crl_pem": crl_pem,
            "extra": "nope",
        },
    )
    assert response.status_code == 422


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
        ("crl_pem", ""),
        ("crl_pem", "   "),
    ],
)
def test_invalid_fields_return_422_without_writing_state(
    client, app, root_id, chain, field, value
):
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    payload = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "crl_pem": crl_pem,
    }
    payload[field] = value
    response = client.post("/v1/crls", json=payload)
    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.scalars(select(CertificateRevocationList)).all() == []
        assert session.scalars(select(CrlRevokedCertificate)).all() == []


def test_wrong_types_return_422(client, root_id, chain):
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    base = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "crl_pem": crl_pem,
    }
    for field, value in [
        ("tenant_id", 123),
        ("workload_id", None),
        ("trust_root_id", ["x"]),
        ("crl_pem", {"x": 1}),
        ("crl_pem", 42),
    ]:
        payload = dict(base)
        payload[field] = value
        response = client.post("/v1/crls", json=payload)
        assert response.status_code == 422, field


@pytest.mark.parametrize(
    "bad_pem",
    [
        "not a pem at all",
        "-----BEGIN X509 CRL-----\nAAAA\n-----END X509 CRL-----\n",
    ],
)
def test_bad_pem_or_asn1_is_422(client, app, root_id, bad_pem):
    response = _register_crl(client, root_id, bad_pem)
    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.scalars(select(CertificateRevocationList)).all() == []


def test_certificate_pem_instead_of_crl_is_422(client, root_id, chain):
    response = _register_crl(client, root_id, pem(chain["root_cert"]))
    assert response.status_code == 422


def test_missing_crl_number_extension_is_422(client, root_id, chain):
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"], [], number=1, include_number=False
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422


def test_missing_next_update_is_422(client, root_id, chain):
    crl_pem = crl_without_next_update(
        chain["root_cert"], chain["root_key"], number=1
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422


def test_this_update_in_the_future_is_422(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"], [], number=1,
        this_update=now + timedelta(days=1),
        next_update=now + timedelta(days=2),
    )
    assert _register_crl(client, root_id, crl_pem).status_code == 422


def test_next_update_not_after_this_update_is_422(client, root_id, chain):
    now = datetime.now(timezone.utc)
    this_update = now - timedelta(days=1)
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"], [], number=1,
        this_update=this_update, next_update=this_update,
    )
    assert _register_crl(client, root_id, crl_pem).status_code == 422


def test_already_expired_next_update_is_422(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"], [], number=1,
        this_update=now - timedelta(days=10),
        next_update=now - timedelta(days=1),
    )
    assert _register_crl(client, root_id, crl_pem).status_code == 422


def test_duplicate_serial_in_crl_is_422(client, app, root_id, chain):
    now = datetime.now(timezone.utc)
    serial = chain["intermediate_cert"].serial_number
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"],
        [(serial, now - timedelta(hours=1)), (serial, now - timedelta(hours=2))],
        number=1,
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.scalars(select(CertificateRevocationList)).all() == []
        assert session.scalars(select(CrlRevokedCertificate)).all() == []


def test_intrinsic_422_takes_precedence_over_unknown_root_404(client, chain):
    # A malformed CRL body is rejected 422 even when the named root does
    # not exist — intrinsic checks never reveal root existence.
    response = _register_crl(
        client,
        "11111111-1111-1111-1111-111111111111",
        "garbage",
    )
    assert response.status_code == 422


def test_issuer_dn_mismatch_is_422(client, root_id, chain):
    other_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "some-other-ca")]
    )
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"], [], number=1,
        issuer_name=other_name,
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422
    assert "BEGIN X509 CRL" not in response.text


def test_signature_not_from_root_key_is_422(client, app, root_id, chain):
    foreign_key, _foreign_cert = make_root("foreign")
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"], [], number=1,
        signer_key=foreign_key,
    )
    response = _register_crl(client, root_id, crl_pem)
    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.scalars(select(CertificateRevocationList)).all() == []


# --- trust root isolation ---------------------------------------------------


def test_unknown_trust_root_returns_404(client, root_id, chain):
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    response = _register_crl(
        client, "11111111-1111-1111-1111-111111111111", crl_pem
    )
    assert response.status_code == 404


def test_trust_root_of_other_tenant_returns_404(client, root_id, chain):
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    response = _register_crl(
        client, root_id, crl_pem, tenant_id=OTHER_TENANT
    )
    assert response.status_code == 404


def test_trust_root_of_other_workload_returns_404(client, root_id, chain):
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    response = _register_crl(
        client, root_id, crl_pem, workload_id=OTHER_WORKLOAD
    )
    assert response.status_code == 404


def test_crl_under_one_tenant_does_not_affect_other(client, chain):
    # The same root certificate anchors two tenants; a CRL registered for
    # one tenant must not reject the other's evidence.
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

    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"],
        [(chain["intermediate_cert"].serial_number, now - timedelta(hours=1))],
        number=1,
    )
    assert _register_crl(client, roots[TENANT], crl_pem).status_code == 201

    _, response_a = _verify(
        client, None, chain["leaf_key"], _full_chain(chain), tenant=TENANT
    )
    assert response_a.status_code == 200
    assert response_a.json()["status"] == "rejected"

    _, response_b = _verify(
        client, None, chain["leaf_key"], _full_chain(chain), tenant=OTHER_TENANT
    )
    assert response_b.status_code == 200
    assert response_b.json()["status"] == "verified"


# --- enforcement during verification ----------------------------------------


def test_crl_revoked_intermediate_rejects(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"],
        [(chain["intermediate_cert"].serial_number, now - timedelta(hours=1))],
        number=1,
    )
    assert _register_crl(client, root_id, crl_pem).status_code == 201

    evidence_id, response = _verify(
        client, None, chain["leaf_key"], _full_chain(chain)
    )
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    with client.app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "rejected"
        assert record.verification_result == "rejected"


def test_crl_revoked_root_certificate_rejects(client, root_id, chain):
    # The self-signed root's issuer DN is the root subject itself, so its
    # serial appearing in the root's CRL rejects a root-only chain.
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"],
        [(chain["root_cert"].serial_number, now - timedelta(hours=1))],
        number=1,
    )
    assert _register_crl(client, root_id, crl_pem).status_code == 201
    _, response = _verify(
        client, None, chain["root_key"], [chain["root_cert"]]
    )
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_crl_revoked_direct_root_issued_leaf_rejects(client, chain):
    # A leaf issued directly by the root (chain [leaf, root]) is a CRL
    # candidate, unlike a leaf issued by an intermediate.
    direct_leaf_key, direct_leaf_cert = make_leaf(
        chain["root_cert"], chain["root_key"], "direct-leaf"
    )
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": "tenant-direct",
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    direct_root_id = response.json()["root_id"]
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"],
        [(direct_leaf_cert.serial_number, now - timedelta(hours=1))],
        number=1,
    )
    assert _register_crl(
        client, direct_root_id, crl_pem, tenant_id="tenant-direct"
    ).status_code == 201
    _, verify_response = _verify(
        client,
        None,
        direct_leaf_key,
        [direct_leaf_cert, chain["root_cert"]],
        tenant="tenant-direct",
    )
    assert verify_response.json()["status"] == "rejected"


def test_crl_does_not_match_certificate_with_different_issuer_dn(
    client, root_id, chain
):
    # The leaf is issued by the intermediate, not the CRL issuer (root):
    # its serial in the root CRL must not revoke it.
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"],
        [(chain["leaf_cert"].serial_number, now - timedelta(hours=1))],
        number=1,
    )
    assert _register_crl(client, root_id, crl_pem).status_code == 201
    _, response = _verify(
        client, None, chain["leaf_key"], _full_chain(chain)
    )
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_future_revocation_date_does_not_reject_yet(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"],
        [(chain["intermediate_cert"].serial_number, now + timedelta(days=5))],
        number=1,
    )
    assert _register_crl(client, root_id, crl_pem).status_code == 201
    _, response = _verify(
        client, None, chain["leaf_key"], _full_chain(chain)
    )
    assert response.json()["status"] == "verified"


def test_replaced_clean_snapshot_allows_fresh_evidence(client, root_id, chain):
    now = datetime.now(timezone.utc)
    revoking = make_crl(
        chain["root_cert"], chain["root_key"],
        [(chain["intermediate_cert"].serial_number, now - timedelta(hours=1))],
        number=1,
    )
    assert _register_crl(client, root_id, revoking).status_code == 201
    _, rejected = _verify(
        client, None, chain["leaf_key"], _full_chain(chain)
    )
    assert rejected.json()["status"] == "rejected"

    clean = make_crl(
        chain["root_cert"], chain["root_key"], [], number=2,
        this_update=now - timedelta(minutes=5),
    )
    assert _register_crl(client, root_id, clean).status_code == 201
    _, allowed = _verify(
        client, None, chain["leaf_key"], _full_chain(chain)
    )
    assert allowed.status_code == 200
    assert allowed.json()["status"] == "verified"


def test_expired_current_crl_fails_closed_until_fresh_snapshot(
    client, app, root_id, chain
):
    now = datetime.now(timezone.utc)
    short = make_crl(
        chain["root_cert"], chain["root_key"], [], number=1,
        this_update=now - timedelta(seconds=10),
        next_update=now + timedelta(seconds=2),
    )
    assert _register_crl(client, root_id, short).status_code == 201

    evidence_id, before = _verify(
        client, None, chain["leaf_key"], _full_chain(chain)
    )
    assert before.json()["status"] == "verified"

    # A different evidence submitted before expiry but verified after must
    # fail closed.
    created = _challenge(client)
    evidence_doc = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": X509_ATTESTED_NONCE_JSON,
            "evidence": evidence_doc,
        },
    )
    pending_id = submitted.json()["evidence_id"]
    threading.Event().wait(2.5)

    expired = client.post(
        f"/v1/evidence/{pending_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence_doc,
        },
    )
    assert expired.status_code == 500
    assert "BEGIN X509 CRL" not in expired.text
    with app.state.session_factory() as session:
        assert session.get(Evidence, pending_id).status == "received"

    # The identical request can be retried but still fails closed...
    retry = client.post(
        f"/v1/evidence/{pending_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence_doc,
        },
    )
    assert retry.status_code == 500

    # ...until a higher-numbered, in-window snapshot is registered.
    fresh = make_crl(
        chain["root_cert"], chain["root_key"], [], number=2,
        this_update=now - timedelta(seconds=5),
    )
    assert _register_crl(client, root_id, fresh).status_code == 201
    recovered = client.post(
        f"/v1/evidence/{pending_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence_doc,
        },
    )
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "verified"


def test_no_crl_registered_keeps_existing_behavior(client, root_id, chain):
    _, response = _verify(
        client, None, chain["leaf_key"], _full_chain(chain)
    )
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


# --- union with fingerprint revocation, retirement precedence ---------------


def test_fingerprint_revocation_still_rejects_with_clean_crl(
    client, root_id, chain
):
    import hashlib
    from cryptography.hazmat.primitives.serialization import Encoding
    from proof_release.envelopes import b64url_encode

    clean = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    assert _register_crl(client, root_id, clean).status_code == 201

    fp = b64url_encode(
        hashlib.sha256(
            chain["leaf_cert"].public_bytes(Encoding.DER)
        ).digest()
    )
    response = client.post(
        "/v1/revocations",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "certificate_fingerprint": fp,
            "effective_at": "2020-01-01T00:00:00Z",
        },
    )
    assert response.status_code == 201
    _, verify_response = _verify(
        client, None, chain["leaf_key"], _full_chain(chain)
    )
    assert verify_response.json()["status"] == "rejected"


def test_expired_crl_fails_closed_even_with_fingerprint_hit(
    client, app, root_id, chain
):
    # The CRL freshness gate is unconditional for a CRL-enrolled root:
    # once the highest CRLNumber is past nextUpdate, even an independent
    # fingerprint revocation cannot settle the evidence; the request fails
    # closed (500, evidence stays received) until a fresh CRL arrives.
    import hashlib
    from cryptography.hazmat.primitives.serialization import Encoding
    from proof_release.envelopes import b64url_encode

    now = datetime.now(timezone.utc)
    short = make_crl(
        chain["root_cert"], chain["root_key"], [], number=1,
        this_update=now - timedelta(seconds=10),
        next_update=now + timedelta(seconds=2),
    )
    assert _register_crl(client, root_id, short).status_code == 201
    fp = b64url_encode(
        hashlib.sha256(
            chain["leaf_cert"].public_bytes(Encoding.DER)
        ).digest()
    )
    assert client.post(
        "/v1/revocations",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "certificate_fingerprint": fp,
            "effective_at": "2020-01-01T00:00:00Z",
        },
    ).status_code == 201

    created = _challenge(client)
    evidence_doc = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    pending_id = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": X509_ATTESTED_NONCE_JSON,
            "evidence": evidence_doc,
        },
    ).json()["evidence_id"]
    threading.Event().wait(2.5)
    expired = client.post(
        f"/v1/evidence/{pending_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence_doc,
        },
    )
    assert expired.status_code == 500
    with app.state.session_factory() as session:
        assert session.get(Evidence, pending_id).status == "received"


def test_crl_revocation_and_fingerprint_revocation_are_union(
    client, root_id, chain
):
    # Only the intermediate is CRL-revoked; the fingerprint registry is
    # empty — rejection still occurs through the CRL.
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"],
        [(chain["intermediate_cert"].serial_number, now - timedelta(hours=1))],
        number=1,
    )
    assert _register_crl(client, root_id, crl_pem).status_code == 201
    _, response = _verify(
        client, None, chain["leaf_key"], _full_chain(chain)
    )
    assert response.json()["status"] == "rejected"


def test_retired_root_rejects_even_when_crl_expired(client, root_id, chain):
    now = datetime.now(timezone.utc)
    short = make_crl(
        chain["root_cert"], chain["root_key"], [], number=1,
        this_update=now - timedelta(seconds=10),
        next_update=now + timedelta(seconds=2),
    )
    assert _register_crl(client, root_id, short).status_code == 201
    assert client.post(
        f"/v1/trust-roots/{root_id}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).status_code == 200
    threading.Event().wait(2.5)
    # Retirement precedes the CRL check: an expired CRL cannot turn the
    # outcome into a 500; the evidence is conclusively rejected.
    _, response = _verify(
        client, None, chain["leaf_key"], _full_chain(chain)
    )
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


# --- non-retroactivity and response hygiene ---------------------------------


def test_crl_registered_after_settlement_is_not_retroactive(
    client, app, root_id, chain
):
    evidence_id, first = _verify(
        client, None, chain["leaf_key"], _full_chain(chain)
    )
    assert first.json()["status"] == "verified"

    # A later CRL never rewrites the settled conclusion; the fresh
    # rejection applies only to evidence that settles afterwards.
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"],
        [(chain["intermediate_cert"].serial_number, now - timedelta(hours=1))],
        number=1,
    )
    assert _register_crl(client, root_id, crl_pem).status_code == 201

    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "verified"
        assert record.verification_result == "accepted"

    _, later = _verify(
        client, None, chain["leaf_key"], _full_chain(chain)
    )
    assert later.json()["status"] == "rejected"


def test_rejection_response_never_contains_crl_or_certificate_material(
    client, root_id, chain
):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"], chain["root_key"],
        [(chain["intermediate_cert"].serial_number, now - timedelta(hours=1))],
        number=1,
    )
    _register_crl(client, root_id, crl_pem)
    _, response = _verify(
        client, None, chain["leaf_key"], _full_chain(chain)
    )
    assert response.status_code == 200
    assert "BEGIN X509 CRL" not in response.text
    assert "BEGIN CERTIFICATE" not in response.text
    assert pem(chain["intermediate_cert"]) not in response.text


# --- storage / service failure semantics ------------------------------------


def test_failed_snapshot_write_leaves_no_partial_registration(
    app, client, root_id, chain
):
    engine = app.state.engine

    def fail_insert(conn, cursor, statement, parameters, context, executemany):
        if "INTO certificate_revocation_lists" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_insert)
    try:
        crl_pem = make_crl(
            chain["root_cert"], chain["root_key"],
            [(chain["intermediate_cert"].serial_number, datetime.now(timezone.utc))],
            number=1,
        )
        response = _register_crl(client, root_id, crl_pem)
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_insert)

    with app.state.session_factory() as session:
        assert session.scalars(select(CertificateRevocationList)).all() == []
        assert session.scalars(select(CrlRevokedCertificate)).all() == []


def test_failed_entry_write_leaves_no_partial_registration(
    app, client, root_id, chain
):
    engine = app.state.engine

    def fail_insert(conn, cursor, statement, parameters, context, executemany):
        if "INTO crl_revoked_certificates" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_insert)
    try:
        now = datetime.now(timezone.utc)
        crl_pem = make_crl(
            chain["root_cert"], chain["root_key"],
            [(chain["intermediate_cert"].serial_number, now - timedelta(hours=1))],
            number=1,
        )
        response = _register_crl(client, root_id, crl_pem)
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_insert)

    with app.state.session_factory() as session:
        assert session.scalars(select(CertificateRevocationList)).all() == []
        assert session.scalars(select(CrlRevokedCertificate)).all() == []


def test_failed_registration_read_returns_500_and_keeps_previous_snapshot(
    app, client, root_id, chain
):
    first = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    created = _register_crl(client, root_id, first)
    assert created.status_code == 201
    first_id = created.json()["crl_id"]
    engine = app.state.engine

    def fail_lookup(conn, cursor, statement, parameters, context, executemany):
        if "FROM trust_roots" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_lookup)
    try:
        second = make_crl(
            chain["root_cert"], chain["root_key"], [], number=2,
            this_update=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        response = _register_crl(client, root_id, second)
        assert response.status_code == 500
    finally:
        event.remove(engine, "before_cursor_execute", fail_lookup)

    # The previous snapshot stays current; the failed higher-numbered
    # registration left nothing behind and can now be retried once.
    with app.state.session_factory() as session:
        rows = session.scalars(select(CertificateRevocationList)).all()
    assert len(rows) == 1
    assert rows[0].crl_id == first_id
    assert rows[0].crl_number == 1
    recovered = _register_crl(
        client,
        root_id,
        make_crl(
            chain["root_cert"], chain["root_key"], [], number=2,
            this_update=datetime.now(timezone.utc) - timedelta(minutes=1),
        ),
    )
    assert recovered.status_code == 201


# --- read-only snapshot query (GET /v1/crls/{crl_id}) ------------------------


def _get_crl(client, crl_id, **scope):
    params = {
        "tenant_id": scope.get("tenant_id", TENANT),
        "workload_id": scope.get("workload_id", WORKLOAD),
        "trust_root_id": scope.get("trust_root_id", scope.get("root_id", "")),
    }
    params.update(scope.get("extra_params", {}))
    return client.get(f"/v1/crls/{crl_id}", params=params)


def test_get_crl_returns_snapshot_and_all_entries(client, root_id, chain):
    now = datetime.now(timezone.utc)
    serials = [1001, 1002, 1003]
    crl_pem = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [
            (serials[1], now - timedelta(seconds=1)),
            # Not yet effective at registration: still listed, never counted.
            (serials[2], now + timedelta(days=3)),
            (serials[0], now - timedelta(hours=2)),
        ],
        number=7,
    )
    created = _register_crl(client, root_id, crl_pem)
    assert created.status_code == 201
    crl_id = created.json()["crl_id"]

    response = _get_crl(client, crl_id, root_id=root_id)
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == [
        "crl_id",
        "tenant_id",
        "workload_id",
        "trust_root_id",
        "crl_number",
        "crl_sha256",
        "this_update",
        "next_update",
        "revoked_count",
        "created_at",
        "entries",
    ]
    assert body["crl_id"] == crl_id
    assert body["tenant_id"] == TENANT
    assert body["workload_id"] == WORKLOAD
    assert body["trust_root_id"] == root_id
    assert body["crl_number"] == 7
    assert isinstance(body["crl_number"], int)
    assert isinstance(body["revoked_count"], int)
    # The registration-time count is kept verbatim, never recomputed.
    assert body["revoked_count"] == 2
    sha256 = body["crl_sha256"]
    assert len(sha256) == 64 and sha256 == sha256.lower()
    int(sha256, 16)
    for field in ("this_update", "next_update", "created_at"):
        assert body[field].endswith("+00:00")

    entries = body["entries"]
    assert len(entries) == 3
    for entry in entries:
        assert list(entry.keys()) == [
            "entry_id",
            "issuer_dn",
            "serial_number",
            "revocation_date",
        ]
        assert entry["revocation_date"].endswith("+00:00")
    # Sorted by revocation_date ascending; the future-dated entry is last.
    assert [e["serial_number"] for e in entries] == ["1001", "1002", "1003"]
    dates = [e["revocation_date"] for e in entries]
    assert dates == sorted(dates)
    # No CRL or certificate material is ever returned.
    assert "BEGIN X509 CRL" not in response.text
    assert "BEGIN CERTIFICATE" not in response.text


def test_get_crl_empty_snapshot_has_no_entries(client, root_id, chain):
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    crl_id = _register_crl(client, root_id, crl_pem).json()["crl_id"]
    response = _get_crl(client, crl_id, root_id=root_id)
    assert response.status_code == 200
    body = response.json()
    assert body["entries"] == []
    assert body["revoked_count"] == 0


def test_get_crl_repeated_queries_are_identical(client, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [(1001, now - timedelta(hours=1))],
        number=1,
    )
    crl_id = _register_crl(client, root_id, crl_pem).json()["crl_id"]
    first = _get_crl(client, crl_id, root_id=root_id)
    second = _get_crl(client, crl_id, root_id=root_id)
    assert first.status_code == 200
    assert first.content == second.content


def test_get_crl_is_read_only(client, app, root_id, chain):
    now = datetime.now(timezone.utc)
    crl_pem = make_crl(
        chain["root_cert"],
        chain["root_key"],
        [(1001, now - timedelta(hours=1))],
        number=1,
    )
    crl_id = _register_crl(client, root_id, crl_pem).json()["crl_id"]
    response = _get_crl(client, crl_id, root_id=root_id)
    assert response.status_code == 200
    with app.state.session_factory() as session:
        snapshots = session.scalars(select(CertificateRevocationList)).all()
        entries = session.scalars(select(CrlRevokedCertificate)).all()
    assert len(snapshots) == 1
    assert snapshots[0].crl_id == crl_id
    assert len(entries) == 1


@pytest.mark.parametrize(
    "bad_id",
    [
        "not-a-uuid",
        "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaZ",
        " aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "",
    ],
)
def test_get_crl_non_canonical_id_is_422(client, root_id, chain, bad_id):
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    crl_id = _register_crl(client, root_id, crl_pem).json()["crl_id"]
    target = bad_id if bad_id else "%20"
    response = _get_crl(client, target, root_id=root_id)
    assert response.status_code == 422
    # A non-canonical spelling of the real id is also rejected.
    assert _get_crl(client, crl_id.upper(), root_id=root_id).status_code == 422


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
    ],
)
def test_get_crl_invalid_scope_params_are_422(client, root_id, chain, field, value):
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    crl_id = _register_crl(client, root_id, crl_pem).json()["crl_id"]
    response = _get_crl(client, crl_id, root_id=root_id, **{field: value})
    assert response.status_code == 422


def test_get_crl_missing_scope_params_are_422(client, root_id, chain):
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    crl_id = _register_crl(client, root_id, crl_pem).json()["crl_id"]
    assert client.get(f"/v1/crls/{crl_id}").status_code == 422
    assert (
        client.get(f"/v1/crls/{crl_id}", params={"tenant_id": TENANT}).status_code
        == 422
    )


def test_get_crl_extra_query_param_is_422(client, root_id, chain):
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    crl_id = _register_crl(client, root_id, crl_pem).json()["crl_id"]
    response = _get_crl(
        client, crl_id, root_id=root_id, extra_params={"cursor": "abc"}
    )
    assert response.status_code == 422


def test_get_crl_empty_path_segment_is_422(client):
    assert client.get("/v1/crls/").status_code == 422


def test_get_crl_unknown_id_is_404(client, root_id, chain):
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    assert _register_crl(client, root_id, crl_pem).status_code == 201
    response = _get_crl(
        client, "11111111-1111-1111-1111-111111111111", root_id=root_id
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "crl not found"


@pytest.mark.parametrize(
    "scope_override",
    [
        {"tenant_id": OTHER_TENANT},
        {"workload_id": OTHER_WORKLOAD},
    ],
)
def test_get_crl_cross_scope_is_indistinguishable_404(
    client, root_id, chain, scope_override
):
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    crl_id = _register_crl(client, root_id, crl_pem).json()["crl_id"]
    response = _get_crl(client, crl_id, root_id=root_id, **scope_override)
    assert response.status_code == 404
    assert response.json()["detail"] == "crl not found"


def test_get_crl_cross_trust_root_is_indistinguishable_404(client, chain):
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
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    crl_id = _register_crl(client, roots[TENANT], crl_pem).json()["crl_id"]
    # The other tenant's root id is a canonical UUID but not the CRL's root.
    response = _get_crl(client, crl_id, root_id=roots[OTHER_TENANT])
    assert response.status_code == 404
    assert response.json()["detail"] == "crl not found"


def test_get_crl_storage_failure_is_500(app, client, root_id, chain):
    crl_pem = make_crl(chain["root_cert"], chain["root_key"], [], number=1)
    crl_id = _register_crl(client, root_id, crl_pem).json()["crl_id"]
    engine = app.state.engine

    def fail_select(conn, cursor, statement, parameters, context, executemany):
        if "FROM certificate_revocation_lists" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(engine, "before_cursor_execute", fail_select)
    try:
        response = _get_crl(client, crl_id, root_id=root_id)
        assert response.status_code == 500
        assert response.json()["detail"] == "CRL registry unavailable"
    finally:
        event.remove(engine, "before_cursor_execute", fail_select)

    # The failure left nothing behind: the same query succeeds afterwards.
    assert _get_crl(client, crl_id, root_id=root_id).status_code == 200
