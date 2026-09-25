"""Tests for POST /v1/revocations and X.509 revocation rejection."""

from __future__ import annotations

import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from cryptography.hazmat.primitives.serialization import Encoding

from proof_release.app import create_app
from proof_release.db import CertificateRevocation
from proof_release.verifiers import (
    ATTESTED_NONCE_JSON,
    X509_ATTESTED_NONCE_JSON,
)

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


def _full_chain(chain):
    return [chain["leaf_cert"], chain["intermediate_cert"], chain["root_cert"]]


def fingerprint(certificate) -> str:
    der = certificate.public_bytes(Encoding.DER)
    return base64.urlsafe_b64encode(hashlib.sha256(der).digest()).rstrip(b"=").decode(
        "ascii"
    )


def rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


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


def _register(
    client,
    root_id_value,
    fingerprint_value,
    effective_at=None,
    **overrides,
):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id_value,
        "certificate_fingerprint": fingerprint_value,
        "effective_at": effective_at
        if effective_at is not None
        else rfc3339(datetime.now(timezone.utc)),
    }
    body.update(overrides)
    return client.post("/v1/revocations", json=body)


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


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_returns_201_compact_json_with_fields(client, root_id, chain):
    effective = rfc3339(datetime.now(timezone.utc) - timedelta(minutes=1))

    response = _register(client, root_id, fingerprint(chain["leaf_cert"]), effective)

    assert response.status_code == 201
    # Compact JSON: no insignificant whitespace.
    assert b"\n" not in response.content
    assert b": " not in response.content
    data = response.json()
    assert set(data) == {
        "revocation_id",
        "trust_root_id",
        "certificate_fingerprint",
        "effective_at",
    }
    assert data["trust_root_id"] == root_id
    assert data["certificate_fingerprint"] == fingerprint(chain["leaf_cert"])
    assert data["revocation_id"]
    # Echoed back in normalized UTC RFC3339.
    parsed = datetime.fromisoformat(data["effective_at"])
    assert parsed.utcoffset() == timedelta(0)


def test_register_accepts_z_suffix_and_past_time_is_immediately_effective(
    client, root_id, chain
):
    stamped = (
        (datetime.now(timezone.utc) - timedelta(minutes=5))
        .isoformat()
        .replace("+00:00", "Z")
    )
    response = _register(client, root_id, fingerprint(chain["leaf_cert"]), stamped)
    assert response.status_code == 201
    assert response.json()["effective_at"].endswith("+00:00")


def test_future_registration_is_stored_but_not_yet_effective(client, root_id, chain):
    future = rfc3339(datetime.now(timezone.utc) + timedelta(hours=1))
    response = _register(client, root_id, fingerprint(chain["leaf_cert"]), future)
    assert response.status_code == 201

    # A future-dated registration does not reject valid evidence yet.
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, verify_response = _submit_and_verify(client, created, evidence)
    assert verify_response.status_code == 200
    assert verify_response.json()["status"] == "verified"


@pytest.mark.parametrize(
    "missing",
    [
        "tenant_id",
        "workload_id",
        "trust_root_id",
        "certificate_fingerprint",
        "effective_at",
    ],
)
def test_register_requires_all_fields(client, root_id, chain, missing):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "trust_root_id": root_id,
        "certificate_fingerprint": fingerprint(chain["leaf_cert"]),
        "effective_at": rfc3339(datetime.now(timezone.utc)),
    }
    del body[missing]

    response = client.post("/v1/revocations", json=body)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"workload_id": ""},
        {"workload_id": "  "},
        {"workload_id": ["x"]},
        {"trust_root_id": ""},
        {"trust_root_id": "   "},
        {"trust_root_id": "not-a-uuid"},
        {"trust_root_id": 123},
        # Uppercase UUID hex is not the canonical shape.
        {"trust_root_id": "01234567-89AB-CDEF-0123-456789ABCDEF"},
        {"certificate_fingerprint": ""},
        {"certificate_fingerprint": "   "},
        {"certificate_fingerprint": 42},
        # Padded base64url.
        {"certificate_fingerprint": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="},
        # 31 bytes (42 unpadded base64url characters).
        {"certificate_fingerprint": "A" * 42},
        # 33 bytes (44 unpadded base64url characters).
        {"certificate_fingerprint": "A" * 44},
        # Non-alphabet characters.
        {"certificate_fingerprint": "!" * 43},
        {"effective_at": ""},
        {"effective_at": "   "},
        {"effective_at": 1700000000},
        # Naive timestamp has no UTC offset.
        {"effective_at": "2026-01-01T00:00:00"},
        # Non-UTC offset is rejected rather than converted.
        {"effective_at": "2026-01-01T00:00:00+02:00"},
        {"effective_at": "not-a-timestamp"},
    ],
)
def test_register_rejects_invalid_fields_with_422(
    client, root_id, chain, overrides
):
    response = _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        **overrides,
    )

    assert response.status_code == 422


def test_invalid_fields_write_no_state(client, app, root_id, chain):
    response = _register(
        client, root_id, "not-base64url", rfc3339(datetime.now(timezone.utc))
    )
    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.query(CertificateRevocation).count() == 0


def test_unknown_trust_root_returns_404(client, chain):
    unknown = "01234567-89ab-cdef-0123-456789abcdef"
    response = _register(
        client, unknown, fingerprint(chain["leaf_cert"]), rfc3339(datetime.now(timezone.utc))
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "trust root not found"}


def test_trust_root_of_other_tenant_returns_404(client, chain):
    created = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": OTHER_TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    other_root_id = created.json()["root_id"]

    response = _register(
        client,
        other_root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )

    assert response.status_code == 404


def test_trust_root_of_other_workload_returns_404(client, chain):
    created = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": OTHER_WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    other_root_id = created.json()["root_id"]

    response = _register(
        client,
        other_root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
        tenant_id=TENANT,
        workload_id=WORKLOAD,
    )

    assert response.status_code == 404


def test_404_does_not_reveal_existence_and_writes_nothing(client, app, chain):
    # Same certificate is a real root for another scope; referencing it
    # cross-scope is indistinguishable from a wholly unknown id.
    created = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": OTHER_TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    other_root_id = created.json()["root_id"]

    response = _register(
        client,
        other_root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )
    assert response.status_code == 404
    with app.state.session_factory() as session:
        assert session.query(CertificateRevocation).count() == 0


def test_duplicate_fingerprint_same_root_returns_409(client, root_id, chain):
    now = rfc3339(datetime.now(timezone.utc))
    first = _register(client, root_id, fingerprint(chain["leaf_cert"]), now)
    assert first.status_code == 201

    second = _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc) + timedelta(days=1)),
    )
    assert second.status_code == 409


def test_distinct_fingerprints_under_same_root_are_retained_independently(
    client, app, root_id, chain
):
    leaf_response = _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )
    intermediate_response = _register(
        client,
        root_id,
        fingerprint(chain["intermediate_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )
    assert leaf_response.status_code == 201
    assert intermediate_response.status_code == 201
    assert (
        leaf_response.json()["revocation_id"]
        != intermediate_response.json()["revocation_id"]
    )
    with app.state.session_factory() as session:
        assert session.query(CertificateRevocation).count() == 2


def test_same_fingerprint_under_different_root_is_independent(client, chain):
    # The same certificate registered as a root for two distinct scopes
    # may each carry their own revocation.
    first_root = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    ).json()["root_id"]
    second_root = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": OTHER_TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    ).json()["root_id"]

    first = _register(
        client,
        first_root,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )
    second = _register(
        client,
        second_root,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
        tenant_id=OTHER_TENANT,
        workload_id=WORKLOAD,
    )
    assert first.status_code == 201
    assert second.status_code == 201


def test_concurrent_registrations_at_most_one_wins(app, client, root_id, chain):
    stamped = rfc3339(datetime.now(timezone.utc))

    def register_once(_):
        # Each thread needs its own client bound to the same app/engine.
        local_client = TestClient(app)
        return _register(
            local_client,
            root_id,
            fingerprint(chain["leaf_cert"]),
            stamped,
        ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(register_once, range(8)))

    assert statuses.count(201) == 1
    assert statuses.count(409) == 7
    with app.state.session_factory() as session:
        assert session.query(CertificateRevocation).count() == 1


def test_registration_persists_across_restart(tmp_path, chain):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    root_id = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    ).json()["root_id"]
    created = _register(
        client1,
        root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )
    assert created.status_code == 201
    revocation_id = created.json()["revocation_id"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        record = session.get(CertificateRevocation, revocation_id)
        assert record is not None
        assert record.trust_root_id == root_id
        assert len(record.fingerprint) == 32
        assert record.effective_at is not None
    app2.state.engine.dispose()


# ---------------------------------------------------------------------------
# Verification-time rejection
# ---------------------------------------------------------------------------


def test_revoked_leaf_settles_rejected(client, root_id, chain):
    _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc) - timedelta(minutes=1)),
    )
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain), {"m": "abc"}
    )

    evidence_id, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    # Settled atomically and never re-evaluated.
    repeat = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert repeat.status_code == 200
    assert repeat.json()["status"] == "rejected"
    assert repeat.json() == response.json()


def test_revoked_intermediate_settles_rejected(client, root_id, chain):
    _register(
        client,
        root_id,
        fingerprint(chain["intermediate_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_revoked_root_settles_rejected(client, root_id, chain):
    _register(
        client,
        root_id,
        fingerprint(chain["root_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_unrelated_fingerprint_does_not_reject(client, root_id, chain):
    _, unrelated = make_leaf(chain["root_cert"], chain["root_key"], "unrelated-leaf")
    _register(
        client,
        root_id,
        fingerprint(unrelated),
        rfc3339(datetime.now(timezone.utc)),
    )
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_future_revocation_does_not_reject_until_effective(client, root_id, chain):
    _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc) + timedelta(hours=2)),
    )
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_revocation_scoped_to_other_tenant_does_not_reject(client, chain):
    # Root exists in both scopes; the revocation is registered only in the
    # other tenant.
    client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    other_root_id = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": OTHER_TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    ).json()["root_id"]
    _register(
        client,
        other_root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
        tenant_id=OTHER_TENANT,
        workload_id=WORKLOAD,
    )

    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_revocation_rejects_across_restart(tmp_path, chain):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    root_id = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    ).json()["root_id"]
    assert (
        _register(
            client1,
            root_id,
            fingerprint(chain["leaf_cert"]),
            rfc3339(datetime.now(timezone.utc)),
        ).status_code
        == 201
    )
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    created = _challenge(client2)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, response = _submit_and_verify(client2, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    app2.state.engine.dispose()


def test_hmac_format_is_unaffected_by_revocations(
    client, app, monkeypatch, root_id, chain
):
    # The HMAC/attested-nonce conclusion must not change even when a
    # fingerprint registration exists.
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", "unit-test-secret")
    _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )

    created = _challenge(client)
    import hmac as _hmac

    secret = b"unit-test-secret"
    mac_key = _hmac.new(
        secret, f"{TENANT}:{WORKLOAD}".encode("utf-8"), hashlib.sha256
    ).digest()
    nonce = created["nonce"]
    signed = json.dumps(
        {"claims": {}, "nonce": nonce}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    mac = _hmac.new(mac_key, signed, hashlib.sha256).hexdigest()
    evidence = json.dumps(
        {"nonce": nonce, "claims": {}, "mac": mac}, sort_keys=True
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


def test_revocation_check_runs_after_chain_validation_bad_chain_still_rejected(
    client, root_id, chain
):
    # An already-invalid chain is rejected whether or not a revocation
    # exists; the revocation never turns a reject into anything else.
    _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )
    created = _challenge(client)
    # Sign with a wrong key: verifier rejects before revocation is consulted.
    evidence = make_evidence(
        created["nonce"],
        generate_key(),
        _full_chain(chain),
    )
    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_revocation_lookup_failure_returns_500_and_keeps_received(
    app, client, monkeypatch, root_id, chain
):
    # Register a live revocation so the query, had it run, would reject.
    _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )
    created = _challenge(client)
    evidence = make_evidence(
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
            "evidence": evidence,
        },
    )
    evidence_id = submitted.json()["evidence_id"]

    import proof_release.app as app_module

    def boom(*args, **kwargs):
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(app_module, "_x509_chain_revocation", boom)

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
    # No certificate or exception detail leaks.
    assert "BEGIN CERTIFICATE" not in response.text
    assert "storage unavailable" not in response.text

    # Evidence stays received and re-verifies normally after recovery.
    with app.state.session_factory() as session:
        from proof_release.db import Evidence

        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verified_at is None

    monkeypatch.undo()
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
    assert recovered.json()["status"] == "rejected"


def test_registration_committed_before_settlement_is_observed(
    app, client, root_id, chain
):
    # A registration that commits before the evidence settles is observed;
    # sanity for the commit-before-settle ordering (registrations and
    # verification share one serialized write lock on SQLite).
    created = _challenge(client)
    evidence = make_evidence(
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
            "evidence": evidence,
        },
    )
    evidence_id = submitted.json()["evidence_id"]

    reg = _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )
    assert reg.status_code == 201

    response = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_registration_after_settlement_does_not_retroactively_change(
    client, root_id, chain
):
    # The evidence settles verified before any registration exists.
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    evidence_id, first = _submit_and_verify(client, created, evidence)
    assert first.status_code == 200
    assert first.json()["status"] == "verified"

    # A registration committed after settlement must not rewrite the
    # terminal conclusion; a repeat verify returns the stored verdict.
    reg = _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )
    assert reg.status_code == 201

    repeat = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert repeat.status_code == 200
    assert repeat.json() == first.json()
    assert repeat.json()["status"] == "verified"


def test_no_certificate_material_is_persisted(app, client, root_id, chain):
    leaf_pem = pem(chain["leaf_cert"])
    response = _register(
        client,
        root_id,
        fingerprint(chain["leaf_cert"]),
        rfc3339(datetime.now(timezone.utc)),
    )
    assert response.status_code == 201
    assert leaf_pem not in response.text
    assert "BEGIN CERTIFICATE" not in response.text
    with app.state.session_factory() as session:
        rows = session.scalars(select(CertificateRevocation)).all()
        for row in rows:
            blob = json.dumps(
                {c.name: str(getattr(row, c.name)) for c in row.__table__.columns}
            )
            assert "BEGIN CERTIFICATE" not in blob
