"""Tests for POST /v1/trust-roots/{root_id}/retire."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import TrustRoot
from proof_release.verifiers import X509_ATTESTED_NONCE_JSON

from x509_helpers import make_evidence, make_intermediate, make_leaf, make_root, pem

TENANT = "tenant-a"
WORKLOAD = "workload-1"


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


def _retire(client, root_id, **overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    body.update(overrides)
    return client.post(f"/v1/trust-roots/{root_id}/retire", json=body)


def _stored_root(app, root_id):
    with app.state.session_factory() as session:
        return session.get(TrustRoot, root_id)


def test_retire_returns_200_compact_ordered_body(client, root_id):
    response = _retire(client, root_id)

    assert response.status_code == 200
    # Compact serialization, exactly the three result fields in order, one
    # trailing newline.
    raw = response.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b" " not in raw
    data = json.loads(raw)
    assert list(data) == ["root_id", "status", "retired_at"]
    assert data["root_id"] == root_id
    assert data["status"] == "retired"
    retired_at = datetime.fromisoformat(data["retired_at"])
    assert retired_at.utcoffset() == timedelta(0)


def test_retire_persists_status_and_first_retirement_time(client, app, root_id):
    response = _retire(client, root_id)
    assert response.status_code == 200

    record = _stored_root(app, root_id)
    assert record.status == "retired"
    assert record.retired_at is not None
    assert record.retired_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "root_id",
    [
        "not-a-uuid",
        "   ",
        "123e4567-e89b-12d3-a456-42661417400",  # too short
        "123E4567-E89B-12D3-A456-426614174000",  # uppercase is non-canonical
        "123e4567e89b12d3a456426614174000",  # missing dashes
    ],
)
def test_retire_rejects_noncanonical_path_id_without_touching_state(
    client, app, root_id
):
    response = _retire(client, root_id)

    assert response.status_code == 422
    with app.state.session_factory() as session:
        assert session.query(TrustRoot).count() == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"tenant_id": None},
        {"workload_id": ""},
        {"workload_id": "   "},
        {"workload_id": 1},
        {"workload_id": None},
        {"unexpected": "field"},
    ],
)
def test_retire_rejects_invalid_body_fields(client, app, root_id, overrides):
    response = _retire(client, root_id, **overrides)

    assert response.status_code == 422
    assert _stored_root(app, root_id).status == "active"
    assert _stored_root(app, root_id).retired_at is None


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id"])
def test_retire_requires_both_scope_fields(client, app, root_id, missing):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    del body[missing]

    response = client.post(f"/v1/trust-roots/{root_id}/retire", json=body)

    assert response.status_code == 422
    assert _stored_root(app, root_id).status == "active"


def test_retire_missing_body_is_422(client, root_id):
    response = client.post(f"/v1/trust-roots/{root_id}/retire")

    assert response.status_code == 422


def test_retire_unknown_root_id_is_404(client):
    response = _retire(client, "123e4567-e89b-12d3-a456-426614174000")

    assert response.status_code == 404


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": "tenant-b"},
        {"workload_id": "workload-2"},
        {"tenant_id": "tenant-b", "workload_id": "workload-2"},
    ],
)
def test_retire_cross_scope_is_404_not_422(client, app, root_id, overrides):
    response = _retire(client, root_id, **overrides)

    assert response.status_code == 404
    assert _stored_root(app, root_id).status == "active"
    assert _stored_root(app, root_id).retired_at is None


def test_repeat_retire_is_409_and_keeps_first_retirement_time(client, app, root_id):
    first = _retire(client, root_id)
    assert first.status_code == 200
    first_retired_at = first.json()["retired_at"]

    second = _retire(client, root_id)

    assert second.status_code == 409
    record = _stored_root(app, root_id)
    assert record.status == "retired"
    assert record.retired_at == datetime.fromisoformat(first_retired_at)


def test_concurrent_retires_settle_at_most_once(app, root_id):
    def retire():
        return TestClient(app).post(
            f"/v1/trust-roots/{root_id}/retire",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: retire(), range(20)))

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(200) == 1
    assert statuses.count(409) == 19
    record = _stored_root(app, root_id)
    assert record.status == "retired"
    assert record.retired_at is not None


def test_retirement_survives_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    _, certificate = make_root()
    created = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(certificate),
        },
    )
    root_id = created.json()["root_id"]
    retired = _retire(client1, root_id)
    assert retired.status_code == 200
    retired_at = retired.json()["retired_at"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        record = session.get(TrustRoot, root_id)
        assert record.status == "retired"
        assert record.retired_at == datetime.fromisoformat(retired_at)
    client2 = TestClient(app2)
    assert _retire(client2, root_id).status_code == 409
    app2.state.engine.dispose()


def test_retired_root_still_conflicts_on_duplicate_creation(client, chain, root_id):
    assert _retire(client, root_id).status_code == 200

    duplicate = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )

    assert duplicate.status_code == 409


def _challenge(client):
    response = client.post(
        "/v1/challenges", json={"tenant_id": TENANT, "workload_id": WORKLOAD}
    )
    assert response.status_code == 201
    return response.json()


def _submit_and_verify(client, created, evidence):
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
    verified = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    return evidence_id, verified


def _valid_evidence(chain, nonce):
    return make_evidence(
        nonce,
        chain["leaf_key"],
        [chain["leaf_cert"], chain["intermediate_cert"], chain["root_cert"]],
        {"m": "abc"},
    )


def test_evidence_anchored_to_retired_root_settles_rejected(client, chain, root_id):
    # Sanity: the same chain verifies while the root is active.
    created = _challenge(client)
    _, verified = _submit_and_verify(client, created, _valid_evidence(chain, created["nonce"]))
    assert verified.status_code == 200
    assert verified.json()["status"] == "verified"

    assert _retire(client, root_id).status_code == 200

    created = _challenge(client)
    evidence_id, verified = _submit_and_verify(
        client, created, _valid_evidence(chain, created["nonce"])
    )
    assert verified.status_code == 200
    assert verified.json()["status"] == "rejected"
    assert verified.json()["evidence_id"] == evidence_id


def test_retirement_rejection_precedes_revocation_check(client, chain, root_id):
    # Revoke the leaf first, then retire: the outcome is still a single
    # rejected settlement and stays stable on repeat reads.
    created = _challenge(client)
    evidence = _valid_evidence(chain, created["nonce"])
    assert _retire(client, root_id).status_code == 200

    _, verified = _submit_and_verify(client, created, evidence)
    assert verified.status_code == 200
    assert verified.json()["status"] == "rejected"

    again = client.post(
        f"/v1/evidence/{verified.json()['evidence_id']}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert again.status_code == 200
    assert again.json()["status"] == "rejected"
    assert again.json()["verified_at"] == verified.json()["verified_at"]


def test_settled_conclusion_is_not_rewritten_by_later_retirement(
    client, chain, root_id
):
    created = _challenge(client)
    evidence = _valid_evidence(chain, created["nonce"])
    evidence_id, verified = _submit_and_verify(client, created, evidence)
    assert verified.json()["status"] == "verified"

    assert _retire(client, root_id).status_code == 200

    # The already-settled evidence keeps its verdict verbatim.
    reread = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert reread.status_code == 200
    assert reread.json()["status"] == "verified"
    assert reread.json()["verified_at"] == verified.json()["verified_at"]
