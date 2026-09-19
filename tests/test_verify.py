import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Evidence
from proof_release.verifiers import (
    VerificationContext,
    get_verifier,
    register_verifier,
    registered_formats,
)


@pytest.fixture()
def app(tmp_path):
    return create_app(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


def _make_evidence(payload: bytes = b"quote-bytes") -> str:
    encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    return f"{encoded}.{digest}"


def _create(client, **overrides):
    body = {"tenant_id": "tenant-a", "workload_id": "workload-1"}
    body.update(overrides)
    return client.post("/v1/challenges", json=body)


def _submit(client, created, **overrides):
    body = {
        "tenant_id": "tenant-a",
        "workload_id": "workload-1",
        "challenge_id": created["challenge_id"],
        "nonce": created["nonce"],
        "evidence_format": "sha256-digest",
        "evidence": _make_evidence(),
    }
    body.update(overrides)
    return client.post("/v1/evidence", json=body)


def _verify(client, evidence_id, created, **overrides):
    body = {
        "tenant_id": "tenant-a",
        "workload_id": "workload-1",
        "nonce": created["nonce"],
        "evidence": _make_evidence(),
    }
    body.update(overrides)
    return client.post(f"/v1/evidence/{evidence_id}/verify", json=body)


def _received(client, **submit_overrides):
    created = _create(client).json()
    submitted = _submit(client, created, **submit_overrides)
    assert submitted.status_code == 201
    return created, submitted.json()["evidence_id"]


def test_verify_success(client):
    created, evidence_id = _received(client)

    response = _verify(client, evidence_id, created)

    assert response.status_code == 200
    data = response.json()
    assert data["evidence_id"] == evidence_id
    assert data["challenge_id"] == created["challenge_id"]
    assert data["status"] == "verified"
    verified_at = datetime.fromisoformat(data["verified_at"])
    assert verified_at.tzinfo is not None
    assert verified_at.utcoffset() == timedelta(0)


def test_verify_rejected_returns_200(client, app):
    payload = base64.urlsafe_b64encode(b"quote-bytes").rstrip(b"=").decode("ascii")
    bad_evidence = f"{payload}.{'0' * 64}"
    created, evidence_id = _received(client, evidence=bad_evidence)

    response = _verify(client, evidence_id, created, evidence=bad_evidence)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["verified_at"]
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "rejected"
        assert record.verified_at is not None


def test_verify_persists_verdict(client, app):
    created, evidence_id = _received(client)
    assert _verify(client, evidence_id, created).status_code == 200

    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "verified"
        assert record.verified_at is not None
        assert record.verified_at.tzinfo is not None
        assert not hasattr(record, "evidence")


def test_verify_unknown_evidence_returns_404(client):
    created = _create(client).json()

    response = _verify(client, "00000000-0000-0000-0000-000000000000", created)

    assert response.status_code == 404


def test_verify_tenant_and_workload_mismatch_return_404(client):
    created, evidence_id = _received(client)

    assert _verify(client, evidence_id, created, tenant_id="tenant-b").status_code == 404
    assert (
        _verify(client, evidence_id, created, workload_id="workload-2").status_code
        == 404
    )


def test_verify_wrong_nonce_returns_401(client):
    created, evidence_id = _received(client)
    wrong = "A" + created["nonce"][1:]

    response = _verify(client, evidence_id, created, nonce=wrong)

    assert response.status_code == 401


def test_verify_evidence_digest_mismatch_returns_422(client):
    created, evidence_id = _received(client)

    response = _verify(
        client, evidence_id, created, evidence=_make_evidence(b"other-bytes")
    )

    assert response.status_code == 422


def test_verify_unsupported_format_returns_422(client):
    created, evidence_id = _received(
        client, evidence_format="tpm-quote", evidence="cXVvdGU="
    )

    response = _verify(client, evidence_id, created, evidence="cXVvdGU=")

    assert response.status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"workload_id": ""},
        {"nonce": ""},
        {"nonce": "with=padding"},
        {"evidence": ""},
        {"evidence": 42},
    ],
)
def test_verify_rejects_invalid_fields(client, overrides):
    created, evidence_id = _received(client)

    response = _verify(client, evidence_id, created, **overrides)

    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "nonce", "evidence"])
def test_verify_requires_all_fields(client, missing):
    created, evidence_id = _received(client)
    body = {
        "tenant_id": "tenant-a",
        "workload_id": "workload-1",
        "nonce": created["nonce"],
        "evidence": _make_evidence(),
    }
    del body[missing]

    response = client.post(f"/v1/evidence/{evidence_id}/verify", json=body)

    assert response.status_code == 422


def test_verify_replay_returns_same_verdict_without_plugin(client):
    calls = []

    class CountingVerifier:
        def verify(self, evidence, context):
            calls.append(evidence)
            return True

    register_verifier("counting-format", CountingVerifier())
    created, evidence_id = _received(
        client, evidence_format="counting-format", evidence="payload"
    )

    first = _verify(client, evidence_id, created, evidence="payload")
    second = _verify(client, evidence_id, created, evidence="payload")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert len(calls) == 1


def test_verify_concurrent_calls_settle_once(app):
    calls = []

    class CountingVerifier:
        def verify(self, evidence, context):
            calls.append(evidence)
            return True

    register_verifier("concurrent-format", CountingVerifier())
    client = TestClient(app)
    created, evidence_id = _received(
        client, evidence_format="concurrent-format", evidence="payload"
    )

    def verify():
        return TestClient(app).post(
            f"/v1/evidence/{evidence_id}/verify",
            json={
                "tenant_id": "tenant-a",
                "workload_id": "workload-1",
                "nonce": created["nonce"],
                "evidence": "payload",
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: verify(), range(16)))

    assert all(r.status_code == 200 for r in responses)
    bodies = {r.json()["verified_at"] for r in responses}
    assert all(r.json()["status"] == "verified" for r in responses)
    assert len(bodies) == 1
    assert len(calls) == 1
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "verified"


def test_verify_plugin_exception_leaves_no_state(client, app):
    attempts = []

    class FlakyVerifier:
        def verify(self, evidence, context):
            attempts.append(evidence)
            if len(attempts) == 1:
                raise RuntimeError("boom")
            return True

    register_verifier("flaky-format", FlakyVerifier())
    created, evidence_id = _received(
        client, evidence_format="flaky-format", evidence="payload"
    )

    failed = _verify(client, evidence_id, created, evidence="payload")
    assert failed.status_code == 500
    assert "payload" not in failed.text

    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verified_at is None

    retried = _verify(client, evidence_id, created, evidence="payload")
    assert retried.status_code == 200
    assert retried.json()["status"] == "verified"


def test_verify_plugin_receives_context(client):
    seen = []

    class ContextVerifier:
        def verify(self, evidence, context):
            seen.append((evidence, context))
            return True

    register_verifier("context-format", ContextVerifier())
    created, evidence_id = _received(
        client, evidence_format="context-format", evidence="payload"
    )

    response = _verify(client, evidence_id, created, evidence="payload")

    assert response.status_code == 200
    evidence, context = seen[0]
    assert evidence == "payload"
    assert isinstance(context, VerificationContext)
    assert context.evidence_id == evidence_id
    assert context.challenge_id == created["challenge_id"]
    assert context.tenant_id == "tenant-a"
    assert context.workload_id == "workload-1"
    assert context.evidence_format == "context-format"


def test_verify_verdict_survives_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    created, evidence_id = _received(client1)
    verified = _verify(client1, evidence_id, created)
    assert verified.status_code == 200
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    replayed = _verify(client2, evidence_id, created)

    assert replayed.status_code == 200
    assert replayed.json() == verified.json()
    with app2.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "verified"
        assert record.verified_at is not None


def test_builtin_verifier_registered():
    assert "sha256-digest" in registered_formats()
    assert get_verifier("sha256-digest") is not None
    assert get_verifier("no-such-format") is None
