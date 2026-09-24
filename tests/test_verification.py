"""Tests for POST /v1/evidence/{evidence_id}/verify."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import String, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from proof_release.app import create_app
from proof_release.db import Challenge, Evidence, UTCDateTime
from proof_release.verifiers import (
    ATTESTED_NONCE_JSON,
    AttestedNonceJSONVerifier,
    VerificationContext,
    VerificationResult,
    Verifier,
    VerifierRegistry,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"
SECRET = "unit-test-secret"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    application = create_app(f"sqlite:///{tmp_path}/test.db")
    return application


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, **overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    body.update(overrides)
    return client.post("/v1/challenges", json=body)


def _mac(nonce: str, claims: dict, tenant_id: str = TENANT, workload_id: str = WORKLOAD) -> str:
    key = hmac.new(
        SECRET.encode("utf-8"),
        f"{tenant_id}:{workload_id}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _attested_evidence(nonce: str, claims: dict | None = None, **fields) -> str:
    document = {"nonce": nonce, "claims": claims or {}, "mac": _mac(nonce, claims or {})}
    document.update(fields)
    return json.dumps(document)


def _submit(client, created, evidence, evidence_format=ATTESTED_NONCE_JSON):
    return client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": evidence_format,
            "evidence": evidence,
        },
    )


def _verify(client, evidence_id, created, evidence, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
    }
    body.update(overrides)
    return client.post(f"/v1/evidence/{evidence_id}/verify", json=body)


@pytest.fixture()
def submitted(client):
    created = _create(client).json()
    evidence = _attested_evidence(created["nonce"], {"measurement": "abc"})
    response = _submit(client, created, evidence)
    assert response.status_code == 201
    return created, evidence, response.json()["evidence_id"]


def test_verify_valid_evidence_returns_verified(submitted, client):
    created, evidence, evidence_id = submitted

    response = _verify(client, evidence_id, created, evidence)

    assert response.status_code == 200
    data = response.json()
    assert data["evidence_id"] == evidence_id
    assert data["challenge_id"] == created["challenge_id"]
    assert data["status"] == "verified"
    verified_at = datetime.fromisoformat(data["verified_at"])
    assert verified_at.utcoffset() == timedelta(0)


def test_verify_rejected_evidence_still_200_with_timestamp(client):
    created = _create(client).json()
    # Structurally valid document with a wrong MAC: the evidence is accepted
    # for receipt (only its digest is stored) and rejected at verification.
    document = {
        "nonce": created["nonce"],
        "claims": {},
        "mac": "0" * 64,
    }
    evidence = json.dumps(document)
    evidence_id = _submit(client, created, evidence).json()["evidence_id"]

    response = _verify(client, evidence_id, created, evidence)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["evidence_id"] == evidence_id
    assert data["challenge_id"] == created["challenge_id"]
    verified_at = datetime.fromisoformat(data["verified_at"])
    assert verified_at.utcoffset() == timedelta(0)


def test_verify_wrong_nonce_in_body_returns_401(submitted, client):
    created, evidence, evidence_id = submitted
    wrong = ("B" if created["nonce"][0] != "B" else "C") + created["nonce"][1:]

    response = _verify(client, evidence_id, created, evidence, nonce=wrong)

    assert response.status_code == 401


def test_verify_unknown_evidence_returns_404(submitted, client):
    created, evidence, _ = submitted

    response = _verify(
        client,
        "00000000-0000-0000-0000-000000000000",
        created,
        evidence,
    )
    assert response.status_code == 404


def test_verify_tenant_and_workload_mismatch_return_404(submitted, client):
    created, evidence, evidence_id = submitted

    assert (
        _verify(client, evidence_id, created, evidence, tenant_id="tenant-b").status_code
        == 404
    )
    assert (
        _verify(
            client, evidence_id, created, evidence, workload_id="workload-2"
        ).status_code
        == 404
    )


def test_verify_digest_mismatch_returns_422(submitted, client):
    created, evidence, evidence_id = submitted

    response = _verify(client, evidence_id, created, evidence + " ")

    assert response.status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"workload_id": ""},
        {"nonce": ""},
        {"nonce": "not base64!!!"},
        {"nonce": "with=padding"},
        {"evidence": ""},
        {"evidence": 42},
        {"evidence": None},
    ],
)
def test_verify_rejects_invalid_fields(submitted, client, overrides):
    created, evidence, evidence_id = submitted
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
    }
    body.update(overrides)

    response = client.post(f"/v1/evidence/{evidence_id}/verify", json=body)

    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["tenant_id", "workload_id", "nonce", "evidence"])
def test_verify_requires_all_fields(submitted, client, missing):
    created, evidence, evidence_id = submitted
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "nonce": created["nonce"],
        "evidence": evidence,
    }
    del body[missing]

    response = client.post(f"/v1/evidence/{evidence_id}/verify", json=body)

    assert response.status_code == 422


def test_verify_unsupported_format_returns_422(client):
    created = _create(client).json()
    evidence = "opaque-blob"
    submitted_response = _submit(client, created, evidence, evidence_format="mystery")
    assert submitted_response.status_code == 201
    evidence_id = submitted_response.json()["evidence_id"]

    response = _verify(client, evidence_id, created, evidence)

    assert response.status_code == 422
    assert response.json()["detail"] == "unsupported evidence format"


def test_first_verification_is_settled_and_repeated(submitted, client, app):
    created, evidence, evidence_id = submitted

    first = _verify(client, evidence_id, created, evidence)
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["status"] == "verified"

    second = _verify(client, evidence_id, created, evidence)
    assert second.status_code == 200
    assert second.json() == first_data

    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "verified"
        assert record.verified_at is not None
        assert record.verification_result == "accepted"


def test_repeated_verify_with_different_evidence_returns_stored_conclusion(
    submitted, client
):
    created, evidence, evidence_id = submitted
    assert _verify(client, evidence_id, created, evidence).json()["status"] == "verified"

    # A second call with a different (digest-mismatching) body still returns
    # the settled conclusion — it is never re-evaluated.
    tampered = evidence + "x"
    response = _verify(client, evidence_id, created, tampered)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_repeated_verify_with_wrong_nonce_still_401(submitted, client):
    created, evidence, evidence_id = submitted
    assert _verify(client, evidence_id, created, evidence).status_code == 200
    wrong = ("B" if created["nonce"][0] != "B" else "C") + created["nonce"][1:]

    response = _verify(client, evidence_id, created, evidence, nonce=wrong)

    assert response.status_code == 401


def test_settlement_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    created = _create(client1).json()
    evidence = _attested_evidence(created["nonce"])
    evidence_id = _submit(client1, created, evidence).json()["evidence_id"]
    verified = _verify(client1, evidence_id, created, evidence)
    assert verified.status_code == 200
    assert verified.json()["status"] == "verified"
    first_body = verified.json()
    app1.state.engine.dispose()

    app2 = create_app(url)
    client2 = TestClient(app2)
    response = _verify(client2, evidence_id, created, evidence)
    assert response.status_code == 200
    assert response.json() == first_body
    with app2.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "verified"
        assert record.verified_at is not None
        assert record.tenant_id == TENANT
        assert record.workload_id == WORKLOAD
        assert record.challenge_id == created["challenge_id"]


def test_raw_evidence_is_never_persisted_or_returned(submitted, client, app, tmp_path):
    created, evidence, evidence_id = submitted
    response = _verify(client, evidence_id, created, evidence)
    assert response.status_code == 200
    assert evidence not in response.text

    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        columns = {
            c.name: getattr(record, c.name)
            for c in record.__table__.columns
        }
    for name, value in columns.items():
        assert evidence not in str(value), f"raw evidence leaked into column {name}"


def test_concurrent_verify_settles_once_with_consistent_status(app, submitted):
    created, evidence, evidence_id = submitted

    def verify():
        return TestClient(app).post(
            f"/v1/evidence/{evidence_id}/verify",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "nonce": created["nonce"],
                "evidence": evidence,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: verify(), range(16)))

    statuses = [r.json()["status"] for r in responses]
    assert all(r.status_code == 200 for r in responses)
    assert set(statuses) == {"verified"}
    timestamps = {r.json()["verified_at"] for r in responses}
    assert len(timestamps) == 1
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "verified"


class AlternatingVerifier(Verifier):
    """Verifier whose verdict flips on every invocation.

    Used to prove that even a non-deterministic plugin cannot produce
    contradictory concurrent states: the first settlement wins and every
    other caller observes it.
    """

    format_name = "alternating"

    def __init__(self):
        self.calls = 0

    def verify(self, context: VerificationContext) -> VerificationResult:
        accepted = self.calls % 2 == 0
        self.calls += 1
        return VerificationResult(accepted=accepted)


def test_concurrent_verify_with_nondeterministic_plugin_settles_once(tmp_path):
    verifier = AlternatingVerifier()
    registry = VerifierRegistry()
    registry.register(verifier)
    application = create_app(
        f"sqlite:///{tmp_path}/alternating.db", verifier_registry=registry
    )
    client = TestClient(application)
    created = _create(client).json()
    evidence = "same-bytes-for-every-caller"
    evidence_id = _submit(
        client, created, evidence, evidence_format="alternating"
    ).json()["evidence_id"]

    def verify():
        return TestClient(application).post(
            f"/v1/evidence/{evidence_id}/verify",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "nonce": created["nonce"],
                "evidence": evidence,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: verify(), range(16)))

    assert all(r.status_code == 200 for r in responses)
    statuses = {r.json()["status"] for r in responses}
    assert len(statuses) == 1
    timestamps = {r.json()["verified_at"] for r in responses}
    assert len(timestamps) == 1
    # Only the caller that performed the first settlement ran the plugin.
    assert verifier.calls == 1
    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status in ("verified", "rejected")
        assert record.status == next(iter(statuses))


class CountingVerifier(Verifier):
    format_name = "counting"

    def __init__(self):
        self.calls = 0

    def verify(self, context: VerificationContext) -> VerificationResult:
        self.calls += 1
        return VerificationResult(accepted=True)


def test_plugin_is_invoked_only_once_across_repeats(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    verifier = CountingVerifier()
    registry = VerifierRegistry()
    registry.register(verifier)
    application = create_app(
        f"sqlite:///{tmp_path}/count.db", verifier_registry=registry
    )
    client = TestClient(application)
    created = _create(client).json()
    evidence = "blob"
    evidence_id = _submit(client, created, evidence, evidence_format="counting").json()[
        "evidence_id"
    ]

    for _ in range(3):
        response = _verify(client, evidence_id, created, evidence)
        assert response.status_code == 200
        assert response.json()["status"] == "verified"

    assert verifier.calls == 1


class RaisingVerifier(Verifier):
    format_name = "raising"

    def __init__(self):
        self.calls = 0

    def verify(self, context: VerificationContext) -> VerificationResult:
        self.calls += 1
        # A faulty plugin must not be able to persist raw evidence by
        # raising; the service rolls back and maps this to 500.
        raise RuntimeError("internal verifier fault")


def test_plugin_exception_leaves_no_half_state_and_can_retry(tmp_path):
    verifier = RaisingVerifier()
    registry = VerifierRegistry()
    registry.register(verifier)
    application = create_app(
        f"sqlite:///{tmp_path}/raising.db", verifier_registry=registry
    )
    client = TestClient(application)
    created = _create(client).json()
    evidence = "blob"
    evidence_id = _submit(client, created, evidence, evidence_format="raising").json()[
        "evidence_id"
    ]

    response = _verify(client, evidence_id, created, evidence)
    assert response.status_code == 500
    assert "blob" not in response.text

    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verified_at is None
        assert record.verification_result is None

    # Replace the faulty verifier with a working one and retry; settlement
    # is still possible exactly once.
    replacement = CountingVerifier()
    replacement.format_name = "raising"
    registry.register(replacement)
    response = _verify(client, evidence_id, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"
    assert replacement.calls == 1
    # The raising verifier was only tried during the failed request.
    assert verifier.calls == 1


def test_plugin_receives_challenge_and_identity_context(tmp_path):
    seen = {}

    class CapturingVerifier(Verifier):
        format_name = "capturing"

        def verify(self, context):
            seen["evidence"] = context.evidence
            seen["tenant_id"] = context.tenant_id
            seen["workload_id"] = context.workload_id
            seen["challenge_id"] = context.challenge.challenge_id
            seen["nonce_digest"] = context.challenge.nonce_digest
            return VerificationResult(accepted=True)

    registry = VerifierRegistry()
    capturing = CapturingVerifier()
    registry.register(capturing)
    application = create_app(
        f"sqlite:///{tmp_path}/capture.db", verifier_registry=registry
    )
    client = TestClient(application)
    created = _create(client).json()
    evidence = "context-blob"
    evidence_id = _submit(
        client, created, evidence, evidence_format="capturing"
    ).json()["evidence_id"]

    response = _verify(client, evidence_id, created, evidence)
    assert response.status_code == 200
    assert seen["evidence"] == evidence
    assert seen["tenant_id"] == TENANT
    assert seen["workload_id"] == WORKLOAD
    assert seen["challenge_id"] == created["challenge_id"]
    assert seen["nonce_digest"] == hashlib.sha256(
        created["nonce"].encode("ascii")
    ).hexdigest()


def test_builtin_verifier_rejects_nonce_bound_to_other_challenge(client):
    created = _create(client).json()
    other = _create(client).json()
    # Evidence submitted under the first challenge (its submission nonce is
    # correct) but the attested document binds the other challenge's nonce.
    evidence = _attested_evidence(other["nonce"])
    submitted_response = _submit(client, created, evidence)
    assert submitted_response.status_code == 201
    evidence_id = submitted_response.json()["evidence_id"]

    # Presenting with the other challenge's nonce fails the bound-challenge
    # nonce check before the plugin runs.
    response = _verify(client, evidence_id, created, evidence, nonce=other["nonce"])
    assert response.status_code == 401

    # Presenting with the bound challenge's nonce reaches the plugin, which
    # rejects because the attested nonce does not match that challenge.
    response = _verify(client, evidence_id, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_builtin_verifier_rejects_wrong_mac_key(client):
    created = _create(client).json()
    # Compute the MAC with a secret the service does not hold.
    foreign_mac = _mac_with_secret(created["nonce"], {}, "some-other-secret")
    evidence = json.dumps(
        {"nonce": created["nonce"], "claims": {}, "mac": foreign_mac}
    )
    evidence_id = _submit(client, created, evidence).json()["evidence_id"]

    response = _verify(client, evidence_id, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def _mac_with_secret(nonce, claims, secret):
    key = hmac.new(
        secret.encode("utf-8"),
        f"{TENANT}:{WORKLOAD}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


@pytest.mark.parametrize(
    "document",
    [
        "not json",
        "[]",
        "42",
        json.dumps({"claims": {}, "mac": "a"}),  # missing nonce
        json.dumps({"nonce": "abc", "claims": {}, "mac": "zz"}),  # non-hex mac
        json.dumps({"nonce": "ab=", "claims": {}, "mac": "a"}),  # padded nonce
        json.dumps({"nonce": "abc", "claims": [], "mac": "a"}),  # bad claims
    ],
)
def test_builtin_verifier_malformed_documents_reject(client, document):
    created = _create(client).json()
    # Digest mismatch would 422 for tampered bodies, so submit exactly what
    # we will verify.
    evidence_id = _submit(client, created, document).json()["evidence_id"]

    response = _verify(client, evidence_id, created, document)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


LEAKED_SECRET = "sk-live-SECRET-key-material-0123456789"


def _build_app_with_verifier(tmp_path, verifier, db_name="plugin.db"):
    registry = VerifierRegistry()
    registry.register(verifier)
    application = create_app(
        f"sqlite:///{tmp_path}/{db_name}", verifier_registry=registry
    )
    return application, TestClient(application)


def _submit_and_verify(application, client, evidence, evidence_format, **verify_overrides):
    created = _create(client).json()
    evidence_id = _submit(
        client, created, evidence, evidence_format=evidence_format
    ).json()["evidence_id"]
    response = _verify(client, evidence_id, created, evidence, **verify_overrides)
    return created, evidence_id, response


def _assert_no_leak_in_record(app, evidence_id, *needles):
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        columns = {c.name: getattr(record, c.name) for c in record.__table__.columns}
    for name, value in columns.items():
        for needle in needles:
            assert needle not in str(value), f"{needle!r} leaked into column {name}"


class LeakingVerifier(Verifier):
    """Malicious plugin that stuffs raw evidence and secrets into its result."""

    format_name = "leaking"

    def __init__(self, accepted=True):
        self.accepted = accepted

    def verify(self, context: VerificationContext) -> VerificationResult:
        return VerificationResult(
            accepted=self.accepted,
            detail=(
                f"evidence={context.evidence}; secret={LEAKED_SECRET}; "
                f"nonce_digest={context.challenge.nonce_digest}"
            ),
        )


def test_malicious_plugin_result_text_is_never_persisted_or_returned(
    tmp_path, caplog
):
    application, client = _build_app_with_verifier(tmp_path, LeakingVerifier())
    evidence = "raw-evidence-blob-with-private-context"
    with caplog.at_level(logging.DEBUG):
        created, evidence_id, response = _submit_and_verify(
            application, client, evidence, "leaking"
        )

    assert response.status_code == 200
    assert response.json()["status"] == "verified"
    assert evidence not in response.text
    assert LEAKED_SECRET not in response.text
    assert evidence not in caplog.text
    assert LEAKED_SECRET not in caplog.text

    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        # Only the fixed, service-defined result code is stored.
        assert record.verification_result == "accepted"
    _assert_no_leak_in_record(application, evidence_id, evidence, LEAKED_SECRET)

    # The settled conclusion stays clean on repeat calls.
    repeat = _verify(client, evidence_id, created, evidence)
    assert repeat.status_code == 200
    assert repeat.json() == response.json()
    assert evidence not in repeat.text
    assert LEAKED_SECRET not in repeat.text


def test_malicious_plugin_rejection_text_is_dropped(tmp_path):
    application, client = _build_app_with_verifier(
        tmp_path, LeakingVerifier(accepted=False)
    )
    evidence = "rejected-evidence-with-secrets"
    _, evidence_id, response = _submit_and_verify(
        application, client, evidence, "leaking"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert evidence not in response.text
    assert LEAKED_SECRET not in response.text
    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.verification_result == "rejected"
    _assert_no_leak_in_record(application, evidence_id, evidence, LEAKED_SECRET)


class DuckTypingVerifier(Verifier):
    """Returns a foreign result object carrying evidence-laden attributes."""

    format_name = "duck-typing"

    def verify(self, context: VerificationContext):
        class ForeignResult:
            accepted = True
            detail = f"{context.evidence}|{LEAKED_SECRET}"
            evidence = context.evidence

        return ForeignResult()


def test_non_model_result_object_is_reduced_to_verdict_only(tmp_path):
    application, client = _build_app_with_verifier(tmp_path, DuckTypingVerifier())
    evidence = "duck-typed-evidence"
    _, evidence_id, response = _submit_and_verify(
        application, client, evidence, "duck-typing"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "verified"
    assert evidence not in response.text
    assert LEAKED_SECRET not in response.text
    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.verification_result == "accepted"
    _assert_no_leak_in_record(application, evidence_id, evidence, LEAKED_SECRET)


class SecretRaisingVerifier(Verifier):
    """Faulty plugin whose exception message embeds evidence and secrets."""

    format_name = "secret-raising"

    def verify(self, context: VerificationContext) -> VerificationResult:
        raise RuntimeError(
            f"boom on {context.evidence} using {LEAKED_SECRET}"
        )


def test_plugin_exception_with_sensitive_message_leaks_nothing(tmp_path, caplog):
    application, client = _build_app_with_verifier(tmp_path, SecretRaisingVerifier())
    evidence = "exception-path-evidence"
    with caplog.at_level(logging.DEBUG):
        _, evidence_id, response = _submit_and_verify(
            application, client, evidence, "secret-raising"
        )

    assert response.status_code == 500
    assert evidence not in response.text
    assert LEAKED_SECRET not in response.text
    assert evidence not in caplog.text
    assert LEAKED_SECRET not in caplog.text

    # The exception rolled back: the record is still received and unsettled.
    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verified_at is None
        assert record.verification_result is None
    _assert_no_leak_in_record(application, evidence_id, evidence, LEAKED_SECRET)


def test_concurrent_verify_with_leaking_plugin_settles_once_cleanly(tmp_path):
    verifier = LeakingVerifier()
    application, client = _build_app_with_verifier(tmp_path, verifier)
    created = _create(client).json()
    evidence = "concurrent-leak-attempt"
    evidence_id = _submit(client, created, evidence, evidence_format="leaking").json()[
        "evidence_id"
    ]

    def verify():
        return TestClient(application).post(
            f"/v1/evidence/{evidence_id}/verify",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "nonce": created["nonce"],
                "evidence": evidence,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: verify(), range(16)))

    assert all(r.status_code == 200 for r in responses)
    assert {r.json()["status"] for r in responses} == {"verified"}
    assert len({r.json()["verified_at"] for r in responses}) == 1
    for r in responses:
        assert evidence not in r.text
        assert LEAKED_SECRET not in r.text
    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "verified"
        assert record.verification_result == "accepted"
    _assert_no_leak_in_record(application, evidence_id, evidence, LEAKED_SECRET)


class LegacyBase(DeclarativeBase):
    """Schema of the previous release: free-form verification_detail, no
    verification_result column."""


class LegacyChallenge(LegacyBase):
    __tablename__ = "challenges"

    challenge_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    nonce_digest: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    issued_at: Mapped[datetime] = mapped_column(UTCDateTime())
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime())
    consumed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class LegacyEvidence(LegacyBase):
    __tablename__ = "evidence"

    evidence_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    challenge_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(256), index=True)
    workload_id: Mapped[str] = mapped_column(String(256))
    evidence_format: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="received")
    received_at: Mapped[datetime] = mapped_column(UTCDateTime())
    evidence_sha256: Mapped[str] = mapped_column(String(64))
    verified_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    verification_detail: Mapped[str | None] = mapped_column(
        String(256), nullable=True
    )


def _legacy_challenge(challenge_id, nonce, **overrides):
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    values = dict(
        challenge_id=challenge_id,
        tenant_id=TENANT,
        workload_id=WORKLOAD,
        nonce_digest=hashlib.sha256(nonce.encode("ascii")).hexdigest(),
        status="consumed",
        issued_at=now - timedelta(minutes=5),
        expires_at=now + timedelta(minutes=5),
        consumed_at=now - timedelta(minutes=4),
    )
    values.update(overrides)
    return LegacyChallenge(**values)


def test_legacy_database_migrates_without_leaking_stored_detail(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    db_path = tmp_path / "legacy.db"
    url = f"sqlite:///{db_path}"

    settled_nonce = "settled-challenge-nonce"
    pending_nonce = "pending-challenge-nonce"
    legacy_detail = f"legacy note with evidence and {LEAKED_SECRET}"
    settled_verified_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    fresh_evidence = _attested_evidence(pending_nonce, {"measurement": "xyz"})

    engine = create_engine(url)
    LegacyBase.metadata.create_all(engine)
    legacy_sessions = sessionmaker(bind=engine)
    with legacy_sessions() as session:
        session.add(_legacy_challenge("challenge-settled", settled_nonce))
        session.add(_legacy_challenge("challenge-pending", pending_nonce))
        session.add(
            LegacyEvidence(
                evidence_id="evidence-settled",
                challenge_id="challenge-settled",
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                evidence_format=ATTESTED_NONCE_JSON,
                status="verified",
                received_at=settled_verified_at - timedelta(minutes=3),
                evidence_sha256=hashlib.sha256(b"old-evidence").hexdigest(),
                verified_at=settled_verified_at,
                verification_detail=legacy_detail,
            )
        )
        session.add(
            LegacyEvidence(
                evidence_id="evidence-pending",
                challenge_id="challenge-pending",
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                evidence_format=ATTESTED_NONCE_JSON,
                status="received",
                received_at=settled_verified_at - timedelta(minutes=2),
                evidence_sha256=hashlib.sha256(
                    fresh_evidence.encode("utf-8")
                ).hexdigest(),
                verified_at=None,
                verification_detail=None,
            )
        )
        session.commit()
    engine.dispose()

    # The current service boots on the legacy database without failure.
    application = create_app(url)
    client = TestClient(application)

    # The migration scrubbed the legacy free-form detail and added the
    # fixed-code column.
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT verification_detail FROM evidence")).fetchall()
        assert rows and all(row[0] is None for row in rows)
        columns = {
            row[1] for row in conn.execute(text("PRAGMA table_info(evidence)"))
        }
        assert "verification_result" in columns
    engine.dispose()

    # The previously settled record still returns its stored conclusion,
    # and the legacy detail never appears in the response.
    response = client.post(
        "/v1/evidence/evidence-settled/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": settled_nonce,
            "evidence": "whatever-was-settled",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "verified"
    assert data["verified_at"] == settled_verified_at.isoformat()
    assert legacy_detail not in response.text
    assert LEAKED_SECRET not in response.text

    # A record still in received state verifies normally after migration.
    response = client.post(
        "/v1/evidence/evidence-pending/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": pending_nonce,
            "evidence": fresh_evidence,
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "verified"
    with application.state.session_factory() as session:
        record = session.get(Evidence, "evidence-pending")
        assert record.verification_result == "accepted"
    application.state.engine.dispose()

    # Migration is idempotent: a second boot on the migrated file works.
    application2 = create_app(url)
    client2 = TestClient(application2)
    response = client2.post(
        "/v1/evidence/evidence-pending/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": pending_nonce,
            "evidence": fresh_evidence,
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "verified"
    application2.state.engine.dispose()
