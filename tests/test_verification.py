"""Tests for POST /v1/evidence/{evidence_id}/verify."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from proof_release.app import create_app
from proof_release.db import Challenge, Evidence, VerificationResultCode
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
    wrong = "B" + created["nonce"][1:]

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
        # Only a service-defined code is stored, derived from the verdict.
        assert (
            record.verification_result_code
            == VerificationResultCode.ACCEPTED.value
        )


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
    wrong = "C" + created["nonce"][1:]

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
        assert record.verification_result_code is None

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


# A distinctive token a malicious plugin might try to exfiltrate: it is both
# the raw evidence and resembles private key material.
SMUGGLED_SECRET = "SUPER-SECRET-KEY-MATERIAL-0xDEADBEEF"
SMUGGLED_EVIDENCE = f'{{"quote":"attested","secret":"{SMUGGLED_SECRET}"}}'


class _SmuggleViaKwargsVerifier(Verifier):
    """Tries to attach a free-text detail to the result object."""

    format_name = "smuggle-kwargs"

    def verify(self, context: VerificationContext) -> VerificationResult:
        return VerificationResult(accepted=True, detail=context.evidence + SMUGGLED_SECRET)  # type: ignore[call-arg]


@dataclass(frozen=True)
class SmuggledResult(VerificationResult):
    """A result subclass carrying an extra smuggled text attribute."""

    payload: str = ""


class _SmuggleViaAttributeVerifier(Verifier):
    format_name = "smuggle-attr"

    def verify(self, context: VerificationContext) -> VerificationResult:
        return SmuggledResult(True, context.evidence + SMUGGLED_SECRET)


class _SmuggleViaForeignObjectVerifier(Verifier):
    format_name = "smuggle-object"

    def verify(self, context: VerificationContext):
        # Not a VerificationResult at all, and carrying no usable verdict.
        return object()


class _NonBoolAcceptedVerifier(Verifier):
    format_name = "nonbool"

    def verify(self, context: VerificationContext) -> VerificationResult:
        # A truthy, non-boolean verdict must not be coerced into "accepted".
        return VerificationResult(accepted=context.evidence)  # type: ignore[arg-type]


class _SecretLeakingException(Exception):
    pass


class _RaiseWithSecretVerifier(Verifier):
    format_name = "raise-secret"

    def verify(self, context: VerificationContext) -> VerificationResult:
        raise _SecretLeakingException(
            f"failure while processing {context.evidence} key={SMUGGLED_SECRET}"
        )


def _smuggle_setup(tmp_path, fmt, verifier):
    registry = VerifierRegistry()
    registry.register(verifier)
    application = create_app(
        f"sqlite:///{tmp_path}/smuggle.db", verifier_registry=registry
    )
    client = TestClient(application)
    created = _create(client).json()
    evidence_id = _submit(
        client, created, SMUGGLED_EVIDENCE, evidence_format=fmt
    ).json()["evidence_id"]
    return application, client, created, evidence_id


def _assert_secret_absent_everywhere(application, response, evidence_id, caplog):
    # Not in the API response.
    assert SMUGGLED_SECRET not in response.text
    assert SMUGGLED_EVIDENCE not in response.text
    # Not in any mapped column.
    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        for column in record.__table__.columns:
            assert SMUGGLED_SECRET not in str(getattr(record, column.name))
    # Not anywhere in the database file on disk.
    application.state.engine.dispose()
    db_path = str(application.state.engine.url.database)
    with open(db_path, "rb") as handle:
        raw = handle.read()
    assert SMUGGLED_SECRET.encode() not in raw
    assert SMUGGLED_EVIDENCE.encode() not in raw
    # Not in logs.
    assert SMUGGLED_SECRET not in caplog.text
    assert SMUGGLED_EVIDENCE not in caplog.text


def test_plugin_detail_kwarg_is_never_accepted(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="proof_release")
    application, client, created, evidence_id = _smuggle_setup(
        tmp_path, "smuggle-kwargs", _SmuggleViaKwargsVerifier()
    )

    response = _verify(client, evidence_id, created, SMUGGLED_EVIDENCE)

    # The smuggled constructor argument is a plugin fault, not a verdict.
    assert response.status_code == 500
    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verification_result_code is None
    _assert_secret_absent_everywhere(application, response, evidence_id, caplog)


def test_plugin_extra_result_attribute_is_ignored(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="proof_release")
    application, client, created, evidence_id = _smuggle_setup(
        tmp_path, "smuggle-attr", _SmuggleViaAttributeVerifier()
    )

    response = _verify(client, evidence_id, created, SMUGGLED_EVIDENCE)

    # The valid boolean verdict is honored; the smuggled attribute is ignored.
    assert response.status_code == 200
    assert response.json()["status"] == "verified"
    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "verified"
        assert record.verification_result_code == "accepted"
        assert not hasattr(record, "payload")
    _assert_secret_absent_everywhere(application, response, evidence_id, caplog)


def test_plugin_foreign_result_object_is_a_fault(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="proof_release")
    application, client, created, evidence_id = _smuggle_setup(
        tmp_path, "smuggle-object", _SmuggleViaForeignObjectVerifier()
    )

    response = _verify(client, evidence_id, created, SMUGGLED_EVIDENCE)

    assert response.status_code == 500
    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verification_result_code is None
    _assert_secret_absent_everywhere(application, response, evidence_id, caplog)


def test_plugin_non_boolean_verdict_is_a_fault(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="proof_release")
    application, client, created, evidence_id = _smuggle_setup(
        tmp_path, "nonbool", _NonBoolAcceptedVerifier()
    )

    response = _verify(client, evidence_id, created, SMUGGLED_EVIDENCE)

    # A truthy non-boolean must never be interpreted as acceptance.
    assert response.status_code == 500
    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verification_result_code is None
    _assert_secret_absent_everywhere(application, response, evidence_id, caplog)


def test_plugin_exception_with_secret_rolls_back_and_can_retry(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="proof_release")
    registry = VerifierRegistry()
    raising = _RaiseWithSecretVerifier()
    registry.register(raising)
    application = create_app(
        f"sqlite:///{tmp_path}/raise-secret.db", verifier_registry=registry
    )
    client = TestClient(application)
    created = _create(client).json()
    evidence_id = _submit(
        client, created, SMUGGLED_EVIDENCE, evidence_format="raise-secret"
    ).json()["evidence_id"]

    response = _verify(client, evidence_id, created, SMUGGLED_EVIDENCE)

    assert response.status_code == 500
    assert response.json() == {"detail": "verification failed"}
    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verified_at is None
        assert record.verification_result_code is None
    _assert_secret_absent_everywhere(application, response, evidence_id, caplog)

    # The logged line names only the exception type, never its message.
    assert "_RaiseWithSecretVerifier" in caplog.text
    assert "_SecretLeakingException" in caplog.text

    # Replace the faulty plugin and retry; the record settles exactly once.
    registry.register(CountingVerifier())  # registered under "counting"
    replacement = CountingVerifier()
    replacement.format_name = "raise-secret"
    registry.register(replacement)
    response = _verify(client, evidence_id, created, SMUGGLED_EVIDENCE)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"
    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.verification_result_code == "accepted"
    with open(str(application.state.engine.url.database), "rb") as handle:
        assert SMUGGLED_SECRET.encode() not in handle.read()


def test_rejected_result_persists_rejected_code_and_is_stable(client, app):
    created = _create(client).json()
    document = {"nonce": created["nonce"], "claims": {}, "mac": "0" * 64}
    evidence = json.dumps(document)
    evidence_id = _submit(client, created, evidence).json()["evidence_id"]

    first = _verify(client, evidence_id, created, evidence)
    assert first.status_code == 200
    assert first.json()["status"] == "rejected"
    second = _verify(client, evidence_id, created, evidence)
    assert second.json() == first.json()

    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "rejected"
        assert record.verification_result_code == "rejected"


def test_concurrent_verify_writes_consistent_result_code(tmp_path):
    verifier = AlternatingVerifier()
    registry = VerifierRegistry()
    registry.register(verifier)
    application = create_app(
        f"sqlite:///{tmp_path}/alternating-code.db", verifier_registry=registry
    )
    created = _create(TestClient(application)).json()
    evidence = "same-bytes-for-every-caller"
    evidence_id = _submit(
        TestClient(application), created, evidence, evidence_format="alternating"
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
    with application.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        expected_code = VerificationResultCode.from_accepted(
            record.status == "verified"
        ).value
        assert record.verification_result_code == expected_code
        assert record.verification_result_code in {"accepted", "rejected"}


# --- Legacy (pre-change) SQLite database migration -----------------------

def _write_legacy_database(path, nonce: str, received_evidence: str) -> dict:
    """Create a database with the *old* schema, including free-text notes."""
    now = datetime.now(timezone.utc)
    issued = (now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S.%f")
    expires = (now + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S.%f")
    consumed = (now - timedelta(minutes=4)).strftime("%Y-%m-%d %H:%M:%S.%f")
    verified = (now - timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M:%S.%f")
    received = (now - timedelta(minutes=4, seconds=5)).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE challenges (
                challenge_id VARCHAR(36) NOT NULL,
                tenant_id VARCHAR(256) NOT NULL,
                workload_id VARCHAR(256) NOT NULL,
                nonce_digest VARCHAR(64) NOT NULL,
                status VARCHAR(16) NOT NULL,
                issued_at DATETIME NOT NULL,
                expires_at DATETIME NOT NULL,
                consumed_at DATETIME,
                PRIMARY KEY (challenge_id)
            );
            CREATE TABLE evidence (
                evidence_id VARCHAR(36) NOT NULL,
                challenge_id VARCHAR(36) NOT NULL,
                tenant_id VARCHAR(256) NOT NULL,
                workload_id VARCHAR(256) NOT NULL,
                evidence_format VARCHAR(128) NOT NULL,
                status VARCHAR(16) NOT NULL,
                received_at DATETIME NOT NULL,
                evidence_sha256 VARCHAR(64) NOT NULL,
                verified_at DATETIME,
                verification_detail VARCHAR(256),
                PRIMARY KEY (evidence_id),
                UNIQUE (challenge_id)
            );
            """
        )

        def add_challenge(cid, status="consumed"):
            conn.execute(
                "INSERT INTO challenges "
                "(challenge_id, tenant_id, workload_id, nonce_digest, status, "
                "issued_at, expires_at, consumed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cid,
                    TENANT,
                    WORKLOAD,
                    hashlib.sha256(nonce.encode("ascii")).hexdigest(),
                    status,
                    issued,
                    expires,
                    consumed if status == "consumed" else None,
                ),
            )

        # verified legacy row carrying a secret-laden free-text note
        verified_cid = "11111111-1111-1111-1111-111111111111"
        rejected_cid = "22222222-2222-2222-2222-222222222222"
        received_cid = "33333333-3333-3333-3333-333333333333"
        add_challenge(verified_cid)
        add_challenge(rejected_cid)
        add_challenge(received_cid)

        legacy_note = f"legacy note with raw evidence {received_evidence} and {SMUGGLED_SECRET}"
        conn.execute(
            "INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                verified_cid,
                TENANT,
                WORKLOAD,
                "attested-nonce-json",
                "verified",
                received,
                hashlib.sha256(b"old-verified-evidence").hexdigest(),
                verified,
                legacy_note,
            ),
        )
        conn.execute(
            "INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                rejected_cid,
                TENANT,
                WORKLOAD,
                "attested-nonce-json",
                "rejected",
                received,
                hashlib.sha256(b"old-rejected-evidence").hexdigest(),
                verified,
                legacy_note,
            ),
        )
        conn.execute(
            "INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "cccccccc-cccc-cccc-cccc-cccccccccccc",
                received_cid,
                TENANT,
                WORKLOAD,
                "legacy-fmt",
                "received",
                received,
                hashlib.sha256(received_evidence.encode("utf-8")).hexdigest(),
                None,
                legacy_note,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "verified_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "rejected_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "received_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
        "verified_cid": "11111111-1111-1111-1111-111111111111",
        "rejected_cid": "22222222-2222-2222-2222-222222222222",
        "received_cid": "33333333-3333-3333-3333-333333333333",
    }


def test_legacy_sqlite_database_migrates_and_scrubs_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    db_path = tmp_path / "legacy.db"
    nonce = base64.urlsafe_b64encode(b"0" * 32).rstrip(b"=").decode("ascii")
    received_evidence = "legacy-pending-evidence"
    ids = _write_legacy_database(str(db_path), nonce, received_evidence)

    # The secret is physically present in the un-migrated file.
    assert SMUGGLED_SECRET.encode() in db_path.read_bytes()

    registry = VerifierRegistry()
    registry.register(CountingVerifier())  # "counting"
    legacy_verifier = CountingVerifier()
    legacy_verifier.format_name = "legacy-fmt"
    registry.register(legacy_verifier)

    application = create_app(
        f"sqlite:///{db_path}", verifier_registry=registry
    )
    client = TestClient(application)

    # The retired free-text column is gone (SQLite >= 3.35) or fully null.
    with application.state.engine.connect() as conn:
        columns = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(evidence)")).fetchall()
        }
    assert "verification_result_code" in columns
    if sqlite3.sqlite_version_info >= (3, 35, 0):
        assert "verification_detail" not in columns

    # Settled conclusions and timestamps survive; codes are backfilled from
    # status, never derived from the old note.
    with application.state.session_factory() as session:
        verified = session.get(Evidence, ids["verified_id"])
        rejected = session.get(Evidence, ids["rejected_id"])
        pending = session.get(Evidence, ids["received_id"])
        assert verified.status == "verified"
        assert verified.verified_at is not None
        assert verified.verification_result_code == "accepted"
        assert rejected.status == "rejected"
        assert rejected.verification_result_code == "rejected"
        assert pending.status == "received"
        assert pending.verification_result_code is None

    # Repeat verification of a settled legacy record returns the stored
    # conclusion with the unchanged response semantics; the old note is gone.
    created_shape = {"challenge_id": ids["verified_cid"], "nonce": nonce}
    response = _verify(
        client,
        ids["verified_id"],
        created_shape,
        "old-verified-evidence",  # digest mismatch ignored once settled
    )
    assert response.status_code == 200
    data = response.json()
    assert data["evidence_id"] == ids["verified_id"]
    assert data["challenge_id"] == ids["verified_cid"]
    assert data["status"] == "verified"
    datetime.fromisoformat(data["verified_at"])
    assert SMUGGLED_SECRET not in response.text

    # A still-pending legacy record verifies normally against the live plugin.
    response = _verify(
        client,
        ids["received_id"],
        {"challenge_id": ids["received_cid"], "nonce": nonce},
        received_evidence,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "verified"
    assert legacy_verifier.calls == 1

    application.state.engine.dispose()
    # Migration (including VACUUM) physically purged the old secret text.
    assert SMUGGLED_SECRET.encode() not in db_path.read_bytes()
