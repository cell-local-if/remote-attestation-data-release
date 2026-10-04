"""Tests for the ``attested-nonce-json-v2`` evidence format.

The v2 format supports shared-key rotation: evidence carries a ``kid``
that selects one of up to 32 shared keys configured through
``PROOF_RELEASE_ATTESTED_NONCE_KEYS``. Configuration failures are plugin
failures (500, evidence stays received and retryable); malformed
evidence, unknown kids and bad MACs settle as rejected (200).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.db import Evidence
from proof_release.verifiers import ATTESTED_NONCE_JSON_V2

TENANT = "tenant-a"
WORKLOAD = "workload-1"
KEYS_ENV = "PROOF_RELEASE_ATTESTED_NONCE_KEYS"

KEY_ONE = bytes(range(32))
KEY_TWO = bytes(range(32, 64))


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _keys_env(mapping: dict[str, bytes]) -> str:
    return json.dumps({kid: _b64(key) for kid, key in mapping.items()})


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv(KEYS_ENV, _keys_env({"key-1": KEY_ONE}))
    return create_app(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, **overrides):
    body = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    body.update(overrides)
    return client.post("/v1/challenges", json=body)


def _mac_v2(
    key: bytes,
    nonce: str,
    kid: str,
    claims: dict,
    tenant_id: str = TENANT,
    workload_id: str = WORKLOAD,
) -> str:
    mac_key = hmac.new(
        key, f"{tenant_id}:{workload_id}".encode("utf-8"), hashlib.sha256
    ).digest()
    payload = (
        b"attested-nonce-json-v2\x00"
        + json.dumps(
            {"claims": claims, "kid": kid, "nonce": nonce},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return hmac.new(mac_key, payload, hashlib.sha256).hexdigest()


def _evidence_v2(
    nonce: str,
    kid: str = "key-1",
    claims: dict | None = None,
    key: bytes = KEY_ONE,
    **mac_kwargs,
) -> str:
    claims = {} if claims is None else claims
    return json.dumps(
        {
            "kid": kid,
            "nonce": nonce,
            "claims": claims,
            "mac": _mac_v2(key, nonce, kid, claims, **mac_kwargs),
        }
    )


def _submit(client, created, evidence, evidence_format=ATTESTED_NONCE_JSON_V2):
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


def _submit_and_verify(client, created, evidence, **verify_overrides):
    """Submit ``evidence`` under ``created`` and run the first verify."""
    submitted = _submit(client, created, evidence)
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    return evidence_id, _verify(
        client, evidence_id, created, evidence, **verify_overrides
    )


def test_valid_evidence_verifies(client):
    created = _create(client).json()
    evidence = _evidence_v2(created["nonce"], claims={"measurement": "abc"})
    evidence_id, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    data = response.json()
    assert data["evidence_id"] == evidence_id
    assert data["status"] == "verified"
    assert data["challenge_id"] == created["challenge_id"]


def test_empty_claims_verify(client):
    created = _create(client).json()
    evidence = _evidence_v2(created["nonce"], claims={})
    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_complex_claims_verify(client):
    claims = {
        "measurement": "abc",
        "nested": {"list": [1, "two", None, True], "float": 1.5},
        "unicode": "héllo-世界",
        "empty": {},
    }
    created = _create(client).json()
    evidence = _evidence_v2(created["nonce"], claims=claims)
    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_old_and_new_keys_coexist(client, monkeypatch):
    monkeypatch.setenv(
        KEYS_ENV, _keys_env({"key-old": KEY_ONE, "key-new": KEY_TWO})
    )
    for kid, key in (("key-old", KEY_ONE), ("key-new", KEY_TWO)):
        created = _create(client).json()
        evidence = _evidence_v2(created["nonce"], kid=kid, key=key)
        _, response = _submit_and_verify(client, created, evidence)
        assert response.status_code == 200
        assert response.json()["status"] == "verified"


def test_removed_key_rejects_new_evidence(client, monkeypatch):
    monkeypatch.setenv(
        KEYS_ENV, _keys_env({"key-old": KEY_ONE, "key-new": KEY_TWO})
    )
    created = _create(client).json()
    old_evidence = _evidence_v2(created["nonce"], kid="key-old", key=KEY_ONE)
    _, response = _submit_and_verify(client, created, old_evidence)
    assert response.json()["status"] == "verified"

    # Rotation completes: the old key is removed from the configuration.
    monkeypatch.setenv(KEYS_ENV, _keys_env({"key-new": KEY_TWO}))

    created = _create(client).json()
    stale_evidence = _evidence_v2(created["nonce"], kid="key-old", key=KEY_ONE)
    _, response = _submit_and_verify(client, created, stale_evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"

    created = _create(client).json()
    new_evidence = _evidence_v2(created["nonce"], kid="key-new", key=KEY_TWO)
    _, response = _submit_and_verify(client, created, new_evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_unknown_kid_rejects(client):
    created = _create(client).json()
    evidence = _evidence_v2(created["nonce"], kid="no-such-key", key=KEY_ONE)
    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_wrong_mac_rejects(client):
    created = _create(client).json()
    document = json.loads(_evidence_v2(created["nonce"]))
    document["mac"] = "0" * 64
    _, response = _submit_and_verify(client, created, json.dumps(document))

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_mac_computed_with_foreign_key_rejects(client):
    created = _create(client).json()
    evidence = _evidence_v2(created["nonce"], key=b"\xff" * 32)
    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


@pytest.mark.parametrize(
    "scope",
    [
        {"tenant_id": "tenant-b"},
        {"workload_id": "workload-2"},
        {"tenant_id": "tenant-b", "workload_id": "workload-2"},
    ],
)
def test_mac_bound_to_other_tenant_or_workload_rejects(client, scope):
    # The MAC key is derived per tenant and workload: a MAC valid in one
    # scope never verifies in another, even with the same shared key.
    created = _create(client).json()
    evidence = _evidence_v2(created["nonce"], **scope)
    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_evidence_isolated_across_tenants(client):
    # Evidence received in one tenant/workload scope cannot be verified in
    # another: the service scopes the evidence lookup itself.
    created = _create(client).json()
    evidence = _evidence_v2(created["nonce"])
    submitted = _submit(client, created, evidence)
    evidence_id = submitted.json()["evidence_id"]

    response = _verify(client, evidence_id, created, evidence, tenant_id="tenant-b")
    assert response.status_code == 404
    response = _verify(client, evidence_id, created, evidence, workload_id="workload-2")
    assert response.status_code == 404


def test_nonce_bound_to_other_challenge_rejects(client):
    created = _create(client).json()
    other = _create(client).json()
    # The document binds the other challenge's nonce; it is submitted under
    # the first challenge with that challenge's correct submission nonce.
    evidence = _evidence_v2(other["nonce"])
    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


@pytest.mark.parametrize(
    "document",
    [
        "not json",
        "[]",
        "42",
        json.dumps({"nonce": "abc", "claims": {}, "mac": "0" * 64}),  # no kid
        json.dumps({"kid": "key-1", "claims": {}, "mac": "0" * 64}),  # no nonce
        json.dumps({"kid": "key-1", "nonce": "abc", "mac": "0" * 64}),  # no claims
        json.dumps({"kid": "key-1", "nonce": "abc", "claims": {}}),  # no mac
        json.dumps(  # extra key
            {"kid": "key-1", "nonce": "abc", "claims": {}, "mac": "0" * 64, "x": 1}
        ),
        json.dumps({"kid": "", "nonce": "abc", "claims": {}, "mac": "0" * 64}),
        json.dumps({"kid": "bad kid!", "nonce": "abc", "claims": {}, "mac": "0" * 64}),
        json.dumps({"kid": "k" * 65, "nonce": "abc", "claims": {}, "mac": "0" * 64}),
        json.dumps({"kid": 7, "nonce": "abc", "claims": {}, "mac": "0" * 64}),
        json.dumps({"kid": "key-1", "nonce": "ab=", "claims": {}, "mac": "0" * 64}),
        json.dumps({"kid": "key-1", "nonce": "abc", "claims": [], "mac": "0" * 64}),
        json.dumps({"kid": "key-1", "nonce": "abc", "claims": {}, "mac": "0" * 63}),
        json.dumps({"kid": "key-1", "nonce": "abc", "claims": {}, "mac": "0" * 65}),
        json.dumps({"kid": "key-1", "nonce": "abc", "claims": {}, "mac": "A" + "0" * 63}),
        json.dumps({"kid": "key-1", "nonce": "abc", "claims": {}, "mac": "z" * 64}),
        json.dumps({"kid": "key-1", "nonce": "abc", "claims": {}, "mac": 42}),
    ],
)
def test_malformed_documents_reject(client, document):
    created = _create(client).json()
    _, response = _submit_and_verify(client, created, document)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_missing_keys_env_returns_500_and_retry_succeeds(client, app, monkeypatch):
    monkeypatch.delenv(KEYS_ENV)
    created = _create(client).json()
    evidence = _evidence_v2(created["nonce"])
    evidence_id, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 500
    assert response.json()["detail"] == "verifier plugin failure"
    assert evidence not in response.text

    # No half state: the evidence is still received and unsettled.
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verified_at is None
        assert record.verification_result is None

    # After the configuration is fixed the identical request settles.
    monkeypatch.setenv(KEYS_ENV, _keys_env({"key-1": KEY_ONE}))
    response = _verify(client, evidence_id, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


@pytest.mark.parametrize(
    "raw_config",
    [
        "{not json",
        "[1, 2]",
        '"just a string"',
        "42",
        "null",
        json.dumps({f"key-{i}": _b64(KEY_ONE) for i in range(33)}),  # 33 entries
        json.dumps({"bad kid!": _b64(KEY_ONE)}),
        json.dumps({"": _b64(KEY_ONE)}),
        json.dumps({"key-1": _b64(b"too-short")}),
        json.dumps({"key-1": _b64(b"x" * 33)}),
        json.dumps({"key-1": _b64(KEY_ONE) + "="}),  # padded encoding
        json.dumps({"key-1": "not base64!!!"}),
        json.dumps({"key-1": 42}),
    ],
)
def test_invalid_key_configuration_returns_500_without_half_state(
    client, app, monkeypatch, raw_config
):
    monkeypatch.setenv(KEYS_ENV, raw_config)
    created = _create(client).json()
    evidence = _evidence_v2(created["nonce"])
    evidence_id, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 500
    assert response.json()["detail"] == "verifier plugin failure"
    assert evidence not in response.text
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verified_at is None
        assert record.verification_result is None


def test_exactly_32_keys_is_accepted(client, monkeypatch):
    keys = {f"key-{i}": bytes([i]) * 32 for i in range(32)}
    keys["key-1"] = KEY_ONE
    monkeypatch.setenv(KEYS_ENV, _keys_env(keys))
    created = _create(client).json()
    evidence = _evidence_v2(created["nonce"])
    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_settled_conclusion_is_returned_without_rereading_config(
    client, app, monkeypatch
):
    created = _create(client).json()
    evidence = _evidence_v2(created["nonce"])
    evidence_id, first = _submit_and_verify(client, created, evidence)
    assert first.status_code == 200
    assert first.json()["status"] == "verified"

    # Break the configuration; the stored conclusion must still be
    # returned verbatim and the plugin (and config) never re-read.
    monkeypatch.delenv(KEYS_ENV)
    second = _verify(client, evidence_id, created, evidence)
    assert second.status_code == 200
    assert second.json() == first.json()

    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "verified"
        assert record.verification_result == "accepted"


def test_rejected_settlement_is_also_idempotent(client, monkeypatch):
    created = _create(client).json()
    evidence = _evidence_v2(created["nonce"], kid="unknown-kid")
    evidence_id, first = _submit_and_verify(client, created, evidence)
    assert first.status_code == 200
    assert first.json()["status"] == "rejected"

    # Even adding the missing key afterwards never reopens the settlement.
    monkeypatch.setenv(
        KEYS_ENV, _keys_env({"key-1": KEY_ONE, "unknown-kid": KEY_ONE})
    )
    second = _verify(client, evidence_id, created, evidence)
    assert second.status_code == 200
    assert second.json() == first.json()


def test_v1_format_is_unaffected_by_v2_configuration(tmp_path, monkeypatch):
    # attested-nonce-json keeps its own env var and dev fallback; the v2
    # key map neither applies to it nor is required by it.
    monkeypatch.delenv(KEYS_ENV, raising=False)
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", "v1-secret")
    application = create_app(f"sqlite:///{tmp_path}/v1.db")
    client = TestClient(application)
    created = _create(client).json()

    mac_key = hmac.new(
        b"v1-secret", f"{TENANT}:{WORKLOAD}".encode("utf-8"), hashlib.sha256
    ).digest()
    payload = json.dumps(
        {"claims": {}, "nonce": created["nonce"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    mac = hmac.new(mac_key, payload, hashlib.sha256).hexdigest()
    evidence = json.dumps(
        {"nonce": created["nonce"], "claims": {}, "mac": mac}
    )
    submitted = _submit(client, created, evidence, evidence_format="attested-nonce-json")
    assert submitted.status_code == 201
    response = _verify(client, submitted.json()["evidence_id"], created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_raw_evidence_and_keys_are_never_persisted_or_returned(client, app):
    created = _create(client).json()
    evidence = _evidence_v2(created["nonce"], claims={"measurement": "secret-claim"})
    evidence_id, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert evidence not in response.text
    assert _b64(KEY_ONE) not in response.text

    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        columns = {c.name: getattr(record, c.name) for c in record.__table__.columns}
    for name, value in columns.items():
        assert evidence not in str(value), f"raw evidence leaked into column {name}"
        assert _b64(KEY_ONE) not in str(value), f"key material leaked into {name}"
