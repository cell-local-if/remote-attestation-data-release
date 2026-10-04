"""Tests for the ``attested-nonce-json-v2`` shared-key-rotation format."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from proof_release.app import create_app
from proof_release.db import Evidence, ProofLifecycleEvent
from proof_release.db import PROOF_EVENT_TYPE_VERIFIED
from proof_release.verifiers import (
    ATTESTED_NONCE_JSON,
    ATTESTED_NONCE_JSON_V2,
    DEMO_ATTESTED_NONCE_SECRET,
)

TENANT = "tenant-a"
WORKLOAD = "workload-1"

KID_A = "key-2026-01"
KID_B = "key-2026-06"
KEY_A = b"a" * 32
KEY_B = b"b" * 32


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _keyring(keys: dict[str, bytes]) -> str:
    return json.dumps({kid: _b64(key) for kid, key in keys.items()})


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PROOF_RELEASE_ATTESTED_NONCE_KEYS", _keyring({KID_A: KEY_A})
    )
    return create_app(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, tenant_id=TENANT, workload_id=WORKLOAD):
    return client.post(
        "/v1/challenges",
        json={"tenant_id": tenant_id, "workload_id": workload_id},
    )


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


def _evidence(
    nonce: str,
    kid: str = KID_A,
    claims: dict | None = None,
    key: bytes = KEY_A,
    tenant_id: str = TENANT,
    workload_id: str = WORKLOAD,
) -> str:
    claims = {} if claims is None else claims
    return json.dumps(
        {
            "kid": kid,
            "nonce": nonce,
            "claims": claims,
            "mac": _mac_v2(key, nonce, kid, claims, tenant_id, workload_id),
        }
    )


def _submit(client, created, evidence, tenant_id=TENANT, workload_id=WORKLOAD):
    return client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant_id,
            "workload_id": workload_id,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": ATTESTED_NONCE_JSON_V2,
            "evidence": evidence,
        },
    )


def _verify(client, evidence_id, created, evidence, tenant_id=TENANT, workload_id=WORKLOAD):
    return client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": tenant_id,
            "workload_id": workload_id,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )


def _submit_and_verify(client, created, evidence, tenant_id=TENANT, workload_id=WORKLOAD):
    submitted = _submit(client, created, evidence, tenant_id, workload_id)
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    return evidence_id, _verify(
        client, evidence_id, created, evidence, tenant_id, workload_id
    )


def test_single_key_valid_evidence_verifies(client):
    created = _create(client).json()
    evidence = _evidence(created["nonce"], claims={"measurement": "abc"})

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_empty_and_complex_claims_verify(client):
    complex_claims = {
        "nested": {"list": [1, "two", {"three": [None, True, 3.5]}], "empty": {}},
        "unicode": "héllo/世界",
    }
    for claims in ({}, complex_claims):
        created = _create(client).json()
        evidence = _evidence(created["nonce"], claims=claims)
        _, response = _submit_and_verify(client, created, evidence)
        assert response.status_code == 200
        assert response.json()["status"] == "verified"


def test_old_and_new_keys_coexist(client, monkeypatch):
    monkeypatch.setenv(
        "PROOF_RELEASE_ATTESTED_NONCE_KEYS",
        _keyring({KID_A: KEY_A, KID_B: KEY_B}),
    )
    for kid, key in ((KID_A, KEY_A), (KID_B, KEY_B)):
        created = _create(client).json()
        evidence = _evidence(created["nonce"], kid=kid, key=key)
        _, response = _submit_and_verify(client, created, evidence)
        assert response.status_code == 200
        assert response.json()["status"] == "verified"


def test_removed_key_rejects_new_evidence_but_keeps_prior_settlement(
    client, monkeypatch
):
    monkeypatch.setenv(
        "PROOF_RELEASE_ATTESTED_NONCE_KEYS",
        _keyring({KID_A: KEY_A, KID_B: KEY_B}),
    )
    # Settle one proof under the old key while it is still configured.
    created = _create(client).json()
    old_evidence = _evidence(created["nonce"], kid=KID_A, key=KEY_A)
    old_id, response = _submit_and_verify(client, created, old_evidence)
    assert response.json()["status"] == "verified"

    # Rotate: the old key is removed from the keyring.
    monkeypatch.setenv(
        "PROOF_RELEASE_ATTESTED_NONCE_KEYS", _keyring({KID_B: KEY_B})
    )

    # The settled proof returns its first conclusion without re-reading
    # the configuration.
    again = _verify(client, old_id, created, old_evidence)
    assert again.status_code == 200
    assert again.json()["status"] == "verified"

    # New evidence naming the removed kid is rejected; the new kid works.
    created = _create(client).json()
    stale = _evidence(created["nonce"], kid=KID_A, key=KEY_A)
    _, response = _submit_and_verify(client, created, stale)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"

    created = _create(client).json()
    fresh = _evidence(created["nonce"], kid=KID_B, key=KEY_B)
    _, response = _submit_and_verify(client, created, fresh)
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_unknown_kid_rejects(client):
    created = _create(client).json()
    evidence = _evidence(created["nonce"], kid="no-such-key", key=KEY_A)

    _, response = _submit_and_verify(client, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_wrong_mac_rejects(client):
    created = _create(client).json()
    document = json.loads(_evidence(created["nonce"]))
    document["mac"] = "0" * 64

    _, response = _submit_and_verify(client, created, json.dumps(document))

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_mac_is_bound_to_tenant_and_workload(client):
    # A MAC derived for another tenant or workload must not verify here.
    for tenant_id, workload_id in (("tenant-b", WORKLOAD), (TENANT, "workload-2")):
        created = _create(client).json()
        evidence = _evidence(
            created["nonce"], tenant_id=tenant_id, workload_id=workload_id
        )
        _, response = _submit_and_verify(client, created, evidence)
        assert response.status_code == 200
        assert response.json()["status"] == "rejected"

    # The same key under a different scope verifies only in that scope.
    created = _create(client, "tenant-b", "workload-9").json()
    evidence = _evidence(
        created["nonce"], tenant_id="tenant-b", workload_id="workload-9"
    )
    _, response = _submit_and_verify(
        client, created, evidence, tenant_id="tenant-b", workload_id="workload-9"
    )
    assert response.status_code == 200
    assert response.json()["status"] == "verified"


@pytest.mark.parametrize(
    "document",
    [
        # missing keys
        {"nonce": "N", "claims": {}, "mac": "m"},
        {"kid": KID_A, "claims": {}, "mac": "m"},
        {"kid": KID_A, "nonce": "N", "mac": "m"},
        {"kid": KID_A, "nonce": "N", "claims": {}},
        # extra key
        {"kid": KID_A, "nonce": "N", "claims": {}, "mac": "m", "extra": 1},
        # type errors
        {"kid": 7, "nonce": "N", "claims": {}, "mac": "m"},
        {"kid": KID_A, "nonce": 7, "claims": {}, "mac": "m"},
        {"kid": KID_A, "nonce": "N", "claims": [], "mac": "m"},
        {"kid": KID_A, "nonce": "N", "claims": {}, "mac": 7},
        {"kid": "bad kid!", "nonce": "N", "claims": {}, "mac": "m"},
        {"kid": "", "nonce": "N", "claims": {}, "mac": "m"},
        {"kid": "x" * 65, "nonce": "N", "claims": {}, "mac": "m"},
        {"kid": KID_A, "nonce": "N", "claims": {}, "mac": "AB" * 32},
        {"kid": KID_A, "nonce": "N", "claims": {}, "mac": "0" * 63},
        {"kid": KID_A, "nonce": "N", "claims": {}, "mac": "z" * 64},
    ],
)
def test_malformed_documents_reject(client, document):
    created = _create(client).json()
    # Keep the nonce well-formed so only the shape is under test.
    if document.get("nonce") == "N":
        document["nonce"] = created["nonce"]
    if document.get("mac") == "m":
        document["mac"] = "0" * 64

    _, response = _submit_and_verify(client, created, json.dumps(document))

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_non_object_and_wrong_nonce_reject(client):
    created = _create(client).json()
    _, response = _submit_and_verify(client, created, json.dumps(["not", "an", "object"]))
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"

    created = _create(client).json()
    other = _create(client).json()
    evidence = _evidence(other["nonce"])
    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


@pytest.mark.parametrize(
    "raw_config",
    [
        pytest.param(None, id="missing-env"),
        pytest.param("not json{", id="broken-json"),
        pytest.param(json.dumps(["a"]), id="not-an-object"),
        pytest.param(
            json.dumps({f"k{i}": _b64(KEY_A) for i in range(33)}),
            id="too-many-entries",
        ),
        pytest.param(json.dumps({"bad kid!": _b64(KEY_A)}), id="illegal-kid"),
        pytest.param(json.dumps({KID_A: 7}), id="non-string-key"),
        pytest.param(json.dumps({KID_A: "!!!"}), id="non-base64url-key"),
        pytest.param(json.dumps({KID_A: _b64(b"short")}), id="short-key"),
        pytest.param(json.dumps({KID_A: _b64(b"x" * 33)}), id="long-key"),
    ],
)
def test_invalid_config_fails_without_half_state(
    client, app, monkeypatch, raw_config
):
    if raw_config is None:
        monkeypatch.delenv("PROOF_RELEASE_ATTESTED_NONCE_KEYS")
    else:
        monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_KEYS", raw_config)
    created = _create(client).json()
    evidence = _evidence(created["nonce"])
    evidence_id = _submit(client, created, evidence).json()["evidence_id"]

    response = _verify(client, evidence_id, created, evidence)

    assert response.status_code == 500
    assert response.json()["detail"] == "verifier plugin failure"
    # No half state: the evidence is still received and no verification
    # event was committed.
    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"
        assert record.verification_result is None
        events = session.scalars(
            select(ProofLifecycleEvent).where(
                ProofLifecycleEvent.evidence_id == evidence_id,
                ProofLifecycleEvent.event_type == PROOF_EVENT_TYPE_VERIFIED,
            )
        ).all()
        assert events == []


def test_fixed_config_retries_successfully(client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_KEYS", "not json{")
    created = _create(client).json()
    evidence = _evidence(created["nonce"])
    evidence_id = _submit(client, created, evidence).json()["evidence_id"]

    assert _verify(client, evidence_id, created, evidence).status_code == 500

    monkeypatch.setenv(
        "PROOF_RELEASE_ATTESTED_NONCE_KEYS", _keyring({KID_A: KEY_A})
    )
    response = _verify(client, evidence_id, created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"


def test_settlement_is_idempotent_and_ignores_later_config(client, monkeypatch):
    created = _create(client).json()
    evidence = _evidence(created["nonce"])
    evidence_id = _submit(client, created, evidence).json()["evidence_id"]

    first = _verify(client, evidence_id, created, evidence)
    assert first.status_code == 200
    assert first.json()["status"] == "verified"

    # The configuration is not re-read once the proof has settled.
    monkeypatch.delenv("PROOF_RELEASE_ATTESTED_NONCE_KEYS")
    second = _verify(client, evidence_id, created, evidence)
    assert second.status_code == 200
    assert second.json() == first.json()


def test_rejection_settlement_is_idempotent(client, monkeypatch):
    created = _create(client).json()
    evidence = _evidence(created["nonce"], kid="no-such-key")
    evidence_id = _submit(client, created, evidence).json()["evidence_id"]

    first = _verify(client, evidence_id, created, evidence)
    assert first.json()["status"] == "rejected"

    # Even adding the missing key afterwards does not reopen the proof.
    monkeypatch.setenv(
        "PROOF_RELEASE_ATTESTED_NONCE_KEYS",
        _keyring({KID_A: KEY_A, "no-such-key": KEY_A}),
    )
    second = _verify(client, evidence_id, created, evidence)
    assert second.status_code == 200
    assert second.json() == first.json()


def test_v1_format_is_unaffected_by_v2_configuration(client, monkeypatch):
    # v1 keeps its own env var and development fallback; the v2 keyring
    # and the v2 document rules do not apply to it.
    monkeypatch.delenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", raising=False)
    created = _create(client).json()
    nonce = created["nonce"]
    mac_key = hmac.new(
        DEMO_ATTESTED_NONCE_SECRET.encode("utf-8"),
        f"{TENANT}:{WORKLOAD}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": {}, "nonce": nonce}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    mac = hmac.new(mac_key, payload, hashlib.sha256).hexdigest()
    # Extra keys (e.g. a kid) are tolerated by the v1 format.
    evidence = json.dumps({"nonce": nonce, "claims": {}, "mac": mac, "kid": KID_A})
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

    response = _verify(client, submitted.json()["evidence_id"], created, evidence)

    assert response.status_code == 200
    assert response.json()["status"] == "verified"
