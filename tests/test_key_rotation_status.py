"""Tests for GET /v1/key-rotation/status: the read-only key-rotation
inventory for a tenant/workload scope."""

from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"

KEY_V1 = b64url_encode(b"0" * 32)
KEY_V2 = b64url_encode(b"1" * 32)
KEY_V3 = b64url_encode(b"2" * 32)

KEYRING_V1_V2 = json.dumps(
    {"current_version": 2, "keys": {"1": KEY_V1, "2": KEY_V2}}
)


def _keyring(current: int, keys: dict[int, str]) -> str:
    return json.dumps(
        {"current_version": current, "keys": {str(v): k for v, k in keys.items()}}
    )


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = app_module.create_app(f"sqlite:///{tmp_path}/status.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, data_id, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": f"payload-for-{data_id}",
        },
    )


def _status(client, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        "/v1/key-rotation/status",
        params={"tenant_id": tenant, "workload_id": workload},
    )


# --- request shape ---------------------------------------------------------


def test_status_empty_scope(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _status(client)
    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == TENANT
    assert body["workload_id"] == WORKLOAD
    assert body["current_key_version"] == 2
    assert body["total"] == 0
    assert body["at_current"] == 0
    assert body["behind"] == 0
    assert body["by_key_version"] == {}
    assert body["unavailable_key_versions"] == []
    assert body["scope_ready_for_key_drop"] is True
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T.*\+00:00", body["generated_at"])


def test_status_response_is_compact_json_with_trailing_newline(client):
    response = _status(client)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    raw = response.content
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b", " not in raw and b'": ' not in raw
    assert json.loads(raw.decode("utf-8"))


def test_status_field_set_and_order(client):
    body = _status(client).json()
    assert list(body) == [
        "tenant_id",
        "workload_id",
        "current_key_version",
        "total",
        "at_current",
        "behind",
        "by_key_version",
        "unavailable_key_versions",
        "scope_ready_for_key_drop",
        "generated_at",
    ]


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"tenant_id": TENANT},
        {"workload_id": WORKLOAD},
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "  ", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "extra": "x"},
    ],
)
def test_status_rejects_bad_query_shape(client, params):
    response = client.get("/v1/key-rotation/status", params=params)
    assert response.status_code == 422


def test_status_rejects_repeated_parameter(client):
    response = client.get(
        f"/v1/key-rotation/status?tenant_id={TENANT}"
        f"&tenant_id=other&workload_id={WORKLOAD}"
    )
    assert response.status_code == 422


def test_status_rejects_non_empty_body(client):
    response = client.request(
        "GET",
        "/v1/key-rotation/status",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_status_rejects_post(client):
    response = client.post(
        "/v1/key-rotation/status",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 405


# --- counting --------------------------------------------------------------


def test_status_counts_envelopes_by_version(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", _keyring(1, {1: KEY_V1}))
    assert _create(client, "item-1").status_code == 201
    assert _create(client, "item-2").status_code == 201

    # Rotate the keyring: new envelopes land on version 2.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _create(client, "item-3").status_code == 201

    body = _status(client).json()
    assert body["current_key_version"] == 2
    assert body["total"] == 3
    assert body["at_current"] == 1
    assert body["behind"] == 2
    assert body["by_key_version"] == {"1": 2, "2": 1}
    assert body["unavailable_key_versions"] == []
    assert body["scope_ready_for_key_drop"] is False
    # total == at_current + behind == sum(by_key_version.values())
    assert body["total"] == body["at_current"] + body["behind"]
    assert body["total"] == sum(body["by_key_version"].values())


def test_status_reflects_rewrap_without_changing_anything_itself(
    app, client, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", _keyring(1, {1: KEY_V1}))
    assert _create(client, "item-1").status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)

    before = _status(client).json()
    assert before["behind"] == 1
    assert before["scope_ready_for_key_drop"] is False

    # The status endpoint itself is read-only: a second call shows the
    # identical inventory and the stored envelope is untouched.
    again = _status(client).json()
    assert again["by_key_version"] == before["by_key_version"]
    stored = client.get(
        "/v1/data-envelopes/item-1",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    assert stored["key_version"] == 1

    # A real rewrap migrates the envelope and the next status shows it.
    response = client.post(
        "/v1/data-envelopes/item-1/rewrap",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 200
    after = _status(client).json()
    assert after["total"] == 1
    assert after["at_current"] == 1
    assert after["behind"] == 0
    assert after["by_key_version"] == {"2": 1}
    assert after["scope_ready_for_key_drop"] is True


def test_status_scope_is_exact(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _create(client, "item-1").status_code == 201
    assert _create(client, "item-2", tenant="tenant-b").status_code == 201
    assert _create(client, "item-3", workload="workload-2").status_code == 201

    body = _status(client).json()
    assert body["total"] == 1
    assert body["by_key_version"] == {"2": 1}

    other = _status(client, tenant="tenant-b").json()
    assert other["total"] == 1
    assert other["tenant_id"] == "tenant-b"


def test_status_reports_unavailable_versions(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", _keyring(1, {1: KEY_V1}))
    assert _create(client, "item-1").status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _create(client, "item-2").status_code == 201

    # The keyring drops version 1 while an envelope still uses it.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", _keyring(2, {2: KEY_V2}))
    body = _status(client).json()
    assert body["total"] == 2
    assert body["at_current"] == 1
    assert body["behind"] == 1
    assert body["by_key_version"] == {"1": 1, "2": 1}
    assert body["unavailable_key_versions"] == ["1"]
    assert body["scope_ready_for_key_drop"] is False


def test_status_ready_for_drop_requires_no_unavailable(app, client, monkeypatch):
    # All envelopes at the current version, but a historical version still
    # in use elsewhere... construct: all envelopes at current, none
    # unavailable -> ready.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _create(client, "item-1").status_code == 201
    body = _status(client).json()
    assert body["scope_ready_for_key_drop"] is True
    assert body["unavailable_key_versions"] == []


def test_status_by_key_version_sorted_numerically(app, client, monkeypatch):
    # Versions 9 and 10 sort numerically, not lexicographically.
    keys = {9: KEY_V1, 10: KEY_V2}
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", _keyring(9, keys))
    assert _create(client, "item-1").status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", _keyring(10, keys))
    assert _create(client, "item-2").status_code == 201

    body = _status(client).json()
    assert list(body["by_key_version"]) == ["9", "10"]
    assert body["current_key_version"] == 10


def test_status_legacy_master_key_is_version_one(client):
    assert _create(client, "item-1").status_code == 201
    body = _status(client).json()
    assert body["current_key_version"] == 1
    assert body["by_key_version"] == {"1": 1}
    assert body["scope_ready_for_key_drop"] is True


def test_status_response_contains_no_material_or_data_ids(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _create(client, "item-secret-id").status_code == 201
    raw = _status(client).content.decode("utf-8")
    assert "item-secret-id" not in raw
    assert "data_id" not in raw
    assert "wrapped_key" not in raw
    assert "ciphertext" not in raw
    assert "payload" not in raw


# --- failure modes ---------------------------------------------------------


def test_status_missing_keyring_is_500(app, client, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    response = _status(client)
    assert response.status_code == 500


def test_status_malformed_keyring_is_500(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not-json{")
    assert _status(client).status_code == 500

    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING",
        json.dumps({"current_version": 3, "keys": {"2": KEY_V2}}),
    )
    assert _status(client).status_code == 500


def test_status_does_not_change_key_version(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _create(client, "item-1").status_code == 201
    _status(client)
    stored = client.get(
        "/v1/data-envelopes/item-1",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    assert stored["key_version"] == 2
