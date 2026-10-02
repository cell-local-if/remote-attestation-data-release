"""Tests for GET /v1/key-rotation/status.

A read-only key-rotation inventory for one tenant/workload scope: how
many committed envelopes still sit on historical master key versions
versus the keyring's current version, which in-use versions have no
configured unwrapping key, and whether the scope has no historical
version left. The endpoint never writes — it rotates, re-wraps, or
deletes nothing, appends no audit row, and returns no data_id, payload,
key, or envelope material.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from proof_release.app import create_app
from proof_release.db import DataEnvelope
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
PAYLOAD = "status-secret-payload"
STATUS_PATH = "/v1/key-rotation/status"

KEY_V1 = b64url_encode(b"0123456789abcdef0123456789abcdef")
KEY_V2 = b64url_encode(b"fedcba9876543210fedcba9876543210")
KEY_V3 = b64url_encode(b"a" * 32)

KEYRING_V1_V2 = json.dumps(
    {"current_version": 2, "keys": {"1": KEY_V1, "2": KEY_V2}}
)
KEYRING_V2_ONLY = json.dumps({"current_version": 2, "keys": {"2": KEY_V2}})

EXPECTED_FIELD_ORDER = [
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


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", KEY_V1)
    application = create_app(f"sqlite:///{tmp_path}/rotation_status.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _status(client, *, tenant=TENANT, workload=WORKLOAD, **kwargs):
    params = kwargs.pop("params", None)
    if params is None:
        params = {"tenant_id": tenant, "workload_id": workload}
    return client.request("GET", STATUS_PATH, params=params, **kwargs)


def _create(client, data_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": PAYLOAD,
        },
    )


def _rewrap(client, data_id, *, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        f"/v1/data-envelopes/{data_id}/rewrap",
        json={"tenant_id": tenant, "workload_id": workload},
    )


# --- shape and wire format ---------------------------------------------------


def test_empty_scope_reports_ready(client):
    response = _status(client)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    # Compact JSON terminated by exactly one newline.
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert b", " not in response.content
    data = response.json()
    assert list(data) == EXPECTED_FIELD_ORDER
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["current_key_version"] == 1
    assert data["total"] == 0
    assert data["at_current"] == 0
    assert data["behind"] == 0
    assert data["by_key_version"] == {}
    assert data["unavailable_key_versions"] == []
    assert data["scope_ready_for_key_drop"] is True
    generated_at = data["generated_at"]
    assert generated_at.endswith("+00:00")
    assert datetime.fromisoformat(generated_at).utcoffset() == timedelta(0)


def test_status_reflects_current_and_historical_versions(app, client, monkeypatch):
    # Three envelopes on version 1 (legacy single-key configuration).
    for data_id in ("d1", "d2", "d3"):
        assert _create(client, data_id).status_code == 201

    # Rotate the keyring and re-wrap two of the three.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _rewrap(client, "d1").status_code == 200
    assert _rewrap(client, "d2").status_code == 200

    response = _status(client)
    assert response.status_code == 200
    data = response.json()
    assert data["current_key_version"] == 2
    assert data["total"] == 3
    assert data["at_current"] == 2
    assert data["behind"] == 1
    assert data["total"] == data["at_current"] + data["behind"]
    assert data["by_key_version"] == {"1": 1, "2": 2}
    assert list(data["by_key_version"]) == ["1", "2"]
    assert sum(data["by_key_version"].values()) == data["total"]
    assert data["unavailable_key_versions"] == []
    assert data["scope_ready_for_key_drop"] is False


def test_status_ready_when_all_envelopes_at_current(app, client, monkeypatch):
    assert _create(client, "d1").status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _rewrap(client, "d1").status_code == 200

    data = _status(client).json()
    assert data["total"] == 1
    assert data["at_current"] == 1
    assert data["behind"] == 0
    assert data["by_key_version"] == {"2": 1}
    assert data["unavailable_key_versions"] == []
    assert data["scope_ready_for_key_drop"] is True


def test_unavailable_versions_listed_and_block_readiness(
    app, client, monkeypatch
):
    assert _create(client, "d1").status_code == 201
    assert _create(client, "d2").status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    assert _rewrap(client, "d2").status_code == 200

    # The keyring drops version 1 while d1 still uses it.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    data = _status(client).json()
    assert data["current_key_version"] == 2
    assert data["total"] == 2
    assert data["at_current"] == 1
    assert data["behind"] == 1
    assert data["by_key_version"] == {"1": 1, "2": 1}
    assert data["unavailable_key_versions"] == [1]
    assert data["scope_ready_for_key_drop"] is False


def test_unavailable_versions_sorted_numerically(app, client, monkeypatch):
    # Envelopes on versions 2..11, then a keyring that knows none of the
    # historical ones: numeric (not lexicographic) ordering is observable.
    for index in range(2, 12):
        monkeypatch.setenv(
            "PROOF_RELEASE_KEYRING",
            json.dumps(
                {"current_version": index, "keys": {str(index): KEY_V1}}
            ),
        )
        assert _create(client, f"d{index}").status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V2_ONLY)
    # Version 2 is configured; versions 3..11 are not.
    data = _status(client).json()
    assert data["unavailable_key_versions"] == list(range(3, 12))
    assert list(data["by_key_version"]) == [str(v) for v in range(2, 12)]
    assert data["current_key_version"] == 2
    assert data["at_current"] == 1
    assert data["behind"] == 9


def test_scope_isolation(app, client, monkeypatch):
    assert _create(client, "d1").status_code == 201
    assert _create(client, "d2", tenant=OTHER_TENANT).status_code == 201
    assert _create(client, "d3", workload=OTHER_WORKLOAD).status_code == 201

    data = _status(client).json()
    assert data["total"] == 1
    assert data["by_key_version"] == {"1": 1}
    assert _status(client, tenant=OTHER_TENANT).json()["total"] == 1
    assert _status(client, workload=OTHER_WORKLOAD).json()["total"] == 1


def test_response_carries_no_envelope_or_key_material(app, client, monkeypatch):
    assert _create(client, "d1").status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    response = _status(client)
    assert response.status_code == 200
    data = response.json()
    assert "d1" not in response.text
    assert "data_id" not in data
    assert PAYLOAD not in response.text
    for secret_name in ("ciphertext", "iv", "tag", "wrapped_key", "payload"):
        assert secret_name not in data
    for key_text in (KEY_V1, KEY_V2):
        assert key_text not in response.text


def test_status_does_not_modify_envelopes_or_keyring(app, client, monkeypatch):
    assert _create(client, "d1").status_code == 201
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    before = client.get(
        "/v1/data-envelopes/d1",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()

    assert _status(client).status_code == 200

    after = client.get(
        "/v1/data-envelopes/d1",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    assert after == before
    assert after["key_version"] == 1
    with app.state.session_factory() as session:
        assert session.query(DataEnvelope).count() == 1


# --- request shape: 422 without reading state --------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"tenant_id": TENANT},
        {"workload_id": WORKLOAD},
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "   ", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": "  "},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "data_id": "d1"},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "limit": "5"},
    ],
)
def test_invalid_query_shape_returns_422(client, params):
    assert _status(client, params=params).status_code == 422


def test_repeated_parameter_returns_422(client):
    response = client.get(
        f"{STATUS_PATH}?tenant_id={TENANT}&tenant_id={OTHER_TENANT}"
        f"&workload_id={WORKLOAD}"
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "content,content_type",
    [
        (b"{}", "application/json"),
        (b'{"tenant_id": "t"}', "application/json"),
        (b" ", "text/plain"),
        (b"x", "text/plain"),
    ],
)
def test_non_empty_body_returns_422(client, content, content_type):
    response = client.request(
        "GET",
        STATUS_PATH,
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=content,
        headers={"content-type": content_type},
    )
    assert response.status_code == 422


def test_invalid_request_reads_no_envelopes(app, client, monkeypatch):
    # A failing keyring would produce 500 if the handler consulted it;
    # shape failures must be rejected before any state is read.
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", "not json")
    assert _status(client, params={}).status_code == 422
    assert (
        _status(client, params={"tenant_id": TENANT, "extra": "1"}).status_code
        == 422
    )


# --- server-side failures: 500 -----------------------------------------------


@pytest.mark.parametrize(
    "keyring",
    [
        "not json",
        "[]",
        "{}",
        json.dumps({"current_version": 0, "keys": {"1": KEY_V1}}),
        json.dumps({"current_version": "1", "keys": {"1": KEY_V1}}),
        json.dumps({"current_version": 1, "keys": {}}),
        json.dumps({"current_version": 1, "keys": {"1": "not-base64!!!"}}),
        # current_version has no configured key.
        json.dumps({"current_version": 2, "keys": {"1": KEY_V1}}),
    ],
)
def test_unusable_keyring_returns_500(app, client, monkeypatch, keyring):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", keyring)
    response = _status(client)
    assert response.status_code == 500
    assert PAYLOAD not in response.text
    for key_text in (KEY_V1, KEY_V2, KEY_V3):
        assert key_text not in response.text


def test_missing_keyring_returns_500(app, client, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    assert _status(client).status_code == 500


def test_database_failure_returns_500(app, client):
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE data_envelopes"))
    response = _status(client)
    assert response.status_code == 500
    # No fabricated/partial tally reaches the client.
    assert response.content == b'{"detail":"key rotation status unavailable"}'
