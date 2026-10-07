"""Tests for the data-envelope governance classification endpoints.

PUT/GET /v1/data-envelopes/{data_id}/classification and
GET /v1/data-envelopes/{data_id}/classification/history.

The classification is governance audit metadata only: it never
participates in proving, policy evaluation, authorization or decryption,
and it never touches the envelope's material, key version or directory
ordering.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
DATA_ID = "item-1"
PAYLOAD = "super-secret-attestation-payload 🔐"

MASTER_KEY = b64url_encode(b"0123456789abcdef0123456789abcdef")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = app_module.create_app(f"sqlite:///{tmp_path}/classification.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create_envelope(client, data_id=DATA_ID, tenant=TENANT, workload=WORKLOAD):
    return client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "data_id": data_id,
            "payload": PAYLOAD,
        },
    )


def _put(client, data_id=DATA_ID, expected_version=0, classification="internal",
         tenant=TENANT, workload=WORKLOAD, **raw):
    body = {
        "tenant_id": tenant,
        "workload_id": workload,
        "classification": classification,
        "expected_version": expected_version,
    }
    body.update(raw)
    return client.put(f"/v1/data-envelopes/{data_id}/classification", json=body)


def _get_current(client, data_id=DATA_ID, tenant=TENANT, workload=WORKLOAD,
                 **params):
    query = {"tenant_id": tenant, "workload_id": workload}
    query.update(params)
    return client.get(f"/v1/data-envelopes/{data_id}/classification", params=query)


def _get_history(client, data_id=DATA_ID, tenant=TENANT, workload=WORKLOAD,
                 **params):
    query = {"tenant_id": tenant, "workload_id": workload}
    query.update(params)
    return client.get(
        f"/v1/data-envelopes/{data_id}/classification/history", params=query
    )


# --- PUT: first registration and reclassification -------------------------


def test_put_first_registration_returns_200_with_version_1(client):
    assert _create_envelope(client).status_code == 201

    response = _put(client, classification="confidential")

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "data_id",
        "tenant_id",
        "workload_id",
        "classification",
        "version",
        "updated_at",
    }
    assert data["data_id"] == DATA_ID
    assert data["tenant_id"] == TENANT
    assert data["workload_id"] == WORKLOAD
    assert data["classification"] == "confidential"
    assert data["version"] == 1
    assert data["updated_at"].endswith("+00:00")
    assert datetime.fromisoformat(data["updated_at"]).utcoffset() == timedelta(0)
    # No envelope material or payload ever appears.
    assert PAYLOAD not in response.text
    for secret_name in ("ciphertext", "iv", "tag", "wrapped_key", "payload"):
        assert secret_name not in data


@pytest.mark.parametrize("value", ["public", "internal", "confidential", "restricted"])
def test_put_accepts_every_defined_classification(client, value):
    assert _create_envelope(client).status_code == 201
    response = _put(client, classification=value)
    assert response.status_code == 200
    assert response.json()["classification"] == value


def test_put_reclassification_increments_version_and_current_follows(client):
    assert _create_envelope(client).status_code == 201
    first = _put(client, classification="internal")
    assert first.status_code == 200

    second = _put(client, expected_version=1, classification="restricted")

    assert second.status_code == 200
    data = second.json()
    assert data["classification"] == "restricted"
    assert data["version"] == 2
    assert data["updated_at"] >= first.json()["updated_at"]

    current = _get_current(client)
    assert current.status_code == 200
    assert current.json() == data


def test_put_stale_expected_version_is_409_and_changes_nothing(client):
    assert _create_envelope(client).status_code == 201
    assert _put(client, classification="internal").status_code == 200
    assert _put(client, expected_version=1, classification="public").status_code == 200

    stale = _put(client, expected_version=1, classification="restricted")

    assert stale.status_code == 409
    assert stale.json()["detail"] == "classification version conflict"
    # The winner's state is untouched.
    current = _get_current(client).json()
    assert current["classification"] == "public"
    assert current["version"] == 2
    history = _get_history(client).json()
    assert [e["version"] for e in history["classifications"]] == [1, 2]


def test_put_expected_version_zero_on_classified_envelope_is_409(client):
    assert _create_envelope(client).status_code == 201
    assert _put(client).status_code == 200

    response = _put(client, expected_version=0, classification="public")

    assert response.status_code == 409
    assert response.json()["detail"] == "classification version conflict"


def test_put_nonzero_expected_version_on_unclassified_envelope_is_409(client):
    assert _create_envelope(client).status_code == 201

    response = _put(client, expected_version=1)

    assert response.status_code == 409
    assert response.json()["detail"] == "classification version conflict"
    assert _get_current(client).status_code == 404


def test_put_unknown_envelope_is_404(client):
    response = _put(client, data_id="nope")

    assert response.status_code == 404
    assert response.json()["detail"] == "data envelope not found"


def test_put_cross_scope_envelope_is_indistinguishable_404(client):
    assert _create_envelope(client).status_code == 201

    for scope in ({"tenant": "tenant-b"}, {"workload": "workload-2"}):
        response = _put(client, **scope)
        assert response.status_code == 404
        assert response.json()["detail"] == "data envelope not found"
    # The real scope still shows no classification.
    assert _get_current(client).status_code == 404


# --- PUT: request shape (all 422) -----------------------------------------


def test_put_rejects_invalid_classification_value(client):
    assert _create_envelope(client).status_code == 201
    for bad in ("Public", "INTERNAL", "secret", "", " public"):
        response = _put(client, classification=bad)
        assert response.status_code == 422, bad
    assert _get_current(client).status_code == 404


def test_put_rejects_bad_expected_version_shapes(client):
    assert _create_envelope(client).status_code == 201
    for bad in (-1, -100, "0", "1", 1.0, True, False, None, [0], {"v": 0}):
        response = _put(client, expected_version=bad)
        assert response.status_code == 422, repr(bad)
    assert _get_current(client).status_code == 404


def test_put_rejects_missing_unknown_and_mistyped_fields(client):
    assert _create_envelope(client).status_code == 201
    base = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "classification": "public",
        "expected_version": 0,
    }
    for field in base:
        body = {k: v for k, v in base.items() if k != field}
        assert client.put(
            f"/v1/data-envelopes/{DATA_ID}/classification", json=body
        ).status_code == 422, field
    # Unknown fields are rejected, not ignored.
    assert client.put(
        f"/v1/data-envelopes/{DATA_ID}/classification",
        json={**base, "data_id": DATA_ID},
    ).status_code == 422
    assert client.put(
        f"/v1/data-envelopes/{DATA_ID}/classification",
        json={**base, "version": 1},
    ).status_code == 422
    # Wrong-typed scope fields.
    for field in ("tenant_id", "workload_id", "classification"):
        assert client.put(
            f"/v1/data-envelopes/{DATA_ID}/classification",
            json={**base, field: 7},
        ).status_code == 422, field
    # Blank scope strings.
    for field in ("tenant_id", "workload_id"):
        assert client.put(
            f"/v1/data-envelopes/{DATA_ID}/classification",
            json={**base, field: "  "},
        ).status_code == 422, field
    assert _get_current(client).status_code == 404


# --- GET current -----------------------------------------------------------


def test_get_current_unclassified_is_404(client):
    assert _create_envelope(client).status_code == 201

    response = _get_current(client)

    assert response.status_code == 404
    assert response.json()["detail"] == "classification not found"


def test_get_current_unknown_or_cross_scope_envelope_is_404(client):
    assert _create_envelope(client).status_code == 201
    assert _put(client).status_code == 200

    assert _get_current(client, data_id="nope").status_code == 404
    for scope in ({"tenant": "tenant-b"}, {"workload": "workload-2"}):
        response = _get_current(client, **scope)
        assert response.status_code == 404
        assert response.json()["detail"] == "data envelope not found"


def test_get_current_shape_errors_are_422(client):
    assert _create_envelope(client).status_code == 201
    assert _put(client).status_code == 200

    # Missing parameters.
    assert client.get(f"/v1/data-envelopes/{DATA_ID}/classification").status_code == 422
    # Unknown parameter.
    assert _get_current(client, cursor="x").status_code == 422
    # Blank parameters.
    assert _get_current(client, tenant="  ").status_code == 422
    assert _get_current(client, workload="").status_code == 422
    # Repeated parameter.
    assert client.get(
        f"/v1/data-envelopes/{DATA_ID}/classification"
        f"?tenant_id={TENANT}&tenant_id={TENANT}&workload_id={WORKLOAD}"
    ).status_code == 422
    # Non-empty body.
    assert client.request(
        "GET",
        f"/v1/data-envelopes/{DATA_ID}/classification",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b"{}",
    ).status_code == 422


# --- History ---------------------------------------------------------------


def test_history_records_every_committed_version_in_order(client):
    assert _create_envelope(client).status_code == 201
    assert _put(client, classification="public").status_code == 200
    assert _put(client, expected_version=1, classification="internal").status_code == 200
    assert (
        _put(client, expected_version=2, classification="restricted").status_code == 200
    )

    response = _get_history(client)

    assert response.status_code == 200
    data = response.json()
    assert list(data) == ["classifications", "next_cursor", "complete"]
    assert data["complete"] is True
    assert data["next_cursor"] == ""
    entries = data["classifications"]
    assert [e["version"] for e in entries] == [1, 2, 3]
    assert [e["classification"] for e in entries] == [
        "public",
        "internal",
        "restricted",
    ]
    for entry in entries:
        assert set(entry) == {"version", "classification", "updated_at"}
        assert entry["updated_at"].endswith("+00:00")
    # The last history entry equals the current value.
    current = _get_current(client).json()
    assert entries[-1]["classification"] == current["classification"]
    assert entries[-1]["version"] == current["version"]
    assert entries[-1]["updated_at"] == current["updated_at"]


def test_history_unclassified_envelope_is_empty_page(client):
    assert _create_envelope(client).status_code == 201

    data = _get_history(client).json()

    assert data == {"classifications": [], "next_cursor": "", "complete": True}


def test_history_unknown_or_cross_scope_envelope_is_404(client):
    assert _create_envelope(client).status_code == 201
    assert _put(client).status_code == 200

    assert _get_history(client, data_id="nope").status_code == 404
    for scope in ({"tenant": "tenant-b"}, {"workload": "workload-2"}):
        response = _get_history(client, **scope)
        assert response.status_code == 404
        assert response.json()["detail"] == "data envelope not found"


def test_history_shape_errors_are_422(client):
    assert _create_envelope(client).status_code == 201
    assert _put(client).status_code == 200

    assert client.get(
        f"/v1/data-envelopes/{DATA_ID}/classification/history"
    ).status_code == 422
    assert _get_history(client, unknown="x").status_code == 422
    assert _get_history(client, tenant=" ").status_code == 422
    assert client.get(
        f"/v1/data-envelopes/{DATA_ID}/classification/history"
        f"?tenant_id={TENANT}&tenant_id={TENANT}&workload_id={WORKLOAD}"
    ).status_code == 422
    assert client.request(
        "GET",
        f"/v1/data-envelopes/{DATA_ID}/classification/history",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
        content=b" ",
    ).status_code == 422


def test_history_pagination_is_complete_and_repeatable(client, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_CLASSIFICATION_PAGE_SIZE", 2)
    assert _create_envelope(client).status_code == 201
    for expected, value in enumerate(
        ["public", "internal", "confidential", "restricted", "public"]
    ):
        assert _put(
            client, expected_version=expected, classification=value
        ).status_code == 200

    first = _get_history(client)
    assert first.status_code == 200
    page1 = first.json()
    assert [e["version"] for e in page1["classifications"]] == [1, 2]
    assert page1["complete"] is False
    assert page1["next_cursor"] != ""

    second = _get_history(client, cursor=page1["next_cursor"])
    page2 = second.json()
    assert [e["version"] for e in page2["classifications"]] == [3, 4]
    assert page2["complete"] is False

    third = _get_history(client, cursor=page2["next_cursor"])
    page3 = third.json()
    assert [e["version"] for e in page3["classifications"]] == [5]
    assert page3["complete"] is True
    assert page3["next_cursor"] == ""

    # Replaying an earlier cursor yields exactly the same page again: no
    # duplicates across pages and no skipped entries.
    replay = _get_history(client, cursor=page1["next_cursor"]).json()
    assert replay == page2
    seen = [
        e["version"]
        for page in (page1, page2, page3)
        for e in page["classifications"]
    ]
    assert seen == [1, 2, 3, 4, 5]


def test_history_empty_cursor_starts_from_beginning(client):
    assert _create_envelope(client).status_code == 201
    assert _put(client).status_code == 200

    data = _get_history(client, cursor="").json()

    assert [e["version"] for e in data["classifications"]] == [1]
    assert data["complete"] is True


def test_history_rejects_forged_tampered_and_cross_context_cursors(client):
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_CLASSIFICATION_PAGE_SIZE", 1)
    assert _create_envelope(client).status_code == 201
    assert _create_envelope(client, data_id="item-2").status_code == 201
    assert _put(client, classification="public").status_code == 200
    assert _put(client, expected_version=1, classification="internal").status_code == 200
    assert _put(client, data_id="item-2", classification="public").status_code == 200
    assert (
        _put(client, data_id="item-2", expected_version=1, classification="internal")
        .status_code
        == 200
    )
    cursor = _get_history(client).json()["next_cursor"]
    assert cursor

    # Tampered token.
    tampered = cursor[:-1] + ("A" if cursor[-1] != "A" else "B")
    assert _get_history(client, cursor=tampered).status_code == 422
    # Garbage and whitespace tokens.
    assert _get_history(client, cursor="not a cursor!!").status_code == 422
    assert _get_history(client, cursor="  ").status_code == 422
    # Cross-envelope replay.
    assert _get_history(client, data_id="item-2", cursor=cursor).status_code == 422
    # Cross-scope replay.
    assert _get_history(client, tenant="tenant-b", cursor=cursor).status_code == 422
    assert _get_history(client, workload="workload-2", cursor=cursor).status_code == 422
    monkeypatch.undo()


def test_history_cursor_from_another_query_family_is_422(client, monkeypatch):
    # A data-envelope *directory* cursor is authenticated with the same
    # secret but carries a different kind tag: replaying it against the
    # classification history is an indistinguishable 422.
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 1)
    assert _create_envelope(client).status_code == 201
    assert _create_envelope(client, data_id="item-2").status_code == 201
    assert _put(client).status_code == 200
    directory = client.get(
        "/v1/data-envelopes", params={"tenant_id": TENANT, "workload_id": WORKLOAD}
    ).json()
    foreign_cursor = directory["next_cursor"]
    assert foreign_cursor

    assert _get_history(client, cursor=foreign_cursor).status_code == 422


# --- Concurrency -----------------------------------------------------------


def test_concurrent_first_registrations_settle_exactly_once(client):
    assert _create_envelope(client).status_code == 201

    results = []

    def register(value):
        results.append(_put(client, classification=value).status_code)

    threads = [threading.Thread(target=register, args=(value,)) for value in ("public", "restricted")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [200, 409]
    current = _get_current(client).json()
    assert current["version"] == 1
    history = _get_history(client).json()["classifications"]
    # Exactly one history entry exists: the loser wrote nothing.
    assert len(history) == 1
    assert history[0]["classification"] == current["classification"]


def test_concurrent_reclassifications_settle_exactly_once(client):
    assert _create_envelope(client).status_code == 201
    assert _put(client, classification="public").status_code == 200

    results = []

    def reclassify(value):
        results.append(
            _put(client, expected_version=1, classification=value).status_code
        )

    threads = [threading.Thread(target=reclassify, args=(value,)) for value in ("internal", "restricted")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [200, 409]
    current = _get_current(client).json()
    assert current["version"] == 2
    history = _get_history(client).json()["classifications"]
    assert [e["version"] for e in history] == [1, 2]
    assert history[-1]["classification"] == current["classification"]


# --- Durability and failure semantics --------------------------------------


def test_classification_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart.db"
    first = app_module.create_app(url)
    first_client = TestClient(first)
    assert _create_envelope(first_client).status_code == 201
    assert _put(first_client, classification="confidential").status_code == 200
    assert (
        _put(first_client, expected_version=1, classification="restricted").status_code
        == 200
    )
    expected_current = _get_current(first_client).json()
    expected_history = _get_history(first_client).json()
    first.state.engine.dispose()

    second = app_module.create_app(url)
    try:
        second_client = TestClient(second)
        assert _get_current(second_client).json() == expected_current
        assert _get_history(second_client).json() == expected_history
        # The version sequence continues across the restart.
        assert (
            _put(second_client, expected_version=2, classification="public").json()[
                "version"
            ]
            == 3
        )
    finally:
        second.state.engine.dispose()


def test_put_failure_is_500_and_rolls_back(client, app, monkeypatch):
    assert _create_envelope(client).status_code == 201

    from proof_release.db import Base
    from proof_release.db import DataEnvelopeClassification

    with app.state.engine.begin() as conn:
        DataEnvelopeClassification.__table__.drop(conn)

    response = _put(client)
    assert response.status_code == 500
    assert response.json()["detail"] == "classification unavailable"

    # Nothing was half-written: once the table is restored the envelope is
    # still unclassified and a first registration starts at version 1.
    with app.state.engine.begin() as conn:
        Base.metadata.create_all(conn)
    assert _get_current(client).status_code == 404
    assert _put(client).status_code == 200
    assert _get_current(client).json()["version"] == 1


def test_get_current_failure_is_500(client, app):
    assert _create_envelope(client).status_code == 201
    assert _put(client).status_code == 200

    from proof_release.db import DataEnvelopeClassification

    with app.state.engine.begin() as conn:
        DataEnvelopeClassification.__table__.drop(conn)

    response = _get_current(client)
    assert response.status_code == 500
    assert response.json()["detail"] == "classification unavailable"


def test_history_failure_is_500(client, app):
    assert _create_envelope(client).status_code == 201
    assert _put(client).status_code == 200

    from proof_release.db import DataEnvelopeClassificationEvent

    with app.state.engine.begin() as conn:
        DataEnvelopeClassificationEvent.__table__.drop(conn)

    response = _get_history(client)
    assert response.status_code == 500
    assert response.json()["detail"] == "classification unavailable"


# --- Isolation from the envelope itself ------------------------------------


def test_classification_does_not_touch_envelope_material_or_directory(client):
    assert _create_envelope(client).status_code == 201
    before = client.get(
        f"/v1/data-envelopes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()

    assert _put(client, classification="restricted").status_code == 200
    assert _put(client, expected_version=1, classification="public").status_code == 200

    after = client.get(
        f"/v1/data-envelopes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    # Material, key version and creation time are byte-identical.
    assert after == before
    directory = client.get(
        "/v1/data-envelopes", params={"tenant_id": TENANT, "workload_id": WORKLOAD}
    ).json()
    assert [e["data_id"] for e in directory["envelopes"]] == [DATA_ID]
    assert directory["envelopes"][0]["key_version"] == before["key_version"]


def test_classification_is_scoped_per_envelope(client):
    assert _create_envelope(client).status_code == 201
    assert _create_envelope(client, data_id="item-2").status_code == 201
    assert _put(client, classification="restricted").status_code == 200

    # The other envelope in the same scope is independent.
    assert _get_current(client, data_id="item-2").status_code == 404
    assert _get_history(client, data_id="item-2").json()["classifications"] == []
    assert _put(client, data_id="item-2", classification="public").status_code == 200
    assert _get_current(client).json()["classification"] == "restricted"
    assert _get_current(client, data_id="item-2").json()["classification"] == "public"
