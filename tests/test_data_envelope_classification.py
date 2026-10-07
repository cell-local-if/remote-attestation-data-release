"""Tests for PUT/GET /v1/data-envelopes/{data_id}/classification (+ /history).

Classification is governance/audit metadata only: it never participates in
proving, policy evaluation, authorization or decryption, and these tests
additionally pin that no envelope material or exception text ever leaves
the service on these paths.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from proof_release import app as app_module
from proof_release.db import (
    DataEnvelope,
    DataEnvelopeClassification,
    DataEnvelopeClassificationHistory,
)
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


def _put(client, data_id=DATA_ID, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "classification": "internal",
        "expected_version": 0,
    }
    body.update(overrides)
    return client.put(f"/v1/data-envelopes/{data_id}/classification", json=body)


def _get(client, data_id=DATA_ID, **params):
    query = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    query.update(params)
    return client.get(f"/v1/data-envelopes/{data_id}/classification", params=query)


def _history(client, data_id=DATA_ID, **params):
    query = {"tenant_id": TENANT, "workload_id": WORKLOAD}
    query.update(params)
    return client.get(
        f"/v1/data-envelopes/{data_id}/classification/history", params=query
    )


def _register(client, classification="internal", expected_version=0, **kwargs):
    response = _put(
        client, classification=classification, expected_version=expected_version,
        **kwargs,
    )
    assert response.status_code == 200, response.text
    return response.json()


# --- PUT: happy path ------------------------------------------------------


def test_first_registration_returns_200_version_1(client):
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
    assert isinstance(data["version"], int)
    updated_at = data["updated_at"]
    assert updated_at.endswith("+00:00")
    assert datetime.fromisoformat(updated_at).utcoffset() == timedelta(0)
    # No envelope material or payload ever appears on this path.
    assert PAYLOAD not in response.text
    for secret_name in ("ciphertext", "iv", "tag", "wrapped_key", "payload"):
        assert secret_name not in data


@pytest.mark.parametrize(
    "label", ["public", "internal", "confidential", "restricted"]
)
def test_every_classification_label_is_accepted(client, label):
    assert _create_envelope(client).status_code == 201
    response = _put(client, classification=label)
    assert response.status_code == 200
    assert response.json()["classification"] == label


def test_update_increments_version_and_refreshes_timestamp(client):
    assert _create_envelope(client).status_code == 201
    first = _register(client, "public", 0)

    response = _put(client, classification="restricted", expected_version=1)

    assert response.status_code == 200
    data = response.json()
    assert data["classification"] == "restricted"
    assert data["version"] == 2
    assert data["updated_at"] >= first["updated_at"]

    third = _register(client, "internal", 2)
    assert third["version"] == 3


def test_stale_expected_version_returns_409_and_changes_nothing(client):
    assert _create_envelope(client).status_code == 201
    _register(client, "public", 0)
    _register(client, "internal", 1)

    loser = _put(client, classification="restricted", expected_version=1)

    assert loser.status_code == 409
    assert loser.json()["detail"] == "classification version conflict"
    # The winner's state is untouched: current value and history intact.
    current = _get(client)
    assert current.status_code == 200
    assert current.json()["classification"] == "internal"
    assert current.json()["version"] == 2
    history = _history(client).json()
    assert [e["version"] for e in history["classifications"]] == [1, 2]
    assert [e["classification"] for e in history["classifications"]] == [
        "public",
        "internal",
    ]


def test_registration_on_classified_envelope_with_zero_conflicts(client):
    assert _create_envelope(client).status_code == 201
    _register(client, "public", 0)

    # 0 means "never classified"; the envelope already has version 1.
    assert _put(client, expected_version=0).status_code == 409


def test_concurrent_registrations_settle_exactly_once(client):
    assert _create_envelope(client).status_code == 201

    results = []

    def race(label):
        results.append(
            _put(client, classification=label, expected_version=0).status_code
        )

    threads = [
        threading.Thread(target=race, args=(label,))
        for label in ("public", "internal", "confidential", "restricted")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Exactly one registration wins; every other matching-expectation
    # request is a stable 409.
    assert sorted(results) == [200, 409, 409, 409]
    current = _get(client)
    assert current.status_code == 200
    assert current.json()["version"] == 1
    history = _history(client).json()
    assert len(history["classifications"]) == 1
    assert history["classifications"][0]["version"] == 1


# --- PUT: validation (422) and lookup (404) -------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": ""},
        {"tenant_id": "   "},
        {"tenant_id": 1},
        {"workload_id": ""},
        {"workload_id": "   "},
        {"classification": ""},
        {"classification": "PUBLIC"},
        {"classification": "secret"},
        {"classification": 1},
        {"classification": None},
        {"expected_version": -1},
        {"expected_version": 1.5},
        {"expected_version": True},
        {"expected_version": "0"},
        {"expected_version": None},
    ],
)
def test_put_rejects_invalid_fields(client, overrides):
    assert _create_envelope(client).status_code == 201
    assert _put(client, **overrides).status_code == 422


@pytest.mark.parametrize(
    "missing", ["tenant_id", "workload_id", "classification", "expected_version"]
)
def test_put_requires_all_fields(client, missing):
    assert _create_envelope(client).status_code == 201
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "classification": "internal",
        "expected_version": 0,
    }
    del body[missing]
    assert (
        client.put(
            f"/v1/data-envelopes/{DATA_ID}/classification", json=body
        ).status_code
        == 422
    )


def test_put_rejects_extra_fields(client):
    assert _create_envelope(client).status_code == 201
    assert _put(client, extra="nope").status_code == 422


def test_put_rejects_blank_path_identifier(client):
    assert _create_envelope(client).status_code == 201
    response = client.put(
        "/v1/data-envelopes/%20/classification",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "classification": "internal",
            "expected_version": 0,
        },
    )
    assert response.status_code == 422


def test_put_unknown_or_cross_scope_envelope_returns_404(client):
    assert _create_envelope(client).status_code == 201

    assert _put(client, data_id="missing").status_code == 404
    assert (
        _put(client, data_id="missing").json()["detail"]
        == "data envelope not found"
    )
    # An existing data_id addressed from another scope is
    # indistinguishable from an unknown one.
    assert _put(client, tenant_id="tenant-b").status_code == 404
    assert _put(client, workload_id="workload-2").status_code == 404
    # Nothing was registered by any of the probes.
    assert _get(client).status_code == 404
    assert _get(client).json()["detail"] == "classification not found"


def test_failed_registration_leaves_no_trace(client, app, monkeypatch):
    from sqlalchemy.orm import Session

    assert _create_envelope(client).status_code == 201

    def raise_commit(self):
        raise OSError("disk full")

    monkeypatch.setattr(Session, "commit", raise_commit)
    response = _put(client)
    assert response.status_code == 500
    assert response.json()["detail"] == "classification unavailable"
    # The raw exception text never leaves the service.
    assert "disk full" not in response.text

    monkeypatch.undo()
    with app.state.session_factory() as session:
        assert session.query(DataEnvelopeClassification).count() == 0
        assert session.query(DataEnvelopeClassificationHistory).count() == 0
    # The envelope itself is untouched and classifiable afterwards.
    assert _put(client).status_code == 200


# --- GET current ----------------------------------------------------------


def test_get_returns_current_classification(client):
    assert _create_envelope(client).status_code == 201
    registered = _register(client, "confidential", 0)

    response = _get(client)

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
    assert data["classification"] == "confidential"
    assert data["version"] == 1
    assert data["updated_at"] == registered["updated_at"]
    assert PAYLOAD not in response.text


def test_get_unclassified_envelope_returns_404_classification_not_found(client):
    assert _create_envelope(client).status_code == 201
    response = _get(client)
    assert response.status_code == 404
    assert response.json()["detail"] == "classification not found"


def test_get_unknown_or_cross_scope_returns_404_envelope_not_found(client):
    assert _create_envelope(client).status_code == 201
    _register(client, "public", 0)

    assert _get(client, data_id="missing").status_code == 404
    assert (
        _get(client, data_id="missing").json()["detail"]
        == "data envelope not found"
    )
    # Cross-scope probes are indistinguishable from unknown envelopes,
    # even though the classification exists in the original scope.
    assert _get(client, tenant_id="tenant-b").status_code == 404
    assert _get(client, workload_id="workload-2").status_code == 404
    assert (
        _get(client, tenant_id="tenant-b").json()["detail"]
        == "data envelope not found"
    )


def test_get_rejects_bad_request_shape(client):
    assert _create_envelope(client).status_code == 201
    base = f"/v1/data-envelopes/{DATA_ID}/classification"

    assert client.get(base).status_code == 422
    assert client.get(base, params={"workload_id": WORKLOAD}).status_code == 422
    assert client.get(base, params={"tenant_id": TENANT}).status_code == 422
    assert (
        client.get(
            base, params={"tenant_id": "  ", "workload_id": WORKLOAD}
        ).status_code
        == 422
    )
    assert (
        client.get(
            base, params={"tenant_id": TENANT, "workload_id": " "}
        ).status_code
        == 422
    )
    # Unknown or repeated query parameters are shape errors.
    assert (
        client.get(
            base,
            params={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "cursor": "abc",
            },
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"{base}?tenant_id={TENANT}&tenant_id={TENANT}"
            f"&workload_id={WORKLOAD}"
        ).status_code
        == 422
    )
    # A non-empty body is never meaningful on a read-only query.
    assert (
        client.request(
            "GET",
            base,
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            content=b"{}",
        ).status_code
        == 422
    )


def test_get_read_failure_returns_500_without_exception_text(
    client, monkeypatch
):
    from sqlalchemy.orm import Session

    assert _create_envelope(client).status_code == 201
    _register(client, "public", 0)

    def raise_scalar(self, *args, **kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(Session, "scalar", raise_scalar)
    response = _get(client)
    assert response.status_code == 500
    assert response.json()["detail"] == "classification unavailable"
    assert "backend exploded" not in response.text


# --- GET history ----------------------------------------------------------


def test_history_is_empty_and_complete_when_unclassified(client):
    assert _create_envelope(client).status_code == 201

    response = _history(client)

    assert response.status_code == 200
    body = response.json()
    assert list(body) == ["classifications", "next_cursor", "complete"]
    assert body == {"classifications": [], "next_cursor": "", "complete": True}


def test_history_unknown_or_cross_scope_returns_404_envelope_not_found(client):
    assert _create_envelope(client).status_code == 201
    _register(client, "public", 0)

    assert _history(client, data_id="missing").status_code == 404
    assert (
        _history(client, data_id="missing").json()["detail"]
        == "data envelope not found"
    )
    assert _history(client, tenant_id="tenant-b").status_code == 404
    assert _history(client, workload_id="workload-2").status_code == 404


def test_history_lists_every_version_ascending(client):
    assert _create_envelope(client).status_code == 201
    _register(client, "public", 0)
    _register(client, "internal", 1)
    _register(client, "restricted", 2)

    response = _history(client)

    assert response.status_code == 200
    body = response.json()
    assert list(body) == ["classifications", "next_cursor", "complete"]
    assert body["complete"] is True
    assert body["next_cursor"] == ""
    entries = body["classifications"]
    assert [e["version"] for e in entries] == [1, 2, 3]
    assert [e["classification"] for e in entries] == [
        "public",
        "internal",
        "restricted",
    ]
    for entry in entries:
        # Entries carry exactly the three history fields — no
        # identifiers, no material, no payload.
        assert set(entry) == {"version", "classification", "updated_at"}
        assert entry["updated_at"].endswith("+00:00")
    assert PAYLOAD not in response.text


def test_history_is_isolated_per_envelope_and_scope(client):
    assert _create_envelope(client).status_code == 201
    assert _create_envelope(client, data_id="item-2").status_code == 201
    _register(client, "public", 0)
    _register(client, "internal", 1)
    _register(client, "restricted", 0, data_id="item-2")

    mine = _history(client).json()["classifications"]
    other = _history(client, data_id="item-2").json()["classifications"]
    assert [e["version"] for e in mine] == [1, 2]
    assert [e["version"] for e in other] == [1]
    assert other[0]["classification"] == "restricted"


def test_history_paginates_with_cursor_without_duplicates_or_gaps(
    client, monkeypatch
):
    monkeypatch.setattr(app_module, "CLASSIFICATION_HISTORY_PAGE_SIZE", 3)
    assert _create_envelope(client).status_code == 201
    labels = ["public", "internal", "confidential", "restricted", "public"]
    for index, label in enumerate(labels):
        _register(client, label, index)

    seen = []
    cursor = ""
    pages = 0
    while True:
        response = _history(client, cursor=cursor)
        assert response.status_code == 200
        body = response.json()
        seen.extend(body["classifications"])
        pages += 1
        if body["complete"]:
            assert body["next_cursor"] == ""
            break
        assert body["next_cursor"]
        cursor = body["next_cursor"]

    assert pages == 2
    assert [e["version"] for e in seen] == [1, 2, 3, 4, 5]
    assert [e["classification"] for e in seen] == labels

    # Replaying a cursor yields the identical page (no drift).
    first_page = _history(client, cursor="").json()
    replay = _history(client, cursor="").json()
    assert replay == first_page
    page_two = _history(client, cursor=first_page["next_cursor"]).json()
    assert [e["version"] for e in page_two["classifications"]] == [4, 5]


def test_history_cursor_boundary_is_exclusive(client, monkeypatch):
    monkeypatch.setattr(app_module, "CLASSIFICATION_HISTORY_PAGE_SIZE", 2)
    assert _create_envelope(client).status_code == 201
    for index in range(3):
        _register(client, "internal", index)

    first = _history(client).json()
    assert [e["version"] for e in first["classifications"]] == [1, 2]
    second = _history(client, cursor=first["next_cursor"]).json()
    assert [e["version"] for e in second["classifications"]] == [3]
    assert second["complete"] is True


def test_history_cursor_rejects_forged_tampered_and_cross_context(
    client, monkeypatch
):
    monkeypatch.setattr(app_module, "CLASSIFICATION_HISTORY_PAGE_SIZE", 1)
    assert _create_envelope(client).status_code == 201
    assert _create_envelope(client, data_id="item-2").status_code == 201
    _register(client, "public", 0)
    _register(client, "internal", 1)
    _register(client, "restricted", 0, data_id="item-2")
    _register(client, "public", 1, data_id="item-2")

    valid_cursor = _history(client).json()["next_cursor"]
    assert valid_cursor

    def check(code, **params):
        assert _history(client, **params).status_code == code

    # Forged token and garbage.
    check(422, cursor="forged-cursor")
    check(422, cursor="!!!not-base64url!!!")
    check(422, cursor=" ")
    # Tampered cursor: flip one character of the valid token.
    flipped = valid_cursor[:-1] + ("A" if valid_cursor[-1] != "A" else "B")
    check(422, cursor=flipped)
    # Cross-envelope and cross-scope replay of a valid cursor.
    check(422, cursor=valid_cursor, data_id="item-2")
    check(422, cursor=valid_cursor, tenant_id="tenant-b")
    check(422, cursor=valid_cursor, workload_id="workload-2")
    # A cursor minted for another envelope is only valid in its own
    # context, and a cursor from a different query family is rejected.
    other_cursor = _history(client, data_id="item-2").json()["next_cursor"]
    assert other_cursor
    check(422, cursor=other_cursor)
    foreign = client.get(
        "/v1/data-envelopes",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    if foreign["next_cursor"]:
        check(422, cursor=foreign["next_cursor"])

    # The valid cursor still works in its own context.
    assert _history(client, cursor=valid_cursor).status_code == 200


def test_history_rejects_bad_request_shape(client):
    assert _create_envelope(client).status_code == 201
    base = f"/v1/data-envelopes/{DATA_ID}/classification/history"

    assert client.get(base).status_code == 422
    assert client.get(base, params={"tenant_id": TENANT}).status_code == 422
    assert (
        client.get(
            base, params={"tenant_id": TENANT, "workload_id": ""}
        ).status_code
        == 422
    )
    assert (
        client.get(
            base,
            params={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "bogus": "1",
            },
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"{base}?tenant_id={TENANT}&workload_id={WORKLOAD}"
            f"&cursor=a&cursor=b"
        ).status_code
        == 422
    )
    assert (
        client.request(
            "GET",
            base,
            params={"tenant_id": TENANT, "workload_id": WORKLOAD},
            content=b" ",
        ).status_code
        == 422
    )


def test_history_read_failure_returns_500_without_exception_text(
    client, monkeypatch
):
    from sqlalchemy.orm import Session

    assert _create_envelope(client).status_code == 201
    _register(client, "public", 0)

    def raise_execute(self, *args, **kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(Session, "execute", raise_execute)
    response = _history(client)
    assert response.status_code == 500
    assert response.json()["detail"] == "classification unavailable"
    assert "backend exploded" not in response.text


# --- durability and isolation ---------------------------------------------


def test_classification_and_history_survive_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    assert _create_envelope(client1).status_code == 201
    _register(client1, "public", 0)
    registered = _register(client1, "confidential", 1)
    app1.state.engine.dispose()

    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    try:
        current = _get(client2)
        assert current.status_code == 200
        assert current.json()["classification"] == "confidential"
        assert current.json()["version"] == 2
        assert current.json()["updated_at"] == registered["updated_at"]

        history = _history(client2).json()
        assert [e["version"] for e in history["classifications"]] == [1, 2]
        assert [e["classification"] for e in history["classifications"]] == [
            "public",
            "confidential",
        ]

        # Versioning continues from the persisted state after restart.
        assert (
            _put(client2, classification="internal", expected_version=1)
            .status_code
            == 409
        )
        third = _register(client2, "internal", 2)
        assert third["version"] == 3
    finally:
        app2.state.engine.dispose()


def test_classification_does_not_change_envelope_or_release_paths(client):
    assert _create_envelope(client).status_code == 201
    before = client.get(
        f"/v1/data-envelopes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()

    _register(client, "restricted", 0)

    # The stored envelope — including all material columns — is byte-for-
    # byte identical after classification.
    after = client.get(
        f"/v1/data-envelopes/{DATA_ID}",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    assert after == before


def test_no_sensitive_material_is_persisted_in_classification_rows(app, client):
    assert _create_envelope(client).status_code == 201
    _register(client, "confidential", 0)
    _register(client, "restricted", 1)

    with app.state.session_factory() as session:
        envelope = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        material = (
            envelope.ciphertext
            + envelope.iv
            + envelope.tag
            + envelope.wrapped_key
        )
        rows = list(session.query(DataEnvelopeClassification).all()) + list(
            session.query(DataEnvelopeClassificationHistory).all()
        )
        assert len(rows) == 3
        for row in rows:
            columns = {
                c.name: getattr(row, c.name) for c in row.__table__.columns
            }
            assert set(columns) == {
                "tenant_id",
                "workload_id",
                "data_id",
                "classification",
                "version",
                "updated_at",
            }
            blob = json.dumps(
                {k: str(v) for k, v in columns.items()}
            ).encode("utf-8")
            assert PAYLOAD.encode("utf-8") not in blob
            for chunk in (
                material[i : i + 16]
                for i in range(0, len(material), 16)
            ):
                if len(chunk) == 16:
                    assert chunk not in blob
