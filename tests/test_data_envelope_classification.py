"""Tests for data-envelope classification metadata.

``POST /v1/data-envelopes` accepts an optional ``classification`` (1..32
lowercase ASCII characters: a leading letter, then letters, digits,
underscores or hyphens) that defaults to ``unclassified``. It is pure
queryable metadata: it never participates in encryption, policy
evaluation, authorization, release or rewrap, and it changes neither the
ciphertext material nor the ``data_id`` uniqueness rules. The creation
response, the detail GET and the read-only directory all report it, and
the directory can be filtered by an exact match with cursors bound to
the active filter. Under an ``Idempotency-Key`` the effective
classification is part of the request identity. Envelopes and
idempotency records written before the metadata existed are upgraded on
open to ``unclassified``.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from proof_release import app as app_module
from proof_release.db import (
    Base,
    DataEnvelope,
    DataEnvelopeCommitCounter,
    DataEnvelopeIdempotencyRecord,
)
from proof_release.envelopes import b64url_encode

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"
DATA_ID = "item-1"
PAYLOAD = "super-secret-attestation-payload"

MASTER_KEY_BYTES = b"0123456789abcdef0123456789abcdef"
MASTER_KEY = b64url_encode(MASTER_KEY_BYTES)

KEY_V1 = b64url_encode(b"11111111111111111111111111111111")
KEY_V2 = b64url_encode(b"22222222222222222222222222222222")
KEYRING_V1 = json.dumps({"current_version": 1, "keys": {"1": KEY_V1}})
KEYRING_V1_V2 = json.dumps(
    {"current_version": 2, "keys": {"1": KEY_V1, "2": KEY_V2}}
)


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = app_module.create_app(f"sqlite:///{tmp_path}/classification.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, data_id=DATA_ID, key=None, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": data_id,
        "payload": PAYLOAD,
    }
    body.update(overrides)
    headers = {"Idempotency-Key": key} if key is not None else {}
    return client.post("/v1/data-envelopes", json=body, headers=headers)


def _query(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    query = {"tenant_id": tenant, "workload_id": workload}
    query.update(params)
    return client.get("/v1/data-envelopes", params=query)


def _detail(client, data_id=DATA_ID, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/data-envelopes/{data_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _count_rows(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


def _scope_seq(app) -> int | None:
    with app.state.session_factory() as session:
        counter = session.get(DataEnvelopeCommitCounter, (TENANT, WORKLOAD))
        return None if counter is None else counter.last_seq


def _walk(client, *, page_size, monkeypatch, **params):
    token = None
    rows = []
    for _ in range(50):
        query_params = dict(params)
        if token is not None:
            query_params["cursor"] = token
        monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", page_size)
        page = _query(client, **query_params)
        assert page.status_code == 200, page.text
        data = page.json()
        rows.extend(data["envelopes"])
        if data["complete"]:
            assert data["next_cursor"] == ""
            return rows
        token = data["next_cursor"]
        assert token
    raise AssertionError("pagination never completed")  # pragma: no cover


# --- creation defaults and validation --------------------------------------


def test_omitted_classification_defaults_to_unclassified(client):
    response = _create(client)
    assert response.status_code == 201
    assert response.json()["classification"] == "unclassified"
    assert _detail(client).json()["classification"] == "unclassified"
    (entry,) = _query(client).json()["envelopes"]
    assert entry["classification"] == "unclassified"


def test_explicit_classification_round_trips(client):
    response = _create(client, classification="confidential_data-1")
    assert response.status_code == 201
    assert response.json()["classification"] == "confidential_data-1"
    assert _detail(client).json()["classification"] == "confidential_data-1"
    (entry,) = _query(client).json()["envelopes"]
    assert entry["classification"] == "confidential_data-1"


@pytest.mark.parametrize(
    "value",
    [
        "a",  # single letter
        "z" * 32,  # maximum length
        "a0_-b9",  # digits, underscore and hyphen after the first letter
        "unclassified",
    ],
)
def test_valid_classification_shapes_are_accepted(client, value):
    response = _create(client, classification=value)
    assert response.status_code == 201, response.text
    assert response.json()["classification"] == value


@pytest.mark.parametrize(
    "value",
    [
        "",  # empty
        "   ",  # whitespace
        "Unclassified",  # uppercase
        "UNCLASSIFIED",
        "aB",
        "1abc",  # leading digit
        "_abc",  # leading underscore
        "-abc",  # leading hyphen
        "a" * 33,  # too long
        "ab c",  # space
        "ab.c",  # other characters
        "ab/c",
        "ab!c",
        "café",  # non-ASCII
    ],
)
def test_invalid_classification_is_422_and_creates_nothing(
    app, client, value
):
    response = _create(client, classification=value)
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid classification"
    # No envelope, no directory sequence, no idempotency record.
    assert _count_rows(app, DataEnvelope) == 0
    assert _scope_seq(app) is None
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 0
    assert _detail(client).status_code == 404


def test_invalid_keyed_classification_leaves_key_free(app, client):
    rejected = _create(client, key="key-1", classification="BAD")
    assert rejected.status_code == 422
    assert rejected.json()["detail"] == "invalid classification"

    created = _create(client, key="key-1", classification="good")
    assert created.status_code == 201
    assert created.json()["classification"] == "good"
    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1


def test_classification_does_not_change_uniqueness_or_material(client):
    first = _create(client, classification="alpha")
    assert first.status_code == 201
    # The data_id conflict rule is unchanged by a differing classification.
    conflict = _create(client, classification="beta")
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "data_id already exists in this scope"

    # Two envelopes with the same classification still encrypt
    # independently (fresh data key and IV per request).
    assert _create(client, "item-2", classification="alpha").status_code == 201
    one = _detail(client).json()
    two = _detail(client, "item-2").json()
    assert one["classification"] == two["classification"] == "alpha"
    assert one["ciphertext"] != two["ciphertext"]
    assert one["iv"] != two["iv"]
    assert one["wrapped_key"] != two["wrapped_key"]


# --- detail and directory reads --------------------------------------------


def test_directory_filters_by_exact_classification(client):
    _create(client, "a", classification="alpha")
    _create(client, "b", classification="beta")
    _create(client, "c", classification="alpha")
    _create(client, "d")  # unclassified

    rows = _query(client, classification="alpha").json()["envelopes"]
    assert [row["data_id"] for row in rows] == ["a", "c"]
    rows = _query(client, classification="beta").json()["envelopes"]
    assert [row["data_id"] for row in rows] == ["b"]
    rows = _query(client, classification="unclassified").json()["envelopes"]
    assert [row["data_id"] for row in rows] == ["d"]

    # A well-formed classification with no members is an empty range.
    page = _query(client, classification="gamma").json()
    assert page["envelopes"] == []
    assert page["complete"] is True
    assert page["next_cursor"] == ""


def test_classification_filter_stays_inside_the_scope(client):
    _create(client, "a", classification="alpha")
    _create(
        client,
        "b",
        classification="alpha",
        tenant_id=OTHER_TENANT,
        workload_id=OTHER_WORKLOAD,
    )
    rows = _query(client, classification="alpha").json()["envelopes"]
    assert [row["data_id"] for row in rows] == ["a"]
    rows = _query(
        client,
        tenant=OTHER_TENANT,
        workload=OTHER_WORKLOAD,
        classification="alpha",
    ).json()["envelopes"]
    assert [row["data_id"] for row in rows] == ["b"]


@pytest.mark.parametrize(
    "value", ["", "   ", "Unclassified", "1abc", "a" * 33, "ab.c"]
)
def test_invalid_classification_filter_is_422(client, value):
    response = _query(client, classification=value)
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid classification"


def test_classification_filter_combines_with_other_filters(client):
    _create(client, "a", classification="alpha")
    _create(client, "b", classification="beta")
    created = _query(client).json()["envelopes"]
    stamp = created[0]["created_at"]

    rows = _query(
        client, classification="alpha", data_id="a"
    ).json()["envelopes"]
    assert [row["data_id"] for row in rows] == ["a"]
    # An explicit data_id whose classification differs is an empty page
    # (the named envelope exists, so this is not a 404).
    page = _query(client, classification="beta", data_id="a").json()
    assert page["envelopes"] == []
    rows = _query(
        client,
        classification="alpha",
        created_after=stamp,
        created_before=stamp,
    ).json()["envelopes"]
    assert [row["data_id"] for row in rows] == ["a"]


def test_filtered_pagination_walks_each_match_once(client, monkeypatch):
    for index in range(5):
        _create(client, f"m{index}", classification="alpha")
    for index in range(3):
        _create(client, f"x{index}", classification="beta")

    rows = _walk(client, page_size=2, monkeypatch=monkeypatch,
                 classification="alpha")
    assert [row["data_id"] for row in rows] == [f"m{index}" for index in range(5)]
    assert all(row["classification"] == "alpha" for row in rows)


def test_cursor_is_bound_to_the_classification_filter(client, monkeypatch):
    for index in range(3):
        _create(client, f"m{index}", classification="alpha")
    _create(client, "x0", classification="beta")

    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 2)
    first = _query(client, classification="alpha").json()
    assert not first["complete"]
    cursor = first["next_cursor"]

    # Replaying under the same filter works.
    resumed = _query(client, classification="alpha", cursor=cursor)
    assert resumed.status_code == 200
    assert [row["data_id"] for row in resumed.json()["envelopes"]] == ["m2"]

    # The same cursor under another classification, or with the filter
    # dropped, is an indistinguishable invalid cursor.
    for params in (
        {"classification": "beta"},
        {"classification": "unclassified"},
        {},
    ):
        response = _query(client, cursor=cursor, **params)
        assert response.status_code == 422
        assert response.json()["detail"] == "invalid cursor"


def test_filtered_snapshot_excludes_later_creations(client, monkeypatch):
    _create(client, "m0", classification="alpha")
    _create(client, "m1", classification="alpha")

    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 1)
    first = _query(client, classification="alpha").json()
    assert [row["data_id"] for row in first["envelopes"]] == ["m0"]

    # Created after the range began; its data_id sorts inside the range
    # but it lies beyond the fixed snapshot.
    _create(client, "m0a", classification="alpha")
    _create(client, "m2", classification="alpha")

    second = _query(client, classification="alpha",
                    cursor=first["next_cursor"]).json()
    assert [row["data_id"] for row in second["envelopes"]] == ["m1"]
    assert second["complete"] is True

    # A fresh query observes the new state.
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 100)
    fresh = _query(client, classification="alpha").json()["envelopes"]
    assert [row["data_id"] for row in fresh] == ["m0", "m0a", "m1", "m2"]


# --- idempotency ------------------------------------------------------------


def test_keyed_replay_requires_the_same_classification(app, client):
    first = _create(client, key="key-1", classification="alpha")
    assert first.status_code == 201

    replay = _create(client, key="key-1", classification="alpha")
    assert replay.status_code == 201
    assert replay.content == first.content

    # Same key with a different (or omitted) classification is a stable
    # 409 that changes nothing.
    for overrides in ({"classification": "beta"}, {}):
        response = _create(client, key="key-1", **overrides)
        assert response.status_code == 409
        assert response.json()["detail"] == (
            "idempotency key reused with a different request"
        )
    assert _count_rows(app, DataEnvelope) == 1
    assert _count_rows(app, DataEnvelopeIdempotencyRecord) == 1
    assert _scope_seq(app) == 1


def test_omitted_and_explicit_unclassified_share_one_identity(client):
    first = _create(client, key="key-1")
    assert first.status_code == 201
    replay = _create(client, key="key-1", classification="unclassified")
    assert replay.status_code == 201
    assert replay.content == first.content


def test_keyed_record_stores_the_effective_classification(app, client):
    response = _create(client, key="key-1", classification="alpha")
    assert response.status_code == 201
    with app.state.session_factory() as session:
        (record,) = session.query(DataEnvelopeIdempotencyRecord).all()
        assert record.classification == "alpha"
        assert json.loads(record.response_body)["classification"] == "alpha"
        envelope = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
        assert envelope.classification == "alpha"


# --- metadata never interferes with rewrap ----------------------------------


def test_rewrap_preserves_classification(app, client, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1)
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    created = _create(client, classification="alpha")
    assert created.status_code == 201
    assert created.json()["key_version"] == 1

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    rewrapped = client.post(
        f"/v1/data-envelopes/{DATA_ID}/rewrap",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert rewrapped.status_code == 200
    assert rewrapped.json()["key_version"] == 2

    detail = _detail(client).json()
    assert detail["key_version"] == 2
    assert detail["classification"] == "alpha"
    (entry,) = _query(client, classification="alpha").json()["envelopes"]
    assert entry["key_version"] == 2


# --- upgrade of pre-classification databases --------------------------------


def _build_legacy_database(url: str) -> None:
    """Create a current database, then strip the classification columns.

    The result matches a deployment written before the metadata existed:
    envelope and idempotency rows carry no classification at all.
    """
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE data_envelopes DROP COLUMN classification"))
        conn.execute(
            text(
                "ALTER TABLE data_envelope_idempotency_records "
                "DROP COLUMN classification"
            )
        )
        conn.execute(
            text(
                "INSERT INTO data_envelopes (tenant_id, workload_id, data_id, "
                "key_version, ciphertext, iv, tag, wrapped_key, created_at, "
                "commit_seq) VALUES ('t1', 'w1', 'legacy', 1, X'63', "
                "X'696969696969696969696969', "
                "X'74747474747474747474747474747474', "
                "X'6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b6b', "
                "'2026-01-01 00:00:00+00:00', 1)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO data_envelope_commit_counters "
                "(tenant_id, workload_id, last_seq) VALUES ('t1', 'w1', 1)"
            )
        )
        legacy_body = json.dumps(
            {
                "data_id": "legacy",
                "tenant_id": "t1",
                "workload_id": "w1",
                "key_version": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            separators=(",", ":"),
        )
        conn.execute(
            text(
                "INSERT INTO data_envelope_idempotency_records "
                "(record_id, tenant_id, workload_id, idempotency_key, "
                "data_id, payload_sha256, response_body, created_at) VALUES "
                "('rec-1', 't1', 'w1', 'oldkey', 'legacy', "
                ":digest, "
                ":body, '2026-01-01 00:00:00+00:00')"
            ),
            {
                "body": legacy_body,
                # SHA-256 of the payload "p" the replay below presents.
                "digest": hashlib.sha256(b"p").hexdigest(),
            },
        )
    engine.dispose()


def test_pre_classification_database_is_upgraded_on_open(tmp_path, monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/legacy.db"
    _build_legacy_database(url)

    application = app_module.create_app(url)
    client = TestClient(application)
    scope = {"tenant_id": "t1", "workload_id": "w1"}

    # Legacy rows read as unclassified everywhere.
    detail = client.get("/v1/data-envelopes/legacy", params=scope)
    assert detail.status_code == 200
    assert detail.json()["classification"] == "unclassified"
    (entry,) = client.get("/v1/data-envelopes", params=scope).json()["envelopes"]
    assert entry["classification"] == "unclassified"
    rows = client.get(
        "/v1/data-envelopes", params={**scope, "classification": "unclassified"}
    ).json()["envelopes"]
    assert [row["data_id"] for row in rows] == ["legacy"]
    page = client.get(
        "/v1/data-envelopes", params={**scope, "classification": "other"}
    ).json()
    assert page["envelopes"] == []

    # The legacy idempotency record belongs to an unclassified request:
    # its stored body (which has no classification field) is replayed
    # byte-for-byte, and a same-key request naming another classification
    # is a 409.
    body = {
        "tenant_id": "t1",
        "workload_id": "w1",
        "data_id": "legacy",
        "payload": "p",
    }
    replay = client.post(
        "/v1/data-envelopes", json=body, headers={"Idempotency-Key": "oldkey"}
    )
    assert replay.status_code == 201
    assert "classification" not in replay.json()
    explicit_default = client.post(
        "/v1/data-envelopes",
        json={**body, "classification": "unclassified"},
        headers={"Idempotency-Key": "oldkey"},
    )
    assert explicit_default.status_code == 201
    assert explicit_default.content == replay.content
    changed = client.post(
        "/v1/data-envelopes",
        json={**body, "classification": "secret"},
        headers={"Idempotency-Key": "oldkey"},
    )
    assert changed.status_code == 409

    # New creations after the upgrade allocate the next sequence and
    # carry their own classification.
    created = client.post(
        "/v1/data-envelopes",
        json={**body, "data_id": "new1", "classification": "secret"},
    )
    assert created.status_code == 201
    assert created.json()["classification"] == "secret"
    rows = client.get(
        "/v1/data-envelopes", params={**scope, "classification": "secret"}
    ).json()["envelopes"]
    assert [row["data_id"] for row in rows] == ["new1"]
    with application.state.session_factory() as session:
        counter = session.get(DataEnvelopeCommitCounter, ("t1", "w1"))
        assert counter.last_seq == 2
    application.state.engine.dispose()
