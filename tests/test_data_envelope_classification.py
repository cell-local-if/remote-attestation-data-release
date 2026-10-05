"""Tests for the classification metadata on data envelopes.

POST /v1/data-envelopes accepts an optional ``classification``: 1..32
lowercase ASCII characters, the first a letter, the rest letters,
digits, underscores or hyphens, defaulting to ``unclassified``. It is
pure metadata — it never participates in encryption and never changes
the ciphertext, IV, tag, wrapped key, key version, creation time or
data_id uniqueness — and it is returned by the detail and directory
queries, where an optional exact-match filter narrows the listing. The
classification joins the idempotency request identity, and envelopes
and idempotency records written before the field existed are treated
as ``unclassified``.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from proof_release import app as app_module
from proof_release.db import (
    DataEnvelope,
    DataEnvelopeCommitCounter,
    DataEnvelopeIdempotencyRecord,
)
from proof_release.envelopes import b64url_decode, b64url_encode
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
DATA_ID = "item-1"
PAYLOAD = "super-secret-attestation-payload 🔐"

MASTER_KEY_BYTES = b"0123456789abcdef0123456789abcdef"
MASTER_KEY = b64url_encode(MASTER_KEY_BYTES)
KEY_V2 = b64url_encode(b"22222222222222222222222222222222")
KEYRING_V1_V2 = json.dumps(
    {"current_version": 2, "keys": {"1": MASTER_KEY, "2": KEY_V2}}
)

IDEMPOTENCY_HEADER = "Idempotency-Key"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    application = app_module.create_app(f"sqlite:///{tmp_path}/classification.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _create(client, data_id=DATA_ID, *, key=None, **overrides):
    body = {
        "tenant_id": TENANT,
        "workload_id": WORKLOAD,
        "data_id": data_id,
        "payload": PAYLOAD,
    }
    body.update(overrides)
    headers = {IDEMPOTENCY_HEADER: key} if key is not None else None
    return client.post("/v1/data-envelopes", json=body, headers=headers)


def _get(client, data_id=DATA_ID, *, tenant=TENANT, workload=WORKLOAD):
    return client.get(
        f"/v1/data-envelopes/{data_id}",
        params={"tenant_id": tenant, "workload_id": workload},
    )


def _query(client, *, tenant=TENANT, workload=WORKLOAD, **params):
    query = {"tenant_id": tenant, "workload_id": workload}
    query.update(params)
    return client.get("/v1/data-envelopes", params=query)


def _count(app, model) -> int:
    with app.state.session_factory() as session:
        return session.query(model).count()


def _decrypt(master_key: bytes, fields: dict) -> bytes:
    data_key = aes_key_unwrap(master_key, b64url_decode(fields["wrapped_key"]))
    iv = b64url_decode(fields["iv"])
    sealed = b64url_decode(fields["ciphertext"]) + b64url_decode(fields["tag"])
    return AESGCM(data_key).decrypt(iv, sealed, None)


# --- creation defaults and round trip ---------------------------------------


def test_default_classification_is_unclassified_on_every_read(client):
    created = _create(client)
    assert created.status_code == 201
    assert created.json()["classification"] == "unclassified"

    detail = _get(client)
    assert detail.status_code == 200
    assert detail.json()["classification"] == "unclassified"

    (entry,) = _query(client).json()["envelopes"]
    assert entry["classification"] == "unclassified"


def test_explicit_classification_round_trips_through_every_read(client):
    created = _create(client, classification="confidential_1")
    assert created.status_code == 201
    assert created.json()["classification"] == "confidential_1"
    assert _get(client).json()["classification"] == "confidential_1"
    (entry,) = _query(client).json()["envelopes"]
    assert entry["classification"] == "confidential_1"


@pytest.mark.parametrize(
    "value",
    [
        "a",  # single letter
        "a" * 32,  # maximum length
        "unclassified",
        "z0",
        "a_0-9_b",
        "q" * 31 + "-",
    ],
)
def test_boundary_and_alphabet_values_are_accepted(client, value):
    response = _create(client, classification=value)
    assert response.status_code == 201, response.text
    assert response.json()["classification"] == value


def test_classification_is_stored_on_the_row(app, client):
    assert _create(client, classification="secret").status_code == 201
    with app.state.session_factory() as session:
        record = session.get(DataEnvelope, (TENANT, WORKLOAD, DATA_ID))
    assert record.classification == "secret"


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "",  # empty
        " ",  # blank
        "Unclassified",  # uppercase first letter
        "UNCLASSIFIED",
        "aB",  # embedded uppercase
        "1abc",  # leading digit
        "_abc",  # leading underscore
        "-abc",  # leading hyphen
        "a b",  # embedded space
        "a!b",  # punctuation
        "a.b",
        "a/b",
        "éclair",  # non-ASCII
        "a" * 33,  # over-long
        42,  # not a string
        True,
        None,
    ],
)
def test_invalid_classification_is_422_and_writes_nothing(app, client, value):
    response = _create(client, classification=value)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid classification"}
    assert PAYLOAD not in response.text
    assert _count(app, DataEnvelope) == 0
    assert _count(app, DataEnvelopeCommitCounter) == 0


@pytest.mark.parametrize("value", ["", "Bad", "a b", "a" * 33])
def test_invalid_classification_with_key_writes_no_record(app, client, value):
    response = _create(client, key="key-1", classification=value)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid classification"}
    assert _count(app, DataEnvelope) == 0
    assert _count(app, DataEnvelopeIdempotencyRecord) == 0
    assert _count(app, DataEnvelopeCommitCounter) == 0
    # The failed judgement consumed nothing: the key is still free.
    assert _create(client, key="key-1").status_code == 201


def test_classification_does_not_change_encryption_or_uniqueness(client):
    first = _create(client, "item-a", classification="alpha")
    second = _create(client, "item-b", classification="beta")
    assert first.status_code == 201 and second.status_code == 201
    assert first.json()["key_version"] == second.json()["key_version"] == 1

    one = _get(client, "item-a").json()
    two = _get(client, "item-b").json()
    # Fresh randomness per envelope regardless of classification, and
    # both decrypt to the exact payload under the same master key.
    assert one["iv"] != two["iv"]
    assert one["wrapped_key"] != two["wrapped_key"]
    assert _decrypt(MASTER_KEY_BYTES, one) == PAYLOAD.encode("utf-8")
    assert _decrypt(MASTER_KEY_BYTES, two) == PAYLOAD.encode("utf-8")
    # The classification string never leaks into the material fields.
    for fields, name in ((one, "alpha"), (two, "beta")):
        for field in ("ciphertext", "iv", "tag", "wrapped_key"):
            assert name.encode() not in b64url_decode(fields[field])

    # data_id uniqueness is unaffected: same data_id, new classification.
    duplicate = _create(client, "item-a", classification="gamma")
    assert duplicate.status_code == 409


# --- directory filter ---------------------------------------------------------


def test_classification_filter_returns_exact_matches_in_order(client):
    _create(client, "a-first", classification="secret")
    _create(client, "b-second", classification="public")
    _create(client, "c-third", classification="secret")
    _create(client, "d-fourth")  # unclassified

    rows = _query(client, classification="secret").json()["envelopes"]
    assert [r["data_id"] for r in rows] == ["a-first", "c-third"]
    assert all(r["classification"] == "secret" for r in rows)

    rows = _query(client, classification="public").json()["envelopes"]
    assert [r["data_id"] for r in rows] == ["b-second"]

    rows = _query(client, classification="unclassified").json()["envelopes"]
    assert [r["data_id"] for r in rows] == ["d-fourth"]

    # No filter still returns everything, ordered by data_id.
    rows = _query(client).json()["envelopes"]
    assert [r["data_id"] for r in rows] == [
        "a-first",
        "b-second",
        "c-third",
        "d-fourth",
    ]


def test_classification_filter_with_no_matches_is_an_empty_page(client):
    _create(client, "item", classification="secret")
    response = _query(client, classification="public")
    assert response.status_code == 200
    assert response.json() == {
        "envelopes": [],
        "next_cursor": "",
        "complete": True,
    }


def test_classification_filter_is_scoped(client):
    _create(client, "own", classification="secret")
    _create(client, "other", tenant_id=OTHER_TENANT, classification="secret")

    rows = _query(client, classification="secret").json()["envelopes"]
    assert [r["data_id"] for r in rows] == ["own"]
    rows = _query(client, tenant=OTHER_TENANT, classification="secret").json()[
        "envelopes"
    ]
    assert [r["data_id"] for r in rows] == ["other"]


def test_classification_filter_composes_with_data_id_and_window(client):
    created = _create(client, "item", classification="secret")
    instant = created.json()["created_at"]

    # Matching classification plus an explicit data_id finds the row.
    rows = _query(client, data_id="item", classification="secret").json()[
        "envelopes"
    ]
    assert [r["data_id"] for r in rows] == ["item"]

    # A non-matching classification filters the named row out: the range
    # is empty, not a 404 (the data_id itself exists).
    response = _query(client, data_id="item", classification="public")
    assert response.status_code == 200
    assert response.json()["envelopes"] == []

    # The filter composes with the inclusive creation-time window.
    rows = _query(
        client,
        classification="secret",
        created_after=instant,
        created_before=instant,
    ).json()["envelopes"]
    assert [r["data_id"] for r in rows] == ["item"]
    assert (
        _query(
            client,
            classification="secret",
            created_after="2030-01-01T00:00:00Z",
        ).json()["envelopes"]
        == []
    )


@pytest.mark.parametrize(
    "value", ["", " ", "Secret", "SECRET", "a b", "a!b", "1abc", "a" * 33]
)
def test_invalid_classification_filter_is_422(client, value):
    _create(client, "item", classification="secret")
    response = _query(client, classification=value)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid classification"}


def test_repeated_classification_parameter_is_422(client):
    response = client.get(
        "/v1/data-envelopes",
        params=[
            ("tenant_id", TENANT),
            ("workload_id", WORKLOAD),
            ("classification", "a"),
            ("classification", "b"),
        ],
    )
    assert response.status_code == 422


# --- cursor binding and snapshot ----------------------------------------------


def test_cursor_is_bound_to_the_classification_filter(client, monkeypatch):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 1)
    _create(client, "a", classification="secret")
    _create(client, "b", classification="secret")

    first = _query(client, classification="secret").json()
    assert [r["data_id"] for r in first["envelopes"]] == ["a"]
    assert not first["complete"]
    token = first["next_cursor"]

    # The same filter resumes the range.
    second = _query(client, classification="secret", cursor=token).json()
    assert [r["data_id"] for r in second["envelopes"]] == ["b"]
    assert second["complete"]

    # The cursor replayed against any other classification filter — or
    # against no filter — is an indistinguishable 422.
    for params in (
        {"classification": "public"},
        {"classification": "unclassified"},
        {},
    ):
        response = _query(client, cursor=token, **params)
        assert response.status_code == 422
        assert response.json() == {"detail": "invalid cursor"}


def test_unfiltered_cursor_is_rejected_under_a_classification_filter(
    client, monkeypatch
):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 1)
    _create(client, "a", classification="secret")
    _create(client, "b", classification="public")

    token = _query(client).json()["next_cursor"]
    assert token
    response = _query(client, classification="secret", cursor=token)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid cursor"}


def test_filtered_snapshot_excludes_later_creations_without_gaps(
    client, monkeypatch
):
    monkeypatch.setattr(app_module, "DATA_ENVELOPE_PAGE_SIZE", 2)
    for data_id in ("a", "b", "c"):
        _create(client, data_id, classification="secret")

    first = _query(client, classification="secret").json()
    assert [r["data_id"] for r in first["envelopes"]] == ["a", "b"]

    # Created after the range's snapshot was fixed; even though its
    # data_id sorts inside the range it never enters a replayed page.
    _create(client, "aa", classification="secret")

    second = _query(
        client, classification="secret", cursor=first["next_cursor"]
    ).json()
    assert [r["data_id"] for r in second["envelopes"]] == ["c"]
    assert second["complete"]

    walked = [r["data_id"] for r in first["envelopes"] + second["envelopes"]]
    assert walked == ["a", "b", "c"]
    assert len(set(walked)) == 3

    # A fresh query sees the full current membership.
    fresh = _query(client, classification="secret").json()
    assert [r["data_id"] for r in fresh["envelopes"]] == ["a", "aa"]


# --- idempotency ----------------------------------------------------------------


def test_keyed_create_replays_classification_byte_for_byte(app, client):
    first = _create(client, key="key-1", classification="secret")
    assert first.status_code == 201
    assert first.json()["classification"] == "secret"

    (record,) = _records(app)
    assert record.classification == "secret"
    assert json.loads(record.response_body)["classification"] == "secret"

    replay = _create(client, key="key-1", classification="secret")
    assert replay.status_code == 201
    assert replay.content == first.content
    assert _count(app, DataEnvelope) == 1
    assert _count(app, DataEnvelopeIdempotencyRecord) == 1


def _records(app):
    with app.state.session_factory() as session:
        return session.query(DataEnvelopeIdempotencyRecord).all()


def test_same_key_different_classification_is_409_and_changes_nothing(
    app, client
):
    first = _create(client, key="key-1", classification="secret")
    assert first.status_code == 201

    for overrides in (
        {"classification": "public"},
        {"classification": "other-secret"},
        {},  # omitted: the default is a different request identity
    ):
        conflict = _create(client, key="key-1", **overrides)
        assert conflict.status_code == 409
        assert conflict.json() == {
            "detail": "idempotency key reused with a different request"
        }

    assert _count(app, DataEnvelope) == 1
    assert _count(app, DataEnvelopeIdempotencyRecord) == 1
    # The original request still replays its first response.
    replay = _create(client, key="key-1", classification="secret")
    assert replay.content == first.content


def test_explicit_unclassified_and_omitted_are_the_same_identity(app, client):
    first = _create(client, key="key-1")
    assert first.status_code == 201
    assert first.json()["classification"] == "unclassified"

    replay = _create(client, key="key-1", classification="unclassified")
    assert replay.status_code == 201
    assert replay.content == first.content
    assert _count(app, DataEnvelope) == 1


# --- legacy rows and records ----------------------------------------------------


def _drop_column(app, table):
    with app.state.engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table} DROP COLUMN classification"))


def test_envelopes_from_before_classification_are_unclassified(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/legacy-envelopes.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    assert _create(client1, "legacy").status_code == 201
    # Simulate a pre-classification database: the column is gone, so no
    # row can carry a classification.
    _drop_column(app1, "data_envelopes")
    app1.state.engine.dispose()

    # Reopening upgrades the database: the column returns and legacy rows
    # are backfilled to unclassified.
    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    try:
        assert _get(client2, "legacy").json()["classification"] == (
            "unclassified"
        )
        rows = _query(client2, classification="unclassified").json()["envelopes"]
        assert [r["data_id"] for r in rows] == ["legacy"]
        assert _query(client2, classification="secret").json()["envelopes"] == []

        # Envelopes created after the upgrade carry their classification
        # alongside the backfilled legacy rows.
        assert (
            _create(client2, "modern", classification="secret").status_code
            == 201
        )
        assert _get(client2, "modern").json()["classification"] == "secret"
        rows = _query(client2, classification="secret").json()["envelopes"]
        assert [r["data_id"] for r in rows] == ["modern"]
        rows = _query(client2).json()["envelopes"]
        assert [r["data_id"] for r in rows] == ["legacy", "modern"]
    finally:
        app2.state.engine.dispose()


def test_legacy_idempotency_record_replays_its_stored_body_verbatim(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROOF_RELEASE_MASTER_KEY", MASTER_KEY)
    url = f"sqlite:///{tmp_path}/legacy-records.db"
    app1 = app_module.create_app(url)
    client1 = TestClient(app1)
    first = _create(client1, key="legacy-key")
    assert first.status_code == 201

    # Simulate a pre-classification record: no classification column, and
    # the stored first response names no classification.
    with app1.state.engine.begin() as conn:
        stored = json.loads(first.content)
        del stored["classification"]
        legacy_body = json.dumps(
            stored, separators=(",", ":"), ensure_ascii=False
        )
        conn.execute(
            text(
                "UPDATE data_envelope_idempotency_records "
                "SET response_body = :body"
            ),
            {"body": legacy_body},
        )
    _drop_column(app1, "data_envelope_idempotency_records")
    app1.state.engine.dispose()

    app2 = app_module.create_app(url)
    client2 = TestClient(app2)
    try:
        # The upgraded record carries no classification (NULL), treated as
        # unclassified: an omitted or explicit-unclassified replay matches
        # and returns the legacy stored body byte-for-byte — without a
        # classification field.
        for overrides in ({}, {"classification": "unclassified"}):
            replay = _create(client2, key="legacy-key", **overrides)
            assert replay.status_code == 201
            assert replay.content.decode("utf-8") == legacy_body
            assert "classification" not in replay.json()

        # A different classification is a different request identity.
        conflict = _create(client2, key="legacy-key", classification="secret")
        assert conflict.status_code == 409
        assert conflict.json() == {
            "detail": "idempotency key reused with a different request"
        }
        assert _count(app2, DataEnvelope) == 1
        assert _count(app2, DataEnvelopeIdempotencyRecord) == 1
    finally:
        app2.state.engine.dispose()


# --- classification is inert elsewhere -----------------------------------------


def test_rewrap_preserves_classification(client, monkeypatch):
    assert _create(client, classification="secret").status_code == 201

    monkeypatch.setenv("PROOF_RELEASE_KEYRING", KEYRING_V1_V2)
    monkeypatch.delenv("PROOF_RELEASE_MASTER_KEY", raising=False)
    rewrap = client.post(
        f"/v1/data-envelopes/{DATA_ID}/rewrap",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert rewrap.status_code == 200
    assert rewrap.json()["key_version"] == 2
    assert "classification" not in rewrap.json()

    detail = _get(client).json()
    assert detail["classification"] == "secret"
    assert detail["key_version"] == 2
    (entry,) = _query(client, classification="secret").json()["envelopes"]
    assert entry["key_version"] == 2
