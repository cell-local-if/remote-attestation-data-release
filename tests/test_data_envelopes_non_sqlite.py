"""Non-SQLite coverage for the data-envelope commit sequence migration.

The read-only envelope directory fixes its replayable snapshot through a
per-scope, gap-free, strictly increasing ``commit_seq`` allocated in each
envelope's own creation transaction. The original ``data_envelopes``
table already carried the scope columns but not the sequence, so
databases written by a deployment before the directory existed must be
upgraded on open on *every* backend: the nullable column added, legacy
rows numbered per scope in the directory listing key (``data_id``) order,
the per-scope allocator seeded at that maximum, and the unique ordering
index built.

These tests run only against a real locking backend (PostgreSQL) because
SQLite is already covered by ``test_data_envelope_directory.py`` and its
``rowid`` backfill path. They are skipped unless
``PROOF_RELEASE_TEST_PG_URL`` names a reachable PostgreSQL database, so
the default, dependency-free test run stays SQLite-only. The named
database is reset (all tables dropped) per test.
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from proof_release.app import create_app
from proof_release.db import Base
from proof_release.envelopes import b64url_encode

PG_URL = os.environ.get("PROOF_RELEASE_TEST_PG_URL")
pytestmark = pytest.mark.skipif(
    PG_URL is None,
    reason="set PROOF_RELEASE_TEST_PG_URL to run non-SQLite migration tests",
)

TENANT = "tA"
WORKLOAD = "w1"
OTHER_TENANT = "tB"

KEY = b64url_encode(b"0123456789abcdef0123456789abcdef")

# 12-byte iv, 16-byte tag and 40-byte wrapped key as bytea hex literals.
_IV = "\\x" + "69" * 12
_TAG = "\\x" + "74" * 16
_WRAPPED = "\\x" + "6b" * 40

_LEGACY_ROWS = [
    ("tB", "w1", "only", "2026-01-01T00:00:00+00:00"),
    ("tA", "w1", "charlie", "2026-01-01T00:00:02+00:00"),
    ("tA", "w1", "alpha", "2026-01-01T00:00:00+00:00"),
    ("tA", "w1", "bravo", "2026-01-01T00:00:01+00:00"),
]
LEGACY_SQL = (
    "INSERT INTO data_envelopes "
    "(tenant_id, workload_id, data_id, key_version, ciphertext, iv, tag, "
    "wrapped_key, created_at) VALUES "
    + ",".join(
        f"('{t}','{w}','{d}',1,'\\x63','{_IV}','{_TAG}','{_WRAPPED}','{ts}')"
        for t, w, d, ts in _LEGACY_ROWS
    )
)


def _build_legacy_database(url: str) -> None:
    engine = create_engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text("DROP INDEX IF EXISTS ix_data_envelopes_scope_commit_seq")
        )
        conn.execute(text("ALTER TABLE data_envelopes DROP COLUMN IF EXISTS commit_seq"))
        conn.execute(text("DROP TABLE IF EXISTS data_envelope_commit_counters"))
        for statement in LEGACY_SQL.split(";"):
            if statement.strip():
                conn.execute(text(statement))
    engine.dispose()


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv(
        "PROOF_RELEASE_KEYRING",
        json.dumps({"current_version": 1, "keys": {"1": KEY}}),
    )
    _build_legacy_database(PG_URL)
    application = create_app(PG_URL)
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def test_open_backfills_legacy_sequence_counter_and_index(app):
    with app.state.engine.begin() as conn:
        backfilled = dict(
            conn.execute(
                text(
                    "SELECT data_id, commit_seq FROM data_envelopes "
                    "WHERE tenant_id = 'tA' AND workload_id = 'w1' "
                    "ORDER BY data_id"
                )
            ).fetchall()
        )
        # On a locking backend the directory listing key (data_id) orders
        # the backfill regardless of the rows' business timestamps.
        assert backfilled == {"alpha": 1, "bravo": 2, "charlie": 3}
        other = conn.execute(
            text(
                "SELECT commit_seq FROM data_envelopes "
                "WHERE tenant_id = 'tB'"
            )
        ).scalar()
        assert other == 1
        counters = dict(
            conn.execute(
                text(
                    "SELECT tenant_id || '/' || workload_id, last_seq "
                    "FROM data_envelope_commit_counters"
                )
            ).fetchall()
        )
        assert counters == {"tA/w1": 3, "tB/w1": 1}
        indexdef = conn.execute(
            text(
                "SELECT indexdef FROM pg_indexes WHERE indexname = "
                "'ix_data_envelopes_scope_commit_seq'"
            )
        ).scalar()
        assert indexdef is not None and "UNIQUE" in indexdef
        assert conn.execute(
            text("SELECT count(*) FROM data_envelopes WHERE commit_seq IS NULL")
        ).scalar() == 0


def test_open_is_idempotent_across_restart(app):
    app.state.engine.dispose()
    reopened = create_app(PG_URL)
    try:
        with reopened.state.engine.begin() as conn:
            assert (
                conn.execute(
                    text(
                        "SELECT count(*) FROM pg_indexes WHERE indexname = "
                        "'ix_data_envelopes_scope_commit_seq'"
                    )
                ).scalar()
                == 1
            )
            counters = dict(
                conn.execute(
                    text(
                        "SELECT tenant_id || '/' || workload_id, last_seq "
                        "FROM data_envelope_commit_counters"
                    )
                ).fetchall()
            )
            assert counters == {"tA/w1": 3, "tB/w1": 1}
    finally:
        reopened.state.engine.dispose()


def test_creation_after_upgrade_allocates_next_sequence_and_directory_lists(client):
    created = client.post(
        "/v1/data-envelopes",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "data_id": "delta",
            "payload": "p",
        },
    )
    assert created.status_code == 201, created.text

    listing = client.get(
        "/v1/data-envelopes",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert listing.status_code == 200
    assert [e["data_id"] for e in listing.json()["envelopes"]] == [
        "alpha",
        "bravo",
        "charlie",
        "delta",
    ]
    with client.app.state.engine.begin() as conn:
        assert conn.execute(
            text(
                "SELECT last_seq FROM data_envelope_commit_counters "
                "WHERE tenant_id = 'tA' AND workload_id = 'w1'"
            )
        ).scalar() == 4
