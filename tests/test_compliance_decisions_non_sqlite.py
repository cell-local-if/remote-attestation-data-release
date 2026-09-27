"""Non-SQLite coverage for the compliance decision sequence migration.

The decision listing's fixed snapshot is bounded by a per-scope commit
sequence, not by ``decided_at``. Databases written by a deployment before
decision scoping/sequencing must be upgraded on open on *every* backend:
scope columns added and copied from the evidence row, legacy decisions
numbered per scope in a stable order, the per-scope allocator seeded at
that maximum, and the unique ordering index built.

These tests run only against a real locking backend (PostgreSQL) because
SQLite is already covered by ``test_compliance_decisions.py`` and its
``rowid`` backfill path. They are skipped unless
``PROOF_RELEASE_TEST_PG_URL`` names a reachable PostgreSQL database, so
the default, dependency-free test run stays SQLite-only. The named
database is reset (all tables dropped) per test.
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from proof_release.app import _next_decision_commit_seq, create_app
from proof_release.db import Base, Decision

PG_URL = os.environ.get("PROOF_RELEASE_TEST_PG_URL")
pytestmark = pytest.mark.skipif(
    PG_URL is None,
    reason="set PROOF_RELEASE_TEST_PG_URL to run non-SQLite migration tests",
)

TENANT = "tA"
WORKLOAD = "w1"
SECRET = "unit-test-secret"

LEGACY_SQL = """
INSERT INTO evidence
(evidence_id, challenge_id, tenant_id, workload_id, evidence_format, status,
 received_at, evidence_sha256)
VALUES
('11111111-1111-4111-8111-111111111111','c1','tA','w1',
 'attested-nonce-json','verified','2026-01-01T00:00:00+00:00','aa'),
('22222222-2222-4222-8222-222222222222','c2','tA','w1',
 'attested-nonce-json','verified','2026-01-01T00:00:00+00:00','bb'),
('33333333-3333-4333-8333-333333333333','c3','tB','w9',
 'attested-nonce-json','verified','2026-01-01T00:00:00+00:00','cc');

INSERT INTO decisions
(decision_id, evidence_id, policy_id, policy_version, status, decided_at)
VALUES
('44444444-4444-4444-8444-444444444444',
 '11111111-1111-4111-8111-111111111111',
 '77777777-7777-4777-8777-777777777777', 1, 'allowed',
 '2026-01-01T00:00:03+00:00'),
('55555555-5555-4555-8555-555555555555',
 '22222222-2222-4222-8222-222222222222',
 '77777777-7777-4777-8777-777777777778', 1, 'denied',
 '2026-01-01T00:00:01+00:00'),
('66666666-6666-4666-8666-666666666666',
 '33333333-3333-4333-8333-333333333333',
 '77777777-7777-4777-8777-777777777779', 1, 'allowed',
 '2026-01-01T00:00:02+00:00');
"""


def _build_legacy_database(url: str) -> None:
    engine = create_engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX IF EXISTS ix_decisions_scope_commit_seq"))
        conn.execute(text("DROP INDEX IF EXISTS ix_decisions_scope_decided"))
        conn.execute(text("ALTER TABLE decisions DROP COLUMN IF EXISTS commit_seq"))
        conn.execute(text("ALTER TABLE decisions DROP COLUMN IF EXISTS tenant_id"))
        conn.execute(text("ALTER TABLE decisions DROP COLUMN IF EXISTS workload_id"))
        conn.execute(text("DROP TABLE IF EXISTS decision_commit_counters"))
        for statement in LEGACY_SQL.split(";"):
            if statement.strip():
                conn.execute(text(statement))
    engine.dispose()


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    _build_legacy_database(PG_URL)
    application = create_app(PG_URL)
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _query(client, **params):
    return client.get(
        "/v1/compliance/decisions",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD, **params},
    )


def test_open_backfills_legacy_scope_sequence_counter_and_indexes(app):
    with app.state.engine.begin() as conn:
        backfilled = {
            decision_id: (tenant, workload, seq)
            for decision_id, tenant, workload, seq in conn.execute(
                text(
                    "SELECT decision_id, tenant_id, workload_id, commit_seq "
                    "FROM decisions ORDER BY decision_id"
                )
            ).fetchall()
        }
        # On a locking backend the stable listing key orders the backfill:
        # the denied row has the older business time so it is sequence 1 in
        # tA even though it was inserted second; the other scope starts 1.
        assert backfilled["44444444-4444-4444-8444-444444444444"] == (
            "tA",
            "w1",
            2,
        )
        assert backfilled["55555555-5555-4555-8555-555555555555"] == (
            "tA",
            "w1",
            1,
        )
        assert backfilled["66666666-6666-4666-8666-666666666666"] == (
            "tB",
            "w9",
            1,
        )
        counters = dict(
            conn.execute(
                text(
                    "SELECT tenant_id || '/' || workload_id, last_seq "
                    "FROM decision_commit_counters"
                )
            ).fetchall()
        )
        assert counters == {"tA/w1": 2, "tB/w9": 1}
        indexdef = conn.execute(
            text(
                "SELECT indexdef FROM pg_indexes WHERE indexname = "
                "'ix_decisions_scope_commit_seq'"
            )
        ).scalar()
        assert indexdef is not None and "UNIQUE" in indexdef
        assert conn.execute(
            text(
                "SELECT count(*) FROM decisions WHERE commit_seq IS NULL "
                "OR tenant_id IS NULL OR workload_id IS NULL"
            )
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
                        "'ix_decisions_scope_commit_seq'"
                    )
                ).scalar()
                == 1
            )
            counters = dict(
                conn.execute(
                    text(
                        "SELECT tenant_id || '/' || workload_id, last_seq "
                        "FROM decision_commit_counters"
                    )
                ).fetchall()
            )
            assert counters == {"tA/w1": 2, "tB/w9": 1}
    finally:
        reopened.state.engine.dispose()


def test_legacy_query_reads_backfilled_decisions_in_business_order(client):
    response = _query(client)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["complete"] is True
    assert [row["decision_id"] for row in data["decisions"]] == [
        "55555555-5555-4555-8555-555555555555",
        "44444444-4444-4444-8444-444444444444",
    ]
    assert [row["status"] for row in data["decisions"]] == ["denied", "allowed"]


def test_new_decisions_continue_gap_free_after_upgrade(app, client):
    import proof_release.app as app_module

    app_module.DECISION_PAGE_SIZE = 1
    late_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    with app.state.session_factory() as session:
        seq = _next_decision_commit_seq(session, TENANT, WORKLOAD)
        assert seq == 3
        session.add(
            Decision(
                decision_id=late_id,
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                evidence_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                policy_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                policy_version=9,
                status="allowed",
                decided_at=datetime(2026, 1, 1, 0, 0, 2),
                commit_seq=seq,
            )
        )
        session.commit()
    data = _query(client).json()
    assert data["complete"] is True
    assert [row["decision_id"] for row in data["decisions"]] == [
        "55555555-5555-4555-8555-555555555555",
        late_id,
        "44444444-4444-4444-8444-444444444444",
    ]
