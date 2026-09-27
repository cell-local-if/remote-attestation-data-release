"""Non-SQLite coverage for the proof-lifecycle commit-sequence migration.

The proof timeline's fixed snapshot is bounded by a per-scope commit
sequence, not by ``occurred_at``. Databases written by a deployment
before the sequence existed must be upgraded on open on *every* backend:
the column added, legacy rows numbered per scope in a stable order, the
per-scope allocator seeded at that maximum, and the unique ordering
index built. These tests run only against a real locking backend
(PostgreSQL) because SQLite is already covered by
``test_proof_lifecycle_events.py`` and its ``rowid`` backfill path.

They are skipped unless ``PROOF_RELEASE_TEST_PG_URL`` names a reachable
PostgreSQL database (a SQLAlchemy psycopg2 URL, e.g.
``postgresql+psycopg2://user:pass@localhost/proofrel_test``), so the
default, dependency-free test run stays SQLite-only. The named database
is reset (all tables dropped) per test.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from proof_release.app import (
    _next_proof_event_commit_seq,
    create_app,
)
from proof_release.db import Base, ProofLifecycleEvent

PG_URL = os.environ.get("PROOF_RELEASE_TEST_PG_URL")
pytestmark = pytest.mark.skipif(
    PG_URL is None,
    reason="set PROOF_RELEASE_TEST_PG_URL to run non-SQLite migration tests",
)

TENANT = "tA"
WORKLOAD = "w1"
SECRET = "unit-test-secret"

LEGACY_SQL = """
INSERT INTO proof_lifecycle_events
(event_id, tenant_id, workload_id, event_type, evidence_id,
 policy_version, evidence_format, status, occurred_at)
VALUES
('11111111-1111-4111-8111-111111111111','tA','w1',
 'proof-received','11111111-1111-4111-8111-111111111111',
 NULL,'f','received','2026-01-01T00:00:03+00:00'),
('22222222-2222-4222-8222-222222222222','tA','w1',
 'proof-verified','11111111-1111-4111-8111-111111111111',
 NULL,NULL,'verified','2026-01-01T00:00:01+00:00'),
('33333333-3333-4333-8333-333333333333','tB','w9',
 'proof-received','33333333-3333-4333-8333-333333333333',
 NULL,'f','received','2026-01-01T00:00:02+00:00')
"""


def _build_legacy_database(url: str) -> None:
    """Create a database that looks like a pre-sequence deployment."""
    engine = create_engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "DROP INDEX IF EXISTS "
                "ix_proof_lifecycle_events_scope_commit_seq"
            )
        )
        conn.execute(
            text("ALTER TABLE proof_lifecycle_events DROP COLUMN commit_seq")
        )
        conn.execute(text("DROP TABLE IF EXISTS proof_event_commit_counters"))
        conn.execute(text(LEGACY_SQL))
    engine.dispose()


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("PROOF_RELEASE_ATTESTED_NONCE_SECRET", SECRET)
    monkeypatch.delenv("PROOF_RELEASE_KEYRING", raising=False)
    monkeypatch.setenv(
        "PROOF_RELEASE_MASTER_KEY",
        # unpadded base64url of 32 zero-free bytes (matches the other suites)
        "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY",
    )
    _build_legacy_database(PG_URL)
    application = create_app(PG_URL)
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


def _query(client, **params):
    return client.get(
        "/v1/compliance/proof-events",
        params={"tenant_id": TENANT, "workload_id": WORKLOAD, **params},
    )


# --- migration on open -----------------------------------------------------


def test_open_backfills_legacy_rows_counter_and_unique_index(app):
    with app.state.engine.begin() as conn:
        backfilled = dict(
            conn.execute(
                text(
                    "SELECT event_id, commit_seq FROM proof_lifecycle_events "
                    "ORDER BY event_id"
                )
            ).fetchall()
        )
        # Ranked by the audit listing key per scope: the verified row has
        # the older business time so it is sequence 1 in tA even though it
        # was inserted second; the other scope restarts at 1.
        assert backfilled == {
            "11111111-1111-4111-8111-111111111111": 2,
            "22222222-2222-4222-8222-222222222222": 1,
            "33333333-3333-4333-8333-333333333333": 1,
        }
        counters = dict(
            conn.execute(
                text(
                    "SELECT tenant_id || '/' || workload_id, last_seq "
                    "FROM proof_event_commit_counters"
                )
            ).fetchall()
        )
        assert counters == {"tA/w1": 2, "tB/w9": 1}
        assert (
            conn.execute(
                text(
                    "SELECT count(*) FROM proof_lifecycle_events "
                    "WHERE commit_seq IS NULL"
                )
            ).scalar()
            == 0
        )
        indexdef = conn.execute(
            text(
                "SELECT indexdef FROM pg_indexes WHERE indexname = "
                "'ix_proof_lifecycle_events_scope_commit_seq'"
            )
        ).scalar()
        assert indexdef is not None and "UNIQUE" in indexdef


def test_open_is_idempotent_across_restart(app):
    # Opening the already-upgraded database a second time changes nothing.
    app.state.engine.dispose()
    reopened = create_app(PG_URL)
    try:
        with reopened.state.engine.begin() as conn:
            assert (
                conn.execute(
                    text(
                        "SELECT count(*) FROM pg_indexes WHERE indexname = "
                        "'ix_proof_lifecycle_events_scope_commit_seq'"
                    )
                ).scalar()
                == 1
            )
            counters = dict(
                conn.execute(
                    text(
                        "SELECT tenant_id || '/' || workload_id, last_seq "
                        "FROM proof_event_commit_counters"
                    )
                ).fetchall()
            )
            assert counters == {"tA/w1": 2, "tB/w9": 1}
    finally:
        reopened.state.engine.dispose()


def test_legacy_query_reads_backfilled_events_in_business_order(client):
    response = _query(client)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["complete"] is True
    assert [row["event_type"] for row in data["events"]] == [
        "proof-verified",
        "proof-received",
    ]


# --- live sequence after the upgrade --------------------------------------


def test_new_events_continue_gap_free_after_upgrade(app, client):
    _full_proof(client)
    with app.state.engine.begin() as conn:
        seqs = [
            row[0]
            for row in conn.execute(
                text(
                    "SELECT commit_seq FROM proof_lifecycle_events "
                    "WHERE tenant_id = :t AND workload_id = :w "
                    "ORDER BY commit_seq"
                ),
                {"t": TENANT, "w": WORKLOAD},
            ).fetchall()
        ]
        assert seqs == [1, 2, 3, 4, 5]
        counter = conn.execute(
            text(
                "SELECT last_seq FROM proof_event_commit_counters "
                "WHERE tenant_id = :t AND workload_id = :w"
            ),
            {"t": TENANT, "w": WORKLOAD},
        ).scalar()
        assert counter == 5


def test_concurrent_allocation_is_gap_free(app):
    results: list[int] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        try:
            with app.state.session_factory() as session:
                barrier.wait()
                seq = _next_proof_event_commit_seq(
                    session, "fresh-scope", "fresh-workload"
                )
                session.commit()
            results.append(seq)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(results) == list(range(1, 9))
    with app.state.engine.begin() as conn:
        assert (
            conn.execute(
                text(
                    "SELECT last_seq FROM proof_event_commit_counters "
                    "WHERE tenant_id = 'fresh-scope'"
                )
            ).scalar()
            == 8
        )


# --- snapshot fixed to the commit boundary ---------------------------------


def test_late_commit_with_older_business_time_keeps_old_snapshot(
    app, client, monkeypatch
):
    _full_proof(client)  # three events at sequences 3,4,5
    import proof_release.app as app_module

    monkeypatch.setattr(app_module, "PROOF_EVENT_PAGE_SIZE", 2)

    first = _query(client).json()
    assert len(first["events"]) == 2 and first["complete"] is False
    cursor = first["next_cursor"]

    t0 = datetime.fromisoformat(first["events"][0]["occurred_at"])
    t1 = datetime.fromisoformat(first["events"][1]["occurred_at"])
    backdated = t0 + (t1 - t0) / 2
    late_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    with app.state.session_factory() as session:
        late_seq = _next_proof_event_commit_seq(session, TENANT, WORKLOAD)
        assert late_seq == 6
        session.add(
            ProofLifecycleEvent(
                event_id=late_id,
                tenant_id=TENANT,
                workload_id=WORKLOAD,
                event_type="proof-received",
                evidence_id=late_id,
                commit_seq=late_seq,
                policy_version=None,
                evidence_format="attested-nonce-json",
                status="received",
                occurred_at=backdated,
            )
        )
        session.commit()

    # The fixed snapshot's remaining pages exclude the late commit even
    # though its business time sorts inside the already-returned window.
    fixed_ids = {row["event_id"] for row in first["events"]}
    token = cursor
    second = None
    for _ in range(10):
        page = _query(client, cursor=token).json()
        if second is None:
            second = page
        fixed_ids.update(row["event_id"] for row in page["events"])
        if page["complete"]:
            break
        token = page["next_cursor"]
    assert page["complete"] is True
    assert len(fixed_ids) == 5
    assert late_id not in fixed_ids
    assert _query(client, cursor=cursor).json() == second

    # A fresh first query observes it exactly once, in business-time order.
    monkeypatch.setattr(app_module, "PROOF_EVENT_PAGE_SIZE", 100)
    fresh = _query(client).json()
    assert fresh["complete"] is True and len(fresh["events"]) == 6
    ids = [row["event_id"] for row in fresh["events"]]
    assert ids.count(late_id) == 1
    keys = [(row["occurred_at"], row["event_id"]) for row in fresh["events"]]
    assert keys == sorted(keys)


# --- API proof builder -----------------------------------------------------


def _mac_for(nonce: str, claims: dict) -> str:
    key = hmac.new(
        SECRET.encode("utf-8"),
        f"{TENANT}:{WORKLOAD}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    payload = json.dumps(
        {"claims": claims, "nonce": nonce},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _full_proof(client) -> tuple[str, str, str]:
    """Receive, verify and decide one proof over the API; return its ids."""
    created = client.post(
        "/v1/challenges",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    ).json()
    evidence = json.dumps(
        {
            "nonce": created["nonce"],
            "claims": {"m": "x"},
            "mac": _mac_for(created["nonce"], {"m": "x"}),
        }
    )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": "attested-nonce-json",
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201, submitted.text
    evidence_id = submitted.json()["evidence_id"]
    verified = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert verified.status_code == 200, verified.text
    policy = client.post(
        "/v1/policies",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "name": f"r-{evidence_id[:8]}",
            "rule": {"claim": "m", "equals": "x"},
        },
    )
    assert policy.status_code == 201, policy.text
    decided = client.post(
        f"/v1/evidence/{evidence_id}/decisions",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
            "policy_id": policy.json()["policy_id"],
        },
    )
    assert decided.status_code == 200, decided.text
    return created, evidence, evidence_id
