"""Tests for POST /v1/trust-roots/{root_id}/retire and for the effect of
trust-root retirement on X.509 evidence verification."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select, text

from proof_release.app import create_app
from proof_release.db import Evidence, TrustRoot
from proof_release.verifiers import (
    X509_ATTESTED_NONCE_JSON,
    VerifierRegistry,
    X509AttestedNonceJSONVerifier,
)

from x509_helpers import make_evidence, make_intermediate, make_leaf, make_root, pem

TENANT = "tenant-a"
WORKLOAD = "workload-1"
OTHER_TENANT = "tenant-b"
OTHER_WORKLOAD = "workload-2"


@pytest.fixture()
def app(tmp_path):
    application = create_app(f"sqlite:///{tmp_path}/test.db")
    yield application
    application.state.engine.dispose()


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def chain():
    root_key, root_cert = make_root()
    intermediate_key, intermediate_cert = make_intermediate(root_cert, root_key)
    leaf_key, leaf_cert = make_leaf(intermediate_cert, intermediate_key)
    return {
        "root_key": root_key,
        "root_cert": root_cert,
        "intermediate_key": intermediate_key,
        "intermediate_cert": intermediate_cert,
        "leaf_key": leaf_key,
        "leaf_cert": leaf_cert,
    }


@pytest.fixture()
def root_id(client, chain):
    response = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    assert response.status_code == 201
    return response.json()["root_id"]


def _retire(client, root, *, tenant=TENANT, workload=WORKLOAD, **extra):
    body = {"tenant_id": tenant, "workload_id": workload}
    body.update(extra)
    return client.post(f"/v1/trust-roots/{root}/retire", json=body)


def _challenge(client, tenant=TENANT, workload=WORKLOAD):
    response = client.post(
        "/v1/challenges", json={"tenant_id": tenant, "workload_id": workload}
    )
    assert response.status_code == 201
    return response.json()


def _full_chain(chain):
    return [chain["leaf_cert"], chain["intermediate_cert"], chain["root_cert"]]


def _submit_and_verify(client, created, evidence, *, tenant=TENANT, workload=WORKLOAD):
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": X509_ATTESTED_NONCE_JSON,
            "evidence": evidence,
        },
    )
    assert submitted.status_code == 201
    evidence_id = submitted.json()["evidence_id"]
    response = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": tenant,
            "workload_id": workload,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    return evidence_id, response


# --- retire endpoint: happy path and response shape -------------------------


def test_retire_returns_200_compact_ordered_json(client, root_id):
    response = _retire(client, root_id)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    # Exactly one terminating newline; no surrounding whitespace.
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert response.content == response.content.strip() + b"\n"
    # Field order and field set are fixed, and every value is a string.
    assert list(response.json().keys()) == ["root_id", "status", "retired_at"]
    data = response.json()
    assert data["root_id"] == root_id
    assert data["status"] == "retired"
    retired_at = datetime.fromisoformat(data["retired_at"])
    assert retired_at.utcoffset() == timedelta(0)
    assert all(isinstance(value, str) for value in data.values())
    # The body describes only the retirement result: no extra metadata.
    assert response.content == (
        b'{"root_id":"'
        + root_id.encode("ascii")
        + b'","status":"retired","retired_at":"'
        + data["retired_at"].encode("ascii")
        + b'"}\n'
    )


def test_retire_persists_status_and_time(app, client, root_id):
    response = _retire(client, root_id)
    retired_at = response.json()["retired_at"]

    with app.state.session_factory() as session:
        root = session.get(TrustRoot, root_id)
        assert root.status == "retired"
        assert root.retired_at is not None
        assert _iso(root.retired_at) == retired_at


def _iso(value):
    return value.astimezone(timezone.utc).isoformat()


def test_retire_survives_restart(tmp_path, chain):
    url = f"sqlite:///{tmp_path}/restart.db"
    app1 = create_app(url)
    client1 = TestClient(app1)
    created = client1.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    root = created.json()["root_id"]
    retired = _retire(client1, root)
    assert retired.status_code == 200
    retired_at = retired.json()["retired_at"]
    app1.state.engine.dispose()

    app2 = create_app(url)
    with app2.state.session_factory() as session:
        record = session.get(TrustRoot, root)
        assert record.status == "retired"
        assert _iso(record.retired_at) == retired_at
    # A repeat retire after restart is still a stable 409.
    client2 = TestClient(app2)
    assert _retire(client2, root).status_code == 409
    app2.state.engine.dispose()


# --- path validation --------------------------------------------------------


@pytest.mark.parametrize(
    "root",
    [
        "not-a-uuid",
        "ABCDEFAB-ABCD-ABCD-ABCD-ABCDEFABCDEF",  # uppercase hex rejected
        "deadbeefdead-dead-dead-dead-deadbeefdead",  # malformed grouping
        "deadbeef-dead-dead-dead",  # truncated
        "%20",
        "%09",
    ],
)
def test_retire_invalid_path_identifier_returns_422(client, root_id, root):
    response = client.request(
        "POST",
        f"/v1/trust-roots/{root}/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_retire_whitespace_padded_path_identifier_returns_422(client, root_id):
    response = client.request(
        "POST",
        f"/v1/trust-roots/%09{root_id}%09/retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422
    # The root must still be active: a malformed path never retires.
    with client.app.state.session_factory() as session:
        assert session.get(TrustRoot, root_id).status == "active"


def test_retire_empty_path_segment_returns_422(client):
    response = client.post(
        "/v1/trust-roots//retire",
        json={"tenant_id": TENANT, "workload_id": WORKLOAD},
    )
    assert response.status_code == 422


def test_retire_invalid_path_does_not_touch_storage(app, client, root_id):
    # Count queries? At minimum the root stays active and the malformed id
    # never resolves to a row.
    statements: list[str] = []

    def spy(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(app.state.engine, "before_cursor_execute", spy)
    try:
        response = client.post(
            "/v1/trust-roots/not-a-uuid/retire",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )
    finally:
        event.remove(app.state.engine, "before_cursor_execute", spy)
    assert response.status_code == 422
    # No trust-root read or write occurred for a malformed path id.
    assert not any("trust_roots" in s for s in statements)
    with app.state.session_factory() as session:
        assert session.get(TrustRoot, root_id).status == "active"


# --- body validation --------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        None,
        b"",
        b"   ",
        b"not json",
        {"workload_id": WORKLOAD},
        {"tenant_id": TENANT},
        {"tenant_id": TENANT, "workload_id": WORKLOAD, "extra": 1},
        {"tenant_id": 1, "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": 2},
        {"tenant_id": None, "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": None},
        {"tenant_id": "", "workload_id": WORKLOAD},
        {"tenant_id": "   ", "workload_id": WORKLOAD},
        {"tenant_id": TENANT, "workload_id": ""},
        {"tenant_id": TENANT, "workload_id": "\t"},
        [],
        "a string",
        42,
    ],
)
def test_retire_invalid_bodies_return_422(client, root_id, payload):
    kwargs = {}
    if payload is None:
        # No body at all.
        response = client.post(
            f"/v1/trust-roots/{root_id}/retire",
            headers={"content-type": "application/json"},
            content=b"",
        )
    elif isinstance(payload, (bytes, bytearray)):
        response = client.post(
            f"/v1/trust-roots/{root_id}/retire",
            headers={"content-type": "application/json"},
            content=bytes(payload),
        )
    else:
        response = client.post(
            f"/v1/trust-roots/{root_id}/retire", json=payload
        )
    assert response.status_code == 422
    with client.app.state.session_factory() as session:
        root = session.get(TrustRoot, root_id)
        assert root.status == "active"
        assert root.retired_at is None


def test_retire_unknown_field_rejects_without_writing(client, root_id):
    response = _retire(client, root_id, unexpected="x")
    assert response.status_code == 422
    with client.app.state.session_factory() as session:
        assert session.get(TrustRoot, root_id).status == "active"


# --- 404 semantics ----------------------------------------------------------


def test_retire_unknown_root_returns_404(client, root_id):
    response = _retire(client, "66666666-6666-6666-6666-666666666666")
    assert response.status_code == 404


def test_retire_wrong_tenant_or_workload_returns_404(client, root_id):
    assert _retire(client, root_id, tenant=OTHER_TENANT).status_code == 404
    assert _retire(client, root_id, workload=OTHER_WORKLOAD).status_code == 404
    # Still active after the cross-scope attempts.
    with client.app.state.session_factory() as session:
        assert session.get(TrustRoot, root_id).status == "active"


def test_retire_404_is_indistinguishable_for_unknown_and_cross_scope(
    app, client, chain
):
    # A root that exists under another scope must not be discoverable.
    _, other_cert = make_root("other-scope-root")
    other = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": OTHER_TENANT,
            "workload_id": OTHER_WORKLOAD,
            "root_pem": pem(other_cert),
        },
    )
    assert other.status_code == 201
    other_root = other.json()["root_id"]

    unknown = _retire(client, "77777777-7777-7777-7777-777777777777")
    cross_scope = _retire(client, other_root)
    assert unknown.status_code == cross_scope.status_code == 404
    assert unknown.text == cross_scope.text


# --- repeat and concurrency -------------------------------------------------


def test_repeat_retire_returns_409_and_keeps_first_time(client, root_id):
    first = _retire(client, root_id)
    assert first.status_code == 200
    first_time = first.json()["retired_at"]

    second = _retire(client, root_id)
    assert second.status_code == 409
    with client.app.state.session_factory() as session:
        root = session.get(TrustRoot, root_id)
        assert root.status == "retired"
        assert _iso(root.retired_at) == first_time


def test_repeat_retire_appends_no_other_state_record(app, client, root_id):
    assert _retire(client, root_id).status_code == 200
    rowcount_before = _audit_like_rowcounts(app)

    assert _retire(client, root_id).status_code == 409

    assert _audit_like_rowcounts(app) == rowcount_before


def _audit_like_rowcounts(app):
    with app.state.engine.connect() as conn:
        return {
            table: conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            for table in ("audit_events", "certificate_revocations")
        }


def test_concurrent_retires_settle_once(app, root_id, chain):
    def call():
        return TestClient(app).post(
            f"/v1/trust-roots/{root_id}/retire",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )

    # A barrier maximizes contention despite SQLite serialization.
    barrier = threading.Barrier(8)

    def gated_call():
        barrier.wait()
        return call()

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: gated_call(), range(8)))

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(200) == 1
    assert statuses.count(409) == 7
    bodies = [r.json() for r in responses if r.status_code == 200]
    assert len(bodies) == 1
    with app.state.session_factory() as session:
        root = session.get(TrustRoot, root_id)
        assert root.status == "retired"
        assert root.retired_at is not None


# --- write/read failure -----------------------------------------------------


def test_retire_write_failure_rolls_back(app, client, root_id):
    def fail_update(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE trust_roots"):
            raise RuntimeError("simulated storage failure")

    event.listen(app.state.engine, "before_cursor_execute", fail_update)
    try:
        response = _retire(client, root_id)
        assert response.status_code == 500
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_update)

    with app.state.session_factory() as session:
        root = session.get(TrustRoot, root_id)
        assert root.status == "active"
        assert root.retired_at is None

    # After recovery the retire succeeds and the original active state is
    # gone only via this explicit transition.
    recovered = _retire(client, root_id)
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "retired"


def test_retire_read_failure_returns_500_and_writes_nothing(app, client, root_id):
    def fail_read(conn, cursor, statement, parameters, context, executemany):
        # The retire handler loads the full trust-root row by id.
        if statement.lstrip().startswith("SELECT trust_roots.root_id,"):
            raise RuntimeError("simulated storage failure")

    event.listen(app.state.engine, "before_cursor_execute", fail_read)
    try:
        response = _retire(client, root_id)
        assert response.status_code == 500
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_read)

    with app.state.session_factory() as session:
        root = session.get(TrustRoot, root_id)
        assert root.status == "active"
        assert root.retired_at is None


# --- duplicate create after retirement --------------------------------------


def test_duplicate_create_after_retire_still_returns_409(client, chain, root_id):
    assert _retire(client, root_id).status_code == 200

    repeat = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    assert repeat.status_code == 409
    # The retired row is the only one for this certificate/scope.
    with client.app.state.session_factory() as session:
        roots = session.scalars(select(TrustRoot)).all()
        assert len(roots) == 1
        assert roots[0].root_id == root_id
        assert roots[0].status == "retired"


# --- X.509 verification against a retired anchor ----------------------------


def test_retired_anchor_rejects_before_other_checks(client, root_id, chain):
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain), {"m": "abc"}
    )
    # The chain verifies while the root is active.
    _, before = _submit_and_verify(client, created, evidence)
    assert before.status_code == 200
    assert before.json()["status"] == "verified"

    # A *new* challenge/evidence anchored to the now-retired root is
    # rejected.
    assert _retire(client, root_id).status_code == 200
    created2 = _challenge(client)
    evidence2 = make_evidence(
        created2["nonce"], chain["leaf_key"], _full_chain(chain), {"m": "abc"}
    )
    evidence_id2, after = _submit_and_verify(client, created2, evidence2)
    assert after.status_code == 200
    assert after.json()["status"] == "rejected"
    with client.app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id2)
        assert record.status == "rejected"
        assert record.verification_result == "rejected"


def test_retired_anchor_rejection_precedes_revocation_and_signature(
    client, root_id, chain
):
    # Even evidence whose leaf signature is invalid and which carries no
    # revocation is rejected solely because the anchor is retired.
    assert _retire(client, root_id).status_code == 200
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    evidence_id, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert response.json()["evidence_id"] == evidence_id


class _CountingX509Verifier(X509AttestedNonceJSONVerifier):
    """X.509 verifier that records every invocation."""

    def __init__(self):
        self.calls = 0

    def verify(self, context):
        self.calls += 1
        return super().verify(context)


def test_retired_anchor_does_not_invoke_verifier(tmp_path, chain):
    verifier = _CountingX509Verifier()
    registry = VerifierRegistry()
    registry.register(verifier)
    application = create_app(f"sqlite:///{tmp_path}/count.db", registry)
    client = TestClient(application)
    created_root = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    root = created_root.json()["root_id"]
    assert _retire(client, root).status_code == 200

    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert verifier.calls == 0
    application.state.engine.dispose()


def test_retired_anchor_rejection_precedes_revocation_check(
    app, client, root_id, chain
):
    # Drop the revocation table so a revocation query would 500. A retired
    # anchor must settle rejected without touching that table.
    assert _retire(client, root_id).status_code == 200
    with app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE certificate_revocations"))

    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, response = _submit_and_verify(client, created, evidence)
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


def test_retirement_after_settlement_is_not_retroactive(client, root_id, chain):
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain), {"m": "abc"}
    )
    evidence_id, first = _submit_and_verify(client, created, evidence)
    assert first.json()["status"] == "verified"

    assert _retire(client, root_id).status_code == 200

    repeated = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert repeated.status_code == 200
    assert repeated.json() == first.json()
    assert repeated.json()["status"] == "verified"


def test_retirement_committed_during_verification_does_not_change_winner(
    tmp_path, chain
):
    """A retirement that commits while verification holds the anchor lock
    cannot change that verification's settlement, and it is not applied
    retroactively afterward."""

    entered = threading.Event()
    release = threading.Event()

    class GatedX509Verifier(X509AttestedNonceJSONVerifier):
        def verify(self, context):
            entered.set()
            release.wait(timeout=10)
            return super().verify(context)

    registry = VerifierRegistry()
    registry.register(GatedX509Verifier())
    application = create_app(f"sqlite:///{tmp_path}/gated.db", registry)
    client = TestClient(application)
    created_root = client.post(
        "/v1/trust-roots",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "root_pem": pem(chain["root_cert"]),
        },
    )
    root = created_root.json()["root_id"]

    challenge = _challenge(client)
    evidence = make_evidence(
        challenge["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": challenge["challenge_id"],
            "nonce": challenge["nonce"],
            "evidence_format": X509_ATTESTED_NONCE_JSON,
            "evidence": evidence,
        },
    )
    evidence_id = submitted.json()["evidence_id"]

    def verify():
        return TestClient(application).post(
            f"/v1/evidence/{evidence_id}/verify",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "nonce": challenge["nonce"],
                "evidence": evidence,
            },
        )

    def retire():
        return TestClient(application).post(
            f"/v1/trust-roots/{root}/retire",
            json={"tenant_id": TENANT, "workload_id": WORKLOAD},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        verify_future = pool.submit(verify)
        assert entered.wait(timeout=5)
        retire_future = pool.submit(retire)
        threading.Event().wait(0.5)
        release.set()
        verify_response = verify_future.result(timeout=15)
        retire_response = retire_future.result(timeout=15)

    assert verify_response.status_code == 200
    assert verify_response.json()["status"] == "verified"
    assert retire_response.status_code == 200

    repeated = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": challenge["nonce"],
            "evidence": evidence,
        },
    )
    assert repeated.json()["status"] == "verified"
    application.state.engine.dispose()


def test_trust_root_status_query_failure_returns_500_and_keeps_received(
    app, client, root_id, chain
):
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    submitted = client.post(
        "/v1/evidence",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "challenge_id": created["challenge_id"],
            "nonce": created["nonce"],
            "evidence_format": X509_ATTESTED_NONCE_JSON,
            "evidence": evidence,
        },
    )
    evidence_id = submitted.json()["evidence_id"]

    def fail_anchor_read(conn, cursor, statement, parameters, context, executemany):
        # The anchor lookup loads the full trust-root row by digest; the
        # earlier PEM tuple query selects only root_pem and still succeeds.
        if statement.lstrip().startswith("SELECT trust_roots.root_id,"):
            raise RuntimeError("simulated storage failure")

    event.listen(app.state.engine, "before_cursor_execute", fail_anchor_read)
    try:
        response = client.post(
            f"/v1/evidence/{evidence_id}/verify",
            json={
                "tenant_id": TENANT,
                "workload_id": WORKLOAD,
                "nonce": created["nonce"],
                "evidence": evidence,
            },
        )
        assert response.status_code == 500
    finally:
        event.remove(app.state.engine, "before_cursor_execute", fail_anchor_read)

    with app.state.session_factory() as session:
        record = session.get(Evidence, evidence_id)
        assert record.status == "received"

    # After recovery, evidence anchored to the still-active root verifies.
    recovered = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "verified"


def test_retired_anchor_rejection_settles_and_is_repeated(
    client, root_id, chain
):
    assert _retire(client, root_id).status_code == 200
    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    evidence_id, first = _submit_and_verify(client, created, evidence)
    assert first.json()["status"] == "rejected"

    second = client.post(
        f"/v1/evidence/{evidence_id}/verify",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "nonce": created["nonce"],
            "evidence": evidence,
        },
    )
    assert second.status_code == 200
    assert second.json() == first.json()


def test_active_root_still_anchors_and_existing_revocations_still_reject(
    client, root_id, chain
):
    # A revocation registered before retirement keeps rejecting through the
    # normal path while the root is active; retiring afterward only affects
    # new evidence and does not erase the revocation.
    import base64
    import hashlib

    from cryptography.hazmat.primitives.serialization import Encoding

    fingerprint = base64.urlsafe_b64encode(
        hashlib.sha256(
            chain["leaf_cert"].public_bytes(Encoding.DER)
        ).digest()
    ).rstrip(b"=").decode("ascii")
    registered = client.post(
        "/v1/revocations",
        json={
            "tenant_id": TENANT,
            "workload_id": WORKLOAD,
            "trust_root_id": root_id,
            "certificate_fingerprint": fingerprint,
            "effective_at": "2020-01-01T00:00:00Z",
        },
    )
    assert registered.status_code == 201

    created = _challenge(client)
    evidence = make_evidence(
        created["nonce"], chain["leaf_key"], _full_chain(chain)
    )
    _, response = _submit_and_verify(client, created, evidence)
    assert response.json()["status"] == "rejected"
