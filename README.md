# Remote Attestation Data Release

Backend foundation for releasing protected data only after a remote workload presents verifiable evidence that satisfies an explicit policy.

The service is intended to grow around attestation formats, verifier trust roots, nonce and freshness controls, policy evaluation, confidential data envelopes, one-time release grants, workload identity, revocation, audit evidence, key rotation, tenant isolation, rate limits, failure recovery, and operational observability. Every release decision must remain explainable and auditable without exposing the protected payload or private key material.

## Development

Requires Python 3.12 or newer.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest -q
uvicorn proof_release.app:app --reload
```

The API exposes `GET /health`, which returns a JSON readiness result, plus a one-time random-challenge flow used to prove workload freshness:

- `POST /v1/challenges` — issues a challenge (`tenant_id`, `workload_id`, optional `ttl_seconds` of 30–900, default 300). Returns `201` with `challenge_id`, `nonce`, `issued_at`, `expires_at`, and `status` (`pending`). The nonce is generated from a cryptographically secure source (32 bytes, unpadded base64url) and appears only in this response; the database stores its SHA-256 digest.
- `POST /v1/challenges/{challenge_id}/consume` — consumes a challenge (`tenant_id`, `workload_id`, `nonce`). Returns `200` with `consumed_at` on success, `404` for unknown/foreign challenges, `401` for a wrong nonce, `409` if already consumed, `410` if expired, and `422` for invalid fields. The pending→consumed transition is atomic, so concurrent consumes have exactly one winner.

Challenges persist in SQLite (`proof_release.db` by default, override with `PROOF_RELEASE_DATABASE_URL`) and survive process restarts; tables are created at startup.
