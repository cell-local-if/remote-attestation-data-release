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

The API exposes:

- `GET /health` — JSON readiness result.
- `POST /v1/challenges` — issue a pending challenge bound to a tenant and workload, returning a one-time unpadded base64url nonce and an expiry.
- `POST /v1/challenges/{challenge_id}/consume` — present the nonce to atomically consume a pending, unexpired challenge.
- `POST /v1/evidence` — submit attestation evidence against a challenge. The JSON body requires non-blank `tenant_id`, `workload_id`, `challenge_id`, `evidence_format`, a non-empty `evidence` string, and an unpadded base64url `nonce`. The challenge is matched to the tenant and workload, the nonce is verified, and only a pending, unexpired challenge is accepted; consuming the challenge and recording the evidence happen in one transaction, so a challenge used here cannot be consumed again (and vice versa). Returns `201` with `evidence_id`, `challenge_id`, `status: "received"`, and a UTC RFC3339 `received_at`. Errors: `404` unknown challenge or ownership mismatch, `401` bad nonce, `409` already consumed/received, `410` expired, `422` invalid request. Only the SHA-256 digest of the evidence (and of the nonce) is persisted; the raw payload is never stored or returned, and all records survive restarts.

