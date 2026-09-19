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

The API exposes `GET /health`, which returns a JSON readiness result, and a random-challenge flow:

- `POST /v1/challenges` issues a one-time random challenge (nonce) bound to a tenant and workload, with a bounded TTL.
- `POST /v1/challenges/{challenge_id}/consume` consumes a pending, unexpired challenge by presenting its nonce to prove freshness.
- `POST /v1/evidence` receives attestation evidence for a pending, unexpired challenge. The JSON body must include `tenant_id`, `workload_id`, `challenge_id`, `nonce` (unpadded base64url), `evidence_format`, and `evidence`. On success it returns `201` with `evidence_id`, `challenge_id`, status `received`, and a UTC RFC3339 `received_at`, and consumes the challenge in the same transaction so it cannot be reused. Only the SHA-256 digest of the evidence is persisted — never the evidence itself. Errors: `404` unknown/mismatched challenge, `401` wrong nonce, `409` already consumed or received, `410` expired, `422` invalid fields.
- `POST /v1/evidence/{evidence_id}/verify` verifies previously received evidence. The JSON body must include `tenant_id`, `workload_id`, `nonce` (unpadded base64url), and the original `evidence` string. The service checks tenant/workload ownership, the challenge nonce, and that the evidence's SHA-256 matches the stored digest, then runs the verifier plugin registered for the record's `evidence_format`. The first verification settles atomically: it returns `200` with `evidence_id`, `challenge_id`, status `verified` or `rejected`, and a UTC RFC3339 `verified_at`, and persists only the verdict and timestamp. Later calls with the same evidence return the stored verdict without re-running the plugin. Errors: `404` unknown/mismatched evidence, `401` wrong nonce, `422` invalid fields, unsupported format, or evidence digest mismatch.

## Verifier plugins

`proof_release.verifiers` exposes the plugin interface: implement `EvidenceVerifier` (`verify(evidence, context) -> bool`) and register it with `register_verifier(evidence_format, verifier)`. The plugin receives the request's evidence plaintext plus a `VerificationContext` (evidence/challenge IDs, tenant, workload, format, challenge timestamps); the service never persists the plaintext. The built-in `sha256-digest` format expects evidence of the form `<payload>.<digest>`, where `payload` is unpadded base64url and `digest` is the lowercase hex SHA-256 of the decoded payload bytes.
