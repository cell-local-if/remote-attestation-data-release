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
- `POST /v1/evidence/{evidence_id}/verify` verifies previously received evidence. The JSON body must include non-empty `tenant_id`, `workload_id`, `nonce` (unpadded base64url) and `evidence` (non-empty string). Only evidence belonging to the same tenant and workload, together with its challenge, is processed: the nonce is checked against the challenge and the evidence is checked to be byte-identical to what was received (SHA-256 digest comparison). The first verification settles the record atomically (`verified` or `rejected`); later calls return the stored conclusion and never re-run the verifier. Both outcomes return `200` with `evidence_id`, `challenge_id`, `status` and a UTC RFC3339 `verified_at`. Errors: `404` unknown/mismatched evidence, `401` wrong nonce, `422` invalid fields/format or evidence digest mismatch (including an `evidence_format` with no registered verifier), `500` verifier plugin failure (no partial state is left behind). As with receipt, the raw evidence is never persisted, logged, or returned — only its digest, the settlement status, timestamp, and a fixed service-defined result code (`accepted`/`rejected`) derived solely from the verifier's verdict are stored.

### Verifier plugins

Formats are verified through a public plugin interface in `proof_release.verifiers`: implement `Verifier` (set `format_name` and `verify(context) -> VerificationResult`) and register it via `register_verifier()` on `default_registry`, or pass a custom `VerifierRegistry` to `create_app(..., verifier_registry=...)`. The verifier receives a `VerificationContext` containing the raw evidence for this call plus the tenant, workload and challenge context; the service layer never stores the raw evidence.

**Plugin security limits.** The service only ever acts on the boolean `accepted` verdict: it persists a fixed, service-defined result code (`accepted`/`rejected`) derived solely from that verdict. Any text a plugin attaches to its result (e.g. `VerificationResult.detail`) is discarded at the service boundary — it is never persisted, logged, or returned in API responses — and plugins must not embed raw evidence, key material, or other private context there. The same applies to exceptions: a raised exception maps to `500` with no state left behind, and only the exception *type* (never its message or traceback) is logged, since a faulty plugin could stuff sensitive data into either.

The built-in format `attested-nonce-json` accepts the JSON object `{"nonce": "<unpadded base64url challenge nonce>", "claims": {...}, "mac": "<hex>"}`, where `mac` is the lowercase-hex HMAC-SHA256 over the canonical JSON of `{"claims": ..., "nonce": ...}` (sorted keys, compact separators) using the per-workload key `HMAC-SHA256(secret, tenant_id + ":" + workload_id)`. The shared secret comes from `PROOF_RELEASE_ATTESTED_NONCE_SECRET` (a clearly labelled development-only constant is used when unset; set the variable outside local development). Malformed documents, nonce mismatches and bad MACs are reported as `rejected`, not as errors.
