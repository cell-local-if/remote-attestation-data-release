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

The initial API exposes `GET /health`, which returns a JSON readiness result. It intentionally contains no attestation or data-release workflow yet.
