# Runtime Common

Shared runtime execution services.

## Files

- `backend.rs`: backend/device selection and CUDA validation.
- `env.rs`: async pool, worker communication, env registry.
- `evaluation.rs`: shared deterministic evaluation harness.
- `telemetry.rs`: tracing/metrics setup and `TrainingContext`.

## Contract

- Runtime modules are algorithm-agnostic.
- Snapshot lifecycle accounting stays at pool boundary.
- Evaluation remains reusable via closure-based policy adapter.

## Related docs

- [Architecture](../../../docs/architecture.md)
- [Telemetry](../../../docs/telemetry.md)
- [Troubleshooting](../../../docs/troubleshooting.md)
