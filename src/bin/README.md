# Binaries

Executable entrypoints.

- `ppo.rs`: PPO bootstrap.
- `spo.rs`: SPO bootstrap.

Each binary performs:

1. argument loading (`PpoArgs` / `SpoArgs`),
2. distributed info resolution,
3. telemetry initialization (`TrainingContext`),
4. trainer execution.

## Related docs

- [Onboarding](../../docs/onboarding.md)
- [Architecture](../../docs/architecture.md)
