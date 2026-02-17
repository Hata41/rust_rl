# Common Layer

This folder contains shared components used by both PPO and SPO.

Subfolders:

- [`runtime/`](runtime/README.md): backend/env/evaluation/telemetry runtime wiring.
- [`model/`](model/README.md): architecture and observation adaptation.
- [`utils/`](utils/README.md): shared low-level training helpers.
- `config.rs`: shared config schema + per-binary loaders.

## Rules

- `common` must not depend on `algorithms`.
- Place cross-algorithm contracts here first, then consume from algorithms.
- Keep runtime/model/utils responsibilities separated.

## Related docs

- [Architecture](../../docs/architecture.md)
- [Configuration](../../docs/configuration.md)
- [Telemetry](../../docs/telemetry.md)
