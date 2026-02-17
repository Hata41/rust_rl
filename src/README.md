# Source Tree Guide

This folder contains the crate implementation split into two stable layers:

- [`algorithms/`](algorithms/README.md): PPO/SPO algorithm-specific logic.
- [`common/`](common/README.md): shared runtime/model/config/utils used by both algorithms.
- [`bin/`](bin/README.md): executable entrypoints (`ppo`, `spo`).

## Dependency direction

- `src/bin/*` -> `rust_rl::algorithms::*` and `rust_rl::common::*`
- `src/algorithms/*` -> `crate::common::*`
- `src/common/*` -> no dependency on algorithm modules

Avoid reverse dependencies from `common` back into `algorithms`.

## Related docs

- [Architecture](../docs/architecture.md)
- [Configuration](../docs/configuration.md)
- [PPO loop](../docs/training-loop.md)
- [SPO loop](../docs/spo-training-loop.md)
- [Telemetry](../docs/telemetry.md)
