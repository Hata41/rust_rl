# SPO Module

SPO-specific implementation.

## Files

- `train.rs`: `SpoTrainer` + `SpoOptimizer` orchestration.
- `search.rs`: snapshot-based SMC search flow.
- `buffer.rs`: replay buffer and sequence sampling.
- `loss.rs`: MPO losses and dual updates.

## Depends on

- `crate::common::config`
- `crate::common::runtime::{env, evaluation, telemetry}`
- `crate::common::model::{models, observation_adapter}`
- `crate::common::utils::{buffer, optimization}`

## Related docs

- [SPO training loop](../../../docs/spo-training-loop.md)
- [Architecture](../../../docs/architecture.md)
- [Troubleshooting](../../../docs/troubleshooting.md)
