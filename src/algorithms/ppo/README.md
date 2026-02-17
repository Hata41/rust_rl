# PPO Module

PPO-specific implementation.

## Files

- `train.rs`: `PpoTrainer` + `PpoOptimizer` orchestration.
- `buffer.rs`: rollout storage and minibatch preparation.
- `loss.rs`: PPO clipped objective and value/entropy terms.

## Depends on

- `crate::common::config`
- `crate::common::runtime::{env, evaluation, telemetry}`
- `crate::common::model::{models, observation_adapter}`
- `crate::common::utils::{buffer, optimization}`

## Related docs

- [PPO training loop](../../../docs/training-loop.md)
- [Architecture](../../../docs/architecture.md)
- [Troubleshooting](../../../docs/troubleshooting.md)
