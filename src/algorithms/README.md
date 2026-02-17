# Algorithms Layer

This folder contains algorithm-owned training logic.

Subfolders:

- [`ppo/`](ppo/README.md): PPO rollout/loss/training.
- [`spo/`](spo/README.md): SPO search/replay/loss/training.

## Rules

- Algorithm code may use `crate::common::*`.
- Shared abstractions should be extracted to `src/common/*` rather than duplicated.
- Keep PPO and SPO internals independent unless sharing is intentional and moved to `common`.

## Related docs

- [Architecture](../../docs/architecture.md)
- [PPO loop](../../docs/training-loop.md)
- [SPO loop](../../docs/spo-training-loop.md)
