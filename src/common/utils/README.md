# Utils Common

Shared low-level utilities consumed by both algorithms.

## Files

- `buffer.rs`: flattening and storage helper contracts.
- `optimization.rs`: learning-rate decay and gradient clipping helpers.

## Contract

- Keep helpers generic and algorithm-neutral.
- Avoid introducing trainer policy decisions at this layer.

## Related docs

- [Architecture](../../../docs/architecture.md)
- [PPO loop](../../../docs/training-loop.md)
- [SPO loop](../../../docs/spo-training-loop.md)
