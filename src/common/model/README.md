# Model Common

Shared model/input adaptation layer.

## Files

- `models.rs`: actor/critic definitions and architecture-aware construction.
- `observation_adapter.rs`: trait-based observation conversion (`dense`, `binpack`).

## Contract

- Input conversion consistency across train/eval/search.
- BinPack tensor shape conventions preserved.
- Adapter selection supports explicit config + metadata fallback.

## Related docs

- [Architecture](../../../docs/architecture.md)
- [Configuration](../../../docs/configuration.md)
- [PPO loop](../../../docs/training-loop.md)
- [SPO loop](../../../docs/spo-training-loop.md)
