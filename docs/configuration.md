# Configuration

## Source of truth

Code defaults in `src/config.rs` are authoritative runtime defaults.

`base_config.yaml` is a template override file and may intentionally differ to provide a lighter local profile.

Source pointers:

- `Args` defaults and CLI schema: [src/config.rs](../src/config.rs#L14-L151)
- Default values implementation: [src/config.rs](../src/config.rs#L138-L179)
- Template config file: [base_config.yaml](../base_config.yaml#L1-L64)

## Precedence

At runtime:

1. Built-in defaults (`Args::default` / clap defaults)
2. YAML values from `--config <path>`
3. Explicit CLI flags provided on command line

Distributed env vars are then resolved in `DistInfo::from_env_or_args`:

- `RANK`
- `WORLD_SIZE`
- `LOCAL_RANK`

If present, env vars override distributed fallback values from args/config.

Source pointers:

- Load + merge pipeline (`Args::load`): [src/config.rs](../src/config.rs#L257-L273)
- YAML application (`apply_config_file`): [src/config.rs](../src/config.rs#L294-L338)
- CLI overrides (`apply_cli_overrides`): [src/config.rs](../src/config.rs#L341-L389)
- Env distributed overrides (`from_env_or_args`): [src/config.rs](../src/config.rs#L414-L450)

## Main groups

- `environment`: task/env cardinality and env-specific limits
- `ppo_core`: rollout/epoch/minibatch core algorithm knobs
- `optimization`: LR, clip, entropy, vf, grad clipping, scaling/standardization
- `architecture`: hidden dim and seed
- `evaluation`: deterministic eval cadence and volume
- `hardware`: backend and device index
- `distributed`: rank/world-size/local-rank fallbacks

## Common pitfalls

- CPU backend with `WORLD_SIZE > 1` is rejected.
- `num_envs` must be divisible by `WORLD_SIZE`.
- `rollout_length * local_num_envs` must be divisible by `num_minibatches`.

Source pointers:

- CPU distributed restriction: [src/bin/rust_ppo.rs](../src/bin/rust_ppo.rs#L211-L219)
- Batch divisibility guardrails: [src/ppo/train.rs](../src/ppo/train.rs#L332-L351)

## Examples

Use template as-is:

```bash
cargo run --release -- --config base_config.yaml
```

Override selected values:

```bash
cargo run --release -- \
  --config base_config.yaml \
  --task-id Maze-v0 \
  --num-envs 128 \
  --epochs 4 \
  --num-minibatches 32
```
