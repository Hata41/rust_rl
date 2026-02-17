# Configuration

This project uses one shared YAML schema with algorithm-specific loader entrypoints:

- PPO binary: `PpoArgs::load()`
- SPO binary: `SpoArgs::load()`

See also:

- [onboarding.md](onboarding.md)
- [architecture.md](architecture.md)
- [src/common/README.md](../src/common/README.md)

## Source of truth

Authoritative defaults live in code:

- `src/common/config.rs`

Template YAML files are curated presets:

- `ppo_config.yaml`
- `spo_config.yaml`

## Merge precedence

Runtime merge order:

1. Built-in defaults (`Args::default` / clap defaults)
2. YAML values from `--config`
3. Explicit CLI flags

## Schema sections

Top-level YAML sections:

- `environment`
- `training_core`
- `optimization`
- `architecture`
- `evaluation`
- `hardware`
- `logging`
- `spo`

Compatibility alias:

- canonical: `training_core`
- legacy accepted: `ppo_core`

## Strict validation behavior

Section structs use `deny_unknown_fields`.

Implications:

- unknown keys fail startup parsing,
- typos fail fast,
- silent misconfiguration is avoided.

## PPO/SPO decoupling behavior

Both binaries share schema, but loaders differ intentionally:

- `PpoArgs::load()` strips `spo` section before strict deserialization.
- `SpoArgs::load()` reads full schema including `spo` section.

This keeps PPO independent from SPO-only parameter payloads.

## Observation adapter selection

`architecture.observation_adapter` is optional.

Allowed values:

- `dense`
- `binpack`

If omitted, adapter selection falls back to metadata detection in `env_model`.

## Section contracts

### `environment`

- `task_id`
- `num_envs`
- `max_items`
- `max_ems`
- `max_episode_steps`

### `training_core`

- `rollout_length`
- `num_updates`
- `epochs`
- `num_minibatches`
- `gamma`
- `gae_lambda`

### `optimization`

Shared:

- `actor_lr`, `critic_lr`, `decay_learning_rates`

PPO-heavy:

- `max_grad_norm`, `clip_eps`, `ent_coef`, `vf_coef`, `standardize_advantages`, `reward_scale`

SPO-specific:

- `tau`

### `architecture`

- `hidden_dim`
- `seed`
- `observation_adapter` (optional)

### `evaluation`

- `eval_interval`
- `num_eval_envs`
- `num_eval_episodes`

### `hardware`

- `device_type` (`cpu` / `cuda`)
- `cuda_device`

### `logging`

- `log_level`
- `backend_logs_visible`
- `otlp_endpoint`
- `mlflow_run_id`

### `spo`

Search/replay/MPO controls used by SPO trainer.

- `num_particles`
- `search_depth`
- `replay_buffer_size`
- `sample_sequence_length`
- `sample_period`
- `dual_lr`
- `epsilon`
- `epsilon_policy`
- `init_log_temperature`
- `init_log_alpha`
- `search_gamma`
- `search_gae_lambda`
- `root_exploration_dirichlet_alpha`
- `root_exploration_dirichlet_fraction`
- `resampling_period`
- `resampling_ess_threshold`
- `adaptive_temperature`
- `fixed_temperature`

## Runtime guardrails

- Runtime remains single-process (`DistInfo` currently resolves to `world_size=1`).
- PPO requires `rollout_length * num_envs` divisible by `num_minibatches`.
- SPO currently supports `Maze-v0` and `BinPack-v0`.

## Usage examples

PPO:

```bash
cargo run --release -- --config ppo_config.yaml
```

SPO:

```bash
cargo run --release --bin spo -- --config spo_config.yaml
```

With overrides:

```bash
cargo run --release -- --config ppo_config.yaml --task-id Maze-v0 --num-envs 128
```
