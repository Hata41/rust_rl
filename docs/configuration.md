# Configuration

This project uses a single shared configuration schema for both binaries:

- `ppo` (PPO)
- `spo` (SPO)

Runtime values are resolved through one merge pipeline in `Args::load`.

See also:

- [onboarding.md](onboarding.md) for first-run workflow
- [architecture.md](architecture.md) for crate-level ownership boundaries

## Source of truth

Authoritative runtime defaults live in code:

- `Args` defaults and CLI schema: [src/config.rs](../src/config.rs)

Template YAML files are curated presets, not default truth:

- PPO-oriented profile: [ppo_config.yaml](../ppo_config.yaml)
- SPO-oriented profile: [spo_config.yaml](../spo_config.yaml)

If template values differ from code defaults, code defaults still apply when a key is omitted from YAML and CLI.

## Merge precedence

At runtime, merge order is:

1. Built-in code defaults (`Args::default` and clap defaults)
2. YAML values from `--config <path>`
3. Explicit CLI flags

Code pointers:

- Loader and merge entrypoint: [src/config.rs](../src/config.rs)
- YAML parsing and application: [src/config.rs](../src/config.rs)
- CLI override application: [src/config.rs](../src/config.rs)

## Shared schema sections

Top-level YAML sections:

- `environment`
- `training_core`
- `optimization`
- `architecture`
- `evaluation`
- `hardware`
- `spo`

Both binaries load the same schema. Algorithm-specific keys are simply ignored by the other trainer unless read at runtime.

## Compatibility: `training_core` and legacy `ppo_core`

Canonical section name is `training_core`.

For backward compatibility, parser still accepts legacy alias `ppo_core` through serde aliasing in `FileConfig`.

Migration guidance:

- New/maintained configs should use `training_core`.
- Existing configs with `ppo_core` continue to load.
- Do not set both keys in one file.

## Strict validation behavior

YAML structs are deserialized with `deny_unknown_fields`.

Implication:

- Any misspelled key or unexpected extra key in a section fails config parsing at startup.

This is intentional to prevent silent misconfiguration.

Typical failure class:

- `failed to parse YAML config ... unknown field ...`

## Section contracts

### `environment`

Core environment topology and env-specific dimensions.

Important keys:

- `task_id`: task selector (`BinPack-v0`, `Maze-v0`)
- `num_envs`: trainer parallelism in each process
- `max_items`, `max_ems`: BinPack geometry controls
- `max_episode_steps`: episode truncation limit used by env construction

### `training_core`

Shared update cadence and discounting knobs.

Important keys:

- `rollout_length`
- `num_updates`
- `epochs`
- `num_minibatches`
- `gamma`
- `gae_lambda`

Notes:

- PPO uses all keys directly.
- SPO uses cadence and `gamma`; `gae_lambda` is currently unused by SPO runtime but remains in shared schema.

### `optimization`

Shared and algorithm-specific optimizer knobs.

Important keys:

- Shared: `actor_lr`, `critic_lr`
- PPO-specific: `max_grad_norm`, `clip_eps`, `ent_coef`, `vf_coef`, `decay_learning_rates`, `standardize_advantages`, `reward_scale`
- SPO-specific: `tau`

### `architecture`

Model width and random seed:

- `hidden_dim`
- `seed`

### `evaluation`

Deterministic evaluation schedule:

- `eval_interval`
- `num_eval_envs`
- `num_eval_episodes`

### `hardware`

Execution backend and device index:

- `device_type` (`cpu` or `cuda`)
- `cuda_device`

### `spo`

SPO search and MPO-dual controls.

| Key | Purpose | Practical constraints |
|---|---|---|
| `num_particles` | Search particles per root env | `> 0`; larger improves search estimate, increases cost |
| `search_depth` | SMC rollout depth | `> 0`; deeper increases latency and snapshot pressure |
| `replay_buffer_size` | Replay capacity (transitions) | Must fit memory; too small destabilizes updates |
| `sample_sequence_length` | Replay sequence length for learner sampling | Replay is sampled as contiguous sequences of this length |
| `dual_lr` | MPO dual optimizer LR | Too large can destabilize alpha/temperature updates |
| `epsilon` | E-step-like constraint | Must be positive |
| `epsilon_policy` | Policy KL constraint | Must be positive |
| `init_log_temperature` | Initial log temperature parameter | Usually moderate positive scalar |
| `init_log_alpha` | Initial log alpha parameter | Usually moderate positive scalar |
| `search_gamma` | Search return discount | Typical `[0, 1]` |
| `search_gae_lambda` | Search-side GAE decay factor | Applied in particle GAE update using $(\text{search\_gamma} \cdot \lambda)^\text{depth}$ |
| `root_exploration_dirichlet_alpha` | Dirichlet concentration | Must be positive when noise fraction > 0 |
| `root_exploration_dirichlet_fraction` | Root noise mix fraction | Range `[0, 1]` |
| `resampling_period` | Periodic resample cadence | `0` disables periodic trigger |
| `resampling_ess_threshold` | ESS-trigger threshold multiplier | Typical `(0, 1]` |
| `adaptive_temperature` | Per-env adaptive softmax temperature | Bool |
| `fixed_temperature` | Fallback temperature when adaptive off | Must be positive |

## Runtime guardrails to remember

Shared runtime checks are not all in config parser; some are validated at training bootstrap:

- Runtime is single-process only (`world_size = 1`).
- PPO requires minibatch divisibility conditions on `rollout_length * num_envs`.
- SPO currently supports `BinPack-v0` and `Maze-v0`.

Code pointers:

- PPO bootstrap constraints: [src/ppo/train.rs](../src/ppo/train.rs)
- SPO bootstrap constraints: [src/spo/train.rs](../src/spo/train.rs)

## Usage examples

PPO run with profile config:

```bash
cargo run --release -- --config ppo_config.yaml
```

SPO run with profile config:

```bash
cargo run --release --bin spo -- --config spo_config.yaml
```

PPO run with selected overrides:

```bash
cargo run --release -- \
  --config ppo_config.yaml \
  --task-id Maze-v0 \
  --num-envs 128 \
  --rollout-length 128 \
  --epochs 4
```

For algorithm runtime semantics, continue with:

- [training-loop.md](training-loop.md) (PPO)
- [spo-training-loop.md](spo-training-loop.md) (SPO)
