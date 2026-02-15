# rust_rl

High-throughput PPO training in Rust on top of `rustpool`, with support for `BinPack-v0` and `Maze-v0`.

This repository is documented for two goals:

1. Execute training reliably.
2. Understand how the system works (data flow and dependencies) before modifying it.

## Documentation

- [Onboarding](docs/onboarding.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [PPO Training Loop](docs/training-loop.md)
- [SPO Training Loop](docs/spo-training-loop.md)
- [Telemetry](docs/telemetry.md)
- [Troubleshooting](docs/troubleshooting.md)

## Quickstart

### 1) Prerequisites

- Rust stable toolchain (`cargo`, `rustc`)
- Python dev tooling accessible to `python3-config` (required by `build.rs`)
- Sibling `rustpool` checkout at `../rustpool` (path dependency)

### 2) Build check

```bash
cargo check
```

### 3) Run with template config

```bash
cargo run --release -- --config ppo_config.yaml
```

### 3b) Run SPO with template config

```bash
cargo run --release --bin spo -- --config spo_config.yaml
```

### 4) Override config values from CLI

```bash
cargo run --release -- \
  --config ppo_config.yaml \
  --task-id BinPack-v0 \
  --num-envs 64 \
  --rollout-length 128
```

## Execute common scenarios

### BinPack on CPU

```bash
cargo run --release -- \
  --config ppo_config.yaml \
  --task-id BinPack-v0 \
  --device-type cpu
```

### Maze on CPU

```bash
cargo run --release -- \
  --config ppo_config.yaml \
  --task-id Maze-v0 \
  --device-type cpu
```

### SPO BinPack on CPU (config-only)

```bash
cargo run --release --bin spo -- --config spo_config.yaml
```

Code pointers for execution path:

- Entry and backend selection (`main`): [src/bin/ppo.rs](src/bin/ppo.rs)
- Config loading and precedence (`Args::load`): [src/config.rs](src/config.rs#L257)
- Training orchestrator (`train::run`): [src/ppo/train.rs](src/ppo/train.rs#L325)
- Environment pool + env factory (`AsyncEnvPool::new`, `make_env`): [src/env.rs](src/env.rs#L32), [src/env.rs](src/env.rs#L167)

## Read in this order

If your priority is execution:

1. [Onboarding](docs/onboarding.md)
2. [Configuration](docs/configuration.md)
3. [Troubleshooting](docs/troubleshooting.md)

If your priority is understanding how it works:

1. [Architecture](docs/architecture.md)
2. [PPO Training Loop](docs/training-loop.md)
3. [SPO Training Loop](docs/spo-training-loop.md)
4. [Telemetry](docs/telemetry.md)

## Shared contracts

Cross-cutting contracts to know before changing internals:

- Shared schema merge and compatibility aliasing (`training_core` + legacy `ppo_core`): [Configuration](docs/configuration.md)
- Shared env-model adapter used by PPO and SPO: [Architecture](docs/architecture.md)
- Snapshot ownership boundary (`rust_rl` accounting, `rustpool` storage backend): [Architecture](docs/architecture.md)
- Shared telemetry schema and categories for PPO/SPO: [Telemetry](docs/telemetry.md)

## Runtime model

- Config precedence: **code defaults < YAML file < explicit CLI flags**.
- Runtime is single-process (`world_size = 1`).

Code pointers:

- Arg merge and overrides (`apply_config_file`, `apply_cli_overrides`): [src/config.rs](src/config.rs#L294), [src/config.rs](src/config.rs#L341)
- Dist info resolution (`from_env_or_args`): [src/config.rs](src/config.rs#L414)
- CPU/CUDA dispatch: [src/bin/ppo.rs](src/bin/ppo.rs)

For details, see [Configuration](docs/configuration.md), [PPO Training Loop](docs/training-loop.md), and [SPO Training Loop](docs/spo-training-loop.md).

## Recent updates

Recent implementation and documentation changes are now reflected across PPO and SPO paths.

- SPO logging now matches PPO style and cadence, including evaluation-phase schema compatibility (`TRAINER` / `ACTOR` / `EVALUATOR` / `MISC`).
- Dashboard formatting and metric-label handling are shared in `src/telemetry.rs` and used by both binaries.
- Shared config schema docs now include precedence, strict unknown-field behavior, and compatibility aliasing (`training_core` with legacy `ppo_core`).
- SPO now has a dedicated runtime contract document at [docs/spo-training-loop.md](docs/spo-training-loop.md) at the same rigor level as PPO docs.
- Crate boundary ownership is explicit: `rust_rl` owns snapshot lifecycle accounting policy, while `rustpool` remains snapshot storage backend.

Suggested read order for these updates:

1. [docs/configuration.md](docs/configuration.md)
2. [docs/architecture.md](docs/architecture.md)
3. [docs/telemetry.md](docs/telemetry.md)
4. [docs/training-loop.md](docs/training-loop.md)
5. [docs/spo-training-loop.md](docs/spo-training-loop.md)
