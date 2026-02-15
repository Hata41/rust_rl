# rust_ppo

High-throughput PPO training in Rust on top of `rustpool`, with support for `BinPack-v0` and `Maze-v0`.

This repository is documented for two goals:

1. Execute training reliably.
2. Understand how the system works (data flow and dependencies) before modifying it.

## Documentation

- [Onboarding](docs/onboarding.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Training Loop](docs/training-loop.md)
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
cargo run --release -- --config base_config.yaml
```

### 4) Override config values from CLI

```bash
cargo run --release -- \
  --config base_config.yaml \
  --task-id BinPack-v0 \
  --num-envs 64 \
  --rollout-length 128
```

## Execute common scenarios

### BinPack on CPU

```bash
cargo run --release -- \
  --config base_config.yaml \
  --task-id BinPack-v0 \
  --device-type cpu
```

### Maze on CPU

```bash
cargo run --release -- \
  --config base_config.yaml \
  --task-id Maze-v0 \
  --device-type cpu
```

### Distributed CUDA (2 processes example)

```bash
WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 cargo run --release -- --config base_config.yaml --device-type cuda
WORLD_SIZE=2 RANK=1 LOCAL_RANK=1 cargo run --release -- --config base_config.yaml --device-type cuda
```

Code pointers for execution path:

- Entry and backend selection (`main`): [src/bin/rust_ppo.rs](src/bin/rust_ppo.rs#L190)
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
2. [Training Loop](docs/training-loop.md)
3. [Telemetry](docs/telemetry.md)

## Runtime model

- Config precedence: **code defaults < YAML file < explicit CLI flags**.
- `RANK`, `WORLD_SIZE`, `LOCAL_RANK` env vars override distributed CLI fallbacks.
- CPU backend is single-process only (`WORLD_SIZE` must be 1).

Code pointers:

- Arg merge and overrides (`apply_config_file`, `apply_cli_overrides`): [src/config.rs](src/config.rs#L294), [src/config.rs](src/config.rs#L341)
- Distributed env resolution (`from_env_or_args`): [src/config.rs](src/config.rs#L414)
- CPU/CUDA guardrails: [src/bin/rust_ppo.rs](src/bin/rust_ppo.rs#L211)

For details, see [Configuration](docs/configuration.md) and [Training Loop](docs/training-loop.md).
