# Onboarding

Use this guide to get from clean checkout to reliable PPO/SPO runs with current config and telemetry semantics.

Source guides:

- [src/README.md](https://github.com/Hata41/rust_ppo/blob/main/src/README.md)
- [src/bin/README.md](https://github.com/Hata41/rust_ppo/blob/main/src/bin/README.md)

## Local setup

1. Install Rust stable toolchain.
2. Ensure Python build tooling is present (`python3-config` in `PATH`).
3. Ensure sibling `../rustpool` checkout exists.

Quick verification:

```bash
cargo check
```

## First successful runs

PPO:

```bash
cargo run --release -- --config ppo_config.yaml
```

SPO:

```bash
cargo run --release --bin spo -- --config spo_config.yaml
```

Expected telemetry categories:

- `TRAINER`
- `ACTOR`
- `EVALUATOR`
- `MISC`

## Runtime mental model

Both binaries follow:

1. load config via dedicated loader (`PpoArgs` / `SpoArgs`)
2. initialize shared telemetry context (`TrainingContext`)
3. run trainer wrapper (`PpoTrainer` / `SpoTrainer`)

## Shared config model

Precedence:

- code defaults
- YAML config
- explicit CLI flags

Compatibility:

- canonical section `training_core`
- legacy alias `ppo_core`

Adapter selection:

- optional `architecture.observation_adapter`
- metadata fallback when omitted

## Read order by intent

If your priority is running experiments:

1. [configuration.md](configuration.md)
2. [telemetry.md](telemetry.md)
3. [troubleshooting.md](troubleshooting.md)

If your priority is modifying internals:

1. [architecture.md](architecture.md)
2. [training-loop.md](training-loop.md)
3. [spo-training-loop.md](spo-training-loop.md)
4. [telemetry.md](telemetry.md)

## Safety checklist before pushing changes

- `cargo check` passes
- PPO smoke run succeeds
- SPO smoke run succeeds
- shared telemetry categories/evaluator schema preserved unless intentionally changed
- docs/templates updated with any schema or contract change
