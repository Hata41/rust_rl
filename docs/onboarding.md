# Onboarding

Use this guide to get from clean checkout to a reliable PPO/SPO run with the expected telemetry and config semantics.

## Local setup

1. Install Rust stable toolchain.
2. Ensure Python build tooling is present and `python3-config` is in `PATH`.
3. Ensure sibling `../rustpool` checkout exists (path dependency).

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

## Recommended execution workflow

1. Build check:

```bash
cargo check
```

2. PPO baseline run:

```bash
cargo run --release -- --config ppo_config.yaml
```

3. SPO baseline run:

```bash
cargo run --release --bin spo -- --config spo_config.yaml
```

4. Override one variable at a time from CLI to isolate effects.

## Shared configuration model

Both binaries use the same config loader and merge path.

Precedence:

- code defaults
- YAML config file
- explicit CLI flags

Compatibility:

- canonical section name: `training_core`
- legacy alias accepted: `ppo_core`

If behavior differs from expected values, inspect merge path in `src/config.rs` before debugging trainers.

## Read order by intent

### If your priority is running experiments

1. [configuration.md](configuration.md)
2. [telemetry.md](telemetry.md)
3. [troubleshooting.md](troubleshooting.md)

### If your priority is internal changes

1. [architecture.md](architecture.md)
2. [training-loop.md](training-loop.md)
3. [spo-training-loop.md](spo-training-loop.md)
4. [telemetry.md](telemetry.md)

## Safety checklist before pushing trainer changes

- `cargo check` passes
- one PPO smoke run succeeds
- one SPO smoke run succeeds
- telemetry categories and eval schema are unchanged unless intentionally migrated
- config docs/templates were updated if schema keys changed

## Common first-debug path

When a run behaves unexpectedly, inspect in this order:

1. config merge and unknown-field validation (`src/config.rs`)
2. env routing and snapshot lifecycle (`src/env.rs`)
3. trainer-specific loop (`src/ppo/train.rs` or `src/spo/train.rs`)
4. rustpool worker/env semantics (`../rustpool/src/core/worker.rs` and env modules)
