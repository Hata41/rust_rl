# Onboarding

## Local setup

1. Install Rust toolchain.
2. Ensure Python build tooling is present and `python3-config` is in `PATH`.
3. Ensure `../rustpool` exists (path dependency in `Cargo.toml`).

Quick verification:

```bash
cargo check
```

## First successful run

```bash
cargo run --release -- --config base_config.yaml
```

If successful, you should see categorized dashboard lines (`TRAINER`, `ACTOR`, `EVALUATOR`, `MISC`).

## Execution workflow (recommended)

1. Validate build:

```bash
cargo check
```

2. Run baseline config:

```bash
cargo run --release -- --config base_config.yaml
```

3. Run with a single explicit override (for example task):

```bash
cargo run --release -- --config base_config.yaml --task-id Maze-v0
```

4. If behavior is unexpected, inspect in this order:
	- config precedence in `src/config.rs`
	- runtime guardrails in `src/ppo/train.rs`
	- env reset/step behavior in `src/env.rs` + rustpool worker loop

This order minimizes debugging time because most runtime surprises come from config/shape/done contracts.

Direct links:

- Config precedence + CLI merge: [src/config.rs](../src/config.rs#L257-L389)
- Runtime guardrails and checks: [src/ppo/train.rs](../src/ppo/train.rs#L325-L351)
- Env pool routing/reset/step: [src/env.rs](../src/env.rs#L32-L141)
- Worker done/reset semantics: [rustpool worker loop](../../rustpool/src/core/worker.rs#L28-L67)

## Performance guardrails

The rollout path is allocation-sensitive by design.

- Keep pre-allocation invariant intact (`Rollout::assert_preallocated`).
- Avoid introducing per-step `Vec` growth in the `rollout_length` loop.
- Be careful with `.clone()` in hot paths (`train.rs` rollout and minibatch sections).
- Prefer pre-sized buffers and in-place copy semantics over per-iteration rebuilds.

### Why this matters

In Anakin-style PPO, throughput comes from large `N` and stable memory behavior. Re-allocating rollout storage each update can dominate runtime and hide model-side gains.

Direct links:

- Pre-allocation invariants: [src/ppo/buffer.rs](../src/ppo/buffer.rs#L245)
- Rollout construction and assertion: [src/ppo/train.rs](../src/ppo/train.rs#L435-L451)

## Safe change checklist

Before opening a PR that touches training or buffer code:

- `cargo check`
- one smoke run (`--config base_config.yaml`)
- verify no shape regressions (especially BinPack path)
- verify no new allocations in rollout hot loop unless intentionally justified

Continue with [Architecture](architecture.md) and [Training Loop](training-loop.md).
