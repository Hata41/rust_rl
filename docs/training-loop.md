# PPO Training Loop

This page explains PPO runtime behavior: update order, seeding, rollout/eval semantics, and where to edit specific behavior.
For static module dependencies and data contracts, see [Architecture](architecture.md).
For SPO runtime internals, see [spo-training-loop.md](spo-training-loop.md).

## Runtime sequence

`src/bin/ppo.rs` -> `train::run` -> `AsyncEnvPool` / `Rollout` / `Agent` / PPO losses.

Direct links:

- Entrypoint and run dispatch: [src/bin/ppo.rs](../src/bin/ppo.rs)
- Training orchestrator: [src/ppo/train.rs](../src/ppo/train.rs#L325)
- Environment layer: [src/env.rs](../src/env.rs#L32)
- Rollout storage: [src/ppo/buffer.rs](../src/ppo/buffer.rs#L167)
- Model forward path: [src/models.rs](../src/models.rs#L402)
- Loss path: [src/ppo/loss.rs](../src/ppo/loss.rs#L1)

Inside one update, the effective sequence is:

1. env step/reset via `AsyncEnvPool`
2. observation conversion (`flatten_obs` or BinPack parse)
3. actor/critic forward (`PolicyInput`)
4. action sampling + logprob
5. rollout store
6. GAE + target computation
7. minibatch tensor assembly
8. PPO losses + backward + optimizer step
9. telemetry + optional deterministic eval

## Where to edit for specific behavior

- Change sampling behavior -> `src/ppo/loss.rs` + rollout action path in `train.rs`.
- Change optimizer schedule -> LR/decay and epoch/minibatch loops in `train.rs`.
- Change evaluation behavior -> deterministic eval function in `train.rs`.
- Change observation handling -> `buffer.rs` parse/flatten + batch tensor assembly in `train.rs`.

Direct links:

- Sampling/loss utilities: [src/ppo/loss.rs](../src/ppo/loss.rs#L1)
- Optimization/eval loops: [src/ppo/train.rs](../src/ppo/train.rs#L640-L1009)
- Obs parsing and minibatch extraction: [src/ppo/buffer.rs](../src/ppo/buffer.rs#L33-L510)

## Determinism & seeding

`Args.seed` influences multiple RNG domains. Keep this model in mind when refactoring:

- **Backend RNG:** `B::seed(&device, args.seed)`
- **Training env pool seed root:** `args.seed`
- **Initial reset seed:** `args.seed + 10_000`
- **Eval pool seed root:** `args.seed + 999`
  - Deterministic eval pool creation and eval resets.
- **Minibatch shuffle RNG:** `StdRng::seed_from_u64(args.seed ^ 0xA11CE)`
  - Governs `all_indices.shuffle(&mut rng)` order.

### Reproducibility note

If you change minibatch ordering, update ordering, or pool reset timing, you are changing reproducibility characteristics even if loss code is untouched.

## Rollout and buffer behavior

- Dense tasks use flattened `obs` storage.
- BinPack uses packed structured storage (`items`, `ems`, validity masks).
- `roll.assert_preallocated()` should remain in update setup to guard memory contracts.

## Invalid action and done semantics

Training receives `done` and `action_mask` from rustpool workers. On `done`, rustpool worker resets env immediately and returns the reset observation with terminal reward/done flag for the just-finished step.

This means a step can carry:

- terminal reward from episode A,
- next observation/mask already from episode B.

See [Troubleshooting](troubleshooting.md) for the One Strike policy details.

## Practical dependency rule

When changing any item below, always re-check the coupled layer:

- observation shape -> re-check buffer parse + model inputs
- action dimension -> re-check masks + actor logits shape
- done/reset semantics -> re-check rollout bookkeeping + eval interpretation
- seed logic -> re-check env pool seeds + shuffle RNG and reproducibility expectations

Direct links:

- Seed propagation + shuffle RNG: [src/ppo/train.rs](../src/ppo/train.rs#L354-L405)
- Env seed fan-out: [src/env.rs](../src/env.rs#L76-L88)
