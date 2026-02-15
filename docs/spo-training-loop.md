# SPO Training Loop

This page documents the SPO runtime path in detail: search rollout, replay population, MPO-style optimization, target-network updates, deterministic evaluation, and failure-mode invariants.

For static crate/module boundaries, see [architecture.md](architecture.md).
For PPO runtime flow, see [training-loop.md](training-loop.md).
For schema/merge semantics, see [configuration.md](configuration.md).
For common failure signatures, see [troubleshooting.md](troubleshooting.md).

## Runtime sequence

`src/bin/spo.rs` -> `spo::train::run` -> `AsyncEnvPool` + SMC search + replay + MPO updates.

Inside one SPO update:

1. Collect `rollout_length` transitions from live envs.
2. For each environment step:
   - snapshot current live env state ids,
   - run SMC search from root snapshots,
   - release root/leaf simulated state ids,
   - execute chosen root action on live env,
   - append transition + root action-weight target into replay.
3. If replay has enough samples, run `epochs * num_minibatches` optimization steps.
4. Emit telemetry blocks.
5. Optionally run deterministic evaluation (search-based policy) on cadence.

Code pointers:

- SPO trainer entry and update loop: [src/spo/train.rs](../src/spo/train.rs)
- Search routine: [src/spo/search.rs](../src/spo/search.rs)
- Replay contract: [src/spo/buffer.rs](../src/spo/buffer.rs)
- MPO losses/duals: [src/spo/loss.rs](../src/spo/loss.rs)
- Env pool and snapshot lifecycle: [src/env.rs](../src/env.rs)

## Search phase contract

`run_smc_search` consumes:

- `root_state_ids`
- root observations
- root action masks
- search config and model adapter context

and produces:

- `root_actions`: sampled actions for live env step
- `root_action_weights`: per-env action distribution target from search particles
- `leaf_state_ids`: simulated states requiring release

### SMC mechanics currently implemented

- Particle rollout for configured depth
- Return accumulation with `search_gamma`
- Per-env particle weighting via softmax over returns
- Optional adaptive/fixed temperature policy
- ESS and periodic resampling triggers
- Optional root Dirichlet mixing (`alpha`, `fraction`)
- Weighted root action sampling

## Replay and optimization contract

Replay stores raw observations and masks, not only flattened vectors.

Stored transition payload:

- `obs`, `next_obs`
- selected environment action
- reward and done
- current/next action masks
- `root_action_weights` (search policy target)

Optimization uses sampled replay mini-batches:

- actor target: search-derived root action weights
- critic target: TD-style target using target critic on next observations
- dual updates: temperature and alpha losses
- online target sync: soft update with `tau`

## Target network update semantics

`soft_update_params` applies element-wise blend per float parameter:

- `target = (1 - tau) * target + tau * online`

This executes each optimizer step during SPO optimization.

## Evaluation phase contract

SPO deterministic evaluation is search-based:

1. Evaluate on separate eval env pool.
2. Use search root-action weights and pick greedy action.
3. Aggregate return and episode length statistics.
4. Emit `EVALUATOR` telemetry with PPO-compatible key schema.

Cadence:

- evaluation runs when `eval_interval > 0`, `update % eval_interval == 0`, and `num_eval_episodes > 0`.

## Shared model-input adapter in SPO

SPO does not hand-build task tensors inline. It uses `env_model` helpers to build actor/critic inputs from raw observations.

Benefits:

- shared observation conversion logic with PPO
- reduced task-specific branching duplication
- consistent behavior across train/eval/search call sites

## Snapshot lifecycle invariants

Critical invariants for correctness and memory stability:

- Every root snapshot id returned by `snapshot` must eventually be released.
- Every simulated leaf state id must eventually be released.
- Releases may be deduplicated; double-release should not corrupt local accounting.
- `AsyncEnvPool` drop must release any remaining tracked ids.

Ownership boundary:

- `rustpool::StateRegistry` stores/removes snapshot clones.
- `rust_rl::AsyncEnvPool` owns active-id accounting and leak/underflow prevention discipline.

## Failure modes and diagnostics

### Replay never optimizes

Symptom:

- TRAINER logs show replay growth but optimization metrics remain near zero or absent.

Likely cause:

- `replay.len() < sample_sequence_length * num_envs`.

### Search instability or action collapse

Symptoms:

- highly peaky `root_action_weights`
- poor return diversity

Knobs to inspect:

- `adaptive_temperature`
- `fixed_temperature`
- `resampling_ess_threshold`
- `root_exploration_dirichlet_fraction`

### Snapshot lifecycle regressions

Symptoms:

- rising memory usage in long runs
- debug assertions or accounting anomalies

Checks:

- verify all paths release root and leaf ids
- verify evaluation path also releases search state ids

For additional shared diagnostics (including One Strike and telemetry contract checks), see [troubleshooting.md](troubleshooting.md).

## Performance notes

Main SPO cost centers:

- `num_particles * search_depth`
- replay sampling and repeated forward/backward passes

Scaling guidance:

- increase particles/depth only after confirming telemetry throughput and stability
- keep eval configuration proportionate to training budget

## Edit map

When changing behavior, edit these primary files:

- Search behavior: [src/spo/search.rs](../src/spo/search.rs)
- Replay storage/sampling: [src/spo/buffer.rs](../src/spo/buffer.rs)
- Loss definitions: [src/spo/loss.rs](../src/spo/loss.rs)
- Trainer schedule/eval/logging: [src/spo/train.rs](../src/spo/train.rs)
- Shared obs/model adaptation: [src/env_model.rs](../src/env_model.rs)
