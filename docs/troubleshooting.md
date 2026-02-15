# Troubleshooting

This guide covers shared runtime contracts plus PPO/SPO-specific failure modes.

## One Strike Policy (invalid action contract)

In this architecture, invalid actions terminate the current episode immediately.

### Maze

In rustpool Maze env (`../rustpool/src/envs/maze/mod.rs`):

- [Maze step function](../../rustpool/src/envs/maze/mod.rs#L124)
- [invalid_action flag](../../rustpool/src/envs/maze/mod.rs#L144)
- [done condition](../../rustpool/src/envs/maze/mod.rs#L145)

- action validity is checked against current action mask,
- invalid action sets `invalid_action = true`,
- `done = reached_goal || truncated || invalid_action`,
- reward remains `1.0` only on goal, otherwise `0.0`.

### BinPack

In rustpool BinPack env (`../rustpool/src/envs/binpack/mod.rs`):

- [BinPack step function](../../rustpool/src/envs/binpack/mod.rs#L640)
- [done condition including failed placement](../../rustpool/src/envs/binpack/mod.rs#L650)

- if placement fails, `placed_successfully = false`,
- `done = all_placed || no_more_moves || !placed_successfully`.

### Worker auto-reset behavior

In rustpool worker loop (`../rustpool/src/core/worker.rs`):

- [worker loop entry](../../rustpool/src/core/worker.rs#L28)
- [step handling branch](../../rustpool/src/core/worker.rs#L36)
- [done-triggered reset behavior](../../rustpool/src/core/worker.rs#L42-L67)

- when `done` is true, worker resets env immediately,
- sends terminal `reward` and `done=true`,
- but observation/action mask already correspond to the **next** episode.

If you see sudden `0.0` rewards and a fresh observation mid-rollout, this contract is likely why.

## Shape mismatch issues

Symptoms:

- panics/errors in BinPack parsing or tensor construction,
- mask/action dimension mismatch.

Checks:

- verify `max_items` and `max_ems` consistency across config and env constructor,
- verify action dim equals `max_items * max_ems` for BinPack,
- verify parsed masks/valid flags sizes in `parse_binpack_obs`.

SPO-specific checks:

- verify `root_action_weights` length matches `action_dim` for replay inserts,
- verify eval and train both use consistent `EnvModelKind` path through `env_model` helpers,
- verify search masks match actor output action space at each depth.

## SPO replay warmup appears stuck

Symptom:

- SPO TRAINER logs appear but optimization losses stay near zero or do not update.

Cause:

- optimization is gated on replay warmup threshold (`sample_sequence_length * num_envs`).

Checks:

- confirm replay length grows each update,
- reduce `sample_sequence_length` for smoke tests,
- ensure rollout phase is executing and transitions are being appended.

## Snapshot lifecycle and accounting regressions

Symptoms:

- memory growth in long search-heavy runs,
- debug assertions or negative-live-counter style symptoms.

Contract:

- every state id produced by snapshot/simulate paths must be released,
- `rust_rl` (`AsyncEnvPool`) owns active-id accounting,
- `rustpool` `StateRegistry` remains storage backend and should not carry trainer accounting policy.

Checks:

- verify root ids are released after each search call,
- verify leaf ids returned by search are released,
- verify eval path also releases root/leaf simulated ids.

## Telemetry mismatch between PPO and SPO

Symptoms:

- dashboards/parsers fail on SPO logs,
- category names differ across binaries.

Expected:

- both binaries use shared formatter in `src/telemetry.rs`,
- categories are `TRAINER`, `ACTOR`, `EVALUATOR`, `MISC`,
- evaluation records use PPO-compatible key schema (`mean_return`, `episode_length_mean`, etc.).

If mismatch appears:

- verify both binaries initialize shared formatter,
- verify log fields in trainer emitters were not renamed inconsistently.

## Build/link issues

If build fails around Python linking, validate `python3-config` availability and Python dev headers/libraries expected by `build.rs`.
