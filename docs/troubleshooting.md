# Troubleshooting

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

## Distributed startup issues

- `WORLD_SIZE must be > 0`
- `RANK must be < WORLD_SIZE`
- CPU mode with distributed world size > 1 is not supported.

## Build/link issues

If build fails around Python linking, validate `python3-config` availability and Python dev headers/libraries expected by `build.rs`.
