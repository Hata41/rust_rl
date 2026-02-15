# Telemetry

This project uses one shared dashboard formatting stack for PPO and SPO:

- formatter and category rendering: [src/telemetry.rs](../src/telemetry.rs)
- PPO wiring: [src/bin/ppo.rs](../src/bin/ppo.rs)
- SPO wiring: [src/bin/spo.rs](../src/bin/spo.rs)

## Categories

Structured metrics are emitted under these categories:

- `TRAINER`
- `ACTOR`
- `EVALUATOR`
- `MISC`

Category names are normalized by formatter-level mapping.

## Dynamic metric labels

Metric labels come from `MetricRegistry` defaults plus optional runtime overrides.

Environment variable format:

```bash
RUST_RL_METRIC_LABELS="global_grad_norm=Grad Norm,steps_per_second=SPS"
```

Rules:

- comma-separated `key=Label` pairs
- unknown keys fall back to title-cased rendering

## Emission producers

- PPO train/eval emissions: [src/ppo/train.rs](../src/ppo/train.rs)
- SPO train/eval emissions: [src/spo/train.rs](../src/spo/train.rs)

## PPO/SPO schema compatibility contract

PPO and SPO are expected to remain log-schema compatible for shared monitoring.

Note:

- some keys can be emitted as compatibility placeholders in one algorithm (for example when a metric is not semantically central to that method), but key presence and category contracts should remain stable for downstream consumers.

### Shared evaluation schema

Both should emit in `EVALUATOR` category:

- `phase`
- `policy_version`
- `episodes`
- `mean_return`
- `max_return`
- `min_return`
- `episode_length_mean`
- `episode_length_max`
- `episode_length_min`
- `duration_ms`

### Shared runtime categories per update

Per update, both algorithms should emit:

- `TRAINER` (optimization metrics)
- `ACTOR` (episodic performance aggregates)
- `MISC` (throughput/runtime metrics)

## Cadence expectations

- PPO: emits train/update logs per update and deterministic eval on configured cadence.
- SPO: emits train/update logs per update and deterministic search-based eval on configured cadence.

Eval cadence contract:

- run when `eval_interval > 0`
- run when `update % eval_interval == 0`
- require `num_eval_episodes > 0`

See runtime sources for exact emission points:

- [training-loop.md](training-loop.md) (PPO)
- [spo-training-loop.md](spo-training-loop.md) (SPO)

## Filtering

Use tracing filter env vars as usual (`RUST_LOG=info`, etc.).

## Maintenance checklist for telemetry changes

When modifying telemetry:

- update both trainer emitters if schema keys are shared,
- keep category names stable,
- keep formatter in shared module,
- update this document and run a quick PPO+SPO smoke check.
