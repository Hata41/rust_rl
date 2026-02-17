# Telemetry

This project uses a dual telemetry pipeline for PPO and SPO:

- Tracing (OTLP): spans/events export to MLflow OTLP (`/v1/traces`)
- Metrics (REST): scalar metrics export to MLflow REST (`/api/2.0/mlflow/runs/log-batch`)

Both are initialized through shared `TrainingContext::initialize(...)` in `src/common/runtime/telemetry.rs`.

Source guide:

- [src/common/runtime/README.md](../src/common/runtime/README.md)

## Initialization path

- PPO: `src/bin/ppo.rs`
- SPO: `src/bin/spo.rs`

Each binary:

1. loads args (`PpoArgs` / `SpoArgs`)
2. resolves dist info
3. initializes `TrainingContext`
4. enters trainer run path

## Categories

Shared category contract:

- `TRAINER`
- `ACTOR`
- `EVALUATOR`
- `MISC`

Category names are normalized by shared formatter logic in `src/common/runtime/telemetry.rs`.

## Metric key mapping

MLflow metric keys are prefixed by category:

- `TRAINER` -> `trainer/`
- `ACTOR` -> `actor/`
- `EVALUATOR` -> `evaluator/`
- fallback -> `misc/`

Only numeric fields are exported as metrics.

## Step fields

Metrics export requires a step-like field; accepted names include:

- `timesteps`
- `policy_version`
- `step`
- `global_step`
- `update`

## Runtime behavior

- Metrics export uses non-blocking channel + background worker.
- Batching flushes periodically and by batch size threshold.
- Missing/failed MLflow run creation disables metrics export only; tracing can still continue.

## Labels

Metric labels are controlled by `MetricRegistry` defaults and optional env override:

```bash
RUST_RL_METRIC_LABELS="global_grad_norm=Grad Norm,steps_per_second=SPS"
```

## Shared schema expectation

PPO and SPO should keep telemetry schema compatibility for downstream dashboards, especially evaluator keys:

- `episodes`
- `mean_return`
- `max_return`
- `min_return`
- `episode_length_mean`
- `episode_length_max`
- `episode_length_min`
- `duration_ms`

## Remote MLflow runbook

Typical reverse-SSH flow:

1. start MLflow locally (`127.0.0.1:5000`)
2. keep reverse tunnel open
3. run training remotely with `otlp_endpoint` pointing to `http://localhost:5000`

For transport failures, validate tunnel and endpoint path (`/v1/traces`).
