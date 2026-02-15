# Telemetry

## Dashboard categories

Logs are formatted into category lines:

- `TRAINER`
- `ACTOR`
- `EVALUATOR`
- `MISC`

Category styling is handled by `DashboardFormatter` in the binary entrypoint.

Source pointers:

- Formatter and category rendering: [src/bin/rust_ppo.rs](../src/bin/rust_ppo.rs#L108-L187)

## Dynamic metric labels

Metric labels are resolved by a registry with defaults and optional runtime overrides.

Environment variable:

```bash
RUST_PPO_METRIC_LABELS="global_grad_norm=Grad Norm,steps_per_second=SPS,mean_return=Eval Mean Return"
```

Rules:

- Format: comma-separated `key=Label` pairs.
- Unknown keys fall back to title-cased formatting (`episode_length_mean` -> `Episode length mean`).
- No code change needed to display new metric keys cleanly.

Source pointers:

- Metric registry defaults: [src/bin/rust_ppo.rs](../src/bin/rust_ppo.rs#L73-L83)
- Runtime env override parsing: [src/bin/rust_ppo.rs](../src/bin/rust_ppo.rs#L86-L100)
- Formatter usage at startup: [src/bin/rust_ppo.rs](../src/bin/rust_ppo.rs#L194-L207)

## Filtering

Use standard tracing filter env vars (for example `RUST_LOG=info` or more specific filters).

## Metric producers

Most training/eval metrics are emitted from `src/ppo/train.rs` via structured `info!`/`debug!` fields. When adding metrics, prefer consistent key names and avoid changing existing semantics unless intentional.

Source pointers:

- Train and runtime metrics blocks: [src/ppo/train.rs](../src/ppo/train.rs#L932-L965)
- Eval metrics block: [src/ppo/train.rs](../src/ppo/train.rs#L982-L1009)
