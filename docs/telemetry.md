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

Filtering precedence:

- If `RUST_LOG` is set, it is used directly.
- Otherwise, defaults come from YAML `logging` section (`log_level`, `backend_logs_visible`).

Default config behavior keeps CubeCL/CUDA backend context logs hidden to reduce dashboard noise.

## MLflow OTLP over reverse SSH (repeatable runbook)

Use this when training runs on a remote host (e.g. `BareMetal`) and MLflow runs on your laptop.

### Required runtime wiring

- OTLP endpoint must be `http://localhost:5000/v1/traces` on the remote training host.
- MLflow OTLP header must be present: `x-mlflow-experiment-id` (configured in `src/telemetry.rs`).

### SSH config (laptop)

Example `~/.ssh/config` entry:

```ssh_config
Host REMOTE_TRAINING_HOST
	HostName YOUR_REMOTE_HOST_OR_IP
	IdentityFile ~/.ssh/YOUR_PRIVATE_KEY
	User YOUR_REMOTE_USER
	Port YOUR_SSH_PORT
	RemoteForward 5000 localhost:5000
```

### Every-run command order

1) **Laptop / Terminal A**: start MLflow first

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

2) **Laptop / Terminal B**: keep reverse tunnel open

```bash
ssh -N BareMetal
```

Optional debug mode:

```bash
ssh -vvv -N BareMetal
```

3) **Remote (`BareMetal`)**: verify tunnel endpoint

```bash
curl -i http://localhost:5000/
```

Expected: `HTTP/1.1 200 OK`.

4) **Remote (`BareMetal`)**: run training

```bash
cargo run --bin ppo -- --config ppo_config.yaml
cargo run --bin spo -- --config spo_config.yaml
```

### Quick diagnostics

- `405 Method Not Allowed` at `http://localhost:5000/`:
	endpoint path is wrong; use `/v1/traces`.
- `Connection refused` to `http://localhost:5000/v1/traces`:
	tunnel or local MLflow is down.
- `ssh -N BareMetal` prints `connect_to localhost port 5000: failed`:
	local MLflow is not listening yet; start MLflow first, then reconnect SSH.

### Optional direct probe of OTLP route

From `BareMetal`, this checks that POST reaches the traces route and header is accepted:

```bash
curl -X POST 'http://localhost:5000/v1/traces' \
	-H 'x-mlflow-experiment-id: 0' \
	-H 'Content-Type: application/x-protobuf' \
	--data-binary '' -i
```

Expected: not `404`/`405`; a `400` with empty payload is acceptable for this probe.

## Maintenance checklist for telemetry changes

When modifying telemetry:

- update both trainer emitters if schema keys are shared,
- keep category names stable,
- keep formatter in shared module,
- update this document and run a quick PPO+SPO smoke check.
