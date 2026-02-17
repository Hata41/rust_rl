use std::collections::HashMap;
use std::fmt;
use std::thread;
use std::time::{Duration, Instant};
use anyhow::{anyhow, Result};
use chrono::Utc;
use crossbeam_channel::{Receiver, Sender};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::fmt::format::{FormatEvent, FormatFields, Writer};
use tracing_subscriber::fmt::FmtContext;
use tracing_subscriber::Layer;
use tracing_subscriber::registry::LookupSpan;

#[derive(Default)]
pub struct DashboardVisitor {
    category: Option<String>,
    fields: Vec<(String, String)>,
}

impl DashboardVisitor {
    fn record_value(&mut self, field: &Field, value: String) {
        let name = field.name();
        if name == "category" || name == "telemetry" {
            self.category = Some(value);
            return;
        }
        if name == "message" {
            return;
        }
        self.fields.push((name.to_string(), value));
    }

    fn format_float(value: f64) -> String {
        let abs = value.abs();
        if value == 0.0 {
            return "0".to_string();
        }
        if (1.0e-3..1.0e4).contains(&abs) {
            format!("{value:.6}")
        } else {
            format!("{value:.6e}")
        }
    }
}

impl Visit for DashboardVisitor {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.record_value(field, Self::format_float(value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record_value(field, value.to_string());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.record_value(field, value.to_string());
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.record_value(field, value.to_string());
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_value(field, value.to_string());
    }

    fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
        self.record_value(field, value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.record_value(field, format!("{value:?}"));
    }
}

#[derive(Default, Clone)]
pub struct MetricRegistry {
    labels: HashMap<String, String>,
}

impl MetricRegistry {
    pub fn with_defaults() -> Self {
        let mut registry = Self::default();
        registry.register("global_grad_norm", "Grad norm");
        registry.register("steps_per_second", "Steps per second");
        registry.register("learning_rate", "Learning rate");
        registry.register("critic_loss", "Critic loss");
        registry.register("episode_length_mean", "Episode length mean");
        registry.register("episode_length_max", "Episode length max");
        registry.register("episode_length_min", "Episode length min");
        registry
    }

    pub fn with_env_overrides(mut self) -> Self {
        if let Ok(raw) = std::env::var("RUST_RL_METRIC_LABELS") {
            for entry in raw.split(',') {
                let Some((key, label)) = entry.split_once('=') else {
                    continue;
                };
                let key = key.trim();
                let label = label.trim();
                if !key.is_empty() && !label.is_empty() {
                    self.register(key, label);
                }
            }
        }
        self
    }

    pub fn register(&mut self, key: impl Into<String>, label: impl Into<String>) {
        self.labels.insert(key.into(), label.into());
    }

    pub fn resolve(&self, key: &str) -> Option<&str> {
        self.labels.get(key).map(String::as_str)
    }
}

pub struct DashboardFormatter {
    registry: MetricRegistry,
}

impl DashboardFormatter {
    pub fn new(registry: MetricRegistry) -> Self {
        Self { registry }
    }

    fn category_label(raw: Option<&str>) -> &'static str {
        match raw.unwrap_or("MISC").to_ascii_uppercase().as_str() {
            "TRAINER" | "TRAIN" => "TRAINER",
            "EVALUATOR" | "EVAL" => "EVALUATOR",
            "ACTOR" | "ACT" => "ACTOR",
            _ => "MISC",
        }
    }

    fn style_category(label: &str, ansi: bool) -> String {
        if !ansi {
            return label.to_string();
        }
        let code = match label {
            "TRAINER" => "\x1b[1;35m",
            "EVALUATOR" => "\x1b[1;32m",
            "ACTOR" => "\x1b[1;36m",
            _ => "\x1b[1;33m",
        };
        format!("{code}{label}\x1b[0m")
    }

    fn pretty_key(&self, key: &str) -> String {
        if let Some(label) = self.registry.resolve(key) {
            return label.to_string();
        }

        let mut chars = key.replace('_', " ").chars().collect::<Vec<_>>();
        if let Some(first) = chars.first_mut() {
            *first = first.to_ascii_uppercase();
        }
        chars.into_iter().collect()
    }
}

impl<S, N> FormatEvent<S, N> for DashboardFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let mut visitor = DashboardVisitor::default();
        event.record(&mut visitor);

        let label = Self::category_label(visitor.category.as_deref());
        let styled = Self::style_category(label, writer.has_ansi_escapes());
        write!(writer, "{styled} - ")?;

        let mut wrote_metric = false;
        for (key, value) in visitor.fields {
            if wrote_metric {
                write!(writer, " | ")?;
            }
            wrote_metric = true;
            write!(writer, "{}: {}", self.pretty_key(&key), value)?;
        }

        if !wrote_metric {
            write!(writer, "No metrics")?;
        }

        writeln!(writer)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Metric {
    pub key: String,
    pub value: f64,
    pub timestamp: i64,
    pub step: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct LogBatch {
    pub run_id: String,
    pub metrics: Vec<Metric>,
}

#[derive(Debug, Serialize)]
struct CreateRunRequest<'a> {
    experiment_id: &'a str,
    start_time: i64,
    run_name: &'a str,
}

#[derive(Debug, Deserialize)]
struct CreateRunResponse {
    run: CreateRun,
}

#[derive(Debug, Deserialize)]
struct CreateRun {
    info: CreateRunInfo,
}

#[derive(Debug, Deserialize)]
struct CreateRunInfo {
    run_id: String,
}

#[derive(Clone)]
pub struct MlflowLayer {
    tx: Sender<Metric>,
    run_id: String,
}

#[derive(Default)]
struct NumericMetricVisitor {
    metrics: Vec<(String, f64)>,
    step: Option<i64>,
    message: Option<String>,
}

impl NumericMetricVisitor {
    fn is_step_field(field_name: &str) -> bool {
        matches!(
            field_name,
            "step" | "global_step" | "update" | "timesteps" | "policy_version"
        )
    }

    fn push_numeric(&mut self, field: &Field, value: f64) {
        let name = field.name();
        if name == "message" || name == "category" || name == "telemetry" {
            return;
        }
        self.metrics.push((name.to_string(), value));
    }

    fn should_skip_event(&self) -> bool {
        matches!(self.message.as_deref(), Some("startup") | Some("config_trace"))
    }
}

impl Visit for NumericMetricVisitor {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.push_numeric(field, value);
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        if Self::is_step_field(field.name()) {
            self.step = Some(value);
            return;
        }
        self.push_numeric(field, value as f64);
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        if Self::is_step_field(field.name()) {
            self.step = Some(value as i64);
            return;
        }
        self.push_numeric(field, value as f64);
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn fmt::Debug) {}
}

impl<S> Layer<S> for MlflowLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = NumericMetricVisitor::default();
        event.record(&mut visitor);

        if visitor.should_skip_event() {
            return;
        }

        let Some(step) = visitor.step else {
            return;
        };

        if visitor.metrics.is_empty() {
            return;
        }

        let timestamp = Utc::now().timestamp_millis();
        for (key, value) in visitor.metrics {
            let metric = Metric {
                key,
                value,
                timestamp,
                step,
            };

            if self.tx.send(metric).is_err() {
                eprintln!(
                    "MLflow metric channel closed for run_id={} (dropping metrics)",
                    self.run_id
                );
                return;
            }
        }
    }
}

fn flush_batch(client: &Client, endpoint: &str, run_id: &str, batch: &mut Vec<Metric>) -> Option<String> {
    if batch.is_empty() {
        return None;
    }

    let payload = LogBatch {
        run_id: run_id.to_string(),
        metrics: batch.clone(),
    };

    let mut last_error = None;
    for attempt in 0..3 {
        let response = client.post(endpoint).json(&payload).send();
        match response {
            Ok(resp) if resp.status().is_success() => {
                batch.clear();
                return None;
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp
                    .text()
                    .unwrap_or_else(|_| "<unavailable response body>".to_string());
                last_error = Some(format!("status={status}, body={body}"));
            }
            Err(error) => {
                last_error = Some(error.to_string());
            }
        }

        if attempt < 2 {
            thread::sleep(Duration::from_millis((attempt as u64 + 1) * 200));
        }
    }

    Some(format!(
        "MLflow log-batch transient failure after retries (run_id={run_id}, buffered_metrics={}): {}",
        batch.len(),
        last_error.unwrap_or_else(|| "unknown error".to_string())
    ))
}

fn worker_loop(rx: Receiver<Metric>, run_id: String, uri: String) {
    let endpoint = format!(
        "{}/api/2.0/mlflow/runs/log-batch",
        uri.trim_end_matches('/')
    );
    let client = Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap_or_else(|_| Client::new());

    let mut batch = Vec::with_capacity(50);
    let mut last_flush = Instant::now();
    let flush_interval = Duration::from_secs(1);
    let mut last_error_log_at: Option<Instant> = None;

    loop {
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(metric) => {
                batch.push(metric);
                if batch.len() >= 50 {
                    if let Some(message) = flush_batch(&client, &endpoint, &run_id, &mut batch) {
                        let should_log = last_error_log_at
                            .map(|t| t.elapsed() >= Duration::from_secs(10))
                            .unwrap_or(true);
                        if should_log {
                            eprintln!("{message}");
                            last_error_log_at = Some(Instant::now());
                        }
                    }
                    last_flush = Instant::now();
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                if let Some(message) = flush_batch(&client, &endpoint, &run_id, &mut batch) {
                    eprintln!("{message}");
                }
                break;
            }
        }

        if !batch.is_empty() && last_flush.elapsed() >= flush_interval {
            if let Some(message) = flush_batch(&client, &endpoint, &run_id, &mut batch) {
                let should_log = last_error_log_at
                    .map(|t| t.elapsed() >= Duration::from_secs(10))
                    .unwrap_or(true);
                if should_log {
                    eprintln!("{message}");
                    last_error_log_at = Some(Instant::now());
                }
            }
            last_flush = Instant::now();
        }
    }
}

pub fn init_mlflow_metrics(run_id: &str, uri: &str) -> MlflowLayer {
    let (tx, rx) = crossbeam_channel::unbounded::<Metric>();
    let thread_run_id = run_id.to_string();
    let thread_uri = uri.to_string();

    let _ = thread::Builder::new()
        .name("mlflow-metrics-worker".to_string())
        .spawn(move || worker_loop(rx, thread_run_id, thread_uri));

    MlflowLayer {
        tx,
        run_id: run_id.to_string(),
    }
}

pub fn create_mlflow_run(uri: &str, experiment_id: &str, run_name: &str) -> Result<String> {
    let endpoint = format!(
        "{}/api/2.0/mlflow/runs/create",
        uri.trim_end_matches('/')
    );

    let payload = CreateRunRequest {
        experiment_id,
        start_time: Utc::now().timestamp_millis(),
        run_name,
    };

    let client = Client::new();
    let response = client
        .post(&endpoint)
        .json(&payload)
        .send()
        .map_err(|error| anyhow!("request failed: {error}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .unwrap_or_else(|_| "<unavailable response body>".to_string());
        return Err(anyhow!("MLflow runs/create failed ({status}): {body}"));
    }

    let payload = response
        .json::<CreateRunResponse>()
        .map_err(|error| anyhow!("invalid runs/create response body: {error}"))?;
    Ok(payload.run.info.run_id)
}
