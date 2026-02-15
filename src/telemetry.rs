use std::collections::HashMap;
use std::fmt;
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::fmt::format::{FormatEvent, FormatFields, Writer};
use tracing_subscriber::fmt::FmtContext;
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
}

impl Visit for DashboardVisitor {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.record_value(field, format!("{value:.3}"));
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
