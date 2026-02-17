use anyhow::Result;
use burn::backend::autodiff::Autodiff;
use burn::backend::cuda::{Cuda, CudaDevice};
use burn_ndarray::{NdArray, NdArrayDevice};
use tracing::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use rust_rl::backend::{resolve_backend, RuntimeBackend};
use rust_rl::config::{Args, DistInfo};
use rust_rl::spo::train;
use rust_rl::telemetry::{
    create_mlflow_run, init_mlflow_metrics, DashboardFormatter, MetricRegistry,
};

fn main() -> Result<()> {
    let args = Args::load()?;
    let dist = DistInfo::from_env_or_args(&args)?;

    if dist.rank == 0 {
        let formatter =
            DashboardFormatter::new(MetricRegistry::with_defaults().with_env_overrides());
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(args.default_tracing_filter()));
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_target(false)
            .with_level(false)
            .without_time()
            .with_file(false)
            .with_line_number(false)
            .with_thread_ids(false)
            .with_thread_names(false)
            .event_format(formatter);
        let run_id = if let Some(run_id) = args.mlflow_run_id.clone() {
            Some(run_id)
        } else {
            let experiment_id =
                std::env::var("MLFLOW_EXPERIMENT_ID").unwrap_or_else(|_| "0".to_string());
            let run_name = format!("rust_rl.spo.pid-{}", std::process::id());
            match create_mlflow_run(&args.otlp_endpoint, &experiment_id, &run_name) {
                Ok(run_id) => {
                    eprintln!(
                        "created MLflow run_id='{run_id}' in experiment_id='{experiment_id}'"
                    );
                    Some(run_id)
                }
                Err(error) => {
                    eprintln!(
                        "warning: missing --mlflow-run-id and auto-create failed: {error}; MLflow metrics export is disabled"
                    );
                    None
                }
            }
        };
        let mlflow_layer = run_id
            .as_deref()
            .map(|resolved_run_id| init_mlflow_metrics(resolved_run_id, &args.otlp_endpoint));
        let _ = tracing_subscriber::registry()
            .with(mlflow_layer)
            .with(filter)
            .with(fmt_layer)
            .try_init();
        info!(category = "MISC", config = ?args, "config_trace");
    }

    let backend = resolve_backend(&args);
    let result = match backend.backend {
        RuntimeBackend::Cpu => train::run::<Autodiff<NdArray<f32>>>(args, dist, NdArrayDevice::Cpu),
        RuntimeBackend::Cuda { device } => {
            train::run::<Autodiff<Cuda<f32, i32>>>(args, dist, CudaDevice::new(device))
        }
    };

    result
}
