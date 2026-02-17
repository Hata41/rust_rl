use anyhow::Result;
use burn::backend::autodiff::Autodiff;
use burn::backend::cuda::{Cuda, CudaDevice};
use burn_ndarray::{NdArray, NdArrayDevice};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use rust_rl::backend::{resolve_backend, RuntimeBackend};
use rust_rl::config::{Args, DistInfo};
use rust_rl::spo::train;
use rust_rl::telemetry::{
    init_otlp_layer, shutdown_otlp_provider, DashboardFormatter, MetricRegistry,
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
        let otlp_layer = match init_otlp_layer("rust_rl.spo", &args.otlp_endpoint) {
            Ok(layer) => Some(layer),
            Err(error) => {
                eprintln!("failed to initialize OTLP tracing: {error}");
                None
            }
        };
        let _ = tracing_subscriber::registry()
            .with(otlp_layer)
            .with(filter)
            .with(fmt_layer)
            .try_init();
    }

    let backend = resolve_backend(&args);
    let result = match backend.backend {
        RuntimeBackend::Cpu => train::run::<Autodiff<NdArray<f32>>>(args, dist, NdArrayDevice::Cpu),
        RuntimeBackend::Cuda { device } => {
            train::run::<Autodiff<Cuda<f32, i32>>>(args, dist, CudaDevice::new(device))
        }
    };

    shutdown_otlp_provider();
    result
}
