use anyhow::Result;
use burn::backend::autodiff::Autodiff;
use burn::backend::cuda::{Cuda, CudaDevice};
use burn_ndarray::{NdArray, NdArrayDevice};
use tracing_subscriber::EnvFilter;

use rust_rl::config::{Args, DeviceType, DistInfo};
use rust_rl::spo::train;
use rust_rl::telemetry::{DashboardFormatter, MetricRegistry};

fn main() -> Result<()> {
    let args = Args::load()?;
    let dist = DistInfo::from_env_or_args(&args)?;

    if dist.rank == 0 {
        let formatter = DashboardFormatter::new(
            MetricRegistry::with_defaults().with_env_overrides(),
        );
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_level(false)
            .without_time()
            .with_file(false)
            .with_line_number(false)
            .with_thread_ids(false)
            .with_thread_names(false)
            .event_format(formatter)
            .try_init();
    }

    match args.device_type {
        DeviceType::Cpu => train::run::<Autodiff<NdArray<f32>>>(args, dist, NdArrayDevice::Cpu),
        DeviceType::Cuda => {
            let device = CudaDevice::new(args.cuda_device);
            train::run::<Autodiff<Cuda<f32, i32>>>(args, dist, device)
        }
    }
}
