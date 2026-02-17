use anyhow::Result;
use burn::backend::autodiff::Autodiff;
use burn::backend::cuda::{Cuda, CudaDevice};
use burn_ndarray::{NdArray, NdArrayDevice};

use rust_rl::common::runtime::backend::{resolve_backend, RuntimeBackend};
use rust_rl::common::config::{DistInfo, SpoArgs};
use rust_rl::algorithms::spo::train;
use rust_rl::common::runtime::telemetry::{shutdown_otlp_provider, TrainingContext};

fn main() -> Result<()> {
    let args = SpoArgs::load()?.into_inner();
    let dist = DistInfo::from_env_or_args(&args)?;
    let context = TrainingContext::initialize("rust_rl.spo", &args, dist);

    let backend = resolve_backend(&args);
    let result = match backend.backend {
        RuntimeBackend::Cpu => {
            train::run::<Autodiff<NdArray<f32>>>(args, dist, &context, NdArrayDevice::Cpu)
        }
        RuntimeBackend::Cuda { device } => {
            train::run::<Autodiff<Cuda<f32, i32>>>(args, dist, &context, CudaDevice::new(device))
        }
    };

    shutdown_otlp_provider();
    result
}
