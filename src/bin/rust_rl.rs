use anyhow::{bail, Result};
use burn::backend::autodiff::Autodiff;
use burn::backend::cuda::{Cuda, CudaDevice};
use burn::collective::{register, CollectiveConfig, PeerId};
use burn_ndarray::{NdArray, NdArrayDevice};
use tracing_subscriber::EnvFilter;

use rust_rl::config::{Args, DeviceType, DistInfo};
use rust_rl::ppo::train;
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
        DeviceType::Cpu => {
            if dist.world_size > 1 {
                bail!(
                    "--device-type cpu does not support distributed training (WORLD_SIZE must be 1)"
                );
            }
            train::run::<Autodiff<NdArray<f32>>>(args, dist, NdArrayDevice::Cpu)
        }
        DeviceType::Cuda => {
            let device_index = if dist.world_size > 1 {
                dist.local_rank
            } else {
                args.cuda_device
            };
            let device = CudaDevice::new(device_index);

            if dist.world_size > 1 {
                let peer_id = PeerId::from(dist.rank);
                register::<Cuda<f32, i32>>(
                    peer_id,
                    device.clone(),
                    CollectiveConfig::default().with_num_devices(dist.world_size),
                )
                .map_err(|e| anyhow::anyhow!("failed to register burn collective: {e:?}"))?;
            }

            train::run::<Autodiff<Cuda<f32, i32>>>(args, dist, device)
        }
    }
}
