use std::any::Any;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use burn::backend::cuda::CudaDevice;
use cudarc::driver::{CudaContext, DriverError};
use cudarc::nvrtc::{compile_ptx, CompileError};
use tracing::warn;

use crate::config::{Args, DeviceType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeBackend {
    Cpu,
    Cuda { device: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaProbeFailureKind {
    NoGpu,
    DriverOnlyNoToolkit,
    MissingNvrtc,
    UnsupportedCudaVersion,
    InitFailure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaProbeFailure {
    pub kind: CudaProbeFailureKind,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDecision {
    pub backend: RuntimeBackend,
    pub warning: Option<String>,
    pub cuda_fallback: Option<CudaProbeFailureKind>,
}

pub fn resolve_backend(args: &Args) -> BackendDecision {
    resolve_backend_with_probe(args, probe_cuda_runtime)
}

pub(crate) fn resolve_backend_with_probe<F>(args: &Args, probe: F) -> BackendDecision
where
    F: Fn(usize) -> Result<(), CudaProbeFailure>,
{
    if matches!(args.device_type, DeviceType::Cpu) {
        return BackendDecision {
            backend: RuntimeBackend::Cpu,
            warning: None,
            cuda_fallback: None,
        };
    }

    match probe(args.cuda_device) {
        Ok(()) => BackendDecision {
            backend: RuntimeBackend::Cuda {
                device: args.cuda_device,
            },
            warning: None,
            cuda_fallback: None,
        },
        Err(err) => {
            let warning = format!(
                "CUDA requested for device {} but probe failed ({}): {}. Falling back to CPU backend.",
                args.cuda_device,
                failure_kind_label(err.kind),
                err.detail,
            );
            warn!("{warning}");
            eprintln!("WARNING: {warning}");
            warn!(
                "If CUDA is expected on this host, verify toolkit/runtime paths (CUDA_PATH/CUDA_HOME and LD_LIBRARY_PATH)."
            );
            eprintln!(
                "WARNING: If CUDA is expected on this host, verify toolkit/runtime paths (CUDA_PATH/CUDA_HOME and LD_LIBRARY_PATH)."
            );
            BackendDecision {
                backend: RuntimeBackend::Cpu,
                warning: Some(warning),
                cuda_fallback: Some(err.kind),
            }
        }
    }
}

fn probe_cuda_runtime(cuda_device: usize) -> Result<(), CudaProbeFailure> {
    let device_count = catch_unwind_silent(CudaContext::device_count)
        .map_err(classify_cuda_context_panic)?
        .map_err(|err| classify_driver_error(err, cuda_device))?;

    if device_count <= 0 {
        return Err(CudaProbeFailure {
            kind: CudaProbeFailureKind::NoGpu,
            detail: "no CUDA device detected by the driver".to_string(),
        });
    }

    if cuda_device >= device_count as usize {
        return Err(CudaProbeFailure {
            kind: CudaProbeFailureKind::NoGpu,
            detail: format!(
                "requested cuda_device={} but only {} device(s) are visible",
                cuda_device, device_count
            ),
        });
    }

    let _context = catch_unwind_silent(|| CudaContext::new(cuda_device))
        .map_err(classify_cuda_context_panic)?
        .map_err(|err| classify_driver_error(err, cuda_device))?;

    let kernel = "extern \"C\" __global__ void rust_rl_probe(float *x) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i == 0) { x[0] = 0.0f; } }";
    catch_unwind_silent(|| compile_ptx(kernel))
        .map_err(classify_nvrtc_panic)?
        .map_err(classify_nvrtc_error)?;

    catch_unwind_silent(|| CudaDevice::new(cuda_device)).map_err(classify_burn_cuda_panic)?;

    Ok(())
}

fn catch_unwind_silent<F, T>(f: F) -> Result<T, String>
where
    F: FnOnce() -> T,
{
    static PANIC_HOOK_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
    let mutex = PANIC_HOOK_MUTEX.get_or_init(|| Mutex::new(()));
    let guard = mutex.lock().expect("panic hook mutex poisoned");

    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .map_err(panic_payload_to_string);

    std::panic::set_hook(previous_hook);
    drop(guard);

    result
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    "panic without string payload".to_string()
}

fn classify_cuda_context_panic(message: String) -> CudaProbeFailure {
    let lowered = message.to_lowercase();
    if lowered.contains("undefined symbol") {
        return CudaProbeFailure {
            kind: CudaProbeFailureKind::UnsupportedCudaVersion,
            detail: format!(
                "CUDA driver/runtime symbol mismatch ({message}); this usually indicates an older driver not exposing symbols required by the selected CUDA runtime"
            ),
        };
    }

    if lowered.contains("libcuda") || lowered.contains("cuda driver") {
        return CudaProbeFailure {
            kind: CudaProbeFailureKind::InitFailure,
            detail: format!(
                "failed to load CUDA driver library ({message}); check NVIDIA driver installation and LD_LIBRARY_PATH"
            ),
        };
    }

    CudaProbeFailure {
        kind: CudaProbeFailureKind::InitFailure,
        detail: format!("CUDA initialization panicked: {message}"),
    }
}

fn classify_driver_error(err: DriverError, cuda_device: usize) -> CudaProbeFailure {
    let name = err
        .error_name()
        .ok()
        .and_then(|s| s.to_str().ok())
        .unwrap_or("CUDA_ERROR_UNKNOWN");
    let description = err
        .error_string()
        .ok()
        .and_then(|s| s.to_str().ok())
        .unwrap_or("unknown CUDA driver error");

    let kind = match name {
        "CUDA_ERROR_NO_DEVICE" | "CUDA_ERROR_INVALID_DEVICE" => CudaProbeFailureKind::NoGpu,
        "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH"
        | "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE"
        | "CUDA_ERROR_UNSUPPORTED_PTX_VERSION" => CudaProbeFailureKind::UnsupportedCudaVersion,
        _ => CudaProbeFailureKind::InitFailure,
    };

    CudaProbeFailure {
        kind,
        detail: format!(
            "driver error while probing cuda_device={cuda_device}: {name} ({description})"
        ),
    }
}

fn classify_nvrtc_panic(message: String) -> CudaProbeFailure {
    let lowered = message.to_lowercase();
    if lowered.contains("nvrtc") {
        if toolkit_root_exists() {
            return CudaProbeFailure {
                kind: CudaProbeFailureKind::MissingNvrtc,
                detail: format!(
                    "NVRTC failed to load despite toolkit hints present ({message}); check LD_LIBRARY_PATH and toolkit installation"
                ),
            };
        }

        return CudaProbeFailure {
            kind: CudaProbeFailureKind::DriverOnlyNoToolkit,
            detail: format!(
                "CUDA driver appears available, but toolkit/NVRTC is missing ({message})"
            ),
        };
    }

    CudaProbeFailure {
        kind: CudaProbeFailureKind::InitFailure,
        detail: format!("NVRTC probe panicked: {message}"),
    }
}

fn classify_nvrtc_error(err: CompileError) -> CudaProbeFailure {
    let detail = format!("{err:?}");
    let lowered = detail.to_lowercase();

    let kind = if lowered.contains("unsupported") || lowered.contains("invalid ptx") {
        CudaProbeFailureKind::UnsupportedCudaVersion
    } else if lowered.contains("nvrtc") && lowered.contains("not found") {
        if toolkit_root_exists() {
            CudaProbeFailureKind::MissingNvrtc
        } else {
            CudaProbeFailureKind::DriverOnlyNoToolkit
        }
    } else {
        CudaProbeFailureKind::InitFailure
    };

    CudaProbeFailure { kind, detail }
}

fn classify_burn_cuda_panic(message: String) -> CudaProbeFailure {
    let lowered = message.to_lowercase();
    if lowered.contains("undefined symbol") {
        return CudaProbeFailure {
            kind: CudaProbeFailureKind::UnsupportedCudaVersion,
            detail: format!(
                "CUDA driver does not expose a symbol required by Burn/CubeCL ({message}); update NVIDIA driver or use a compatible CUDA runtime"
            ),
        };
    }

    if lowered.contains("libcuda") {
        return CudaProbeFailure {
            kind: CudaProbeFailureKind::InitFailure,
            detail: format!(
                "Burn CUDA backend failed to load CUDA driver library ({message})"
            ),
        };
    }

    CudaProbeFailure {
        kind: CudaProbeFailureKind::InitFailure,
        detail: format!("Burn CUDA backend initialization panicked: {message}"),
    }
}

fn toolkit_root_exists() -> bool {
    std::env::var("CUDA_PATH")
        .ok()
        .filter(|p| Path::new(p).exists())
        .is_some()
        || std::env::var("CUDA_HOME")
            .ok()
            .filter(|p| Path::new(p).exists())
            .is_some()
        || Path::new("/usr/local/cuda").exists()
}

fn failure_kind_label(kind: CudaProbeFailureKind) -> &'static str {
    match kind {
        CudaProbeFailureKind::NoGpu => "no-gpu",
        CudaProbeFailureKind::DriverOnlyNoToolkit => "driver-only-no-toolkit",
        CudaProbeFailureKind::MissingNvrtc => "missing-nvrtc",
        CudaProbeFailureKind::UnsupportedCudaVersion => "unsupported-cuda-version",
        CudaProbeFailureKind::InitFailure => "init-failure",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Args, DeviceType};

    fn base_args(device_type: DeviceType) -> Args {
        let mut args = Args::default();
        args.device_type = device_type;
        args.cuda_device = 0;
        args
    }

    #[test]
    fn cpu_request_skips_probe() {
        let args = base_args(DeviceType::Cpu);
        let decision = resolve_backend_with_probe(&args, |_device| {
            panic!("probe should not be called for cpu")
        });

        assert_eq!(decision.backend, RuntimeBackend::Cpu);
        assert!(decision.warning.is_none());
        assert!(decision.cuda_fallback.is_none());
    }

    #[test]
    fn cuda_request_keeps_cuda_when_probe_ok() {
        let args = base_args(DeviceType::Cuda);
        let decision = resolve_backend_with_probe(&args, |_device| Ok(()));

        assert_eq!(decision.backend, RuntimeBackend::Cuda { device: 0 });
        assert!(decision.warning.is_none());
        assert!(decision.cuda_fallback.is_none());
    }

    #[test]
    fn cuda_request_falls_back_on_no_gpu() {
        let args = base_args(DeviceType::Cuda);
        let decision = resolve_backend_with_probe(&args, |_device| {
            Err(CudaProbeFailure {
                kind: CudaProbeFailureKind::NoGpu,
                detail: "no CUDA device detected".to_string(),
            })
        });

        assert_eq!(decision.backend, RuntimeBackend::Cpu);
        assert_eq!(decision.cuda_fallback, Some(CudaProbeFailureKind::NoGpu));
        assert!(decision
            .warning
            .as_deref()
            .unwrap_or_default()
            .contains("no-gpu"));
    }

    #[test]
    fn cuda_request_falls_back_on_missing_nvrtc() {
        let args = base_args(DeviceType::Cuda);
        let decision = resolve_backend_with_probe(&args, |_device| {
            Err(CudaProbeFailure {
                kind: CudaProbeFailureKind::MissingNvrtc,
                detail: "libnvrtc.so not found".to_string(),
            })
        });

        assert_eq!(decision.backend, RuntimeBackend::Cpu);
        assert_eq!(
            decision.cuda_fallback,
            Some(CudaProbeFailureKind::MissingNvrtc)
        );
        assert!(decision
            .warning
            .as_deref()
            .unwrap_or_default()
            .contains("missing-nvrtc"));
    }

    #[test]
    fn cuda_request_falls_back_on_unsupported_version() {
        let args = base_args(DeviceType::Cuda);
        let decision = resolve_backend_with_probe(&args, |_device| {
            Err(CudaProbeFailure {
                kind: CudaProbeFailureKind::UnsupportedCudaVersion,
                detail: "driver/toolkit mismatch".to_string(),
            })
        });

        assert_eq!(decision.backend, RuntimeBackend::Cpu);
        assert_eq!(
            decision.cuda_fallback,
            Some(CudaProbeFailureKind::UnsupportedCudaVersion)
        );
        assert!(decision
            .warning
            .as_deref()
            .unwrap_or_default()
            .contains("unsupported-cuda-version"));
    }
}
