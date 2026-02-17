use std::any::Any;
use std::ffi::{CStr, CString};
use std::sync::{Mutex, OnceLock};

use burn::backend::cuda::CudaDevice;
use cudarc::driver::{CudaContext, DriverError};
use tracing::warn;

use crate::common::config::{Args, DeviceType};

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
                "CUDA requested for device {} but probe failed ({}): {}.",
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
            panic!("{}", warning);
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

    probe_nvrtc_target_arch(cuda_device)?;

    catch_unwind_silent(|| CudaDevice::new(cuda_device)).map_err(classify_burn_cuda_panic)?;

    Ok(())
}

fn probe_nvrtc_target_arch(cuda_device: usize) -> Result<(), CudaProbeFailure> {
    let arch_version = query_cuda_arch_version(cuda_device)?;
    let mut logs = Vec::new();

    for candidate in nvrtc_arch_candidates(arch_version) {
        let arch_plain = format!("sm_{candidate}");
        match try_compile_with_arch(&arch_plain) {
            Ok(()) => return Ok(()),
            Err(log) => {
                logs.push(format!("{arch_plain}: {log}"));
                if !is_invalid_arch_log(&log) {
                    return Err(CudaProbeFailure {
                        kind: CudaProbeFailureKind::UnsupportedCudaVersion,
                        detail: format!(
                            "NVRTC failed for '--gpu-architecture={arch_plain}': {log}"
                        ),
                    });
                }
            }
        }
    }

    Err(CudaProbeFailure {
        kind: CudaProbeFailureKind::UnsupportedCudaVersion,
        detail: format!(
            "NVRTC rejected all tested architecture targets: {}",
            logs.join(" | ")
        ),
    })
}

fn query_cuda_arch_version(cuda_device: usize) -> Result<u32, CudaProbeFailure> {
    cudarc::driver::result::init().map_err(|err| CudaProbeFailure {
        kind: CudaProbeFailureKind::InitFailure,
        detail: format!("unable to initialize CUDA driver while probing architecture: {err:?}"),
    })?;

    let device_ptr = cudarc::driver::result::device::get(cuda_device as i32).map_err(|err| {
        classify_driver_error(err, cuda_device)
    })?;

    let major = unsafe {
        cudarc::driver::result::device::get_attribute(
            device_ptr,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .map_err(|err| classify_driver_error(err, cuda_device))?;

    let minor = unsafe {
        cudarc::driver::result::device::get_attribute(
            device_ptr,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .map_err(|err| classify_driver_error(err, cuda_device))?;

    Ok((major * 10 + minor) as u32)
}

fn try_compile_with_arch(arch: &str) -> Result<(), String> {
    let source = CString::new("extern \"C\" __global__ void rust_rl_arch_probe() {}")
        .map_err(|err| format!("failed to build NVRTC probe source: {err}"))?;
    let program = cudarc::nvrtc::result::create_program(source.as_c_str(), None)
        .map_err(|err| format!("failed to create NVRTC program: {err:?}"))?;
    let arch_option = format!("--gpu-architecture={arch}");
    let options = [arch_option.as_str()];

    if unsafe { cudarc::nvrtc::result::compile_program(program, &options) }.is_ok() {
        return Ok(());
    }

    let log_raw = unsafe { cudarc::nvrtc::result::get_program_log(program) }
        .map_err(|err| format!("NVRTC compile failed and log retrieval failed: {err:?}"))?;
    let log = unsafe {
        CStr::from_ptr(log_raw.as_ptr())
            .to_str()
            .unwrap_or("NVRTC compile failed (non UTF-8 log)")
    };
    Err(log.to_string())
}

fn nvrtc_arch_candidates(device_arch: u32) -> Vec<u32> {
    let mut values = vec![device_arch];
    for fallback in [120, 110, 100, 90, 89, 86, 80, 75, 70, 60, 52] {
        if fallback <= device_arch && !values.contains(&fallback) {
            values.push(fallback);
        }
    }
    values
}

fn is_invalid_arch_log(log: &str) -> bool {
    let lowered = log.to_lowercase();
    lowered.contains("invalid value for --gpu-architecture") || lowered.contains("(-arch)")
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
    use crate::common::config::{Args, DeviceType};

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
        let result = std::panic::catch_unwind(|| {
            resolve_backend_with_probe(&args, |_device| {
                Err(CudaProbeFailure {
                    kind: CudaProbeFailureKind::NoGpu,
                    detail: "no CUDA device detected".to_string(),
                })
            })
        });

        assert!(result.is_err());
    }

    #[test]
    fn cuda_request_falls_back_on_missing_nvrtc() {
        let args = base_args(DeviceType::Cuda);
        let result = std::panic::catch_unwind(|| {
            resolve_backend_with_probe(&args, |_device| {
                Err(CudaProbeFailure {
                    kind: CudaProbeFailureKind::MissingNvrtc,
                    detail: "libnvrtc.so not found".to_string(),
                })
            })
        });

        assert!(result.is_err());
    }

    #[test]
    fn cuda_request_falls_back_on_unsupported_version() {
        let args = base_args(DeviceType::Cuda);
        let result = std::panic::catch_unwind(|| {
            resolve_backend_with_probe(&args, |_device| {
                Err(CudaProbeFailure {
                    kind: CudaProbeFailureKind::UnsupportedCudaVersion,
                    detail: "driver/toolkit mismatch".to_string(),
                })
            })
        });

        assert!(result.is_err());
    }
}
