use cubecl_common::backtrace::BackTrace;
use cubecl_cpp::formatter::format_cpp;
use cubecl_cpp::{cuda::arch::CudaArchitecture, shared::CompilationOptions};
use cubecl_runtime::compiler::CompilationError;

use super::storage::gpu::GpuResource;
use crate::install::{cccl_include_path, include_path};
use crate::{CudaCompiler, compute::stream::Stream};
use cubecl_core::prelude::*;
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use cubecl_runtime::{compiler::CubeTask, logging::ServerLogger};
use cudarc::driver::sys::CUfunc_st;
use cudarc::driver::sys::{CUctx_st, CUfunction_attribute, CUtensorMap};
use std::collections::HashMap;
use std::ffi::CString;
use std::ffi::c_char;
use std::str::FromStr;
use std::sync::Arc;
use std::{ffi::CStr, os::raw::c_void};

use cubecl_common::cache::{Cache, CacheOption};

#[derive(Debug)]
pub(crate) struct CudaContext {
    pub context: *mut CUctx_st,
    pub module_names: HashMap<KernelId, CompiledKernel>,
    ptx_cache: Option<Cache<String, PtxCacheEntry>>,
    pub timestamps: TimestampProfiler,
    pub arch: CudaArchitecture,
    pub compilation_options: CompilationOptions,
}

#[derive(Debug)]
pub struct CompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct PtxCacheEntry {
    entrypoint_name: String,
    cube_dim: (u32, u32, u32),
    shared_mem_bytes: usize,
    cluster_dim: Option<(u32, u32, u32)>,
    ptx: Vec<std::ffi::c_char>,
}

impl CudaContext {
    pub fn new(
        compilation_options: CompilationOptions,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
    ) -> Self {
        Self {
            context,
            module_names: HashMap::new(),
            ptx_cache: {
                let config = cubecl_runtime::config::GlobalConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(Cache::new(
                        "ptx",
                        CacheOption::default().name("cuda").root(root),
                    ))
                } else {
                    None
                }
            },
            arch,
            timestamps: TimestampProfiler::default(),
            compilation_options,
        }
    }

    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        kernel: Box<dyn CubeTask<CudaCompiler>>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) -> Result<(), CompilationError> {
        let name = if let Some(cache) = &self.ptx_cache {
            let name = kernel_id.stable_format();

            if let Some(entry) = cache.get(&name) {
                log::trace!("Using PTX cache");

                return self.load_ptx(
                    entry.ptx.clone(),
                    kernel_id.clone(),
                    entry.entrypoint_name.clone(),
                    CubeDim {
                        x: entry.cube_dim.0,
                        y: entry.cube_dim.1,
                        z: entry.cube_dim.2,
                    },
                    entry.shared_mem_bytes,
                );
            }
            Some(name)
        } else {
            None
        };

        log::trace!("Compiling kernel");

        let mut kernel_compiled = kernel.compile(
            &mut Default::default(),
            &self.compilation_options,
            mode,
            kernel.address_type(),
        )?;

        if logger.compilation_activated() {
            kernel_compiled.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&kernel_compiled.source) {
                kernel_compiled.source = formatted;
            }
        }

        let compute_kernel = kernel_compiled.repr.as_ref().unwrap();
        let cube_dim = kernel_compiled.cube_dim;
        let arch_flags = nvrtc_arch_flags(self.arch.version);

        let include_path = include_path();
        let include_option = format!("--include-path={}", include_path.to_str().unwrap());
        let cccl_include_path = cccl_include_path();
        let cccl_include_option = format!("--include-path={}", cccl_include_path.to_str().unwrap());
        let mut base_options = vec![include_option, "-lineinfo".to_string()];
        if cccl_include_path.exists() {
            base_options.push(cccl_include_option);
        }

        let cluster_dim = compute_kernel.cluster_dim;

        logger.log_compilation(&kernel_compiled);

        let ptx = unsafe {
            // I'd like to set the name to the kernel name, but keep getting UTF-8 errors so let's
            // leave it `None` for now
            let source = CString::from_str(&kernel_compiled.source).unwrap();
            let mut last_log = String::new();
            let mut compiled_ptx: Option<Vec<c_char>> = None;

            'compile_attempt: for arch_flag in arch_flags.iter() {
                let program =
                    cudarc::nvrtc::result::create_program(source.as_c_str(), None).map_err(|err| {
                        CompilationError::Generic {
                            reason: format!("{err:?}"),
                            backtrace: BackTrace::capture(),
                        }
                    })?;

                let mut options = Vec::with_capacity(1 + base_options.len());
                options.push(arch_flag.as_str());
                for option in base_options.iter() {
                    options.push(option.as_str());
                }

                if cudarc::nvrtc::result::compile_program(program, &options).is_ok() {
                    let ptx = cudarc::nvrtc::result::get_ptx(program).map_err(|err| {
                        CompilationError::Generic {
                            reason: format!("{err:?}"),
                            backtrace: BackTrace::capture(),
                        }
                    })?;
                    compiled_ptx = Some(ptx);
                    break 'compile_attempt;
                }

                let log_raw = cudarc::nvrtc::result::get_program_log(program).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err:?}"),
                        backtrace: BackTrace::capture(),
                    }
                })?;
                let log = CStr::from_ptr(log_raw.as_ptr()).to_str().unwrap_or_default();
                last_log = log.to_string();

                if !is_invalid_arch_log(log) {
                    break 'compile_attempt;
                }
            }

            if let Some(ptx) = compiled_ptx {
                ptx
            } else {
                let mut message = "[Compilation Error] ".to_string();
                for line in last_log.split('\n') {
                    if !line.is_empty() {
                        message += format!("\n    {line}").as_str();
                    }
                }
                let source = kernel
                    .compile(
                        &mut Default::default(),
                        &self.compilation_options,
                        mode,
                        kernel.address_type(),
                    )?
                    .source;
                Err(CompilationError::Generic {
                    reason: format!("{message}\n[Source]  \n{source}"),
                    backtrace: BackTrace::capture(),
                })?
            }
        };

        let repr = kernel_compiled.repr.unwrap();

        if let Some(cache) = &mut self.ptx_cache {
            let result = cache.insert(
                name.unwrap(),
                PtxCacheEntry {
                    entrypoint_name: kernel_compiled.entrypoint_name.clone(),
                    cube_dim: (cube_dim.x, cube_dim.y, cube_dim.z),
                    shared_mem_bytes: repr.shared_memory_size(),
                    cluster_dim: cluster_dim.map(|cluster| (cluster.x, cluster.y, cluster.z)),
                    ptx: ptx.clone(),
                },
            );
            if let Err(err) = result {
                log::warn!("Unable to save the ptx {err:?}");
            }
        }

        self.load_ptx(
            ptx,
            kernel_id.clone(),
            kernel_compiled.entrypoint_name,
            cube_dim,
            repr.shared_memory_size(),
        )
    }

    fn load_ptx(
        &mut self,
        ptx: Vec<c_char>,
        kernel_id: KernelId,
        entrypoint_name: String,
        cube_dim: CubeDim,
        shared_mem_bytes: usize,
    ) -> Result<(), CompilationError> {
        let func_name = CString::new(entrypoint_name).unwrap();
        let func = unsafe {
            let module = cudarc::driver::result::module::load_data(ptx.as_ptr() as *const _)
                .map_err(|err| CompilationError::Generic {
                    reason: format!("Unable to load the PTX: {err:?}"),
                    backtrace: BackTrace::capture(),
                })?;

            cudarc::driver::result::module::get_function(module, func_name).map_err(|err| {
                CompilationError::Generic {
                    reason: format!("Unable to fetch the function from the module: {err:?}"),
                    backtrace: BackTrace::capture(),
                }
            })?
        };

        self.module_names.insert(
            kernel_id.clone(),
            CompiledKernel {
                cube_dim,
                shared_mem_bytes,
                func,
            },
        );

        Ok(())
    }

    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        tensor_maps: &[CUtensorMap],
        resources: &[GpuResource],
        scalars: &[*mut c_void],
    ) -> Result<(), String> {
        let mut bindings = tensor_maps
            .iter()
            .map(|map| map as *const _ as *mut c_void)
            .collect::<Vec<_>>();
        bindings.extend(resources.iter().map(|memory| memory.binding));
        bindings.extend(scalars);

        let kernel = self.module_names.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;
        unsafe {
            cudarc::driver::result::function::set_function_attribute(
                kernel.func,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                kernel.shared_mem_bytes as i32,
            )
            .map_err(|err| format!("{err:?}"))?;
            cudarc::driver::result::launch_kernel(
                kernel.func,
                dispatch_count,
                (cube_dim.x, cube_dim.y, cube_dim.z),
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                &mut bindings,
            )
            .map_err(|err| format!("{err:?}"))?;
        };

        Ok(())
    }
}

fn nvrtc_arch_flags(device_arch: u32) -> Vec<String> {
    let mut versions = vec![device_arch];
    for fallback in [120, 110, 100, 90, 89, 86, 80, 75, 70, 60, 52] {
        if fallback <= device_arch && !versions.contains(&fallback) {
            versions.push(fallback);
        }
    }
    versions
        .into_iter()
        .map(|version| format!("--gpu-architecture=sm_{version}"))
        .collect()
}

fn is_invalid_arch_log(log: &str) -> bool {
    let lowered = log.to_lowercase();
    lowered.contains("invalid value for --gpu-architecture") || lowered.contains("(-arch)")
}
