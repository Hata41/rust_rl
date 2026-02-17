use anyhow::{Context, Result};
use clap::{parser::ValueSource, ArgMatches, CommandFactory, Parser, ValueEnum};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::ops::Deref;

#[derive(Copy, Clone, Debug, ValueEnum, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    Cuda,
    Cpu,
}

#[derive(Copy, Clone, Debug, ValueEnum, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ObservationAdapterKind {
    Dense,
    Binpack,
}

#[derive(Clone, Copy)]
enum TrainingAlgorithm {
    Unified,
    Ppo,
    Spo,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "rust_rl")]
pub struct Args {
    /// Optional YAML config file path.
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// rustpool task id: "Maze-v0", "BinPack-v0"
    #[arg(long, default_value = "BinPack-v0")]
    pub task_id: String,

    /// Number of parallel envs (Anakin-style throughput comes mostly from large N here).
    #[arg(long, default_value_t = 64)]
    pub num_envs: usize,

    /// PPO rollout length (T)
    #[arg(long, default_value_t = 128)]
    pub rollout_length: usize,

    /// Total PPO updates
    #[arg(long, default_value_t = 2000)]
    pub num_updates: usize,

    /// Number of updates between deterministic evaluations
    #[arg(long, default_value_t = 20)]
    pub eval_interval: usize,

    /// Number of environments to use for deterministic evaluation
    #[arg(long, default_value_t = 32)]
    pub num_eval_envs: usize,

    /// Total completed episodes to collect per deterministic evaluation
    #[arg(long, default_value_t = 32)]
    pub num_eval_episodes: usize,

    /// PPO epochs per update
    #[arg(long, default_value_t = 4)]
    pub epochs: usize,

    /// Number of minibatches per epoch (total batch size must be divisible by this)
    #[arg(long, default_value_t = 32)]
    pub num_minibatches: usize,

    /// Discount gamma
    #[arg(long, default_value_t = 0.99)]
    pub gamma: f32,

    /// GAE lambda
    #[arg(long, default_value_t = 0.95)]
    pub gae_lambda: f32,

    /// PPO clip epsilon
    #[arg(long, default_value_t = 0.2)]
    pub clip_eps: f32,

    /// Entropy coefficient
    #[arg(long, default_value_t = 0.01)]
    pub ent_coef: f32,

    /// Value loss coefficient
    #[arg(long, default_value_t = 0.5)]
    pub vf_coef: f32,

    /// Actor LR (Adam)
    #[arg(long, default_value_t = 3e-4)]
    pub actor_lr: f64,

    /// Critic LR (Adam)
    #[arg(long, default_value_t = 1e-3)]
    pub critic_lr: f64,

    /// Global gradient clipping threshold (L2 norm)
    #[arg(long, default_value_t = 0.5)]
    pub max_grad_norm: f32,

    /// Soft target-network update coefficient for SPO
    #[arg(long, default_value_t = 0.005)]
    pub tau: f64,

    /// Linearly decay actor/critic learning rates over updates
    #[arg(long, default_value_t = true)]
    pub decay_learning_rates: bool,

    /// Reward scaling (multiply env reward by this before GAE)
    #[arg(long, default_value_t = 1.0)]
    pub reward_scale: f32,

    /// Standardize advantages per update
    #[arg(long, default_value_t = true)]
    pub standardize_advantages: bool,

    /// Hidden dim of MLP
    #[arg(long, default_value_t = 256)]
    pub hidden_dim: usize,

    /// Explicit observation adapter selection. If omitted, adapter is auto-detected from env metadata.
    #[arg(long, value_enum)]
    pub observation_adapter: Option<ObservationAdapterKind>,

    /// RNG seed
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// CUDA device index (0 = cuda:0)
    #[arg(long, default_value_t = 0)]
    pub cuda_device: usize,

    /// Device backend to use: cuda or cpu
    #[arg(long, value_enum, default_value_t = DeviceType::Cuda)]
    pub device_type: DeviceType,

    #[arg(long, default_value_t = 20)]
    pub max_items: usize,

    #[arg(long, default_value_t = 40)]
    pub max_ems: usize,

    /// Max episode steps (used by some envs internally; rustpool auto-resets on done)
    #[arg(long, default_value_t = 200)]
    pub max_episode_steps: usize,

    /// SPO: number of search particles per environment.
    #[arg(long, default_value_t = 16)]
    pub num_particles: usize,

    /// SPO: search depth for SMC rollout.
    #[arg(long, default_value_t = 4)]
    pub search_depth: usize,

    /// SPO: replay buffer capacity in transitions.
    #[arg(long, default_value_t = 65_536)]
    pub replay_buffer_size: usize,

    /// SPO: sampled sequence length from replay.
    #[arg(long, default_value_t = 32)]
    pub sample_sequence_length: usize,

    /// SPO: period between sampled sequence start indices in replay.
    #[arg(long, default_value_t = 1)]
    pub sample_period: usize,

    /// SPO: dual optimizer learning rate.
    #[arg(long, default_value_t = 1e-3)]
    pub dual_lr: f64,

    /// SPO/MPO: E-step KL constraint.
    #[arg(long, default_value_t = 0.5)]
    pub epsilon: f32,

    /// SPO/MPO: policy KL constraint.
    #[arg(long, default_value_t = 1e-3)]
    pub epsilon_policy: f32,

    /// SPO/MPO: initial log temperature.
    #[arg(long, default_value_t = 3.0)]
    pub init_log_temperature: f32,

    /// SPO/MPO: initial log alpha.
    #[arg(long, default_value_t = 3.0)]
    pub init_log_alpha: f32,

    /// SPO: search discount factor.
    #[arg(long, default_value_t = 1.0)]
    pub search_gamma: f32,

    /// SPO: search GAE lambda.
    #[arg(long, default_value_t = 1.0)]
    pub search_gae_lambda: f32,

    /// SPO: root Dirichlet alpha for exploration.
    #[arg(long, default_value_t = 1.0)]
    pub root_exploration_dirichlet_alpha: f32,

    /// SPO: root Dirichlet mixing fraction.
    #[arg(long, default_value_t = 0.0)]
    pub root_exploration_dirichlet_fraction: f32,

    /// SPO: resampling period in steps.
    #[arg(long, default_value_t = 4)]
    pub resampling_period: usize,

    /// SPO: effective sample size threshold for resampling.
    #[arg(long, default_value_t = 0.5)]
    pub resampling_ess_threshold: f32,

    /// SPO: adaptive temperature flag.
    #[arg(long, default_value_t = true)]
    pub adaptive_temperature: bool,

    /// SPO: fixed temperature fallback when adaptive is disabled.
    #[arg(long, default_value_t = 0.5)]
    pub fixed_temperature: f32,

    /// Base tracing level used when RUST_LOG is not set.
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Show backend (CubeCL/CUDA) context logs in dashboard output.
    #[arg(long, default_value_t = false)]
    pub backend_logs_visible: bool,

    /// OTLP endpoint for telemetry export (use reverse SSH tunnel target, e.g. localhost).
    #[arg(long, default_value = "http://localhost:5000")]
    pub otlp_endpoint: String,

    /// MLflow run ID used to log scalar metrics.
    #[arg(long)]
    pub mlflow_run_id: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            config: None,
            task_id: "BinPack-v0".to_string(),
            num_envs: 64,
            rollout_length: 128,
            num_updates: 2000,
            eval_interval: 20,
            num_eval_envs: 32,
            num_eval_episodes: 32,
            epochs: 4,
            num_minibatches: 32,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_eps: 0.2,
            ent_coef: 0.01,
            vf_coef: 0.5,
            actor_lr: 3e-4,
            critic_lr: 1e-3,
            max_grad_norm: 0.5,
            tau: 0.005,
            decay_learning_rates: true,
            reward_scale: 1.0,
            standardize_advantages: true,
            hidden_dim: 256,
            observation_adapter: None,
            seed: 0,
            cuda_device: 0,
            device_type: DeviceType::Cuda,
            max_items: 20,
            max_ems: 40,
            max_episode_steps: 200,
            num_particles: 16,
            search_depth: 4,
            replay_buffer_size: 65_536,
            sample_sequence_length: 32,
            sample_period: 1,
            dual_lr: 1e-3,
            epsilon: 0.5,
            epsilon_policy: 1e-3,
            init_log_temperature: 3.0,
            init_log_alpha: 3.0,
            search_gamma: 1.0,
            search_gae_lambda: 1.0,
            root_exploration_dirichlet_alpha: 1.0,
            root_exploration_dirichlet_fraction: 0.0,
            resampling_period: 4,
            resampling_ess_threshold: 0.5,
            adaptive_temperature: true,
            fixed_temperature: 0.5,
            log_level: "info".to_string(),
            backend_logs_visible: false,
            otlp_endpoint: "http://localhost:5000".to_string(),
            mlflow_run_id: None,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct FileConfig {
    environment: EnvironmentConfig,
    #[serde(alias = "ppo_core")]
    training_core: TrainingCoreConfig,
    optimization: OptimizationConfig,
    architecture: ArchitectureConfig,
    evaluation: EvaluationConfig,
    hardware: HardwareConfig,
    spo: SpoConfig,
    logging: LoggingConfig,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct LoggingConfig {
    log_level: Option<String>,
    backend_logs_visible: Option<bool>,
    otlp_endpoint: Option<String>,
    mlflow_run_id: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct EnvironmentConfig {
    task_id: Option<String>,
    num_envs: Option<usize>,
    max_items: Option<usize>,
    max_ems: Option<usize>,
    max_episode_steps: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct TrainingCoreConfig {
    rollout_length: Option<usize>,
    num_updates: Option<usize>,
    epochs: Option<usize>,
    num_minibatches: Option<usize>,
    gamma: Option<f32>,
    gae_lambda: Option<f32>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct OptimizationConfig {
    actor_lr: Option<f64>,
    critic_lr: Option<f64>,
    max_grad_norm: Option<f32>,
    clip_eps: Option<f32>,
    ent_coef: Option<f32>,
    vf_coef: Option<f32>,
    decay_learning_rates: Option<bool>,
    standardize_advantages: Option<bool>,
    reward_scale: Option<f32>,
    tau: Option<f64>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct SpoConfig {
    num_particles: Option<usize>,
    search_depth: Option<usize>,
    replay_buffer_size: Option<usize>,
    sample_sequence_length: Option<usize>,
    sample_period: Option<usize>,
    dual_lr: Option<f64>,
    epsilon: Option<f32>,
    epsilon_policy: Option<f32>,
    init_log_temperature: Option<f32>,
    init_log_alpha: Option<f32>,
    search_gamma: Option<f32>,
    search_gae_lambda: Option<f32>,
    root_exploration_dirichlet_alpha: Option<f32>,
    root_exploration_dirichlet_fraction: Option<f32>,
    resampling_period: Option<usize>,
    resampling_ess_threshold: Option<f32>,
    adaptive_temperature: Option<bool>,
    fixed_temperature: Option<f32>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct ArchitectureConfig {
    hidden_dim: Option<usize>,
    observation_adapter: Option<ObservationAdapterKind>,
    seed: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct PpoArgs(Args);

impl PpoArgs {
    pub fn load() -> Result<Self> {
        Ok(Self(Args::load_for_algorithm(TrainingAlgorithm::Ppo)?))
    }

    pub fn into_inner(self) -> Args {
        self.0
    }
}

impl Deref for PpoArgs {
    type Target = Args;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct SpoArgs(Args);

impl SpoArgs {
    pub fn load() -> Result<Self> {
        Ok(Self(Args::load_for_algorithm(TrainingAlgorithm::Spo)?))
    }

    pub fn into_inner(self) -> Args {
        self.0
    }
}

impl Deref for SpoArgs {
    type Target = Args;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct EvaluationConfig {
    eval_interval: Option<usize>,
    num_eval_envs: Option<usize>,
    num_eval_episodes: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct HardwareConfig {
    device_type: Option<DeviceType>,
    cuda_device: Option<usize>,
}

impl Args {
    pub fn default_tracing_filter(&self) -> String {
        let mut directives = vec![self.log_level.trim().to_string()];

        if !self.backend_logs_visible {
            directives.push("cubecl_cuda=off".to_string());
            directives.push("cubecl_runtime=off".to_string());
        }

        directives.join(",")
    }

    pub fn load() -> Result<Self> {
        Self::load_for_algorithm(TrainingAlgorithm::Unified)
    }

    fn load_for_algorithm(algo: TrainingAlgorithm) -> Result<Self> {
        let argv = std::env::args_os().collect::<Vec<_>>();
        let cli_args = Self::try_parse_from(&argv)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .context("failed to parse CLI arguments")?;
        let matches = Self::command()
            .try_get_matches_from(&argv)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .context("failed to parse CLI arguments")?;

        let mut merged = Self::default();

        if let Some(config_path) = cli_args.config.as_deref() {
            let file_config = Self::load_file_config(config_path, algo)?;
            merged.apply_config_file(file_config);
        }

        merged.apply_cli_overrides(&cli_args, &matches);
        merged.config = cli_args.config;

        Ok(merged)
    }

    fn load_file_config(path: &Path, algo: TrainingAlgorithm) -> Result<FileConfig> {
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .context("failed to get current working directory")?
                .join(path)
        };

        let content = std::fs::read_to_string(&resolved)
            .with_context(|| format!("failed to read config file at {}", resolved.display()))?;

        if matches!(algo, TrainingAlgorithm::Ppo) {
            let mut value = serde_yaml::from_str::<serde_yaml::Value>(&content).with_context(|| {
                format!("failed to parse YAML config at {}", resolved.display())
            })?;
            if let Some(mapping) = value.as_mapping_mut() {
                mapping.remove(serde_yaml::Value::String("spo".to_string()));
            }
            return serde_yaml::from_value::<FileConfig>(value)
                .with_context(|| format!("failed to parse YAML config at {}", resolved.display()));
        }

        serde_yaml::from_str::<FileConfig>(&content)
            .with_context(|| format!("failed to parse YAML config at {}", resolved.display()))
    }

    fn apply_config_file(&mut self, file: FileConfig) {
        macro_rules! set_if_some {
            ($field:ident, $value:expr) => {
                if let Some(value) = $value {
                    self.$field = value;
                }
            };
        }

        set_if_some!(task_id, file.environment.task_id);
        set_if_some!(num_envs, file.environment.num_envs);
        set_if_some!(max_items, file.environment.max_items);
        set_if_some!(max_ems, file.environment.max_ems);
        set_if_some!(max_episode_steps, file.environment.max_episode_steps);

        set_if_some!(rollout_length, file.training_core.rollout_length);
        set_if_some!(num_updates, file.training_core.num_updates);
        set_if_some!(epochs, file.training_core.epochs);
        set_if_some!(num_minibatches, file.training_core.num_minibatches);
        set_if_some!(gamma, file.training_core.gamma);
        set_if_some!(gae_lambda, file.training_core.gae_lambda);

        set_if_some!(actor_lr, file.optimization.actor_lr);
        set_if_some!(critic_lr, file.optimization.critic_lr);
        set_if_some!(max_grad_norm, file.optimization.max_grad_norm);
        set_if_some!(clip_eps, file.optimization.clip_eps);
        set_if_some!(ent_coef, file.optimization.ent_coef);
        set_if_some!(vf_coef, file.optimization.vf_coef);
        set_if_some!(decay_learning_rates, file.optimization.decay_learning_rates);
        set_if_some!(
            standardize_advantages,
            file.optimization.standardize_advantages
        );
        set_if_some!(reward_scale, file.optimization.reward_scale);
        set_if_some!(tau, file.optimization.tau);

        set_if_some!(hidden_dim, file.architecture.hidden_dim);
        if let Some(value) = file.architecture.observation_adapter {
            self.observation_adapter = Some(value);
        }
        set_if_some!(seed, file.architecture.seed);

        set_if_some!(eval_interval, file.evaluation.eval_interval);
        set_if_some!(num_eval_envs, file.evaluation.num_eval_envs);
        set_if_some!(num_eval_episodes, file.evaluation.num_eval_episodes);

        set_if_some!(device_type, file.hardware.device_type);
        set_if_some!(cuda_device, file.hardware.cuda_device);

        set_if_some!(log_level, file.logging.log_level);
        set_if_some!(backend_logs_visible, file.logging.backend_logs_visible);
        set_if_some!(otlp_endpoint, file.logging.otlp_endpoint);
        if let Some(value) = file.logging.mlflow_run_id {
            self.mlflow_run_id = Some(value);
        }

        set_if_some!(num_particles, file.spo.num_particles);
        set_if_some!(search_depth, file.spo.search_depth);
        set_if_some!(replay_buffer_size, file.spo.replay_buffer_size);
        set_if_some!(sample_sequence_length, file.spo.sample_sequence_length);
        set_if_some!(sample_period, file.spo.sample_period);
        set_if_some!(dual_lr, file.spo.dual_lr);
        set_if_some!(epsilon, file.spo.epsilon);
        set_if_some!(epsilon_policy, file.spo.epsilon_policy);
        set_if_some!(init_log_temperature, file.spo.init_log_temperature);
        set_if_some!(init_log_alpha, file.spo.init_log_alpha);
        set_if_some!(search_gamma, file.spo.search_gamma);
        set_if_some!(search_gae_lambda, file.spo.search_gae_lambda);
        set_if_some!(
            root_exploration_dirichlet_alpha,
            file.spo.root_exploration_dirichlet_alpha
        );
        set_if_some!(
            root_exploration_dirichlet_fraction,
            file.spo.root_exploration_dirichlet_fraction
        );
        set_if_some!(resampling_period, file.spo.resampling_period);
        set_if_some!(resampling_ess_threshold, file.spo.resampling_ess_threshold);
        set_if_some!(adaptive_temperature, file.spo.adaptive_temperature);
        set_if_some!(fixed_temperature, file.spo.fixed_temperature);
    }

    fn apply_cli_overrides(&mut self, cli: &Self, matches: &ArgMatches) {
        macro_rules! set_if_cli {
            ($field:ident, $arg_name:literal) => {
                if Self::provided_on_cli(matches, $arg_name) {
                    self.$field = cli.$field.clone();
                }
            };
        }

        set_if_cli!(task_id, "task_id");
        set_if_cli!(num_envs, "num_envs");
        set_if_cli!(max_items, "max_items");
        set_if_cli!(max_ems, "max_ems");
        set_if_cli!(max_episode_steps, "max_episode_steps");

        set_if_cli!(rollout_length, "rollout_length");
        set_if_cli!(num_updates, "num_updates");
        set_if_cli!(epochs, "epochs");
        set_if_cli!(num_minibatches, "num_minibatches");
        set_if_cli!(gamma, "gamma");
        set_if_cli!(gae_lambda, "gae_lambda");

        set_if_cli!(actor_lr, "actor_lr");
        set_if_cli!(critic_lr, "critic_lr");
        set_if_cli!(max_grad_norm, "max_grad_norm");
        set_if_cli!(clip_eps, "clip_eps");
        set_if_cli!(ent_coef, "ent_coef");
        set_if_cli!(vf_coef, "vf_coef");
        set_if_cli!(decay_learning_rates, "decay_learning_rates");
        set_if_cli!(standardize_advantages, "standardize_advantages");
        set_if_cli!(reward_scale, "reward_scale");
        set_if_cli!(tau, "tau");

        set_if_cli!(hidden_dim, "hidden_dim");
        set_if_cli!(observation_adapter, "observation_adapter");
        set_if_cli!(seed, "seed");

        set_if_cli!(eval_interval, "eval_interval");
        set_if_cli!(num_eval_envs, "num_eval_envs");
        set_if_cli!(num_eval_episodes, "num_eval_episodes");

        set_if_cli!(device_type, "device_type");
        set_if_cli!(cuda_device, "cuda_device");

        set_if_cli!(log_level, "log_level");
        set_if_cli!(backend_logs_visible, "backend_logs_visible");
        set_if_cli!(otlp_endpoint, "otlp_endpoint");
        set_if_cli!(mlflow_run_id, "mlflow_run_id");

        set_if_cli!(num_particles, "num_particles");
        set_if_cli!(search_depth, "search_depth");
        set_if_cli!(replay_buffer_size, "replay_buffer_size");
        set_if_cli!(sample_sequence_length, "sample_sequence_length");
        set_if_cli!(sample_period, "sample_period");
        set_if_cli!(dual_lr, "dual_lr");
        set_if_cli!(epsilon, "epsilon");
        set_if_cli!(epsilon_policy, "epsilon_policy");
        set_if_cli!(init_log_temperature, "init_log_temperature");
        set_if_cli!(init_log_alpha, "init_log_alpha");
        set_if_cli!(search_gamma, "search_gamma");
        set_if_cli!(search_gae_lambda, "search_gae_lambda");
        set_if_cli!(
            root_exploration_dirichlet_alpha,
            "root_exploration_dirichlet_alpha"
        );
        set_if_cli!(
            root_exploration_dirichlet_fraction,
            "root_exploration_dirichlet_fraction"
        );
        set_if_cli!(resampling_period, "resampling_period");
        set_if_cli!(resampling_ess_threshold, "resampling_ess_threshold");
        set_if_cli!(adaptive_temperature, "adaptive_temperature");
        set_if_cli!(fixed_temperature, "fixed_temperature");
    }

    fn provided_on_cli(matches: &ArgMatches, arg_name: &str) -> bool {
        matches.value_source(arg_name) == Some(ValueSource::CommandLine)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DistInfo {
    pub rank: usize,
    pub world_size: usize,
    pub local_rank: usize,
}

impl DistInfo {
    pub fn from_env_or_args(_args: &Args) -> Result<Self> {
        Ok(Self {
            rank: 0,
            world_size: 1,
            local_rank: 0,
        })
    }
}
