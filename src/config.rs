use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, Debug, ValueEnum, Eq, PartialEq)]
pub enum DeviceType {
    Cuda,
    Cpu,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "rust_ppo")]
pub struct Args {
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

    /// RNG seed
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// CUDA device index (0 = cuda:0)
    #[arg(long, default_value_t = 0)]
    pub cuda_device: usize,

    /// Device backend to use: cuda or cpu
    #[arg(long, value_enum, default_value_t = DeviceType::Cuda)]
    pub device_type: DeviceType,

    /// Global rank (used if RANK env var is not set)
    #[arg(long, default_value_t = 0)]
    pub rank: usize,

    /// World size (used if WORLD_SIZE env var is not set)
    #[arg(long, default_value_t = 1)]
    pub world_size: usize,

    /// Local rank / local GPU index (used if LOCAL_RANK env var is not set)
    #[arg(long, default_value_t = 0)]
    pub local_rank: usize,

    #[arg(long, default_value_t = 20)]
    pub max_items: usize,

    #[arg(long, default_value_t = 40)]
    pub max_ems: usize,

    /// Max episode steps (used by some envs internally; rustpool auto-resets on done)
    #[arg(long, default_value_t = 200)]
    pub max_episode_steps: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct DistInfo {
    pub rank: usize,
    pub world_size: usize,
    pub local_rank: usize,
}

impl DistInfo {
    pub fn from_env_or_args(args: &Args) -> Result<Self> {
        let rank = std::env::var("RANK")
            .ok()
            .map(|v| v.parse::<usize>().context("invalid RANK"))
            .transpose()?
            .unwrap_or(args.rank);

        let world_size = std::env::var("WORLD_SIZE")
            .ok()
            .map(|v| v.parse::<usize>().context("invalid WORLD_SIZE"))
            .transpose()?
            .unwrap_or(args.world_size);

        let local_rank = std::env::var("LOCAL_RANK")
            .ok()
            .map(|v| v.parse::<usize>().context("invalid LOCAL_RANK"))
            .transpose()?
            .unwrap_or(args.local_rank);

        if world_size == 0 {
            bail!("WORLD_SIZE must be > 0");
        }
        if rank >= world_size {
            bail!("RANK ({rank}) must be < WORLD_SIZE ({world_size})");
        }

        Ok(Self {
            rank,
            world_size,
            local_rank,
        })
    }
}