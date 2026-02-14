use anyhow::{bail, Context, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::collections::HashMap;
use std::thread;

use rustpool::core::rl_env::RlEnv;
use rustpool::core::types::GenericObs;
use rustpool::core::worker::{worker_loop, WorkerAction, WorkerMessage};
use rustpool::envs::binpack::{BinPackConfig, BinPackEnv, RewardFnType};
use rustpool::envs::maze::{MazeConfig, MazeEnv};

use crate::config::Args;

#[derive(Clone, Debug)]
pub struct StepOut {
    pub obs: GenericObs,
    pub reward: f32,
    pub done: bool,
    pub action_mask: Vec<bool>,
}

/// Minimal Rust envpool (pure Rust) built on rustpool's worker_loop.
pub struct AsyncEnvPool {
    num_envs: usize,
    num_threads: usize,
    action_txs: Vec<Sender<WorkerAction>>,
    state_rx: Receiver<WorkerMessage>,
    worker_handles: Vec<thread::JoinHandle<()>>,
}

impl AsyncEnvPool {
    pub fn new<F>(num_envs: usize, base_seed: u64, mut factory: F) -> Result<Self>
    where
        F: FnMut(u64) -> Box<dyn RlEnv> + Send + 'static,
    {
        if num_envs == 0 {
            bail!("num_envs must be > 0");
        }

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(num_envs);

        let mut shards: Vec<HashMap<usize, Box<dyn RlEnv>>> =
            (0..num_threads).map(|_| HashMap::new()).collect();

        for env_id in 0..num_envs {
            let shard_id = env_id % num_threads;
            let env = factory(base_seed + env_id as u64);
            shards[shard_id].insert(env_id, env);
        }

        let (state_tx, state_rx) = unbounded::<WorkerMessage>();
        let mut action_txs = Vec::with_capacity(num_threads);
        let mut worker_handles = Vec::with_capacity(num_threads);

        for shard_id in 0..num_threads {
            let (action_tx, action_rx) = unbounded::<WorkerAction>();
            action_txs.push(action_tx);

            let envs_for_worker = std::mem::take(&mut shards[shard_id]);
            let state_tx_clone = state_tx.clone();

            let handle = thread::spawn(move || {
                worker_loop(shard_id, envs_for_worker, action_rx, state_tx_clone)
            });
            worker_handles.push(handle);
        }

        Ok(Self {
            num_envs,
            num_threads,
            action_txs,
            state_rx,
            worker_handles,
        })
    }

    fn route_tx(&self, env_id: usize) -> &Sender<WorkerAction> {
        let shard = env_id % self.num_threads;
        &self.action_txs[shard]
    }

    fn async_reset(&self, seed: Option<u64>) -> Result<()> {
        let seed_map = seed.map(|s| {
            (0..self.num_envs)
                .map(|env_id| (env_id, s + env_id as u64))
                .collect::<Vec<_>>()
        });

        for tx in &self.action_txs {
            tx.send(WorkerAction::Reset(seed_map.clone()))
                .context("failed to send reset")?;
        }
        Ok(())
    }

    fn recv_n(&self, n: usize) -> Result<Vec<StepOut>> {
        let mut out = vec![
            StepOut {
                obs: vec![],
                reward: 0.0,
                done: false,
                action_mask: vec![],
            };
            n
        ];

        let mut got = 0usize;
        while got < n {
            match self.state_rx.recv().context("envpool recv failed")? {
                WorkerMessage::StepResult {
                    env_id,
                    obs,
                    reward,
                    done,
                    action_mask,
                } => {
                    if env_id < n {
                        out[env_id] = StepOut {
                            obs,
                            reward,
                            done,
                            action_mask,
                        };
                    }
                    got += 1;
                }
                WorkerMessage::SnapshotResult { .. } => {}
            }
        }
        Ok(out)
    }

    pub fn reset_all(&self, seed: Option<u64>) -> Result<Vec<StepOut>> {
        self.async_reset(seed)?;
        self.recv_n(self.num_envs)
    }

    pub fn step_all(&self, actions: &[i32]) -> Result<Vec<StepOut>> {
        if actions.len() != self.num_envs {
            bail!("actions length must match num_envs");
        }

        for (env_id, action) in actions.iter().enumerate() {
            self.route_tx(env_id)
                .send(WorkerAction::Step(env_id, *action))
                .context("failed to send step")?;
        }

        self.recv_n(self.num_envs)
    }
}

impl Drop for AsyncEnvPool {
    fn drop(&mut self) {
        for tx in &self.action_txs {
            let _ = tx.send(WorkerAction::Close);
        }
        for h in self.worker_handles.drain(..) {
            let _ = h.join();
        }
    }
}

pub fn make_env(task_id: &str, args: &Args, seed: u64) -> Result<Box<dyn RlEnv>> {
    match task_id {
        "Maze-v0" => {
            let cfg = MazeConfig {
                width: 10,
                height: 10,
                max_episode_steps: args.max_episode_steps as i32,
            };
            Ok(Box::new(MazeEnv::new(cfg, seed)))
        }
        "BinPack-v0" => {
            let cfg = BinPackConfig {
                max_items: args.max_items,
                max_ems: args.max_ems,
                split_eps: 0.001,
                prob_split_one_item: 0.3,
                split_num_same_items: 1,
            };
            Ok(Box::new(BinPackEnv::new(cfg, seed, RewardFnType::Dense)))
        }
        other => bail!("unknown task_id: {other}"),
    }
}