use anyhow::{bail, Context, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use rayon::prelude::*;
use rustpool::core::mcts::StateRegistry;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Mutex;
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
    pub state_ids: Vec<i32>,
}

/// Minimal Rust envpool (pure Rust) built on rustpool's worker_loop.
pub struct AsyncEnvPool {
    num_envs: usize,
    num_threads: usize,
    action_txs: Vec<Sender<WorkerAction>>,
    state_rx: Receiver<WorkerMessage>,
    message_buffer: Mutex<VecDeque<WorkerMessage>>,
    registry: StateRegistry,
    active_state_ids: Mutex<HashSet<i32>>,
    live_state_ids: AtomicI64,
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
            message_buffer: Mutex::new(VecDeque::new()),
            registry: StateRegistry::new(),
            active_state_ids: Mutex::new(HashSet::new()),
            live_state_ids: AtomicI64::new(0),
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
                state_ids: vec![],
            };
            n
        ];

        let mut got = 0usize;
        while got < n {
            match self.recv_step_message()? {
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
                            state_ids: vec![],
                        };
                    }
                    got += 1;
                }
                WorkerMessage::SnapshotResult { .. } => {
                    unreachable!("recv_step_message returns only step results")
                }
            }
        }
        Ok(out)
    }

    fn recv_step_message(&self) -> Result<WorkerMessage> {
        {
            let mut buffer = self
                .message_buffer
                .lock()
                .expect("message buffer mutex poisoned");
            if let Some(idx) = buffer
                .iter()
                .position(|msg| matches!(msg, WorkerMessage::StepResult { .. }))
            {
                if let Some(msg) = buffer.remove(idx) {
                    return Ok(msg);
                }
            }
        }

        loop {
            let msg = self.state_rx.recv().context("envpool recv failed")?;
            match msg {
                WorkerMessage::StepResult { .. } => return Ok(msg),
                WorkerMessage::SnapshotResult { .. } => {
                    let mut buffer = self
                        .message_buffer
                        .lock()
                        .expect("message buffer mutex poisoned");
                    buffer.push_back(msg);
                }
            }
        }
    }

    fn recv_snapshot_message(&self, env_ids: &[usize]) -> Result<(usize, Box<dyn RlEnv>)> {
        {
            let mut buffer = self
                .message_buffer
                .lock()
                .expect("message buffer mutex poisoned");
            if let Some(idx) = buffer.iter().position(|msg| {
                matches!(msg, WorkerMessage::SnapshotResult { env_id, .. } if env_ids.contains(env_id))
            }) {
                if let Some(WorkerMessage::SnapshotResult { env_id, snapshot }) = buffer.remove(idx) {
                    return Ok((env_id, snapshot));
                }
            }
        }

        loop {
            let msg = self.state_rx.recv().context("envpool recv failed")?;
            match msg {
                WorkerMessage::SnapshotResult { env_id, snapshot } if env_ids.contains(&env_id) => {
                    return Ok((env_id, snapshot));
                }
                other => {
                    let mut buffer = self
                        .message_buffer
                        .lock()
                        .expect("message buffer mutex poisoned");
                    buffer.push_back(other);
                }
            }
        }
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

    pub fn snapshot(&self, env_ids: &[usize]) -> Result<Vec<i32>> {
        if env_ids.is_empty() {
            return Ok(vec![]);
        }

        let mut shard_requests: Vec<Vec<usize>> =
            (0..self.num_threads).map(|_| Vec::new()).collect();
        for &env_id in env_ids {
            if env_id >= self.num_envs {
                bail!(
                    "env_id {env_id} out of bounds for num_envs {}",
                    self.num_envs
                );
            }
            let shard_id = env_id % self.num_threads;
            shard_requests[shard_id].push(env_id);
        }

        for (shard_id, ids) in shard_requests.into_iter().enumerate() {
            if !ids.is_empty() {
                self.action_txs[shard_id]
                    .send(WorkerAction::Snapshot(ids))
                    .context("failed to send snapshot request")?;
            }
        }

        let mut snapshots = HashMap::new();
        let mut remaining = env_ids.len();
        while remaining > 0 {
            let (env_id, snapshot) = self.recv_snapshot_message(env_ids)?;
            if snapshots.insert(env_id, snapshot).is_none() {
                remaining -= 1;
            }
        }

        let ordered = env_ids
            .iter()
            .map(|env_id| snapshots.remove(env_id).expect("snapshot should exist"))
            .collect::<Vec<_>>();

        let ids = self.registry.snapshot(ordered);
        {
            let mut active = self
                .active_state_ids
                .lock()
                .expect("active_state_ids mutex poisoned");
            let mut inserted = 0i64;
            for &id in ids.iter() {
                if active.insert(id) {
                    inserted += 1;
                }
            }
            if inserted > 0 {
                self.live_state_ids.fetch_add(inserted, Ordering::Relaxed);
            }
        }
        Ok(ids)
    }

    pub fn simulate_batch(&self, state_ids: &[i32], actions: &[i32]) -> Result<Vec<StepOut>> {
        if state_ids.len() != actions.len() {
            bail!(
                "state_ids length ({}) must match actions length ({})",
                state_ids.len(),
                actions.len()
            );
        }
        if state_ids.is_empty() {
            return Ok(vec![]);
        }

        let envs = self
            .registry
            .get_clones(state_ids)
            .map_err(anyhow::Error::msg)?;
        let rollouts = envs
            .into_par_iter()
            .zip(actions.par_iter().copied())
            .map(|(mut env, action)| {
                let (obs, reward, done, action_mask) = env.step(action);
                (env, obs, reward, done, action_mask)
            })
            .collect::<Vec<_>>();

        let mut next_envs = Vec::with_capacity(rollouts.len());
        let mut out = Vec::with_capacity(rollouts.len());
        for (env, obs, reward, done, action_mask) in rollouts {
            next_envs.push(env);
            out.push(StepOut {
                obs,
                reward,
                done,
                action_mask,
                state_ids: vec![],
            });
        }

        let new_state_ids = self.registry.snapshot(next_envs);
        {
            let mut active = self
                .active_state_ids
                .lock()
                .expect("active_state_ids mutex poisoned");
            let mut inserted = 0i64;
            for &id in new_state_ids.iter() {
                if active.insert(id) {
                    inserted += 1;
                }
            }
            if inserted > 0 {
                self.live_state_ids.fetch_add(inserted, Ordering::Relaxed);
            }
        }
        for (step, sid) in out.iter_mut().zip(new_state_ids.into_iter()) {
            step.state_ids.push(sid);
        }

        Ok(out)
    }

    pub fn release_batch(&self, state_ids: &[i32]) {
        if !state_ids.is_empty() {
            let unique_state_ids = state_ids
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            self.registry.release(&unique_state_ids);

            let removed = {
                let mut active = self
                    .active_state_ids
                    .lock()
                    .expect("active_state_ids mutex poisoned");
                let mut removed = 0i64;
                for id in unique_state_ids.iter() {
                    if active.remove(id) {
                        removed += 1;
                    }
                }
                removed
            };

            if removed > 0 {
                self.live_state_ids.fetch_sub(removed, Ordering::Relaxed);
            }
        }

        #[cfg(debug_assertions)]
        {
            let live = self.live_state_ids.load(Ordering::Relaxed);
            debug_assert!(
                live >= 0,
                "release_batch underflow detected: live state counter is {}",
                live
            );
        }
    }
}

impl Drop for AsyncEnvPool {
    fn drop(&mut self) {
        let remaining_ids = {
            let active = self
                .active_state_ids
                .lock()
                .expect("active_state_ids mutex poisoned");
            active.iter().copied().collect::<Vec<_>>()
        };
        if !remaining_ids.is_empty() {
            self.registry.release(&remaining_ids);
            self.live_state_ids.store(0, Ordering::Relaxed);
        }

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
