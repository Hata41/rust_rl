use rand::Rng;
use rustpool::core::types::GenericObs;
use std::sync::Arc;

use crate::common::utils::buffer::{flatten_obs_nonempty, BufferStorage};

#[derive(Clone, Debug)]
pub struct ReplayBatch {
    pub obs: Vec<Arc<GenericObs>>,
    pub next_obs: Vec<Arc<GenericObs>>,
    pub actions: Vec<i32>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub action_masks: Vec<bool>,
    pub next_action_masks: Vec<bool>,
    pub root_action_weights: Vec<f32>,
    pub sampled_actions: Vec<i32>,
    pub sampled_advantages: Vec<f32>,
    pub action_dim: usize,
    pub num_particles: usize,
}

#[derive(Debug)]
pub struct ReplayBuffer {
    add_batch_size: usize,
    max_length_time_axis: usize,
    current_index: usize,
    is_full: bool,
    action_dim: usize,
    num_particles: usize,
    obs: Vec<Arc<GenericObs>>,
    next_obs: Vec<Arc<GenericObs>>,
    actions: Vec<i32>,
    rewards: Vec<f32>,
    dones: Vec<bool>,
    action_masks: Vec<bool>,
    next_action_masks: Vec<bool>,
    root_action_weights: Vec<f32>,
    sampled_actions: Vec<i32>,
    sampled_advantages: Vec<f32>,
}

impl ReplayBuffer {
    pub fn new(
        capacity: usize,
        add_batch_size: usize,
        action_dim: usize,
        num_particles: usize,
    ) -> Self {
        assert!(add_batch_size > 0, "add_batch_size must be > 0");
        assert!(num_particles > 0, "num_particles must be > 0");
        let max_length_time_axis = (capacity / add_batch_size).max(1);
        let slots = add_batch_size * max_length_time_axis;

        Self {
            add_batch_size,
            max_length_time_axis,
            current_index: 0,
            is_full: false,
            action_dim,
            num_particles,
            obs: vec![Arc::new(Vec::new()); slots],
            next_obs: vec![Arc::new(Vec::new()); slots],
            actions: vec![0; slots],
            rewards: vec![0.0; slots],
            dones: vec![false; slots],
            action_masks: vec![false; slots * action_dim],
            next_action_masks: vec![false; slots * action_dim],
            root_action_weights: vec![0.0; slots * action_dim],
            sampled_actions: vec![0; slots * num_particles],
            sampled_advantages: vec![0.0; slots * num_particles],
        }
    }

    pub fn len(&self) -> usize {
        if self.is_full {
            self.add_batch_size * self.max_length_time_axis
        } else {
            self.add_batch_size * self.current_index
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn idx(&self, env_idx: usize, time_idx: usize) -> usize {
        env_idx * self.max_length_time_axis + time_idx
    }

    pub fn can_sample(&self, min_length_time_axis: usize) -> bool {
        self.is_full || self.current_index >= min_length_time_axis
    }

    pub fn add_timestep(
        &mut self,
        obs_batch: &[&GenericObs],
        next_obs_batch: &[&GenericObs],
        actions: &[i32],
        rewards: &[f32],
        dones: &[bool],
        action_masks: &[&[bool]],
        next_action_masks: &[&[bool]],
        root_action_weights: &[f32],
        sampled_actions: &[i32],
        sampled_advantages: &[f32],
    ) -> anyhow::Result<()> {
        if obs_batch.len() != self.add_batch_size
            || next_obs_batch.len() != self.add_batch_size
            || actions.len() != self.add_batch_size
            || rewards.len() != self.add_batch_size
            || dones.len() != self.add_batch_size
            || action_masks.len() != self.add_batch_size
            || next_action_masks.len() != self.add_batch_size
        {
            anyhow::bail!(
                "add_timestep batch mismatch: expected {}, got obs={}, next_obs={}, actions={}, rewards={}, dones={}, action_masks={}, next_action_masks={}",
                self.add_batch_size,
                obs_batch.len(),
                next_obs_batch.len(),
                actions.len(),
                rewards.len(),
                dones.len(),
                action_masks.len(),
                next_action_masks.len(),
            );
        }

        if root_action_weights.len() != self.add_batch_size * self.action_dim {
            anyhow::bail!(
                "replay root_action_weights mismatch: got {}, expected {}",
                root_action_weights.len(),
                self.add_batch_size * self.action_dim
            );
        }

        if sampled_actions.len() != self.add_batch_size * self.num_particles {
            anyhow::bail!(
                "replay sampled_actions mismatch: got {}, expected {}",
                sampled_actions.len(),
                self.add_batch_size * self.num_particles
            );
        }
        if sampled_advantages.len() != self.add_batch_size * self.num_particles {
            anyhow::bail!(
                "replay sampled_advantages mismatch: got {}, expected {}",
                sampled_advantages.len(),
                self.add_batch_size * self.num_particles
            );
        }

        let write_time = self.current_index;
        for env_idx in 0..self.add_batch_size {
            if action_masks[env_idx].len() != self.action_dim
                || next_action_masks[env_idx].len() != self.action_dim
            {
                anyhow::bail!(
                    "replay action_dim mismatch at env {}: got ({}, {}), expected {}",
                    env_idx,
                    action_masks[env_idx].len(),
                    next_action_masks[env_idx].len(),
                    self.action_dim
                );
            }

            let idx = self.idx(env_idx, write_time);
            let m0 = idx * self.action_dim;
            let rw0 = env_idx * self.action_dim;
            let p0 = idx * self.num_particles;
            let ps0 = env_idx * self.num_particles;

            self.obs[idx] = Arc::new((*obs_batch[env_idx]).clone());
            self.next_obs[idx] = Arc::new((*next_obs_batch[env_idx]).clone());
            self.actions[idx] = actions[env_idx];
            self.rewards[idx] = rewards[env_idx];
            self.dones[idx] = dones[env_idx];
            self.action_masks[m0..m0 + self.action_dim].copy_from_slice(&action_masks[env_idx]);
            self.next_action_masks[m0..m0 + self.action_dim]
                .copy_from_slice(&next_action_masks[env_idx]);
            self.root_action_weights[m0..m0 + self.action_dim]
                .copy_from_slice(&root_action_weights[rw0..rw0 + self.action_dim]);
            self.sampled_actions[p0..p0 + self.num_particles]
                .copy_from_slice(&sampled_actions[ps0..ps0 + self.num_particles]);
            self.sampled_advantages[p0..p0 + self.num_particles]
                .copy_from_slice(&sampled_advantages[ps0..ps0 + self.num_particles]);
        }

        let next_index = self.current_index + 1;
        self.is_full = self.is_full || next_index >= self.max_length_time_axis;
        self.current_index = next_index % self.max_length_time_axis;

        Ok(())
    }

    pub fn sample_sequences(
        &self,
        num_sequences: usize,
        sequence_len: usize,
        period: usize,
        rng: &mut impl Rng,
    ) -> ReplayBatch {
        if sequence_len == 0 || num_sequences == 0 || period == 0 {
            return ReplayBatch {
                obs: Vec::new(),
                next_obs: Vec::new(),
                actions: Vec::new(),
                rewards: Vec::new(),
                dones: Vec::new(),
                action_masks: Vec::new(),
                next_action_masks: Vec::new(),
                root_action_weights: Vec::new(),
                sampled_actions: Vec::new(),
                sampled_advantages: Vec::new(),
                action_dim: self.action_dim,
                num_particles: self.num_particles,
            };
        }

        let max_time = if self.is_full {
            self.max_length_time_axis
        } else {
            self.current_index
        };
        let head = if self.is_full { self.current_index } else { 0 };

        if max_time < sequence_len {
            return ReplayBatch {
                obs: Vec::new(),
                next_obs: Vec::new(),
                actions: Vec::new(),
                rewards: Vec::new(),
                dones: Vec::new(),
                action_masks: Vec::new(),
                next_action_masks: Vec::new(),
                root_action_weights: Vec::new(),
                sampled_actions: Vec::new(),
                sampled_advantages: Vec::new(),
                action_dim: self.action_dim,
                num_particles: self.num_particles,
            };
        }

        let max_start = max_time - sequence_len;
        let num_valid_items = (max_start / period) + 1;
        let batch = num_sequences * sequence_len;

        let mut obs = Vec::with_capacity(batch);
        let mut next_obs = Vec::with_capacity(batch);
        let mut actions = Vec::with_capacity(batch);
        let mut rewards = Vec::with_capacity(batch);
        let mut dones = Vec::with_capacity(batch);
        let mut action_masks = Vec::with_capacity(batch * self.action_dim);
        let mut next_action_masks = Vec::with_capacity(batch * self.action_dim);
        let mut root_action_weights = Vec::with_capacity(batch * self.action_dim);
        let mut sampled_actions = Vec::with_capacity(batch * self.num_particles);
        let mut sampled_advantages = Vec::with_capacity(batch * self.num_particles);

        for _ in 0..num_sequences {
            let sampled_item_idx = if num_valid_items == 1 {
                0
            } else {
                rng.gen_range(0..num_valid_items)
            };
            let logical_start = sampled_item_idx * period;
            let physical_start = (head + logical_start) % self.max_length_time_axis;
            let sampled_env_idx = if self.add_batch_size == 1 {
                0
            } else {
                rng.gen_range(0..self.add_batch_size)
            };

            for offset in 0..sequence_len {
                let time_idx = (physical_start + offset) % self.max_length_time_axis;
                let idx = self.idx(sampled_env_idx, time_idx);
                let m0 = idx * self.action_dim;

                obs.push(self.obs[idx].clone());
                next_obs.push(self.next_obs[idx].clone());
                actions.push(self.actions[idx]);
                rewards.push(self.rewards[idx]);
                dones.push(self.dones[idx]);
                action_masks.extend_from_slice(&self.action_masks[m0..m0 + self.action_dim]);
                next_action_masks
                    .extend_from_slice(&self.next_action_masks[m0..m0 + self.action_dim]);
                root_action_weights
                    .extend_from_slice(&self.root_action_weights[m0..m0 + self.action_dim]);
                let p0 = idx * self.num_particles;
                sampled_actions
                    .extend_from_slice(&self.sampled_actions[p0..p0 + self.num_particles]);
                sampled_advantages
                    .extend_from_slice(&self.sampled_advantages[p0..p0 + self.num_particles]);
            }
        }

        ReplayBatch {
            obs,
            next_obs,
            actions,
            rewards,
            dones,
            action_masks,
            next_action_masks,
            root_action_weights,
            sampled_actions,
            sampled_advantages,
            action_dim: self.action_dim,
            num_particles: self.num_particles,
        }
    }
}

impl BufferStorage for ReplayBuffer {
    fn len(&self) -> usize {
        if self.is_full {
            self.add_batch_size * self.max_length_time_axis
        } else {
            self.add_batch_size * self.current_index
        }
    }
}

pub fn flatten_obs_once(obs: &GenericObs) -> anyhow::Result<Vec<f32>> {
    flatten_obs_nonempty(obs)
}
