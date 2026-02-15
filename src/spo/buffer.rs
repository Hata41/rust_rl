use rustpool::core::types::GenericObs;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct ReplayBatch {
    pub obs: Vec<GenericObs>,
    pub next_obs: Vec<GenericObs>,
    pub actions: Vec<i32>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub action_masks: Vec<bool>,
    pub next_action_masks: Vec<bool>,
    pub root_action_weights: Vec<f32>,
    pub action_dim: usize,
}

#[derive(Debug)]
pub struct ReplayBuffer {
    capacity: usize,
    len: usize,
    write_idx: usize,
    action_dim: usize,
    obs: Vec<GenericObs>,
    next_obs: Vec<GenericObs>,
    actions: Vec<i32>,
    rewards: Vec<f32>,
    dones: Vec<bool>,
    action_masks: Vec<bool>,
    next_action_masks: Vec<bool>,
    root_action_weights: Vec<f32>,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, action_dim: usize) -> Self {
        Self {
            capacity,
            len: 0,
            write_idx: 0,
            action_dim,
            obs: vec![Vec::new(); capacity],
            next_obs: vec![Vec::new(); capacity],
            actions: vec![0; capacity],
            rewards: vec![0.0; capacity],
            dones: vec![false; capacity],
            action_masks: vec![false; capacity * action_dim],
            next_action_masks: vec![false; capacity * action_dim],
            root_action_weights: vec![0.0; capacity * action_dim],
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn add(
        &mut self,
        obs: &GenericObs,
        next_obs: &GenericObs,
        action: i32,
        reward: f32,
        done: bool,
        action_mask: &[bool],
        next_action_mask: &[bool],
        root_action_weights: &[f32],
    ) -> anyhow::Result<()> {
        if action_mask.len() != self.action_dim || next_action_mask.len() != self.action_dim {
            anyhow::bail!(
                "replay action_dim mismatch: got ({}, {}), expected {}",
                action_mask.len(),
                next_action_mask.len(),
                self.action_dim
            );
        }
        if root_action_weights.len() != self.action_dim {
            anyhow::bail!(
                "replay root_action_weights mismatch: got {}, expected {}",
                root_action_weights.len(),
                self.action_dim
            );
        }

        let idx = self.write_idx;
        let m0 = idx * self.action_dim;

    self.obs[idx] = obs.clone();
    self.next_obs[idx] = next_obs.clone();
        self.actions[idx] = action;
        self.rewards[idx] = reward;
        self.dones[idx] = done;
        self.action_masks[m0..m0 + self.action_dim].copy_from_slice(action_mask);
        self.next_action_masks[m0..m0 + self.action_dim].copy_from_slice(next_action_mask);
        self.root_action_weights[m0..m0 + self.action_dim].copy_from_slice(root_action_weights);

        self.write_idx = (self.write_idx + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
        Ok(())
    }

    fn logical_to_physical_idx(&self, logical_idx: usize) -> usize {
        if self.len < self.capacity {
            logical_idx
        } else {
            (self.write_idx + logical_idx) % self.capacity
        }
    }

    pub fn sample_random(&self, batch_size: usize, rng: &mut impl Rng) -> ReplayBatch {
        let batch = batch_size.min(self.len);
        let mut obs = Vec::with_capacity(batch);
        let mut next_obs = Vec::with_capacity(batch);
        let mut actions = Vec::with_capacity(batch);
        let mut rewards = Vec::with_capacity(batch);
        let mut dones = Vec::with_capacity(batch);
        let mut action_masks = Vec::with_capacity(batch * self.action_dim);
        let mut next_action_masks = Vec::with_capacity(batch * self.action_dim);
        let mut root_action_weights = Vec::with_capacity(batch * self.action_dim);

        let sampled = rand::seq::index::sample(rng, self.len, batch);
        for logical_idx in sampled.iter() {
            let idx = self.logical_to_physical_idx(logical_idx);
            let m0 = idx * self.action_dim;
            obs.push(self.obs[idx].clone());
            next_obs.push(self.next_obs[idx].clone());
            actions.push(self.actions[idx]);
            rewards.push(self.rewards[idx]);
            dones.push(self.dones[idx]);
            action_masks.extend_from_slice(&self.action_masks[m0..m0 + self.action_dim]);
            next_action_masks.extend_from_slice(&self.next_action_masks[m0..m0 + self.action_dim]);
            root_action_weights.extend_from_slice(&self.root_action_weights[m0..m0 + self.action_dim]);
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
            action_dim: self.action_dim,
        }
    }
}

pub fn flatten_obs_once(obs: &GenericObs) -> anyhow::Result<Vec<f32>> {
    let mut out = Vec::new();
    for array in obs {
        match array {
            rustpool::core::types::ArrayData::Float32(values) => out.extend_from_slice(values),
            rustpool::core::types::ArrayData::Int32(values) => {
                out.extend(values.iter().map(|v| *v as f32));
            }
            rustpool::core::types::ArrayData::Bool(values) => {
                out.extend(values.iter().map(|v| if *v { 1.0 } else { 0.0 }));
            }
        }
    }
    if out.is_empty() {
        anyhow::bail!("observation flatten produced empty vector");
    }
    Ok(out)
}
