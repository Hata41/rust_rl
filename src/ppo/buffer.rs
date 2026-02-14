use bitvec::prelude::*;
use rustpool::core::types::{ArrayData, GenericObs};

pub fn flatten_obs(obs: &GenericObs) -> Vec<f32> {
    let mut out = Vec::new();
    for a in obs {
        match a {
            ArrayData::Float32(v) => out.extend_from_slice(v),
            ArrayData::Int32(v) => out.extend(v.iter().map(|x| *x as f32)),
            ArrayData::Bool(v) => out.extend(v.iter().map(|b| if *b { 1.0 } else { 0.0 })),
        }
    }
    out
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AdvantageStats {
    pub pre_mean: f32,
    pub pre_std: f32,
    pub post_mean: f32,
    pub post_std: f32,
}

fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    (mean, var.sqrt())
}

#[derive(Clone)]
pub struct PackedMasks {
    action_dim: usize,
    words_per_mask: usize,
    bits: BitVec<u64, Lsb0>,
}

impl PackedMasks {
    pub fn new(action_dim: usize, num_samples: usize) -> Self {
        let words_per_mask = action_dim.div_ceil(64);
        let total_bits = num_samples * words_per_mask * 64;
        Self {
            action_dim,
            words_per_mask,
            bits: bitvec![u64, Lsb0; 0; total_bits],
        }
    }

    pub fn set_mask(&mut self, sample_idx: usize, mask: &[bool]) {
        let base_bit = sample_idx * self.words_per_mask * 64;
        for (i, &m) in mask.iter().enumerate().take(self.action_dim) {
            self.bits.set(base_bit + i, m);
        }
    }

    pub fn unpack_to_f32(&self, indices: &[usize]) -> Vec<f32> {
        let mut out = vec![0.0f32; indices.len() * self.action_dim];
        for (row, &sample_idx) in indices.iter().enumerate() {
            let base_bit = sample_idx * self.words_per_mask * 64;
            let row_base = row * self.action_dim;
            for a in 0..self.action_dim {
                out[row_base + a] = if self.bits[base_bit + a] { 1.0 } else { 0.0 };
            }
        }
        out
    }
}

pub struct Rollout {
    t: usize,
    n: usize,
    obs_dim: usize,
    _action_dim: usize,

    obs: Vec<f32>,
    actions: Vec<i32>,
    old_logp: Vec<f32>,
    values: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<u8>,
    masks: PackedMasks,

    advantages: Vec<f32>,
    targets: Vec<f32>,
    adv_stats: AdvantageStats,
}

impl Rollout {
    pub fn new(t: usize, n: usize, obs_dim: usize, action_dim: usize) -> Self {
        let num_samples = t * n;
        Self {
            t,
            n,
            obs_dim,
            _action_dim: action_dim,
            obs: vec![0.0; num_samples * obs_dim],
            actions: vec![0; num_samples],
            old_logp: vec![0.0; num_samples],
            values: vec![0.0; num_samples],
            rewards: vec![0.0; num_samples],
            dones: vec![0; num_samples],
            masks: PackedMasks::new(action_dim, num_samples),
            advantages: vec![0.0; num_samples],
            targets: vec![0.0; num_samples],
            adv_stats: AdvantageStats::default(),
        }
    }

    #[inline]
    fn idx(&self, t: usize, env: usize) -> usize {
        t * self.n + env
    }

    pub fn store_step(
        &mut self,
        t: usize,
        env: usize,
        obs_flat: &[f32],
        mask: &[bool],
        action: i32,
        logp: f32,
        value: f32,
        reward: f32,
        done: bool,
    ) {
        let i = self.idx(t, env);

        let obs_base = i * self.obs_dim;
        self.obs[obs_base..obs_base + self.obs_dim].copy_from_slice(obs_flat);

        self.masks.set_mask(i, mask);

        self.actions[i] = action;
        self.old_logp[i] = logp;
        self.values[i] = value;
        self.rewards[i] = reward;
        self.dones[i] = if done { 1 } else { 0 };
    }

    pub fn compute_gae(
        &mut self,
        last_values: &[f32],
        gamma: f32,
        lam: f32,
        reward_scale: f32,
        standardize: bool,
    ) {
        for (env, &next_value) in last_values.iter().enumerate().take(self.n) {
            let mut gae = 0.0f32;
            let mut next_v = next_value;

            for t in (0..self.t).rev() {
                let i = self.idx(t, env);
                let done = self.dones[i] != 0;

                let r = self.rewards[i] * reward_scale;
                let v = self.values[i];

                let not_done = if done { 0.0 } else { 1.0 };
                let delta = r + gamma * not_done * next_v - v;
                gae = delta + gamma * lam * not_done * gae;

                self.advantages[i] = gae;
                self.targets[i] = gae + v;

                next_v = v;
            }
        }

        let (pre_mean, pre_std) = mean_std(&self.advantages);

        if standardize {
            let n = (self.t * self.n) as f32;
            let mean = self.advantages.iter().sum::<f32>() / n;
            let var = self
                .advantages
                .iter()
                .map(|a| {
                    let d = *a - mean;
                    d * d
                })
                .sum::<f32>()
                / n;
            let std = (var + 1.0e-5).sqrt();

            for a in &mut self.advantages {
                *a = (*a - mean) / std;
            }
        }

        let (post_mean, post_std) = mean_std(&self.advantages);
        self.adv_stats = AdvantageStats {
            pre_mean,
            pre_std,
            post_mean,
            post_std,
        };
    }

    pub fn advantage_stats(&self) -> AdvantageStats {
        self.adv_stats
    }

    pub fn minibatch(
        &self,
        indices: &[usize],
    ) -> (Vec<f32>, Vec<i32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let bsz = indices.len();

        let mut obs_mb = vec![0.0f32; bsz * self.obs_dim];
        for (row, &idx) in indices.iter().enumerate() {
            let src = idx * self.obs_dim;
            let dst = row * self.obs_dim;
            obs_mb[dst..dst + self.obs_dim].copy_from_slice(&self.obs[src..src + self.obs_dim]);
        }

        let mask_mb = self.masks.unpack_to_f32(indices);

        let mut act_mb = vec![0i32; bsz];
        let mut old_lp_mb = vec![0.0f32; bsz];
        let mut old_v_mb = vec![0.0f32; bsz];
        let mut adv_mb = vec![0.0f32; bsz];
        let mut tgt_mb = vec![0.0f32; bsz];

        for (row, &idx) in indices.iter().enumerate() {
            act_mb[row] = self.actions[idx];
            old_lp_mb[row] = self.old_logp[idx];
            old_v_mb[row] = self.values[idx];
            adv_mb[row] = self.advantages[idx];
            tgt_mb[row] = self.targets[idx];
        }

        (obs_mb, act_mb, mask_mb, old_lp_mb, old_v_mb, adv_mb, tgt_mb)
    }
}