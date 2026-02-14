use anyhow::{bail, Result};
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

#[derive(Clone, Debug)]
pub struct BinPackObsView {
    pub items: Vec<f32>,
    pub ems: Vec<f32>,
    pub items_valid: Vec<bool>,
    pub ems_valid: Vec<bool>,
}

pub fn parse_binpack_obs(obs: &GenericObs, max_items: usize, max_ems: usize) -> Result<BinPackObsView> {
    if obs.len() < 5 {
        bail!("binpack obs must have at least 5 entries, got {}", obs.len());
    }

    let items = match &obs[0] {
        ArrayData::Float32(v) => v.clone(),
        _ => bail!("binpack items must be Float32"),
    };
    let ems = match &obs[1] {
        ArrayData::Float32(v) => v.clone(),
        _ => bail!("binpack ems must be Float32"),
    };
    let items_valid = match &obs[2] {
        ArrayData::Bool(v) => v.clone(),
        _ => bail!("binpack items_mask must be Bool"),
    };
    let ems_valid = match &obs[4] {
        ArrayData::Bool(v) => v.clone(),
        _ => bail!("binpack ems_mask must be Bool"),
    };

    if items.len() != max_items * 3 {
        bail!(
            "binpack items length mismatch: expected {}, got {}",
            max_items * 3,
            items.len()
        );
    }
    if ems.len() != max_ems * 6 {
        bail!(
            "binpack ems length mismatch: expected {}, got {}",
            max_ems * 6,
            ems.len()
        );
    }
    if items_valid.len() != max_items {
        bail!(
            "binpack items_mask length mismatch: expected {}, got {}",
            max_items,
            items_valid.len()
        );
    }
    if ems_valid.len() != max_ems {
        bail!(
            "binpack ems_mask length mismatch: expected {}, got {}",
            max_ems,
            ems_valid.len()
        );
    }

    Ok(BinPackObsView {
        items,
        ems,
        items_valid,
        ems_valid,
    })
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
    max_items: usize,
    max_ems: usize,
    is_binpack: bool,

    obs: Vec<f32>,
    items: Vec<f32>,
    ems: Vec<f32>,
    items_valid: Vec<u8>,
    ems_valid: Vec<u8>,
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
            max_items: 0,
            max_ems: 0,
            is_binpack: false,
            obs: vec![0.0; num_samples * obs_dim],
            items: vec![],
            ems: vec![],
            items_valid: vec![],
            ems_valid: vec![],
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

    pub fn new_binpack(t: usize, n: usize, max_items: usize, max_ems: usize, action_dim: usize) -> Self {
        let num_samples = t * n;
        Self {
            t,
            n,
            obs_dim: 0,
            _action_dim: action_dim,
            max_items,
            max_ems,
            is_binpack: true,
            obs: vec![],
            items: vec![0.0; num_samples * max_items * 3],
            ems: vec![0.0; num_samples * max_ems * 6],
            items_valid: vec![0; num_samples * max_items],
            ems_valid: vec![0; num_samples * max_ems],
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

    pub fn store_step_binpack(
        &mut self,
        t: usize,
        env: usize,
        obs: &GenericObs,
        mask: &[bool],
        action: i32,
        logp: f32,
        value: f32,
        reward: f32,
        done: bool,
    ) -> Result<()> {
        if !self.is_binpack {
            bail!("store_step_binpack called on non-binpack rollout");
        }
        let i = self.idx(t, env);
        let parsed = parse_binpack_obs(obs, self.max_items, self.max_ems)?;

        let items_base = i * self.max_items * 3;
        self.items[items_base..items_base + self.max_items * 3].copy_from_slice(&parsed.items);

        let ems_base = i * self.max_ems * 6;
        self.ems[ems_base..ems_base + self.max_ems * 6].copy_from_slice(&parsed.ems);

        let item_mask_base = i * self.max_items;
        for (offset, valid) in parsed.items_valid.iter().enumerate() {
            self.items_valid[item_mask_base + offset] = if *valid { 1 } else { 0 };
        }

        let ems_mask_base = i * self.max_ems;
        for (offset, valid) in parsed.ems_valid.iter().enumerate() {
            self.ems_valid[ems_mask_base + offset] = if *valid { 1 } else { 0 };
        }

        self.masks.set_mask(i, mask);
        self.actions[i] = action;
        self.old_logp[i] = logp;
        self.values[i] = value;
        self.rewards[i] = reward;
        self.dones[i] = if done { 1 } else { 0 };
        Ok(())
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

    pub fn minibatch_binpack(
        &self,
        indices: &[usize],
    ) -> Result<(
        Vec<f32>,
        Vec<f32>,
        Vec<bool>,
        Vec<bool>,
        Vec<i32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    )> {
        if !self.is_binpack {
            bail!("minibatch_binpack called on non-binpack rollout");
        }

        let bsz = indices.len();
        let mut items_mb = vec![0.0f32; bsz * self.max_items * 3];
        let mut ems_mb = vec![0.0f32; bsz * self.max_ems * 6];
        let mut items_mask_mb = vec![false; bsz * self.max_items];
        let mut ems_mask_mb = vec![false; bsz * self.max_ems];
        let mut act_mb = vec![0i32; bsz];
        let mut mask_mb = vec![0.0f32; bsz * self._action_dim];
        let mut old_lp_mb = vec![0.0f32; bsz];
        let mut old_v_mb = vec![0.0f32; bsz];
        let mut adv_mb = vec![0.0f32; bsz];
        let mut tgt_mb = vec![0.0f32; bsz];

        let unpacked_mask_mb = self.masks.unpack_to_f32(indices);
        mask_mb.copy_from_slice(&unpacked_mask_mb);

        for (row, &idx) in indices.iter().enumerate() {
            let src_items = idx * self.max_items * 3;
            let dst_items = row * self.max_items * 3;
            items_mb[dst_items..dst_items + self.max_items * 3]
                .copy_from_slice(&self.items[src_items..src_items + self.max_items * 3]);

            let src_ems = idx * self.max_ems * 6;
            let dst_ems = row * self.max_ems * 6;
            ems_mb[dst_ems..dst_ems + self.max_ems * 6]
                .copy_from_slice(&self.ems[src_ems..src_ems + self.max_ems * 6]);

            let src_item_mask = idx * self.max_items;
            let dst_item_mask = row * self.max_items;
            for j in 0..self.max_items {
                items_mask_mb[dst_item_mask + j] = self.items_valid[src_item_mask + j] != 0;
            }

            let src_ems_mask = idx * self.max_ems;
            let dst_ems_mask = row * self.max_ems;
            for j in 0..self.max_ems {
                ems_mask_mb[dst_ems_mask + j] = self.ems_valid[src_ems_mask + j] != 0;
            }

            act_mb[row] = self.actions[idx];
            old_lp_mb[row] = self.old_logp[idx];
            old_v_mb[row] = self.values[idx];
            adv_mb[row] = self.advantages[idx];
            tgt_mb[row] = self.targets[idx];
        }

        Ok((
            items_mb,
            ems_mb,
            items_mask_mb,
            ems_mask_mb,
            act_mb,
            mask_mb,
            old_lp_mb,
            old_v_mb,
            adv_mb,
            tgt_mb,
        ))
    }
}