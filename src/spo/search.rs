use anyhow::{bail, Result};
use burn::tensor::backend::Backend;
use rand::distributions::{Distribution as RandDistribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Gamma;
use rayon::prelude::*;
use rustpool::core::types::GenericObs;

use crate::config::Args;
use crate::env::{AsyncEnvPool, StepOut};
use crate::env_model::{build_actor_input_batch, build_critic_input_batch, EnvModelKind};
use crate::models::Agent;
use crate::ppo::loss::sample_actions_categorical;

#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    pub num_particles: usize,
    pub search_depth: usize,
    pub search_gamma: f32,
    pub search_gae_lambda: f32,
    pub resampling_period: usize,
    pub resampling_ess_threshold: f32,
    pub adaptive_temperature: bool,
    pub fixed_temperature: f32,
    pub root_exploration_dirichlet_alpha: f32,
    pub root_exploration_dirichlet_fraction: f32,
}

pub struct SearchOut {
    pub root_actions: Vec<i32>,
    pub root_action_weights: Vec<f32>,
    pub sampled_actions: Vec<i32>,
    pub sampled_advantages: Vec<f32>,
    pub leaf_steps: Vec<StepOut>,
    pub leaf_state_ids: Vec<i32>,
}

struct EnvTransitionChunk {
    state_ids: Vec<i32>,
    obs: Vec<GenericObs>,
    masks: Vec<Vec<bool>>,
    td_weights: Vec<f32>,
    gae: Vec<f32>,
    terminal: Vec<bool>,
    weights: Vec<f32>,
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max_v = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| if a > b { a } else { b });
    let exps = values.iter().map(|v| (v - max_v).exp()).collect::<Vec<_>>();
    let denom = exps.iter().sum::<f32>().max(1.0e-12);
    exps.into_iter().map(|x| x / denom).collect()
}

fn normalize_weights(weights: &mut [f32]) {
    let sum_w = weights.iter().sum::<f32>();
    if sum_w <= 0.0 || !sum_w.is_finite() {
        let uniform = 1.0 / (weights.len() as f32).max(1.0);
        for w in weights.iter_mut() {
            *w = uniform;
        }
        return;
    }
    for w in weights.iter_mut() {
        *w /= sum_w;
    }
}

fn effective_sample_size(weights: &[f32]) -> f32 {
    let denom = weights.iter().map(|w| w * w).sum::<f32>().max(1.0e-12);
    1.0 / denom
}

fn resample_indices<R: Rng + ?Sized>(
    weights: &[f32],
    num_samples: usize,
    rng: &mut R,
) -> Vec<usize> {
    let dist = WeightedIndex::new(weights)
        .ok()
        .or_else(|| WeightedIndex::new(vec![1.0f32; weights.len()]).ok());
    match dist {
        Some(d) => (0..num_samples).map(|_| d.sample(rng)).collect(),
        None => (0..num_samples).collect(),
    }
}

fn sample_dirichlet<R: Rng + ?Sized>(alpha: f32, dim: usize, rng: &mut R) -> Vec<f32> {
    let safe_alpha = alpha.max(1.0e-3) as f64;
    let gamma = Gamma::new(safe_alpha, 1.0).ok();
    let mut draws = vec![0.0f32; dim];
    let mut sum_draw = 0.0f32;
    for d in draws.iter_mut() {
        let v = gamma.as_ref().map(|g| g.sample(rng) as f32).unwrap_or(1.0);
        *d = v;
        sum_draw += v;
    }
    if sum_draw <= 0.0 || !sum_draw.is_finite() {
        let uniform = 1.0 / dim.max(1) as f32;
        return vec![uniform; dim];
    }
    for d in draws.iter_mut() {
        *d /= sum_draw;
    }
    draws
}

pub fn flatten_particle_actions(actions: &[i32], num_particles: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(actions.len() * num_particles);
    for &a in actions {
        for _ in 0..num_particles {
            out.push(a);
        }
    }
    out
}

pub fn expand_state_ids(state_ids: &[i32], num_particles: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(state_ids.len() * num_particles);
    for &sid in state_ids {
        for _ in 0..num_particles {
            out.push(sid);
        }
    }
    out
}

pub fn one_depth_simulation(
    env: &AsyncEnvPool,
    root_state_ids: &[i32],
    root_actions: &[i32],
    num_particles: usize,
) -> Result<Vec<StepOut>> {
    if root_state_ids.len() != root_actions.len() {
        bail!(
            "state/action mismatch at search root: {} vs {}",
            root_state_ids.len(),
            root_actions.len()
        );
    }
    let expanded_state_ids = expand_state_ids(root_state_ids, num_particles);
    let expanded_actions = flatten_particle_actions(root_actions, num_particles);

    let simulated = env.simulate_batch(&expanded_state_ids, &expanded_actions)?;
    Ok(simulated)
}

pub fn run_smc_search<B: Backend>(
    env: &AsyncEnvPool,
    agent: &Agent<B>,
    root_state_ids: &[i32],
    root_obs: &[&rustpool::core::types::GenericObs],
    root_action_masks: &[&[bool]],
    model_kind: EnvModelKind,
    args: &Args,
    cfg: SearchConfig,
    obs_dim: usize,
    action_dim: usize,
    device: &B::Device,
    rng: &mut impl Rng,
) -> Result<SearchOut> {
    if root_state_ids.len() != root_obs.len() || root_state_ids.len() != root_action_masks.len() {
        bail!(
            "state/obs/mask mismatch at search root: {} vs {} vs {}",
            root_state_ids.len(),
            root_obs.len(),
            root_action_masks.len()
        );
    }
    if cfg.num_particles == 0 {
        bail!("num_particles must be > 0");
    }
    if cfg.search_depth == 0 {
        bail!("search_depth must be > 0");
    }

    let batch = root_state_ids.len();
    let mut current_state_ids = expand_state_ids(root_state_ids, cfg.num_particles);
    let mut current_obs = Vec::with_capacity(batch * cfg.num_particles);
    let mut current_masks = Vec::with_capacity(batch * cfg.num_particles);
    for idx in 0..batch {
        for _ in 0..cfg.num_particles {
            current_obs.push((*root_obs[idx]).clone());
            current_masks.push(root_action_masks[idx].to_vec());
        }
    }

    debug_assert_eq!(current_state_ids.len(), batch * cfg.num_particles);
    debug_assert_eq!(current_obs.len(), batch * cfg.num_particles);

    let mut last_steps = Vec::new();
    let mut root_actions = vec![0i32; batch];
    let mut root_particle_actions = vec![0i32; batch * cfg.num_particles];
    let mut particle_td_weights = vec![0.0f32; batch * cfg.num_particles];
    let mut particle_gae = vec![0.0f32; batch * cfg.num_particles];
    let mut particle_terminal = vec![false; batch * cfg.num_particles];
    let mut particle_weights = vec![1.0f32 / (cfg.num_particles as f32); batch * cfg.num_particles];

    // Reused scratch buffer for action masks across search depths.
    // We keep this allocation outside the depth loop to reduce allocator pressure.
    let mut mask_flat = Vec::with_capacity(batch * cfg.num_particles * action_dim);

    for depth_idx in 0..cfg.search_depth {
        let n = current_obs.len();
        mask_flat.clear();
        mask_flat.reserve(n * action_dim);
        for mask in current_masks.iter() {
            if mask.len() != action_dim {
                bail!(
                    "search action mask mismatch: got {}, expected {}",
                    mask.len(),
                    action_dim
                );
            }
            for &m in mask {
                mask_flat.push(if m { 1.0 } else { 0.0 });
            }
        }

        let actor_input =
            build_actor_input_batch::<B>(&current_obs, model_kind, args, obs_dim, device)?;
        let current_values_t = agent
            .critic_values(build_critic_input_batch::<B>(
                &current_obs,
                model_kind,
                args,
                obs_dim,
                device,
            )?)
            .reshape([n]);
        let current_values = current_values_t
            .to_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("failed to convert current values: {e:?}"))?;

        let logits = agent.actor_logits(actor_input);
        // Burn's TensorData constructor takes ownership; cloning here keeps the reusable
        // scratch vector available for the next depth iteration.
        let mask_t = burn::tensor::Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(mask_flat.clone(), [n, action_dim]),
            device,
        );
        let actions_t = sample_actions_categorical(logits, mask_t, device);
        let actions_data = actions_t.to_data();
        let actions = match actions_data.clone().to_vec::<i32>() {
            Ok(v) => v,
            Err(_) => actions_data
                .to_vec::<i64>()
                .map_err(|e| anyhow::anyhow!("failed to convert sampled actions: {e:?}"))?
                .into_iter()
                .map(|v| v as i32)
                .collect(),
        };

        if depth_idx == 0 {
            root_particle_actions.copy_from_slice(&actions);
        }

        let steps = env.simulate_batch(&current_state_ids, &actions)?;
        let next_obs = steps.iter().map(|s| s.obs.clone()).collect::<Vec<_>>();
        let next_values_t = agent
            .critic_values(build_critic_input_batch::<B>(
                &next_obs, model_kind, args, obs_dim, device,
            )?)
            .reshape([n]);
        let next_values = next_values_t
            .to_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("failed to convert next values: {e:?}"))?;

        for i in 0..steps.len() {
            let td_error = steps[i].reward + next_values[i] - current_values[i];
            let terminal_mask = if particle_terminal[i] { 0.0 } else { 1.0 };
            particle_td_weights[i] += td_error * terminal_mask;

            let discount = if steps[i].done { 0.0 } else { 1.0 };
            let gae_decay =
                (cfg.search_gamma * cfg.search_gae_lambda * discount).powi(depth_idx as i32);
            particle_gae[i] += td_error * gae_decay;

            particle_terminal[i] = particle_terminal[i] || steps[i].done;
        }

        // Per-environment normalization is independent, so we parallelize over env chunks.
        particle_weights
            .par_chunks_mut(cfg.num_particles)
            .enumerate()
            .for_each(|(env_idx, weights_out)| {
                let start = env_idx * cfg.num_particles;
                let end = start + cfg.num_particles;
                let td_weights = &particle_td_weights[start..end];
                let temperature = if cfg.adaptive_temperature {
                    let mean = td_weights.iter().sum::<f32>() / cfg.num_particles as f32;
                    let var = td_weights
                        .iter()
                        .map(|r| {
                            let d = *r - mean;
                            d * d
                        })
                        .sum::<f32>()
                        / cfg.num_particles as f32;
                    var.sqrt().max(1.0e-3)
                } else {
                    cfg.fixed_temperature.max(1.0e-3)
                };

                let mut logits_buf = vec![0.0f32; cfg.num_particles];
                for (i, &w) in td_weights.iter().enumerate() {
                    logits_buf[i] = w / temperature;
                }
                let weights = softmax(&logits_buf);
                weights_out.copy_from_slice(&weights);
            });

        if depth_idx + 1 < cfg.search_depth {
            let periodic_trigger =
                cfg.resampling_period > 0 && (depth_idx + 1) % cfg.resampling_period == 0;
            // Generate deterministic per-env seeds serially, then run each env transition in
            // parallel with its own local RNG. This avoids mutable RNG contention in Rayon tasks.
            let env_seeds = (0..batch).map(|_| rng.gen::<u64>()).collect::<Vec<_>>();

            // Build per-env transition chunks in parallel, then flatten once.
            // This keeps ownership simple while still parallelizing resampling-heavy work.
            let chunk_results = (0..batch)
                .into_par_iter()
                .map(|env_idx| -> Result<EnvTransitionChunk> {
                    let start = env_idx * cfg.num_particles;
                    let end = start + cfg.num_particles;
                    let w = &particle_weights[start..end];
                    let ess = effective_sample_size(w);
                    let ess_trigger = ess < cfg.resampling_ess_threshold * cfg.num_particles as f32;
                    let do_resample = periodic_trigger || ess_trigger;

                    let mut chunk = EnvTransitionChunk {
                        state_ids: Vec::with_capacity(cfg.num_particles),
                        obs: Vec::with_capacity(cfg.num_particles),
                        masks: Vec::with_capacity(cfg.num_particles),
                        td_weights: Vec::with_capacity(cfg.num_particles),
                        gae: Vec::with_capacity(cfg.num_particles),
                        terminal: Vec::with_capacity(cfg.num_particles),
                        weights: Vec::with_capacity(cfg.num_particles),
                    };

                    if do_resample {
                        let mut local_rng = StdRng::seed_from_u64(env_seeds[env_idx]);
                        let sampled = resample_indices(w, cfg.num_particles, &mut local_rng);
                        for &j in sampled.iter() {
                            let idx = start + j;
                            chunk.state_ids.push(
                                steps[idx].state_ids.first().copied().ok_or_else(|| {
                                    anyhow::anyhow!("simulate_batch returned empty state_ids")
                                })?,
                            );
                            chunk.obs.push(steps[idx].obs.clone());
                            chunk.masks.push(steps[idx].action_mask.clone());
                            chunk.td_weights.push(0.0);
                            chunk.gae.push(particle_gae[idx]);
                            chunk.terminal.push(particle_terminal[idx]);
                        }
                        let uniform = 1.0 / cfg.num_particles as f32;
                        for _ in 0..cfg.num_particles {
                            chunk.weights.push(uniform);
                        }
                    } else {
                        for idx in start..end {
                            chunk.state_ids.push(
                                steps[idx].state_ids.first().copied().ok_or_else(|| {
                                    anyhow::anyhow!("simulate_batch returned empty state_ids")
                                })?,
                            );
                            chunk.obs.push(steps[idx].obs.clone());
                            chunk.masks.push(steps[idx].action_mask.clone());
                            chunk.td_weights.push(particle_td_weights[idx]);
                            chunk.gae.push(particle_gae[idx]);
                            chunk.terminal.push(particle_terminal[idx]);
                            chunk.weights.push(particle_weights[idx]);
                        }
                    }

                    Ok(chunk)
                })
                .collect::<Vec<_>>();

            // Flatten chunked outputs with exact-capacity vectors to avoid repeated reallocations.
            let mut next_state_ids = Vec::with_capacity(batch * cfg.num_particles);
            let mut next_obs = Vec::with_capacity(batch * cfg.num_particles);
            let mut next_masks = Vec::with_capacity(batch * cfg.num_particles);
            let mut next_td_weights = Vec::with_capacity(batch * cfg.num_particles);
            let mut next_gae = Vec::with_capacity(batch * cfg.num_particles);
            let mut next_terminal = Vec::with_capacity(batch * cfg.num_particles);
            let mut next_weights = Vec::with_capacity(batch * cfg.num_particles);

            for chunk_result in chunk_results {
                let chunk = chunk_result?;
                next_state_ids.extend(chunk.state_ids);
                next_obs.extend(chunk.obs);
                next_masks.extend(chunk.masks);
                next_td_weights.extend(chunk.td_weights);
                next_gae.extend(chunk.gae);
                next_terminal.extend(chunk.terminal);
                next_weights.extend(chunk.weights);
            }

            let mut released = current_state_ids.clone();
            released.sort_unstable();
            released.dedup();
            env.release_batch(&released);

            current_state_ids = next_state_ids;
            current_obs = next_obs;
            current_masks = next_masks;
            particle_td_weights = next_td_weights;
            particle_gae = next_gae;
            particle_terminal = next_terminal;
            particle_weights = next_weights;
        } else {
            let mut released = current_state_ids.clone();
            released.sort_unstable();
            released.dedup();
            env.release_batch(&released);

            current_state_ids = steps
                .iter()
                .map(|step| {
                    step.state_ids
                        .first()
                        .copied()
                        .ok_or_else(|| anyhow::anyhow!("simulate_batch returned empty state_ids"))
                })
                .collect::<Result<Vec<_>>>()?;
        }

        last_steps = steps;
    }

    let mut root_action_weights = vec![0.0f32; batch * action_dim];
    for env_idx in 0..batch {
        let start = env_idx * cfg.num_particles;
        let end = start + cfg.num_particles;
        let mut action_probs = vec![0.0f32; action_dim];
        for p in start..end {
            let action = root_particle_actions[p] as usize;
            if action < action_dim {
                action_probs[action] += particle_weights[p];
            }
        }

        if cfg.root_exploration_dirichlet_fraction > 0.0 {
            let frac = cfg.root_exploration_dirichlet_fraction.clamp(0.0, 1.0);
            let dir_noise = sample_dirichlet(cfg.root_exploration_dirichlet_alpha, action_dim, rng);
            for a in 0..action_dim {
                action_probs[a] = (1.0 - frac) * action_probs[a] + frac * dir_noise[a];
            }
        }

        normalize_weights(&mut action_probs);
        let chooser = WeightedIndex::new(&action_probs)
            .ok()
            .or_else(|| WeightedIndex::new(vec![1.0f32; action_dim]).ok())
            .ok_or_else(|| anyhow::anyhow!("failed to build root action sampler"))?;
        root_actions[env_idx] = chooser.sample(rng) as i32;
        let row0 = env_idx * action_dim;
        root_action_weights[row0..row0 + action_dim].copy_from_slice(&action_probs);
    }

    let leaf_state_ids = last_steps
        .iter()
        .filter_map(|s| s.state_ids.first().copied())
        .collect::<Vec<_>>();

    Ok(SearchOut {
        root_actions,
        root_action_weights,
        sampled_actions: root_particle_actions,
        sampled_advantages: particle_gae,
        leaf_steps: last_steps,
        leaf_state_ids,
    })
}
