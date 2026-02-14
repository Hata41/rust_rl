use anyhow::{bail, Result};
use burn::backend::cuda::Cuda;
use burn::collective::{finish_collective, PeerId, ReduceOperation};
use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Bool, ElementConversion, Tensor, TensorData};
use burn::tensor::Int;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;
use tracing::{debug, info, span, Level};
use rustpool::core::types::GenericObs;

use crate::config::{Args, DistInfo};
use crate::env::{make_env, AsyncEnvPool};
use crate::models::{Actor, Agent, Critic};
use crate::ppo::buffer::{flatten_obs, parse_binpack_obs, Rollout};
use crate::ppo::loss::{
    compute_ppo_losses, logprob_and_entropy, masked_logits, sample_actions_gumbel,
};

fn build_binpack_batch_tensors<B: AutodiffBackend>(
    obs_batch: &[GenericObs],
    max_items: usize,
    max_ems: usize,
    device: &B::Device,
) -> Result<(
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 2, Bool>,
    Tensor<B, 2, Bool>,
    Tensor<B, 2>,
    Tensor<B, 2>,
)> {
    let batch = obs_batch.len();
    let mut items = vec![0.0f32; batch * max_items * 3];
    let mut ems = vec![0.0f32; batch * max_ems * 6];
    let mut items_pad = vec![false; batch * max_items];
    let mut ems_pad = vec![false; batch * max_ems];
    let mut items_valid_f32 = vec![0.0f32; batch * max_items];
    let mut ems_valid_f32 = vec![0.0f32; batch * max_ems];

    obs_batch
        .par_iter()
        .zip(items.par_chunks_mut(max_items * 3))
        .zip(ems.par_chunks_mut(max_ems * 6))
        .zip(items_pad.par_chunks_mut(max_items))
        .zip(ems_pad.par_chunks_mut(max_ems))
        .zip(items_valid_f32.par_chunks_mut(max_items))
        .zip(ems_valid_f32.par_chunks_mut(max_ems))
        .try_for_each(
            |((((((obs, items_row), ems_row), items_pad_row), ems_pad_row), items_valid_row), ems_valid_row)| -> Result<()> {
                let parsed = parse_binpack_obs(obs, max_items, max_ems)?;

                items_row.copy_from_slice(&parsed.items);
                ems_row.copy_from_slice(&parsed.ems);

                for i in 0..max_items {
                    let valid = parsed.items_valid[i];
                    items_pad_row[i] = !valid;
                    items_valid_row[i] = if valid { 1.0 } else { 0.0 };
                }

                for i in 0..max_ems {
                    let valid = parsed.ems_valid[i];
                    ems_pad_row[i] = !valid;
                    ems_valid_row[i] = if valid { 1.0 } else { 0.0 };
                }

                Ok(())
            },
        )?;

    let items_t = Tensor::<B, 3>::from_data(TensorData::new(items, [batch, max_items, 3]), device);
    let ems_t = Tensor::<B, 3>::from_data(TensorData::new(ems, [batch, max_ems, 6]), device);
    let items_pad_t = Tensor::<B, 2, Bool>::from_data(TensorData::new(items_pad, [batch, max_items]), device);
    let ems_pad_t = Tensor::<B, 2, Bool>::from_data(TensorData::new(ems_pad, [batch, max_ems]), device);
    let items_valid_t = Tensor::<B, 2>::from_data(TensorData::new(items_valid_f32, [batch, max_items]), device);
    let ems_valid_t = Tensor::<B, 2>::from_data(TensorData::new(ems_valid_f32, [batch, max_ems]), device);

    Ok((items_t, ems_t, items_pad_t, ems_pad_t, items_valid_t, ems_valid_t))
}

fn linear_decay_alpha(update: usize, num_updates: usize) -> f64 {
    if num_updates == 0 {
        return 1.0;
    }
    let progress = (update as f64) / (num_updates as f64);
    1.0 - progress
}

struct GradSqAccumulator<'a> {
    grads: &'a GradientsParams,
    sum_sq: f64,
}

impl<'a> GradSqAccumulator<'a> {
    fn new(grads: &'a GradientsParams) -> Self {
        Self { grads, sum_sq: 0.0 }
    }
}

impl<Bk: AutodiffBackend> ModuleVisitor<Bk> for GradSqAccumulator<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<Bk, D>>) {
        if let Some(grad) = self.grads.get::<Bk::InnerBackend, D>(param.id) {
            if let Ok(values) = grad.to_data().to_vec::<Bk::FloatElem>() {
                self.sum_sq += values
                    .iter()
                    .map(|v| {
                        let x: f64 = (*v).elem();
                        x * x
                    })
                    .sum::<f64>();
            }
        }
    }
}

struct GradScaler<'a> {
    grads: &'a mut GradientsParams,
    scale: f64,
}

impl<Bk: AutodiffBackend> ModuleVisitor<Bk> for GradScaler<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<Bk, D>>) {
        if let Some(grad) = self.grads.remove::<Bk::InnerBackend, D>(param.id) {
            self.grads
                .register::<Bk::InnerBackend, D>(param.id, grad.mul_scalar(self.scale));
        }
    }
}

fn clip_global_grad_norm<Bk: AutodiffBackend>(
    actor: &Actor<Bk>,
    critic: &Critic<Bk>,
    grads_actor: &mut GradientsParams,
    grads_critic: &mut GradientsParams,
    max_grad_norm: f32,
) -> f32 {

    let mut actor_acc = GradSqAccumulator::new(grads_actor);
    <Actor<Bk> as Module<Bk>>::visit(actor, &mut actor_acc);

    let mut critic_acc = GradSqAccumulator::new(grads_critic);
    <Critic<Bk> as Module<Bk>>::visit(critic, &mut critic_acc);

    let total_norm = (actor_acc.sum_sq + critic_acc.sum_sq).sqrt() as f32;
    if max_grad_norm > 0.0 && total_norm > max_grad_norm {
        let scale = (max_grad_norm / (total_norm + 1.0e-6)) as f64;

        let mut actor_scaler = GradScaler {
            grads: grads_actor,
            scale,
        };
        <Actor<Bk> as Module<Bk>>::visit(actor, &mut actor_scaler);

        let mut critic_scaler = GradScaler {
            grads: grads_critic,
            scale,
        };
        <Critic<Bk> as Module<Bk>>::visit(critic, &mut critic_scaler);
    }

    total_norm
}

#[derive(Debug, Clone, Copy)]
struct EvalStats {
    mean_return: f32,
    max_return: f32,
    min_return: f32,
    mean_ep_len: f32,
    max_ep_len: usize,
    min_ep_len: usize,
    episodes: usize,
}

fn run_deterministic_eval<B: AutodiffBackend>(
    agent: &Agent<B>,
    eval_pool: &AsyncEnvPool,
    eval_seed: u64,
    num_eval_envs: usize,
    num_eval_episodes: usize,
    is_binpack: bool,
    obs_dim: usize,
    action_dim: usize,
    max_items: usize,
    max_ems: usize,
    device: &B::Device,
) -> Result<EvalStats> {
    if num_eval_envs == 0 || num_eval_episodes == 0 {
        return Ok(EvalStats {
            mean_return: 0.0,
            max_return: 0.0,
            min_return: 0.0,
            mean_ep_len: 0.0,
            max_ep_len: 0,
            min_ep_len: 0,
            episodes: 0,
        });
    }

    let reset_out = eval_pool.reset_all(Some(eval_seed))?;
    let mut cur_obs = reset_out.iter().map(|s| s.obs.clone()).collect::<Vec<_>>();
    let mut cur_mask = reset_out
        .iter()
        .map(|s| s.action_mask.clone())
        .collect::<Vec<_>>();

    let mut ep_return = vec![0.0f32; num_eval_envs];
    let mut ep_len = vec![0usize; num_eval_envs];
    let mut completed_returns = Vec::<f32>::with_capacity(num_eval_episodes);
    let mut completed_lengths = Vec::<usize>::with_capacity(num_eval_episodes);

    while completed_returns.len() < num_eval_episodes {
        let mut mask_f = vec![0.0f32; num_eval_envs * action_dim];
        for e in 0..num_eval_envs {
            let row = e * action_dim;
            for a in 0..action_dim {
                mask_f[row + a] = if cur_mask[e][a] { 1.0 } else { 0.0 };
            }
        }
        let mask_t = Tensor::<B, 2>::from_data(
            TensorData::new(mask_f, [num_eval_envs, action_dim]),
            device,
        );

        let logits = if is_binpack {
            let (items_t, ems_t, items_pad_t, ems_pad_t, _items_valid_t, _ems_valid_t) =
                build_binpack_batch_tensors::<B>(&cur_obs, max_items, max_ems, device)?;

            agent
                .actor
                .forward_binpack(ems_t, items_t, ems_pad_t, items_pad_t)
                .detach()
        } else {
            let mut obs_flat_all = vec![0.0f32; num_eval_envs * obs_dim];
            for e in 0..num_eval_envs {
                let flat = flatten_obs(&cur_obs[e]);
                let base = e * obs_dim;
                obs_flat_all[base..base + obs_dim].copy_from_slice(&flat);
            }
            let obs_t = Tensor::<B, 2>::from_data(
                TensorData::new(obs_flat_all, [num_eval_envs, obs_dim]),
                device,
            );
            agent.actor.forward(obs_t).detach()
        };

        let actions_t = masked_logits(logits, mask_t).argmax(1).squeeze::<1>();

        let actions_data = actions_t.to_data();
        let actions_vec: Vec<i32> = match actions_data.clone().to_vec::<i32>() {
            Ok(v) => v,
            Err(_) => actions_data
                .to_vec::<i64>()
                .map_err(|e| anyhow::anyhow!("failed to convert greedy actions: {e}"))?
                .into_iter()
                .map(|v| v as i32)
                .collect(),
        };

        let step_out = eval_pool.step_all(&actions_vec)?;

        for e in 0..num_eval_envs {
            ep_return[e] += step_out[e].reward;
            ep_len[e] += 1;

            if step_out[e].done {
                if completed_returns.len() < num_eval_episodes {
                    completed_returns.push(ep_return[e]);
                    completed_lengths.push(ep_len[e]);
                }
                ep_return[e] = 0.0;
                ep_len[e] = 0;
            }
        }

        for e in 0..num_eval_envs {
            cur_obs[e] = step_out[e].obs.clone();
            cur_mask[e] = step_out[e].action_mask.clone();
        }
    }

    let episodes = completed_returns.len();
    let mean_return = completed_returns.iter().sum::<f32>() / (episodes as f32);
    let max_return = completed_returns
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_return = completed_returns
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let mean_ep_len = completed_lengths
        .iter()
        .map(|v| *v as f32)
        .sum::<f32>()
        / (episodes as f32);
    let max_ep_len = *completed_lengths.iter().max().unwrap_or(&0);
    let min_ep_len = *completed_lengths.iter().min().unwrap_or(&0);

    Ok(EvalStats {
        mean_return,
        max_return,
        min_return,
        mean_ep_len,
        max_ep_len,
        min_ep_len,
        episodes,
    })
}

pub fn run<B: AutodiffBackend>(args: Args, dist: DistInfo, device: B::Device) -> Result<()> {
    let is_lead = dist.rank == 0;

    if args.num_envs == 0 {
        bail!("num_envs must be > 0");
    }

    if args.num_envs % dist.world_size != 0 {
        bail!(
            "num_envs ({}) must be divisible by WORLD_SIZE ({})",
            args.num_envs,
            dist.world_size
        );
    }
    let local_num_envs = args.num_envs / dist.world_size;
    if local_num_envs == 0 {
        bail!("local_num_envs must be > 0");
    }

    let local_batch = args.rollout_length * local_num_envs;
    if local_batch % args.num_minibatches != 0 {
        bail!(
            "rollout_length*local_num_envs = {local_batch} must be divisible by num_minibatches = {}",
            args.num_minibatches
        );
    }

    let peer_id = PeerId::from(dist.rank);

    B::seed(&device, args.seed ^ ((dist.rank as u64) << 32));

    let rank_env_seed_offset = (dist.rank as u64) * (local_num_envs as u64);

    let env_pool = AsyncEnvPool::new(local_num_envs, args.seed + rank_env_seed_offset, {
        let args = args.clone();
        move |seed| make_env(&args.task_id, &args, seed).unwrap()
    })?;

    let reset_out = env_pool.reset_all(Some(args.seed + 10_000 + rank_env_seed_offset))?;
    let mut cur_obs = reset_out.iter().map(|s| s.obs.clone()).collect::<Vec<_>>();
    let mut cur_mask = reset_out
        .iter()
        .map(|s| s.action_mask.clone())
        .collect::<Vec<_>>();

    let is_binpack = args.task_id == "BinPack-v0";

    let obs_dim = if is_binpack {
        args.max_items * 3 + args.max_ems * 6
    } else {
        flatten_obs(&cur_obs[0]).len()
    };
    let action_dim = cur_mask[0].len();

    if is_lead {
        info!(
            category = "MISC",
            task = %args.task_id,
            world_size = dist.world_size,
            global_num_envs = args.num_envs,
            local_num_envs,
            rollout_length = args.rollout_length,
            local_batch,
            obs_dim,
            action_dim,
            "startup"
        );
    }

    let mut agent: Agent<B> = Agent::new(obs_dim, args.hidden_dim, action_dim, &device);

    let mut actor_optim = AdamConfig::new().init::<B, Actor<B>>();
    let mut critic_optim = AdamConfig::new().init::<B, Critic<B>>();

    let mut ep_return = vec![0.0f32; local_num_envs];
    let mut ep_len = vec![0usize; local_num_envs];
    let mut recent_returns: Vec<f32> = Vec::with_capacity(256);
    let mut recent_lengths: Vec<usize> = Vec::with_capacity(256);

    let mut rng = StdRng::seed_from_u64(args.seed ^ 0xA11CE);

    let eval_num_envs = args.num_eval_envs;
    if is_lead && eval_num_envs == 0 {
        bail!("num_eval_envs must be > 0 on lead rank");
    }

    let eval_pool = if is_lead {
        Some(AsyncEnvPool::new(eval_num_envs, args.seed + 999, {
            let args = args.clone();
            move |seed| make_env(&args.task_id, &args, seed).unwrap()
        })?)
    } else {
        None
    };

    for update in 0..args.num_updates {
        let update_span = span!(
            Level::INFO,
            "update_span",
            update,
            num_updates = args.num_updates,
            rank = dist.rank,
            world_size = dist.world_size
        );
        let _update_guard = update_span.enter();

        let alpha = if args.decay_learning_rates {
            linear_decay_alpha(update, args.num_updates)
        } else {
            1.0
        };
        let current_actor_lr = args.actor_lr * alpha;
        let current_critic_lr = args.critic_lr * alpha;

        let mut roll = if is_binpack {
            Rollout::new_binpack(
                args.rollout_length,
                local_num_envs,
                args.max_items,
                args.max_ems,
                action_dim,
            )
        } else {
            Rollout::new(args.rollout_length, local_num_envs, obs_dim, action_dim)
        };

        let rollout_started = Instant::now();
        {
            let rollout_span = span!(
                Level::DEBUG,
                "rollout_span",
                update,
                rollout_length = args.rollout_length,
                local_num_envs
            );
            let _rollout_guard = rollout_span.enter();

            for t in 0..args.rollout_length {
                let mut mask_f = vec![0.0f32; local_num_envs * action_dim];
                for e in 0..local_num_envs {
                    let row = e * action_dim;
                    for a in 0..action_dim {
                        mask_f[row + a] = if cur_mask[e][a] { 1.0 } else { 0.0 };
                    }
                }
                let mask_t = Tensor::<B, 2>::from_data(
                    TensorData::new(mask_f, [local_num_envs, action_dim]),
                    &device,
                );

                let (logits, values2) = if is_binpack {
                    let (items_t, ems_t, items_pad_t, ems_pad_t, items_valid_t, ems_valid_t) =
                        build_binpack_batch_tensors::<B>(&cur_obs, args.max_items, args.max_ems, &device)?;

                    let logits = agent
                        .actor
                        .forward_binpack(ems_t.clone(), items_t.clone(), ems_pad_t.clone(), items_pad_t.clone())
                        .detach();
                    let values2 = agent
                        .critic
                        .forward_binpack(ems_t, items_t, ems_pad_t, items_pad_t, ems_valid_t, items_valid_t)
                        .detach();
                    (logits, values2)
                } else {
                    let mut obs_flat_all = vec![0.0f32; local_num_envs * obs_dim];
                    for e in 0..local_num_envs {
                        let flat = flatten_obs(&cur_obs[e]);
                        let base = e * obs_dim;
                        obs_flat_all[base..base + obs_dim].copy_from_slice(&flat);
                    }
                    let obs_t = Tensor::<B, 2>::from_data(
                        TensorData::new(obs_flat_all, [local_num_envs, obs_dim]),
                        &device,
                    );

                    let logits = agent.actor.forward(obs_t.clone()).detach();
                    let values2 = agent.critic.forward(obs_t).detach();
                    (logits, values2)
                };
                let values = values2.reshape([local_num_envs]);

                let actions_t = sample_actions_gumbel::<B>(logits.clone(), mask_t.clone(), &device);

                let (logp_t, _ent_t) = logprob_and_entropy::<B>(logits, mask_t, actions_t.clone());

                let actions_data = actions_t.to_data();
                let actions_vec: Vec<i32> = match actions_data.clone().to_vec::<i32>() {
                    Ok(v) => v,
                    Err(_) => actions_data
                        .to_vec::<i64>()
                        .map_err(|e| anyhow::anyhow!("failed to convert sampled actions: {e}"))?
                        .into_iter()
                        .map(|v| v as i32)
                        .collect(),
                };
                let logp_vec: Vec<f32> = logp_t.to_data().to_vec().unwrap();
                let values_vec: Vec<f32> = values.to_data().to_vec().unwrap();

                let step_out = env_pool.step_all(&actions_vec)?;

                for e in 0..local_num_envs {
                    if is_binpack {
                        roll.store_step_binpack(
                            t,
                            e,
                            &cur_obs[e],
                            &cur_mask[e],
                            actions_vec[e],
                            logp_vec[e],
                            values_vec[e],
                            step_out[e].reward,
                            step_out[e].done,
                        )?;
                    } else {
                        let obs_flat = flatten_obs(&cur_obs[e]);
                        roll.store_step(
                            t,
                            e,
                            &obs_flat,
                            &cur_mask[e],
                            actions_vec[e],
                            logp_vec[e],
                            values_vec[e],
                            step_out[e].reward,
                            step_out[e].done,
                        );
                    }

                    ep_return[e] += step_out[e].reward;
                    ep_len[e] += 1;
                    if step_out[e].done {
                        recent_returns.push(ep_return[e]);
                        recent_lengths.push(ep_len[e]);
                        ep_return[e] = 0.0;
                        ep_len[e] = 0;
                    }
                }

                for e in 0..local_num_envs {
                    cur_obs[e] = step_out[e].obs.clone();
                    cur_mask[e] = step_out[e].action_mask.clone();
                }
            }
        }
        let rollout_elapsed = rollout_started.elapsed();

        let last_v2 = if is_binpack {
            let (items_t, ems_t, items_pad_t, ems_pad_t, items_valid_t, ems_valid_t) =
                build_binpack_batch_tensors::<B>(&cur_obs, args.max_items, args.max_ems, &device)?;
            agent
                .critic
                .forward_binpack(ems_t, items_t, ems_pad_t, items_pad_t, ems_valid_t, items_valid_t)
                .detach()
        } else {
            let mut obs_flat_all = vec![0.0f32; local_num_envs * obs_dim];
            for e in 0..local_num_envs {
                let flat = flatten_obs(&cur_obs[e]);
                let base = e * obs_dim;
                obs_flat_all[base..base + obs_dim].copy_from_slice(&flat);
            }
            let obs_last = Tensor::<B, 2>::from_data(
                TensorData::new(obs_flat_all, [local_num_envs, obs_dim]),
                &device,
            );
            agent.critic.forward(obs_last).detach()
        };
        let last_v = last_v2.reshape([local_num_envs]);
        let last_values: Vec<f32> = last_v.to_data().to_vec().unwrap();

        roll.compute_gae(
            &last_values,
            args.gamma,
            args.gae_lambda,
            args.reward_scale,
            args.standardize_advantages,
        );

        let mb_size = local_batch / args.num_minibatches;
        let mut all_indices: Vec<usize> = (0..local_batch).collect();

        let mut actor_loss_sum = 0.0f64;
        let mut critic_loss_sum = 0.0f64;
        let mut entropy_sum = 0.0f64;
        let mut grad_norm_sum = 0.0f64;

        let optimization_started = Instant::now();
        {
            let optimization_span = span!(
                Level::DEBUG,
                "optimization_span",
                update,
                epochs = args.epochs,
                num_minibatches = args.num_minibatches,
                mb_size
            );
            let _optimization_guard = optimization_span.enter();

            for epoch in 0..args.epochs {
                all_indices.shuffle(&mut rng);

                for mb in 0..args.num_minibatches {
                    let start = mb * mb_size;
                    let end = start + mb_size;
                    let mb_idx = &all_indices[start..end];

                    let (logits, act_t, mask_t, old_lp_t, old_v_t, adv_t, tgt_t, v2) = if is_binpack {
                        let (
                            items_mb,
                            ems_mb,
                            items_valid_mb,
                            ems_valid_mb,
                            act_mb,
                            mask_mb,
                            old_lp_mb,
                            old_v_mb,
                            adv_mb,
                            tgt_mb,
                        ) = roll.minibatch_binpack(mb_idx)?;

                        let items_t = Tensor::<B, 3>::from_data(
                            TensorData::new(items_mb, [mb_size, args.max_items, 3]),
                            &device,
                        );
                        let ems_t = Tensor::<B, 3>::from_data(
                            TensorData::new(ems_mb, [mb_size, args.max_ems, 6]),
                            &device,
                        );

                        let mut items_pad_mb = vec![false; mb_size * args.max_items];
                        let mut ems_pad_mb = vec![false; mb_size * args.max_ems];
                        let mut items_valid_f32_mb = vec![0.0f32; mb_size * args.max_items];
                        let mut ems_valid_f32_mb = vec![0.0f32; mb_size * args.max_ems];

                        for i in 0..(mb_size * args.max_items) {
                            let valid = items_valid_mb[i];
                            items_pad_mb[i] = !valid;
                            items_valid_f32_mb[i] = if valid { 1.0 } else { 0.0 };
                        }
                        for i in 0..(mb_size * args.max_ems) {
                            let valid = ems_valid_mb[i];
                            ems_pad_mb[i] = !valid;
                            ems_valid_f32_mb[i] = if valid { 1.0 } else { 0.0 };
                        }

                        let items_pad_t = Tensor::<B, 2, Bool>::from_data(
                            TensorData::new(items_pad_mb, [mb_size, args.max_items]),
                            &device,
                        );
                        let ems_pad_t = Tensor::<B, 2, Bool>::from_data(
                            TensorData::new(ems_pad_mb, [mb_size, args.max_ems]),
                            &device,
                        );
                        let items_valid_t = Tensor::<B, 2>::from_data(
                            TensorData::new(items_valid_f32_mb, [mb_size, args.max_items]),
                            &device,
                        );
                        let ems_valid_t = Tensor::<B, 2>::from_data(
                            TensorData::new(ems_valid_f32_mb, [mb_size, args.max_ems]),
                            &device,
                        );

                        let logits = agent
                            .actor
                            .forward_binpack(ems_t.clone(), items_t.clone(), ems_pad_t.clone(), items_pad_t.clone());
                        let v2 = agent
                            .critic
                            .forward_binpack(ems_t, items_t, ems_pad_t, items_pad_t, ems_valid_t, items_valid_t);

                        let act_t = Tensor::<B, 1, Int>::from_data(
                            TensorData::new(act_mb, [mb_size]),
                            &device,
                        );
                        let mask_t = Tensor::<B, 2>::from_data(
                            TensorData::new(mask_mb, [mb_size, action_dim]),
                            &device,
                        );
                        let old_lp_t = Tensor::<B, 1>::from_data(
                            TensorData::new(old_lp_mb, [mb_size]),
                            &device,
                        );
                        let old_v_t = Tensor::<B, 1>::from_data(
                            TensorData::new(old_v_mb, [mb_size]),
                            &device,
                        );
                        let adv_t = Tensor::<B, 1>::from_data(
                            TensorData::new(adv_mb, [mb_size]),
                            &device,
                        );
                        let tgt_t = Tensor::<B, 1>::from_data(
                            TensorData::new(tgt_mb, [mb_size]),
                            &device,
                        );

                        (logits, act_t, mask_t, old_lp_t, old_v_t, adv_t, tgt_t, v2)
                    } else {
                        let (obs_mb, act_mb, mask_mb, old_lp_mb, old_v_mb, adv_mb, tgt_mb) =
                            roll.minibatch(mb_idx);

                        let obs_t = Tensor::<B, 2>::from_data(
                            TensorData::new(obs_mb, [mb_size, obs_dim]),
                            &device,
                        );
                        let act_t = Tensor::<B, 1, Int>::from_data(
                            TensorData::new(act_mb, [mb_size]),
                            &device,
                        );
                        let mask_t = Tensor::<B, 2>::from_data(
                            TensorData::new(mask_mb, [mb_size, action_dim]),
                            &device,
                        );
                        let old_lp_t = Tensor::<B, 1>::from_data(
                            TensorData::new(old_lp_mb, [mb_size]),
                            &device,
                        );
                        let old_v_t = Tensor::<B, 1>::from_data(
                            TensorData::new(old_v_mb, [mb_size]),
                            &device,
                        );
                        let adv_t = Tensor::<B, 1>::from_data(
                            TensorData::new(adv_mb, [mb_size]),
                            &device,
                        );
                        let tgt_t = Tensor::<B, 1>::from_data(
                            TensorData::new(tgt_mb, [mb_size]),
                            &device,
                        );

                        let logits = agent.actor.forward(obs_t.clone());
                        let v2 = agent.critic.forward(obs_t);

                        (logits, act_t, mask_t, old_lp_t, old_v_t, adv_t, tgt_t, v2)
                    };

                    let v = v2.reshape([mb_size]);

                    let (new_lp, ent) = logprob_and_entropy::<B>(logits, mask_t, act_t);

                    let parts = compute_ppo_losses::<B>(
                        new_lp,
                        old_lp_t,
                        adv_t,
                        ent,
                        v,
                        old_v_t,
                        tgt_t,
                        args.clip_eps,
                        args.ent_coef,
                        args.vf_coef,
                    );

                    let mut grads = parts.total_loss.backward();

                    let mut grads_actor =
                        GradientsParams::from_module::<B, Actor<B>>(&mut grads, &agent.actor);
                    let mut grads_critic =
                        GradientsParams::from_module::<B, Critic<B>>(&mut grads, &agent.critic);

                    if dist.world_size > 1 {
                        grads_actor = grads_actor
                            .all_reduce::<Cuda<f32, i32>>(peer_id, ReduceOperation::Mean)
                            .map_err(|e| anyhow::anyhow!("failed actor gradient all-reduce: {e:?}"))?;
                        grads_critic = grads_critic
                            .all_reduce::<Cuda<f32, i32>>(peer_id, ReduceOperation::Mean)
                            .map_err(|e| anyhow::anyhow!("failed critic gradient all-reduce: {e:?}"))?;
                    }

                    let global_grad_norm = clip_global_grad_norm(
                        &agent.actor,
                        &agent.critic,
                        &mut grads_actor,
                        &mut grads_critic,
                        args.max_grad_norm,
                    );

                    agent.actor = actor_optim.step(current_actor_lr, agent.actor, grads_actor);
                    agent.critic = critic_optim.step(current_critic_lr, agent.critic, grads_critic);

                    let actor_loss = parts
                        .actor_loss
                        .to_data()
                        .to_vec::<f32>()
                        .unwrap_or_default()
                        .first()
                        .copied()
                        .unwrap_or(0.0);
                    let critic_loss = parts
                        .value_loss
                        .to_data()
                        .to_vec::<f32>()
                        .unwrap_or_default()
                        .first()
                        .copied()
                        .unwrap_or(0.0);
                    let entropy = parts
                        .entropy_mean
                        .to_data()
                        .to_vec::<f32>()
                        .unwrap_or_default()
                        .first()
                        .copied()
                        .unwrap_or(0.0);

                    actor_loss_sum += actor_loss as f64;
                    critic_loss_sum += critic_loss as f64;
                    entropy_sum += entropy as f64;
                    grad_norm_sum += global_grad_norm as f64;

                    if is_lead {
                        debug!(
                            category = "TRAINER",
                            update,
                            epoch,
                            minibatch = mb,
                            actor_loss,
                            critic_loss,
                            entropy,
                            global_grad_norm,
                            learning_rate = current_actor_lr,
                            "minibatch"
                        );
                    }
                }
            }
        }
        let optimization_elapsed = optimization_started.elapsed();

        if is_lead {
            let num_updates_in_cycle = (args.epochs * args.num_minibatches) as f64;
            let denom = num_updates_in_cycle.max(1.0);
            let mean_actor_loss = actor_loss_sum / denom;
            let mean_critic_loss = critic_loss_sum / denom;
            let mean_entropy = entropy_sum / denom;
            let mean_global_grad_norm = grad_norm_sum / denom;

            let mean_return = if recent_returns.is_empty() {
                0.0
            } else {
                let k = recent_returns.len().min(100);
                recent_returns[recent_returns.len() - k..].iter().sum::<f32>() / (k as f32)
            };
            let max_return = if recent_returns.is_empty() {
                0.0
            } else {
                let k = recent_returns.len().min(100);
                recent_returns[recent_returns.len() - k..]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
            };
            let min_return = if recent_returns.is_empty() {
                0.0
            } else {
                let k = recent_returns.len().min(100);
                recent_returns[recent_returns.len() - k..]
                    .iter()
                    .copied()
                    .fold(f32::INFINITY, f32::min)
            };
            let mean_ep_len = if recent_lengths.is_empty() {
                0.0
            } else {
                let k = recent_lengths.len().min(100);
                recent_lengths[recent_lengths.len() - k..]
                    .iter()
                    .map(|v| *v as f32)
                    .sum::<f32>()
                    / (k as f32)
            };
            let max_ep_len = if recent_lengths.is_empty() {
                0
            } else {
                let k = recent_lengths.len().min(100);
                *recent_lengths[recent_lengths.len() - k..]
                    .iter()
                    .max()
                    .unwrap_or(&0)
            };
            let min_ep_len = if recent_lengths.is_empty() {
                0
            } else {
                let k = recent_lengths.len().min(100);
                *recent_lengths[recent_lengths.len() - k..]
                    .iter()
                    .min()
                    .unwrap_or(&0)
            };

            let adv_stats = roll.advantage_stats();

            let rollout_steps = (args.rollout_length * args.num_envs) as f64;
            let total_update_secs =
                (rollout_elapsed + optimization_elapsed).as_secs_f64().max(1.0e-9);
            let steps_per_second = rollout_steps / total_update_secs;

            let timesteps = (update + 1) * local_batch * dist.world_size;
            info!(
                category = "TRAINER",
                policy_version = update + 1,
                actor_loss = mean_actor_loss,
                critic_loss = mean_critic_loss,
                entropy = mean_entropy,
                global_grad_norm = mean_global_grad_norm,
                "train"
            );

            info!(
                category = "ACTOR",
                phase = "Training/Actor",
                mean_return,
                max_return,
                min_return,
                episode_length_mean = mean_ep_len,
                episode_length_max = max_ep_len,
                episode_length_min = min_ep_len,
                "actor"
            );

            info!(
                category = "MISC",
                timesteps,
                steps_per_second,
                learning_rate = current_actor_lr,
                "misc"
            );

            debug!(
                category = "MISC",
                timesteps,
                rollout_duration_ms = rollout_elapsed.as_secs_f64() * 1_000.0,
                optimization_duration_ms = optimization_elapsed.as_secs_f64() * 1_000.0,
                advantage_pre_mean = adv_stats.pre_mean,
                advantage_pre_std = adv_stats.pre_std,
                advantage_post_mean = adv_stats.post_mean,
                advantage_post_std = adv_stats.post_std,
                "details"
            );
        }

        if is_lead
            && args.eval_interval > 0
            && (update % args.eval_interval == 0)
            && args.num_eval_episodes > 0
        {
            if let Some(eval_pool) = eval_pool.as_ref() {
                let eval_started = Instant::now();
                let eval_stats = run_deterministic_eval::<B>(
                    &agent,
                    eval_pool,
                    args.seed + 999,
                    eval_num_envs,
                    args.num_eval_episodes,
                    is_binpack,
                    obs_dim,
                    action_dim,
                    args.max_items,
                    args.max_ems,
                    &device,
                )?;

                info!(
                    category = "EVALUATOR",
                    phase = "Evaluator",
                    policy_version = update + 1,
                    episodes = eval_stats.episodes,
                    mean_return = eval_stats.mean_return,
                    max_return = eval_stats.max_return,
                    min_return = eval_stats.min_return,
                    episode_length_mean = eval_stats.mean_ep_len,
                    episode_length_max = eval_stats.max_ep_len,
                    episode_length_min = eval_stats.min_ep_len,
                    duration_ms = eval_started.elapsed().as_secs_f64() * 1_000.0,
                    "deterministic_eval"
                );
            }
        }
    }

    if dist.world_size > 1 {
        finish_collective::<Cuda<f32, i32>>(peer_id)
            .map_err(|e| anyhow::anyhow!("failed to finish burn collective: {e:?}"))?;
    }

    Ok(())
}