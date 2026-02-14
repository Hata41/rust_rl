use anyhow::{bail, Result};
use burn::backend::cuda::Cuda;
use burn::collective::{finish_collective, PeerId, ReduceOperation};
use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor, TensorData};
use burn::tensor::Int;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;
use tracing::{debug, info, span, Level};

use crate::config::{Args, DistInfo};
use crate::env::{make_env, AsyncEnvPool};
use crate::models::{Actor, Agent, Critic};
use crate::ppo::buffer::{flatten_obs, Rollout};
use crate::ppo::loss::{compute_ppo_losses, logprob_and_entropy, sample_actions_gumbel};

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

    let obs0 = flatten_obs(&cur_obs[0]);
    let obs_dim = obs0.len();
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

        let mut roll = Rollout::new(args.rollout_length, local_num_envs, obs_dim, action_dim);

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

                let logits = agent.actor.forward(obs_t.clone()).detach();
                let values2 = agent.critic.forward(obs_t).detach();
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
        let last_v2 = agent.critic.forward(obs_last).detach();
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

        let mut last_actor_loss = 0.0f32;
        let mut last_critic_loss = 0.0f32;
        let mut last_entropy = 0.0f32;
        let mut last_global_grad_norm = 0.0f32;

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

                    last_global_grad_norm = clip_global_grad_norm(
                        &agent.actor,
                        &agent.critic,
                        &mut grads_actor,
                        &mut grads_critic,
                        args.max_grad_norm,
                    );

                    agent.actor = actor_optim.step(current_actor_lr, agent.actor, grads_actor);
                    agent.critic = critic_optim.step(current_critic_lr, agent.critic, grads_critic);

                    last_actor_loss = parts.actor_loss.to_data().to_vec::<f32>().unwrap()[0];
                    last_critic_loss = parts.value_loss.to_data().to_vec::<f32>().unwrap()[0];
                    last_entropy = parts.entropy_mean.to_data().to_vec::<f32>().unwrap()[0];

                    if is_lead {
                        debug!(
                            category = "TRAINER",
                            update,
                            epoch,
                            minibatch = mb,
                            actor_loss = last_actor_loss,
                            value_loss = last_critic_loss,
                            entropy = last_entropy,
                            global_grad_norm = last_global_grad_norm,
                            learning_rate = current_actor_lr,
                            "minibatch"
                        );
                    }
                }
            }
        }
        let optimization_elapsed = optimization_started.elapsed();

        if is_lead && update % 10 == 0 {
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

            let adv_stats = roll.advantage_stats();

            let rollout_steps = (args.rollout_length * local_num_envs * dist.world_size) as f64;
            let rollout_secs = rollout_elapsed.as_secs_f64().max(1.0e-9);
            let steps_per_second = rollout_steps / rollout_secs;

            let timesteps = (update + 1) * local_batch * dist.world_size;
            info!(
                category = "TRAINER",
                actor_loss = last_actor_loss,
                value_loss = last_critic_loss,
                entropy = last_entropy,
                global_grad_norm = last_global_grad_norm,
                "train"
            );

            info!(
                category = "ACTOR",
                mean_return,
                max_return,
                episode_length = mean_ep_len,
                "actor"
            );

            info!(
                category = "MISC",
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
                episode_length_max = max_ep_len,
                "details"
            );
        }
    }

    if dist.world_size > 1 {
        finish_collective::<Cuda<f32, i32>>(peer_id)
            .map_err(|e| anyhow::anyhow!("failed to finish burn collective: {e:?}"))?;
    }

    Ok(())
}