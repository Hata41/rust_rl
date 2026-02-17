use anyhow::{bail, Result};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Int;
use burn::tensor::{Tensor, TensorData};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::{debug, info, span, Level};

use crate::common::config::{Args, DistInfo};
use crate::common::runtime::env::{make_env, AsyncEnvPool};
use crate::common::model::observation_adapter::{
    ObservationAdapter, resolve_observation_adapter,
};
use crate::common::runtime::evaluation::{run_eval_with_policy, EvalStats};
use crate::common::model::models::{Actor, Agent, Critic, PolicyInput};
use crate::algorithms::ppo::buffer::{flatten_obs_into, Rollout};
use crate::algorithms::ppo::loss::{
    compute_ppo_losses, logprob_and_entropy, masked_logits, sample_actions_categorical,
};
use crate::common::runtime::telemetry::TrainingContext;
use crate::common::utils::optimization::{clip_global_grad_norm, linear_decay_alpha};

struct PpoOptimizationSummary {
    elapsed: Duration,
    actor_loss_sum: f64,
    critic_loss_sum: f64,
    entropy_sum: f64,
    grad_norm_sum: f64,
    data_prep_duration_ms: f64,
    forward_loss_duration_ms: f64,
    backward_step_duration_ms: f64,
}

struct PpoOptimizer;

impl PpoOptimizer {
    #[allow(clippy::too_many_arguments)]
    fn optimize_update<B: AutodiffBackend>(
        roll: &Rollout,
        is_binpack: bool,
        local_batch: usize,
        obs_dim: usize,
        action_dim: usize,
        args: &Args,
        update: usize,
        is_lead: bool,
        adapter: &dyn ObservationAdapter<B>,
        device: &B::Device,
        rng: &mut StdRng,
        agent: &mut Agent<B>,
        actor_optim: &mut impl Optimizer<Actor<B>, B>,
        critic_optim: &mut impl Optimizer<Critic<B>, B>,
        current_actor_lr: f64,
        current_critic_lr: f64,
    ) -> Result<PpoOptimizationSummary> {
        let mb_size = local_batch / args.num_minibatches;
        let mut all_indices: Vec<usize> = (0..local_batch).collect();

        let mut actor_loss_acc = Tensor::<B, 1>::zeros([1], device);
        let mut critic_loss_acc = Tensor::<B, 1>::zeros([1], device);
        let mut entropy_acc = Tensor::<B, 1>::zeros([1], device);
        let mut grad_norm_sum = 0.0f64;
        let mut optimization_data_prep_duration_ms = 0.0f64;
        let mut optimization_forward_loss_duration_ms = 0.0f64;
        let mut optimization_backward_step_duration_ms = 0.0f64;

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
                all_indices.shuffle(rng);

                for mb in 0..args.num_minibatches {
                    let start = mb * mb_size;
                    let end = start + mb_size;
                    let mb_idx = &all_indices[start..end];

                    let data_prep_started = Instant::now();
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

                        let policy_input = adapter.build_policy_input_from_binpack_parts(
                            items_mb,
                            ems_mb,
                            items_valid_mb,
                            ems_valid_mb,
                            mb_size,
                            args,
                            device,
                        )?;
                        let (logits, v2) = agent.policy_value(policy_input);

                        let act_t = Tensor::<B, 1, Int>::from_data(
                            TensorData::new(act_mb, [mb_size]),
                            device,
                        );
                        let mask_t = Tensor::<B, 2>::from_data(
                            TensorData::new(mask_mb, [mb_size, action_dim]),
                            device,
                        );
                        let old_lp_t = Tensor::<B, 1>::from_data(
                            TensorData::new(old_lp_mb, [mb_size]),
                            device,
                        );
                        let old_v_t = Tensor::<B, 1>::from_data(
                            TensorData::new(old_v_mb, [mb_size]),
                            device,
                        );
                        let adv_t = Tensor::<B, 1>::from_data(TensorData::new(adv_mb, [mb_size]), device);
                        let tgt_t = Tensor::<B, 1>::from_data(TensorData::new(tgt_mb, [mb_size]), device);

                        (logits, act_t, mask_t, old_lp_t, old_v_t, adv_t, tgt_t, v2)
                    } else {
                        let (obs_mb, act_mb, mask_mb, old_lp_mb, old_v_mb, adv_mb, tgt_mb) =
                            roll.minibatch(mb_idx);

                        let obs_t = Tensor::<B, 2>::from_data(
                            TensorData::new(obs_mb, [mb_size, obs_dim]),
                            device,
                        );
                        let act_t = Tensor::<B, 1, Int>::from_data(
                            TensorData::new(act_mb, [mb_size]),
                            device,
                        );
                        let mask_t = Tensor::<B, 2>::from_data(
                            TensorData::new(mask_mb, [mb_size, action_dim]),
                            device,
                        );
                        let old_lp_t = Tensor::<B, 1>::from_data(
                            TensorData::new(old_lp_mb, [mb_size]),
                            device,
                        );
                        let old_v_t = Tensor::<B, 1>::from_data(
                            TensorData::new(old_v_mb, [mb_size]),
                            device,
                        );
                        let adv_t = Tensor::<B, 1>::from_data(TensorData::new(adv_mb, [mb_size]), device);
                        let tgt_t = Tensor::<B, 1>::from_data(TensorData::new(tgt_mb, [mb_size]), device);

                        let (logits, v2) = agent.policy_value(PolicyInput::Dense { obs: obs_t });

                        (logits, act_t, mask_t, old_lp_t, old_v_t, adv_t, tgt_t, v2)
                    };
                    optimization_data_prep_duration_ms +=
                        data_prep_started.elapsed().as_secs_f64() * 1_000.0;

                    let forward_loss_started = Instant::now();
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
                    optimization_forward_loss_duration_ms +=
                        forward_loss_started.elapsed().as_secs_f64() * 1_000.0;

                    let backward_step_started = Instant::now();
                    let mut grads = parts.total_loss.backward();

                    let mut grads_actor =
                        GradientsParams::from_module::<B, Actor<B>>(&mut grads, &agent.actor);
                    let mut grads_critic =
                        GradientsParams::from_module::<B, Critic<B>>(&mut grads, &agent.critic);

                    let global_grad_norm = clip_global_grad_norm(
                        &agent.actor,
                        &agent.critic,
                        &mut grads_actor,
                        &mut grads_critic,
                        args.max_grad_norm,
                    );

                    agent.actor = actor_optim.step(current_actor_lr, agent.actor.clone(), grads_actor);
                    agent.critic = critic_optim.step(current_critic_lr, agent.critic.clone(), grads_critic);
                    optimization_backward_step_duration_ms +=
                        backward_step_started.elapsed().as_secs_f64() * 1_000.0;

                    actor_loss_acc = actor_loss_acc + parts.actor_loss.clone().detach();
                    critic_loss_acc = critic_loss_acc + parts.value_loss.clone().detach();
                    entropy_acc = entropy_acc + parts.entropy_mean.clone().detach();
                    grad_norm_sum += global_grad_norm as f64;

                    if is_lead {
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

        let actor_loss_sum = actor_loss_acc
            .to_data()
            .to_vec::<f32>()
            .unwrap_or_default()
            .first()
            .copied()
            .unwrap_or(0.0) as f64;
        let critic_loss_sum = critic_loss_acc
            .to_data()
            .to_vec::<f32>()
            .unwrap_or_default()
            .first()
            .copied()
            .unwrap_or(0.0) as f64;
        let entropy_sum = entropy_acc
            .to_data()
            .to_vec::<f32>()
            .unwrap_or_default()
            .first()
            .copied()
            .unwrap_or(0.0) as f64;

        Ok(PpoOptimizationSummary {
            elapsed: optimization_started.elapsed(),
            actor_loss_sum,
            critic_loss_sum,
            entropy_sum,
            grad_norm_sum,
            data_prep_duration_ms: optimization_data_prep_duration_ms,
            forward_loss_duration_ms: optimization_forward_loss_duration_ms,
            backward_step_duration_ms: optimization_backward_step_duration_ms,
        })
    }
}

fn run_deterministic_eval<B: AutodiffBackend>(
    agent: &Agent<B>,
    eval_pool: &AsyncEnvPool,
    eval_seed: u64,
    num_eval_envs: usize,
    num_eval_episodes: usize,
    adapter: &dyn ObservationAdapter<B>,
    args: &Args,
    obs_dim: usize,
    action_dim: usize,
    device: &B::Device,
) -> Result<EvalStats> {
    run_eval_with_policy(
        eval_pool,
        eval_seed,
        num_eval_envs,
        num_eval_episodes,
        |_, cur_steps| {
            let cur_obs = cur_steps.iter().map(|s| &s.obs).collect::<Vec<_>>();
            let cur_mask = cur_steps
                .iter()
                .map(|s| s.action_mask.as_slice())
                .collect::<Vec<_>>();

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

            let logits = agent
                .actor_logits(adapter.build_actor_input_batch(&cur_obs, args, obs_dim, device)?)
                .detach();

            let masked = masked_logits(logits, mask_t);
            let probs = burn::tensor::activation::softmax(masked, 1);
            let actions_t = probs.argmax(1).reshape([num_eval_envs]);

            let actions_data = actions_t.to_data();
            let actions_vec: Vec<i32> = match actions_data.clone().to_vec::<i32>() {
                Ok(v) => v,
                Err(_) => actions_data
                    .to_vec::<i64>()
                    .map_err(|e| anyhow::anyhow!("failed to convert greedy actions: {e:?}"))?
                    .into_iter()
                    .map(|v| v as i32)
                    .collect(),
            };

            Ok(actions_vec)
        },
    )
}

fn run_loop<B: AutodiffBackend>(
    args: Args,
    dist: DistInfo,
    device: B::Device,
) -> Result<()> {
    let is_lead = dist.rank == 0;

    if args.num_envs == 0 {
        bail!("num_envs must be > 0");
    }

    let local_num_envs = args.num_envs;

    let local_batch = args.rollout_length * local_num_envs;
    if local_batch % args.num_minibatches != 0 {
        bail!(
            "rollout_length*local_num_envs = {local_batch} must be divisible by num_minibatches = {}",
            args.num_minibatches
        );
    }

    B::seed(&device, args.seed);

    let model_probe = make_env(&args.task_id, &args, args.seed)?;
    let adapter = resolve_observation_adapter::<B>(&*model_probe, &args);

    let env_pool = AsyncEnvPool::new(local_num_envs, args.seed, {
        let args = args.clone();
        move |seed| make_env(&args.task_id, &args, seed).unwrap()
    })?;

    let reset_out = env_pool.reset_all(Some(args.seed + 10_000))?;
    let mut cur_obs = reset_out.iter().map(|s| s.obs.clone()).collect::<Vec<_>>();
    let mut cur_mask = reset_out
        .iter()
        .map(|s| s.action_mask.clone())
        .collect::<Vec<_>>();

    let is_binpack = adapter.uses_binpack_architecture();

    let obs_dim = adapter.infer_obs_dim(&cur_obs[0], &args);
    let action_dim = cur_mask[0].len();

    if is_lead {
        info!(
            category = "MISC",
            task = %args.task_id,
            world_size = 1,
            global_num_envs = args.num_envs,
            local_num_envs,
            rollout_length = args.rollout_length,
            local_batch,
            obs_dim,
            action_dim,
            "startup"
        );
    }

    let mut agent: Agent<B> = Agent::new(
        obs_dim,
        args.hidden_dim,
        action_dim,
        is_binpack,
        &device,
    );

    let mut actor_optim = AdamConfig::new().init::<B, Actor<B>>();
    let mut critic_optim = AdamConfig::new().init::<B, Critic<B>>();

    let mut ep_return = vec![0.0f32; local_num_envs];
    let mut ep_len = vec![0usize; local_num_envs];
    let recent_window = 100usize;
    let mut recent_returns: VecDeque<f32> = VecDeque::with_capacity(recent_window);
    let mut recent_lengths: VecDeque<usize> = VecDeque::with_capacity(recent_window);

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
            world_size = 1
        );
        let _update_guard = update_span.enter();

        let alpha = if args.decay_learning_rates {
            linear_decay_alpha(update, args.num_updates)
        } else {
            1.0
        };
        let current_actor_lr = args.actor_lr * alpha;
        let current_critic_lr = args.critic_lr * alpha;
        let timesteps = (update + 1) * local_batch;

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
        roll.assert_preallocated();
        let mut obs_flat_all = if is_binpack {
            Vec::new()
        } else {
            vec![0.0f32; local_num_envs * obs_dim]
        };

        let rollout_started = Instant::now();
        let mut rollout_model_duration_ms = 0.0f64;
        let mut rollout_env_step_duration_ms = 0.0f64;
        let mut rollout_store_duration_ms = 0.0f64;
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

                if !is_binpack {
                    for e in 0..local_num_envs {
                        let base = e * obs_dim;
                        flatten_obs_into(&cur_obs[e], &mut obs_flat_all[base..base + obs_dim]);
                    }
                }

                let rollout_model_started = Instant::now();
                let cur_obs_refs = cur_obs.iter().collect::<Vec<_>>();
                let (logits, values2) = agent.policy_value(
                    adapter.build_policy_input_batch(&cur_obs_refs, &args, obs_dim, &device)?,
                );
                let (logits, values2) = (logits.detach(), values2.detach());
                let values = values2.reshape([local_num_envs]);

                let actions_t =
                    sample_actions_categorical::<B>(logits.clone(), mask_t.clone(), &device);

                let (logp_t, _ent_t) = logprob_and_entropy::<B>(logits, mask_t, actions_t.clone());
                rollout_model_duration_ms +=
                    rollout_model_started.elapsed().as_secs_f64() * 1_000.0;

                let actions_data = actions_t.to_data();
                let actions_vec: Vec<i32> = match actions_data.clone().to_vec::<i32>() {
                    Ok(v) => v,
                    Err(_) => actions_data
                        .to_vec::<i64>()
                        .map_err(|e| anyhow::anyhow!("failed to convert sampled actions: {e:?}"))?
                        .into_iter()
                        .map(|v| v as i32)
                        .collect(),
                };
                let logp_vec: Vec<f32> = logp_t.to_data().to_vec().unwrap();
                let values_vec: Vec<f32> = values.to_data().to_vec().unwrap();

                let rollout_env_started = Instant::now();
                let step_out = env_pool.step_all(&actions_vec)?;
                rollout_env_step_duration_ms +=
                    rollout_env_started.elapsed().as_secs_f64() * 1_000.0;

                let rollout_store_started = Instant::now();
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
                        let base = e * obs_dim;
                        let obs_flat = &obs_flat_all[base..base + obs_dim];
                        roll.store_step(
                            t,
                            e,
                            obs_flat,
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
                        recent_returns.push_back(ep_return[e]);
                        if recent_returns.len() > recent_window {
                            recent_returns.pop_front();
                        }
                        recent_lengths.push_back(ep_len[e]);
                        if recent_lengths.len() > recent_window {
                            recent_lengths.pop_front();
                        }
                        ep_return[e] = 0.0;
                        ep_len[e] = 0;
                    }
                }
                rollout_store_duration_ms +=
                    rollout_store_started.elapsed().as_secs_f64() * 1_000.0;

                for e in 0..local_num_envs {
                    cur_obs[e].clone_from(&step_out[e].obs);
                    cur_mask[e].clone_from(&step_out[e].action_mask);
                }
            }

            if !is_binpack {
                for e in 0..local_num_envs {
                    let base = e * obs_dim;
                    flatten_obs_into(&cur_obs[e], &mut obs_flat_all[base..base + obs_dim]);
                }
            }
        }
        let rollout_elapsed = rollout_started.elapsed();

        let last_v2 = if is_binpack {
            let cur_obs_refs = cur_obs.iter().collect::<Vec<_>>();
            agent
                .critic_values(adapter.build_critic_input_batch(&cur_obs_refs, &args, obs_dim, &device)?)
                .detach()
        } else {
            let obs_last = Tensor::<B, 2>::from_data(
                TensorData::new(obs_flat_all, [local_num_envs, obs_dim]),
                &device,
            );
            let policy_input = PolicyInput::Dense { obs: obs_last };
            agent.policy_value(policy_input).1.detach()
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

        let optimization_summary = PpoOptimizer::optimize_update(
            &roll,
            is_binpack,
            local_batch,
            obs_dim,
            action_dim,
            &args,
            update,
            is_lead,
            adapter.as_ref(),
            &device,
            &mut rng,
            &mut agent,
            &mut actor_optim,
            &mut critic_optim,
            current_actor_lr,
            current_critic_lr,
        )?;
        let optimization_elapsed = optimization_summary.elapsed;

        if is_lead {
            let num_updates_in_cycle = (args.epochs * args.num_minibatches) as f64;
            let denom = num_updates_in_cycle.max(1.0);
            let actor_loss_sum = optimization_summary.actor_loss_sum;
            let critic_loss_sum = optimization_summary.critic_loss_sum;
            let entropy_sum = optimization_summary.entropy_sum;
            let mean_actor_loss = actor_loss_sum / denom;
            let mean_critic_loss = critic_loss_sum / denom;
            let mean_entropy = entropy_sum / denom;
            let mean_global_grad_norm = optimization_summary.grad_norm_sum / denom;

            let mean_return = if recent_returns.is_empty() {
                0.0
            } else {
                recent_returns.iter().copied().sum::<f32>() / (recent_returns.len() as f32)
            };
            let max_return = if recent_returns.is_empty() {
                0.0
            } else {
                recent_returns
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
            };
            let min_return = if recent_returns.is_empty() {
                0.0
            } else {
                recent_returns.iter().copied().fold(f32::INFINITY, f32::min)
            };
            let mean_ep_len = if recent_lengths.is_empty() {
                0.0
            } else {
                recent_lengths.iter().map(|v| *v as f32).sum::<f32>()
                    / (recent_lengths.len() as f32)
            };
            let max_ep_len = if recent_lengths.is_empty() {
                0
            } else {
                recent_lengths.iter().copied().max().unwrap_or(0)
            };
            let min_ep_len = if recent_lengths.is_empty() {
                0
            } else {
                recent_lengths.iter().copied().min().unwrap_or(0)
            };

            let adv_stats = roll.advantage_stats();

            let rollout_steps = (args.rollout_length * args.num_envs) as f64;
            let total_update_secs = (rollout_elapsed + optimization_elapsed)
                .as_secs_f64()
                .max(1.0e-9);
            let steps_per_second = rollout_steps / total_update_secs;

            info!(
                category = "TRAINER",
                timesteps,
                policy_version = update + 1,
                actor_loss = mean_actor_loss,
                critic_loss = mean_critic_loss,
                entropy = mean_entropy,
                global_grad_norm = mean_global_grad_norm,
                "train"
            );

            info!(
                category = "ACTOR",
                timesteps,
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
                rollout_model_duration_ms,
                rollout_env_step_duration_ms,
                rollout_store_duration_ms,
                optimization_duration_ms = optimization_elapsed.as_secs_f64() * 1_000.0,
                optimization_data_prep_duration_ms = optimization_summary.data_prep_duration_ms,
                optimization_forward_loss_duration_ms = optimization_summary.forward_loss_duration_ms,
                optimization_backward_step_duration_ms = optimization_summary.backward_step_duration_ms,
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
                    args.seed.wrapping_add(999).wrapping_add(update as u64),
                    eval_num_envs,
                    args.num_eval_episodes,
                    adapter.as_ref(),
                    &args,
                    obs_dim,
                    action_dim,
                    &device,
                )?;

                info!(
                    category = "EVALUATOR",
                    timesteps,
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

    Ok(())
}

pub struct PpoTrainer<'a, B: AutodiffBackend> {
    args: Args,
    dist: DistInfo,
    _context: &'a TrainingContext,
    device: B::Device,
}

impl<'a, B: AutodiffBackend> PpoTrainer<'a, B> {
    pub fn new(args: Args, dist: DistInfo, context: &'a TrainingContext, device: B::Device) -> Self {
        Self {
            args,
            dist,
            _context: context,
            device,
        }
    }

    pub fn run(self) -> Result<()> {
        run_loop::<B>(self.args, self.dist, self.device)
    }
}

pub fn run<B: AutodiffBackend>(
    args: Args,
    dist: DistInfo,
    context: &TrainingContext,
    device: B::Device,
) -> Result<()> {
    PpoTrainer::<B>::new(args, dist, context, device).run()
}
