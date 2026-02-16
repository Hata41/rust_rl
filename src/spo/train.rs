use anyhow::{bail, Result};
use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use rand::{rngs::StdRng, SeedableRng};
use std::any::Any;
use std::collections::VecDeque;
use std::time::Instant;
use tracing::{debug, info};

use crate::config::{Args, DistInfo};
use crate::env::{make_env, AsyncEnvPool};
use crate::env_model::{
    build_actor_input_batch, build_critic_input_batch, detect_env_model_from_metadata,
    infer_obs_dim, EnvModelKind,
};
use crate::models::Agent;
use crate::spo::buffer::ReplayBuffer;
use crate::spo::loss::{compute_discrete_mpo_losses, MpoDuals};
use crate::spo::search::{run_smc_search, SearchConfig};

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

fn linear_decay_alpha(update: usize, num_updates: usize) -> f64 {
    if num_updates == 0 {
        return 1.0;
    }
    let progress = (update as f64) / (num_updates as f64);
    1.0 - progress
}

fn greedy_actions_from_weights(
    root_action_weights: &[f32],
    batch: usize,
    action_dim: usize,
) -> Vec<i32> {
    let mut actions = vec![0i32; batch];
    for env_idx in 0..batch {
        let row0 = env_idx * action_dim;
        let row = &root_action_weights[row0..row0 + action_dim];
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &v) in row.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = idx;
            }
        }
        actions[env_idx] = best_idx as i32;
    }
    actions
}

fn run_search_eval<B: AutodiffBackend>(
    env_pool: &AsyncEnvPool,
    agent: &Agent<B>,
    args: &Args,
    model_kind: EnvModelKind,
    obs_dim: usize,
    action_dim: usize,
    device: &B::Device,
    eval_seed: u64,
) -> Result<EvalStats> {
    if args.num_eval_envs == 0 || args.num_eval_episodes == 0 {
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

    let mut eval_rng = StdRng::seed_from_u64(eval_seed ^ 0x5EEDu64);
    let mut cur_steps = env_pool.reset_all(Some(eval_seed))?;
    let eval_env_ids = (0..args.num_eval_envs).collect::<Vec<_>>();

    let mut ep_return = vec![0.0f32; args.num_eval_envs];
    let mut ep_len = vec![0usize; args.num_eval_envs];
    let mut completed_returns = Vec::<f32>::with_capacity(args.num_eval_episodes);
    let mut completed_lengths = Vec::<usize>::with_capacity(args.num_eval_episodes);

    while completed_returns.len() < args.num_eval_episodes {
        let cur_obs = cur_steps.iter().map(|s| &s.obs).collect::<Vec<_>>();
        let cur_masks = cur_steps
            .iter()
            .map(|s| s.action_mask.as_slice())
            .collect::<Vec<_>>();

        let root_state_ids = env_pool.snapshot(&eval_env_ids)?;
        let search_out = run_smc_search(
            env_pool,
            agent,
            &root_state_ids,
            &cur_obs,
            &cur_masks,
            model_kind,
            args,
            SearchConfig {
                num_particles: args.num_particles,
                search_depth: args.search_depth,
                search_gamma: args.search_gamma,
                search_gae_lambda: args.search_gae_lambda,
                resampling_period: args.resampling_period,
                resampling_ess_threshold: args.resampling_ess_threshold,
                adaptive_temperature: args.adaptive_temperature,
                fixed_temperature: args.fixed_temperature,
                root_exploration_dirichlet_alpha: args.root_exploration_dirichlet_alpha,
                root_exploration_dirichlet_fraction: args.root_exploration_dirichlet_fraction,
            },
            obs_dim,
            action_dim,
            device,
            &mut eval_rng,
        )?;

        env_pool.release_batch(&root_state_ids);
        env_pool.release_batch(&search_out.leaf_state_ids);

        let actions = greedy_actions_from_weights(
            &search_out.root_action_weights,
            args.num_eval_envs,
            action_dim,
        );
        let next_steps = env_pool.step_all(&actions)?;

        for env_idx in 0..args.num_eval_envs {
            ep_return[env_idx] += next_steps[env_idx].reward;
            ep_len[env_idx] += 1;
            if next_steps[env_idx].done {
                if completed_returns.len() < args.num_eval_episodes {
                    completed_returns.push(ep_return[env_idx]);
                    completed_lengths.push(ep_len[env_idx]);
                }
                ep_return[env_idx] = 0.0;
                ep_len[env_idx] = 0;
            }
        }

        cur_steps = next_steps;
    }

    let episodes = completed_returns.len();
    let mean_return = completed_returns.iter().sum::<f32>() / episodes as f32;
    let max_return = completed_returns
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_return = completed_returns
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let mean_ep_len = completed_lengths.iter().map(|v| *v as f32).sum::<f32>() / episodes as f32;
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

struct FloatTensorCollector {
    tensors: Vec<Box<dyn Any>>,
}

impl<B: Backend> ModuleVisitor<B> for FloatTensorCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        // Keep tensors on device and preserve typed shape information.
        // This avoids host TensorData materialization during target-network updates.
        self.tensors.push(Box::new(param.val()));
    }
}

struct SoftUpdateMapper {
    tau: f64,
    online_tensors: VecDeque<Box<dyn Any>>,
}

impl<B: Backend> ModuleMapper<B> for SoftUpdateMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, target_tensor, mapper) = param.consume();
        let online_any = self
            .online_tensors
            .pop_front()
            .expect("online parameter stream exhausted during soft update");
        let online_tensor = *online_any
            .downcast::<Tensor<B, D>>()
            .expect("online parameter type mismatch during soft update");
        let mixed = target_tensor.mul_scalar((1.0 - self.tau) as f32)
            + online_tensor.mul_scalar(self.tau as f32);
        Param::from_mapped_value(id, mixed, mapper)
    }
}

pub fn soft_update_params<M, B>(target: &mut M, online: &M, tau: f64)
where
    M: Module<B>,
    B: Backend,
{
    // Device-resident soft update stream:
    // target <- (1 - tau) * target + tau * online
    // without host synchronization on parameter tensors.
    let mut collector = FloatTensorCollector {
        tensors: Vec::new(),
    };
    <M as Module<B>>::visit(online, &mut collector);

    let mut mapper = SoftUpdateMapper {
        tau,
        online_tensors: collector.tensors.into(),
    };

    let mapped = target.clone().map(&mut mapper);
    *target = mapped;
}

pub fn run<B: AutodiffBackend>(args: Args, dist: DistInfo, device: B::Device) -> Result<()> {
    if args.task_id != "Maze-v0" && args.task_id != "BinPack-v0" {
        bail!("SPO currently supports Maze-v0 and BinPack-v0");
    }

    let model_probe = make_env(&args.task_id, &args, args.seed)?;
    let model_kind = detect_env_model_from_metadata(&*model_probe);

    let env_pool = AsyncEnvPool::new(args.num_envs, args.seed, {
        let task = args.task_id.clone();
        let args_clone = args.clone();
        move |seed| make_env(&task, &args_clone, seed).expect("failed to create environment")
    })?;
    let is_lead = dist.rank == 0;
    let eval_pool = if is_lead && args.num_eval_envs > 0 && args.num_eval_episodes > 0 {
        Some(AsyncEnvPool::new(
            args.num_eval_envs,
            args.seed ^ 0xA11CEu64,
            {
                let task = args.task_id.clone();
                let args_clone = args.clone();
                move |seed| {
                    make_env(&task, &args_clone, seed).expect("failed to create eval environment")
                }
            },
        )?)
    } else {
        None
    };

    let reset = env_pool.reset_all(Some(args.seed))?;
    let first_obs = reset
        .first()
        .ok_or_else(|| anyhow::anyhow!("reset returned no environments"))?;

    let obs_dim = infer_obs_dim(&first_obs.obs, model_kind, &args);
    let action_dim = first_obs.action_mask.len();

    let mut replay = ReplayBuffer::new(
        args.replay_buffer_size,
        args.num_envs,
        action_dim,
        args.num_particles,
    );

    let mut agent_online = Agent::<B>::new(obs_dim, args.hidden_dim, action_dim, &device);
    let mut agent_target = agent_online.clone();
    let mut duals = MpoDuals::<B>::new(args.init_log_temperature, args.init_log_alpha, &device);
    let mut actor_optim = AdamConfig::new().init::<B, crate::models::Actor<B>>();
    let mut critic_optim = AdamConfig::new().init::<B, crate::models::Critic<B>>();
    let mut dual_optim = AdamConfig::new().init::<B, MpoDuals<B>>();
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut ep_return = vec![0.0f32; args.num_envs];
    let mut ep_len = vec![0usize; args.num_envs];
    let recent_window = 100usize;
    let mut recent_returns: VecDeque<f32> = VecDeque::with_capacity(recent_window);
    let mut recent_lengths: VecDeque<usize> = VecDeque::with_capacity(recent_window);

    if is_lead {
        info!(
            category = "MISC",
            task = %args.task_id,
            world_size = 1,
            global_num_envs = args.num_envs,
            local_num_envs = args.num_envs,
            rollout_length = args.rollout_length,
            local_batch = args.rollout_length * args.num_envs,
            obs_dim,
            action_dim,
            num_particles = args.num_particles,
            search_depth = args.search_depth,
            replay_capacity = args.replay_buffer_size,
            replay_period = args.sample_period,
            "startup"
        );
    }

    let mut cur_steps = reset;
    let env_ids = (0..args.num_envs).collect::<Vec<_>>();
    for update in 0..args.num_updates {
        let alpha = if args.decay_learning_rates {
            linear_decay_alpha(update, args.num_updates)
        } else {
            1.0
        };
        let current_actor_lr = args.actor_lr * alpha;
        let current_critic_lr = args.critic_lr * alpha;

        let rollout_started = Instant::now();
        let mut search_duration_ms = 0.0f64;
        let mut env_step_duration_ms = 0.0f64;
        let mut replay_add_duration_ms = 0.0f64;
        for _ in 0..args.rollout_length {
            let cur_obs = cur_steps.iter().map(|s| &s.obs).collect::<Vec<_>>();
            let cur_masks = cur_steps
                .iter()
                .map(|s| s.action_mask.as_slice())
                .collect::<Vec<_>>();

            let root_state_ids = env_pool.snapshot(&env_ids)?;
            let search_started = Instant::now();
            let search_out = run_smc_search(
                &env_pool,
                &agent_online,
                &root_state_ids,
                &cur_obs,
                &cur_masks,
                model_kind,
                &args,
                SearchConfig {
                    num_particles: args.num_particles,
                    search_depth: args.search_depth,
                    search_gamma: args.search_gamma,
                    search_gae_lambda: args.search_gae_lambda,
                    resampling_period: args.resampling_period,
                    resampling_ess_threshold: args.resampling_ess_threshold,
                    adaptive_temperature: args.adaptive_temperature,
                    fixed_temperature: args.fixed_temperature,
                    root_exploration_dirichlet_alpha: args.root_exploration_dirichlet_alpha,
                    root_exploration_dirichlet_fraction: args.root_exploration_dirichlet_fraction,
                },
                obs_dim,
                action_dim,
                &device,
                &mut rng,
            )?;
            search_duration_ms += search_started.elapsed().as_secs_f64() * 1_000.0;

            env_pool.release_batch(&root_state_ids);
            let actions = search_out.root_actions.clone();
            env_pool.release_batch(&search_out.leaf_state_ids);

            let env_step_started = Instant::now();
            let next_steps = env_pool.step_all(&actions)?;
            env_step_duration_ms += env_step_started.elapsed().as_secs_f64() * 1_000.0;
            for env_idx in 0..args.num_envs {
                ep_return[env_idx] += next_steps[env_idx].reward;
                ep_len[env_idx] += 1;
                if next_steps[env_idx].done {
                    if recent_returns.len() >= recent_window {
                        recent_returns.pop_front();
                    }
                    if recent_lengths.len() >= recent_window {
                        recent_lengths.pop_front();
                    }
                    recent_returns.push_back(ep_return[env_idx]);
                    recent_lengths.push_back(ep_len[env_idx]);
                    ep_return[env_idx] = 0.0;
                    ep_len[env_idx] = 0;
                }
            }

            let obs_batch = cur_steps.iter().map(|s| &s.obs).collect::<Vec<_>>();
            let next_obs_batch = next_steps.iter().map(|s| &s.obs).collect::<Vec<_>>();
            let rewards = next_steps.iter().map(|s| s.reward).collect::<Vec<_>>();
            let dones = next_steps.iter().map(|s| s.done).collect::<Vec<_>>();
            let action_masks = cur_steps
                .iter()
                .map(|s| s.action_mask.as_slice())
                .collect::<Vec<_>>();
            let next_action_masks = next_steps
                .iter()
                .map(|s| s.action_mask.as_slice())
                .collect::<Vec<_>>();

            let replay_add_started = Instant::now();
            replay.add_timestep(
                &obs_batch,
                &next_obs_batch,
                &actions,
                &rewards,
                &dones,
                &action_masks,
                &next_action_masks,
                &search_out.root_action_weights,
                &search_out.sampled_actions,
                &search_out.sampled_advantages,
            )?;
            replay_add_duration_ms += replay_add_started.elapsed().as_secs_f64() * 1_000.0;

            cur_steps = next_steps;
        }
        let rollout_elapsed = rollout_started.elapsed();

        let optimization_started = Instant::now();
        // Accumulate metrics on-device and read back once per update to reduce host syncs.
        let mut actor_loss_acc = Tensor::<B, 1>::zeros([1], &device);
        let mut critic_loss_acc = Tensor::<B, 1>::zeros([1], &device);
        let mut temp_loss_acc = Tensor::<B, 1>::zeros([1], &device);
        let mut alpha_loss_acc = Tensor::<B, 1>::zeros([1], &device);
        let mut mpo_total_loss_acc = Tensor::<B, 1>::zeros([1], &device);
        let mut optimization_updates = 0usize;
        let mut sample_duration_ms = 0.0f64;
        let mut model_forward_loss_duration_ms = 0.0f64;
        let mut backward_step_duration_ms = 0.0f64;

        if replay.can_sample(args.sample_sequence_length) {
            for _ in 0..args.epochs {
                for _ in 0..args.num_minibatches {
                    let sample_started = Instant::now();
                    let sampled = replay.sample_sequences(
                        args.num_envs,
                        args.sample_sequence_length,
                        args.sample_period,
                        &mut rng,
                    );
                    sample_duration_ms += sample_started.elapsed().as_secs_f64() * 1_000.0;
                    let batch = sampled.actions.len();
                    if batch == 0 {
                        continue;
                    }

                    let forward_started = Instant::now();
                    let policy_logits = agent_online.actor_logits(build_actor_input_batch::<B>(
                        &sampled.obs,
                        model_kind,
                        &args,
                        obs_dim,
                        &device,
                    )?);
                    let target_action_weights = Tensor::<B, 2>::from_data(
                        TensorData::new(sampled.root_action_weights, [batch, sampled.action_dim]),
                        &device,
                    );
                    let sampled_advantages = Tensor::<B, 2>::from_data(
                        TensorData::new(sampled.sampled_advantages, [batch, sampled.num_particles]),
                        &device,
                    );

                    let values = agent_online
                        .critic_values(build_critic_input_batch::<B>(
                            &sampled.obs,
                            model_kind,
                            &args,
                            obs_dim,
                            &device,
                        )?)
                        .reshape([batch]);
                    let target_v_tm1 = agent_target
                        .critic_values(build_critic_input_batch::<B>(
                            &sampled.obs,
                            model_kind,
                            &args,
                            obs_dim,
                            &device,
                        )?)
                        .reshape([batch]);
                    let target_v_t = agent_target
                        .critic_values(build_critic_input_batch::<B>(
                            &sampled.next_obs,
                            model_kind,
                            &args,
                            obs_dim,
                            &device,
                        )?)
                        .reshape([batch]);

                    let target_v_tm1_vec = target_v_tm1
                        .clone()
                        .to_data()
                        .to_vec::<f32>()
                        .map_err(|e| anyhow::anyhow!("failed to convert target_v_tm1: {e:?}"))?;
                    let target_v_t_vec = target_v_t
                        .detach()
                        .to_data()
                        .to_vec::<f32>()
                        .map_err(|e| anyhow::anyhow!("failed to convert target_v_t: {e:?}"))?;

                    let mut target_values = vec![0.0f32; batch];
                    let sequence_len = args.sample_sequence_length;
                    let num_sequences = batch / sequence_len;
                    for sequence_idx in 0..num_sequences {
                        let mut gae = 0.0f32;
                        for t in (0..sequence_len).rev() {
                            let i = sequence_idx * sequence_len + t;
                            let not_done = if sampled.dones[i] { 0.0 } else { 1.0 };
                            let delta = sampled.rewards[i]
                                + args.gamma * not_done * target_v_t_vec[i]
                                - target_v_tm1_vec[i];
                            gae = delta + args.gamma * args.gae_lambda * not_done * gae;
                            target_values[i] = gae + target_v_tm1_vec[i];
                        }
                    }
                    let critic_targets =
                        Tensor::<B, 1>::from_data(TensorData::new(target_values, [batch]), &device);

                    let parts = compute_discrete_mpo_losses(
                        policy_logits,
                        target_action_weights,
                        sampled_advantages,
                        values,
                        critic_targets,
                        &duals,
                        args.epsilon,
                        args.epsilon_policy,
                    );
                    model_forward_loss_duration_ms +=
                        forward_started.elapsed().as_secs_f64() * 1_000.0;

                    let backward_started = Instant::now();
                    let mut grads = parts.total_loss.backward();

                    let grads_actor = GradientsParams::from_module::<B, crate::models::Actor<B>>(
                        &mut grads,
                        &agent_online.actor,
                    );
                    let grads_critic = GradientsParams::from_module::<B, crate::models::Critic<B>>(
                        &mut grads,
                        &agent_online.critic,
                    );
                    let grads_duals =
                        GradientsParams::from_module::<B, MpoDuals<B>>(&mut grads, &duals);

                    agent_online.actor =
                        actor_optim.step(current_actor_lr, agent_online.actor, grads_actor);
                    agent_online.critic =
                        critic_optim.step(current_critic_lr, agent_online.critic, grads_critic);
                    duals = dual_optim.step(args.dual_lr, duals, grads_duals);

                    soft_update_params::<Agent<B>, B>(&mut agent_target, &agent_online, args.tau);
                    backward_step_duration_ms += backward_started.elapsed().as_secs_f64() * 1_000.0;

                    actor_loss_acc = actor_loss_acc + parts.actor_loss.detach();
                    critic_loss_acc = critic_loss_acc + parts.critic_loss.detach();
                    temp_loss_acc = temp_loss_acc + parts.loss_temperature.detach();
                    alpha_loss_acc = alpha_loss_acc + parts.loss_alpha.detach();
                    mpo_total_loss_acc = mpo_total_loss_acc + parts.total_loss.detach();
                    optimization_updates += 1;
                }
            }
        }
        let optimization_elapsed = optimization_started.elapsed();

        if is_lead {
            let denom = (optimization_updates as f64).max(1.0);
            let actor_loss_sum = actor_loss_acc
                .to_data()
                .to_vec::<f32>()
                .ok()
                .and_then(|v| v.into_iter().next())
                .unwrap_or(0.0) as f64;
            let critic_loss_sum = critic_loss_acc
                .to_data()
                .to_vec::<f32>()
                .ok()
                .and_then(|v| v.into_iter().next())
                .unwrap_or(0.0) as f64;
            let temp_loss_sum = temp_loss_acc
                .to_data()
                .to_vec::<f32>()
                .ok()
                .and_then(|v| v.into_iter().next())
                .unwrap_or(0.0) as f64;
            let alpha_loss_sum = alpha_loss_acc
                .to_data()
                .to_vec::<f32>()
                .ok()
                .and_then(|v| v.into_iter().next())
                .unwrap_or(0.0) as f64;
            let mpo_total_loss_sum = mpo_total_loss_acc
                .to_data()
                .to_vec::<f32>()
                .ok()
                .and_then(|v| v.into_iter().next())
                .unwrap_or(0.0) as f64;

            let mean_actor_loss = actor_loss_sum / denom;
            let mean_critic_loss = critic_loss_sum / denom;
            let mean_temp_loss = temp_loss_sum / denom;
            let mean_alpha_loss = alpha_loss_sum / denom;
            let mean_mpo_total_loss = mpo_total_loss_sum / denom;

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
            let max_ep_len = recent_lengths.iter().copied().max().unwrap_or(0);
            let min_ep_len = recent_lengths.iter().copied().min().unwrap_or(0);

            let rollout_steps = (args.rollout_length * args.num_envs) as f64;
            let total_update_secs = (rollout_elapsed + optimization_elapsed)
                .as_secs_f64()
                .max(1.0e-9);
            let steps_per_second = rollout_steps / total_update_secs;
            let timesteps = (update + 1) * args.rollout_length * args.num_envs;

            info!(
                category = "TRAINER",
                policy_version = update + 1,
                actor_loss = mean_actor_loss,
                critic_loss = mean_critic_loss,
                entropy = 0.0,
                global_grad_norm = 0.0,
                mpo_temperature_loss = mean_temp_loss,
                mpo_alpha_loss = mean_alpha_loss,
                mpo_total_loss = mean_mpo_total_loss,
                replay_len = replay.len(),
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
                critic_learning_rate = current_critic_lr,
                "misc"
            );

            debug!(
                category = "MISC",
                timesteps,
                rollout_duration_ms = rollout_elapsed.as_secs_f64() * 1_000.0,
                rollout_search_duration_ms = search_duration_ms,
                rollout_env_step_duration_ms = env_step_duration_ms,
                rollout_replay_add_duration_ms = replay_add_duration_ms,
                optimization_duration_ms = optimization_elapsed.as_secs_f64() * 1_000.0,
                optimization_sample_duration_ms = sample_duration_ms,
                optimization_forward_loss_duration_ms = model_forward_loss_duration_ms,
                optimization_backward_step_duration_ms = backward_step_duration_ms,
                replay_len = replay.len(),
                mpo_total_loss = mean_mpo_total_loss,
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
                let eval_stats = run_search_eval(
                    eval_pool,
                    &agent_online,
                    &args,
                    model_kind,
                    obs_dim,
                    action_dim,
                    &device,
                    args.seed.wrapping_add(update as u64),
                )?;
                let eval_duration_ms = eval_started.elapsed().as_secs_f64() * 1_000.0;

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
                    duration_ms = eval_duration_ms,
                    "deterministic_eval"
                );
            }
        }
    }

    Ok(())
}
