use anyhow::{bail, Result};
use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::VecDeque;
use tracing::info;

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
    episodes: usize,
}

fn greedy_actions_from_weights(root_action_weights: &[f32], batch: usize, action_dim: usize) -> Vec<i32> {
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
        let cur_obs = cur_steps.iter().map(|s| s.obs.clone()).collect::<Vec<_>>();
        let cur_masks = cur_steps
            .iter()
            .map(|s| s.action_mask.clone())
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

    Ok(EvalStats {
        mean_return,
        max_return,
        min_return,
        mean_ep_len,
        episodes,
    })
}

struct FloatTensorCollector {
    tensors: Vec<TensorData>,
}

impl<B: Backend> ModuleVisitor<B> for FloatTensorCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.tensors.push(param.val().to_data());
    }
}

struct SoftUpdateMapper {
    tau: f64,
    online_tensors: VecDeque<TensorData>,
}

impl<B: Backend> ModuleMapper<B> for SoftUpdateMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, target_tensor, mapper) = param.consume();
        let online_data = self
            .online_tensors
            .pop_front()
            .expect("online parameter stream exhausted during soft update");
        let online_tensor = Tensor::<B, D>::from_data(online_data, &target_tensor.device());
        let blended = target_tensor.mul_scalar((1.0 - self.tau) as f32)
            + online_tensor.mul_scalar(self.tau as f32);
        Param::from_mapped_value(id, blended, mapper)
    }
}

pub fn soft_update_params<M, B>(target: &mut M, online: &M, tau: f64)
where
    M: Module<B>,
    B: Backend,
{
    let mut collector = FloatTensorCollector { tensors: Vec::new() };
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
        Some(AsyncEnvPool::new(args.num_eval_envs, args.seed ^ 0xA11CEu64, {
            let task = args.task_id.clone();
            let args_clone = args.clone();
            move |seed| make_env(&task, &args_clone, seed).expect("failed to create eval environment")
        })?)
    } else {
        None
    };

    let reset = env_pool.reset_all(Some(args.seed))?;
    let first_obs = reset
        .first()
        .ok_or_else(|| anyhow::anyhow!("reset returned no environments"))?;

    let obs_dim = infer_obs_dim(&first_obs.obs, model_kind, &args);
    let action_dim = first_obs.action_mask.len();

    let mut replay = ReplayBuffer::new(args.replay_buffer_size, action_dim);

    let mut agent_online = Agent::<B>::new(obs_dim, args.hidden_dim, action_dim, &device);
    let mut agent_target = agent_online.clone();
    let mut duals = MpoDuals::<B>::new(
        args.init_log_temperature,
        args.init_log_alpha,
        &device,
    );
    let mut actor_optim = AdamConfig::new().init::<B, crate::models::Actor<B>>();
    let mut critic_optim = AdamConfig::new().init::<B, crate::models::Critic<B>>();
    let mut dual_optim = AdamConfig::new().init::<B, MpoDuals<B>>();
    let mut rng = StdRng::seed_from_u64(args.seed + dist.rank as u64);

    if dist.rank == 0 {
        info!(
            category = "TRAINER",
            mode = "spo",
            num_envs = args.num_envs,
            num_particles = args.num_particles,
            search_depth = args.search_depth,
            replay_capacity = args.replay_buffer_size,
            "initialized SPO trainer"
        );
    }

    let mut cur_steps = reset;
    let env_ids = (0..args.num_envs).collect::<Vec<_>>();
    for update in 0..args.num_updates {
        for _ in 0..args.rollout_length {
            let cur_obs = cur_steps.iter().map(|s| s.obs.clone()).collect::<Vec<_>>();
            let cur_masks = cur_steps
                .iter()
                .map(|s| s.action_mask.clone())
                .collect::<Vec<_>>();

            let root_state_ids = env_pool.snapshot(&env_ids)?;
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

            env_pool.release_batch(&root_state_ids);
            let actions = search_out.root_actions.clone();
            env_pool.release_batch(&search_out.leaf_state_ids);

            let next_steps = env_pool.step_all(&actions)?;
            for (env_idx, ((cur, nxt), action)) in cur_steps
                .iter()
                .zip(next_steps.iter())
                .zip(actions.iter())
                .enumerate()
            {
                let row0 = env_idx * action_dim;
                replay.add(
                    &cur.obs,
                    &nxt.obs,
                    *action,
                    nxt.reward,
                    nxt.done,
                    &cur.action_mask,
                    &nxt.action_mask,
                    &search_out.root_action_weights[row0..row0 + action_dim],
                )?;
            }

            cur_steps = next_steps;
        }

        if replay.len() >= args.sample_sequence_length * args.num_envs {
            let mut last_mpo_total = 0.0f64;
            for _ in 0..args.epochs {
                for _ in 0..args.num_minibatches {
                    let sampled = replay.sample_random(args.num_envs, &mut rng);
                    let batch = sampled.actions.len();

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

                    let values = agent_online
                        .critic_values(build_critic_input_batch::<B>(
                            &sampled.obs,
                            model_kind,
                            &args,
                            obs_dim,
                            &device,
                        )?)
                        .reshape([batch]);
                    let next_values = agent_target
                        .critic_values(build_critic_input_batch::<B>(
                            &sampled.next_obs,
                            model_kind,
                            &args,
                            obs_dim,
                            &device,
                        )?)
                        .reshape([batch]);

                    let not_done = sampled
                        .dones
                        .iter()
                        .map(|d| if *d { 0.0f32 } else { 1.0f32 })
                        .collect::<Vec<_>>();
                    let reward_t =
                        Tensor::<B, 1>::from_data(TensorData::new(sampled.rewards, [batch]), &device);
                    let not_done_t =
                        Tensor::<B, 1>::from_data(TensorData::new(not_done, [batch]), &device);
                    let critic_targets = reward_t + next_values.detach() * not_done_t * args.gamma;

                    let parts = compute_discrete_mpo_losses(
                        policy_logits,
                        target_action_weights,
                        values,
                        critic_targets,
                        &duals,
                        args.epsilon,
                        args.epsilon_policy,
                    );

                    let mut grads = parts.total_loss.backward();

                    let grads_actor = GradientsParams::from_module::<B, crate::models::Actor<B>>(
                        &mut grads,
                        &agent_online.actor,
                    );
                    let grads_critic =
                        GradientsParams::from_module::<B, crate::models::Critic<B>>(
                            &mut grads,
                            &agent_online.critic,
                        );
                    let grads_duals = GradientsParams::from_module::<B, MpoDuals<B>>(&mut grads, &duals);

                    agent_online.actor = actor_optim.step(args.actor_lr, agent_online.actor, grads_actor);
                    agent_online.critic =
                        critic_optim.step(args.critic_lr, agent_online.critic, grads_critic);
                    duals = dual_optim.step(args.dual_lr, duals, grads_duals);

                    soft_update_params::<Agent<B>, B>(&mut agent_target, &agent_online, args.tau);

                    last_mpo_total = parts
                        .total_loss
                        .to_data()
                        .to_vec::<f32>()
                        .ok()
                        .and_then(|v| v.into_iter().next())
                        .unwrap_or(0.0) as f64;
                }
            }

            if dist.rank == 0 && update % 10 == 0 {
                info!(
                    category = "TRAINER",
                    mode = "spo",
                    update = update,
                    replay_len = replay.len(),
                    mpo_total_loss = last_mpo_total,
                    "SPO update step"
                );
            }
        }

        if is_lead && args.eval_interval > 0 && update > 0 && update % args.eval_interval == 0 {
            if let Some(eval_pool) = eval_pool.as_ref() {
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

                info!(
                    category = "EVAL",
                    mode = "spo",
                    update = update,
                    episodes = eval_stats.episodes,
                    return_mean = eval_stats.mean_return as f64,
                    return_max = eval_stats.max_return as f64,
                    return_min = eval_stats.min_return as f64,
                    ep_len_mean = eval_stats.mean_ep_len as f64,
                    "SPO evaluation"
                );
            }
        }
    }

    Ok(())
}
