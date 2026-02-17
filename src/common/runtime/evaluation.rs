use anyhow::{bail, Result};

use crate::common::runtime::env::{AsyncEnvPool, StepOut};

#[derive(Debug, Clone, Copy)]
pub struct EvalStats {
    pub mean_return: f32,
    pub max_return: f32,
    pub min_return: f32,
    pub mean_ep_len: f32,
    pub max_ep_len: usize,
    pub min_ep_len: usize,
    pub episodes: usize,
}

pub fn run_eval_with_policy<F>(
    eval_pool: &AsyncEnvPool,
    eval_seed: u64,
    num_eval_envs: usize,
    num_eval_episodes: usize,
    mut select_actions: F,
) -> Result<EvalStats>
where
    F: FnMut(&AsyncEnvPool, &[StepOut]) -> Result<Vec<i32>>,
{
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

    let mut cur_steps = eval_pool.reset_all(Some(eval_seed))?;
    let mut ep_return = vec![0.0f32; num_eval_envs];
    let mut ep_len = vec![0usize; num_eval_envs];
    let mut completed_returns = Vec::<f32>::with_capacity(num_eval_episodes);
    let mut completed_lengths = Vec::<usize>::with_capacity(num_eval_episodes);
    let base_quota = num_eval_episodes / num_eval_envs;
    let extra = num_eval_episodes % num_eval_envs;
    let per_env_quota = (0..num_eval_envs)
        .map(|env_idx| base_quota + usize::from(env_idx < extra))
        .collect::<Vec<_>>();
    let mut per_env_count = vec![0usize; num_eval_envs];

    while completed_returns.len() < num_eval_episodes {
        let actions = select_actions(eval_pool, &cur_steps)?;
        if actions.len() != num_eval_envs {
            bail!(
                "evaluation policy produced {} actions for {} environments",
                actions.len(),
                num_eval_envs
            );
        }

        let next_steps = eval_pool.step_all(&actions)?;
        for env_idx in 0..num_eval_envs {
            ep_return[env_idx] += next_steps[env_idx].reward;
            ep_len[env_idx] += 1;

            if next_steps[env_idx].done {
                if completed_returns.len() < num_eval_episodes
                    && per_env_count[env_idx] < per_env_quota[env_idx]
                {
                    completed_returns.push(ep_return[env_idx]);
                    completed_lengths.push(ep_len[env_idx]);
                    per_env_count[env_idx] += 1;
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
