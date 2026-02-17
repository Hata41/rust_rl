use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor, TensorData};
use rustpool::core::rl_env::RlEnv;
use rustpool::core::types::GenericObs;

use crate::common::config::{Args, ObservationAdapterKind};
use crate::common::model::models::{ActorInput, CriticInput, PolicyInput};
use crate::algorithms::ppo::buffer::{flatten_obs, flatten_obs_into, parse_binpack_obs};

pub trait ObservationAdapter<B: Backend>: Send + Sync {
    fn uses_binpack_architecture(&self) -> bool;

    fn infer_obs_dim(&self, first_obs: &GenericObs, args: &Args) -> usize;

    fn build_actor_input_batch(
        &self,
        obs_batch: &[&GenericObs],
        args: &Args,
        obs_dim: usize,
        device: &B::Device,
    ) -> Result<ActorInput<B>>;

    fn build_critic_input_batch(
        &self,
        obs_batch: &[&GenericObs],
        args: &Args,
        obs_dim: usize,
        device: &B::Device,
    ) -> Result<CriticInput<B>>;

    fn build_policy_input_batch(
        &self,
        obs_batch: &[&GenericObs],
        args: &Args,
        obs_dim: usize,
        device: &B::Device,
    ) -> Result<PolicyInput<B>>;

    fn build_policy_input_from_binpack_parts(
        &self,
        items_mb: Vec<f32>,
        ems_mb: Vec<f32>,
        items_valid_mb: Vec<bool>,
        ems_valid_mb: Vec<bool>,
        batch_size: usize,
        args: &Args,
        device: &B::Device,
    ) -> Result<PolicyInput<B>>;
}

#[derive(Default)]
pub struct DenseObservationAdapter;

#[derive(Default)]
pub struct BinPackObservationAdapter;

pub fn detect_env_model_from_metadata(env: &dyn RlEnv) -> ObservationAdapterKind {
    let keys = env
        .obs_keys()
        .into_iter()
        .map(|k| k.to_ascii_lowercase())
        .collect::<Vec<_>>();

    let has_items = keys.iter().any(|k| k.contains("item"));
    let has_ems = keys.iter().any(|k| k.contains("ems"));
    if has_items && has_ems {
        ObservationAdapterKind::Binpack
    } else {
        ObservationAdapterKind::Dense
    }
}

pub fn resolve_observation_adapter<B: Backend>(
    env: &dyn RlEnv,
    args: &Args,
) -> Box<dyn ObservationAdapter<B>> {
    let adapter_kind = args
        .observation_adapter
        .unwrap_or_else(|| detect_env_model_from_metadata(env));

    match adapter_kind {
        ObservationAdapterKind::Dense => Box::new(DenseObservationAdapter),
        ObservationAdapterKind::Binpack => Box::new(BinPackObservationAdapter),
    }
}

impl<B: Backend> ObservationAdapter<B> for DenseObservationAdapter {
    fn uses_binpack_architecture(&self) -> bool {
        false
    }

    fn infer_obs_dim(&self, first_obs: &GenericObs, _args: &Args) -> usize {
        flatten_obs(first_obs).len()
    }

    fn build_actor_input_batch(
        &self,
        obs_batch: &[&GenericObs],
        _args: &Args,
        obs_dim: usize,
        device: &B::Device,
    ) -> Result<ActorInput<B>> {
        let batch = obs_batch.len();
        let mut obs_flat = vec![0.0f32; batch * obs_dim];
        for (e, obs) in obs_batch.iter().enumerate() {
            let base = e * obs_dim;
            flatten_obs_into(obs, &mut obs_flat[base..base + obs_dim]);
        }
        let obs_t = Tensor::<B, 2>::from_data(TensorData::new(obs_flat, [batch, obs_dim]), device);
        Ok(ActorInput::Dense { obs: obs_t })
    }

    fn build_critic_input_batch(
        &self,
        obs_batch: &[&GenericObs],
        _args: &Args,
        obs_dim: usize,
        device: &B::Device,
    ) -> Result<CriticInput<B>> {
        let batch = obs_batch.len();
        let mut obs_flat = vec![0.0f32; batch * obs_dim];
        for (e, obs) in obs_batch.iter().enumerate() {
            let base = e * obs_dim;
            flatten_obs_into(obs, &mut obs_flat[base..base + obs_dim]);
        }
        let obs_t = Tensor::<B, 2>::from_data(TensorData::new(obs_flat, [batch, obs_dim]), device);
        Ok(CriticInput::Dense { obs: obs_t })
    }

    fn build_policy_input_batch(
        &self,
        obs_batch: &[&GenericObs],
        _args: &Args,
        obs_dim: usize,
        device: &B::Device,
    ) -> Result<PolicyInput<B>> {
        let batch = obs_batch.len();
        let mut obs_flat = vec![0.0f32; batch * obs_dim];
        for (e, obs) in obs_batch.iter().enumerate() {
            let base = e * obs_dim;
            flatten_obs_into(obs, &mut obs_flat[base..base + obs_dim]);
        }
        let obs_t = Tensor::<B, 2>::from_data(TensorData::new(obs_flat, [batch, obs_dim]), device);
        Ok(PolicyInput::Dense { obs: obs_t })
    }

    fn build_policy_input_from_binpack_parts(
        &self,
        _items_mb: Vec<f32>,
        _ems_mb: Vec<f32>,
        _items_valid_mb: Vec<bool>,
        _ems_valid_mb: Vec<bool>,
        _batch_size: usize,
        _args: &Args,
        _device: &B::Device,
    ) -> Result<PolicyInput<B>> {
        anyhow::bail!("dense observation adapter does not support binpack policy tensor construction")
    }
}

type BinPackBatch<B> = (
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 2, Bool>,
    Tensor<B, 2, Bool>,
    Tensor<B, 2>,
    Tensor<B, 2>,
);

fn build_binpack_batch_tensors<B: Backend>(
    obs_batch: &[&GenericObs],
    args: &Args,
    device: &B::Device,
) -> Result<BinPackBatch<B>> {
    let batch = obs_batch.len();
    let mut items = vec![0.0f32; batch * args.max_items * 3];
    let mut ems = vec![0.0f32; batch * args.max_ems * 6];
    let mut items_pad = vec![false; batch * args.max_items];
    let mut ems_pad = vec![false; batch * args.max_ems];
    let mut items_valid_f32 = vec![0.0f32; batch * args.max_items];
    let mut ems_valid_f32 = vec![0.0f32; batch * args.max_ems];

    for (row, obs) in obs_batch.iter().enumerate() {
        let parsed = parse_binpack_obs(obs, args.max_items, args.max_ems)?;

        let items_base = row * args.max_items * 3;
        let ems_base = row * args.max_ems * 6;
        let item_mask_base = row * args.max_items;
        let ems_mask_base = row * args.max_ems;

        items[items_base..items_base + args.max_items * 3].copy_from_slice(parsed.items);
        ems[ems_base..ems_base + args.max_ems * 6].copy_from_slice(parsed.ems);

        for i in 0..args.max_items {
            let valid = parsed.items_valid[i];
            items_pad[item_mask_base + i] = !valid;
            items_valid_f32[item_mask_base + i] = if valid { 1.0 } else { 0.0 };
        }
        for i in 0..args.max_ems {
            let valid = parsed.ems_valid[i];
            ems_pad[ems_mask_base + i] = !valid;
            ems_valid_f32[ems_mask_base + i] = if valid { 1.0 } else { 0.0 };
        }
    }

    let items_t =
        Tensor::<B, 3>::from_data(TensorData::new(items, [batch, args.max_items, 3]), device);
    let ems_t = Tensor::<B, 3>::from_data(TensorData::new(ems, [batch, args.max_ems, 6]), device);
    let items_pad_t = Tensor::<B, 2, Bool>::from_data(
        TensorData::new(items_pad, [batch, args.max_items]),
        device,
    );
    let ems_pad_t =
        Tensor::<B, 2, Bool>::from_data(TensorData::new(ems_pad, [batch, args.max_ems]), device);
    let items_valid_t = Tensor::<B, 2>::from_data(
        TensorData::new(items_valid_f32, [batch, args.max_items]),
        device,
    );
    let ems_valid_t = Tensor::<B, 2>::from_data(
        TensorData::new(ems_valid_f32, [batch, args.max_ems]),
        device,
    );

    Ok((
        items_t,
        ems_t,
        items_pad_t,
        ems_pad_t,
        items_valid_t,
        ems_valid_t,
    ))
}

impl<B: Backend> ObservationAdapter<B> for BinPackObservationAdapter {
    fn uses_binpack_architecture(&self) -> bool {
        true
    }

    fn infer_obs_dim(&self, _first_obs: &GenericObs, args: &Args) -> usize {
        args.max_items * 3 + args.max_ems * 6
    }

    fn build_actor_input_batch(
        &self,
        obs_batch: &[&GenericObs],
        args: &Args,
        _obs_dim: usize,
        device: &B::Device,
    ) -> Result<ActorInput<B>> {
        let (items_t, ems_t, items_pad_t, ems_pad_t, _, _) =
            build_binpack_batch_tensors::<B>(obs_batch, args, device)?;
        Ok(ActorInput::BinPack {
            ems: ems_t,
            items: items_t,
            ems_pad_mask: ems_pad_t,
            items_pad_mask: items_pad_t,
        })
    }

    fn build_critic_input_batch(
        &self,
        obs_batch: &[&GenericObs],
        args: &Args,
        _obs_dim: usize,
        device: &B::Device,
    ) -> Result<CriticInput<B>> {
        let (items_t, ems_t, items_pad_t, ems_pad_t, items_valid_t, ems_valid_t) =
            build_binpack_batch_tensors::<B>(obs_batch, args, device)?;
        Ok(CriticInput::BinPack {
            ems: ems_t,
            items: items_t,
            ems_pad_mask: ems_pad_t,
            items_pad_mask: items_pad_t,
            ems_valid_f32: ems_valid_t,
            items_valid_f32: items_valid_t,
        })
    }

    fn build_policy_input_batch(
        &self,
        obs_batch: &[&GenericObs],
        args: &Args,
        _obs_dim: usize,
        device: &B::Device,
    ) -> Result<PolicyInput<B>> {
        let (items_t, ems_t, items_pad_t, ems_pad_t, items_valid_t, ems_valid_t) =
            build_binpack_batch_tensors::<B>(obs_batch, args, device)?;
        Ok(PolicyInput::BinPack {
            ems: ems_t,
            items: items_t,
            ems_pad_mask: ems_pad_t,
            items_pad_mask: items_pad_t,
            ems_valid_f32: ems_valid_t,
            items_valid_f32: items_valid_t,
        })
    }

    fn build_policy_input_from_binpack_parts(
        &self,
        items_mb: Vec<f32>,
        ems_mb: Vec<f32>,
        items_valid_mb: Vec<bool>,
        ems_valid_mb: Vec<bool>,
        batch_size: usize,
        args: &Args,
        device: &B::Device,
    ) -> Result<PolicyInput<B>> {
    let items_t = Tensor::<B, 3>::from_data(
        TensorData::new(items_mb, [batch_size, args.max_items, 3]),
        device,
    );
    let ems_t = Tensor::<B, 3>::from_data(
        TensorData::new(ems_mb, [batch_size, args.max_ems, 6]),
        device,
    );

    let mut items_pad_mb = vec![false; batch_size * args.max_items];
    let mut ems_pad_mb = vec![false; batch_size * args.max_ems];
    let mut items_valid_f32_mb = vec![0.0f32; batch_size * args.max_items];
    let mut ems_valid_f32_mb = vec![0.0f32; batch_size * args.max_ems];

    for i in 0..(batch_size * args.max_items) {
        let valid = items_valid_mb[i];
        items_pad_mb[i] = !valid;
        items_valid_f32_mb[i] = if valid { 1.0 } else { 0.0 };
    }
    for i in 0..(batch_size * args.max_ems) {
        let valid = ems_valid_mb[i];
        ems_pad_mb[i] = !valid;
        ems_valid_f32_mb[i] = if valid { 1.0 } else { 0.0 };
    }

    let items_pad_t = Tensor::<B, 2, Bool>::from_data(
        TensorData::new(items_pad_mb, [batch_size, args.max_items]),
        device,
    );
    let ems_pad_t = Tensor::<B, 2, Bool>::from_data(
        TensorData::new(ems_pad_mb, [batch_size, args.max_ems]),
        device,
    );
    let items_valid_t = Tensor::<B, 2>::from_data(
        TensorData::new(items_valid_f32_mb, [batch_size, args.max_items]),
        device,
    );
    let ems_valid_t = Tensor::<B, 2>::from_data(
        TensorData::new(ems_valid_f32_mb, [batch_size, args.max_ems]),
        device,
    );

        Ok(PolicyInput::BinPack {
        ems: ems_t,
        items: items_t,
        ems_pad_mask: ems_pad_t,
        items_pad_mask: items_pad_t,
        ems_valid_f32: ems_valid_t,
        items_valid_f32: items_valid_t,
        })
    }
}
