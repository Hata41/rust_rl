use burn::module::Module;
use burn::tensor::activation::{log_softmax, softplus};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

const MPO_FLOAT_EPSILON: f32 = 1.0e-8;
const MIN_LOG_DUAL: f32 = -18.0;

#[derive(Module, Debug)]
pub struct MpoDuals<B: Backend> {
    pub log_temperature: Tensor<B, 1>,
    pub log_alpha: Tensor<B, 1>,
}

impl<B: Backend> MpoDuals<B> {
    pub fn new(init_log_temperature: f32, init_log_alpha: f32, device: &B::Device) -> Self {
        let log_temperature =
            Tensor::<B, 1>::from_data(TensorData::new(vec![init_log_temperature], [1]), device);
        let log_alpha =
            Tensor::<B, 1>::from_data(TensorData::new(vec![init_log_alpha], [1]), device);
        Self {
            log_temperature,
            log_alpha,
        }
    }

    pub fn temperature(&self) -> Tensor<B, 1> {
        softplus(
            self.log_temperature
                .clone()
                .clamp(MIN_LOG_DUAL, f32::INFINITY),
            1.0,
        ) + MPO_FLOAT_EPSILON
    }

    pub fn alpha(&self) -> Tensor<B, 1> {
        softplus(
            self.log_alpha.clone().clamp(MIN_LOG_DUAL, f32::INFINITY),
            1.0,
        ) + MPO_FLOAT_EPSILON
    }
}

pub struct MpoLossParts<B: Backend> {
    pub actor_loss: Tensor<B, 1>,
    pub critic_loss: Tensor<B, 1>,
    pub loss_kl_penalty: Tensor<B, 1>,
    pub loss_temperature: Tensor<B, 1>,
    pub loss_alpha: Tensor<B, 1>,
    pub total_loss: Tensor<B, 1>,
}

pub fn compute_discrete_mpo_losses<B: Backend>(
    policy_logits: Tensor<B, 2>,
    target_action_weights: Tensor<B, 2>,
    sampled_advantages: Tensor<B, 2>,
    critic_values: Tensor<B, 1>,
    critic_targets: Tensor<B, 1>,
    duals: &MpoDuals<B>,
    epsilon: f32,
    epsilon_policy: f32,
) -> MpoLossParts<B> {
    let device = policy_logits.device();

    let temperature = duals.temperature().clamp(1.0e-6, 1.0e6);
    let alpha = duals.alpha().clamp(1.0e-6, 1.0e6);

    let mut target_dist = target_action_weights.detach();
    target_dist = target_dist.clamp(1.0e-8, 1.0);
    let target_sum = target_dist.clone().sum_dim(1).reshape([-1, 1]);
    let target_dist = target_dist / target_sum;
    let log_target_dist = target_dist.clone().log();

    let log_policy = log_softmax(policy_logits.clone(), 1);

    let actor_loss = (target_dist.clone().detach() * log_policy.clone())
        .sum_dim(1)
        .mean()
        .neg();

    let critic_loss = (critic_values - critic_targets.detach())
        .powf_scalar(2.0)
        .mean();

    let kl_tp = (target_dist.clone().detach() * (log_target_dist.clone().detach() - log_policy))
        .sum_dim(1)
        .mean();

    let eps_t = Tensor::<B, 1>::from_data(TensorData::new(vec![epsilon], [1]), &device);
    let eps_policy_t =
        Tensor::<B, 1>::from_data(TensorData::new(vec![epsilon_policy], [1]), &device);

    let [_, num_particles] = sampled_advantages.dims();
    let tempered_adv = sampled_advantages.detach() / temperature.clone().reshape([1, 1]);
    let q_logsumexp = tempered_adv.exp().sum_dim(1).log();
    let q_logsumexp_mean = q_logsumexp.mean();
    let log_num_actions = Tensor::<B, 1>::from_data(
        TensorData::new(vec![(num_particles as f32).ln()], [1]),
        &device,
    );
    let loss_temperature =
        (temperature.clone() * (eps_t + q_logsumexp_mean - log_num_actions)).mean();
    let loss_kl_penalty = (alpha.clone().detach() * kl_tp.clone()).mean();
    let loss_alpha = (alpha.clone() * (eps_policy_t - kl_tp.detach())).mean();

    let total_loss = actor_loss.clone()
        + critic_loss.clone()
        + loss_kl_penalty.clone()
        + loss_temperature.clone()
        + loss_alpha.clone();

    MpoLossParts {
        actor_loss,
        critic_loss,
        loss_kl_penalty,
        loss_temperature,
        loss_alpha,
        total_loss,
    }
}
