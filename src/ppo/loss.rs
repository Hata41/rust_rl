use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{Distribution, Tensor};
use burn::tensor::Int;

pub fn masked_logits<Bk: burn::tensor::backend::Backend>(
    logits: Tensor<Bk, 2>,
    mask_f32: Tensor<Bk, 2>,
) -> Tensor<Bk, 2> {
    let one = Tensor::<Bk, 2>::ones_like(&mask_f32);
    logits + (mask_f32 - one) * 1.0e9
}

pub fn logprob_and_entropy<Bk: burn::tensor::backend::Backend>(
    logits: Tensor<Bk, 2>,
    mask_f32: Tensor<Bk, 2>,
    actions: Tensor<Bk, 1, Int>,
) -> (Tensor<Bk, 1>, Tensor<Bk, 1>) {
    let masked = masked_logits(logits, mask_f32);

    let lp = log_softmax(masked.clone(), 1);
    let probs = softmax(masked, 1);

    let bsz = actions.dims()[0];
    let idx2 = actions.reshape([bsz, 1]);
    let chosen_lp = lp.clone().gather(1, idx2).reshape([bsz]);

    let ent = (probs * lp).sum_dim(1).neg().reshape([bsz]);

    (chosen_lp, ent)
}

pub fn sample_actions_categorical<Bk: burn::tensor::backend::Backend>(
    logits: Tensor<Bk, 2>,
    mask_f32: Tensor<Bk, 2>,
    device: &Bk::Device,
) -> Tensor<Bk, 1, Int> {
    let masked = masked_logits(logits, mask_f32);
    let probs = softmax(masked, 1);
    let [batch_size, _action_dim] = probs.dims();

    let u = Tensor::<Bk, 2>::random(
        [batch_size, 1],
        Distribution::Uniform(1.0e-6, 1.0 - 1.0e-6),
        device,
    );
    let cdf = probs.cumsum(1);

    cdf.lower(u).int().sum_dim(1).reshape([batch_size])
}

pub struct PpoLossParts<Bk: burn::tensor::backend::Backend> {
    pub actor_loss: Tensor<Bk, 1>,
    pub value_loss: Tensor<Bk, 1>,
    pub entropy_mean: Tensor<Bk, 1>,
    pub total_loss: Tensor<Bk, 1>,
}

pub fn compute_ppo_losses<Bk: burn::tensor::backend::Backend>(
    new_lp: Tensor<Bk, 1>,
    old_lp_t: Tensor<Bk, 1>,
    adv_t: Tensor<Bk, 1>,
    ent: Tensor<Bk, 1>,
    v: Tensor<Bk, 1>,
    old_v_t: Tensor<Bk, 1>,
    tgt_t: Tensor<Bk, 1>,
    clip_eps: f32,
    ent_coef: f32,
    vf_coef: f32,
) -> PpoLossParts<Bk> {
    let ratio = (new_lp - old_lp_t).exp();
    let clipped = ratio.clone().clamp(1.0 - clip_eps, 1.0 + clip_eps);

    let surr1 = ratio * adv_t.clone();
    let surr2 = clipped * adv_t;

    let min_surr = (surr1.clone() + surr2.clone() - (surr1 - surr2).abs()) * 0.5;
    let policy_loss = min_surr.mean().neg();

    let entropy_mean = ent.mean();
    let actor_loss = policy_loss - entropy_mean.clone() * ent_coef;

    let v_clipped = old_v_t.clone() + (v.clone() - old_v_t).clamp(-clip_eps, clip_eps);
    let l1 = (v.clone() - tgt_t.clone()).powf_scalar(2.0);
    let l2 = (v_clipped - tgt_t).powf_scalar(2.0);

    let max_l = (l1.clone() + l2.clone() + (l1 - l2).abs()) * 0.5;
    let value_loss = max_l.mean() * (0.5 * vf_coef);

    let total_loss = actor_loss.clone() + value_loss.clone();

    PpoLossParts {
        actor_loss,
        value_loss,
        entropy_mean,
        total_loss,
    }
}