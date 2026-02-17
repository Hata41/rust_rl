use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

use crate::common::model::models::{Actor, Critic};

pub fn linear_decay_alpha(update: usize, num_updates: usize) -> f64 {
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
        let id = param.id;
        if let Some(grad) = self.grads.get::<Bk::InnerBackend, D>(id) {
            if let Ok(values) = grad.to_data().to_vec::<f32>() {
                self.sum_sq += values
                    .iter()
                    .map(|v| {
                        let x = (*v) as f64;
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
        let id = param.id;
        if let Some(grad) = self.grads.remove::<Bk::InnerBackend, D>(id) {
            self.grads
                .register::<Bk::InnerBackend, D>(id, grad.mul_scalar(self.scale));
        }
    }
}

pub fn clip_global_grad_norm<Bk: AutodiffBackend>(
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
