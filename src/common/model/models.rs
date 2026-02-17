use burn::module::Module;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

#[derive(Module, Debug)]
pub struct Mlp<B: burn::tensor::backend::Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> Mlp<B> {
    pub fn new(in_dim: usize, hidden: usize, device: &B::Device) -> Self {
        let l1 = LinearConfig::new(in_dim, hidden).init(device);
        let l2 = LinearConfig::new(hidden, hidden).init(device);
        Self { l1, l2 }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = burn::tensor::activation::relu(self.l1.forward(x));
        burn::tensor::activation::relu(self.l2.forward(x))
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: burn::tensor::backend::Backend> {
    ems_self_attn: MultiHeadAttention<B>,
    item_self_attn: MultiHeadAttention<B>,
    ems_to_item_attn: MultiHeadAttention<B>,
    item_to_ems_attn: MultiHeadAttention<B>,
    ems_ff1: Linear<B>,
    ems_ff2: Linear<B>,
    item_ff1: Linear<B>,
    item_ff2: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> TransformerBlock<B> {
    pub fn new(embed_dim: usize, num_heads: usize, dropout: f64, device: &B::Device) -> Self {
        let ems_self_attn = MultiHeadAttentionConfig::new(embed_dim, num_heads)
            .with_dropout(dropout)
            .init(device);
        let item_self_attn = MultiHeadAttentionConfig::new(embed_dim, num_heads)
            .with_dropout(dropout)
            .init(device);

        let ems_to_item_attn = MultiHeadAttentionConfig::new(embed_dim, num_heads)
            .with_dropout(dropout)
            .init(device);
        let item_to_ems_attn = MultiHeadAttentionConfig::new(embed_dim, num_heads)
            .with_dropout(dropout)
            .init(device);

        let ff_dim = embed_dim * 4;
        let ems_ff1 = LinearConfig::new(embed_dim, ff_dim).init(device);
        let ems_ff2 = LinearConfig::new(ff_dim, embed_dim).init(device);
        let item_ff1 = LinearConfig::new(embed_dim, ff_dim).init(device);
        let item_ff2 = LinearConfig::new(ff_dim, embed_dim).init(device);

        Self {
            ems_self_attn,
            item_self_attn,
            ems_to_item_attn,
            item_to_ems_attn,
            ems_ff1,
            ems_ff2,
            item_ff1,
            item_ff2,
        }
    }

    pub fn forward(
        &self,
        ems: Tensor<B, 3>,
        items: Tensor<B, 3>,
        ems_pad_mask: Tensor<B, 2, Bool>,
        items_pad_mask: Tensor<B, 2, Bool>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let ems_self = self
            .ems_self_attn
            .forward(MhaInput::self_attn(ems.clone()).mask_pad(ems_pad_mask.clone()))
            .context;
        let items_self = self
            .item_self_attn
            .forward(MhaInput::self_attn(items.clone()).mask_pad(items_pad_mask.clone()))
            .context;

        let ems = ems + ems_self;
        let items = items + items_self;

        let ems_cross = self
            .ems_to_item_attn
            .forward(
                MhaInput::new(ems.clone(), items.clone(), items.clone()).mask_pad(items_pad_mask),
            )
            .context;
        let items_cross = self
            .item_to_ems_attn
            .forward(MhaInput::new(items.clone(), ems.clone(), ems.clone()).mask_pad(ems_pad_mask))
            .context;

        let ems = ems + ems_cross;
        let items = items + items_cross;

        let ems_ff = self.ems_ff2.forward(burn::tensor::activation::relu(
            self.ems_ff1.forward(ems.clone()),
        ));
        let items_ff = self.item_ff2.forward(burn::tensor::activation::relu(
            self.item_ff1.forward(items.clone()),
        ));

        (ems + ems_ff, items + items_ff)
    }
}

#[derive(Module, Debug)]
pub struct BinPackTorso<B: burn::tensor::backend::Backend> {
    ems_encoder: Linear<B>,
    item_encoder: Linear<B>,
    blocks: Vec<TransformerBlock<B>>,
}

impl<B: burn::tensor::backend::Backend> BinPackTorso<B> {
    pub fn new(
        embed_dim: usize,
        num_layers: usize,
        num_heads: usize,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let ems_encoder = LinearConfig::new(6, embed_dim).init(device);
        let item_encoder = LinearConfig::new(3, embed_dim).init(device);
        let blocks = (0..num_layers)
            .map(|_| TransformerBlock::new(embed_dim, num_heads, dropout, device))
            .collect();

        Self {
            ems_encoder,
            item_encoder,
            blocks,
        }
    }

    pub fn forward(
        &self,
        ems: Tensor<B, 3>,
        items: Tensor<B, 3>,
        ems_pad_mask: Tensor<B, 2, Bool>,
        items_pad_mask: Tensor<B, 2, Bool>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mut ems_h = self.ems_encoder.forward(ems);
        let mut item_h = self.item_encoder.forward(items);

        for block in self.blocks.iter() {
            let (next_ems, next_items) =
                block.forward(ems_h, item_h, ems_pad_mask.clone(), items_pad_mask.clone());
            ems_h = next_ems;
            item_h = next_items;
        }

        (ems_h, item_h)
    }
}

#[derive(Module, Debug)]
pub struct BinPackActor<B: burn::tensor::backend::Backend> {
    torso: BinPackTorso<B>,
    ems_proj: Linear<B>,
    item_proj: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> BinPackActor<B> {
    pub fn new(hidden: usize, device: &B::Device) -> Self {
        let torso = BinPackTorso::new(hidden, 2, 4, 0.0, device);
        let ems_proj = LinearConfig::new(hidden, hidden).init(device);
        let item_proj = LinearConfig::new(hidden, hidden).init(device);
        Self {
            torso,
            ems_proj,
            item_proj,
        }
    }

    pub fn forward(
        &self,
        ems: Tensor<B, 3>,
        items: Tensor<B, 3>,
        ems_pad_mask: Tensor<B, 2, Bool>,
        items_pad_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 2> {
        let (ems_h, item_h) = self.torso.forward(ems, items, ems_pad_mask, items_pad_mask);
        let ems_h = self.ems_proj.forward(ems_h);
        let item_h = self.item_proj.forward(item_h);

        let pair_scores = item_h.matmul(ems_h.swap_dims(1, 2));
        let [batch, item_count, ems_count] = pair_scores.dims();
        pair_scores.reshape([batch, item_count * ems_count])
    }
}

#[derive(Module, Debug)]
pub struct BinPackCritic<B: burn::tensor::backend::Backend> {
    torso: BinPackTorso<B>,
    value_mlp: Mlp<B>,
    value_head: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> BinPackCritic<B> {
    pub fn new(hidden: usize, device: &B::Device) -> Self {
        let torso = BinPackTorso::new(hidden, 2, 4, 0.0, device);
        let value_mlp = Mlp::new(hidden * 2, hidden, device);
        let value_head = LinearConfig::new(hidden, 1).init(device);
        Self {
            torso,
            value_mlp,
            value_head,
        }
    }

    pub fn forward(
        &self,
        ems: Tensor<B, 3>,
        items: Tensor<B, 3>,
        ems_pad_mask: Tensor<B, 2, Bool>,
        items_pad_mask: Tensor<B, 2, Bool>,
        ems_valid_f32: Tensor<B, 2>,
        items_valid_f32: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let (ems_h, item_h) = self.torso.forward(ems, items, ems_pad_mask, items_pad_mask);

        let [batch, ems_len, hidden] = ems_h.dims();
        let [_, item_len, _] = item_h.dims();
        let ems_mask = ems_valid_f32.reshape([batch, ems_len, 1]);
        let item_mask = items_valid_f32.reshape([batch, item_len, 1]);

        let ems_pooled = (ems_h * ems_mask).sum_dim(1).reshape([batch, hidden]);
        let items_pooled = (item_h * item_mask).sum_dim(1).reshape([batch, hidden]);
        let joined = Tensor::cat(vec![ems_pooled, items_pooled], 1);

        let h = self.value_mlp.forward(joined);
        self.value_head.forward(h)
    }
}

#[derive(Module, Debug)]
pub struct DenseActor<B: burn::tensor::backend::Backend> {
    torso: Mlp<B>,
    head: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> DenseActor<B> {
    pub fn new(obs_dim: usize, hidden: usize, action_dim: usize, device: &B::Device) -> Self {
        let torso = Mlp::new(obs_dim, hidden, device);
        let head = LinearConfig::new(hidden, action_dim).init(device);
        Self { torso, head }
    }

    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.torso.forward(obs);
        self.head.forward(h)
    }
}

#[derive(Module, Debug)]
pub struct Actor<B: burn::tensor::backend::Backend> {
    dense: Option<DenseActor<B>>,
    binpack: Option<BinPackActor<B>>,
}

impl<B: burn::tensor::backend::Backend> Actor<B> {
    pub fn new(
        obs_dim: usize,
        hidden: usize,
        action_dim: usize,
        use_binpack_architecture: bool,
        device: &B::Device,
    ) -> Self {
        if use_binpack_architecture {
            Self {
                dense: None,
                binpack: Some(BinPackActor::new(hidden, device)),
            }
        } else {
            Self {
                dense: Some(DenseActor::new(obs_dim, hidden, action_dim, device)),
                binpack: None,
            }
        }
    }

    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.dense
            .as_ref()
            .expect("dense actor requested but dense architecture is disabled")
            .forward(obs)
    }

    pub fn forward_binpack(
        &self,
        ems: Tensor<B, 3>,
        items: Tensor<B, 3>,
        ems_pad_mask: Tensor<B, 2, Bool>,
        items_pad_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 2> {
        self.binpack
            .as_ref()
            .expect("binpack actor requested but binpack architecture is disabled")
            .forward(ems, items, ems_pad_mask, items_pad_mask)
    }

    pub fn forward_input(&self, input: ActorInput<B>) -> Tensor<B, 2> {
        match input {
            ActorInput::Dense { obs } => self.forward(obs),
            ActorInput::BinPack {
                ems,
                items,
                ems_pad_mask,
                items_pad_mask,
            } => self.forward_binpack(ems, items, ems_pad_mask, items_pad_mask),
        }
    }
}

#[derive(Module, Debug)]
pub struct DenseCritic<B: burn::tensor::backend::Backend> {
    torso: Mlp<B>,
    head: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> DenseCritic<B> {
    pub fn new(obs_dim: usize, hidden: usize, device: &B::Device) -> Self {
        let torso = Mlp::new(obs_dim, hidden, device);
        let head = LinearConfig::new(hidden, 1).init(device);
        Self { torso, head }
    }

    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.torso.forward(obs);
        self.head.forward(h)
    }
}

#[derive(Module, Debug)]
pub struct Critic<B: burn::tensor::backend::Backend> {
    dense: Option<DenseCritic<B>>,
    binpack: Option<BinPackCritic<B>>,
}

impl<B: burn::tensor::backend::Backend> Critic<B> {
    pub fn new(obs_dim: usize, hidden: usize, use_binpack_architecture: bool, device: &B::Device) -> Self {
        if use_binpack_architecture {
            Self {
                dense: None,
                binpack: Some(BinPackCritic::new(hidden, device)),
            }
        } else {
            Self {
                dense: Some(DenseCritic::new(obs_dim, hidden, device)),
                binpack: None,
            }
        }
    }

    pub fn forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.dense
            .as_ref()
            .expect("dense critic requested but dense architecture is disabled")
            .forward(obs)
    }

    pub fn forward_binpack(
        &self,
        ems: Tensor<B, 3>,
        items: Tensor<B, 3>,
        ems_pad_mask: Tensor<B, 2, Bool>,
        items_pad_mask: Tensor<B, 2, Bool>,
        ems_valid_f32: Tensor<B, 2>,
        items_valid_f32: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        self.binpack
            .as_ref()
            .expect("binpack critic requested but binpack architecture is disabled")
            .forward(
            ems,
            items,
            ems_pad_mask,
            items_pad_mask,
            ems_valid_f32,
            items_valid_f32,
            )
    }

    pub fn forward_input(&self, input: CriticInput<B>) -> Tensor<B, 2> {
        match input {
            CriticInput::Dense { obs } => self.forward(obs),
            CriticInput::BinPack {
                ems,
                items,
                ems_pad_mask,
                items_pad_mask,
                ems_valid_f32,
                items_valid_f32,
            } => self.forward_binpack(
                ems,
                items,
                ems_pad_mask,
                items_pad_mask,
                ems_valid_f32,
                items_valid_f32,
            ),
        }
    }
}

pub enum ActorInput<B: Backend> {
    Dense {
        obs: Tensor<B, 2>,
    },
    BinPack {
        ems: Tensor<B, 3>,
        items: Tensor<B, 3>,
        ems_pad_mask: Tensor<B, 2, Bool>,
        items_pad_mask: Tensor<B, 2, Bool>,
    },
}

pub enum CriticInput<B: Backend> {
    Dense {
        obs: Tensor<B, 2>,
    },
    BinPack {
        ems: Tensor<B, 3>,
        items: Tensor<B, 3>,
        ems_pad_mask: Tensor<B, 2, Bool>,
        items_pad_mask: Tensor<B, 2, Bool>,
        ems_valid_f32: Tensor<B, 2>,
        items_valid_f32: Tensor<B, 2>,
    },
}

pub enum PolicyInput<B: Backend> {
    Dense {
        obs: Tensor<B, 2>,
    },
    BinPack {
        ems: Tensor<B, 3>,
        items: Tensor<B, 3>,
        ems_pad_mask: Tensor<B, 2, Bool>,
        items_pad_mask: Tensor<B, 2, Bool>,
        ems_valid_f32: Tensor<B, 2>,
        items_valid_f32: Tensor<B, 2>,
    },
}

#[derive(Module, Debug)]
pub struct Agent<B: burn::tensor::backend::Backend> {
    pub actor: Actor<B>,
    pub critic: Critic<B>,
}

impl<B: burn::tensor::backend::Backend> Agent<B> {
    pub fn new(
        obs_dim: usize,
        hidden: usize,
        action_dim: usize,
        use_binpack_architecture: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            actor: Actor::new(
                obs_dim,
                hidden,
                action_dim,
                use_binpack_architecture,
                device,
            ),
            critic: Critic::new(obs_dim, hidden, use_binpack_architecture, device),
        }
    }

    pub fn policy_value(&self, input: PolicyInput<B>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        match input {
            PolicyInput::Dense { obs } => {
                let logits = self
                    .actor
                    .forward_input(ActorInput::Dense { obs: obs.clone() });
                let values = self.critic.forward_input(CriticInput::Dense { obs });
                (logits, values)
            }
            PolicyInput::BinPack {
                ems,
                items,
                ems_pad_mask,
                items_pad_mask,
                ems_valid_f32,
                items_valid_f32,
            } => {
                let logits = self.actor.forward_input(ActorInput::BinPack {
                    ems: ems.clone(),
                    items: items.clone(),
                    ems_pad_mask: ems_pad_mask.clone(),
                    items_pad_mask: items_pad_mask.clone(),
                });
                let values = self.critic.forward_input(CriticInput::BinPack {
                    ems,
                    items,
                    ems_pad_mask,
                    items_pad_mask,
                    ems_valid_f32,
                    items_valid_f32,
                });
                (logits, values)
            }
        }
    }

    pub fn actor_logits(&self, input: ActorInput<B>) -> Tensor<B, 2> {
        self.actor.forward_input(input)
    }

    pub fn critic_values(&self, input: CriticInput<B>) -> Tensor<B, 2> {
        self.critic.forward_input(input)
    }
}
