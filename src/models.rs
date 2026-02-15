use burn::module::Module;
use burn::nn::attention::{CrossAttention, CrossAttentionConfig, MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

#[derive(Module, Clone, Debug)]
pub struct Mlp<Bk: burn::tensor::backend::Backend> {
    l1: Linear<Bk>,
    l2: Linear<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> Mlp<Bk> {
    pub fn new(in_dim: usize, hidden: usize, device: &Bk::Device) -> Self {
        let l1 = LinearConfig::new(in_dim, hidden).init(device);
        let l2 = LinearConfig::new(hidden, hidden).init(device);
        Self { l1, l2 }
    }

    pub fn forward(&self, x: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let x = burn::tensor::activation::relu(self.l1.forward(x));
        burn::tensor::activation::relu(self.l2.forward(x))
    }
}

#[derive(Module, Clone, Debug)]
pub struct TransformerBlock<Bk: burn::tensor::backend::Backend> {
    ems_self_attn: MultiHeadAttention<Bk>,
    item_self_attn: MultiHeadAttention<Bk>,
    ems_to_item_attn: CrossAttention<Bk>,
    item_to_ems_attn: CrossAttention<Bk>,
    ems_ff1: Linear<Bk>,
    ems_ff2: Linear<Bk>,
    item_ff1: Linear<Bk>,
    item_ff2: Linear<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> TransformerBlock<Bk> {
    pub fn new(embed_dim: usize, num_heads: usize, dropout: f64, device: &Bk::Device) -> Self {
        let ems_self_attn = MultiHeadAttentionConfig::new(embed_dim, num_heads)
            .with_dropout(dropout)
            .init(device);
        let item_self_attn = MultiHeadAttentionConfig::new(embed_dim, num_heads)
            .with_dropout(dropout)
            .init(device);

        let head_dim = (embed_dim / num_heads).max(1);
        let ems_to_item_attn = CrossAttentionConfig::new(embed_dim, embed_dim, num_heads, num_heads, head_dim)
            .with_dropout(dropout)
            .init(device);
        let item_to_ems_attn = CrossAttentionConfig::new(embed_dim, embed_dim, num_heads, num_heads, head_dim)
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
        ems: Tensor<Bk, 3>,
        items: Tensor<Bk, 3>,
        ems_pad_mask: Tensor<Bk, 2, Bool>,
        items_pad_mask: Tensor<Bk, 2, Bool>,
    ) -> (Tensor<Bk, 3>, Tensor<Bk, 3>) {
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
            .forward(ems.clone(), items.clone(), Some(items_pad_mask));
        let items_cross = self
            .item_to_ems_attn
            .forward(items.clone(), ems.clone(), Some(ems_pad_mask));

        let ems = ems + ems_cross;
        let items = items + items_cross;

        let ems_ff = self
            .ems_ff2
            .forward(burn::tensor::activation::relu(self.ems_ff1.forward(ems.clone())));
        let items_ff = self
            .item_ff2
            .forward(burn::tensor::activation::relu(self.item_ff1.forward(items.clone())));

        (ems + ems_ff, items + items_ff)
    }
}

#[derive(Module, Clone, Debug)]
pub struct BinPackTorso<Bk: burn::tensor::backend::Backend> {
    ems_encoder: Linear<Bk>,
    item_encoder: Linear<Bk>,
    blocks: Vec<TransformerBlock<Bk>>,
}

impl<Bk: burn::tensor::backend::Backend> BinPackTorso<Bk> {
    pub fn new(embed_dim: usize, num_layers: usize, num_heads: usize, dropout: f64, device: &Bk::Device) -> Self {
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
        ems: Tensor<Bk, 3>,
        items: Tensor<Bk, 3>,
        ems_pad_mask: Tensor<Bk, 2, Bool>,
        items_pad_mask: Tensor<Bk, 2, Bool>,
    ) -> (Tensor<Bk, 3>, Tensor<Bk, 3>) {
        let mut ems_h = self.ems_encoder.forward(ems);
        let mut item_h = self.item_encoder.forward(items);

        for block in self.blocks.iter() {
            let (next_ems, next_items) = block.forward(
                ems_h,
                item_h,
                ems_pad_mask.clone(),
                items_pad_mask.clone(),
            );
            ems_h = next_ems;
            item_h = next_items;
        }

        (ems_h, item_h)
    }
}

#[derive(Module, Clone, Debug)]
pub struct BinPackActor<Bk: burn::tensor::backend::Backend> {
    torso: BinPackTorso<Bk>,
    ems_proj: Linear<Bk>,
    item_proj: Linear<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> BinPackActor<Bk> {
    pub fn new(hidden: usize, device: &Bk::Device) -> Self {
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
        ems: Tensor<Bk, 3>,
        items: Tensor<Bk, 3>,
        ems_pad_mask: Tensor<Bk, 2, Bool>,
        items_pad_mask: Tensor<Bk, 2, Bool>,
    ) -> Tensor<Bk, 2> {
        let (ems_h, item_h) = self
            .torso
            .forward(ems, items, ems_pad_mask, items_pad_mask);
        let ems_h = self.ems_proj.forward(ems_h);
        let item_h = self.item_proj.forward(item_h);

        let pair_scores = item_h.matmul(ems_h.swap_dims(1, 2));
        let [batch, item_count, ems_count] = pair_scores.dims();
        pair_scores.reshape([batch, item_count * ems_count])
    }
}

#[derive(Module, Clone, Debug)]
pub struct BinPackCritic<Bk: burn::tensor::backend::Backend> {
    torso: BinPackTorso<Bk>,
    value_mlp: Mlp<Bk>,
    value_head: Linear<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> BinPackCritic<Bk> {
    pub fn new(hidden: usize, device: &Bk::Device) -> Self {
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
        ems: Tensor<Bk, 3>,
        items: Tensor<Bk, 3>,
        ems_pad_mask: Tensor<Bk, 2, Bool>,
        items_pad_mask: Tensor<Bk, 2, Bool>,
        ems_valid_f32: Tensor<Bk, 2>,
        items_valid_f32: Tensor<Bk, 2>,
    ) -> Tensor<Bk, 2> {
        let (ems_h, item_h) = self
            .torso
            .forward(ems, items, ems_pad_mask, items_pad_mask);

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

#[derive(Module, Clone, Debug)]
pub struct Actor<Bk: burn::tensor::backend::Backend> {
    torso: Mlp<Bk>,
    head: Linear<Bk>,
    binpack: BinPackActor<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> Actor<Bk> {
    pub fn new(obs_dim: usize, hidden: usize, action_dim: usize, device: &Bk::Device) -> Self {
        let torso = Mlp::new(obs_dim, hidden, device);
        let head = LinearConfig::new(hidden, action_dim).init(device);
        let binpack = BinPackActor::new(hidden, device);
        Self { torso, head, binpack }
    }

    pub fn forward(&self, obs: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let h = self.torso.forward(obs);
        self.head.forward(h)
    }

    pub fn forward_binpack(
        &self,
        ems: Tensor<Bk, 3>,
        items: Tensor<Bk, 3>,
        ems_pad_mask: Tensor<Bk, 2, Bool>,
        items_pad_mask: Tensor<Bk, 2, Bool>,
    ) -> Tensor<Bk, 2> {
        self.binpack.forward(ems, items, ems_pad_mask, items_pad_mask)
    }

    pub fn forward_input(&self, input: ActorInput<Bk>) -> Tensor<Bk, 2> {
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

#[derive(Module, Clone, Debug)]
pub struct Critic<Bk: burn::tensor::backend::Backend> {
    torso: Mlp<Bk>,
    head: Linear<Bk>,
    binpack: BinPackCritic<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> Critic<Bk> {
    pub fn new(obs_dim: usize, hidden: usize, device: &Bk::Device) -> Self {
        let torso = Mlp::new(obs_dim, hidden, device);
        let head = LinearConfig::new(hidden, 1).init(device);
        let binpack = BinPackCritic::new(hidden, device);
        Self { torso, head, binpack }
    }

    pub fn forward(&self, obs: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let h = self.torso.forward(obs);
        self.head.forward(h)
    }

    pub fn forward_binpack(
        &self,
        ems: Tensor<Bk, 3>,
        items: Tensor<Bk, 3>,
        ems_pad_mask: Tensor<Bk, 2, Bool>,
        items_pad_mask: Tensor<Bk, 2, Bool>,
        ems_valid_f32: Tensor<Bk, 2>,
        items_valid_f32: Tensor<Bk, 2>,
    ) -> Tensor<Bk, 2> {
        self.binpack.forward(
            ems,
            items,
            ems_pad_mask,
            items_pad_mask,
            ems_valid_f32,
            items_valid_f32,
        )
    }

    pub fn forward_input(&self, input: CriticInput<Bk>) -> Tensor<Bk, 2> {
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

pub enum ActorInput<Bk: Backend> {
    Dense {
        obs: Tensor<Bk, 2>,
    },
    BinPack {
        ems: Tensor<Bk, 3>,
        items: Tensor<Bk, 3>,
        ems_pad_mask: Tensor<Bk, 2, Bool>,
        items_pad_mask: Tensor<Bk, 2, Bool>,
    },
}

pub enum CriticInput<Bk: Backend> {
    Dense {
        obs: Tensor<Bk, 2>,
    },
    BinPack {
        ems: Tensor<Bk, 3>,
        items: Tensor<Bk, 3>,
        ems_pad_mask: Tensor<Bk, 2, Bool>,
        items_pad_mask: Tensor<Bk, 2, Bool>,
        ems_valid_f32: Tensor<Bk, 2>,
        items_valid_f32: Tensor<Bk, 2>,
    },
}

pub enum PolicyInput<Bk: Backend> {
    Dense {
        obs: Tensor<Bk, 2>,
    },
    BinPack {
        ems: Tensor<Bk, 3>,
        items: Tensor<Bk, 3>,
        ems_pad_mask: Tensor<Bk, 2, Bool>,
        items_pad_mask: Tensor<Bk, 2, Bool>,
        ems_valid_f32: Tensor<Bk, 2>,
        items_valid_f32: Tensor<Bk, 2>,
    },
}

#[derive(Module, Clone, Debug)]
pub struct Agent<Bk: burn::tensor::backend::Backend> {
    pub actor: Actor<Bk>,
    pub critic: Critic<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> Agent<Bk> {
    pub fn new(obs_dim: usize, hidden: usize, action_dim: usize, device: &Bk::Device) -> Self {
        Self {
            actor: Actor::new(obs_dim, hidden, action_dim, device),
            critic: Critic::new(obs_dim, hidden, device),
        }
    }

    pub fn policy_value(&self, input: PolicyInput<Bk>) -> (Tensor<Bk, 2>, Tensor<Bk, 2>) {
        match input {
            PolicyInput::Dense { obs } => {
                let logits = self.actor.forward_input(ActorInput::Dense { obs: obs.clone() });
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

    pub fn actor_logits(&self, input: ActorInput<Bk>) -> Tensor<Bk, 2> {
        self.actor.forward_input(input)
    }

    pub fn critic_values(&self, input: CriticInput<Bk>) -> Tensor<Bk, 2> {
        self.critic.forward_input(input)
    }
}