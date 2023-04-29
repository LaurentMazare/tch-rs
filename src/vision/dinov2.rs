//! DINOv2: Learning Robust Visual Features without Supervision
//! https://github.com/facebookresearch/dinov2
use crate::{nn, Kind, Result, Tensor};

#[derive(Debug)]
struct Attention {
    qkv: nn::Linear,
    proj: nn::Linear,
    num_heads: i64,
    scale: f64,
}

impl Attention {
    fn new(vs: nn::Path, dim: i64, num_heads: i64, qkv_bias: bool, proj_bias: bool) -> Self {
        let qkv_config = nn::LinearConfig { bias: qkv_bias, ..Default::default() };
        let proj_config = nn::LinearConfig { bias: proj_bias, ..Default::default() };
        let qkv = nn::linear(&vs / "qkv", dim, dim * 3, qkv_config);
        let proj = nn::linear(&vs / "proj", dim, dim, proj_config);
        let scale = 1. / ((dim / num_heads) as f64).sqrt();
        Self { qkv, proj, num_heads, scale }
    }
}

impl nn::Module for Attention {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (b, n, c) = xs.size3().unwrap();
        let qkv = self
            .qkv
            .forward(&xs)
            .reshape([b, n, 3, self.num_heads, c / self.num_heads])
            .permute([2, 0, 3, 1, 4]);
        let q = qkv.get(0) * self.scale;
        let k = qkv.get(1);
        let v = qkv.get(2);
        let attn = q.matmul(&k.transpose(-2, -1)).softmax(-1, Kind::Float);
        attn.matmul(&v).transpose(1, 2).reshape([b, n, c]).apply(&self.proj)
    }
}

#[derive(Debug)]
struct LayerScale {
    gamma: Tensor,
}

impl LayerScale {
    fn new(vs: nn::Path, dim: i64) -> Self {
        let gamma = vs.var("gamma", &[dim], nn::Init::Const(0.));
        Self { gamma }
    }
}

impl nn::Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs * &self.gamma
    }
}

#[derive(Debug)]
struct Mlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Mlp {
    fn new(vs: nn::Path, in_features: i64, hidden_features: i64, bias: bool) -> Self {
        let out_features = in_features;
        let config = nn::LinearConfig { bias, ..Default::default() };
        let fc1 = nn::linear(&vs / "fc1", in_features, hidden_features, config);
        let fc2 = nn::linear(&vs / "fc2", hidden_features, out_features, config);
        Self { fc1, fc2 }
    }
}

impl nn::Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1).gelu("none").apply(&self.fc2)
    }
}

#[derive(Debug)]
struct Block {
    norm1: nn::LayerNorm,
    attn: Attention,
    ls1: LayerScale,
    norm2: nn::LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl Block {
    fn new(vs: nn::Path, dim: i64, num_heads: i64) -> Self {
        let norm1 = nn::layer_norm(&vs / "norm1", vec![dim], Default::default());
        let attn = Attention::new(&vs / "attn", dim, num_heads, false, true);
        let ls1 = LayerScale::new(&vs / "ls1", dim);
        let norm2 = nn::layer_norm(&vs / "norm2", vec![dim], Default::default());
        let mlp = Mlp::new(&vs / "mlp", dim, dim * 4, true);
        let ls2 = LayerScale::new(&vs / "ls2", dim);
        Self { norm1, attn, ls1, norm2, mlp, ls2 }
    }
}

impl nn::Module for Block {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = xs + xs.apply(&self.norm1).apply(&self.attn).apply(&self.ls1);
        &xs + xs.apply(&self.norm2).apply(&self.mlp).apply(&self.ls2)
    }
}

pub struct DinoVisionTransformer {
    head: nn::Linear,
}

impl DinoVisionTransformer {
    pub fn new(vs: nn::Path) -> Self {
        let head = nn::linear(&vs / "head", 0, 0, Default::default());
        Self { head }
    }
}
