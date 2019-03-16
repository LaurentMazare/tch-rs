//! A batch-normalization layer.
use crate::Tensor;
use std::borrow::Borrow;

#[derive(Builder, Debug, Clone, Copy)]
#[builder(default)]
pub struct BatchNorm2DConfig {
    pub cudnn_enabled: bool,
    pub eps: f64,
    pub momentum: f64,
    pub ws_init: super::Init,
    pub bs_init: super::Init,
}

impl Default for BatchNorm2DConfig {
    fn default() -> Self {
        BatchNorm2DConfig {
            cudnn_enabled: true,
            eps: 1e-5,
            momentum: 0.1,
            ws_init: super::Init::Uniform { lo: 0., up: 1. },
            bs_init: super::Init::Const(0.),
        }
    }
}

/// A batch-normalization layer.
#[derive(Debug)]
pub struct BatchNorm2D {
    config: BatchNorm2DConfig,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub ws: Tensor,
    pub bs: Tensor,
}

impl BatchNorm2D {
    pub fn new<'a, T: Borrow<super::Path<'a>>>(
        vs: T,
        out_dim: i64,
        config: BatchNorm2DConfig,
    ) -> BatchNorm2D {
        let vs = vs.borrow();
        BatchNorm2D {
            config,
            running_mean: vs.zeros_no_train("running_mean", &[out_dim]),
            running_var: vs.ones_no_train("running_var", &[out_dim]),
            ws: vs.var("weight", &[out_dim], config.ws_init),
            bs: vs.var("bias", &[out_dim], config.bs_init),
        }
    }
}

impl super::module::ModuleT for BatchNorm2D {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        Tensor::batch_norm(
            xs,
            Some(&self.ws),
            Some(&self.bs),
            Some(&self.running_mean),
            Some(&self.running_var),
            train,
            self.config.momentum,
            self.config.eps,
            self.config.cudnn_enabled,
        )
    }
}
