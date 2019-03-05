use std::borrow::Borrow;

use crate::Tensor;

#[derive(Builder, Debug, Clone, Copy)]
#[builder(setter(into))]
pub struct BatchNorm2DConfig {
    cudnn_enabled: bool,
    eps: f64,
    momentum: f64,
}

impl Default for BatchNorm2DConfig {
    fn default() -> Self {
        BatchNorm2DConfig {
            cudnn_enabled: true,
            eps: 1e-5,
            momentum: 0.1,
        }
    }
}

/// A batch-normalization layer.
#[derive(Debug)]
pub struct BatchNorm2D {
    config: BatchNorm2DConfig,
    running_mean: Tensor,
    running_var: Tensor,
    ws: Tensor,
    bs: Tensor,
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
            ws: vs.uniform("weight", &[out_dim], 0.0, 1.0),
            bs: vs.zeros("bias", &[out_dim]),
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
