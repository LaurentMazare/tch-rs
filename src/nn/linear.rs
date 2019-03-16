//! A linear fully-connected layer.
use crate::Tensor;
use std::borrow::Borrow;

#[derive(Builder, Debug, Clone, Copy)]
#[builder(default)]
pub struct LinearConfig {
    pub ws_init: super::Init,
    pub bs_init: Option<super::Init>,
}

impl Default for LinearConfig {
    fn default() -> Self {
        LinearConfig {
            ws_init: super::Init::KaimingUniform,
            bs_init: None,
        }
    }
}

/// A linear fully-connected layer.
#[derive(Debug)]
pub struct Linear {
    pub ws: Tensor,
    pub bs: Tensor,
}

impl Linear {
    pub fn new<'a, T: Borrow<super::Path<'a>>>(
        vs: T,
        in_dim: i64,
        out_dim: i64,
        c: LinearConfig,
    ) -> Linear {
        let vs = vs.borrow();
        let bs_init = c.bs_init.unwrap_or_else(|| {
            let bound = 1.0 / (in_dim as f64).sqrt();
            super::Init::Uniform {
                lo: -bound,
                up: bound,
            }
        });
        Linear {
            ws: vs.var("weight", &[out_dim, in_dim], c.ws_init),
            bs: vs.var("bias", &[out_dim], bs_init),
        }
    }
}

impl super::module::Module for Linear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.matmul(&self.ws.tr()) + &self.bs
    }
}
