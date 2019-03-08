//! A linear fully-connected layer.
use crate::Tensor;
use std::borrow::Borrow;

/// A linear fully-connected layer.
#[derive(Debug)]
pub struct Linear {
    ws: Tensor,
    bs: Tensor,
}

impl Linear {
    pub fn new<'a, T: Borrow<super::Path<'a>>>(vs: T, in_dim: i64, out_dim: i64) -> Linear {
        let vs = vs.borrow();
        let bound = 1.0 / (in_dim as f64).sqrt();
        Linear {
            ws: vs.kaiming_uniform("weight", &[out_dim, in_dim]),
            bs: vs.uniform("bias", &[out_dim], -bound, bound),
        }
    }
}

impl super::module::Module for Linear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.matmul(&self.ws.tr()) + &self.bs
    }
}
