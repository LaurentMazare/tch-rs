use crate::tensor::Tensor;

pub struct Linear {
    ws: Tensor,
    bs: Tensor,
}

impl Linear {
    pub fn new(vs: &super::var_store::Path, in_dim: i64, out_dim: i64) -> Linear {
        let bound = 1.0 / (in_dim as f64).sqrt();
        Linear {
            ws: vs.kaiming_uniform("weight", &[out_dim, in_dim]),
            bs: vs.uniform("bias", &[out_dim], -bound, bound),
        }
    }
}

impl super::module::Module for Linear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.mm(&self.ws.tr()) + &self.bs
    }
}
