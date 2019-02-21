use crate::tensor::Tensor;

pub struct Linear {
    ws: Tensor,
    bs: Tensor,
}

impl super::module::Module for Linear {
    type Config = (i64, i64); // input-dim, output-dim

    fn new(vs: &mut super::var_store::VarStore, cfg: &Self::Config) -> Linear {
        let (in_dim, out_dim) = *cfg;
        let bound = 1.0 / (in_dim as f64).sqrt();
        Linear {
            ws: vs.kaiming_uniform(&[out_dim, in_dim]),
            bs: vs.uniform(&[out_dim], -bound, bound),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.mm(&self.ws.tr()) + &self.bs
    }
}
