use crate::tensor::Tensor;

pub struct Linear {
    ws: Tensor,
    bs: Tensor,
}

impl super::module::Module for Linear {
    type Config = (i64, i64); // input-dim, output-dim

    fn new(vs: &mut super::var_store::VarStore, cfg: &Self::Config) -> Linear {
        let (in_dim, out_dim) = *cfg;
        Linear {
            ws: vs.zeros(&[in_dim, out_dim]),
            bs: vs.zeros(&[out_dim]),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.mm(&self.ws) + &self.bs
    }
}
