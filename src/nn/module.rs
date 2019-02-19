// TODO: add training/testing.
// TODO: add layer names when registering inner modules?
use crate::tensor::Tensor;

pub trait Module {
    type Config;
    fn new(vs: &mut super::var_store::VarStore, cfg: &Self::Config) -> Self;
    fn forward(&self, xs: &Tensor) -> Tensor;
}

impl Tensor {
    pub fn apply<M: Module>(&self, m: &M) -> Tensor {
        m.forward(&self)
    }
}
