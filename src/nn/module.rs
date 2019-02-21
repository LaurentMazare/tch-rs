// TODO: add layer names when registering inner modules?
use crate::tensor::Tensor;

pub trait Module {
    fn forward(&self, xs: &Tensor) -> Tensor;
}

pub trait ModuleT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor;
}

impl<T> ModuleT for T
where
    T: Module,
{
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(&xs)
    }
}

impl Tensor {
    pub fn apply<M: Module>(&self, m: &M) -> Tensor {
        m.forward(&self)
    }
    pub fn apply_t<M: ModuleT>(&self, m: &M, train: bool) -> Tensor {
        m.forward_t(&self, train)
    }
}
