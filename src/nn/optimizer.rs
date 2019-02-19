use super::c_optimizer::COptimizer;
use super::var_store::VarStore;
use crate::tensor::Tensor;

pub struct Optimizer {
    opt: COptimizer,
}

impl Optimizer {
    pub fn sgd(vs: &VarStore, lr: f64) -> Optimizer {
        let mut opt = COptimizer::sgd(lr, 0., 0., 0., false);
        opt.add_parameters(vs.variables());
        Optimizer { opt }
    }

    pub fn adam(vs: &VarStore, lr: f64) -> Optimizer {
        let mut opt = COptimizer::adam(lr, 0.9, 0.999, 0.);
        opt.add_parameters(vs.variables());
        Optimizer { opt }
    }

    pub fn zero_grad(&self) {
        self.opt.zero_grad()
    }

    pub fn step(&self) {
        self.opt.step()
    }

    pub fn backward_step(&self, loss: &Tensor) {
        self.zero_grad();
        loss.backward();
        self.step();
    }
}
