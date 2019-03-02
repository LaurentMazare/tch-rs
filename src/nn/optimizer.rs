use super::c_optimizer::COptimizer;
use super::var_store::VarStore;
use crate::{Scalar, Tensor};

pub struct Optimizer {
    opt: COptimizer,
    trainable_variables: Vec<Tensor>,
}

pub struct Sgd {
    momentum: f64,
    dampening: f64,
    wd: f64,
    nesterov: bool,
}

impl Default for Sgd {
    fn default() -> Self {
        Sgd {
            momentum: 0.,
            dampening: 0.,
            wd: 0.,
            nesterov: false,
        }
    }
}

pub struct Adam {
    beta1: f64,
    beta2: f64,
    wd: f64,
}

impl Default for Adam {
    fn default() -> Self {
        Adam {
            beta1: 0.9,
            beta2: 0.999,
            wd: 0.,
        }
    }
}

pub struct RmsProp {
    alpha: f64,
    eps: f64,
    wd: f64,
    momentum: f64,
    centered: bool,
}

impl Default for RmsProp {
    fn default() -> Self {
        RmsProp {
            alpha: 0.99,
            eps: 1e-8,
            wd: 0.,
            momentum: 0.,
            centered: false,
        }
    }
}

impl Optimizer {
    pub fn sgd(vs: &VarStore, lr: f64, s: Sgd) -> Optimizer {
        let mut opt = COptimizer::sgd(lr, s.momentum, s.dampening, s.wd, s.nesterov);
        let trainable_variables = vs.trainable_variables();
        opt.add_parameters(&trainable_variables);
        Optimizer { opt, trainable_variables }
    }

    pub fn adam(vs: &VarStore, lr: f64, a: Adam) -> Optimizer {
        let mut opt = COptimizer::adam(lr, a.beta1, a.beta2, a.wd);
        let trainable_variables = vs.trainable_variables();
        opt.add_parameters(&trainable_variables);
        Optimizer { opt, trainable_variables }
    }

    pub fn rms_prop(vs: &VarStore, lr: f64, r: RmsProp) -> Optimizer {
        let mut opt = COptimizer::rms_prop(lr, r.alpha, r.eps, r.wd, r.momentum, r.centered);
        let trainable_variables = vs.trainable_variables();
        opt.add_parameters(&trainable_variables);
        Optimizer { opt, trainable_variables }
    }

    pub fn zero_grad(&self) {
        self.opt.zero_grad()
    }

    pub fn clip_grad_value(&self, max: f64) {
        for tensor in self.trainable_variables.iter() {
            let _t = tensor.grad().clamp_(&Scalar::float(-max), &Scalar::float(max));
        }
    }

    pub fn step(&self) {
        self.opt.step()
    }

    pub fn backward_step(&self, loss: &Tensor) {
        self.zero_grad();
        loss.backward();
        self.step();
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.opt.set_learning_rate(lr)
    }

    pub fn set_momentum(&mut self, m: f64) {
        self.opt.set_momentum(m)
    }
}
