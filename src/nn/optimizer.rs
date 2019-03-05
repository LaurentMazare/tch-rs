/// Optimizers to be used for gradient-descent based training.
use super::c_optimizer::COptimizer;
use super::var_store::VarStore;
use crate::{Scalar, Tensor};

pub struct Optimizer {
    opt: COptimizer,
    trainable_variables: Vec<Tensor>,
}

/// Parameters for the SGD optimizer.
#[derive(Builder, Debug, Default, Clone, Copy)]
#[builder(setter(into))]
pub struct Sgd {
    pub momentum: f64,
    pub dampening: f64,
    pub wd: f64,
    pub nesterov: bool,
}

/// Parameters for the Adam optimizer.
#[derive(Builder, Debug, Clone, Copy)]
#[builder(setter(into))]
pub struct Adam {
    pub beta1: f64,
    pub beta2: f64,
    pub wd: f64,
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

/// Parameters for the RmsProp optimizer.
#[derive(Builder, Debug, Clone, Copy)]
#[builder(setter(into))]
pub struct RmsProp {
    pub alpha: f64,
    pub eps: f64,
    pub wd: f64,
    pub momentum: f64,
    pub centered: bool,
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
        Optimizer {
            opt,
            trainable_variables,
        }
    }

    pub fn adam(vs: &VarStore, lr: f64, a: Adam) -> Optimizer {
        let mut opt = COptimizer::adam(lr, a.beta1, a.beta2, a.wd);
        let trainable_variables = vs.trainable_variables();
        opt.add_parameters(&trainable_variables);
        Optimizer {
            opt,
            trainable_variables,
        }
    }

    pub fn rms_prop(vs: &VarStore, lr: f64, r: RmsProp) -> Optimizer {
        let mut opt = COptimizer::rms_prop(lr, r.alpha, r.eps, r.wd, r.momentum, r.centered);
        let trainable_variables = vs.trainable_variables();
        opt.add_parameters(&trainable_variables);
        Optimizer {
            opt,
            trainable_variables,
        }
    }

    pub fn zero_grad(&self) {
        self.opt.zero_grad()
    }

    /// Clips gradient value at some specified maximum value.
    pub fn clip_grad_value(&self, max: f64) {
        for tensor in self.trainable_variables.iter() {
            let _t = tensor
                .grad()
                .clamp_(&Scalar::float(-max), &Scalar::float(max));
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

    pub fn backward_step_clip(&self, loss: &Tensor, max: f64) {
        self.zero_grad();
        loss.backward();
        self.clip_grad_value(max);
        self.step();
    }

    /// Sets the optimizer learning rate.
    pub fn set_lr(&mut self, lr: f64) {
        self.opt.set_learning_rate(lr)
    }

    /// Sets the optimizer momentum.
    pub fn set_momentum(&mut self, m: f64) {
        self.opt.set_momentum(m)
    }
}

pub fn sgd(
    vs: &VarStore,
    lr: f64,
    momentum: f64,
    dampening: f64,
    wd: f64,
    nesterov: bool,
) -> Optimizer {
    let sgd = Sgd {
        momentum,
        dampening,
        wd,
        nesterov,
    };
    Optimizer::sgd(vs, lr, sgd)
}

pub fn adam(vs: &VarStore, lr: f64, beta1: f64, beta2: f64, wd: f64) -> Optimizer {
    let adam = Adam { beta1, beta2, wd };
    Optimizer::adam(vs, lr, adam)
}

pub fn rms_prop(
    vs: &VarStore,
    lr: f64,
    alpha: f64,
    eps: f64,
    wd: f64,
    momentum: f64,
    centered: bool,
) -> Optimizer {
    let rmsprop = RmsProp {
        alpha,
        eps,
        wd,
        momentum,
        centered,
    };
    Optimizer::rms_prop(vs, lr, rmsprop)
}
