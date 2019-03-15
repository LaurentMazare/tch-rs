//! Optimizers to be used for gradient-descent based training.
use super::var_store::VarStore;
use crate::wrappers::optimizer::COptimizer;
use crate::Tensor;
use failure::Fallible;

#[derive(Debug)]
pub struct Optimizer {
    opt: COptimizer,
    trainable_variables: Vec<Tensor>,
}

/// Parameters for the SGD optimizer.
#[derive(Debug, Copy, Clone)]
pub struct Sgd {
    pub momentum: f64,
    pub dampening: f64,
    pub wd: f64,
    pub nesterov: bool,
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

/// Parameters for the Adam optimizer.
#[derive(Debug, Copy, Clone)]
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
#[derive(Debug, Copy, Clone)]
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
    pub fn sgd(vs: &VarStore, lr: f64, s: Sgd) -> Fallible<Optimizer> {
        let mut opt = COptimizer::sgd(lr, s.momentum, s.dampening, s.wd, s.nesterov)?;
        let trainable_variables = vs.trainable_variables();
        opt.add_parameters(&trainable_variables)?;
        Ok(Optimizer {
            opt,
            trainable_variables,
        })
    }

    pub fn adam(vs: &VarStore, lr: f64, a: Adam) -> Fallible<Optimizer> {
        let mut opt = COptimizer::adam(lr, a.beta1, a.beta2, a.wd)?;
        let trainable_variables = vs.trainable_variables();
        opt.add_parameters(&trainable_variables)?;
        Ok(Optimizer {
            opt,
            trainable_variables,
        })
    }

    pub fn rms_prop(vs: &VarStore, lr: f64, r: RmsProp) -> Fallible<Optimizer> {
        let mut opt = COptimizer::rms_prop(lr, r.alpha, r.eps, r.wd, r.momentum, r.centered)?;
        let trainable_variables = vs.trainable_variables();
        opt.add_parameters(&trainable_variables)?;
        Ok(Optimizer {
            opt,
            trainable_variables,
        })
    }

    pub fn zero_grad(&self) {
        self.opt.zero_grad().unwrap()
    }

    /// Clips gradient value at some specified maximum value.
    pub fn clip_grad_value(&self, max: f64) {
        for tensor in self.trainable_variables.iter() {
            let _t = tensor.grad().clamp_(-max, max);
        }
    }

    pub fn step(&self) {
        self.opt.step().unwrap()
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
        self.opt.set_learning_rate(lr).unwrap()
    }

    /// Sets the optimizer momentum.
    pub fn set_momentum(&mut self, m: f64) {
        self.opt.set_momentum(m).unwrap()
    }
}

pub fn sgd(
    vs: &VarStore,
    lr: f64,
    momentum: f64,
    dampening: f64,
    wd: f64,
    nesterov: bool,
) -> Fallible<Optimizer> {
    let sgd = Sgd {
        momentum,
        dampening,
        wd,
        nesterov,
    };
    Optimizer::sgd(vs, lr, sgd)
}

pub fn adam(vs: &VarStore, lr: f64, beta1: f64, beta2: f64, wd: f64) -> Fallible<Optimizer> {
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
) -> Fallible<Optimizer> {
    let rmsprop = RmsProp {
        alpha,
        eps,
        wd,
        momentum,
        centered,
    };
    Optimizer::rms_prop(vs, lr, rmsprop)
}
