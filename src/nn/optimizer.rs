//! Optimizers to be used for gradient-descent based training.
use super::var_store::{VarStore, Variables};
use crate::wrappers::optimizer::COptimizer;
use crate::{TchError, Tensor};
use std::sync::{Arc, Mutex};

/// An optimizer to run gradient descent.
#[derive(Debug)]
pub struct Optimizer<T> {
    opt: COptimizer,
    variables: Arc<Mutex<Variables>>,
    variables_in_optimizer: usize,
    config: T,
}

/// Optimizer configurations. These configs can be used to build optimizer.
pub trait OptimizerConfig
where
    Self: std::marker::Sized,
{
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError>;

    /// Builds an optimizer with the specified learning rate handling variables stored in `vs`.
    fn build(self, vs: &VarStore, lr: f64) -> Result<Optimizer<Self>, TchError> {
        let mut opt = self.build_copt(lr)?;
        let v = vs.variables_.lock().unwrap();
        opt.add_parameters(&v.trainable_variables)?;
        Ok(Optimizer {
            opt,
            variables: vs.variables_.clone(),
            variables_in_optimizer: v.trainable_variables.len(),
            config: self,
        })
    }
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

/// Creates the configuration for a Stochastic Gradient Descent (SGD) optimizer.
pub fn sgd(momentum: f64, dampening: f64, wd: f64, nesterov: bool) -> Sgd {
    Sgd {
        momentum,
        dampening,
        wd,
        nesterov,
    }
}

impl OptimizerConfig for Sgd {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        COptimizer::sgd(lr, self.momentum, self.dampening, self.wd, self.nesterov)
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

/// Creates the configuration for the Adam optimizer.
pub fn adam(beta1: f64, beta2: f64, wd: f64) -> Adam {
    Adam { beta1, beta2, wd }
}

impl OptimizerConfig for Adam {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        COptimizer::adam(lr, self.beta1, self.beta2, self.wd)
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

/// Creates the configuration for the RmsProp optimizer.
pub fn rms_prop(alpha: f64, eps: f64, wd: f64, momentum: f64, centered: bool) -> RmsProp {
    RmsProp {
        alpha,
        eps,
        wd,
        momentum,
        centered,
    }
}

impl OptimizerConfig for RmsProp {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        COptimizer::rms_prop(
            lr,
            self.alpha,
            self.eps,
            self.wd,
            self.momentum,
            self.centered,
        )
    }
}

impl<T> Optimizer<T> {
    fn add_missing_variables(&mut self) {
        let v = self.variables.lock().unwrap();
        let missing_variables = v.trainable_variables.len() - self.variables_in_optimizer;

        if missing_variables > 0 {
            self.opt
                .add_parameters(&v.trainable_variables[self.variables_in_optimizer..])
                .unwrap();
            self.variables_in_optimizer = v.trainable_variables.len();
        }
    }

    /// Zeroes the gradient for the tensors tracked by this optimizer.
    pub fn zero_grad(&mut self) {
        self.add_missing_variables();
        self.opt.zero_grad().unwrap()
    }

    /// Clips gradient value at some specified maximum value.
    pub fn clip_grad_value(&self, max: f64) {
        let v = self.variables.lock().unwrap();
        for tensor in v.trainable_variables.iter() {
            let _t = tensor.grad().clamp_(-max, max);
        }
    }

    /// Performs an optimization step, updating the tracked tensors based on their gradients.
    pub fn step(&mut self) {
        self.add_missing_variables();
        self.opt.step().unwrap()
    }

    /// Applies a backward step pass, update the gradients, and performs an optimization step.
    pub fn backward_step(&mut self, loss: &Tensor) {
        self.add_missing_variables();
        self.opt.zero_grad().unwrap();
        loss.backward();
        self.opt.step().unwrap()
    }

    /// Applies a backward step pass, update the gradients, and performs an optimization step.
    ///
    /// The gradients are clipped based on `max` before being applied.
    pub fn backward_step_clip(&mut self, loss: &Tensor, max: f64) {
        self.add_missing_variables();
        self.opt.zero_grad().unwrap();
        loss.backward();
        self.clip_grad_value(max);
        self.opt.step().unwrap()
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
