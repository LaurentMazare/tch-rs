// TODO: add gradient clipping
use crate::tensor::C_tensor;
use crate::utils::read_and_clean_error;
use crate::Tensor;
use libc::c_int;

#[repr(C)]
pub struct C_optimizer {
    _private: [u8; 0],
}

extern "C" {
    fn ato_adam(lr: f64, beta1: f64, beta2: f64, wd: f64) -> *mut C_optimizer;
    fn ato_rms_prop(
        lr: f64,
        alpha: f64,
        eps: f64,
        wd: f64,
        momentum: f64,
        centered: c_int,
    ) -> *mut C_optimizer;
    fn ato_sgd(
        lr: f64,
        momentum: f64,
        dampening: f64,
        wd: f64,
        nesterov: c_int,
    ) -> *mut C_optimizer;
    fn ato_add_parameters(arg: *mut C_optimizer, ts: *const *mut C_tensor, n: c_int);
    fn ato_set_learning_rate(arg: *mut C_optimizer, lr: f64);
    fn ato_set_momentum(arg: *mut C_optimizer, momentum: f64);
    fn ato_zero_grad(arg: *mut C_optimizer);
    fn ato_step(arg: *mut C_optimizer);
    fn ato_free(arg: *mut C_optimizer);
}

pub struct COptimizer {
    c_optimizer: *mut C_optimizer,
}

impl COptimizer {
    pub fn adam(lr: f64, beta1: f64, beta2: f64, wd: f64) -> COptimizer {
        let c_optimizer = unsafe { ato_adam(lr, beta1, beta2, wd) };
        read_and_clean_error();
        COptimizer { c_optimizer }
    }

    // Maybe we should use the builder pattern to provide default values for these ?
    pub fn rms_prop(
        lr: f64,
        alpha: f64,
        eps: f64,
        wd: f64,
        momentum: f64,
        centered: bool,
    ) -> COptimizer {
        let centered = if centered { 1 } else { 0 };
        let c_optimizer = unsafe { ato_rms_prop(lr, alpha, eps, wd, momentum, centered) };
        read_and_clean_error();
        COptimizer { c_optimizer }
    }

    pub fn sgd(lr: f64, momentum: f64, dampening: f64, wd: f64, nesterov: bool) -> COptimizer {
        let nesterov = if nesterov { 1 } else { 0 };
        let c_optimizer = unsafe { ato_sgd(lr, momentum, dampening, wd, nesterov) };
        read_and_clean_error();
        COptimizer { c_optimizer }
    }

    pub fn add_parameters(&mut self, ts: &[Tensor]) {
        let ts: Vec<_> = ts.iter().map(|x| x.c_tensor).collect();
        unsafe { ato_add_parameters(self.c_optimizer, ts.as_ptr(), ts.len() as c_int) };
        read_and_clean_error()
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        unsafe { ato_set_learning_rate(self.c_optimizer, lr) };
        read_and_clean_error()
    }

    pub fn set_momentum(&mut self, m: f64) {
        unsafe { ato_set_momentum(self.c_optimizer, m) };
        read_and_clean_error()
    }

    pub fn zero_grad(&self) {
        unsafe { ato_zero_grad(self.c_optimizer) };
        read_and_clean_error()
    }

    pub fn step(&self) {
        unsafe { ato_step(self.c_optimizer) };
        read_and_clean_error()
    }
}

impl Drop for COptimizer {
    fn drop(&mut self) {
        unsafe { ato_free(self.c_optimizer) }
    }
}
