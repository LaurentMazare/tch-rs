use crate::Tensor;
use libc::c_int;

pub struct COptimizer {
    c_optimizer: *mut torch_sys::C_optimizer,
}

impl COptimizer {
    pub fn adam(lr: f64, beta1: f64, beta2: f64, wd: f64) -> COptimizer {
        let c_optimizer = unsafe_torch!({ torch_sys::ato_adam(lr, beta1, beta2, wd) });
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
        let c_optimizer = unsafe_torch!({ torch_sys::ato_rms_prop(lr, alpha, eps, wd, momentum, centered) });
        COptimizer { c_optimizer }
    }

    pub fn sgd(lr: f64, momentum: f64, dampening: f64, wd: f64, nesterov: bool) -> COptimizer {
        let nesterov = if nesterov { 1 } else { 0 };
        let c_optimizer = unsafe_torch!({ torch_sys::ato_sgd(lr, momentum, dampening, wd, nesterov) });
        COptimizer { c_optimizer }
    }

    pub fn add_parameters(&mut self, ts: &[Tensor]) {
        let ts: Vec<_> = ts.iter().map(|x| x.c_tensor).collect();
        unsafe_torch!({ torch_sys::ato_add_parameters(self.c_optimizer, ts.as_ptr(), ts.len() as c_int) })
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        unsafe_torch!({ torch_sys::ato_set_learning_rate(self.c_optimizer, lr) })
    }

    pub fn set_momentum(&mut self, m: f64) {
        unsafe_torch!({ torch_sys::ato_set_momentum(self.c_optimizer, m) })
    }

    pub fn zero_grad(&self) {
        unsafe_torch!({ torch_sys::ato_zero_grad(self.c_optimizer) })
    }

    pub fn step(&self) {
        unsafe_torch!({ torch_sys::ato_step(self.c_optimizer) })
    }
}

impl Drop for COptimizer {
    fn drop(&mut self) {
        unsafe { torch_sys::ato_free(self.c_optimizer) }
    }
}
