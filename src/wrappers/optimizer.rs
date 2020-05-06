use super::tensor::Tensor;
use crate::TchError;
use libc::c_int;

pub struct COptimizer {
    c_optimizer: *mut torch_sys::C_optimizer,
}

impl std::fmt::Debug for COptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "optimizer")
    }
}

impl COptimizer {
    pub fn adam(lr: f64, beta1: f64, beta2: f64, wd: f64) -> Result<COptimizer, TchError> {
        let c_optimizer = unsafe_torch_err!(torch_sys::ato_adam(lr, beta1, beta2, wd));
        Ok(COptimizer { c_optimizer })
    }

    // Maybe we should use the builder pattern to provide default values for these ?
    pub fn rms_prop(
        lr: f64,
        alpha: f64,
        eps: f64,
        wd: f64,
        momentum: f64,
        centered: bool,
    ) -> Result<COptimizer, TchError> {
        let centered = if centered { 1 } else { 0 };
        let c_optimizer = unsafe_torch_err!(torch_sys::ato_rms_prop(
            lr, alpha, eps, wd, momentum, centered
        ));
        Ok(COptimizer { c_optimizer })
    }

    pub fn sgd(
        lr: f64,
        momentum: f64,
        dampening: f64,
        wd: f64,
        nesterov: bool,
    ) -> Result<COptimizer, TchError> {
        let nesterov = if nesterov { 1 } else { 0 };
        let c_optimizer =
            unsafe_torch_err!(torch_sys::ato_sgd(lr, momentum, dampening, wd, nesterov));
        Ok(COptimizer { c_optimizer })
    }

    pub fn add_parameters(&mut self, ts: &[Tensor]) -> Result<(), TchError> {
        let ts: Vec<_> = ts.iter().map(|x| x.c_tensor).collect();
        unsafe_torch_err!({
            torch_sys::ato_add_parameters(self.c_optimizer, ts.as_ptr(), ts.len() as c_int)
        });
        Ok(())
    }

    pub fn set_learning_rate(&mut self, lr: f64) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::ato_set_learning_rate(self.c_optimizer, lr));
        Ok(())
    }

    pub fn set_momentum(&mut self, m: f64) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::ato_set_momentum(self.c_optimizer, m));
        Ok(())
    }

    pub fn zero_grad(&self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::ato_zero_grad(self.c_optimizer));
        Ok(())
    }

    pub fn step(&self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::ato_step(self.c_optimizer));
        Ok(())
    }
}

impl Drop for COptimizer {
    fn drop(&mut self) {
        unsafe { torch_sys::ato_free(self.c_optimizer) }
    }
}
