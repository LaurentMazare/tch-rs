use crate::Tensor;
use failure::Fallible;
use libc::c_int;

pub(in crate::nn) struct COptimizer {
    c_optimizer: *mut torch_sys::C_optimizer,
}

impl COptimizer {
    pub(in crate::nn) fn adam(lr: f64, beta1: f64, beta2: f64, wd: f64) -> Fallible<COptimizer> {
        let c_optimizer = unsafe_torch_err!({ torch_sys::ato_adam(lr, beta1, beta2, wd) });
        Ok(COptimizer { c_optimizer })
    }

    // Maybe we should use the builder pattern to provide default values for these ?
    pub(in crate::nn) fn rms_prop(
        lr: f64,
        alpha: f64,
        eps: f64,
        wd: f64,
        momentum: f64,
        centered: bool,
    ) -> Fallible<COptimizer> {
        let c_optimizer = unsafe_torch_err!({
            torch_sys::ato_rms_prop(lr, alpha, eps, wd, momentum, centered as i32)
        });
        Ok(COptimizer { c_optimizer })
    }

    pub(in crate::nn) fn sgd(
        lr: f64,
        momentum: f64,
        dampening: f64,
        wd: f64,
        nesterov: bool,
    ) -> Fallible<COptimizer> {
        let c_optimizer =
            unsafe_torch_err!({ torch_sys::ato_sgd(lr, momentum, dampening, wd, nesterov as i32) });
        Ok(COptimizer { c_optimizer })
    }

    pub(in crate::nn) fn add_parameters(&mut self, ts: &[Tensor]) -> Fallible<()> {
        let ts: Vec<_> = ts.iter().map(|x| x.c_tensor).collect();
        unsafe_torch_err!({
            torch_sys::ato_add_parameters(self.c_optimizer, ts.as_ptr(), ts.len() as c_int)
        });
        Ok(())
    }

    pub(in crate::nn) fn set_learning_rate(&mut self, lr: f64) -> Fallible<()> {
        unsafe_torch_err!({ torch_sys::ato_set_learning_rate(self.c_optimizer, lr) });
        Ok(())
    }

    pub(in crate::nn) fn set_momentum(&mut self, m: f64) -> Fallible<()> {
        unsafe_torch_err!({ torch_sys::ato_set_momentum(self.c_optimizer, m) });
        Ok(())
    }

    pub(in crate::nn) fn zero_grad(&self) -> Fallible<()> {
        unsafe_torch_err!({ torch_sys::ato_zero_grad(self.c_optimizer) });
        Ok(())
    }

    pub(in crate::nn) fn step(&self) -> Fallible<()> {
        unsafe_torch_err!({ torch_sys::ato_step(self.c_optimizer) });
        Ok(())
    }
}

impl Drop for COptimizer {
    fn drop(&mut self) {
        unsafe { torch_sys::ato_free(self.c_optimizer) }
    }
}
