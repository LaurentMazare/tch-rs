//! Exponential moving average.
use super::var_store::VarStore;
use crate::Tensor;

/// A ExponentialMovingAverage maintains moving averages of the trainable variables.
///
/// It keeps shadow copies of trainable variables in VarStore. In each step,
/// it applies average by `shadow_variable -= (1 - decay) * (shadow_variable - variable)`,
/// then copies updated shadow variables back to VarStore.
#[derive(Debug)]
pub struct ExponentialMovingAverage {
    decay: f64,
    shadow_target_vars: Vec<(Tensor, Tensor)>,
}

impl ExponentialMovingAverage {
    /// Constructs a new ExponentialMovingAverage instance.
    pub fn new(vs: &VarStore, decay: f64) -> ExponentialMovingAverage {
        let shadow_target_vars = vs
            .trainable_variables()
            .iter()
            .map(|var| (var.copy(), var.shallow_clone()))
            .collect();

        ExponentialMovingAverage {
            decay,
            shadow_target_vars,
        }
    }

    /// Sets the decay parameter.
    pub fn set_decay(&mut self, decay: f64) {
        self.decay = decay;
    }

    /// Apply weighted average and update VarStore.
    pub fn step(&mut self) {
        let decay = self.decay;
        self.shadow_target_vars
            .iter_mut()
            .for_each(|(shadow, target)| {
                crate::no_grad(|| {
                    *shadow -= (1. - decay) * (&*shadow - &*target);
                    target.copy_(shadow);
                });
            });
    }
}

#[test]
fn ema_test() {
    let vs = VarStore::new(crate::Device::cuda_if_available());
    let root = vs.root();

    let t = Tensor::from(1.0_f64);
    let mut var = root.var_copy("dummy", &t);

    let mut ema = ExponentialMovingAverage::new(&vs, 0.99);

    let t = Tensor::from(0.5_f64);
    crate::no_grad(|| var.copy_(&t));
    ema.step();

    let t = Tensor::from(0.1_f64);
    crate::no_grad(|| var.copy_(&t));
    ema.set_decay(0.98);
    ema.step();

    let expect = 1.0 * 0.99 * 0.98 + 0.5 * (1. - 0.99) * 0.98 + 0.1 * (1. - 0.98);
    assert!((f64::from(var) - expect).abs() <= 0.0000001);
}
