//! A small neural-network library based on Torch.
//!
//! This library tries to stay as close as possible to the original
//! Python and C++ implementations.
mod init;
pub use init::{f_init, init, Init};

mod var_store;
pub use var_store::{Path, VarStore, Variables};

mod module;
pub use module::{batch_accuracy_for_logits, Module, ModuleT};

mod linear;
pub use linear::*;

mod conv;
pub use conv::*;

mod conv_transpose;
pub use conv_transpose::*;

mod batch_norm;
pub use batch_norm::*;

mod layer_norm;
pub use layer_norm::*;

mod sparse;
pub use sparse::*;

mod rnn;
pub use rnn::*;

mod func;
pub use func::*;

mod sequential;
pub use sequential::*;

mod optimizer;
pub use optimizer::{
    adam, adamw, rms_prop, sgd, Adam, AdamW, Optimizer, OptimizerConfig, RmsProp, Sgd,
};

/// An identity layer. This just propagates its tensor input as output.
#[derive(Debug)]
pub struct Id();

impl ModuleT for Id {
    type Input = crate::Tensor;
    type Output = crate::Tensor;

    fn forward_t(&self, xs: &crate::Tensor, _train: bool) -> crate::Tensor {
        xs.shallow_clone()
    }

    fn batch_accuracy_for_logits(
        &self,
        xs: &Self::Input,
        ys: &Self::Input,
        d: crate::Device,
        batch_size: i64,
    ) -> f64 {
        module::batch_accuracy_for_logits(self, xs, ys, d, batch_size)
    }
}

impl Module for crate::CModule {
    type Input = crate::Tensor;
    type Output = crate::Tensor;

    fn forward(&self, xs: &Self::Input) -> Self::Output {
        self.forward_ts(&[xs]).unwrap()
    }
}

impl ModuleT for crate::TrainableCModule {
    type Input = crate::Tensor;
    type Output = crate::Tensor;

    fn forward_t(&self, xs: &Self::Input, _train: bool) -> Self::Output {
        self.inner.forward_ts(&[xs]).unwrap()
    }

    fn batch_accuracy_for_logits(
        &self,
        xs: &Self::Input,
        ys: &Self::Input,
        d: crate::Device,
        batch_size: i64,
    ) -> f64 {
        module::batch_accuracy_for_logits(self, xs, ys, d, batch_size)
    }
}
