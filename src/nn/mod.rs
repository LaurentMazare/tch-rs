//! A small neural-network library based on Torch.
//!
//! This library tries to stay as close as possible to the original
//! Python and C++ implementations.
pub mod init;
pub use init::{f_init, init, Init};

mod var_store;
pub use var_store::{Path, VarStore, Variables};

mod module;
pub use module::{Module, ModuleT};

mod linear;
pub use linear::*;

mod conv;
pub use conv::*;

mod conv_transpose;
pub use conv_transpose::*;

mod batch_norm;
pub use batch_norm::*;

mod group_norm;
pub use group_norm::*;

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

use crate::{TchError, Tensor};

impl ModuleT for Id {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Result<Tensor, TchError> {
        Ok(xs.shallow_clone())
    }
}

impl Module for crate::CModule {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, TchError> {
        self.forward_ts(&[xs])
    }
}

impl ModuleT for crate::TrainableCModule {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Result<Tensor, TchError> {
        self.inner.forward_ts(&[xs])
    }
}
