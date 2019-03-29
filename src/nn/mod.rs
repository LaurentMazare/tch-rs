//! A small neural-network library based on Torch.
//!
//! This library tries to stay as close as possible to the original
//! Python and C++ implementations.
mod init;
pub use init::{init, Init};

mod var_store;
pub use var_store::{Path, VarStore};

mod module;
pub use module::{Module, ModuleT};

mod linear;
pub use linear::{Linear, LinearConfig};

mod conv;
pub use conv::{Conv1D, Conv2D, Conv3D, ConvConfig, ConvConfigND};

mod conv_transpose2d;
pub use conv_transpose2d::{ConvTranspose2D, ConvTranspose2DConfig};

mod batch_norm;
pub use batch_norm::{BatchNorm2D, BatchNorm2DConfig};

mod rnn;
pub use rnn::{Config, GRU, LSTM, RNN};

mod func;
pub use func::{Func, FuncT};

mod sequential;
pub use sequential::Sequential;
pub use sequential::SequentialT;

mod optimizer;
pub use optimizer::{adam, rms_prop, sgd, Adam, Optimizer, OptimizerConfig, RmsProp, Sgd};

#[derive(Debug)]
pub struct Id();

impl ModuleT for Id {
    fn forward_t(&self, xs: &crate::Tensor, _train: bool) -> crate::Tensor {
        xs.shallow_clone()
    }
}
