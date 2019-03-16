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
pub use linear::{Linear, LinearConfig, LinearConfigBuilder};

mod conv2d;
pub use conv2d::{Conv2D, Conv2DConfig, Conv2DConfigBuilder};

mod conv_transpose2d;
pub use conv_transpose2d::{ConvTranspose2D, ConvTranspose2DConfig, ConvTranspose2DConfigBuilder};

mod batch_norm;
pub use batch_norm::{BatchNorm2D, BatchNorm2DConfig, BatchNorm2DConfigBuilder};

mod rnn;
pub use rnn::{Config, ConfigBuilder, GRU, LSTM, RNN};

mod func;
pub use func::{Func, FuncT};

mod sequential;
pub use sequential::Sequential;
pub use sequential::SequentialT;

mod optimizer;
pub use optimizer::{
    Adam, AdamBuilder, Optimizer, OptimizerConfig, RmsProp, RmsPropBuilder, Sgd, SgdBuilder,
};

#[derive(Debug)]
pub struct Id();

impl ModuleT for Id {
    fn forward_t(&self, xs: &crate::Tensor, _train: bool) -> crate::Tensor {
        xs.shallow_clone()
    }
}
