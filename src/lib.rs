#[macro_use]
extern crate lazy_static;

pub mod data;

mod error;
pub use error::TchError;

pub(crate) mod wrappers;
pub use wrappers::device::{Cuda, Device};
pub use wrappers::jit::{self, CModule, IValue, TrainableCModule};
pub use wrappers::kind::{self, Kind};
pub use wrappers::optimizer::COptimizer;
pub use wrappers::scalar::Scalar;
pub use wrappers::utils;
pub use wrappers::{
    get_num_interop_threads, get_num_threads, manual_seed, set_num_interop_threads,
    set_num_threads, QEngine,
};

mod tensor;
pub use tensor::{
    autocast, display, index, no_grad, no_grad_guard, with_grad, IndexOp, NewAxis, NoGradGuard,
    Reduction, Shape, Tensor, TensorIndexer,
};

pub mod nn;
pub mod vision;

pub fn maybe_init_cuda() {
    unsafe {
        torch_sys::dummy_cuda_dependency();
    }
}
