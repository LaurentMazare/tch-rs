#[macro_use]
extern crate lazy_static;

extern crate libc;
extern crate thiserror;
extern crate zip;

pub mod data;

mod error;
pub use error::TchError;

pub(crate) mod wrappers;
pub use wrappers::{
    device::{Cuda, Device},
    get_num_interop_threads, get_num_threads,
    jit::{CModule, IValue},
    kind::Kind,
    manual_seed,
    scalar::Scalar,
    set_num_interop_threads, set_num_threads,
    tensor::{no_grad, no_grad_guard, NoGradGuard, Reduction, Tensor},
};

mod tensor;
pub use tensor::{index, IndexOp, NewAxis, Shape, TensorIndexer};

pub mod nn;
pub mod vision;

pub mod kind {
    pub(crate) use super::wrappers::kind::T;
    pub use super::wrappers::kind::*;
}
