#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate failure;
extern crate libc;
extern crate zip;

pub mod data;

mod wrappers;
pub use wrappers::device::{Cpu, Cuda, Device};
pub use wrappers::jit::{CModule, IValue};
pub use wrappers::kind::Kind;
pub use wrappers::manual_seed;
pub use wrappers::scalar::Scalar;

mod tensor;
pub use tensor::{no_grad, no_grad_guard, NoGradGuard, Reduction, Tensor};

pub mod nn;
pub mod vision;

pub mod kind {
    pub(crate) use super::wrappers::kind::T;
    pub use super::wrappers::kind::*;
}
