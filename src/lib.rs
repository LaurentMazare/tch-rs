#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate failure;
extern crate libc;

#[macro_use]
mod utils;
pub use utils::manual_seed;

pub mod data;

mod device;
mod device_wrapper;
pub use device::{Cpu, Cuda, Device};

pub mod kind;
pub use kind::Kind;

mod jit;
pub use jit::{CModule, IValue};

mod scalar;
pub use scalar::Scalar;

mod tensor;
pub use tensor::{no_grad, NoGradGuard, Tensor};

pub mod nn;
pub mod vision;
