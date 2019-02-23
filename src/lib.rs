extern crate libc;

#[macro_use]
mod utils;

mod device;
mod device_wrapper;
pub use device::{Cpu, Cuda, Device};

pub mod kind;
pub use kind::Kind;

mod scalar;
pub use scalar::Scalar;

mod tensor;
pub use tensor::{no_grad, Tensor};

pub mod nn;
pub mod vision;
