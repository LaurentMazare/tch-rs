extern crate libc;

mod utils;

mod device;
pub use device::Device;

pub mod kind;
pub use kind::Kind;

mod scalar;
pub use scalar::Scalar;

mod tensor;
pub use tensor::{no_grad, Tensor};

pub mod vision;

mod optimizer;
pub use optimizer::Optimizer;
