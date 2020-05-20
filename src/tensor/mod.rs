//! A Torch tensor.

mod conversion;
mod impls;
pub mod index;
mod iter;
mod npy;
mod shape;

pub use index::{IndexOp, NewAxis, TensorIndexer};
pub use shape::Shape;

#[used]
static INIT_ARRAY: [unsafe extern "C" fn(); 1] = [torch_sys::dummy_cuda_dependency];
