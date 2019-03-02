/// A small neural-network library based on Torch.
///
/// This library tries to stay as close as possible to the original
/// Python and C++ implementations.

mod var_store;
pub use var_store::Path;
pub use var_store::VarStore;

mod module;
pub use module::{Module, ModuleT};

mod linear;
pub use linear::Linear;

mod conv2d;
pub use conv2d::{Conv2D, Conv2DConfig};

mod batch_norm;
pub use batch_norm::BatchNorm2D;

mod func;
pub use func::{Func, FuncT};

mod sequential;
pub use sequential::Sequential;
pub use sequential::SequentialT;

mod c_optimizer;
mod optimizer;
pub use optimizer::Optimizer;
