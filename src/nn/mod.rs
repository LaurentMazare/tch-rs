mod var_store;
pub use var_store::VarStore;

mod module;
pub use module::Module;

mod linear;
pub use linear::Linear;

mod c_optimizer;
mod optimizer;
pub use optimizer::Optimizer;

// TODO: do not expose this
pub use c_optimizer::COptimizer;
