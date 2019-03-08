//! Layers defined by closures.
use crate::Tensor;

/// A layer defined by a simple closure.
pub struct Func<'a> {
    f: Box<Fn(&Tensor) -> Tensor + 'a>,
}

impl<'a> std::fmt::Debug for Func<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "func")
    }
}

impl<'a> Func<'a> {
    pub fn new<F>(f: F) -> Func<'a>
    where
        F: 'a,
        F: Fn(&Tensor) -> Tensor,
    {
        Func { f: Box::new(f) }
    }
}

impl<'a> super::module::Module for Func<'a> {
    fn forward(&self, xs: &Tensor) -> Tensor {
        (*self.f)(xs)
    }
}

/// A layer defined by a closure with an additional training parameter.
pub struct FuncT<'a> {
    f: Box<Fn(&Tensor, bool) -> Tensor + 'a>,
}

impl<'a> std::fmt::Debug for FuncT<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "funcT")
    }
}

impl<'a> FuncT<'a> {
    pub fn new<F>(f: F) -> FuncT<'a>
    where
        F: 'a,
        F: Fn(&Tensor, bool) -> Tensor,
    {
        FuncT { f: Box::new(f) }
    }
}

impl<'a> super::module::ModuleT for FuncT<'a> {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        (*self.f)(xs, train)
    }
}
