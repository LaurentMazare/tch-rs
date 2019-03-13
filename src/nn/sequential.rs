//! A sequential layer used to chain multiple layers and closures.
use super::{Module, ModuleT};
use crate::Tensor;

#[derive(Debug)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Sequential {
        Sequential { layers: vec![] }
    }

    pub fn len(&self) -> i64 {
        self.layers.len() as i64
    }
}

impl Module for Sequential {
    fn forward(&self, xs: &Tensor) -> Tensor {
        if self.layers.is_empty() {
            xs.shallow_clone()
        } else {
            let xs = self.layers[0].forward(xs);
            self.layers
                .iter()
                .skip(1)
                .fold(xs, |xs, layer| layer.forward(&xs))
        }
    }
}

impl Sequential {
    /// Appends a layer after all the current layers.
    pub fn add<M: Module + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn<F>(self, f: F) -> Self
    where
        F: 'static,
        F: Fn(&Tensor) -> Tensor,
    {
        self.add(super::func::Func::new(f))
    }

    /// Applies the forward pass and returns the output for each layer.
    pub fn forward_all(&self, xs: &Tensor) -> Vec<Tensor> {
        if self.layers.is_empty() {
            vec![xs.shallow_clone()]
        } else {
            let xs = self.layers[0].forward(xs);
            let mut vec = vec![];
            let out = self.layers.iter().skip(1).fold(xs, |xs, layer| {
                let out = layer.forward(&xs);
                vec.push(xs);
                out
            });
            vec.push(out);
            vec
        }
    }
}

#[derive(Debug)]
pub struct SequentialT {
    layers: Vec<Box<dyn ModuleT>>,
}

impl SequentialT {
    pub fn new() -> SequentialT {
        SequentialT { layers: vec![] }
    }

    pub fn len(&self) -> i64 {
        self.layers.len() as i64
    }
}

impl ModuleT for SequentialT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        if self.layers.is_empty() {
            xs.shallow_clone()
        } else {
            let xs = self.layers[0].forward_t(xs, train);
            self.layers
                .iter()
                .skip(1)
                .fold(xs, |xs, layer| layer.forward_t(&xs, train))
        }
    }
}

impl SequentialT {
    /// Appends a layer after all the current layers.
    pub fn add<M: ModuleT + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn<F>(self, f: F) -> Self
    where
        F: 'static,
        F: Fn(&Tensor) -> Tensor,
    {
        self.add(super::func::Func::new(f))
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn_t<F>(self, f: F) -> Self
    where
        F: 'static,
        F: Fn(&Tensor, bool) -> Tensor,
    {
        self.add(super::func::FuncT::new(f))
    }

    /// Applies the forward pass and returns the output for each layer.
    pub fn forward_all_t(&self, xs: &Tensor, train: bool) -> Vec<Tensor> {
        if self.layers.is_empty() {
            vec![xs.shallow_clone()]
        } else {
            let xs = self.layers[0].forward_t(xs, train);
            let mut vec = vec![];
            let out = self.layers.iter().skip(1).fold(xs, |xs, layer| {
                let out = layer.forward_t(&xs, train);
                vec.push(xs);
                out
            });
            vec.push(out);
            vec
        }
    }
}
