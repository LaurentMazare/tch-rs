use crate::tensor::Tensor;

pub struct Sequential {
    layers: Vec<Box<super::module::Module>>,
}

impl Sequential {
    pub fn new() -> Sequential {
        Sequential { layers: vec![] }
    }
}

impl super::module::Module for Sequential {
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
    pub fn add<M: super::module::Module + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn add_fn<F>(self, f: F) -> Self
    where
        F: 'static,
        F: Fn(&Tensor) -> Tensor,
    {
        self.add(super::func::Func::new(f))
    }
}

pub struct SequentialT {
    layers: Vec<Box<super::module::ModuleT>>,
}

impl SequentialT {
    pub fn new() -> SequentialT {
        SequentialT { layers: vec![] }
    }
}

impl super::module::ModuleT for SequentialT {
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
    pub fn add<M: super::module::ModuleT + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn add_fn<F>(self, f: F) -> Self
    where
        F: 'static,
        F: Fn(&Tensor) -> Tensor,
    {
        self.add(super::func::Func::new(f))
    }

    pub fn add_fn_t<F>(self, f: F) -> Self
    where
        F: 'static,
        F: Fn(&Tensor, bool) -> Tensor,
    {
        self.add(super::func::FuncT::new(f))
    }
}
