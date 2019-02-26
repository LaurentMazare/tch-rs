use crate::tensor::Tensor;
use crate::{Device, Kind};

/// A VarStore is used to store variables used by one or multiple layers.
/// It specifies a single device where all variables are stored.
pub struct VarStore {
    variables: Vec<Tensor>,
    device: Device,
}

pub struct Path<'a> {
    path: Vec<String>,
    var_store: &'a mut VarStore,
}

impl VarStore {
    pub fn new(device: Device) -> VarStore {
        VarStore {
            variables: Vec::new(),
            device,
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn variables(&self) -> &[Tensor] {
        self.variables.as_slice()
    }

    pub fn root(&mut self) -> Path {
        Path {
            path: vec![],
            var_store: self,
        }
    }
}

impl<'a> Path<'a> {
    pub fn sub(&'a mut self, s: &str) -> Path<'a> {
        let mut path = self.path.clone();
        path.push(s.to_owned());
        Path {
            path,
            var_store: self.var_store,
        }
    }

    pub fn device(&self) -> Device {
        self.var_store.device
    }

    pub fn zeros(&mut self, dims: &[i64]) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.device())).set_requires_grad(true);
        self.var_store.variables.push(z.shallow_clone());
        z
    }

    pub fn ones(&mut self, dims: &[i64]) -> Tensor {
        let z = Tensor::ones(dims, (Kind::Float, self.device())).set_requires_grad(true);
        self.var_store.variables.push(z.shallow_clone());
        z
    }

    pub fn randn_standard(&mut self, dims: &[i64]) -> Tensor {
        let z = Tensor::randn(dims, (Kind::Float, self.device())).set_requires_grad(true);
        self.var_store.variables.push(z.shallow_clone());
        z
    }

    pub fn randn(&mut self, dims: &[i64], mean: f64, stdev: f64) -> Tensor {
        let z = Tensor::randn(dims, (Kind::Float, self.device()));
        let z = (z * stdev + mean).set_requires_grad(true);
        self.var_store.variables.push(z.shallow_clone());
        z
    }

    pub fn uniform(&mut self, dims: &[i64], lo: f64, up: f64) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.device()))
            .uniform_(lo, up)
            .set_requires_grad(true);
        self.var_store.variables.push(z.shallow_clone());
        z
    }

    pub fn kaiming_uniform(&mut self, dims: &[i64]) -> Tensor {
        let fan_in: i64 = dims.iter().skip(1).product();
        let bound = (1.0 / fan_in as f64).sqrt();
        self.uniform(dims, -bound, bound)
    }
}
