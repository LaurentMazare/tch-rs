use crate::tensor::Tensor;
use crate::{Device, Kind};

pub struct VarStore {
    variables: Vec<Tensor>,
    kind_device: (Kind, Device),
}

pub struct Path<'a> {
    path: Vec<String>,
    var_store: &'a VarStore,
}

impl VarStore {
    pub fn new(device: Device) -> VarStore {
        VarStore {
            variables: Vec::new(),
            kind_device: (Kind::Float, device),
        }
    }

    pub fn device(&self) -> Device {
        self.kind_device.1
    }

    pub fn zeros(&mut self, dims: &[i64]) -> Tensor {
        let z = Tensor::zeros(dims, self.kind_device).set_requires_grad(true);
        self.variables.push(z.shallow_clone());
        z
    }

    pub fn ones(&mut self, dims: &[i64]) -> Tensor {
        let z = Tensor::ones(dims, self.kind_device).set_requires_grad(true);
        self.variables.push(z.shallow_clone());
        z
    }

    pub fn randn_standard(&mut self, dims: &[i64]) -> Tensor {
        let z = Tensor::randn(dims, self.kind_device).set_requires_grad(true);
        self.variables.push(z.shallow_clone());
        z
    }

    pub fn randn(&mut self, dims: &[i64], mean: f64, stdev: f64) -> Tensor {
        let z = Tensor::randn(dims, self.kind_device);
        let z = (z * stdev + mean).set_requires_grad(true);
        self.variables.push(z.shallow_clone());
        z
    }

    pub fn uniform(&mut self, dims: &[i64], lo: f64, up: f64) -> Tensor {
        let z = Tensor::zeros(dims, self.kind_device)
            .uniform_(lo, up)
            .set_requires_grad(true);
        self.variables.push(z.shallow_clone());
        z
    }

    pub fn kaiming_uniform(&mut self, dims: &[i64]) -> Tensor {
        let fan_in: i64 = dims.iter().skip(1).product();
        let bound = (1.0 / fan_in as f64).sqrt();
        self.uniform(dims, -bound, bound)
    }

    pub fn variables(&self) -> &[Tensor] {
        self.variables.as_slice()
    }

    pub fn root(&self) -> Path {
        Path {
            path: vec![],
            var_store: &self,
        }
    }
}
