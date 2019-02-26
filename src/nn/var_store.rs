use crate::tensor::Tensor;
use crate::utils::TorchError;
use crate::{Device, Kind};
use std::collections::HashMap;
use std::ops::Div;

/// A VarStore is used to store variables used by one or multiple layers.
/// It specifies a single device where all variables are stored.
pub struct VarStore {
    variables: HashMap<String, Tensor>,
    device: Device,
    counter: i64,
}

pub struct Path<'a> {
    path: Vec<String>,
    var_store: &'a mut VarStore,
}

impl VarStore {
    pub fn new(device: Device) -> VarStore {
        VarStore {
            variables: HashMap::new(),
            device,
            counter: 0,
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn variables(&self) -> Vec<Tensor> {
        self.variables.values().map(|x| x.shallow_clone()).collect()
    }

    pub fn root(&mut self) -> Path {
        Path {
            path: vec![],
            var_store: self,
        }
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), TorchError> {
        let named_tensors = self
            .variables
            .iter()
            .map(|(x, y)| (&x[..], y))
            .collect::<Vec<_>>();
        Tensor::save_multi(named_tensors.as_slice(), path)
    }

    pub fn load(&self, path: &std::path::Path) -> Result<(), TorchError> {
        let named_tensors = Tensor::load_multi(path)?;
        let named_tensors: HashMap<_, _> = named_tensors.into_iter().collect();
        for (name, tensor) in self.variables.iter() {
            match named_tensors.get(name) {
                Some(src) => crate::no_grad(|| tensor.copy_(src)),
                None => Err(TorchError::new(format!(
                    "cannot find {} in {:?}",
                    name, path
                )))?,
            }
        }
        Ok(())
    }
}

impl<'a> Path<'a> {
    pub fn sub(&'a mut self, s: &str) -> Path<'a> {
        if s.chars().any(|x| x == '.') {
            panic!("sub name cannot contain a dot {}", s);
        }
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

    fn path(&self, name: &str) -> String {
        if name.chars().any(|x| x == '.') {
            panic!("variable name cannot contain a dot {}", name);
        }
        if self.path.len() == 0 {
            name.to_string()
        } else {
            format!("{}.{}", self.path.join("."), name)
        }
    }

    fn add(&mut self, name: &str, tensor: Tensor) -> Tensor {
        let path = self.path(name);
        let path = if self.var_store.variables.contains_key(&path) {
            self.var_store.counter = self.var_store.counter + 1;
            format!("{}__{}", path, self.var_store.counter)
        } else {
            path
        };
        self.var_store
            .variables
            .insert(path, tensor.shallow_clone());
        tensor
    }

    pub fn zeros_no_train(&mut self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.device()));
        self.add(name, z)
    }

    pub fn ones_no_train(&mut self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::ones(dims, (Kind::Float, self.device()));
        self.add(name, z)
    }

    pub fn zeros(&mut self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.device())).set_requires_grad(true);
        self.add(name, z)
    }

    pub fn ones(&mut self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::ones(dims, (Kind::Float, self.device())).set_requires_grad(true);
        self.add(name, z)
    }

    pub fn randn_standard(&mut self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::randn(dims, (Kind::Float, self.device())).set_requires_grad(true);
        self.add(name, z)
    }

    pub fn randn(&mut self, name: &str, dims: &[i64], mean: f64, stdev: f64) -> Tensor {
        let z = Tensor::randn(dims, (Kind::Float, self.device()));
        let z = (z * stdev + mean).set_requires_grad(true);
        self.add(name, z)
    }

    pub fn uniform(&mut self, name: &str, dims: &[i64], lo: f64, up: f64) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.device()))
            .uniform_(lo, up)
            .set_requires_grad(true);
        self.add(name, z)
    }

    pub fn kaiming_uniform(&mut self, name: &str, dims: &[i64]) -> Tensor {
        let fan_in: i64 = dims.iter().skip(1).product();
        let bound = (1.0 / fan_in as f64).sqrt();
        self.uniform(name, dims, -bound, bound)
    }
}

impl<'a> Div<&str> for &'a mut Path<'a> {
    type Output = Path<'a>;

    fn div(self, rhs: &str) -> Self::Output {
        self.sub(&rhs)
    }
}
