use std::collections::HashMap;
use std::ops::Div;
use std::sync::Mutex;

use failure::Fallible;

use crate::{tensor::Tensor, Device, Kind};

/// A VarStore is used to store variables used by one or multiple layers.
/// It specifies a single device where all variables are stored.
pub struct VarStore {
    variables: Mutex<HashMap<String, Tensor>>,
    device: Device,
}

pub struct Path<'a> {
    path: Vec<String>,
    var_store: &'a VarStore,
}

impl VarStore {
    pub fn new(device: Device) -> VarStore {
        VarStore {
            variables: Mutex::new(HashMap::new()),
            device,
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn trainable_variables(&self) -> Vec<Tensor> {
        let variables = self.variables.lock().expect("Poisoned MutexGuard");
        variables
            .values()
            .filter_map(|x| {
                if x.requires_grad() {
                    Some(x.shallow_clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn root(&self) -> Path {
        Path {
            path: vec![],
            var_store: self,
        }
    }

    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Fallible<()> {
        let variables = self.variables.lock().expect("Poisoned MutexGuard");
        let named_tensors = variables
            .iter()
            .map(|(x, y)| (&x[..], y))
            .collect::<Vec<_>>();
        Tensor::save_multi(named_tensors.as_slice(), path.as_ref())
    }

    pub fn load<P: AsRef<std::path::Path>>(&self, path: P) -> Fallible<()> {
        let named_tensors = Tensor::load_multi(path.as_ref())?;
        let named_tensors: HashMap<_, _> = named_tensors.into_iter().collect();
        let variables = self.variables.lock().expect("Poisoned MutexGuard");
        for (name, tensor) in variables.iter() {
            match named_tensors.get(name) {
                Some(src) => crate::no_grad(|| tensor.copy_(src)),
                None => Err(format_err!("cannot find {} in {:?}", name, path.as_ref()))?,
            }
        }
        Ok(())
    }
}

impl<'a> Path<'a> {
    pub fn sub(&'a self, s: &str) -> Path<'a> {
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
        if self.path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.path.join("."), name)
        }
    }

    fn add(&self, name: &str, tensor: Tensor) -> Tensor {
        let path = self.path(name);
        let mut variables = self.var_store.variables.lock().unwrap();
        let path = if variables.contains_key(&path) {
            format!("{}__{}", path, variables.len())
        } else {
            path
        };
        variables.insert(path, tensor.shallow_clone());
        tensor
    }

    pub fn zeros_no_train(&self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.device()));
        self.add(name, z)
    }

    pub fn ones_no_train(&self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::ones(dims, (Kind::Float, self.device()));
        self.add(name, z)
    }

    pub fn zeros(&self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.device())).set_requires_grad(true);
        self.add(name, z)
    }

    pub fn ones(&self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::ones(dims, (Kind::Float, self.device())).set_requires_grad(true);
        self.add(name, z)
    }

    pub fn randn_standard(&self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::randn(dims, (Kind::Float, self.device())).set_requires_grad(true);
        self.add(name, z)
    }

    pub fn randn(&self, name: &str, dims: &[i64], mean: f64, stdev: f64) -> Tensor {
        let z = Tensor::randn(dims, (Kind::Float, self.device()));
        let z = (z * stdev + mean).set_requires_grad(true);
        self.add(name, z)
    }

    pub fn uniform(&self, name: &str, dims: &[i64], lo: f64, up: f64) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.device()))
            .uniform_(lo, up)
            .set_requires_grad(true);
        self.add(name, z)
    }

    pub fn kaiming_uniform(&self, name: &str, dims: &[i64]) -> Tensor {
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

impl<'a> Div<&str> for &'a Path<'a> {
    type Output = Path<'a>;

    fn div(self, rhs: &str) -> Self::Output {
        self.sub(&rhs)
    }
}
