//! Variable stores.
use crate::tensor::Tensor;
use crate::{Device, Kind};
use failure::Fallible;
use std::collections::HashMap;
use std::ops::Div;
use std::sync::Mutex;

/// The separator is used to separate path elements in the tensor names.
const SEP: char = '|';

#[derive(Debug, Copy, Clone)]
pub enum Init {
    Const(f64),
    Randn { mean: f64, std: f64 },
    Uniform { lo: f64, up: f64 },
    KaimingUniform,
}

// When the variable store is frozen, trainable still is set to tree,
// however the tensor is not set to require gradients.
#[derive(Debug)]
struct Variable {
    tensor: Tensor,
    trainable: bool,
}

/// A VarStore is used to store variables used by one or multiple layers.
/// It specifies a single device where all variables are stored.
#[derive(Debug)]
pub struct VarStore {
    variables: Mutex<HashMap<String, Variable>>,
    device: Device,
}

/// A variable store with an associated path for variables naming.
#[derive(Debug)]
pub struct Path<'a> {
    path: Vec<String>,
    var_store: &'a VarStore,
}

impl VarStore {
    /// Creates a new var-store located on the specified device.
    pub fn new(device: Device) -> VarStore {
        VarStore {
            variables: Mutex::new(HashMap::new()),
            device,
        }
    }

    /// Gets the device for this var-store.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns all the trainable variables for this var-store.
    pub fn trainable_variables(&self) -> Vec<Tensor> {
        let variables = self.variables.lock().unwrap();
        variables
            .values()
            .filter_map(|v| {
                if v.trainable {
                    Some(v.tensor.shallow_clone())
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

    /// Saves the var-store variable values to a file.
    pub fn save<T: AsRef<std::path::Path>>(&self, path: T) -> Fallible<()> {
        let variables = self.variables.lock().unwrap();
        let named_tensors = variables
            .iter()
            .map(|(x, y)| (&x[..], &y.tensor))
            .collect::<Vec<_>>();
        Tensor::save_multi(named_tensors.as_slice(), path)
    }

    /// Loads the var-store variable values from a file.
    pub fn load<T: AsRef<std::path::Path>>(&self, path: T) -> Fallible<()> {
        let named_tensors = Tensor::load_multi(&path)?;
        let named_tensors: HashMap<_, _> = named_tensors.into_iter().collect();
        let variables = self.variables.lock().unwrap();
        for (name, var) in variables.iter() {
            match named_tensors.get(name) {
                Some(src) => crate::no_grad(|| var.tensor.copy_(src)),
                None => Err(format_err!("cannot find {} in {:?}", name, path.as_ref()))?,
            }
        }
        Ok(())
    }

    pub fn freeze(&mut self) {
        let variables = self.variables.lock().unwrap();
        for variable in variables.values() {
            if variable.trainable {
                let _v = variable.tensor.set_requires_grad(false);
            }
        }
    }

    pub fn unfreeze(&mut self) {
        let variables = self.variables.lock().unwrap();
        for variable in variables.values() {
            if variable.trainable {
                let _v = variable.tensor.set_requires_grad(true);
            }
        }
    }
}

impl<'a> Path<'a> {
    pub fn sub(&'a self, s: &str) -> Path<'a> {
        if s.chars().any(|x| x == SEP) {
            panic!("sub name cannot contain {} {}", SEP, s);
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
        if name.chars().any(|x| x == SEP) {
            panic!("variable name cannot contain {} {}", SEP, name);
        }
        if self.path.is_empty() {
            name.to_string()
        } else {
            format!("{}{}{}", self.path.join(&SEP.to_string()), SEP, name)
        }
    }

    fn add(&self, name: &str, tensor: Tensor, trainable: bool) -> Tensor {
        let path = self.path(name);
        let mut variables = self.var_store.variables.lock().unwrap();
        let path = if variables.contains_key(&path) {
            format!("{}__{}", path, variables.len())
        } else {
            path
        };
        let tensor = if trainable {
            tensor.set_requires_grad(true)
        } else {
            tensor
        };
        let var = Variable {
            tensor: tensor.shallow_clone(),
            trainable,
        };
        variables.insert(path, var);
        tensor
    }

    pub fn zeros_no_train(&self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.device()));
        self.add(name, z, false)
    }

    pub fn ones_no_train(&self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::ones(dims, (Kind::Float, self.device()));
        self.add(name, z, false)
    }

    pub fn zeros(&self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.device()));
        self.add(name, z, true)
    }

    pub fn ones(&self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::ones(dims, (Kind::Float, self.device()));
        self.add(name, z, true)
    }

    pub fn randn_standard(&self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::randn(dims, (Kind::Float, self.device()));
        self.add(name, z, true)
    }

    pub fn randn(&self, name: &str, dims: &[i64], mean: f64, stdev: f64) -> Tensor {
        let z = Tensor::randn(dims, (Kind::Float, self.device()));
        self.add(name, z * stdev + mean, true)
    }

    pub fn uniform(&self, name: &str, dims: &[i64], lo: f64, up: f64) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.device())).uniform_(lo, up);
        self.add(name, z, true)
    }

    pub fn kaiming_uniform(&self, name: &str, dims: &[i64]) -> Tensor {
        let fan_in: i64 = dims.iter().skip(1).product();
        let bound = (1.0 / fan_in as f64).sqrt();
        self.uniform(name, dims, -bound, bound)
    }

    pub fn var(&self, name: &str, dims: &[i64], init: Init) -> Tensor {
        match init {
            Init::Const(cst) => {
                // Optimize the case for which a single C++ code can be done.
                if cst == 0. {
                    self.zeros(name, dims)
                } else if cst == 1. {
                    self.ones(name, dims)
                } else {
                    self.ones(name, dims) * cst
                }
            }
            Init::Uniform { lo, up } => self.uniform(name, dims, lo, up),
            Init::Randn { mean, std } => {
                if mean == 0. && std == 1. {
                    self.randn_standard(name, dims)
                } else {
                    self.randn(name, dims, mean, std)
                }
            }
            Init::KaimingUniform => self.kaiming_uniform(name, dims),
        }
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
