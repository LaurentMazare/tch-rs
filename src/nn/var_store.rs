//! Variable stores.
use super::Init;
use crate::tensor::Tensor;
use crate::{Device, Kind};
use failure::Fallible;
use std::collections::HashMap;
use std::ops::Div;
use std::sync::{Mutex, MutexGuard};

/// The separator is used to separate path elements in the tensor names.
const SEP: char = '|';

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

/// An Entry holds an entry corresponding to a given name in Path.
#[derive(Debug)]
pub struct Entry<'a> {
    name: &'a str,
    variables: MutexGuard<'a, HashMap<String, Variable>>, // This field holds the mutex lock
    path: &'a Path<'a>,
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

    /// Returns the number of tensors currently stored on this var-store.
    pub fn len(&self) -> usize {
        let variables = self.variables.lock().unwrap();
        variables.len()
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

    /// Gets the root path for this variable store.
    ///
    /// Variables are named and organized using paths. This function returns
    /// the top level path for the var store and can be combined with '/'
    /// to create sub-paths.
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
    pub fn load<T: AsRef<std::path::Path>>(&mut self, path: T) -> Fallible<()> {
        let named_tensors = Tensor::load_multi(&path)?;
        let named_tensors: HashMap<_, _> = named_tensors.into_iter().collect();
        let mut variables = self.variables.lock().unwrap();
        for (name, var) in variables.iter_mut() {
            match named_tensors.get(name) {
                Some(src) => crate::no_grad(|| {
                    var.tensor
                        .f_copy_(src)
                        .map_err(|e| format_err!("{}: {}", name, e))
                })?,
                None => return Err(format_err!("cannot find {} in {:?}", name, path.as_ref())),
            }
        }
        Ok(())
    }

    /// Freezes a var store.
    ///
    /// Gradients for the variables in this store are not tracked
    /// anymore.
    pub fn freeze(&mut self) {
        let variables = self.variables.lock().unwrap();
        for variable in variables.values() {
            if variable.trainable {
                let _v = variable.tensor.set_requires_grad(false);
            }
        }
    }

    /// Unfreezes a var store.
    ///
    /// Gradients for the variables in this store are tracked again.
    pub fn unfreeze(&mut self) {
        let variables = self.variables.lock().unwrap();
        for variable in variables.values() {
            if variable.trainable {
                let _v = variable.tensor.set_requires_grad(true);
            }
        }
    }

    /// Copies variable values from a source var store to this var store.
    ///
    /// All the variables in this var store have to exist with the same
    /// name in the source var store, otherwise an error is returned.
    pub fn copy(&mut self, src: &VarStore) -> Fallible<()> {
        let mut variables = self.variables.lock().unwrap();
        let src_variables = src.variables.lock().unwrap();
        let device = self.device;
        for name in variables.keys() {
            if !src_variables.contains_key(name) {
                bail!("cannot find {} in the source var store", name);
            }
        }
        for (name, var) in variables.iter_mut() {
            let src_var = src_variables.get(name).unwrap();
            crate::no_grad(|| var.tensor.f_copy_(&src_var.tensor.to_device(device)))?;
        }
        Ok(())
    }
}

impl<'a> Path<'a> {
    /// Gets a sub-path of the given path.
    pub fn sub<T: std::string::ToString>(&'a self, s: T) -> Path<'a> {
        let s = s.to_string();
        if s.chars().any(|x| x == SEP) {
            panic!("sub name cannot contain {} {}", SEP, s);
        }
        let mut path = self.path.clone();
        path.push(s);
        Path {
            path,
            var_store: self.var_store,
        }
    }

    /// Gets the device where the var-store variables are stored.
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

    fn get_or_add_with_lock(&self, name: &str, tensor: Tensor, trainable: bool, mut variables: MutexGuard<HashMap<String, Variable>>) -> Tensor {
        let path = self.path(name);
        if let Some(var) = variables.get(&path) {
            return var.tensor.shallow_clone();
        }

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

    /// Creates a new variable initialized with zeros.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable will not be trainable so
    /// gradients will not be tracked.
    /// The variable uses a float tensor initialized with zeros.
    pub fn zeros_no_train(&self, name: &str, dims: &[i64]) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.device()));
        self.add(name, z, false)
    }

    /// Creates a new variable initialized with ones.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable will not be trainable so
    /// gradients will not be tracked.
    /// The variable uses a float tensor initialized with ones.
    pub fn ones_no_train(&self, name: &str, dims: &[i64]) -> Tensor {
        let o = Tensor::ones(dims, (Kind::Float, self.device()));
        self.add(name, o, false)
    }

    /// Creates a new variable.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized as per the
    /// related argument.
    pub fn var(&self, name: &str, dims: &[i64], init: Init) -> Tensor {
        let v = super::init(init, dims, self.device());
        self.add(name, v, true)
    }

    /// Creates a new variable initialized with zeros.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized with zeros.
    pub fn zeros(&self, name: &str, dims: &[i64]) -> Tensor {
        self.var(name, dims, Init::Const(0.))
    }

    /// Creates a new variable initialized with ones.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized with ones.
    pub fn ones(&self, name: &str, dims: &[i64]) -> Tensor {
        self.var(name, dims, Init::Const(1.))
    }

    /// Creates a new variable initialized randomly with normal distribution.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// standard normal distribution.
    pub fn randn_standard(&self, name: &str, dims: &[i64]) -> Tensor {
        let init = Init::Randn {
            mean: 0.,
            stdev: 1.,
        };
        self.var(name, dims, init)
    }

    /// Creates a new variable initialized randomly with normal distribution.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// normal distribution with the specified mean and standard deviation.
    pub fn randn(&self, name: &str, dims: &[i64], mean: f64, stdev: f64) -> Tensor {
        self.var(name, dims, Init::Randn { mean, stdev })
    }

    /// Creates a new variable initialized randomly with uniform distribution.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// uniform distribution between the specified bounds.
    pub fn uniform(&self, name: &str, dims: &[i64], lo: f64, up: f64) -> Tensor {
        self.var(name, dims, Init::Uniform { lo, up })
    }

    /// Creates a new variable initialized randomly with kaiming uniform.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized randomly using a
    /// uniform distribution which bounds follow Kaiming initialization.
    pub fn kaiming_uniform(&self, name: &str, dims: &[i64]) -> Tensor {
        self.var(name, dims, Init::KaimingUniform)
    }

    /// Creates a new variable initialized by copying an existing tensor.
    ///
    /// The new variable is named according to the name parameter and
    /// has the specified shape. The variable is trainable, its gradient
    /// will be tracked.
    /// The variable uses a float tensor initialized by copying some
    /// given tensor.
    pub fn var_copy(&self, name: &str, t: &Tensor) -> Tensor {
        let mut v = self.zeros(name, &t.size());
        crate::no_grad(|| v.copy_(&t));
        v
    }

    /// Gets the tensor corresponding to a given name if present.
    pub fn get(&self, name: &str) -> Option<Tensor> {
        let path = self.path(name);
        let variables = self.var_store.variables.lock().unwrap();
        variables.get(&path).map(|v| v.tensor.shallow_clone())
    }

    /// Gets the entry corresponding to a given name for in-place manipulation.
    pub fn entry<'b>(&'b self, name: &'b str) -> Entry<'b> {
        let variables = self.var_store.variables.lock().unwrap();
        Entry {
            name,
            variables,
            path: &self,
        }
    }
}

impl<'a> Entry<'a> {
    pub fn or_var(self, dims: &[i64], init: Init) -> Tensor {
        let v = super::init(init, dims, self.path.device());
        self.path.get_or_add_with_lock(self.name, v, true, self.variables)
    }

    pub fn or_var_copy(self, tensor: &Tensor) -> Tensor {
        let mut v = self.or_zeros(&tensor.size());
        crate::no_grad(|| v.copy_(&tensor));
        v
    }

    pub fn or_kaiming_uniform(self, dims: &[i64]) -> Tensor {
        self.or_var(dims, Init::KaimingUniform)
    }

    pub fn or_ones(self, dims: &[i64]) -> Tensor {
        self.or_var(dims, Init::Const(1.))
    }

    pub fn or_ones_no_train(self, dims: &[i64]) -> Tensor {
        let o = Tensor::ones(dims, (Kind::Float, self.path.device()));
        self.path.get_or_add_with_lock(self.name, o, true, self.variables)
    }

    pub fn or_randn(self, dims: &[i64], mean: f64, stdev: f64) -> Tensor {
        self.or_var(dims, Init::Randn { mean, stdev })
    }

    pub fn or_randn_standard(self, dims: &[i64]) -> Tensor {
        let init = Init::Randn {
            mean: 0.,
            stdev: 1.,
        };
        self.or_var(dims, init)
    }

    pub fn or_uniform(self, dims: &[i64], lo: f64, up: f64) -> Tensor {
        self.or_var(dims, Init::Uniform { lo, up })
    }

    pub fn or_zeros(self, dims: &[i64]) -> Tensor {
        self.or_var(dims, Init::Const(0.))
    }

    pub fn or_zeros_no_train(self, dims: &[i64]) -> Tensor {
        let z = Tensor::zeros(dims, (Kind::Float, self.path.device()));
        self.path.get_or_add_with_lock(self.name, z, true, self.variables)
    }
}

impl<'a, T> Div<T> for &'a mut Path<'a>
where
    T: std::string::ToString,
{
    type Output = Path<'a>;

    fn div(self, rhs: T) -> Self::Output {
        self.sub(rhs.to_string())
    }
}

impl<'a, T> Div<T> for &'a Path<'a>
where
    T: std::string::ToString,
{
    type Output = Path<'a>;

    fn div(self, rhs: T) -> Self::Output {
        self.sub(rhs.to_string())
    }
}
