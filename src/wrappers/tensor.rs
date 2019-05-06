use super::utils::{path_to_cstring, ptr_to_string};
use super::{device::Device, kind, kind::Kind};
use failure::Fallible;
use libc::{c_char, c_int, c_void};
use std::borrow::Borrow;
use std::path::Path;
use torch_sys::*;

/// A tensor object.
#[must_use]
pub struct Tensor {
    pub(super) c_tensor: *mut C_tensor,
}

unsafe impl Send for Tensor {}

extern "C" fn add_callback(data: *mut c_void, name: *const c_char, c_tensor: *mut C_tensor) {
    let name = unsafe { std::ffi::CStr::from_ptr(name).to_str().unwrap() };
    let v: &mut Vec<(String, Tensor)> = unsafe { &mut *(data as *mut Vec<(String, Tensor)>) };
    v.push((name.to_owned(), Tensor { c_tensor }))
}

impl Tensor {
    /// Creates a new tensor.
    pub fn new() -> Tensor {
        let c_tensor = unsafe_torch!({ at_new_tensor() });
        Tensor { c_tensor }
    }

    /// Returns the number of dimension of the tensor.
    pub fn dim(&self) -> usize {
        unsafe_torch!({ at_dim(self.c_tensor) as usize })
    }

    /// Returns the shape of the input tensor.
    pub fn size(&self) -> Vec<i64> {
        let dim = unsafe_torch!({ at_dim(self.c_tensor) });
        let mut sz = vec![0i64; dim];
        unsafe_torch!({ at_shape(self.c_tensor, sz.as_mut_ptr()) });
        sz
    }

    /// Returns the tensor size for single dimension tensors.
    pub fn size1(&self) -> Fallible<i64> {
        match self.size().as_slice() {
            &[s0] => Ok(s0),
            size => bail!("expected one dim, got {:?}", size),
        }
    }

    /// Returns the tensor sizes for two dimension tensors.
    pub fn size2(&self) -> Fallible<(i64, i64)> {
        match self.size().as_slice() {
            &[s0, s1] => Ok((s0, s1)),
            size => bail!("expected two dims, got {:?}", size),
        }
    }

    /// Returns the tensor sizes for three dimension tensors.
    pub fn size3(&self) -> Fallible<(i64, i64, i64)> {
        match self.size().as_slice() {
            &[s0, s1, s2] => Ok((s0, s1, s2)),
            size => bail!("expected three dims, got {:?}", size),
        }
    }

    /// Returns the tensor sizes for four dimension tensors.
    pub fn size4(&self) -> Fallible<(i64, i64, i64, i64)> {
        match self.size().as_slice() {
            &[s0, s1, s2, s3] => Ok((s0, s1, s2, s3)),
            size => bail!("expected four dims, got {:?}", size),
        }
    }

    /// Returns the kind of elements stored in the input tensor.
    pub fn kind(&self) -> Kind {
        let kind = unsafe_torch!({ at_scalar_type(self.c_tensor) });
        Kind::of_c_int(kind)
    }

    /// Returns the device on which the input tensor is located.
    pub fn device(&self) -> Device {
        let device = unsafe_torch!({ at_device(self.c_tensor) });
        Device::of_c_int(device)
    }

    /// Prints the input tensor.
    ///
    /// Caution: this uses the C++ printer which prints the whole tensor even if
    /// it is very large.
    pub fn print(&self) {
        unsafe_torch!({ at_print(self.c_tensor) })
    }

    /// Returns a double value on tensors holding a single element. An error is
    /// returned otherwise.
    pub fn f_double_value(&self, idx: &[i64]) -> Fallible<f64> {
        Ok(unsafe_torch_err!({
            at_double_value_at_indexes(self.c_tensor, idx.as_ptr(), idx.len() as i32)
        }))
    }

    /// Returns an int value on tensors holding a single element. An error is
    /// returned otherwise.
    pub fn f_int64_value(&self, idx: &[i64]) -> Fallible<i64> {
        Ok(unsafe_torch!({
            at_int64_value_at_indexes(self.c_tensor, idx.as_ptr(), idx.len() as i32)
        }))
    }

    /// Returns a double value on tensors holding a single element. Panics otherwise.
    pub fn double_value(&self, idx: &[i64]) -> f64 {
        self.f_double_value(idx).unwrap()
    }

    /// Returns an int value on tensors holding a single element. Panics otherwise.
    pub fn int64_value(&self, idx: &[i64]) -> i64 {
        self.f_int64_value(idx).unwrap()
    }

    /// Returns true if gradient are currently tracked for this tensor.
    pub fn requires_grad(&self) -> bool {
        unsafe_torch!({ at_requires_grad(self.c_tensor) }) != 0
    }

    /// Returns true is the tensor is defined.
    pub fn defined(&self) -> bool {
        unsafe_torch!({ at_defined(self.c_tensor) != 0 })
    }

    /// Zeroes the gradient tensor attached to this tensor if defined.
    pub fn zero_grad(&mut self) {
        let mut grad = self.grad();
        if grad.defined() {
            let _ = grad.detach_().zero_();
        }
    }

    /// Runs the backward pass, populating the gradient tensors for tensors
    /// which gradients are tracked.
    ///
    /// Gradients tracking can be turned on via `set_requires_grad`.
    pub fn f_backward(&self) -> Fallible<()> {
        unsafe_torch_err!({ at_backward(self.c_tensor, 0, 0) });
        Ok(())
    }

    /// Runs the backward pass, populating the gradient tensors for tensors
    /// which gradients are tracked.
    ///
    /// Gradients tracking can be turned on via `set_requires_grad`.
    /// Panics if the C++ api returns an exception.
    pub fn backward(&self) {
        self.f_backward().unwrap()
    }

    pub fn f_run_backward<T1, T2>(
        tensors: &[T1],
        inputs: &[T2],
        keep_graph: bool,
        create_graph: bool,
    ) -> Fallible<Vec<Tensor>>
    where
        T1: Borrow<Tensor>,
        T2: Borrow<Tensor>,
    {
        let mut outputs = vec![std::ptr::null_mut(); inputs.len()];
        let tensors: Vec<_> = tensors.iter().map(|x| x.borrow().c_tensor).collect();
        let inputs: Vec<_> = inputs.iter().map(|x| x.borrow().c_tensor).collect();
        unsafe_torch_err!({
            at_run_backward(
                tensors.as_ptr(),
                tensors.len() as c_int,
                inputs.as_ptr(),
                inputs.len() as c_int,
                outputs.as_mut_ptr(),
                keep_graph as c_int,
                create_graph as c_int,
            )
        });
        Ok(outputs
            .into_iter()
            .map(|c_tensor| Tensor { c_tensor })
            .collect())
    }

    pub fn run_backward<T1, T2>(
        tensors: &[T1],
        inputs: &[T2],
        keep_graph: bool,
        create_graph: bool,
    ) -> Vec<Tensor>
    where
        T1: Borrow<Tensor>,
        T2: Borrow<Tensor>,
    {
        Tensor::f_run_backward(tensors, inputs, keep_graph, create_graph).unwrap()
    }

    /// Copies `numel` elements from `src` to this tensor.
    pub fn f_copy_data<T>(&self, dst: &mut [T], numel: i64) -> Fallible<()> {
        let kind = self.kind();
        unsafe_torch_err!({
            at_copy_data(
                self.c_tensor,
                dst.as_mut_ptr() as *const c_void,
                numel,
                kind.elt_size_in_bytes(),
            )
        });
        Ok(())
    }

    /// Copies `numel` elements from `src` to this tensor.
    pub fn copy_data<T>(&self, dst: &mut [T], numel: i64) {
        self.f_copy_data(dst, numel).unwrap()
    }

    /// Returns the total number of elements stored in a tensor.
    pub fn numel(&self) -> i64 {
        self.size().iter().product()
    }

    // This is similar to vec_... but faster as it directly blits the data.
    /// Converts a slice to a tensor.
    pub fn f_of_slice<T: kind::T>(data: &[T]) -> Fallible<Tensor> {
        let data_len = data.len();
        let data = data.as_ptr() as *const c_void;
        let c_tensor = unsafe_torch_err!({
            at_tensor_of_data(
                data,
                [data_len as i64].as_ptr(),
                1,
                T::KIND.elt_size_in_bytes(),
                T::KIND.c_int(),
            )
        });
        Ok(Tensor { c_tensor })
    }

    /// Converts a slice to a tensor.
    pub fn of_slice<T: kind::T>(data: &[T]) -> Tensor {
        Self::f_of_slice(data).unwrap()
    }

    /// Converts some byte data to a tensor with some specified kind and shape.
    pub fn f_of_data_size(data: &[u8], size: &[i64], kind: Kind) -> Fallible<Tensor> {
        let data = data.as_ptr() as *const c_void;
        let elt_size_in_bytes = kind.elt_size_in_bytes();
        let c_tensor = unsafe_torch_err!({
            at_tensor_of_data(
                data,
                size.as_ptr(),
                size.len() as i64,
                elt_size_in_bytes,
                kind.c_int(),
            )
        });
        Ok(Tensor { c_tensor })
    }

    /// Converts some byte data to a tensor with some specified kind and shape.
    pub fn of_data_size(data: &[u8], size: &[i64], kind: Kind) -> Tensor {
        Self::f_of_data_size(data, size, kind).unwrap()
    }

    /// Returns a new tensor that share storage with the input tensor.
    pub fn shallow_clone(&self) -> Tensor {
        let c_tensor = unsafe_torch!({ at_shallow_clone(self.c_tensor) });
        Tensor { c_tensor }
    }

    /// Gets the sub-tensor at the given index.
    pub fn f_get(&self, index: i64) -> Fallible<Tensor> {
        let c_tensor = unsafe_torch_err!({ at_get(self.c_tensor, index as c_int) });
        Ok(Tensor { c_tensor })
    }

    /// Gets the sub-tensor at the given index.
    pub fn get(&self, index: i64) -> Tensor {
        self.f_get(index).unwrap()
    }

    /// Copies values from the argument tensor to the input tensor.
    pub fn f_copy_(&mut self, src: &Tensor) -> Fallible<()> {
        unsafe_torch_err!({ at_copy_(self.c_tensor, src.c_tensor) });
        Ok(())
    }

    /// Copies values from the argument tensor to the input tensor.
    pub fn copy_(&mut self, src: &Tensor) {
        self.f_copy_(src).unwrap()
    }

    /// Loads a tensor from a file.
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn load<T: AsRef<Path>>(path: T) -> Fallible<Tensor> {
        let path = path_to_cstring(path)?;
        let c_tensor = unsafe_torch_err!({ at_load(path.as_ptr()) });
        Ok(Tensor { c_tensor })
    }

    /// Saves a tensor to a file.
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn save<T: AsRef<Path>>(&self, path: T) -> Fallible<()> {
        let path = path_to_cstring(path)?;
        unsafe_torch_err!({ at_save(self.c_tensor, path.as_ptr()) });
        Ok(())
    }

    /// Saves some named tensors to a file
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn save_multi<S: AsRef<str>, T: AsRef<Tensor>, P: AsRef<Path>>(
        named_tensors: &[(S, T)],
        path: P,
    ) -> Fallible<()> {
        let path = path_to_cstring(path)?;
        let c_tensors = named_tensors
            .iter()
            .map(|nt| nt.1.as_ref().c_tensor)
            .collect::<Vec<_>>();
        let names = named_tensors
            .iter()
            .map(|nt| std::ffi::CString::new(nt.0.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let name_ptrs = names.iter().map(|n| n.as_ptr()).collect::<Vec<_>>();
        unsafe_torch_err!({
            at_save_multi(
                c_tensors.as_ptr(),
                name_ptrs.as_ptr(),
                names.len() as i32,
                path.as_ptr(),
            )
        });
        Ok(())
    }

    /// Loads some named tensors from a file
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn load_multi<T: AsRef<Path>>(path: T) -> Fallible<Vec<(String, Tensor)>> {
        let path = path_to_cstring(path)?;
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!({
            at_load_callback(path.as_ptr(), &mut v as *mut _ as *mut c_void, add_callback)
        });
        Ok(v)
    }

    /// Returns a string representation for the tensor.
    ///
    /// The representation will contain all the tensor element hence may be huge for
    /// large tensors.
    pub fn to_string(&self, lw: i64) -> Fallible<String> {
        let s = unsafe_torch_err!({
            ptr_to_string(torch_sys::at_to_string(self.c_tensor, lw as c_int))
        });
        match s {
            None => bail!("nullptr representation"),
            Some(s) => Ok(s),
        }
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe_torch!({ at_free(self.c_tensor) })
    }
}

fn grad_set_enabled(b: bool) -> bool {
    unsafe_torch!({ at_grad_set_enabled(if b { 1 } else { 0 }) != 0 })
}

/// Runs a closure without keeping track of gradients.
pub fn no_grad<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let prev = grad_set_enabled(false);
    let result = f();
    let _false = grad_set_enabled(prev);
    result
}

/// A RAII guard that prevents gradient tracking until deallocated.
pub struct NoGradGuard {
    enabled: bool,
}

/// Disables gradient tracking, this will be enabled back when the
/// returned value gets deallocated.
pub fn no_grad_guard() -> NoGradGuard {
    NoGradGuard {
        enabled: grad_set_enabled(false),
    }
}

impl std::convert::AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Self {
        &self
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        let _enabled = grad_set_enabled(self.enabled);
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Reduction {
    /// Do not reduce.
    None,
    /// Mean of losses.
    Mean,
    /// Sum of losses.
    Sum,
}

impl Reduction {
    // This has to stay in sync with
    // pytorch/aten/src/ATen/core/Reduction.h
    pub fn to_int(&self) -> i64 {
        match self {
            Reduction::None => 0,
            Reduction::Mean => 1,
            Reduction::Sum => 2,
        }
    }
}
