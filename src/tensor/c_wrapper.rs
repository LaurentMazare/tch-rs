use crate::utils::path_to_cstring;
use crate::{Device, Kind};
use failure::Fallible;
use libc::{c_char, c_int, c_void};
use std::path::Path;
use torch_sys::*;

/// A tensor object.
pub struct Tensor {
    pub(crate) c_tensor: *mut C_tensor,
}

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

    /// Creates a one-dimension tensor using some integer values.
    pub fn int_vec(v: &[i64]) -> Tensor {
        let c_tensor =
            unsafe_torch!({ at_int_vec(v.as_ptr(), v.len() as i32, Kind::Int64.c_int()) });
        Tensor { c_tensor }
    }

    /// Creates a one-dimension tensor using some float values.
    pub fn float_vec(v: &[f64]) -> Tensor {
        let c_tensor =
            unsafe_torch!({ at_float_vec(v.as_ptr(), v.len() as i32, Kind::Float.c_int()) });
        Tensor { c_tensor }
    }

    /// Returns the shape of the input tensor.
    pub fn size(&self) -> Vec<i64> {
        let dim = unsafe_torch!({ at_dim(self.c_tensor) as usize });
        let mut sz = vec![0i64; dim];
        unsafe_torch!({ at_shape(self.c_tensor, sz.as_mut_ptr()) });
        sz
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

    pub fn double_value(&self, idx: &[i32]) -> f64 {
        unsafe_torch!({ at_double_value_at_indexes(self.c_tensor, idx.as_ptr(), idx.len() as i32) })
    }

    pub fn int64_value(&self, idx: &[i32]) -> i64 {
        unsafe_torch!({ at_int64_value_at_indexes(self.c_tensor, idx.as_ptr(), idx.len() as i32) })
    }

    pub fn requires_grad(&self) -> bool {
        unsafe_torch!({ at_requires_grad(self.c_tensor) }) != 0
    }

    pub fn defined(&self) -> bool {
        unsafe_torch!({ at_defined(self.c_tensor) != 0 })
    }

    pub fn zero_grad(&mut self) {
        let grad = self.grad();
        if grad.defined() {
            let _ = grad.detach_().zero_();
        }
    }

    pub fn backward(&self) {
        unsafe_torch!({ at_backward(self.c_tensor, 0, 0) })
    }

    pub fn copy_data<T>(&self, dst: &mut [T], numel: i64) {
        let kind = self.kind();
        unsafe_torch!({
            at_copy_data(
                self.c_tensor,
                dst.as_mut_ptr() as *const c_void,
                numel,
                kind.elt_size_in_bytes(),
            )
        })
    }

    /// Returns the total number of elements stored in a tensor.
    pub fn numel(&self) -> i64 {
        self.size().iter().product()
    }

    // This is similar to vec_... but faster as it directly blits the data.
    pub fn of_data(data: &[u8], kind: Kind) -> Tensor {
        let data_len = data.len();
        let data = data.as_ptr() as *const c_void;
        let c_tensor = unsafe_torch!({
            at_tensor_of_data(
                data,
                [data_len as i64].as_ptr(),
                1,
                kind.elt_size_in_bytes(),
                kind.c_int(),
            )
        });
        Tensor { c_tensor }
    }

    /// Returns a new tensor that share storage with the input tensor.
    pub fn shallow_clone(&self) -> Tensor {
        let c_tensor = unsafe_torch!({ at_shallow_clone(self.c_tensor) });
        Tensor { c_tensor }
    }

    pub fn get(&self, index: i64) -> Tensor {
        let c_tensor = unsafe_torch!({ at_get(self.c_tensor, index as c_int) });
        Tensor { c_tensor }
    }

    /// Copies values from the argument tensor to the input tensor.
    pub fn copy_(&self, src: &Tensor) {
        unsafe_torch!({ at_copy_(self.c_tensor, src.c_tensor) })
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
    pub fn save_multi<T: AsRef<Path>>(named_tensors: &[(&str, &Tensor)], path: T) -> Fallible<()> {
        let path = path_to_cstring(path)?;
        let c_tensors = named_tensors
            .iter()
            .map(|nt| nt.1.c_tensor)
            .collect::<Vec<_>>();
        let names = named_tensors
            .iter()
            .map(|nt| std::ffi::CString::new(nt.0))
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
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe_torch!({ at_free(self.c_tensor) })
    }
}

fn grad_set_enabled(b: bool) -> bool {
    unsafe_torch!({ at_grad_set_enabled(if b { 1 } else { 0 }) != 0 })
}
pub fn no_grad<F>(f: F)
where
    F: FnOnce(),
{
    let prev = grad_set_enabled(false);
    f();
    let _false = grad_set_enabled(prev);
}

pub struct NoGradGuard {
    enabled: bool,
}

impl NoGradGuard {
    pub fn new() -> NoGradGuard {
        NoGradGuard {
            enabled: grad_set_enabled(false),
        }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        let _enabled = grad_set_enabled(self.enabled);
    }
}
