use crate::kind::Kind;
use crate::utils::{read_and_clean_error, TorchError};
use libc::{c_char, c_int, c_void};

#[repr(C)]
pub(crate) struct C_tensor {
    _private: [u8; 0],
}

extern "C" {
    fn at_new_tensor() -> *mut C_tensor;
    fn at_shallow_clone(arg: *mut C_tensor) -> *mut C_tensor;
    fn at_copy_(dst: *mut C_tensor, src: *mut C_tensor);
    fn at_int_vec(v: *const i64, v_len: c_int, type_: c_int) -> *mut C_tensor;
    fn at_float_vec(v: *const f64, v_len: c_int, type_: c_int) -> *mut C_tensor;
    fn at_defined(arg: *mut C_tensor) -> c_int;
    fn at_backward(arg: *mut C_tensor, keep_graph: c_int, create_graph: c_int);
    fn at_print(arg: *mut C_tensor);
    fn at_dim(arg: *mut C_tensor) -> c_int;
    fn at_requires_grad(arg: *mut C_tensor) -> c_int;
    fn at_shape(arg: *mut C_tensor, sz: *mut c_int);
    fn at_double_value_at_indexes(arg: *mut C_tensor, idx: *const c_int, idx_len: c_int) -> f64;
    fn at_int64_value_at_indexes(arg: *mut C_tensor, idx: *const c_int, idx_len: c_int) -> i64;
    fn at_free(arg: *mut C_tensor);
    fn at_copy_data(arg: *mut C_tensor, vs: *const c_void, numel: i64, elt_size_in_bytes: c_int);
    fn at_scalar_type(arg: *mut C_tensor) -> c_int;
    fn at_tensor_of_data(
        vs: *const c_void,
        dims: *const i64,
        ndims: i64,
        elt_size_in_bytes: c_int,
        kind: c_int,
    ) -> *mut C_tensor;
    fn at_grad_set_enabled(b: c_int) -> c_int;
    fn at_save(arg: *mut C_tensor, filename: *const c_char);
    fn at_load(filename: *const c_char) -> *mut C_tensor;
}

pub struct Tensor {
    pub(crate) c_tensor: *mut C_tensor,
}

fn path_to_str(path: &std::path::Path) -> Result<&str, TorchError> {
    match path.to_str() {
        Some(path) => Ok(path),
        None => Err(TorchError::new(format!("path {:?} is none", path))),
    }
}

impl Tensor {
    pub fn new() -> Tensor {
        let c_tensor = unsafe_torch!({ at_new_tensor() });
        Tensor { c_tensor }
    }

    pub fn int_vec(v: &[i64]) -> Tensor {
        let c_tensor =
            unsafe_torch!({ at_int_vec(v.as_ptr(), v.len() as i32, Kind::Int64.c_int()) });
        Tensor { c_tensor }
    }

    pub fn float_vec(v: &[f64]) -> Tensor {
        let c_tensor =
            unsafe_torch!({ at_float_vec(v.as_ptr(), v.len() as i32, Kind::Float.c_int()) });
        Tensor { c_tensor }
    }

    pub fn size(&self) -> Vec<i32> {
        let dim = unsafe_torch!({ at_dim(self.c_tensor) as usize });
        let mut sz = vec![0i32; dim];
        unsafe_torch!({ at_shape(self.c_tensor, sz.as_mut_ptr()) });
        sz
    }

    pub fn kind(&self) -> Kind {
        let kind = unsafe_torch!({ at_scalar_type(self.c_tensor) });
        Kind::of_c_int(kind)
    }

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

    pub fn numel(&self) -> i64 {
        self.size().iter().fold(1, |acc, &v| acc * i64::from(v))
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

    pub fn shallow_clone(&self) -> Tensor {
        let c_tensor = unsafe_torch!({ at_shallow_clone(self.c_tensor) });
        Tensor { c_tensor }
    }

    pub fn copy_(&self, src: &Tensor) {
        unsafe_torch!({ at_copy_(self.c_tensor, src.c_tensor) })
    }

    pub fn load(path: &std::path::Path) -> Result<Tensor, TorchError> {
        let path = std::ffi::CString::new(path_to_str(path)?)?;
        let c_tensor = unsafe_torch_err!({ at_load(path.as_ptr()) });
        Ok(Tensor { c_tensor })
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), TorchError> {
        let path = std::ffi::CString::new(path_to_str(path)?)?;
        unsafe_torch_err!({ at_save(self.c_tensor, path.as_ptr()) });
        Ok(())
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
