extern crate libc;
use libc::c_int;

#[repr(C)] pub struct C_tensor { _private: [u8; 0] }

extern  {
    fn at_int_vec(v: *const i64, v_len: c_int, type_: c_int) -> *mut C_tensor;
    fn atg_randn(out: *mut *mut C_tensor, size: *const i64, size_len: c_int, kind: c_int, device: c_int);
    fn at_print(arg: *mut C_tensor);
    fn at_free(arg: *mut C_tensor);
}

pub struct Tensor {
    c_tensor : *mut C_tensor
}

impl Tensor {
    pub fn randn(size: &[i64]) -> Tensor {
        unsafe {
            let mut c_tensor = std::ptr::null_mut();
            atg_randn(&mut c_tensor, size.as_ptr(), size.len() as i32, 6, 0);
            Tensor { c_tensor }
        }
    }

    pub fn int_vec(v: &[i64]) -> Tensor {
        unsafe {
            Tensor { c_tensor: at_int_vec(v.as_ptr(), v.len() as i32, 4) }
        }
    }

    pub fn print(&self) {
        unsafe {
            at_print(self.c_tensor)
        }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            at_free(self.c_tensor)
        }
    }
}

impl From<&[i64]> for Tensor {
    fn from(v: &[i64]) -> Tensor {
        Tensor::int_vec(v)
    }
}
