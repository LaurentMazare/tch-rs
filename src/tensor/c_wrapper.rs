use crate::device::Device;
use crate::kind::Kind;
use crate::scalar::{C_scalar, Scalar};
use crate::utils::read_and_clean_error;
use libc::c_int;

#[repr(C)]
pub struct C_tensor {
    _private: [u8; 0],
}

extern "C" {
    fn at_int_vec(v: *const i64, v_len: c_int, type_: c_int) -> *mut C_tensor;
    fn at_float_vec(v: *const f64, v_len: c_int, type_: c_int) -> *mut C_tensor;
    fn atg_randn(
        out: *mut *mut C_tensor,
        size: *const i64,
        size_len: c_int,
        kind: c_int,
        device: c_int,
    );
    fn atg_add(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_tensor);
    fn atg_add1(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_scalar);
    fn atg_add_(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_tensor);
    fn atg_add_1(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_scalar);
    fn atg_mul(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_tensor);
    fn atg_mul1(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_scalar);
    fn atg_mul_(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_tensor);
    fn atg_mul_1(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_scalar);
    fn at_print(arg: *mut C_tensor);
    fn at_dim(arg: *mut C_tensor) -> c_int;
    fn at_shape(arg: *mut C_tensor, sz: *mut c_int);
    fn at_double_value_at_indexes(arg: *mut C_tensor, idx: *const c_int, idx_len: c_int) -> f64;
    fn at_int64_value_at_indexes(arg: *mut C_tensor, idx: *const c_int, idx_len: c_int) -> i64;
    fn at_free(arg: *mut C_tensor);

}

pub struct Tensor {
    c_tensor: *mut C_tensor,
}

impl Tensor {
    pub fn randn(size: &[i64], kind: Kind) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe {
            atg_randn(
                &mut c_tensor,
                size.as_ptr(),
                size.len() as i32,
                kind.c_int(),
                Device::Cpu.c_int(),
            );
            read_and_clean_error();
        };
        Tensor { c_tensor }
    }

    pub fn int_vec(v: &[i64]) -> Tensor {
        let c_tensor = unsafe { at_int_vec(v.as_ptr(), v.len() as i32, Kind::Int64.c_int()) };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn float_vec(v: &[f64]) -> Tensor {
        let c_tensor = unsafe { at_float_vec(v.as_ptr(), v.len() as i32, Kind::Float.c_int()) };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn size(&self) -> Vec<i32> {
        let dim = unsafe { at_dim(self.c_tensor) as usize };
        read_and_clean_error();
        let mut sz = vec![0i32; dim];
        unsafe { at_shape(self.c_tensor, sz.as_mut_ptr()) };
        read_and_clean_error();
        sz
    }

    pub fn add_tensor(&self, rhs: Tensor) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_add(&mut c_tensor, self.c_tensor, rhs.c_tensor) };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn add_scalar(&self, rhs: Scalar) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_add1(&mut c_tensor, self.c_tensor, rhs.c_scalar) };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn add_tensor_(&self, rhs: Tensor) {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_add_(&mut c_tensor, self.c_tensor, rhs.c_tensor) };
        read_and_clean_error()
    }

    pub fn add_scalar_(&self, rhs: Scalar) {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_add_1(&mut c_tensor, self.c_tensor, rhs.c_scalar) };
        read_and_clean_error()
    }

    pub fn mul_tensor(&self, rhs: Tensor) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_mul(&mut c_tensor, self.c_tensor, rhs.c_tensor) };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn mul_scalar(&self, rhs: Scalar) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_mul1(&mut c_tensor, self.c_tensor, rhs.c_scalar) };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn mul_tensor_(&self, rhs: Tensor) {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_mul_(&mut c_tensor, self.c_tensor, rhs.c_tensor) };
        read_and_clean_error()
    }

    pub fn mul_scalar_(&self, rhs: Scalar) {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_mul_1(&mut c_tensor, self.c_tensor, rhs.c_scalar) };
        read_and_clean_error()
    }

    pub fn print(&self) {
        unsafe { at_print(self.c_tensor) };
        read_and_clean_error()
    }

    pub fn double_value(&self, idx: &[i32]) -> f64 {
        unsafe { at_double_value_at_indexes(self.c_tensor, idx.as_ptr(), idx.len() as i32) }
    }

    pub fn int64_value(&self, idx: &[i32]) -> i64 {
        unsafe { at_int64_value_at_indexes(self.c_tensor, idx.as_ptr(), idx.len() as i32) }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { at_free(self.c_tensor) }
    }
}
