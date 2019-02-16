extern crate libc;
use libc::c_int;

#[repr(C)] pub struct C_tensor { _private: [u8; 0] }
#[repr(C)] pub struct C_scalar { _private: [u8; 0] }

extern  {
    // This is not thread-safe but neither is libtorch in general
    // A thread-local variable would be better.
    static mut torch_last_err: *mut libc::c_char;

    fn at_int_vec(v: *const i64, v_len: c_int, type_: c_int) -> *mut C_tensor;
    fn at_float_vec(v: *const f64, v_len: c_int, type_: c_int) -> *mut C_tensor;
    fn atg_randn(out: *mut *mut C_tensor, size: *const i64, size_len: c_int, kind: c_int, device: c_int);
    fn atg_add(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_tensor);
    fn atg_add1(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_scalar);
    fn atg_mul(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_tensor);
    fn atg_mul1(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_scalar);
    fn at_print(arg: *mut C_tensor);
    fn at_free(arg: *mut C_tensor);

    fn ats_int(v: i64) -> *mut C_scalar;
    fn ats_float(v: f64) -> *mut C_scalar;
    fn ats_free(arg: *mut C_scalar);
}

pub struct Scalar {
    c_scalar : *mut C_scalar
}

pub struct Tensor {
    c_tensor : *mut C_tensor
}

fn read_and_clean_error() {
    unsafe {
        if !torch_last_err.is_null() {
            let err =
                std::ffi::CStr::from_ptr(torch_last_err)
                    .to_string_lossy()
                    .into_owned();
            libc::free(torch_last_err as *mut libc::c_void);
            torch_last_err = std::ptr::null_mut();
            panic!(err)
        }
    }
}

impl Scalar {
    pub fn int(v: i64) -> Scalar {
        unsafe {
            let c_scalar = ats_int(v);
            read_and_clean_error();
            Scalar { c_scalar }
        }
    }

    pub fn float(v: f64) -> Scalar {
        unsafe {
            let c_scalar = ats_float(v);
            read_and_clean_error();
            Scalar { c_scalar }
        }
    }
}

impl Drop for Scalar {
    fn drop(&mut self) {
        unsafe {
            ats_free(self.c_scalar)
        }
    }
}

impl From<i64> for Scalar {
    fn from(v: i64) -> Scalar {
        Scalar::int(v)
    }
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Scalar {
        Scalar::float(v)
    }
}

impl Tensor {
    pub fn randn(size: &[i64]) -> Tensor {
        unsafe {
            let mut c_tensor = std::ptr::null_mut();
            atg_randn(&mut c_tensor, size.as_ptr(), size.len() as i32, 6, 0);
            read_and_clean_error();
            Tensor { c_tensor }
        }
    }

    pub fn int_vec(v: &[i64]) -> Tensor {
        unsafe {
            let c_tensor = at_int_vec(v.as_ptr(), v.len() as i32, 4);
            read_and_clean_error();
            Tensor { c_tensor }
        }
    }

    pub fn float_vec(v: &[f64]) -> Tensor {
        unsafe {
            let c_tensor = at_float_vec(v.as_ptr(), v.len() as i32, 4);
            read_and_clean_error();
            Tensor { c_tensor }
        }
    }

    pub fn add_tensor(&self, rhs: Tensor) -> Tensor {
        unsafe {
            let mut c_tensor = std::ptr::null_mut();
            atg_add(&mut c_tensor, self.c_tensor, rhs.c_tensor);
            read_and_clean_error();
            Tensor { c_tensor }
        }
    }

    pub fn add_scalar(&self, rhs: Scalar) -> Tensor {
        unsafe {
            let mut c_tensor = std::ptr::null_mut();
            atg_add1(&mut c_tensor, self.c_tensor, rhs.c_scalar);
            read_and_clean_error();
            Tensor { c_tensor }
        }
    }

    pub fn mul_tensor(&self, rhs: Tensor) -> Tensor {
        unsafe {
            let mut c_tensor = std::ptr::null_mut();
            atg_mul(&mut c_tensor, self.c_tensor, rhs.c_tensor);
            read_and_clean_error();
            Tensor { c_tensor }
        }
    }

    pub fn mul_scalar(&self, rhs: Scalar) -> Tensor {
        unsafe {
            let mut c_tensor = std::ptr::null_mut();
            atg_mul1(&mut c_tensor, self.c_tensor, rhs.c_scalar);
            read_and_clean_error();
            Tensor { c_tensor }
        }
    }

    pub fn print(&self) {
        unsafe {
            at_print(self.c_tensor);
            read_and_clean_error()
        }
    }
}

impl std::ops::Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Tensor {
        self.add_tensor(rhs)
    }
}

impl std::ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        self.mul_tensor(rhs)
    }
}

impl std::ops::Add<Scalar> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Scalar) -> Tensor {
        self.add_scalar(rhs)
    }
}

impl std::ops::Mul<Scalar> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Scalar) -> Tensor {
        self.mul_scalar(rhs)
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

impl From<&[f64]> for Tensor {
    fn from(v: &[f64]) -> Tensor {
        Tensor::float_vec(v)
    }
}
