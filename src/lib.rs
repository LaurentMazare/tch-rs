extern crate libc;
use libc::c_int;

#[repr(C)] pub struct C_tensor { _private: [u8; 0] }
#[repr(C)] pub struct C_scalar { _private: [u8; 0] }

pub enum Kind {
    Uint8,
    Int8,
    Int16,
    Int,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
}

impl Kind {
    fn c_int(&self) -> c_int {
        match self {
            Kind::Uint8 => 0,
            Kind::Int8 => 1,
            Kind::Int16 => 2,
            Kind::Int => 3,
            Kind::Int64 => 4,
            Kind::Half => 5,
            Kind::Float => 6,
            Kind::Double => 7,
            Kind::ComplexHalf => 8,
            Kind::ComplexFloat => 9,
            Kind::ComplexDouble => 10,
        }
    }
}

pub enum Device {
    Cpu,
    Cuda,
}

impl Device {
    fn c_int(&self) -> c_int {
        match self {
            Device::Cpu => 0,
            Device::Cuda => 1,
        }
    }
}

extern  {
    fn get_and_reset_last_err() -> *mut libc::c_char;
    fn at_int_vec(v: *const i64, v_len: c_int, type_: c_int) -> *mut C_tensor;
    fn at_float_vec(v: *const f64, v_len: c_int, type_: c_int) -> *mut C_tensor;
    fn atg_randn(out: *mut *mut C_tensor, size: *const i64, size_len: c_int, kind: c_int, device: c_int);
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
        let torch_last_err = get_and_reset_last_err();
        if !torch_last_err.is_null() {
            let err =
                std::ffi::CStr::from_ptr(torch_last_err)
                    .to_string_lossy()
                    .into_owned();
            libc::free(torch_last_err as *mut libc::c_void);
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
    pub fn randn(size: &[i64], kind: Kind) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe {
            atg_randn(&mut c_tensor,
                      size.as_ptr(),
                      size.len() as i32,
                      kind.c_int(),
                      Device::Cpu.c_int());
            read_and_clean_error();
        };
        Tensor { c_tensor }
    }

    pub fn int_vec(v: &[i64]) -> Tensor {
        let c_tensor = unsafe {
            at_int_vec(v.as_ptr(), v.len() as i32, Kind::Int64.c_int())
        };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn float_vec(v: &[f64]) -> Tensor {
        let c_tensor = unsafe {
            at_float_vec(v.as_ptr(), v.len() as i32, Kind::Float.c_int())
        };
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

impl std::ops::Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Tensor {
        self.add_scalar(rhs.into())
    }
}

impl std::ops::Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Tensor {
        self.mul_scalar(rhs.into())
    }
}

impl std::ops::Add<i64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: i64) -> Tensor {
        self.add_scalar(rhs.into())
    }
}

impl std::ops::Mul<i64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: i64) -> Tensor {
        self.mul_scalar(rhs.into())
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

impl std::ops::AddAssign<Tensor> for Tensor {
    fn add_assign(&mut self, rhs: Tensor) {
        self.add_tensor_(rhs)
    }
}

impl std::ops::MulAssign<Tensor> for Tensor {
    fn mul_assign(&mut self, rhs: Tensor) {
        self.mul_tensor_(rhs)
    }
}

impl std::ops::AddAssign<f64> for Tensor {
    fn add_assign(&mut self, rhs: f64) {
        self.add_scalar_(rhs.into())
    }
}

impl std::ops::MulAssign<f64> for Tensor {
    fn mul_assign(&mut self, rhs: f64) {
        self.mul_scalar_(rhs.into())
    }
}

impl std::ops::AddAssign<i64> for Tensor {
    fn add_assign(&mut self, rhs: i64) {
        self.add_scalar_(rhs.into())
    }
}

impl std::ops::MulAssign<i64> for Tensor {
    fn mul_assign(&mut self, rhs: i64) {
        self.mul_scalar_(rhs.into())
    }
}

impl std::ops::AddAssign<Scalar> for Tensor {
    fn add_assign(&mut self, rhs: Scalar) {
        self.add_scalar_(rhs)
    }
}

impl std::ops::MulAssign<Scalar> for Tensor {
    fn mul_assign(&mut self, rhs: Scalar) {
        self.mul_scalar_(rhs)
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
