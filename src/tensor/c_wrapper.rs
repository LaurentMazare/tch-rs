use crate::device::Device;
use crate::kind::Kind;
use crate::scalar::{C_scalar, Scalar};
use crate::utils::read_and_clean_error;
use libc::{c_int, c_void};

#[repr(C)]
pub struct C_tensor {
    _private: [u8; 0],
}

extern "C" {
    fn at_new_tensor() -> *mut C_tensor;
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

    fn atg_randn(
        out: *mut *mut C_tensor,
        size: *const i64,
        size_len: c_int,
        kind: c_int,
        device: c_int,
    );
    fn atg_zeros(
        out: *mut *mut C_tensor,
        size: *const i64,
        size_len: c_int,
        kind: c_int,
        device: c_int,
    );
    fn atg_add1(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_scalar);
    fn atg_add_(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_tensor);
    fn atg_add_1(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_scalar);
    fn atg_mul1(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_scalar);
    fn atg_mul_(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_tensor);
    fn atg_mul_1(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_scalar);
    fn atg_log_softmax(out: *mut *mut C_tensor, arg: *mut C_tensor, dim: i64);
    fn atg_view(out: *mut *mut C_tensor, arg: *mut C_tensor, size: *const i64, size_len: c_int);
    fn atg_set_requires_grad(out: *mut *mut C_tensor, arg: *mut C_tensor, r: c_int);
    fn atg_detach_(out: *mut *mut C_tensor, arg: *mut C_tensor);
    fn atg_zero_(out: *mut *mut C_tensor, arg: *mut C_tensor);
    fn atg_totype(out: *mut *mut C_tensor, arg: *mut C_tensor, kind: c_int);
    fn atg_nll_loss(
        out: *mut *mut C_tensor,
        arg: *mut C_tensor,
        targets: *mut C_tensor,
        weights: *mut C_tensor,
        reduction: i64,
        ignore_index: i64,
    );
    fn atg_narrow(out: *mut *mut C_tensor, arg: *mut C_tensor, dim: i64, start: i64, len: i64);
    fn atg_argmax1(out: *mut *mut C_tensor, arg: *mut C_tensor, dim: i64, keepdim: c_int);
}

pub struct Tensor {
    c_tensor: *mut C_tensor,
}

macro_rules! binary_op {
    ($op:ident, $c_op:ident) => {
        extern "C" {
            fn $c_op(out: *mut *mut C_tensor, lhs: *mut C_tensor, rhs: *mut C_tensor);
        }

        impl Tensor {
            pub fn $op(&self, rhs: &Tensor) -> Tensor {
                let mut c_tensor = std::ptr::null_mut();
                unsafe { $c_op(&mut c_tensor, self.c_tensor, rhs.c_tensor) };
                read_and_clean_error();
                Tensor { c_tensor }
            }
        }
    };
}

macro_rules! unary_op {
    ($op:ident, $c_op:ident) => {
        extern "C" {
            fn $c_op(out: *mut *mut C_tensor, arg: *mut C_tensor);
        }

        impl Tensor {
            pub fn $op(&self) -> Tensor {
                let mut c_tensor = std::ptr::null_mut();
                unsafe { $c_op(&mut c_tensor, self.c_tensor) };
                read_and_clean_error();
                Tensor { c_tensor }
            }
        }
    };
}

binary_op!(add_tensor, atg_add);
binary_op!(mul_tensor, atg_mul);
binary_op!(mm, atg_matmul);
binary_op!(eq, atg_eq1);
unary_op!(grad, atg_grad);
unary_op!(mean, atg_mean);
unary_op!(sum, atg_sum);

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
        };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn zeros(size: &[i64], kind: Kind) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe {
            atg_zeros(
                &mut c_tensor,
                size.as_ptr(),
                size.len() as i32,
                kind.c_int(),
                Device::Cpu.c_int(),
            );
        };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn new() -> Tensor {
        let c_tensor = unsafe { at_new_tensor() };
        read_and_clean_error();
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

    pub fn add_scalar(&self, rhs: &Scalar) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_add1(&mut c_tensor, self.c_tensor, rhs.c_scalar) };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn add_tensor_(&self, rhs: &Tensor) {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_add_(&mut c_tensor, self.c_tensor, rhs.c_tensor) };
        read_and_clean_error()
    }

    pub fn add_scalar_(&self, rhs: &Scalar) {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_add_1(&mut c_tensor, self.c_tensor, rhs.c_scalar) };
        read_and_clean_error()
    }

    pub fn mul_scalar(&self, rhs: &Scalar) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_mul1(&mut c_tensor, self.c_tensor, rhs.c_scalar) };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn mul_tensor_(&self, rhs: &Tensor) {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_mul_(&mut c_tensor, self.c_tensor, rhs.c_tensor) };
        read_and_clean_error()
    }

    pub fn mul_scalar_(&self, rhs: &Scalar) {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_mul_1(&mut c_tensor, self.c_tensor, rhs.c_scalar) };
        read_and_clean_error()
    }

    // The same storage is shared so maybe this should consume self ?
    pub fn view(&self, size: &[i64]) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe {
            atg_view(
                &mut c_tensor,
                self.c_tensor,
                size.as_ptr(),
                size.len() as i32,
            )
        };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn to_kind(&self, kind: Kind) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_totype(&mut c_tensor, self.c_tensor, kind.c_int()) };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn kind(&self) -> Kind {
        let kind = unsafe { at_scalar_type(self.c_tensor) };
        read_and_clean_error();
        Kind::of_c_int(kind)
    }

    pub fn print(&self) {
        unsafe { at_print(self.c_tensor) };
        read_and_clean_error()
    }

    pub fn double_value(&self, idx: &[i32]) -> f64 {
        let v =
            unsafe { at_double_value_at_indexes(self.c_tensor, idx.as_ptr(), idx.len() as i32) };
        read_and_clean_error();
        v
    }

    pub fn int64_value(&self, idx: &[i32]) -> i64 {
        let v = unsafe { at_int64_value_at_indexes(self.c_tensor, idx.as_ptr(), idx.len() as i32) };
        read_and_clean_error();
        v
    }

    pub fn requires_grad(&self) -> bool {
        let r = unsafe { at_requires_grad(self.c_tensor) };
        read_and_clean_error();
        r != 0
    }

    pub fn defined(&self) -> bool {
        let defined = unsafe { at_defined(self.c_tensor) != 0 };
        read_and_clean_error();
        defined
    }

    pub fn zero_(self) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_zero_(&mut c_tensor, self.c_tensor) };
        read_and_clean_error();
        self
    }

    pub fn detach_(self) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_detach_(&mut c_tensor, self.c_tensor) };
        read_and_clean_error();
        self
    }

    pub fn zero_grad(&mut self) {
        let grad = self.grad();
        if grad.defined() {
            let _ = grad.detach_().zero_();
        }
    }

    pub fn backward(&self) {
        unsafe { at_backward(self.c_tensor, 0, 0) };
        read_and_clean_error()
    }

    // TODO: add some trait/generic version for this.
    pub fn copy_data(self, data: &[u8]) {
        let data_len = data.len();
        let data = data.as_ptr() as *const c_void;
        unsafe { at_copy_data(self.c_tensor, data, data_len as i64, 1) };
        read_and_clean_error()
    }

    // This is similar to vec_... but faster as it directly blits the data.
    pub fn of_data(data: &[u8]) -> Tensor {
        let data_len = data.len();
        let data = data.as_ptr() as *const c_void;
        let c_tensor = unsafe {
            at_tensor_of_data(data, [data_len as i64].as_ptr(), 1, 1, Kind::Uint8.c_int())
        };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn set_requires_grad(self, r: bool) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_set_requires_grad(&mut c_tensor, self.c_tensor, if r { 1 } else { 0 }) };
        read_and_clean_error();
        self
    }

    pub fn nll_loss(self, targets: &Tensor) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        let weights = Tensor::new();
        unsafe {
            atg_nll_loss(
                &mut c_tensor,
                self.c_tensor,
                targets.c_tensor,
                weights.c_tensor,
                1, // 0: no reduction, 1: mean, 2: sum
                -100,
            )
        };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn narrow(&self, dim: i64, start: i64, len: i64) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_narrow(&mut c_tensor, self.c_tensor, dim, start, len) };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn log_softmax(&self, dim: i64) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_log_softmax(&mut c_tensor, self.c_tensor, dim) };
        read_and_clean_error();
        Tensor { c_tensor }
    }

    pub fn argmax(&self, dim: i64) -> Tensor {
        let mut c_tensor = std::ptr::null_mut();
        unsafe { atg_argmax1(&mut c_tensor, self.c_tensor, dim, 0) };
        read_and_clean_error();
        Tensor { c_tensor }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { at_free(self.c_tensor) }
    }
}

fn grad_set_enabled(b: bool) -> bool {
    unsafe { at_grad_set_enabled(if b { 1 } else { 0 }) != 0 }
}
pub fn no_grad<F>(f: F)
where
    F: FnOnce(),
{
    let prev = grad_set_enabled(false);
    f();
    let _false = grad_set_enabled(prev);
}
