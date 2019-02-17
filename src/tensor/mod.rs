use crate::scalar::Scalar;
use std::ops::{Add, AddAssign, Mul, MulAssign};

mod c_wrapper;

pub use c_wrapper::{no_grad, Tensor};

macro_rules! impl_op {
    ($trait:ident, $rhs:ident, $func:ident, $op:ident) => {
        impl $trait<$rhs> for Tensor {
            type Output = Tensor;

            fn $func(self, rhs: $rhs) -> Self::Output {
                self.$op(&rhs)
            }
        }

        impl $trait<&$rhs> for Tensor {
            type Output = Tensor;

            fn $func(self, rhs: &$rhs) -> Self::Output {
                self.$op(rhs)
            }
        }

        impl<'a> $trait<&$rhs> for &'a Tensor {
            type Output = Tensor;

            fn $func<'b>(self, rhs: &'b $rhs) -> Self::Output {
                self.$op(rhs)
            }
        }

        impl $trait<$rhs> for &Tensor {
            type Output = Tensor;

            fn $func(self, rhs: $rhs) -> Self::Output {
                self.$op(&rhs)
            }
        }
    };
}

macro_rules! impl_op_basic {
    ($trait:ident, $func:ident, $op:ident) => {
        impl $trait<i64> for Tensor {
            type Output = Tensor;

            fn $func(self, rhs: i64) -> Self::Output {
                self.$op(&rhs.into())
            }
        }

        impl $trait<f64> for Tensor {
            type Output = Tensor;

            fn $func(self, rhs: f64) -> Self::Output {
                self.$op(&rhs.into())
            }
        }

        impl<'a> $trait<i64> for &'a Tensor {
            type Output = Tensor;

            fn $func(self, rhs: i64) -> Self::Output {
                self.$op(&rhs.into())
            }
        }

        impl $trait<f64> for &Tensor {
            type Output = Tensor;

            fn $func(self, rhs: f64) -> Self::Output {
                self.$op(&rhs.into())
            }
        }
    };
}

macro_rules! impl_op_assign {
    ($trait:ident, $rhs:ident, $func:ident, $op:ident) => {
        impl $trait<$rhs> for Tensor {
            fn $func(&mut self, rhs: $rhs) {
                self.$op(&rhs)
            }
        }

        impl $trait<&$rhs> for Tensor {
            fn $func(&mut self, rhs: &$rhs) {
                self.$op(rhs)
            }
        }
    };
}

macro_rules! impl_op_assign_basic {
    ($trait:ident, $func:ident, $op:ident) => {
        impl $trait<i64> for Tensor {
            fn $func(&mut self, rhs: i64) {
                self.$op(&rhs.into())
            }
        }
        impl $trait<f64> for Tensor {
            fn $func(&mut self, rhs: f64) {
                self.$op(&rhs.into())
            }
        }
    };
}

impl_op!(Add, Tensor, add, add_tensor);
impl_op!(Mul, Tensor, mul, mul_tensor);
impl_op!(Add, Scalar, add, add_scalar);
impl_op!(Mul, Scalar, mul, mul_scalar);
impl_op_basic!(Add, add, add_scalar);
impl_op_basic!(Mul, mul, mul_scalar);
impl_op_assign!(AddAssign, Tensor, add_assign, add_tensor_);
impl_op_assign!(MulAssign, Tensor, mul_assign, mul_tensor_);
impl_op_assign!(AddAssign, Scalar, add_assign, add_scalar_);
impl_op_assign!(MulAssign, Scalar, mul_assign, mul_scalar_);
impl_op_assign_basic!(AddAssign, add_assign, add_scalar_);
impl_op_assign_basic!(MulAssign, mul_assign, mul_scalar_);

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

impl From<i64> for Tensor {
    fn from(v: i64) -> Tensor {
        Tensor::int_vec(&[v]).view(&[])
    }
}

impl From<f64> for Tensor {
    fn from(v: f64) -> Tensor {
        Tensor::float_vec(&[v]).view(&[])
    }
}
