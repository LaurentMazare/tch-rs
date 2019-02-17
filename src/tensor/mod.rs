use crate::scalar::Scalar;
use crate::Kind;
use std::ops::{Add, AddAssign, Mul, MulAssign};

mod c_wrapper;
mod c_wrapper_generated;

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
                let _ = self.$op(&rhs);
            }
        }

        impl $trait<&$rhs> for Tensor {
            fn $func(&mut self, rhs: &$rhs) {
                let _ = self.$op(rhs);
            }
        }
    };
}

macro_rules! impl_op_assign_basic {
    ($trait:ident, $func:ident, $op:ident) => {
        impl $trait<i64> for Tensor {
            fn $func(&mut self, rhs: i64) {
                let _ = self.$op(&rhs.into());
            }
        }
        impl $trait<f64> for Tensor {
            fn $func(&mut self, rhs: f64) {
                let _ = self.$op(&rhs.into());
            }
        }
    };
}

impl_op!(Add, Tensor, add, g_add);
impl_op!(Mul, Tensor, mul, g_mul);
impl_op!(Add, Scalar, add, g_add1);
impl_op!(Mul, Scalar, mul, g_mul1);
impl_op_basic!(Add, add, g_add1);
impl_op_basic!(Mul, mul, g_mul1);
impl_op_assign!(AddAssign, Tensor, add_assign, g_add_);
impl_op_assign!(MulAssign, Tensor, mul_assign, g_mul_);
impl_op_assign!(AddAssign, Scalar, add_assign, g_add_1);
impl_op_assign!(MulAssign, Scalar, mul_assign, g_mul_1);
impl_op_assign_basic!(AddAssign, add_assign, g_add_1);
impl_op_assign_basic!(MulAssign, mul_assign, g_mul_1);

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

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Tensor[{:?}, {:?}]", self.size(), self.kind())
    }
}

impl Tensor {
    pub fn to_kind(&self, kind: &Kind) -> Tensor {
        self.totype(kind)
    }

    pub fn nll_loss(&self, targets: &Tensor) -> Tensor {
        let weights = Tensor::new();
        self.g_nll_loss(targets, &weights, 1, -100)
    }
}
