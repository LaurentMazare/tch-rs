use crate::{Device, Kind, Reduction, Shape, TchError, Tensor};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

impl Tensor {
    pub fn f_view<T: Shape>(&self, s: T) -> Result<Tensor, TchError> {
        self.f_view_(&*s.to_shape())
    }

    pub fn view<T: Shape>(&self, s: T) -> Tensor {
        self.view_(&*s.to_shape())
    }

    pub fn f_zero_pad1d(&self, left: i64, right: i64) -> Result<Tensor, TchError> {
        if self.dim() != 3 {
            return Err(TchError::Shape(format!(
                "expected a 3 dimension tensor, got {:?}",
                self.size()
            )));
        }
        self.f_constant_pad_nd(&[left, right])
    }

    pub fn zero_pad1d(&self, left: i64, right: i64) -> Tensor {
        self.f_zero_pad1d(left, right).unwrap()
    }

    pub fn f_zero_pad2d(
        &self,
        left: i64,
        right: i64,
        top: i64,
        bottom: i64,
    ) -> Result<Tensor, TchError> {
        if self.dim() != 4 {
            return Err(TchError::Shape(format!(
                "expected a 4 dimension tensor, got {:?}",
                self.size()
            )));
        }
        self.f_constant_pad_nd(&[left, right, top, bottom])
    }

    pub fn zero_pad2d(&self, left: i64, right: i64, top: i64, bottom: i64) -> Tensor {
        self.f_zero_pad2d(left, right, top, bottom).unwrap()
    }

    /// Casts a tensor to a specified kind.
    pub fn to_kind(&self, kind: Kind) -> Tensor {
        self.totype(kind)
    }

    pub fn f_to_kind(&self, kind: Kind) -> Result<Tensor, TchError> {
        self.f_totype(kind)
    }

    pub fn nll_loss(&self, targets: &Tensor) -> Tensor {
        self.g_nll_loss::<Tensor>(targets, None, Reduction::Mean, -100)
    }
    /// Computes the cross-entropy loss based on some logits and targets.
    pub fn cross_entropy_for_logits(&self, targets: &Tensor) -> Tensor {
        self.log_softmax(-1, Kind::Float).nll_loss(&targets)
    }

    /// Returns the average accuracy for some given logits assuming that
    /// targets represent ground-truth.
    pub fn accuracy_for_logits(&self, targets: &Tensor) -> Tensor {
        self.argmax(-1, false)
            .eq1(&targets)
            .to_kind(Kind::Float)
            .mean(Kind::Float)
    }

    pub fn random_batch(&self, batch_size: i64) -> Tensor {
        let len: i64 = self.size()[0];
        let index = Tensor::randint(len, &[batch_size], (Kind::Int64, self.device()));
        self.index_select(0, &index)
    }

    pub fn random_batch2(t1: &Tensor, t2: &Tensor, batch_size: i64) -> (Tensor, Tensor) {
        let len1: i64 = t1.size()[0];
        let len2: i64 = t2.size()[0];
        if len1 != len2 {
            panic!(
                "random_batch2: shape mismatch {:?} {:?}",
                t1.size(),
                t2.size()
            )
        }
        let device1 = t1.device();
        let device2 = t2.device();
        if device1 != device2 {
            panic!("random_batch2: device mismatch {:?} {:?}", device1, device2)
        }
        let index = Tensor::randint(len1, &[batch_size], (Kind::Int64, device1));
        let batch1 = t1.index_select(0, &index);
        let batch2 = t2.index_select(0, &index);
        (batch1, batch2)
    }

    /// Moves a tensor to a specified device.
    pub fn to_device(&self, device: Device) -> Tensor {
        self.to(device)
    }

    pub fn f_to_device(&self, device: Device) -> Result<Tensor, TchError> {
        self.f_to(device)
    }

    pub fn avg_pool2d_default(&self, ksize: i64) -> Tensor {
        self.avg_pool2d(&[ksize, ksize], &[ksize, ksize], &[0, 0], false, true, 1)
    }

    pub fn max_pool2d_default(&self, ksize: i64) -> Tensor {
        self.max_pool2d(&[ksize, ksize], &[ksize, ksize], &[0, 0], &[1, 1], false)
    }

    /// Flattens a tensor.
    ///
    /// This returns a flattened version of the given tensor. The first dimension
    /// is preserved as it is assumed to be the mini-batch dimension.
    pub fn flat_view(&self) -> Tensor {
        let batch_size = self.size()[0] as i64;
        self.view((batch_size, -1))
    }

    /// Converts a tensor to a one-hot encoded version.
    ///
    /// If the input has a size [N1, N2, ..., Nk], the returned tensor has a size
    /// [N1, ..., Nk, labels]. The returned tensor uses float values.
    /// Elements of the input vector are expected to be between 0 and labels-1.
    pub fn onehot(&self, labels: i64) -> Tensor {
        Tensor::zeros(
            &[self.size(), vec![labels]].concat(),
            crate::wrappers::kind::FLOAT_CPU,
        )
        .scatter_1(-1, &self.unsqueeze(-1).to_kind(Kind::Int64), 1.0)
    }

    /// Copies a tensor to a newly allocated tensor using the same shape and device.
    pub fn copy(&self) -> Tensor {
        let mut result = self.zeros_like();
        result.copy_(&self);
        result
    }
}

macro_rules! impl_op {
    ($trait:ident, $func:ident, $op:ident) => {
        impl $trait<Tensor> for Tensor {
            type Output = Tensor;

            fn $func(self, rhs: Tensor) -> Self::Output {
                self.$op(&rhs)
            }
        }

        impl $trait<&Tensor> for Tensor {
            type Output = Tensor;

            fn $func(self, rhs: &Tensor) -> Self::Output {
                self.$op(rhs)
            }
        }

        impl<'a> $trait<&Tensor> for &'a Tensor {
            type Output = Tensor;

            fn $func(self, rhs: &Tensor) -> Self::Output {
                self.$op(rhs)
            }
        }

        impl $trait<Tensor> for &Tensor {
            type Output = Tensor;

            fn $func(self, rhs: Tensor) -> Self::Output {
                self.$op(&rhs)
            }
        }
    };
}

macro_rules! impl_op_basic {
    /* rev such that rev(op(b, a)) = op(a, b) */
    ($trait:ident, $func:ident, $op:ident, $rev:ident) => {
        impl $trait<i64> for Tensor {
            type Output = Tensor;

            fn $func(self, rhs: i64) -> Self::Output {
                self.$op(rhs)
            }
        }

        impl $trait<f64> for Tensor {
            type Output = Tensor;

            fn $func(self, rhs: f64) -> Self::Output {
                self.$op(rhs)
            }
        }

        impl $trait<i64> for &Tensor {
            type Output = Tensor;

            fn $func(self, rhs: i64) -> Self::Output {
                self.$op(rhs)
            }
        }

        impl $trait<f64> for &Tensor {
            type Output = Tensor;

            fn $func(self, rhs: f64) -> Self::Output {
                self.$op(rhs)
            }
        }

        impl $trait<Tensor> for i64 {
            type Output = Tensor;

            fn $func(self, rhs: Tensor) -> Self::Output {
                $rev(rhs.$op(self))
            }
        }

        impl $trait<Tensor> for f64 {
            type Output = Tensor;

            fn $func(self, rhs: Tensor) -> Self::Output {
                $rev(rhs.$op(self))
            }
        }

        impl $trait<&Tensor> for i64 {
            type Output = Tensor;

            fn $func(self, rhs: &Tensor) -> Self::Output {
                $rev(rhs.$op(self))
            }
        }

        impl $trait<&Tensor> for f64 {
            type Output = Tensor;

            fn $func(self, rhs: &Tensor) -> Self::Output {
                $rev(rhs.$op(self))
            }
        }
    };
}

macro_rules! impl_op_assign {
    ($trait:ident, $func:ident, $op:ident) => {
        impl $trait<Tensor> for Tensor {
            fn $func(&mut self, rhs: Tensor) {
                let _ = self.$op(&rhs);
            }
        }

        impl $trait<&Tensor> for Tensor {
            fn $func(&mut self, rhs: &Tensor) {
                let _ = self.$op(rhs);
            }
        }
    };
}

macro_rules! impl_op_assign_basic {
    ($trait:ident, $func:ident, $op:ident) => {
        impl $trait<i64> for Tensor {
            fn $func(&mut self, rhs: i64) {
                let _ = self.$op(rhs);
            }
        }
        impl $trait<f64> for Tensor {
            fn $func(&mut self, rhs: f64) {
                let _ = self.$op(rhs);
            }
        }
    };
}

fn id<T>(v: T) -> T {
    v
}

fn neg(t: Tensor) -> Tensor {
    t.neg()
}

fn inv(t: Tensor) -> Tensor {
    t.pow(-1)
}

impl_op!(Add, add, g_add);
impl_op_basic!(Add, add, g_add1, id);
impl_op_assign!(AddAssign, add_assign, g_add_);
impl_op_assign_basic!(AddAssign, add_assign, g_add_1);

impl_op!(Mul, mul, g_mul);
impl_op_basic!(Mul, mul, g_mul1, id);
impl_op_assign!(MulAssign, mul_assign, g_mul_);
impl_op_assign_basic!(MulAssign, mul_assign, g_mul_1);

impl_op!(Div, div, g_div);
impl_op_basic!(Div, div, g_div1, inv);
impl_op_assign!(DivAssign, div_assign, g_div_);
impl_op_assign_basic!(DivAssign, div_assign, g_div_1);

impl_op!(Sub, sub, g_sub);
impl_op_basic!(Sub, sub, g_sub1, neg);
impl_op_assign!(SubAssign, sub_assign, g_sub_);
impl_op_assign_basic!(SubAssign, sub_assign, g_sub_1);

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self.f_neg().unwrap()
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self.f_neg().unwrap()
    }
}

impl std::iter::Sum for Tensor {
    fn sum<I: Iterator<Item = Tensor>>(mut iter: I) -> Tensor {
        match iter.next() {
            None => Tensor::from(0.),
            Some(t) => iter.fold(t, |acc, x| x + acc),
        }
    }
}

impl<'a> std::iter::Sum<&'a Tensor> for Tensor {
    fn sum<I: Iterator<Item = &'a Tensor>>(mut iter: I) -> Tensor {
        match iter.next() {
            None => Tensor::from(0.),
            Some(t) => iter.fold(t.shallow_clone(), |acc, x| x + acc),
        }
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        if self.size() != other.size() {
            return false;
        }
        match self.f_eq1(&other) {
            Err(_) => false,
            Ok(v) => match v.f_all() {
                Err(_) => false,
                Ok(v) => i64::from(v) > 0,
            },
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (is_int, is_float) = match self.kind() {
            Kind::Int | Kind::Int8 | Kind::Uint8 | Kind::Int16 | Kind::Int64 => (true, false),
            Kind::Half | Kind::Float | Kind::Double => (false, true),
            Kind::Bool | Kind::ComplexHalf | Kind::ComplexFloat | Kind::ComplexDouble => {
                (false, false)
            }
        };
        match (self.size().as_slice(), is_int, is_float) {
            ([], true, false) => write!(f, "[{}]", i64::from(self)),
            ([s], true, false) if *s < 10 => write!(f, "{:?}", Vec::<i64>::from(self)),
            ([], false, true) => write!(f, "[{}]", f64::from(self)),
            ([s], false, true) if *s < 10 => write!(f, "{:?}", Vec::<f64>::from(self)),
            _ => write!(f, "Tensor[{:?}, {:?}]", self.size(), self.kind()),
        }
    }
}
