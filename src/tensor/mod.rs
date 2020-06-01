//! A Torch tensor.
use crate::{Device, Kind, TchError};
use std::{
    convert::TryFrom,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    slice,
};
use torch_sys::*;

pub mod index;
mod iter;
mod npy;

pub use super::wrappers::tensor::{no_grad, no_grad_guard, NoGradGuard, Reduction, Tensor};
pub use index::{IndexOp, NewAxis, TensorIndexer};

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

pub trait Shape {
    fn to_shape(&self) -> Box<[i64]>;
}

macro_rules! impl_shape {
    ($v:expr) => {
        impl Shape for [i64; $v] {
            fn to_shape(&self) -> Box<[i64]> {
                Box::new(*self)
            }
        }
    };
}

impl_shape!(0);
impl_shape!(1);
impl_shape!(2);
impl_shape!(3);
impl_shape!(4);
impl_shape!(5);
impl_shape!(6);

impl Shape for () {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([])
    }
}

impl Shape for &[i64] {
    fn to_shape(&self) -> Box<[i64]> {
        (*self).into()
    }
}

impl Shape for i64 {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([*self])
    }
}

impl Shape for usize {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([*self as i64])
    }
}

impl Shape for i32 {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([i64::from(*self)])
    }
}

impl Shape for (i64,) {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([self.0])
    }
}

impl Shape for (i64, i64) {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([self.0, self.1])
    }
}

impl Shape for (i64, i64, i64) {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([self.0, self.1, self.2])
    }
}

impl Shape for (i64, i64, i64, i64) {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([self.0, self.1, self.2, self.3])
    }
}

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

    /// Creates a flattened vector of tensor elements.
    pub fn f_to_flat_vec<T>(&self) -> Result<Vec<T>, TchError>
    where
        T: crate::kind::T,
    {
        let numel = self.numel();
        let mut v = Vec::<T>::with_capacity(numel);
        unsafe {
            let s = slice::from_raw_parts_mut(v.as_mut_ptr(), numel);
            self.f_copy_data::<T>(s, numel)?;
            v.set_len(numel);
        }
        Ok(v)
    }

    /// See [f_to_flat_vec](Tensor::f_to_flat_vec).
    pub fn to_flat_vec<T>(&self) -> Vec<T>
    where
        T: crate::kind::T,
    {
        self.f_to_flat_vec().unwrap()
    }

    /// Creates a tensor from a slice-like object and a shape.
    pub fn from_shape<T, Shp, S>(shape: Shp, data: S) -> Result<Self, TchError>
    where
        Shp: Shape,
        S: AsRef<[T]>,
        T: crate::kind::T,
    {
        let shape = shape.to_shape();
        let slice = data.as_ref();
        Self::from(slice).f_reshape(&shape)
    }

    /// Creates a tensor with values generated by the callback `f`.
    ///
    /// The callback function has the signature `Fn(&[i64]) -> T`. If the shape
    /// has zero dimension, the callback will never be called.
    pub fn from_shape_fn<T, S, F>(shape: S, f: F) -> Result<Self, TchError>
    where
        S: Shape,
        F: Fn(&[i64]) -> T,
        T: crate::kind::T,
    {
        let shape_ = shape.to_shape();
        let numel_opt = shape_
            .iter()
            .cloned()
            .fold(None, |prod_opt, value| {
                Some(prod_opt.map(|prod| prod * value).unwrap_or(value))
            })
            .map(|numel| numel as usize);

        match numel_opt {
            Some(numel) => {
                let mut data = Vec::with_capacity(numel);
                let uninit_slice = unsafe { slice::from_raw_parts_mut(data.as_mut_ptr(), numel) };
                Self::from_fn_recursive(
                    shape_.as_ref(),
                    &f,
                    0,
                    0,
                    &mut Vec::with_capacity(shape_.len()),
                    uninit_slice,
                );
                unsafe {
                    data.set_len(numel);
                }
                let tensor = Tensor::from_shape(shape, &data)?;
                Ok(tensor)
            }
            None => Ok(Tensor::empty(&[], (T::KIND, Device::Cpu))),
        }
    }

    fn from_fn_recursive<F, T>(
        shape: &[i64],
        f: &F,
        depth: usize,
        data_index: i64,
        indexes: &mut Vec<i64>,
        data: &mut [T],
    ) where
        F: Fn(&[i64]) -> T,
        T: crate::kind::T,
    {
        if depth < shape.len() {
            let max_index = shape[depth];
            for index in 0..max_index {
                indexes.push(index);
                Self::from_fn_recursive(
                    shape,
                    f,
                    depth + 1,
                    data_index * max_index + index,
                    indexes,
                    data,
                );
                indexes.pop();
            }
        } else {
            data[data_index as usize] = f(indexes.as_ref());
        }
    }
}

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
            ([], true, false) => write!(f, "[{}]", i64::try_from(self).unwrap()),
            ([s], true, false) if *s < 10 => write!(f, "{:?}", Vec::<i64>::try_from(self).unwrap()),
            ([], false, true) => write!(f, "[{}]", f64::try_from(self).unwrap()),
            ([s], false, true) if *s < 10 => write!(f, "{:?}", Vec::<f64>::try_from(self).unwrap()),
            _ => write!(f, "Tensor[{:?}, {:?}]", self.size(), self.kind()),
        }
    }
}

impl Tensor {
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
}

impl Tensor {
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
                Ok(v) => bool::try_from(v).unwrap(),
            },
        }
    }
}

// from primitive types to tensor

impl<T: crate::kind::T> From<T> for Tensor {
    fn from(v: T) -> Tensor {
        Tensor::of_slice(&[v]).view(())
    }
}

// try from tensor to primitive types

macro_rules! impl_tryfrom_tensor_for_primitive {
    ($typ:ty) => {
        impl TryFrom<&Tensor> for $typ
        where
            $typ: crate::kind::T,
        {
            type Error = TchError;

            fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
                let numel = tensor.numel();
                if numel != 1 {
                    return Err(TchError::Shape(format!(
                        "expected exactly one element, got {}",
                        numel
                    )));
                }
                let mut v = Vec::<$typ>::with_capacity(1);
                unsafe {
                    let s = slice::from_raw_parts_mut(v.as_mut_ptr(), 1);
                    tensor.f_copy_data::<$typ>(s, numel)?;
                    v.set_len(1);
                }
                Ok(v[0])
            }
        }

        impl TryFrom<Tensor> for $typ
        where
            $typ: crate::kind::T,
        {
            type Error = TchError;

            fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
                TryFrom::try_from(&tensor)
            }
        }
    };
}

impl_tryfrom_tensor_for_primitive!(u8);
impl_tryfrom_tensor_for_primitive!(i8);
impl_tryfrom_tensor_for_primitive!(i16);
impl_tryfrom_tensor_for_primitive!(i32);
impl_tryfrom_tensor_for_primitive!(i64);
impl_tryfrom_tensor_for_primitive!(f32);
impl_tryfrom_tensor_for_primitive!(f64);
impl_tryfrom_tensor_for_primitive!(bool);

// from slice or vec to tensor

impl<T> From<&[T]> for Tensor
where
    T: crate::kind::T,
{
    fn from(v: &[T]) -> Tensor {
        Tensor::of_slice(v)
    }
}

impl<T> From<&Vec<T>> for Tensor
where
    T: crate::kind::T,
{
    fn from(v: &Vec<T>) -> Tensor {
        From::from(v.as_slice())
    }
}

impl<T> From<Vec<T>> for Tensor
where
    T: crate::kind::T,
{
    fn from(v: Vec<T>) -> Tensor {
        From::from(v.as_slice())
    }
}

// from tensor to vec

impl<T> TryFrom<&Tensor> for Vec<T>
where
    T: crate::kind::T,
{
    type Error = TchError;

    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        // dbg!(tensor.size());
        let numel = match tensor.size().as_slice() {
            &[numel] => numel as usize,
            _ => {
                return Err(TchError::Shape(
                    "the number of dimensions does not match".into(),
                ))
            }
        };
        let mut v = Vec::<T>::with_capacity(numel);
        unsafe {
            let s = slice::from_raw_parts_mut(v.as_mut_ptr(), numel);
            tensor.f_copy_data::<T>(s, numel)?;
            v.set_len(numel);
        }
        Ok(v)
    }
}

impl<T> TryFrom<Tensor> for Vec<T>
where
    T: crate::kind::T,
{
    type Error = TchError;

    fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
        TryFrom::try_from(&tensor)
    }
}

// from tensor to multi-dimensional vecs

macro_rules! make_vecs_t {
    (Vec) => {
        Vec<T>
    };
    (Vec Vec $($vec:ident)*) => {
        Vec< make_vecs_t!(Vec $($vec)*) >
    };
}

macro_rules! impl_tryfrom_tensor_for_vecs {
    (Vec Vec) => {
        impl<T> TryFrom<&Tensor> for Vec<Vec<T>>
        where
            T: crate::kind::T,
        {
            type Error = TchError;

            fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
                // dbg!(tensor.size());
                let first_dim = *tensor.size().first().ok_or_else(|| {
                    TchError::Shape("the number of dimensions does not match".into())
                })?;

                (0..first_dim)
                    .map(|i| Vec::<T>::try_from(tensor.get(i)))
                    .collect::<Result<Vec<_>, TchError>>()
            }
        }

        impl<T> TryFrom<Tensor> for Vec<Vec<T>>
        where
            T: crate::kind::T,
        {
            type Error = TchError;

            fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
                TryFrom::try_from(&tensor)
            }
        }
    };
    (Vec Vec Vec $($vec:ident)*) => {
        impl_tryfrom_tensor_for_vecs!(Vec Vec $($vec)*);

        impl<T> TryFrom<&Tensor> for make_vecs_t!(Vec Vec Vec $($vec)*)
        where
            T: crate::kind::T,
        {
            type Error = TchError;

            fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
                let first_dim = *tensor.size().first().ok_or_else(|| {
                    TchError::Shape("the number of dimensions does not match".into())
                })?;

                (0..first_dim)
                    .map(|i| Vec::<make_vecs_t!(Vec $($vec)*)>::try_from(tensor.get(i)))
                    .collect::<Result<Vec<_>, TchError>>()
            }
        }

        impl<T> TryFrom<Tensor> for make_vecs_t!(Vec Vec Vec $($vec)*)
        where
            T: crate::kind::T,
        {
            type Error = TchError;

            fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
                TryFrom::try_from(&tensor)
            }
        }
    };
}

impl_tryfrom_tensor_for_vecs!(Vec Vec Vec Vec Vec Vec Vec Vec Vec Vec Vec Vec Vec Vec Vec Vec);

// from/to ndarray

impl<T, D> TryFrom<&ndarray::Array<T, D>> for Tensor
where
    T: crate::kind::T,
    D: ndarray::Dimension,
{
    type Error = TchError;

    fn try_from(value: &ndarray::Array<T, D>) -> Result<Self, Self::Error> {
        let slice = value
            .as_slice()
            .ok_or_else(|| TchError::Convert("cannot convert to slice".into()))?;
        let tn = Self::f_of_slice(slice)?;
        let shape: Vec<i64> = value.shape().iter().map(|s| *s as i64).collect();
        Ok(tn.f_reshape(&shape)?)
    }
}

impl<T, D> TryFrom<ndarray::Array<T, D>> for Tensor
where
    T: crate::kind::T,
    D: ndarray::Dimension,
{
    type Error = TchError;

    fn try_from(value: ndarray::Array<T, D>) -> Result<Self, Self::Error> {
        TryFrom::try_from(&value)
    }
}

impl<T> TryFrom<&Tensor> for ndarray::ArrayD<T>
where
    T: crate::kind::T,
{
    type Error = TchError;

    fn try_from(tensor: &Tensor) -> Result<ndarray::ArrayD<T>, Self::Error> {
        let numel = tensor.numel();
        let mut v = Vec::<T>::with_capacity(numel);
        unsafe {
            let s = slice::from_raw_parts_mut(v.as_mut_ptr(), numel);
            tensor.f_copy_data::<T>(s, numel)?;
            v.set_len(numel);
        }
        let shape: Vec<usize> = tensor.size().into_iter().map(|s| s as usize).collect();
        Ok(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), v).unwrap())
    }
}

impl<T> TryFrom<Tensor> for ndarray::ArrayD<T>
where
    T: crate::kind::T,
{
    type Error = TchError;

    fn try_from(tensor: Tensor) -> Result<ndarray::ArrayD<T>, Self::Error> {
        TryFrom::try_from(&tensor)
    }
}

#[used]
static INIT_ARRAY: [unsafe extern "C" fn(); 1] = [dummy_cuda_dependency];
