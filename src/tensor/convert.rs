//! Implement conversion traits for tensors
use super::Tensor;
use crate::{kind::Element, TchError};
use half::f16;
use std::convert::{TryFrom, TryInto};

impl<T: Element> From<&Tensor> for Vec<T> {
    fn from(tensor: &Tensor) -> Vec<T> {
        let numel = tensor.numel();
        let mut vec = vec![T::ZERO; numel as usize];
        tensor.to_kind(T::KIND).copy_data(&mut vec, numel);
        vec
    }
}

impl<T: Element> From<&Tensor> for Vec<Vec<T>> {
    fn from(tensor: &Tensor) -> Vec<Vec<T>> {
        let first_dim = tensor.size()[0];
        (0..first_dim).map(|i| Vec::<T>::from(tensor.get(i))).collect()
    }
}

impl<T: Element> From<&Tensor> for Vec<Vec<Vec<T>>> {
    fn from(tensor: &Tensor) -> Vec<Vec<Vec<T>>> {
        let first_dim = tensor.size()[0];
        (0..first_dim).map(|i| Vec::<Vec<T>>::from(tensor.get(i))).collect()
    }
}

impl<T: Element> From<Tensor> for Vec<T> {
    fn from(tensor: Tensor) -> Vec<T> {
        Vec::<T>::from(&tensor)
    }
}

impl<T: Element> From<Tensor> for Vec<Vec<T>> {
    fn from(tensor: Tensor) -> Vec<Vec<T>> {
        Vec::<Vec<T>>::from(&tensor)
    }
}

impl<T: Element> From<Tensor> for Vec<Vec<Vec<T>>> {
    fn from(tensor: Tensor) -> Vec<Vec<Vec<T>>> {
        Vec::<Vec<Vec<T>>>::from(&tensor)
    }
}

macro_rules! from_tensor {
    ($typ:ident) => {
        impl From<&Tensor> for $typ {
            fn from(tensor: &Tensor) -> $typ {
                let numel = tensor.numel();
                if numel != 1 {
                    panic!("expected exactly one element, got {}", numel)
                }
                Vec::from(tensor)[0]
            }
        }

        impl From<Tensor> for $typ {
            fn from(tensor: Tensor) -> $typ {
                $typ::from(&tensor)
            }
        }
    };
}

from_tensor!(f64);
from_tensor!(f32);
from_tensor!(f16);
from_tensor!(i64);
from_tensor!(i32);
from_tensor!(i8);
from_tensor!(u8);
from_tensor!(bool);

impl<T: Element> TryInto<ndarray::ArrayD<T>> for &Tensor {
    type Error = ndarray::ShapeError;

    fn try_into(self) -> Result<ndarray::ArrayD<T>, Self::Error> {
        let v: Vec<T> = self.into();
        let shape: Vec<usize> = self.size().iter().map(|s| *s as usize).collect();
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), v)
    }
}

impl<T, D> TryFrom<&ndarray::ArrayBase<T, D>> for Tensor
where
    T: ndarray::Data,
    T::Elem: Element,
    D: ndarray::Dimension,
{
    type Error = TchError;

    fn try_from(value: &ndarray::ArrayBase<T, D>) -> Result<Self, Self::Error> {
        let slice = value
            .as_slice()
            .ok_or_else(|| TchError::Convert("cannot convert to slice".to_string()))?;
        let tn = Self::f_of_slice(slice)?;
        let shape: Vec<i64> = value.shape().iter().map(|s| *s as i64).collect();
        tn.f_reshape(&shape)
    }
}

impl<T, D> TryFrom<ndarray::ArrayBase<T, D>> for Tensor
where
    T: ndarray::Data,
    T::Elem: Element,
    D: ndarray::Dimension,
{
    type Error = TchError;

    fn try_from(value: ndarray::ArrayBase<T, D>) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl<T: Element> TryFrom<&Vec<T>> for Tensor {
    type Error = TchError;

    fn try_from(value: &Vec<T>) -> Result<Self, Self::Error> {
        Self::f_of_slice(value.as_slice())
    }
}

impl<T: Element> TryFrom<Vec<T>> for Tensor {
    type Error = TchError;

    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}
