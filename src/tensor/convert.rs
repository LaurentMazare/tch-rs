//! Implement conversion traits for tensors
use super::Tensor;
use crate::{kind::Element, Device, TchError};
use half::f16;
use std::convert::TryFrom;

impl<T: Element> From<&Tensor> for Vec<T> {
    fn from(tensor: &Tensor) -> Vec<T> {
        let numel = tensor.numel();
        let mut vec = Vec::with_capacity(numel);
        unsafe {
            vec.set_len(numel);
        }
        tensor.to_kind(T::KIND).copy_data(&mut vec, numel);
        vec
    }
}

impl<T: Element> From<&Tensor> for Vec<Vec<T>> {
    fn from(tensor: &Tensor) -> Vec<Vec<T>> {
        let first_dim = tensor.size()[0];
        (0..first_dim)
            .map(|i| Vec::<T>::from(tensor.get(i)))
            .collect()
    }
}

impl<T: Element> From<&Tensor> for Vec<Vec<Vec<T>>> {
    fn from(tensor: &Tensor) -> Vec<Vec<Vec<T>>> {
        let first_dim = tensor.size()[0];
        (0..first_dim)
            .map(|i| Vec::<Vec<T>>::from(tensor.get(i)))
            .collect()
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

impl<T: Element, D: ndarray::Dimension> TryFrom<&Tensor> for ndarray::Array<T, D> {
    type Error = ndarray::ShapeError;

    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        let shape: Vec<usize> = value.size().iter().map(|s| *s as usize).collect();
        let v: Vec<T> = value.into();
        let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), v)?;
        array.into_dimensionality::<D>()
    }
}

impl<'a, T: Element, D: ndarray::Dimension> TryFrom<&'a Tensor> for ndarray::ArrayView<'a, T, D> {
    type Error = ndarray::ShapeError;

    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        let shape: Vec<usize> = value.size().iter().map(|s| *s as usize).collect();
        assert_eq!(value.kind(), T::KIND);
        assert_eq!(value.device(), Device::Cpu);
        let array;
        unsafe {
            array = ndarray::ArrayViewD::from_shape_ptr(
                ndarray::IxDyn(&shape),
                value.data_ptr() as *const T,
            );
        }
        array.into_dimensionality::<D>()
    }
}

impl<'a, T: Element, D: ndarray::Dimension> TryFrom<&'a Tensor>
    for ndarray::ArrayViewMut<'a, T, D>
{
    type Error = ndarray::ShapeError;

    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        let shape: Vec<usize> = value.size().iter().map(|s| *s as usize).collect();
        assert_eq!(value.kind(), T::KIND);
        assert_eq!(value.device(), Device::Cpu);
        let array;
        unsafe {
            array = ndarray::ArrayViewMutD::from_shape_ptr(
                ndarray::IxDyn(&shape),
                value.data_ptr() as *mut T,
            );
        }
        array.into_dimensionality::<D>()
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
