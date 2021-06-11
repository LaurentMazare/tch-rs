//! Implement conversion traits for tensors
use super::Tensor;
use crate::{Kind, TchError};
use half::f16;
use std::convert::{TryFrom, TryInto};

macro_rules! from_tensor {
    ($typ:ident, $zero:expr, $kind:ident) => {
        impl From<&Tensor> for Vec<$typ> {
            fn from(tensor: &Tensor) -> Vec<$typ> {
                let numel = tensor.numel();
                let mut vec = vec![$zero; numel as usize];
                tensor.to_kind(Kind::$kind).copy_data(&mut vec, numel);
                vec
            }
        }

        impl From<&Tensor> for Vec<Vec<$typ>> {
            fn from(tensor: &Tensor) -> Vec<Vec<$typ>> {
                let first_dim = tensor.size()[0];
                (0..first_dim)
                    .map(|i| Vec::<$typ>::from(tensor.get(i)))
                    .collect()
            }
        }

        impl From<&Tensor> for Vec<Vec<Vec<$typ>>> {
            fn from(tensor: &Tensor) -> Vec<Vec<Vec<$typ>>> {
                let first_dim = tensor.size()[0];
                (0..first_dim)
                    .map(|i| Vec::<Vec<$typ>>::from(tensor.get(i)))
                    .collect()
            }
        }

        impl From<&Tensor> for $typ {
            fn from(tensor: &Tensor) -> $typ {
                let numel = tensor.numel();
                if numel != 1 {
                    panic!("expected exactly one element, got {}", numel)
                }
                Vec::from(tensor)[0]
            }
        }

        impl From<Tensor> for Vec<$typ> {
            fn from(tensor: Tensor) -> Vec<$typ> {
                Vec::<$typ>::from(&tensor)
            }
        }

        impl From<Tensor> for Vec<Vec<$typ>> {
            fn from(tensor: Tensor) -> Vec<Vec<$typ>> {
                Vec::<Vec<$typ>>::from(&tensor)
            }
        }

        impl From<Tensor> for Vec<Vec<Vec<$typ>>> {
            fn from(tensor: Tensor) -> Vec<Vec<Vec<$typ>>> {
                Vec::<Vec<Vec<$typ>>>::from(&tensor)
            }
        }

        impl From<Tensor> for $typ {
            fn from(tensor: Tensor) -> $typ {
                $typ::from(&tensor)
            }
        }
    };
}

from_tensor!(f64, 0f64, Double);
from_tensor!(f32, 0f32, Float);
from_tensor!(f16, f16::from_f64(0.0), Half);
from_tensor!(i64, 0i64, Int64);
from_tensor!(i32, 0i32, Int);
from_tensor!(i8, 0i8, Int8);
from_tensor!(u8, 0u8, Uint8);
from_tensor!(bool, false, Bool);

macro_rules! try_into_impl {
    ($type:ident) => {
        impl TryInto<ndarray::ArrayD<$type>> for &Tensor {
            type Error = ndarray::ShapeError;

            fn try_into(self) -> Result<ndarray::ArrayD<$type>, Self::Error> {
                let v: Vec<$type> = self.into();
                let shape: Vec<usize> = self.size().iter().map(|s| *s as usize).collect();
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), v)
            }
        }
    };
}

try_into_impl!(f16);
try_into_impl!(f32);
try_into_impl!(i32);
try_into_impl!(f64);
try_into_impl!(i64);
try_into_impl!(bool);

macro_rules! try_from_impl {
    ($type:ident) => {
        impl<D> TryFrom<ndarray::Array<$type, D>> for Tensor
        where
            D: ndarray::Dimension,
        {
            type Error = TchError;

            fn try_from(value: ndarray::Array<$type, D>) -> Result<Self, Self::Error> {
                // TODO: Replace this with `?` once `std::option::NoneError` has been stabilized.
                let slice = match value.as_slice() {
                    None => return Err(TchError::Convert("cannot convert to slice".to_string())),
                    Some(v) => v,
                };
                let tn = Self::f_of_slice(slice)?;
                let shape: Vec<i64> = value.shape().iter().map(|s| *s as i64).collect();
                tn.f_reshape(&shape)
            }
        }

        impl TryFrom<Vec<$type>> for Tensor {
            type Error = TchError;

            fn try_from(value: Vec<$type>) -> Result<Self, Self::Error> {
                let tn = Self::f_of_slice(value.as_slice())?;
                Ok(tn)
            }
        }
    };
}

try_from_impl!(f16);
try_from_impl!(f32);
try_from_impl!(i32);
try_from_impl!(f64);
try_from_impl!(i64);
try_from_impl!(bool);
