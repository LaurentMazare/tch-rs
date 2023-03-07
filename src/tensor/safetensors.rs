//! Safetensors support for tensors.
//!
//! Format spec:
//! https://github.com/huggingface/safetensors
use crate::{Kind, TchError, Tensor};

use std::borrow::Cow;
use std::convert::{TryFrom, TryInto};
use std::path::Path;

use safetensors::tensor;
use safetensors::tensor::{Dtype, SafeTensorError, SafeTensors, TensorView, View};

impl TryFrom<Kind> for Dtype {
    type Error = TchError;
    fn try_from(kind: Kind) -> Result<Self, Self::Error> {
        let dtype = match kind {
            Kind::Bool => Dtype::BOOL,
            Kind::Uint8 => Dtype::U8,
            Kind::Int8 => Dtype::I8,
            Kind::Int16 => Dtype::I16,
            Kind::Int => Dtype::I32,
            Kind::Int64 => Dtype::I64,
            Kind::BFloat16 => Dtype::BF16,
            Kind::Half => Dtype::F16,
            Kind::Float => Dtype::F32,
            Kind::Double => Dtype::F64,
            kind => return Err(TchError::Convert(format!("unsupported kind ({kind:?})"))),
        };
        Ok(dtype)
    }
}

impl TryFrom<Dtype> for Kind {
    type Error = TchError;
    fn try_from(dtype: Dtype) -> Result<Self, Self::Error> {
        let kind = match dtype {
            Dtype::BOOL => Kind::Bool,
            Dtype::U8 => Kind::Uint8,
            Dtype::I8 => Kind::Int8,
            Dtype::I16 => Kind::Int16,
            Dtype::I32 => Kind::Int,
            Dtype::I64 => Kind::Int64,
            Dtype::BF16 => Kind::BFloat16,
            Dtype::F16 => Kind::Half,
            Dtype::F32 => Kind::Float,
            Dtype::F64 => Kind::Double,
            dtype => return Err(TchError::Convert(format!("unsupported dtype {dtype:?}"))),
        };
        Ok(kind)
    }
}

impl<'a> TryFrom<TensorView<'a>> for Tensor {
    type Error = TchError;
    fn try_from(view: TensorView<'a>) -> Result<Self, Self::Error> {
        let size: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
        let kind: Kind = view.dtype().try_into()?;
        Tensor::f_of_data_size(view.data(), &size, kind)
    }
}

struct SafeView<'a> {
    tensor: &'a Tensor,
    shape: Vec<usize>,
    dtype: Dtype,
}

impl<'a> TryFrom<&'a Tensor> for SafeView<'a> {
    type Error = TchError;

    fn try_from(tensor: &'a Tensor) -> Result<Self, Self::Error> {
        if tensor.is_sparse() {
            return Err(TchError::Convert("Cannot save sparse tensors".to_string()));
        }

        if !tensor.is_contiguous() {
            return Err(TchError::Convert("Cannot save non contiguous tensors".to_string()));
        }
        // TODO here it would be better to know if tensors `is_contiguous` which
        // doesn't seem available.
        let dtype = tensor.kind().try_into()?;
        let shape: Vec<usize> = tensor.size().iter().map(|&x| x as usize).collect();
        Ok(Self { tensor, shape, dtype })
    }
}

impl<'a> View for SafeView<'a> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<[u8]> {
        let mut data = vec![0; self.data_len()];
        let numel = self.tensor.numel();
        self.tensor.f_copy_data_u8(&mut data, numel).unwrap();
        data.into()
    }

    fn data_len(&self) -> usize {
        let numel = self.tensor.numel();
        numel * self.tensor.kind().elt_size_in_bytes()
    }
}

impl crate::Tensor {
    /// Reads a safetensors file and returns some named tensors.
    pub fn read_safetensors<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, TchError> {
        let file = std::fs::read(&path)?;

        let safetensors = match SafeTensors::deserialize(&file) {
            Ok(value) => value,
            Err(e) => match e {
                SafeTensorError::IoError(e) => return Err(TchError::Io(e)),
                // Always reoutput the error message from the underlying error
                // For easier debugging
                e => {
                    return Err(TchError::FileFormat(format!(
                        "unable to load safetensor file : {e}"
                    )))
                }
            },
        };

        safetensors
            .tensors()
            .into_iter()
            .map(|(name, view)| -> Result<(String, Tensor), TchError> {
                let tensor: Tensor = view.try_into()?;
                Ok((name, tensor))
            })
            .collect()
    }

    /// Writes a tensor in the safetensors format.
    pub fn write_safetensors<S: AsRef<str>, T: AsRef<Tensor>, P: AsRef<Path>>(
        tensors: &[(S, T)],
        path: P,
    ) -> Result<(), TchError> {
        let views: Result<Vec<_>, TchError> = tensors
            .into_iter()
            .map(|(name, tensor)| -> Result<(&str, SafeView), TchError> {
                let name = name.as_ref();
                let tensor = tensor.as_ref();
                let view: SafeView = tensor.try_into()?;
                Ok((name, view))
            })
            .collect();
        let views = views?;

        tensor::serialize_to_file(views, &None, path.as_ref())
            .map_err(|e| TchError::Convert(format!("Error while saving safetensor file {e}")))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{convert::TryInto};

    use safetensors::Dtype;
    use crate::{Kind};

    #[test]
    fn parse() {
        // From Kind to Dtype
        assert_eq!(TryInto::<Dtype>::try_into(Kind::Double).unwrap(), Dtype::F64);
        assert_eq!(TryInto::<Dtype>::try_into(Kind::Float).unwrap(), Dtype::F32);
        assert_eq!(TryInto::<Dtype>::try_into(Kind::Half).unwrap(), Dtype::F16);

        assert_eq!(TryInto::<Dtype>::try_into(Kind::Int8).unwrap(), Dtype::I8);
        assert_eq!(TryInto::<Dtype>::try_into(Kind::Uint8).unwrap(), Dtype::U8);

        // From Dtype to Kind
        assert_eq!(TryInto::<Kind>::try_into(Dtype::F64).unwrap(), Kind::Double);
        assert_eq!(TryInto::<Kind>::try_into(Dtype::F32).unwrap(), Kind::Float);
        assert_eq!(TryInto::<Kind>::try_into(Dtype::F16).unwrap(), Kind::Half);

        assert_eq!(TryInto::<Kind>::try_into(Dtype::I8).unwrap(), Kind::Int8);
        assert_eq!(TryInto::<Kind>::try_into(Dtype::U8).unwrap(), Kind::Uint8);
    }
}
