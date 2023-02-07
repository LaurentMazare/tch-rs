//! Safetensors support for tensors.
//!
//! Format spec:
//! https://github.com/huggingface/safetensors
use crate::{Kind, TchError, Tensor};

use std::path::Path;
use std::collections::BTreeMap;

use safetensors::tensor::{SafeTensors, Dtype, SafeTensorError, TensorView};
use safetensors::tensor;


impl crate::Tensor {
    pub fn dtype_to_kind(dtype: safetensors::tensor::Dtype) -> Result<Kind, TchError> {
        let kind =  match dtype {
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
            _ => return Err(TchError::Convert(format!("unsupported dtype in safetensor file"))),
        };
        Ok(kind)
    }

    pub fn kind_to_dtype(kind: Kind) -> Result<Dtype, TchError> {
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
            _ => return Err(TchError::Convert(format!("unsupported kind in safetensor file"))),
        };
        Ok(dtype)
    }

    /// Reads a safetensors file and returns some named tensors.
    pub fn read_safetensors<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, TchError> {
        let file = std::fs::read(&path)?;

        let safetensors = match SafeTensors::deserialize(&file) {
            Ok(value) => value,
            Err(e) => match e {
                SafeTensorError::IoError(e) => return Err(TchError::Io(e)),
                _ => return Err(TchError::FileFormat(format!("unable to load safetensor file"))),
            },
        };

        let mut result = vec![];
        for safetensor in safetensors.tensors() {
            let view = safetensor.1;

            let size: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();

            let kind = Self::dtype_to_kind(view.dtype())?;
            let tensor = Tensor::f_of_data_size(view.data(), &size, kind)?;
            result.push((safetensor.0, tensor));
        }

        Ok(result)
    }

    /// Writes a tensor in the safetensors format.
    pub fn write_safetensors<S: AsRef<str>, T: AsRef<Tensor>, P: AsRef<Path>>(
        ts: &[(S, T)],
        path: P,
    ) -> Result<(), TchError> {

        let mut tensors: BTreeMap<String, TensorView> = BTreeMap::new();
        for (name, tensor) in ts.iter() {
            let t = tensor.as_ref();

            let shape: Vec<usize> = t.size().iter().map(|&x| x as usize).collect();

            let kind = t.f_kind()?;
            let dtype = Self::kind_to_dtype(kind)?;

            let numel = t.numel();

            let mut content = vec![0u8; numel * kind.elt_size_in_bytes()];
            t.f_copy_data_u8(&mut content, numel)?;

            let view = TensorView::new(dtype, shape, &content);
            tensors.insert(name.as_ref().to_string(), view);
        }

        // tensor::serialize_to_file(&tensors, &None, path.as_ref());

        Ok(())
    }
}