use crate::{Device, Kind, TchError, Tensor};

use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::path::Path;

use crate::nn::VarStore;
use safetensors::tensor;
use safetensors::tensor::{Dtype, SafeTensorError, SafeTensors, TensorView};
use tensor::serialize_to_file;

impl From<Dtype> for Kind {
    fn from(dtype: Dtype) -> Self {
        match dtype {
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
            _ => panic!("unsupported dtype `{:?}` in safetensor file", dtype),
        }
    }
}

impl From<Kind> for Dtype {
    fn from(kind: Kind) -> Self {
        match kind {
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
            _ => panic!("unsupported kind `{:?}` in safetensor file", kind),
        }
    }
}

impl<'a> TryFrom<TensorView<'a>> for Tensor {
    type Error = TchError;

    fn try_from(view: TensorView<'a>) -> Result<Self, Self::Error> {
        let kind = Kind::from(view.dtype());
        let size = shape_to_size(view.shape());
        Tensor::f_of_data_size(view.data(), &size, kind)
    }
}

impl<'a> TryFrom<Tensor> for TensorView<'a> {
    type Error = TchError;

    fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
        let cpu_tensor = tensor.f_to_device(Device::Cpu)?.f_contiguous()?;
        let shape: Vec<usize> = tensor.size().iter().map(|&x| x as usize).collect();
        let kind = cpu_tensor.f_kind()?;
        let dtype = Dtype::from(kind);
        let numel = cpu_tensor.numel();
        let size_of_elt = kind.elt_size_in_bytes();
        let length = numel * size_of_elt;
        let content: &[u8] =
            unsafe { std::slice::from_raw_parts(cpu_tensor.data_ptr() as *const u8, length) };
        Ok(TensorView::new(dtype, shape, content))
    }
}

#[inline]
fn shape_to_size(shape: &[usize]) -> Vec<i64> {
    shape.iter().map(|x| *x as i64).collect()
}

impl VarStore {
    /// Read data from safe tensor file, missing tensors will raise a error.
    ///
    /// Used to load in disk safe tensor data
    pub fn read_safe_tensors<P: AsRef<Path>>(&self, path: P) -> Result<(), TchError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        let data = load_safe_tensors(&bytes)?;
        let data: BTreeMap<String, TensorView> = data.tensors().into_iter().collect();
        for (name, tensor) in self.variables_.lock().unwrap().named_variables.iter_mut() {
            match data.get(name) {
                Some(s) => {
                    let kind = Kind::from(s.dtype());
                    let size = shape_to_size(s.shape());
                    let new = Tensor::f_of_data_size(s.data(), &size, kind)?;
                    tensor.f_copy_(&new)?
                }
                None => {
                    Err(TchError::TensorNameNotFound(name.to_string(), path.display().to_string()))?
                }
            }
        }
        Ok(())
    }
    /// Fill data from safe tensor, extra tensors in the file will be ignored.
    ///
    /// Used to load in memory safe tensor data
    pub fn fill_safe_tensor(&self, safe_tensor: SafeTensors) -> Result<(), TchError> {
        let data: BTreeMap<String, TensorView> = safe_tensor.tensors().into_iter().collect();
        for (name, tensor) in data {
            match self.variables_.lock().unwrap().named_variables.get_mut(&name) {
                Some(s) => {
                    let new = Tensor::try_from(tensor)?;
                    s.f_copy_(&new)?
                }
                None => {
                    // TODO: should log extra name here
                    continue;
                }
            }
        }
        Ok(())
    }
    /// Writes a tensor to file with the safe tensors format.
    pub fn save_safe_tensors<'a>(&self, path: &'a Path) -> Result<(), TchError> {
        let mut tensors: BTreeMap<String, TensorView<'a>> = BTreeMap::new();
        for (name, tensor) in self.variables() {
            if tensor.is_sparse() {
                Err(TchError::Convert("Cannot save sparse tensors".to_string()))?;
            }
            let view = TensorView::try_from(tensor)?;
            tensors.insert(name.to_string(), view);
        }
        match serialize_to_file(&tensors, &None, path.as_ref()) {
            Ok(_) => Ok(()),
            Err(e) => Err(TchError::Convert(format!("Error while saving safetensor file {e}"))),
        }
    }
}

fn load_safe_tensors(bytes: &[u8]) -> Result<SafeTensors, TchError> {
    let st = match SafeTensors::deserialize(&bytes) {
        Ok(o) => o,
        Err(e) => match e {
            SafeTensorError::IoError(e) => Err(TchError::Io(e))?,
            _ => Err(TchError::FileFormat(e.to_string()))?,
        },
    };
    Ok(st)
}
