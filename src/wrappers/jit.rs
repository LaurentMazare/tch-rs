//! JIT interface to run model trained/saved using PyTorch Python API.
use super::utils::{path_to_cstring, ptr_to_string};
use super::{device::Device, kind::Kind};
use crate::{TchError, Tensor};
use libc::{c_int, c_void};
use std::borrow::Borrow;
use torch_sys::*;

/// Argument and output values for JIT models.
#[derive(Debug, PartialEq)]
pub enum IValue {
    None,
    Tensor(crate::Tensor),
    Double(f64),
    Int(i64),
    Bool(bool),
    Tuple(Vec<IValue>),
    IntList(Vec<i64>),
    DoubleList(Vec<f64>),
    BoolList(Vec<bool>),
    String(String),
    StringList(Vec<String>),
    TensorList(Vec<crate::Tensor>),
    GenericList(Vec<IValue>),
    // We use a vec to represent dictionaries as f64 does not implement
    // Eq or Hash out of the box in rust. TODO: improve this?
    GenericDict(Vec<(IValue, IValue)>),
}

impl From<()> for IValue {
    fn from((): ()) -> Self {
        IValue::None
    }
}

impl<T1: Into<IValue>, T2: Into<IValue>> From<(T1, T2)> for IValue {
    fn from((p1, p2): (T1, T2)) -> Self {
        IValue::Tuple(vec![p1.into(), p2.into()])
    }
}

impl<T1: Into<IValue>, T2: Into<IValue>, T3: Into<IValue>> From<(T1, T2, T3)> for IValue {
    fn from((p1, p2, p3): (T1, T2, T3)) -> Self {
        IValue::Tuple(vec![p1.into(), p2.into(), p3.into()])
    }
}

impl<T1: Into<IValue>, T2: Into<IValue>, T3: Into<IValue>, T4: Into<IValue>> From<(T1, T2, T3, T4)>
    for IValue
{
    fn from((p1, p2, p3, p4): (T1, T2, T3, T4)) -> Self {
        IValue::Tuple(vec![p1.into(), p2.into(), p3.into(), p4.into()])
    }
}

macro_rules! impl_from {
    ($type_:ty, $cons:ident) => {
        impl From<$type_> for IValue {
            fn from(v: $type_) -> Self {
                IValue::$cons(v)
            }
        }
    };
}

impl_from!(i64, Int);
impl_from!(f64, Double);
impl_from!(bool, Bool);
impl_from!(String, String);
impl_from!(Tensor, Tensor);
impl_from!(Vec<i64>, IntList);
impl_from!(Vec<f64>, DoubleList);
impl_from!(Vec<bool>, BoolList);
impl_from!(Vec<String>, StringList);
impl_from!(Vec<crate::Tensor>, TensorList);
impl_from!(Vec<IValue>, GenericList);
impl_from!(Vec<(IValue, IValue)>, GenericDict);

impl From<&str> for IValue {
    fn from(s: &str) -> Self {
        IValue::String(s.to_string())
    }
}

impl IValue {
    pub(super) fn to_c(&self) -> Result<*mut CIValue, TchError> {
        let c = unsafe_torch_err!(match self {
            IValue::Tensor(tensor) => ati_tensor(tensor.c_tensor),
            IValue::Int(i) => ati_int(*i),
            IValue::None => ati_none(),
            IValue::Double(f) => ati_double(*f),
            IValue::Bool(b) => ati_bool(if *b { 1 } else { 0 }),
            IValue::Tuple(v) => {
                let v = v
                    .iter()
                    .map(Self::to_c)
                    .collect::<Result<Vec<_>, TchError>>()?;
                let tuple = ati_tuple(v.as_ptr(), v.len() as c_int);
                for x in v {
                    ati_free(x);
                }

                tuple
            }
            IValue::GenericList(v) => {
                let v = v
                    .iter()
                    .map(Self::to_c)
                    .collect::<Result<Vec<_>, TchError>>()?;
                let list = ati_generic_list(v.as_ptr(), v.len() as c_int);
                for x in v {
                    ati_free(x);
                }
                list
            }
            IValue::IntList(v) => ati_int_list(v.as_ptr(), v.len() as c_int),
            IValue::DoubleList(v) => ati_double_list(v.as_ptr(), v.len() as c_int),
            IValue::BoolList(v) => {
                let v: Vec<libc::c_char> = v.iter().map(|&b| if b { 1 } else { 0 }).collect();
                ati_bool_list(v.as_ptr(), v.len() as c_int)
            }
            IValue::TensorList(v) => {
                let v = v.iter().map(|t| t.c_tensor).collect::<Vec<_>>();
                ati_tensor_list(v.as_ptr(), v.len() as c_int)
            }
            IValue::String(string) => {
                let c_str = std::ffi::CString::new(string.as_str())?;
                ati_string(c_str.as_ptr())
            }
            IValue::StringList(strings) => {
                let mut v = vec![];
                for s in strings {
                    v.push(std::ffi::CString::new(s.as_str())?);
                }
                let v_ptr: Vec<_> = v.iter().map(|s| s.as_ptr()).collect();
                ati_string_list(v_ptr.as_ptr(), v.len() as c_int)
            }
            IValue::GenericDict(dict) => {
                let v = dict
                    .iter()
                    .flat_map(|(k, v)| vec![Self::to_c(k), Self::to_c(v)])
                    .collect::<Result<Vec<_>, TchError>>()?;
                let dict = ati_generic_dict(v.as_ptr(), dict.len() as c_int);
                for x in v {
                    ati_free(x);
                }
                dict
            }
        });
        Ok(c)
    }

    // This consumes the pointer and frees the associated memory.
    pub(super) fn of_c(c_ivalue: *mut CIValue) -> Result<Self, TchError> {
        let tag = unsafe_torch_err!(ati_tag(c_ivalue));
        let v = match tag {
            0 => IValue::None,
            1 => {
                let c_tensor = unsafe_torch_err!(ati_to_tensor(c_ivalue));
                IValue::Tensor(crate::Tensor { c_tensor })
            }
            2 => IValue::Double(unsafe_torch_err!(ati_to_double(c_ivalue))),
            3 => IValue::Int(unsafe_torch_err!(ati_to_int(c_ivalue))),
            4 => {
                let b = unsafe_torch_err!(ati_to_bool(c_ivalue));
                if b < 0 {
                    return Err(TchError::Kind(format!("unexpected bool value {}", b)));
                }
                IValue::Bool(b != 0)
            }
            5 => {
                let len = unsafe_torch_err!(ati_tuple_length(c_ivalue));
                let mut c_ivalues: Vec<_> =
                    (0..len).map(|_| std::ptr::null_mut::<CIValue>()).collect();
                unsafe_torch_err!(ati_to_tuple(c_ivalue, c_ivalues.as_mut_ptr(), len));
                let vec: Result<Vec<_>, _> = c_ivalues
                    .iter()
                    .map(|&c_ivalue| (Self::of_c(c_ivalue)))
                    .collect();
                IValue::Tuple(vec?)
            }
            6 => {
                let len = unsafe_torch_err!(ati_length(c_ivalue));
                let mut c_array = vec![0i64; len as usize];
                unsafe_torch_err!(ati_to_int_list(c_ivalue, c_array.as_mut_ptr(), len));
                IValue::IntList(c_array)
            }
            7 => {
                let len = unsafe_torch_err!(ati_length(c_ivalue));
                let mut c_array = vec![0f64; len as usize];
                unsafe_torch_err!(ati_to_double_list(c_ivalue, c_array.as_mut_ptr(), len));
                IValue::DoubleList(c_array)
            }
            8 => {
                let len = unsafe_torch_err!(ati_length(c_ivalue));
                let mut c_array = vec![0 as libc::c_char; len as usize];
                unsafe_torch_err!(ati_to_bool_list(c_ivalue, c_array.as_mut_ptr(), len));
                IValue::BoolList(c_array.iter().map(|&x| x != 0).collect())
            }
            9 => {
                let ptr = unsafe_torch_err!(ati_to_string(c_ivalue));
                let string = match unsafe { ptr_to_string(ptr) } {
                    None => return Err(TchError::Kind("nullptr representation".to_string())),
                    Some(s) => s,
                };
                IValue::String(string)
            }
            10 => {
                let len = unsafe_torch_err!(ati_length(c_ivalue));
                let mut c_tensors: Vec<_> =
                    (0..len).map(|_| std::ptr::null_mut::<C_tensor>()).collect();
                unsafe_torch_err!(ati_to_tensor_list(c_ivalue, c_tensors.as_mut_ptr(), len));
                let vec: Vec<_> = c_tensors
                    .iter()
                    .map(|&c_tensor| (Tensor { c_tensor }))
                    .collect();
                IValue::TensorList(vec)
            }
            12 => {
                let len = unsafe_torch_err!(ati_length(c_ivalue));
                let mut c_ivalues: Vec<_> =
                    (0..len).map(|_| std::ptr::null_mut::<CIValue>()).collect();
                unsafe_torch_err!(ati_to_generic_list(c_ivalue, c_ivalues.as_mut_ptr(), len));
                let vec: Result<Vec<_>, _> = c_ivalues
                    .iter()
                    .map(|&c_ivalue| (Self::of_c(c_ivalue)))
                    .collect();
                IValue::GenericList(vec?)
            }
            13 => {
                let len = unsafe_torch_err!(ati_length(c_ivalue));
                let mut c_ivalues: Vec<_> = (0..2 * len)
                    .map(|_| std::ptr::null_mut::<CIValue>())
                    .collect();
                unsafe_torch_err!(ati_to_generic_dict(c_ivalue, c_ivalues.as_mut_ptr(), len));
                let mut res: Vec<(IValue, IValue)> = vec![];
                for i in 0..(len as usize) {
                    let key = Self::of_c(c_ivalues[2 * i])?;
                    let value = Self::of_c(c_ivalues[2 * i + 1])?;
                    res.push((key, value))
                }
                IValue::GenericDict(res)
            }
            _ => return Err(TchError::Kind(format!("unhandled tag {}", tag))),
        };
        unsafe_torch_err!(ati_free(c_ivalue));
        Ok(v)
    }
}

/// A jit PyTorch module.
///
/// These modules can be created via the
/// [TorchScript python api](https://pytorch.org/docs/stable/jit.html).
#[derive(Debug)]
pub struct CModule {
    pub(super) c_module: *mut CModule_,
}

unsafe impl Send for CModule {}

unsafe impl Sync for CModule {}

impl Drop for CModule {
    fn drop(&mut self) {
        unsafe_torch!(atm_free(self.c_module))
    }
}

impl CModule {
    /// Loads a PyTorch saved JIT model from a file.
    pub fn load<T: AsRef<std::path::Path>>(path: T) -> Result<CModule, TchError> {
        let path = path_to_cstring(path)?;
        let c_module = unsafe_torch_err!(atm_load(path.as_ptr()));
        Ok(CModule { c_module })
    }

    /// Loads a PyTorch saved JIT model from a file onto the given device.
    ///
    /// This function loads the model directly on the specified device,
    /// which means it also allows loading a GPU model on the CPU without having a CUDA enabled GPU.
    pub fn load_on_device<T: AsRef<std::path::Path>>(
        path: T,
        device: Device,
    ) -> Result<CModule, TchError> {
        let path = path_to_cstring(path)?;
        let c_module = unsafe_torch_err!(atm_load_on_device(path.as_ptr(), device.c_int()));
        Ok(CModule { c_module })
    }

    /// Loads a PyTorch saved JIT model from a read instance.
    pub fn load_data<T: std::io::Read>(f: &mut T) -> Result<CModule, TchError> {
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;
        let buffer_ptr = buffer.as_ptr() as *const libc::c_char;
        let c_module = unsafe_torch_err!(atm_load_str(buffer_ptr, buffer.len()));
        Ok(CModule { c_module })
    }

    /// Loads a PyTorch saved JIT model from a read instance.
    ///
    /// This function loads the model directly on the specified device,
    /// which means it also allows loading a GPU model on the CPU without having a CUDA enabled GPU.
    pub fn load_data_on_device<T: std::io::Read>(
        f: &mut T,
        device: Device,
    ) -> Result<CModule, TchError> {
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;
        let buffer_ptr = buffer.as_ptr() as *const libc::c_char;
        let c_module = unsafe_torch_err!(atm_load_str_on_device(
            buffer_ptr,
            buffer.len(),
            device.c_int()
        ));
        Ok(CModule { c_module })
    }

    /// Performs the forward pass for a model on some specified tensor inputs.
    pub fn forward_ts<T: Borrow<Tensor>>(&self, ts: &[T]) -> Result<Tensor, TchError> {
        let ts: Vec<_> = ts.iter().map(|x| x.borrow().c_tensor).collect();
        let c_tensor =
            unsafe_torch_err!(atm_forward(self.c_module, ts.as_ptr(), ts.len() as c_int));
        Ok(Tensor { c_tensor })
    }

    /// Performs the forward pass for a model on some specified ivalue inputs.
    pub fn forward_is<T: Borrow<IValue>>(&self, ts: &[T]) -> Result<IValue, TchError> {
        let ts = ts
            .iter()
            .map(|x| x.borrow().to_c())
            .collect::<Result<Vec<_>, TchError>>()?;
        let c_ivalue =
            unsafe_torch_err!(atm_forward_(self.c_module, ts.as_ptr(), ts.len() as c_int));
        for x in ts {
            unsafe { ati_free(x) }
        }
        IValue::of_c(c_ivalue)
    }

    /// Runs a specified entry point for a model on some given tensor inputs.
    pub fn method_ts<T: Borrow<Tensor>>(
        &self,
        method_name: &str,
        ts: &[T],
    ) -> Result<Tensor, TchError> {
        let ts: Vec<_> = ts.iter().map(|x| x.borrow().c_tensor).collect();
        let method_name = std::ffi::CString::new(method_name)?;
        let c_tensor = unsafe_torch_err!(atm_method(
            self.c_module,
            method_name.as_ptr(),
            ts.as_ptr(),
            ts.len() as c_int
        ));
        Ok(Tensor { c_tensor })
    }

    /// Runs a specified entry point for a model on some given ivalue inputs.
    pub fn method_is<T: Borrow<IValue>>(
        &self,
        method_name: &str,
        ts: &[T],
    ) -> Result<IValue, TchError> {
        let ts = ts
            .iter()
            .map(|x| x.borrow().to_c())
            .collect::<Result<Vec<_>, TchError>>()?;
        let method_name = std::ffi::CString::new(method_name)?;
        let c_ivalue = unsafe_torch_err!(atm_method_(
            self.c_module,
            method_name.as_ptr(),
            ts.as_ptr(),
            ts.len() as c_int
        ));
        for x in ts {
            unsafe { ati_free(x) }
        }
        IValue::of_c(c_ivalue)
    }

    pub fn to(&mut self, device: Device, kind: Kind, non_blocking: bool) {
        unsafe_torch!(atm_to(
            self.c_module,
            device.c_int(),
            kind.c_int(),
            non_blocking
        ));
    }

    /// Saves a module to a given path.
    pub fn save<T: AsRef<std::path::Path>>(&self, path: T) -> Result<(), TchError> {
        let path = path_to_cstring(path)?;
        unsafe_torch_err!(atm_save(self.c_module, path.as_ptr()));
        Ok(())
    }

    /// Loads some named tensors from a module
    pub fn named_parameters(&self) -> Result<Vec<(String, Tensor)>, TchError> {
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(atm_named_parameters(
            self.c_module,
            &mut v as *mut _ as *mut c_void,
            super::tensor::add_callback
        ));
        Ok(v)
    }
}

pub fn f_get_profiling_mode() -> Result<bool, TchError> {
    Ok(unsafe_torch_err!(atm_get_profiling_mode()) != 0)
}

pub fn get_profiling_mode() -> bool {
    f_get_profiling_mode().unwrap()
}

pub fn f_set_profiling_mode(b: bool) -> Result<(), TchError> {
    unsafe_torch_err!(atm_set_profiling_mode(b as c_int));
    Ok(())
}

pub fn set_profiling_mode(b: bool) {
    f_set_profiling_mode(b).unwrap()
}

#[cfg(test)]
mod tests {
    use super::IValue;
    fn round_trip<T: Into<IValue>>(t: T) {
        let ivalue: IValue = t.into();
        let ivalue2 = IValue::of_c(ivalue.to_c().unwrap()).unwrap();
        assert_eq!(ivalue, ivalue2);
    }
    #[test]
    fn ivalue_round_trip() {
        round_trip(());
        round_trip(true);
        round_trip(false);
        round_trip(-1);
        round_trip(42);
        round_trip(3.1415);
        round_trip("".to_string());
        round_trip("foobar".to_string());
        round_trip((42, 3.1415));
        round_trip(vec![42, 1337]);
        round_trip(vec![2.71828, 3.141592, 299792458.00001]);
        round_trip((
            vec![true, false, true, true],
            vec![2.71828, 3.141592, 299792458.00001],
        ));
        round_trip(vec![IValue::from(42), IValue::from("foobar")]);
        round_trip(vec![
            (IValue::from(42), IValue::from("foobar")),
            (IValue::from("foo"), IValue::from("bar")),
        ]);
    }
}
