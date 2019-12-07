//! JIT interface to run model trained/saved using PyTorch Python API.
use super::utils::{path_to_cstring, ptr_to_string};
use crate::Tensor;
use failure::Fallible;
use libc::c_int;
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
    TensorList(Vec<crate::Tensor>),
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
    ($type_:ident, $cons:ident) => {
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

impl From<&str> for IValue {
    fn from(s: &str) -> Self {
        IValue::String(s.to_string())
    }
}

impl IValue {
    pub(super) fn to_c(&self) -> Fallible<*mut CIValue> {
        let c = unsafe_torch_err!({
            match self {
                IValue::Tensor(tensor) => ati_tensor(tensor.c_tensor),
                IValue::Int(i) => ati_int(*i),
                IValue::None => ati_none(),
                IValue::Double(f) => ati_double(*f),
                IValue::Bool(b) => ati_bool(if *b { 1 } else { 0 }),
                IValue::Tuple(v) => {
                    let v = v.iter().map(Self::to_c).collect::<Fallible<Vec<_>>>()?;
                    let tuple = ati_tuple(v.as_ptr(), v.len() as c_int);
                    for x in v {
                        ati_free(x);
                    }

                    tuple
                }
                IValue::IntList(v) => ati_int_list(v.as_ptr(), v.len() as c_int),
                IValue::DoubleList(v) => ati_double_list(v.as_ptr(), v.len() as c_int),
                IValue::BoolList(v) => {
                    let v: Vec<i8> = v.iter().map(|&b| if b { 1 } else { 0 }).collect();
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
            }
        });
        Ok(c)
    }

    // This consumes the pointer and frees the associated memory.
    pub(super) fn of_c(c_ivalue: *mut CIValue) -> Fallible<Self> {
        let tag = unsafe_torch_err!({ ati_tag(c_ivalue) });
        let v = match tag {
            0 => IValue::None,
            1 => {
                let c_tensor = unsafe_torch_err!({ ati_to_tensor(c_ivalue) });
                IValue::Tensor(crate::Tensor { c_tensor })
            }
            2 => IValue::Double(unsafe_torch_err!({ ati_to_double(c_ivalue) })),
            3 => IValue::Int(unsafe_torch_err!({ ati_to_int(c_ivalue) })),
            4 => {
                let b = unsafe_torch_err!({ ati_to_bool(c_ivalue) });
                ensure!(b >= 0, "unexpected bool value {}", b);
                IValue::Bool(b != 0)
            }
            5 => {
                let len = unsafe_torch_err!({ ati_tuple_length(c_ivalue) });
                let mut c_ivalues: Vec<_> =
                    (0..len).map(|_| std::ptr::null_mut::<CIValue>()).collect();
                unsafe_torch_err!(ati_to_tuple(c_ivalue, c_ivalues.as_mut_ptr(), len));
                let vec: Result<Vec<_>, _> = c_ivalues
                    .iter()
                    .map(|&c_ivalue| (Self::of_c(c_ivalue)))
                    .collect();
                IValue::Tuple(vec?)
            }
            6 => bail!("IntList is not currently supported"),
            7 => bail!("DoubleList is not currently supported"),
            8 => bail!("BoolList is not currently supported"),
            9 => {
                let ptr = unsafe_torch_err!({ ati_to_string(c_ivalue) });
                let string = match unsafe { ptr_to_string(ptr) } {
                    None => bail!("unable to decode string"),
                    Some(s) => s,
                };
                IValue::String(string)
            }
            10 => bail!("TensorList is not currently supported"),
            12 => bail!("GenericList is not currently supported"),
            13 => bail!("GenericDict is not currently supported"),
            _ => bail!("unhandled tag {}", tag),
        };
        unsafe_torch_err!({ ati_free(c_ivalue) });
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
        unsafe_torch!({ atm_free(self.c_module) })
    }
}

impl CModule {
    /// Loads a PyTorch saved JIT model from a file.
    pub fn load<T: AsRef<std::path::Path>>(path: T) -> Fallible<CModule> {
        let path = path_to_cstring(path)?;
        let c_module = unsafe_torch_err!({ atm_load(path.as_ptr()) });
        Ok(CModule { c_module })
    }

    /// Performs the forward pass for a model on some specified tensor inputs.
    pub fn forward_ts<T: Borrow<Tensor>>(&self, ts: &[T]) -> Fallible<Tensor> {
        let ts: Vec<_> = ts.iter().map(|x| x.borrow().c_tensor).collect();
        let c_tensor =
            unsafe_torch_err!({ atm_forward(self.c_module, ts.as_ptr(), ts.len() as c_int) });
        Ok(Tensor { c_tensor })
    }

    /// Performs the forward pass for a model on some specified ivalue input.
    pub fn forward_is<T: Borrow<IValue>>(&self, ts: &[T]) -> Fallible<IValue> {
        let ts = ts
            .iter()
            .map(|x| x.borrow().to_c())
            .collect::<Fallible<Vec<_>>>()?;
        let c_ivalue =
            unsafe_torch_err!({ atm_forward_(self.c_module, ts.as_ptr(), ts.len() as c_int) });
        for x in ts {
            unsafe { ati_free(x) }
        }
        IValue::of_c(c_ivalue)
    }
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
    }
}
