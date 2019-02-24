use super::tensor::{C_tensor, Tensor};
use crate::utils::{path_to_str, TorchError};
use libc::{c_char, c_int};

#[repr(C)]
pub struct CIValue {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CModule_ {
    _private: [u8; 0],
}

extern "C" {
    // Constructors
    fn ati_int(v: i64) -> *mut CIValue;
    fn ati_double(v: f64) -> *mut CIValue;
    fn ati_tensor(v: *mut C_tensor) -> *mut CIValue;
    fn ati_tuple(v: *const *mut CIValue, n: c_int) -> *mut CIValue;

    // Type query
    fn ati_tag(arg: *mut CIValue) -> c_int;

    // Getters
    fn ati_to_int(arg: *mut CIValue) -> i64;
    fn ati_to_double(arg: *mut CIValue) -> f64;
    fn ati_to_tensor(arg: *mut CIValue) -> *mut C_tensor;
    fn ati_tuple_length(arg: *mut CIValue) -> c_int;
    fn ati_to_tuple(arg: *mut CIValue, outputs: *mut *mut CIValue, n: c_int) -> c_int;

    fn ati_free(arg: *mut CIValue);

    fn atm_load(filename: *const c_char) -> *mut CModule_;
    fn atm_forward(m: *mut CModule_, args: *const *mut C_tensor, n: c_int) -> *mut C_tensor;
    fn atm_forward_(m: *mut CModule_, args: *const *mut CIValue, n: c_int) -> *mut CIValue;
    fn atm_free(m: *mut CModule_);
}

#[derive(Debug)]
pub enum IValue {
    Tensor(crate::Tensor),
    Int(i64),
    Double(f64),
    Tuple(Vec<IValue>),
}

impl IValue {
    pub(super) fn to_c(&self) -> *mut CIValue {
        unsafe_torch!({
            match self {
                IValue::Tensor(tensor) => ati_tensor(tensor.c_tensor),
                IValue::Int(i) => ati_int(*i),
                IValue::Double(f) => ati_double(*f),
                IValue::Tuple(v) => {
                    let v: Vec<_> = v.iter().map(|x| x.to_c()).collect();
                    ati_tuple(v.as_ptr(), v.len() as c_int)
                }
            }
        })
    }

    // This consumes the pointer and frees the associated memory.
    pub(super) fn of_c(c_ivalue: *mut CIValue) -> Result<Self, TorchError> {
        let tag = unsafe_torch_err!({ ati_tag(c_ivalue) });
        let v = match tag {
            0 => {
                let c_tensor = unsafe_torch_err!({ ati_to_tensor(c_ivalue) });
                IValue::Tensor(crate::Tensor { c_tensor })
            }
            1 => IValue::Int(unsafe_torch_err!({ ati_to_int(c_ivalue) })),
            2 => IValue::Double(unsafe_torch_err!({ ati_to_double(c_ivalue) })),
            3 => {
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
            _ => Err(TorchError::new(format!("unhandled tag {}", tag)))?,
        };
        unsafe_torch_err!({ ati_free(c_ivalue) });
        Ok(v)
    }
}

pub struct CModule {
    pub(crate) c_module: *mut CModule_,
}

impl Drop for CModule {
    fn drop(&mut self) {
        unsafe_torch!({ atm_free(self.c_module) })
    }
}

impl CModule {
    pub fn load(path: &std::path::Path) -> Result<CModule, TorchError> {
        let path = std::ffi::CString::new(path_to_str(path)?)?;
        let c_module = unsafe_torch_err!({ atm_load(path.as_ptr()) });
        Ok(CModule { c_module })
    }

    pub fn forward(&self, ts: &[&Tensor]) -> Result<Tensor, TorchError> {
        let ts: Vec<_> = ts.iter().map(|x| x.c_tensor).collect();
        let c_tensor = unsafe_torch_err!({ atm_forward(self.c_module, ts.as_ptr(), ts.len() as c_int) });
        Ok(Tensor { c_tensor })
    }

    pub fn forward_(&self, ts: &[&IValue]) -> Result<IValue, TorchError> {
        let ts: Vec<_> = ts.iter().map(|x| x.to_c()).collect();
        let c_ivalue = unsafe_torch_err!({ atm_forward_(self.c_module, ts.as_ptr(), ts.len() as c_int) });
        IValue::of_c(c_ivalue)
    }
}

#[cfg(test)]
mod tests {
    use super::IValue;
    #[test]
    fn ivalue() {
        let ivalue = IValue::Tuple(vec![IValue::Int(42), IValue::Double(3.1415)]);
        let ivalue2 = IValue::of_c(ivalue.to_c()).unwrap();
        assert_eq!(format!("{:?}", ivalue), format!("{:?}", ivalue2));
    }
}
