use super::utils::TorchError;
use libc::c_int;

#[repr(C)]
pub struct CIValue {
    _private: [u8; 0],
}

extern "C" {
    // Constructors
    fn ati_int(v: i64) -> *mut CIValue;
    fn ati_double(v: f64) -> *mut CIValue;
    fn ati_tensor(v: *mut super::tensor::C_tensor) -> *mut CIValue;
    fn ati_tuple(v: *const *mut CIValue, n: c_int) -> *mut CIValue;

    // Type query
    fn ati_tag(arg: *mut CIValue) -> c_int;

    // Getters
    fn ati_to_int(arg: *mut CIValue) -> i64;
    fn ati_to_double(arg: *mut CIValue) -> f64;
    fn ati_to_tensor(arg: *mut CIValue) -> *mut super::tensor::C_tensor;
    fn ati_tuple_length(arg: *mut CIValue) -> c_int;
    fn ati_to_tuple(arg: *mut CIValue, outputs: *mut *mut CIValue, n: c_int) -> c_int;

    fn ati_free(arg: *mut CIValue);
}

#[derive(Debug)]
pub enum IValue {
    Tensor(crate::Tensor),
    Int(i64),
    Double(f64),
    Tuple(Vec<IValue>),
}

impl IValue {
    pub fn to_c(&self) -> *mut CIValue {
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
    pub fn of_c(c_ivalue: *mut CIValue) -> Result<Self, TorchError> {
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
