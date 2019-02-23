use crate::utils::read_and_clean_error;

#[repr(C)]
pub struct C_scalar {
    _private: [u8; 0],
}

extern "C" {
    fn ats_int(v: i64) -> *mut C_scalar;
    fn ats_float(v: f64) -> *mut C_scalar;
    fn ats_free(arg: *mut C_scalar);
}

pub struct Scalar {
    pub(crate) c_scalar: *mut C_scalar,
}

impl Scalar {
    pub fn int(v: i64) -> Scalar {
        let c_scalar = unsafe_torch!({ ats_int(v) });
        Scalar { c_scalar }
    }

    pub fn float(v: f64) -> Scalar {
        let c_scalar = unsafe_torch!({ ats_float(v) });
        Scalar { c_scalar }
    }
}

impl Drop for Scalar {
    fn drop(&mut self) {
        unsafe_torch!({ ats_free(self.c_scalar) })
    }
}

impl From<i64> for Scalar {
    fn from(v: i64) -> Scalar {
        Scalar::int(v)
    }
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Scalar {
        Scalar::float(v)
    }
}
