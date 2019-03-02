/// Scalar elements.

pub struct Scalar {
    pub(crate) c_scalar: *mut torch_sys::C_scalar,
}

impl Scalar {
    /// Creates an integer scalar.
    pub fn int(v: i64) -> Scalar {
        let c_scalar = unsafe_torch!({ torch_sys::ats_int(v) });
        Scalar { c_scalar }
    }

    /// Creates a float scalar scalar.
    pub fn float(v: f64) -> Scalar {
        let c_scalar = unsafe_torch!({ torch_sys::ats_float(v) });
        Scalar { c_scalar }
    }
}

impl Drop for Scalar {
    fn drop(&mut self) {
        unsafe_torch!({ torch_sys::ats_free(self.c_scalar) })
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
