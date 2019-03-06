//! Scalar elements.
use failure::Fallible;

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

    /// Returns an integer value.
    pub fn to_int(&self) -> Fallible<i64> {
        let i = unsafe_torch_err!({ torch_sys::ats_to_int(self.c_scalar) });
        Ok(i)
    }

    /// Returns a float value.
    pub fn to_float(&self) -> Fallible<f64> {
        let f = unsafe_torch_err!({ torch_sys::ats_to_float(self.c_scalar) });
        Ok(f)
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

impl From<Scalar> for i64 {
    fn from(s: Scalar) -> i64 {
        s.to_int().unwrap()
    }
}

impl From<Scalar> for f64 {
    fn from(s: Scalar) -> f64 {
        s.to_float().unwrap()
    }
}

impl From<&Scalar> for i64 {
    fn from(s: &Scalar) -> i64 {
        s.to_int().unwrap()
    }
}

impl From<&Scalar> for f64 {
    fn from(s: &Scalar) -> f64 {
        s.to_float().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::Scalar;
    #[test]
    fn scalar() {
        let pi = Scalar::float(3.14159265358979);
        assert_eq!(i64::from(&pi), 3);
        assert_eq!(f64::from(&pi), 3.14159265358979);
        let leet = Scalar::int(1337);
        assert_eq!(i64::from(&leet), 1337);
        assert_eq!(f64::from(&leet), 1337.);
    }
}
