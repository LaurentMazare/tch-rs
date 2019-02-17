use crate::scalar::Scalar;

mod c_wrapper;

pub use c_wrapper::Tensor;

// TODO: create binary operator for references via a macro ?
impl std::ops::Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Tensor {
        self.add_tensor(&rhs)
    }
}

impl std::ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        self.mul_tensor(&rhs)
    }
}

impl std::ops::Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Tensor {
        self.add_scalar(&rhs.into())
    }
}

impl std::ops::Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Tensor {
        self.mul_scalar(&rhs.into())
    }
}

impl std::ops::Add<i64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: i64) -> Tensor {
        self.add_scalar(&rhs.into())
    }
}

impl std::ops::Mul<i64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: i64) -> Tensor {
        self.mul_scalar(&rhs.into())
    }
}

impl std::ops::Add<Scalar> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Scalar) -> Tensor {
        self.add_scalar(&rhs)
    }
}

impl std::ops::Mul<Scalar> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Scalar) -> Tensor {
        self.mul_scalar(&rhs)
    }
}

impl std::ops::AddAssign<Tensor> for Tensor {
    fn add_assign(&mut self, rhs: Tensor) {
        self.add_tensor_(&rhs)
    }
}

impl std::ops::MulAssign<Tensor> for Tensor {
    fn mul_assign(&mut self, rhs: Tensor) {
        self.mul_tensor_(&rhs)
    }
}

impl std::ops::AddAssign<f64> for Tensor {
    fn add_assign(&mut self, rhs: f64) {
        self.add_scalar_(&rhs.into())
    }
}

impl std::ops::MulAssign<f64> for Tensor {
    fn mul_assign(&mut self, rhs: f64) {
        self.mul_scalar_(&rhs.into())
    }
}

impl std::ops::AddAssign<i64> for Tensor {
    fn add_assign(&mut self, rhs: i64) {
        self.add_scalar_(&rhs.into())
    }
}

impl std::ops::MulAssign<i64> for Tensor {
    fn mul_assign(&mut self, rhs: i64) {
        self.mul_scalar_(&rhs.into())
    }
}

impl std::ops::AddAssign<Scalar> for Tensor {
    fn add_assign(&mut self, rhs: Scalar) {
        self.add_scalar_(&rhs)
    }
}

impl std::ops::MulAssign<Scalar> for Tensor {
    fn mul_assign(&mut self, rhs: Scalar) {
        self.mul_scalar_(&rhs)
    }
}

impl From<&[i64]> for Tensor {
    fn from(v: &[i64]) -> Tensor {
        Tensor::int_vec(v)
    }
}

impl From<&[f64]> for Tensor {
    fn from(v: &[f64]) -> Tensor {
        Tensor::float_vec(v)
    }
}

impl From<i64> for Tensor {
    fn from(v: i64) -> Tensor {
        Tensor::int_vec(&[v]).view(&[])
    }
}

impl From<f64> for Tensor {
    fn from(v: f64) -> Tensor {
        Tensor::float_vec(&[v]).view(&[])
    }
}
