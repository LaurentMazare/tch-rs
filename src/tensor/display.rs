use crate::{Kind, Tensor};

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.defined() {
            match self.f_kind() {
                Err(err) => write!(f, "Tensor[{:?}, {:?}]", self.size(), err),
                Ok(kind) => {
                    let (is_int, is_float) = match kind {
                        Kind::Int | Kind::Int8 | Kind::Uint8 | Kind::Int16 | Kind::Int64 => {
                            (true, false)
                        }
                        Kind::BFloat16
                        | Kind::QInt8
                        | Kind::QUInt8
                        | Kind::QInt32
                        | Kind::Half
                        | Kind::Float
                        | Kind::Double => (false, true),
                        Kind::Bool
                        | Kind::ComplexHalf
                        | Kind::ComplexFloat
                        | Kind::ComplexDouble => (false, false),
                    };
                    match (self.size().as_slice(), is_int, is_float) {
                        ([], true, false) => write!(f, "[{}]", i64::from(self)),
                        ([s], true, false) if *s < 10 => write!(f, "{:?}", Vec::<i64>::from(self)),
                        ([], false, true) => write!(f, "[{}]", f64::from(self)),
                        ([s], false, true) if *s < 10 => write!(f, "{:?}", Vec::<f64>::from(self)),
                        _ => write!(f, "Tensor[{:?}, {:?}]", self.size(), self.f_kind()),
                    }
                }
            }
        } else {
            write!(f, "Tensor[Undefined]")
        }
    }
}
