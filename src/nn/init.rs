//! Variable initialization.
use crate::{Device, Kind, TchError, Tensor};

/// Variable initializations.
#[derive(Debug, Copy, Clone)]
pub enum Init {
    /// Constant value.
    Const(f64),

    /// Random normal with some mean and standard deviation.
    Randn { mean: f64, stdev: f64 },

    /// Uniform initialization between some lower and upper bounds.
    Uniform { lo: f64, up: f64 },

    /// Kaiming uniform initialization.
    KaimingUniform,

    /// Orthogonal initialization
    Orthogonal { gain: f64 },
}

/// Creates a new float tensor with the specified shape, device, and initialization.
pub fn f_init(i: Init, dims: &[i64], device: Device) -> Result<Tensor, TchError> {
    match i {
        Init::Const(cst) => {
            // Optimize the case for which a single C++ code can be done.
            if cst == 0. {
                Tensor::f_zeros(dims, (Kind::Float, device))
            } else if (cst - 1.).abs() <= std::f64::EPSILON {
                Tensor::f_ones(dims, (Kind::Float, device))
            } else {
                Tensor::f_ones(dims, (Kind::Float, device)).map(|t| t * cst)
            }
        }
        Init::Uniform { lo, up } => {
            Tensor::f_zeros(dims, (Kind::Float, device))?.f_uniform_(lo, up)
        }
        Init::Randn { mean, stdev } => {
            if mean == 0. && (stdev - 1.).abs() <= std::f64::EPSILON {
                Tensor::f_randn(dims, (Kind::Float, device))
            } else {
                Tensor::f_randn(dims, (Kind::Float, device)).map(|t| t * stdev + mean)
            }
        }
        Init::KaimingUniform => {
            let fan_in: i64 = dims.iter().skip(1).product();
            let bound = (1.0 / fan_in as f64).sqrt();
            Tensor::f_zeros(dims, (Kind::Float, device))?.f_uniform_(-bound, bound)
        }
        Init::Orthogonal { gain } => {
            if dims.len() < 2 {
                return Err(TchError::Shape(
                    "Only tensors with 2 or more dimensions are supported".to_string(),
                ));
            }
            let rows = dims[0];
            let cols: i64 = dims.iter().skip(1).product();

            let mut flattened =
                Tensor::f_empty(&[rows, cols], (Kind::Float, device))?.f_normal_(0.0, 1.0)?;
            let flattened = if rows < cols { flattened.f_t_()? } else { flattened };

            let (mut q, r) = flattened.f_linalg_qr("reduced")?;
            let d = r.f_diag(0)?;
            let ph = d.f_sign()?;
            q *= ph;

            let mut q = if rows < cols { q.f_t_()? } else { q };
            crate::no_grad(|| q *= gain);

            q.f_contiguous()
        }
    }
}

/// Creates a new float tensor with the specified shape, device, and initialization.
pub fn init(i: Init, dims: &[i64], device: Device) -> Tensor {
    f_init(i, dims, device).unwrap()
}

impl Init {
    /// Re-initializes an existing tensor with the specified initialization
    pub fn set(self, tensor: &mut Tensor) {
        match self {
            Init::Const(cst) => {
                let _ = tensor.fill_(cst);
            }
            Init::Uniform { lo, up } => {
                let _ = tensor.uniform_(lo, up);
            }
            Init::KaimingUniform => {
                let fan_in: i64 = tensor.size().iter().skip(1).product();
                let bound = (1.0 / fan_in as f64).sqrt();
                let _ = tensor.uniform_(-bound, bound);
            }
            Init::Randn { mean, stdev } => {
                tensor.copy_(&(tensor.randn_like() * stdev + mean));
            }
            Init::Orthogonal { gain } => {
                let q = f_init(Init::Orthogonal { gain }, &tensor.size(), tensor.device()).unwrap();
                crate::no_grad(|| tensor.view_as(&q).copy_(&q));
            }
        }
    }
}

impl Tensor {
    /// Re-initializes the tensor using the specified initialization.
    pub fn init(&mut self, i: Init) {
        i.set(self)
    }
}
