//! Variable initialization.
use crate::{Device, Kind, Tensor};

#[derive(Debug, Copy, Clone)]
pub enum Init {
    Const(f64),
    Randn { mean: f64, stdev: f64 },
    Uniform { lo: f64, up: f64 },
    KaimingUniform,
}

pub fn init(i: Init, dims: &[i64], device: Device) -> Tensor {
    match i {
        Init::Const(cst) => {
            // Optimize the case for which a single C++ code can be done.
            if cst == 0. {
                Tensor::zeros(dims, (Kind::Float, device))
            } else if cst == 1. {
                Tensor::ones(dims, (Kind::Float, device))
            } else {
                Tensor::ones(dims, (Kind::Float, device)) * cst
            }
        }
        Init::Uniform { lo, up } => Tensor::zeros(dims, (Kind::Float, device)).uniform_(lo, up),
        Init::Randn { mean, stdev } => {
            if mean == 0. && stdev == 1. {
                Tensor::randn(dims, (Kind::Float, device))
            } else {
                Tensor::randn(dims, (Kind::Float, device)) * stdev + mean
            }
        }
        Init::KaimingUniform => {
            let fan_in: i64 = dims.iter().skip(1).product();
            let bound = (1.0 / fan_in as f64).sqrt();
            Tensor::zeros(dims, (Kind::Float, device)).uniform_(-bound, bound)
        }
    }
}

impl Init {
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
        }
    }
}

impl Tensor {
    pub fn init(&mut self, i: Init) {
        i.set(self)
    }
}
