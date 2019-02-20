// TODO: support alternative devices.
use crate::kind;
use crate::tensor::Tensor;

pub struct VarStore {
    variables: Vec<Tensor>,
}

impl VarStore {
    pub fn new() -> VarStore {
        VarStore {
            variables: Vec::new(),
        }
    }

    pub fn zeros(&mut self, dims: &[i64]) -> Tensor {
        let z = Tensor::zeros(dims, &kind::FLOAT_CPU).set_requires_grad(true);
        self.variables.push(z.shallow_clone());
        z
    }

    pub fn ones(&mut self, dims: &[i64]) -> Tensor {
        let z = Tensor::ones(dims, &kind::FLOAT_CPU).set_requires_grad(true);
        self.variables.push(z.shallow_clone());
        z
    }

    pub fn randn_standard(&mut self, dims: &[i64]) -> Tensor {
        let z = Tensor::randn(dims, &kind::FLOAT_CPU).set_requires_grad(true);
        self.variables.push(z.shallow_clone());
        z
    }

    pub fn randn(&mut self, dims: &[i64], mean: f64, stdev: f64) -> Tensor {
        let z = Tensor::randn(dims, &kind::FLOAT_CPU);
        let z = (z * stdev + mean).set_requires_grad(true);
        self.variables.push(z.shallow_clone());
        z
    }

    pub fn uniform(&mut self, dims: &[i64], lo:f64, up:f64) -> Tensor {
        let z = Tensor::zeros(dims, &kind::FLOAT_CPU)
            .uniform_(lo, up)
            .set_requires_grad(true);
        self.variables.push(z.shallow_clone());
        z
    }

    pub fn kaiming_uniform(&mut self, dims: &[i64]) -> Tensor {
        let fan_in: i64 = dims.iter().skip(1).product();
        let bound = (1.0 / fan_in as f64).sqrt();
        self.uniform(dims, - bound, bound)
    }

    pub fn variables(&self) -> &[Tensor] {
        self.variables.as_slice()
    }
}
