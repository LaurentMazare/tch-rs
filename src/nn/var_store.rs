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
        self.variables.push(z.clone());
        z
    }

    pub fn variables(&self) -> &[Tensor] {
        self.variables.as_slice()
    }
}
