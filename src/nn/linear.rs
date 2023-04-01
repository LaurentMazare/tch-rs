//! A linear fully-connected layer.
use crate::Tensor;
use std::borrow::Borrow;

/// Configuration for a linear layer.
#[derive(Debug, Clone, Copy)]
pub struct LinearConfig {
    pub ws_init: super::Init,
    pub bs_init: Option<super::Init>,
    pub bias: bool,
}

impl Default for LinearConfig {
    fn default() -> Self {
        LinearConfig { ws_init: super::init::DEFAULT_KAIMING_UNIFORM, bs_init: None, bias: true }
    }
}

/// A linear fully-connected layer.
#[derive(Debug)]
pub struct Linear {
    pub ws: Tensor,
    pub bs: Option<Tensor>,
}

/// Creates a new linear layer.
pub fn linear<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    c: LinearConfig,
) -> Linear {
    let vs = vs.borrow();
    let bs = if c.bias {
        let bs_init = c.bs_init.unwrap_or_else(|| {
            let bound = 1.0 / (in_dim as f64).sqrt();
            super::Init::Uniform { lo: -bound, up: bound }
        });
        Some(vs.var("bias", &[out_dim], bs_init))
    } else {
        None
    };

    Linear { ws: vs.var("weight", &[out_dim, in_dim], c.ws_init), bs }
}

impl super::module::Module for Linear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.linear(&self.ws, self.bs.as_ref())
    }
}

#[test]
/// To run this test, generate test tensors:
///
/// ```shell
/// cd tests/linear
/// python gen-test-data.py
/// ```
#[ignore]
fn matches_pytorch() {
    use crate::nn::Module;

    let input = Tensor::read_npy("tests/linear/in.npy").unwrap();
    let expected_output = Tensor::read_npy("tests/linear/out.npy").unwrap();
    let ws = Tensor::read_npy("tests/linear/ws.npy").unwrap();
    let bs = Tensor::read_npy("tests/linear/bs.npy").unwrap();

    let linear = Linear { ws, bs: Some(bs) };
    let output = linear.forward(&input);
    assert!(output.allclose(&expected_output, 1e-5, 1e-8, false));
}
