/// A two dimension convolution layer.
use crate::Tensor;
use std::borrow::Borrow;

#[derive(Builder, Debug, Clone, Copy)]
#[builder(setter(into))]
pub struct Conv2DConfig {
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub groups: i64,
    pub bias: bool,
}

impl Default for Conv2DConfig {
    fn default() -> Self {
        Conv2DConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
        }
    }
}

/// A two dimension convolution layer.
pub struct Conv2D {
    ws: Tensor,
    bs: Tensor,
    stride: [i64; 2],
    padding: [i64; 2],
    dilation: [i64; 2],
    groups: i64,
}

impl Conv2D {
    pub fn new<'a, T: Borrow<super::Path<'a>>>(
        vs: T,
        in_dim: i64,
        out_dim: i64,
        ksize: i64,
        config: Conv2DConfig,
    ) -> Conv2D {
        let vs = vs.borrow();
        let Conv2DConfig {
            stride,
            padding,
            dilation,
            groups,
            bias,
        } = config;
        let bs = if bias {
            vs.zeros("bias", &[out_dim])
        } else {
            Tensor::zeros(&[out_dim], (crate::Kind::Float, vs.device()))
        };
        let ws = vs.kaiming_uniform("weight", &[out_dim, in_dim, ksize, ksize]);
        Conv2D {
            ws,
            bs,
            stride: [stride, stride],
            padding: [padding, padding],
            dilation: [dilation, dilation],
            groups,
        }
    }
}

impl super::module::Module for Conv2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::conv2d(
            &xs,
            &self.ws,
            &self.bs,
            &self.stride,
            &self.padding,
            &self.dilation,
            self.groups,
        )
    }
}
