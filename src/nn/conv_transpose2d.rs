//! A two dimension transposed convolution layer.
use crate::Tensor;
use std::borrow::Borrow;

pub struct ConvTranspose2DConfig {
    pub stride: i64,
    pub padding: i64,
    pub output_padding: i64,
    pub groups: i64,
    pub bias: bool,
    pub dilation: i64,
}

impl Default for ConvTranspose2DConfig {
    fn default() -> Self {
        ConvTranspose2DConfig {
            stride: 1,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
        }
    }
}

/// A two dimension transposed convolution layer.
pub struct ConvTranspose2D {
    ws: Tensor,
    bs: Tensor,
    stride: [i64; 2],
    padding: [i64; 2],
    output_padding: [i64; 2],
    dilation: [i64; 2],
    groups: i64,
}

impl ConvTranspose2D {
    pub fn new<'a, T: Borrow<super::Path<'a>>>(
        vs: T,
        in_dim: i64,
        out_dim: i64,
        ksize: i64,
        config: ConvTranspose2DConfig,
    ) -> ConvTranspose2D {
        let vs = vs.borrow();
        let ConvTranspose2DConfig {
            stride,
            padding,
            output_padding,
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
        ConvTranspose2D {
            ws,
            bs,
            stride: [stride, stride],
            padding: [padding, padding],
            output_padding: [output_padding, output_padding],
            dilation: [dilation, dilation],
            groups,
        }
    }
}

impl super::module::Module for ConvTranspose2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::conv_transpose2d(
            &xs,
            &self.ws,
            &self.bs,
            &self.stride,
            &self.padding,
            &self.output_padding,
            self.groups,
            &self.dilation,
        )
    }
}
