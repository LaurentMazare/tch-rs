//! A two dimension transposed convolution layer.
use crate::Tensor;
use std::borrow::Borrow;

#[derive(Builder, Debug, Clone, Copy)]
#[builder(default)]
pub struct ConvTranspose2DConfig {
    pub stride: i64,
    pub padding: i64,
    pub output_padding: i64,
    pub groups: i64,
    pub bias: bool,
    pub dilation: i64,
    pub ws_init: super::Init,
    pub bs_init: super::Init,
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
            ws_init: super::Init::KaimingUniform,
            bs_init: super::Init::Const(0.),
        }
    }
}

/// A two dimension transposed convolution layer.
#[derive(Debug)]
pub struct ConvTranspose2D {
    pub ws: Tensor,
    pub bs: Tensor,
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
            ws_init,
            bs_init,
        } = config;
        let bs = if bias {
            vs.var("bias", &[out_dim], bs_init)
        } else {
            Tensor::zeros(&[out_dim], (crate::Kind::Float, vs.device()))
        };
        let ws = vs.var("weight", &[in_dim, out_dim, ksize, ksize], ws_init);
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
        xs.conv_transpose2d(
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
