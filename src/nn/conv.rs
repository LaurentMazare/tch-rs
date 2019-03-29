//! N-dimensional convolution layers.
use super::Path;
use crate::Tensor;
use std::borrow::Borrow;

#[derive(Debug, Clone, Copy)]
pub struct ConvConfig {
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub groups: i64,
    pub bias: bool,
    pub ws_init: super::Init,
    pub bs_init: super::Init,
}

impl Default for ConvConfig {
    fn default() -> Self {
        ConvConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
            ws_init: super::Init::KaimingUniform,
            bs_init: super::Init::Const(0.),
        }
    }
}

// Use const generics when they have landed in stable rust.
/// A N-dimensional convolution layer.
#[derive(Debug)]
pub struct Conv<ND> {
    pub ws: Tensor,
    pub bs: Tensor,
    stride: ND,
    padding: ND,
    dilation: ND,
    groups: i64,
}

/// One dimension convolution layer.
pub type Conv1D = Conv<[i64; 1]>;

/// Two dimensions convolution layer.
pub type Conv2D = Conv<[i64; 2]>;

/// Three dimensions convolution layer.
pub type Conv3D = Conv<[i64; 3]>;

trait Create: std::convert::AsRef<[i64]> + std::marker::Sized {
    fn make_array(i: i64) -> Self;

    fn conv<'a, T: Borrow<super::Path<'a>>>(
        vs: T,
        in_dim: i64,
        out_dim: i64,
        ksize: i64,
        config: ConvConfig,
    ) -> Conv<Self> {
        let vs = vs.borrow();
        let ConvConfig {
            stride,
            padding,
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
        let mut weight_size = vec![out_dim, in_dim];
        weight_size.extend(Self::make_array(ksize).as_ref().into_iter());
        let ws = vs.var("weight", weight_size.as_slice(), ws_init);
        Conv {
            ws,
            bs,
            stride: Self::make_array(stride),
            padding: Self::make_array(padding),
            dilation: Self::make_array(dilation),
            groups,
        }
    }
}

impl Create for [i64; 1] {
    fn make_array(i: i64) -> Self {
        [i]
    }
}

impl Create for [i64; 2] {
    fn make_array(i: i64) -> Self {
        [i, i]
    }
}

impl Create for [i64; 3] {
    fn make_array(i: i64) -> Self {
        [i, i, i]
    }
}

impl Conv1D {
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T, i: i64, o: i64, k: i64, c: ConvConfig) -> Self {
        <[i64; 1]>::conv(vs, i, o, k, c)
    }
}

impl Conv2D {
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T, i: i64, o: i64, k: i64, c: ConvConfig) -> Self {
        <[i64; 2]>::conv(vs, i, o, k, c)
    }
}

impl Conv3D {
    pub fn new<'a, T: Borrow<Path<'a>>>(vs: T, i: i64, o: i64, k: i64, c: ConvConfig) -> Self {
        <[i64; 3]>::conv(vs, i, o, k, c)
    }
}

impl super::module::Module for Conv1D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::conv1d(
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

impl super::module::Module for Conv3D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::conv3d(
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
