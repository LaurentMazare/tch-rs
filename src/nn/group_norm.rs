//! A group-normalization layer.
use crate::Tensor;
use std::borrow::Borrow;

/// Group-normalization config.
#[derive(Debug, Clone, Copy)]
pub struct GroupNormConfig {
    pub cudnn_enabled: bool,
    pub eps: f64,
    pub ws_init: super::Init,
    pub bs_init: super::Init,
}

impl Default for GroupNormConfig {
    fn default() -> Self {
        GroupNormConfig {
            cudnn_enabled: true,
            eps: 1e-5,
            ws_init: super::Init::Uniform { lo: 0., up: 1. },
            bs_init: super::Init::Const(0.),
        }
    }
}

/// A group-normalization layer.
#[derive(Debug)]
pub struct GroupNorm {
    num_groups: i64,
    config: GroupNormConfig,
    pub ws: Tensor,
    pub bs: Tensor,
    pub nd: usize,
}

fn group_norm<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    nd: usize,
    num_groups: i64,
    input_channels: i64,
    config: GroupNormConfig,
) -> GroupNorm {
    let vs = vs.borrow();
    GroupNorm {
        num_groups,
        config,
        ws: vs.var("weight", &[input_channels], config.ws_init),
        bs: vs.var("bias", &[input_channels], config.bs_init),
        nd,
    }
}

/// Applies Group Normalization over a three dimension input.
///
/// The input shape is assumed to be (N, C, L). Normalization
/// is performed over the channel dimension C.
pub fn group_norm1d<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    num_groups: i64,
    input_channels: i64,
    config: GroupNormConfig,
) -> GroupNorm {
    group_norm(vs, 1, num_groups, input_channels, config)
}

/// Applies Group Normalization over a four dimension input.
///
/// The input shape is assumed to be (N, C, H, W). Normalization
/// is performed over the channel dimension C.
pub fn group_norm2d<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    num_groups: i64,
    input_channels: i64,
    config: GroupNormConfig,
) -> GroupNorm {
    group_norm(vs, 2, num_groups, input_channels, config)
}

/// Applies Group Normalization over a five dimension input.
///
/// The input shape is assumed to be (N, C, D, H, W). Normalization
/// is performed over the channel dimension C.
pub fn group_norm3d<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    num_groups: i64,
    input_channels: i64,
    config: GroupNormConfig,
) -> GroupNorm {
    group_norm(vs, 3, num_groups, input_channels, config)
}

impl super::module::Module for GroupNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let dim = xs.dim();
        if self.nd == 1 && dim != 2 && dim != 3 {
            panic!(
                "expected an input tensor with 2 or 3 dims, got {:?}",
                xs.size()
            )
        }
        if self.nd > 1 && xs.dim() != self.nd + 2 {
            panic!(
                "expected an input tensor with {} dims, got {:?}",
                self.nd + 2,
                xs.size()
            )
        };
        Tensor::group_norm(
            xs,
            self.num_groups,
            Some(&self.ws),
            Some(&self.bs),
            self.config.eps,
            self.config.cudnn_enabled,
        )
    }
}
