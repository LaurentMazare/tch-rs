//! An instance-normalization layer.
use crate::Tensor;

use std::borrow::Borrow;

/// Instance-normalization config.
#[derive(Debug, Clone, Copy)]
pub struct InstanceNormConfig {
    pub cudnn_enabled: bool,
    pub eps: f64,
    pub momentum: f64,
    pub affine: bool,
    pub track_running_stats: bool,
    pub ws_init: super::Init,
    pub bs_init: super::Init,
}

impl Default for InstanceNormConfig {
    fn default() -> Self {
        InstanceNormConfig {
            cudnn_enabled: true,
            eps: 1e-5,
            momentum: 0.1,
            affine: false,
            track_running_stats: false,
            ws_init: super::Init::Uniform { lo: 0., up: 1. },
            bs_init: super::Init::Const(0.),
        }
    }
}

/// A Instance-normalization layer.
#[derive(Debug)]
pub struct InstanceNorm {
    config: InstanceNormConfig,
    pub running_stats: Option<(Tensor, Tensor)>,
    pub ws: Option<Tensor>,
    pub bs: Option<Tensor>,
    pub nd: usize,
}

fn instance_norm<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    nd: usize,
    out_dim: i64,
    config: InstanceNormConfig,
) -> InstanceNorm {
    let vs = vs.borrow();
    let running_stats = if config.track_running_stats {
        Some((
            vs.zeros_no_train("running_mean", &[out_dim]),
            vs.ones_no_train("running_var", &[out_dim]),
        ))
    } else {
        None
    };

    let (ws, bs) = if config.affine {
        (
            Some(vs.var("weight", &[out_dim], config.ws_init)),
            Some(vs.var("bias", &[out_dim], config.bs_init)),
        )
    } else {
        (None, None)
    };

    InstanceNorm { config, running_stats, ws, bs, nd }
}

/// Applies Instance Normalization over a three dimension input.
///
/// The input shape is assumed to be (N, C, L). Normalization
/// is performed over the first instance dimension N.
pub fn instance_norm1d<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    out_dim: i64,
    config: InstanceNormConfig,
) -> InstanceNorm {
    instance_norm(vs, 1, out_dim, config)
}

/// Applies Instance Normalization over a four dimension input.
///
/// The input shape is assumed to be (N, C, H, W). Normalization
/// is performed over the first instance dimension N.
pub fn instance_norm2d<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    out_dim: i64,
    config: InstanceNormConfig,
) -> InstanceNorm {
    instance_norm(vs, 2, out_dim, config)
}

/// Applies Instance Normalization over a five dimension input.
///
/// The input shape is assumed to be (N, C, D, H, W). Normalization
/// is performed over the first instance dimension N.
pub fn instance_norm3d<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    out_dim: i64,
    config: InstanceNormConfig,
) -> InstanceNorm {
    instance_norm(vs, 3, out_dim, config)
}

impl super::module::ModuleT for InstanceNorm {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let dim = xs.dim();
        if self.nd == 1 && dim != 2 && dim != 3 {
            panic!("expected an input tensor with 2 or 3 dims, got {:?}", xs.size())
        }
        if self.nd > 1 && xs.dim() != self.nd + 2 {
            panic!("expected an input tensor with {} dims, got {:?}", self.nd + 2, xs.size())
        };

        let (track_running_stats, running_mean, running_var) = match self.running_stats {
            Some((ref running_mean, ref running_var)) => {
                (true, Some(running_mean), Some(running_var))
            }
            None => (false, None, None),
        };

        Tensor::instance_norm(
            xs,
            self.ws.as_ref(),
            self.bs.as_ref(),
            running_mean,
            running_var,
            train || !track_running_stats,
            self.config.momentum,
            self.config.eps,
            self.config.cudnn_enabled,
        )
    }
}
