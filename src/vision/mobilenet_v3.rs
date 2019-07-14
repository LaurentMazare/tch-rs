//! MobileNetV3 implementation.
//!
//! See "Searching for MobileNetV3" Andrew Howard et al. 2019
//! https://arxiv.org/abs/1905.02244
use crate::{
    nn::{self, Module, ModuleT, Path},
    Tensor,
};
use std::borrow::Borrow;

#[derive(Debug, Copy, Clone)]
enum Mode {
    Small,
    Large,
}

#[derive(Debug, Copy, Clone)]
enum NL {
    ReLU,
    HSwish,
}

#[derive(Debug, Copy, Clone)]
enum SE {
    SEModule,
    Identity,
}

#[derive(Debug, Clone)]
pub struct MobileNetV3Config {
    pub dropout: f64,
    pub width_mult: f64,
}

/// Create default model parameters for MobileNetV3
///
/// It suggests 0.8 for dropout and 1.0 for width-mult.
impl Default for MobileNetV3Config {
    fn default() -> Self {
        MobileNetV3Config {
            dropout: 0.8,
            width_mult: 1.0,
        }
    }
}

fn mobile_module(
    path: &Path,
    c_in: i64,
    c_out: i64,
    k: i64,
    s: i64,
    c_exp: i64,
    se: SE,
    nl: NL,
) -> impl ModuleT {
    assert!(vec![1, 2].contains(&s), "stride should be either 1 or 2");
    assert!(
        vec![3, 5].contains(&k),
        "kernel size should be either 3 or 5"
    );

    let padding = (k - 1) / 2;
    let use_res_connect = s == 1 && c_in == c_out;
    let se_module = squeeze_module(&(path / "squeeze_module"), c_exp, None);
    let se_layer = nn::func(move |xs: &Tensor| match se {
        SE::SEModule => xs.apply(&se_module),
        SE::Identity => xs.shallow_clone(),
    });

    // Build conv blocks
    let conv_blocks = nn::seq_t()
        .add(conv_layer(path / "conv_pw", c_in, c_exp, 1, 1, 0, None))
        .add(norm_layer(path / "bn_pw", c_exp))
        .add(nonlinear_layer(nl))
        .add(conv_layer(
            path / "conv_dw",
            c_exp,
            c_exp,
            k,
            s,
            padding,
            Some(c_exp),
        ))
        .add(norm_layer(path / "bn_dw", c_exp))
        .add(se_layer)
        .add(conv_layer(
            path / "conv_pw_linear",
            c_exp,
            c_out,
            1,
            1,
            0,
            None,
        ))
        .add(norm_layer(path / "bn_pw_linear", c_out))
        .add(nonlinear_layer(nl));

    nn::func_t(move |xs, train| match use_res_connect {
        true => xs.apply_t(&conv_blocks, train) + xs,
        false => xs.apply_t(&conv_blocks, train),
    })
}

fn squeeze_module(path: &Path, channel: i64, reduction_opt: Option<i64>) -> impl Module {
    let reduction = match reduction_opt {
        Some(r) => r,
        None => 4,
    };

    let squeeze_channel = channel / reduction;

    let linear_layer_1 = nn::linear(
        path / "linear1",
        channel,
        squeeze_channel,
        nn::LinearConfig {
            bs_init: None,
            ..Default::default()
        },
    );

    let linear_layer_2 = nn::linear(
        path / "linear2",
        squeeze_channel,
        channel,
        nn::LinearConfig {
            bs_init: None,
            ..Default::default()
        },
    );

    let fc = nn::seq()
        .add(linear_layer_1)
        .add_fn(|xs| xs.relu())
        .add(linear_layer_2)
        .add(h_sigmoid());

    nn::func(move |xs| {
        let size = xs.size();
        let bsize = size[0];
        let channel = size[1];
        let y = xs
            .adaptive_avg_pool2d(&[1, 1])
            .view([bsize, channel])
            .apply(&fc)
            .view([bsize, channel, 1, 1]);
        xs * y.expand_as(&xs)
    })
}

// Helper functions

/// Conolution layer
fn conv_layer<'a, P: Borrow<Path<'a>>>(
    path: P,
    input_channel: i64,
    output_channel: i64,
    kernel: i64,
    stride: i64,
    padding: i64,
    groups: Option<i64>,
) -> nn::Conv2D {
    let mut config = nn::ConvConfig {
        stride: stride,
        padding: padding,
        bias: false,
        ..Default::default()
    };

    if let Some(g) = groups {
        config.groups = g;
    }

    nn::conv2d(path, input_channel, output_channel, kernel, config)
}

/// Batch-norm layer
fn norm_layer<'a, P: Borrow<Path<'a>>>(path: P, channel: i64) -> nn::BatchNorm {
    nn::batch_norm2d(
        path,
        channel,
        nn::BatchNormConfig {
            cudnn_enabled: true,
            momentum: 0.01,
            ..Default::default()
        },
    )
}

/// Non-linear layer
fn nonlinear_layer<'a>(nl: NL) -> nn::Func<'a> {
    match nl {
        NL::ReLU => nn::func(|xs| xs.relu()),
        NL::HSwish => nn::func(|xs| xs.apply(&h_swish())),
    }
}

/// H-Sigmoid activation
fn h_sigmoid<'a>() -> nn::Func<'a> {
    nn::func(|xs| (xs + 3.).relu().clamp_max(6.) / 6.)
}

/// H-Swish activation
fn h_swish<'a>() -> nn::Func<'a> {
    nn::func(|xs| xs * xs.apply(&h_sigmoid()))
}

/// Main model building function
fn build_mobilenet_v3(
    path: &Path,
    input_channel: i64,
    n_classes: i64,
    dropout: f64,
    width_mult: f64,
    mode: Mode,
) -> impl ModuleT {
    assert!(
        input_channel.is_positive(),
        "input_channel must be positive integer, but get {}",
        input_channel
    );
    assert!(
        n_classes.is_positive(),
        "n_classes must be positive integer, but get {}",
        n_classes
    );
    assert!(
        !dropout.is_sign_negative(),
        "dropout should be non-negative, but get {}",
        dropout
    );
    assert!(
        width_mult.is_sign_positive(),
        "width_mult should be positive, but get {}",
        width_mult
    );

    let mobile_setting = match mode {
        Mode::Small => {
            vec![
                // k, exp, c, se, nl, s
                (3, 16, 16, SE::SEModule, NL::ReLU, 2),
                (3, 72, 24, SE::Identity, NL::ReLU, 2),
                (3, 88, 24, SE::Identity, NL::ReLU, 1),
                (5, 96, 40, SE::SEModule, NL::HSwish, 2),
                (5, 240, 40, SE::SEModule, NL::HSwish, 1),
                (5, 240, 40, SE::SEModule, NL::HSwish, 1),
                (5, 120, 48, SE::SEModule, NL::HSwish, 1),
                (5, 144, 48, SE::SEModule, NL::HSwish, 1),
                (5, 288, 96, SE::SEModule, NL::HSwish, 2),
                (5, 576, 96, SE::SEModule, NL::HSwish, 1),
                (5, 576, 96, SE::SEModule, NL::HSwish, 1),
            ]
        }
        Mode::Large => {
            vec![
                // k, exp, c, se, nl, s
                (3, 16, 16, SE::Identity, NL::ReLU, 1),
                (3, 64, 24, SE::Identity, NL::ReLU, 2),
                (3, 72, 24, SE::Identity, NL::ReLU, 1),
                (5, 72, 40, SE::SEModule, NL::ReLU, 2),
                (5, 120, 40, SE::SEModule, NL::ReLU, 1),
                (5, 120, 40, SE::SEModule, NL::ReLU, 1),
                (3, 240, 80, SE::Identity, NL::HSwish, 2),
                (3, 200, 80, SE::Identity, NL::HSwish, 1),
                (3, 184, 80, SE::Identity, NL::HSwish, 1),
                (3, 184, 80, SE::Identity, NL::HSwish, 1),
                (3, 480, 112, SE::SEModule, NL::HSwish, 1),
                (3, 672, 112, SE::SEModule, NL::HSwish, 1),
                (5, 672, 160, SE::SEModule, NL::HSwish, 2),
                (5, 960, 160, SE::SEModule, NL::HSwish, 1),
                (5, 960, 160, SE::SEModule, NL::HSwish, 1),
            ]
        }
    };

    // Channel scaling function
    let multiplied_channel = |val| {
        let divisor = 8.;
        let multiplied = val as f64 * width_mult;
        let mut new_val = (multiplied / divisor + 0.5).floor() * divisor;
        if new_val < val as f64 * 0.9 {
            new_val += divisor;
        }
        new_val as i64
    };

    // Start to build model here
    let mut layers = nn::seq_t();

    // First conv layer
    let mut c_in = input_channel;
    let mut c_out = multiplied_channel(16);
    {
        let path_first = path / "init_block";
        layers = layers
            .add(conv_layer(&path_first / "conv", c_in, c_out, 3, 2, 1, None))
            .add(norm_layer(&path_first / "bn", c_out))
            .add(nonlinear_layer(NL::HSwish));
    }
    c_in = c_out;

    // Mobile blocks
    for (ind, (k, exp, c, se, nl, s)) in mobile_setting.into_iter().enumerate() {
        c_out = multiplied_channel(c);
        let path_block = path / format!("mobile_{}", ind);
        let c_exp = multiplied_channel(exp);
        let block = mobile_module(&path_block, c_in, c_out, k, s, c_exp, se, nl);
        layers = layers.add(block);
        c_in = c_out;
    }

    // Conv block
    c_out = match mode {
        Mode::Large => multiplied_channel(960),
        Mode::Small => multiplied_channel(576),
    };
    {
        let path_1x1 = path / "conv_1x1_bn";
        layers = layers
            .add(conv_layer(&path_1x1 / "conv", c_in, c_out, 1, 1, 0, None))
            .add(norm_layer(&path_1x1 / "bn", c_out))
            .add(h_swish());
    }
    c_in = c_out;

    // Average pooling layer
    layers = layers.add_fn(|xs| xs.adaptive_avg_pool2d(&[1, 1]));

    // Conv block
    c_out = multiplied_channel(1280);
    {
        let path_1x1 = path / "conv_1x1_nbn";
        layers = layers
            .add(conv_layer(&path_1x1, c_in, c_out, 1, 1, 0, None))
            .add(h_swish());
    }
    c_in = c_out;

    // Conv block
    c_out = n_classes;
    {
        let path_1x1 = path / "conv_1x1_nbn_last";
        layers = layers
            .add(conv_layer(&path_1x1, c_in, c_out, 1, 1, 0, None))
            .add(h_swish());
    }

    // Flatten
    layers.add_fn(|xs| xs.flatten(1, -1))
}

/// Build large MobileNetV3 model
pub fn v3_large(
    path: &Path,
    input_channel: i64,
    n_classes: i64,
    config: MobileNetV3Config,
) -> impl ModuleT {
    build_mobilenet_v3(
        path,
        input_channel,
        n_classes,
        config.dropout,
        config.width_mult,
        Mode::Large,
    )
}

/// A layer defined by a closure with an additional training parameter.
pub fn v3_small(
    path: &Path,
    input_channel: i64,
    n_classes: i64,
    config: MobileNetV3Config,
) -> impl ModuleT {
    build_mobilenet_v3(
        path,
        input_channel,
        n_classes,
        config.dropout,
        config.width_mult,
        Mode::Small,
    )
}
