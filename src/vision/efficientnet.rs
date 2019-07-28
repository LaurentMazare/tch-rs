//! EfficientNet implementation.
use crate::nn::{self, ModuleT};
use crate::Tensor;

const BATCH_NORM_MOMENTUM: f64 = 0.99;
const BATCH_NORM_EPSILON: f64 = 1e-3;

#[derive(Debug, Clone, Copy)]
pub struct BlockArgs {
    kernel_size: i64,
    num_repeat: i64,
    input_filters: i64,
    output_filters: i64,
    expand_ratio: i64,
    id_skip: bool,
    se_ratio: Option<f64>,
    stride: i64,
}

#[derive(Debug, Clone, Copy)]
pub struct Params {
    width: f64,
    depth: f64,
    res: i64,
    dropout: f64,
}

impl Params {
    pub fn of_tuple(width: f64, depth: f64, res: i64, dropout: f64) -> Params {
        Params {
            width,
            depth,
            res,
            dropout,
        }
    }
    pub fn b0() -> Params {
        Params::of_tuple(1.0, 1.0, 224, 0.2)
    }
    pub fn b1() -> Params {
        Params::of_tuple(1.0, 1.1, 240, 0.2)
    }
    pub fn b2() -> Params {
        Params::of_tuple(1.1, 1.2, 260, 0.3)
    }
    pub fn b3() -> Params {
        Params::of_tuple(1.2, 1.3, 300, 0.3)
    }
    pub fn b4() -> Params {
        Params::of_tuple(1.4, 1.8, 380, 0.4)
    }
    pub fn b5() -> Params {
        Params::of_tuple(1.6, 2.2, 456, 0.4)
    }
    pub fn b6() -> Params {
        Params::of_tuple(1.8, 2.6, 528, 0.5)
    }
    pub fn b7() -> Params {
        Params::of_tuple(2.0, 3.1, 600, 0.5)
    }
}

impl Tensor {
    fn swish(&self) -> Tensor {
        self * self.sigmoid()
    }
}

fn block(p: nn::Path, args: BlockArgs) -> impl ModuleT {
    let inp = args.input_filters;
    let oup = args.input_filters * args.expand_ratio;
    let final_oup = args.output_filters;
    let bn2d = nn::BatchNormConfig {
        momentum: 1.0 - BATCH_NORM_MOMENTUM,
        eps: BATCH_NORM_EPSILON,
        ..Default::default()
    };
    let conv_no_bias = nn::ConvConfig {
        bias: false,
        ..Default::default()
    };
    let depthwise_conv = nn::ConvConfig {
        stride: args.stride,
        groups: oup,
        bias: false,
        ..Default::default()
    };

    let expansion = if args.expand_ratio != 1 {
        nn::seq_t()
            .add(nn::conv2d(&p, inp, oup, 1, conv_no_bias))
            .add(nn::batch_norm2d(&p, oup, bn2d))
            .add_fn(|xs| xs.swish())
    } else {
        nn::seq_t()
    };
    let depthwise_conv = nn::conv2d(&p, oup, oup, args.kernel_size, depthwise_conv);
    let depthwise_bn = nn::batch_norm2d(&p, oup, bn2d);
    let se = args.se_ratio.map(|se_ratio| {
        let nsc = i64::max(1, (inp as f64 * se_ratio) as i64);
        nn::seq_t()
            .add(nn::conv2d(&p, oup, nsc, 1, Default::default()))
            .add_fn(|xs| xs.swish())
            .add(nn::conv2d(&p, nsc, oup, 1, Default::default()))
    });
    let project_conv = nn::conv2d(&p, oup, final_oup, 1, conv_no_bias);
    let project_bn = nn::batch_norm2d(&p, final_oup, bn2d);
    nn::func_t(move |xs, train| {
        let ys = xs
            .apply_t(&expansion, train)
            .apply(&depthwise_conv)
            .apply_t(&depthwise_bn, train)
            .swish();
        let ys = match &se {
            None => ys,
            Some(seq) => ys.adaptive_avg_pool2d(&[1, 1]).apply_t(seq, train) * ys,
        };
        let ys = ys.apply(&project_conv).apply_t(&project_bn, train);
        if args.id_skip && args.stride == 1 && inp == final_oup {
            // Maybe add a drop_connect layer here ?
            ys + xs
        } else {
            ys
        }
    })
}

pub fn efficientnet(p: nn::Path, args: Vec<BlockArgs>, nclasses: i64) -> impl ModuleT {
    let bn2d = nn::BatchNormConfig {
        momentum: 1.0 - BATCH_NORM_MOMENTUM,
        eps: BATCH_NORM_EPSILON,
        ..Default::default()
    };
    let conv_no_bias = nn::ConvConfig {
        bias: false,
        ..Default::default()
    };
    let conv_s2 = nn::ConvConfig {
        stride: 2,
        bias: false,
        ..Default::default()
    };
    let conv_stem = nn::conv2d(&p, 3, 32, 3, conv_s2);
    let bn0 = nn::batch_norm2d(&p, 32, bn2d);
    let mut blocks = nn::seq_t();
    for &arg in args.iter() {
        blocks = blocks.add(block(&p / "bl0", arg));
        let arg = BlockArgs {
            input_filters: arg.output_filters,
            stride: 1,
            ..arg
        };
        for i in 1..arg.num_repeat {
            blocks = blocks.add(block(&p / i, arg));
        }
    }
    let in_channels = args.last().unwrap().output_filters;
    let out_channels = 1280;
    let conv_head = nn::conv2d(&p, in_channels, out_channels, 1, conv_no_bias);
    let bn1 = nn::batch_norm2d(&p, out_channels, bn2d);
    let classifier = nn::seq_t()
        .add_fn_t(|xs, train| xs.dropout(0.2, train))
        .add(nn::linear(&p, out_channels, nclasses, Default::default()));
    nn::func_t(move |xs, train| {
        xs.apply(&conv_stem)
            .apply_t(&bn0, train)
            .swish()
            .apply_t(&blocks, train)
            .apply(&conv_head)
            .apply_t(&bn1, train)
            .swish()
            .adaptive_avg_pool2d(&[1, 1])
            .squeeze1(-1)
            .squeeze1(-1)
            .apply_t(&classifier, train)
    })
}
