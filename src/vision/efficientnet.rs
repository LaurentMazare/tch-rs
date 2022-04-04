//! EfficientNet implementation.
use crate::nn::{self, ConvConfig, Module, ModuleT};
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
    se_ratio: Option<f64>,
    stride: i64,
}

#[allow(clippy::many_single_char_names)]
fn ba(k: i64, r: i64, i: i64, o: i64, er: i64, sr: f64, s: i64) -> BlockArgs {
    BlockArgs {
        kernel_size: k,
        num_repeat: r,
        input_filters: i,
        output_filters: o,
        expand_ratio: er,
        se_ratio: Some(sr),
        stride: s,
    }
}

fn block_args() -> Vec<BlockArgs> {
    vec![
        ba(3, 1, 32, 16, 1, 0.25, 1),
        ba(3, 2, 16, 24, 6, 0.25, 2),
        ba(5, 2, 24, 40, 6, 0.25, 2),
        ba(3, 3, 40, 80, 6, 0.25, 2),
        ba(5, 3, 80, 112, 6, 0.25, 1),
        ba(5, 4, 112, 192, 6, 0.25, 2),
        ba(3, 1, 192, 320, 6, 0.25, 1),
    ]
}

#[derive(Debug, Clone, Copy)]
struct Params {
    width: f64,
    depth: f64,
}

impl Params {
    fn round_repeats(&self, repeats: i64) -> i64 {
        (self.depth * repeats as f64).ceil() as i64
    }

    fn round_filters(&self, filters: i64) -> i64 {
        let divisor = 8;
        let filters = self.width * filters as f64;
        let filters_ = (filters + divisor as f64 / 2.) as i64;
        let new_filters = i64::max(divisor, filters_ / divisor * divisor);
        if (new_filters as f64) < 0.9 * filters {
            new_filters + divisor
        } else {
            new_filters
        }
    }
}

/// Conv2D with same padding.
#[allow(clippy::many_single_char_names)]
pub fn conv2d_same(
    vs: nn::Path,
    i: i64,
    o: i64,
    k: i64,
    c: ConvConfig,
) -> impl Module<Input = Tensor, Output = Tensor> {
    let conv2d = nn::conv2d(vs, i, o, k, c);
    let s = c.stride;
    nn::func(move |xs: &Tensor| {
        let size = xs.size();
        let ih = size[2];
        let iw = size[3];
        let oh = (ih + s - 1) / s;
        let ow = (iw + s - 1) / s;
        let pad_h = i64::max((oh - 1) * s + k - ih, 0);
        let pad_w = i64::max((ow - 1) * s + k - iw, 0);
        if pad_h > 0 || pad_w > 0 {
            xs.zero_pad2d(pad_w / 2, pad_w - pad_w / 2, pad_h / 2, pad_h - pad_h / 2).apply(&conv2d)
        } else {
            xs.apply(&conv2d)
        }
    })
}

impl Params {
    fn of_tuple(width: f64, depth: f64) -> Params {
        Params { width, depth }
    }
    fn b0() -> Params {
        Params::of_tuple(1.0, 1.0)
    }
    fn b1() -> Params {
        Params::of_tuple(1.0, 1.1)
    }
    fn b2() -> Params {
        Params::of_tuple(1.1, 1.2)
    }
    fn b3() -> Params {
        Params::of_tuple(1.2, 1.4)
    }
    fn b4() -> Params {
        Params::of_tuple(1.4, 1.8)
    }
    fn b5() -> Params {
        Params::of_tuple(1.6, 2.2)
    }
    fn b6() -> Params {
        Params::of_tuple(1.8, 2.6)
    }
    fn b7() -> Params {
        Params::of_tuple(2.0, 3.1)
    }
}

impl Tensor {
    fn swish(&self) -> Tensor {
        self * self.sigmoid()
    }
}

fn block(p: nn::Path, args: BlockArgs) -> impl ModuleT<Input = Tensor, Output = Tensor> {
    let inp = args.input_filters;
    let oup = args.input_filters * args.expand_ratio;
    let final_oup = args.output_filters;
    let bn2d = nn::BatchNormConfig {
        momentum: 1.0 - BATCH_NORM_MOMENTUM,
        eps: BATCH_NORM_EPSILON,
        ..Default::default()
    };
    let conv_no_bias = nn::ConvConfig { bias: false, ..Default::default() };
    let depthwise_conv =
        nn::ConvConfig { stride: args.stride, groups: oup, bias: false, ..Default::default() };

    let expansion = if args.expand_ratio != 1 {
        nn::seq_t()
            .add(conv2d_same(&p / "_expand_conv", inp, oup, 1, conv_no_bias))
            .add(nn::batch_norm2d(&p / "_bn0", oup, bn2d))
            .add_fn(|xs| xs.swish())
    } else {
        nn::seq_t()
    };
    let depthwise_conv =
        conv2d_same(&p / "_depthwise_conv", oup, oup, args.kernel_size, depthwise_conv);
    let depthwise_bn = nn::batch_norm2d(&p / "_bn1", oup, bn2d);
    let se = args.se_ratio.map(|se_ratio| {
        let nsc = i64::max(1, (inp as f64 * se_ratio) as i64);
        nn::seq_t()
            .add(conv2d_same(&p / "_se_reduce", oup, nsc, 1, Default::default()))
            .add_fn(|xs| xs.swish())
            .add(conv2d_same(&p / "_se_expand", nsc, oup, 1, Default::default()))
    });
    let project_conv = conv2d_same(&p / "_project_conv", oup, final_oup, 1, conv_no_bias);
    let project_bn = nn::batch_norm2d(&p / "_bn2", final_oup, bn2d);
    nn::func_t(
        move |xs: &Tensor, train| {
            let ys = if args.expand_ratio != 1 {
                xs.apply_t(&expansion, train)
            } else {
                xs.shallow_clone()
            };
            let ys = ys.apply(&depthwise_conv).apply_t(&depthwise_bn, train).swish();
            let ys = match &se {
                None => ys,
                Some(seq) => ys.adaptive_avg_pool2d(&[1, 1]).apply_t(seq, train).sigmoid() * ys,
            };
            let ys = ys.apply(&project_conv).apply_t(&project_bn, train);
            if args.stride == 1 && inp == final_oup {
                // Maybe add a drop_connect layer here ?
                ys + xs
            } else {
                ys
            }
        },
        |m, xs, ys, d, bs| nn::batch_accuracy_for_logits(m, xs, ys, d, bs),
    )
}

fn efficientnet(
    p: &nn::Path,
    params: Params,
    nclasses: i64,
) -> impl ModuleT<Input = Tensor, Output = Tensor> {
    let args = block_args();
    let bn2d = nn::BatchNormConfig {
        momentum: 1.0 - BATCH_NORM_MOMENTUM,
        eps: BATCH_NORM_EPSILON,
        ..Default::default()
    };
    let conv_no_bias = nn::ConvConfig { bias: false, ..Default::default() };
    let conv_s2 = nn::ConvConfig { stride: 2, bias: false, ..Default::default() };
    let out_c = params.round_filters(32);
    let conv_stem = conv2d_same(p / "_conv_stem", 3, out_c, 3, conv_s2);
    let bn0 = nn::batch_norm2d(p / "_bn0", out_c, bn2d);
    let mut blocks = nn::seq_t();
    let block_p = p / "_blocks";
    let mut block_idx = 0;
    for &arg in args.iter() {
        let arg = BlockArgs {
            input_filters: params.round_filters(arg.input_filters),
            output_filters: params.round_filters(arg.output_filters),
            ..arg
        };
        blocks = blocks.add(block(&block_p / block_idx, arg));
        block_idx += 1;
        let arg = BlockArgs { input_filters: arg.output_filters, stride: 1, ..arg };
        for _i in 1..params.round_repeats(arg.num_repeat) {
            blocks = blocks.add(block(&block_p / block_idx, arg));
            block_idx += 1;
        }
    }
    let in_channels = params.round_filters(args.last().unwrap().output_filters);
    let out_c = params.round_filters(1280);
    let conv_head = conv2d_same(p / "_conv_head", in_channels, out_c, 1, conv_no_bias);
    let bn1 = nn::batch_norm2d(p / "_bn1", out_c, bn2d);
    let classifier = nn::seq_t().add_fn_t(|xs, train| xs.dropout(0.2, train)).add(nn::linear(
        p / "_fc",
        out_c,
        nclasses,
        Default::default(),
    ));
    nn::func_t(
        move |xs: &Tensor, train| {
            xs.apply(&conv_stem)
                .apply_t(&bn0, train)
                .swish()
                .apply_t(&blocks, train)
                .apply(&conv_head)
                .apply_t(&bn1, train)
                .swish()
                .adaptive_avg_pool2d(&[1, 1])
                .squeeze_dim(-1)
                .squeeze_dim(-1)
                .apply_t(&classifier, train)
        },
        |m, xs, ys, d, bs| nn::batch_accuracy_for_logits(m, xs, ys, d, bs),
    )
}

pub fn b0(p: &nn::Path, nclasses: i64) -> impl ModuleT<Input = Tensor, Output = Tensor> {
    efficientnet(p, Params::b0(), nclasses)
}
pub fn b1(p: &nn::Path, nclasses: i64) -> impl ModuleT<Input = Tensor, Output = Tensor> {
    efficientnet(p, Params::b1(), nclasses)
}
pub fn b2(p: &nn::Path, nclasses: i64) -> impl ModuleT<Input = Tensor, Output = Tensor> {
    efficientnet(p, Params::b2(), nclasses)
}
pub fn b3(p: &nn::Path, nclasses: i64) -> impl ModuleT<Input = Tensor, Output = Tensor> {
    efficientnet(p, Params::b3(), nclasses)
}
pub fn b4(p: &nn::Path, nclasses: i64) -> impl ModuleT<Input = Tensor, Output = Tensor> {
    efficientnet(p, Params::b4(), nclasses)
}
pub fn b5(p: &nn::Path, nclasses: i64) -> impl ModuleT<Input = Tensor, Output = Tensor> {
    efficientnet(p, Params::b5(), nclasses)
}
pub fn b6(p: &nn::Path, nclasses: i64) -> impl ModuleT<Input = Tensor, Output = Tensor> {
    efficientnet(p, Params::b6(), nclasses)
}
pub fn b7(p: &nn::Path, nclasses: i64) -> impl ModuleT<Input = Tensor, Output = Tensor> {
    efficientnet(p, Params::b7(), nclasses)
}
