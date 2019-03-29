//! Inception V3.
use crate::nn;
use crate::nn::{BatchNorm2D, Conv2D, FuncT, Linear, ModuleT, SequentialT};
use crate::Tensor;

fn conv_bn(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, pad: i64, stride: i64) -> impl ModuleT {
    let conv2d_cfg = nn::ConvConfig {
        stride,
        padding: pad,
        bias: false,
        ..Default::default()
    };
    let bn_cfg = nn::BatchNorm2DConfig {
        eps: 0.001,
        ..Default::default()
    };
    SequentialT::new()
        .add(Conv2D::new(&p / "conv", c_in, c_out, ksize, conv2d_cfg))
        .add(BatchNorm2D::new(&p / "bn", c_out, bn_cfg))
        .add_fn(|xs| xs.relu())
}

fn max_pool2d(xs: Tensor, ksize: i64, stride: i64) -> Tensor {
    xs.max_pool2d(&[ksize, ksize], &[stride, stride], &[0, 0], &[1, 1], false)
}

fn inception_a(p: nn::Path, c_in: i64, c_pool: i64) -> impl ModuleT {
    let b1 = conv_bn(&p / "branch1x1", c_in, 64, 1, 0, 1);
    let b2_1 = conv_bn(&p / "branch5x5_1", c_in, 48, 1, 0, 1);
    let b2_2 = conv_bn(&p / "branch5x5_2", 48, 64, 5, 2, 1);
    let b3_1 = conv_bn(&p / "branch3x3dbl_1", c_in, 64, 1, 0, 1);
    let b3_2 = conv_bn(&p / "branch3x3dbl_2", 64, 96, 3, 1, 1);
    let b3_3 = conv_bn(&p / "branch3x3dbl_3", 96, 96, 3, 1, 1);
    let bpool = conv_bn(&p / "branch_pool", c_in, c_pool, 1, 0, 1);
    FuncT::new(move |xs, tr| {
        let b1 = xs.apply_t(&b1, tr);
        let b2 = xs.apply_t(&b2_1, tr).apply_t(&b2_2, tr);
        let b3 = xs.apply_t(&b3_1, tr).apply_t(&b3_2, tr).apply_t(&b3_3, tr);
        let bpool = xs
            .avg_pool2d(&[3, 3], &[1, 1], &[1, 1], false, true)
            .apply_t(&bpool, tr);
        Tensor::cat(&[b1, b2, b3, bpool], 1)
    })
}

fn inception_b(p: nn::Path, c_in: i64) -> impl ModuleT {
    let b1 = conv_bn(&p / "branch3x3", c_in, 384, 3, 0, 2);
    let b2_1 = conv_bn(&p / "branch3x3dbl_1", c_in, 64, 1, 0, 1);
    let b2_2 = conv_bn(&p / "branch3x3dbl_2", 64, 96, 3, 1, 1);
    let b2_3 = conv_bn(&p / "branch3x3dbl_3", 96, 96, 3, 0, 2);
    FuncT::new(move |xs, tr| {
        let b1 = xs.apply_t(&b1, tr);
        let b2 = xs.apply_t(&b2_1, tr).apply_t(&b2_2, tr).apply_t(&b2_3, tr);
        let bpool = xs.avg_pool2d(&[3, 3], &[2, 2], &[0, 0], false, true);
        Tensor::cat(&[b1, b2, bpool], 1)
    })
}

pub fn v3(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    SequentialT::new()
        .add(conv_bn(p / "Conv2d_1a_3x3", 3, 32, 3, 0, 2))
        .add(conv_bn(p / "Conv2d_2a_3x3", 32, 32, 3, 0, 1))
        .add(conv_bn(p / "Conv2d_2b_3x3", 32, 64, 3, 1, 1))
        .add_fn(|xs| max_pool2d(xs.relu(), 3, 2))
        .add(conv_bn(p / "Conv2d_3b_1x1", 64, 80, 1, 0, 1))
        .add(conv_bn(p / "Conv2d_4a_3x3", 80, 192, 3, 0, 1))
        .add_fn(|xs| max_pool2d(xs.relu(), 3, 2))
        .add(inception_a(p / "Mixed_5b", 192, 32))
        .add(inception_a(p / "Mixed_5c", 256, 64))
        .add(inception_a(p / "Mixed_5d", 288, 64))
        .add(inception_b(p / "Mixed_6a", 288))
        .add_fn_t(|xs, train| {
            xs.adaptive_avg_pool2d(&[1, 1])
                .dropout(0.5, train)
                .flat_view()
        })
        .add(Linear::new(p / "fc", 2048, nclasses, Default::default()))
}
