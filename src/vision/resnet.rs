//! ResNet implementation.
//!
//! See "Deep Residual Learning for Image Recognition" He et al. 2015
//! https://arxiv.org/abs/1512.03385

use crate::nn;
use crate::nn::{BatchNorm2D, Conv2D, FuncT, Linear, ModuleT, SequentialT};

fn conv2d(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::Conv2DConfig {
        stride,
        padding,
        bias: false,
        ..Default::default()
    };
    Conv2D::new(&p, c_in, c_out, ksize, conv2d_cfg)
}

fn downsample(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
    if stride != 1 || c_in != c_out {
        SequentialT::new()
            .add(conv2d(&p / "0", c_in, c_out, 1, 0, stride))
            .add(BatchNorm2D::new(&p / "1", c_out, Default::default()))
    } else {
        SequentialT::new()
    }
}

fn basic_block(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
    let conv1 = conv2d(&p / "conv1", c_in, c_out, 3, 1, stride);
    let bn1 = BatchNorm2D::new(&p / "bn1", c_out, Default::default());
    let conv2 = conv2d(&p / "conv2", c_out, c_out, 3, 1, 1);
    let bn2 = BatchNorm2D::new(&p / "bn2", c_out, Default::default());
    let downsample = downsample(&p / "downsample", c_in, c_out, stride);
    FuncT::new(move |xs, train| {
        let ys = xs
            .apply(&conv1)
            .apply_t(&bn1, train)
            .relu()
            .apply(&conv2)
            .apply_t(&bn2, train);
        (xs.apply_t(&downsample, train) + ys).relu()
    })
}

fn make_layer(p: nn::Path, c_in: i64, c_out: i64, stride: i64, cnt: i64) -> impl ModuleT {
    let mut layer = SequentialT::new().add(basic_block(&p / "0", c_in, c_out, stride));
    for block_index in 1..cnt {
        layer = layer.add(basic_block(&p / &block_index.to_string(), c_out, c_out, 1))
    }
    layer
}

fn resnet(p: &nn::Path, nclasses: Option<i64>, c1: i64, c2: i64, c3: i64, c4: i64) -> impl ModuleT {
    let conv1 = conv2d(p / "conv1", 3, 64, 7, 3, 2);
    let bn1 = BatchNorm2D::new(p / "bn1", 64, Default::default());
    let layer1 = make_layer(p / "layer1", 64, 64, 1, c1);
    let layer2 = make_layer(p / "layer2", 64, 128, 2, c2);
    let layer3 = make_layer(p / "layer3", 128, 256, 2, c3);
    let layer4 = make_layer(p / "layer4", 256, 512, 2, c4);
    let fc = nclasses.map(|n| Linear::new(p / "fc", 512, n));
    FuncT::new(move |xs, train| {
        xs.apply(&conv1)
            .apply_t(&bn1, train)
            .relu()
            .max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[1, 1], false)
            .apply_t(&layer1, train)
            .apply_t(&layer2, train)
            .apply_t(&layer3, train)
            .apply_t(&layer4, train)
            .adaptive_avg_pool2d(&[1, 1])
            .flat_view()
            .apply_opt(&fc)
    })
}

/// Creates a ResNet-18 model.
///
/// Pre-trained weights can be downloaded at the following link:
/// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet18.ot
pub fn resnet18(p: &nn::Path, num_classes: i64) -> impl ModuleT {
    resnet(p, Some(num_classes), 2, 2, 2, 2)
}

pub fn resnet18_no_final_layer(p: &nn::Path) -> impl ModuleT {
    resnet(p, None, 2, 2, 2, 2)
}

/// Creates a ResNet-34 model.
///
/// Pre-trained weights can be downloaded at the following link:
/// https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/resnet34.ot
pub fn resnet34(p: &nn::Path, num_classes: i64) -> impl ModuleT {
    resnet(p, Some(num_classes), 3, 4, 6, 3)
}

pub fn resnet34_no_final_layer(p: &nn::Path) -> impl ModuleT {
    resnet(p, None, 3, 4, 6, 3)
}
