/// ResNet implementation.
///
/// See "Deep Residual Learning for Image Recognition" He et al. 2015
/// https://arxiv.org/abs/1512.03385
use crate::nn;
use crate::nn::{BatchNorm2D, Conv2D, FuncT, Linear, ModuleT, SequentialT};

fn conv2d(vs: nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::Conv2DConfig {
        stride,
        padding,
        bias: false,
        ..Default::default()
    };
    Conv2D::new(&vs, c_in, c_out, ksize, conv2d_cfg)
}

fn downsample(vs: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
    if stride != 1 || c_in != c_out {
        SequentialT::new()
            .add(conv2d(&vs / "0", c_in, c_out, 1, 0, stride))
            .add(BatchNorm2D::new(&vs / "1", c_out, Default::default()))
    } else {
        SequentialT::new()
    }
}

fn basic_block(vs: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
    let conv1 = conv2d(&vs / "conv1", c_in, c_out, 3, 1, stride);
    let bn1 = BatchNorm2D::new(&vs / "bn1", c_out, Default::default());
    let conv2 = conv2d(&vs / "conv2", c_out, c_out, 3, 1, 1);
    let bn2 = BatchNorm2D::new(&vs / "bn2", c_out, Default::default());
    let downsample = downsample(&vs / "downsample", c_in, c_out, stride);
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

fn make_layer(vs: nn::Path, c_in: i64, c_out: i64, stride: i64, cnt: i64) -> impl ModuleT {
    let mut layer = SequentialT::new().add(basic_block(&vs / "0", c_in, c_out, stride));
    for block_index in 1..cnt {
        layer = layer.add(basic_block(&vs / &block_index.to_string(), c_out, c_out, 1))
    }
    layer
}

fn resnet(vs: nn::Path, num_classes: i64, c1: i64, c2: i64, c3: i64, c4: i64) -> impl ModuleT {
    let conv1 = conv2d(&vs / "conv1", 3, 64, 7, 3, 2);
    let bn1 = BatchNorm2D::new(&vs / "bn1", 64, Default::default());
    let layer1 = make_layer(&vs / "layer1", 64, 64, 1, c1);
    let layer2 = make_layer(&vs / "layer2", 64, 128, 2, c2);
    let layer3 = make_layer(&vs / "layer3", 128, 256, 2, c3);
    let layer4 = make_layer(&vs / "layer4", 256, 512, 2, c4);
    let fc = Linear::new(&vs / "fc", 512, num_classes);
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
            .apply(&fc)
    })
}

pub fn resnet18(vs: nn::Path, num_classes: i64) -> impl ModuleT {
    resnet(vs, num_classes, 2, 2, 2, 2)
}

pub fn resnet34(vs: nn::Path, num_classes: i64) -> impl ModuleT {
    resnet(vs, num_classes, 3, 4, 6, 3)
}
