//! DenseNet implementation.
//!
//! See "Densely Connected Convolutional Networks", Huang et al 2016.
//! https://arxiv.org/abs/1608.06993
use crate::nn::{BatchNorm2D, Conv2D, FuncT, Linear, ModuleT, SequentialT};
use crate::{nn, Tensor};

fn conv2d(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::Conv2DConfig {
        stride,
        padding,
        bias: false,
        ..Default::default()
    };
    Conv2D::new(&p, c_in, c_out, ksize, conv2d_cfg)
}

fn dense_layer(p: nn::Path, c_in: i64, bn_size: i64, growth: i64) -> impl ModuleT {
    let c_inter = bn_size * growth;
    let bn1 = BatchNorm2D::new(&p / "norm1", c_in, Default::default());
    let conv1 = conv2d(&p / "conv1", c_in, c_inter, 1, 0, 1);
    let bn2 = BatchNorm2D::new(&p / "norm2", c_inter, Default::default());
    let conv2 = conv2d(&p / "conv2", c_inter, growth, 3, 1, 1);
    FuncT::new(move |xs, train| {
        let ys = xs
            .apply_t(&bn1, train)
            .relu()
            .apply(&conv1)
            .apply_t(&bn2, train)
            .relu()
            .apply(&conv2);
        Tensor::cat(&[xs, &ys], 1)
    })
}

fn dense_block(p: nn::Path, c_in: i64, bn_size: i64, growth: i64, nlayers: i64) -> impl ModuleT {
    let mut seq = SequentialT::new();
    for i in 0..nlayers {
        seq = seq.add(dense_layer(
            &p / &format!("denselayer{}", 1 + i),
            c_in + i * growth,
            bn_size,
            growth,
        ));
    }
    seq
}

fn transition(p: nn::Path, c_in: i64, c_out: i64) -> impl ModuleT {
    SequentialT::new()
        .add(BatchNorm2D::new(&p / "norm", c_in, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(conv2d(&p / "conv1", c_in, c_out, 1, 0, 1))
        .add_fn(|xs| xs.avg_pool2d_default(2))
}

fn densenet(
    p: &nn::Path,
    init_dim: i64,
    bn_size: i64,
    growth: i64,
    block_config: &[i64],
    nclasses: i64,
) -> impl ModuleT {
    let features_p = p / "features";
    let mut seq = SequentialT::new()
        .add(conv2d(&features_p / "conv0", 3, init_dim, 7, 3, 2))
        .add(BatchNorm2D::new(
            &features_p / "norm0",
            init_dim,
            Default::default(),
        ))
        .add_fn(|xs| {
            xs.relu()
                .max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[1, 1], false)
        });
    let mut nfeatures = init_dim;
    for (i, &nlayers) in block_config.iter().enumerate() {
        seq = seq.add(dense_block(
            &features_p / &format!("denseblock{}", 1 + i),
            nfeatures,
            bn_size,
            growth,
            nlayers,
        ));
        nfeatures += nlayers * growth;
        if i + 1 != block_config.len() {
            seq = seq.add(transition(
                &features_p / &format!("transition{}", 1 + i),
                nfeatures,
                nfeatures / 2,
            ));
            nfeatures /= 2
        }
    }
    seq.add(BatchNorm2D::new(
        &features_p / "norm5",
        init_dim,
        Default::default(),
    ))
    .add_fn(|xs| {
        xs.relu()
            .avg_pool2d(&[7, 7], &[1, 1], &[0, 0], false, true)
            .flat_view()
    })
    .add(Linear::new(p / "classifier", nfeatures, nclasses))
}

pub fn densenet121(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    densenet(p, 64, 4, 32, &[6, 12, 24, 16], nclasses)
}

pub fn densenet161(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    densenet(p, 96, 4, 48, &[6, 12, 36, 24], nclasses)
}

pub fn densenet169(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    densenet(p, 64, 4, 32, &[6, 12, 32, 32], nclasses)
}

pub fn densenet201(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    densenet(p, 64, 4, 32, &[6, 12, 48, 32], nclasses)
}
