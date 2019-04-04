//! MobileNet V2 implementation.
//! https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html
use crate::nn::{self, BatchNorm2D, Conv2D, FuncT, Linear, ModuleT, SequentialT};

// Conv2D + BatchNorm2D + ReLU6
fn cbr(p: nn::Path, c_in: i64, c_out: i64, ks: i64, stride: i64, g: i64) -> impl ModuleT {
    let conv2d = nn::ConvConfig {
        stride,
        padding: (ks - 1) / 2,
        groups: g,
        ..Default::default()
    };
    SequentialT::new()
        .add(Conv2D::new(&p / "1", c_in, c_out, ks, conv2d))
        .add(BatchNorm2D::new(&p / "2", c_out, Default::default()))
        .add_fn(|xs| xs.relu().clamp_max(6.))
}

// Inverted Residual block.
fn inv(p: nn::Path, c_in: i64, c_out: i64, stride: i64, er: i64) -> impl ModuleT {
    let c_hidden = er * c_in;
    let mut conv = SequentialT::new();
    if er != 1 {
        conv = conv.add(cbr(&p / "0", c_in, c_hidden, 1, 1, 1));
    }
    conv = conv
        .add(cbr(&p / "1", c_hidden, c_hidden, 3, stride, c_hidden))
        .add(Conv2D::new(&p / "2", c_hidden, c_out, 1, nn::no_bias()))
        .add(BatchNorm2D::new(&p / "3", c_out, Default::default()));
    FuncT::new(move |xs, train| {
        let ys = xs.apply_t(&conv, train);
        if stride == 1 && c_in == c_out {
            xs + ys
        } else {
            ys
        }
    })
}

const INVERTED_RESIDUAL_SETTINGS: [(i64, i64, i64, i64); 7] = [
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
];

pub fn v2(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    let f_p = p / "features";
    let c_p = p / "classifier";
    let mut c_in = 32;
    let mut features = SequentialT::new().add(cbr(&f_p / "0", 3, c_in, 3, 2, 1));
    for &(er, c_out, n, stride) in INVERTED_RESIDUAL_SETTINGS.iter() {
        for i in 0..n {
            let stride = if i == 0 { stride } else { 1 };
            features = features.add(inv(&f_p / "1", c_in, c_out, stride, er));
            c_in = c_out;
        }
    }
    features = features.add(cbr(&f_p / "x", c_in, 1280, 1, 1, 1));
    let classifier = SequentialT::new()
        .add_fn_t(|xs, train| xs.dropout(0.2, train))
        .add(Linear::new(&c_p, 1280, nclasses, Default::default()));
    FuncT::new(move |xs, train| {
        xs.apply_t(&features, train)
            .mean2(2, false)
            .mean2(2, false)
            .apply_t(&classifier, train)
    })
}
