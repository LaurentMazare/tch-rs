//! VGG models.
use crate::nn;
use crate::nn::{BatchNorm2D, Conv2D, Linear, ModuleT, SequentialT};

// Each list element contains multiple convolutions with some specified number
// of features followed by a single max-pool layer.
fn layers_a() -> Vec<Vec<i64>> {
    vec![
        vec![64],
        vec![128],
        vec![256, 256],
        vec![512, 512],
        vec![512, 512],
    ]
}

fn layers_b() -> Vec<Vec<i64>> {
    vec![
        vec![64, 64],
        vec![128, 128],
        vec![256, 256],
        vec![512, 512],
        vec![512, 512],
    ]
}
fn layers_d() -> Vec<Vec<i64>> {
    vec![
        vec![64, 64],
        vec![128, 128],
        vec![256, 256, 256],
        vec![512, 512, 512],
        vec![512, 512, 512],
    ]
}
fn layers_e() -> Vec<Vec<i64>> {
    vec![
        vec![64, 64],
        vec![128, 128],
        vec![256, 256, 256, 256],
        vec![512, 512, 512, 512],
        vec![512, 512, 512, 512],
    ]
}

fn conv2d(p: nn::Path, c_in: i64, c_out: i64) -> Conv2D {
    let conv2d_cfg = nn::Conv2DConfig {
        stride: 1,
        padding: 1,
        ..Default::default()
    };
    Conv2D::new(&p, c_in, c_out, 3, conv2d_cfg)
}

fn make(p: nn::Path, layers: Vec<Vec<i64>>, batch_norm: bool) -> impl ModuleT {
    let mut seq = SequentialT::new();
    let mut c_in = 3;
    for channels in layers.into_iter() {
        for &c_out in channels.iter() {
            let l = seq.len();
            seq = seq.add(conv2d(&p / &l.to_string(), c_in, c_out));
            if batch_norm {
                let l = seq.len();
                seq = seq.add(BatchNorm2D::new(
                    &p / &l.to_string(),
                    c_out,
                    Default::default(),
                ));
            };
            seq = seq.add_fn(|xs| xs.relu());
            c_in = c_out;
        }
        seq = seq.add_fn(|xs| xs.max_pool2d_default(2));
    }
    seq
}

fn vgg(p: nn::Path, cfg: Vec<Vec<i64>>, nclasses: i64, batch_norm: bool) -> impl ModuleT {
    let c = &p / "classifier";
    SequentialT::new()
        .add(make(&p / "features", cfg, batch_norm))
        .add_fn(|xs| xs.flat_view())
        .add(Linear::new(&c / "0", 512 * 7 * 7, 4096, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn_t(|xs, train| xs.dropout(0.5, train))
        .add(Linear::new(&c / "3", 4096, 4096, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn_t(|xs, train| xs.dropout(0.5, train))
        .add(Linear::new(&c / "6", 4096, nclasses, Default::default()))
}

pub fn vgg11(p: nn::Path, nclasses: i64) -> impl ModuleT {
    vgg(p, layers_a(), nclasses, false)
}

pub fn vgg11_bn(p: nn::Path, nclasses: i64) -> impl ModuleT {
    vgg(p, layers_a(), nclasses, true)
}

pub fn vgg13(p: nn::Path, nclasses: i64) -> impl ModuleT {
    vgg(p, layers_b(), nclasses, false)
}

pub fn vgg13_bn(p: nn::Path, nclasses: i64) -> impl ModuleT {
    vgg(p, layers_b(), nclasses, true)
}

pub fn vgg16(p: nn::Path, nclasses: i64) -> impl ModuleT {
    vgg(p, layers_d(), nclasses, false)
}

pub fn vgg16_bn(p: nn::Path, nclasses: i64) -> impl ModuleT {
    vgg(p, layers_d(), nclasses, true)
}

pub fn vgg19(p: nn::Path, nclasses: i64) -> impl ModuleT {
    vgg(p, layers_e(), nclasses, false)
}

pub fn vgg19_bn(p: nn::Path, nclasses: i64) -> impl ModuleT {
    vgg(p, layers_e(), nclasses, true)
}
