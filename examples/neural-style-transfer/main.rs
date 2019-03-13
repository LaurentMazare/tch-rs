// This is inspired by the Neural Style tutorial from PyTorch.org
//   https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
#[macro_use]
extern crate failure;
extern crate tch;
use tch::vision::{imagenet, vgg};
use tch::{nn, Device, Tensor};

static STYLE_WEIGHT: f64 = 1e-6;
static LEARNING_RATE: f64 = 1e-1;
static TOTAL_STEPS: i64 = 3000;
static STYLE_INDEXES: [usize; 5] = [0, 2, 5, 7, 10];
static CONTENT_INDEXES: [usize; 1] = [7];

fn gram_matrix(m: &Tensor) -> Tensor {
    let (a, b, c, d) = m.size4().unwrap();
    let m = m.view(&[a * b, c * d]);
    let g = m.matmul(&m.tr());
    g / (a * b * c * d)
}

fn style_loss(m1: &Tensor, m2: &Tensor) -> Tensor {
    gram_matrix(m1).mse_loss(&gram_matrix(m2), 1)
}

pub fn main() -> failure::Fallible<()> {
    let device = Device::cuda_if_available();
    let args: Vec<_> = std::env::args().collect();
    let (style_img, content_img, weights) = match args.as_slice() {
        [_, s, c, w] => (s.to_owned(), c.to_owned(), w.to_owned()),
        _ => bail!("usage: main style.jpg content.jpg vgg16.ot"),
    };

    let mut net_vs = tch::nn::VarStore::new(device);
    let net = vgg::vgg16(&net_vs.root(), imagenet::CLASS_COUNT);
    net_vs.load(weights)?;
    net_vs.freeze();

    let style_img = imagenet::load_image(style_img)?
        .unsqueeze(0)
        .to_device(device);
    let content_img = imagenet::load_image(content_img)?
        .unsqueeze(0)
        .to_device(device);
    let max_layer = STYLE_INDEXES.iter().max().unwrap() + 1;
    let style_layers = net.forward_all_t(&style_img, false, Some(max_layer));
    let content_layers = net.forward_all_t(&content_img, false, Some(max_layer));

    let vs = nn::VarStore::new(device);
    let input_var = vs.root().var_copy("img", &content_img);
    let opt = nn::Optimizer::adam(&vs, LEARNING_RATE, Default::default())?;

    for step_idx in 1..(1 + TOTAL_STEPS) {
        let input_layers = net.forward_all_t(&input_var, false, Some(max_layer));
        let mut sloss = Tensor::float_vec(&[0.]).to_device(device);
        for &i in STYLE_INDEXES.iter() {
            sloss = sloss + style_loss(&input_layers[i], &style_layers[i]);
        }
        let mut closs = Tensor::float_vec(&[0.]).to_device(device);
        for &i in CONTENT_INDEXES.iter() {
            closs = closs + input_layers[i].mse_loss(&content_layers[i], 1);
        }
        let loss = sloss * STYLE_WEIGHT + closs;
        opt.backward_step(&loss);
        if step_idx % 1000 == 0 {
            println!("{} {}", step_idx, f64::from(loss));
            imagenet::save_image(
                &input_var.to_device(Device::Cpu).squeeze1(0),
                &format!("out{}.jpg", step_idx),
            )?;
        }
    }

    Ok(())
}
