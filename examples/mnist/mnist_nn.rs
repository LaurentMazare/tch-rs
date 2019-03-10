// This should rearch 97% accuracy.

use tch::{nn, nn::Module, Device, Tensor};

static IMAGE_DIM: i64 = 784;
static HIDDEN_NODES: i64 = 128;
static LABELS: i64 = 10;

#[derive(Debug)]
struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        let fc1 = nn::Linear::new(vs, IMAGE_DIM, HIDDEN_NODES, Default::default());
        let fc2 = nn::Linear::new(vs, HIDDEN_NODES, LABELS, Default::default());
        Net { fc1, fc2 }
    }
}

impl nn::Module for Net {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1).relu().apply(&self.fc2)
    }
}

pub fn run() -> failure::Fallible<()> {
    let m = tch::vision::mnist::load_dir("data")?;
    let vs = nn::VarStore::new(Device::Cpu);
    let net = Net::new(&vs.root());
    let opt = nn::Optimizer::adam(&vs, 1e-3, Default::default())?;
    for epoch in 1..200 {
        let loss = net
            .forward(&m.train_images)
            .cross_entropy_for_logits(&m.train_labels);
        opt.backward_step(&loss);
        let test_accuracy = net
            .forward(&m.test_images)
            .accuracy_for_logits(&m.test_labels);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::from(&loss),
            100. * f64::from(&test_accuracy),
        );
    }
    Ok(())
}
