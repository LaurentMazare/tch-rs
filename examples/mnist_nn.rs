/* Hidden layer model for the MNIST dataset.
   The 4 following dataset files can be downloaded from http://yann.lecun.com/exdb/mnist/
   These files should be extracted in the 'data' directory.
     train-images-idx3-ubyte.gz
     train-labels-idx1-ubyte.gz
     t10k-images-idx3-ubyte.gz
     t10k-labels-idx1-ubyte.gz

   This should rearch XX.X% accuracy.
*/

extern crate torchr;
use torchr::{nn, nn::Module, nn::Optimizer, nn::VarStore, Tensor};

static IMAGE_DIM: i64 = 784;
static HIDDEN_NODES: i64 = 128;
static LABELS: i64 = 10;

struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &mut VarStore) -> Net {
        let fc1 = nn::Linear::new(vs, IMAGE_DIM, HIDDEN_NODES);
        let fc2 = nn::Linear::new(vs, HIDDEN_NODES, LABELS);
        Net { fc1, fc2 }
    }
}

impl Module for Net {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1).relu().apply(&self.fc2)
    }
}

fn main() {
    let m = torchr::vision::Mnist::load_dir(std::path::Path::new("data")).unwrap();
    let mut vs = VarStore::new();
    let net = Net::new(&mut vs);
    let opt = Optimizer::adam(&vs, 1e-3, Default::default());
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
}
