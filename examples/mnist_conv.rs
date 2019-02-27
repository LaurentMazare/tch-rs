/* CNN model for the MNIST dataset.

   This should rearch 99.1% accuracy.
*/

extern crate tch;
use tch::{nn, nn::ModuleT, Device, Tensor};

struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        let conv1 = nn::Conv2D::new(vs, 1, 32, 5, Default::default());
        let conv2 = nn::Conv2D::new(vs, 32, 64, 5, Default::default());
        let fc1 = nn::Linear::new(vs, 1024, 1024);
        let fc2 = nn::Linear::new(vs, 1024, 10);
        Net {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view(&[-1, 1, 28, 28])
            .apply(&self.conv1)
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .max_pool2d_default(2)
            .view(&[-1, 1024])
            .apply(&self.fc1)
            .relu()
            .dropout_(0.5, train)
            .apply(&self.fc2)
    }
}

fn main() {
    let m = tch::vision::mnist::load_dir(std::path::Path::new("data")).unwrap();
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let net = Net::new(&vs.root());
    let opt = nn::Optimizer::adam(&vs, 1e-4, Default::default());
    for epoch in 1..6000 {
        let (bimages, blabels) =
            Tensor::random_batch2(&m.train_images, &m.train_labels, 256, vs.device());
        let loss = net
            .forward_t(&bimages, true)
            .cross_entropy_for_logits(&blabels);
        opt.backward_step(&loss);
        if epoch % 50 == 0 {
            let test_accuracy =
                net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 1024);
            println!(
                "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
                epoch,
                f64::from(&loss),
                100. * test_accuracy,
            );
        }
    }
}
