/* CNN model for the MNIST dataset.

   This should rearch XX.X% accuracy.
*/

extern crate torchr;
use torchr::{nn, nn::ModuleT, Device, Tensor};

struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
    device: Device,
}

impl Net {
    fn new(vs: &mut nn::VarStore) -> Net {
        let conv1 = nn::Conv2D::new(vs, 1, 32, 5, Default::default());
        let conv2 = nn::Conv2D::new(vs, 32, 64, 5, Default::default());
        let fc1 = nn::Linear::new(vs, 1024, 1024);
        let fc2 = nn::Linear::new(vs, 1024, 10);
        let device = vs.device();
        Net {
            conv1,
            conv2,
            fc1,
            fc2,
            device,
        }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.to_device(self.device)
            .view(&[-1, 1, 28, 28])
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
    let m = torchr::vision::mnist::load_dir(std::path::Path::new("data")).unwrap();
    let mut vs = nn::VarStore::new(Device::cuda_if_available());
    let net = Net::new(&mut vs);
    let opt = nn::Optimizer::adam(&vs, 1e-3, Default::default());
    for epoch in 1..200 {
        let (bimages, blabels) = Tensor::random_batch2(&m.train_images, &m.train_labels, 256);
        let loss = net
            .forward_t(&bimages, true)
            .cross_entropy_for_logits(&blabels);
        opt.backward_step(&loss);
        let test_accuracy = net
            .forward_t(&m.test_images, false)
            .accuracy_for_logits(&m.test_labels);
        if epoch % 50 == 0 {
            println!(
                "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
                epoch,
                f64::from(&loss),
                100. * f64::from(&test_accuracy),
            );
        }
    }
}
