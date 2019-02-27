/* Training various models on the CIFAR-10 dataset.

   The dataset can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html, files
   should be placed in the data/ directory.

   The resnet model reaches XX.X% accuracy.
*/

extern crate tch;
use tch::nn::{BatchNorm2D, Conv2D, Linear, ModuleT, Path};
use tch::{nn, Device, Tensor};

struct ConvBN {
    conv2d: Conv2D,
    batch_norm2d: BatchNorm2D,
}

impl ConvBN {
    fn new(vs: &Path, c_in: i64, c_out: i64) -> ConvBN {
        let conv2d_cfg = nn::Conv2DConfig {
            padding: 1,
            bias: false,
            ..Default::default()
        };
        ConvBN {
            conv2d: Conv2D::new(vs, c_in, c_out, 3, conv2d_cfg),
            batch_norm2d: BatchNorm2D::new(vs, c_out, Default::default()),
        }
    }
}

impl ModuleT for ConvBN {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply(&self.conv2d)
            .apply_t(&self.batch_norm2d, train)
            .relu()
    }
}

struct Layer {
    pre: ConvBN,
    block1: ConvBN,
    block2: ConvBN,
}

impl Layer {
    fn new(vs: &Path, c_in: i64, c_out: i64) -> Layer {
        Layer {
            pre: ConvBN::new(&vs.sub("pre"), c_in, c_out),
            block1: ConvBN::new(&vs.sub("b1"), c_out, c_out),
            block2: ConvBN::new(&vs.sub("b2"), c_out, c_out),
        }
    }
}

impl ModuleT for Layer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let pre = xs.apply_t(&self.pre, train).max_pool2d_default(2);
        let ys = pre
            .apply_t(&self.block1, train)
            .apply_t(&self.block2, train);
        pre + ys
    }
}

struct FastResnet {
    pre: ConvBN,
    layer1: Layer,
    inter: ConvBN,
    layer2: Layer,
    linear: Linear,
}

impl FastResnet {
    fn new(vs: &nn::Path) -> FastResnet {
        FastResnet {
            pre: ConvBN::new(&vs.sub("pre"), 3, 64),
            layer1: Layer::new(&vs.sub("layer1"), 64, 128),
            inter: ConvBN::new(&vs.sub("inter"), 128, 256),
            layer2: Layer::new(&vs.sub("layer2"), 256, 512),
            linear: Linear::new(&vs.sub("linear"), 512, 10),
        }
    }
}

impl nn::ModuleT for FastResnet {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let batch_size = xs.size()[0] as i64;
        xs.apply_t(&self.pre, train)
            .apply_t(&self.layer1, train)
            .apply_t(&self.inter, train)
            .max_pool2d_default(2)
            .apply_t(&self.layer2, train)
            .max_pool2d_default(4)
            .view(&[batch_size, 512])
            .apply(&self.linear)
            * 0.125
    }
}

pub fn main() {
    let m = tch::vision::cifar::load_dir(std::path::Path::new("data")).unwrap();
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let net = FastResnet::new(&vs.root());
    let opt = nn::Optimizer::adam(&vs, 1e-4, Default::default());
    for epoch in 1..6000 {
        let (bimages, blabels) =
            Tensor::random_batch2(&m.train_images, &m.train_labels, 64, vs.device());
        let loss = net
            .forward_t(&bimages, true)
            .cross_entropy_for_logits(&blabels);
        opt.backward_step(&loss);
        if epoch % 50 == 0 {
            let test_accuracy =
                net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 512);
            println!(
                "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
                epoch,
                f64::from(&loss),
                100. * test_accuracy,
            );
        }
    }
}
