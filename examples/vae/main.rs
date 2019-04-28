/* Variational Auto-Encoder on MNIST.
   The implementation is based on:
     https://github.com/pytorch/examples/blob/master/vae/main.py

   The 4 following dataset files can be downloaded from http://yann.lecun.com/exdb/mnist/
   These files should be extracted in the 'data' directory.
     train-images-idx3-ubyte.gz
     train-labels-idx1-ubyte.gz
     t10k-images-idx3-ubyte.gz
     t10k-labels-idx1-ubyte.gz
*/

extern crate tch;
use tch::{nn, nn::Module, nn::OptimizerConfig, Reduction, Tensor};

struct VAE {
    fc1: nn::Linear,
    fc21: nn::Linear,
    fc22: nn::Linear,
    fc3: nn::Linear,
    fc4: nn::Linear,
}

impl VAE {
    fn new(vs: &nn::Path) -> Self {
        VAE {
            fc1: nn::linear(vs / "fc1", 784, 400, Default::default()),
            fc21: nn::linear(vs / "fc21", 400, 20, Default::default()),
            fc22: nn::linear(vs / "fc22", 400, 20, Default::default()),
            fc3: nn::linear(vs / "fc3", 20, 400, Default::default()),
            fc4: nn::linear(vs / "fc4", 400, 784, Default::default()),
        }
    }

    fn encode(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let h1 = xs.apply(&self.fc1).relu();
        (self.fc21.forward(&h1), self.fc22.forward(&h1))
    }

    fn decode(&self, zs: &Tensor) -> Tensor {
        zs.apply(&self.fc3).relu().apply(&self.fc4).sigmoid()
    }

    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor, Tensor) {
        let (mu, logvar) = self.encode(&xs.view(&[-1, 784]));
        let std = (&logvar * 0.5).exp();
        let eps = std.randn_like();
        (self.decode(&(&mu + eps * std)), mu, logvar)
    }
}

// Reconstruction + KL divergence losses summed over all elements and batch dimension.
fn loss(recon_x: &Tensor, x: &Tensor, mu: &Tensor, logvar: &Tensor) -> Tensor {
    let bce =
        recon_x.binary_cross_entropy(&x.view(&[-1, 784]), &Tensor::new(), Reduction::Sum.to_int());
    // See Appendix B from VAE paper:
    //     Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    // https://arxiv.org/abs/1312.6114
    // 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    let kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum();
    bce + kld
}

pub fn main() -> failure::Fallible<()> {
    let m = tch::vision::mnist::load_dir("data")?;
    let vs = nn::VarStore::new(tch::Device::cuda_if_available());
    let vae = VAE::new(&vs.root());
    let opt = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 1..21 {
        let mut train_loss = 0f64;
        let mut batches = 0f64;
        for (bimages, _) in m.train_iter(128).shuffle().to_device(vs.device()) {
            let (recon_batch, mu, logvar) = vae.forward(&bimages);
            let loss = loss(&recon_batch, &bimages, &mu, &logvar);
            opt.backward_step(&loss);
            train_loss += f64::from(&loss);
            batches += 1.0;
        }
        println!("Epoch: {}, loss: {}", epoch, train_loss / batches);
    }
    Ok(())
}
