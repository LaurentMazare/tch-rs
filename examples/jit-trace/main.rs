use anyhow::Result;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, CModule, Device, Tensor};

#[derive(Debug)]
struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        let conv1 = nn::conv2d(vs, 1, 32, 5, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 5, Default::default());
        let fc1 = nn::linear(vs, 1024, 1024, Default::default());
        let fc2 = nn::linear(vs, 1024, 10, Default::default());
        Net { conv1, conv2, fc1, fc2 }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, 28, 28])
            .apply(&self.conv1)
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .max_pool2d_default(2)
            .view([-1, 1024])
            .apply(&self.fc1)
            .relu()
            .dropout(0.5, train)
            .apply(&self.fc2)
    }
}

fn trace_model(net: &Net, vs: &nn::VarStore) -> std::result::Result<CModule, tch::TchError> {
    let mut closure = |input: &[Tensor]| vec![net.forward_t(&input[0], false)];
    CModule::create_by_tracing(
        "MyModule",
        "forward",
        &[Tensor::zeros([784], (tch::Kind::Float, vs.device()))],
        &mut closure,
    )
}

fn main() -> Result<()> {
    let m = tch::vision::mnist::load_dir("data")?;
    let mut vs = nn::VarStore::new(Device::cuda_if_available());
    let net = Net::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
    for epoch in 1..10 {
        for (bimages, blabels) in m.train_iter(256).shuffle().to_device(vs.device()) {
            let loss = net.forward_t(&bimages, true).cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
        }
        let test_accuracy =
            net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 1024);
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }

    vs.freeze();
    let model = trace_model(&net, &vs)?;
    model.save("model.pt")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_example_net_traces_saves_and_reloads() -> Result<()> {
        tch::set_num_threads(1);
        let mut vs = nn::VarStore::new(Device::Cpu);
        let net = Net::new(&vs.root());
        vs.freeze();
        let input = Tensor::arange(784, (tch::Kind::Float, Device::Cpu)) / 784.;
        let expected = net.forward_t(&input, false);
        let model = trace_model(&net, &vs)?;
        let output = model.forward_ts(&[&input])?;
        assert_eq!(output.size(), [1, 10]);
        assert!(output.allclose(&expected, 1e-5, 1e-6, false));
        let path =
            std::env::temp_dir().join(format!("tch-jit-trace-cpu-{}.pt", std::process::id()));
        assert!(!path.exists());
        model.save(&path)?;
        let restored = CModule::load(&path)?;
        let restored_output = restored.forward_ts(&[&input]);
        std::fs::remove_file(path)?;
        assert!(restored_output?.allclose(&expected, 1e-5, 1e-6, false));
        Ok(())
    }
}
