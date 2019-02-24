# tch-rs
Experimental rust bindings for PyTorch.
The code generation part for the C api on top of libtorch comes from
[ocaml-torch](https://github.com/LaurentMazare/ocaml-torch).

## Instructions

- Get `libtorch` from the
[PyTorch website download section](https://pytorch.org/get-started/locally/) and extract
the content of the zip file.
- Add the following to your `.bashrc` or equivalent.
```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```
- You should now be able to run some examples, e.g. `cargo run --example basics`.

## Examples

The following code defines a simple model with one hidden layer.

```rust
struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &mut nn::VarStore) -> Net {
        let fc1 = nn::Linear::new(vs, IMAGE_DIM, HIDDEN_NODES);
        let fc2 = nn::Linear::new(vs, HIDDEN_NODES, LABELS);
        Net { fc1, fc2 }
    }
}

impl nn::Module for Net {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1).relu().apply(&self.fc2)
    }
}
```

This model can then be trained on the MNIST dataset with the following code.

```ocaml
fn main() {
    let m = tch::vision::mnist::load_dir(std::path::Path::new("data")).unwrap();
    let mut vs = nn::VarStore::new(Device::Cpu);
    let net = Net::new(&mut vs);
    let opt = nn::Optimizer::adam(&vs, 1e-3, Default::default());
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
```
More examples can be found in the `examples` directory. They can be run
using the following command:

```bash
LD_LIBRARY_PATH=/.../libtorch/lib LIBTORCH=/.../libtorch cargo run --example mnist_nn
```
