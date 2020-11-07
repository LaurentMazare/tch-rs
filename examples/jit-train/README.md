To run this example, you will have to unzip the [MNIST data
files](http://yann.lecun.com/exdb/mnist/) in `data/`.

# Loading and Training a PyTorch Model in Rust

In this tutorial this is illustrated using a demo model that has not been pre-trained.
We start by defining and serializing the model using the Python api.
The resulting model file is later loaded from Rust and trained on the MNIST dataset.

## Converting a Python PyTorch Model to Torch Script

There are various ways to create the Torch Script as detailed
in the original [tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html).

Here we will use tracing. The following Python script runs the
defined model on some random image and creates the Torch Script
file based on this evaluation.

```python
import torch
from torch.nn import Module


class DemoModule(Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(1)
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(5, 5), padding=(2, 2))
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(16 * 28 * 28, 100)
        self.linear2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return self.linear2(x)


module = DemoModule()
example = torch.rand(1, 1, 28, 28)
traced_script_module = torch.jit.trace(module, example)
traced_script_module.save("model.pt")
```

The last line creates the `model.pt` Torch Script file which includes both the model
architecture and the weight values.

## Loading the Torch Script Model from Rust

The `model.pt` file can then be loaded and trained from Rust.

```rust
fn train_and_save_model(dataset: &Dataset, device: Device) -> Result<()> {
    let vs = VarStore::new(device);

    let trainable = TrainableCModule::load("model.pt", vs.root())?;
    let initial_acc = trainable.batch_accuracy_for_logits(
        &dataset.test_images,
        &dataset.test_labels,
        vs.device(),
        1024,
    );
    println!("Initial accuracy: {:5.2}%", 100. * initial_acc);

    let mut opt = Adam::default().build(&vs, 1e-4)?;
    for epoch in 1..20 {
        for (images, labels) in dataset
            .train_iter(128)
            .shuffle()
            .to_device(vs.device())
            .take(50)
        {
            let loss = trainable
                .forward_t(&images, true)
                .cross_entropy_for_logits(&labels);
            opt.backward_step(&loss);
        }
        let test_accuracy = trainable.batch_accuracy_for_logits(
            &dataset.test_images,
            &dataset.test_labels,
            vs.device(),
            1024,
        );
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }

    trainable.save("trained_model.pt")?;
    Ok(())
}
```

After the model was trained, the updated Torch Script is saved to file `trained_model.pt`. 
This can be used for executing the module later on either via Python or
using Rust as per the following.

```rust
fn load_trained_and_test_acc(dataset: &Dataset, device: Device) -> Result<()> {
    let module = CModule::load_on_device("trained_model.pt", device)?;
    let accuracy =
        module.batch_accuracy_for_logits(&dataset.test_images, &dataset.test_labels, device, 1024);
    println!("Updated accuracy: {:5.2}%", 100. * accuracy);
    Ok(())
}
```

Below is the typical output that this example would produce.

```
Initial accuracy: 13.39%
epoch:    1 test acc: 87.49%
epoch:    2 test acc: 89.72%
epoch:    3 test acc: 90.59%
epoch:    4 test acc: 90.95%
epoch:    5 test acc: 91.41%
epoch:    6 test acc: 91.40%
epoch:    7 test acc: 91.70%
epoch:    8 test acc: 91.62%
epoch:    9 test acc: 91.59%
epoch:   10 test acc: 91.76%
epoch:   11 test acc: 91.59%
epoch:   12 test acc: 91.39%
epoch:   13 test acc: 91.94%
epoch:   14 test acc: 91.74%
epoch:   15 test acc: 91.60%
epoch:   16 test acc: 91.78%
epoch:   17 test acc: 91.77%
epoch:   18 test acc: 91.72%
epoch:   19 test acc: 91.65%
Updated accuracy: 91.65%
```
