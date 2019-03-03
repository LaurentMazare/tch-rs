# tch-rs
Rust bindings for PyTorch. The goal of the `tch` crate is to provide some thin wrappers
around the C++ PyTorch api (a.k.a. libtorch). It aims at staying as close as
possible to the original C++ api. More idiomatic rust bindings could then be
developed on top of this. The [documentation](https://docs.rs/tch/) can be found on docs.rs.

[![Build Status](https://travis-ci.org/LaurentMazare/tch-rs.svg?branch=master)](https://travis-ci.org/LaurentMazare/tch-rs)
[![Crate](http://meritbadge.herokuapp.com/tch)](https://crates.io/crates/tch)


The code generation part for the C api on top of libtorch comes from
[ocaml-torch](https://github.com/LaurentMazare/ocaml-torch).

## Getting Started

This crate requires the C++ version of PyTorch (libtorch) to be available on
your system. You can either install it manually and let the build script now about
it via the `LIBTORCH` environment variable. If not set, the build script will
try downloading and extracting a pre-built binary version of libtorch.

### Libtorch Manual Install

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

This model can be trained on the MNIST dataset by running the following command.

```bash
cargo run --example mnist
```

More details on the training loop can be found in the
[detailed tutorial](https://github.com/LaurentMazare/tch-rs/tree/master/examples/mnist).

Further examples include:
* A simplified version of
  [char-rnn](https://github.com/LaurentMazare/tch-rs/blob/master/examples/char-rnn)
  illustrating character level language modeling using Recurrent Neural Networks.
* Some [ResNet examples on CIFAR-10](https://github.com/LaurentMazare/tch-rs/tree/master/examples/cifar).
