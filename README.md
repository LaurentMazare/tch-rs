# torch-rust
Some very experimental rust bindings for PyTorch.
The code generation part for the C api on top of libtorch comes from
[ocaml-torch](https://github.com/LaurentMazare/ocaml-torch).

## Instructions

- Get `libtorch` from the
[PyTorch website download section](https://pytorch.org/get-started/locally/) and extract
the content of the zip file.
- Run the following command:
```bash
LD_LIBRARY_PATH=/.../libtorch/lib LIBTORCH=/.../libtorch cargo run --example basics
```

## Examples

The following code trains a linear classifier on MNIST as a proof of concept.

```rust
    let m = vision::Mnist::load_dir(std::path::Path::new("data")).unwrap();
    let mut ws = Tensor::zeros(&[IMAGE_DIM, LABELS], Kind::Float).set_requires_grad(true);
    let mut bs = Tensor::zeros(&[LABELS], Kind::Float).set_requires_grad(true);
    for epoch in 1..200 {
        let logits = m.train_images.mm(&ws) + &bs;
        let loss = logits.log_softmax(-1).nll_loss(&m.train_labels);
        ws.zero_grad();
        bs.zero_grad();
        loss.backward();
        no_grad(|| {
            ws += ws.grad() * (-1);
            bs += bs.grad() * (-1);
        });
        let test_logits = m.test_images.mm(&ws) + &bs;
        let test_accuracy = test_logits
            .argmax(-1)
            .eq(&m.test_labels)
            .to_kind(Kind::Float)
            .mean()
            .double_value(&[]);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            loss.double_value(&[]),
            100. * test_accuracy
        );
    }
```

This can be run with this command.

```bash
LD_LIBRARY_PATH=/.../libtorch/lib LIBTORCH=/.../libtorch cargo run --example mnist
```
