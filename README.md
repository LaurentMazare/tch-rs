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
