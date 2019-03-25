# Reinforcement Learning Examples

These examples illustrate how to implement a couple reinforcement learning
algorithms to play Atari games.
This uses the [https://github.com/openai/gym](OpenAI Gym) through its Python
api so the gym Python package has to be installed.
The Python library is executed in Rust using
[https://github.com/dgrunwald/rust-cpython](rust-cpython).

The policy gradient example uses the `CartPole-v0` environment. It can be
run using the following command.

```bash
cargo run --example reinforcement-learning  --features=python pg
```

The A2C example can use any Atari game supported in the OpenAI gym.
Training can be launched by running:

```bash
cargo run --example reinforcement-learning  --features=python a2c
```

This produces some weight files in the current directory.

If you get some Python import error when using conda, this can
be related to `rust-cpython` not being able to use the Python
runtime as a shared library. `rust-cpython` uses
`Py_ENABLE_SHARED` to determine the link mode which does not
seem to work properly with conda as reported in this
[https://github.com/conda/conda-build/issues/2738](issue).
