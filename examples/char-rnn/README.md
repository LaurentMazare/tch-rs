This example implements a character-level language model heavily inspired 
by [char-rnn](https://github.com/karpathy/char-rnn).
- For training, the model takes as input a text file and learns to predict the
  next character following a given sequence.
- For generation, the model uses a fixed seed, returns the next character
  distribution from which a character is randomly sampled and this is iterated
  to generate a text.

At the end of each training epoch, some sample text is generated and printed.

Any text file can be used as an input, as long as it's large enough for training.
A typical example would be the
[tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
The training text file should be stored in `data/input.txt`.

Compiling and running the example is done via the following command lines:
```bash
cargo run --example char-rnn
```

Here is an example of generated data when training on the Shakespeare dataset after a couple epochs. 
```
...TODO...
```

