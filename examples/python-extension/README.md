## Python extensions using tch

This sample crate shows how to use `tch` to write a Python extension
that manipulates PyTorch tensors via [PyO3](https://github.com/PyO3/pyo3).

This is currently experimental hence requires some unsafe code until this has
been stabilized.

In order to build the extension and test the plugin, run the following in a
Python environment that has torch installed from the root of the github repo.

```bash
LIBTORCH_USE_PYTORCH=1 cargo build -p tch-ext && cp -f target/debug/libtch_ext.so tch_ext.so
python examples/python-extension/main.py
```

It is recommended to run the build with `LIBTORCH_USE_PYTORCH` set, this will
result in using the libtorch C++ library from the Python install in `tch` and
will ensure that this is at the proper version (having `tch` using a different
libtorch version from the one used by the Python runtime may result in segfaults).

## Colab Notebook

`tch` based plugins can easily be used from colab (though it might be a bit slow
to download all the crates and compile), see this [example
notebook](https://colab.research.google.com/drive/1bXVQ2TaKABI4bBG9IL0QFkmvhhf8Tsyl?usp=sharing).
