# Python Extension Example

## Instructions
Run the following commands to download the latest tch-rs version and build  `python-entropy` extension:

```bash
git clone https://github.com/LaurentMazare/tch-rs.git
cd tch-rs
```

```bash
cd examples/python-entropy
pip install maturin
maturin develop
```

Run `main.py` to test the extension:
```bash
python main.py
```