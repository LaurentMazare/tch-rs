import os
import shutil

# Copy the shared library to the current directory and rename it
# so that it's easy to import.
SHARED_LIB = "../../target/debug/libtch_ext.so"
TMP_LIB = "./tch_ext.so"

if os.path.exists(TMP_LIB):
    os.remove(TMP_LIB)
shutil.copy(SHARED_LIB, TMP_LIB)

import torch
import tch_ext
print(tch_ext.__file__)

t = torch.tensor([[1., -1.], [1., -1.]])
print(t)
print(tch_ext.add_one(t))
