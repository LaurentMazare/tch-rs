import tch_ext
import torch
print(tch_ext.__file__)

t = torch.tensor([[1., -1.], [1., -1.]])
print(t)
print(tch_ext.add_one(t))
