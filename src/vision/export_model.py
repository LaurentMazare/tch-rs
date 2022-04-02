# This script exports pre-trained model weights in numpy format.
# These weights can then be converted to the libtorch native format via:
# cargo run --example tensor-tools -- cp resnet18.npz resnet18.ot
import numpy as np
import torch
import torchvision

m = torchvision.models.resnet34(pretrained=True)
nps = {}
for k, v in m.state_dict().items(): nps[k] = v.numpy()
np.savez('resnet34.npz', **nps)
