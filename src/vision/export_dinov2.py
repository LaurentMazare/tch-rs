import torch
from safetensors.torch import save_file

def normalize_key(k):
    if k.startswith("backbone."):
        k = k[9:]
    if k.startswith("linear_head."):
        k = k[7:]
    return k

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc', layers=1)
print(dinov2_vits14)
weights = dinov2_vits14.state_dict()
weights = {normalize_key(k): v for k, v in weights.items()}
save_file(weights, "dinov2_vits14.safetensors")
