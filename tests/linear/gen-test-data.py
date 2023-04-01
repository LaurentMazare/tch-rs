import numpy as np
import torch

SIZE=512

linear = torch.nn.Linear(SIZE, SIZE)
input = torch.rand(SIZE, SIZE)

np.save('ws.npy', linear.weight.data)
np.save('bs.npy', linear.bias.data)
np.save('in.npy', input.data)
np.save('out.npy', linear(input).data)
