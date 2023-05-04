import python_entropy as pe
import torch

metric = pe.EntropyMetric(2)
print(f'Initial counter: {metric.get_counter()}')
metric.update(torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
print(f'Updated counter: {metric.get_counter()}')
result = metric.compute()
print(f'Entropy: {result}')