import torch
from rich import print






x = torch.tensor([1, 1, 1, 0, 0])
a = torch.outer(x, x)
print(a)
