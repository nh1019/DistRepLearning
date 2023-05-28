import torch.nn as nn

class LinearClassifier(nn.Module):
  def __init__(self, input_dims, output_dims):
    super(LinearClassifier, self).__init__()
    self.linear = nn.Linear(input_dims, output_dims)

  def forward(self, x):
    return self.linear(x)