import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
  def __init__(self, input_dims, output_dims):
    super(LinearClassifier, self).__init__()
    self.linear = nn.Linear(input_dims, output_dims)

  def forward(self, x):
    return F.log_softmax(self.linear(x), dim=1)