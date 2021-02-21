import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self,):
        super(SimpleMLP, self).__init__()

        self.linear = nn.Sequential(nn.Linear(5, 20), nn.ReLU(), nn.Linear(20, 1))

    def forward(self, inputs):

        return self.linear(inputs)
