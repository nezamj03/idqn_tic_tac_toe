import torch
from torch import nn

class DQN(nn.Module):
    """ Deep Q Network with one hidden layer """
    
    def __init__(self, in_dim, hidden_dim, out_dim):

        self.in_dimension = in_dim
        self.out_dimension = out_dim

        super().__init__()
        activ = nn.ReLU
        network = [
            nn.Linear(in_dim, hidden_dim),
            activ(),
            nn.Linear(hidden_dim, out_dim),
        ]
        self.network = nn.Sequential(*network) # sequential model

    def forward(self, input : torch.Tensor):
        assert input.size()[0] == self.in_dimension # ensure correct input size
        return self.network.forward(input)