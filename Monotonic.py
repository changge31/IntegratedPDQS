from parameters import parameters
import torch
import torch.nn as nn


class Monotonic(nn.Module):
    def __init__(self):
        super().__init__()

        self.hyper_w_1 = nn.Sequential(nn.Linear(parameters.n, parameters.embed),
                                       nn.ReLU(),
                                       nn.Linear(parameters.embed, 1))

        self.hyper_w_final = nn.Sequential(nn.Linear(parameters.n, parameters.embed),
                                           nn.ReLU(),
                                           nn.Linear(parameters.embed, 1))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(parameters.n, 1)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(parameters.n, parameters.embed),
                               nn.ReLU(),
                               nn.Linear(parameters.embed, 1))

    def forward(self, theta):

        # First layer
        w1 = -torch.abs(self.hyper_w_1(theta))

        b1 = self.hyper_b_1(theta)

        hidden = w1 * theta + b1

        # Second layer
        w_final = torch.abs(self.hyper_w_final(theta))

        # State-dependent bias
        v = self.V(theta)

        # Compute final output
        y = hidden * w_final + v

        # Reshape and return
        return y.sigmoid() * 0.9
