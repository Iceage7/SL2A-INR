import torch
import torch.nn as nn

from ChebyKANLayer import ChebyKANLayer

class ChebyLayer(nn.Module):
    def __init__(self, in_features, out_features, deg=512):
        super(ChebyLayer, self).__init__()
        self.cheby = ChebyKANLayer(in_features, out_features, deg)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        x = self.cheby(x)
        x = self.norm(x)
        return x

class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ReLULayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return nn.functional.relu(self.linear(input))

class ChebyINR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, deg=512, outermost_linear=True):
        super(ChebyINR, self).__init__()
        self.nonlin = ChebyLayer
        self.net = nn.ModuleList()
        self.net.append(self.nonlin(in_features, hidden_features, deg=deg))

        self.nonlin = ReLULayer
        self.linear = nn.Linear(hidden_features, out_features)

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features))

        if outermost_linear:
            # self.net.append(self.nonlin(hidden_features, out_features))
            self.net.append(self.linear)
        else:
            raise NotImplementedError("")

    def forward(self, coords):
        for layer in self.net:
            coords = layer(coords)
        return coords
