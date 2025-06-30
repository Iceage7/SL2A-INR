
import torch
from torch import nn
import numpy as np


## FINER sin⁡(𝑜𝑚𝑒𝑔𝑎∗𝑠𝑐𝑎𝑙𝑒∗(𝑊𝑥+𝑏𝑖𝑎𝑠)) 𝑠𝑐𝑎𝑙𝑒=|𝑊𝑥+𝑏𝑖𝑎𝑠|+1
class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale != None:
            self.init_first_bias()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
                # print('init fbs', self.first_bias_scale)

    def generate_scale(self, x):
        if self.scale_req_grad: 
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale
        
    def forward(self, input):
        x = self.linear(input)
        scale = self.generate_scale(x)
        out = torch.sin(self.omega_0 * scale * x)
        return out
    

class Finer(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, first_omega_0=30, hidden_omega_0=30.0, bias=True, 
                 first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad))

        for i in range(hidden_layers):
            self.net.append(FinerLayer(hidden_features, hidden_features, omega_0=hidden_omega_0, scale_req_grad=scale_req_grad))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output        

    

