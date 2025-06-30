import torch
import torch.nn as nn
import math
from ChebyKANLayer import ChebyKANLayer

class ChebyLayer(nn.Module):
    def __init__(self, in_features, out_features, deg, init_method):
        super(ChebyLayer, self).__init__()
        self.cheby = ChebyKANLayer(in_features, out_features, deg, init_method)
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
    

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)  
        
    def forward(self, input):
        return self.linear(input)   


    

class LowRankReLULayer(nn.Module):
    def __init__(self, in_features, out_features, rank=32, bias=True, nonlinearity='relu', linear_init_type='kaiming_uniform'):
        super(LowRankReLULayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.nonlinearity = nonlinearity
        
        # Create two smaller matrices for low-rank approximation
        self.weight_left = nn.Parameter(torch.Tensor(in_features, rank))
        self.weight_right = nn.Parameter(torch.Tensor(rank, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters(linear_init_type)
    
    def reset_parameters(self, linear_init_type='kaiming_uniform'):
        if linear_init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.weight_left, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_right, a=math.sqrt(5))
        elif linear_init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(self.weight_left, a=math.sqrt(5))
            nn.init.kaiming_normal_(self.weight_right, a=math.sqrt(5))
        elif linear_init_type == 'orthogonal':
            nn.init.orthogonal_(self.weight_left)
            nn.init.orthogonal_(self.weight_right)
        elif linear_init_type == 'uniform':
            nn.init.uniform_(self.weight_left, a=-0.5, b=0.5)
            nn.init.uniform_(self.weight_right, a=-0.5, b=0.5)
        elif linear_init_type == 'normal':
            nn.init.normal_(self.weight_left, mean=0.0, std=1 / (self.in_features * self.rank))
            nn.init.normal_(self.weight_right, mean=0.0, std=1 / (self.rank * self.out_features))
        elif linear_init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(self.weight_left)
            nn.init.xavier_uniform_(self.weight_right)
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_left)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        # Compute the low-rank approximation of the weight matrix
        weight = torch.matmul(self.weight_left, self.weight_right)
        
        # Apply the linear transformation
        output = torch.matmul(input, weight)
        
        if self.bias is not None:
            output += self.bias
        
        if self.nonlinearity == 'relu':
            return nn.functional.relu(output)
        
        elif self.nonlinearity == None or self.nonlinearity == 'none':
            return output

    
class SL2A(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, deg=512, outermost_linear=True, nonlinearity='relu', rank=32, init_method='xavier_uniform', linear_init_type='kaiming_uniform'):
        super(SL2A, self).__init__()
        self.nonlin = ChebyLayer
        self.net = nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.rank = rank
        self.net.append(self.nonlin(in_features, hidden_features, deg=deg, init_method=init_method))

        self.linear = nn.Linear(hidden_features, out_features)

        self.nonlin = LowRankReLULayer

        for i in range(hidden_layers):
            if i == 0:
              self.net.append(self.nonlin(hidden_features, hidden_features, rank=self.rank, nonlinearity=self.nonlinearity, linear_init_type=linear_init_type))
            else:
              self.net.append(self.nonlin(hidden_features, hidden_features, rank=self.rank, nonlinearity=self.nonlinearity, linear_init_type=linear_init_type))

        if outermost_linear:
            self.net.append(self.linear)
        else:
            raise NotImplementedError("")
        
        
    def forward(self, coords):
        coords = coords.squeeze()
        # activations = [] 
        for i, layer in enumerate(self.net):
            if i == 0:
              x = layer(coords)
              y = x
            else:
                y = layer(torch.einsum('ij,ij->ij', x,y))

        return y
    
