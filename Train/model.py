import torch
import torch.nn as nn
import numpy as np

class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, is_last=False):
        super().__init__()
        self.is_last = is_last
        
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU(inplace=True)
        
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # ReLU Kaiming (He) Initialization
            nn.init.kaiming_normal_(self.linear.weight, a=0, mode='fan_in', nonlinearity='relu')
            
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        
        if self.is_last:
            return x
            
        return self.activation(x)

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, is_last=False):
        super().__init__()
        self.in_features = in_features
        self.w0 = 30.0
        self.is_first = is_first
        self.is_last = is_last
        
        self.linear = nn.Linear(in_features, out_features)
        
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_features if self.is_first else np.sqrt(6 / self.in_features) / self.w0
        
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        
        if self.is_last:
            return x
            
        return torch.sin(self.w0 * x)

class VoxelMLP(nn.Module):
    """
    Voxel MLP Model
    Maps 3D coordinates (XYZ) to 3D color values (RGB).
    """
    def __init__(self, config):
        super(VoxelMLP, self).__init__()
        self.config = config

        if config.activation == 'siren':
            LayerType = SirenLayer
        elif config.activation == 'relu':
            LayerType = ReLULayer
        else:
            raise ValueError(f"Unknown layer type: {config.layer_type}")
        
        # Build network layers
        layers = []
        
        # Input layer -> First hidden layer
        layers.append(LayerType(
            config.input_dim, 
            config.hidden_dims[0], 
            is_first=True,
        ))
        
        # Hidden layers
        for i in range(1, len(config.hidden_dims)):
            layers.append(LayerType(
                config.hidden_dims[i - 1], 
                config.hidden_dims[i], 
            ))
            
        # Last hidden layer -> Output layer
        layers.append(LayerType(
            config.hidden_dims[-1], 
            config.output_dim, 
            is_last=True
        ))
        
        # Combine into Sequential container
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
