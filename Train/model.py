import torch.nn as nn

class VoxelMLP(nn.Module):
    """
    Voxel MLP Model
    Maps 3D coordinates (XYZ) to 3D color values (RGB).
    """
    def __init__(self, config):
        super(VoxelMLP, self).__init__()
        self.config = config
        
        # Build network layers
        layers = []
        
        # Input layer -> First hidden layer
        layers.append(nn.Linear(config.input_dim, config.hidden_dims[0]))
        layers.append(config.activation)
        
        # Hidden layers
        for i in range(1, len(config.hidden_dims)):
            layers.append(nn.Linear(config.hidden_dims[i-1], config.hidden_dims[i]))
            layers.append(config.activation)
            
        # Last hidden layer -> Output layer
        layers.append(nn.Linear(config.hidden_dims[-1], config.output_dim))
        
        # Combine into Sequential container
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        return self.model(x)
