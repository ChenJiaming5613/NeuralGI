import os
import json
from pathlib import Path
import torch
import torch.nn as nn

class Config:
    def __init__(self, json_path: str):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise f'JSON file not found: {json_path}'
        
        brick_size = data['brickSize']
        padded_brick_size = brick_size + 1
        indirection_data_dim = data['indirectionTextureDimensions']
        self.volume_dim = [
            indirection_data_dim['x'] * padded_brick_size,
            indirection_data_dim['y'] * padded_brick_size,
            indirection_data_dim['z'] * padded_brick_size
        ]

        current_dir = Path(__file__).resolve().parent

        # ===================== Data Settings =====================
        self.data_path = os.path.join(current_dir, 'data/train.npy') # Path to training data
        
        # ===================== Model Settings =====================
        self.input_dim = 3                    # XYZ coordinates
        self.hidden_dims = [64, 64, 64, 64, 64, 64]           # Hidden layer dimensions
        self.output_dim = 3                   # RGB labels
        self.activation = nn.ReLU()           # Activation function
        
        # ===================== Training Settings =====================
        self.batch_size = 512                 # Batch size
        self.lr = 1e-3                        # Learning rate
        self.epochs = 200                     # Number of training epochs
        # Device configuration (GPU/CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ===================== Saving Settings =====================
        self.save_dir = os.path.join(current_dir, 'model_checkpoints')  # Directory to save models
        self.model_path = os.path.join(self.save_dir, 'mlp_final.pth')  # Path to trained model
        self.save_freq = 100                  # Save model every N epochs
        self.plot_freq = 100                  # Plot loss curve every N epochs
