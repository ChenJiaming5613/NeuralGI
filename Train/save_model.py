import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from config import Config
from model import VoxelMLP

def save_model(config: Config, saving_path: str):
    print(f"\nLoading model: {config.model_path}")
    model = VoxelMLP(config).to('cpu')

    # Load Weights
    try:
        checkpoint = torch.load(config.model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Failed to load model! Error: {e}")
        return
    
    # state_dict = model.state_dict()

    # for name, param in state_dict.items():
    #     data = param.detach().cpu().numpy().astype(np.float32)
    #     print(name, data.shape)

    all_params = []
    for layer in model.model:
        if isinstance(layer, nn.Linear):
            w = layer.weight.detach().numpy().astype(np.float32)
            b = layer.bias.detach().numpy().astype(np.float32)            
            print(f"Exporting Layer: W shape {w.shape}, Bias shape {b.shape}")
            all_params.append(w.flatten())
            all_params.append(b.flatten())

    flat_data = np.concatenate(all_params)
    flat_data.tofile(saving_path)

    print(f'Sample', flat_data[:10], flat_data[-10:])
    print(f"\nSaved to {saving_path}")
    print(f"Total float count: {len(flat_data)}")
    print(f"File size: {len(flat_data) * 4} bytes")

    print("\n=== C++ Config Info ===")
    print(f"Input Dim: {config.input_dim}")
    print(f"Hidden Dims: {config.hidden_dims}")
    print(f"Output Dim: {config.output_dim}")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    json_path = os.path.join(current_dir, 'data/VLM_ThirdPersonExampleMap.json')
    config = Config(json_path)
    save_model(config, os.path.join(config.save_dir, "model.bin"))
