import os
from pathlib import Path
import torch
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
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Failed to load model! Error: {e}")
        return

    all_params = []
    layer_count = 0

    print("\n--- Exporting Layers ---")
    for i, layer in enumerate(model.model):
        if hasattr(layer, 'linear'):
            w = layer.linear.weight.detach().cpu().numpy().astype(np.float32)
            
            if layer.linear.bias is not None:
                b = layer.linear.bias.detach().cpu().numpy().astype(np.float32)
            else:
                b = np.zeros(w.shape[0], dtype=np.float32)

            print(f"Layer {i}: Weight {w.shape} | Bias {b.shape}")
            
            all_params.append(w.flatten())
            all_params.append(b.flatten())
            
            layer_count += 1
        else:
            print(f"Warning: Layer {i} does not have a 'linear' attribute, skipping.")

    if layer_count == 0:
        print("Error: No layers were exported! Check your model structure.")
        return

    flat_data = np.concatenate(all_params)
    
    os.makedirs(os.path.dirname(saving_path), exist_ok=True)
    
    flat_data.tofile(saving_path)

    print(f"\n--- Success ---")
    print(f"Saved to: {saving_path}")
    print(f"Total layers exported: {layer_count}")
    print(f"Total float count: {len(flat_data)}")
    print(f"File size: {len(flat_data) * 4 / 1024:.2f} KB")
    
    print(f"Head (first 5): {flat_data[:5]}")
    print(f"Tail (last 5): {flat_data[-5:]}")
