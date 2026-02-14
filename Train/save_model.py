import os
import torch
import numpy as np
from loguru import logger
from config import Config
from model import VoxelMLP

def save_model(config: Config, saving_path: str):
    logger.info(f"Loading model: {config.model_path}")
    model = VoxelMLP(config).to('cpu')

    # Load Weights
    try:
        checkpoint = torch.load(config.model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model! Error: {e}")
        return

    all_params = []
    layer_count = 0

    logger.info("--- Exporting Layers ---")
    for i, layer in enumerate(model.model):
        if hasattr(layer, 'linear'):
            w = layer.linear.weight.detach().cpu().numpy().astype(np.float32)
            
            if layer.linear.bias is not None:
                b = layer.linear.bias.detach().cpu().numpy().astype(np.float32)
            else:
                b = np.zeros(w.shape[0], dtype=np.float32)

            logger.info(f"Layer {i}: Weight {w.shape} | Bias {b.shape}")
            
            all_params.append(w.flatten())
            all_params.append(b.flatten())
            
            layer_count += 1
        else:
            logger.info(f"Warning: Layer {i} does not have a 'linear' attribute, skipping.")

    if layer_count == 0:
        logger.info("Error: No layers were exported! Check your model structure.")
        return

    flat_data = np.concatenate(all_params)
    
    os.makedirs(os.path.dirname(saving_path), exist_ok=True)
    
    flat_data.tofile(saving_path)

    logger.info(f"--- Success ---")
    logger.info(f"Saved to: {saving_path}")
    logger.info(f"Total layers exported: {layer_count}")
    logger.info(f"Total float count: {len(flat_data)}")
    logger.info(f"File size: {len(flat_data) * 4 / 1024:.2f} KB")
    
    logger.info(f"Head (first 5): {flat_data[:5]}")
    logger.info(f"Tail (last 5): {flat_data[-5:]}")
