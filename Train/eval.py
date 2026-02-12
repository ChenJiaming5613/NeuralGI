import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from config import Config
from model import VoxelMLP
from dataset import load_and_preprocess_data
import exr_util

def evaluate_model(config):
    """
    Main Evaluation Logic:
    1. Load model and set to inference mode
    2. Iterate through samples, denormalize, and convert to int
    3. Calculate L2 loss between predictions and labels
    4. Calculate average pixel loss per sample
    """
    # Step 1: Load Model
    print(f"\nLoading model: {config.model_path}")
    model = VoxelMLP(config).to(config.device)
    
    # Load Weights
    try:
        checkpoint = torch.load(config.model_path, map_location=config.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Failed to load model! Error: {e}")
        return
    
    # Step 2: Set to Evaluation Mode
    model.eval()
    
    # Step 3: Initialize Loss Statistics
    total_l2_loss = 0.0
    l2_loss_fn = nn.MSELoss(reduction="sum")
    
    # Determine which dataloader to use
    if getattr(config, 'dataloader', None) is None:
         # If not provided, load evaluation data
         # load_and_preprocess_data returns (train_loader, eval_loader, total_samples)
         _, eval_loader, total_samples = load_and_preprocess_data(config, eval_mode=True)
         dataloader = eval_loader 
         config.total_samples = len(eval_loader.dataset) # Update total samples for metrics
    else:
         dataloader = config.dataloader

    if dataloader is None:
        print("No data to evaluate.")
        return

    print(f"\nStarting Inference Evaluation (Device: {config.device})...")
    
    # Step 4: Batch Inference + Loss Calculation
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, (batch_inputs, batch_labels) in pbar:
            # Forward: Normalized RGB
            batch_preds = model(batch_inputs)
            
            # Calculate Batch L2 Loss
            batch_l2_loss = l2_loss_fn(batch_preds, batch_labels)
            
            # Accumulate Total Loss
            total_l2_loss += batch_l2_loss.item()
            
            # Update Progress Bar
            pbar.set_postfix({
                "Batch Avg L2 Loss": f"{batch_l2_loss.item() / batch_inputs.size(0):.4f}"
            })
    
    # Step 5: Calculate Average Loss
    avg_sample_l2_loss = total_l2_loss / config.total_samples
    print(f"\nEvaluation Results:")
    print(f"   Total Cumulative L2 Loss: {total_l2_loss:.4f}")
    print(f"   Total Samples: {config.total_samples}")
    print(f"   Avg Pixel L2 Loss per Sample: {avg_sample_l2_loss:.4f}")
    
    # Calculate RMSE
    avg_sample_rmse = np.sqrt(avg_sample_l2_loss)
    print(f"   Avg Pixel RMSE per Sample: {avg_sample_rmse:.4f}")

    volume_dim_x, volume_dim_y, volume_dim_z  = config.volume_dim
    pred_data = torch.zeros((volume_dim_z, volume_dim_y, volume_dim_x, 3), dtype=torch.float32)
    with torch.no_grad():
        with tqdm(total=volume_dim_x * volume_dim_y * volume_dim_z, desc='Generate Pred Data') as pbar:
            for z in range(volume_dim_z):
                for y in range(volume_dim_y):
                    for x in range(volume_dim_x):
                        input = torch.tensor([
                            x / (volume_dim_x - 1),
                            y / (volume_dim_y - 1),
                            z / (volume_dim_z - 1)], dtype=torch.float32, device=config.device)
                        rgb = model(input)
                        pred_data[z, y, x] = rgb
                        pbar.update(1)
                        # print(rgb)
                        # return
    print('Generate Done')
    os.makedirs(os.path.join(config.save_dir, 'eval'), exist_ok=True)
    for z in range(pred_data.shape[0]):
        slice_data = pred_data[z].detach().cpu().numpy() # (Height, Width, 3)
        filename = f'pred_ambient_slice_{z}.exr'
        path = os.path.join(config.save_dir, 'eval', filename)
        exr_util.write_exr(path, slice_data)
    print(f'Saved exr textures')

def main(config):
    print("======= Voxel MLP Model Evaluation Script =======")
    print(f"Eval Config:")
    print(f"   Device: {config.device}")
    print(f"   Batch Size: {config.batch_size}")
    
    # Load Data (Use Full Dataset)
    # The load_and_preprocess_data function now returns the eval_loader as the second argument
    _, eval_loader, total_samples = load_and_preprocess_data(config)
    config.dataloader = eval_loader
    config.total_samples = len(eval_loader.dataset)
    
    # Execute Evaluation
    evaluate_model(config)
    
    print("\nEvaluation Complete!")
