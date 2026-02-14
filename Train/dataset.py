import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from loguru import logger

def load_and_preprocess_data(config):
    # Load data
    logger.info(f"Loading data: {config.data_path}")
    if not hasattr(config, "data_path"):
        logger.error(f"Config missing data_path")
        raise ValueError("Config missing data_path")
        
    try:
        data = np.load(config.data_path).astype(np.float32)
    except FileNotFoundError:
        logger.error(f"Data file not found: {config.data_path}")
        return None, None, 0

    logger.info(f"Original data shape: {data.shape} (Samples x 6)")
    
    # Split inputs and labels
    inputs = data[:, :3]   # First 3 columns: XYZ coordinates
    labels = data[:, 3:]   # Last 3 columns: RGB values
    
    # Convert to tensors
    inputs_tensor = torch.from_numpy(inputs).to(config.device)
    labels_tensor = torch.from_numpy(labels).to(config.device)
    
    total_samples = len(inputs)
    
    # Create DataLoaders
    train_dataset = TensorDataset(inputs_tensor, labels_tensor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,  # Shuffle training data
        num_workers=0  # Set to 0 for Windows to avoid multi-threading issues
    )
    
    # Loader for evaluation (using full dataset, no shuffle)
    eval_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False, # Don't shuffle for evaluation
        num_workers=0
    )
    
    logger.info(f"Data preprocessing complete:")
    logger.info(f"   Total samples: {total_samples}")
    logger.info(f"   Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
    logger.info(f"   Label range: [{labels.min():.4f}, {labels.max():.4f}]")
    
    # Return train_loader (shuffled) and eval_loader (not shuffled)
    return train_loader, eval_loader, total_samples
