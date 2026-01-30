import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import Config
from model import VoxelMLP
from dataset import load_and_preprocess_data
import eval

# ===================== Training & Validation Functions =====================
def train_one_epoch(model, train_loader, criterion, optimizer, config, epoch):
    """Train for one epoch."""
    model.train()  # Set to training mode
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
    
    for batch_inputs, batch_labels in pbar:
        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item() * batch_inputs.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    # Calculate average loss
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

# ===================== Visualization =====================
def plot_loss(train_losses, config, epoch):
    """Plot Training Loss Curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss (Epoch {epoch+1})")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(config.save_dir, f"loss_curve_epoch_{epoch+1}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved: {save_path}")

# ===================== Main Training Flow =====================
def main(config : Config):
    print(f"Training Config:")
    print(f"   Device: {config.device}")
    print(f"   Model Arch: {config.input_dim} -> {config.hidden_dims} -> {config.output_dim}")
    print(f"   Epochs: {config.epochs} | Batch Size: {config.batch_size} | LR: {config.lr}")
    
    # Create model save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Load Data
    # unpack train_loader, eval_loader (unused in train), total_samples
    train_loader, _, total_samples = load_and_preprocess_data(config)
    
    if train_loader is None:
        return
    
    # Initialize Model, Criterion, Optimizer
    model = VoxelMLP(config).to(config.device)
    criterion = nn.MSELoss()  # MSE Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Track Losses
    train_losses = []
    
    # Start Training
    print("\nStarting Training on FULL Dataset...")
    for epoch in range(config.epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config, epoch)
        
        # Record Losses
        train_losses.append(train_loss)
        
        # Step Scheduler
        scheduler.step()
        
        # Log Progress
        # print(f"\nEpoch {epoch+1}/{config.epochs} | "
        #       f"Train Loss: {train_loss:.6f} | "
        #       f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save Model
        if (epoch + 1) % config.save_freq == 0:
            model_path = os.path.join(config.save_dir, f"mlp_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
            }, model_path)
            print(f"Model saved: {model_path}")
        
        # Plot Loss
        if (epoch + 1) % config.plot_freq == 0:
            plot_loss(train_losses, config, epoch)
    
    # Training Complete: Save Final Model
    final_model_path = os.path.join(config.save_dir, "mlp_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining Complete! Final model saved to: {final_model_path}")
    
    # Plot Final Loss Curve
    plot_loss(train_losses, config, config.epochs-1)

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    json_path = os.path.join(current_dir, 'data/VLM_ThirdPersonExampleMap.json')
    config = Config(json_path)
    main(config)
    # Run evaluation after training
    eval.main(config)