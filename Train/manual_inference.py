import os
import random
from pathlib import Path
import torch
import numpy as np
from config import Config
from model import VoxelMLP

def generate_random_xyz(n):
    return [(random.random(), random.random(), random.random()) for _ in range(n)]

def manual_inference_numpy(input_data, state_dict, config):
    """
    解析 state_dict 并使用 numpy 手动进行矩阵乘法
    """
    # 确保输入是 numpy 数组
    x = np.array(input_data, dtype=np.float32)
    
    # 计算总共的 Linear 层数
    # 结构: Linear -> ReLU -> ... -> Linear
    num_linear_layers = len(config.hidden_dims) + 1
    
    # 在 nn.Sequential 中，Linear 层的索引通常是 0, 2, 4, 6...
    # 对应的 ReLU 层索引是 1, 3, 5...
    
    for i in range(num_linear_layers):
        layer_idx_in_sequential = i * 2 # 0, 2, 4...
        
        # 1. 构建 state_dict 中的键名
        weight_key = f'model.{layer_idx_in_sequential}.weight'
        bias_key   = f'model.{layer_idx_in_sequential}.bias'
        
        if weight_key not in state_dict:
            raise KeyError(f"Key {weight_key} not found in state_dict. Check layer indexing.")

        # 2. 提取权重和偏置，转为 numpy
        # PyTorch Linear 权重形状是 [out_features, in_features]
        w_torch = state_dict[weight_key].cpu()
        b_torch = state_dict[bias_key].cpu()
        
        W = w_torch.numpy()
        b = b_torch.numpy()
        
        # 3. 线性计算: y = x * W^T + b
        # input shape: [batch, in], W shape: [out, in], W.T shape: [in, out]
        # result shape: [batch, out]
        x = np.dot(x, W.T) + b
        
        # 4. 激活函数 (最后一层不加激活)
        if i < num_linear_layers - 1:
            # ReLU: max(0, x)
            x = np.maximum(0, x)
    
    return x

def run_verification():
    current_dir = Path(__file__).resolve().parent
    json_path = os.path.join(current_dir, 'data/VLM_ThirdPersonExampleMap.json')
    config = Config(json_path)
    
    device = torch.device('cpu')
    model = VoxelMLP(config).to(device)
    
    # --- Load Weights ---
    if os.path.exists(config.model_path):
        print(f"Loading model from {config.model_path}...")
        checkpoint = torch.load(config.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
    else:
        print(f"Model file not found at {config.model_path}. Using RANDOM weights for functionality test.")
        state_dict = model.state_dict()
    
    model.eval()

    print("\n>>> Defining User Input (Batch Size = 10)")
    # user_input = generate_random_xyz(10)
    user_input = [(0.5, 0.2, 0.9)]
    user_input_np = np.array(user_input, dtype=np.float32)
    user_input_tensor = torch.from_numpy(user_input_np).to(device)
    print(user_input_np)
    # --- 1. Torch Inference ---
    print(">>> Running Torch Inference...")
    with torch.no_grad():
        out_torch = model(user_input_tensor)
    out_torch_np = out_torch.numpy()

    # --- 2. Manual Inference ---
    print(">>> Running Manual NumPy Inference...")
    out_manual_np = manual_inference_numpy(user_input_np, state_dict, config)

    # --- 3. Compare Results ---
    print("\n" + "="*30)
    print("Comparison Result")
    print("="*30)
    
    print(f"Torch Output:\n{out_torch_np}")
    print(f"Manual Output:\n{out_manual_np}")
    
    diff = np.abs(out_torch_np - out_manual_np)
    max_diff = np.max(diff)
    
    is_consistent = np.allclose(out_torch_np, out_manual_np, atol=1e-5)
    
    print("-" * 30)
    print(f"Max Difference: {max_diff:.8f}")
    
    if is_consistent:
        print("\n✅ Success: The results are mathematically consistent.")
    else:
        print("\n❌ Failed: The results differ significantly.")

if __name__ == "__main__":
    run_verification()