import os
import json
import struct
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
import exr_util

def decode_r11g11b10f(b0, b1, b2, b3):
    """
    4 bytes -> R11G11B10 float
    Little Endian
    """
    packed_int = struct.unpack('<I', bytes([b0, b1, b2, b3]))[0]
    
    r_bits = packed_int & 0x7FF
    g_bits = (packed_int >> 11) & 0x7FF
    b_bits = (packed_int >> 22) & 0x3FF
    
    def decode_float(raw_bits, mantissa_len):
        exponent_len = 5
        bias = 15
        
        mantissa_mask = (1 << mantissa_len) - 1
        mantissa = raw_bits & mantissa_mask
        exponent = raw_bits >> mantissa_len
        
        if exponent == 0:
            if mantissa == 0:
                return 0.0
            else:
                # Denormalized
                return (mantissa / (1 << mantissa_len)) * (2 ** (1 - bias))
        
        if exponent == 31:
            return float('inf') 

        # Normalized
        return (2 ** (exponent - bias)) * (1 + mantissa / (1 << mantissa_len))

    r_val = decode_float(r_bits, 6)
    g_val = decode_float(g_bits, 6)
    b_val = decode_float(b_bits, 5)
    
    return r_val, g_val, b_val

def generate_random_samples(volume, num_samples, scale_factor=1.0):
    """
    基于 3D volume 生成随机连续坐标，并进行三线性插值采样。
    用于让 SIREN 学习体素间的平滑过渡。
    
    Args:
        volume: shape (D, H, W, 3) 的 float32 数组
        num_samples: 生成的随机点数量
        scale_factor: 颜色缩放因子
    Returns:
        samples: shape (num_samples, 6) 的数组，包含 [x, y, z, r, g, b]
    """
    print(f"Generating {num_samples} random interpolated samples...")
    D, H, W, C = volume.shape

    # 1. 生成随机浮点坐标 (对应 Z, Y, X 轴的索引范围)
    # 范围: [0, D-1], [0, H-1], [0, W-1]
    rand_z = np.random.rand(num_samples) * (D - 1)
    rand_y = np.random.rand(num_samples) * (H - 1)
    rand_x = np.random.rand(num_samples) * (W - 1)

    # 2. 准备插值坐标 (3, N) -> map_coordinates 需要 (Dim0, Dim1, Dim2) 顺序
    coords = np.stack([rand_z, rand_y, rand_x])

    # 3. 三线性插值 (Order=1)，边缘截断 (Mode='nearest' 对应 UE Clamp)
    # 分别对 R, G, B 通道采样
    r_vals = map_coordinates(volume[..., 0], coords, order=1, mode='nearest')
    g_vals = map_coordinates(volume[..., 1], coords, order=1, mode='nearest')
    b_vals = map_coordinates(volume[..., 2], coords, order=1, mode='nearest')

    # 4. 归一化坐标到 0~1 (作为神经网络输入)
    # 注意输出顺序是: x, y, z, r, g, b
    norm_x = rand_x / (W - 1)
    norm_y = rand_y / (H - 1)
    norm_z = rand_z / (D - 1)

    # 5. 应用 Scale Factor 并组合
    r_vals /= scale_factor
    g_vals /= scale_factor
    b_vals /= scale_factor

    # 堆叠成 [N, 6]
    return np.stack([norm_x, norm_y, norm_z, r_vals, g_vals, b_vals], axis=1).astype(np.float32)

def make_train_dataset(json_path: str, output_dir: str, scale_factor: float=1.0, save_exr: bool=False):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'Loading JSON: {json_path}')
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'JSON file not found: {json_path}')
        return
    
    # parse indirection data
    indirection_data = data['indirectionTextureData']
    indirection_data_dim = data['indirectionTextureDimensions']
    assert(len(indirection_data) == indirection_data_dim.get('x', -1)
                                  * indirection_data_dim.get('y', -1)
                                  * indirection_data_dim.get('z', -1) * 4)
    indirection = []
    for i in range(0, len(indirection_data), 4):
        chunk = indirection_data[i:i+4]
        if len(chunk) < 4:
            raise 'len(chunk) = ' + len(chunk)
        indirection.append([chunk[0], chunk[1], chunk[2]])
    indirection = np.array(indirection, dtype=np.int32).reshape((
        indirection_data_dim['z'], indirection_data_dim['y'], indirection_data_dim['x'], 3))
    print(indirection.shape)
    
    # parse ambient data
    ambient_data = data['ambientVectorData']
    ambient_data_dim = data['brickDataDimensions']
    width = ambient_data_dim.get('x', -1)
    height = ambient_data_dim.get('y', -1)
    depth = ambient_data_dim.get('z', -1)
    assert(len(ambient_data) == width * height * depth * 4)

    pixels_list = []
    for i in range(0, len(ambient_data), 4):
        chunk = ambient_data[i:i+4]
        if len(chunk) < 4:
            break
        r, g, b = decode_r11g11b10f(chunk[0], chunk[1], chunk[2], chunk[3])
        pixels_list.append([r, g, b])
        
    pixels_float = np.array(pixels_list, dtype=np.float32)
    volume = pixels_float.reshape((depth, height, width, 3))
    print(volume.shape)

    brick_size = data['brickSize']
    padded_brick_size = brick_size + 1
    output_volume = np.zeros((
        indirection.shape[0] * padded_brick_size,
        indirection.shape[1] * padded_brick_size,
        indirection.shape[2] * padded_brick_size, 3), dtype=np.float32)
    print(output_volume.shape)

    for z in range(indirection.shape[0]):
        for y in range(indirection.shape[1]):
            for x in range(indirection.shape[2]):
                # print((z, y, x), indirection[z, y, x])
                phy_x, phy_y, phy_z = indirection[z, y, x]
                src_range = np.array([phy_z, phy_y, phy_x]) * padded_brick_size
                dst_range = np.array([z, y, x]) * padded_brick_size
                output_volume[
                    dst_range[0] : dst_range[0] + padded_brick_size,
                    dst_range[1] : dst_range[1] + padded_brick_size,
                    dst_range[2] : dst_range[2] + padded_brick_size
                ] = volume[
                    src_range[0] : src_range[0] + padded_brick_size,
                    src_range[1] : src_range[1] + padded_brick_size,
                    src_range[2] : src_range[2] + padded_brick_size
                ]
                # print(dst_range, dst_range + padded_brick_size, '<-', src_range, src_range + padded_brick_size)
    
    output_dataset = []
    output_min = None
    output_max = None
    for z in range(output_volume.shape[0]):
        for y in range(output_volume.shape[1]):
            for x in range(output_volume.shape[2]):
                r, g, b = output_volume[z, y, x] / scale_factor
                output_dataset.append([
                    x / (output_volume.shape[2] - 1),
                    y / (output_volume.shape[1] - 1),
                    z / (output_volume.shape[0] - 1),
                    r, g, b
                ])
                if output_min is None and output_max is None:
                    output_min = min(r, g, b)
                    output_max = max(r, g, b)
                else:
                    output_min = min(output_min, r, g, b)
                    output_max = max(output_max, r, g, b)
                # print(x, y, z, r, g, b)
    grid_dataset = np.array(output_dataset, dtype=np.float32)
    print(f"Grid samples shape: {grid_dataset.shape}")

    # 2. 生成随机插值数据 (数量建议与网格点一致，或者更多)
    num_random = grid_dataset.shape[0] * 16
    random_dataset = generate_random_samples(output_volume, num_random, scale_factor)
    
    # 3. 合并与打乱
    final_dataset = np.concatenate([grid_dataset, random_dataset], axis=0)
    np.random.shuffle(final_dataset) # 这一步对训练非常关键

    print(f'min: {output_min}, max: {output_max}, scale_factor: {scale_factor}')
    saved_path = os.path.join(output_dir, 'train.npy')
    np.save(saved_path, final_dataset)
    print(f'Saved {saved_path}, Shape={final_dataset.shape}')

    if save_exr:
        os.makedirs(os.path.join(output_dir, 'textures'), exist_ok=True)
        for z in range(output_volume.shape[0]):
            slice_data = output_volume[z] # (Height, Width, 3)
            filename = f'ambient_slice_{z}.exr'
            path = os.path.join(output_dir, 'textures', filename)
            exr_util.write_exr(path, slice_data)
        print(f'Saved exr textures')

if __name__ == '__main__':
    current_dir = Path(__file__).resolve().parent
    json_path = os.path.join(current_dir, 'data/VLM_ThirdPersonExampleMap.json')
    output_dir = os.path.join(current_dir, 'data')
    make_train_dataset(json_path, output_dir, scale_factor=1.0, save_exr=True)