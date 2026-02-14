import os
import numpy as np
import OpenEXR
import Imath
from loguru import logger

def load_image(path):
    if not os.path.exists(path):
        return None
    
    try:
        file = OpenEXR.InputFile(path)
        dw = file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        
        img_channels = [file.channel(c, pt) for c in ['R', 'G', 'B']]
        
        img_arrs = [np.frombuffer(c, dtype=np.float32).reshape(size[1], size[0]) for c in img_channels]
        img = np.dstack(img_arrs)
        return img

    except Exception as e:
        logger.error(f"Failed to read exr file: {e}")
        return None

def write_exr(path, data, dtype=np.float32):
    if data.ndim == 2:
        # (H, W) -> (H, W, 1)
        data = data[:, :, np.newaxis]
    
    height, width, channels = data.shape
    
    if dtype == np.float32:
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    elif dtype == np.float16:
        pixel_type = Imath.PixelType(Imath.PixelType.HALF)
        data = data.astype(np.float16) 
    else:
        raise ValueError("Unsupported data type, only support np.float32 or np.float16")

    header = OpenEXR.Header(width, height)
    
    if channels == 3:
        channel_names = ['R', 'G', 'B']
    elif channels == 4:
        channel_names = ['R', 'G', 'B', 'A']
    elif channels == 1:
        channel_names = ['Y']
    else:
        raise ValueError(f"Unsupported num of channels: {channels}")

    header['channels'] = {name: Imath.Channel(pixel_type) for name in channel_names}

    channel_data = {}
    for i, name in enumerate(channel_names):
        channel_data[name] = data[:, :, i].tobytes()

    out = OpenEXR.OutputFile(path, header)
    out.writePixels(channel_data)
    out.close()