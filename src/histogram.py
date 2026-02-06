import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))


def calculate_histogram(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    return hist


def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    
    img_equalized = (cdf_normalized[img] * 255).astype(np.uint8)
    
    return img_equalized, hist, cdf


def local_histogram_equalization(img, window_size=3):
    h, w = img.shape
    pad = window_size // 2
    img_padded = np.pad(img, pad, mode='reflect')
    img_local_eq = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            window = img_padded[i:i+window_size, j:j+window_size]
            
            hist, _ = np.histogram(window.flatten(), bins=256, range=[0, 256])
            cdf = hist.cumsum()
            
            if cdf[-1] > 0:
                cdf_normalized = cdf / cdf[-1]
            else:
                cdf_normalized = cdf
            
            center_val = img[i, j]
            img_local_eq[i, j] = (cdf_normalized[center_val] * 255).astype(np.uint8)
    
    return img_local_eq


def histogram_matching(img, target_hist):
    src_hist = calculate_histogram(img)
    src_cdf = src_hist.cumsum()
    src_cdf = src_cdf / src_cdf[-1]
    
    target_cdf = target_hist.cumsum()
    target_cdf = target_cdf / target_cdf[-1]
    
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(target_cdf - src_cdf[i])
        mapping[i] = np.argmin(diff)
    
    matched = mapping[img]
    return matched


def adaptive_histogram_equalization(img, clip_limit=2.0, grid_size=(8, 8)):
    h, w = img.shape
    grid_h, grid_w = grid_size
    
    tile_h = h // grid_h
    tile_w = w // grid_w
    
    result = np.zeros_like(img)
    
    for i in range(grid_h):
        for j in range(grid_w):
            y_start = i * tile_h
            y_end = (i + 1) * tile_h if i < grid_h - 1 else h
            x_start = j * tile_w
            x_end = (j + 1) * tile_w if j < grid_w - 1 else w
            
            tile = img[y_start:y_end, x_start:x_end]
            
            hist = calculate_histogram(tile)
            
            clip_val = clip_limit * hist.mean()
            excess = 0
            for k in range(256):
                if hist[k] > clip_val:
                    excess += hist[k] - clip_val
                    hist[k] = clip_val
            
            bonus = excess // 256
            hist += bonus
            
            cdf = hist.cumsum()
            if cdf[-1] > 0:
                cdf = cdf / cdf[-1]
                tile_eq = (cdf[tile] * 255).astype(np.uint8)
            else:
                tile_eq = tile
            
            result[y_start:y_end, x_start:x_end] = tile_eq
    
    return result
