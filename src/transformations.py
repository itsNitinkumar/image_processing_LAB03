import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))


def image_negative(img):
    negative = 255 - img.astype(np.float32)
    return np.clip(negative, 0, 255).astype(np.uint8)


def log_transformation(img, c=None):
    img_float = img.astype(np.float32)
    
    if c is None:
        c = 255.0 / np.log(1 + np.max(img_float))
    
    log_img = c * np.log(1 + img_float)
    return np.clip(log_img, 0, 255).astype(np.uint8)


def gamma_correction(img, gamma=1.0):
    normalized = img.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    return (corrected * 255.0).astype(np.uint8)


def bit_plane_slicing(img):
    bit_planes = []
    for k in range(8):
        bit_plane = ((img >> k) & 1) * 255
        bit_planes.append(bit_plane)
    return bit_planes


def contrast_stretching(img, r1=None, s1=None, r2=None, s2=None):
    if r1 is None or r2 is None:
        r1 = np.percentile(img, 5)
        r2 = np.percentile(img, 95)
    
    if s1 is None:
        s1 = 0
    if s2 is None:
        s2 = 255
    
    result = np.zeros_like(img, dtype=np.float32)
    
    mask1 = img < r1
    result[mask1] = (s1 / r1) * img[mask1]
    
    mask2 = (img >= r1) & (img <= r2)
    result[mask2] = ((s2 - s1) / (r2 - r1)) * (img[mask2] - r1) + s1
    
    mask3 = img > r2
    result[mask3] = ((255 - s2) / (255 - r2)) * (img[mask3] - r2) + s2
    
    return np.clip(result, 0, 255).astype(np.uint8)
