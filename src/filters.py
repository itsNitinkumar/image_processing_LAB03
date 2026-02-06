import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import apply_convolution
from config import LAPLACIAN_KERNEL, SOBEL_X_KERNEL, SOBEL_Y_KERNEL


def box_filter(img, size):
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    filtered = apply_convolution(img, kernel)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel = kernel / np.sum(kernel)
    return kernel


def gaussian_filter(img, size, sigma):
    kernel = gaussian_kernel(size, sigma)
    filtered = apply_convolution(img, kernel)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def median_filter(img, size=3):
    pad = size // 2
    img_padded = np.pad(img, pad, mode='reflect')
    h, w = img.shape
    result = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            window = img_padded[i:i+size, j:j+size]
            result[i, j] = np.median(window)
    
    return result


def laplacian_filter(img, kernel=None):
    if kernel is None:
        kernel = LAPLACIAN_KERNEL
    
    laplacian = apply_convolution(img, kernel)
    return laplacian


def sharpen_with_laplacian(img, c=1.0, kernel=None):
    laplacian = laplacian_filter(img, kernel)
    sharpened = img.astype(np.float32) + c * laplacian
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def sobel_gradient(img):
    grad_x = apply_convolution(img, SOBEL_X_KERNEL)
    grad_y = apply_convolution(img, SOBEL_Y_KERNEL)
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude


def unsharp_masking(img, blur_size=5, k=1.0):
    blurred = box_filter(img, blur_size)
    
    mask = img.astype(np.float32) - blurred.astype(np.float32)
    
    result = img.astype(np.float32) + k * mask
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return blurred, mask, result


def high_boost_filter(img, blur_size=5, A=1.5):
    blurred = box_filter(img, blur_size)
    result = A * img.astype(np.float32) - blurred.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def mixed_spatial_enhancement(img, gamma=0.5):
    steps = {}
    steps['original'] = img
    
    laplacian = laplacian_filter(img)
    steps['laplacian'] = laplacian
    
    sharpened = img.astype(np.float32) + laplacian
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    steps['sharpened'] = sharpened
    
    sobel_grad = sobel_gradient(img)
    steps['sobel_gradient'] = sobel_grad
    
    sobel_smoothed = box_filter(sobel_grad.astype(np.uint8), 5).astype(np.float32)
    steps['sobel_smoothed'] = sobel_smoothed
    
    laplacian_norm = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min() + 1e-10)
    sobel_norm = (sobel_smoothed - sobel_smoothed.min()) / (sobel_smoothed.max() - sobel_smoothed.min() + 1e-10)
    multiplied = laplacian_norm * sobel_norm * 255
    steps['multiplied'] = multiplied
    
    result_step6 = img.astype(np.float32) + multiplied
    result_step6 = np.clip(result_step6, 0, 255).astype(np.uint8)
    steps['added_to_original'] = result_step6
    
    from src.transformations import gamma_correction
    final_result = gamma_correction(result_step6, gamma)
    steps['final'] = final_result
    
    return steps
