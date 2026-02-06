import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import IMAGES_DIR, RESULTS_DIR, DPI


def load_image(filename, grayscale=True):
    filepath = IMAGES_DIR / filename
    
    if not filepath.exists():
        print(f"Warning: {filename} not found. Creating placeholder.")
        return np.zeros((256, 256), dtype=np.uint8)
    
    if grayscale:
        img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img is None:
        print(f"Warning: Could not load {filename}. Creating placeholder.")
        return np.zeros((256, 256), dtype=np.uint8)
    
    return img


def save_image(img, filename):
    filepath = RESULTS_DIR / filename
    
    if len(img.shape) == 2:
        cv2.imwrite(str(filepath), img)
    else:  # Color
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), img_bgr)
    
    print(f"Saved: {filename}")


def show_comparison(original, processed, title_original="Original", 
                   title_processed="Processed", figsize=(12, 5), 
                   save_name=None):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    axes[0].set_title(title_original, fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(processed, cmap='gray' if len(processed.shape) == 2 else None)
    axes[1].set_title(title_processed, fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_name:
        save_path = RESULTS_DIR / save_name
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved figure: {save_name}")
    
    plt.show()
    plt.close()


def show_multiple_images(images, titles, grid_shape, figsize=(16, 8), 
                        suptitle=None, save_name=None):
    rows, cols = grid_shape
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        if idx < len(axes):
            axes[idx].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[idx].set_title(title, fontweight='bold', fontsize=10)
            axes[idx].axis('off')
    
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    if suptitle:
        plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_name:
        save_path = RESULTS_DIR / save_name
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved figure: {save_name}")
    
    plt.show()
    plt.close()


def normalize_for_display(img):
    img_min, img_max = img.min(), img.max()
    if img_max - img_min < 1e-10:
        return np.zeros_like(img, dtype=np.uint8)
    
    normalized = ((img - img_min) / (img_max - img_min) * 255)
    return normalized.astype(np.uint8)


def apply_convolution(img, kernel):
    kernel = np.array(kernel, dtype=np.float32)
    size = kernel.shape[0]
    pad = size // 2
    
    img_padded = np.pad(img.astype(np.float32), pad, mode='reflect')
    h, w = img.shape
    result = np.zeros_like(img, dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            window = img_padded[i:i+size, j:j+size]
            result[i, j] = np.sum(window * kernel)
    
    return result
