"""
Configuration file for Digital Image Processing Assignment
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
IMAGES_DIR = PROJECT_ROOT / "Images"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)

# Image processing parameters
DEFAULT_GAMMA_BRIGHT = [0.6, 0.4, 0.3]
DEFAULT_GAMMA_DARK = [3.0, 4.0, 5.0]
DEFAULT_BIT_PLANES = 8

# Filter parameters
BOX_FILTER_SIZES = [3, 11, 21]
GAUSSIAN_SIGMAS = [0.8, 2.5, 5.0]
LOCAL_HIST_WINDOW_SIZE = 3

# Sharpening parameters
LAPLACIAN_KERNEL = [[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]]

SOBEL_X_KERNEL = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]

SOBEL_Y_KERNEL = [[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]]

# Display settings
FIGURE_SIZE_COMPARISON = (12, 5)
FIGURE_SIZE_GRID_2x4 = (16, 8)
FIGURE_SIZE_GRID_4x2 = (12, 16)
DPI = 100
