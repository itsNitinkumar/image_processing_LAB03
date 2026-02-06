"""
Digital Image Processing - Source Package
Contains modules for image transformations, histogram processing, and filtering.
"""

from .transformations import (
    image_negative,
    log_transformation,
    gamma_correction,
    bit_plane_slicing,
    contrast_stretching
)

from .histogram import (
    calculate_histogram,
    histogram_equalization,
    local_histogram_equalization,
    histogram_matching,
    adaptive_histogram_equalization
)

from .filters import (
    box_filter,
    gaussian_filter,
    median_filter,
    laplacian_filter,
    sharpen_with_laplacian,
    sobel_gradient,
    unsharp_masking,
    high_boost_filter,
    mixed_spatial_enhancement
)

from .utils import (
    load_image,
    save_image,
    show_comparison,
    show_multiple_images,
    normalize_for_display,
    apply_convolution
)

__all__ = [
    # Transformations
    'image_negative',
    'log_transformation',
    'gamma_correction',
    'bit_plane_slicing',
    'contrast_stretching',
    
    # Histogram
    'calculate_histogram',
    'histogram_equalization',
    'local_histogram_equalization',
    'histogram_matching',
    'adaptive_histogram_equalization',
    
    # Filters
    'box_filter',
    'gaussian_filter',
    'median_filter',
    'laplacian_filter',
    'sharpen_with_laplacian',
    'sobel_gradient',
    'unsharp_masking',
    'high_boost_filter',
    'mixed_spatial_enhancement',
    
    # Utils
    'load_image',
    'save_image',
    'show_comparison',
    'show_multiple_images',
    'normalize_for_display',
    'apply_convolution'
]
