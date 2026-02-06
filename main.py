import numpy as np

from config import (
    RESULTS_DIR, DEFAULT_GAMMA_BRIGHT, DEFAULT_GAMMA_DARK,
    BOX_FILTER_SIZES, GAUSSIAN_SIGMAS, LOCAL_HIST_WINDOW_SIZE
)

from src.utils import (
    load_image, save_image, show_comparison, show_multiple_images,
    normalize_for_display
)

from src.transformations import (
    image_negative, log_transformation, gamma_correction, bit_plane_slicing
)

from src.histogram import (
    histogram_equalization, local_histogram_equalization
)

from src.filters import (
    box_filter, gaussian_filter, laplacian_filter, sharpen_with_laplacian,
    sobel_gradient, unsharp_masking, mixed_spatial_enhancement
)


def task_1_image_negatives():
    print("\n" + "="*60)
    print("TASK 1: Image Negatives")
    print("="*60)
    
    img = load_image('Fig0304(a)(breast_digital_Xray)_1.jpg')
    img_negative = image_negative(img)
    
    save_image(img_negative, 'task1_negative.png')
    show_comparison(img, img_negative, 
                   "Original (Dark Mammogram)", "Image Negative",
                   save_name='task1_comparison.png')
    
    print("✓ Task 1 completed")


def task_2_log_transformation():
    print("\n" + "="*60)
    print("TASK 2: Log Transformation")
    print("="*60)
    
    img = load_image('Fig0305(a)(DFT_no_log)_1.jpg')
    img_log = log_transformation(img)
    
    save_image(img_log, 'task2_log_transform.png')
    show_comparison(img, img_log,
                   "Original (Low Dynamic Range)", "Log Transformation",
                   save_name='task2_comparison.png')
    
    print("✓ Task 2 completed")


def task_3_gamma_correction():
    print("\n" + "="*60)
    print("TASK 3: Gamma Correction")
    print("="*60)
    
    print("\nTask 3a: Brightening dark images")
    img_dark = load_image('Fig0308(a)(fractured_spine).jpg')
    
    images = [img_dark]
    titles = ['Original (Dark Spine)']
    
    for gamma in DEFAULT_GAMMA_BRIGHT:
        img_gamma = gamma_correction(img_dark, gamma)
        images.append(img_gamma)
        titles.append(f'Gamma = {gamma}')
        save_image(img_gamma, f'task3a_gamma_{gamma}.png')
    
    show_multiple_images(images, titles, (2, 2), 
                        suptitle='Gamma Correction - Brightening',
                        save_name='task3a_brightening.png')
    
    print("\nTask 3b: Darkening washed-out images")
    img_light = load_image('Fig0309(a)(washed_out_aerial_image).jpg')
    
    images = [img_light]
    titles = ['Original (Washed-out)']
    
    for gamma in DEFAULT_GAMMA_DARK:
        img_gamma = gamma_correction(img_light, gamma)
        images.append(img_gamma)
        titles.append(f'Gamma = {gamma}')
        save_image(img_gamma, f'task3b_gamma_{gamma}.png')
    
    show_multiple_images(images, titles, (2, 2),
                        suptitle='Gamma Correction - Darkening',
                        save_name='task3b_darkening.png')
    
    print("✓ Task 3 completed")


def task_4_bit_plane_slicing():
    print("\n" + "="*60)
    print("TASK 4: Bit-Plane Slicing")
    print("="*60)
    
    img = load_image('Fig0314(a)(100-dollars)_1.jpg')
    bit_planes = bit_plane_slicing(img)
    
    titles = []
    for k in range(8):
        bit_name = "LSB" if k == 0 else ("MSB" if k == 7 else f"Bit {k}")
        titles.append(f'Bit Plane {k} ({bit_name})')
        save_image(bit_planes[k], f'task4_bitplane_{k}.png')
    
    show_multiple_images(bit_planes, titles, (2, 4),
                        suptitle='Bit-Plane Slicing - All 8 Planes',
                        save_name='task4_bitplanes.png')
    
    print("✓ Task 4 completed")


def task_5_histogram_equalization():
    print("\n" + "="*60)
    print("TASK 5: Global Histogram Equalization")
    print("="*60)
    
    image_files = [
        'Fig0316(1)(top_left)_1.jpg',
        'Fig0316(2)(2nd_from_top)_1.jpg',
        'Fig0316(3)(third_from_top)_1.jpg',
        'Fig0316(4)(bottom_left)_1.jpg'
    ]
    
    titles = ['Dark Image', 'Light Image', 'Low Contrast', 'High Contrast']
    
    all_images = []
    all_titles = []
    
    for idx, (filename, title) in enumerate(zip(image_files, titles)):
        img = load_image(filename)
        img_eq, _, _ = histogram_equalization(img)
        
        all_images.extend([img, img_eq])
        all_titles.extend([f'{title} - Original', f'{title} - Equalized'])
        
        save_image(img_eq, f'task5_equalized_{idx+1}.png')
    
    show_multiple_images(all_images, all_titles, (4, 2), suptitle='Global Histogram Equalization', save_name='task5_histogram_eq.png')
    
    print("✓ Task 5 completed")


def task_6_local_histogram_processing():
    print("\n" + "="*60)
    print("TASK 6: Local Histogram Equalization")
    print("="*60)
    
    img = load_image('Fig0326(a)(embedded_square_noisy_512).jpg')
    
    print("Computing global histogram equalization...")
    img_global_eq, _, _ = histogram_equalization(img)
    
    print(f"Computing local histogram equalization (window size: {LOCAL_HIST_WINDOW_SIZE}x{LOCAL_HIST_WINDOW_SIZE})...")
    img_local_eq = local_histogram_equalization(img, LOCAL_HIST_WINDOW_SIZE)
    
    save_image(img_global_eq, 'task6_global_eq.png')
    save_image(img_local_eq, 'task6_local_eq.png')
    
    images = [img, img_global_eq, img_local_eq]
    titles = ['Original (Hidden Squares)', 
             'Global Histogram Equalization',
             f'Local Histogram Equalization ({LOCAL_HIST_WINDOW_SIZE}x{LOCAL_HIST_WINDOW_SIZE})']
    
    show_multiple_images(images, titles, (1, 3), figsize=(18, 6),
                        save_name='task6_local_vs_global.png')
    
    print("✓ Task 6 completed")


def task_7_smoothing_filters():
    print("\n" + "="*60)
    print("TASK 7: Smoothing Filters")
    print("="*60)
    
    img = load_image('Fig0333(a)(test_pattern_blurring_orig)_1.jpg')
    
    images = [img]
    titles = ['Original']
    
    for size in BOX_FILTER_SIZES:
        box_img = box_filter(img, size)
        images.append(box_img)
        titles.append(f'Box Filter {size}x{size}')
        save_image(box_img, f'task7_box_{size}x{size}.png')
    
    for size, sigma in zip(BOX_FILTER_SIZES, GAUSSIAN_SIGMAS):
        gauss_img = gaussian_filter(img, size, sigma)
        images.append(gauss_img)
        titles.append(f'Gaussian {size}x{size} (σ={sigma})')
        save_image(gauss_img, f'task7_gaussian_{size}x{size}_sigma{sigma}.png')
    
    show_multiple_images(images, titles, (2, 4), figsize=(20, 10),
                        suptitle='Smoothing Filters - Box vs Gaussian',
                        save_name='task7_smoothing_filters.png')
    
    print("✓ Task 7 completed")


def task_8_laplacian_sharpening():
    print("\n" + "="*60)
    print("TASK 8: Laplacian Sharpening")
    print("="*60)
    
    img = load_image('Fig0338(a)(blurry_moon).jpg')
    
    laplacian = laplacian_filter(img)
    laplacian_display = normalize_for_display(laplacian)
    
    sharpened_c1 = sharpen_with_laplacian(img, c=1.0)
    sharpened_c2 = sharpen_with_laplacian(img, c=2.0)
    
    save_image(laplacian_display, 'task8_laplacian.png')
    save_image(sharpened_c1, 'task8_sharpened_c1.png')
    save_image(sharpened_c2, 'task8_sharpened_c2.png')
    
    images = [img, laplacian_display, sharpened_c1, sharpened_c2]
    titles = ['Original (Blurry Moon)', 'Laplacian', 'Sharpened (c=1.0)', 'Sharpened (c=2.0)']
    
    show_multiple_images(images, titles, (2, 2), figsize=(12, 12),
                        suptitle='Laplacian Sharpening',
                        save_name='task8_laplacian_sharpening.png')
    
    print("✓ Task 8 completed")


def task_9_unsharp_masking():
    print("\n" + "="*60)
    print("TASK 9: Unsharp Masking")
    print("="*60)
    
    img = load_image('Fig0340(a)(dipxe_text).jpg')
    
    blurred, mask, result_k1 = unsharp_masking(img, blur_size=5, k=1.0)
    _, _, result_k45 = unsharp_masking(img, blur_size=5, k=4.5)
    
    mask_display = normalize_for_display(mask)
    
    save_image(blurred, 'task9_blurred.png')
    save_image(mask_display, 'task9_mask.png')
    save_image(result_k1, 'task9_unsharp_k1.png')
    save_image(result_k45, 'task9_unsharp_k45.png')
    
    images = [img, blurred, mask_display, img, result_k1, result_k45]
    titles = ['Original (Blurred Text)', 'Blurred Image', 'Mask (Original - Blurred)', 'Original (Reference)', 'Unsharp Masking (k=1.0)', 'Unsharp Masking (k=4.5)']
    
    show_multiple_images(images, titles, (2, 3), figsize=(18, 12),
                        suptitle='Unsharp Masking',
                        save_name='task9_unsharp_masking.png')
    
    print("✓ Task 9 completed")


def task_10_mixed_spatial_enhancement():
    print("\n" + "="*60)
    print("TASK 10: Mixed Spatial Enhancement")
    print("="*60)
    
    img = load_image('Fig0343(a)(skeleton_orig)_1.jpg')
    
    steps = mixed_spatial_enhancement(img, gamma=0.5)
    
    for key, value in steps.items():
        if isinstance(value, np.ndarray):
            display_img = normalize_for_display(value) if value.dtype != np.uint8 else value
            save_image(display_img, f'task10_{key}.png')
    
    step_names = ['original', 'laplacian', 'sharpened', 'sobel_gradient', 'sobel_smoothed', 'multiplied', 'added_to_original', 'final']
    
    images = []
    titles = []
    title_mapping = {
        'original': 'Step 0: Original',
        'laplacian': 'Step 1: Laplacian',
        'sharpened': 'Step 2: Sharpened',
        'sobel_gradient': 'Step 3: Sobel Gradient',
        'sobel_smoothed': 'Step 4: Smoothed Gradient',
        'multiplied': 'Step 5: Laplacian x Gradient',
        'added_to_original': 'Step 6: Original + Step 5',
        'final': 'Step 7: Final (y=0.5)'
    }
    
    for name in step_names:
        if name in steps:
            img_step = steps[name]
            if img_step.dtype != np.uint8:
                img_step = normalize_for_display(img_step)
            images.append(img_step)
            titles.append(title_mapping[name])
    
    show_multiple_images(images, titles, (2, 4), figsize=(20, 10),
                        suptitle='Mixed Spatial Enhancement - All Steps',
                        save_name='task10_all_steps.png')
    
    show_comparison(img, steps['final'], 
                   "Original", "Final Enhanced Result",
                   save_name='task10_final_comparison.png')
    
    print("✓ Task 10 completed")


def main():
    print("\n" + "="*60)
    print("Intensity Transformations and Spatial Filtering")
    print("="*60)
    
    print(f"\nResults will be saved to: {RESULTS_DIR}")
    
    tasks = [
        task_1_image_negatives,
        task_2_log_transformation,
        task_3_gamma_correction,
        task_4_bit_plane_slicing,
        task_5_histogram_equalization,
        task_6_local_histogram_processing,
        task_7_smoothing_filters,
        task_8_laplacian_sharpening,
        task_9_unsharp_masking,
        task_10_mixed_spatial_enhancement
    ]
    
    for task in tasks:
        try:
            task()
        except Exception as e:
            print(f"\n✗ Error in {task.__name__}: {str(e)}")
            continue
    
    print("\n" + "="*60)
    print("ALL TASKS COMPLETED!")
    print(f"Check the '{RESULTS_DIR.name}' folder for all results.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
