import os
import sys
import csv
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from scipy import ndimage
from scipy.interpolate import griddata, RBFInterpolator
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from sklearn.cluster import KMeans
import random

# Add the current directory to the path to import the data modules
sys.path.append('.')
from data.dataset import ArlFeAl2O3EdxDataset

# Set fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_edx_dataset(data_dir='./data', split='train'):
    """Load the EDX dataset without transforms for analysis."""
    dataset = ArlFeAl2O3EdxDataset(
        root=data_dir,
        split=split,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        discriminative=True,
        transform=None,  # No transforms to keep PIL images
        seed=SEED
    )
    return dataset

def analyze_spatial_sparsity(image_array):
    """
    Analyze spatial sparsity (zero pixels) in an image.
    
    Parameters:
    -----------
    image_array : np.ndarray
        2D image array
        
    Returns:
    --------
    dict
        Dictionary containing sparsity metrics
    """
    total_pixels = image_array.size
    zero_pixels = np.sum(image_array == 0)
    nonzero_pixels = total_pixels - zero_pixels
    
    # Calculate sparsity ratio
    sparsity_ratio = zero_pixels / total_pixels
    
    # Calculate largest connected component of zeros
    zero_mask = (image_array == 0)
    labeled_zeros, num_components = ndimage.label(zero_mask)
    if num_components > 0:
        component_sizes = np.bincount(labeled_zeros.ravel())
        if len(component_sizes) > 1:  # Exclude background (0)
            component_sizes = component_sizes[1:]
            largest_zero_component = np.max(component_sizes) if len(component_sizes) > 0 else 0
        else:
            largest_zero_component = 0
    else:
        largest_zero_component = 0
    
    # Calculate largest connected component of non-zeros
    nonzero_mask = (image_array > 0)
    labeled_nonzeros, num_nonzero_components = ndimage.label(nonzero_mask)
    if num_nonzero_components > 0:
        nonzero_component_sizes = np.bincount(labeled_nonzeros.ravel())
        if len(nonzero_component_sizes) > 1:
            nonzero_component_sizes = nonzero_component_sizes[1:]
            largest_nonzero_component = np.max(nonzero_component_sizes) if len(nonzero_component_sizes) > 0 else 0
        else:
            largest_nonzero_component = 0
    else:
        largest_nonzero_component = 0
    
    return {
        'total_pixels': total_pixels,
        'zero_pixels': zero_pixels,
        'nonzero_pixels': nonzero_pixels,
        'sparsity_ratio': sparsity_ratio,
        'density_ratio': 1 - sparsity_ratio,
        'num_zero_components': num_components,
        'largest_zero_component': largest_zero_component,
        'num_nonzero_components': num_nonzero_components,
        'largest_nonzero_component': largest_nonzero_component
    }

def analyze_discrete_values(image_array):
    """
    Analyze discrete values in an image (which values from 0-255 are used).
    
    Parameters:
    -----------
    image_array : np.ndarray
        2D image array
        
    Returns:
    --------
    dict
        Dictionary containing discrete value metrics
    """
    unique_values = np.unique(image_array)
    num_unique_values = len(unique_values)
    value_range = np.max(image_array) - np.min(image_array)
    
    # Calculate value distribution
    value_counts = np.bincount(image_array.ravel(), minlength=256)
    value_distribution = value_counts / np.sum(value_counts)
    
    # Calculate entropy
    nonzero_probs = value_distribution[value_distribution > 0]
    entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
    
    # Calculate coverage (how much of 0-255 range is used)
    coverage_ratio = num_unique_values / 256
    
    return {
        'unique_values': unique_values,
        'num_unique_values': num_unique_values,
        'min_value': np.min(image_array),
        'max_value': np.max(image_array),
        'value_range': value_range,
        'mean_value': np.mean(image_array),
        'std_value': np.std(image_array),
        'entropy': entropy,
        'coverage_ratio': coverage_ratio,
        'value_distribution': value_distribution
    }

# Density map calculation functions
def gaussian_density_map(image_array, sigma=2.0):
    """Calculate density map using Gaussian filter."""
    normalized = image_array.astype(np.float64)
    if normalized.max() - normalized.min() > 0:
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    density_map = ndimage.gaussian_filter(normalized, sigma=sigma)
    return density_map

def morphological_density_map(image_array, kernel_size=5):
    """Calculate density map using morphological operations."""
    normalized = image_array.astype(np.float64)
    if normalized.max() - normalized.min() > 0:
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    
    # Create circular kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    y, x = np.ogrid[:kernel_size, :kernel_size]
    mask = (x - center) ** 2 + (y - center) ** 2 <= center ** 2
    kernel[mask] = 1
    
    # Calculate local density using convolution
    density_map = ndimage.convolve(normalized, kernel, mode='constant') / np.sum(kernel)
    return density_map

def kernel_density_map(image_array, bandwidth=0.1, subsample_factor=4):
    """Calculate density map using kernel density estimation with subsampling for efficiency."""
    normalized = image_array.astype(np.float64)
    if normalized.max() - normalized.min() > 0:
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    
    h, w = normalized.shape
    
    # Subsample for efficiency
    step = max(1, min(h, w) // 32)  # Adjust based on image size
    y_coords, x_coords = np.mgrid[0:h:step, 0:w:step]
    
    # Create sample points weighted by pixel intensity
    sample_points = []
    weights = []
    
    for i in range(0, h, step):
        for j in range(0, w, step):
            if normalized[i, j] > 0.1:  # Only consider pixels above threshold
                sample_points.append([i, j])
                weights.append(normalized[i, j])
    
    if len(sample_points) == 0:
        return np.zeros_like(normalized)
    
    sample_points = np.array(sample_points)
    weights = np.array(weights)
    
    # Use a simpler approach for large images
    if len(sample_points) > 1000:
        # Use simple distance-based weighting instead of full KDE
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        density_map = np.zeros((h, w))
        
        for i, (sy, sx) in enumerate(sample_points):
            # Calculate distance-based contribution
            dist = np.sqrt((y_grid - sy)**2 + (x_grid - sx)**2)
            contribution = weights[i] * np.exp(-dist**2 / (2 * bandwidth**2 * 100))
            density_map += contribution
    else:
        # Use full KDE for smaller datasets
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(sample_points, sample_weight=weights)
        
        # Evaluate KDE on a grid
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        grid_points = np.column_stack([y_grid.ravel(), x_grid.ravel()])
        density_scores = np.exp(kde.score_samples(grid_points))
        density_map = density_scores.reshape(h, w)
    
    return density_map

# Resampling methods
def interpolation_resample(image_array, method='cubic', noise_factor=0.02):
    """
    Resample using interpolation methods.
    
    Parameters:
    -----------
    image_array : np.ndarray
        2D image array
    method : str
        Interpolation method ('linear', 'cubic', 'nearest')
    noise_factor : float
        Amount of noise to add for augmentation
    """
    h, w = image_array.shape
    
    # Get non-zero pixel coordinates and values
    y_coords, x_coords = np.where(image_array > 0)
    values = image_array[y_coords, x_coords]
    
    if len(values) == 0:
        return image_array.copy()
    
    # Add small random perturbations to coordinates for augmentation
    y_coords_perturbed = y_coords + np.random.normal(0, 0.5, len(y_coords))
    x_coords_perturbed = x_coords + np.random.normal(0, 0.5, len(x_coords))
    
    # Create target grid
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    points = np.column_stack([y_coords_perturbed, x_coords_perturbed])
    
    # Interpolate
    try:
        resampled = griddata(points, values, (y_grid, x_grid), method=method, fill_value=0)
        resampled = np.nan_to_num(resampled, nan=0.0)
    except:
        # Fallback to nearest neighbor if method fails
        resampled = griddata(points, values, (y_grid, x_grid), method='nearest', fill_value=0)
        resampled = np.nan_to_num(resampled, nan=0.0)
    
    # Add noise for augmentation
    if noise_factor > 0:
        noise = np.random.normal(0, noise_factor * np.std(values), resampled.shape)
        resampled = np.maximum(0, resampled + noise)
    
    return resampled.astype(image_array.dtype)

def morphological_resample(image_array, operation='closing', kernel_size=3):
    """
    Resample using morphological operations.
    
    Parameters:
    -----------
    image_array : np.ndarray
        2D image array
    operation : str
        Morphological operation ('closing', 'opening', 'dilation', 'erosion')
    kernel_size : int
        Size of the morphological kernel
    """
    # Create circular kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    y, x = np.ogrid[:kernel_size, :kernel_size]
    mask = (x - center) ** 2 + (y - center) ** 2 <= center ** 2
    kernel[mask] = 1
    
    # Apply morphological operation
    if operation == 'closing':
        resampled = ndimage.binary_closing(image_array > 0, structure=kernel).astype(float)
        resampled = resampled * np.mean(image_array[image_array > 0]) if np.any(image_array > 0) else resampled
    elif operation == 'opening':
        resampled = ndimage.binary_opening(image_array > 0, structure=kernel).astype(float)
        resampled = resampled * np.mean(image_array[image_array > 0]) if np.any(image_array > 0) else resampled
    elif operation == 'dilation':
        resampled = ndimage.binary_dilation(image_array > 0, structure=kernel).astype(float)
        resampled = resampled * np.mean(image_array[image_array > 0]) if np.any(image_array > 0) else resampled
    elif operation == 'erosion':
        resampled = ndimage.binary_erosion(image_array > 0, structure=kernel).astype(float)
        resampled = resampled * np.mean(image_array[image_array > 0]) if np.any(image_array > 0) else resampled
    else:
        resampled = image_array.copy()
    
    return resampled.astype(image_array.dtype)

def clustering_resample(image_array, n_clusters=8, redistribution='random'):
    """
    Resample using clustering and redistribution.
    
    Parameters:
    -----------
    image_array : np.ndarray
        2D image array
    n_clusters : int
        Number of clusters for K-means
    redistribution : str
        How to redistribute cluster centers ('random', 'smooth')
    """
    if np.all(image_array == 0):
        return image_array.copy()
    
    h, w = image_array.shape
    
    # Get non-zero pixel coordinates and values
    nonzero_mask = image_array > 0
    if not np.any(nonzero_mask):
        return image_array.copy()
    
    y_coords, x_coords = np.where(nonzero_mask)
    values = image_array[nonzero_mask]
    
    # Create feature vectors (position + intensity)
    features = np.column_stack([y_coords, x_coords, values])
    
    # Perform clustering
    n_clusters = min(n_clusters, len(features))
    if n_clusters < 2:
        return image_array.copy()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init='auto')
    cluster_labels = kmeans.fit_predict(features)
    
    # Create resampled image
    resampled = np.zeros_like(image_array, dtype=float)
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        if not np.any(cluster_mask):
            continue
        
        cluster_points = features[cluster_mask]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        
        if redistribution == 'random':
            # Randomly redistribute points around cluster center
            center_y, center_x = cluster_center[0], cluster_center[1]
            spread = np.std(cluster_points[:, :2], axis=0) if len(cluster_points) > 1 else [2, 2]
            
            for point in cluster_points:
                new_y = int(np.clip(center_y + np.random.normal(0, spread[0]), 0, h-1))
                new_x = int(np.clip(center_x + np.random.normal(0, spread[1]), 0, w-1))
                resampled[new_y, new_x] = point[2]  # Use original intensity
        else:
            # Smooth redistribution
            for point in cluster_points:
                y, x, val = point
                resampled[int(y), int(x)] = val
    
    return resampled.astype(image_array.dtype)

def elastic_deformation_resample(image_array, sigma=4, alpha=34):
    """
    Resample using elastic deformation.
    
    Parameters:
    -----------
    image_array : np.ndarray
        2D image array
    sigma : float
        Standard deviation for Gaussian filter
    alpha : float
        Scaling factor for deformation
    """
    h, w = image_array.shape
    
    # Generate random displacement fields
    dx = np.random.randn(h, w) * alpha
    dy = np.random.randn(h, w) * alpha
    
    # Smooth the displacement fields
    dx = ndimage.gaussian_filter(dx, sigma, mode='constant', cval=0)
    dy = ndimage.gaussian_filter(dy, sigma, mode='constant', cval=0)
    
    # Create coordinate arrays
    y, x = np.mgrid[0:h, 0:w]
    
    # Apply deformation
    y_new = np.clip(y + dy, 0, h-1)
    x_new = np.clip(x + dx, 0, w-1)
    
    # Map coordinates
    resampled = ndimage.map_coordinates(image_array, [y_new, x_new], order=1, mode='constant', cval=0)
    
    return resampled.astype(image_array.dtype)

def statistical_resample(image_array, method='poisson'):
    """
    Resample using statistical noise models.
    
    Parameters:
    -----------
    image_array : np.ndarray
        2D image array
    method : str
        Statistical method ('poisson', 'gaussian', 'gamma')
    """
    if method == 'poisson':
        # Add Poisson noise
        resampled = np.random.poisson(image_array).astype(image_array.dtype)
    elif method == 'gaussian':
        # Add Gaussian noise
        noise_std = np.std(image_array[image_array > 0]) * 0.1 if np.any(image_array > 0) else 0
        noise = np.random.normal(0, noise_std, image_array.shape)
        resampled = np.maximum(0, image_array + noise).astype(image_array.dtype)
    elif method == 'gamma':
        # Apply gamma correction with random gamma
        gamma = np.random.uniform(0.7, 1.5)
        normalized = image_array.astype(float) / 255.0
        resampled = (255 * np.power(normalized, gamma)).astype(image_array.dtype)
    else:
        resampled = image_array.copy()
    
    return resampled

def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two images."""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index between two images."""
    # Ensure images are in the right range for SSIM
    img1_norm = img1.astype(float)
    img2_norm = img2.astype(float)
    
    # Normalize to 0-1 range
    if img1_norm.max() > 1:
        img1_norm = img1_norm / 255.0
    if img2_norm.max() > 1:
        img2_norm = img2_norm / 255.0
    
    try:
        return ssim(img1_norm, img2_norm, data_range=1.0)
    except:
        return 0.0

def save_image(image_array, filepath):
    """Save image array as PNG file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if image_array.dtype != np.uint8:
        # Normalize to 0-255 range
        if image_array.max() > 0:
            image_normalized = (255 * image_array / image_array.max()).astype(np.uint8)
        else:
            image_normalized = image_array.astype(np.uint8)
    else:
        image_normalized = image_array
    
    Image.fromarray(image_normalized).save(filepath)

def process_sample(dataset, image_idx, output_dir='data-report'):
    """
    Process a single sample from the dataset with comprehensive analysis and resampling.
    
    Parameters:
    -----------
    dataset : ArlFeAl2O3EdxDataset
        The dataset to process
    image_idx : int
        Index of the sample to process
    output_dir : str
        Output directory for results
    
    Returns:
    --------
    list
        List of dictionaries containing analysis results for each resampling method
    """
    # Get the sample
    bse_img, edx_img, label = dataset[image_idx]
    
    # Convert to numpy arrays
    bse_array = np.array(bse_img)
    edx_array = np.array(edx_img)
    
    # Extract individual EDX channels
    channels = {
        'Fe': edx_array[:, :, 0],
        'Al': edx_array[:, :, 1],
        'O': edx_array[:, :, 2]
    }
    
    # Create output directory for this sample
    sample_dir = os.path.join(output_dir, str(image_idx))
    os.makedirs(sample_dir, exist_ok=True)
    
    # Define resampling methods and their parameters
    resampling_methods = [
        ('interpolation_linear', lambda img: interpolation_resample(img, method='linear')),
        ('interpolation_cubic', lambda img: interpolation_resample(img, method='cubic')),
        ('interpolation_nearest', lambda img: interpolation_resample(img, method='nearest')),
        ('morphological_closing', lambda img: morphological_resample(img, operation='closing', kernel_size=3)),
        ('morphological_opening', lambda img: morphological_resample(img, operation='opening', kernel_size=3)),
        ('morphological_dilation', lambda img: morphological_resample(img, operation='dilation', kernel_size=3)),
        ('clustering_random', lambda img: clustering_resample(img, n_clusters=8, redistribution='random')),
        ('clustering_smooth', lambda img: clustering_resample(img, n_clusters=8, redistribution='smooth')),
        ('elastic_deformation', lambda img: elastic_deformation_resample(img, sigma=4, alpha=34)),
        ('statistical_poisson', lambda img: statistical_resample(img, method='poisson')),
        ('statistical_gaussian', lambda img: statistical_resample(img, method='gaussian')),
        ('statistical_gamma', lambda img: statistical_resample(img, method='gamma'))
    ]
    
    # Define density methods
    density_methods = [
        ('gaussian_sigma1', lambda img: gaussian_density_map(img, sigma=1.0)),
        ('gaussian_sigma2', lambda img: gaussian_density_map(img, sigma=2.0)),
        ('gaussian_sigma4', lambda img: gaussian_density_map(img, sigma=4.0)),
        ('morphological_k3', lambda img: morphological_density_map(img, kernel_size=3)),
        ('morphological_k5', lambda img: morphological_density_map(img, kernel_size=5)),
        # ('kernel_bw01', lambda img: kernel_density_map(img, bandwidth=0.1))
    ]
    
    results = []
    
    for channel_name, channel_array in channels.items():
        # Analyze original image
        sparsity_analysis = analyze_spatial_sparsity(channel_array)
        discrete_analysis = analyze_discrete_values(channel_array)
        
        # Save original image
        save_image(channel_array, os.path.join(sample_dir, f'original-{channel_name}.png'))
        
        # Calculate and save original density maps
        for density_name, density_func in density_methods:
            original_density = density_func(channel_array)
            density_filepath = os.path.join(sample_dir, f'density-original-{channel_name}-{density_name}.png')
            save_image(original_density, density_filepath)
        
        # Process each resampling method
        for resample_name, resample_func in resampling_methods:
            # Apply resampling
            resampled_array = resample_func(channel_array)
            
            # Analyze resampled image
            resampled_sparsity = analyze_spatial_sparsity(resampled_array)
            resampled_discrete = analyze_discrete_values(resampled_array)
            
            # Save resampled image
            resampled_filepath = os.path.join(sample_dir, f'resampled-{channel_name}-{resample_name}.png')
            save_image(resampled_array, resampled_filepath)
            
            # Calculate and save resampled density maps, compare with original
            density_results = {}
            for density_name, density_func in density_methods:
                # print(f"Running channel {channel_name} with resample method {resample_name} and density method {density_name}", end="\r", flush=True)
                # Calculate density maps
                original_density = density_func(channel_array)
                resampled_density = density_func(resampled_array)
                
                # Save resampled density map
                density_filepath = os.path.join(sample_dir, f'density-resampled-{channel_name}-{resample_name}-{density_name}.png')
                save_image(resampled_density, density_filepath)
                
                # Calculate comparison metrics
                mse_score = calculate_mse(original_density, resampled_density)
                ssim_score = calculate_ssim(original_density, resampled_density)
                
                density_results[f'{density_name}_mse'] = mse_score
                density_results[f'{density_name}_ssim'] = ssim_score
            
            # Compile results for this resampling method
            result = {
                'image_idx': image_idx,
                'channel': channel_name,
                'resample_method': resample_name,
                'fe_percent': label[0],
                'laser_speed': label[1],
                
                # Original image analysis
                'orig_total_pixels': sparsity_analysis['total_pixels'],
                'orig_zero_pixels': sparsity_analysis['zero_pixels'],
                'orig_sparsity_ratio': sparsity_analysis['sparsity_ratio'],
                'orig_num_unique_values': discrete_analysis['num_unique_values'],
                'orig_value_range': discrete_analysis['value_range'],
                'orig_entropy': discrete_analysis['entropy'],
                'orig_coverage_ratio': discrete_analysis['coverage_ratio'],
                'orig_mean_value': discrete_analysis['mean_value'],
                'orig_std_value': discrete_analysis['std_value'],
                
                # Resampled image analysis
                'resampled_total_pixels': resampled_sparsity['total_pixels'],
                'resampled_zero_pixels': resampled_sparsity['zero_pixels'],
                'resampled_sparsity_ratio': resampled_sparsity['sparsity_ratio'],
                'resampled_num_unique_values': resampled_discrete['num_unique_values'],
                'resampled_value_range': resampled_discrete['value_range'],
                'resampled_entropy': resampled_discrete['entropy'],
                'resampled_coverage_ratio': resampled_discrete['coverage_ratio'],
                'resampled_mean_value': resampled_discrete['mean_value'],
                'resampled_std_value': resampled_discrete['std_value'],
                
                # Image comparison metrics
                'pixel_mse': calculate_mse(channel_array, resampled_array),
                'pixel_ssim': calculate_ssim(channel_array, resampled_array),
            }
            
            # Add density comparison results
            result.update(density_results)
            
            results.append(result)
    
    return results

def main():
    """Main function to process all samples in the dataset."""
    # Load dataset
    print("Loading EDX dataset...")
    dataset = load_edx_dataset()
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Create output directory
    output_dir = 'data-report'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all samples
    all_results = []
    
    for image_idx in range(len(dataset)):
        print(f"Processing sample {image_idx + 1}/{len(dataset)}...")
        
        try:
            sample_results = process_sample(dataset, image_idx, output_dir)
            all_results.extend(sample_results)
            
            # Save individual CSV for this sample
            sample_df = pd.DataFrame(sample_results)
            csv_filepath = os.path.join(output_dir, str(image_idx), f'analysis_report_{image_idx}.csv')
            sample_df.to_csv(csv_filepath, index=False)
            
        except Exception as e:
            print(f"Error processing sample {image_idx}: {e}")
            continue
    
    # Save comprehensive CSV report
    if all_results:
        print("Saving comprehensive analysis report...")
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(output_dir, 'comprehensive_analysis_report.csv'), index=False)
        
        # Print summary statistics
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Total samples processed: {len(dataset)}")
        print(f"Total resampling experiments: {len(all_results)}")
        print(f"Channels analyzed: {df['channel'].unique()}")
        print(f"Resampling methods: {df['resample_method'].unique()}")
        
        # Print average metrics by method
        print("\n=== AVERAGE METRICS BY RESAMPLING METHOD ===")
        method_summary = df.groupby('resample_method').agg({
            'pixel_mse': 'mean',
            'pixel_ssim': 'mean',
            'orig_sparsity_ratio': 'mean',
            'resampled_sparsity_ratio': 'mean'
        }).round(4)
        print(method_summary)
        
    print(f"\nAnalysis complete! Results saved in '{output_dir}' directory.")
    print("Check individual sample directories for detailed images and analysis.")

if __name__ == "__main__":
    main()
