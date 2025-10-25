"""
Wavelet Transform Module
Continuous Wavelet Transform (CWT) for XCO₂ time series

Converts temporal XCO₂ sequences into 2D spectrograms using CWT
for multi-scale pattern extraction.
"""

import numpy as np
import pywt
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt


def generate_wavelet_spectrogram(
    time_series: np.ndarray,
    wavelet: str = 'morl',
    scales: int = 64,
    sampling_period: float = 1.0
) -> np.ndarray:
    """
    Generate wavelet spectrogram from time series using CWT.
    
    Args:
        time_series: 1D array of XCO₂ values
        wavelet: Wavelet type (default: 'morl' - Morlet)
        scales: Number of scales for CWT (default: 64)
        sampling_period: Sampling period in days (default: 1.0)
        
    Returns:
        2D array of shape (scales, len(time_series)) containing spectrogram
    """
    # Handle NaN values
    if np.any(np.isnan(time_series)):
        # Interpolate NaNs
        mask = ~np.isnan(time_series)
        if mask.sum() < 2:
            # Not enough valid points, return zeros
            return np.zeros((scales, len(time_series)))
        
        indices = np.arange(len(time_series))
        time_series = np.interp(indices, indices[mask], time_series[mask])
    
    # Generate scales
    scale_range = np.arange(1, scales + 1)
    
    # Compute CWT
    coefficients, frequencies = pywt.cwt(
        time_series,
        scale_range,
        wavelet,
        sampling_period=sampling_period
    )
    
    # Normalize to [0, 1]
    spec_min = coefficients.min()
    spec_max = coefficients.max()
    
    if spec_max > spec_min:
        coefficients = (coefficients - spec_min) / (spec_max - spec_min)
    else:
        coefficients = np.zeros_like(coefficients)
    
    return coefficients


def batch_generate_spectrograms(
    time_series_list: list,
    wavelet: str = 'morl',
    scales: int = 64,
    sampling_period: float = 1.0,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate spectrograms for multiple time series.
    
    Args:
        time_series_list: List of 1D arrays
        wavelet: Wavelet type
        scales: Number of scales
        sampling_period: Sampling period
        show_progress: Whether to print progress
        
    Returns:
        3D array of shape (n_series, scales, time_length)
    """
    spectrograms = []
    
    for i, ts in enumerate(time_series_list):
        if show_progress and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(time_series_list)} spectrograms...")
        
        spec = generate_wavelet_spectrogram(ts, wavelet, scales, sampling_period)
        spectrograms.append(spec)
    
    return np.array(spectrograms)


def pad_or_crop_spectrogram(
    spectrogram: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Pad or crop spectrogram to target size.
    
    Args:
        spectrogram: 2D spectrogram array
        target_size: (height, width) target dimensions
        
    Returns:
        Resized spectrogram
    """
    current_h, current_w = spectrogram.shape
    target_h, target_w = target_size
    
    # Create output array
    output = np.zeros(target_size)
    
    # Calculate copy dimensions
    copy_h = min(current_h, target_h)
    copy_w = min(current_w, target_w)
    
    # Copy data
    output[:copy_h, :copy_w] = spectrogram[:copy_h, :copy_w]
    
    return output


def create_spectrogram_from_grid(
    xco2_grid: np.ndarray,
    coordinates: np.ndarray,
    has_data: np.ndarray,
    window_size: int = 64,
    wavelet: str = 'morl',
    scales: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create spectrograms from spatial grid by extracting temporal windows.
    
    Args:
        xco2_grid: 3D array (time, lat, lon) or 2D array (lat, lon)
        coordinates: Array of (lat, lon) coordinates
        has_data: Boolean mask indicating valid data points
        window_size: Temporal window size for CWT
        wavelet: Wavelet type
        scales: Number of CWT scales
        
    Returns:
        spectrograms: Array of shape (n_points, scales, scales)
        valid_mask: Boolean mask for successfully created spectrograms
    """
    n_points = coordinates.shape[0]
    spectrograms = np.zeros((n_points, scales, scales))
    valid_mask = np.zeros(n_points, dtype=bool)
    
    print(f"Generating {n_points} spectrograms...")
    
    for i in range(n_points):
        if not has_data[i]:
            continue
        
        # For 2D grids, create synthetic time series (placeholder)
        # In real implementation, this would extract actual temporal data
        if xco2_grid.ndim == 2:
            # Create synthetic temporal variation
            base_value = xco2_grid[coordinates[i, 0], coordinates[i, 1]]
            time_series = base_value + np.random.normal(0, 0.5, window_size)
        else:
            # Extract temporal profile from 3D grid
            time_series = xco2_grid[:, coordinates[i, 0], coordinates[i, 1]]
        
        # Generate spectrogram
        spec = generate_wavelet_spectrogram(time_series, wavelet, scales)
        
        # Resize to square
        spec = pad_or_crop_spectrogram(spec, (scales, scales))
        
        spectrograms[i] = spec
        valid_mask[i] = True
        
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{n_points}")
    
    return spectrograms, valid_mask


def visualize_spectrogram(
    spectrogram: np.ndarray,
    title: str = "Wavelet Spectrogram",
    save_path: Optional[str] = None
):
    """
    Visualize a single spectrogram.
    
    Args:
        spectrogram: 2D spectrogram array
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        spectrogram,
        aspect='auto',
        cmap='viridis',
        origin='lower',
        interpolation='bilinear'
    )
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Scale (Frequency)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spectrogram saved to: {save_path}")
    
    plt.show()


def compare_spectrograms(
    spectrograms: list,
    titles: list,
    save_path: Optional[str] = None
):
    """
    Compare multiple spectrograms side by side.
    
    Args:
        spectrograms: List of 2D spectrogram arrays
        titles: List of titles for each spectrogram
        save_path: Optional path to save figure
    """
    n = len(spectrograms)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    
    if n == 1:
        axes = [axes]
    
    for ax, spec, title in zip(axes, spectrograms, titles):
        im = ax.imshow(
            spec,
            aspect='auto',
            cmap='viridis',
            origin='lower',
            interpolation='bilinear'
        )
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Scale', fontsize=10)
        ax.set_title(title, fontsize=12, weight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()


def get_available_wavelets():
    """
    Get list of available wavelet types.
    
    Returns:
        Dictionary of wavelet families and their members
    """
    wavelets = {
        'Continuous': pywt.wavelist(kind='continuous'),
        'Discrete': pywt.wavelist(kind='discrete')
    }
    return wavelets


def recommend_wavelet_scales(
    time_series_length: int,
    target_resolution: int = 64
) -> int:
    """
    Recommend number of scales based on time series length.
    
    Args:
        time_series_length: Length of time series
        target_resolution: Desired spectrogram resolution
        
    Returns:
        Recommended number of scales
    """
    # Rule of thumb: scales should be <= time_series_length
    return min(target_resolution, time_series_length)


if __name__ == "__main__":
    # Example usage
    print("Wavelet Transform Module - Example")
    print("=" * 50)
    
    # Create synthetic XCO₂ time series
    np.random.seed(42)
    time_length = 100
    t = np.linspace(0, 10, time_length)
    
    # Simulate XCO₂ with trend and seasonal variation
    xco2 = 410 + 2 * t + 5 * np.sin(2 * np.pi * t / 2) + np.random.normal(0, 0.5, time_length)
    
    print(f"\nTime series length: {time_length}")
    print(f"XCO₂ range: {xco2.min():.2f} - {xco2.max():.2f} ppm")
    
    # Generate spectrogram
    spectrogram = generate_wavelet_spectrogram(xco2, wavelet='morl', scales=64)
    print(f"Spectrogram shape: {spectrogram.shape}")
    
    # Visualize
    visualize_spectrogram(
        spectrogram,
        title="Example XCO₂ Wavelet Spectrogram (Morlet)"
    )
    
    # Show available wavelets
    available = get_available_wavelets()
    print(f"\nAvailable continuous wavelets: {len(available['Continuous'])}")
    print(f"Examples: {available['Continuous'][:5]}")
