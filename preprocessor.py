"""
Data Preprocessing Module
Handles complete preprocessing pipeline for XCO₂ reconstruction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import warnings

from wavelet_transform import generate_wavelet_spectrogram, batch_generate_spectrograms


def load_xco2_observations(filepath: str) -> pd.DataFrame:
    """
    Load OCO-2/3 XCO₂ observations.
    
    Args:
        filepath: Path to CSV file with columns: lat, lon, xco2, date
        
    Returns:
        DataFrame with XCO₂ observations
    """
    df = pd.read_csv(filepath)
    
    required_cols = ['lat', 'lon', 'xco2']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Data must contain columns: {required_cols}")
    
    # Filter quality
    if 'quality_flag' in df.columns:
        df = df[df['quality_flag'] == 0]
    
    # Remove NaN values
    df = df.dropna(subset=required_cols)
    
    print(f"Loaded {len(df)} XCO₂ observations")
    return df


def create_grid(lat_range: Tuple[float, float], 
                lon_range: Tuple[float, float],
                resolution: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create regular lat-lon grid.
    
    Returns:
        lats, lons, coordinates (n_points, 2)
    """
    lats = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lons = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    coordinates = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    
    return lats, lons, coordinates


def assign_to_grid(df: pd.DataFrame, 
                   coordinates: np.ndarray,
                   resolution: float = 0.25) -> np.ndarray:
    """
    Assign observations to nearest grid cells.
    
    Returns:
        Array of XCO₂ values for each grid cell (with NaN for missing)
    """
    n_cells = len(coordinates)
    xco2_grid = np.full(n_cells, np.nan)
    
    for idx, (lat, lon) in enumerate(coordinates):
        # Find observations within grid cell
        mask = (np.abs(df['lat'] - lat) <= resolution/2) & \
               (np.abs(df['lon'] - lon) <= resolution/2)
        
        if mask.sum() > 0:
            # Take mean of observations in cell
            xco2_grid[idx] = df.loc[mask, 'xco2'].mean()
    
    return xco2_grid


def preprocess_xco2_data(
    xco2_filepath: str,
    auxiliary_data: Dict,
    lat_range: Tuple[float, float] = (41.0, 60.0),
    lon_range: Tuple[float, float] = (-141.0, -52.0),
    resolution: float = 0.25,
    wavelet: str = 'morl',
    scales: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline.
    
    Returns:
        spectrograms: (n_points, scales, scales)
        features: (n_points, n_features)
        targets: (n_points,)
        coordinates: (n_points, 2)
        has_data: (n_points,) boolean mask
    """
    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Load observations
    df = load_xco2_observations(xco2_filepath)
    
    # Create grid
    lats, lons, coordinates = create_grid(lat_range, lon_range, resolution)
    n_points = len(coordinates)
    print(f"Grid: {len(lats)} × {len(lons)} = {n_points} cells")
    
    # Assign to grid
    xco2_grid = assign_to_grid(df, coordinates, resolution)
    has_data = ~np.isnan(xco2_grid)
    print(f"Valid cells: {has_data.sum()} ({100*has_data.sum()/n_points:.1f}%)")
    
    # Generate spectrograms
    print("\nGenerating spectrograms...")
    # Placeholder: In real implementation, extract temporal sequences
    spectrograms = np.zeros((n_points, scales, scales))
    
    for i in range(n_points):
        if has_data[i]:
            # Create synthetic time series (replace with real temporal data)
            ts = xco2_grid[i] + np.random.normal(0, 0.5, scales)
            spec = generate_wavelet_spectrogram(ts, wavelet, scales)
            spectrograms[i] = spec[:scales, :scales]  # Ensure square
    
    print(f"Generated {n_points} spectrograms")
    
    # Extract auxiliary features (placeholder)
    print("\nExtracting auxiliary features...")
    n_features = 31
    features = np.random.randn(n_points, n_features)  # Replace with real GEE data
    
    # Prepare targets
    targets = np.nan_to_num(xco2_grid, nan=0.0)
    
    print(f"\nPreprocessing complete!")
    print(f"  Spectrograms: {spectrograms.shape}")
    print(f"  Features: {features.shape}")
    print(f"  Targets: {targets.shape}")
    
    return spectrograms, features, targets, coordinates, has_data


def normalize_features(features: np.ndarray, 
                      targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """
    Normalize features and targets using StandardScaler.
    
    Returns:
        normalized_features, normalized_targets, scaler_features, scaler_targets
    """
    scaler_features = StandardScaler()
    scaler_targets = StandardScaler()
    
    features_scaled = scaler_features.fit_transform(features)
    targets_scaled = scaler_targets.fit_transform(targets.reshape(-1, 1)).flatten()
    
    return features_scaled, targets_scaled, scaler_features, scaler_targets
