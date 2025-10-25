"""
Google Earth Engine Auxiliary Data Extractor
Extracts geospatial datasets for XCO₂ reconstruction

Supports:
- ERA5-Land: Meteorological data
- MODIS MOD13A1: Vegetation indices
- SRTM: Topography
- ESA WorldCover: Land cover
- AAFC: Crop inventory
"""

import ee
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime


class GEEAuxiliaryDataExtractor:
    """
    Extract auxiliary geospatial datasets from Google Earth Engine.
    Handles temporal aggregation and spatial sampling for XCO₂ reconstruction.
    
    Args:
        region: ee.Geometry defining study area
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        resolution: Spatial resolution in degrees (default: 0.25)
    """
    
    def __init__(self, region: ee.Geometry, start_date: str, 
                 end_date: str, resolution: float = 0.25):
        self.region = region
        self.start_date = start_date
        self.end_date = end_date
        self.resolution = resolution
        self.auxiliary_data = {}
        
        # Verify GEE initialization
        try:
            ee.Initialize()
        except Exception as e:
            raise RuntimeError(f"Google Earth Engine not initialized: {e}")
    
    def extract_era5_data(self) -> Optional[ee.Image]:
        """
        Extract ERA5-Land Daily Aggregated meteorological data.
        
        Returns:
            ee.Image with bands: temperature_2m, total_precipitation_sum,
            u_component_of_wind_10m, v_component_of_wind_10m, surface_pressure,
            dewpoint_temperature_2m
        """
        print("\n[1/5] Extracting ERA5-Land meteorological data...")
        
        try:
            era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
                .filterDate(self.start_date, self.end_date) \
                .filterBounds(self.region) \
                .select([
                    'temperature_2m',
                    'total_precipitation_sum',
                    'u_component_of_wind_10m',
                    'v_component_of_wind_10m',
                    'surface_pressure',
                    'dewpoint_temperature_2m'
                ])
            
            # Temporal mean (annual average)
            era5_mean = era5.mean().clip(self.region)
            
            self.auxiliary_data['era5'] = era5_mean
            print("  ✓ Temperature, Precipitation, Wind, Pressure, Dewpoint")
            return era5_mean
            
        except Exception as e:
            print(f"  ✗ Error extracting ERA5 data: {e}")
            return None
    
    def extract_modis_vegetation(self) -> Optional[ee.Image]:
        """
        Extract MODIS MOD13A1 vegetation indices.
        
        Returns:
            ee.Image with bands: NDVI, EVI
        """
        print("\n[2/5] Extracting MODIS vegetation indices...")
        
        try:
            modis = ee.ImageCollection('MODIS/061/MOD13A1') \
                .filterDate(self.start_date, self.end_date) \
                .filterBounds(self.region) \
                .select(['NDVI', 'EVI'])
            
            # Scale factors (MODIS values are scaled by 10000)
            def scale_modis(image):
                return image.multiply(0.0001)
            
            # Temporal mean
            modis_mean = modis.map(scale_modis).mean().clip(self.region)
            
            self.auxiliary_data['modis'] = modis_mean
            print("  ✓ NDVI, EVI")
            return modis_mean
            
        except Exception as e:
            print(f"  ✗ Error extracting MODIS data: {e}")
            return None
    
    def extract_srtm_topography(self) -> Optional[ee.Image]:
        """
        Extract SRTM elevation and derived slope.
        
        Returns:
            ee.Image with bands: elevation, slope
        """
        print("\n[3/5] Extracting SRTM topography...")
        
        try:
            srtm = ee.Image('USGS/SRTMGL1_003').select('elevation')
            slope = ee.Terrain.slope(srtm)
            
            topo = srtm.addBands(slope.rename('slope')).clip(self.region)
            
            self.auxiliary_data['srtm'] = topo
            print("  ✓ Elevation, Slope")
            return topo
            
        except Exception as e:
            print(f"  ✗ Error extracting SRTM data: {e}")
            return None
    
    def extract_esa_landcover(self) -> Optional[ee.Image]:
        """
        Extract ESA WorldCover land cover classification.
        
        Returns:
            ee.Image with band: landcover
        """
        print("\n[4/5] Extracting ESA WorldCover land cover...")
        
        try:
            # Use 2021 data (most recent available)
            esa = ee.ImageCollection('ESA/WorldCover/v200') \
                .first() \
                .select('Map') \
                .rename('landcover') \
                .clip(self.region)
            
            self.auxiliary_data['esa'] = esa
            print("  ✓ Land Cover Classification")
            return esa
            
        except Exception as e:
            print(f"  ✗ Error extracting ESA data: {e}")
            return None
    
    def extract_aafc_crop(self) -> Optional[ee.Image]:
        """
        Extract AAFC Annual Crop Inventory for Canada.
        
        Returns:
            ee.Image with band: crop_type
        """
        print("\n[5/5] Extracting AAFC crop inventory...")
        
        try:
            # Use most recent year available
            year = int(self.end_date.split('-')[0])
            
            aafc = ee.ImageCollection('AAFC/ACI') \
                .filterDate(f'{year}-01-01', f'{year}-12-31') \
                .first() \
                .select('landcover') \
                .rename('crop_type') \
                .clip(self.region)
            
            self.auxiliary_data['aafc'] = aafc
            print("  ✓ Crop Type Classification")
            return aafc
            
        except Exception as e:
            print(f"  ✗ Error extracting AAFC data: {e}")
            return None
    
    def combine_all_bands(self) -> Optional[ee.Image]:
        """
        Combine all extracted datasets into a single multi-band image.
        
        Returns:
            ee.Image with all auxiliary bands
        """
        if not self.auxiliary_data:
            print("No auxiliary data extracted. Run extract methods first.")
            return None
        
        # Start with first available dataset
        combined = None
        for key, data in self.auxiliary_data.items():
            if data is not None:
                if combined is None:
                    combined = data
                else:
                    combined = combined.addBands(data)
        
        return combined
    
    def sample_at_points(self, coordinates: np.ndarray) -> pd.DataFrame:
        """
        Sample auxiliary data at specified coordinates.
        
        Args:
            coordinates: Array of shape (n_points, 2) with [lat, lon]
            
        Returns:
            DataFrame with sampled values for each coordinate
        """
        if not self.auxiliary_data:
            raise ValueError("No auxiliary data extracted. Run extract methods first.")
        
        # Combine all bands
        combined = self.combine_all_bands()
        
        # Create feature collection from coordinates
        points = []
        for i, (lat, lon) in enumerate(coordinates):
            point = ee.Feature(ee.Geometry.Point([lon, lat]), {'index': i})
            points.append(point)
        
        fc = ee.FeatureCollection(points)
        
        # Sample at points
        sampled = combined.sampleRegions(
            collection=fc,
            scale=self.resolution * 111000,  # Convert degrees to meters
            geometries=True
        )
        
        # Convert to DataFrame
        data = sampled.getInfo()
        features = data['features']
        
        rows = []
        for feature in features:
            props = feature['properties']
            rows.append(props)
        
        df = pd.DataFrame(rows)
        return df.sort_values('index').reset_index(drop=True)
    
    def create_grid_samples(self, lat_range: Tuple[float, float],
                           lon_range: Tuple[float, float]) -> np.ndarray:
        """
        Create regular grid of coordinates for sampling.
        
        Args:
            lat_range: (min_lat, max_lat)
            lon_range: (min_lon, max_lon)
            
        Returns:
            Array of shape (n_points, 2) with [lat, lon]
        """
        lats = np.arange(lat_range[0], lat_range[1] + self.resolution, self.resolution)
        lons = np.arange(lon_range[0], lon_range[1] + self.resolution, self.resolution)
        
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        coordinates = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        
        return coordinates
    
    def extract_all_data(self) -> Dict[str, ee.Image]:
        """
        Extract all auxiliary datasets.
        
        Returns:
            Dictionary mapping dataset names to ee.Image objects
        """
        print("=" * 60)
        print("Extracting Auxiliary Data from Google Earth Engine")
        print("=" * 60)
        
        self.extract_era5_data()
        self.extract_modis_vegetation()
        self.extract_srtm_topography()
        self.extract_esa_landcover()
        self.extract_aafc_crop()
        
        print("\n" + "=" * 60)
        print(f"Extraction Complete: {len(self.auxiliary_data)}/5 datasets")
        print("=" * 60)
        
        return self.auxiliary_data
    
    def get_band_names(self) -> List[str]:
        """
        Get list of all band names from extracted data.
        
        Returns:
            List of band names
        """
        if not self.auxiliary_data:
            return []
        
        combined = self.combine_all_bands()
        if combined is None:
            return []
        
        return combined.bandNames().getInfo()
    
    def export_to_drive(self, filename: str, folder: str = 'GEE_Exports'):
        """
        Export combined auxiliary data to Google Drive.
        
        Args:
            filename: Output filename (without extension)
            folder: Google Drive folder name
        """
        combined = self.combine_all_bands()
        
        if combined is None:
            print("No data to export.")
            return
        
        task = ee.batch.Export.image.toDrive(
            image=combined,
            description=filename,
            folder=folder,
            region=self.region,
            scale=self.resolution * 111000,  # Convert to meters
            crs='EPSG:4326',
            maxPixels=1e13
        )
        
        task.start()
        print(f"Export task started: {filename}")
        print(f"Monitor at: https://code.earthengine.google.com/tasks")


def example_usage():
    """Example usage of GEE Auxiliary Data Extractor."""
    
    # Initialize Earth Engine
    try:
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()
    
    # Define study area (Canada)
    region = ee.Geometry.Rectangle([-141.0, 41.0, -52.0, 60.0])
    
    # Create extractor
    extractor = GEEAuxiliaryDataExtractor(
        region=region,
        start_date='2024-01-01',
        end_date='2024-12-31',
        resolution=0.25
    )
    
    # Extract all data
    auxiliary_data = extractor.extract_all_data()
    
    # Get band names
    bands = extractor.get_band_names()
    print(f"\nTotal bands extracted: {len(bands)}")
    print(f"Band names: {bands}")
    
    # Sample at specific coordinates
    coords = np.array([
        [45.5, -73.6],  # Montreal
        [43.7, -79.4],  # Toronto
        [51.0, -114.1]  # Calgary
    ])
    
    samples = extractor.sample_at_points(coords)
    print(f"\nSampled data shape: {samples.shape}")
    print(samples.head())


if __name__ == "__main__":
    example_usage()
