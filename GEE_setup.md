# Google Earth Engine Setup Guide

Complete guide for setting up and using Google Earth Engine with the ST-ViWT project.

## Prerequisites

- Google account
- Python 3.8 or higher

## Step 1: Create GEE Account

1. Visit: https://earthengine.google.com/signup/
2. Sign in with your Google account
3. Fill out the registration form
4. Select your usage type:
   - **Research/Education**: For academic use
   - **Commercial**: For commercial applications
5. Wait for approval email (usually within 24-48 hours)

## Step 2: Install Earth Engine Python API

```bash
pip install earthengine-api geemap
```

## Step 3: Authenticate

### Option A: Command Line Authentication

```bash
earthengine authenticate
```

This will:
1. Open a browser window
2. Ask you to sign in to Google
3. Request permission to access Earth Engine
4. Provide an authorization code
5. Paste the code back in terminal

### Option B: Python Authentication

```python
import ee

# First time setup
ee.Authenticate()

# After authentication, initialize
ee.Initialize(project='your-project-id')
```

## Step 4: Verify Installation

```python
import ee

# Initialize
ee.Initialize()

# Test with simple query
image = ee.Image('USGS/SRTMGL1_003')
print('Earth Engine initialized successfully!')
```

## Step 5: Project Setup

If using a Cloud Project:

```python
import ee

# Initialize with project
ee.Initialize(project='your-cloud-project-id')
```


## Common Issues and Solutions

### Issue 1: Authentication Failed

**Solution**:
```bash
# Clear existing credentials
rm -rf ~/.config/earthengine

# Re-authenticate
earthengine authenticate
```

### Issue 2: Quota Exceeded

**Solutions**:
- Wait for quota reset (daily limits)
- Optimize queries to reduce computation
- Consider upgrading to commercial license

### Issue 3: Import Error

**Solution**:
```bash
# Reinstall dependencies
pip uninstall earthengine-api
pip install earthengine-api --upgrade
```

## Best Practices

### 1. Efficient Data Access

```python
# Use filters to reduce data
collection = ee.ImageCollection('MODIS/006/MOD13A1') \
    .filterDate('2024-01-01', '2024-12-31') \
    .filterBounds(region) \
    .select(['NDVI'])
```

### 2. Batch Processing

```python
# Process multiple regions
def process_region(region):
    data = extract_data(region)
    return data

regions = [region1, region2, region3]
results = [process_region(r) for r in regions]
```

### 3. Export Large Results

```python
# Export to Google Drive
task = ee.batch.Export.image.toDrive(
    image=image,
    description='export_task',
    folder='GEE_Exports',
    scale=1000,
    region=region
)
task.start()
```

## Resources

### Documentation
- **Earth Engine Guides**: https://developers.google.com/earth-engine/guides
- **API Reference**: https://developers.google.com/earth-engine/apidocs
- **Code Editor**: https://code.earthengine.google.com/

### Support
- **Forum**: https://groups.google.com/forum/#!forum/google-earth-engine-developers
- **Stack Overflow**: Tag `google-earth-engine`
- **GitHub Issues**: https://github.com/google/earthengine-api/issues

## Datasets Used in ST-ViWT

| Dataset | Collection ID | Documentation |
|---------|---------------|---------------|
| ERA5-Land | `ECMWF/ERA5_LAND/DAILY_AGGR` | [Link](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR) |
| MODIS Vegetation | `MODIS/061/MOD13A1` | [Link](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13A1) |
| SRTM | `USGS/SRTMGL1_003` | [Link](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003) |
| ESA WorldCover | `ESA/WorldCover/v200` | [Link](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200) |
| AAFC Crop | `AAFC/ACI` | [Link](https://developers.google.com/earth-engine/datasets/catalog/AAFC_ACI) |

## Tips for Canada-Specific Data

### 1. Regional Filtering

```python
# Canada bounding box
canada_bbox = ee.Geometry.Rectangle([-141.0, 41.0, -52.0, 60.0])

# Filter to Canada
data = collection.filterBounds(canada_bbox)
```

### 2. Handling High Latitudes

```python
# Account for lower data quality at high latitudes
# Apply quality filters
def quality_filter(image):
    qa = image.select('quality')
    return image.updateMask(qa.eq(0))

filtered = collection.map(quality_filter)
```

### 3. Seasonal Considerations

```python
# Summer months (better data quality)
summer = collection.filter(ee.Filter.calendarRange(6, 8, 'month'))
```

## Quota Information

- **Free Tier**: 
  - 10,000 requests per day
  - 250,000 requests per month
  - 1TB storage in Assets

- **Commercial Tier**: Higher limits, contact sales
---
**Last Updated**: October 24, 2025  
**Version**: 1.0
