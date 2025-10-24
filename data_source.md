Data Sources Documentation
This document provides comprehensive information about all data sources used in the ST-ViWT XCO₂ reconstruction framework.

Primary Data: Satellite XCO₂ Observations
OCO-2/OCO-3 Satellites
Source: NASA Orbiting Carbon Observatory missions
Variables: Column-averaged dry-air mole fraction of CO₂ (XCO₂)
Temporal Coverage: 2014-present (OCO-2), 2019-present (OCO-3)
Spatial Resolution: ~1.29 km × 2.25 km footprint
Accuracy: ~0.5 ppm (target), ~1.5 ppm (achieved)
Access: NASA GES DISC (https://disc.gsfc.nasa.gov/)

Key Features:

High-precision measurements of atmospheric CO₂
Nadir and glint viewing modes
Quality-filtered retrievals (XCO₂_quality_flag = 0)
Bias-corrected values
Data Product: OCO2_L2_Lite_FP v10r
Reference: O'Dell et al. (2018), Atmospheric Measurement Techniques

Auxiliary Data: Google Earth Engine
All auxiliary datasets are accessed through Google Earth Engine platform for consistent spatial-temporal processing.

1. ERA5-Land Daily Aggregated
GEE Collection: ECMWF/ERA5_LAND/DAILY_AGGR
Provider: European Centre for Medium-Range Weather Forecasts (ECMWF)
Temporal Resolution: Daily
Spatial Resolution: 0.1° (~9 km)
Temporal Coverage: 1950-present

Variables Extracted (6 features):

temperature_2m (K): 2-meter air temperature
total_precipitation_sum (m): Daily total precipitation
u_component_of_wind_10m (m/s): Eastward wind component at 10m
v_component_of_wind_10m (m/s): Northward wind component at 10m
surface_pressure (Pa): Surface atmospheric pressure
dewpoint_temperature_2m (K): 2-meter dewpoint temperature
Processing: Annual mean aggregation for 2024
Reference: Muñoz-Sabater et al. (2021), Earth System Science Data
DOI: 10.5194/essd-13-4349-2021

Relevance to XCO₂:

Temperature affects photosynthesis and respiration rates
Precipitation influences vegetation growth and CO₂ uptake
Wind patterns transport CO₂ across regions
Pressure relates to atmospheric column properties
2. MODIS Vegetation Indices
GEE Collection: MODIS/061/MOD13A1
Provider: NASA LP DAAC
Temporal Resolution: 16-day composite
Spatial Resolution: 500m
Temporal Coverage: 2000-present

Variables Extracted (2 features):

NDVI (Normalized Difference Vegetation Index): -1 to +1
EVI (Enhanced Vegetation Index): -1 to +1
Processing:

Scaled by 0.0001 (original values × 0.0001)
Annual mean aggregation for 2024
Quality-filtered pixels
Reference: Didan, K. (2021), MODIS/Terra Vegetation Indices
DOI: 10.5067/MODIS/MOD13A1.061

Relevance to XCO₂:

Vegetation vigor indicator (photosynthetic activity)
Proxy for carbon uptake by terrestrial ecosystems
Seasonal phenology patterns
NDVI correlates with gross primary productivity (GPP)
3. SRTM Topography
GEE Collection: USGS/SRTMGL1_003
Provider: NASA/USGS
Temporal Resolution: Static (2000)
Spatial Resolution: 30m (~1 arc-second)
Coverage: 60°N to 56°S

Variables Extracted (2 features):

elevation (m): Height above sea level
slope (degrees): Terrain slope derived from elevation
Processing:

Slope computed using ee.Terrain.slope()
Clipped to study region
Reference: NASA JPL (2013), SRTM Global 1 arc-second
DOI: 10.5067/MEaSUREs/SRTM/SRTMGL1.003

Relevance to XCO₂:

Elevation affects atmospheric pressure and column height
Topography influences local meteorology and vegetation
Slope impacts solar radiation receipt
Mountainous regions show distinct CO₂ patterns
4. ESA WorldCover
GEE Collection: ESA/WorldCover/v200
Provider: European Space Agency (ESA)
Temporal Resolution: Annual (2020, 2021)
Spatial Resolution: 10m
Temporal Coverage: 2020-present

Variables Extracted (1 feature):

landcover: Discrete land cover classification (11 classes)
Land Cover Classes:

10: Tree cover
20: Shrubland
30: Grassland
40: Cropland
50: Built-up
60: Bare/sparse vegetation
70: Snow and ice
80: Permanent water bodies
90: Herbaceous wetland
95: Mangroves
100: Moss and lichen
Processing: Most recent year (2021) used
Reference: Zanaga et al. (2021), Remote Sensing
DOI: 10.3390/rs13101951

Relevance to XCO₂:

Different land covers have distinct CO₂ exchange patterns
Forest vs. cropland differences in carbon uptake
Urban areas show elevated CO₂ (fossil fuel emissions)
Wetlands are significant carbon sinks/sources
5. AAFC Annual Crop Inventory
GEE Collection: AAFC/ACI
Provider: Agriculture and Agri-Food Canada
Temporal Resolution: Annual
Spatial Resolution: 30m
Temporal Coverage: 2009-present
Geographic Coverage: Canada only

Variables Extracted (1 feature):

crop_type: Crop classification (50+ classes)
Major Crop Types:

110: Cloud
120: Water
131: Grassland
132: Pasture/Forages
133: Cereals
147: Corn
153: Soybeans
158: Canola/Rapeseed
And 40+ additional crop types
Processing: Current year data (2024)
Reference: Agriculture and Agri-Food Canada (2024)
Access: https://open.canada.ca/data/en/dataset/ba2645d5-4458-414d-b196-6303ac06c1c9

Relevance to XCO₂:

Agricultural management affects CO₂ fluxes
Different crops have varying photosynthetic rates
Tillage practices influence soil carbon
Growing season timing impacts atmospheric CO₂
Data Integration Strategy
Spatial Harmonization
All datasets are resampled/aggregated to 0.25° grid resolution to match OCO-2/3 observation scale.

Temporal Alignment
Static data (SRTM, ESA WorldCover, AAFC): Single representative value per grid cell
Temporal data (ERA5-Land, MODIS): Annual mean for 2024 study period
Feature Engineering
Total of 31 auxiliary features derived from base datasets:

ERA5-Land: 6 features
MODIS: 2 features
SRTM: 2 features
ESA WorldCover: 1 feature (one-hot encoded → 11 features)
AAFC: 1 feature (one-hot encoded → 9 major classes)
Normalization
All features standardized using sklearn.preprocessing.StandardScaler:

z = (x - μ) / σ
where μ = mean, σ = standard deviation

Data Quality and Filtering
OCO-2/3 Quality Control
XCO₂_quality_flag = 0 (good quality)
Warn_level < 10
Cloud-free conditions
Valid retrieval geometry
Auxiliary Data Quality
ERA5-Land: No additional filtering (reanalysis product)
MODIS: Quality flags applied in GEE
SRTM: Void-filled version used
ESA WorldCover: Validated global product (accuracy >74%)
AAFC: Canadian-specific, annually updated
Data Access Instructions
Google Earth Engine Setup
Create GEE Account:

https://earthengine.google.com/signup/
Authenticate Python API:

earthengine authenticate
Initialize in Code:

import ee
ee.Initialize(project='your-project-id')
OCO-2/3 Data Access
Create NASA Earthdata Account:

https://urs.earthdata.nasa.gov/
Access GES DISC:

https://disc.gsfc.nasa.gov/
Download Using wget:

wget --user=YOUR_USERNAME --password=YOUR_PASSWORD \
     https://oco2.gesdisc.eosdis.nasa.gov/data/OCO2_DATA/...
Citation Requirements
When using these datasets, please cite:

ERA5-Land:

Muñoz-Sabater, J., et al. (2021). ERA5-Land: A state-of-the-art global 
reanalysis dataset for land applications. Earth System Science Data, 
13(9), 4349-4383.
MODIS:

Didan, K. (2021). MODIS/Terra Vegetation Indices 16-Day L3 Global 500m 
SIN Grid V061. NASA EOSDIS Land Processes DAAC.
SRTM:

NASA JPL (2013). NASA Shuttle Radar Topography Mission Global 1 arc 
second. NASA EOSDIS Land Processes DAAC.
ESA WorldCover:

Zanaga, D., et al. (2021). ESA WorldCover 10 m 2020 v100. 
https://doi.org/10.5281/zenodo.5571936
AAFC:

Agriculture and Agri-Food Canada (2024). Annual Crop Inventory. 
https://open.canada.ca/
Data Limitations
Spatial Coverage
AAFC: Canada only
SRTM: Limited to ±60° latitude
Temporal Latency
ERA5-Land: ~5 days behind real-time
MODIS: ~1-2 days
ESA WorldCover: Annual updates
AAFC: Annual updates (released following year)
Known Issues
Cloud contamination in optical data (MODIS)
Missing values in high-latitude regions
Seasonal bias in vegetation indices
Urban heat island effects in temperature data
Contact and Support
Google Earth Engine: https://developers.google.com/earth-engine/
NASA Earthdata Help: https://support.earthdata.nasa.gov/
ESA Support: https://earth.esa.int/eogateway/

Last Updated: October 24, 2025
Document Version: 1.0
