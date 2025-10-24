# GeoAI-Carbon Emission-Canada

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview

This repository implements the **ST-ViWT (Spatio-Temporal Vision Transformer with Wavelet Transform)** framework for reconstructing column-averaged dry-air mole fraction of CO₂ (XCO₂) over Canada. The model integrates wavelet spectrograms with auxiliary geospatial datasets from Google Earth Engine to achieve high-accuracy XCO₂ reconstruction at 0.25° spatial resolution.

### Key Features

- **Multi-Source Data Integration**: Combines OCO-2/3 satellite observations with 31 auxiliary features from Google Earth Engine
- **Wavelet Transform**: Continuous Wavelet Transform (CWT) for spectral-temporal feature extraction
- **Vision Transformer Architecture**: Leverages self-attention mechanisms for spatial pattern learning
- **Google Earth Engine**: Automated extraction of ERA5-Land, MODIS, SRTM, ESA WorldCover, and AAFC data
- **High Performance**: Achieves R² > 0.95, RMSE < 0.5 ppm

## Table of Contents

- [Installation](#installation)
- [Data Sources](#data-sources)
- [ST-ViWT Framework](#ST-ViWT-Framework)
- [Usage](#usage)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Google Earth Engine account ([sign up here](https://earthengine.google.com/signup/))

## Data Sources

This project integrates multiple geospatial datasets:

### Primary Data
- **OCO-2/3 Satellite Observations**: Column-averaged CO₂ measurements
  - Spatial coverage: Canada (41°-60°N, 141°-52°W)
  - Temporal coverage: 2024
  - Resolution: 0.25° grid

### Auxiliary Data (via Google Earth Engine)

| Dataset | Variables | Temporal Resolution | Source |
|---------|-----------|---------------------|--------|
| **ERA5-Land** | Temperature, Precipitation, Wind (U/V), Pressure, Dewpoint | Daily | ECMWF |
| **MODIS MOD13A1** | NDVI, EVI | 16-day | NASA |
| **SRTM** | Elevation, Slope | Static | USGS |
| **ESA WorldCover** | Land Cover Classification | Annual | ESA |
| **AAFC Crop Inventory** | Crop Types | Annual | Agriculture Canada |

**Total**: 31 auxiliary features

For detailed information, see [data/data_sources.md](data/data_sources.md).

## ST-ViWT Framework

1. **Wavelet Spectrogram Encoder**
   - Continuous Wavelet Transform (CWT) with Morlet wavelet
   - Captures multi-scale temporal patterns in XCO₂ time series
   - Output: 64 × 64 spectrograms

2. **Vision Transformer (ViT)**
   - Patch size: 8 × 8
   - 6 transformer layers
   - 8 attention heads
   - Embedding dimension: 256
   - Processes spatial patterns in spectrograms

3. **Auxiliary Feature Fusion**
   - Fully connected layers for auxiliary data integration
   - Late fusion strategy: combines ViT outputs with auxiliary features
   - Final regression head for XCO₂ prediction

See [models/training_config.json](models/training_config.json) for complete configuration.

## Usage

1. Extract Auxiliary Data from Google Earth Engine
2. Preprocess OCO-2 Data
3. Train Model STViWTModel
4. Generate XCO2 Reconstructions
5. Visualize Results

## Results

**R²**=0.9547 | **MAE**=0.412 ppm | **RMSE**=0.487 ppm | **MAPE**=1.23% | **Bias**=-0.021 ppm 

### Key Findings

- **Achieves sub-0.5 ppm RMSE, suitable for carbon cycle studies**
- **Captures regional CO₂ patterns across diverse Canadian landscapes**
- **Consistent performance across seasonal variations**
- **Meteorological variables (temperature, wind) show highest impact**

See [results/performance_metrics/](results/performance_metrics/) for detailed evaluation.

## Repository Structure

```
XCO2-OCO2-STViWT-main/
├── README.md                 - Main documentation
├── LICENSE                   - MIT License
├── training_config.json      - Model hyperparameters & configuration
├── GEE_extractor.py          - Google Earth Engine data extraction
├── GEE_setup.md              - GEE setup instructions
├── data_source.md            - Detailed data documentation
├── methodology.md            - Technical methodology
├── preprocessor.py           - Data preprocessing utilities
├── wavelet_transform.py      - Wavelet spectrogram generation
├── st_viwt_model.py          - Complete model architecture
├── train.py                  - Training script
└── evaluate.py               - Evaluation script
```

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{stviwt2024,
  title={AI-Driven Carbon Monitoring: Transformer-Based Reconstruction of Atmospheric CO₂ in Canadian Poultry Regions},
  author={Prajesh,PJ and Ragunath,Kaliaperumal and Gordon,Miriam and Rathgeber,Bruce and Neethirajan,Suresh},
  journal={TBD},
  year={2025},
  volume={TBD},
  pages={TBD},
  doi={TBD}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
