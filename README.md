# Sentinel-2 High Resolution Albedo - Python Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Python implementation of high spatial resolution albedo retrieval from Sentinel-2 surface reflectance and MODIS BRDF information.

**Original R implementation by**: Andre Bertoncini (Centre for Hydrology - University of Saskatchewan)  
**Python port by**: Claude AI Assistant

## 🎯 Overview

This project generates high spatial resolution (20m) albedo maps by combining:
- **Sentinel-2** surface reflectance images
- **MODIS MOD09GA** surface reflectance and BRDF information  
- **Meteorological data** (temperature, humidity, radiation)
- **Topographic corrections** using DEM data

## 📚 Scientific Background

Based on the methodology from:
- Li et al. (2018) "Preliminary assessment of 20-m surface albedo retrievals from sentinel-2A surface reflectance and MODIS/VIIRS surface anisotropy measures" - *Remote Sensing of Environment*
- Shuai et al. (2011) "An algorithm for the retrieval of 30-m snow-free albedo from Landsat surface reflectance and MODIS BRDF" - *Remote Sensing of Environment*
- Jiao et al. (2019) "Development of a snow kernel to better model the anisotropic reflectance of pure snow" - *Remote Sensing of Environment*

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/tofunori/sentinel2-albedo-python.git
cd sentinel2-albedo-python
pip install -r requirements.txt
```

### Basic Usage

```python
from sentinel2_albedo import S2AlbedoProcessor

# Initialize processor
processor = S2AlbedoProcessor(
    target_date="2018-08-08",
    sensor="S2A",  # or "S2B"
    study_area_extent=(421290, 496851.5, 5759734, 5800008.8)
)

# Process BRDF kernels (choose one approach)
processor.compute_rtlsr_kernels()  # Standard RTLSR kernel
# OR
processor.compute_snow_kernels()   # Snow-specific kernel

# Compute albedo-to-nadir ratios
processor.compute_an_ratios()

# Generate final albedo maps
albedo_maps = processor.compute_s2_albedo()

print("Albedo computation complete!")
```

## 🔧 Workflow

The processing follows a 3-step workflow:

### 1. BRDF Kernel Computation
- **RTLSR Kernel**: Standard Ross-Thick Li-Sparse kernel model
- **Snow Kernel**: Specialized kernel for snow/ice surfaces
- Processes 16-day MODIS time series
- Applies quality filtering and corrections

### 2. Albedo-to-Nadir Ratios
- Calculates BSA (Black-Sky Albedo) ratios
- Calculates WSA (White-Sky Albedo) ratios  
- Integrates BRDF over illumination/observation hemispheres

### 3. Sentinel-2 Albedo Generation
- Applies ratios to Sentinel-2 surface reflectance
- K-means clustering for surface type classification
- Generates final 20m resolution albedo maps

## 🌟 Status

🚧 **Work in Progress** - Currently porting from original R implementation

- ✅ Project structure created
- 🔄 Core algorithms in development
- ⏳ Testing and validation pending

## 📞 Support

For questions or issues:
- Open an [issue](https://github.com/tofunori/sentinel2-albedo-python/issues)
- Check back for updates as development progresses

## 🙏 Acknowledgments

- **Andre Bertoncini** for the original R implementation
- **Centre for Hydrology, University of Saskatchewan** for the scientific methodology
