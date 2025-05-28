"""
Data handling modules for Sentinel-2 and MODIS datasets.

This package contains modules for:
- Sentinel-2 data loading and processing
- MODIS data handling and time series management
- Atmospheric and topographic corrections
- Spectral band adjustment factors (SBAF)
"""

from . import sentinel2, modis, corrections

__all__ = ["sentinel2", "modis", "corrections"]
