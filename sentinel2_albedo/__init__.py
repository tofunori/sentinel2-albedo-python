"""
Sentinel-2 High Resolution Albedo - Python Implementation

Python port of Andre Bertoncini's R scripts for generating high spatial resolution
albedo from Sentinel-2 surface reflectance and MODIS BRDF information.

Original R implementation by Andre Bertoncini (Centre for Hydrology - University of Saskatchewan)
Python port by Claude AI Assistant
"""

__version__ = "1.0.0"
__author__ = "Claude AI Assistant"
__email__ = ""
__license__ = "GPL-3.0"

# Main processor class
from .core.processor import S2AlbedoProcessor

# Version info
__all__ = [
    "S2AlbedoProcessor",
    "__version__"
]
