"""
Core processing modules for Sentinel-2 albedo computation.

This package contains the main algorithmic components:
- BRDF kernel calculations
- Albedo-to-nadir ratio computations  
- Geometric calculations for sun/view angles
- Main processor orchestration
"""

from .processor import S2AlbedoProcessor

__all__ = ["S2AlbedoProcessor"]
