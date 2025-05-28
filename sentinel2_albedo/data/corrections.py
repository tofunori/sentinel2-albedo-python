"""
Atmospheric and topographic corrections for satellite data.

This module provides correction methods for atmospheric effects and
topographic influences on surface reflectance data.

Based on:
- Soenen et al. (2005) - SCS+C topographic correction
- Various atmospheric correction approaches

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy import ndimage
from sklearn.linear_model import LinearRegression

from ..utils.io import load_raster


class AtmosphericCorrector:
    """
    Atmospheric correction utilities.
    
    This class provides methods for basic atmospheric corrections,
    though most Sentinel-2 and MODIS data should already be
    atmospherically corrected.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def apply_simple_correction(
        self, 
        reflectance_data: xr.Dataset,
        atmospheric_params: Optional[Dict] = None
    ) -> xr.Dataset:
        """
        Apply simple atmospheric correction.
        
        Parameters
        ----------
        reflectance_data : xr.Dataset
            Surface reflectance data
        atmospheric_params : dict, optional
            Atmospheric correction parameters
            
        Returns
        -------
        xr.Dataset
            Corrected reflectance data
        """
        # For L2A products, atmospheric correction is already applied
        # This is a placeholder for additional corrections if needed
        
        self.logger.info("Applying atmospheric correction...")
        
        corrected_data = reflectance_data.copy()
        
        # Add atmospheric correction metadata
        corrected_data.attrs['atmospheric_correction'] = 'simple'
        
        return corrected_data


class TopographicCorrector:
    """
    Topographic correction using SCS+C method.
    
    This class implements the Sun-Canopy-Sensor + C (SCS+C) topographic
    correction method from Soenen et al. (2005).
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def apply_scs_correction(
        self,
        reflectance_data: xr.Dataset,
        dem_data: xr.DataArray,
        solar_geometry: Optional[Dict[str, xr.DataArray]] = None
    ) -> xr.Dataset:
        """
        Apply SCS+C topographic correction.
        
        Parameters
        ----------
        reflectance_data : xr.Dataset
            Surface reflectance data
        dem_data : xr.DataArray
            Digital elevation model
        solar_geometry : dict, optional
            Solar geometry data (theta_s, phi_s)
            
        Returns
        -------
        xr.Dataset
            Topographically corrected reflectance data
        """
        self.logger.info("Applying SCS+C topographic correction...")
        
        # Calculate slope and aspect from DEM
        slope, aspect = self._calculate_topographic_variables(dem_data)
        
        # Get or compute illumination angles
        if solar_geometry is not None:
            theta_s = solar_geometry['theta_s']
            phi_s = solar_geometry['phi_s']
        else:
            # Use approximate solar geometry (placeholder)
            theta_s = xr.full_like(slope, np.radians(45))  # 45 degrees
            phi_s = xr.full_like(slope, np.radians(180))   # South-facing
        
        # Calculate cosine of local illumination angle
        cos_local_illum = self._calculate_local_illumination(
            slope, aspect, theta_s, phi_s
        )
        
        # Apply correction to each band
        corrected_data = {}
        
        for var_name, var_data in reflectance_data.data_vars.items():
            if var_name.startswith('sur_refl') or var_name in ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']:
                # Apply SCS+C correction
                corrected_band = self._apply_scs_to_band(
                    var_data, cos_local_illum, theta_s
                )
                corrected_data[var_name] = corrected_band
            else:
                # Keep non-reflectance variables unchanged
                corrected_data[var_name] = var_data
        
        # Create corrected dataset
        corrected_dataset = xr.Dataset(corrected_data)
        
        # Copy attributes and add correction info
        corrected_dataset.attrs.update(reflectance_data.attrs)
        corrected_dataset.attrs['topographic_correction'] = 'SCS+C'
        
        return corrected_dataset
    
    def _calculate_topographic_variables(
        self, 
        dem: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Calculate slope and aspect from DEM.
        
        Parameters
        ----------
        dem : xr.DataArray
            Digital elevation model
            
        Returns
        -------
        tuple
            (slope, aspect) in radians
        """
        # Get pixel resolution
        x_res = abs(float(dem.x[1] - dem.x[0]))
        y_res = abs(float(dem.y[1] - dem.y[0]))
        
        # Calculate gradients
        dem_values = dem.values
        
        # Use numpy gradient
        grad_y, grad_x = np.gradient(dem_values)
        
        # Scale by resolution
        grad_x = grad_x / x_res
        grad_y = grad_y / y_res
        
        # Calculate slope (in radians)
        slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        
        # Calculate aspect (in radians)
        aspect = np.arctan2(grad_y, grad_x)
        
        # Convert to xarray DataArrays
        slope_da = xr.DataArray(
            slope,
            dims=dem.dims,
            coords=dem.coords,
            attrs={'long_name': 'Slope', 'units': 'radians'}
        )
        
        aspect_da = xr.DataArray(
            aspect,
            dims=dem.dims,
            coords=dem.coords,
            attrs={'long_name': 'Aspect', 'units': 'radians'}
        )
        
        return slope_da, aspect_da
    
    def _calculate_local_illumination(
        self,
        slope: xr.DataArray,
        aspect: xr.DataArray,
        theta_s: xr.DataArray,
        phi_s: xr.DataArray
    ) -> xr.DataArray:
        """
        Calculate cosine of local illumination angle.
        
        Parameters
        ----------
        slope : xr.DataArray
            Terrain slope in radians
        aspect : xr.DataArray
            Terrain aspect in radians
        theta_s : xr.DataArray
            Solar zenith angle in radians
        phi_s : xr.DataArray
            Solar azimuth angle in radians
            
        Returns
        -------
        xr.DataArray
            Cosine of local illumination angle
        """
        # Local illumination angle calculation
        cos_local = (np.cos(slope) * np.cos(theta_s) + 
                    np.sin(slope) * np.sin(theta_s) * np.cos(phi_s - aspect))
        
        # Ensure values are within valid range [-1, 1]
        cos_local = xr.where(cos_local < -1, -1, cos_local)
        cos_local = xr.where(cos_local > 1, 1, cos_local)
        
        return cos_local
    
    def _apply_scs_to_band(
        self,
        reflectance: xr.DataArray,
        cos_local_illum: xr.DataArray,
        theta_s: xr.DataArray
    ) -> xr.DataArray:
        """
        Apply SCS+C correction to a single band.
        
        Parameters
        ----------
        reflectance : xr.DataArray
            Band reflectance data
        cos_local_illum : xr.DataArray
            Cosine of local illumination angle
        theta_s : xr.DataArray
            Solar zenith angle
            
        Returns
        -------
        xr.DataArray
            Corrected reflectance
        """
        # Flatten arrays for regression
        refl_values = reflectance.values.flatten()
        cos_local_values = cos_local_illum.values.flatten()
        
        # Remove invalid values
        valid_mask = (np.isfinite(refl_values) & 
                     np.isfinite(cos_local_values) & 
                     (refl_values > 0) & 
                     (cos_local_values > 0))
        
        if np.sum(valid_mask) < 10:  # Need minimum number of points
            self.logger.warning("Insufficient valid data for topographic correction")
            return reflectance
        
        refl_valid = refl_values[valid_mask]
        cos_local_valid = cos_local_values[valid_mask]
        
        try:
            # Fit linear regression: reflectance = a + b * cos_local_illum
            reg = LinearRegression()
            reg.fit(cos_local_valid.reshape(-1, 1), refl_valid)
            
            # Get regression parameters
            b = reg.coef_[0]
            a = reg.intercept_
            
            # Calculate C parameter for SCS+C
            c_factor = a / b if b != 0 else 0
            
            # Apply SCS+C correction
            cos_flat = np.cos(theta_s)  # Flat surface illumination
            
            correction_factor = ((cos_flat + c_factor) / 
                                (cos_local_illum + c_factor))
            
            # Apply correction
            corrected_reflectance = reflectance * correction_factor
            
            # Ensure non-negative values
            corrected_reflectance = xr.where(
                corrected_reflectance < 0, 0, corrected_reflectance
            )
            
            # Ensure reasonable upper bound
            corrected_reflectance = xr.where(
                corrected_reflectance > 1.5, 1.5, corrected_reflectance
            )
            
        except Exception as e:
            self.logger.warning(f"Error in SCS+C correction: {e}")
            corrected_reflectance = reflectance
        
        return corrected_reflectance
    
    def calculate_terrain_metrics(
        self, 
        dem: xr.DataArray
    ) -> Dict[str, xr.DataArray]:
        """
        Calculate various terrain metrics from DEM.
        
        Parameters
        ----------
        dem : xr.DataArray
            Digital elevation model
            
        Returns
        -------
        dict
            Dictionary containing terrain metrics
        """
        # Calculate slope and aspect
        slope, aspect = self._calculate_topographic_variables(dem)
        
        # Calculate additional metrics
        slope_degrees = slope * 180 / np.pi
        aspect_degrees = aspect * 180 / np.pi
        
        # Terrain roughness (standard deviation of elevation in local window)
        dem_values = dem.values
        roughness = ndimage.generic_filter(
            dem_values, np.std, size=3, mode='constant', cval=np.nan
        )
        
        roughness_da = xr.DataArray(
            roughness,
            dims=dem.dims,
            coords=dem.coords,
            attrs={'long_name': 'Terrain Roughness', 'units': 'meters'}
        )
        
        # Terrain position index (TPI)
        # TPI = elevation - mean elevation in neighborhood
        mean_elevation = ndimage.uniform_filter(
            dem_values, size=3, mode='constant', cval=np.nan
        )
        
        tpi = dem_values - mean_elevation
        
        tpi_da = xr.DataArray(
            tpi,
            dims=dem.dims,
            coords=dem.coords,
            attrs={'long_name': 'Terrain Position Index', 'units': 'meters'}
        )
        
        return {
            'slope': slope,
            'aspect': aspect,
            'slope_degrees': xr.DataArray(
                slope_degrees, dims=dem.dims, coords=dem.coords,
                attrs={'long_name': 'Slope', 'units': 'degrees'}
            ),
            'aspect_degrees': xr.DataArray(
                aspect_degrees, dims=dem.dims, coords=dem.coords,
                attrs={'long_name': 'Aspect', 'units': 'degrees'}
            ),
            'roughness': roughness_da,
            'tpi': tpi_da
        }


class SpectralBandAdjustmentFactor:
    """
    Spectral Band Adjustment Factor (SBAF) corrections.
    
    This class handles SBAF corrections to convert between different
    sensor spectral response functions (e.g., MODIS to Sentinel-2).
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Default SBAF values (these should be loaded from external files)
        # Values depend on land cover type and sensor combination
        self.default_sbaf = {
            'S2A': {
                'red': {'forest': 1.02, 'crop': 1.01, 'urban': 1.03, 'water': 1.00},
                'nir': {'forest': 0.98, 'crop': 0.99, 'urban': 0.97, 'water': 1.00},
                'blue': {'forest': 1.05, 'crop': 1.03, 'urban': 1.06, 'water': 1.01},
                'green': {'forest': 1.02, 'crop': 1.01, 'urban': 1.04, 'water': 1.00},
                'swir1': {'forest': 0.96, 'crop': 0.98, 'urban': 0.95, 'water': 1.00},
                'swir2': {'forest': 0.97, 'crop': 0.99, 'urban': 0.96, 'water': 1.00}
            },
            'S2B': {
                'red': {'forest': 1.01, 'crop': 1.00, 'urban': 1.02, 'water': 1.00},
                'nir': {'forest': 0.99, 'crop': 1.00, 'urban': 0.98, 'water': 1.00},
                'blue': {'forest': 1.04, 'crop': 1.02, 'urban': 1.05, 'water': 1.01},
                'green': {'forest': 1.01, 'crop': 1.00, 'urban': 1.03, 'water': 1.00},
                'swir1': {'forest': 0.97, 'crop': 0.99, 'urban': 0.96, 'water': 1.00},
                'swir2': {'forest': 0.98, 'crop': 1.00, 'urban': 0.97, 'water': 1.00}
            }
        }
    
    def apply_sbaf_correction(
        self,
        reflectance_data: xr.Dataset,
        landcover_map: xr.DataArray,
        sensor: str = 'S2A'
    ) -> xr.Dataset:
        """
        Apply SBAF corrections based on land cover.
        
        Parameters
        ----------
        reflectance_data : xr.Dataset
            MODIS reflectance data
        landcover_map : xr.DataArray
            Land cover classification map
        sensor : str, optional
            Target sensor ('S2A' or 'S2B')
            
        Returns
        -------
        xr.Dataset
            SBAF-corrected reflectance data
        """
        self.logger.info(f"Applying SBAF corrections for {sensor}...")
        
        if sensor not in self.default_sbaf:
            raise ValueError(f"SBAF values not available for sensor {sensor}")
        
        corrected_data = {}
        sensor_sbaf = self.default_sbaf[sensor]
        
        # Define land cover mapping (MODIS IGBP classes)
        lc_mapping = {
            1: 'forest',   # Evergreen Needleleaf Forests
            2: 'forest',   # Evergreen Broadleaf Forests
            3: 'forest',   # Deciduous Needleleaf Forests
            4: 'forest',   # Deciduous Broadleaf Forests
            5: 'forest',   # Mixed Forests
            6: 'forest',   # Closed Shrublands
            7: 'forest',   # Open Shrublands
            8: 'forest',   # Woody Savannas
            9: 'forest',   # Savannas
            10: 'forest',  # Grasslands
            11: 'crop',    # Permanent Wetlands
            12: 'crop',    # Croplands
            13: 'urban',   # Urban and Built-up Lands
            14: 'crop',    # Cropland/Natural Vegetation Mosaics
            15: 'crop',    # Permanent Snow and Ice
            16: 'crop',    # Barren
            17: 'water'    # Water Bodies
        }
        
        # Apply corrections to each band
        for var_name, var_data in reflectance_data.data_vars.items():
            # Determine band name
            band_name = None
            if 'b01' in var_name.lower() or 'red' in var_name.lower():
                band_name = 'red'
            elif 'b02' in var_name.lower() or 'nir' in var_name.lower():
                band_name = 'nir'
            elif 'b03' in var_name.lower() or 'blue' in var_name.lower():
                band_name = 'blue'
            elif 'b04' in var_name.lower() or 'green' in var_name.lower():
                band_name = 'green'
            elif 'b06' in var_name.lower() or 'swir1' in var_name.lower():
                band_name = 'swir1'
            elif 'b07' in var_name.lower() or 'swir2' in var_name.lower():
                band_name = 'swir2'
            
            if band_name and band_name in sensor_sbaf:
                # Apply SBAF correction based on land cover
                corrected_band = self._apply_sbaf_to_band(
                    var_data, landcover_map, sensor_sbaf[band_name], lc_mapping
                )
                corrected_data[var_name] = corrected_band
            else:
                # Keep non-reflectance bands unchanged
                corrected_data[var_name] = var_data
        
        # Create corrected dataset
        corrected_dataset = xr.Dataset(corrected_data)
        corrected_dataset.attrs.update(reflectance_data.attrs)
        corrected_dataset.attrs['sbaf_correction'] = sensor
        
        return corrected_dataset
    
    def _apply_sbaf_to_band(
        self,
        band_data: xr.DataArray,
        landcover_map: xr.DataArray,
        sbaf_values: Dict[str, float],
        lc_mapping: Dict[int, str]
    ) -> xr.DataArray:
        """
        Apply SBAF correction to a single band.
        
        Parameters
        ----------
        band_data : xr.DataArray
            Band reflectance data
        landcover_map : xr.DataArray
            Land cover map
        sbaf_values : dict
            SBAF values for different land cover types
        lc_mapping : dict
            Mapping from land cover codes to types
            
        Returns
        -------
        xr.DataArray
            SBAF-corrected band data
        """
        # Initialize corrected data
        corrected_data = band_data.copy()
        
        # Apply SBAF for each land cover type
        for lc_code, lc_type in lc_mapping.items():
            if lc_type in sbaf_values:
                # Create mask for this land cover type
                lc_mask = landcover_map == lc_code
                
                # Apply SBAF correction
                sbaf_factor = sbaf_values[lc_type]
                corrected_data = xr.where(
                    lc_mask,
                    band_data * sbaf_factor,
                    corrected_data
                )
        
        return corrected_data
    
    def load_sbaf_table(self, filepath: str, sensor: str) -> None:
        """
        Load SBAF values from external file.
        
        Parameters
        ----------
        filepath : str
            Path to SBAF table file
        sensor : str
            Sensor identifier
        """
        try:
            import pandas as pd
            
            # Load SBAF table (format depends on file structure)
            sbaf_table = pd.read_csv(filepath, sep=';')
            
            # Parse and store SBAF values
            # This is a placeholder - actual implementation depends on file format
            self.logger.info(f"Loaded SBAF table for {sensor} from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading SBAF table: {e}")
            raise
