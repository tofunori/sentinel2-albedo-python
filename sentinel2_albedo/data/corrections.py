"""
Atmospheric and topographic corrections for satellite data.

This module provides correction methods for satellite reflectance data,
including SBAF (Spectral Band Adjustment Factor) corrections and
topographic corrections using the SCS+C method.

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import logging
from typing import Dict, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

from ..utils.io import load_raster


class AtmosphericCorrector:
    """
    Atmospheric corrections for satellite data.
    
    This class provides methods for atmospheric corrections including
    Rayleigh scattering and aerosol corrections.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def apply_rayleigh_correction(
        self, 
        reflectance: xr.DataArray,
        solar_zenith: xr.DataArray,
        view_zenith: xr.DataArray,
        relative_azimuth: xr.DataArray,
        wavelength: float
    ) -> xr.DataArray:
        """
        Apply Rayleigh scattering correction.
        
        Parameters
        ----------
        reflectance : xr.DataArray
            Top-of-atmosphere reflectance
        solar_zenith : xr.DataArray
            Solar zenith angle in radians
        view_zenith : xr.DataArray
            View zenith angle in radians
        relative_azimuth : xr.DataArray
            Relative azimuth angle in radians
        wavelength : float
            Wavelength in micrometers
            
        Returns
        -------
        xr.DataArray
            Rayleigh-corrected reflectance
        """
        # Simplified Rayleigh correction
        # In practice, would use more sophisticated atmospheric correction
        
        # Rayleigh optical depth (simplified)
        tau_r = 0.008569 * wavelength**(-4) * (1 + 0.0113 * wavelength**(-2) + 0.00013 * wavelength**(-4))
        
        # Air mass factors
        air_mass_sun = 1 / np.cos(solar_zenith)
        air_mass_view = 1 / np.cos(view_zenith)
        
        # Rayleigh reflectance (simplified)
        phase_function = 0.75 * (1 + np.cos(relative_azimuth)**2)
        
        rayleigh_refl = tau_r * phase_function / (4 * np.pi * np.cos(solar_zenith))
        
        # Apply correction
        corrected_refl = reflectance - rayleigh_refl
        
        return corrected_refl.where(corrected_refl > 0, 0)


class TopographicCorrector:
    """
    Topographic corrections for satellite data.
    
    This class implements the SCS+C (Sun-Canopy-Sensor + C) method
    for topographic correction of surface reflectance.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def apply_scs_correction(
        self, 
        reflectance_data: xr.Dataset,
        dem_data: xr.DataArray,
        solar_zenith: Optional[xr.DataArray] = None,
        solar_azimuth: Optional[xr.DataArray] = None
    ) -> xr.Dataset:
        """
        Apply SCS+C topographic correction.
        
        Parameters
        ----------
        reflectance_data : xr.Dataset
            Surface reflectance data
        dem_data : xr.DataArray
            Digital elevation model
        solar_zenith : xr.DataArray, optional
            Solar zenith angle (will be extracted from data if not provided)
        solar_azimuth : xr.DataArray, optional
            Solar azimuth angle (will be extracted from data if not provided)
            
        Returns
        -------
        xr.Dataset
            Topographically corrected reflectance data
        """
        self.logger.info("Applying SCS+C topographic correction...")
        
        # Extract solar geometry if not provided
        if solar_zenith is None:
            if 'solar_zenith' in reflectance_data.data_vars:
                solar_zenith = reflectance_data['solar_zenith']
            else:
                raise ValueError("Solar zenith angle not found in data")
        
        if solar_azimuth is None:
            if 'solar_azimuth' in reflectance_data.data_vars:
                solar_azimuth = reflectance_data['solar_azimuth']
            else:
                raise ValueError("Solar azimuth angle not found in data")
        
        # Compute terrain parameters
        slope, aspect = self._compute_terrain_parameters(dem_data)
        
        # Compute illumination angle
        cos_i = self._compute_illumination_angle(solar_zenith, solar_azimuth, slope, aspect)
        
        # Apply correction to each reflectance band
        corrected_data = {}
        
        for var_name, var_data in reflectance_data.data_vars.items():
            if var_name.startswith('sur_refl') or 'reflectance' in var_name.lower():
                # Apply SCS+C correction
                corrected_refl = self._apply_scsc_correction(var_data, cos_i, solar_zenith)
                corrected_data[var_name] = corrected_refl
            else:
                # Keep non-reflectance variables unchanged
                corrected_data[var_name] = var_data
        
        # Create corrected dataset
        corrected_dataset = xr.Dataset(corrected_data)
        
        # Add correction metadata
        corrected_dataset.attrs.update(reflectance_data.attrs)
        corrected_dataset.attrs['topographic_correction'] = 'SCS+C'
        
        return corrected_dataset
    
    def _compute_terrain_parameters(self, dem: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute slope and aspect from DEM.
        
        Parameters
        ----------
        dem : xr.DataArray
            Digital elevation model
            
        Returns
        -------
        tuple
            (slope, aspect) in radians
        """
        # Compute gradients
        grad_x = dem.differentiate('x')
        grad_y = dem.differentiate('y')
        
        # Compute slope
        slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        
        # Compute aspect
        aspect = np.arctan2(-grad_x, grad_y)
        
        # Ensure aspect is in [0, 2π]
        aspect = xr.where(aspect < 0, aspect + 2*np.pi, aspect)
        
        return slope, aspect
    
    def _compute_illumination_angle(
        self, 
        solar_zenith: xr.DataArray,
        solar_azimuth: xr.DataArray,
        slope: xr.DataArray,
        aspect: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute illumination angle on sloped terrain.
        
        Parameters
        ----------
        solar_zenith : xr.DataArray
            Solar zenith angle in radians
        solar_azimuth : xr.DataArray
            Solar azimuth angle in radians
        slope : xr.DataArray
            Terrain slope in radians
        aspect : xr.DataArray
            Terrain aspect in radians
            
        Returns
        -------
        xr.DataArray
            Cosine of illumination angle
        """
        # Compute cosine of illumination angle
        cos_i = (np.cos(slope) * np.cos(solar_zenith) + 
                np.sin(slope) * np.sin(solar_zenith) * np.cos(solar_azimuth - aspect))
        
        # Ensure cos_i is within valid range
        cos_i = xr.where(cos_i < 0.01, 0.01, cos_i)  # Avoid division by zero
        
        return cos_i
    
    def _apply_scsc_correction(
        self, 
        reflectance: xr.DataArray,
        cos_i: xr.DataArray,
        solar_zenith: xr.DataArray
    ) -> xr.DataArray:
        """
        Apply SCS+C correction to reflectance data.
        
        Parameters
        ----------
        reflectance : xr.DataArray
            Surface reflectance
        cos_i : xr.DataArray
            Cosine of illumination angle
        solar_zenith : xr.DataArray
            Solar zenith angle
            
        Returns
        -------
        xr.DataArray
            Corrected reflectance
        """
        # Compute regression between reflectance and cos_i
        # This is a simplified implementation
        
        # Flatten arrays for regression
        refl_flat = reflectance.values.flatten()
        cos_i_flat = cos_i.values.flatten()
        
        # Remove NaN values
        valid_mask = np.isfinite(refl_flat) & np.isfinite(cos_i_flat)
        
        if np.sum(valid_mask) < 10:  # Need minimum points for regression
            self.logger.warning("Insufficient valid points for topographic correction")
            return reflectance
        
        refl_valid = refl_flat[valid_mask]
        cos_i_valid = cos_i_flat[valid_mask]
        
        try:
            # Linear regression: reflectance = a * cos_i + b
            slope_reg, intercept, r_value, p_value, std_err = linregress(cos_i_valid, refl_valid)
            
            # SCS+C correction factor
            c_factor = intercept / slope_reg if slope_reg != 0 else 0
            
            # Apply correction
            correction_factor = ((np.cos(solar_zenith) + c_factor) / (cos_i + c_factor))
            
            corrected_refl = reflectance * correction_factor
            
            # Ensure corrected values are reasonable
            corrected_refl = xr.where(corrected_refl < 0, 0, corrected_refl)
            corrected_refl = xr.where(corrected_refl > 1.5, 1.5, corrected_refl)
            
            return corrected_refl
            
        except Exception as e:
            self.logger.warning(f"Error in topographic correction: {e}")
            return reflectance


class SBAFCorrector:
    """
    Spectral Band Adjustment Factor (SBAF) corrections.
    
    This class applies SBAF corrections to convert between different
    sensor spectral response functions (e.g., MODIS to Sentinel-2).
    """
    
    def __init__(self, sbaf_table_path: Optional[Path] = None):
        """
        Initialize SBAF corrector.
        
        Parameters
        ----------
        sbaf_table_path : Path, optional
            Path to SBAF lookup table
        """
        self.sbaf_table_path = sbaf_table_path
        self.logger = logging.getLogger(__name__)
        self.sbaf_table = None
        
        if sbaf_table_path and Path(sbaf_table_path).exists():
            self._load_sbaf_table()
    
    def _load_sbaf_table(self):
        """
        Load SBAF lookup table from file.
        """
        try:
            import pandas as pd
            self.sbaf_table = pd.read_csv(self.sbaf_table_path, sep=';')
            self.logger.info(f"Loaded SBAF table with {len(self.sbaf_table)} entries")
        except Exception as e:
            self.logger.error(f"Error loading SBAF table: {e}")
            self.sbaf_table = None
    
    def apply_sbaf_correction(
        self, 
        reflectance_data: xr.Dataset,
        landcover_data: xr.DataArray,
        sensor: str = 'S2A',
        bands: Optional[list] = None
    ) -> xr.Dataset:
        """
        Apply SBAF corrections to reflectance data.
        
        Parameters
        ----------
        reflectance_data : xr.Dataset
            Input reflectance data
        landcover_data : xr.DataArray
            Land cover classification
        sensor : str, optional
            Target sensor ('S2A' or 'S2B')
        bands : list, optional
            List of bands to correct
            
        Returns
        -------
        xr.Dataset
            SBAF-corrected reflectance data
        """
        if self.sbaf_table is None:
            self.logger.warning("No SBAF table loaded, using default corrections")
            return self._apply_default_sbaf(reflectance_data, sensor)
        
        self.logger.info(f"Applying SBAF corrections for {sensor}...")
        
        # Define band mapping
        band_mapping = {
            1: 'Red',
            2: 'NIR', 
            3: 'Blue',
            4: 'Green',
            6: 'SWIR1',
            7: 'SWIR2'
        }
        
        if bands is None:
            bands = list(band_mapping.keys())
        
        corrected_data = {}
        
        # Apply corrections to each band
        for var_name, var_data in reflectance_data.data_vars.items():
            if any(f'b{band:02d}' in var_name.lower() for band in bands):
                # Extract band number
                band_num = self._extract_band_number(var_name)
                
                if band_num in band_mapping:
                    band_name = band_mapping[band_num]
                    
                    # Apply SBAF correction
                    corrected_refl = self._apply_band_sbaf(
                        var_data, landcover_data, band_name, sensor
                    )
                    corrected_data[var_name] = corrected_refl
                else:
                    corrected_data[var_name] = var_data
            else:
                corrected_data[var_name] = var_data
        
        # Create corrected dataset
        corrected_dataset = xr.Dataset(corrected_data)
        corrected_dataset.attrs.update(reflectance_data.attrs)
        corrected_dataset.attrs['sbaf_correction'] = f'{sensor}_applied'
        
        return corrected_dataset
    
    def _extract_band_number(self, var_name: str) -> Optional[int]:
        """
        Extract band number from variable name.
        
        Parameters
        ----------
        var_name : str
            Variable name
            
        Returns
        -------
        int or None
            Band number if found
        """
        import re
        
        # Look for patterns like 'b01', 'b02', etc.
        match = re.search(r'b(\d{2})', var_name.lower())
        if match:
            return int(match.group(1))
        
        # Look for patterns like 'band_1', 'band_2', etc.
        match = re.search(r'band[_\s]*(\d+)', var_name.lower())
        if match:
            return int(match.group(1))
        
        return None
    
    def _apply_band_sbaf(
        self, 
        reflectance: xr.DataArray,
        landcover: xr.DataArray,
        band_name: str,
        sensor: str
    ) -> xr.DataArray:
        """
        Apply SBAF correction for a specific band.
        
        Parameters
        ----------
        reflectance : xr.DataArray
            Reflectance data for the band
        landcover : xr.DataArray
            Land cover classification
        band_name : str
            Band name (e.g., 'Red', 'NIR')
        sensor : str
            Sensor identifier
            
        Returns
        -------
        xr.DataArray
            SBAF-corrected reflectance
        """
        corrected_refl = reflectance.copy()
        
        # Get unique land cover classes
        unique_classes = np.unique(landcover.values)
        unique_classes = unique_classes[np.isfinite(unique_classes)]
        
        # Apply SBAF for each land cover class
        for lc_class in unique_classes:
            try:
                # Look up SBAF value
                sbaf_row = self.sbaf_table[
                    (self.sbaf_table['class_n'] == lc_class) & 
                    (self.sbaf_table['band'] == band_name) &
                    (self.sbaf_table['sensor'] == sensor)
                ]
                
                if not sbaf_row.empty:
                    sbaf_value = sbaf_row['sbaf'].iloc[0]
                    
                    # Apply correction to pixels of this land cover class
                    class_mask = landcover == lc_class
                    corrected_refl = xr.where(class_mask, reflectance * sbaf_value, corrected_refl)
                    
            except Exception as e:
                self.logger.warning(f"Error applying SBAF for class {lc_class}, band {band_name}: {e}")
        
        return corrected_refl
    
    def _apply_default_sbaf(self, reflectance_data: xr.Dataset, sensor: str) -> xr.Dataset:
        """
        Apply default SBAF corrections when no lookup table is available.
        
        Parameters
        ----------
        reflectance_data : xr.Dataset
            Input reflectance data
        sensor : str
            Sensor identifier
            
        Returns
        -------
        xr.Dataset
            Dataset with default corrections applied
        """
        # Default SBAF values (simplified)
        default_sbaf = {
            'S2A': {
                'red': 1.001,
                'nir': 0.999,
                'blue': 1.002,
                'green': 1.000,
                'swir1': 0.998,
                'swir2': 0.997
            },
            'S2B': {
                'red': 1.002,
                'nir': 0.998,
                'blue': 1.003,
                'green': 1.001,
                'swir1': 0.997,
                'swir2': 0.996
            }
        }
        
        sensor_sbaf = default_sbaf.get(sensor, default_sbaf['S2A'])
        
        corrected_data = {}
        
        for var_name, var_data in reflectance_data.data_vars.items():
            # Apply default correction based on band type
            if 'red' in var_name.lower() or 'b04' in var_name.lower():
                corrected_data[var_name] = var_data * sensor_sbaf['red']
            elif 'nir' in var_name.lower() or 'b08' in var_name.lower():
                corrected_data[var_name] = var_data * sensor_sbaf['nir']
            elif 'blue' in var_name.lower() or 'b02' in var_name.lower():
                corrected_data[var_name] = var_data * sensor_sbaf['blue']
            elif 'green' in var_name.lower() or 'b03' in var_name.lower():
                corrected_data[var_name] = var_data * sensor_sbaf['green']
            elif 'swir1' in var_name.lower() or 'b11' in var_name.lower():
                corrected_data[var_name] = var_data * sensor_sbaf['swir1']
            elif 'swir2' in var_name.lower() or 'b12' in var_name.lower():
                corrected_data[var_name] = var_data * sensor_sbaf['swir2']
            else:
                corrected_data[var_name] = var_data
        
        corrected_dataset = xr.Dataset(corrected_data)
        corrected_dataset.attrs.update(reflectance_data.attrs)
        corrected_dataset.attrs['sbaf_correction'] = f'{sensor}_default'
        
        return corrected_dataset
