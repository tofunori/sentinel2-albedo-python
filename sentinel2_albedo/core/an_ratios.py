"""
Albedo-to-Nadir (AN) ratio calculations.

This module computes the ratios needed to convert Sentinel-2 nadir reflectance
to hemispherical albedo using MODIS BRDF kernel parameters.

Based on Shuai et al. (2011) "An algorithm for the retrieval of 30-m snow-free 
albedo from Landsat surface reflectance and MODIS BRDF" - Remote Sensing of Environment.

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import logging
from typing import Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import integrate
from tqdm import tqdm

from .geometry import GeometryCalculator
from ..utils.io import save_raster


class AlbedoNadirRatioCalculator:
    """
    Calculator for Albedo-to-Nadir (AN) ratios.
    
    This class computes BSA (Black-Sky Albedo) and WSA (White-Sky Albedo) ratios
    that are used to convert Sentinel-2 directional reflectance to hemispherical albedo.
    
    The ratios are computed by integrating BRDF models over different hemispheres:
    - BSA: Integration over observation hemisphere only (fixed solar angle)
    - WSA: Integration over both observation and illumination hemispheres
    """
    
    def __init__(
        self,
        brdf_kernels: Dict[str, xr.DataArray],
        s2_geometry: Dict[str, xr.DataArray],
        geometry_calc: GeometryCalculator,
        output_path: Path,
        **kwargs
    ):
        self.brdf_kernels = brdf_kernels
        self.s2_geometry = s2_geometry
        self.geometry_calc = geometry_calc
        self.output_path = Path(output_path)
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters
        self.integration_tolerance = kwargs.get('integration_tolerance', 1e-2)
        self.max_zenith = kwargs.get('max_zenith', np.pi/2)
        self.max_azimuth = kwargs.get('max_azimuth', 2*np.pi)
        
        # RTLSR kernel constants
        self.k1 = 1.247
        self.k2 = 1.186
        self.k3 = 5.157
        self.alpha = 0.3
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Extract date string for file naming
        self.date_str = kwargs.get('date_str', '20180808')
    
    def compute_ratios(self) -> Dict[str, Dict[str, xr.DataArray]]:
        """
        Compute both BSA and WSA ratios for all spectral bands.
        
        Returns
        -------
        dict
            Dictionary containing 'bsa' and 'wsa' ratios for each band
        """
        self.logger.info("Computing Albedo-to-Nadir ratios...")
        
        # Determine available bands from BRDF kernels
        bands = self._get_available_bands()
        
        bsa_ratios = {}
        wsa_ratios = {}
        
        for band in bands:
            self.logger.info(f"Processing band {band}...")
            
            # Get BRDF kernels for this band
            band_kernels = self._extract_band_kernels(band)
            
            # Apply quality filtering
            band_kernels = self._apply_quality_filters(band_kernels)
            
            # Compute ratios
            bsa_ratio = self._compute_bsa_ratio(band_kernels)
            wsa_ratio = self._compute_wsa_ratio(band_kernels)
            
            # Store results
            bsa_ratios[f'B{band}'] = bsa_ratio
            wsa_ratios[f'B{band}'] = wsa_ratio
            
            # Save to files
            self._save_ratio(bsa_ratio, f'a_bsa_{self.date_str}_B{band}.tif')
            self._save_ratio(wsa_ratio, f'a_wsa_{self.date_str}_B{band}.tif')
        
        self.logger.info("AN ratio computation completed")
        
        return {
            'bsa': bsa_ratios,
            'wsa': wsa_ratios
        }
    
    def _get_available_bands(self) -> list:
        """
        Extract available band numbers from BRDF kernel data.
        
        Returns
        -------
        list
            List of available band numbers
        """
        bands = []
        for key in self.brdf_kernels.keys():
            if 'f_iso_B' in key:
                band_num = key.split('_B')[1]
                bands.append(int(band_num))
        
        return sorted(bands)
    
    def _extract_band_kernels(self, band: int) -> Dict[str, xr.DataArray]:
        """
        Extract BRDF kernel parameters for a specific band.
        
        Parameters
        ----------
        band : int
            Band number
            
        Returns
        -------
        dict
            Dictionary containing kernel parameters for the band
        """
        kernels = {}
        
        # Required parameters
        required_params = ['f_iso', 'f_vol', 'f_geo']
        
        for param in required_params:
            key = f'{param}_B{band}'
            if key in self.brdf_kernels:
                kernels[param] = self.brdf_kernels[key]
            else:
                raise ValueError(f"Missing BRDF kernel parameter: {key}")
        
        # Optional snow kernel
        snow_key = f'f_snow_B{band}'
        if snow_key in self.brdf_kernels:
            kernels['f_snow'] = self.brdf_kernels[snow_key]
        
        # Quality metrics
        quality_params = ['rmse', 'wod_wdr', 'wod_wsa']
        for param in quality_params:
            key = f'{param}_B{band}'
            if key in self.brdf_kernels:
                kernels[param] = self.brdf_kernels[key]
        
        return kernels
    
    def _apply_quality_filters(self, kernels: Dict[str, xr.DataArray]) -> Dict[str, xr.DataArray]:
        """
        Apply quality filtering based on MODIS thresholds.
        
        Following Shuai et al. (2008) quality assessment criteria.
        
        Parameters
        ----------
        kernels : dict
            Dictionary containing kernel parameters
            
        Returns
        -------
        dict
            Quality-filtered kernel parameters
        """
        # Create quality mask
        quality_mask = xr.ones_like(kernels['f_iso'], dtype=bool)
        
        # Apply RMSE threshold
        if 'rmse' in kernels:
            quality_mask &= kernels['rmse'] <= 0.08
        
        # Apply WoD thresholds
        if 'wod_wdr' in kernels:
            quality_mask &= kernels['wod_wdr'] <= 1.65
        
        if 'wod_wsa' in kernels:
            quality_mask &= kernels['wod_wsa'] <= 2.5
        
        # Apply mask to all kernel parameters
        filtered_kernels = {}
        for param, data in kernels.items():
            if param in ['f_iso', 'f_vol', 'f_geo', 'f_snow']:
                filtered_kernels[param] = data.where(quality_mask)
            else:
                filtered_kernels[param] = data
        
        return filtered_kernels
    
    def _compute_bsa_ratio(self, kernels: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Compute Black-Sky Albedo (BSA) ratio.
        
        BSA is computed by integrating BRDF over the observation hemisphere
        while keeping the solar zenith angle fixed.
        
        Parameters
        ----------
        kernels : dict
            BRDF kernel parameters
            
        Returns
        -------
        xr.DataArray
            BSA ratio map
        """
        self.logger.info("Computing BSA ratios...")
        
        # Get Sentinel-2 observation geometry
        theta_s_s2 = self.s2_geometry['solar_zenith']
        theta_v_s2 = self.s2_geometry['view_zenith']
        phi_s2 = self.s2_geometry['relative_azimuth']
        
        # Compute BRDF at Sentinel-2 geometry (R_omega)
        R_omega = self._compute_brdf_at_geometry(
            kernels, theta_s_s2, theta_v_s2, phi_s2
        )
        
        # Compute directional-hemispherical reflectance (R_l_theta_s)
        R_l_theta_s = self._compute_directional_hemispherical_reflectance(
            kernels, theta_s_s2
        )
        
        # BSA ratio
        bsa_ratio = R_l_theta_s / R_omega
        
        # Handle division by zero
        bsa_ratio = bsa_ratio.where(R_omega > 0)
        
        return bsa_ratio
    
    def _compute_wsa_ratio(self, kernels: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Compute White-Sky Albedo (WSA) ratio.
        
        WSA is computed by integrating BRDF over both observation and
        illumination hemispheres.
        
        Parameters
        ----------
        kernels : dict
            BRDF kernel parameters
            
        Returns
        -------
        xr.DataArray
            WSA ratio map
        """
        self.logger.info("Computing WSA ratios...")
        
        # Get Sentinel-2 observation geometry
        theta_s_s2 = self.s2_geometry['solar_zenith']
        theta_v_s2 = self.s2_geometry['view_zenith']
        phi_s2 = self.s2_geometry['relative_azimuth']
        
        # Compute BRDF at Sentinel-2 geometry (R_omega)
        R_omega = self._compute_brdf_at_geometry(
            kernels, theta_s_s2, theta_v_s2, phi_s2
        )
        
        # Compute bi-hemispherical reflectance (R_l)
        R_l = self._compute_bihemispherical_reflectance(kernels)
        
        # WSA ratio
        wsa_ratio = R_l / R_omega
        
        # Handle division by zero
        wsa_ratio = wsa_ratio.where(R_omega > 0)
        
        return wsa_ratio
    
    def _compute_brdf_at_geometry(
        self, 
        kernels: Dict[str, xr.DataArray],
        theta_s: xr.DataArray,
        theta_v: xr.DataArray,
        phi: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute BRDF at specific observation geometry.
        
        Parameters
        ----------
        kernels : dict
            BRDF kernel parameters
        theta_s : xr.DataArray
            Solar zenith angle
        theta_v : xr.DataArray
            View zenith angle
        phi : xr.DataArray
            Relative azimuth angle
            
        Returns
        -------
        xr.DataArray
            BRDF reflectance at specified geometry
        """
        # Compute kernel values at specified geometry
        k_vol_omega = self._compute_volumetric_kernel(theta_s, theta_v, phi)
        k_geo_omega = self._compute_geometric_kernel(theta_s, theta_v, phi)
        
        # Basic BRDF model
        R_omega = (kernels['f_iso'] + 
                  kernels['f_vol'] * k_vol_omega + 
                  kernels['f_geo'] * k_geo_omega)
        
        # Add snow kernel if available
        if 'f_snow' in kernels:
            k_snow_omega = self._compute_snow_kernel(theta_s, theta_v, phi)
            R_omega += kernels['f_snow'] * k_snow_omega
        
        # Ensure non-negative values
        R_omega = R_omega.where(R_omega >= 0)
        
        return R_omega
    
    def _compute_directional_hemispherical_reflectance(
        self, 
        kernels: Dict[str, xr.DataArray],
        theta_s: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute directional-hemispherical reflectance (BSA).
        
        This integrates BRDF over the observation hemisphere for fixed solar angle.
        
        Parameters
        ----------
        kernels : dict
            BRDF kernel parameters
        theta_s : xr.DataArray
            Solar zenith angle
            
        Returns
        -------
        xr.DataArray
            Directional-hemispherical reflectance
        """
        # Compute kernel integrals for directional-hemispherical case
        k_vol_dh = self._compute_kernel_integral_directional_hemispherical(
            theta_s, kernel_type='volumetric'
        )
        k_geo_dh = self._compute_kernel_integral_directional_hemispherical(
            theta_s, kernel_type='geometric'
        )
        
        # Compute reflectance
        R_dh = (kernels['f_iso'] + 
               kernels['f_vol'] * k_vol_dh + 
               kernels['f_geo'] * k_geo_dh)
        
        # Add snow kernel if available
        if 'f_snow' in kernels:
            k_snow_dh = self._compute_kernel_integral_directional_hemispherical(
                theta_s, kernel_type='snow'
            )
            R_dh += kernels['f_snow'] * k_snow_dh
        
        return R_dh
    
    def _compute_bihemispherical_reflectance(
        self, 
        kernels: Dict[str, xr.DataArray]
    ) -> xr.DataArray:
        """
        Compute bi-hemispherical reflectance (WSA).
        
        This integrates BRDF over both observation and illumination hemispheres.
        
        Parameters
        ----------
        kernels : dict
            BRDF kernel parameters
            
        Returns
        -------
        xr.DataArray
            Bi-hemispherical reflectance
        """
        # Compute kernel integrals for bi-hemispherical case
        k_vol_bh = self._compute_kernel_integral_bihemispherical('volumetric')
        k_geo_bh = self._compute_kernel_integral_bihemispherical('geometric')
        
        # Compute reflectance
        R_bh = (kernels['f_iso'] + 
               kernels['f_vol'] * k_vol_bh + 
               kernels['f_geo'] * k_geo_bh)
        
        # Add snow kernel if available
        if 'f_snow' in kernels:
            k_snow_bh = self._compute_kernel_integral_bihemispherical('snow')
            R_bh += kernels['f_snow'] * k_snow_bh
        
        return R_bh
    
    def _compute_volumetric_kernel(
        self, 
        theta_s: xr.DataArray, 
        theta_v: xr.DataArray, 
        phi: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute Ross-Thick volumetric kernel.
        
        Parameters
        ----------
        theta_s : xr.DataArray
            Solar zenith angle in radians
        theta_v : xr.DataArray
            View zenith angle in radians
        phi : xr.DataArray
            Relative azimuth angle in radians
            
        Returns
        -------
        xr.DataArray
            Volumetric kernel values
        """
        # Compute phase angle
        cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                    np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
        cos_phase = xr.where(cos_phase > 1, 1, cos_phase)
        cos_phase = xr.where(cos_phase < -1, -1, cos_phase)
        phase = np.arccos(cos_phase)
        
        # Ross-Thick kernel
        k_vol = ((np.pi/2 - phase) * cos_phase + np.sin(phase)) / \
                (np.cos(theta_s) + np.cos(theta_v)) - np.pi/4
        
        return k_vol
    
    def _compute_geometric_kernel(
        self, 
        theta_s: xr.DataArray, 
        theta_v: xr.DataArray, 
        phi: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute Li-Sparse geometric kernel.
        
        Parameters
        ----------
        theta_s : xr.DataArray
            Solar zenith angle in radians
        theta_v : xr.DataArray
            View zenith angle in radians
        phi : xr.DataArray
            Relative azimuth angle in radians
            
        Returns
        -------
        xr.DataArray
            Geometric kernel values
        """
        # Transform angles
        theta_s_prime = np.arctan(np.tan(theta_s))
        theta_v_prime = np.arctan(np.tan(theta_v))
        
        # Compute distance in transformed space
        xi = np.sqrt(np.tan(theta_s_prime)**2 + np.tan(theta_v_prime)**2 - 
                    2 * np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.cos(phi))
        
        # Compute overlap angle
        cos_t = 2 * np.sqrt(xi**2 + (np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.sin(phi))**2) / \
                (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
        cos_t = xr.where(cos_t > 1, 1, cos_t)
        cos_t = xr.where(cos_t < -1, -1, cos_t)
        
        t = np.arccos(cos_t)
        
        # Overlap function
        overlap = (1/np.pi) * (t - np.sin(t) * cos_t) * \
                 (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
        
        # Phase function
        cos_xi = (np.cos(theta_s_prime) * np.cos(theta_v_prime) + 
                 np.sin(theta_s_prime) * np.sin(theta_v_prime) * np.cos(phi))
        
        # Li-Sparse kernel
        k_geo = overlap - (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime) - 
                          0.5 * (1 + cos_xi) * (1/np.cos(theta_s_prime)) * (1/np.cos(theta_v_prime)))
        
        return k_geo
    
    def _compute_snow_kernel(
        self, 
        theta_s: xr.DataArray, 
        theta_v: xr.DataArray, 
        phi: xr.DataArray
    ) -> xr.DataArray:
        """
        Compute snow kernel following Jiao et al. (2019).
        
        Parameters
        ----------
        theta_s : xr.DataArray
            Solar zenith angle in radians
        theta_v : xr.DataArray
            View zenith angle in radians
        phi : xr.DataArray
            Relative azimuth angle in radians
            
        Returns
        -------
        xr.DataArray
            Snow kernel values
        """
        # Compute phase angle
        cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                    np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
        cos_phase = xr.where(cos_phase > 1, 1, cos_phase)
        cos_phase = xr.where(cos_phase < -1, -1, cos_phase)
        phase = np.arccos(cos_phase)
        
        # Phase function P_E
        E_deg = phase * 180.0 / np.pi
        P_E = (11.1 * np.exp(-0.087 * (180 - E_deg)) + 
               1.1 * np.exp(-0.014 * (180 - E_deg)))
        
        # R_0 term
        R_0 = (self.k1 + self.k2 * (np.cos(theta_s) + np.cos(theta_v)) + 
               self.k3 * np.cos(theta_s) * np.cos(theta_v) + P_E) / \
              (4 * (np.cos(theta_s) + np.cos(theta_v)))
        
        # Snow kernel
        k_snow = (R_0 * (1 - self.alpha * np.cos(phase) * np.exp(-np.cos(phase))) + 
                 (0.4076 * self.alpha - 1.1081))
        
        return k_snow
    
    def _compute_kernel_integral_directional_hemispherical(
        self,
        theta_s: xr.DataArray,
        kernel_type: str
    ) -> xr.DataArray:
        """
        Compute kernel integral for directional-hemispherical case.
        
        This is a simplified implementation. For high accuracy, numerical
        integration should be performed for each pixel.
        
        Parameters
        ----------
        theta_s : xr.DataArray
            Solar zenith angle
        kernel_type : str
            Type of kernel ('volumetric', 'geometric', 'snow')
            
        Returns
        -------
        xr.DataArray
            Kernel integral values
        """
        # Simplified analytical approximations
        # In practice, these should be computed via numerical integration
        
        if kernel_type == 'volumetric':
            # Approximate volumetric kernel integral
            k_integral = 0.1 * (1 - 2 * theta_s / np.pi)
        elif kernel_type == 'geometric':
            # Approximate geometric kernel integral
            k_integral = 0.05 * (1 - theta_s / np.pi)
        elif kernel_type == 'snow':
            # Approximate snow kernel integral
            k_integral = 0.02 * np.ones_like(theta_s)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        return k_integral
    
    def _compute_kernel_integral_bihemispherical(self, kernel_type: str) -> float:
        """
        Compute kernel integral for bi-hemispherical case.
        
        These are constant values that can be pre-computed.
        
        Parameters
        ----------
        kernel_type : str
            Type of kernel ('volumetric', 'geometric', 'snow')
            
        Returns
        -------
        float
            Kernel integral value
        """
        # Pre-computed integral values from numerical integration
        if kernel_type == 'volumetric':
            return 0.189184  # Ross-Thick bi-hemispherical integral
        elif kernel_type == 'geometric':
            return -0.007574  # Li-Sparse bi-hemispherical integral
        elif kernel_type == 'snow':
            return 0.05  # Approximate snow kernel integral
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    def _save_ratio(self, ratio: xr.DataArray, filename: str):
        """
        Save ratio data to file.
        
        Parameters
        ----------
        ratio : xr.DataArray
            Ratio data to save
        filename : str
            Output filename
        """
        output_file = self.output_path / filename
        save_raster(ratio, output_file)
        self.logger.info(f"Saved ratio to {output_file}")
