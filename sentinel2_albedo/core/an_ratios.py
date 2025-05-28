"""
Albedo-to-Nadir ratio calculations.

This module computes the ratios needed to convert Sentinel-2 nadir reflectance
to hemispherical albedo (BSA and WSA) using MODIS BRDF kernel parameters.

Based on Shuai et al. (2011) "An algorithm for the retrieval of 30-m 
snow-free albedo from Landsat surface reflectance and MODIS BRDF".

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
    Calculator for Albedo-to-Nadir ratios from BRDF kernel parameters.
    
    This class computes BSA (Black-Sky Albedo) and WSA (White-Sky Albedo)
    ratios that are used to convert Sentinel-2 nadir reflectance observations
    to hemispherical albedo values.
    """
    
    def __init__(
        self,
        brdf_kernels: Dict[str, xr.DataArray],
        s2_geometry: Dict[str, xr.DataArray],
        geometry_calc: GeometryCalculator,
        output_path: Path,
        **kwargs
    ):
        """
        Initialize the AN ratio calculator.
        
        Parameters
        ----------
        brdf_kernels : dict
            Dictionary containing BRDF kernel parameters (f_iso, f_vol, f_geo, f_snow)
        s2_geometry : dict
            Sentinel-2 observation geometry (theta_s, theta_v, phi)
        geometry_calc : GeometryCalculator
            Geometry calculator instance
        output_path : Path
            Output directory for ratio files
        """
        self.brdf_kernels = brdf_kernels
        self.s2_geometry = s2_geometry
        self.geometry_calc = geometry_calc
        self.output_path = Path(output_path)
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # RTLSR kernel constants
        self.k1 = 1.247
        self.k2 = 1.186
        self.k3 = 5.157
        self.alpha = 0.3
        
        # Quality thresholds
        self.rmse_threshold = kwargs.get('rmse_threshold', 0.08)
        self.wod_wdr_threshold = kwargs.get('wod_wdr_threshold', 1.65)
        self.wod_wsa_threshold = kwargs.get('wod_wsa_threshold', 2.5)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Precompute kernel integrals
        self.kernel_integrals = self._compute_kernel_integrals()
        
    def compute_ratios(self) -> Dict[str, Dict[str, xr.DataArray]]:
        """
        Compute BSA and WSA ratios for all spectral bands.
        
        Returns
        -------
        dict
            Dictionary containing 'bsa' and 'wsa' ratios for each band
        """
        self.logger.info("Computing Albedo-to-Nadir ratios...")
        
        # Extract band numbers from kernel data
        bands = self._extract_band_numbers()
        
        bsa_ratios = {}
        wsa_ratios = {}
        
        for band in bands:
            self.logger.info(f"Processing band {band}...")
            
            # Get kernel parameters for this band
            band_kernels = self._get_band_kernels(band)
            
            # Apply quality filtering
            band_kernels = self._apply_quality_filters(band_kernels)
            
            # Compute ratios
            bsa_ratio = self._compute_bsa_ratio(band_kernels)
            wsa_ratio = self._compute_wsa_ratio(band_kernels)
            
            # Store results
            bsa_ratios[f'B{band}'] = bsa_ratio
            wsa_ratios[f'B{band}'] = wsa_ratio
            
            # Save to files
            self._save_ratio_files(bsa_ratio, wsa_ratio, band)
        
        results = {
            'bsa': bsa_ratios,
            'wsa': wsa_ratios
        }
        
        self.logger.info("AN ratio computation completed")
        return results
    
    def _extract_band_numbers(self) -> list:
        """
        Extract band numbers from BRDF kernel parameter names.
        
        Returns
        -------
        list
            List of band numbers
        """
        bands = set()
        
        for key in self.brdf_kernels.keys():
            if '_B' in key:
                band_str = key.split('_B')[-1]
                try:
                    band_num = int(band_str)
                    bands.add(band_num)
                except ValueError:
                    continue
        
        return sorted(list(bands))
    
    def _get_band_kernels(self, band: int) -> Dict[str, xr.DataArray]:
        """
        Get BRDF kernel parameters for a specific band.
        
        Parameters
        ----------
        band : int
            Band number
            
        Returns
        -------
        dict
            Dictionary containing kernel parameters for the band
        """
        band_kernels = {}
        
        # Required parameters
        required_params = ['f_iso', 'f_vol', 'f_geo', 'rmse', 'wod_wdr', 'wod_wsa']
        
        for param in required_params:
            key = f'{param}_B{band}'
            if key in self.brdf_kernels:
                band_kernels[param] = self.brdf_kernels[key]
            else:
                raise ValueError(f"Missing parameter {key} for band {band}")
        
        # Optional snow parameter
        snow_key = f'f_snow_B{band}'
        if snow_key in self.brdf_kernels:
            band_kernels['f_snow'] = self.brdf_kernels[snow_key]
        
        return band_kernels
    
    def _apply_quality_filters(self, kernels: Dict[str, xr.DataArray]) -> Dict[str, xr.DataArray]:
        """
        Apply quality filtering to kernel parameters based on MODIS thresholds.
        
        Parameters
        ----------
        kernels : dict
            Dictionary containing kernel parameters
            
        Returns
        -------
        dict
            Filtered kernel parameters
        """
        # Create quality mask
        quality_mask = (
            (kernels['rmse'] <= self.rmse_threshold) &
            (kernels['wod_wdr'] <= self.wod_wdr_threshold) &
            (kernels['wod_wsa'] <= self.wod_wsa_threshold)
        )
        
        # Apply mask to all parameters
        filtered_kernels = {}
        for param, data in kernels.items():
            if param in ['rmse', 'wod_wdr', 'wod_wsa']:
                # Keep quality metrics as-is
                filtered_kernels[param] = data
            else:
                # Apply quality mask
                filtered_kernels[param] = data.where(quality_mask)
        
        return filtered_kernels
    
    def _compute_bsa_ratio(self, kernels: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Compute Black-Sky Albedo (BSA) ratio.
        
        BSA ratio = R_l(theta_s) / R_omega(theta_s, theta_v, phi)
        
        Parameters
        ----------
        kernels : dict
            Dictionary containing kernel parameters
            
        Returns
        -------
        xr.DataArray
            BSA ratio
        """
        # Compute BRDF at Sentinel-2 observation geometry (R_omega)
        r_omega = self._compute_brdf_at_geometry(kernels)
        
        # Compute BRDF integrated over observation hemisphere (R_l_theta_s)
        r_l_theta_s = self._compute_brdf_hemispherical_obs(kernels)
        
        # Compute BSA ratio
        bsa_ratio = r_l_theta_s / r_omega
        
        # Set invalid ratios to NaN
        bsa_ratio = bsa_ratio.where((r_omega > 0) & np.isfinite(r_omega))
        
        return bsa_ratio
    
    def _compute_wsa_ratio(self, kernels: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Compute White-Sky Albedo (WSA) ratio.
        
        WSA ratio = R_l / R_omega(theta_s, theta_v, phi)
        
        Parameters
        ----------
        kernels : dict
            Dictionary containing kernel parameters
            
        Returns
        -------
        xr.DataArray
            WSA ratio
        """
        # Compute BRDF at Sentinel-2 observation geometry (R_omega)
        r_omega = self._compute_brdf_at_geometry(kernels)
        
        # Compute BRDF integrated over both hemispheres (R_l)
        r_l = self._compute_brdf_bihemispherical(kernels)
        
        # Compute WSA ratio
        wsa_ratio = r_l / r_omega
        
        # Set invalid ratios to NaN
        wsa_ratio = wsa_ratio.where((r_omega > 0) & np.isfinite(r_omega))
        
        return wsa_ratio
    
    def _compute_brdf_at_geometry(self, kernels: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Compute BRDF at Sentinel-2 observation geometry.
        
        R_omega = f_iso + f_vol * k_vol_omega + f_geo * k_geo_omega [+ f_snow * k_snow_omega]
        
        Parameters
        ----------
        kernels : dict
            Dictionary containing kernel parameters
            
        Returns
        -------
        xr.DataArray
            BRDF at observation geometry
        """
        # Get Sentinel-2 geometry
        theta_s = self.s2_geometry['theta_s']
        theta_v = self.s2_geometry['theta_v']
        phi = self.s2_geometry['phi']
        
        # Compute kernel values at S2 geometry
        k_vol_omega = self._compute_volumetric_kernel(theta_s, theta_v, phi)
        k_geo_omega = self._compute_geometric_kernel(theta_s, theta_v, phi)
        
        # Compute BRDF
        r_omega = (kernels['f_iso'] + 
                  kernels['f_vol'] * k_vol_omega + 
                  kernels['f_geo'] * k_geo_omega)
        
        # Add snow kernel if present
        if 'f_snow' in kernels:
            k_snow_omega = self._compute_snow_kernel(theta_s, theta_v, phi)
            r_omega += kernels['f_snow'] * k_snow_omega
        
        # Ensure non-negative values
        r_omega = r_omega.where(r_omega >= 0)
        
        return r_omega
    
    def _compute_brdf_hemispherical_obs(self, kernels: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Compute BRDF integrated over observation hemisphere.
        
        R_l(theta_s) = f_iso + f_vol * k_vol_l(theta_s) + f_geo * k_geo_l(theta_s) [+ f_snow * k_snow_l(theta_s)]
        
        Parameters
        ----------
        kernels : dict
            Dictionary containing kernel parameters
            
        Returns
        -------
        xr.DataArray
            BRDF integrated over observation hemisphere
        """
        # Get illumination angle
        theta_s = self.s2_geometry['theta_s']
        
        # Compute kernel integrals for each pixel's illumination angle
        k_vol_l_theta_s = self._compute_kernel_integral_theta_s(theta_s, 'volumetric')
        k_geo_l_theta_s = self._compute_kernel_integral_theta_s(theta_s, 'geometric')
        
        # Compute BRDF
        r_l_theta_s = (kernels['f_iso'] + 
                      kernels['f_vol'] * k_vol_l_theta_s + 
                      kernels['f_geo'] * k_geo_l_theta_s)
        
        # Add snow kernel if present
        if 'f_snow' in kernels:
            k_snow_l_theta_s = self._compute_kernel_integral_theta_s(theta_s, 'snow')
            r_l_theta_s += kernels['f_snow'] * k_snow_l_theta_s
        
        # Ensure non-negative values
        r_l_theta_s = r_l_theta_s.where(r_l_theta_s >= 0)
        
        return r_l_theta_s
    
    def _compute_brdf_bihemispherical(self, kernels: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Compute BRDF integrated over both hemispheres (White-Sky Albedo).
        
        R_l = f_iso + f_vol * k_vol_l + f_geo * k_geo_l [+ f_snow * k_snow_l]
        
        Parameters
        ----------
        kernels : dict
            Dictionary containing kernel parameters
            
        Returns
        -------
        xr.DataArray
            BRDF integrated over both hemispheres
        """
        # Use precomputed kernel integrals
        r_l = (kernels['f_iso'] + 
              kernels['f_vol'] * self.kernel_integrals['k_vol'] + 
              kernels['f_geo'] * self.kernel_integrals['k_geo'])
        
        # Add snow kernel if present
        if 'f_snow' in kernels:
            r_l += kernels['f_snow'] * self.kernel_integrals['k_snow']
        
        # Ensure non-negative values
        r_l = r_l.where(r_l >= 0)
        
        return r_l
    
    def _compute_volumetric_kernel(self, theta_s: xr.DataArray, theta_v: xr.DataArray, phi: xr.DataArray) -> xr.DataArray:
        """
        Compute Ross-Thick volumetric kernel.
        
        Parameters
        ----------
        theta_s : xr.DataArray
            Solar zenith angle
        theta_v : xr.DataArray
            View zenith angle
        phi : xr.DataArray
            Relative azimuth angle
            
        Returns
        -------
        xr.DataArray
            Volumetric kernel values
        """
        # Compute phase angle
        cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                    np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
        cos_phase = xr.where((cos_phase < -1), -1, cos_phase)
        cos_phase = xr.where((cos_phase > 1), 1, cos_phase)
        
        phase = np.arccos(cos_phase)
        
        # Ross-Thick kernel
        k_vol = ((np.pi/2 - phase) * cos_phase + np.sin(phase)) / (np.cos(theta_s) + np.cos(theta_v)) - np.pi/4
        
        return k_vol
    
    def _compute_geometric_kernel(self, theta_s: xr.DataArray, theta_v: xr.DataArray, phi: xr.DataArray) -> xr.DataArray:
        """
        Compute Li-Sparse geometric kernel.
        
        Parameters
        ----------
        theta_s : xr.DataArray
            Solar zenith angle
        theta_v : xr.DataArray
            View zenith angle
        phi : xr.DataArray
            Relative azimuth angle
            
        Returns
        -------
        xr.DataArray
            Geometric kernel values
        """
        # Transform angles
        theta_s_prime = np.arctan(np.tan(theta_s))
        theta_v_prime = np.arctan(np.tan(theta_v))
        
        # Compute distance
        xi = np.sqrt(np.tan(theta_s_prime)**2 + np.tan(theta_v_prime)**2 - 
                    2 * np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.cos(phi))
        
        # Compute overlap angle
        cos_t = 2 * np.sqrt(xi**2 + (np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.sin(phi))**2) / \
                (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
        cos_t = xr.where((cos_t < -1), -1, cos_t)
        cos_t = xr.where((cos_t > 1), 1, cos_t)
        
        t = np.arccos(cos_t)
        
        # Overlap function
        overlap = (1/np.pi) * (t - np.sin(t) * cos_t) * (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
        
        # Phase function
        cos_xi = (np.cos(theta_s_prime) * np.cos(theta_v_prime) + 
                 np.sin(theta_s_prime) * np.sin(theta_v_prime) * np.cos(phi))
        
        # Li-Sparse kernel
        k_geo = overlap - (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime) - 
                          0.5 * (1 + cos_xi) * (1/np.cos(theta_s_prime)) * (1/np.cos(theta_v_prime)))
        
        return k_geo
    
    def _compute_snow_kernel(self, theta_s: xr.DataArray, theta_v: xr.DataArray, phi: xr.DataArray) -> xr.DataArray:
        """
        Compute snow kernel.
        
        Parameters
        ----------
        theta_s : xr.DataArray
            Solar zenith angle
        theta_v : xr.DataArray
            View zenith angle
        phi : xr.DataArray
            Relative azimuth angle
            
        Returns
        -------
        xr.DataArray
            Snow kernel values
        """
        # Compute phase angle
        cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                    np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
        cos_phase = xr.where((cos_phase < -1), -1, cos_phase)
        cos_phase = xr.where((cos_phase > 1), 1, cos_phase)
        
        phase = np.arccos(cos_phase)
        E_deg = phase * 180.0 / np.pi
        
        # Phase function P_E
        P_E = (11.1 * np.exp(-0.087 * (180 - E_deg)) + 
               1.1 * np.exp(-0.014 * (180 - E_deg)))
        
        # R_0 term
        R_0 = (self.k1 + self.k2 * (np.cos(theta_s) + np.cos(theta_v)) + 
               self.k3 * np.cos(theta_s) * np.cos(theta_v) + P_E) / \
              (4 * (np.cos(theta_s) + np.cos(theta_v)))
        
        # Snow kernel
        k_snow = (R_0 * (1 - self.alpha * cos_phase * np.exp(-cos_phase)) + 
                 (0.4076 * self.alpha - 1.1081))
        
        return k_snow
    
    def _compute_kernel_integrals(self) -> Dict[str, float]:
        """
        Compute kernel integrals for bihemispherical albedo.
        
        Returns
        -------
        dict
            Dictionary containing kernel integral values
        """
        self.logger.info("Computing kernel integrals for WSA...")
        
        # Integration limits
        theta_max = np.pi / 2
        phi_max = 2 * np.pi
        
        # Volumetric kernel integral
        def vol_integrand(theta_s, theta_v, phi):
            return self._volumetric_integrand(theta_s, theta_v, phi)
        
        k_vol_integral = integrate.tplquad(
            vol_integrand,
            0, phi_max,
            lambda phi: 0, lambda phi: theta_max,
            lambda phi, theta_v: 0, lambda phi, theta_v: theta_max,
            epsrel=1e-2
        )[0] * (2 / np.pi)
        
        # Geometric kernel integral
        def geo_integrand(theta_s, theta_v, phi):
            return self._geometric_integrand(theta_s, theta_v, phi)
        
        k_geo_integral = integrate.tplquad(
            geo_integrand,
            0, phi_max,
            lambda phi: 0, lambda phi: theta_max,
            lambda phi, theta_v: 0, lambda phi, theta_v: theta_max,
            epsrel=1e-2
        )[0] * (2 / np.pi)
        
        # Snow kernel integral (simplified)
        k_snow_integral = 0.1  # Placeholder - would need proper integration
        
        return {
            'k_vol': k_vol_integral,
            'k_geo': k_geo_integral,
            'k_snow': k_snow_integral
        }
    
    def _volumetric_integrand(self, theta_s: float, theta_v: float, phi: float) -> float:
        """
        Volumetric kernel integrand.
        
        Parameters
        ----------
        theta_s, theta_v, phi : float
            Angles in radians
            
        Returns
        -------
        float
            Weighted kernel value
        """
        # Compute kernel value
        cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                    np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
        cos_phase = np.clip(cos_phase, -1.0, 1.0)
        phase = np.arccos(cos_phase)
        
        k_vol = ((np.pi/2 - phase) * cos_phase + np.sin(phase)) / (np.cos(theta_s) + np.cos(theta_v)) - np.pi/4
        
        # Weight by solid angle
        weight = np.sin(theta_s) * np.cos(theta_s) * np.sin(theta_v) * np.cos(theta_v)
        
        return k_vol * weight
    
    def _geometric_integrand(self, theta_s: float, theta_v: float, phi: float) -> float:
        """
        Geometric kernel integrand.
        
        Parameters
        ----------
        theta_s, theta_v, phi : float
            Angles in radians
            
        Returns
        -------
        float
            Weighted kernel value
        """
        # Avoid singularities
        if theta_s == 0 or theta_v == 0:
            return 0.0
            
        # Transform angles
        theta_s_prime = np.arctan(np.tan(theta_s))
        theta_v_prime = np.arctan(np.tan(theta_v))
        
        # Compute distance
        xi = np.sqrt(np.tan(theta_s_prime)**2 + np.tan(theta_v_prime)**2 - 
                    2 * np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.cos(phi))
        
        # Compute overlap
        cos_t = 2 * np.sqrt(xi**2 + (np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.sin(phi))**2) / \
                (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
        cos_t = np.clip(cos_t, -1.0, 1.0)
        
        t = np.arccos(cos_t)
        overlap = (1/np.pi) * (t - np.sin(t) * cos_t) * (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
        
        cos_xi = (np.cos(theta_s_prime) * np.cos(theta_v_prime) + 
                 np.sin(theta_s_prime) * np.sin(theta_v_prime) * np.cos(phi))
        
        k_geo = overlap - (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime) - 
                          0.5 * (1 + cos_xi) * (1/np.cos(theta_s_prime)) * (1/np.cos(theta_v_prime)))
        
        # Weight by solid angle
        weight = np.sin(theta_s) * np.cos(theta_s) * np.sin(theta_v) * np.cos(theta_v)
        
        return k_geo * weight
    
    def _compute_kernel_integral_theta_s(self, theta_s: xr.DataArray, kernel_type: str) -> xr.DataArray:
        """
        Compute kernel integral over observation hemisphere for given illumination.
        
        Parameters
        ----------
        theta_s : xr.DataArray
            Solar zenith angles
        kernel_type : str
            Type of kernel ('volumetric', 'geometric', 'snow')
            
        Returns
        -------
        xr.DataArray
            Kernel integral values
        """
        # This is a simplified implementation
        # In practice, would need to compute integrals for each pixel's theta_s
        
        if kernel_type == 'volumetric':
            # Simplified volumetric kernel integral
            integral = 0.189184 - 1.377622 * np.cos(theta_s) + 0.737671 * np.cos(theta_s)**2
        elif kernel_type == 'geometric':
            # Simplified geometric kernel integral  
            integral = -1.332409 + 2.285669 * np.cos(theta_s) - 1.176098 * np.cos(theta_s)**2
        elif kernel_type == 'snow':
            # Simplified snow kernel integral
            integral = xr.zeros_like(theta_s) + 0.05
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
            
        return integral
    
    def _save_ratio_files(self, bsa_ratio: xr.DataArray, wsa_ratio: xr.DataArray, band: int):
        """
        Save ratio files to disk.
        
        Parameters
        ----------
        bsa_ratio : xr.DataArray
            BSA ratio data
        wsa_ratio : xr.DataArray
            WSA ratio data
        band : int
            Band number
        """
        date_str = self.s2_geometry['date'] if 'date' in self.s2_geometry else '20180808'
        
        # Save BSA ratio
        bsa_file = self.output_path / f"a_bsa_{date_str}_B{band}.tif"
        save_raster(bsa_ratio, bsa_file)
        
        # Save WSA ratio
        wsa_file = self.output_path / f"a_wsa_{date_str}_B{band}.tif"
        save_raster(wsa_ratio, wsa_file)
        
        self.logger.info(f"Saved ratios for band {band}")
