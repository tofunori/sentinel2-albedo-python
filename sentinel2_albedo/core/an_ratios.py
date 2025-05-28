"""
Albedo-to-Nadir (AN) ratio calculations.

This module computes the ratios needed to convert Sentinel-2 nadir reflectance
to hemispherical albedo (BSA and WSA), following the methodology from
Shuai et al. (2011).

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
    Calculator for Albedo-to-Nadir ratios.
    
    This class computes BSA (Black-Sky Albedo) and WSA (White-Sky Albedo) ratios
    that are used to convert Sentinel-2 nadir reflectance to hemispherical albedo.
    
    The ratios are computed by integrating BRDF over different hemispheres:
    - BSA: Integration over observation hemisphere for fixed illumination
    - WSA: Integration over both illumination and observation hemispheres
    
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
        
        # Pre-compute kernel integrals
        self._precompute_kernel_integrals()
    
    def _precompute_kernel_integrals(self):
        """
        Pre-compute kernel integrals needed for albedo calculations.
        
        These integrals are used for both BSA and WSA calculations and are
        computationally expensive, so we compute them once.
        """
        self.logger.info("Pre-computing kernel integrals...")
        
        # Integration limits
        theta_max = np.pi / 2  # 90 degrees
        phi_max = 2 * np.pi    # 360 degrees
        
        # Volumetric kernel integral (full hemisphere)
        self.k_vol_wsa = integrate.tplquad(
            self._k_vol_integrand_wsa,
            0, phi_max,
            lambda phi: 0, lambda phi: theta_max,
            lambda phi, theta_v: 0, lambda phi, theta_v: theta_max,
            epsrel=1e-2
        )[0] * (2 / np.pi)
        
        # Geometric kernel integral (full hemisphere)
        self.k_geo_wsa = integrate.tplquad(
            self._k_geo_integrand_wsa,
            0, phi_max,
            lambda phi: 0, lambda phi: theta_max,
            lambda phi, theta_v: 0, lambda phi, theta_v: theta_max,
            epsrel=1e-2
        )[0] * (2 / np.pi)
        
        # Snow kernel integral (if applicable)
        if self._has_snow_kernel():
            self.k_snow_wsa = integrate.tplquad(
                self._k_snow_integrand_wsa,
                0, phi_max,
                lambda phi: 0, lambda phi: theta_max,
                lambda phi, theta_v: 0, lambda phi, theta_v: theta_max,
                epsrel=1e-2
            )[0] * (2 / np.pi)
        
        self.logger.info(f"Kernel integrals computed: k_vol={self.k_vol_wsa:.4f}, k_geo={self.k_geo_wsa:.4f}")
    
    def _has_snow_kernel(self) -> bool:
        """
        Check if snow kernel parameters are available.
        
        Returns
        -------
        bool
            True if snow kernel parameters exist
        """
        return any('f_snow' in key for key in self.brdf_kernels.keys())
    
    def _k_vol_integrand_wsa(self, theta_s: float, theta_v: float, phi: float) -> float:
        """
        Volumetric kernel integrand for WSA calculation.
        
        Parameters
        ----------
        theta_s : float
            Solar zenith angle
        theta_v : float
            View zenith angle
        phi : float
            Relative azimuth angle
            
        Returns
        -------
        float
            Weighted kernel value
        """
        # Compute phase angle
        cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                    np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
        cos_phase = np.clip(cos_phase, -1.0, 1.0)
        phase = np.arccos(cos_phase)
        
        # Volumetric kernel (Ross-Thick)
        k_vol = ((np.pi/2 - phase) * cos_phase + np.sin(phase)) / (np.cos(theta_s) + np.cos(theta_v)) - np.pi/4
        
        # Weight by solid angle
        weight = np.sin(theta_s) * np.cos(theta_s) * np.sin(theta_v) * np.cos(theta_v)
        
        return k_vol * weight
    
    def _k_geo_integrand_wsa(self, theta_s: float, theta_v: float, phi: float) -> float:
        """
        Geometric kernel integrand for WSA calculation.
        
        Parameters
        ----------
        theta_s : float
            Solar zenith angle
        theta_v : float
            View zenith angle
        phi : float
            Relative azimuth angle
            
        Returns
        -------
        float
            Weighted kernel value
        """
        # Compute Li-Sparse kernel
        theta_s_prime = np.arctan(np.tan(theta_s))
        theta_v_prime = np.arctan(np.tan(theta_v))
        
        xi = np.sqrt(np.tan(theta_s_prime)**2 + np.tan(theta_v_prime)**2 - 
                    2 * np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.cos(phi))
        
        cos_t = 2 * np.sqrt(xi**2 + (np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.sin(phi))**2) / \
                (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
        cos_t = np.clip(cos_t, -1.0, 1.0)
        
        t = np.arccos(cos_t)
        
        overlap = (1/np.pi) * (t - np.sin(t) * cos_t) * (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
        
        cos_phase = (np.cos(theta_s_prime) * np.cos(theta_v_prime) + 
                    np.sin(theta_s_prime) * np.sin(theta_v_prime) * np.cos(phi))
        
        k_geo = overlap - (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime) - 
                          0.5 * (1 + cos_phase) * (1/np.cos(theta_s_prime)) * (1/np.cos(theta_v_prime)))
        
        # Weight by solid angle
        weight = np.sin(theta_s) * np.cos(theta_s) * np.sin(theta_v) * np.cos(theta_v)
        
        return k_geo * weight
    
    def _k_snow_integrand_wsa(self, theta_s: float, theta_v: float, phi: float) -> float:
        """
        Snow kernel integrand for WSA calculation.
        
        Parameters
        ----------
        theta_s : float
            Solar zenith angle
        theta_v : float
            View zenith angle
        phi : float
            Relative azimuth angle
            
        Returns
        -------
        float
            Weighted kernel value
        """
        # Compute phase angle
        cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                    np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
        cos_phase = np.clip(cos_phase, -1.0, 1.0)
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
        
        # Weight by solid angle
        weight = np.sin(theta_s) * np.cos(theta_s) * np.sin(theta_v) * np.cos(theta_v)
        
        return k_snow * weight
    
    def compute_ratios(self) -> Dict[str, Dict[str, xr.DataArray]]:
        """
        Compute BSA and WSA ratios for all bands.
        
        Returns
        -------
        dict
            Dictionary containing 'bsa' and 'wsa' ratios for each band
        """
        self.logger.info("Computing Albedo-to-Nadir ratios...")
        
        # Identify bands to process
        bands = self._identify_bands()
        self.logger.info(f"Processing {len(bands)} bands: {bands}")
        
        # Initialize results
        bsa_ratios = {}
        wsa_ratios = {}
        
        # Process each band
        for band in tqdm(bands, desc="Computing AN ratios"):
            self.logger.info(f"Processing band {band}...")
            
            # Apply quality filters
            filtered_kernels = self._apply_quality_filters(band)
            
            # Compute ratios for this band
            bsa_ratio = self._compute_bsa_ratio(filtered_kernels, band)
            wsa_ratio = self._compute_wsa_ratio(filtered_kernels, band)
            
            # Store results
            bsa_ratios[band] = bsa_ratio
            wsa_ratios[band] = wsa_ratio
            
            # Save to files
            self._save_ratio_results(bsa_ratio, wsa_ratio, band)
        
        results = {
            'bsa': bsa_ratios,
            'wsa': wsa_ratios
        }
        
        self.logger.info("AN ratio computation completed")
        return results
    
    def _identify_bands(self) -> List[str]:
        """
        Identify available bands from BRDF kernel data.
        
        Returns
        -------
        list
            List of band identifiers
        """
        bands = set()
        
        for key in self.brdf_kernels.keys():
            if '_B' in key:
                # Extract band number (e.g., 'f_iso_B1' -> 'B1')
                band = key.split('_B')[-1]
                bands.add(f'B{band}')
        
        return sorted(list(bands))
    
    def _apply_quality_filters(self, band: str) -> Dict[str, xr.DataArray]:
        """
        Apply quality filters to BRDF kernel parameters.
        
        Parameters
        ----------
        band : str
            Band identifier (e.g., 'B1')
            
        Returns
        -------
        dict
            Quality-filtered kernel parameters
        """
        filtered_kernels = {}
        
        # Get quality metrics
        rmse_key = f'rmse_{band}'
        wod_wdr_key = f'wod_wdr_{band}'
        wod_wsa_key = f'wod_wsa_{band}'
        
        if all(key in self.brdf_kernels for key in [rmse_key, wod_wdr_key, wod_wsa_key]):
            # Create quality mask
            quality_mask = (
                (self.brdf_kernels[rmse_key] <= self.rmse_threshold) &
                (self.brdf_kernels[wod_wdr_key] <= self.wod_wdr_threshold) &
                (self.brdf_kernels[wod_wsa_key] <= self.wod_wsa_threshold)
            )
            
            # Apply mask to kernel parameters
            for param in ['f_iso', 'f_vol', 'f_geo', 'f_snow']:
                param_key = f'{param}_{band}'
                if param_key in self.brdf_kernels:
                    filtered_kernels[param] = self.brdf_kernels[param_key].where(quality_mask)
        else:
            # No quality filtering if metrics are missing
            self.logger.warning(f"Quality metrics missing for band {band}, skipping filtering")
            for param in ['f_iso', 'f_vol', 'f_geo', 'f_snow']:
                param_key = f'{param}_{band}'
                if param_key in self.brdf_kernels:
                    filtered_kernels[param] = self.brdf_kernels[param_key]
        
        return filtered_kernels
    
    def _compute_bsa_ratio(self, kernels: Dict[str, xr.DataArray], band: str) -> xr.DataArray:
        """
        Compute Black-Sky Albedo (BSA) ratio.
        
        BSA ratio converts nadir reflectance to directional-hemispherical reflectance
        (black-sky albedo) for the given solar illumination geometry.
        
        Parameters
        ----------
        kernels : dict
            Quality-filtered kernel parameters
        band : str
            Band identifier
            
        Returns
        -------
        xr.DataArray
            BSA ratio array
        """
        # Get Sentinel-2 geometry for this computation
        theta_s = self.s2_geometry['theta_s']
        
        # Compute BSA kernel integrals for each solar zenith angle
        bsa_ratio = xr.zeros_like(kernels['f_iso'])
        
        # Process each pixel
        theta_s_values = theta_s.values.flatten()
        unique_theta_s = np.unique(theta_s_values[~np.isnan(theta_s_values)])
        
        for theta_s_val in unique_theta_s:
            # Find pixels with this solar zenith angle
            mask = np.abs(theta_s - theta_s_val) < 1e-6
            
            if mask.sum() > 0:
                # Compute BSA kernel integrals for this solar zenith
                k_vol_bsa = self._compute_bsa_kernel_integral(theta_s_val, 'volumetric')
                k_geo_bsa = self._compute_bsa_kernel_integral(theta_s_val, 'geometric')
                
                # Compute BRDF reflectance at nadir view
                R_omega = self._compute_nadir_brdf(kernels, theta_s_val)
                
                # Compute BSA reflectance
                R_bsa = (kernels['f_iso'] + 
                        kernels['f_vol'] * k_vol_bsa + 
                        kernels['f_geo'] * k_geo_bsa)
                
                if 'f_snow' in kernels:
                    k_snow_bsa = self._compute_bsa_kernel_integral(theta_s_val, 'snow')
                    R_bsa = R_bsa + kernels['f_snow'] * k_snow_bsa
                
                # Compute ratio
                ratio = R_bsa / R_omega
                
                # Apply to masked pixels
                bsa_ratio = xr.where(mask, ratio, bsa_ratio)
        
        return bsa_ratio
    
    def _compute_wsa_ratio(self, kernels: Dict[str, xr.DataArray], band: str) -> xr.DataArray:
        """
        Compute White-Sky Albedo (WSA) ratio.
        
        WSA ratio converts nadir reflectance to bi-hemispherical reflectance
        (white-sky albedo) under isotropic illumination conditions.
        
        Parameters
        ----------
        kernels : dict
            Quality-filtered kernel parameters
        band : str
            Band identifier
            
        Returns
        -------
        xr.DataArray
            WSA ratio array
        """
        # Compute WSA reflectance using pre-computed integrals
        R_wsa = (kernels['f_iso'] + 
                kernels['f_vol'] * self.k_vol_wsa + 
                kernels['f_geo'] * self.k_geo_wsa)
        
        if 'f_snow' in kernels and hasattr(self, 'k_snow_wsa'):
            R_wsa = R_wsa + kernels['f_snow'] * self.k_snow_wsa
        
        # Compute nadir BRDF for reference
        theta_s = self.s2_geometry['theta_s']
        R_omega = self._compute_nadir_brdf_array(kernels, theta_s)
        
        # Compute ratio
        wsa_ratio = R_wsa / R_omega
        
        return wsa_ratio
    
    def _compute_bsa_kernel_integral(self, theta_s: float, kernel_type: str) -> float:
        """
        Compute BSA kernel integral for fixed solar zenith angle.
        
        Parameters
        ----------
        theta_s : float
            Solar zenith angle in radians
        kernel_type : str
            Type of kernel ('volumetric', 'geometric', 'snow')
            
        Returns
        -------
        float
            Kernel integral value
        """
        # Integration limits
        theta_v_max = np.pi / 2
        phi_max = 2 * np.pi
        
        if kernel_type == 'volumetric':
            integrand = lambda phi, theta_v: self._k_vol_bsa_integrand(theta_s, theta_v, phi)
        elif kernel_type == 'geometric':
            integrand = lambda phi, theta_v: self._k_geo_bsa_integrand(theta_s, theta_v, phi)
        elif kernel_type == 'snow':
            integrand = lambda phi, theta_v: self._k_snow_bsa_integrand(theta_s, theta_v, phi)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        integral = integrate.dblquad(
            integrand,
            0, theta_v_max,
            lambda theta_v: 0, lambda theta_v: phi_max,
            epsrel=1e-2
        )[0] * (1 / np.pi)
        
        return integral
    
    def _k_vol_bsa_integrand(self, theta_s: float, theta_v: float, phi: float) -> float:
        """
        Volumetric kernel integrand for BSA calculation.
        
        Parameters
        ----------
        theta_s : float
            Fixed solar zenith angle
        theta_v : float
            View zenith angle
        phi : float
            Relative azimuth angle
            
        Returns
        -------
        float
            Weighted kernel value
        """
        # Compute phase angle
        cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                    np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
        cos_phase = np.clip(cos_phase, -1.0, 1.0)
        phase = np.arccos(cos_phase)
        
        # Volumetric kernel
        k_vol = ((np.pi/2 - phase) * cos_phase + np.sin(phase)) / (np.cos(theta_s) + np.cos(theta_v)) - np.pi/4
        
        # Weight by view solid angle only
        weight = np.sin(theta_v) * np.cos(theta_v)
        
        return k_vol * weight
    
    def _k_geo_bsa_integrand(self, theta_s: float, theta_v: float, phi: float) -> float:
        """
        Geometric kernel integrand for BSA calculation.
        
        Parameters
        ----------
        theta_s : float
            Fixed solar zenith angle
        theta_v : float
            View zenith angle
        phi : float
            Relative azimuth angle
            
        Returns
        -------
        float
            Weighted kernel value
        """
        # Compute Li-Sparse kernel
        theta_s_prime = np.arctan(np.tan(theta_s))
        theta_v_prime = np.arctan(np.tan(theta_v))
        
        xi = np.sqrt(np.tan(theta_s_prime)**2 + np.tan(theta_v_prime)**2 - 
                    2 * np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.cos(phi))
        
        cos_t = 2 * np.sqrt(xi**2 + (np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.sin(phi))**2) / \
                (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
        cos_t = np.clip(cos_t, -1.0, 1.0)
        
        t = np.arccos(cos_t)
        
        overlap = (1/np.pi) * (t - np.sin(t) * cos_t) * (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
        
        cos_phase = (np.cos(theta_s_prime) * np.cos(theta_v_prime) + 
                    np.sin(theta_s_prime) * np.sin(theta_v_prime) * np.cos(phi))
        
        k_geo = overlap - (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime) - 
                          0.5 * (1 + cos_phase) * (1/np.cos(theta_s_prime)) * (1/np.cos(theta_v_prime)))
        
        # Weight by view solid angle only
        weight = np.sin(theta_v) * np.cos(theta_v)
        
        return k_geo * weight
    
    def _k_snow_bsa_integrand(self, theta_s: float, theta_v: float, phi: float) -> float:
        """
        Snow kernel integrand for BSA calculation.
        
        Parameters
        ----------
        theta_s : float
            Fixed solar zenith angle
        theta_v : float
            View zenith angle
        phi : float
            Relative azimuth angle
            
        Returns
        -------
        float
            Weighted kernel value
        """
        # Compute phase angle
        cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                    np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
        cos_phase = np.clip(cos_phase, -1.0, 1.0)
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
        
        # Weight by view solid angle only
        weight = np.sin(theta_v) * np.cos(theta_v)
        
        return k_snow * weight
    
    def _compute_nadir_brdf(self, kernels: Dict[str, xr.DataArray], theta_s: float) -> xr.DataArray:
        """
        Compute BRDF reflectance at nadir view for given solar zenith.
        
        Parameters
        ----------
        kernels : dict
            Kernel parameters
        theta_s : float
            Solar zenith angle in radians
            
        Returns
        -------
        xr.DataArray
            Nadir BRDF reflectance
        """
        # Nadir geometry
        theta_v = 0.0  # Nadir view
        phi = 0.0      # Arbitrary azimuth for nadir
        
        # Compute kernels for nadir geometry
        k_vol_nadir = self._compute_kernel_value(theta_s, theta_v, phi, 'volumetric')
        k_geo_nadir = self._compute_kernel_value(theta_s, theta_v, phi, 'geometric')
        
        # Compute BRDF
        R_nadir = (kernels['f_iso'] + 
                  kernels['f_vol'] * k_vol_nadir + 
                  kernels['f_geo'] * k_geo_nadir)
        
        if 'f_snow' in kernels:
            k_snow_nadir = self._compute_kernel_value(theta_s, theta_v, phi, 'snow')
            R_nadir = R_nadir + kernels['f_snow'] * k_snow_nadir
        
        return R_nadir
    
    def _compute_nadir_brdf_array(self, kernels: Dict[str, xr.DataArray], theta_s: xr.DataArray) -> xr.DataArray:
        """
        Compute BRDF reflectance at nadir view for array of solar zenith angles.
        
        Parameters
        ----------
        kernels : dict
            Kernel parameters
        theta_s : xr.DataArray
            Solar zenith angles in radians
            
        Returns
        -------
        xr.DataArray
            Nadir BRDF reflectance array
        """
        # Nadir geometry
        theta_v = xr.zeros_like(theta_s)  # Nadir view
        phi = xr.zeros_like(theta_s)      # Arbitrary azimuth
        
        # Compute kernels for nadir geometry
        k_vol_nadir = self._compute_kernel_array(theta_s, theta_v, phi, 'volumetric')
        k_geo_nadir = self._compute_kernel_array(theta_s, theta_v, phi, 'geometric')
        
        # Compute BRDF
        R_nadir = (kernels['f_iso'] + 
                  kernels['f_vol'] * k_vol_nadir + 
                  kernels['f_geo'] * k_geo_nadir)
        
        if 'f_snow' in kernels:
            k_snow_nadir = self._compute_kernel_array(theta_s, theta_v, phi, 'snow')
            R_nadir = R_nadir + kernels['f_snow'] * k_snow_nadir
        
        return R_nadir
    
    def _compute_kernel_value(self, theta_s: float, theta_v: float, phi: float, kernel_type: str) -> float:
        """
        Compute kernel value for specific geometry.
        
        Parameters
        ----------
        theta_s : float
            Solar zenith angle
        theta_v : float
            View zenith angle
        phi : float
            Relative azimuth angle
        kernel_type : str
            Type of kernel
            
        Returns
        -------
        float
            Kernel value
        """
        if kernel_type == 'volumetric':
            return self._k_vol_integrand_wsa(theta_s, theta_v, phi) / (np.sin(theta_s) * np.cos(theta_s) * np.sin(theta_v) * np.cos(theta_v))
        elif kernel_type == 'geometric':
            return self._k_geo_integrand_wsa(theta_s, theta_v, phi) / (np.sin(theta_s) * np.cos(theta_s) * np.sin(theta_v) * np.cos(theta_v))
        elif kernel_type == 'snow':
            return self._k_snow_integrand_wsa(theta_s, theta_v, phi) / (np.sin(theta_s) * np.cos(theta_s) * np.sin(theta_v) * np.cos(theta_v))
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    def _compute_kernel_array(self, theta_s: xr.DataArray, theta_v: xr.DataArray, phi: xr.DataArray, kernel_type: str) -> xr.DataArray:
        """
        Compute kernel values for arrays of geometry.
        
        Parameters
        ----------
        theta_s : xr.DataArray
            Solar zenith angles
        theta_v : xr.DataArray
            View zenith angles
        phi : xr.DataArray
            Relative azimuth angles
        kernel_type : str
            Type of kernel
            
        Returns
        -------
        xr.DataArray
            Kernel values
        """
        if kernel_type == 'volumetric':
            # Compute phase angle
            cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                        np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
            cos_phase = np.clip(cos_phase, -1.0, 1.0)
            phase = np.arccos(cos_phase)
            
            # Volumetric kernel
            k_vol = ((np.pi/2 - phase) * cos_phase + np.sin(phase)) / (np.cos(theta_s) + np.cos(theta_v)) - np.pi/4
            return k_vol
            
        elif kernel_type == 'geometric':
            # Li-Sparse kernel computation
            theta_s_prime = np.arctan(np.tan(theta_s))
            theta_v_prime = np.arctan(np.tan(theta_v))
            
            xi = np.sqrt(np.tan(theta_s_prime)**2 + np.tan(theta_v_prime)**2 - 
                        2 * np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.cos(phi))
            
            cos_t = 2 * np.sqrt(xi**2 + (np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.sin(phi))**2) / \
                    (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
            cos_t = np.clip(cos_t, -1.0, 1.0)
            
            t = np.arccos(cos_t)
            
            overlap = (1/np.pi) * (t - np.sin(t) * cos_t) * (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
            
            cos_phase = (np.cos(theta_s_prime) * np.cos(theta_v_prime) + 
                        np.sin(theta_s_prime) * np.sin(theta_v_prime) * np.cos(phi))
            
            k_geo = overlap - (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime) - 
                              0.5 * (1 + cos_phase) * (1/np.cos(theta_s_prime)) * (1/np.cos(theta_v_prime)))
            
            return k_geo
            
        elif kernel_type == 'snow':
            # Snow kernel computation
            cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                        np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
            cos_phase = np.clip(cos_phase, -1.0, 1.0)
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
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    def _save_ratio_results(self, bsa_ratio: xr.DataArray, wsa_ratio: xr.DataArray, band: str):
        """
        Save BSA and WSA ratio results to files.
        
        Parameters
        ----------
        bsa_ratio : xr.DataArray
            BSA ratio data
        wsa_ratio : xr.DataArray
            WSA ratio data
        band : str
            Band identifier
        """
        # Generate date string for filenames
        date_str = self.s2_geometry.get('date_str', 'unknown')
        
        # Save BSA ratio
        bsa_filename = self.output_path / f"a_bsa_{date_str}_{band}.tif"
        save_raster(bsa_ratio, bsa_filename)
        
        # Save WSA ratio
        wsa_filename = self.output_path / f"a_wsa_{date_str}_{band}.tif"
        save_raster(wsa_ratio, wsa_filename)
        
        self.logger.info(f"Saved ratios for band {band}: {bsa_filename.name}, {wsa_filename.name}")
