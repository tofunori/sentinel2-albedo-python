"""
BRDF kernel calculations for MODIS data processing.

This module implements both RTLSR (Ross-Thick Li-Sparse) and Snow kernel models
for modeling bidirectional reflectance distribution functions.

Based on:
- Li and Strahler (1992) - Geometric-optical modeling
- Roujean et al. (1992) - Kernel-driven models  
- Jiao et al. (2019) - Snow kernel development

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import integrate, optimize
from scipy.linalg import lstsq
from scipy.stats import laplace
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..data.modis import MODISDataHandler
from ..data.corrections import TopographicCorrector
from ..utils.io import save_raster
from .geometry import GeometryCalculator


class BRDFKernelProcessor:
    """
    Base class for BRDF kernel processing.
    
    This class provides common functionality for both RTLSR and Snow kernel
    implementations, including data loading, quality filtering, and
    constrained linear system solving.
    """
    
    def __init__(
        self,
        target_date: datetime,
        study_area_extent: Tuple[float, float, float, float],
        modis_handler: MODISDataHandler,
        geometry_calc: GeometryCalculator,
        output_path: Path,
        time_window: int = 16,
        **kwargs
    ):
        self.target_date = target_date
        self.study_area_extent = study_area_extent
        self.modis_handler = modis_handler
        self.geometry_calc = geometry_calc
        self.output_path = Path(output_path)
        self.time_window = time_window
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters
        self.max_na_days = kwargs.get('max_na_days', 12)
        self.rmse_threshold = kwargs.get('rmse_threshold', 0.08)
        self.wod_wdr_threshold = kwargs.get('wod_wdr_threshold', 1.65)
        self.wod_wsa_threshold = kwargs.get('wod_wsa_threshold', 2.5)
        
        # RTLSR kernel constants
        self.k1 = 1.247
        self.k2 = 1.186
        self.k3 = 5.157
        self.alpha = 0.3
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize topographic corrector
        self.topo_corrector = TopographicCorrector()
    
    def _load_modis_timeseries(self, bands: List[int]) -> Dict[str, xr.Dataset]:
        """
        Load 16-day MODIS time series for specified bands.
        
        Parameters
        ----------
        bands : list
            List of MODIS band numbers to process
            
        Returns
        -------
        dict
            Dictionary containing MODIS data for each band
        """
        self.logger.info(f"Loading MODIS time series for bands {bands}")
        
        # Calculate date range (target date ± 8 days)
        start_date = self.target_date - timedelta(days=8)
        end_date = self.target_date + timedelta(days=7)
        
        modis_data = {}
        
        for band in bands:
            band_data = self.modis_handler.load_timeseries(
                start_date=start_date,
                end_date=end_date,
                band=band,
                extent=self.study_area_extent
            )
            
            # Apply quality filtering and corrections
            band_data = self._apply_quality_filters(band_data)
            band_data = self._apply_corrections(band_data)
            
            modis_data[f'band_{band}'] = band_data
        
        return modis_data
    
    def _apply_quality_filters(self, data: xr.Dataset) -> xr.Dataset:
        """
        Apply quality filtering to MODIS data.
        
        Parameters
        ----------
        data : xr.Dataset
            Raw MODIS data
            
        Returns
        -------
        xr.Dataset
            Quality-filtered data
        """
        # Apply MODIS quality flags
        if 'qa_500m' in data.data_vars:
            # Extract cloud mask from QA bits
            qa_data = data['qa_500m']
            cloud_mask = self._extract_cloud_mask(qa_data)
            
            # Apply cloud mask to reflectance data
            for var in data.data_vars:
                if var.startswith('sur_refl'):
                    data[var] = data[var].where(cloud_mask)
        
        # Filter out-of-range values
        for var in data.data_vars:
            if var.startswith('sur_refl'):
                data[var] = data[var].where(
                    (data[var] >= -100) & (data[var] <= 16000)
                )
                # Convert to reflectance
                data[var] = data[var] * 0.0001
        
        return data
    
    def _extract_cloud_mask(self, qa_data: xr.DataArray) -> xr.DataArray:
        """
        Extract cloud mask from MODIS QA data.
        
        Parameters
        ----------
        qa_data : xr.DataArray
            MODIS QA 500m data
            
        Returns
        -------
        xr.DataArray
            Cloud mask (True = clear, False = cloudy)
        """
        # Extract first bit (cloud state)
        cloud_bit = qa_data & 1
        
        # Clear pixels have bit = 0
        clear_mask = cloud_bit == 0
        
        return clear_mask
    
    def _apply_corrections(self, data: xr.Dataset) -> xr.Dataset:
        """
        Apply atmospheric and topographic corrections.
        
        Parameters
        ----------
        data : xr.Dataset
            MODIS data
            
        Returns
        -------
        xr.Dataset
            Corrected data
        """
        # Apply topographic correction if DEM is available
        if hasattr(self, 'dem_data') and self.dem_data is not None:
            data = self.topo_corrector.apply_scs_correction(
                data, self.dem_data
            )
        
        return data
    
    def _compute_kernel_integrals(self) -> Dict[str, float]:
        """
        Compute kernel integrals for albedo calculations.
        
        Returns
        -------
        dict
            Dictionary containing kernel integral values
        """
        self.logger.info("Computing kernel integrals...")
        
        # Define integration limits
        theta_s_max = np.pi / 2  # 90 degrees
        theta_v_max = np.pi / 2  # 90 degrees
        phi_max = 2 * np.pi      # 360 degrees
        
        # Compute volumetric kernel integral
        k_vol_integral = integrate.tplquad(
            self._k_vol_integrand,
            0, phi_max,
            lambda phi: 0, lambda phi: theta_v_max,
            lambda phi, theta_v: 0, lambda phi, theta_v: theta_s_max,
            epsrel=1e-2
        )[0] * (2 / np.pi)
        
        # Compute geometric kernel integral
        k_geo_integral = integrate.tplquad(
            self._k_geo_integrand,
            0, phi_max,
            lambda phi: 0, lambda phi: theta_v_max,
            lambda phi, theta_v: 0, lambda phi, theta_v: theta_s_max,
            epsrel=1e-2
        )[0] * (2 / np.pi)
        
        return {
            'k_vol': k_vol_integral,
            'k_geo': k_geo_integral
        }
    
    def _k_vol_integrand(self, theta_s: float, theta_v: float, phi: float) -> float:
        """
        Volumetric kernel integrand for numerical integration.
        
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
            Kernel value weighted by solid angle
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
    
    def _k_geo_integrand(self, theta_s: float, theta_v: float, phi: float) -> float:
        """
        Geometric kernel integrand for numerical integration.
        
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
            Kernel value weighted by solid angle
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
    
    def _solve_kernel_system(
        self, 
        reflectance: np.ndarray, 
        kernels: Dict[str, np.ndarray],
        weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Solve weighted constrained linear system for kernel parameters.
        
        Parameters
        ----------
        reflectance : np.ndarray
            Observed reflectance values
        kernels : dict
            Dictionary of kernel values
        weights : np.ndarray
            Observation weights
            
        Returns
        -------
        dict
            Dictionary containing kernel parameters and quality metrics
        """
        # Remove NaN values
        valid_mask = np.isfinite(reflectance) & np.all(np.isfinite(list(kernels.values())), axis=0)
        
        if np.sum(valid_mask) < 4:  # Minimum observations needed
            return {
                'f_iso': np.nan,
                'f_vol': np.nan,
                'f_geo': np.nan,
                'rmse': np.nan,
                'wod_wdr': np.nan,
                'wod_wsa': np.nan
            }
        
        # Apply mask
        refl_valid = reflectance[valid_mask]
        weights_valid = weights[valid_mask]
        kernels_valid = {k: v[valid_mask] for k, v in kernels.items()}
        
        # Construct design matrix
        n_obs = len(refl_valid)
        if 'k_snow' in kernels_valid:
            # Snow kernel model (4 parameters)
            A = np.column_stack([
                np.ones(n_obs),  # Isotropic
                kernels_valid['k_vol'],  # Volumetric
                kernels_valid['k_geo'],  # Geometric
                kernels_valid['k_snow']  # Snow
            ])
        else:
            # RTLSR model (3 parameters)
            A = np.column_stack([
                np.ones(n_obs),  # Isotropic
                kernels_valid['k_vol'],  # Volumetric
                kernels_valid['k_geo']   # Geometric
            ])
        
        # Apply weights
        W = np.diag(weights_valid)
        A_weighted = W @ A
        b_weighted = W @ refl_valid
        
        # Solve constrained least squares (all parameters >= 0)
        try:
            result = optimize.lsq_linear(
                A_weighted, 
                b_weighted, 
                bounds=(0, np.inf),
                method='trf'
            )
            
            if result.success:
                params = result.x
                residuals = A @ params - refl_valid
                rmse = np.sqrt(np.mean((residuals**2) * weights_valid))
                
                # Compute weight of determination (WoD)
                wod_metrics = self._compute_wod_metrics(A_weighted, params, weights_valid)
                
                if 'k_snow' in kernels_valid:
                    return {
                        'f_iso': params[0],
                        'f_vol': params[1],
                        'f_geo': params[2],
                        'f_snow': params[3],
                        'rmse': rmse,
                        'wod_wdr': wod_metrics['wdr'],
                        'wod_wsa': wod_metrics['wsa']
                    }
                else:
                    return {
                        'f_iso': params[0],
                        'f_vol': params[1],
                        'f_geo': params[2],
                        'rmse': rmse,
                        'wod_wdr': wod_metrics['wdr'],
                        'wod_wsa': wod_metrics['wsa']
                    }
            else:
                self.logger.warning(f"Optimization failed: {result.message}")
                
        except Exception as e:
            self.logger.warning(f"Error in kernel system solving: {e}")
        
        # Return NaN values if solving failed
        nan_result = {
            'f_iso': np.nan,
            'f_vol': np.nan,
            'f_geo': np.nan,
            'rmse': np.nan,
            'wod_wdr': np.nan,
            'wod_wsa': np.nan
        }
        
        if 'k_snow' in kernels_valid:
            nan_result['f_snow'] = np.nan
            
        return nan_result
    
    def _compute_wod_metrics(self, A: np.ndarray, params: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """
        Compute Weight of Determination (WoD) metrics.
        
        Parameters
        ----------
        A : np.ndarray
            Design matrix
        params : np.ndarray
            Estimated parameters
        weights : np.ndarray
            Observation weights
            
        Returns
        -------
        dict
            WoD metrics for WDR and WSA
        """
        try:
            # Compute covariance matrix
            AtWA = A.T @ np.diag(weights) @ A
            cov_matrix = np.linalg.pinv(AtWA)
            
            # WoD for White-sky Directional Reflectance (WDR) - nadir view
            u_wdr = np.array([1.0, 0.0, 0.0])  # Nadir observation
            if len(params) == 4:  # Snow kernel
                u_wdr = np.array([1.0, 0.0, 0.0, 0.0])
            
            wod_wdr = u_wdr.T @ cov_matrix @ u_wdr
            
            # WoD for White-sky Albedo (WSA) - hemispherical integral
            kernel_integrals = self._compute_kernel_integrals()
            u_wsa = np.array([1.0, kernel_integrals['k_vol'], kernel_integrals['k_geo']])
            if len(params) == 4:  # Snow kernel
                # Add snow kernel integral (simplified)
                u_wsa = np.array([1.0, kernel_integrals['k_vol'], kernel_integrals['k_geo'], 0.1])
            
            wod_wsa = u_wsa.T @ cov_matrix @ u_wsa
            
            return {'wdr': wod_wdr, 'wsa': wod_wsa}
            
        except Exception as e:
            self.logger.warning(f"Error computing WoD metrics: {e}")
            return {'wdr': np.nan, 'wsa': np.nan}


class RTLSRKernelProcessor(BRDFKernelProcessor):
    """
    RTLSR (Ross-Thick Li-Sparse) kernel processor.
    
    Implements the standard RTLSR kernel model for BRDF characterization.
    """
    
    def process(self) -> Dict[str, xr.DataArray]:
        """
        Process RTLSR BRDF kernels.
        
        Returns
        -------
        dict
            Dictionary containing BRDF kernel parameters
        """
        self.logger.info("Processing RTLSR BRDF kernels...")
        
        # MODIS bands to process (1=Red, 2=NIR, 3=Blue, 4=Green, 6=SWIR1, 7=SWIR2)
        bands = [1, 2, 3, 4, 6, 7]
        
        # Load MODIS time series
        modis_data = self._load_modis_timeseries(bands)
        
        # Process each band
        results = {}
        
        for band in bands:
            self.logger.info(f"Processing band {band}...")
            
            band_data = modis_data[f'band_{band}']
            band_results = self._process_band_rtlsr(band_data)
            
            # Save results
            for param, data in band_results.items():
                if f'{param}_B{band}' not in results:
                    results[f'{param}_B{band}'] = data
                    
                # Save to file
                output_file = self.output_path / f"{param}_{self.target_date.strftime('%Y%m%d')}_B{band}.tif"
                save_raster(data, output_file)
        
        self.logger.info("RTLSR kernel processing completed")
        return results
    
    def _process_band_rtlsr(self, band_data: xr.Dataset) -> Dict[str, xr.DataArray]:
        """
        Process RTLSR kernels for a single band.
        
        Parameters
        ----------
        band_data : xr.Dataset
            MODIS band data time series
            
        Returns
        -------
        dict
            Dictionary containing kernel parameters for this band
        """
        # Extract geometry and reflectance data
        geometry_data = self._extract_geometry_data(band_data)
        reflectance_data = self._extract_reflectance_data(band_data)
        
        # Get data shape
        ny, nx = reflectance_data.shape[:2]
        
        # Initialize output arrays
        f_iso = np.full((ny, nx), np.nan)
        f_vol = np.full((ny, nx), np.nan)
        f_geo = np.full((ny, nx), np.nan)
        rmse = np.full((ny, nx), np.nan)
        wod_wdr = np.full((ny, nx), np.nan)
        wod_wsa = np.full((ny, nx), np.nan)
        
        # Process each pixel
        total_pixels = ny * nx
        
        with tqdm(total=total_pixels, desc="Processing pixels") as pbar:
            for i in range(ny):
                for j in range(nx):
                    # Extract pixel time series
                    pixel_refl = reflectance_data[i, j, :]
                    pixel_geom = {k: v[i, j, :] for k, v in geometry_data.items()}
                    
                    # Compute kernels
                    kernels = self._compute_rtlsr_kernels(pixel_geom)
                    
                    # Generate weights (Laplace distribution + coverage)
                    weights = self._generate_weights(pixel_refl, pixel_geom)
                    
                    # Solve kernel system
                    result = self._solve_kernel_system(pixel_refl, kernels, weights)
                    
                    # Store results
                    f_iso[i, j] = result['f_iso']
                    f_vol[i, j] = result['f_vol']
                    f_geo[i, j] = result['f_geo']
                    rmse[i, j] = result['rmse']
                    wod_wdr[i, j] = result['wod_wdr']
                    wod_wsa[i, j] = result['wod_wsa']
                    
                    pbar.update(1)
        
        # Convert to xarray DataArrays
        coords = {'y': band_data['sur_refl_b01'].y, 'x': band_data['sur_refl_b01'].x}
        
        return {
            'f_iso': xr.DataArray(f_iso, dims=['y', 'x'], coords=coords),
            'f_vol': xr.DataArray(f_vol, dims=['y', 'x'], coords=coords),
            'f_geo': xr.DataArray(f_geo, dims=['y', 'x'], coords=coords),
            'rmse': xr.DataArray(rmse, dims=['y', 'x'], coords=coords),
            'wod_wdr': xr.DataArray(wod_wdr, dims=['y', 'x'], coords=coords),
            'wod_wsa': xr.DataArray(wod_wsa, dims=['y', 'x'], coords=coords)
        }
    
    def _compute_rtlsr_kernels(self, geometry: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute RTLSR kernel values for given geometry.
        
        Parameters
        ----------
        geometry : dict
            Dictionary containing theta_s, theta_v, phi arrays
            
        Returns
        -------
        dict
            Dictionary containing kernel values
        """
        theta_s = geometry['theta_s']
        theta_v = geometry['theta_v']
        phi = geometry['phi']
        
        # Compute phase angle
        cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                    np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
        cos_phase = np.clip(cos_phase, -1.0, 1.0)
        phase = np.arccos(cos_phase)
        
        # Volumetric kernel (Ross-Thick)
        k_vol = ((np.pi/2 - phase) * cos_phase + np.sin(phase)) / (np.cos(theta_s) + np.cos(theta_v)) - np.pi/4
        
        # Geometric kernel (Li-Sparse)
        theta_s_prime = np.arctan(np.tan(theta_s))
        theta_v_prime = np.arctan(np.tan(theta_v))
        
        xi = np.sqrt(np.tan(theta_s_prime)**2 + np.tan(theta_v_prime)**2 - 
                    2 * np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.cos(phi))
        
        cos_t = 2 * np.sqrt(xi**2 + (np.tan(theta_s_prime) * np.tan(theta_v_prime) * np.sin(phi))**2) / \
                (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
        cos_t = np.clip(cos_t, -1.0, 1.0)
        
        t = np.arccos(cos_t)
        
        overlap = (1/np.pi) * (t - np.sin(t) * cos_t) * (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime))
        
        cos_xi = (np.cos(theta_s_prime) * np.cos(theta_v_prime) + 
                 np.sin(theta_s_prime) * np.sin(theta_v_prime) * np.cos(phi))
        
        k_geo = overlap - (1/np.cos(theta_s_prime) + 1/np.cos(theta_v_prime) - 
                          0.5 * (1 + cos_xi) * (1/np.cos(theta_s_prime)) * (1/np.cos(theta_v_prime)))
        
        return {
            'k_vol': k_vol,
            'k_geo': k_geo
        }
    
    def _extract_geometry_data(self, data: xr.Dataset) -> Dict[str, np.ndarray]:
        """
        Extract geometry data from MODIS dataset.
        
        Parameters
        ----------
        data : xr.Dataset
            MODIS dataset
            
        Returns
        -------
        dict
            Dictionary containing geometry arrays
        """
        # Extract angles (assuming they're in the dataset)
        theta_s = data['solar_zenith'].values * np.pi / 180.0
        theta_v = data['view_zenith'].values * np.pi / 180.0
        phi_s = data['solar_azimuth'].values * np.pi / 180.0
        phi_v = data['view_azimuth'].values * np.pi / 180.0
        
        # Compute relative azimuth
        phi = self.geometry_calc.compute_relative_azimuth(phi_s, phi_v)
        
        return {
            'theta_s': theta_s,
            'theta_v': theta_v,
            'phi': phi
        }
    
    def _extract_reflectance_data(self, data: xr.Dataset) -> np.ndarray:
        """
        Extract reflectance data from MODIS dataset.
        
        Parameters
        ----------
        data : xr.Dataset
            MODIS dataset
            
        Returns
        -------
        np.ndarray
            Reflectance data array
        """
        # Find the surface reflectance variable
        refl_vars = [var for var in data.data_vars if var.startswith('sur_refl')]
        
        if not refl_vars:
            raise ValueError("No surface reflectance variables found in dataset")
        
        # Use the first reflectance variable
        refl_data = data[refl_vars[0]].values
        
        return refl_data
    
    def _generate_weights(self, reflectance: np.ndarray, geometry: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate observation weights based on Laplace distribution and coverage.
        
        Parameters
        ----------
        reflectance : np.ndarray
            Reflectance values
        geometry : dict
            Geometry data
            
        Returns
        -------
        np.ndarray
            Observation weights
        """
        n_obs = len(reflectance)
        
        # Laplace distribution weights (from original R code)
        probs = np.concatenate([
            np.arange(0.5, 0.96, 0.05625),
            np.arange(0.90, 0.54, -0.05625)
        ])
        
        # Truncate to match observations
        if len(probs) > n_obs:
            probs = probs[:n_obs]
        elif len(probs) < n_obs:
            # Extend with equal weights
            probs = np.concatenate([probs, np.full(n_obs - len(probs), 0.5)])
        
        # Generate Laplace weights
        temp_weights = laplace.ppf(probs, loc=0.5, scale=0.219)
        temp_weights = np.round(temp_weights, 2)
        
        # Add coverage weights (simplified - would need actual coverage data)
        coverage_weights = np.ones(n_obs)  # Placeholder
        
        # Combine weights
        weights = (temp_weights + coverage_weights) / 2
        
        return weights


class SnowKernelProcessor(RTLSRKernelProcessor):
    """
    Snow kernel processor.
    
    Extends RTLSR processor to include snow kernel for better modeling
    of snow and ice surface anisotropy.
    """
    
    def _compute_snow_kernels(self, geometry: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute snow kernel values in addition to RTLSR kernels.
        
        Parameters
        ----------
        geometry : dict
            Dictionary containing theta_s, theta_v, phi arrays
            
        Returns
        -------
        dict
            Dictionary containing all kernel values including snow kernel
        """
        # Compute standard RTLSR kernels
        kernels = self._compute_rtlsr_kernels(geometry)
        
        # Add snow kernel
        theta_s = geometry['theta_s']
        theta_v = geometry['theta_v']
        phi = geometry['phi']
        
        # Compute phase angle
        cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                    np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
        cos_phase = np.clip(cos_phase, -1.0, 1.0)
        phase = np.arccos(cos_phase)
        
        # Phase function P_E from original code
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
        
        kernels['k_snow'] = k_snow
        
        return kernels
    
    def _process_band_snow(self, band_data: xr.Dataset) -> Dict[str, xr.DataArray]:
        """
        Process snow kernels for a single band.
        
        This method is similar to _process_band_rtlsr but includes the snow kernel.
        
        Parameters
        ----------
        band_data : xr.Dataset
            MODIS band data time series
            
        Returns
        -------
        dict
            Dictionary containing kernel parameters including snow kernel
        """
        # Extract geometry and reflectance data
        geometry_data = self._extract_geometry_data(band_data)
        reflectance_data = self._extract_reflectance_data(band_data)
        
        # Get data shape
        ny, nx = reflectance_data.shape[:2]
        
        # Initialize output arrays (including snow kernel)
        f_iso = np.full((ny, nx), np.nan)
        f_vol = np.full((ny, nx), np.nan)
        f_geo = np.full((ny, nx), np.nan)
        f_snow = np.full((ny, nx), np.nan)
        rmse = np.full((ny, nx), np.nan)
        wod_wdr = np.full((ny, nx), np.nan)
        wod_wsa = np.full((ny, nx), np.nan)
        
        # Process each pixel
        total_pixels = ny * nx
        
        with tqdm(total=total_pixels, desc="Processing pixels (Snow)") as pbar:
            for i in range(ny):
                for j in range(nx):
                    # Extract pixel time series
                    pixel_refl = reflectance_data[i, j, :]
                    pixel_geom = {k: v[i, j, :] for k, v in geometry_data.items()}
                    
                    # Compute kernels (including snow)
                    kernels = self._compute_snow_kernels(pixel_geom)
                    
                    # Generate weights
                    weights = self._generate_weights(pixel_refl, pixel_geom)
                    
                    # Solve kernel system
                    result = self._solve_kernel_system(pixel_refl, kernels, weights)
                    
                    # Store results
                    f_iso[i, j] = result['f_iso']
                    f_vol[i, j] = result['f_vol']
                    f_geo[i, j] = result['f_geo']
                    f_snow[i, j] = result['f_snow']
                    rmse[i, j] = result['rmse']
                    wod_wdr[i, j] = result['wod_wdr']
                    wod_wsa[i, j] = result['wod_wsa']
                    
                    pbar.update(1)
        
        # Convert to xarray DataArrays
        coords = {'y': band_data['sur_refl_b01'].y, 'x': band_data['sur_refl_b01'].x}
        
        return {
            'f_iso': xr.DataArray(f_iso, dims=['y', 'x'], coords=coords),
            'f_vol': xr.DataArray(f_vol, dims=['y', 'x'], coords=coords),
            'f_geo': xr.DataArray(f_geo, dims=['y', 'x'], coords=coords),
            'f_snow': xr.DataArray(f_snow, dims=['y', 'x'], coords=coords),
            'rmse': xr.DataArray(rmse, dims=['y', 'x'], coords=coords),
            'wod_wdr': xr.DataArray(wod_wdr, dims=['y', 'x'], coords=coords),
            'wod_wsa': xr.DataArray(wod_wsa, dims=['y', 'x'], coords=coords)
        }
    
    def process(self) -> Dict[str, xr.DataArray]:
        """
        Process Snow BRDF kernels.
        
        Returns
        -------
        dict
            Dictionary containing BRDF kernel parameters including snow kernel
        """
        self.logger.info("Processing Snow BRDF kernels...")
        
        # MODIS bands to process
        bands = [1, 2, 3, 4, 6, 7]
        
        # Load MODIS time series
        modis_data = self._load_modis_timeseries(bands)
        
        # Process each band
        results = {}
        
        for band in bands:
            self.logger.info(f"Processing band {band} with snow kernel...")
            
            band_data = modis_data[f'band_{band}']
            band_results = self._process_band_snow(band_data)
            
            # Save results
            for param, data in band_results.items():
                if f'{param}_B{band}' not in results:
                    results[f'{param}_B{band}'] = data
                    
                # Save to file
                output_file = self.output_path / f"{param}_{self.target_date.strftime('%Y%m%d')}_B{band}.tif"
                save_raster(data, output_file)
        
        self.logger.info("Snow kernel processing completed")
        return results
