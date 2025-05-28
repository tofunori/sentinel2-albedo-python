"""
Geometric calculations for sun and view angles.

This module handles the calculation of observation and illumination geometry
for both MODIS and Sentinel-2 data, including angle conversions and
relative azimuth computations.

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import logging
from typing import Tuple, Dict, Optional
from datetime import datetime

import numpy as np
import xarray as xr
from scipy import interpolate


class GeometryCalculator:
    """
    Calculator for sun and view geometry angles.
    
    This class provides methods to compute and process geometric angles
    needed for BRDF calculations, including:
    - Sun zenith and azimuth angles
    - View zenith and azimuth angles  
    - Relative azimuth angles
    - Phase angles and scattering angles
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compute_phase_angle(
        self, 
        theta_s: np.ndarray, 
        theta_v: np.ndarray, 
        phi: np.ndarray
    ) -> np.ndarray:
        """
        Compute phase angle from sun/view geometry.
        
        Parameters
        ----------
        theta_s : np.ndarray
            Solar zenith angle in radians
        theta_v : np.ndarray
            View zenith angle in radians
        phi : np.ndarray
            Relative azimuth angle in radians
            
        Returns
        -------
        np.ndarray
            Phase angle in radians
        """
        cos_phase = (np.cos(theta_s) * np.cos(theta_v) + 
                    np.sin(theta_s) * np.sin(theta_v) * np.cos(phi))
        
        # Ensure cos_phase is within valid range [-1, 1]
        cos_phase = np.clip(cos_phase, -1.0, 1.0)
        
        return np.arccos(cos_phase)
    
    def compute_scattering_angle(
        self, 
        theta_s: np.ndarray, 
        theta_v: np.ndarray, 
        phi: np.ndarray
    ) -> np.ndarray:
        """
        Compute scattering angle (supplement of phase angle).
        
        Parameters
        ----------
        theta_s : np.ndarray
            Solar zenith angle in radians
        theta_v : np.ndarray
            View zenith angle in radians
        phi : np.ndarray
            Relative azimuth angle in radians
            
        Returns
        -------
        np.ndarray
            Scattering angle in degrees
        """
        phase_angle = self.compute_phase_angle(theta_s, theta_v, phi)
        scattering_angle = (np.pi - phase_angle) * (180.0 / np.pi)
        
        return scattering_angle
    
    def normalize_azimuth(
        self, 
        azimuth: np.ndarray, 
        to_positive: bool = True
    ) -> np.ndarray:
        """
        Normalize azimuth angles to [0, 2π] or [-π, π] range.
        
        Parameters
        ----------
        azimuth : np.ndarray
            Azimuth angles in radians
        to_positive : bool, optional
            If True, normalize to [0, 2π], else to [-π, π]
            
        Returns
        -------
        np.ndarray
            Normalized azimuth angles
        """
        if to_positive:
            # Convert negative angles to positive equivalent
            azimuth = np.where(azimuth < 0, azimuth + 2*np.pi, azimuth)
            # Ensure within [0, 2π]
            azimuth = azimuth % (2*np.pi)
        else:
            # Normalize to [-π, π]
            azimuth = ((azimuth + np.pi) % (2*np.pi)) - np.pi
            
        return azimuth
    
    def compute_relative_azimuth(
        self, 
        phi_s: np.ndarray, 
        phi_v: np.ndarray
    ) -> np.ndarray:
        """
        Compute relative azimuth angle between sun and view directions.
        
        This function handles the angle wrapping to ensure the relative
        azimuth is always the acute angle between the two directions.
        
        Parameters
        ----------
        phi_s : np.ndarray
            Solar azimuth angle in radians
        phi_v : np.ndarray
            View azimuth angle in radians
            
        Returns
        -------
        np.ndarray
            Relative azimuth angle in radians [0, π]
        """
        # Normalize both azimuths to [0, 2π]
        phi_s_norm = self.normalize_azimuth(phi_s, to_positive=True)
        phi_v_norm = self.normalize_azimuth(phi_v, to_positive=True)
        
        # Compute absolute difference
        phi_rel = np.abs(phi_s_norm - phi_v_norm)
        
        # Take the acute angle (≤ π)
        phi_rel = np.where(phi_rel > np.pi, 2*np.pi - phi_rel, phi_rel)
        
        return phi_rel
    
    def interpolate_geometry(
        self, 
        geometry_data: xr.DataArray, 
        target_grid: xr.DataArray,
        method: str = 'linear'
    ) -> xr.DataArray:
        """
        Interpolate geometry angles to target grid resolution.
        
        Parameters
        ----------
        geometry_data : xr.DataArray
            Source geometry data
        target_grid : xr.DataArray
            Target grid for interpolation
        method : str, optional
            Interpolation method ('linear', 'nearest', 'cubic')
            
        Returns
        -------
        xr.DataArray
            Interpolated geometry data
        """
        # Use xarray's interpolation functionality
        interpolated = geometry_data.interp(
            x=target_grid.x,
            y=target_grid.y,
            method=method
        )
        
        return interpolated
    
    def validate_angles(
        self, 
        theta_s: np.ndarray, 
        theta_v: np.ndarray, 
        phi: np.ndarray
    ) -> Dict[str, bool]:
        """
        Validate geometric angles for physical consistency.
        
        Parameters
        ----------
        theta_s : np.ndarray
            Solar zenith angles in radians
        theta_v : np.ndarray
            View zenith angles in radians
        phi : np.ndarray
            Relative azimuth angles in radians
            
        Returns
        -------
        dict
            Validation results
        """
        validation = {
            'theta_s_valid': np.all((theta_s >= 0) & (theta_s <= np.pi/2)),
            'theta_v_valid': np.all((theta_v >= 0) & (theta_v <= np.pi/2)),
            'phi_valid': np.all((phi >= 0) & (phi <= np.pi)),
            'no_nan': np.all(np.isfinite([theta_s, theta_v, phi])),
        }
        
        validation['all_valid'] = all(validation.values())
        
        if not validation['all_valid']:
            self.logger.warning("Invalid angles detected in geometry validation")
            for key, value in validation.items():
                if not value:
                    self.logger.warning(f"  {key}: {value}")
        
        return validation
    
    def compute_sun_position(
        self, 
        lat: float, 
        lon: float, 
        datetime_utc: datetime
    ) -> Tuple[float, float]:
        """
        Compute sun position (zenith and azimuth) for given location and time.
        
        This is a simplified solar position calculation. For high precision
        applications, consider using specialized libraries like pvlib or ephem.
        
        Parameters
        ----------
        lat : float
            Latitude in degrees
        lon : float
            Longitude in degrees
        datetime_utc : datetime
            UTC datetime
            
        Returns
        -------
        tuple
            (solar_zenith, solar_azimuth) in degrees
        """
        # Day of year
        day_of_year = datetime_utc.timetuple().tm_yday
        
        # Hour angle
        hour = datetime_utc.hour + datetime_utc.minute/60.0
        hour_angle = (hour - 12) * 15  # degrees
        
        # Solar declination (simplified)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Convert to radians
        lat_rad = np.radians(lat)
        decl_rad = np.radians(declination)
        hour_rad = np.radians(hour_angle)
        
        # Solar elevation
        sin_elevation = (np.sin(lat_rad) * np.sin(decl_rad) + 
                        np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad))
        elevation = np.arcsin(np.clip(sin_elevation, -1, 1))
        
        # Solar zenith
        zenith = np.pi/2 - elevation
        
        # Solar azimuth (simplified)
        cos_azimuth = ((np.sin(decl_rad) - np.sin(elevation) * np.sin(lat_rad)) /
                      (np.cos(elevation) * np.cos(lat_rad)))
        cos_azimuth = np.clip(cos_azimuth, -1, 1)
        azimuth = np.arccos(cos_azimuth)
        
        # Adjust azimuth based on hour angle
        if hour_angle > 0:
            azimuth = 2*np.pi - azimuth
            
        return np.degrees(zenith), np.degrees(azimuth)
    
    def create_angle_masks(
        self, 
        theta_s: np.ndarray, 
        theta_v: np.ndarray,
        max_zenith: float = 70.0
    ) -> np.ndarray:
        """
        Create masks for valid observation geometries.
        
        Parameters
        ----------
        theta_s : np.ndarray
            Solar zenith angles in degrees
        theta_v : np.ndarray
            View zenith angles in degrees
        max_zenith : float, optional
            Maximum zenith angle in degrees for valid observations
            
        Returns
        -------
        np.ndarray
            Boolean mask where True indicates valid geometry
        """
        valid_mask = (
            (theta_s <= max_zenith) & 
            (theta_v <= max_zenith) &
            np.isfinite(theta_s) &
            np.isfinite(theta_v)
        )
        
        return valid_mask
