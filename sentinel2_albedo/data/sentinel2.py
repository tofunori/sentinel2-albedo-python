"""
Sentinel-2 data handling and processing.

This module provides functionality for loading and processing Sentinel-2 data,
including surface reflectance extraction, metadata parsing, and geometric
information retrieval.

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from scipy import interpolate


class Sentinel2DataHandler:
    """
    Handler for Sentinel-2 data loading and processing.
    
    This class provides methods to load Sentinel-2 L2A surface reflectance data,
    extract observation geometry from metadata, and process spectral bands.
    
    Parameters
    ----------
    data_path : Path, optional
        Base path to Sentinel-2 data directories
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path else None
        self.logger = logging.getLogger(__name__)
        
        # Sentinel-2 band mapping (L2A)
        self.band_mapping = {
            'B02': {'name': 'Blue', 'resolution': 10},
            'B03': {'name': 'Green', 'resolution': 10},
            'B04': {'name': 'Red', 'resolution': 10},
            'B08': {'name': 'NIR', 'resolution': 10},
            'B05': {'name': 'VegRE1', 'resolution': 20},
            'B06': {'name': 'VegRE2', 'resolution': 20},
            'B07': {'name': 'VegRE3', 'resolution': 20},
            'B8A': {'name': 'NIR_narrow', 'resolution': 20},
            'B11': {'name': 'SWIR1', 'resolution': 20},
            'B12': {'name': 'SWIR2', 'resolution': 20},
        }
        
        # Default bands for albedo processing
        self.albedo_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    
    def find_product_directory(self, target_date: datetime, tile_id: Optional[str] = None) -> Optional[Path]:
        """
        Find Sentinel-2 product directory for given date.
        
        Parameters
        ----------
        target_date : datetime
            Target acquisition date
        tile_id : str, optional
            Specific tile identifier (e.g., 'T11UMT')
            
        Returns
        -------
        Path or None
            Path to product directory if found
        """
        if self.data_path is None:
            raise ValueError("Data path not set")
        
        date_str = target_date.strftime('%Y%m%d')
        
        # Search for matching directories
        pattern = f"*{date_str}*"
        if tile_id:
            pattern = f"*{tile_id}*{date_str}*"
        
        matching_dirs = list(self.data_path.glob(pattern))
        
        if not matching_dirs:
            self.logger.warning(f"No Sentinel-2 products found for {date_str}")
            return None
        
        if len(matching_dirs) > 1:
            self.logger.warning(f"Multiple products found for {date_str}, using first: {matching_dirs[0]}")
        
        return matching_dirs[0]
    
    def load_surface_reflectance(
        self, 
        target_date: datetime, 
        extent: Tuple[float, float, float, float],
        bands: Optional[List[str]] = None,
        resolution: int = 20
    ) -> xr.Dataset:
        """
        Load Sentinel-2 surface reflectance data.
        
        Parameters
        ----------
        target_date : datetime
            Target acquisition date
        extent : tuple
            Spatial extent as (xmin, xmax, ymin, ymax)
        bands : list, optional
            List of bands to load (default: albedo_bands)
        resolution : int, optional
            Target resolution in meters (default: 20)
            
        Returns
        -------
        xr.Dataset
            Dataset containing surface reflectance bands
        """
        if bands is None:
            bands = self.albedo_bands
        
        # Find product directory
        product_dir = self.find_product_directory(target_date)
        if product_dir is None:
            raise FileNotFoundError(f"No Sentinel-2 product found for {target_date}")
        
        self.logger.info(f"Loading Sentinel-2 data from {product_dir}")
        
        # Load bands
        data_vars = {}
        
        for band in bands:
            band_path = self._find_band_file(product_dir, band, resolution)
            if band_path:
                band_data = self._load_band(band_path, extent)
                data_vars[band] = band_data
            else:
                self.logger.warning(f"Band {band} not found in product")
        
        # Create dataset
        dataset = xr.Dataset(data_vars)
        
        # Add metadata
        dataset.attrs['product_dir'] = str(product_dir)
        dataset.attrs['target_date'] = target_date.isoformat()
        dataset.attrs['extent'] = extent
        
        return dataset
    
    def _find_band_file(self, product_dir: Path, band: str, resolution: int) -> Optional[Path]:
        """
        Find band file in product directory.
        
        Parameters
        ----------
        product_dir : Path
            Product directory path
        band : str
            Band identifier (e.g., 'B04')
        resolution : int
            Target resolution
            
        Returns
        -------
        Path or None
            Path to band file if found
        """
        # Look in IMG_DATA subdirectories
        img_data_dir = product_dir / "GRANULE" / "*" / "IMG_DATA"
        
        # Search for band file with specific resolution
        pattern = f"*{band}_{resolution}m.jp2"
        
        for img_dir in product_dir.glob("GRANULE/*/IMG_DATA"):
            # Try resolution-specific subdirectory first
            res_dir = img_dir / f"R{resolution}m"
            if res_dir.exists():
                band_files = list(res_dir.glob(pattern))
                if band_files:
                    return band_files[0]
            
            # Try main IMG_DATA directory
            band_files = list(img_dir.glob(pattern))
            if band_files:
                return band_files[0]
        
        return None
    
    def _load_band(self, band_path: Path, extent: Tuple[float, float, float, float]) -> xr.DataArray:
        """
        Load single band with spatial cropping.
        
        Parameters
        ----------
        band_path : Path
            Path to band file
        extent : tuple
            Spatial extent for cropping
            
        Returns
        -------
        xr.DataArray
            Band data array
        """
        with rasterio.open(band_path) as src:
            # Calculate window for cropping
            window = rasterio.windows.from_bounds(
                extent[0], extent[2], extent[1], extent[3], 
                src.transform
            )
            
            # Read data
            data = src.read(1, window=window)
            
            # Get transform for windowed data
            transform = rasterio.windows.transform(window, src.transform)
            
            # Create coordinate arrays
            height, width = data.shape
            x_coords = np.linspace(
                transform[2], 
                transform[2] + width * transform[0], 
                width
            )
            y_coords = np.linspace(
                transform[5], 
                transform[5] + height * transform[4], 
                height
            )
            
            # Create DataArray
            da = xr.DataArray(
                data,
                dims=['y', 'x'],
                coords={'y': y_coords, 'x': x_coords},
                attrs={
                    'crs': src.crs.to_string(),
                    'transform': transform,
                    'nodata': src.nodata
                }
            )
            
            # Apply scale factor (Sentinel-2 L2A scale factor is 10000)
            da = da / 10000.0
            
            # Mask invalid values
            if src.nodata is not None:
                da = da.where(da != src.nodata / 10000.0)
            
            return da
    
    def get_observation_geometry(
        self, 
        target_date: datetime, 
        extent: Tuple[float, float, float, float]
    ) -> Dict[str, xr.DataArray]:
        """
        Extract observation geometry from Sentinel-2 metadata.
        
        Parameters
        ----------
        target_date : datetime
            Target acquisition date
        extent : tuple
            Spatial extent
            
        Returns
        -------
        dict
            Dictionary containing geometry arrays
        """
        # Find product directory
        product_dir = self.find_product_directory(target_date)
        if product_dir is None:
            raise FileNotFoundError(f"No Sentinel-2 product found for {target_date}")
        
        # Find metadata file
        metadata_file = self._find_metadata_file(product_dir)
        if metadata_file is None:
            raise FileNotFoundError("Metadata file not found")
        
        self.logger.info(f"Extracting geometry from {metadata_file}")
        
        # Parse metadata
        geometry = self._parse_geometry_metadata(metadata_file, extent)
        
        return geometry
    
    def _find_metadata_file(self, product_dir: Path) -> Optional[Path]:
        """
        Find tile-level metadata file.
        
        Parameters
        ----------
        product_dir : Path
            Product directory
            
        Returns
        -------
        Path or None
            Path to metadata file
        """
        # Look for MTD_TL.xml in granule directory
        mtd_files = list(product_dir.glob("GRANULE/*/MTD_TL.xml"))
        
        if mtd_files:
            return mtd_files[0]
        
        # Fallback to main metadata file
        main_mtd = list(product_dir.glob("MTD_*.xml"))
        if main_mtd:
            return main_mtd[0]
        
        return None
    
    def _parse_geometry_metadata(self, metadata_file: Path, extent: Tuple[float, float, float, float]) -> Dict[str, xr.DataArray]:
        """
        Parse geometry information from metadata XML.
        
        Parameters
        ----------
        metadata_file : Path
            Path to metadata file
        extent : tuple
            Target spatial extent
            
        Returns
        -------
        dict
            Dictionary containing geometry arrays
        """
        try:
            tree = ET.parse(metadata_file)
            root = tree.getroot()
            
            # Extract sun angles
            sun_angles = self._extract_sun_angles(root, extent)
            
            # Extract viewing angles
            view_angles = self._extract_viewing_angles(root, extent)
            
            # Combine geometry
            geometry = {
                'theta_s': sun_angles['zenith'],
                'phi_s': sun_angles['azimuth'],
                'theta_v': view_angles['zenith'],
                'phi_v': view_angles['azimuth'],
                'date_str': metadata_file.parent.parent.name.split('_')[2]
            }
            
            # Compute relative azimuth
            phi_rel = np.abs(geometry['phi_s'] - geometry['phi_v'])
            phi_rel = np.where(phi_rel > 180, 360 - phi_rel, phi_rel)
            geometry['phi'] = phi_rel * np.pi / 180.0  # Convert to radians
            
            # Convert angles to radians
            geometry['theta_s'] = geometry['theta_s'] * np.pi / 180.0
            geometry['theta_v'] = geometry['theta_v'] * np.pi / 180.0
            
            return geometry
            
        except Exception as e:
            self.logger.error(f"Error parsing metadata: {e}")
            raise
    
    def _extract_sun_angles(self, root: ET.Element, extent: Tuple[float, float, float, float]) -> Dict[str, xr.DataArray]:
        """
        Extract sun angles from metadata.
        
        Parameters
        ----------
        root : ET.Element
            XML root element
        extent : tuple
            Spatial extent
            
        Returns
        -------
        dict
            Dictionary containing sun angle arrays
        """
        # Find sun angles grid
        sun_zenith_elem = root.find(".//Sun_Angles_Grid/Zenith/Values_List")
        sun_azimuth_elem = root.find(".//Sun_Angles_Grid/Azimuth/Values_List")
        
        if sun_zenith_elem is None or sun_azimuth_elem is None:
            raise ValueError("Sun angles not found in metadata")
        
        # Parse angle values
        zenith_values = self._parse_angle_values(sun_zenith_elem.text)
        azimuth_values = self._parse_angle_values(sun_azimuth_elem.text)
        
        # Create coordinate arrays
        coords = self._create_angle_coordinates(extent, zenith_values.shape)
        
        # Create DataArrays
        sun_zenith = xr.DataArray(
            zenith_values,
            dims=['y', 'x'],
            coords=coords
        )
        
        sun_azimuth = xr.DataArray(
            azimuth_values,
            dims=['y', 'x'],
            coords=coords
        )
        
        return {'zenith': sun_zenith, 'azimuth': sun_azimuth}
    
    def _extract_viewing_angles(self, root: ET.Element, extent: Tuple[float, float, float, float]) -> Dict[str, xr.DataArray]:
        """
        Extract viewing angles from metadata.
        
        Parameters
        ----------
        root : ET.Element
            XML root element
        extent : tuple
            Spatial extent
            
        Returns
        -------
        dict
            Dictionary containing viewing angle arrays
        """
        # Find viewing angles (may have multiple detectors)
        view_zenith_elems = root.findall(".//Viewing_Incidence_Angles_Grids/Zenith/Values_List")
        view_azimuth_elems = root.findall(".//Viewing_Incidence_Angles_Grids/Azimuth/Values_List")
        
        if not view_zenith_elems or not view_azimuth_elems:
            raise ValueError("Viewing angles not found in metadata")
        
        # Average across all detectors
        zenith_arrays = []
        azimuth_arrays = []
        
        for elem in view_zenith_elems:
            zenith_values = self._parse_angle_values(elem.text)
            zenith_arrays.append(zenith_values)
        
        for elem in view_azimuth_elems:
            azimuth_values = self._parse_angle_values(elem.text)
            azimuth_arrays.append(azimuth_values)
        
        # Compute mean across detectors
        mean_zenith = np.nanmean(np.stack(zenith_arrays), axis=0)
        mean_azimuth = np.nanmean(np.stack(azimuth_arrays), axis=0)
        
        # Create coordinate arrays
        coords = self._create_angle_coordinates(extent, mean_zenith.shape)
        
        # Create DataArrays
        view_zenith = xr.DataArray(
            mean_zenith,
            dims=['y', 'x'],
            coords=coords
        )
        
        view_azimuth = xr.DataArray(
            mean_azimuth,
            dims=['y', 'x'],
            coords=coords
        )
        
        return {'zenith': view_zenith, 'azimuth': view_azimuth}
    
    def _parse_angle_values(self, text: str) -> np.ndarray:
        """
        Parse angle values from XML text.
        
        Parameters
        ----------
        text : str
            XML text containing angle values
            
        Returns
        -------
        np.ndarray
            2D array of angle values
        """
        # Split text into lines and then into values
        lines = text.strip().split('\n')
        values = []
        
        for line in lines:
            line_values = [float(x) for x in line.strip().split()]
            values.append(line_values)
        
        return np.array(values)
    
    def _create_angle_coordinates(self, extent: Tuple[float, float, float, float], shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Create coordinate arrays for angle grids.
        
        Parameters
        ----------
        extent : tuple
            Spatial extent
        shape : tuple
            Array shape (ny, nx)
            
        Returns
        -------
        dict
            Dictionary containing coordinate arrays
        """
        ny, nx = shape
        
        x_coords = np.linspace(extent[0], extent[1], nx)
        y_coords = np.linspace(extent[3], extent[2], ny)  # Y decreases
        
        return {'y': y_coords, 'x': x_coords}
    
    def get_band_info(self) -> Dict[str, Dict[str, Union[str, int]]]:
        """
        Get information about available bands.
        
        Returns
        -------
        dict
            Dictionary containing band information
        """
        return self.band_mapping.copy()
    
    def validate_product(self, product_dir: Path) -> bool:
        """
        Validate Sentinel-2 product structure.
        
        Parameters
        ----------
        product_dir : Path
            Product directory to validate
            
        Returns
        -------
        bool
            True if product structure is valid
        """
        try:
            # Check for required directories
            required_dirs = ['GRANULE', 'DATASTRIP']
            
            for dir_name in required_dirs:
                if not (product_dir / dir_name).exists():
                    self.logger.error(f"Missing required directory: {dir_name}")
                    return False
            
            # Check for metadata file
            if not self._find_metadata_file(product_dir):
                self.logger.error("Metadata file not found")
                return False
            
            # Check for some band files
            granule_dirs = list(product_dir.glob("GRANULE/*"))
            if not granule_dirs:
                self.logger.error("No granule directories found")
                return False
            
            img_data_dir = granule_dirs[0] / "IMG_DATA"
            if not img_data_dir.exists():
                self.logger.error("IMG_DATA directory not found")
                return False
            
            # Check for at least one band file
            band_files = list(img_data_dir.rglob("*.jp2"))
            if not band_files:
                self.logger.error("No band files found")
                return False
            
            self.logger.info(f"Product validation successful: {product_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating product: {e}")
            return False
