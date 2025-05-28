"""
Sentinel-2 data handling and processing.

This module provides functionality for loading Sentinel-2 L2A surface reflectance
data, extracting observation geometry from metadata, and preparing data for
albedo computation.

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import box


class Sentinel2DataHandler:
    """
    Handler for Sentinel-2 L2A surface reflectance data.
    
    This class provides methods to load Sentinel-2 data, extract geometry
    information from metadata XML files, and prepare data for albedo processing.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path else None
        self.logger = logging.getLogger(__name__)
        
        # Sentinel-2 band mapping (L2A products)
        self.band_mapping = {
            'B02': 'blue',      # 490 nm - 10m
            'B03': 'green',     # 560 nm - 10m  
            'B04': 'red',       # 665 nm - 10m
            'B08': 'nir',       # 842 nm - 10m
            'B11': 'swir1',     # 1610 nm - 20m
            'B12': 'swir2',     # 2190 nm - 20m
        }
        
        # Target resolution for processing
        self.target_resolution = 20  # meters
        self.target_crs = CRS.from_epsg(32611)  # UTM Zone 11N (default)
    
    def find_product_folder(self, target_date: datetime) -> Optional[Path]:
        """
        Find Sentinel-2 product folder for target date.
        
        Parameters
        ----------
        target_date : datetime
            Target acquisition date
            
        Returns
        -------
        Path or None
            Path to product folder if found
        """
        if not self.data_path or not self.data_path.exists():
            raise ValueError("Sentinel-2 data path not set or does not exist")
        
        date_str = target_date.strftime("%Y%m%d")
        
        # Search for folders containing the target date
        for folder in self.data_path.iterdir():
            if folder.is_dir() and date_str in folder.name:
                # Check if it's a valid Sentinel-2 folder
                if self._is_valid_s2_folder(folder):
                    return folder
        
        self.logger.warning(f"No Sentinel-2 product found for {date_str}")
        return None
    
    def _is_valid_s2_folder(self, folder: Path) -> bool:
        """
        Check if folder is a valid Sentinel-2 product folder.
        
        Parameters
        ----------
        folder : Path
            Folder to check
            
        Returns
        -------
        bool
            True if valid Sentinel-2 folder
        """
        # Look for MTD_TL.xml file
        mtd_file = folder / "MTD_TL.xml"
        if not mtd_file.exists():
            # Try alternative structure
            granule_folders = list(folder.glob("GRANULE/*/"))
            if granule_folders:
                mtd_file = granule_folders[0] / "MTD_TL.xml"
        
        return mtd_file.exists()
    
    def load_surface_reflectance(
        self, 
        target_date: datetime,
        extent: Tuple[float, float, float, float],
        bands: Optional[List[str]] = None
    ) -> xr.Dataset:
        """
        Load Sentinel-2 surface reflectance data.
        
        Parameters
        ----------
        target_date : datetime
            Target acquisition date
        extent : tuple
            Bounding box as (xmin, xmax, ymin, ymax)
        bands : list, optional
            List of bands to load (default: all bands)
            
        Returns
        -------
        xr.Dataset
            Sentinel-2 surface reflectance dataset
        """
        self.logger.info(f"Loading Sentinel-2 data for {target_date.strftime('%Y-%m-%d')}")
        
        # Find product folder
        product_folder = self.find_product_folder(target_date)
        if not product_folder:
            raise FileNotFoundError(f"No Sentinel-2 product found for {target_date}")
        
        # Default bands if not specified
        if bands is None:
            bands = list(self.band_mapping.keys())
        
        # Load each band
        band_arrays = {}
        
        for band in bands:
            band_file = self._find_band_file(product_folder, band)
            if band_file:
                band_array = self._load_band(band_file, extent)
                band_name = self.band_mapping.get(band, band)
                band_arrays[band_name] = band_array
            else:
                self.logger.warning(f"Band {band} file not found")
        
        # Create dataset
        dataset = xr.Dataset(band_arrays)
        
        # Add metadata
        dataset.attrs['product_folder'] = str(product_folder)
        dataset.attrs['acquisition_date'] = target_date.isoformat()
        dataset.attrs['extent'] = extent
        
        self.logger.info(f"Loaded {len(band_arrays)} bands")
        return dataset
    
    def _find_band_file(self, product_folder: Path, band: str) -> Optional[Path]:
        """
        Find the file for a specific band.
        
        Parameters
        ----------
        product_folder : Path
            Sentinel-2 product folder
        band : str
            Band identifier (e.g., 'B04')
            
        Returns
        -------
        Path or None
            Path to band file if found
        """
        # Look for band files in IMG_DATA folder
        img_data_folders = list(product_folder.glob("**/IMG_DATA/**"))
        
        for img_folder in img_data_folders:
            # Look for files matching the band pattern
            band_files = list(img_folder.glob(f"*{band}*.jp2"))
            if band_files:
                return band_files[0]
        
        return None
    
    def _load_band(
        self, 
        band_file: Path, 
        extent: Tuple[float, float, float, float]
    ) -> xr.DataArray:
        """
        Load a single band file.
        
        Parameters
        ----------
        band_file : Path
            Path to band file
        extent : tuple
            Bounding box for cropping
            
        Returns
        -------
        xr.DataArray
            Band data array
        """
        with rasterio.open(band_file) as src:
            # Calculate window for extent
            window = rasterio.windows.from_bounds(
                extent[0], extent[2], extent[1], extent[3], 
                src.transform
            )
            
            # Read data
            data = src.read(1, window=window)
            
            # Get transform for windowed data
            transform = src.window_transform(window)
            
            # Create coordinates
            height, width = data.shape
            x_coords = np.linspace(
                transform.c, 
                transform.c + width * transform.a, 
                width
            )
            y_coords = np.linspace(
                transform.f, 
                transform.f + height * transform.e, 
                height
            )
            
            # Create DataArray
            da = xr.DataArray(
                data,
                dims=['y', 'x'],
                coords={'y': y_coords, 'x': x_coords},
                attrs={
                    'crs': src.crs,
                    'transform': transform,
                    'nodata': src.nodata
                }
            )
            
            # Convert to reflectance (Sentinel-2 L2A is already in reflectance * 10000)
            da = da.astype(np.float32) / 10000.0
            
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
            Bounding box for processing
            
        Returns
        -------
        dict
            Dictionary containing geometry arrays
        """
        self.logger.info("Extracting Sentinel-2 observation geometry")
        
        # Find product folder
        product_folder = self.find_product_folder(target_date)
        if not product_folder:
            raise FileNotFoundError(f"No Sentinel-2 product found for {target_date}")
        
        # Find MTD_TL.xml file
        mtd_file = self._find_metadata_file(product_folder)
        
        # Parse geometry from XML
        geometry_data = self._parse_geometry_from_xml(mtd_file, extent)
        
        return geometry_data
    
    def _find_metadata_file(self, product_folder: Path) -> Path:
        """
        Find the metadata XML file.
        
        Parameters
        ----------
        product_folder : Path
            Sentinel-2 product folder
            
        Returns
        -------
        Path
            Path to MTD_TL.xml file
        """
        # Try different locations
        possible_locations = [
            product_folder / "MTD_TL.xml",
            product_folder / "GRANULE" / "*" / "MTD_TL.xml"
        ]
        
        for location in possible_locations:
            if '*' in str(location):
                # Handle wildcard
                matches = list(product_folder.glob(str(location).replace(str(product_folder) + "/", "")))
                if matches:
                    return matches[0]
            elif location.exists():
                return location
        
        raise FileNotFoundError(f"MTD_TL.xml not found in {product_folder}")
    
    def _parse_geometry_from_xml(
        self, 
        mtd_file: Path,
        extent: Tuple[float, float, float, float]
    ) -> Dict[str, xr.DataArray]:
        """
        Parse observation geometry from MTD_TL.xml file.
        
        Parameters
        ----------
        mtd_file : Path
            Path to MTD_TL.xml file
        extent : tuple
            Bounding box for geometry grid
            
        Returns
        -------
        dict
            Dictionary containing geometry arrays
        """
        tree = ET.parse(mtd_file)
        root = tree.getroot()
        
        # Extract sun angles
        sun_zenith_grid = self._extract_angle_grid(root, 'Sun_Angles_Grid/Zenith')
        sun_azimuth_grid = self._extract_angle_grid(root, 'Sun_Angles_Grid/Azimuth')
        
        # Extract viewing angles (averaged across detectors)
        view_zenith_grid = self._extract_viewing_angles(root, 'Zenith')
        view_azimuth_grid = self._extract_viewing_angles(root, 'Azimuth')
        
        # Convert to DataArrays and interpolate to target extent
        geometry_arrays = {}
        
        for name, grid_data in [
            ('solar_zenith', sun_zenith_grid),
            ('solar_azimuth', sun_azimuth_grid),
            ('view_zenith', view_zenith_grid),
            ('view_azimuth', view_azimuth_grid)
        ]:
            # Create DataArray from grid (typically 23x23)
            da = xr.DataArray(
                grid_data,
                dims=['y', 'x']
            )
            
            # Interpolate to target extent
            # This is a simplified approach - in practice, you'd need to
            # properly georeference the angle grids
            da_interp = self._interpolate_to_extent(da, extent)
            
            # Convert to radians
            geometry_arrays[name] = da_interp * np.pi / 180.0
        
        # Compute relative azimuth
        phi_rel = np.abs(geometry_arrays['solar_azimuth'] - geometry_arrays['view_azimuth'])
        phi_rel = xr.where(phi_rel > np.pi, 2*np.pi - phi_rel, phi_rel)
        geometry_arrays['relative_azimuth'] = phi_rel
        
        return geometry_arrays
    
    def _extract_angle_grid(self, root: ET.Element, xpath: str) -> np.ndarray:
        """
        Extract angle grid from XML.
        
        Parameters
        ----------
        root : ET.Element
            XML root element
        xpath : str
            XPath to angle values
            
        Returns
        -------
        np.ndarray
            Angle grid as 2D array
        """
        # Find the Values_List element
        values_element = root.find(f".//{xpath}/Values_List")
        
        if values_element is None:
            raise ValueError(f"Could not find {xpath} in XML")
        
        # Extract values
        values_text = values_element.text.strip()
        values_lines = values_text.split('\n')
        
        # Parse into 2D array
        grid_data = []
        for line in values_lines:
            if line.strip():
                row_values = [float(x) for x in line.split()]
                grid_data.append(row_values)
        
        return np.array(grid_data)
    
    def _extract_viewing_angles(self, root: ET.Element, angle_type: str) -> np.ndarray:
        """
        Extract viewing angles, averaging across detectors.
        
        Parameters
        ----------
        root : ET.Element
            XML root element
        angle_type : str
            'Zenith' or 'Azimuth'
            
        Returns
        -------
        np.ndarray
            Averaged viewing angle grid
        """
        # Find all viewing angle grids for different bands/detectors
        viewing_elements = root.findall(f".//Viewing_Incidence_Angles_Grids/{angle_type}/Values_List")
        
        if not viewing_elements:
            raise ValueError(f"Could not find viewing {angle_type} angles in XML")
        
        # Extract all grids
        grids = []
        for element in viewing_elements:
            values_text = element.text.strip()
            values_lines = values_text.split('\n')
            
            grid_data = []
            for line in values_lines:
                if line.strip():
                    row_values = [float(x) for x in line.split()]
                    grid_data.append(row_values)
            
            grids.append(np.array(grid_data))
        
        # Average across all detectors/bands
        stacked_grids = np.stack(grids, axis=0)
        averaged_grid = np.nanmean(stacked_grids, axis=0)
        
        return averaged_grid
    
    def _interpolate_to_extent(
        self, 
        angle_grid: xr.DataArray,
        extent: Tuple[float, float, float, float],
        target_resolution: float = 20.0
    ) -> xr.DataArray:
        """
        Interpolate angle grid to target extent and resolution.
        
        This is a simplified implementation. In practice, you would need
        to properly georeference the angle grids using the tile geometry.
        
        Parameters
        ----------
        angle_grid : xr.DataArray
            Input angle grid (typically 23x23)
        extent : tuple
            Target extent (xmin, xmax, ymin, ymax)
        target_resolution : float
            Target pixel resolution in meters
            
        Returns
        -------
        xr.DataArray
            Interpolated angle grid
        """
        # Calculate target grid dimensions
        width = int((extent[1] - extent[0]) / target_resolution)
        height = int((extent[3] - extent[2]) / target_resolution)
        
        # Create target coordinates
        x_coords = np.linspace(extent[0], extent[1], width)
        y_coords = np.linspace(extent[3], extent[2], height)  # Reverse for image coordinates
        
        # Simple interpolation (bilinear)
        interpolated = angle_grid.interp(
            x=x_coords,
            y=y_coords,
            method='linear'
        )
        
        return interpolated
    
    def apply_cloud_mask(
        self, 
        dataset: xr.Dataset,
        product_folder: Path
    ) -> xr.Dataset:
        """
        Apply cloud mask to Sentinel-2 data.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Sentinel-2 dataset
        product_folder : Path
            Path to Sentinel-2 product folder
            
        Returns
        -------
        xr.Dataset
            Cloud-masked dataset
        """
        # Find cloud mask file (SCL - Scene Classification Layer)
        scl_file = self._find_band_file(product_folder, 'SCL')
        
        if scl_file:
            self.logger.info("Applying cloud mask")
            
            # Load SCL data
            with rasterio.open(scl_file) as src:
                scl_data = src.read(1)
                
            # Create cloud mask (SCL values: 1=saturated, 3=cloud shadow, 
            # 8=cloud medium prob, 9=cloud high prob, 10=thin cirrus)
            cloud_values = [1, 3, 8, 9, 10]
            cloud_mask = np.isin(scl_data, cloud_values)
            
            # Apply mask to all bands
            for var in dataset.data_vars:
                dataset[var] = dataset[var].where(~cloud_mask)
        
        return dataset
