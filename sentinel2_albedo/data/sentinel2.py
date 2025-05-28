"""
Sentinel-2 data handling and processing.

This module provides functionality for loading and processing Sentinel-2 L2A
surface reflectance data, including metadata extraction and geometric calculations.

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
import xml.etree.ElementTree as ET
from lxml import etree

from ..utils.io import load_raster


class Sentinel2DataHandler:
    """
    Handler for Sentinel-2 L2A surface reflectance data.
    
    This class provides methods to load Sentinel-2 data, extract metadata,
    and compute observation geometry from XML files.
    """
    
    def __init__(self, sentinel2_data_path: Optional[Path] = None):
        """
        Initialize Sentinel-2 data handler.
        
        Parameters
        ----------
        sentinel2_data_path : Path, optional
            Path to Sentinel-2 data directory
        """
        self.data_path = Path(sentinel2_data_path) if sentinel2_data_path else None
        self.logger = logging.getLogger(__name__)
        
        # Sentinel-2 band mapping
        self.band_mapping = {
            'B02': 'blue',     # Blue (490 nm)
            'B03': 'green',    # Green (560 nm) 
            'B04': 'red',      # Red (665 nm)
            'B08': 'nir',      # NIR (842 nm)
            'B11': 'swir1',    # SWIR1 (1610 nm)
            'B12': 'swir2'     # SWIR2 (2190 nm)
        }
        
        # Resolution mapping
        self.resolution_mapping = {
            'B02': 10, 'B03': 10, 'B04': 10, 'B08': 10,  # 10m bands
            'B11': 20, 'B12': 20  # 20m bands
        }
    
    def find_s2_products(self, target_date: datetime, tile_id: Optional[str] = None) -> List[Path]:
        """
        Find Sentinel-2 products for a given date.
        
        Parameters
        ----------
        target_date : datetime
            Target date for data search
        tile_id : str, optional
            Specific tile ID to search for
            
        Returns
        -------
        list
            List of paths to Sentinel-2 product directories
        """
        if not self.data_path or not self.data_path.exists():
            raise ValueError("Sentinel-2 data path not set or doesn't exist")
        
        date_str = target_date.strftime('%Y%m%d')
        products = []
        
        # Search for products matching the date
        for product_dir in self.data_path.iterdir():
            if product_dir.is_dir() and date_str in product_dir.name:
                if tile_id is None or tile_id in product_dir.name:
                    products.append(product_dir)
        
        if not products:
            self.logger.warning(f"No Sentinel-2 products found for {date_str}")
        
        return products
    
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
            Target date
        extent : tuple
            Spatial extent (xmin, xmax, ymin, ymax)
        bands : list, optional
            List of bands to load (default: all bands)
            
        Returns
        -------
        xr.Dataset
            Sentinel-2 surface reflectance dataset
        """
        if bands is None:
            bands = list(self.band_mapping.keys())
        
        # Find product directories
        products = self.find_s2_products(target_date)
        
        if not products:
            raise ValueError(f"No Sentinel-2 products found for {target_date}")
        
        # Use the first product found
        product_dir = products[0]
        self.logger.info(f"Loading data from {product_dir.name}")
        
        # Load bands
        band_data = {}
        
        for band in bands:
            band_file = self._find_band_file(product_dir, band)
            
            if band_file:
                # Load raster data
                data_array = load_raster(band_file, extent=extent)
                
                # Convert to reflectance (scale factor 0.0001)
                data_array = data_array * 0.0001
                
                # Store with standardized name
                band_name = self.band_mapping.get(band, band.lower())
                band_data[band_name] = data_array
            else:
                self.logger.warning(f"Band file not found for {band}")
        
        # Create dataset
        dataset = xr.Dataset(band_data)
        
        # Add metadata
        dataset.attrs['product_dir'] = str(product_dir)
        dataset.attrs['target_date'] = target_date.isoformat()
        dataset.attrs['extent'] = extent
        
        return dataset
    
    def _find_band_file(self, product_dir: Path, band: str) -> Optional[Path]:
        """
        Find the file for a specific band in the product directory.
        
        Parameters
        ----------
        product_dir : Path
            Sentinel-2 product directory
        band : str
            Band identifier (e.g., 'B04')
            
        Returns
        -------
        Path or None
            Path to band file if found
        """
        # Look for .jp2 files in GRANULE subdirectories
        granule_dirs = list(product_dir.glob('GRANULE/*/IMG_DATA/**/'))
        
        for granule_dir in granule_dirs:
            # Try different resolution directories
            resolution = self.resolution_mapping.get(band, 10)
            
            # Look in appropriate resolution directory
            if resolution == 10:
                img_dirs = [granule_dir / 'R10m', granule_dir]
            else:
                img_dirs = [granule_dir / 'R20m', granule_dir]
            
            for img_dir in img_dirs:
                if img_dir.exists():
                    # Find files matching the band pattern
                    pattern = f'*_{band}_*.jp2'
                    band_files = list(img_dir.glob(pattern))
                    
                    if band_files:
                        return band_files[0]
        
        return None
    
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
            Target date
        extent : tuple
            Spatial extent
            
        Returns
        -------
        dict
            Dictionary containing geometry arrays
        """
        # Find product
        products = self.find_s2_products(target_date)
        
        if not products:
            raise ValueError(f"No Sentinel-2 products found for {target_date}")
        
        product_dir = products[0]
        
        # Find MTD_TL.xml file
        mtd_files = list(product_dir.glob('GRANULE/*/MTD_TL.xml'))
        
        if not mtd_files:
            raise ValueError(f"MTD_TL.xml file not found in {product_dir}")
        
        mtd_file = mtd_files[0]
        
        # Parse XML
        geometry_data = self._parse_mtd_xml(mtd_file, extent)
        
        # Add date information
        geometry_data['date'] = target_date.strftime('%Y%m%d')
        
        return geometry_data
    
    def _parse_mtd_xml(self, mtd_file: Path, extent: Tuple[float, float, float, float]) -> Dict[str, xr.DataArray]:
        """
        Parse MTD_TL.xml file to extract geometry information.
        
        Parameters
        ----------
        mtd_file : Path
            Path to MTD_TL.xml file
        extent : tuple
            Spatial extent for interpolation
            
        Returns
        -------
        dict
            Dictionary containing geometry arrays
        """
        try:
            # Parse XML file
            tree = etree.parse(str(mtd_file))
            root = tree.getroot()
            
            # Extract sun angles
            sun_zenith = self._extract_angle_grid(root, './/Sun_Angles_Grid/Zenith/Values_List')
            sun_azimuth = self._extract_angle_grid(root, './/Sun_Angles_Grid/Azimuth/Values_List')
            
            # Extract view angles (average across detectors)
            view_zenith = self._extract_view_angles(root, 'Zenith')
            view_azimuth = self._extract_view_angles(root, 'Azimuth')
            
            # Create coordinate system
            ny, nx = sun_zenith.shape
            
            # Create spatial coordinates based on extent
            x_coords = np.linspace(extent[0], extent[1], nx)
            y_coords = np.linspace(extent[3], extent[2], ny)  # Note: reversed for correct orientation
            
            coords = {'y': y_coords, 'x': x_coords}
            
            # Convert to radians
            theta_s = xr.DataArray(sun_zenith * np.pi / 180, dims=['y', 'x'], coords=coords)
            phi_s = xr.DataArray(sun_azimuth * np.pi / 180, dims=['y', 'x'], coords=coords)
            theta_v = xr.DataArray(view_zenith * np.pi / 180, dims=['y', 'x'], coords=coords)
            phi_v = xr.DataArray(view_azimuth * np.pi / 180, dims=['y', 'x'], coords=coords)
            
            # Compute relative azimuth
            phi_rel = np.abs(phi_s - phi_v)
            phi_rel = xr.where(phi_rel > np.pi, 2*np.pi - phi_rel, phi_rel)
            
            return {
                'theta_s': theta_s,
                'theta_v': theta_v,
                'phi': phi_rel,
                'phi_s': phi_s,
                'phi_v': phi_v
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing MTD_TL.xml: {e}")
            raise
    
    def _extract_angle_grid(self, root, xpath: str) -> np.ndarray:
        """
        Extract angle grid from XML.
        
        Parameters
        ----------
        root : xml.etree.ElementTree.Element
            XML root element
        xpath : str
            XPath expression for angle values
            
        Returns
        -------
        np.ndarray
            2D array of angle values
        """
        values_elements = root.xpath(xpath)
        
        if not values_elements:
            raise ValueError(f"No elements found for xpath: {xpath}")
        
        # Parse values from first element
        values_text = values_elements[0].text.strip()
        
        # Split into lines and then into individual values
        lines = values_text.split('\n')
        angle_grid = []
        
        for line in lines:
            if line.strip():
                row_values = [float(x) for x in line.strip().split()]
                angle_grid.append(row_values)
        
        return np.array(angle_grid)
    
    def _extract_view_angles(self, root, angle_type: str) -> np.ndarray:
        """
        Extract and average view angles across all detectors.
        
        Parameters
        ----------
        root : xml.etree.ElementTree.Element
            XML root element
        angle_type : str
            'Zenith' or 'Azimuth'
            
        Returns
        -------
        np.ndarray
            2D array of averaged view angles
        """
        xpath = f'.//Viewing_Incidence_Angles_Grids/{angle_type}/Values_List'
        values_elements = root.xpath(xpath)
        
        if not values_elements:
            raise ValueError(f"No view {angle_type.lower()} angles found")
        
        # Parse all detector angles
        all_angles = []
        
        for values_element in values_elements:
            values_text = values_element.text.strip()
            
            # Parse angle grid
            lines = values_text.split('\n')
            angle_grid = []
            
            for line in lines:
                if line.strip():
                    row_values = [float(x) for x in line.strip().split()]
                    angle_grid.append(row_values)
            
            all_angles.append(np.array(angle_grid))
        
        # Average across all detectors
        if all_angles:
            # Stack along new axis and compute mean
            stacked_angles = np.stack(all_angles, axis=0)
            mean_angles = np.nanmean(stacked_angles, axis=0)
            return mean_angles
        else:
            raise ValueError("No valid view angles found")
    
    def load_auxiliary_data(self, product_dir: Path) -> Dict[str, xr.DataArray]:
        """
        Load auxiliary data (DEM, cloud masks, etc.) from Sentinel-2 product.
        
        Parameters
        ----------
        product_dir : Path
            Sentinel-2 product directory
            
        Returns
        -------
        dict
            Dictionary containing auxiliary datasets
        """
        aux_data = {}
        
        # Look for auxiliary data in GRANULE directory
        granule_dirs = list(product_dir.glob('GRANULE/*/AUX_DATA/'))
        
        for aux_dir in granule_dirs:
            # Load DEM if available
            dem_files = list(aux_dir.glob('*DEM*.jp2'))
            if dem_files:
                aux_data['dem'] = load_raster(dem_files[0])
            
            # Load other auxiliary data as needed
        
        return aux_data
    
    def get_product_metadata(self, product_dir: Path) -> Dict:
        """
        Extract product-level metadata.
        
        Parameters
        ----------
        product_dir : Path
            Sentinel-2 product directory
            
        Returns
        -------
        dict
            Product metadata
        """
        # Find main metadata file
        mtd_files = list(product_dir.glob('MTD_*.xml'))
        
        if not mtd_files:
            return {}
        
        try:
            tree = etree.parse(str(mtd_files[0]))
            root = tree.getroot()
            
            # Extract key metadata
            metadata = {
                'product_type': self._get_xml_text(root, './/PRODUCT_TYPE'),
                'processing_level': self._get_xml_text(root, './/PROCESSING_LEVEL'),
                'spacecraft': self._get_xml_text(root, './/SPACECRAFT_NAME'),
                'sensing_time': self._get_xml_text(root, './/PRODUCT_START_TIME'),
                'tile_id': self._get_xml_text(root, './/TILE_ID'),
                'cloud_coverage': self._get_xml_text(root, './/Cloud_Coverage_Assessment')
            }
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error reading product metadata: {e}")
            return {}
    
    def _get_xml_text(self, root, xpath: str) -> Optional[str]:
        """
        Get text content from XML element.
        
        Parameters
        ----------
        root : xml.etree.ElementTree.Element
            XML root
        xpath : str
            XPath expression
            
        Returns
        -------
        str or None
            Text content if found
        """
        elements = root.xpath(xpath)
        return elements[0].text if elements else None
