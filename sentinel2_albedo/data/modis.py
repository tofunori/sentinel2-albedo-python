"""
MODIS data handling and processing.

This module provides functionality for loading and processing MODIS MOD09GA
daily surface reflectance data, including time series management and
geometry extraction.

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from ..utils.io import load_raster


class MODISDataHandler:
    """
    Handler for MODIS MOD09GA daily surface reflectance data.
    
    This class provides methods to load MODIS data, manage time series,
    and extract observation geometry information.
    """
    
    def __init__(self, modis_data_path: Optional[Path] = None):
        """
        Initialize MODIS data handler.
        
        Parameters
        ----------
        modis_data_path : Path, optional
            Path to MODIS data directory
        """
        self.data_path = Path(modis_data_path) if modis_data_path else None
        self.logger = logging.getLogger(__name__)
        
        # MODIS band mapping (MOD09GA)
        self.band_mapping = {
            1: 'red',        # Band 1: Red (620-670 nm)
            2: 'nir',        # Band 2: NIR (841-876 nm)
            3: 'blue',       # Band 3: Blue (459-479 nm)
            4: 'green',      # Band 4: Green (545-565 nm)
            5: 'nir_wide',   # Band 5: NIR (1230-1250 nm)
            6: 'swir1',      # Band 6: SWIR1 (1628-1652 nm)
            7: 'swir2'       # Band 7: SWIR2 (2105-2155 nm)
        }
        
        # File patterns for different data types
        self.file_patterns = {
            'surface_reflectance': 'MOD09GA*.sur_refl_b{:02d}*.tif',
            'view_zenith': 'MOD09GA*.view_zenith*.tif',
            'view_azimuth': 'MOD09GA*.view_azimuth*.tif',
            'solar_zenith': 'MOD09GA*.solar_zenith*.tif',
            'solar_azimuth': 'MOD09GA*.solar_azimuth*.tif',
            'qa_500m': 'MOD09GA*.QC_500m*.tif'
        }
    
    def find_modis_files(
        self, 
        date: datetime, 
        data_type: str = 'surface_reflectance',
        band: Optional[int] = None
    ) -> List[Path]:
        """
        Find MODIS files for a specific date and data type.
        
        Parameters
        ----------
        date : datetime
            Target date
        data_type : str, optional
            Type of data to find
        band : int, optional
            Specific band number (for surface reflectance)
            
        Returns
        -------
        list
            List of matching file paths
        """
        if not self.data_path or not self.data_path.exists():
            raise ValueError("MODIS data path not set or doesn't exist")
        
        # Convert date to day of year
        day_of_year = date.timetuple().tm_yday
        year = date.year
        date_pattern = f"{year}{day_of_year:03d}"
        
        # Get file pattern
        if data_type == 'surface_reflectance' and band is not None:
            pattern = self.file_patterns[data_type].format(band)
        else:
            pattern = self.file_patterns.get(data_type, f"MOD09GA*.{data_type}*.tif")
        
        # Search for files
        matching_files = []
        
        for file_path in self.data_path.rglob(pattern):
            if date_pattern in file_path.name:
                matching_files.append(file_path)
        
        return sorted(matching_files)
    
    def load_timeseries(
        self,
        start_date: datetime,
        end_date: datetime,
        band: int,
        extent: Tuple[float, float, float, float]
    ) -> xr.Dataset:
        """
        Load MODIS time series for a specific band and date range.
        
        Parameters
        ----------
        start_date : datetime
            Start date of time series
        end_date : datetime
            End date of time series
        band : int
            MODIS band number
        extent : tuple
            Spatial extent (xmin, xmax, ymin, ymax)
            
        Returns
        -------
        xr.Dataset
            Time series dataset
        """
        self.logger.info(f"Loading MODIS time series for band {band}...")
        
        # Generate date range
        date_range = []
        current_date = start_date
        
        while current_date <= end_date:
            date_range.append(current_date)
            current_date += timedelta(days=1)
        
        # Load data for each date
        time_series_data = {}
        
        for date in date_range:
            try:
                # Load surface reflectance
                refl_files = self.find_modis_files(date, 'surface_reflectance', band)
                
                if refl_files:
                    refl_data = load_raster(refl_files[0], extent=extent)
                    time_series_data[f'sur_refl_b{band:02d}_{date.strftime("%Y%j")}'] = refl_data
                
                # Load geometry data
                geometry_data = self._load_geometry_data(date, extent)
                
                for geom_var, geom_data in geometry_data.items():
                    time_series_data[f'{geom_var}_{date.strftime("%Y%j")}'] = geom_data
                
                # Load QA data
                qa_files = self.find_modis_files(date, 'qa_500m')
                
                if qa_files:
                    qa_data = load_raster(qa_files[0], extent=extent)
                    time_series_data[f'qa_500m_{date.strftime("%Y%j")}'] = qa_data
                    
            except Exception as e:
                self.logger.warning(f"Error loading data for {date}: {e}")
                continue
        
        if not time_series_data:
            raise ValueError(f"No MODIS data found for band {band} in date range")
        
        # Create dataset
        dataset = xr.Dataset(time_series_data)
        
        # Add metadata
        dataset.attrs.update({
            'band': band,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'extent': extent,
            'n_dates': len(date_range)
        })
        
        return dataset
    
    def _load_geometry_data(
        self, 
        date: datetime, 
        extent: Tuple[float, float, float, float]
    ) -> Dict[str, xr.DataArray]:
        """
        Load geometry data for a specific date.
        
        Parameters
        ----------
        date : datetime
            Target date
        extent : tuple
            Spatial extent
            
        Returns
        -------
        dict
            Dictionary containing geometry data
        """
        geometry_data = {}
        
        geometry_types = ['view_zenith', 'view_azimuth', 'solar_zenith', 'solar_azimuth']
        
        for geom_type in geometry_types:
            try:
                geom_files = self.find_modis_files(date, geom_type)
                
                if geom_files:
                    geom_data = load_raster(geom_files[0], extent=extent)
                    geometry_data[geom_type] = geom_data
                    
            except Exception as e:
                self.logger.warning(f"Error loading {geom_type} for {date}: {e}")
        
        return geometry_data
    
    def load_single_date(
        self,
        date: datetime,
        bands: List[int],
        extent: Tuple[float, float, float, float],
        include_geometry: bool = True,
        include_qa: bool = True
    ) -> xr.Dataset:
        """
        Load MODIS data for a single date.
        
        Parameters
        ----------
        date : datetime
            Target date
        bands : list
            List of band numbers to load
        extent : tuple
            Spatial extent
        include_geometry : bool, optional
            Whether to include geometry data
        include_qa : bool, optional
            Whether to include QA data
            
        Returns
        -------
        xr.Dataset
            MODIS dataset for the date
        """
        data_vars = {}
        
        # Load surface reflectance bands
        for band in bands:
            try:
                refl_files = self.find_modis_files(date, 'surface_reflectance', band)
                
                if refl_files:
                    refl_data = load_raster(refl_files[0], extent=extent)
                    band_name = f'sur_refl_b{band:02d}'
                    data_vars[band_name] = refl_data
                else:
                    self.logger.warning(f"No surface reflectance file found for band {band} on {date}")
                    
            except Exception as e:
                self.logger.error(f"Error loading band {band} for {date}: {e}")
        
        # Load geometry data
        if include_geometry:
            geometry_data = self._load_geometry_data(date, extent)
            data_vars.update(geometry_data)
        
        # Load QA data
        if include_qa:
            try:
                qa_files = self.find_modis_files(date, 'qa_500m')
                
                if qa_files:
                    qa_data = load_raster(qa_files[0], extent=extent)
                    data_vars['qa_500m'] = qa_data
                    
            except Exception as e:
                self.logger.warning(f"Error loading QA data for {date}: {e}")
        
        # Create dataset
        dataset = xr.Dataset(data_vars)
        
        # Add metadata
        dataset.attrs.update({
            'date': date.isoformat(),
            'bands': bands,
            'extent': extent
        })
        
        return dataset
    
    def get_available_dates(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[datetime]:
        """
        Get list of available dates within a date range.
        
        Parameters
        ----------
        start_date : datetime
            Start date
        end_date : datetime
            End date
            
        Returns
        -------
        list
            List of available dates
        """
        if not self.data_path or not self.data_path.exists():
            return []
        
        available_dates = set()
        
        # Search for any MODIS files in the date range
        current_date = start_date
        
        while current_date <= end_date:
            day_of_year = current_date.timetuple().tm_yday
            year = current_date.year
            date_pattern = f"{year}{day_of_year:03d}"
            
            # Check if any files exist for this date
            matching_files = list(self.data_path.rglob(f"MOD09GA*{date_pattern}*.tif"))
            
            if matching_files:
                available_dates.add(current_date)
            
            current_date += timedelta(days=1)
        
        return sorted(list(available_dates))
    
    def get_file_info(self, filepath: Path) -> Dict[str, any]:
        """
        Extract information from MODIS filename.
        
        Parameters
        ----------
        filepath : Path
            Path to MODIS file
            
        Returns
        -------
        dict
            Dictionary containing file information
        """
        filename = filepath.name
        
        # Parse MODIS filename (e.g., MOD09GA.A2018220.h10v03.006.2018222050516.hdf)
        parts = filename.split('.')
        
        info = {
            'product': parts[0] if len(parts) > 0 else None,
            'date_str': parts[1] if len(parts) > 1 else None,
            'tile': parts[2] if len(parts) > 2 else None,
            'collection': parts[3] if len(parts) > 3 else None,
            'processing_date': parts[4] if len(parts) > 4 else None,
            'filepath': filepath
        }
        
        # Parse date string
        if info['date_str'] and info['date_str'].startswith('A'):
            try:
                year = int(info['date_str'][1:5])
                day_of_year = int(info['date_str'][5:8])
                date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
                info['date'] = date
            except ValueError:
                info['date'] = None
        
        return info
    
    def create_coverage_mask(
        self, 
        dataset: xr.Dataset, 
        coverage_threshold: float = 0.6
    ) -> xr.DataArray:
        """
        Create coverage mask based on pixel coverage percentage.
        
        Parameters
        ----------
        dataset : xr.Dataset
            MODIS dataset
        coverage_threshold : float, optional
            Minimum coverage threshold (0-1)
            
        Returns
        -------
        xr.DataArray
            Coverage mask
        """
        # This is a placeholder implementation
        # In practice, would use actual coverage data from MODIS files
        
        # For now, create a simple mask based on data availability
        if 'sur_refl_b01' in dataset.data_vars:
            reference_data = dataset['sur_refl_b01']
            coverage_mask = xr.ones_like(reference_data, dtype=bool)
        else:
            # Use first available variable
            var_name = list(dataset.data_vars.keys())[0]
            reference_data = dataset[var_name]
            coverage_mask = xr.ones_like(reference_data, dtype=bool)
        
        return coverage_mask
