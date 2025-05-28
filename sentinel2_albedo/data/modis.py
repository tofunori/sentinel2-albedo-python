"""
MODIS data handling and processing.

This module provides functionality for loading and processing MODIS MOD09GA
data, including time series management, quality assessment, and geometric
information extraction.

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling


class MODISDataHandler:
    """
    Handler for MODIS MOD09GA data loading and processing.
    
    This class provides methods to load MODIS daily surface reflectance data,
    create time series, extract geometric information, and apply quality filtering.
    
    Parameters
    ----------
    data_path : Path, optional
        Base path to MODIS data directories
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path else None
        self.logger = logging.getLogger(__name__)
        
        # MODIS MOD09GA band mapping
        self.band_mapping = {
            1: {'name': 'Red', 'wavelength': 645, 'sds_name': 'sur_refl_b01'},
            2: {'name': 'NIR', 'wavelength': 858, 'sds_name': 'sur_refl_b02'},
            3: {'name': 'Blue', 'wavelength': 469, 'sds_name': 'sur_refl_b03'},
            4: {'name': 'Green', 'wavelength': 555, 'sds_name': 'sur_refl_b04'},
            5: {'name': 'NIR_1240', 'wavelength': 1240, 'sds_name': 'sur_refl_b05'},
            6: {'name': 'SWIR1', 'wavelength': 1640, 'sds_name': 'sur_refl_b06'},
            7: {'name': 'SWIR2', 'wavelength': 2130, 'sds_name': 'sur_refl_b07'}
        }
        
        # Quality flags
        self.qa_band = 'QC_500m'
        
        # Scale factors
        self.reflectance_scale = 0.0001
        self.angle_scale = 0.01
        
        # Fill values
        self.reflectance_fill = -28672
        self.angle_fill = -32767
    
    def find_modis_files(
        self, 
        start_date: datetime, 
        end_date: datetime,
        tile: Optional[str] = None
    ) -> List[Path]:
        """
        Find MODIS files within date range.
        
        Parameters
        ----------
        start_date : datetime
            Start date for search
        end_date : datetime
            End date for search
        tile : str, optional
            MODIS tile identifier (e.g., 'h10v03')
            
        Returns
        -------
        list
            List of MODIS file paths
        """
        if self.data_path is None:
            raise ValueError("Data path not set")
        
        files = []
        current_date = start_date
        
        while current_date <= end_date:
            # Generate expected filename pattern
            year = current_date.year
            doy = current_date.timetuple().tm_yday
            
            pattern = f"MOD09GA.A{year}{doy:03d}*.hdf"
            if tile:
                pattern = f"MOD09GA.A{year}{doy:03d}.{tile}.*.hdf"
            
            # Search for files matching pattern
            matching_files = list(self.data_path.glob(pattern))
            files.extend(matching_files)
            
            current_date += timedelta(days=1)
        
        files.sort()
        self.logger.info(f"Found {len(files)} MODIS files between {start_date} and {end_date}")
        
        return files
    
    def load_timeseries(
        self,
        start_date: datetime,
        end_date: datetime,
        band: Union[int, List[int]],
        extent: Tuple[float, float, float, float],
        tile: Optional[str] = None
    ) -> xr.Dataset:
        """
        Load MODIS time series data.
        
        Parameters
        ----------
        start_date : datetime
            Start date
        end_date : datetime
            End date
        band : int or list
            Band number(s) to load
        extent : tuple
            Spatial extent as (xmin, xmax, ymin, ymax)
        tile : str, optional
            MODIS tile identifier
            
        Returns
        -------
        xr.Dataset
            Time series dataset
        """
        # Find MODIS files
        files = self.find_modis_files(start_date, end_date, tile)
        
        if not files:
            raise FileNotFoundError(f"No MODIS files found for date range {start_date} to {end_date}")
        
        # Ensure band is a list
        if isinstance(band, int):
            bands = [band]
        else:
            bands = band
        
        # Load data from each file
        datasets = []
        
        for file_path in files:
            try:
                daily_data = self.load_daily_data(file_path, bands, extent)
                if daily_data is not None:
                    datasets.append(daily_data)
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No valid MODIS data could be loaded")
        
        # Concatenate along time dimension
        timeseries = xr.concat(datasets, dim='time')
        
        # Add metadata
        timeseries.attrs['start_date'] = start_date.isoformat()
        timeseries.attrs['end_date'] = end_date.isoformat()
        timeseries.attrs['extent'] = extent
        timeseries.attrs['bands'] = bands
        
        return timeseries
    
    def load_daily_data(
        self,
        file_path: Path,
        bands: List[int],
        extent: Tuple[float, float, float, float]
    ) -> Optional[xr.Dataset]:
        """
        Load daily MODIS data from single file.
        
        Parameters
        ----------
        file_path : Path
            Path to MODIS HDF file
        bands : list
            List of band numbers to load
        extent : tuple
            Spatial extent for cropping
            
        Returns
        -------
        xr.Dataset or None
            Daily dataset or None if loading failed
        """
        try:
            # Extract date from filename
            filename = file_path.name
            date_str = filename.split('.')[1][1:]  # Remove 'A' prefix
            date = datetime.strptime(date_str, '%Y%j')
            
            data_vars = {}
            
            # Load surface reflectance bands
            for band in bands:
                if band in self.band_mapping:
                    sds_name = self.band_mapping[band]['sds_name']
                    band_data = self._load_sds(file_path, sds_name, extent)
                    
                    if band_data is not None:
                        # Apply scale factor and mask fill values
                        band_data = band_data.where(band_data != self.reflectance_fill)
                        band_data = band_data * self.reflectance_scale
                        data_vars[sds_name] = band_data
            
            # Load quality data
            qa_data = self._load_sds(file_path, self.qa_band, extent)
            if qa_data is not None:
                data_vars['qa_500m'] = qa_data
            
            # Load geometric data
            geom_data = self._load_geometry_data(file_path, extent)
            data_vars.update(geom_data)
            
            if not data_vars:
                return None
            
            # Create dataset
            dataset = xr.Dataset(data_vars)
            dataset = dataset.expand_dims('time')
            dataset['time'] = [date]
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _load_sds(self, file_path: Path, sds_name: str, extent: Tuple[float, float, float, float]) -> Optional[xr.DataArray]:
        """
        Load a specific SDS (Scientific Data Set) from MODIS HDF file.
        
        Parameters
        ----------
        file_path : Path
            Path to HDF file
        sds_name : str
            Name of the SDS to load
        extent : tuple
            Spatial extent for cropping
            
        Returns
        -------
        xr.DataArray or None
            Data array or None if loading failed
        """
        try:
            # Open HDF file as rasterio dataset
            hdf_path = f"HDF4_EOS:EOS_GRID:{file_path}:MODIS_Grid_500m_2D:{sds_name}"
            
            with rasterio.open(hdf_path) as src:
                # Calculate window for spatial cropping
                window = rasterio.windows.from_bounds(
                    extent[0], extent[2], extent[1], extent[3],
                    src.transform
                )
                
                # Read data
                data = src.read(1, window=window)
                
                # Get windowed transform
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
                        'crs': src.crs.to_string() if src.crs else None,
                        'transform': transform,
                        'sds_name': sds_name
                    }
                )
                
                return da
                
        except Exception as e:
            self.logger.warning(f"Failed to load SDS {sds_name} from {file_path}: {e}")
            return None
    
    def _load_geometry_data(self, file_path: Path, extent: Tuple[float, float, float, float]) -> Dict[str, xr.DataArray]:
        """
        Load geometric data (angles) from MODIS file.
        
        Parameters
        ----------
        file_path : Path
            Path to MODIS file
        extent : tuple
            Spatial extent
            
        Returns
        -------
        dict
            Dictionary containing geometric data arrays
        """
        geom_data = {}
        
        # Geometric SDSs to load
        geom_sds = {
            'solar_zenith': 'SolarZenith',
            'solar_azimuth': 'SolarAzimuth',
            'view_zenith': 'SensorZenith',
            'view_azimuth': 'SensorAzimuth'
        }
        
        for var_name, sds_name in geom_sds.items():
            try:
                # Try 1km grid first
                hdf_path = f"HDF4_EOS:EOS_GRID:{file_path}:MODIS_Grid_1km_2D:{sds_name}"
                
                with rasterio.open(hdf_path) as src:
                    # Read full data (1km resolution)
                    data = src.read(1)
                    
                    # Mask fill values and apply scale
                    data = np.where(data == self.angle_fill, np.nan, data)
                    data = data * self.angle_scale
                    
                    # Create coordinates
                    height, width = data.shape
                    x_coords = np.linspace(
                        src.bounds.left, src.bounds.right, width
                    )
                    y_coords = np.linspace(
                        src.bounds.top, src.bounds.bottom, height
                    )
                    
                    # Create DataArray
                    da = xr.DataArray(
                        data,
                        dims=['y', 'x'],
                        coords={'y': y_coords, 'x': x_coords},
                        attrs={'sds_name': sds_name}
                    )
                    
                    geom_data[var_name] = da
                    
            except Exception as e:
                self.logger.warning(f"Failed to load {sds_name}: {e}")
                continue
        
        return geom_data
    
    def apply_quality_mask(self, dataset: xr.Dataset, cloud_threshold: int = 1) -> xr.Dataset:
        """
        Apply quality mask to MODIS data.
        
        Parameters
        ----------
        dataset : xr.Dataset
            MODIS dataset
        cloud_threshold : int, optional
            Cloud state threshold (0=clear, 1=cloudy, 2=mixed)
            
        Returns
        -------
        xr.Dataset
            Quality-masked dataset
        """
        if 'qa_500m' not in dataset.data_vars:
            self.logger.warning("Quality data not available, skipping quality filtering")
            return dataset
        
        # Extract cloud state from QA (first 2 bits)
        qa_data = dataset['qa_500m']
        cloud_state = qa_data & 3  # Extract first 2 bits
        
        # Create quality mask (True = good quality)
        quality_mask = cloud_state <= cloud_threshold
        
        # Apply mask to all surface reflectance variables
        masked_dataset = dataset.copy()
        
        for var_name in dataset.data_vars:
            if var_name.startswith('sur_refl'):
                masked_dataset[var_name] = dataset[var_name].where(quality_mask)
        
        # Add quality mask as variable
        masked_dataset['quality_mask'] = quality_mask
        
        return masked_dataset
    
    def resample_to_target(
        self,
        dataset: xr.Dataset,
        target_grid: xr.DataArray,
        method: str = 'bilinear'
    ) -> xr.Dataset:
        """
        Resample MODIS data to target grid.
        
        Parameters
        ----------
        dataset : xr.Dataset
            MODIS dataset
        target_grid : xr.DataArray
            Target grid for resampling
        method : str, optional
            Resampling method ('bilinear', 'nearest')
            
        Returns
        -------
        xr.Dataset
            Resampled dataset
        """
        # Resample each variable
        resampled_vars = {}
        
        for var_name, var_data in dataset.data_vars.items():
            if 'x' in var_data.dims and 'y' in var_data.dims:
                # Resample using xarray interpolation
                resampled_var = var_data.interp(
                    x=target_grid.x,
                    y=target_grid.y,
                    method=method
                )
                resampled_vars[var_name] = resampled_var
            else:
                # Keep non-spatial variables as-is
                resampled_vars[var_name] = var_data
        
        # Create resampled dataset
        resampled_dataset = xr.Dataset(resampled_vars)
        
        # Copy attributes
        resampled_dataset.attrs = dataset.attrs.copy()
        resampled_dataset.attrs['resampled'] = True
        resampled_dataset.attrs['resampling_method'] = method
        
        return resampled_dataset
    
    def compute_statistics(self, dataset: xr.Dataset) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for MODIS time series.
        
        Parameters
        ----------
        dataset : xr.Dataset
            MODIS time series dataset
            
        Returns
        -------
        dict
            Dictionary containing statistics for each variable
        """
        stats = {}
        
        for var_name, var_data in dataset.data_vars.items():
            if var_name.startswith('sur_refl'):
                var_stats = {
                    'mean': float(var_data.mean().values),
                    'std': float(var_data.std().values),
                    'min': float(var_data.min().values),
                    'max': float(var_data.max().values),
                    'valid_pixels': int((~np.isnan(var_data)).sum().values),
                    'total_pixels': int(var_data.size)
                }
                
                # Add temporal statistics if time dimension exists
                if 'time' in var_data.dims:
                    var_stats['temporal_coverage'] = int((~np.isnan(var_data)).sum(dim=['x', 'y']).mean().values)
                
                stats[var_name] = var_stats
        
        return stats
    
    def get_band_info(self) -> Dict[int, Dict[str, Union[str, int]]]:
        """
        Get information about MODIS bands.
        
        Returns
        -------
        dict
            Dictionary containing band information
        """
        return self.band_mapping.copy()
    
    def validate_file(self, file_path: Path) -> bool:
        """
        Validate MODIS file structure and content.
        
        Parameters
        ----------
        file_path : Path
            Path to MODIS file
            
        Returns
        -------
        bool
            True if file is valid
        """
        try:
            # Check if file exists and is readable
            if not file_path.exists():
                return False
            
            # Try to open and read basic info
            test_sds = f"HDF4_EOS:EOS_GRID:{file_path}:MODIS_Grid_500m_2D:sur_refl_b01"
            
            with rasterio.open(test_sds) as src:
                # Check dimensions
                if src.width <= 0 or src.height <= 0:
                    return False
                
                # Try to read a small sample
                sample = src.read(1, window=rasterio.windows.Window(0, 0, 10, 10))
                if sample.size == 0:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"File validation failed for {file_path}: {e}")
            return False
