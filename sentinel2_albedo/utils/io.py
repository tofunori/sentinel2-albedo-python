"""
Input/output utilities for raster data handling.

This module provides functions for loading and saving raster data
using rasterio and xarray, with proper coordinate system handling.

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import logging
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path

import numpy as np
import xarray as xr
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds


def load_raster(
    filepath: Union[str, Path], 
    extent: Optional[Tuple[float, float, float, float]] = None,
    band: Optional[int] = None
) -> xr.DataArray:
    """
    Load a raster file as xarray DataArray.
    
    Parameters
    ----------
    filepath : str or Path
        Path to raster file
    extent : tuple, optional
        Spatial extent (xmin, xmax, ymin, ymax) to crop to
    band : int, optional
        Specific band to load (1-indexed)
        
    Returns
    -------
    xr.DataArray
        Loaded raster data
    """
    logger = logging.getLogger(__name__)
    
    try:
        with rasterio.open(filepath) as src:
            # Determine which bands to read
            if band is not None:
                bands = [band]
            else:
                bands = list(range(1, src.count + 1))
            
            # Calculate window if extent is provided
            if extent is not None:
                window = from_bounds(*extent, src.transform)
                # Read data within window
                data = src.read(bands, window=window)
                # Get transform for the window
                transform = src.window_transform(window)
            else:
                # Read all data
                data = src.read(bands)
                transform = src.transform
                window = None
            
            # Remove singleton dimension if single band
            if len(bands) == 1:
                data = data[0]
            
            # Create coordinate arrays
            if window is not None:
                height, width = window.height, window.width
            else:
                height, width = src.height, src.width
            
            # Generate coordinates
            x_coords = np.arange(width) * transform[0] + transform[2] + transform[0] / 2
            y_coords = np.arange(height) * transform[4] + transform[5] + transform[4] / 2
            
            # Create DataArray
            if len(bands) == 1:
                dims = ['y', 'x']
                coords = {'y': y_coords, 'x': x_coords}
            else:
                dims = ['band', 'y', 'x']
                coords = {'band': bands, 'y': y_coords, 'x': x_coords}
            
            data_array = xr.DataArray(
                data,
                dims=dims,
                coords=coords,
                attrs={
                    'crs': src.crs.to_string() if src.crs else None,
                    'transform': transform,
                    'nodata': src.nodata,
                    'source_file': str(filepath)
                }
            )
            
            return data_array
            
    except Exception as e:
        logger.error(f"Error loading raster {filepath}: {e}")
        raise


def save_raster(
    data_array: xr.DataArray, 
    filepath: Union[str, Path],
    crs: Optional[Union[str, CRS]] = None,
    compress: str = 'lzw',
    dtype: Optional[str] = None,
    nodata: Optional[Union[int, float]] = None
) -> None:
    """
    Save xarray DataArray as raster file.
    
    Parameters
    ----------
    data_array : xr.DataArray
        Data to save
    filepath : str or Path
        Output file path
    crs : str or CRS, optional
        Coordinate reference system
    compress : str, optional
        Compression method (default: 'lzw')
    dtype : str, optional
        Output data type
    nodata : int or float, optional
        NoData value
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure output directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Get CRS from data attributes or parameter
        if crs is None:
            crs = data_array.attrs.get('crs')
        
        if isinstance(crs, str):
            crs = CRS.from_string(crs)
        elif crs is None:
            logger.warning("No CRS specified, using default EPSG:4326")
            crs = CRS.from_epsg(4326)
        
        # Get transform from data attributes or calculate
        if 'transform' in data_array.attrs:
            transform = data_array.attrs['transform']
        else:
            # Calculate transform from coordinates
            x_coords = data_array.x.values
            y_coords = data_array.y.values
            
            x_res = np.abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
            y_res = np.abs(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0
            
            # Note: y_res should be negative for north-up images
            if len(y_coords) > 1 and y_coords[1] < y_coords[0]:
                y_res = -y_res
            
            transform = transform_from_bounds(
                x_coords[0] - x_res/2,
                y_coords[-1] - abs(y_res)/2, 
                x_coords[-1] + x_res/2,
                y_coords[0] + abs(y_res)/2,
                len(x_coords),
                len(y_coords)
            )
        
        # Prepare data
        data = data_array.values
        
        # Handle different dimensionalities
        if data.ndim == 2:
            count = 1
            height, width = data.shape
            data = data[np.newaxis, :, :]  # Add band dimension
        elif data.ndim == 3:
            count, height, width = data.shape
        else:
            raise ValueError(f"Unsupported data dimensionality: {data.ndim}")
        
        # Determine dtype
        if dtype is None:
            if data.dtype == np.float64:
                dtype = rasterio.float32
            else:
                dtype = data.dtype
        
        # Get nodata value
        if nodata is None:
            nodata = data_array.attrs.get('nodata')
        
        # Write raster
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype=dtype,
            crs=crs,
            transform=transform,
            compress=compress,
            nodata=nodata
        ) as dst:
            dst.write(data)
            
        logger.info(f"Raster saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving raster {filepath}: {e}")
        raise


def load_raster_stack(
    filepaths: list, 
    extent: Optional[Tuple[float, float, float, float]] = None
) -> xr.Dataset:
    """
    Load multiple raster files as xarray Dataset.
    
    Parameters
    ----------
    filepaths : list
        List of raster file paths
    extent : tuple, optional
        Spatial extent to crop to
        
    Returns
    -------
    xr.Dataset
        Dataset containing all rasters
    """
    data_arrays = {}
    
    for i, filepath in enumerate(filepaths):
        # Generate variable name from filename
        var_name = Path(filepath).stem
        
        # Load raster
        data_array = load_raster(filepath, extent=extent)
        data_arrays[var_name] = data_array
    
    return xr.Dataset(data_arrays)


def create_spatial_grid(
    extent: Tuple[float, float, float, float], 
    resolution: Union[float, Tuple[float, float]],
    crs: Union[str, CRS] = 'EPSG:4326'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create spatial coordinate grids.
    
    Parameters
    ----------
    extent : tuple
        Spatial extent (xmin, xmax, ymin, ymax)
    resolution : float or tuple
        Spatial resolution (single value or (x_res, y_res))
    crs : str or CRS, optional
        Coordinate reference system
        
    Returns
    -------
    tuple
        (x_coords, y_coords) coordinate arrays
    """
    xmin, xmax, ymin, ymax = extent
    
    if isinstance(resolution, (int, float)):
        x_res = y_res = resolution
    else:
        x_res, y_res = resolution
    
    # Create coordinate arrays
    x_coords = np.arange(xmin + x_res/2, xmax, x_res)
    y_coords = np.arange(ymax - y_res/2, ymin, -y_res)
    
    return x_coords, y_coords


def reproject_raster(
    data_array: xr.DataArray,
    target_crs: Union[str, CRS],
    resolution: Optional[float] = None,
    resampling: str = 'bilinear'
) -> xr.DataArray:
    """
    Reproject raster data to target CRS.
    
    Parameters
    ----------
    data_array : xr.DataArray
        Input raster data
    target_crs : str or CRS
        Target coordinate reference system
    resolution : float, optional
        Target resolution
    resampling : str, optional
        Resampling method
        
    Returns
    -------
    xr.DataArray
        Reprojected raster data
    """
    try:
        import rioxarray as rxr
        
        # Ensure raster has CRS information
        if not hasattr(data_array, 'rio'):
            if 'crs' in data_array.attrs:
                data_array = data_array.rio.write_crs(data_array.attrs['crs'])
            else:
                raise ValueError("No CRS information available")
        
        # Reproject
        reprojected = data_array.rio.reproject(
            target_crs,
            resolution=resolution,
            resampling=getattr(rasterio.enums.Resampling, resampling.lower())
        )
        
        return reprojected
        
    except ImportError:
        raise ImportError("rioxarray required for reprojection")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error reprojecting raster: {e}")
        raise


def clip_raster(
    data_array: xr.DataArray,
    extent: Tuple[float, float, float, float]
) -> xr.DataArray:
    """
    Clip raster data to specified extent.
    
    Parameters
    ----------
    data_array : xr.DataArray
        Input raster data
    extent : tuple
        Clipping extent (xmin, xmax, ymin, ymax)
        
    Returns
    -------
    xr.DataArray
        Clipped raster data
    """
    xmin, xmax, ymin, ymax = extent
    
    # Select data within extent
    clipped = data_array.sel(
        x=slice(xmin, xmax),
        y=slice(ymax, ymin)  # Note: y coordinates may be in descending order
    )
    
    return clipped


def get_raster_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a raster file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to raster file
        
    Returns
    -------
    dict
        Dictionary containing raster information
    """
    try:
        with rasterio.open(filepath) as src:
            info = {
                'driver': src.driver,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': str(src.dtypes[0]),
                'crs': src.crs.to_string() if src.crs else None,
                'transform': src.transform,
                'bounds': src.bounds,
                'nodata': src.nodata,
                'compression': src.compression,
                'interleave': src.interleave
            }
            
            return info
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error reading raster info from {filepath}: {e}")
        raise
