"""
Main processor class for Sentinel-2 albedo computation.

This module orchestrates the complete workflow:
1. BRDF kernel computation (RTLSR or Snow kernel)
2. Albedo-to-nadir ratio calculation  
3. Sentinel-2 albedo generation

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import os
import logging
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

import numpy as np
import xarray as xr
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from .geometry import GeometryCalculator
from .brdf_kernels import RTLSRKernelProcessor, SnowKernelProcessor
from .an_ratios import AlbedoNadirRatioCalculator
from ..data.sentinel2 import Sentinel2DataHandler
from ..data.modis import MODISDataHandler
from ..data.corrections import AtmosphericCorrector, TopographicCorrector
from ..utils.clustering import SurfaceClassifier
from ..utils.validation import QualityAssessment
from ..utils.io import save_raster, load_raster


class S2AlbedoProcessor:
    """
    Main processor for generating high-resolution albedo from Sentinel-2 and MODIS data.
    
    This class orchestrates the complete albedo computation workflow, following the
    methodology from Li et al. (2018) and Shuai et al. (2011).
    
    Parameters
    ----------
    target_date : str
        Target date in format 'YYYY-MM-DD'
    sensor : str
        Sentinel-2 sensor, either 'S2A' or 'S2B'
    study_area_extent : tuple
        Study area bounds as (xmin, xmax, ymin, ymax) in UTM coordinates
    modis_data_path : str, optional
        Path to MODIS MOD09GA data directory
    sentinel2_data_path : str, optional
        Path to Sentinel-2 data directory
    output_path : str, optional
        Output directory for results
    dem_path : str, optional
        Path to digital elevation model
    landcover_path : str, optional
        Path to land cover classification
    crs : str, optional
        Coordinate reference system (default: 'EPSG:32611')
    """
    
    def __init__(
        self,
        target_date: str,
        sensor: str,
        study_area_extent: Tuple[float, float, float, float],
        modis_data_path: Optional[str] = None,
        sentinel2_data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        dem_path: Optional[str] = None,
        landcover_path: Optional[str] = None,
        crs: str = "EPSG:32611",
        **kwargs
    ):
        # Parse target date
        self.target_date = datetime.strptime(target_date, "%Y-%m-%d")
        self.date_str = target_date.replace("-", "")
        self.day_of_year = self.target_date.timetuple().tm_yday
        
        # Sensor configuration
        if sensor not in ["S2A", "S2B"]:
            raise ValueError("Sensor must be 'S2A' or 'S2B'")
        self.sensor = sensor
        
        # Spatial configuration
        self.study_area_extent = study_area_extent
        self.crs = CRS.from_string(crs)
        
        # Paths configuration
        self.modis_data_path = Path(modis_data_path) if modis_data_path else None
        self.sentinel2_data_path = Path(sentinel2_data_path) if sentinel2_data_path else None
        self.output_path = Path(output_path) if output_path else Path("./outputs")
        self.dem_path = Path(dem_path) if dem_path else None
        self.landcover_path = Path(landcover_path) if landcover_path else None
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.geometry_calc = GeometryCalculator()
        self.s2_handler = Sentinel2DataHandler(self.sentinel2_data_path)
        self.modis_handler = MODISDataHandler(self.modis_data_path)
        self.atm_corrector = AtmosphericCorrector()
        self.topo_corrector = TopographicCorrector()
        self.classifier = SurfaceClassifier()
        self.quality_assessor = QualityAssessment()
        
        # Processing state
        self.brdf_kernels = None
        self.an_ratios = None
        self.s2_albedo = None
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Initialized S2AlbedoProcessor for {target_date} ({sensor})")
        self.logger.info(f"Study area: {study_area_extent}")
        
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def compute_rtlsr_kernels(self, **kwargs) -> Dict[str, xr.DataArray]:
        """
        Compute BRDF kernel parameters using RTLSR (Ross-Thick Li-Sparse) model.
        
        This method processes 16-day MODIS time series to derive BRDF kernel parameters
        following Li and Strahler (1992) and Roujean et al. (1992).
        
        Returns
        -------
        dict
            Dictionary containing f_iso, f_vol, f_geo parameters and quality metrics
        """
        self.logger.info("Computing RTLSR BRDF kernels...")
        
        # Initialize RTLSR processor
        rtlsr_processor = RTLSRKernelProcessor(
            target_date=self.target_date,
            study_area_extent=self.study_area_extent,
            modis_handler=self.modis_handler,
            geometry_calc=self.geometry_calc,
            output_path=self.output_path / "brdf_kernels",
            **kwargs
        )
        
        # Process BRDF kernels
        self.brdf_kernels = rtlsr_processor.process()
        
        self.logger.info("RTLSR kernel computation completed")
        return self.brdf_kernels
    
    def compute_snow_kernels(self, **kwargs) -> Dict[str, xr.DataArray]:
        """
        Compute BRDF kernel parameters using Snow kernel model.
        
        This method uses the specialized snow kernel from Jiao et al. (2019)
        for better modeling of snow and ice surface anisotropy.
        
        Returns
        -------
        dict
            Dictionary containing f_iso, f_vol, f_geo, f_snow parameters and quality metrics
        """
        self.logger.info("Computing Snow BRDF kernels...")
        
        # Initialize Snow processor
        snow_processor = SnowKernelProcessor(
            target_date=self.target_date,
            study_area_extent=self.study_area_extent,
            modis_handler=self.modis_handler,
            geometry_calc=self.geometry_calc,
            output_path=self.output_path / "brdf_kernels",
            **kwargs
        )
        
        # Process BRDF kernels
        self.brdf_kernels = snow_processor.process()
        
        self.logger.info("Snow kernel computation completed")
        return self.brdf_kernels
    
    def compute_an_ratios(self, **kwargs) -> Dict[str, xr.DataArray]:
        """
        Compute Albedo-to-Nadir (AN) ratios for BSA and WSA.
        
        This method calculates the ratios needed to convert Sentinel-2 nadir reflectance
        to hemispherical albedo, following Shuai et al. (2011).
        
        Returns
        -------
        dict
            Dictionary containing BSA and WSA ratios for each spectral band
        """
        if self.brdf_kernels is None:
            raise ValueError("BRDF kernels must be computed first")
            
        self.logger.info("Computing Albedo-to-Nadir ratios...")
        
        # Get Sentinel-2 geometry
        s2_geometry = self.s2_handler.get_observation_geometry(
            self.target_date, 
            self.study_area_extent
        )
        
        # Initialize AN ratio calculator
        an_calculator = AlbedoNadirRatioCalculator(
            brdf_kernels=self.brdf_kernels,
            s2_geometry=s2_geometry,
            geometry_calc=self.geometry_calc,
            output_path=self.output_path / "an_ratios",
            **kwargs
        )
        
        # Calculate ratios
        self.an_ratios = an_calculator.compute_ratios()
        
        self.logger.info("AN ratio computation completed")
        return self.an_ratios
    
    def compute_s2_albedo(
        self, 
        cluster_range: Tuple[int, int] = (6, 20),
        **kwargs
    ) -> Dict[str, xr.DataArray]:
        """
        Generate final Sentinel-2 albedo maps.
        
        This method applies the AN ratios to Sentinel-2 surface reflectance
        to produce high-resolution albedo maps.
        
        Parameters
        ----------
        cluster_range : tuple
            Range of cluster numbers for sensitivity analysis (min, max)
        
        Returns
        -------
        dict
            Dictionary containing BSA and WSA albedo maps
        """
        if self.an_ratios is None:
            raise ValueError("AN ratios must be computed first")
            
        self.logger.info("Computing Sentinel-2 albedo maps...")
        
        # Load Sentinel-2 data
        s2_data = self.s2_handler.load_surface_reflectance(
            self.target_date,
            self.study_area_extent
        )
        
        # Apply clustering and generate albedo for each cluster count
        albedo_results = {}
        
        for n_clusters in range(cluster_range[0], cluster_range[1] + 1):
            self.logger.info(f"Processing with {n_clusters} clusters...")
            
            # Perform surface classification
            cluster_map = self.classifier.classify_surfaces(
                s2_data, 
                n_clusters=n_clusters
            )
            
            # Apply AN ratios to generate albedo
            bsa_albedo = self._apply_an_ratios(
                s2_data, cluster_map, self.an_ratios['bsa'], n_clusters
            )
            wsa_albedo = self._apply_an_ratios(
                s2_data, cluster_map, self.an_ratios['wsa'], n_clusters
            )
            
            # Store results
            albedo_results[n_clusters] = {
                'bsa': bsa_albedo,
                'wsa': wsa_albedo,
                'clusters': cluster_map
            }
            
            # Save intermediate results
            self._save_albedo_results(bsa_albedo, wsa_albedo, n_clusters)
        
        self.s2_albedo = albedo_results
        self.logger.info("Sentinel-2 albedo computation completed")
        
        return self.s2_albedo
    
    def _apply_an_ratios(
        self, 
        s2_data: xr.Dataset, 
        cluster_map: xr.DataArray, 
        an_ratios: Dict[str, xr.DataArray],
        n_clusters: int
    ) -> xr.Dataset:
        """
        Apply AN ratios to Sentinel-2 data based on cluster classification.
        
        Parameters
        ----------
        s2_data : xr.Dataset
            Sentinel-2 surface reflectance data
        cluster_map : xr.DataArray
            Surface classification map
        an_ratios : dict
            AN ratios for each spectral band
        n_clusters : int
            Number of clusters used
            
        Returns
        -------
        xr.Dataset
            Albedo dataset
        """
        albedo_bands = {}
        
        # Process each spectral band
        for band_name, band_data in s2_data.data_vars.items():
            if band_name in an_ratios:
                # Get AN ratio for this band
                ratio_data = an_ratios[band_name]
                
                # Initialize albedo array
                albedo = xr.zeros_like(band_data)
                
                # Apply ratios for each cluster
                for cluster_id in range(1, n_clusters + 1):
                    # Create mask for this cluster
                    cluster_mask = cluster_map == cluster_id
                    
                    if cluster_mask.sum() > 0:
                        # Get ratio value for this cluster
                        if cluster_id <= len(ratio_data):
                            ratio_value = ratio_data[cluster_id - 1]
                            
                            # Apply ratio
                            albedo = xr.where(
                                cluster_mask,
                                band_data * ratio_value,
                                albedo
                            )
                
                albedo_bands[band_name] = albedo
        
        return xr.Dataset(albedo_bands)
    
    def _save_albedo_results(
        self, 
        bsa_albedo: xr.Dataset, 
        wsa_albedo: xr.Dataset, 
        n_clusters: int
    ):
        """Save albedo results to files."""
        output_dir = self.output_path / "albedo" / f"clusters_{n_clusters}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save BSA albedo
        for band_name, band_data in bsa_albedo.data_vars.items():
            filename = output_dir / f"albedo_bsa_{self.date_str}_{band_name}.tif"
            save_raster(band_data, filename, self.crs)
        
        # Save WSA albedo
        for band_name, band_data in wsa_albedo.data_vars.items():
            filename = output_dir / f"albedo_wsa_{self.date_str}_{band_name}.tif"
            save_raster(band_data, filename, self.crs)
    
    def validate_results(self, reference_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Validate computed albedo against reference data or quality metrics.
        
        Parameters
        ----------
        reference_data : dict, optional
            Reference albedo data for validation
            
        Returns
        -------
        dict
            Validation statistics
        """
        if self.s2_albedo is None:
            raise ValueError("Albedo must be computed first")
            
        self.logger.info("Validating results...")
        
        validation_stats = self.quality_assessor.assess_quality(
            self.s2_albedo,
            reference_data
        )
        
        self.logger.info("Validation completed")
        return validation_stats
    
    def export_summary_report(self, output_file: Optional[str] = None) -> str:
        """
        Export a summary report of the processing results.
        
        Parameters
        ----------
        output_file : str, optional
            Output file path for the report
            
        Returns
        -------
        str
            Report content as string
        """
        report_lines = [
            "# Sentinel-2 Albedo Processing Report",
            f"\nProcessing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Target Date: {self.target_date.strftime('%Y-%m-%d')}",
            f"Sensor: {self.sensor}",
            f"Study Area: {self.study_area_extent}",
            f"\n## Processing Status",
        ]
        
        if self.brdf_kernels is not None:
            report_lines.append("✅ BRDF kernels computed")
        else:
            report_lines.append("❌ BRDF kernels not computed")
            
        if self.an_ratios is not None:
            report_lines.append("✅ AN ratios computed")
        else:
            report_lines.append("❌ AN ratios not computed")
            
        if self.s2_albedo is not None:
            report_lines.append("✅ Sentinel-2 albedo computed")
            report_lines.append(f"   - Cluster configurations: {list(self.s2_albedo.keys())}")
        else:
            report_lines.append("❌ Sentinel-2 albedo not computed")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, "w") as f:
                f.write(report_content)
            self.logger.info(f"Report saved to {output_file}")
        
        return report_content
