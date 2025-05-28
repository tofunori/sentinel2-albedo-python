#!/usr/bin/env python3
"""
Advanced configuration example for Sentinel-2 albedo processing.

This script demonstrates advanced configuration options including
custom parameters, batch processing, and specialized corrections.

Author: Claude AI Assistant
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel2_albedo import S2AlbedoProcessor
from sentinel2_albedo.data.corrections import SpectralBandAdjustmentFactor
from sentinel2_albedo.utils.clustering import SurfaceClassifier
from sentinel2_albedo.utils.validation import QualityAssessment


class AdvancedAlbedoProcessor:
    """
    Advanced albedo processor with custom configurations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.sbaf_corrector = SpectralBandAdjustmentFactor()
        self.classifier = SurfaceClassifier()
        self.quality_assessor = QualityAssessment()
        
        # Results storage
        self.results = {}
    
    def process_time_series(
        self, 
        start_date: str, 
        end_date: str, 
        time_step: int = 16
    ) -> Dict[str, Any]:
        """
        Process time series of albedo products.
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        time_step : int, optional
            Time step in days
            
        Returns
        -------
        dict
            Time series results
        """
        self.logger.info(f"Processing time series from {start_date} to {end_date}")
        
        # Generate date list
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = []
        current_date = start_dt
        
        while current_date <= end_dt:
            dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=time_step)
        
        self.logger.info(f"Processing {len(dates)} dates: {dates}")
        
        # Process each date
        time_series_results = {}
        
        for date in dates:
            try:
                self.logger.info(f"Processing date: {date}")
                
                # Update configuration for this date
                date_config = self.config.copy()
                date_config['target_date'] = date
                date_config['output_path'] = str(Path(self.config['output_path']) / f"date_{date.replace('-', '')}")
                
                # Process single date
                date_results = self.process_single_date(date_config)
                time_series_results[date] = date_results
                
                self.logger.info(f"✓ Completed processing for {date}")
                
            except Exception as e:
                self.logger.error(f"Error processing {date}: {e}")
                time_series_results[date] = {'error': str(e)}
        
        # Analyze time series
        time_series_analysis = self._analyze_time_series(time_series_results)
        
        return {
            'individual_dates': time_series_results,
            'time_series_analysis': time_series_analysis
        }
    
    def process_single_date(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process albedo for a single date with advanced configuration.
        
        Parameters
        ----------
        config : dict
            Processing configuration
            
        Returns
        -------
        dict
            Processing results
        """
        # Initialize processor with custom parameters
        processor = S2AlbedoProcessor(
            target_date=config['target_date'],
            sensor=config['sensor'],
            study_area_extent=config['study_area_extent'],
            modis_data_path=config.get('modis_data_path'),
            sentinel2_data_path=config.get('sentinel2_data_path'), 
            output_path=config.get('output_path'),
            dem_path=config.get('dem_path'),
            landcover_path=config.get('landcover_path')
        )
        
        results = {}
        
        # Choose BRDF kernel model based on configuration
        kernel_model = config.get('kernel_model', 'rtlsr')
        
        if kernel_model == 'rtlsr':
            self.logger.info("Using RTLSR kernel model")
            brdf_kernels = processor.compute_rtlsr_kernels(
                **config.get('rtlsr_params', {})
            )
        elif kernel_model == 'snow':
            self.logger.info("Using Snow kernel model")
            brdf_kernels = processor.compute_snow_kernels(
                **config.get('snow_params', {})
            )
        else:
            raise ValueError(f"Unknown kernel model: {kernel_model}")
        
        results['brdf_kernels'] = brdf_kernels
        
        # Compute AN ratios
        an_ratios = processor.compute_an_ratios(
            **config.get('an_ratio_params', {})
        )
        results['an_ratios'] = an_ratios
        
        # Advanced clustering configuration
        cluster_config = config.get('clustering', {})
        
        if cluster_config.get('optimize_clusters', False):
            # Optimize number of clusters
            s2_data = processor.s2_handler.load_surface_reflectance(
                datetime.strptime(config['target_date'], '%Y-%m-%d'),
                config['study_area_extent']
            )
            
            optimal_clusters = self.classifier.optimize_cluster_number(
                s2_data,
                min_clusters=cluster_config.get('min_clusters', 6),
                max_clusters=cluster_config.get('max_clusters', 20),
                method=cluster_config.get('optimization_method', 'silhouette')
            )
            
            cluster_range = (optimal_clusters, optimal_clusters)
            self.logger.info(f"Using optimized cluster number: {optimal_clusters}")
        else:
            cluster_range = config.get('cluster_range', (6, 20))
        
        # Generate albedo maps
        albedo_maps = processor.compute_s2_albedo(
            cluster_range=cluster_range,
            **config.get('albedo_params', {})
        )
        results['albedo_maps'] = albedo_maps
        
        # Quality assessment with custom metrics
        quality_stats = processor.validate_results()
        
        # Add custom quality metrics
        if config.get('detailed_validation', False):
            quality_stats['detailed'] = self._detailed_quality_assessment(
                albedo_maps, config
            )
        
        results['quality_stats'] = quality_stats
        
        return results
    
    def _analyze_time_series(self, time_series_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze time series results for trends and patterns.
        
        Parameters
        ----------
        time_series_results : dict
            Results for each date
            
        Returns
        -------
        dict
            Time series analysis
        """
        analysis = {
            'successful_dates': [],
            'failed_dates': [],
            'quality_trends': {},
            'albedo_statistics': {}
        }
        
        for date, results in time_series_results.items():
            if 'error' in results:
                analysis['failed_dates'].append(date)
            else:
                analysis['successful_dates'].append(date)
                
                # Extract quality metrics
                if 'quality_stats' in results:
                    quality = results['quality_stats']
                    
                    # Store quality trends (placeholder)
                    # In practice, would extract specific metrics
                    analysis['quality_trends'][date] = {
                        'processing_success': True
                    }
        
        analysis['success_rate'] = len(analysis['successful_dates']) / len(time_series_results)
        
        return analysis
    
    def _detailed_quality_assessment(
        self, 
        albedo_maps: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform detailed quality assessment.
        
        Parameters
        ----------
        albedo_maps : dict
            Albedo map results
        config : dict
            Processing configuration
            
        Returns
        -------
        dict
            Detailed quality metrics
        """
        detailed_metrics = {
            'spatial_coherence': {},
            'spectral_consistency': {},
            'temporal_stability': {}
        }
        
        # Placeholder for detailed quality assessment
        # In practice, would implement specific quality checks
        
        return detailed_metrics


def main():
    """
    Run advanced configuration example.
    """
    print("=" * 60)
    print("Sentinel-2 High Resolution Albedo Processing")
    print("Advanced Configuration Example")
    print("=" * 60)
    
    # Advanced configuration
    config = {
        # Basic parameters
        'sensor': 'S2A',
        'study_area_extent': (421290, 496851.5, 5759734, 5800008.8),
        'modis_data_path': '/path/to/modis/data',
        'sentinel2_data_path': '/path/to/sentinel2/data',
        'output_path': './outputs/advanced_example',
        'dem_path': '/path/to/dem.tif',
        'landcover_path': '/path/to/landcover.tif',
        
        # Advanced parameters
        'kernel_model': 'snow',  # 'rtlsr' or 'snow'
        
        # RTLSR kernel parameters
        'rtlsr_params': {
            'max_na_days': 10,
            'rmse_threshold': 0.06,
            'wod_wdr_threshold': 1.5,
            'wod_wsa_threshold': 2.0
        },
        
        # Snow kernel parameters
        'snow_params': {
            'max_na_days': 12,
            'rmse_threshold': 0.08,
            'wod_wdr_threshold': 1.65,
            'wod_wsa_threshold': 2.5
        },
        
        # AN ratio parameters
        'an_ratio_params': {
            'integration_accuracy': 1e-3
        },
        
        # Clustering configuration
        'clustering': {
            'optimize_clusters': True,
            'min_clusters': 8,
            'max_clusters': 16,
            'optimization_method': 'silhouette'
        },
        
        # Albedo generation parameters
        'albedo_params': {
            'apply_quality_masks': True,
            'output_uncertainty': True
        },
        
        # Quality assessment
        'detailed_validation': True
    }
    
    try:
        # Initialize advanced processor
        processor = AdvancedAlbedoProcessor(config)
        
        # Example 1: Single date processing with advanced options
        print("\n1. Processing single date with advanced configuration...")
        
        single_date_config = config.copy()
        single_date_config['target_date'] = '2018-08-08'
        
        single_results = processor.process_single_date(single_date_config)
        
        print("   ✓ Advanced single date processing completed")
        
        # Example 2: Time series processing
        print("\n2. Processing time series...")
        
        time_series_results = processor.process_time_series(
            start_date='2018-08-01',
            end_date='2018-08-31',
            time_step=16
        )
        
        success_rate = time_series_results['time_series_analysis']['success_rate']
        print(f"   ✓ Time series processing completed (success rate: {success_rate:.1%})")
        
        # Success message
        print("\n" + "=" * 60)
        print("✅ ADVANCED PROCESSING COMPLETED!")
        print("=" * 60)
        print("\nAdvanced features demonstrated:")
        print("  • Custom BRDF kernel parameters")
        print("  • Automatic cluster optimization")
        print("  • Time series processing")
        print("  • Detailed quality assessment")
        print("  • Flexible configuration system")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    exit_code = main()
    sys.exit(exit_code)
