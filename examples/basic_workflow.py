#!/usr/bin/env python3
"""
Basic workflow example for Sentinel-2 albedo processing.

This example demonstrates the complete workflow from loading data
to generating albedo maps using the default configuration.
"""

import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the main processor
from sentinel2_albedo import S2AlbedoProcessor


def main():
    """
    Run basic albedo processing workflow.
    """
    logger.info("Starting basic Sentinel-2 albedo processing workflow")
    
    # Configuration
    config = {
        'target_date': '2018-08-08',
        'sensor': 'S2A',
        'study_area_extent': (421290, 496851.5, 5759734, 5800008.8),  # UTM coordinates
        'modis_data_path': '/path/to/modis/data',
        'sentinel2_data_path': '/path/to/sentinel2/data',
        'output_path': './outputs/basic_workflow',
        'dem_path': '/path/to/dem.tif',
        'landcover_path': '/path/to/landcover.tif'
    }
    
    try:
        # Initialize processor
        logger.info("Initializing S2AlbedoProcessor...")
        processor = S2AlbedoProcessor(**config)
        
        # Step 1: Compute BRDF kernels
        logger.info("Step 1: Computing BRDF kernels...")
        # Choose either RTLSR or Snow kernel
        brdf_kernels = processor.compute_rtlsr_kernels()
        # Alternative: brdf_kernels = processor.compute_snow_kernels()
        
        logger.info(f"BRDF kernels computed for {len(brdf_kernels)} parameters")
        
        # Step 2: Compute Albedo-to-Nadir ratios
        logger.info("Step 2: Computing AN ratios...")
        an_ratios = processor.compute_an_ratios()
        
        logger.info(f"AN ratios computed for BSA and WSA")
        
        # Step 3: Generate Sentinel-2 albedo maps
        logger.info("Step 3: Generating albedo maps...")
        albedo_maps = processor.compute_s2_albedo(
            cluster_range=(8, 16)  # Test cluster range from 8 to 16
        )
        
        logger.info(f"Albedo maps generated for {len(albedo_maps)} cluster configurations")
        
        # Step 4: Validate results
        logger.info("Step 4: Validating results...")
        validation_stats = processor.validate_results()
        
        # Step 5: Export summary report
        logger.info("Step 5: Generating summary report...")
        report_file = Path(config['output_path']) / 'processing_report.md'
        report_content = processor.export_summary_report(str(report_file))
        
        logger.info(f"Processing completed successfully!")
        logger.info(f"Results saved to: {config['output_path']}")
        logger.info(f"Report saved to: {report_file}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Target Date: {config['target_date']}")
        print(f"Sensor: {config['sensor']}")
        print(f"Study Area: {config['study_area_extent']}")
        print(f"\nBRDF Kernels: {len(brdf_kernels)} parameters")
        print(f"AN Ratios: BSA and WSA computed")
        print(f"Albedo Maps: {len(albedo_maps)} cluster configurations")
        
        if validation_stats:
            print(f"\nValidation completed with {len(validation_stats)} metrics")
        
        print(f"\nOutput Directory: {config['output_path']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in processing workflow: {e}")
        raise


if __name__ == '__main__':
    main()
