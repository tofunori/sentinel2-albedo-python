#!/usr/bin/env python3
"""
Basic workflow example for Sentinel-2 albedo processing.

This script demonstrates the complete workflow from MODIS BRDF computation
to final Sentinel-2 albedo generation.

Author: Claude AI Assistant
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel2_albedo import S2AlbedoProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """
    Run the basic albedo processing workflow.
    """
    print("=" * 60)
    print("Sentinel-2 High Resolution Albedo Processing")
    print("Basic Workflow Example")
    print("=" * 60)
    
    # Configuration parameters
    config = {
        'target_date': '2018-08-08',
        'sensor': 'S2A',
        'study_area_extent': (421290, 496851.5, 5759734, 5800008.8),  # UTM coordinates
        'modis_data_path': '/path/to/modis/data',
        'sentinel2_data_path': '/path/to/sentinel2/data',
        'output_path': './outputs/basic_example',
        'dem_path': '/path/to/dem.tif',
        'landcover_path': '/path/to/landcover.tif'
    }
    
    print(f"Processing date: {config['target_date']}")
    print(f"Sensor: {config['sensor']}")
    print(f"Study area: {config['study_area_extent']}")
    print()
    
    try:
        # Initialize processor
        print("1. Initializing processor...")
        processor = S2AlbedoProcessor(
            target_date=config['target_date'],
            sensor=config['sensor'],
            study_area_extent=config['study_area_extent'],
            modis_data_path=config['modis_data_path'],
            sentinel2_data_path=config['sentinel2_data_path'],
            output_path=config['output_path'],
            dem_path=config['dem_path'],
            landcover_path=config['landcover_path']
        )
        
        # Step 1: Compute BRDF kernels
        print("\n2. Computing BRDF kernels...")
        print("   Using RTLSR (Ross-Thick Li-Sparse) kernel model")
        
        brdf_kernels = processor.compute_rtlsr_kernels(
            max_na_days=12,
            rmse_threshold=0.08,
            wod_wdr_threshold=1.65,
            wod_wsa_threshold=2.5
        )
        
        print(f"   ✓ BRDF kernels computed for {len([k for k in brdf_kernels.keys() if 'f_iso' in k])} bands")
        
        # Step 2: Compute albedo-to-nadir ratios
        print("\n3. Computing Albedo-to-Nadir ratios...")
        
        an_ratios = processor.compute_an_ratios()
        
        print(f"   ✓ AN ratios computed (BSA and WSA)")
        print(f"   ✓ Ratios available for {len(an_ratios['bsa'])} bands")
        
        # Step 3: Generate Sentinel-2 albedo
        print("\n4. Generating Sentinel-2 albedo maps...")
        print("   Testing cluster configurations from 6 to 20")
        
        albedo_maps = processor.compute_s2_albedo(
            cluster_range=(6, 20)
        )
        
        print(f"   ✓ Albedo maps generated for {len(albedo_maps)} cluster configurations")
        
        # Step 4: Quality assessment
        print("\n5. Performing quality assessment...")
        
        quality_stats = processor.validate_results()
        
        print("   ✓ Quality assessment completed")
        
        if quality_stats.get('overall', {}).get('recommended_config'):
            rec_config = quality_stats['overall']['recommended_config']
            print(f"   ✓ Recommended configuration: {rec_config} clusters")
        
        # Step 5: Generate summary report
        print("\n6. Generating summary report...")
        
        report_file = Path(config['output_path']) / 'processing_report.md'
        report_content = processor.export_summary_report(str(report_file))
        
        print(f"   ✓ Report saved to {report_file}")
        
        # Success message
        print("\n" + "=" * 60)
        print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Output directory: {config['output_path']}")
        print("\nGenerated products:")
        print("  • BRDF kernel parameters (f_iso, f_vol, f_geo)")
        print("  • Albedo-to-Nadir ratios (BSA and WSA)")
        print("  • High-resolution albedo maps (multiple configurations)")
        print("  • Quality assessment metrics")
        print("  • Processing summary report")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting tips:")
        print("  • Check that all input data paths are correct")
        print("  • Ensure MODIS and Sentinel-2 data are available for the target date")
        print("  • Verify that the study area extent is valid")
        print("  • Check the log output for detailed error information")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
