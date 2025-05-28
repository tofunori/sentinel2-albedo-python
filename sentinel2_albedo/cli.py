#!/usr/bin/env python3
"""
Command-line interface for Sentinel-2 albedo processing.

This module provides a command-line interface to the sentinel2-albedo
package, allowing users to process albedo products from the terminal.

Author: Claude AI Assistant
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import S2AlbedoProcessor, __version__


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    verbose : bool, optional
        Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_date(date_string: str) -> str:
    """
    Validate date string format.
    
    Parameters
    ----------
    date_string : str
        Date string in YYYY-MM-DD format
        
    Returns
    -------
    str
        Validated date string
    """
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return date_string
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {date_string}. Use YYYY-MM-DD format."
        )


def validate_extent(extent_string: str) -> tuple:
    """
    Validate and parse extent string.
    
    Parameters
    ----------
    extent_string : str
        Extent string in format "xmin,xmax,ymin,ymax"
        
    Returns
    -------
    tuple
        Parsed extent tuple
    """
    try:
        parts = extent_string.split(',')
        if len(parts) != 4:
            raise ValueError("Extent must have 4 values")
        
        extent = tuple(float(x) for x in parts)
        return extent
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid extent format: {extent_string}. Use 'xmin,xmax,ymin,ymax' format. {e}"
        )


def create_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.
    
    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Process high-resolution albedo from Sentinel-2 and MODIS data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  s2-albedo --date 2018-08-08 --sensor S2A --extent "421290,496851.5,5759734,5800008.8" \
            --modis-path /data/modis --s2-path /data/sentinel2 --output ./results
  
  # Processing with snow kernel
  s2-albedo --date 2018-12-15 --sensor S2A --extent "421290,496851.5,5759734,5800008.8" \
            --modis-path /data/modis --s2-path /data/sentinel2 --output ./results \
            --kernel-model snow --clusters 8,12
  
  # Processing with custom parameters
  s2-albedo --date 2018-08-08 --sensor S2B --extent "421290,496851.5,5759734,5800008.8" \
            --modis-path /data/modis --s2-path /data/sentinel2 --output ./results \
            --rmse-threshold 0.06 --max-na-days 10 --verbose
"""
    )
    
    # Version
    parser.add_argument(
        '--version', action='version', version=f'sentinel2-albedo {__version__}'
    )
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    
    required.add_argument(
        '--date', type=validate_date, required=True,
        help='Target date in YYYY-MM-DD format'
    )
    
    required.add_argument(
        '--sensor', choices=['S2A', 'S2B'], required=True,
        help='Sentinel-2 sensor (S2A or S2B)'
    )
    
    required.add_argument(
        '--extent', type=validate_extent, required=True,
        help='Study area extent as "xmin,xmax,ymin,ymax" in UTM coordinates'
    )
    
    required.add_argument(
        '--modis-path', type=Path, required=True,
        help='Path to MODIS MOD09GA data directory'
    )
    
    required.add_argument(
        '--s2-path', type=Path, required=True,
        help='Path to Sentinel-2 data directory'
    )
    
    required.add_argument(
        '--output', type=Path, required=True,
        help='Output directory for results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--dem-path', type=Path,
        help='Path to digital elevation model (for topographic correction)'
    )
    
    parser.add_argument(
        '--landcover-path', type=Path,
        help='Path to land cover classification (for SBAF correction)'
    )
    
    parser.add_argument(
        '--kernel-model', choices=['rtlsr', 'snow'], default='rtlsr',
        help='BRDF kernel model to use (default: rtlsr)'
    )
    
    parser.add_argument(
        '--clusters', type=str, default='6,20',
        help='Cluster range as "min,max" (default: 6,20)'
    )
    
    # Quality parameters
    quality_group = parser.add_argument_group('quality parameters')
    
    quality_group.add_argument(
        '--max-na-days', type=int, default=12,
        help='Maximum number of days with NA values (default: 12)'
    )
    
    quality_group.add_argument(
        '--rmse-threshold', type=float, default=0.08,
        help='RMSE threshold for quality filtering (default: 0.08)'
    )
    
    quality_group.add_argument(
        '--wod-wdr-threshold', type=float, default=1.65,
        help='WoD WDR threshold for quality filtering (default: 1.65)'
    )
    
    quality_group.add_argument('
        '--wod-wsa-threshold', type=float, default=2.5,
        help='WoD WSA threshold for quality filtering (default: 2.5)'
    )
    
    # Processing options
    processing_group = parser.add_argument_group('processing options')
    
    processing_group.add_argument(
        '--skip-brdf', action='store_true',
        help='Skip BRDF kernel computation (use existing results)'
    )
    
    processing_group.add_argument(
        '--skip-ratios', action='store_true',
        help='Skip AN ratio computation (use existing results)'
    )
    
    processing_group.add_argument(
        '--optimize-clusters', action='store_true',
        help='Automatically optimize number of clusters'
    )
    
    # Output options
    output_group = parser.add_argument_group('output options')
    
    output_group.add_argument(
        '--no-report', action='store_true',
        help='Skip generation of summary report'
    )
    
    output_group.add_argument(
        '--save-intermediate', action='store_true',
        help='Save intermediate processing results'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress all output except errors'
    )
    
    return parser


def main() -> int:
    """
    Main CLI function.
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        logging.disable(logging.CRITICAL)
    else:
        setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        if not args.modis_path.exists():
            raise FileNotFoundError(f"MODIS data path does not exist: {args.modis_path}")
        
        if not args.s2_path.exists():
            raise FileNotFoundError(f"Sentinel-2 data path does not exist: {args.s2_path}")
        
        if args.dem_path and not args.dem_path.exists():
            raise FileNotFoundError(f"DEM path does not exist: {args.dem_path}")
        
        if args.landcover_path and not args.landcover_path.exists():
            raise FileNotFoundError(f"Land cover path does not exist: {args.landcover_path}")
        
        # Parse cluster range
        try:
            cluster_parts = args.clusters.split(',')
            if len(cluster_parts) != 2:
                raise ValueError("Cluster range must have 2 values")
            cluster_range = (int(cluster_parts[0]), int(cluster_parts[1]))
        except ValueError as e:
            raise ValueError(f"Invalid cluster range format: {args.clusters}. {e}")
        
        # Print processing information
        if not args.quiet:
            print("=" * 60)
            print("Sentinel-2 High Resolution Albedo Processing")
            print("=" * 60)
            print(f"Date: {args.date}")
            print(f"Sensor: {args.sensor}")
            print(f"Study area: {args.extent}")
            print(f"Kernel model: {args.kernel_model}")
            print(f"Cluster range: {cluster_range}")
            print(f"Output: {args.output}")
            print()
        
        # Initialize processor
        logger.info("Initializing albedo processor...")
        
        processor = S2AlbedoProcessor(
            target_date=args.date,
            sensor=args.sensor,
            study_area_extent=args.extent,
            modis_data_path=args.modis_path,
            sentinel2_data_path=args.s2_path,
            output_path=args.output,
            dem_path=args.dem_path,
            landcover_path=args.landcover_path
        )
        
        # Step 1: BRDF kernel computation
        if not args.skip_brdf:
            logger.info("Computing BRDF kernels...")
            
            kernel_params = {
                'max_na_days': args.max_na_days,
                'rmse_threshold': args.rmse_threshold,
                'wod_wdr_threshold': args.wod_wdr_threshold,
                'wod_wsa_threshold': args.wod_wsa_threshold
            }
            
            if args.kernel_model == 'rtlsr':
                brdf_kernels = processor.compute_rtlsr_kernels(**kernel_params)
            else:  # snow
                brdf_kernels = processor.compute_snow_kernels(**kernel_params)
        
        # Step 2: AN ratio computation
        if not args.skip_ratios:
            logger.info("Computing Albedo-to-Nadir ratios...")
            an_ratios = processor.compute_an_ratios()
        
        # Step 3: Albedo generation
        logger.info("Generating Sentinel-2 albedo maps...")
        
        if args.optimize_clusters:
            # Use automatic cluster optimization
            from sentinel2_albedo.utils.clustering import SurfaceClassifier
            
            s2_data = processor.s2_handler.load_surface_reflectance(
                datetime.strptime(args.date, '%Y-%m-%d'),
                args.extent
            )
            
            classifier = SurfaceClassifier()
            optimal_clusters = classifier.optimize_cluster_number(
                s2_data,
                min_clusters=cluster_range[0],
                max_clusters=cluster_range[1]
            )
            
            cluster_range = (optimal_clusters, optimal_clusters)
            logger.info(f"Using optimized cluster number: {optimal_clusters}")
        
        albedo_maps = processor.compute_s2_albedo(cluster_range=cluster_range)
        
        # Step 4: Quality assessment
        logger.info("Performing quality assessment...")
        quality_stats = processor.validate_results()
        
        # Step 5: Generate report
        if not args.no_report:
            logger.info("Generating summary report...")
            report_file = args.output / 'processing_report.md'
            processor.export_summary_report(str(report_file))
        
        # Success message
        if not args.quiet:
            print("\n" + "=" * 60)
            print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Results saved to: {args.output}")
        
        logger.info("Processing completed successfully")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        
        if not args.quiet:
            print(f"\n❌ ERROR: {e}")
            print("\nFor more detailed error information, use --verbose flag")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
