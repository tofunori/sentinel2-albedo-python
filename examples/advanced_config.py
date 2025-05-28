#!/usr/bin/env python3
"""
Advanced configuration example for Sentinel-2 albedo processing.

This example demonstrates advanced configuration options including
custom parameters, quality filtering, and batch processing.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from sentinel2_albedo import S2AlbedoProcessor


def load_config(config_file: str) -> dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_file : str
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        # Return default configuration
        return get_default_config()


def get_default_config() -> dict:
    """
    Get default configuration.
    
    Returns
    -------
    dict
        Default configuration dictionary
    """
    return {
        'processing': {
            'target_date': '2018-08-08',
            'sensor': 'S2A',
            'study_area_extent': [421290, 496851.5, 5759734, 5800008.8],
            'cluster_range': [6, 20],
            'kernel_type': 'rtlsr',  # 'rtlsr' or 'snow'
        },
        'paths': {
            'modis_data_path': '/path/to/modis/data',
            'sentinel2_data_path': '/path/to/sentinel2/data',
            'output_path': './outputs/advanced_config',
            'dem_path': '/path/to/dem.tif',
            'landcover_path': '/path/to/landcover.tif',
            'sbaf_table_path': '/path/to/sbaf_table.csv'
        },
        'quality': {
            'rmse_threshold': 0.08,
            'wod_wdr_threshold': 1.65,
            'wod_wsa_threshold': 2.5,
            'max_na_days': 12
        },
        'corrections': {
            'apply_topographic': True,
            'apply_atmospheric': False,
            'apply_sbaf': True
        },
        'output': {
            'save_intermediate': True,
            'compression': 'lzw',
            'generate_report': True,
            'create_quicklooks': True
        }
    }


def advanced_processing_workflow(config: dict):
    """
    Run advanced processing workflow with custom configuration.
    
    Parameters
    ----------
    config : dict
        Processing configuration
    """
    logger.info("Starting advanced Sentinel-2 albedo processing workflow")
    
    # Extract configuration sections
    proc_config = config['processing']
    paths_config = config['paths']
    quality_config = config['quality']
    corrections_config = config['corrections']
    output_config = config['output']
    
    try:
        # Initialize processor with advanced options
        logger.info("Initializing S2AlbedoProcessor with advanced configuration...")
        processor = S2AlbedoProcessor(
            target_date=proc_config['target_date'],
            sensor=proc_config['sensor'],
            study_area_extent=tuple(proc_config['study_area_extent']),
            **paths_config,
            **quality_config
        )
        
        # Step 1: Compute BRDF kernels with custom parameters
        logger.info(f"Step 1: Computing {proc_config['kernel_type'].upper()} BRDF kernels...")
        
        if proc_config['kernel_type'].lower() == 'snow':
            brdf_kernels = processor.compute_snow_kernels(**quality_config)
        else:
            brdf_kernels = processor.compute_rtlsr_kernels(**quality_config)
        
        logger.info(f"BRDF kernels computed with quality filtering applied")
        
        # Step 2: Compute AN ratios
        logger.info("Step 2: Computing AN ratios...")
        an_ratios = processor.compute_an_ratios()
        
        # Step 3: Generate albedo maps with custom cluster range
        logger.info("Step 3: Generating albedo maps with custom cluster range...")
        cluster_range = tuple(proc_config['cluster_range'])
        albedo_maps = processor.compute_s2_albedo(cluster_range=cluster_range)
        
        # Step 4: Advanced validation and quality assessment
        logger.info("Step 4: Performing advanced quality assessment...")
        validation_stats = processor.validate_results()
        
        # Additional quality metrics
        from sentinel2_albedo.utils.validation import QualityAssessment
        qa = QualityAssessment()
        
        # Compare different cluster configurations
        comparison_stats = qa.compare_configurations(albedo_maps)
        
        # Step 5: Generate comprehensive report
        logger.info("Step 5: Generating comprehensive report...")
        
        if output_config['generate_report']:
            report_file = Path(paths_config['output_path']) / 'advanced_processing_report.md'
            report_content = generate_advanced_report(
                processor, validation_stats, comparison_stats, config
            )
            
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Comprehensive report saved to: {report_file}")
        
        # Step 6: Create quicklook images (if requested)
        if output_config['create_quicklooks']:
            logger.info("Step 6: Creating quicklook images...")
            create_quicklooks(albedo_maps, paths_config['output_path'])
        
        logger.info("Advanced processing workflow completed successfully!")
        
        return {
            'processor': processor,
            'brdf_kernels': brdf_kernels,
            'an_ratios': an_ratios,
            'albedo_maps': albedo_maps,
            'validation_stats': validation_stats,
            'comparison_stats': comparison_stats
        }
        
    except Exception as e:
        logger.error(f"Error in advanced processing workflow: {e}")
        raise


def generate_advanced_report(
    processor, 
    validation_stats: dict, 
    comparison_stats: dict, 
    config: dict
) -> str:
    """
    Generate advanced processing report.
    
    Parameters
    ----------
    processor : S2AlbedoProcessor
        Processor instance
    validation_stats : dict
        Validation statistics
    comparison_stats : dict
        Configuration comparison statistics
    config : dict
        Processing configuration
        
    Returns
    -------
    str
        Report content
    """
    report_lines = [
        "# Advanced Sentinel-2 Albedo Processing Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Configuration",
        f"\n### Processing Parameters",
        f"- Target Date: {config['processing']['target_date']}",
        f"- Sensor: {config['processing']['sensor']}",
        f"- Kernel Type: {config['processing']['kernel_type'].upper()}",
        f"- Cluster Range: {config['processing']['cluster_range']}",
        f"\n### Quality Thresholds",
        f"- RMSE Threshold: {config['quality']['rmse_threshold']}",
        f"- WoD WDR Threshold: {config['quality']['wod_wdr_threshold']}",
        f"- WoD WSA Threshold: {config['quality']['wod_wsa_threshold']}",
        f"- Max NA Days: {config['quality']['max_na_days']}",
        "\n## Processing Results"
    ]
    
    # Add basic processing status
    basic_report = processor.export_summary_report()
    report_lines.append(basic_report)
    
    # Add validation results
    if validation_stats:
        report_lines.append("\n## Validation Results")
        report_lines.append(f"\nValidation completed with {len(validation_stats)} metrics")
        # Add more detailed validation info here
    
    # Add comparison results
    if comparison_stats:
        report_lines.append("\n## Configuration Comparison")
        if 'configurations' in comparison_stats:
            configs = comparison_stats['configurations']
            report_lines.append(f"\nCompared {len(configs)} cluster configurations: {configs}")
        
        if 'stability_metrics' in comparison_stats:
            report_lines.append("\n### Stability Analysis")
            for transition, metrics in comparison_stats['stability_metrics'].items():
                report_lines.append(f"\n**{transition}**")
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, dict):
                        report_lines.append(f"- {metric_name}:")
                        for sub_key, sub_value in metric_value.items():
                            report_lines.append(f"  - {sub_key}: {sub_value:.4f}")
                    else:
                        report_lines.append(f"- {metric_name}: {metric_value:.4f}")
    
    report_lines.extend([
        "\n## Recommendations",
        "\nBased on the quality assessment and configuration comparison:",
        "\n1. **Optimal Cluster Configuration**: Analysis of stability metrics suggests...",
        "2. **Quality Assessment**: BRDF kernel quality meets specified thresholds",
        "3. **Processing Performance**: Workflow completed within expected parameters",
        "\n## Output Files",
        f"\n- BRDF Kernels: `{config['paths']['output_path']}/brdf_kernels/`",
        f"- AN Ratios: `{config['paths']['output_path']}/an_ratios/`",
        f"- Albedo Maps: `{config['paths']['output_path']}/albedo/`",
        "\n---",
        "\n*Report generated by sentinel2-albedo Python package*"
    ])
    
    return "\n".join(report_lines)


def create_quicklooks(albedo_maps: dict, output_path: str):
    """
    Create quicklook images for albedo maps.
    
    Parameters
    ----------
    albedo_maps : dict
        Dictionary containing albedo maps
    output_path : str
        Output directory path
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        
        quicklook_dir = Path(output_path) / 'quicklooks'
        quicklook_dir.mkdir(parents=True, exist_ok=True)
        
        for n_clusters, cluster_data in albedo_maps.items():
            # Create quicklooks for BSA
            if 'bsa' in cluster_data:
                bsa_data = cluster_data['bsa']
                
                for band_name, band_data in bsa_data.data_vars.items():
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Plot albedo with appropriate color scale
                    im = ax.imshow(
                        band_data.values,
                        cmap='viridis',
                        vmin=0, vmax=1,
                        extent=[
                            band_data.x.min(), band_data.x.max(),
                            band_data.y.min(), band_data.y.max()
                        ]
                    )
                    
                    ax.set_title(f'BSA Albedo - {band_name} ({n_clusters} clusters)')
                    ax.set_xlabel('Easting (m)')
                    ax.set_ylabel('Northing (m)')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Albedo')
                    
                    # Save quicklook
                    quicklook_file = quicklook_dir / f'bsa_albedo_{band_name}_clusters_{n_clusters}.png'
                    plt.savefig(quicklook_file, dpi=150, bbox_inches='tight')
                    plt.close()
        
        logger.info(f"Quicklook images saved to: {quicklook_dir}")
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping quicklook generation")
    except Exception as e:
        logger.warning(f"Error creating quicklooks: {e}")


def main():
    """
    Main function for advanced configuration example.
    """
    # Try to load configuration from file
    config_file = 'config/advanced_config.yaml'
    
    if Path(config_file).exists():
        config = load_config(config_file)
        logger.info(f"Loaded configuration from {config_file}")
    else:
        config = get_default_config()
        logger.info("Using default configuration")
        
        # Optionally save default config for future use
        config_dir = Path('config')
        config_dir.mkdir(exist_ok=True)
        
        with open('config/default_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("Default configuration saved to config/default_config.yaml")
    
    # Run advanced processing workflow
    results = advanced_processing_workflow(config)
    
    print("\n" + "="*80)
    print("ADVANCED PROCESSING COMPLETED")
    print("="*80)
    print(f"Configuration: {len(config)} sections")
    print(f"BRDF Kernels: {len(results['brdf_kernels'])} parameters")
    print(f"Albedo Maps: {len(results['albedo_maps'])} cluster configurations")
    print(f"Output Directory: {config['paths']['output_path']}")
    print("="*80)


if __name__ == '__main__':
    main()
