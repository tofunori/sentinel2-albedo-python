#!/usr/bin/env python3
"""
Batch processing example for multiple dates and tiles.

This example demonstrates how to process multiple Sentinel-2 scenes
in batch mode, with parallel processing and error handling.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from sentinel2_albedo import S2AlbedoProcessor


class BatchProcessor:
    """
    Batch processor for multiple Sentinel-2 albedo processing tasks.
    """
    
    def __init__(self, base_config: dict, max_workers: int = 4):
        """
        Initialize batch processor.
        
        Parameters
        ----------
        base_config : dict
            Base configuration for processing
        max_workers : int, optional
            Maximum number of parallel workers
        """
        self.base_config = base_config
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Track processing results
        self.results = []
        self.errors = []
    
    def generate_processing_tasks(
        self, 
        date_range: Tuple[str, str],
        tiles: List[str],
        sensors: List[str] = ['S2A', 'S2B']
    ) -> List[Dict]:
        """
        Generate list of processing tasks.
        
        Parameters
        ----------
        date_range : tuple
            (start_date, end_date) as strings
        tiles : list
            List of tile identifiers
        sensors : list, optional
            List of sensors to process
            
        Returns
        -------
        list
            List of processing task configurations
        """
        start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
        end_date = datetime.strptime(date_range[1], '%Y-%m-%d')
        
        tasks = []
        current_date = start_date
        
        while current_date <= end_date:
            for tile in tiles:
                for sensor in sensors:
                    task_config = self.base_config.copy()
                    task_config.update({
                        'target_date': current_date.strftime('%Y-%m-%d'),
                        'sensor': sensor,
                        'tile_id': tile,
                        'output_path': str(Path(self.base_config['output_path']) / 
                                        f"{current_date.strftime('%Y%m%d')}_{sensor}_{tile}")
                    })
                    
                    tasks.append(task_config)
            
            # Move to next date (e.g., weekly intervals)
            current_date += timedelta(days=7)
        
        self.logger.info(f"Generated {len(tasks)} processing tasks")
        return tasks
    
    def process_single_task(self, task_config: dict) -> Dict:
        """
        Process a single albedo computation task.
        
        Parameters
        ----------
        task_config : dict
            Configuration for this specific task
            
        Returns
        -------
        dict
            Processing result
        """
        task_id = f"{task_config['target_date']}_{task_config['sensor']}_{task_config.get('tile_id', 'unknown')}"
        
        try:
            self.logger.info(f"Starting task: {task_id}")
            
            # Initialize processor
            processor = S2AlbedoProcessor(**task_config)
            
            # Execute processing steps
            start_time = datetime.now()
            
            # Step 1: BRDF kernels
            brdf_kernels = processor.compute_rtlsr_kernels()
            
            # Step 2: AN ratios
            an_ratios = processor.compute_an_ratios()
            
            # Step 3: Albedo maps
            albedo_maps = processor.compute_s2_albedo(cluster_range=(8, 12))
            
            # Step 4: Validation
            validation_stats = processor.validate_results()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'task_id': task_id,
                'status': 'success',
                'target_date': task_config['target_date'],
                'sensor': task_config['sensor'],
                'tile_id': task_config.get('tile_id'),
                'output_path': task_config['output_path'],
                'processing_time_seconds': processing_time,
                'n_brdf_kernels': len(brdf_kernels),
                'n_albedo_configs': len(albedo_maps),
                'validation_metrics': len(validation_stats) if validation_stats else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Task completed successfully: {task_id} ({processing_time:.1f}s)")
            return result
            
        except Exception as e:
            error_result = {
                'task_id': task_id,
                'status': 'error',
                'target_date': task_config['target_date'],
                'sensor': task_config['sensor'],
                'tile_id': task_config.get('tile_id'),
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.error(f"Task failed: {task_id} - {e}")
            return error_result
    
    def run_batch_processing(
        self, 
        tasks: List[Dict],
        parallel: bool = True
    ) -> Dict:
        """
        Run batch processing for all tasks.
        
        Parameters
        ----------
        tasks : list
            List of processing tasks
        parallel : bool, optional
            Whether to run tasks in parallel
            
        Returns
        -------
        dict
            Batch processing summary
        """
        self.logger.info(f"Starting batch processing for {len(tasks)} tasks")
        batch_start_time = datetime.now()
        
        if parallel and self.max_workers > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(self.process_single_task, task): task 
                                for task in tasks}
                
                # Collect results as they complete
                for future in as_completed(future_to_task):
                    result = future.result()
                    
                    if result['status'] == 'success':
                        self.results.append(result)
                    else:
                        self.errors.append(result)
        else:
            # Sequential processing
            for task in tasks:
                result = self.process_single_task(task)
                
                if result['status'] == 'success':
                    self.results.append(result)
                else:
                    self.errors.append(result)
        
        batch_processing_time = (datetime.now() - batch_start_time).total_seconds()
        
        # Generate summary
        summary = {
            'total_tasks': len(tasks),
            'successful_tasks': len(self.results),
            'failed_tasks': len(self.errors),
            'success_rate': len(self.results) / len(tasks) * 100 if tasks else 0,
            'total_processing_time_seconds': batch_processing_time,
            'average_task_time_seconds': (
                sum(r['processing_time_seconds'] for r in self.results) / len(self.results)
                if self.results else 0
            ),
            'processing_mode': 'parallel' if parallel else 'sequential',
            'max_workers': self.max_workers if parallel else 1,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Batch processing completed: {summary['successful_tasks']}/{len(tasks)} tasks successful")
        
        return summary
    
    def save_results(
        self, 
        output_dir: str,
        summary: Dict
    ):
        """
        Save batch processing results and summary.
        
        Parameters
        ----------
        output_dir : str
            Output directory
        summary : dict
            Batch processing summary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_file = output_path / 'batch_processing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save successful results
        if self.results:
            results_file = output_path / 'successful_tasks.json'
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
        
        # Save errors
        if self.errors:
            errors_file = output_path / 'failed_tasks.json'
            with open(errors_file, 'w') as f:
                json.dump(self.errors, f, indent=2)
        
        # Generate report
        report_content = self.generate_batch_report(summary)
        report_file = output_path / 'batch_processing_report.md'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Batch processing results saved to: {output_path}")
    
    def generate_batch_report(self, summary: Dict) -> str:
        """
        Generate batch processing report.
        
        Parameters
        ----------
        summary : dict
            Batch processing summary
            
        Returns
        -------
        str
            Report content
        """
        report_lines = [
            "# Batch Processing Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary",
            f"\n- **Total Tasks**: {summary['total_tasks']}",
            f"- **Successful**: {summary['successful_tasks']}",
            f"- **Failed**: {summary['failed_tasks']}",
            f"- **Success Rate**: {summary['success_rate']:.1f}%",
            f"- **Total Processing Time**: {summary['total_processing_time_seconds']:.1f} seconds",
            f"- **Average Task Time**: {summary['average_task_time_seconds']:.1f} seconds",
            f"- **Processing Mode**: {summary['processing_mode']}",
            f"- **Workers**: {summary['max_workers']}"
        ]
        
        # Add successful tasks details
        if self.results:
            report_lines.extend([
                "\n## Successful Tasks",
                "\n| Task ID | Date | Sensor | Processing Time (s) | Output Path |",
                "|---------|------|--------|-------------------|-------------|"            ])
            
            for result in self.results:
                report_lines.append(
                    f"| {result['task_id']} | {result['target_date']} | "
                    f"{result['sensor']} | {result['processing_time_seconds']:.1f} | "
                    f"{result['output_path']} |"
                )
        
        # Add error details
        if self.errors:
            report_lines.extend([
                "\n## Failed Tasks",
                "\n| Task ID | Date | Sensor | Error Message |",
                "|---------|------|--------|---------------|"            ])
            
            for error in self.errors:
                error_msg = error['error_message'][:50] + '...' if len(error['error_message']) > 50 else error['error_message']
                report_lines.append(
                    f"| {error['task_id']} | {error['target_date']} | "
                    f"{error['sensor']} | {error_msg} |"
                )
        
        report_lines.extend([
            "\n## Recommendations",
            "\n- Review failed tasks for common error patterns",
            "- Consider adjusting processing parameters for better success rates",
            "- Monitor processing times to optimize workflow efficiency",
            "\n---",
            "\n*Report generated by sentinel2-albedo batch processor*"
        ])
        
        return "\n".join(report_lines)


def main():
    """
    Main function for batch processing example.
    """
    logger.info("Starting batch processing example")
    
    # Base configuration
    base_config = {
        'study_area_extent': (421290, 496851.5, 5759734, 5800008.8),
        'modis_data_path': '/path/to/modis/data',
        'sentinel2_data_path': '/path/to/sentinel2/data',
        'output_path': './outputs/batch_processing',
        'dem_path': '/path/to/dem.tif',
        'landcover_path': '/path/to/landcover.tif'
    }
    
    # Initialize batch processor
    batch_processor = BatchProcessor(
        base_config=base_config,
        max_workers=2  # Adjust based on available resources
    )
    
    # Generate processing tasks
    tasks = batch_processor.generate_processing_tasks(
        date_range=('2018-08-01', '2018-08-31'),
        tiles=['T11UMT', 'T11UMS'],  # Example tile IDs
        sensors=['S2A', 'S2B']
    )
    
    # Limit tasks for demo purposes
    tasks = tasks[:4]  # Process only first 4 tasks
    
    logger.info(f"Processing {len(tasks)} tasks...")
    
    # Run batch processing
    summary = batch_processor.run_batch_processing(
        tasks=tasks,
        parallel=True
    )
    
    # Save results
    batch_processor.save_results(
        output_dir='./outputs/batch_results',
        summary=summary
    )
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETED")
    print("="*80)
    print(f"Total Tasks: {summary['total_tasks']}")
    print(f"Successful: {summary['successful_tasks']}")
    print(f"Failed: {summary['failed_tasks']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Total Time: {summary['total_processing_time_seconds']:.1f} seconds")
    print(f"Average Task Time: {summary['average_task_time_seconds']:.1f} seconds")
    print("="*80)
    
    if batch_processor.errors:
        print("\nErrors occurred:")
        for error in batch_processor.errors:
            print(f"- {error['task_id']}: {error['error_message']}")


if __name__ == '__main__':
    main()
