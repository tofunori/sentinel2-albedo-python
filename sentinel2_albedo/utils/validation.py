"""
Quality assessment and validation utilities.

This module provides functions for validating albedo products and
assessing their quality using various statistical metrics.

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import logging
from typing import Dict, Optional, Tuple, Any

import numpy as np
import xarray as xr
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class QualityAssessment:
    """
    Quality assessment tools for albedo products.
    
    This class provides methods to evaluate the quality of computed
    albedo products using statistical metrics and comparison with
    reference data.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def assess_quality(
        self, 
        albedo_data: Dict[str, Dict[str, xr.DataArray]],
        reference_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Assess overall quality of albedo products.
        
        Parameters
        ----------
        albedo_data : dict
            Dictionary containing albedo data for different cluster configurations
        reference_data : dict, optional
            Reference data for validation
            
        Returns
        -------
        dict
            Dictionary containing quality assessment results
        """
        self.logger.info("Assessing albedo product quality...")
        
        quality_stats = {
            'overall': {},
            'by_cluster_config': {},
            'validation': {}
        }
        
        # Assess each cluster configuration
        for n_clusters, cluster_data in albedo_data.items():
            self.logger.info(f"Assessing quality for {n_clusters} clusters...")
            
            config_stats = self._assess_config_quality(cluster_data)
            quality_stats['by_cluster_config'][n_clusters] = config_stats
        
        # Overall statistics
        quality_stats['overall'] = self._compute_overall_stats(albedo_data)
        
        # Validation against reference data
        if reference_data is not None:
            quality_stats['validation'] = self._validate_against_reference(
                albedo_data, reference_data
            )
        
        return quality_stats
    
    def _assess_config_quality(self, cluster_data: Dict[str, xr.DataArray]) -> Dict[str, Any]:
        """
        Assess quality for a specific cluster configuration.
        
        Parameters
        ----------
        cluster_data : dict
            Dictionary containing BSA, WSA, and cluster data
            
        Returns
        -------
        dict
            Quality statistics for this configuration
        """
        stats = {
            'bsa': {},
            'wsa': {},
            'clusters': {}
        }
        
        # Assess BSA quality
        if 'bsa' in cluster_data:
            stats['bsa'] = self._assess_albedo_quality(cluster_data['bsa'])
        
        # Assess WSA quality
        if 'wsa' in cluster_data:
            stats['wsa'] = self._assess_albedo_quality(cluster_data['wsa'])
        
        # Assess clustering quality
        if 'clusters' in cluster_data:
            stats['clusters'] = self._assess_clustering_quality(cluster_data['clusters'])
        
        return stats
    
    def _assess_albedo_quality(self, albedo_dataset: xr.Dataset) -> Dict[str, Any]:
        """
        Assess quality of albedo dataset.
        
        Parameters
        ----------
        albedo_dataset : xr.Dataset
            Albedo dataset containing multiple bands
            
        Returns
        -------
        dict
            Quality statistics
        """
        band_stats = {}
        
        for band_name, band_data in albedo_dataset.data_vars.items():
            values = band_data.values
            valid_values = values[np.isfinite(values)]
            
            if len(valid_values) > 0:
                band_stats[band_name] = {
                    'mean': float(np.mean(valid_values)),
                    'std': float(np.std(valid_values)),
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                    'median': float(np.median(valid_values)),
                    'percentile_5': float(np.percentile(valid_values, 5)),
                    'percentile_95': float(np.percentile(valid_values, 95)),
                    'valid_pixels': len(valid_values),
                    'total_pixels': values.size,
                    'valid_fraction': len(valid_values) / values.size,
                    'physical_range_check': self._check_physical_range(valid_values)
                }
            else:
                band_stats[band_name] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'median': np.nan,
                    'percentile_5': np.nan,
                    'percentile_95': np.nan,
                    'valid_pixels': 0,
                    'total_pixels': values.size,
                    'valid_fraction': 0.0,
                    'physical_range_check': False
                }
        
        # Overall dataset statistics
        overall_stats = self._compute_dataset_stats(band_stats)
        
        return {
            'bands': band_stats,
            'overall': overall_stats
        }
    
    def _assess_clustering_quality(self, cluster_map: xr.DataArray) -> Dict[str, Any]:
        """
        Assess quality of clustering results.
        
        Parameters
        ----------
        cluster_map : xr.DataArray
            Cluster classification map
            
        Returns
        -------
        dict
            Clustering quality statistics
        """
        values = cluster_map.values
        valid_values = values[np.isfinite(values)]
        
        if len(valid_values) == 0:
            return {
                'n_clusters': 0,
                'valid_pixels': 0,
                'cluster_sizes': {},
                'size_balance': np.nan,
                'silhouette_score': cluster_map.attrs.get('silhouette_score', np.nan)
            }
        
        # Get cluster statistics
        unique_clusters = np.unique(valid_values)
        n_clusters = len(unique_clusters)
        
        cluster_sizes = {}
        for cluster_id in unique_clusters:
            cluster_sizes[int(cluster_id)] = int(np.sum(valid_values == cluster_id))
        
        # Compute size balance (coefficient of variation)
        sizes = list(cluster_sizes.values())
        size_balance = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else np.nan
        
        return {
            'n_clusters': n_clusters,
            'valid_pixels': len(valid_values),
            'cluster_sizes': cluster_sizes,
            'size_balance': size_balance,
            'silhouette_score': cluster_map.attrs.get('silhouette_score', np.nan)
        }
    
    def _check_physical_range(self, values: np.ndarray) -> bool:
        """
        Check if albedo values are within physically reasonable range.
        
        Parameters
        ----------
        values : np.ndarray
            Albedo values
            
        Returns
        -------
        bool
            True if values are within reasonable range
        """
        # Albedo should be between 0 and 1
        return np.all((values >= 0) & (values <= 1))
    
    def _compute_dataset_stats(self, band_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute overall dataset statistics from band statistics.
        
        Parameters
        ----------
        band_stats : dict
            Statistics for individual bands
            
        Returns
        -------
        dict
            Overall dataset statistics
        """
        if not band_stats:
            return {}
        
        # Aggregate statistics across bands
        means = [stats['mean'] for stats in band_stats.values() if np.isfinite(stats['mean'])]
        valid_fractions = [stats['valid_fraction'] for stats in band_stats.values()]
        physical_checks = [stats['physical_range_check'] for stats in band_stats.values()]
        
        overall_stats = {
            'n_bands': len(band_stats),
            'mean_albedo': float(np.mean(means)) if means else np.nan,
            'std_albedo': float(np.std(means)) if means else np.nan,
            'mean_valid_fraction': float(np.mean(valid_fractions)),
            'min_valid_fraction': float(np.min(valid_fractions)),
            'physical_range_ok': all(physical_checks)
        }
        
        return overall_stats
    
    def _compute_overall_stats(self, albedo_data: Dict[str, Dict[str, xr.DataArray]]) -> Dict[str, Any]:
        """
        Compute overall statistics across all cluster configurations.
        
        Parameters
        ----------
        albedo_data : dict
            Albedo data for all configurations
            
        Returns
        -------
        dict
            Overall statistics
        """
        n_configs = len(albedo_data)
        config_keys = list(albedo_data.keys())
        
        return {
            'n_configurations': n_configs,
            'cluster_range': (min(config_keys), max(config_keys)) if config_keys else (0, 0),
            'recommended_config': self._recommend_configuration(albedo_data)
        }
    
    def _recommend_configuration(self, albedo_data: Dict[str, Dict[str, xr.DataArray]]) -> Optional[int]:
        """
        Recommend optimal cluster configuration based on quality metrics.
        
        Parameters
        ----------
        albedo_data : dict
            Albedo data for all configurations
            
        Returns
        -------
        int or None
            Recommended number of clusters
        """
        if not albedo_data:
            return None
        
        # Simple recommendation based on silhouette scores
        best_score = -1
        best_config = None
        
        for n_clusters, cluster_data in albedo_data.items():
            if 'clusters' in cluster_data:
                silhouette = cluster_data['clusters'].attrs.get('silhouette_score', -1)
                
                if silhouette > best_score:
                    best_score = silhouette
                    best_config = n_clusters
        
        return best_config
    
    def _validate_against_reference(
        self, 
        albedo_data: Dict[str, Dict[str, xr.DataArray]],
        reference_data: Dict
    ) -> Dict[str, Any]:
        """
        Validate albedo products against reference data.
        
        Parameters
        ----------
        albedo_data : dict
            Computed albedo data
        reference_data : dict
            Reference data for validation
            
        Returns
        -------
        dict
            Validation statistics
        """
        validation_stats = {}
        
        # This is a placeholder implementation
        # In practice, would compare against ground truth or other albedo products
        
        self.logger.info("Reference data validation not yet implemented")
        
        return validation_stats
    
    def compare_configurations(
        self, 
        albedo_data: Dict[str, Dict[str, xr.DataArray]]
    ) -> Dict[str, Any]:
        """
        Compare different cluster configurations.
        
        Parameters
        ----------
        albedo_data : dict
            Albedo data for different configurations
            
        Returns
        -------
        dict
            Comparison statistics
        """
        if len(albedo_data) < 2:
            return {'message': 'Need at least 2 configurations for comparison'}
        
        comparison_stats = {
            'configurations': list(albedo_data.keys()),
            'stability_metrics': {},
            'convergence_analysis': {}
        }
        
        # Analyze stability across configurations
        config_keys = sorted(albedo_data.keys())
        
        for i in range(len(config_keys) - 1):
            curr_config = config_keys[i]
            next_config = config_keys[i + 1]
            
            # Compare BSA values between consecutive configurations
            if ('bsa' in albedo_data[curr_config] and 
                'bsa' in albedo_data[next_config]):
                
                stability = self._compute_stability_metric(
                    albedo_data[curr_config]['bsa'],
                    albedo_data[next_config]['bsa']
                )
                
                comparison_stats['stability_metrics'][f'{curr_config}_to_{next_config}'] = stability
        
        return comparison_stats
    
    def _compute_stability_metric(
        self, 
        data1: xr.Dataset, 
        data2: xr.Dataset
    ) -> Dict[str, float]:
        """
        Compute stability metric between two datasets.
        
        Parameters
        ----------
        data1, data2 : xr.Dataset
            Datasets to compare
            
        Returns
        -------
        dict
            Stability metrics
        """
        stability_metrics = {}
        
        # Compare common variables
        common_vars = set(data1.data_vars.keys()) & set(data2.data_vars.keys())
        
        for var in common_vars:
            values1 = data1[var].values
            values2 = data2[var].values
            
            # Find common valid pixels
            valid_mask = np.isfinite(values1) & np.isfinite(values2)
            
            if np.sum(valid_mask) > 0:
                v1 = values1[valid_mask]
                v2 = values2[valid_mask]
                
                # Compute various metrics
                rmse = np.sqrt(mean_squared_error(v1, v2))
                mae = mean_absolute_error(v1, v2)
                correlation = stats.pearsonr(v1, v2)[0] if len(v1) > 1 else np.nan
                
                stability_metrics[var] = {
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'correlation': float(correlation),
                    'n_pixels': int(np.sum(valid_mask))
                }
            else:
                stability_metrics[var] = {
                    'rmse': np.nan,
                    'mae': np.nan,
                    'correlation': np.nan,
                    'n_pixels': 0
                }
        
        return stability_metrics
