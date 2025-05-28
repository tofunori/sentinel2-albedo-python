"""
Surface classification and clustering utilities.

This module provides functionality for K-means clustering of Sentinel-2
surface reflectance data to identify homogeneous surface types.

Original R implementation by Andre Bertoncini
Python port by Claude AI Assistant
"""

import logging
from typing import Optional, Dict, Tuple

import numpy as np
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class SurfaceClassifier:
    """
    Surface classification using K-means clustering.
    
    This class performs unsupervised classification of Sentinel-2 surface
    reflectance data to identify spectrally homogeneous surface types.
    """
    
    def __init__(self, random_state: int = 99):
        """
        Initialize surface classifier.
        
        Parameters
        ----------
        random_state : int, optional
            Random state for reproducible results
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def classify_surfaces(
        self, 
        reflectance_data: xr.Dataset,
        n_clusters: int = 8,
        max_iter: int = 1000,
        n_init: int = 10,
        standardize: bool = True
    ) -> xr.DataArray:
        """
        Perform K-means clustering on surface reflectance data.
        
        Parameters
        ----------
        reflectance_data : xr.Dataset
            Sentinel-2 surface reflectance data
        n_clusters : int, optional
            Number of clusters
        max_iter : int, optional
            Maximum number of iterations
        n_init : int, optional
            Number of initializations
        standardize : bool, optional
            Whether to standardize data before clustering
            
        Returns
        -------
        xr.DataArray
            Cluster classification map
        """
        self.logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        
        # Prepare data for clustering
        features, valid_mask, coords = self._prepare_features(reflectance_data)
        
        if features.shape[0] < n_clusters:
            self.logger.warning(f"Not enough valid pixels ({features.shape[0]}) for {n_clusters} clusters")
            # Return empty classification
            return self._create_empty_classification(reflectance_data, coords)
        
        # Standardize features if requested
        if standardize:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = features
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=self.random_state,
            algorithm='lloyd'
        )
        
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Convert labels to 1-based indexing (to match R implementation)
        cluster_labels = cluster_labels + 1
        
        # Create classification map
        classification_map = self._create_classification_map(
            cluster_labels, valid_mask, coords
        )
        
        # Compute clustering quality metrics
        if len(np.unique(cluster_labels)) > 1:
            silhouette = silhouette_score(features_scaled, cluster_labels - 1)  # Back to 0-based for sklearn
            self.logger.info(f"Clustering silhouette score: {silhouette:.3f}")
            classification_map.attrs['silhouette_score'] = silhouette
        
        # Store clustering parameters
        classification_map.attrs.update({
            'n_clusters': n_clusters,
            'standardized': standardize,
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_
        })
        
        return classification_map
    
    def _prepare_features(self, data: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Prepare feature matrix for clustering.
        
        Parameters
        ----------
        data : xr.Dataset
            Input reflectance data
            
        Returns
        -------
        tuple
            (features, valid_mask, coords) where features is 2D array,
            valid_mask is boolean array, and coords contains spatial coordinates
        """
        # Stack all data variables into a single array
        data_vars = list(data.data_vars.keys())
        
        if not data_vars:
            raise ValueError("No data variables found in dataset")
        
        # Create feature array
        feature_arrays = []
        
        for var in data_vars:
            var_data = data[var].values
            
            # Flatten spatial dimensions
            if var_data.ndim == 2:
                feature_arrays.append(var_data.flatten())
            else:
                raise ValueError(f"Unexpected dimensionality for variable {var}: {var_data.ndim}")
        
        # Stack features
        features = np.column_stack(feature_arrays)
        
        # Create valid data mask (no NaN or infinite values)
        valid_mask = np.all(np.isfinite(features), axis=1)
        
        # Get valid features only
        features_valid = features[valid_mask]
        
        # Store coordinate information
        coords = {
            'y': data[data_vars[0]].y,
            'x': data[data_vars[0]].x,
            'shape': data[data_vars[0]].shape
        }
        
        return features_valid, valid_mask, coords
    
    def _create_classification_map(
        self, 
        cluster_labels: np.ndarray, 
        valid_mask: np.ndarray, 
        coords: Dict
    ) -> xr.DataArray:
        """
        Create classification map from cluster labels.
        
        Parameters
        ----------
        cluster_labels : np.ndarray
            Cluster labels for valid pixels
        valid_mask : np.ndarray
            Boolean mask indicating valid pixels
        coords : dict
            Coordinate information
            
        Returns
        -------
        xr.DataArray
            Classification map
        """
        # Initialize classification array with NaN
        classification = np.full(valid_mask.shape, np.nan)
        
        # Assign cluster labels to valid pixels
        classification[valid_mask] = cluster_labels
        
        # Reshape to original spatial dimensions
        classification = classification.reshape(coords['shape'])
        
        # Create DataArray
        classification_map = xr.DataArray(
            classification,
            dims=['y', 'x'],
            coords={'y': coords['y'], 'x': coords['x']},
            attrs={
                'long_name': 'Surface classification',
                'description': 'K-means cluster labels',
                'valid_pixels': np.sum(valid_mask),
                'total_pixels': len(valid_mask)
            }
        )
        
        return classification_map
    
    def _create_empty_classification(self, data: xr.Dataset, coords: Dict) -> xr.DataArray:
        """
        Create empty classification map when clustering fails.
        
        Parameters
        ----------
        data : xr.Dataset
            Reference dataset for dimensions
        coords : dict
            Coordinate information
            
        Returns
        -------
        xr.DataArray
            Empty classification map
        """
        # Get shape from first variable
        var_name = list(data.data_vars.keys())[0]
        shape = data[var_name].shape
        
        # Create empty array
        empty_classification = np.full(shape, np.nan)
        
        # Create DataArray
        classification_map = xr.DataArray(
            empty_classification,
            dims=['y', 'x'],
            coords={'y': coords['y'], 'x': coords['x']},
            attrs={
                'long_name': 'Surface classification',
                'description': 'Empty classification (insufficient data)',
                'valid_pixels': 0,
                'total_pixels': np.prod(shape)
            }
        )
        
        return classification_map
    
    def optimize_cluster_number(
        self, 
        reflectance_data: xr.Dataset,
        min_clusters: int = 2,
        max_clusters: int = 20,
        method: str = 'silhouette'
    ) -> int:
        """
        Optimize the number of clusters using various metrics.
        
        Parameters
        ----------
        reflectance_data : xr.Dataset
            Surface reflectance data
        min_clusters : int, optional
            Minimum number of clusters to test
        max_clusters : int, optional
            Maximum number of clusters to test
        method : str, optional
            Optimization method ('silhouette', 'elbow')
            
        Returns
        -------
        int
            Optimal number of clusters
        """
        self.logger.info(f"Optimizing cluster number using {method} method...")
        
        # Prepare features
        features, valid_mask, coords = self._prepare_features(reflectance_data)
        
        if features.shape[0] < max_clusters:
            self.logger.warning(f"Not enough pixels for optimization, using {min_clusters} clusters")
            return min_clusters
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        scores = []
        n_clusters_range = range(min_clusters, max_clusters + 1)
        
        for n_clusters in n_clusters_range:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=5  # Reduced for speed during optimization
            )
            
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            if method == 'silhouette':
                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(features_scaled, cluster_labels)
                else:
                    score = -1
            elif method == 'elbow':
                score = -kmeans.inertia_  # Negative inertia for maximization
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            scores.append(score)
            self.logger.debug(f"n_clusters={n_clusters}, {method}_score={score:.3f}")
        
        # Find optimal number of clusters
        if method == 'silhouette':
            optimal_idx = np.argmax(scores)
        elif method == 'elbow':
            # Use elbow method (simplified)
            optimal_idx = self._find_elbow_point(scores)
        
        optimal_n_clusters = n_clusters_range[optimal_idx]
        
        self.logger.info(f"Optimal number of clusters: {optimal_n_clusters} (score: {scores[optimal_idx]:.3f})")
        
        return optimal_n_clusters
    
    def _find_elbow_point(self, scores: list) -> int:
        """
        Find elbow point in scores using knee-point detection.
        
        Parameters
        ----------
        scores : list
            List of scores (negative inertia values)
            
        Returns
        -------
        int
            Index of elbow point
        """
        # Convert to numpy array
        scores = np.array(scores)
        
        # Calculate second derivatives to find elbow
        if len(scores) < 3:
            return 0
        
        # Calculate differences
        first_diff = np.diff(scores)
        second_diff = np.diff(first_diff)
        
        # Find point with maximum second derivative (elbow)
        elbow_idx = np.argmax(second_diff) + 1  # +1 to account for diff operations
        
        return min(elbow_idx, len(scores) - 1)
    
    def get_cluster_statistics(
        self, 
        reflectance_data: xr.Dataset,
        classification_map: xr.DataArray
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute statistics for each cluster.
        
        Parameters
        ----------
        reflectance_data : xr.Dataset
            Surface reflectance data
        classification_map : xr.DataArray
            Cluster classification map
            
        Returns
        -------
        dict
            Dictionary containing statistics for each cluster
        """
        cluster_stats = {}
        
        # Get unique cluster labels (excluding NaN)
        unique_clusters = np.unique(classification_map.values)
        unique_clusters = unique_clusters[np.isfinite(unique_clusters)]
        
        for cluster_id in unique_clusters:
            cluster_id = int(cluster_id)
            
            # Create mask for this cluster
            cluster_mask = classification_map == cluster_id
            
            # Compute statistics for each band
            band_stats = {}
            
            for var_name, var_data in reflectance_data.data_vars.items():
                cluster_values = var_data.where(cluster_mask).values
                cluster_values = cluster_values[np.isfinite(cluster_values)]
                
                if len(cluster_values) > 0:
                    band_stats[var_name] = {
                        'mean': float(np.mean(cluster_values)),
                        'std': float(np.std(cluster_values)),
                        'min': float(np.min(cluster_values)),
                        'max': float(np.max(cluster_values)),
                        'count': len(cluster_values)
                    }
                else:
                    band_stats[var_name] = {
                        'mean': np.nan,
                        'std': np.nan,
                        'min': np.nan,
                        'max': np.nan,
                        'count': 0
                    }
            
            cluster_stats[cluster_id] = band_stats
        
        return cluster_stats
