"""
Model Aggregation Module for Federated Learning

This module provides optimized model aggregation methods for federated learning in QTrust blockchain.
It implements various state-of-the-art aggregation strategies to enhance security, fairness, 
and performance in distributed model training.

Key features:
- Multiple aggregation methods (weighted average, median, trimmed mean, Krum)
- Byzantine-resilient aggregation techniques
- Adaptive methods based on trust and performance scores
- Regularization techniques like FedProx to maintain model stability

The module includes two main classes:
- ModelAggregator: Implements core aggregation algorithms
- ModelAggregationManager: Provides intelligent selection and management of aggregation methods
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import copy
import logging
from qtrust.utils.cache import lru_cache, tensor_cache, compute_hash

logger = logging.getLogger("qtrust.federated.aggregation")

class ModelAggregator:
    """
    Class implementing optimized model aggregation methods for Federated Learning.
    """
    def __init__(self):
        """
        Initialize the optimized model aggregation class.
        """
        # Historical data
        self.aggregation_history = []
        
    @staticmethod
    def weighted_average(params_list: List[Dict[str, torch.Tensor]], 
                        weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Perform model aggregation using weighted average.
        
        Args:
            params_list: List of model parameter dictionaries from clients
            weights: List of weights for each client
            
        Returns:
            Dict: Aggregated model parameters
        """
        if not params_list:
            raise ValueError("No parameters to aggregate")
            
        if len(params_list) != len(weights):
            raise ValueError("Number of parameters and weights don't match")
        
        # Normalize weights
        weights_sum = sum(weights)
        if weights_sum == 0:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        else:
            normalized_weights = [w / weights_sum for w in weights]
        
        # Initialize result parameters from the first client
        result_params = copy.deepcopy(params_list[0])
        
        # Perform weighted aggregation
        for key in result_params:
            result_params[key] = torch.zeros_like(result_params[key])
            for i, params in enumerate(params_list):
                result_params[key] += normalized_weights[i] * params[key]
                
        return result_params
    
    @staticmethod
    def median(params_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Perform model aggregation using median value.
        This method helps defend against Byzantine attacks.
        
        Args:
            params_list: List of model parameter dictionaries from clients
            
        Returns:
            Dict: Aggregated model parameters
        """
        if not params_list:
            raise ValueError("No parameters to aggregate")
            
        # Initialize result parameters from the first client
        result_params = copy.deepcopy(params_list[0])
        
        # Perform median aggregation for each parameter
        for key in result_params.keys():
            # Get all values from clients for current parameter
            param_tensors = [params[key].cpu().numpy() for params in params_list]
            
            # Calculate median values along the client axis
            median_values = np.median(param_tensors, axis=0)
            
            # Convert back to tensor
            result_params[key] = torch.tensor(median_values, dtype=result_params[key].dtype, 
                                            device=result_params[key].device)
                
        return result_params
    
    @staticmethod
    def trimmed_mean(params_list: List[Dict[str, torch.Tensor]], 
                    trim_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Perform model aggregation using trimmed mean.
        This method removes % of highest and lowest values before calculating the mean.
        
        Args:
            params_list: List of model parameter dictionaries from clients
            trim_ratio: Percentage of trimming (0.1 = trim 10% highest and 10% lowest)
            
        Returns:
            Dict: Aggregated model parameters
        """
        if not params_list:
            raise ValueError("No parameters to aggregate")
            
        if trim_ratio < 0 or trim_ratio > 0.5:
            raise ValueError("Trim ratio must be in range [0, 0.5]")
            
        # Initialize result parameters from the first client
        result_params = copy.deepcopy(params_list[0])
        
        # Perform trimmed mean aggregation for each parameter
        for key in result_params.keys():
            # Get all values from clients for current parameter
            param_tensors = [params[key].cpu().numpy() for params in params_list]
            
            # Number of clients to trim at each end
            num_clients = len(param_tensors)
            k = int(np.ceil(num_clients * trim_ratio))
            
            if 2*k >= num_clients:
                # If trimming too many, use regular mean method
                mean_values = np.mean(param_tensors, axis=0)
            else:
                # Sort values in ascending order along client axis
                sorted_params = np.sort(param_tensors, axis=0)
                
                # Calculate mean of remaining values after trimming
                mean_values = np.mean(sorted_params[k:num_clients-k], axis=0)
            
            # Convert back to tensor
            result_params[key] = torch.tensor(mean_values, dtype=result_params[key].dtype, 
                                            device=result_params[key].device)
                
        return result_params
    
    @staticmethod
    def krum(params_list: List[Dict[str, torch.Tensor]], 
            num_byzantine: int = 0) -> Dict[str, torch.Tensor]:
        """
        Perform model aggregation using the Krum algorithm.
        Krum selects the client with the smallest sum of Euclidean distances to n-f-2 closest clients.
        
        Args:
            params_list: List of model parameter dictionaries from clients
            num_byzantine: Maximum number of Byzantine clients possible
            
        Returns:
            Dict: Aggregated model parameters
        """
        if not params_list:
            raise ValueError("No parameters to aggregate")
            
        num_clients = len(params_list)
        if num_byzantine >= num_clients / 2:
            raise ValueError("Number of Byzantine clients too large")
            
        # Convert parameters to flat vectors
        flat_params = []
        for params in params_list:
            flat_param = torch.cat([param.flatten() for param in params.values()])
            flat_params.append(flat_param)
        
        # Calculate distances between client pairs
        distances = torch.zeros(num_clients, num_clients)
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                dist = torch.norm(flat_params[i] - flat_params[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Krum score is the sum of distances to n-f-2 closest clients
        scores = torch.zeros(num_clients)
        for i in range(num_clients):
            # Get n-f-2 closest clients
            closest_distances = torch.topk(distances[i], k=num_clients-num_byzantine-2, 
                                         largest=False).values
            scores[i] = torch.sum(closest_distances)
        
        # Select client with lowest score
        best_client_idx = torch.argmin(scores).item()
        
        return copy.deepcopy(params_list[best_client_idx])
    
    @staticmethod
    def adaptive_federated_averaging(params_list: List[Dict[str, torch.Tensor]],
                                   trust_scores: List[float],
                                   performance_scores: List[float],
                                   adaptive_alpha: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Perform model aggregation using adaptive averaging based on trust and performance scores.
        
        Args:
            params_list: List of model parameter dictionaries from clients
            trust_scores: Trust scores for each client
            performance_scores: Performance scores for each client
            adaptive_alpha: Ratio for combining trust and performance
            
        Returns:
            Dict: Aggregated model parameters
        """
        if not params_list:
            raise ValueError("No parameters to aggregate")
            
        if len(params_list) != len(trust_scores) or len(params_list) != len(performance_scores):
            raise ValueError("Number of parameters, trust scores, and performance scores don't match")
        
        # Calculate weights based on combination of trust and performance
        combined_scores = [adaptive_alpha * trust + (1 - adaptive_alpha) * perf 
                          for trust, perf in zip(trust_scores, performance_scores)]
        
        # Normalize scores to weights
        total_score = sum(combined_scores)
        if total_score == 0:
            weights = [1.0 / len(combined_scores)] * len(combined_scores)
        else:
            weights = [score / total_score for score in combined_scores]
        
        # Aggregate using weighted average
        return ModelAggregator.weighted_average(params_list, weights)
    
    @staticmethod
    def fedprox(params_list: List[Dict[str, torch.Tensor]],
               global_params: Dict[str, torch.Tensor],
               weights: List[float],
               mu: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Perform model aggregation using FedProx, adding proximity constraint to global model.
        
        Args:
            params_list: List of model parameter dictionaries from clients
            global_params: Current global model parameters
            weights: List of weights for each client
            mu: Regularization coefficient
            
        Returns:
            Dict: Aggregated model parameters
        """
        if not params_list:
            raise ValueError("No parameters to aggregate")
            
        if len(params_list) != len(weights):
            raise ValueError("Number of parameters and weights don't match")
        
        # Basic aggregation using weighted average
        aggregated_params = ModelAggregator.weighted_average(params_list, weights)
        
        # Add regularization to keep new model close to global model
        for key in aggregated_params:
            aggregated_params[key] = (1 - mu) * aggregated_params[key] + mu * global_params[key]
                
        return aggregated_params

class ModelAggregationManager:
    """
    Manages model aggregation with optimized methods.
    """
    def __init__(self, default_method: str = 'weighted_average'):
        """
        Initialize the model aggregation manager.
        
        Args:
            default_method: Default aggregation method
        """
        self.default_method = default_method
        self.aggregator = ModelAggregator()
        
        # Store information about aggregation sessions
        self.session_history = []
        self.performance_metrics = {}
        
    def recommend_method(self, 
                        num_clients: int,
                        has_trust_scores: bool = False,
                        suspected_byzantine: bool = False) -> str:
        """
        Recommend the best aggregation method based on current conditions.
        
        Args:
            num_clients: Number of participating clients
            has_trust_scores: Whether trust scores are available
            suspected_byzantine: Whether Byzantine clients are suspected
            
        Returns:
            str: Name of recommended aggregation method
        """
        if suspected_byzantine:
            if num_clients >= 4:
                return 'median'  # Robust against Byzantine
            else:
                return 'trimmed_mean'  # Fewer clients but still robust
        
        if has_trust_scores:
            return 'adaptive_fedavg'  # Use trust scores
            
        if num_clients >= 10:
            return 'weighted_average'  # Efficient with many clients
            
        return self.default_method
    
    def aggregate(self, 
                 method: str,
                 params_list: List[Dict[str, torch.Tensor]],
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        Perform model aggregation with the chosen method.
        
        Args:
            method: Name of aggregation method
            params_list: List of model parameters from clients
            **kwargs: Additional parameters (weights, trust_scores, etc.)
            
        Returns:
            Dict: Aggregated model parameters
        """
        if method == 'weighted_average':
            weights = kwargs.get('weights', [1.0/len(params_list)] * len(params_list))
            return self.aggregator.weighted_average(params_list, weights)
            
        elif method == 'adaptive_fedavg':
            # Combine trust and performance scores
            trust_scores = kwargs.get('trust_scores', [1.0] * len(params_list))
            performance_scores = kwargs.get('performance_scores', [1.0] * len(params_list))
            
            # Calculate combined weights
            combined_weights = []
            for trust, perf in zip(trust_scores, performance_scores):
                weight = 0.7 * trust + 0.3 * perf  # 70% trust, 30% performance
                combined_weights.append(weight)
                
            return self.aggregator.weighted_average(params_list, combined_weights)
            
        elif method == 'median':
            # Use median for each parameter
            result_params = copy.deepcopy(params_list[0])
            for key in result_params:
                stacked_params = torch.stack([p[key] for p in params_list])
                result_params[key] = torch.median(stacked_params, dim=0)[0]
            return result_params
            
        elif method == 'trimmed_mean':
            # Remove outliers and calculate mean
            trim_ratio = kwargs.get('trim_ratio', 0.2)
            result_params = copy.deepcopy(params_list[0])
            
            for key in result_params:
                stacked_params = torch.stack([p[key] for p in params_list])
                k = int(len(params_list) * trim_ratio)
                if k > 0:
                    sorted_values, _ = torch.sort(stacked_params, dim=0)
                    trimmed_values = sorted_values[k:-k] if len(sorted_values) > 2*k else sorted_values
                    result_params[key] = torch.mean(trimmed_values, dim=0)
                else:
                    result_params[key] = torch.mean(stacked_params, dim=0)
            return result_params
            
        elif method == 'fedprox':
            # FedProx with regularization
            global_params = kwargs.get('global_params', None)
            mu = kwargs.get('mu', 0.01)
            
            if global_params is None:
                return self.aggregator.weighted_average(params_list, [1.0/len(params_list)] * len(params_list))
                
            # Calculate weighted average with regularization
            weights = kwargs.get('weights', [1.0/len(params_list)] * len(params_list))
            result_params = self.aggregator.weighted_average(params_list, weights)
            
            # Add regularization term
            for key in result_params:
                result_params[key] = (1 - mu) * result_params[key] + mu * global_params[key]
                
            return result_params
        
        else:
            raise ValueError(f"Invalid aggregation method: {method}")
    
    def update_performance_metrics(self, 
                                method: str,
                                metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for an aggregation method.
        
        Args:
            method: Method name
            metrics: Dictionary containing metrics
        """
        if method not in self.performance_metrics:
            self.performance_metrics[method] = []
        self.performance_metrics[method].append(metrics)
        
        # Limit history size
        if len(self.performance_metrics[method]) > 100:
            self.performance_metrics[method].pop(0)
    
    def get_best_method(self) -> str:
        """
        Get the best performing aggregation method based on history.
        
        Returns:
            str: Name of the best method
        """
        if not self.performance_metrics:
            return self.default_method
            
        # Calculate average score for each method
        avg_scores = {}
        for method, metrics_list in self.performance_metrics.items():
            if metrics_list:
                # Prioritize recent metrics
                recent_metrics = metrics_list[-10:]  # 10 most recent
                scores = [m.get('score', 0) for m in recent_metrics]
                avg_scores[method] = np.mean(scores) if scores else 0
                
        # Select method with highest score
        return max(avg_scores.items(), key=lambda x: x[1])[0] if avg_scores else self.default_method 