"""
Unit tests for Model Aggregation in Federated Learning

This module contains test cases for different federated learning aggregation methods,
including weighted average, median, trimmed mean, Krum, adaptive federated averaging,
and FedProx. It also tests the ModelAggregationManager that selects appropriate methods.
"""

import unittest
import torch
import numpy as np
import sys
import os
from collections import OrderedDict

# Add root directory to sys.path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from qtrust.federated.model_aggregation import ModelAggregator, ModelAggregationManager

class TestModelAggregator(unittest.TestCase):
    def setUp(self):
        # Create sample PyTorch data for testing
        self.params1 = OrderedDict({
            'layer1.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            'layer1.bias': torch.tensor([0.1, 0.2]),
            'layer2.weight': torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
            'layer2.bias': torch.tensor([0.01, 0.02])
        })
        
        self.params2 = OrderedDict({
            'layer1.weight': torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
            'layer1.bias': torch.tensor([0.3, 0.4]),
            'layer2.weight': torch.tensor([[0.9, 1.0], [1.1, 1.2]]),
            'layer2.bias': torch.tensor([0.03, 0.04])
        })
        
        self.params3 = OrderedDict({
            'layer1.weight': torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
            'layer1.bias': torch.tensor([0.5, 0.6]),
            'layer2.weight': torch.tensor([[1.3, 1.4], [1.5, 1.6]]),
            'layer2.bias': torch.tensor([0.05, 0.06])
        })
        
        # Create abnormal parameters to test Byzantine fault tolerance
        self.byzantine_params = OrderedDict({
            'layer1.weight': torch.tensor([[30.0, 40.0], [50.0, 60.0]]),  # Abnormal values
            'layer1.bias': torch.tensor([5.0, 6.0]),  # Abnormal values
            'layer2.weight': torch.tensor([[13.0, 14.0], [15.0, 16.0]]),  # Abnormal values
            'layer2.bias': torch.tensor([0.5, 0.6])   # Abnormal values
        })
        
        self.params_list = [self.params1, self.params2, self.params3, self.byzantine_params]
        self.normal_params_list = [self.params1, self.params2, self.params3]
        
        # Weights for clients
        self.weights = [0.3, 0.3, 0.3, 0.1]
        self.normal_weights = [0.33, 0.33, 0.34]
        
        # Trust and performance information
        self.trust_scores = [0.9, 0.8, 0.7, 0.1]  # Last client is Byzantine
        self.performance_scores = [0.85, 0.9, 0.8, 0.3]  # Last client has low performance
        
        # Initialize aggregator
        self.aggregator = ModelAggregator()

    def test_weighted_average(self):
        """Test weighted average aggregation method."""
        aggregated = ModelAggregator.weighted_average(self.normal_params_list, weights=self.normal_weights)
        
        # Check results
        for key in aggregated:
            expected = sum(params[key] * w for params, w in zip(self.normal_params_list, self.normal_weights))
            self.assertTrue(torch.allclose(aggregated[key], expected, rtol=1e-5))

    def test_median(self):
        """Test median aggregation method."""
        aggregated = ModelAggregator.median(self.params_list)
        
        # Check results - Median calculation should be accurate
        for key in aggregated:
            # Create tensor to calculate median
            values = [params[key].cpu().numpy() for params in self.params_list]
            expected = np.median(values, axis=0)
            expected_tensor = torch.tensor(expected, dtype=aggregated[key].dtype)
            self.assertTrue(torch.allclose(aggregated[key], expected_tensor, rtol=1e-5))

    def test_trimmed_mean(self):
        """Test trimmed mean aggregation method."""
        aggregated = ModelAggregator.trimmed_mean(self.params_list, trim_ratio=0.25)  # Trim 25% highest and lowest values
        
        # Check results - Must eliminate Byzantine values
        for key in aggregated:
            # Create tensor to calculate trimmed mean (remove highest and lowest values)
            values = [params[key].cpu().numpy() for params in self.params_list]
            
            # Trim 25% highest and lowest values
            n = len(values)
            k = int(np.ceil(n * 0.25))
            
            if 2*k >= n:
                # If trimming too much, use regular mean
                expected = np.mean(values, axis=0)
            else:
                # Sort and trim
                sorted_values = np.sort(values, axis=0)
                expected = np.mean(sorted_values[k:n-k], axis=0)
            
            expected_tensor = torch.tensor(expected, dtype=aggregated[key].dtype)
            self.assertTrue(torch.allclose(aggregated[key], expected_tensor, rtol=1e-5))

    def test_krum(self):
        """Test Krum aggregation method."""
        aggregated = ModelAggregator.krum(self.params_list, num_byzantine=1)
        
        # Check results - Must select one of the normal parameters
        is_one_of_normal = False
        for normal_params in self.normal_params_list:
            if all(torch.allclose(aggregated[key], normal_params[key]) for key in aggregated):
                is_one_of_normal = True
                break
        
        self.assertTrue(is_one_of_normal, "Krum result must be one of the normal parameters")

    def test_adaptive_federated_averaging(self):
        """Test adaptive federated averaging aggregation method."""
        # Adaptive federated averaging uses trust_scores and performance_scores instead of weights
        aggregated = ModelAggregator.adaptive_federated_averaging(
            self.params_list, 
            trust_scores=self.trust_scores,
            performance_scores=self.performance_scores,
            adaptive_alpha=0.5
        )
        
        # Calculate combined scores as in implementation
        combined_scores = [0.5 * trust + 0.5 * perf 
                         for trust, perf in zip(self.trust_scores, self.performance_scores)]
        
        # Normalize to weights
        total_score = sum(combined_scores)
        expected_weights = [score / total_score for score in combined_scores]
        
        # Calculate weighted average manually
        expected_result = OrderedDict()
        for key in self.params1.keys():
            expected_result[key] = sum(params[key] * w for params, w in zip(self.params_list, expected_weights))
        
        # Compare results
        for key in aggregated:
            self.assertTrue(torch.allclose(aggregated[key], expected_result[key], rtol=1e-5))

    def test_fedprox(self):
        """Test FedProx aggregation method."""
        global_params = OrderedDict({
            'layer1.weight': torch.tensor([[1.5, 2.5], [3.5, 4.5]]),
            'layer1.bias': torch.tensor([0.25, 0.35]),
            'layer2.weight': torch.tensor([[0.75, 0.85], [0.95, 1.05]]),
            'layer2.bias': torch.tensor([0.025, 0.035])
        })
        
        # Mu coefficient
        mu = 0.01
        
        # Calculate average using weighted_average
        weighted_avg = ModelAggregator.weighted_average(self.normal_params_list, weights=self.normal_weights)
        
        # Apply FedProx
        aggregated = ModelAggregator.fedprox(
            self.normal_params_list, 
            global_params=global_params,
            weights=self.normal_weights,
            mu=mu
        )
        
        # Check results
        for key in aggregated:
            # FedProx: (1 - mu) * weighted_avg + mu * global_params
            expected = (1 - mu) * weighted_avg[key] + mu * global_params[key]
            self.assertTrue(torch.allclose(aggregated[key], expected, rtol=1e-5))


class TestModelAggregationManager(unittest.TestCase):
    def setUp(self):
        # Initialize manager with default method
        self.manager = ModelAggregationManager(default_method='weighted_average')
        
        # Create simple sample data for testing
        self.params1 = OrderedDict({'weight': torch.tensor([1.0, 2.0]), 'bias': torch.tensor([0.1])})
        self.params2 = OrderedDict({'weight': torch.tensor([3.0, 4.0]), 'bias': torch.tensor([0.2])})
        self.params3 = OrderedDict({'weight': torch.tensor([5.0, 6.0]), 'bias': torch.tensor([0.3])})
        
        self.params_list = [self.params1, self.params2, self.params3]
        self.weights = [0.2, 0.3, 0.5]

    def test_aggregate_with_default_method(self):
        """Test aggregation with default method."""
        # Method must be explicitly specified
        aggregated = self.manager.aggregate('weighted_average', params_list=self.params_list, weights=self.weights)
        
        # Check results - Must use default method (weighted_average)
        for key in aggregated:
            expected = sum(params[key] * w for params, w in zip(self.params_list, self.weights))
            self.assertTrue(torch.allclose(aggregated[key], expected, rtol=1e-5))

    def test_aggregate_with_specified_method(self):
        """Test aggregation with specified method."""
        aggregated = self.manager.aggregate('median', params_list=self.params_list)
        
        # Check results - Must use median method
        for key in aggregated:
            values = [params[key].cpu().numpy() for params in self.params_list]
            expected = np.median(values, axis=0)
            expected_tensor = torch.tensor(expected, dtype=aggregated[key].dtype)
            self.assertTrue(torch.allclose(aggregated[key], expected_tensor, rtol=1e-5))

    def test_recommend_method(self):
        """Test ability to recommend method based on conditions."""
        # If there are Byzantine clients, should use median or trimmed_mean
        method = self.manager.recommend_method(num_clients=10, suspected_byzantine=True)
        self.assertEqual(method, 'median')
        
        # If there are trust scores, should use adaptive_fedavg
        method = self.manager.recommend_method(num_clients=10, has_trust_scores=True)
        self.assertEqual(method, 'adaptive_fedavg')
        
        # If few clients and no Byzantine, should use weighted_average
        method = self.manager.recommend_method(num_clients=3)
        self.assertEqual(method, 'weighted_average')


if __name__ == '__main__':
    unittest.main() 