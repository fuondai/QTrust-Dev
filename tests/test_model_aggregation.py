"""
Tests for model aggregation methods in Federated Learning

This module tests the various model aggregation methods implemented in the
model_aggregation.py module, including Byzantine-resistant aggregation methods
and adaptive federated averaging techniques.
"""

import unittest
import torch
import numpy as np
import sys
import os
from collections import OrderedDict
from pathlib import Path

# Add the project root to the path for imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from qtrust.federated.model_aggregation import ModelAggregator, ModelAggregationManager

class TestModelAggregator(unittest.TestCase):
    """Test cases for the ModelAggregator class."""
    
    def setUp(self):
        """Set up test data for model aggregation tests."""
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
        
        # Create an abnormal parameter set to test Byzantine resistance
        self.byzantine_params = OrderedDict({
            'layer1.weight': torch.tensor([[30.0, 40.0], [50.0, 60.0]]),  # Abnormally large values
            'layer1.bias': torch.tensor([5.0, 6.0]),  # Abnormally large values
            'layer2.weight': torch.tensor([[13.0, 14.0], [15.0, 16.0]]),  # Abnormally large values
            'layer2.bias': torch.tensor([0.5, 0.6])   # Abnormally large values
        })
        
        self.params_list = [self.params1, self.params2, self.params3, self.byzantine_params]
        self.normal_params_list = [self.params1, self.params2, self.params3]
        
        # Weights for clients
        self.weights = [0.3, 0.3, 0.3, 0.1]
        self.normal_weights = [0.33, 0.33, 0.34]
        
        # Trust and performance information
        self.trust_scores = [0.9, 0.8, 0.7, 0.1]  # Last client is Byzantine
        self.performance_scores = [0.85, 0.9, 0.8, 0.3]  # Last client has low performance
        
        # Initialize the aggregator
        self.aggregator = ModelAggregator()

    def test_weighted_average(self):
        """Test weighted average aggregation method."""
        aggregated = self.aggregator.weighted_average(self.normal_params_list, weights=self.normal_weights)
        
        # Check results
        for key in aggregated:
            expected = sum(params[key] * w for params, w in zip(self.normal_params_list, self.normal_weights))
            self.assertTrue(torch.allclose(aggregated[key], expected, rtol=1e-5))

    def test_median(self):
        """Test median aggregation method."""
        aggregated = self.aggregator.median(self.params_list)
        
        # Check results - Median calculation should be accurate
        for key in aggregated:
            # Create tensors for median calculation
            values = [params[key].cpu().numpy() for params in self.params_list]
            expected = np.median(values, axis=0)
            expected_tensor = torch.tensor(expected, dtype=aggregated[key].dtype)
            self.assertTrue(torch.allclose(aggregated[key], expected_tensor, rtol=1e-5))

    def test_trimmed_mean(self):
        """Test trimmed mean aggregation method."""
        aggregated = self.aggregator.trimmed_mean(self.params_list, trim_ratio=0.25)  # Trim 25% highest and lowest values
        
        # Check results - Should exclude Byzantine values
        for key in aggregated:
            # Create tensors for trimmed mean calculation
            values = [params[key].cpu().numpy() for params in self.params_list]
            
            # Trim 25% of highest and lowest values
            n = len(values)
            k = int(np.ceil(n * 0.25))
            
            if 2*k >= n:
                # If trimming too many, use regular mean
                expected = np.mean(values, axis=0)
            else:
                # Sort and trim
                sorted_values = np.sort(values, axis=0)
                expected = np.mean(sorted_values[k:n-k], axis=0)
            
            expected_tensor = torch.tensor(expected, dtype=aggregated[key].dtype)
            self.assertTrue(torch.allclose(aggregated[key], expected_tensor, rtol=1e-5))

    def test_krum(self):
        """Test Krum aggregation method."""
        aggregated = self.aggregator.krum(self.params_list, num_byzantine=1)
        
        # Check results - Should select one of the normal parameters
        is_one_of_normal = False
        for normal_params in self.normal_params_list:
            if all(torch.allclose(aggregated[key], normal_params[key]) for key in aggregated):
                is_one_of_normal = True
                break
        
        self.assertTrue(is_one_of_normal, "Krum result should be one of the normal parameters")

    def test_adaptive_federated_averaging(self):
        """Test adaptive federated averaging method."""
        # Adaptive federated averaging uses trust_scores and performance_scores instead of weights
        aggregated = self.aggregator.adaptive_federated_averaging(
            self.params_list, 
            trust_scores=self.trust_scores,
            performance_scores=self.performance_scores,
            adaptive_alpha=0.5
        )
        
        # Calculate combined scores as in the implementation
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
        weighted_avg = self.aggregator.weighted_average(self.normal_params_list, weights=self.normal_weights)
        
        # Apply FedProx
        aggregated = self.aggregator.fedprox(
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
    """Test cases for the ModelAggregationManager class."""
    
    def setUp(self):
        """Set up test environment for aggregation manager tests."""
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
        
        # Check results - Should use default method (weighted_average)
        for key in aggregated:
            expected = sum(params[key] * w for params, w in zip(self.params_list, self.weights))
            self.assertTrue(torch.allclose(aggregated[key], expected, rtol=1e-5))

    def test_aggregate_with_specified_method(self):
        """Test aggregation with a specified method."""
        aggregated = self.manager.aggregate('median', params_list=self.params_list)
        
        # Check results - Should use median method
        for key in aggregated:
            values = [params[key].cpu().numpy() for params in self.params_list]
            expected = np.median(values, axis=0)
            expected_tensor = torch.tensor(expected, dtype=aggregated[key].dtype)
            self.assertTrue(torch.allclose(aggregated[key], expected_tensor, rtol=1e-5))

    def test_recommend_method(self):
        """Test the ability to recommend the best method."""
        # With few clients and no Byzantine suspicion -> weighted_average
        method = self.manager.recommend_method(num_clients=3, has_trust_scores=False, suspected_byzantine=False)
        self.assertEqual(method, 'weighted_average')
        
        # With suspected Byzantine issue -> median or trimmed_mean
        method = self.manager.recommend_method(num_clients=10, has_trust_scores=False, suspected_byzantine=True)
        self.assertEqual(method, 'median')
        
        # With trust scores available -> adaptive_fedavg
        method = self.manager.recommend_method(num_clients=5, has_trust_scores=True, suspected_byzantine=False)
        self.assertEqual(method, 'adaptive_fedavg')


if __name__ == '__main__':
    unittest.main() 