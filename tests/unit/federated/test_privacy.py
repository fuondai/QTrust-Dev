"""
Tests for privacy mechanisms in QTrust Federated Learning

This module tests the privacy protection mechanisms implemented in the privacy.py module,
including differential privacy and secure aggregation for federated learning.
"""

import unittest
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from qtrust.federated.privacy import PrivacyManager, SecureAggregator

class TestPrivacyManager(unittest.TestCase):
    """Test cases for the PrivacyManager class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a PrivacyManager with default parameters
        self.privacy_manager = PrivacyManager(
            epsilon=1.0,
            delta=1e-5,
            clip_norm=1.0,
            noise_multiplier=1.1
        )
        
        # Create test gradients
        self.test_gradients = torch.randn(10, 5)
        self.test_model_params = {
            'layer1.weight': torch.randn(5, 10),
            'layer1.bias': torch.randn(5),
            'layer2.weight': torch.randn(1, 5),
            'layer2.bias': torch.randn(1)
        }
        self.test_num_samples = 32
    
    def test_initialization(self):
        """Test initialization of PrivacyManager."""
        # Check that parameters are correctly initialized
        self.assertEqual(self.privacy_manager.epsilon, 1.0)
        self.assertEqual(self.privacy_manager.delta, 1e-5)
        self.assertEqual(self.privacy_manager.clip_norm, 1.0)
        self.assertEqual(self.privacy_manager.noise_multiplier, 1.1)
        
        # Check initial state
        self.assertEqual(self.privacy_manager.consumed_budget, 0.0)
        self.assertEqual(len(self.privacy_manager.privacy_metrics), 0)
    
    def test_add_noise_to_gradients(self):
        """Test adding noise to gradients."""
        # Get original gradients
        original_gradients = self.test_gradients.clone()
        
        # Add noise to gradients
        noisy_gradients = self.privacy_manager.add_noise_to_gradients(
            self.test_gradients, self.test_num_samples
        )
        
        # Check that noise was added (gradients should be different)
        self.assertFalse(torch.allclose(original_gradients, noisy_gradients))
        
        # Check that privacy budget was updated
        self.assertGreater(self.privacy_manager.consumed_budget, 0.0)
        self.assertEqual(len(self.privacy_manager.privacy_metrics), 1)
    
    def test_add_noise_to_model(self):
        """Test adding noise to model parameters."""
        # Get original parameters
        original_params = {k: v.clone() for k, v in self.test_model_params.items()}
        
        # Add noise to model parameters
        noisy_params = self.privacy_manager.add_noise_to_model(
            self.test_model_params, self.test_num_samples
        )
        
        # Check that all parameter tensors have noise added
        for key in original_params:
            self.assertIn(key, noisy_params)
            self.assertFalse(torch.allclose(original_params[key], noisy_params[key]))
        
        # Check that privacy budget was updated for each parameter
        expected_updates = len(self.test_model_params)
        self.assertEqual(len(self.privacy_manager.privacy_metrics), expected_updates)
    
    def test_privacy_accounting(self):
        """Test privacy accounting mechanism."""
        # Initial state
        self.assertEqual(self.privacy_manager.consumed_budget, 0.0)
        
        # Add noise multiple times
        for _ in range(5):
            self.privacy_manager.add_noise_to_gradients(
                self.test_gradients, self.test_num_samples
            )
        
        # Check that budget is tracked properly
        self.assertGreater(self.privacy_manager.consumed_budget, 0.0)
        self.assertEqual(len(self.privacy_manager.privacy_metrics), 5)
        
        # Check that each update is properly recorded
        for i in range(5):
            metrics = self.privacy_manager.privacy_metrics[i]
            self.assertIn('epsilon', metrics)
            self.assertIn('total_budget', metrics)
            self.assertIn('remaining_budget', metrics)
            self.assertIn('num_samples', metrics)
            self.assertEqual(metrics['num_samples'], self.test_num_samples)
    
    def test_privacy_report(self):
        """Test generation of privacy report."""
        # Initial report (empty)
        initial_report = self.privacy_manager.get_privacy_report()
        self.assertEqual(initial_report['status'], 'No privacy metrics available')
        self.assertEqual(initial_report['consumed_budget'], 0.0)
        self.assertEqual(initial_report['remaining_budget'], self.privacy_manager.epsilon)
        
        # Add noise to update metrics
        self.privacy_manager.add_noise_to_gradients(
            self.test_gradients, self.test_num_samples
        )
        
        # Get report after update
        updated_report = self.privacy_manager.get_privacy_report()
        self.assertIn('status', updated_report)
        self.assertIn('consumed_budget', updated_report)
        self.assertIn('remaining_budget', updated_report)
        self.assertIn('noise_multiplier', updated_report)
        self.assertIn('clip_norm', updated_report)
        self.assertIn('last_update', updated_report)
        self.assertIn('total_updates', updated_report)
        
        # Check status based on budget consumption
        if self.privacy_manager.consumed_budget > self.privacy_manager.epsilon:
            self.assertEqual(updated_report['status'], 'Privacy budget exceeded')
        else:
            self.assertEqual(updated_report['status'], 'Active')
            
        # Check update counts
        self.assertEqual(updated_report['total_updates'], 1)


class TestSecureAggregator(unittest.TestCase):
    """Test cases for the SecureAggregator class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create PrivacyManager
        self.privacy_manager = PrivacyManager(
            epsilon=1.0,
            delta=1e-5,
            clip_norm=1.0,
            noise_multiplier=0.1  # Lower noise for testing
        )
        
        # Create SecureAggregator
        self.secure_aggregator = SecureAggregator(
            privacy_manager=self.privacy_manager,
            secure_communication=True,
            threshold=2  # Lower threshold for easier testing
        )
        
        # Create test client updates
        self.client_updates = {
            0: {
                'params': {
                    'layer1.weight': torch.ones(2, 3),
                    'layer1.bias': torch.ones(2)
                },
                'num_samples': 100  # Increase sample size to reduce noise impact
            },
            1: {
                'params': {
                    'layer1.weight': torch.ones(2, 3) * 2,
                    'layer1.bias': torch.ones(2) * 2
                },
                'num_samples': 100
            },
            2: {
                'params': {
                    'layer1.weight': torch.ones(2, 3) * 3,
                    'layer1.bias': torch.ones(2) * 3
                },
                'num_samples': 100
            }
        }
    
    def test_initialization(self):
        """Test initialization of SecureAggregator."""
        # Check that parameters are correctly initialized
        self.assertEqual(self.secure_aggregator.privacy_manager, self.privacy_manager)
        self.assertTrue(self.secure_aggregator.secure_communication)
        self.assertEqual(self.secure_aggregator.threshold, 2)
        
        # Check initial state
        self.assertEqual(len(self.secure_aggregator.key_shares), 0)
        self.assertEqual(len(self.secure_aggregator.masked_models), 0)
    
    def test_aggregate_secure(self):
        """Test secure aggregation of client updates."""
        # Perform secure aggregation
        aggregated_params = self.secure_aggregator.aggregate_secure(
            client_updates=self.client_updates
        )
        
        # Check that aggregated parameters contain expected keys
        self.assertIn('layer1.weight', aggregated_params)
        self.assertIn('layer1.bias', aggregated_params)
        
        # Check shape of aggregated parameters
        self.assertEqual(aggregated_params['layer1.weight'].shape, (2, 3))
        self.assertEqual(aggregated_params['layer1.bias'].shape, (2,))
        
        # Test that the result is not exactly any of the individual client updates
        # This ensures aggregation + noise happened
        for client_id, update in self.client_updates.items():
            client_weight = update['params']['layer1.weight']
            self.assertFalse(torch.allclose(aggregated_params['layer1.weight'], client_weight, atol=1e-5))
    
    def test_aggregate_secure_with_weights(self):
        """Test secure aggregation with explicit weights."""
        # Weights for clients (should sum to 1)
        weights = [0.2, 0.3, 0.5]
        
        # Perform secure aggregation with weights
        aggregated_params = self.secure_aggregator.aggregate_secure(
            client_updates=self.client_updates,
            weights=weights
        )
        
        # Check shape
        self.assertEqual(aggregated_params['layer1.weight'].shape, (2, 3))
        self.assertEqual(aggregated_params['layer1.bias'].shape, (2,))
        
        # Simplified test: just verify the aggregation happens without errors
        # and that noise is added (so result is different from a simple weighted average)
        # Calculate what the result would be without noise
        expected_weight = sum(
            weights[i] * update['params']['layer1.weight'] 
            for i, (_, update) in enumerate(self.client_updates.items())
        )
        
        # Verify that noise was added (result should differ from simple weighted average)
        self.assertFalse(torch.allclose(aggregated_params['layer1.weight'], expected_weight, atol=1e-5))
    
    def test_threshold_enforcement(self):
        """Test that threshold is properly enforced."""
        # Create updates with fewer clients than threshold
        insufficient_updates = {
            0: self.client_updates[0]
        }
        
        # Attempt to aggregate with insufficient clients should raise error
        with self.assertRaises(ValueError):
            self.secure_aggregator.aggregate_secure(insufficient_updates)


if __name__ == "__main__":
    unittest.main() 