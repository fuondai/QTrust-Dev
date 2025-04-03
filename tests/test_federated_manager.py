"""
Tests for Federated Learning Manager

This module contains tests for the FederatedLearningManager class which coordinates
the federated learning process including client selection, model distribution,
aggregation, and evaluation.
"""

import unittest
import os
import numpy as np
import torch
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from collections import OrderedDict

import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from qtrust.federated.manager import FederatedLearningManager
from qtrust.federated.protocol import FederatedProtocol
from qtrust.federated.model_aggregation import ModelAggregator


class TestFederatedLearningManager(unittest.TestCase):
    """Test cases for the FederatedLearningManager class."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for model saving
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data
        self.num_clients = 5
        self.client_fraction = 0.6
        self.mock_protocol = MagicMock(spec=FederatedProtocol)
        
        # Initialize the manager with mock protocol
        self.manager = FederatedLearningManager(
            num_clients=self.num_clients,
            client_fraction=self.client_fraction,
            aggregation_method="fedavg",
            protocol=self.mock_protocol,
            model_path=self.test_dir
        )
        
        # Create sample model parameters using OrderedDict with torch tensors for compatibility with model_aggregation.py
        self.global_model = OrderedDict({
            'layer1': torch.rand(10, 20),
            'layer2': torch.rand(20, 5),
            'bias': torch.rand(5)
        })
        
        # Convert to numpy for saving/loading tests
        self.global_model_numpy = {
            'layer1': self.global_model['layer1'].numpy(),
            'layer2': self.global_model['layer2'].numpy(),
            'bias': self.global_model['bias'].numpy()
        }
        
        # Create sample client models with torch tensors
        self.client_models = [
            OrderedDict({
                'layer1': torch.rand(10, 20),
                'layer2': torch.rand(20, 5),
                'bias': torch.rand(5)
            })
            for _ in range(3)
        ]
        
        # Sample client weights
        self.client_weights = [0.4, 0.3, 0.3]

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test manager initialization with default and custom parameters."""
        # Test with default parameters
        default_manager = FederatedLearningManager()
        self.assertEqual(default_manager.num_clients, 10)
        self.assertEqual(default_manager.client_fraction, 0.8)
        self.assertEqual(default_manager.aggregation_method, "fedavg")
        
        # Test with custom parameters
        custom_manager = FederatedLearningManager(
            num_clients=15,
            client_fraction=0.7,
            aggregation_method="median"
        )
        self.assertEqual(custom_manager.num_clients, 15)
        self.assertEqual(custom_manager.client_fraction, 0.7)
        self.assertEqual(custom_manager.aggregation_method, "median")
        
        # Check if directory was created
        self.assertTrue(os.path.exists(self.test_dir))

    def test_select_clients(self):
        """Test client selection based on trust scores."""
        # Test with trust weighting enabled
        self.manager.enable_trust_weighting = True
        
        # Set specific trust scores for deterministic testing
        for i in range(self.num_clients):
            self.manager.client_metrics[i]['trust'] = 0.5 + i * 0.1
        
        # Test multiple selections to ensure they're influenced by trust
        selection_counts = {i: 0 for i in range(self.num_clients)}
        for _ in range(100):
            selected = self.manager.select_clients()
            for client_id in selected:
                selection_counts[client_id] += 1
                
        # Higher trust clients should be selected more often
        for i in range(1, self.num_clients):
            # This might occasionally fail due to randomness, but should pass most times
            self.assertGreaterEqual(selection_counts[i], selection_counts[i-1] * 0.5)
        
        # Test with trust weighting disabled
        self.manager.enable_trust_weighting = False
        selected = self.manager.select_clients()
        
        # Check correct number of clients selected
        expected_num_selected = int(self.num_clients * self.client_fraction)
        self.assertEqual(len(selected), expected_num_selected)

    def test_distribute_model(self):
        """Test model distribution to selected clients."""
        # Setup mock behavior
        self.mock_protocol.send_model.return_value = True
        
        # Test distribution
        selected_clients = [0, 2, 4]
        result = self.manager.distribute_model(self.global_model_numpy, selected_clients)
        
        # Check result
        self.assertTrue(result)
        
        # Verify calls to protocol
        self.assertEqual(self.mock_protocol.send_model.call_count, len(selected_clients))
        for i, client_id in enumerate(selected_clients):
            args, _ = self.mock_protocol.send_model.call_args_list[i]
            self.assertEqual(args[1], client_id)
        
        # Test failure scenario
        self.mock_protocol.send_model.reset_mock()
        self.mock_protocol.send_model.return_value = False
        
        result = self.manager.distribute_model(self.global_model_numpy, selected_clients)
        self.assertFalse(result)
        
        # Should have attempted to send to first client only
        self.assertEqual(self.mock_protocol.send_model.call_count, 1)

    def test_collect_models(self):
        """Test collection of models from clients."""
        # Setup mock behavior to return models only for specific clients
        def receive_model_side_effect(client_id):
            if client_id in [1, 3]:
                return {k: v.numpy() for k, v in self.client_models[client_id % len(self.client_models)].items()}
            return None
            
        self.mock_protocol.receive_model.side_effect = receive_model_side_effect
        
        # Test collection
        selected_clients = [0, 1, 2, 3, 4]
        client_models, client_weights = self.manager.collect_models(selected_clients)
        
        # Check results - should have received models only from clients 1 and 3
        self.assertEqual(len(client_models), 2)
        self.assertEqual(len(client_weights), 2)
        
        # Verify calls to protocol
        self.assertEqual(self.mock_protocol.receive_model.call_count, len(selected_clients))

        # Test with trust weighting disabled
        self.manager.enable_trust_weighting = False
        
        self.mock_protocol.receive_model.reset_mock()
        self.mock_protocol.receive_model.side_effect = receive_model_side_effect
        
        _, client_weights = self.manager.collect_models(selected_clients)
        
        # Weights should be 1.0 when trust weighting is disabled
        for weight in client_weights:
            self.assertEqual(weight, 1.0)

    def test_aggregate_models_with_real_aggregator(self):
        """Test model aggregation with real ModelAggregator instance."""
        # Create a test manager with a real aggregator
        test_manager = FederatedLearningManager(
            aggregation_method="fedavg",
            model_path=self.test_dir
        )
        
        # Temporarily modify aggregate_models to skip model format conversion
        original_aggregate = test_manager.aggregate_models
        
        def aggregate_models_wrapper(client_models, client_weights):
            # For testing purposes, convert numpy arrays to torch tensors
            return original_aggregate(client_models, client_weights)
        
        test_manager.aggregate_models = aggregate_models_wrapper
        
        # Convert numpy model to tensors for aggregation
        torch_models = self.client_models
        
        # Test aggregation with different methods
        result_fedavg = test_manager.aggregate_models(torch_models, self.client_weights)
        
        # Check result has proper format
        self.assertIsInstance(result_fedavg, dict)
        for key in result_fedavg:
            self.assertIsInstance(result_fedavg[key], torch.Tensor)
        
        # Test median
        test_manager.aggregation_method = "median"
        result_median = test_manager.aggregate_models(torch_models, self.client_weights)
        
        self.assertIsInstance(result_median, dict)
        for key in result_median:
            self.assertIsInstance(result_median[key], torch.Tensor)
        
        # Test trimmed_mean
        test_manager.aggregation_method = "trimmed_mean"
        result_trimmed = test_manager.aggregate_models(torch_models, self.client_weights)
        
        self.assertIsInstance(result_trimmed, dict)
        for key in result_trimmed:
            self.assertIsInstance(result_trimmed[key], torch.Tensor)

    @patch('time.sleep')
    @patch('builtins.print')  # Patch the print function to avoid I/O errors
    def test_run_round(self, mock_print, mock_sleep):
        """Test running a complete federated learning round."""
        # Setup mocks for all component methods
        self.manager.select_clients = MagicMock(return_value=[0, 1, 2])
        self.manager.distribute_model = MagicMock(return_value=True)
        self.manager.collect_models = MagicMock(return_value=(self.client_models, self.client_weights))
        
        # Create expected result
        expected_new_model = {'layer1': torch.ones(10, 20)}
        self.manager.aggregate_models = MagicMock(return_value=expected_new_model)
        self.manager.save_model = MagicMock()
        
        # Test running a round
        result = self.manager.run_round(self.global_model)
        
        # Verify result
        self.assertEqual(result, expected_new_model)
        
        # Verify all methods were called correctly
        self.manager.select_clients.assert_called_once()
        self.manager.distribute_model.assert_called_once_with(self.global_model, [0, 1, 2])
        self.manager.collect_models.assert_called_once_with([0, 1, 2])
        self.manager.aggregate_models.assert_called_once_with(self.client_models, self.client_weights)
        self.manager.save_model.assert_called_once_with(expected_new_model)
        
        # Check if round counter was incremented
        self.assertEqual(self.manager.round, 1)
        
        # Test failure scenario - distribute_model fails
        self.manager.round = 0  # Reset round counter
        self.manager.distribute_model.return_value = False
        
        result = self.manager.run_round(self.global_model)
        
        # Should return the original model unchanged
        self.assertEqual(result, self.global_model)
        
        # Test failure scenario - no client models received
        self.manager.round = 0  # Reset round counter
        self.manager.distribute_model.return_value = True
        self.manager.collect_models.return_value = ([], [])
        
        result = self.manager.run_round(self.global_model)
        
        # Should return the original model unchanged
        self.assertEqual(result, self.global_model)
        
        # Test failure scenario - aggregation fails
        self.manager.round = 0  # Reset round counter
        self.manager.collect_models.return_value = (self.client_models, self.client_weights)
        self.manager.aggregate_models.return_value = {}
        
        result = self.manager.run_round(self.global_model)
        
        # Should return the original model unchanged
        self.assertEqual(result, self.global_model)

    def test_update_client_metrics(self):
        """Test updating metrics for a specific client."""
        # Initial metrics should be 1.0
        self.assertEqual(self.manager.client_metrics[0]['trust'], 1.0)
        self.assertEqual(self.manager.client_metrics[0]['performance'], 1.0)
        
        # Update metrics
        self.manager.update_client_metrics(0, {'trust': 0.8, 'performance': 0.9})
        
        # Check updated values
        self.assertEqual(self.manager.client_metrics[0]['trust'], 0.8)
        self.assertEqual(self.manager.client_metrics[0]['performance'], 0.9)
        
        # Test partial update
        self.manager.update_client_metrics(0, {'trust': 0.7})
        
        # Only trust should be updated
        self.assertEqual(self.manager.client_metrics[0]['trust'], 0.7)
        self.assertEqual(self.manager.client_metrics[0]['performance'], 0.9)

    def test_save_model(self):
        """Test saving global model to disk."""
        # Save model with numpy arrays - the manager should handle this correctly
        self.manager.round = 5
        self.manager.save_model(self.global_model_numpy)
        
        # Check if file was created
        expected_path = os.path.join(self.test_dir, "global_model_round_5.npz")
        self.assertTrue(os.path.exists(expected_path))
        
        # Load saved model to verify contents
        loaded_model = np.load(expected_path)
        
        # Check if all keys are present
        for key in self.global_model_numpy.keys():
            self.assertIn(key, loaded_model.files)
            np.testing.assert_array_equal(loaded_model[key], self.global_model_numpy[key])
        
        # Test with suffix
        self.manager.save_model(self.global_model_numpy, suffix="test")
        
        # Check if file with suffix was created
        expected_path_with_suffix = os.path.join(self.test_dir, "global_model_round_5_test.npz")
        self.assertTrue(os.path.exists(expected_path_with_suffix))


if __name__ == '__main__':
    unittest.main() 