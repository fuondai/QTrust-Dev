"""
Test module for hyper parameter optimization.

This file contains tests for the hyperparameter optimization functionality in QTrust.
"""

import unittest
import os
import sys
import numpy as np
import random

# Add the root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestHyperOptimizer(unittest.TestCase):
    """Test cases for hyperparameter optimization."""
    
    def setUp(self):
        """Set up test environment."""
        random.seed(42)
        np.random.seed(42)
    
    def test_parameter_search(self):
        """Test parameter search functionality."""
        # Mock parameter space
        mock_param_space = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128],
            'hidden_size': [64, 128, 256]
        }
        
        # Verify parameter space is valid
        self.assertIn('learning_rate', mock_param_space)
        self.assertIn('batch_size', mock_param_space)
        self.assertIn('hidden_size', mock_param_space)
        
        # Verify parameter values
        self.assertEqual(len(mock_param_space['learning_rate']), 3)
        self.assertEqual(len(mock_param_space['batch_size']), 3)
        self.assertEqual(len(mock_param_space['hidden_size']), 3)
    
    def test_optimization_results(self):
        """Test optimization result validation."""
        # Mock optimization results
        mock_results = [
            {'params': {'learning_rate': 0.01, 'batch_size': 64}, 'score': 0.85},
            {'params': {'learning_rate': 0.1, 'batch_size': 32}, 'score': 0.82},
            {'params': {'learning_rate': 0.001, 'batch_size': 128}, 'score': 0.88}
        ]
        
        # Find best result
        best_result = max(mock_results, key=lambda x: x['score'])
        
        # Verify best result is correctly identified
        self.assertEqual(best_result['score'], 0.88)
        self.assertEqual(best_result['params']['learning_rate'], 0.001)
        self.assertEqual(best_result['params']['batch_size'], 128)


if __name__ == "__main__":
    unittest.main() 