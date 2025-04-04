"""
Test module for data generation utilities.

This file contains tests for the data generation functionality in the QTrust blockchain system.
"""

import unittest
import os
import sys
import numpy as np
import random

# Add the root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDataGeneration(unittest.TestCase):
    """Test cases for data generation utilities."""
    
    def setUp(self):
        """Set up test environment."""
        # Set a fixed seed for reproducibility
        random.seed(42)
        np.random.seed(42)
    
    def test_random_transaction_generation(self):
        """Test random transaction generation."""
        # This is a placeholder test function
        # In a real scenario, this would test transaction generation logic
        num_transactions = 10
        mock_transactions = [{'id': i, 'value': random.random() * 100} for i in range(num_transactions)]
        
        self.assertEqual(len(mock_transactions), num_transactions)
        for tx in mock_transactions:
            self.assertIn('id', tx)
            self.assertIn('value', tx)
            self.assertGreaterEqual(tx['value'], 0)
            self.assertLessEqual(tx['value'], 100)
    
    def test_network_topology_generation(self):
        """Test network topology generation."""
        # This is a placeholder test function
        # In a real scenario, this would test network topology generation
        mock_nodes = 5
        mock_edges = [(i, (i+1) % mock_nodes) for i in range(mock_nodes)]
        
        self.assertEqual(len(mock_edges), mock_nodes)
        for edge in mock_edges:
            self.assertIsInstance(edge, tuple)
            self.assertEqual(len(edge), 2)
            self.assertLess(edge[0], mock_nodes)
            self.assertLess(edge[1], mock_nodes)


# Allow the file to be run as a standalone test
if __name__ == "__main__":
    unittest.main() 