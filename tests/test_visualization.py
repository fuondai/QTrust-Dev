"""
Test file for the QTrust visualization module.
"""

import os
import sys
import unittest
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from typing import Dict, List

# Add the project root directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrust.utils.visualization import (
    plot_blockchain_network,
    plot_transaction_flow,
    plot_shard_graph,
    plot_consensus_comparison,
    plot_learning_curve
)

class TestVisualization(unittest.TestCase):
    """Tests for visualization utilities."""
    
    def setUp(self):
        """Set up for each test."""
        # Create a temporary directory for output files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test network
        self.network = nx.watts_strogatz_graph(20, 4, 0.3, seed=42)
        
        # Create test shards
        self.shards = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]
        ]
        
        # Create trust scores
        self.trust_scores = {
            node: 0.5 + 0.5 * np.random.random() for node in self.network.nodes()
        }
        
        # Create transactions
        self.transactions = [
            {'id': 1, 'source': 0, 'destination': 8, 'value': 10},
            {'id': 2, 'source': 5, 'destination': 15, 'value': 20},
            {'id': 3, 'source': 10, 'destination': 3, 'value': 15}
        ]
        
        # Create paths
        self.paths = {
            1: [0, 2, 5, 8],
            2: [5, 7, 12, 15],
            3: [10, 11, 6, 3]
        }
        
        # Create shard metrics
        self.shard_metrics = {
            0: {'throughput': 120, 'latency': 50, 'energy_consumption': 80},
            1: {'throughput': 100, 'latency': 60, 'energy_consumption': 90},
            2: {'throughput': 150, 'latency': 40, 'energy_consumption': 70},
            3: {'throughput': 90, 'latency': 70, 'energy_consumption': 100}
        }
        
        # Create consensus results
        self.consensus_results = {
            'PoW': {
                'latency': [50, 55, 60, 45, 52],
                'energy': [100, 95, 105, 98, 102],
                'success_rate': [0.98, 0.97, 0.99, 0.98, 0.97]
            },
            'PoS': {
                'latency': [25, 30, 28, 26, 29],
                'energy': [40, 42, 38, 41, 39],
                'success_rate': [0.97, 0.98, 0.96, 0.97, 0.98]
            },
            'PBFT': {
                'latency': [15, 18, 14, 16, 17],
                'energy': [60, 58, 62, 59, 61],
                'success_rate': [0.99, 0.99, 0.98, 0.99, 0.99]
            }
        }
        
        # Create rewards for learning curve
        self.rewards = [np.random.normal(i/10, 1) for i in range(100)]
        
        # Disable matplotlib interactive mode to avoid showing plots
        plt.ioff()
    
    def test_plot_blockchain_network(self):
        """Test blockchain network plotting."""
        # Test without saving
        plot_blockchain_network(
            network=self.network,
            shards=self.shards,
            trust_scores=self.trust_scores,
            title="Test Blockchain Network",
            figsize=(8, 6)
        )
        plt.close()
        
        # Test with saving
        save_path = os.path.join(self.test_dir, "blockchain_network.png")
        plot_blockchain_network(
            network=self.network,
            shards=self.shards,
            trust_scores=self.trust_scores,
            title="Test Blockchain Network",
            figsize=(8, 6),
            save_path=save_path
        )
        
        # Check if the file was created
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_transaction_flow(self):
        """Test transaction flow plotting."""
        # Test without saving
        plot_transaction_flow(
            network=self.network,
            transactions=self.transactions,
            paths=self.paths,
            title="Test Transaction Flow",
            figsize=(8, 6)
        )
        plt.close()
        
        # Test with saving
        save_path = os.path.join(self.test_dir, "transaction_flow.png")
        plot_transaction_flow(
            network=self.network,
            transactions=self.transactions,
            paths=self.paths,
            title="Test Transaction Flow",
            figsize=(8, 6),
            save_path=save_path
        )
        
        # Check if the file was created
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_shard_graph(self):
        """Test shard graph plotting."""
        # Test without saving for different metrics
        for metric in ['throughput', 'latency', 'energy_consumption']:
            plot_shard_graph(
                network=self.network,
                shards=self.shards,
                shard_metrics=self.shard_metrics,
                metric_name=metric,
                title=f"Test Shard Performance - {metric}",
                figsize=(8, 6)
            )
            plt.close()
        
        # Test with saving
        save_path = os.path.join(self.test_dir, "shard_graph.png")
        plot_shard_graph(
            network=self.network,
            shards=self.shards,
            shard_metrics=self.shard_metrics,
            metric_name='throughput',
            title="Test Shard Performance",
            figsize=(8, 6),
            save_path=save_path
        )
        
        # Check if the file was created
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_consensus_comparison(self):
        """Test consensus comparison plotting."""
        # Test without saving
        plot_consensus_comparison(
            consensus_results=self.consensus_results,
            figsize=(12, 4)
        )
        plt.close()
        
        # Test with saving
        save_path = os.path.join(self.test_dir, "consensus_comparison.png")
        plot_consensus_comparison(
            consensus_results=self.consensus_results,
            figsize=(12, 4),
            save_path=save_path
        )
        
        # Check if the file was created
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_learning_curve(self):
        """Test learning curve plotting."""
        # Test without saving
        plot_learning_curve(
            rewards=self.rewards,
            avg_window=10,
            title="Test Learning Curve",
            figsize=(8, 4)
        )
        plt.close()
        
        # Test with saving
        save_path = os.path.join(self.test_dir, "learning_curve.png")
        plot_learning_curve(
            rewards=self.rewards,
            avg_window=10,
            title="Test Learning Curve",
            figsize=(8, 4),
            save_path=save_path
        )
        
        # Check if the file was created
        self.assertTrue(os.path.exists(save_path))
    
    def test_edge_cases(self):
        """Test edge cases for visualization functions."""
        # Empty network
        empty_network = nx.Graph()
        plot_blockchain_network(
            network=empty_network,
            shards=[[]],
            title="Empty Network",
            figsize=(6, 4)
        )
        plt.close()
        
        # Empty shards
        plot_blockchain_network(
            network=self.network,
            shards=[[]],
            title="Empty Shards",
            figsize=(6, 4)
        )
        plt.close()
        
        # Empty transactions
        plot_transaction_flow(
            network=self.network,
            transactions=[],
            paths={},
            title="Empty Transactions",
            figsize=(6, 4)
        )
        plt.close()
        
        # Small window for learning curve
        plot_learning_curve(
            rewards=self.rewards[:5],
            avg_window=10,  # Window bigger than rewards length
            title="Small Rewards",
            figsize=(6, 4)
        )
        plt.close()

if __name__ == "__main__":
    unittest.main() 