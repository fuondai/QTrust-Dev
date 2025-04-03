"""
Test file for the QTrust metrics module.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from typing import Dict, List

# Add the project root directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrust.utils.metrics import (
    calculate_transaction_throughput,
    calculate_latency_metrics,
    calculate_energy_efficiency,
    calculate_security_metrics,
    calculate_cross_shard_metrics,
    generate_performance_report,
    calculate_throughput,
    calculate_cross_shard_transaction_ratio,
    SecurityMetrics
)

class TestMetrics(unittest.TestCase):
    """Tests for metrics utilities."""
    
    def setUp(self):
        """Set up for each test."""
        # Create a temporary directory for output files
        self.test_dir = tempfile.mkdtemp()
        
        # Disable matplotlib interactive mode to avoid showing plots
        plt.ioff()
    
    def test_calculate_transaction_throughput(self):
        """Test transaction throughput calculation."""
        # Test with normal values
        self.assertEqual(calculate_transaction_throughput(100, 10), 10.0)
        # Test with zero time
        self.assertEqual(calculate_transaction_throughput(100, 0), 0.0)
    
    def test_calculate_latency_metrics(self):
        """Test latency metrics calculation."""
        # Test with normal values
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        metrics = calculate_latency_metrics(latencies)
        
        self.assertEqual(metrics['avg_latency'], 55.0)
        self.assertEqual(metrics['median_latency'], 55.0)
        self.assertEqual(metrics['min_latency'], 10.0)
        self.assertEqual(metrics['max_latency'], 100.0)
        self.assertAlmostEqual(metrics['p95_latency'], 95.5, places=2)
        self.assertAlmostEqual(metrics['p99_latency'], 99.1, places=1)
        
        # Test with empty list
        empty_metrics = calculate_latency_metrics([])
        self.assertEqual(empty_metrics['avg_latency'], 0.0)
    
    def test_calculate_energy_efficiency(self):
        """Test energy efficiency calculation."""
        # Test with normal values
        self.assertEqual(calculate_energy_efficiency(1000, 100), 10.0)
        # Test with zero transactions
        self.assertEqual(calculate_energy_efficiency(1000, 0), float('inf'))
    
    def test_calculate_security_metrics(self):
        """Test security metrics calculation."""
        # Create test data
        trust_scores = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.3, 4: 0.2}
        malicious_nodes = [3, 4]
        
        # Calculate metrics
        metrics = calculate_security_metrics(trust_scores, malicious_nodes)
        
        # Check results
        self.assertAlmostEqual(metrics['avg_trust'], 0.58, delta=0.01)
        self.assertAlmostEqual(metrics['malicious_ratio'], 0.4, delta=0.01)
        self.assertGreater(metrics['trust_variance'], 0.0)
        
        # Test with empty trust scores
        empty_metrics = calculate_security_metrics({}, [])
        self.assertEqual(empty_metrics['avg_trust'], 0.0)
    
    def test_calculate_cross_shard_metrics(self):
        """Test cross-shard metrics calculation."""
        # Create test data
        cross_shard_txs = 30
        total_txs = 100
        cross_shard_latencies = [20, 25, 30, 35, 40]
        intra_shard_latencies = [10, 12, 14, 16, 18]
        
        # Calculate metrics
        metrics = calculate_cross_shard_metrics(
            cross_shard_txs, total_txs, cross_shard_latencies, intra_shard_latencies)
        
        # Check results
        self.assertEqual(metrics['cross_shard_ratio'], 0.3)
        self.assertEqual(metrics['cross_shard_avg_latency'], 30.0)
        self.assertEqual(metrics['intra_shard_avg_latency'], 14.0)
        self.assertAlmostEqual(metrics['latency_overhead'], 2.143, delta=0.001)
        
        # Test with empty latencies
        empty_metrics = calculate_cross_shard_metrics(10, 100, [], [])
        self.assertEqual(empty_metrics['cross_shard_avg_latency'], 0.0)
    
    def test_generate_performance_report(self):
        """Test performance report generation."""
        # Create test metrics
        metrics = {
            'throughput': 0.1,  # tx/ms
            'latency': {
                'avg_latency': 50.0,
                'median_latency': 45.0,
                'p95_latency': 95.0
            },
            'energy_per_tx': 10.0,
            'security': {
                'avg_trust': 0.8,
                'malicious_ratio': 0.1
            },
            'cross_shard': {
                'cross_shard_ratio': 0.3,
                'latency_overhead': 2.0
            }
        }
        
        # Generate report
        report = generate_performance_report(metrics)
        
        # Check report structure
        self.assertIsInstance(report, pd.DataFrame)
        self.assertEqual(len(report), 9)  # 9 metrics
        
        # Check some values
        self.assertEqual(report.loc[0, 'Value'], 100.0)  # throughput converted to tx/s
        self.assertEqual(report.loc[1, 'Value'], 50.0)  # avg latency
    
    def test_calculate_throughput(self):
        """Test throughput calculation."""
        # Test with normal values
        self.assertEqual(calculate_throughput(100, 10), 10.0)
        # Test with zero time
        self.assertEqual(calculate_throughput(100, 0), 0.0)
    
    def test_calculate_cross_shard_transaction_ratio(self):
        """Test cross-shard transaction ratio calculation."""
        # Test with normal values
        self.assertEqual(calculate_cross_shard_transaction_ratio(30, 100), 0.3)
        # Test with zero transactions
        self.assertEqual(calculate_cross_shard_transaction_ratio(0, 0), 0.0)
    
    def test_security_metrics_class(self):
        """Test SecurityMetrics class."""
        # Create SecurityMetrics instance
        sec_metrics = SecurityMetrics(window_size=5)
        
        # Test initial state
        self.assertEqual(sec_metrics.analysis_window, 5)
        self.assertIsNone(sec_metrics.current_attack)
        self.assertEqual(sec_metrics.attack_confidence, 0.0)
        
        # Test attack indicators calculation
        network_metrics = {'congestion': 0.7, 'fork_rate': 0.3, 'block_withholding': 0.4, 'voting_deviation': 0.5}
        transactions = [{'value': 60, 'status': 'completed'}, {'value': 70, 'status': 'failed'}, {'value': 30, 'status': 'completed'}]
        
        indicators = sec_metrics.calculate_attack_indicators(
            failed_tx_ratio=0.4,
            node_trust_variance=0.2,
            latency_deviation=0.6,
            network_metrics=network_metrics,
            transactions=transactions
        )
        
        # Check indicators
        self.assertIn('51_percent', indicators)
        self.assertIn('ddos', indicators)
        self.assertIn('mixed', indicators)
        self.assertIn('selfish_mining', indicators)
        self.assertIn('bribery', indicators)
        
        # Test attack detection
        attack_type, confidence = sec_metrics.detect_attack()
        self.assertIn(attack_type, [None, '51_percent', 'ddos', 'mixed', 'selfish_mining', 'bribery'])

if __name__ == "__main__":
    unittest.main() 