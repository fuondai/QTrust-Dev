#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the Blockchain Comparison Utilities module.
"""

import os
import sys
import unittest
import tempfile
import importlib
import pandas as pd
from unittest.mock import patch, Mock
from typing import Dict

# Add the root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qtrust.benchmark.blockchain_comparison_utils import (
    plot_heatmap_comparison,
    plot_relationship_comparison,
    generate_comparison_table,
    run_all_comparisons
)


class TestBlockchainComparisonUtils(unittest.TestCase):
    """Test cases for blockchain comparison utilities module."""
    
    def setUp(self):
        """Set up the test environment."""
        self.test_systems = {
            'System1': {
                'throughput': 5000,
                'latency': 2.5,
                'security': 0.95,
                'energy': 15,
                'scalability': 0.9,
                'decentralization': 0.85,
                'cross_shard_efficiency': 0.92,
                'attack_resistance': 0.94
            },
            'System2': {
                'throughput': 3000,
                'latency': 5.0,
                'security': 0.90,
                'energy': 20,
                'scalability': 0.85,
                'decentralization': 0.80,
                'cross_shard_efficiency': 0.88,
                'attack_resistance': 0.92
            },
            'System3': {
                'throughput': 10000,
                'latency': 1.0,
                'security': 0.85,
                'energy': 10,
                'scalability': 0.95,
                'decentralization': 0.75,
                'cross_shard_efficiency': 0.90,
                'attack_resistance': 0.88
            }
        }
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def test_plot_heatmap_comparison(self):
        """Test the heatmap comparison plot generation."""
        plot_heatmap_comparison(self.test_systems, self.temp_dir)
        
        # Check if at least one PNG file was created
        png_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        self.assertTrue(any('heatmap' in f for f in png_files))
        
        # Check if CSV file was created
        csv_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.csv')]
        self.assertTrue(any('normalized_metrics' in f for f in csv_files))
    
    def test_plot_relationship_comparison(self):
        """Test the relationship comparison plot generation."""
        plot_relationship_comparison(self.test_systems, self.temp_dir)
        
        # Check if the PNG file was created
        png_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        self.assertTrue(any('relationship' in f for f in png_files))
    
    def test_generate_comparison_table(self):
        """Test generating the comparison table."""
        df = generate_comparison_table(self.test_systems, self.temp_dir)
        
        # Check if the DataFrame was returned
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)  # 3 systems
        
        # Check if the CSV and HTML files were created
        csv_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.csv')]
        self.assertTrue(any('comparison_table' in f for f in csv_files))
        
        html_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.html')]
        self.assertTrue(any('comparison_table' in f for f in html_files))
        
        # Check if Performance Score was calculated
        self.assertIn('Performance Score', df.columns)
        self.assertIn('Rank', df.columns)
    
    @patch('qtrust.benchmark.blockchain_comparison.generate_comparison_report')
    def test_run_all_comparisons(self, mock_generate_report):
        """Test running all comparisons."""
        # Call the function
        run_all_comparisons(self.temp_dir)
        
        # Check if generate_comparison_report was called
        mock_generate_report.assert_called_once_with(self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)


if __name__ == '__main__':
    unittest.main() 