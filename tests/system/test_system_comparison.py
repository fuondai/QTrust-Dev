#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import shutil
import pandas as pd

from qtrust.benchmark.system_comparison import (
    update_system_data,
    save_system_data,
    plot_throughput_vs_security,
    generate_performance_metrics_table,
    plot_performance_radar,
    generate_system_comparison_report,
    TRANSACTION_SYSTEMS
)


class TestSystemComparison(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_update_system_data(self):
        # Save original data to restore later
        original_data = TRANSACTION_SYSTEMS.copy()
        
        try:
            # Update existing system
            test_metrics = {
                'throughput': 6000,
                'security_score': 0.97
            }
            update_system_data('QTrust', test_metrics)
            
            # Check if data was updated
            self.assertEqual(TRANSACTION_SYSTEMS['QTrust']['throughput'], 6000)
            self.assertEqual(TRANSACTION_SYSTEMS['QTrust']['security_score'], 0.97)
            
            # Add new system
            new_system_metrics = {
                'throughput': 3000,
                'latency': 5.0,
                'overhead': 12,
                'security_score': 0.88,
                'energy_efficiency': 0.85,
                'resource_utilization': 0.82,
                'fault_tolerance': 0.91
            }
            update_system_data('NewSystem', new_system_metrics)
            
            # Check if new system was added
            self.assertIn('NewSystem', TRANSACTION_SYSTEMS)
            self.assertEqual(TRANSACTION_SYSTEMS['NewSystem']['throughput'], 3000)
            
        finally:
            # Restore original data
            for system in list(TRANSACTION_SYSTEMS.keys()):
                if system not in original_data:
                    del TRANSACTION_SYSTEMS[system]
            
            for system, metrics in original_data.items():
                TRANSACTION_SYSTEMS[system] = metrics.copy()
    
    @patch('json.dump')
    def test_save_system_data(self, mock_json_dump):
        # Test saving system data
        save_system_data(self.test_dir)
        
        # Check if json.dump was called
        self.assertTrue(mock_json_dump.called)
        
        # Check if file exists in directory
        files = os.listdir(self.test_dir)
        self.assertTrue(any(file.startswith('transaction_systems_data_') for file in files))
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_throughput_vs_security(self, mock_savefig):
        # Test plotting function
        plot_throughput_vs_security(self.test_dir)
        
        # Check if savefig was called
        mock_savefig.assert_called_once()
        
        # Check the filename pattern
        args, kwargs = mock_savefig.call_args
        filepath = args[0]
        self.assertTrue(filepath.startswith(os.path.join(self.test_dir, 'throughput_vs_security_')))
    
    def test_generate_performance_metrics_table(self):
        # Test generating metrics table
        df = generate_performance_metrics_table(self.test_dir)
        
        # Check if dataframe was created correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('Performance Index', df.columns)
        self.assertIn('Rank', df.columns)
        
        # Check files created
        files = os.listdir(self.test_dir)
        self.assertTrue(any(file.startswith('system_performance_table_') for file in files))
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_performance_radar(self, mock_savefig):
        # Test radar plot function
        plot_performance_radar(self.test_dir)
        
        # Check if savefig was called
        mock_savefig.assert_called_once()
        
        # Check the filename pattern
        args, kwargs = mock_savefig.call_args
        filepath = args[0]
        self.assertTrue(filepath.startswith(os.path.join(self.test_dir, 'system_performance_radar_')))
    
    @patch('qtrust.benchmark.system_comparison.plot_throughput_vs_security')
    @patch('qtrust.benchmark.system_comparison.plot_performance_radar')
    @patch('qtrust.benchmark.system_comparison.generate_performance_metrics_table')
    @patch('qtrust.benchmark.system_comparison.save_system_data')
    def test_generate_system_comparison_report(
        self, mock_save, mock_table, mock_radar, mock_throughput
    ):
        # Mock the table return value
        mock_table.return_value = pd.DataFrame({
            'Performance Index': [0.95, 0.85],
            'Rank': [1, 2],
            'Throughput': [5000, 4000],
            'Latency': [2.5, 3.0]
        }, index=['QTrust', 'OtherSystem'])
        
        # Test generating report
        generate_system_comparison_report(self.test_dir)
        
        # Check if all component functions were called
        mock_throughput.assert_called_once_with(self.test_dir)
        mock_radar.assert_called_once_with(self.test_dir)
        mock_table.assert_called_once_with(self.test_dir)
        mock_save.assert_called_once_with(self.test_dir)
        
        # Check if report file was created
        files = os.listdir(self.test_dir)
        self.assertTrue(any(file.startswith('system_comparison_report_') for file in files))


if __name__ == '__main__':
    unittest.main() 