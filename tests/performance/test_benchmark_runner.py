#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the QTrust Benchmark Runner module.
"""

import os
import sys
import unittest
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path

# Add the root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qtrust.benchmark.benchmark_runner import (
    run_benchmark, run_all_benchmarks, generate_comparison_report,
    plot_comparison_charts, parse_args, main
)

# Mock benchmark scenario
class MockScenario:
    def __init__(self):
        self.id = "test_scenario"
        self.name = "Test Scenario"
        self.description = "Description for testing"
        self.num_shards = 2
        self.nodes_per_shard = 4
        self.max_steps = 100
        
        # Mock objects for nested attributes
        self.network_conditions = MagicMock()
        self.network_conditions.latency_base = 50
        self.network_conditions.latency_variance = 10
        self.network_conditions.packet_loss_rate = 0.01
        self.network_conditions.bandwidth_limit = 10000
        self.network_conditions.congestion_probability = 0.05
        self.network_conditions.jitter = 5
        
        self.attack_profile = MagicMock()
        self.attack_profile.attack_type = "none"
        self.attack_profile.malicious_node_percentage = 0
        self.attack_profile.attack_intensity = 0
        self.attack_profile.attack_target = "none"
        self.attack_profile.attack_duration = 0
        self.attack_profile.attack_start_step = 0
        
        self.workload_profile = MagicMock()
        self.workload_profile.transactions_per_step_base = 100
        self.workload_profile.transactions_per_step_variance = 20
        self.workload_profile.cross_shard_transaction_ratio = 0.3
        self.workload_profile.transaction_value_mean = 50
        self.workload_profile.transaction_value_variance = 10
        self.workload_profile.transaction_size_mean = 200
        self.workload_profile.transaction_size_variance = 50
        self.workload_profile.bursty_traffic = False
        self.workload_profile.burst_interval = 0
        self.workload_profile.burst_multiplier = 1
        
        self.node_profile = MagicMock()
        self.node_profile.processing_power_mean = 100
        self.node_profile.processing_power_variance = 20
        self.node_profile.energy_efficiency_mean = 80
        self.node_profile.energy_efficiency_variance = 10
        self.node_profile.reliability_mean = 0.95
        self.node_profile.reliability_variance = 0.05
        self.node_profile.node_failure_rate = 0.01
        self.node_profile.node_recovery_rate = 0.8
        
        self.enable_dynamic_resharding = True
        self.min_shards = 1
        self.max_shards = 4
        self.enable_adaptive_consensus = True
        self.enable_bls = True
        self.enable_adaptive_pos = True
        self.enable_lightweight_crypto = True
        self.enable_federated = True
        self.seed = 42
    
    def get_command_line_args(self):
        return "--num-shards 2 --nodes-per-shard 4 --steps 100"


class TestBenchmarkRunner(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for test results
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch('qtrust.benchmark.benchmark_runner.get_scenario')
    @patch('subprocess.Popen')
    def test_run_benchmark(self, mock_popen, mock_get_scenario):
        # Configure mocks
        mock_get_scenario.return_value = MockScenario()
        
        # Mock subprocess.Popen
        process_mock = MagicMock()
        process_mock.stdout = ["Test output line 1\n", "Test output line 2\n"]
        process_mock.returncode = 0
        mock_popen.return_value = process_mock
        
        # Create a mock results file
        results_dir = os.path.join(self.temp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "final_metrics.json"), "w") as f:
            json.dump({
                "average_throughput": 150,
                "average_latency": 75,
                "average_energy": 30,
                "security_score": 0.95,
                "cross_shard_ratio": 0.3
            }, f)
        
        # Run test function
        with patch('qtrust.benchmark.benchmark_runner.RESULTS_DIR', self.temp_dir):
            result = run_benchmark("test_scenario", output_dir=self.temp_dir, verbose=False)
        
        # Assertions
        self.assertEqual(result["scenario_id"], "test_scenario")
        self.assertEqual(result["scenario_name"], "Test Scenario")
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("execution_time", result)
        self.assertIn("timestamp", result)
        self.assertIn("results", result)
        
        # Check results were parsed correctly
        self.assertEqual(result["results"]["average_throughput"], 150)
        self.assertEqual(result["results"]["average_latency"], 75)
        self.assertEqual(result["results"]["average_energy"], 30)
        self.assertEqual(result["results"]["security_score"], 0.95)
        
        # Verify config file creation
        config_path = os.path.join(self.temp_dir, "scenario_config.json")
        self.assertTrue(os.path.exists(config_path))
        
        # Verify calls
        mock_get_scenario.assert_called_once_with("test_scenario")
        mock_popen.assert_called_once()
    
    @patch('qtrust.benchmark.benchmark_runner.get_all_scenario_ids')
    @patch('qtrust.benchmark.benchmark_runner.run_benchmark')
    def test_run_all_benchmarks(self, mock_run_benchmark, mock_get_all_scenario_ids):
        # Configure mocks
        mock_get_all_scenario_ids.return_value = ["scenario1", "scenario2"]
        
        mock_results = {
            "scenario_id": "scenario1",
            "scenario_name": "Scenario 1",
            "execution_time": 10.5,
            "results": {"average_throughput": 120}
        }
        mock_run_benchmark.return_value = mock_results
        
        # Run test function
        with patch('qtrust.benchmark.benchmark_runner.RESULTS_DIR', self.temp_dir):
            results = run_all_benchmarks(parallel=False, verbose=False)
        
        # Assertions
        self.assertEqual(len(results), 2)
        self.assertIn("scenario1", results)
        self.assertIn("scenario2", results)
        self.assertEqual(results["scenario1"], mock_results)
        
        # Verify summary file creation
        summary_files = [f for f in os.listdir(self.temp_dir) if f.startswith("batch_")]
        self.assertEqual(len(summary_files), 1)
        
        # Verify calls
        mock_get_all_scenario_ids.assert_called_once()
        self.assertEqual(mock_run_benchmark.call_count, 2)
    
    def test_generate_comparison_report(self):
        # Create test data
        batch_dir = os.path.join(self.temp_dir, "batch_20250101_000000")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Create scenario results
        scenarios = ["scenario1", "scenario2"]
        for scenario in scenarios:
            scenario_dir = os.path.join(batch_dir, scenario)
            os.makedirs(scenario_dir, exist_ok=True)
            
            # Create benchmark results
            with open(os.path.join(scenario_dir, "benchmark_results.json"), "w") as f:
                json.dump({
                    "scenario_id": scenario,
                    "scenario_name": f"Scenario {scenario[-1]}",
                    "execution_time": 10.5,
                    "results": {
                        "average_throughput": 120 + int(scenario[-1]) * 10,
                        "average_latency": 80 - int(scenario[-1]) * 5,
                        "average_energy": 40 - int(scenario[-1]) * 2,
                        "security_score": 0.9 + int(scenario[-1]) * 0.05,
                        "cross_shard_ratio": 0.3
                    }
                }, f)
            
            # Create config file
            with open(os.path.join(scenario_dir, "scenario_config.json"), "w") as f:
                json.dump({
                    "num_shards": 2,
                    "nodes_per_shard": 4,
                    "attack_profile": {"attack_type": "none"}
                }, f)
        
        # Run test function with mocked plot function to avoid actual plotting
        with patch('qtrust.benchmark.benchmark_runner.RESULTS_DIR', self.temp_dir):
            with patch('qtrust.benchmark.benchmark_runner.plot_comparison_charts'):
                df = generate_comparison_report(results_dir=batch_dir)
        
        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("Scenario ID", df.columns)
        self.assertIn("Throughput (tx/s)", df.columns)
        
        # Verify CSV file creation
        csv_files = [f for f in os.listdir(batch_dir) if f.endswith(".csv")]
        self.assertEqual(len(csv_files), 1)

    @patch('argparse.ArgumentParser.parse_args')
    @patch('qtrust.benchmark.benchmark_runner.run_benchmark')
    def test_main_run_command(self, mock_run_benchmark, mock_parse_args):
        # Configure mock
        args = MagicMock()
        args.command = "run"
        args.scenario_id = "test_scenario"
        args.output_dir = None
        args.quiet = False
        mock_parse_args.return_value = args
        
        # Run main function
        main()
        
        # Verify calls
        mock_run_benchmark.assert_called_once_with("test_scenario", None, True)

if __name__ == '__main__':
    unittest.main() 