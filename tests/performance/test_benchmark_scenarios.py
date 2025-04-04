#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the QTrust Benchmark Scenarios module.
"""

import os
import sys
import unittest
from typing import Dict, List

# Add the root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qtrust.benchmark.benchmark_scenarios import (
    NetworkCondition, AttackProfile, WorkloadProfile, NodeProfile, 
    BenchmarkScenario, get_scenario, get_all_scenario_ids, get_all_scenarios
)


class TestBenchmarkScenarios(unittest.TestCase):
    """Test cases for benchmark scenarios module."""
    
    def test_network_condition_initialization(self):
        """Test initialization of NetworkCondition."""
        # Default initialization
        network_condition = NetworkCondition()
        self.assertEqual(network_condition.latency_base, 10.0)
        self.assertEqual(network_condition.latency_variance, 5.0)
        self.assertEqual(network_condition.packet_loss_rate, 0.01)
        self.assertEqual(network_condition.bandwidth_limit, 1000.0)
        self.assertEqual(network_condition.congestion_probability, 0.05)
        self.assertEqual(network_condition.jitter, 2.0)
        
        # Custom initialization
        custom_network_condition = NetworkCondition(
            latency_base=20.0,
            latency_variance=10.0,
            packet_loss_rate=0.05,
            bandwidth_limit=500.0,
            congestion_probability=0.1,
            jitter=5.0
        )
        self.assertEqual(custom_network_condition.latency_base, 20.0)
        self.assertEqual(custom_network_condition.latency_variance, 10.0)
        self.assertEqual(custom_network_condition.packet_loss_rate, 0.05)
        self.assertEqual(custom_network_condition.bandwidth_limit, 500.0)
        self.assertEqual(custom_network_condition.congestion_probability, 0.1)
        self.assertEqual(custom_network_condition.jitter, 5.0)
    
    def test_attack_profile_initialization(self):
        """Test initialization of AttackProfile."""
        # Default initialization
        attack_profile = AttackProfile()
        self.assertEqual(attack_profile.attack_type, "none")
        self.assertEqual(attack_profile.malicious_node_percentage, 0.0)
        self.assertEqual(attack_profile.attack_intensity, 0.0)
        self.assertEqual(attack_profile.attack_target, "random")
        self.assertEqual(attack_profile.attack_duration, 0)
        self.assertEqual(attack_profile.attack_start_step, 0)
        
        # Custom initialization
        custom_attack_profile = AttackProfile(
            attack_type="ddos",
            malicious_node_percentage=0.2,
            attack_intensity=0.8,
            attack_target="validators",
            attack_duration=300,
            attack_start_step=200
        )
        self.assertEqual(custom_attack_profile.attack_type, "ddos")
        self.assertEqual(custom_attack_profile.malicious_node_percentage, 0.2)
        self.assertEqual(custom_attack_profile.attack_intensity, 0.8)
        self.assertEqual(custom_attack_profile.attack_target, "validators")
        self.assertEqual(custom_attack_profile.attack_duration, 300)
        self.assertEqual(custom_attack_profile.attack_start_step, 200)
    
    def test_workload_profile_initialization(self):
        """Test initialization of WorkloadProfile."""
        # Default initialization
        workload_profile = WorkloadProfile()
        self.assertEqual(workload_profile.transactions_per_step_base, 100)
        self.assertEqual(workload_profile.transactions_per_step_variance, 20)
        self.assertEqual(workload_profile.cross_shard_transaction_ratio, 0.3)
        self.assertEqual(workload_profile.transaction_value_mean, 25.0)
        self.assertEqual(workload_profile.transaction_value_variance, 10.0)
        self.assertEqual(workload_profile.transaction_size_mean, 1.0)
        self.assertEqual(workload_profile.transaction_size_variance, 0.2)
        self.assertEqual(workload_profile.bursty_traffic, False)
        self.assertEqual(workload_profile.burst_interval, 50)
        self.assertEqual(workload_profile.burst_multiplier, 3.0)
        
        # Custom initialization
        custom_workload_profile = WorkloadProfile(
            transactions_per_step_base=500,
            transactions_per_step_variance=100,
            cross_shard_transaction_ratio=0.5,
            transaction_value_mean=50.0,
            transaction_value_variance=20.0,
            transaction_size_mean=2.0,
            transaction_size_variance=0.5,
            bursty_traffic=True,
            burst_interval=100,
            burst_multiplier=5.0
        )
        self.assertEqual(custom_workload_profile.transactions_per_step_base, 500)
        self.assertEqual(custom_workload_profile.transactions_per_step_variance, 100)
        self.assertEqual(custom_workload_profile.cross_shard_transaction_ratio, 0.5)
        self.assertEqual(custom_workload_profile.transaction_value_mean, 50.0)
        self.assertEqual(custom_workload_profile.transaction_value_variance, 20.0)
        self.assertEqual(custom_workload_profile.transaction_size_mean, 2.0)
        self.assertEqual(custom_workload_profile.transaction_size_variance, 0.5)
        self.assertEqual(custom_workload_profile.bursty_traffic, True)
        self.assertEqual(custom_workload_profile.burst_interval, 100)
        self.assertEqual(custom_workload_profile.burst_multiplier, 5.0)
    
    def test_node_profile_initialization(self):
        """Test initialization of NodeProfile."""
        # Default initialization
        node_profile = NodeProfile()
        self.assertEqual(node_profile.processing_power_mean, 1.0)
        self.assertEqual(node_profile.processing_power_variance, 0.2)
        self.assertEqual(node_profile.energy_efficiency_mean, 0.8)
        self.assertEqual(node_profile.energy_efficiency_variance, 0.1)
        self.assertEqual(node_profile.reliability_mean, 0.95)
        self.assertEqual(node_profile.reliability_variance, 0.05)
        self.assertEqual(node_profile.node_failure_rate, 0.01)
        self.assertEqual(node_profile.node_recovery_rate, 0.8)
        
        # Custom initialization
        custom_node_profile = NodeProfile(
            processing_power_mean=2.0,
            processing_power_variance=0.5,
            energy_efficiency_mean=0.9,
            energy_efficiency_variance=0.2,
            reliability_mean=0.98,
            reliability_variance=0.02,
            node_failure_rate=0.005,
            node_recovery_rate=0.9
        )
        self.assertEqual(custom_node_profile.processing_power_mean, 2.0)
        self.assertEqual(custom_node_profile.processing_power_variance, 0.5)
        self.assertEqual(custom_node_profile.energy_efficiency_mean, 0.9)
        self.assertEqual(custom_node_profile.energy_efficiency_variance, 0.2)
        self.assertEqual(custom_node_profile.reliability_mean, 0.98)
        self.assertEqual(custom_node_profile.reliability_variance, 0.02)
        self.assertEqual(custom_node_profile.node_failure_rate, 0.005)
        self.assertEqual(custom_node_profile.node_recovery_rate, 0.9)
    
    def test_benchmark_scenario_initialization(self):
        """Test initialization of BenchmarkScenario."""
        scenario = BenchmarkScenario(
            id="test_scenario",
            name="Test Scenario",
            description="Scenario for testing",
            num_shards=16,
            nodes_per_shard=10,
            max_steps=500,
            network_conditions=NetworkCondition(),
            attack_profile=AttackProfile(),
            workload_profile=WorkloadProfile(),
            node_profile=NodeProfile()
        )
        
        self.assertEqual(scenario.id, "test_scenario")
        self.assertEqual(scenario.name, "Test Scenario")
        self.assertEqual(scenario.description, "Scenario for testing")
        self.assertEqual(scenario.num_shards, 16)
        self.assertEqual(scenario.nodes_per_shard, 10)
        self.assertEqual(scenario.max_steps, 500)
        self.assertTrue(scenario.enable_dynamic_resharding)
        self.assertEqual(scenario.min_shards, 4)
        self.assertEqual(scenario.max_shards, 32)
        self.assertTrue(scenario.enable_adaptive_consensus)
        self.assertTrue(scenario.enable_bls)
        self.assertTrue(scenario.enable_adaptive_pos)
        self.assertTrue(scenario.enable_lightweight_crypto)
        self.assertFalse(scenario.enable_federated)
        self.assertEqual(scenario.seed, 42)
    
    def test_get_command_line_args(self):
        """Test generation of command line arguments."""
        scenario = BenchmarkScenario(
            id="test_scenario",
            name="Test Scenario",
            description="Scenario for testing",
            num_shards=16,
            nodes_per_shard=10,
            max_steps=500,
            network_conditions=NetworkCondition(),
            attack_profile=AttackProfile(),
            workload_profile=WorkloadProfile(),
            node_profile=NodeProfile()
        )
        
        args = scenario.get_command_line_args()
        self.assertIn("--num-shards 16", args)
        self.assertIn("--nodes-per-shard 10", args)
        self.assertIn("--max-steps 500", args)
        self.assertIn("--enable-bls", args)
        self.assertIn("--enable-adaptive-pos", args)
        self.assertIn("--enable-lightweight-crypto", args)
        self.assertIn("--seed 42", args)
        
        # Test with attack scenario
        attack_scenario = BenchmarkScenario(
            id="attack_scenario",
            name="Attack Scenario",
            description="Scenario with attack",
            num_shards=16,
            nodes_per_shard=10,
            max_steps=500,
            network_conditions=NetworkCondition(),
            attack_profile=AttackProfile(attack_type="ddos"),
            workload_profile=WorkloadProfile(),
            node_profile=NodeProfile()
        )
        
        attack_args = attack_scenario.get_command_line_args()
        self.assertIn("--attack-scenario ddos", attack_args)
        
        # Test with federated enabled
        federated_scenario = BenchmarkScenario(
            id="federated_scenario",
            name="Federated Scenario",
            description="Scenario with federated learning",
            num_shards=16,
            nodes_per_shard=10,
            max_steps=500,
            network_conditions=NetworkCondition(),
            attack_profile=AttackProfile(),
            workload_profile=WorkloadProfile(),
            node_profile=NodeProfile(),
            enable_federated=True
        )
        
        federated_args = federated_scenario.get_command_line_args()
        self.assertIn("--enable-federated", federated_args)
    
    def test_get_scenario(self):
        """Test retrieving scenario by ID."""
        # Test retrieving existing scenario
        basic_scenario = get_scenario("basic")
        self.assertEqual(basic_scenario.id, "basic")
        self.assertEqual(basic_scenario.name, "Basic")
        
        high_load_scenario = get_scenario("high_load")
        self.assertEqual(high_load_scenario.id, "high_load")
        self.assertEqual(high_load_scenario.name, "High Load")
        
        # Test retrieving non-existent scenario
        with self.assertRaises(ValueError):
            get_scenario("non_existent_scenario")
    
    def test_get_all_scenario_ids(self):
        """Test retrieving all scenario IDs."""
        scenario_ids = get_all_scenario_ids()
        self.assertIsInstance(scenario_ids, list)
        self.assertIn("basic", scenario_ids)
        self.assertIn("high_load", scenario_ids)
        self.assertIn("ddos_attack", scenario_ids)
        self.assertIn("fifty_one_percent", scenario_ids)
        self.assertIn("sybil_attack", scenario_ids)
        self.assertIn("eclipse_attack", scenario_ids)
        self.assertIn("real_world_conditions", scenario_ids)
        self.assertIn("large_scale", scenario_ids)
        self.assertIn("mixed_attack", scenario_ids)
    
    def test_get_all_scenarios(self):
        """Test retrieving all scenarios."""
        scenarios = get_all_scenarios()
        self.assertIsInstance(scenarios, dict)
        self.assertIn("basic", scenarios)
        self.assertIn("high_load", scenarios)
        self.assertEqual(scenarios["basic"].id, "basic")
        self.assertEqual(scenarios["high_load"].id, "high_load")


if __name__ == '__main__':
    unittest.main() 