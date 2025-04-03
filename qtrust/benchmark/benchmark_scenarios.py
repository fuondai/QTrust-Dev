#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark Scenarios - Collection of benchmark scenarios for QTrust

This file defines standard benchmark scenarios to evaluate QTrust performance
under realistic network conditions and compare it with other blockchain systems.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any

@dataclass
class NetworkCondition:
    """Describes network conditions for benchmark scenarios."""
    latency_base: float = 10.0  # Base latency (ms)
    latency_variance: float = 5.0  # Latency variance (ms)
    packet_loss_rate: float = 0.01  # Packet loss rate (0-1)
    bandwidth_limit: float = 1000.0  # Bandwidth limit (MB/s)
    congestion_probability: float = 0.05  # Congestion probability (0-1)
    jitter: float = 2.0  # Latency jitter (ms)

@dataclass
class AttackProfile:
    """Describes attacks in benchmark scenarios."""
    attack_type: str = "none"  # Attack type: none, ddos, 51_percent, sybil, eclipse, mixed
    malicious_node_percentage: float = 0.0  # Percentage of malicious nodes (0-1)
    attack_intensity: float = 0.0  # Attack intensity (0-1)
    attack_target: str = "random"  # Attack target: random, specific_shard, validators
    attack_duration: int = 0  # Attack duration (steps)
    attack_start_step: int = 0  # Step to start attack

@dataclass
class WorkloadProfile:
    """Describes workload for benchmark scenarios."""
    transactions_per_step_base: int = 100  # Base transactions per step
    transactions_per_step_variance: int = 20  # Variance in transactions per step
    cross_shard_transaction_ratio: float = 0.3  # Cross-shard transaction ratio (0-1)
    transaction_value_mean: float = 25.0  # Mean transaction value
    transaction_value_variance: float = 10.0  # Variance in transaction value
    transaction_size_mean: float = 1.0  # Mean transaction size (KB)
    transaction_size_variance: float = 0.2  # Variance in transaction size (KB)
    bursty_traffic: bool = False  # Whether traffic has bursts
    burst_interval: int = 50  # Interval between burst peaks (steps)
    burst_multiplier: float = 3.0  # Multiplier for burst traffic

@dataclass
class NodeProfile:
    """Describes node configuration in benchmark scenarios."""
    processing_power_mean: float = 1.0  # Mean processing power
    processing_power_variance: float = 0.2  # Variance in processing power
    energy_efficiency_mean: float = 0.8  # Mean energy efficiency (0-1)
    energy_efficiency_variance: float = 0.1  # Variance in energy efficiency
    reliability_mean: float = 0.95  # Mean reliability (0-1)
    reliability_variance: float = 0.05  # Variance in reliability
    node_failure_rate: float = 0.01  # Node failure rate (0-1)
    node_recovery_rate: float = 0.8  # Node recovery rate (0-1)

@dataclass
class BenchmarkScenario:
    """Full definition of a benchmark scenario."""
    id: str  # Unique ID for the scenario
    name: str  # Descriptive name for the scenario
    description: str  # Detailed description of the scenario
    num_shards: int  # Number of shards
    nodes_per_shard: int  # Number of nodes per shard
    max_steps: int  # Maximum simulation steps
    network_conditions: NetworkCondition  # Network conditions
    attack_profile: AttackProfile  # Attack information
    workload_profile: WorkloadProfile  # Workload information
    node_profile: NodeProfile  # Node configuration information
    enable_dynamic_resharding: bool = True  # Allow dynamic resharding
    min_shards: int = 4  # Minimum number of shards
    max_shards: int = 32  # Maximum number of shards
    enable_adaptive_consensus: bool = True  # Use adaptive consensus
    enable_bls: bool = True  # Use BLS signature aggregation
    enable_adaptive_pos: bool = True  # Use Adaptive PoS
    enable_lightweight_crypto: bool = True  # Use lightweight cryptography
    enable_federated: bool = False  # Use Federated Learning
    seed: Optional[int] = 42  # Seed for reproducibility
    
    def get_command_line_args(self) -> str:
        """Convert scenario to command line arguments for main.py."""
        args = []
        args.append(f"--num-shards {self.num_shards}")
        args.append(f"--nodes-per-shard {self.nodes_per_shard}")
        args.append(f"--max-steps {self.max_steps}")
        
        if self.attack_profile.attack_type != "none":
            args.append(f"--attack-scenario {self.attack_profile.attack_type}")
            
        if self.enable_bls:
            args.append("--enable-bls")
            
        if self.enable_adaptive_pos:
            args.append("--enable-adaptive-pos")
            
        if self.enable_lightweight_crypto:
            args.append("--enable-lightweight-crypto")
            
        if self.enable_federated:
            args.append("--enable-federated")
            
        if self.seed is not None:
            args.append(f"--seed {self.seed}")
            
        return " ".join(args)

# Define standard benchmark scenarios
BENCHMARK_SCENARIOS = {
    # Basic scenario - stable system with average load
    "basic": BenchmarkScenario(
        id="basic",
        name="Basic",
        description="Basic simulation with stable network conditions and average load",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(
            latency_base=10.0,
            latency_variance=5.0,
            packet_loss_rate=0.01,
            bandwidth_limit=1000.0,
            congestion_probability=0.05,
            jitter=2.0
        ),
        attack_profile=AttackProfile(),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=200,
            transactions_per_step_variance=50,
            cross_shard_transaction_ratio=0.3
        ),
        node_profile=NodeProfile()
    ),
    
    # High load scenario - test performance under high load
    "high_load": BenchmarkScenario(
        id="high_load",
        name="High Load",
        description="Simulation with high transaction load to test scalability",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(
            latency_base=15.0,
            latency_variance=8.0,
            packet_loss_rate=0.02,
            bandwidth_limit=800.0,
            congestion_probability=0.1,
            jitter=3.0
        ),
        attack_profile=AttackProfile(),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=500,
            transactions_per_step_variance=100,
            cross_shard_transaction_ratio=0.4,
            bursty_traffic=True,
            burst_interval=100,
            burst_multiplier=3.0
        ),
        node_profile=NodeProfile(
            processing_power_variance=0.3,
            node_failure_rate=0.02
        )
    ),
    
    # DDoS attack scenario
    "ddos_attack": BenchmarkScenario(
        id="ddos_attack",
        name="DDoS Attack",
        description="Simulation under distributed denial of service attack",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(
            latency_base=20.0,
            latency_variance=10.0,
            packet_loss_rate=0.05,
            bandwidth_limit=600.0,
            congestion_probability=0.2,
            jitter=5.0
        ),
        attack_profile=AttackProfile(
            attack_type="ddos",
            malicious_node_percentage=0.1,
            attack_intensity=0.8,
            attack_target="random",
            attack_duration=300,
            attack_start_step=200
        ),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=300,
            cross_shard_transaction_ratio=0.35
        ),
        node_profile=NodeProfile(
            reliability_mean=0.9,
            node_failure_rate=0.03
        )
    ),
    
    # 51% attack scenario
    "fifty_one_percent": BenchmarkScenario(
        id="fifty_one_percent",
        name="51% Attack",
        description="Simulation under 51% attack focused on a single shard",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(),
        attack_profile=AttackProfile(
            attack_type="51_percent",
            malicious_node_percentage=0.15,
            attack_intensity=0.9,
            attack_target="specific_shard",
            attack_duration=400,
            attack_start_step=300
        ),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=250,
            cross_shard_transaction_ratio=0.35,
            transaction_value_mean=40.0
        ),
        node_profile=NodeProfile()
    ),
    
    # Sybil attack scenario
    "sybil_attack": BenchmarkScenario(
        id="sybil_attack",
        name="Sybil Attack",
        description="Simulation under Sybil attack with many malicious nodes spoofing identities",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(),
        attack_profile=AttackProfile(
            attack_type="sybil",
            malicious_node_percentage=0.2,
            attack_intensity=0.7,
            attack_target="validators",
            attack_duration=500,
            attack_start_step=250
        ),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=200
        ),
        node_profile=NodeProfile(
            reliability_variance=0.15
        )
    ),
    
    # Eclipse attack scenario
    "eclipse_attack": BenchmarkScenario(
        id="eclipse_attack",
        name="Eclipse Attack",
        description="Simulation under Eclipse attack aimed at isolating a shard from the network",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(
            latency_variance=15.0,
            packet_loss_rate=0.08
        ),
        attack_profile=AttackProfile(
            attack_type="eclipse",
            malicious_node_percentage=0.25,
            attack_intensity=0.85,
            attack_target="specific_shard",
            attack_duration=350,
            attack_start_step=400
        ),
        workload_profile=WorkloadProfile(
            cross_shard_transaction_ratio=0.5
        ),
        node_profile=NodeProfile()
    ),
    
    # Real world network conditions scenario - high latency, packet loss
    "real_world_conditions": BenchmarkScenario(
        id="real_world_conditions",
        name="Real World Conditions",
        description="Simulation under real-world network conditions with high latency and packet loss",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(
            latency_base=50.0,
            latency_variance=25.0,
            packet_loss_rate=0.1,
            bandwidth_limit=500.0,
            congestion_probability=0.15,
            jitter=15.0
        ),
        attack_profile=AttackProfile(),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=150,
            cross_shard_transaction_ratio=0.4,
            bursty_traffic=True
        ),
        node_profile=NodeProfile(
            processing_power_variance=0.4,
            reliability_mean=0.85,
            node_failure_rate=0.05
        )
    ),
    
    # Large scale scenario - test with high number of shards and nodes
    "large_scale": BenchmarkScenario(
        id="large_scale",
        name="Large Scale",
        description="Large-scale simulation with high number of shards and nodes",
        num_shards=32,
        nodes_per_shard=30,
        max_steps=500,
        network_conditions=NetworkCondition(),
        attack_profile=AttackProfile(),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=1000,
            transactions_per_step_variance=200,
            cross_shard_transaction_ratio=0.45
        ),
        node_profile=NodeProfile(),
        enable_federated=True
    ),
    
    # Mixed attack scenario - multiple attack types simultaneously
    "mixed_attack": BenchmarkScenario(
        id="mixed_attack",
        name="Mixed Attack",
        description="Simulation under multiple combined attack types (DDoS + Sybil)",
        num_shards=24,
        nodes_per_shard=20,
        max_steps=1000,
        network_conditions=NetworkCondition(
            latency_base=30.0,
            packet_loss_rate=0.07
        ),
        attack_profile=AttackProfile(
            attack_type="mixed",
            malicious_node_percentage=0.3,
            attack_intensity=0.9,
            attack_target="random",
            attack_duration=600,
            attack_start_step=200
        ),
        workload_profile=WorkloadProfile(
            transactions_per_step_base=200
        ),
        node_profile=NodeProfile(
            node_failure_rate=0.04
        )
    )
}

def get_scenario(scenario_id: str) -> BenchmarkScenario:
    """Get a benchmark scenario by ID."""
    if scenario_id not in BENCHMARK_SCENARIOS:
        raise ValueError(f"Benchmark scenario not found with ID: {scenario_id}")
    return BENCHMARK_SCENARIOS[scenario_id]

def get_all_scenario_ids() -> List[str]:
    """Get a list of IDs for all benchmark scenarios."""
    return list(BENCHMARK_SCENARIOS.keys())

def get_all_scenarios() -> Dict[str, BenchmarkScenario]:
    """Get all benchmark scenarios."""
    return BENCHMARK_SCENARIOS

if __name__ == "__main__":
    # Display all scenarios and corresponding commands
    for scenario_id, scenario in BENCHMARK_SCENARIOS.items():
        print(f"Scenario: {scenario.name} ({scenario_id})")
        print(f"Description: {scenario.description}")
        print(f"Command: py -3.10 -m main {scenario.get_command_line_args()}")
        print("-" * 80) 