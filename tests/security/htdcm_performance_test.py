#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HTDCM Performance Testing Module

This module provides functionality for testing and analyzing the performance 
of the Hierarchical Trust-based Distributed Consensus Mechanism (HTDCM) 
under various network configurations and attack scenarios.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, List, Tuple
from datetime import datetime
import argparse
from collections import defaultdict

# Add the current directory to PYTHONPATH to import other modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from qtrust.trust.htdcm import HTDCM, HTDCMNode

def simulate_network_with_attacks(
    num_shards: int = 4,
    nodes_per_shard: int = 10,
    malicious_percentage: float = 20,
    attack_type: str = "sybil",
    simulation_steps: int = 100,
    tx_per_step: int = 10
) -> Dict[str, Any]:
    """
    Simulate a blockchain network with HTDCM and different attacks.
    
    Args:
        num_shards: Number of shards in the network
        nodes_per_shard: Number of nodes per shard
        malicious_percentage: Percentage of malicious nodes
        attack_type: Type of attack ('sybil', 'eclipse', '51_percent', 'selfish_mining', 'ddos', 'mixed')
        simulation_steps: Number of simulation steps
        tx_per_step: Number of transactions per step
        
    Returns:
        Dict containing simulation metrics
    """
    print(f"Simulating network with {num_shards} shards, {nodes_per_shard} nodes/shard, {malicious_percentage}% malicious nodes, {attack_type} attack")
    
    # Create blockchain network graph
    G = nx.Graph()
    
    # Create list of shards and nodes in each shard
    total_nodes = num_shards * nodes_per_shard
    shards = []
    
    # Track malicious nodes
    malicious_nodes = []
    
    for s in range(num_shards):
        shard_nodes = []
        for n in range(nodes_per_shard):
            node_id = s * nodes_per_shard + n
            # Determine if node is malicious
            is_malicious = np.random.rand() < (malicious_percentage / 100)
            
            # Add node to graph
            G.add_node(node_id, 
                      shard_id=s, 
                      trust_score=0.7,
                      is_malicious=is_malicious,
                      attack_type=attack_type if is_malicious else None)
            
            if is_malicious:
                malicious_nodes.append(node_id)
                
            shard_nodes.append(node_id)
        shards.append(shard_nodes)
    
    print(f"Created {total_nodes} nodes, with {len(malicious_nodes)} malicious nodes")
    
    # Add intra-shard connections (fully connected)
    for shard_nodes in shards:
        for i in range(len(shard_nodes)):
            for j in range(i + 1, len(shard_nodes)):
                G.add_edge(shard_nodes[i], shard_nodes[j], weight=1)
    
    # Add cross-shard connections (random connections)
    cross_shard_connections = int(total_nodes * 0.3)  # 30% of nodes
    
    for _ in range(cross_shard_connections):
        shard1 = np.random.randint(0, num_shards)
        shard2 = np.random.randint(0, num_shards)
        while shard1 == shard2:
            shard2 = np.random.randint(0, num_shards)
            
        node1 = np.random.choice(shards[shard1])
        node2 = np.random.choice(shards[shard2])
        
        if not G.has_edge(node1, node2):
            G.add_edge(node1, node2, weight=2)  # Cross-shard connections have weight 2
    
    # Set up HTDCM
    htdcm = HTDCM(network=G, shards=shards)
    
    # Implement specific attack configurations
    if attack_type == "eclipse":
        # Eclipse Attack: Isolate certain nodes
        if malicious_nodes:
            # Select random nodes to isolate
            non_malicious = [n for n in G.nodes() if not G.nodes[n].get('is_malicious', False)]
            targets = np.random.choice(non_malicious, 
                                      size=min(3, len(non_malicious)), 
                                      replace=False)
            
            # Isolate target nodes by connecting them only to malicious nodes
            for target in targets:
                # Remove all existing connections
                target_edges = list(G.edges(target))
                for u, v in target_edges:
                    G.remove_edge(u, v)
                
                # Add new connections only to malicious nodes
                for m_node in malicious_nodes:
                    G.add_edge(target, m_node, weight=1)
                print(f"Node {target} has been isolated in Eclipse attack")
    
    elif attack_type == "51_percent":
        # 51% Attack: Ensure at least 51% malicious nodes
        malicious_count = len(malicious_nodes)
        if malicious_count < total_nodes * 0.51:
            # Select additional random nodes to reach 51%
            non_malicious = [n for n in G.nodes() if not G.nodes[n].get('is_malicious', False)]
            additional_needed = int(total_nodes * 0.51) - malicious_count
            
            if additional_needed > 0 and len(non_malicious) > 0:
                additional = np.random.choice(non_malicious, 
                                            size=min(additional_needed, len(non_malicious)), 
                                            replace=False)
                
                for node in additional:
                    G.nodes[node]['is_malicious'] = True
                    G.nodes[node]['attack_type'] = attack_type
                    malicious_nodes.append(node)
                
                print(f"Added {len(additional)} malicious nodes to reach 51%, total: {len(malicious_nodes)}")
    
    # Run simulation and return results
    # This is just a placeholder - actual implementation would run the simulation
    # and collect metrics
    results = {
        "setup": {
            "num_shards": num_shards,
            "nodes_per_shard": nodes_per_shard,
            "malicious_percentage": malicious_percentage,
            "attack_type": attack_type,
            "simulation_steps": simulation_steps,
            "tx_per_step": tx_per_step
        },
        "metrics": {
            "trust_scores": [],
            "transaction_success_rate": 0.0,
            "malicious_node_impact": 0.0
        }
    }
    
    return results

def run_htdcm_performance_analysis(args):
    """
    Analyze HTDCM performance under various scenarios and configurations.
    
    Args:
        args: Command line arguments
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Starting HTDCM Performance Analysis ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up simulation scenarios - vary based on args
    if args.mode == "attack_comparison":
        # Compare different attack types - reduced number of attacks for debugging
        scenarios = []
        for attack_type in ["no_attack", "sybil", "eclipse"]:
            malicious_pct = 20
            if attack_type == "51_percent":
                malicious_pct = 51
            elif attack_type == "sybil":
                malicious_pct = 30
                
            scenarios.append({
                "num_shards": args.num_shards,
                "nodes_per_shard": args.nodes_per_shard,
                "malicious_percentage": malicious_pct,
                "attack_type": attack_type,
                "simulation_steps": args.steps,
                "tx_per_step": args.tx_per_step
            })
    
        # Run simulations for each scenario
        results = []
        for scenario in scenarios:
            print(f"\nRunning scenario with attack type: {scenario['attack_type']}")
            scenario_result = simulate_network_with_attacks(**scenario)
            results.append(scenario_result)
            
        # Save results
        result_file = os.path.join(output_dir, f"attack_comparison_{timestamp}.npy")
        np.save(result_file, results)
        print(f"Results saved to {result_file}")
            
    elif args.mode == "shard_scaling":
        # Test performance with different numbers of shards
        scenarios = []
        for num_shards in [2, 4, 8, 16]:
            scenarios.append({
                "num_shards": num_shards,
                "nodes_per_shard": args.nodes_per_shard,
                "malicious_percentage": 10,  # Lower percentage for scaling tests
                "attack_type": "no_attack",  # No attacks for scaling tests
                "simulation_steps": args.steps,
                "tx_per_step": args.tx_per_step
            })
            
        # Run simulations for each scenario
        results = []
        for scenario in scenarios:
            print(f"\nRunning scenario with {scenario['num_shards']} shards")
            scenario_result = simulate_network_with_attacks(**scenario)
            results.append(scenario_result)
            
        # Save results
        result_file = os.path.join(output_dir, f"shard_scaling_{timestamp}.npy")
        np.save(result_file, results)
        print(f"Results saved to {result_file}")
    
    else:
        print(f"Unknown mode: {args.mode}")
        return
    
    print("=== HTDCM Performance Analysis Complete ===")

def main():
    """Main function to parse command line arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="HTDCM Performance Testing")
    
    parser.add_argument("--mode", type=str, default="attack_comparison",
                      choices=["attack_comparison", "shard_scaling"],
                      help="Testing mode: attack_comparison or shard_scaling")
    
    parser.add_argument("--num-shards", type=int, default=4,
                      help="Number of shards")
    
    parser.add_argument("--nodes-per-shard", type=int, default=10,
                      help="Number of nodes per shard")
    
    parser.add_argument("--steps", type=int, default=100,
                      help="Number of simulation steps")
    
    parser.add_argument("--tx-per-step", type=int, default=10,
                      help="Number of transactions per step")
    
    parser.add_argument("--output-dir", type=str, default="./results",
                      help="Directory to save results")
    
    args = parser.parse_args()
    
    run_htdcm_performance_analysis(args)

if __name__ == "__main__":
    main() 