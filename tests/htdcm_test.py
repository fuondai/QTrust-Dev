#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HTDCM Testing Module

This module provides functionality for testing the malicious node detection capabilities 
of the Hierarchical Trust-based Distributed Consensus Mechanism (HTDCM).
It simulates a blockchain network with both normal and malicious nodes,
tracks trust scores over time, and evaluates detection accuracy metrics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any
import time

# Add the current directory to PYTHONPATH to import other modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from qtrust.trust.htdcm import HTDCM, HTDCMNode

def test_htdcm_malicious_detection():
    """
    Test the malicious node detection capability of HTDCM.
    """
    print("=== Starting HTDCM Testing ===")
    
    # Simple network configuration
    num_shards = 3
    nodes_per_shard = 8
    total_nodes = num_shards * nodes_per_shard
    malicious_percentage = 25  # 25% malicious nodes
    
    # Create network graph
    G = nx.Graph()
    
    # Create list of shards and nodes in each shard
    shards = []
    malicious_nodes = []
    
    print(f"Creating network with {num_shards} shards, each shard has {nodes_per_shard} nodes")
    
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
                      is_malicious=is_malicious)
            
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
    
    # Simulate transactions and detect malicious nodes
    sim_steps = 50
    tx_per_step = 10
    detected_history = []
    trust_score_history = {node_id: [] for node_id in G.nodes()}
    
    print(f"Simulating {sim_steps} steps, each step has {tx_per_step} transactions")
    
    for step in range(sim_steps):
        # Generate transactions
        for _ in range(tx_per_step):
            # Randomly select source and target
            source_shard = np.random.randint(0, num_shards)
            target_shard = np.random.randint(0, num_shards)
            
            source_node = np.random.choice(shards[source_shard])
            target_node = np.random.choice(shards[target_shard])
            
            # Determine if transaction is successful
            # Malicious nodes have higher failure probability
            source_is_malicious = G.nodes[source_node].get('is_malicious', False)
            target_is_malicious = G.nodes[target_node].get('is_malicious', False)
            
            # Success probability based on malicious status
            if source_is_malicious or target_is_malicious:
                success_prob = 0.3  # Malicious nodes have low success probability
            else:
                success_prob = 0.9  # Normal nodes have high success probability
            
            # Determine transaction result
            tx_success = np.random.rand() < success_prob
            
            # Simulate response time
            if source_is_malicious or target_is_malicious:
                # Malicious nodes have variable response time
                response_time = np.random.uniform(15, 30)
            else:
                # Normal nodes have stable response time
                response_time = np.random.uniform(5, 15)
            
            # Update trust information for nodes
            htdcm.update_node_trust(source_node, tx_success, response_time, True)
            htdcm.update_node_trust(target_node, tx_success, response_time, False)
        
        # Save trust scores
        for node_id in G.nodes():
            trust_score_history[node_id].append(htdcm.nodes[node_id].trust_score)
        
        # Detect malicious nodes
        detected_nodes = htdcm.identify_malicious_nodes(min_malicious_activities=1, advanced_filtering=True)
        detected_history.append(len(detected_nodes))
        
        # Report progress
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step + 1}/{sim_steps}: {len(detected_nodes)} malicious nodes detected")
            
            # Calculate performance metrics
            true_malicious = set(malicious_nodes)
            detected_malicious = set(detected_nodes)
            
            true_positives = len(true_malicious.intersection(detected_malicious))
            false_positives = len(detected_malicious - true_malicious)
            false_negatives = len(true_malicious - detected_malicious)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f}")
    
    # Final analysis
    final_detected = htdcm.identify_malicious_nodes(min_malicious_activities=2, advanced_filtering=True)
    true_malicious = set(malicious_nodes)
    detected_malicious = set(final_detected)
    
    # Calculate final performance metrics
    true_positives = len(true_malicious.intersection(detected_malicious))
    false_positives = len(detected_malicious - true_malicious)
    false_negatives = len(true_malicious - detected_malicious)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n=== Final Results ===")
    print(f"Total actual malicious nodes: {len(malicious_nodes)}")
    print(f"Number of detected malicious nodes: {len(final_detected)}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # 1. Chart of detected malicious nodes count
    plt.subplot(2, 2, 1)
    plt.plot(range(sim_steps), detected_history, marker='o')
    plt.axhline(y=len(malicious_nodes), color='r', linestyle='--', label='Actual')
    plt.xlabel('Simulation Step')
    plt.ylabel('Number of Detected Malicious Nodes')
    plt.title('Malicious Node Detection Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Chart of average trust scores for malicious and non-malicious nodes
    malicious_trust = np.mean([np.array(trust_score_history[n]) for n in malicious_nodes], axis=0)
    non_malicious = [n for n in G.nodes() if n not in malicious_nodes]
    non_malicious_trust = np.mean([np.array(trust_score_history[n]) for n in non_malicious], axis=0)
    
    plt.subplot(2, 2, 2)
    plt.plot(range(sim_steps), malicious_trust, 'r-', label='Malicious Nodes')
    plt.plot(range(sim_steps), non_malicious_trust, 'g-', label='Normal Nodes')
    plt.xlabel('Simulation Step')
    plt.ylabel('Average Trust Score')
    plt.title('Average Trust Score Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Histogram of final trust scores
    plt.subplot(2, 2, 3)
    malicious_final_trust = [trust_score_history[n][-1] for n in malicious_nodes]
    non_malicious_final_trust = [trust_score_history[n][-1] for n in non_malicious]
    
    plt.hist([non_malicious_final_trust, malicious_final_trust], bins=10,
            label=['Normal Nodes', 'Malicious Nodes'], alpha=0.7, color=['g', 'r'])
    plt.axvline(x=htdcm.malicious_threshold, color='k', linestyle='--', label='Malicious Threshold')
    plt.xlabel('Trust Score')
    plt.ylabel('Number of Nodes')
    plt.title('Final Trust Score Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Chart of performance metrics over time
    precision_history = []
    recall_history = []
    f1_history = []
    
    for step in range(sim_steps):
        # Get trust scores at this step
        trust_at_step = {node_id: trust_score_history[node_id][step] for node_id in G.nodes()}
        
        # Identify detected nodes at this step (completely replaced section)
        min_activities = 1 if step < sim_steps // 2 else 2
        detected_at_step = []
        
        for node_id, score in trust_at_step.items():
            if score < htdcm.malicious_threshold and node_id in htdcm.nodes:
                node = htdcm.nodes[node_id]
                
                # Advanced checks
                enough_malicious_activities = node.malicious_activities >= min_activities
                
                # Check success rate
                total_txs = node.successful_txs + node.failed_txs
                low_success_rate = False
                if total_txs >= 5:
                    success_rate = node.successful_txs / total_txs if total_txs > 0 else 0
                    low_success_rate = success_rate < 0.4
                
                # Check response time
                high_response_time = False
                if node.response_times and len(node.response_times) >= 3:
                    avg_response_time = np.mean(node.response_times)
                    high_response_time = avg_response_time > 20
                
                # Check peer feedback
                poor_peer_rating = False
                if hasattr(node, 'peer_ratings') and node.peer_ratings:
                    avg_peer_rating = np.mean(list(node.peer_ratings.values()))
                    poor_peer_rating = avg_peer_rating < 0.4
                
                # Calculate evidence score
                evidence_count = sum([
                    enough_malicious_activities,
                    low_success_rate,
                    high_response_time,
                    poor_peer_rating
                ])
                
                if evidence_count >= 2:
                    detected_at_step.append(node_id)
        
        # Calculate metrics
        true_positives = len(set(malicious_nodes).intersection(set(detected_at_step)))
        false_positives = len(set(detected_at_step) - set(malicious_nodes))
        false_negatives = len(set(malicious_nodes) - set(detected_at_step))
        
        p = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        r = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precision_history.append(p)
        recall_history.append(r)
        f1_history.append(f1)
    
    plt.subplot(2, 2, 4)
    plt.plot(range(sim_steps), precision_history, 'b-', label='Precision')
    plt.plot(range(sim_steps), recall_history, 'g-', label='Recall')
    plt.plot(range(sim_steps), f1_history, 'r-', label='F1 Score')
    plt.xlabel('Simulation Step')
    plt.ylabel('Metric')
    plt.title('Performance Metrics Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('htdcm_test_results.png', dpi=300)
    plt.show()
    
    print(f"Saved results chart to htdcm_test_results.png")

if __name__ == "__main__":
    test_htdcm_malicious_detection() 