"""
Test file for the QTrust data generation module.
"""

import os
import sys
import numpy as np
import networkx as nx
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import time

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrust.utils.data_generation import (
    generate_network_topology,
    assign_nodes_to_shards,
    generate_transactions,
    generate_network_events,
    generate_malicious_activities,
    assign_trust_scores
)
from qtrust.utils.paths import get_chart_path

def test_network_topology_generation():
    """Test network topology generation."""
    print("\nTesting network topology generation...")
    
    # Generate a small network for testing
    num_nodes = 20
    network = generate_network_topology(
        num_nodes=num_nodes,
        avg_degree=4.0,
        p_rewire=0.1,
        seed=42
    )
    
    # Verify the network properties
    assert isinstance(network, nx.Graph)
    assert network.number_of_nodes() == num_nodes
    
    # Check that all nodes have attributes
    for node in network.nodes():
        assert 'bandwidth' in network.nodes[node]
        assert 'processing_power' in network.nodes[node]
        assert 'storage' in network.nodes[node]
        assert 'trust_score' in network.nodes[node]
    
    # Check that all edges have attributes
    for u, v in network.edges():
        assert 'latency' in network[u][v]
        assert 'bandwidth' in network[u][v]
    
    print(f"✓ Network created with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
    
    # Optional: Visualize the network
    try:
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(network, seed=42)
        nx.draw(network, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
        plt.title("Generated Blockchain Network")
        chart_path = get_chart_path("network_topology.png", "test")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"✓ Network visualization saved to {chart_path}")
    except Exception as e:
        print(f"Could not visualize network: {e}")
    
    return network

def test_shard_assignment(network: nx.Graph):
    """Test shard assignment functionality."""
    print("\nTesting shard assignment...")
    
    num_shards = 4
    
    # Test random sharding
    random_shards = assign_nodes_to_shards(
        network=network,
        num_shards=num_shards,
        shard_method='random'
    )
    
    assert len(random_shards) == num_shards
    # Check that all nodes are assigned to exactly one shard
    all_nodes = []
    for shard in random_shards:
        all_nodes.extend(shard)
    assert len(all_nodes) == network.number_of_nodes()
    assert len(set(all_nodes)) == network.number_of_nodes()  # No duplicates
    
    print(f"✓ Random sharding: {[len(shard) for shard in random_shards]} nodes per shard")
    
    # Test balanced sharding
    balanced_shards = assign_nodes_to_shards(
        network=network,
        num_shards=num_shards,
        shard_method='balanced'
    )
    
    assert len(balanced_shards) == num_shards
    # Check that all nodes are assigned
    all_balanced_nodes = []
    for shard in balanced_shards:
        all_balanced_nodes.extend(shard)
    assert len(all_balanced_nodes) == network.number_of_nodes()
    
    print(f"✓ Balanced sharding: {[len(shard) for shard in balanced_shards]} nodes per shard")
    
    return balanced_shards

def test_transaction_generation(shards: List[List[int]]):
    """Test transaction generation."""
    print("\nTesting transaction generation...")
    
    num_transactions = 50
    num_nodes = sum(len(shard) for shard in shards)
    
    transactions = generate_transactions(
        num_transactions=num_transactions,
        num_nodes=num_nodes,
        shards=shards,
        value_range=(1.0, 50.0),
        cross_shard_prob=0.3,
        seed=42
    )
    
    assert len(transactions) == num_transactions
    
    # Check transaction attributes
    for tx in transactions:
        assert 'id' in tx
        assert 'timestamp' in tx
        assert 'value' in tx
        assert 'source' in tx
        assert 'destination' in tx
        assert 'cross_shard' in tx
    
    # Count cross-shard transactions
    cross_shard_count = sum(1 for tx in transactions if tx['cross_shard'])
    cross_shard_percentage = (cross_shard_count / num_transactions) * 100
    
    print(f"✓ Generated {num_transactions} transactions")
    print(f"✓ {cross_shard_count} cross-shard transactions ({cross_shard_percentage:.1f}%)")
    
    return transactions

def test_network_events_generation(num_nodes: int):
    """Test network events generation."""
    print("\nTesting network events generation...")
    
    num_events = 30
    
    events = generate_network_events(
        num_events=num_events,
        num_nodes=num_nodes,
        duration=1000.0,
        seed=42
    )
    
    assert len(events) == num_events
    
    # Check event attributes
    for event in events:
        assert 'id' in event
        assert 'timestamp' in event
        assert 'type' in event
        assert 'affected_node' in event
        assert 'duration' in event
    
    # Count events by type
    event_types = {}
    for event in events:
        event_type = event['type']
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    print(f"✓ Generated {num_events} network events")
    print(f"✓ Event types distribution: {event_types}")
    
    return events

def test_malicious_activities_generation(num_nodes: int, shards: List[List[int]]):
    """Test malicious activities generation."""
    print("\nTesting malicious activities generation...")
    
    num_activities = 20
    
    activities, is_honest = generate_malicious_activities(
        num_activities=num_activities,
        num_nodes=num_nodes,
        shards=shards,
        honest_node_prob=0.8,
        seed=42
    )
    
    assert len(activities) == num_activities
    assert len(is_honest) == num_nodes
    
    # Check activity attributes
    for activity in activities:
        assert 'id' in activity
        assert 'type' in activity
        assert 'node' in activity
        assert 'timestamp' in activity
        assert 'severity' in activity
    
    # Count dishonest nodes
    dishonest_count = sum(1 for node, honest in is_honest.items() if not honest)
    dishonest_percentage = (dishonest_count / num_nodes) * 100
    
    print(f"✓ Generated {num_activities} malicious activities")
    print(f"✓ {dishonest_count} dishonest nodes ({dishonest_percentage:.1f}%)")
    
    return activities, is_honest

def test_trust_score_assignment(num_nodes: int, is_honest: Dict[int, bool]):
    """Test trust score assignment."""
    print("\nTesting trust score assignment...")
    
    trust_scores = assign_trust_scores(
        num_nodes=num_nodes,
        is_honest=is_honest,
        base_honest_score=0.8,
        honest_variance=0.1,
        base_dishonest_score=0.4,
        dishonest_variance=0.2,
        seed=42
    )
    
    assert len(trust_scores) == num_nodes
    
    # Check trust score ranges
    for node, score in trust_scores.items():
        assert 0.1 <= score <= 1.0
        if is_honest[node]:
            assert 0.5 <= score <= 1.0
        else:
            assert 0.1 <= score <= 0.7
    
    honest_scores = [score for node, score in trust_scores.items() if is_honest[node]]
    dishonest_scores = [score for node, score in trust_scores.items() if not is_honest[node]]
    
    print(f"✓ Assigned trust scores to {num_nodes} nodes")
    if honest_scores:
        print(f"✓ Honest nodes avg trust: {sum(honest_scores) / len(honest_scores):.2f}")
    if dishonest_scores:
        print(f"✓ Dishonest nodes avg trust: {sum(dishonest_scores) / len(dishonest_scores):.2f}")
    
    return trust_scores

def run_all_tests():
    """Run all tests for the data generation module."""
    print("Running data generation module tests...")
    
    # Chain the tests to reuse outputs
    network = test_network_topology_generation()
    shards = test_shard_assignment(network)
    num_nodes = network.number_of_nodes()
    
    transactions = test_transaction_generation(shards)
    events = test_network_events_generation(num_nodes)
    activities, is_honest = test_malicious_activities_generation(num_nodes, shards)
    trust_scores = test_trust_score_assignment(num_nodes, is_honest)
    
    print("\n✓ All data generation tests passed!")

if __name__ == "__main__":
    run_all_tests() 