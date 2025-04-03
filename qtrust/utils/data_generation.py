"""
Simulation data generation module for the blockchain system.

This module provides functions to generate realistic simulation data for the QTrust blockchain system,
including network topology generation, transaction creation, network events, malicious activities,
and trust scoring. It supports various network models, sharding strategies, and realistic 
transaction patterns to enable thorough testing and evaluation of the system.
"""

import numpy as np
import networkx as nx
import random
from typing import Dict, List, Tuple, Any, Optional

def generate_network_topology(num_nodes: int, 
                             avg_degree: float = 3.0,
                             p_rewire: float = 0.1,
                             seed: Optional[int] = None) -> nx.Graph:
    """
    Generate blockchain network model with Watts-Strogatz (small-world) topology.
    
    Args:
        num_nodes: Number of nodes in the network
        avg_degree: Average degree of each node (number of connections)
        p_rewire: Rewiring probability
        seed: Seed for random generator
        
    Returns:
        nx.Graph: Blockchain network graph
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Calculate k based on avg_degree (k must be even in Watts-Strogatz)
    k = max(2, int(avg_degree))
    if k % 2 == 1:
        k += 1
    
    # Create small-world network
    network = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p_rewire, seed=seed)
    
    # Add attributes to nodes and edges
    for node in network.nodes():
        # Random bandwidth and latency
        network.nodes[node]['bandwidth'] = np.random.uniform(10, 100)  # Mbps
        network.nodes[node]['processing_power'] = np.random.uniform(1, 10)  # Arbitrary units
        network.nodes[node]['storage'] = np.random.uniform(50, 500)  # GB
        network.nodes[node]['trust_score'] = np.random.uniform(0.5, 1.0)  # Initial trust
    
    for u, v in network.edges():
        # Random latency between nodes
        latency = np.random.uniform(5, 50)  # ms
        bandwidth = min(network.nodes[u]['bandwidth'], network.nodes[v]['bandwidth'])
        network[u][v]['latency'] = latency
        network[u][v]['bandwidth'] = bandwidth * 0.8  # Reduce bandwidth by 20% due to overhead
    
    return network

def assign_nodes_to_shards(network: nx.Graph, 
                          num_shards: int,
                          shard_method: str = 'random') -> List[List[int]]:
    """
    Divide nodes into shards.
    
    Args:
        network: Blockchain network graph
        num_shards: Number of shards to create
        shard_method: Division method ('random', 'balanced', 'spectral')
        
    Returns:
        List[List[int]]: List of shards and nodes in each shard
    """
    num_nodes = network.number_of_nodes()
    shards = [[] for _ in range(num_shards)]
    
    if shard_method == 'random':
        # Random allocation
        nodes = list(network.nodes())
        random.shuffle(nodes)
        
        for idx, node in enumerate(nodes):
            shard_id = idx % num_shards
            shards[shard_id].append(node)
            
    elif shard_method == 'balanced':
        # Balanced allocation based on processing power
        nodes = list(network.nodes())
        # Sort nodes by processing power (high to low)
        nodes.sort(key=lambda x: network.nodes[x]['processing_power'], reverse=True)
        
        # Allocate using round-robin for balance
        processing_power = [0] * num_shards
        
        for node in nodes:
            # Find shard with lowest total processing power
            min_power_shard = processing_power.index(min(processing_power))
            shards[min_power_shard].append(node)
            processing_power[min_power_shard] += network.nodes[node]['processing_power']
            
    elif shard_method == 'spectral':
        # Use spectral clustering to create closely connected shards
        try:
            import sklearn.cluster as cluster
            
            # Create adjacency matrix
            adj_matrix = nx.to_numpy_array(network)
            
            # Apply spectral clustering
            spectral = cluster.SpectralClustering(n_clusters=num_shards, 
                                                affinity='precomputed',
                                                random_state=0)
            spectral.fit(adj_matrix)
            
            # Assign nodes to shards based on clustering results
            for node, label in enumerate(spectral.labels_):
                shards[label].append(node)
        except ImportError:
            print("sklearn is not installed, using random sharding method")
            return assign_nodes_to_shards(network, num_shards, 'random')
    else:
        raise ValueError(f"Invalid sharding method: {shard_method}")
    
    return shards

def generate_transactions(num_transactions: int, 
                        num_nodes: int,
                        shards: List[List[int]],
                        value_range: Tuple[float, float] = (0.1, 100.0),
                        cross_shard_prob: float = 0.3,
                        seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Generate random transaction data.
    
    Args:
        num_transactions: Number of transactions to generate
        num_nodes: Total number of nodes in the network
        shards: List of shards and nodes in each shard
        value_range: Transaction value range (min, max)
        cross_shard_prob: Probability of cross-shard transactions
        seed: Seed for random generator
        
    Returns:
        List[Dict[str, Any]]: List of transactions
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    transactions = []
    
    # Map node to shard
    node_to_shard = {}
    for shard_id, nodes in enumerate(shards):
        for node in nodes:
            node_to_shard[node] = shard_id
    
    for i in range(num_transactions):
        # Create new transaction
        transaction = {
            'id': i,
            'timestamp': np.random.uniform(0, 1000),  # Relative time
            'value': np.random.uniform(value_range[0], value_range[1]),
            'gas_price': np.random.uniform(1, 10),
            'size': np.random.uniform(0.5, 2.0),  # KB
        }
        
        # Determine source shard and node
        source_shard_id = np.random.randint(0, len(shards))
        source_node = random.choice(shards[source_shard_id])
        transaction['source'] = source_node
        
        # Determine destination node
        is_cross_shard = random.random() < cross_shard_prob
        
        if is_cross_shard and len(shards) > 1:
            # Choose different shard from source
            dest_shard_candidates = [s for s in range(len(shards)) if s != source_shard_id]
            dest_shard_id = random.choice(dest_shard_candidates)
        else:
            # Same shard as source
            dest_shard_id = source_shard_id
        
        # Choose destination node from destination shard
        dest_node = random.choice(shards[dest_shard_id])
        while dest_node == source_node:  # Avoid self-transactions
            dest_node = random.choice(shards[dest_shard_id])
            
        transaction['destination'] = dest_node
        transaction['cross_shard'] = source_shard_id != dest_shard_id
        
        # Add to list
        transactions.append(transaction)
    
    # Sort by timestamp
    transactions.sort(key=lambda x: x['timestamp'])
    
    return transactions

def generate_network_events(num_events: int, 
                          num_nodes: int,
                          duration: float = 1000.0,
                          seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Generate network events such as node failure, network congestion, latency spikes.
    
    Args:
        num_events: Number of events to generate
        num_nodes: Total number of nodes in the network
        duration: Simulation time (relative units)
        seed: Seed for random generator
        
    Returns:
        List[Dict[str, Any]]: List of network events
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    events = []
    event_types = ['node_failure', 'congestion', 'latency_spike', 'bandwidth_drop']
    
    for i in range(num_events):
        # Event time
        timestamp = np.random.uniform(0, duration)
        
        # Event type
        event_type = random.choice(event_types)
        
        # Affected node or edge
        affected_node = np.random.randint(0, num_nodes)
        
        # Create event
        event = {
            'id': i,
            'timestamp': timestamp,
            'type': event_type,
            'affected_node': affected_node,
            'duration': np.random.uniform(10, 100),  # Event duration
        }
        
        # Add details based on event type
        if event_type == 'node_failure':
            event['severity'] = np.random.uniform(0.7, 1.0)  # Severity level
            
        elif event_type == 'congestion':
            event['severity'] = np.random.uniform(0.3, 0.9)
            event['affected_links'] = []
            
            # Affect some related links
            num_affected_links = np.random.randint(1, 5)
            for _ in range(num_affected_links):
                target_node = np.random.randint(0, num_nodes)
                while target_node == affected_node:
                    target_node = np.random.randint(0, num_nodes)
                event['affected_links'].append((affected_node, target_node))
                
        elif event_type == 'latency_spike':
            event['multiplier'] = np.random.uniform(1.5, 5.0)  # Latency increase factor
            
        elif event_type == 'bandwidth_drop':
            event['reduction'] = np.random.uniform(0.3, 0.8)  # % bandwidth reduction
        
        events.append(event)
    
    # Sort by timestamp
    events.sort(key=lambda x: x['timestamp'])
    
    return events

def generate_malicious_activities(num_activities: int,
                                 num_nodes: int,
                                 shards: List[List[int]],
                                 honest_node_prob: float = 0.9,
                                 seed: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[int, bool]]:
    """
    Generate malicious activities and identify dishonest nodes.
    
    Args:
        num_activities: Number of malicious activities
        num_nodes: Total number of nodes in the network
        shards: List of shards and nodes in each shard
        honest_node_prob: Probability that a node is honest
        seed: Seed for random generator
        
    Returns:
        Tuple[List[Dict[str, Any]], Dict[int, bool]]: 
            - List of malicious activities
            - Dict mapping node ID to honesty status (True if honest)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Determine honest/dishonest nodes
    is_honest = {}
    for node in range(num_nodes):
        is_honest[node] = random.random() < honest_node_prob
    
    # List of dishonest nodes
    dishonest_nodes = [node for node, honest in is_honest.items() if not honest]
    
    if not dishonest_nodes:
        # If no dishonest nodes, create one malicious node
        dishonest_node = random.randint(0, num_nodes - 1)
        dishonest_nodes = [dishonest_node]
        is_honest[dishonest_node] = False
    
    # Generate malicious activities
    activities = []
    activity_types = [
        'double_spending', 'transaction_withholding', 'block_withholding',
        'sybil_attack', 'selfish_mining', 'eclipse_attack'
    ]
    
    for i in range(num_activities):
        # Choose activity type
        activity_type = random.choice(activity_types)
        
        # Choose malicious node to perform activity
        malicious_node = random.choice(dishonest_nodes)
        
        # Create activity
        activity = {
            'id': i,
            'type': activity_type,
            'node': malicious_node,
            'timestamp': np.random.uniform(0, 1000),
            'severity': np.random.uniform(0.3, 1.0),
        }
        
        # Add details based on activity type
        if activity_type == 'double_spending':
            activity['target_shard'] = random.randint(0, len(shards) - 1)
            activity['amount'] = np.random.uniform(10, 100)
            
        elif activity_type == 'transaction_withholding':
            activity['num_transactions'] = random.randint(1, 10)
            
        elif activity_type == 'block_withholding':
            activity['duration'] = np.random.uniform(10, 50)
            
        elif activity_type == 'sybil_attack':
            activity['fake_identities'] = random.randint(2, 5)
            
        elif activity_type == 'selfish_mining':
            activity['private_blocks'] = random.randint(1, 3)
            
        elif activity_type == 'eclipse_attack':
            victim_candidates = [node for node, honest in is_honest.items() if honest]
            if victim_candidates:
                activity['victim'] = random.choice(victim_candidates)
            else:
                activity['victim'] = random.randint(0, num_nodes - 1)
        
        activities.append(activity)
    
    # Sort by timestamp
    activities.sort(key=lambda x: x['timestamp'])
    
    return activities, is_honest

def assign_trust_scores(num_nodes: int, 
                       is_honest: Dict[int, bool],
                       base_honest_score: float = 0.8,
                       honest_variance: float = 0.1,
                       base_dishonest_score: float = 0.4,
                       dishonest_variance: float = 0.2,
                       seed: Optional[int] = None) -> Dict[int, float]:
    """
    Initialize trust scores for nodes based on honesty status.
    
    Args:
        num_nodes: Number of nodes in the network
        is_honest: Dictionary mapping node ID to honesty status
        base_honest_score: Base score for honest nodes
        honest_variance: Variance for honest node base score
        base_dishonest_score: Base score for dishonest nodes
        dishonest_variance: Variance for dishonest node base score
        seed: Seed for random generator
        
    Returns:
        Dict[int, float]: Dictionary mapping node ID to trust score
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    trust_scores = {}
    
    for node in range(num_nodes):
        if is_honest.get(node, True):
            # Honest node
            score = base_honest_score + np.random.normal(0, honest_variance)
            # Limit to range [0.5, 1.0]
            score = min(1.0, max(0.5, score))
        else:
            # Dishonest node
            score = base_dishonest_score + np.random.normal(0, dishonest_variance)
            # Limit to range [0.1, 0.7]
            score = min(0.7, max(0.1, score))
        
        trust_scores[node] = score
    
    return trust_scores 