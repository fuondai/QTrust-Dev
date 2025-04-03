"""
Large Scale Blockchain Simulation

This module simulates a large-scale blockchain network with multiple shards, nodes, and transactions.
It focuses on performance analysis, security under various attack scenarios, and optimization of
cross-shard transactions processing. The simulation includes detailed tracking of metrics like
throughput, latency, energy consumption, and security indicators.

Key components:
- Node: Represents validators with trust scores and processing capabilities
- Transaction: Models transactions with routing and resource consumption metrics
- TransactionPipeline: Optimizes transaction processing through a multistage pipeline
- Shard: Manages groups of nodes and transaction queues with congestion modeling
- LargeScaleBlockchainSimulation: Orchestrates the entire simulation environment
"""

import os
import sys
import time
import random
import argparse
import multiprocessing
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# Configure encoding for output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add current directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

class Node:
    __slots__ = ('node_id', 'shard_id', 'is_malicious', 'attack_type', 'trust_score', 
                'processing_power', 'connections', 'transactions_processed', 'uptime',
                'energy_efficiency', 'last_active', 'reputation_history')
    
    def __init__(self, node_id, shard_id, is_malicious=False, attack_type=None):
        self.node_id = node_id
        self.shard_id = shard_id
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.trust_score = 1.0
        self.processing_power = random.uniform(0.8, 1.2)
        self.connections = []
        self.transactions_processed = 0
        self.uptime = 100.0  # Uptime percentage (%)
        self.energy_efficiency = random.uniform(0.7, 1.0)  # Energy efficiency
        self.last_active = time.time()
        self.reputation_history = []
    
    def __str__(self):
        return f"Node {self.node_id} (Shard {self.shard_id})"
    
    def update_trust_score(self, success_rate):
        """Update trust score based on success rate."""
        self.trust_score = 0.9 * self.trust_score + 0.1 * success_rate
        self.reputation_history.append(self.trust_score)
        return self.trust_score

class Transaction:
    __slots__ = ('tx_id', 'source_shard', 'target_shard', 'size', 'is_cross_shard', 
                'route', 'hops', 'latency', 'energy', 'is_processed', 'timestamp',
                'priority', 'data_integrity', 'processing_attempts', 'completion_time',
                'resource_cost')
    
    def __init__(self, tx_id, source_shard, target_shard, size=1.0):
        self.tx_id = tx_id
        self.source_shard = source_shard
        self.target_shard = target_shard
        self.size = size
        self.is_cross_shard = source_shard != target_shard
        self.route = []
        self.hops = 0
        self.latency = 0
        self.energy = 0
        self.is_processed = False
        self.timestamp = time.time()
        self.priority = random.uniform(0, 1)  # Priority of the transaction
        self.data_integrity = 1.0  # Data integrity
        self.processing_attempts = 0  # Number of processing attempts
        self.completion_time = None
        self.resource_cost = 0.0  # Resource cost for processing
        
    def is_cross_shard_tx(self):
        return self.is_cross_shard
    
    def calculate_resource_cost(self):
        """Calculate resource cost based on size and number of hops."""
        base_cost = self.size * 0.5
        hop_factor = 1.0 + (self.hops * 0.2)
        self.resource_cost = base_cost * hop_factor
        return self.resource_cost
    
    def mark_completed(self):
        """Mark transaction as completed and record time."""
        self.is_processed = True
        self.completion_time = time.time()
        self.calculate_resource_cost()

class TransactionPipeline:
    """Class for processing transactions in a pipeline, optimizing transaction processing."""
    
    __slots__ = ('shards', 'max_workers', 'optimal_workers', 'pipeline_metrics',
                'processing_cache', 'validation_result_cache', 'routing_result_cache')
    
    def __init__(self, shards, max_workers=None):
        self.shards = shards
        self.max_workers = max_workers
        # Determine optimal number of workers
        self.optimal_workers = min(
            multiprocessing.cpu_count() * 2,  # 2x number of CPU cores
            32  # Maximum limit
        ) if max_workers is None else max_workers
        
        # Count number of transactions processed through each stage
        self.pipeline_metrics = {
            'validation': 0,
            'routing': 0,
            'consensus': 0,
            'execution': 0,
            'commit': 0
        }
        
        # Cache for speeding up processing
        self.processing_cache = {}  # Cache for processing results
        self.validation_result_cache = {}  # Cache for validation results
        self.routing_result_cache = {}  # Cache for routing results
    
    def validate_transaction(self, tx):
        """Stage 1: Check transaction validity."""
        # Check cache first
        if tx.tx_id in self.validation_result_cache:
            return self.validation_result_cache[tx.tx_id]
            
        # Simulate transaction validation
        self.pipeline_metrics['validation'] += 1
        
        # Check if source and target shard are valid
        if tx.source_shard < 0 or tx.source_shard >= len(self.shards) or \
           tx.target_shard < 0 or tx.target_shard >= len(self.shards):
            self.validation_result_cache[tx.tx_id] = False
            return False
        
        # Check transaction size
        if tx.size <= 0 or tx.size > 10:  # Size limit from 0-10
            self.validation_result_cache[tx.tx_id] = False
            return False
        
        # Save result in cache
        self.validation_result_cache[tx.tx_id] = True
        return True
    
    def route_transaction(self, tx):
        """Stage 2: Calculate transaction route."""
        # Check cache
        cache_key = f"{tx.tx_id}_{tx.source_shard}_{tx.target_shard}"
        if cache_key in self.routing_result_cache:
            cached_result = self.routing_result_cache[cache_key]
            tx.route = cached_result['route']
            tx.hops = cached_result['hops']
            tx.latency = cached_result['latency']
            return True
            
        self.pipeline_metrics['routing'] += 1
        
        if tx.is_cross_shard:
            # Calculate number of hops
            tx.hops = max(1, abs(tx.target_shard - tx.source_shard))
            
            # Create route
            tx.route = self._generate_route(tx.source_shard, tx.target_shard)
            
            # Calculate latency based on route and congestion
            route_congestion = [self.shards[shard].congestion_level for shard in tx.route]
            avg_congestion = sum(route_congestion) / len(route_congestion) if route_congestion else 0
            tx.latency = (10 + (5 * tx.hops)) * (1 + avg_congestion * 2)
        else:
            # Transaction within the same shard
            tx.hops = 1
            tx.route = [tx.source_shard]
            tx.latency = 5 * (1 + self.shards[tx.source_shard].congestion_level)
        
        # Save result in cache
        self.routing_result_cache[cache_key] = {
            'route': tx.route,
            'hops': tx.hops,
            'latency': tx.latency
        }
        
        return True
    
    def _generate_route(self, source, target):
        """Create optimal route between shards."""
        # Cache key for route
        cache_key = f"route_{source}_{target}"
        if cache_key in self.processing_cache:
            return self.processing_cache[cache_key]
            
        if source == target:
            return [source]
        
        route = [source]
        current = source
        
        # Find shortest path (simplified)
        while current != target:
            if current < target:
                current += 1
            else:
                current -= 1
            route.append(current)
        
        # Save in cache
        self.processing_cache[cache_key] = route
        return route
    
    def reach_consensus(self, tx):
        """Stage 3: Reach consensus among nodes in the shard."""
        self.pipeline_metrics['consensus'] += 1
        
        source_shard = self.shards[tx.source_shard]
        target_shard = self.shards[tx.target_shard]
        
        # Calculate malicious node ratio
        malicious_node_ratio = (
            len(source_shard.get_malicious_nodes()) / max(1, len(source_shard.nodes)) +
            len(target_shard.get_malicious_nodes()) / max(1, len(target_shard.nodes))
        ) / 2
        
        # Probability of reaching consensus based on malicious node ratio
        consensus_probability = 1.0 - (malicious_node_ratio * 0.8)
        
        # Check if consensus is reached
        return random.random() < consensus_probability
    
    def execute_transaction(self, tx):
        """Stage 4: Execute transaction."""
        self.pipeline_metrics['execution'] += 1
        
        # Cache key for energy
        cache_key = f"energy_{tx.tx_id}_{tx.is_cross_shard}_{tx.size}_{tx.hops}"
        if cache_key in self.processing_cache:
            tx.energy = self.processing_cache[cache_key]
            return True
        
        # Calculate energy consumption
        if tx.is_cross_shard:
            base_energy = 2.0 * tx.size
            hop_energy = 0.5 * tx.hops
            tx.energy = base_energy + hop_energy
        else:
            tx.energy = 1.0 * tx.size
        
        # Save in cache
        self.processing_cache[cache_key] = tx.energy
        
        # Check if transaction is successful
        success_probability = 0.98  # High success rate at this stage
        return random.random() < success_probability
    
    def commit_transaction(self, tx):
        """Stage 5: Commit transaction."""
        self.pipeline_metrics['commit'] += 1
        
        # Mark transaction as completed
        tx.mark_completed()
        
        # Calculate data integrity
        source_shard = self.shards[tx.source_shard]
        target_shard = self.shards[tx.target_shard]
        
        malicious_node_ratio = (
            len(source_shard.get_malicious_nodes()) / max(1, len(source_shard.nodes)) +
            len(target_shard.get_malicious_nodes()) / max(1, len(target_shard.nodes))
        ) / 2
        
        if malicious_node_ratio > 0:
            tx.data_integrity = max(0.7, 1.0 - (malicious_node_ratio * 0.5))
        
        return True
    
    def process_transaction(self, tx, sim_context=None):
        """Process transaction through the entire pipeline."""
        # Cache key for processing result
        cache_key = f"process_{tx.tx_id}_{tx.processing_attempts}"
        if cache_key in self.processing_cache:
            return self.processing_cache[cache_key]
            
        # Stage 1: Check transaction validity
        if not self.validate_transaction(tx):
            result = (tx, False)
            self.processing_cache[cache_key] = result
            return result
        
        # Stage 2: Calculate route
        if not self.route_transaction(tx):
            result = (tx, False)
            self.processing_cache[cache_key] = result
            return result
        
        # Stage 3: Reach consensus
        if not self.reach_consensus(tx):
            tx.processing_attempts += 1
            result = (tx, False)
            self.processing_cache[cache_key] = result
            return result
        
        # Stage 4: Execute transaction
        if not self.execute_transaction(tx):
            tx.processing_attempts += 1
            result = (tx, False)
            self.processing_cache[cache_key] = result
            return result
        
        # Stage 5: Commit transaction
        self.commit_transaction(tx)
        
        result = (tx, True)
        self.processing_cache[cache_key] = result
        return result
    
    def process_transactions_batch(self, transactions, sim_context=None):
        """Process transactions in parallel through the pipeline."""
        results = []
        
        # Group transactions by complexity for more efficient processing
        simple_transactions = [tx for tx in transactions if not tx.is_cross_shard]
        complex_transactions = [tx for tx in transactions if tx.is_cross_shard]
        
        with ThreadPoolExecutor(max_workers=self.optimal_workers) as executor:
            # Process simple transactions with higher priority
            simple_futures = [executor.submit(self.process_transaction, tx, sim_context) for tx in simple_transactions]
            # Process complex transactions
            complex_futures = [executor.submit(self.process_transaction, tx, sim_context) for tx in complex_transactions]
            
            # Collect results from simple transactions
            for future in as_completed(simple_futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in pipeline processing (simple transaction): {e}")
            
            # Collect results from complex transactions
            for future in as_completed(complex_futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in pipeline processing (complex transaction): {e}")
        
        # Clear cache to prevent memory overflow
        if len(self.processing_cache) > 10000:
            self.processing_cache = {}
        if len(self.validation_result_cache) > 10000:
            self.validation_result_cache = {}
        if len(self.routing_result_cache) > 10000:
            self.routing_result_cache = {}
            
        return results
    
    def process_shard_transactions(self, shard_transactions, sim_context=None):
        """Process transactions within each shard for locality optimization."""
        all_results = []
        
        # Create list of all transactions to process
        all_transactions = []
        for shard_id, transactions in shard_transactions.items():
            if transactions:
                all_transactions.extend(transactions)
        
        # If there are few transactions, process them at once
        if len(all_transactions) <= self.optimal_workers * 2:
            return self.process_transactions_batch(all_transactions, sim_context)
        
        # If there are many transactions, process them shard by shard for locality
        with ThreadPoolExecutor(max_workers=min(len(shard_transactions), 8)) as executor:
            # Submit tasks for processing each shard
            future_to_shard = {}
            for shard_id, transactions in shard_transactions.items():
                if transactions:
                    future = executor.submit(self.process_transactions_batch, transactions, sim_context)
                    future_to_shard[future] = shard_id
            
            # Collect results
            for future in as_completed(future_to_shard):
                try:
                    shard_results = future.result()
                    all_results.extend(shard_results)
                except Exception as e:
                    shard_id = future_to_shard[future]
                    print(f"Error processing transactions for shard {shard_id}: {e}")
        
        return all_results
    
    def get_pipeline_metrics(self):
        """Return pipeline statistics."""
        return self.pipeline_metrics.copy()
    
    def clear_caches(self):
        """Clear all caches to free up memory."""
        self.processing_cache.clear()
        self.validation_result_cache.clear()
        self.routing_result_cache.clear()
        return True

class Shard:
    __slots__ = ('shard_id', 'nodes', 'congestion_level', 'transactions_queue', 'processed_transactions',
                'blocked_transactions', 'network_stability', 'resource_utilization', 'consensus_difficulty',
                'last_metrics_update', 'historical_congestion', '_non_malicious_nodes_cache', '_malicious_nodes_cache',
                '_last_cache_update')
    
    def __init__(self, shard_id, num_nodes, malicious_percentage=0, attack_types=None):
        self.shard_id = shard_id
        self.nodes = []
        self.congestion_level = 0.0
        self.transactions_queue = []
        self.processed_transactions = []
        self.blocked_transactions = []  # Transactions blocked
        self.network_stability = 1.0  # Network stability
        self.resource_utilization = 0.0  # Resource utilization
        self.consensus_difficulty = random.uniform(0.5, 1.5)  # Difficulty to reach consensus
        self.last_metrics_update = time.time()
        self.historical_congestion = []  # Historical congestion level
        
        # Cache for optimizing queries
        self._non_malicious_nodes_cache = None
        self._malicious_nodes_cache = None
        self._last_cache_update = 0
        
        # Calculate number of malicious nodes
        num_malicious = int(num_nodes * malicious_percentage / 100)
        
        # Create list of attack types if specified
        if attack_types is None:
            attack_types = []
        
        # Create nodes
        for i in range(num_nodes):
            is_malicious = i < num_malicious
            attack_type = None
            if is_malicious and attack_types:
                attack_type = random.choice(attack_types)
            
            node = Node(
                node_id=f"{shard_id}_{i}", 
                shard_id=shard_id,
                is_malicious=is_malicious,
                attack_type=attack_type
            )
            self.nodes.append(node)
    
    def get_non_malicious_nodes(self):
        """Return list of non-malicious nodes, using cache for optimization."""
        # Check if cache is valid
        current_time = time.time()
        if self._non_malicious_nodes_cache is None or current_time - self._last_cache_update > 1.0:
            # Update cache if expired or not created yet
            self._non_malicious_nodes_cache = [node for node in self.nodes if not node.is_malicious]
            self._last_cache_update = current_time
        return self._non_malicious_nodes_cache
    
    def get_malicious_nodes(self):
        """Return list of malicious nodes, using cache for optimization."""
        # Check if cache is valid
        current_time = time.time()
        if self._malicious_nodes_cache is None or current_time - self._last_cache_update > 1.0:
            # Update cache if expired or not created yet
            self._malicious_nodes_cache = [node for node in self.nodes if node.is_malicious]
            self._last_cache_update = current_time
        return self._malicious_nodes_cache
    
    def compute_power_distribution(self):
        """Calculate power distribution among nodes."""
        total_power = sum(node.processing_power for node in self.nodes)
        return [(node.node_id, node.processing_power / total_power) for node in self.nodes]

    def update_congestion(self):
        """Update congestion level of the shard based on number of transactions waiting to be processed."""
        # Update congestion level based on number of transactions in the queue
        queue_size = len(self.transactions_queue)
        prev_congestion = self.congestion_level
        self.congestion_level = min(1.0, queue_size / 100)  # Maximum is 1.0
        
        # Add to history
        self.historical_congestion.append(self.congestion_level)
        
        # Limit history size to prevent excessive memory usage
        if len(self.historical_congestion) > 100:
            self.historical_congestion = self.historical_congestion[-100:]
        
        # Update resource utilization based on congestion delta
        congestion_delta = abs(self.congestion_level - prev_congestion)
        self.resource_utilization = 0.8 * self.resource_utilization + 0.2 * (0.5 + congestion_delta * 2)
        
        # Adjust network stability based on congestion
        if self.congestion_level > 0.8:
            self.network_stability = max(0.5, self.network_stability * 0.95)
        else:
            self.network_stability = min(1.0, self.network_stability * 1.01)
            
        return self.congestion_level
    
    def get_shard_health(self):
        """Get overall health metrics for the shard."""
        health_metrics = {
            'congestion_level': self.congestion_level,
            'network_stability': self.network_stability,
            'resource_utilization': self.resource_utilization,
            'consensus_difficulty': self.consensus_difficulty,
            'transaction_queue_size': len(self.transactions_queue),
            'processed_transactions': len(self.processed_transactions),
            'blocked_transactions': len(self.blocked_transactions),
            'non_malicious_nodes': len(self.get_non_malicious_nodes()),
            'malicious_nodes': len(self.get_malicious_nodes()),
            'total_nodes': len(self.nodes)
        }
        return health_metrics
    
    def add_transaction_to_queue(self, tx):
        """Add a transaction to the shard's processing queue."""
        self.transactions_queue.append(tx)
        self.update_congestion()
        return len(self.transactions_queue)
    
    def clear_old_data(self, max_processed=1000, max_blocked=500):
        """Clean old data to prevent memory overuse."""
        # Keep only the most recent transactions
        if len(self.processed_transactions) > max_processed:
            overflow = len(self.processed_transactions) - max_processed
            self.processed_transactions = self.processed_transactions[overflow:]
        
        # Keep only recent blocked transactions
        if len(self.blocked_transactions) > max_blocked:
            overflow = len(self.blocked_transactions) - max_blocked
            self.blocked_transactions = self.blocked_transactions[overflow:]
            
        return True
    
    def __str__(self):
        """String representation of the shard."""
        return f"Shard {self.shard_id} ({len(self.nodes)} nodes, {len(self.transactions_queue)} pending tx)"

# Function for processing single transaction in a separate thread
def _process_single_transaction(tx, sim_context):
    """Helper function to process a single transaction in parallel execution."""
    # Get necessary components from simulation context
    pipeline = sim_context['pipeline']
    shards = sim_context['shards']
    metrics = sim_context['metrics']
    batch_id = sim_context.get('batch_id', 0)
    
    # Start time tracking
    start_time = time.time()
    
    # Process the transaction
    processed_tx, success = pipeline.process_transaction(tx, sim_context)
    
    # Record processing time
    processing_time = time.time() - start_time
    
    if success:
        # If transaction was successful
        # Add to processed transactions list for source and target shards
        shards[tx.source_shard].processed_transactions.append(tx)
        if tx.is_cross_shard:
            shards[tx.target_shard].processed_transactions.append(tx)
        
        # Update metrics
        metrics['total_processed'] += 1
        metrics['processing_times'].append(processing_time)
        metrics['latencies'].append(tx.latency)
        metrics['hops'].append(tx.hops)
        metrics['energy_consumption'].append(tx.energy)
        
        if tx.is_cross_shard:
            metrics['cross_shard_processed'] += 1
            metrics['cross_shard_latencies'].append(tx.latency)
        else:
            metrics['same_shard_processed'] += 1
            metrics['same_shard_latencies'].append(tx.latency)
            
        # Update data integrity metrics
        metrics['data_integrity_values'].append(tx.data_integrity)
        
        # Record resource cost
        metrics['resource_costs'].append(tx.resource_cost)
    else:
        # If transaction failed
        # Add to blocked transactions for tracking
        shards[tx.source_shard].blocked_transactions.append(tx)
        
        # Update failure metrics
        metrics['total_failed'] += 1
        
        if tx.is_cross_shard:
            metrics['cross_shard_failed'] += 1
        else:
            metrics['same_shard_failed'] += 1
    
    # Return processed transaction and success status
    return processed_tx, success

class LargeScaleBlockchainSimulation:
    __slots__ = ('num_shards', 'nodes_per_shard', 'malicious_percentage', 'attack_scenario',
                'attack_types', 'max_workers', 'shards', 'transactions', 'processed_transactions',
                'blocked_transactions', 'current_step', 'metrics_history', 'pipeline', 'total_processed_tx',
                'total_cross_shard_tx', 'avg_latency', 'avg_energy', 'pipeline_stats',
                'tx_counter', 'start_time')
    
    def __init__(self, 
                 num_shards=10, 
                 nodes_per_shard=20,
                 malicious_percentage=10,
                 attack_scenario=None,
                 max_workers=None):
        self.num_shards = num_shards
        self.nodes_per_shard = nodes_per_shard
        self.malicious_percentage = malicious_percentage
        self.attack_scenario = attack_scenario
        
        # Timestamp for metrics calculations
        self.start_time = time.time()
        self.current_step = 0
        
        # Create attack types based on scenario
        self.attack_types = []
        if attack_scenario == "51_percent":
            self.attack_types = ["51_percent"] 
        elif attack_scenario == "sybil":
            self.attack_types = ["sybil"]
        elif attack_scenario == "eclipse":
            self.attack_types = ["eclipse"]
        elif attack_scenario == "selfish_mining":
            self.attack_types = ["selfish_mining"]
        elif attack_scenario == "random":
            self.attack_types = ["eclipse", "sybil", "51_percent", "selfish_mining"]
        
        self.max_workers = max_workers
        
        # Initialize blockchain
        self.shards = []
        self._initialize_blockchain()
        
        # Create network connections
        self._create_network_connections()
        
        # Set up attack if specified
        if 'eclipse' in self.attack_types:
            self._setup_eclipse_attack()
        
        # Initialize variables for tracking
        self.transactions = []
        self.processed_transactions = []
        self.tx_counter = 0  # Transaction ID counter
        
        # Store performance metrics
        self.metrics_history = {
            'throughput': [],
            'latency': [],
            'energy': [],
            'security': [],
            'cross_shard_ratio': [],
            'transaction_success_rate': [],
            'network_stability': [],
            'resource_utilization': [],
            'consensus_efficiency': [],
            'shard_health': [],
            'avg_hops': [],
            'network_resilience': [],
            'avg_block_size': [],
            'network_partition_events': []
        }
        
        # Initialize transaction processing pipeline
        self.pipeline = TransactionPipeline(self.shards, self.max_workers)
        
        # Additional tracking variables
        self.total_processed_tx = 0
        self.total_cross_shard_tx = 0
        self.avg_latency = 0
        self.avg_energy = 0
        self.pipeline_stats = {}
        
        # Blocked transactions list
        self.blocked_transactions = []
    
    def _initialize_blockchain(self):
        # Create shards
        for i in range(self.num_shards):
            shard = Shard(
                shard_id=i,
                num_nodes=self.nodes_per_shard,
                malicious_percentage=self.malicious_percentage,
                attack_types=self.attack_types
            )
            self.shards.append(shard)
        
        # Create connections between nodes
        self._create_network_connections()
        
        print(f"Initialized blockchain with {self.num_shards} shards, each shard has {self.nodes_per_shard} nodes")
        print(f"Malicious node percentage: {self.malicious_percentage}%")
        if self.attack_scenario:
            print(f"Attack scenario: {self.attack_scenario}")
    
    def _create_network_connections(self):
        # Create inter-shard connections
        for source_shard in self.shards:
            for target_shard in self.shards:
                if source_shard.shard_id != target_shard.shard_id:
                    # Select nodes from each shard randomly for connection
                    source_nodes = random.sample(source_shard.nodes, min(5, len(source_shard.nodes)))
                    target_nodes = random.sample(target_shard.nodes, min(5, len(target_shard.nodes)))
                    
                    for s_node in source_nodes:
                        for t_node in target_nodes:
                            s_node.connections.append(t_node)
                            t_node.connections.append(s_node)
        
        # Add intra-shard connections
        for shard in self.shards:
            for i, node in enumerate(shard.nodes):
                # Each node connects to 80% of other nodes in the shard
                potential_connections = [n for n in shard.nodes if n != node]
                num_connections = int(len(potential_connections) * 0.8)
                connections = random.sample(potential_connections, num_connections)
                
                for conn in connections:
                    if conn not in node.connections:
                        node.connections.append(conn)
                    if node not in conn.connections:
                        conn.connections.append(node)
                        
        # If there is an Eclipse attack scenario, change connections
        if 'eclipse' in self.attack_types:
            self._setup_eclipse_attack()
    
    def _setup_eclipse_attack(self):
        # Select a shard randomly to perform attack
        target_shard = random.choice(self.shards)
        malicious_nodes = target_shard.get_malicious_nodes()
        
        if malicious_nodes:
            # Select a node randomly to be isolated
            victim_nodes = random.sample(target_shard.get_non_malicious_nodes(), 
                                         min(3, len(target_shard.get_non_malicious_nodes())))
            
            for victim in victim_nodes:
                print(f"Setting up Eclipse attack on node {victim.node_id}")
                
                # Remove all current connections
                for conn in victim.connections[:]:
                    if conn in victim.connections:
                        victim.connections.remove(conn)
                    if victim in conn.connections:
                        conn.connections.remove(victim)
                
                # Connect only to malicious nodes
                for attacker in malicious_nodes:
                    victim.connections.append(attacker)
                    attacker.connections.append(victim)
    
    def _generate_transactions(self, num_transactions):
        new_transactions = []
        
        # Create list of source/destination pairs in advance to reduce processing load
        pairs = []
        cross_shard_count = int(num_transactions * 0.3)  # 30% are cross-shard transactions
        same_shard_count = num_transactions - cross_shard_count
        
        # Create pairs within the same shard
        for _ in range(same_shard_count):
            shard_id = random.randint(0, self.num_shards - 1)
            pairs.append((shard_id, shard_id))
        
        # Create cross-shard pairs
        for _ in range(cross_shard_count):
            source_shard = random.randint(0, self.num_shards - 1)
            target_shard = random.randint(0, self.num_shards - 1)
            while target_shard == source_shard:
                target_shard = random.randint(0, self.num_shards - 1)
            pairs.append((source_shard, target_shard))
        
        # Shuffle pairs to avoid processing in batches
        random.shuffle(pairs)
        
        # Create transactions from created pairs
        for source_shard, target_shard in pairs:
            # Create transaction with random size
            tx = Transaction(
                tx_id=f"tx_{self.tx_counter}",
                source_shard=source_shard,
                target_shard=target_shard,
                size=random.uniform(0.5, 2.0)
            )
            self.tx_counter += 1
            new_transactions.append(tx)
            
            # Add to source shard's queue - using new optimized method
            self.shards[source_shard].add_transaction_to_queue(tx)
        
        # Add to list of all transactions
        self.transactions.extend(new_transactions)
        
        # Clear cache of pipeline periodically to prevent excessive memory usage
        if self.tx_counter % 1000 == 0:
            self.pipeline.clear_caches()
        
        return new_transactions
    
    def _process_transactions(self):
        processed_count = 0
        blocked_count = 0
        total_hops = 0
        
        # Update congestion level for all shards
        for shard in self.shards:
            shard.update_congestion()
        
        # Prepare context for multi-threaded processing
        sim_context = {
            'shards': self.shards,
            'attack_types': [self.attack_scenario] if self.attack_scenario else []
        }
        
        # Collect transactions to process from all shards and group them by shard
        shard_transactions = {i: [] for i in range(len(self.shards))}
        
        # Process more efficiently by limiting number of transactions processed per shard
        max_tx_per_shard = 20
        
        for idx, shard in enumerate(self.shards):
            # Get number of non-malicious nodes
            non_malicious_nodes = shard.get_non_malicious_nodes()
            
            # If there are not enough nodes to reach consensus, skip this shard
            if len(non_malicious_nodes) < self.nodes_per_shard / 2:
                continue
                
            # Get transactions with highest priority (already sorted in add_transaction_to_queue)
            transactions_to_process = shard.transactions_queue[:max_tx_per_shard]
            
            # Add to list of transactions to process
            shard_transactions[idx].extend(transactions_to_process)
        
        # Use pipeline to process transactions in parallel through each stage
        results = self.pipeline.process_shard_transactions(shard_transactions, sim_context)
        
        # Store pipeline stats
        self.pipeline_stats = self.pipeline.get_pipeline_metrics()
        
        # Track node workload allocation
        node_work_allocation = {}
        
        # Process results
        for tx, success in results:
            # Remove from source shard's queue - optimizing by finding efficient nodes
            source_shard = self.shards[tx.source_shard]
            try:
                source_shard.transactions_queue.remove(tx)
            except ValueError:
                # Transaction may have been removed by another process
                pass
            
            if success:
                # Transaction successful
                self.processed_transactions.append(tx)
                source_shard.processed_transactions.append(tx)
                processed_count += 1
                
                # Update statistics
                if tx.is_cross_shard:
                    self.total_cross_shard_tx += 1
                    total_hops += tx.hops
                
                self.total_processed_tx += 1
                
                # Update number of transactions processed for each node
                valid_nodes = source_shard.get_non_malicious_nodes()
                if valid_nodes:
                    # Select least busy node to distribute workload evenly
                    selected_node = min(valid_nodes, key=lambda node: node_work_allocation.get(node.node_id, 0))
                    selected_node.transactions_processed += 1
                    
                    # Update workload distribution
                    node_work_allocation[selected_node.node_id] = node_work_allocation.get(selected_node.node_id, 0) + 1
            else:
                # Transaction failed
                blocked_count += 1
                source_shard.blocked_transactions.append(tx)
        
        # Calculate average number of hops
        if processed_count > 0:
            avg_hops = total_hops / max(1, self.total_cross_shard_tx)
        else:
            avg_hops = 0
            
        # Clear old data periodically to save memory
        if self.current_step % 10 == 0:
            for shard in self.shards:
                shard.clear_old_data()
        
        # Update trust score of nodes
        for shard in self.shards:
            for node in shard.nodes:
                # Calculate success rate based on number of transactions processed
                success_rate = min(1.0, node.transactions_processed / max(1, self.total_processed_tx / len(self.shards)))
                node.update_trust_score(success_rate)
        
        return processed_count, blocked_count, avg_hops
    
    def _calculate_metrics(self):
        # Avoid division by zero
        if len(self.processed_transactions) == 0:
            return {
                'throughput': 0,
                'latency': 0,
                'energy': 0,
                'security': 0,
                'cross_shard_ratio': 0,
                'transaction_success_rate': 0,
                'network_stability': 0,
                'resource_utilization': 0,
                'consensus_efficiency': 0,
                'shard_health': 0,
                'avg_hops': 0,
                'network_resilience': 0,
                'avg_block_size': 0,
                'network_partition_events': 0
            }
        
        # Calculate average metrics
        time_elapsed = time.time() - self.start_time + 0.001  # Avoid division by zero
        total_transactions = len(self.processed_transactions) + len(self.blocked_transactions)
        success_rate = len(self.processed_transactions) / max(1, total_transactions)
        
        # Calculate throughput (transactions per second)
        throughput = len(self.processed_transactions) / time_elapsed
        
        # Average latency
        avg_latency = sum(tx.latency for tx in self.processed_transactions) / len(self.processed_transactions)
        
        # Average energy consumption
        avg_energy = sum(tx.energy for tx in self.processed_transactions) / len(self.processed_transactions)
        
        # Calculate cross-shard transaction ratio
        cross_shard_count = sum(1 for tx in self.processed_transactions if tx.is_cross_shard)
        cross_shard_ratio = cross_shard_count / len(self.processed_transactions)
        
        # Calculate network stability
        network_stability = sum(shard.network_stability for shard in self.shards) / len(self.shards)
        
        # Calculate resource utilization
        resource_utilization = sum(shard.resource_utilization for shard in self.shards) / len(self.shards)
        
        # Calculate consensus efficiency
        consensus_efficiency = self.pipeline.pipeline_metrics['commit'] / max(1, self.pipeline.pipeline_metrics['consensus'])
        
        # Calculate average trust score
        avg_trust = sum(node.trust_score for shard in self.shards for node in shard.nodes) / sum(len(shard.nodes) for shard in self.shards)
        
        # Count number of malicious nodes
        malicious_nodes = sum(len(shard.get_malicious_nodes()) for shard in self.shards)
        max_malicious_threshold = 0.33  # Maximum threshold for malicious nodes
        
        # Calculate shard health average (using key metrics from shard health)
        # Updated to handle dictionary return from get_shard_health
        shard_health_metrics = []
        for shard in self.shards:
            health_dict = shard.get_shard_health()
            # Calculate a health score from the metrics in the dictionary
            health_score = (
                (1 - health_dict['congestion_level']) * 0.3 +
                health_dict['network_stability'] * 0.3 +
                (1 - health_dict['resource_utilization']) * 0.2 +
                (health_dict['non_malicious_nodes'] / health_dict['total_nodes']) * 0.2
            )
            shard_health_metrics.append(health_score)
        
        shard_health = sum(shard_health_metrics) / len(self.shards)
        
        # Calculate average number of hops
        avg_hops = sum(tx.hops for tx in self.processed_transactions) / len(self.processed_transactions)
        
        # Calculate security score - improved formula
        malicious_ratio = sum(len(shard.get_malicious_nodes()) for shard in self.shards) / sum(len(shard.nodes) for shard in self.shards)
        if malicious_ratio >= 0.51:  # 51% attack
            security = max(0, 0.2 - (malicious_ratio - 0.51) * 2) * avg_trust
        elif malicious_ratio > max_malicious_threshold:  # Close to dangerous threshold
            security = max(0, 1 - ((malicious_ratio - max_malicious_threshold) / (0.51 - max_malicious_threshold))) * avg_trust
        else:
            security = (1 - (malicious_ratio / max_malicious_threshold)) * avg_trust
        
        # Calculate network resilience
        # Improved resilience to 51% attack
        if '51_percent' in self.attack_types and malicious_ratio >= 0.51:
            # Low trust score nodes detected (already detected)
            detected_malicious = sum(1 for shard in self.shards 
                                   for node in shard.get_malicious_nodes() 
                                   if node.trust_score < 0.5)
            detection_rate = detected_malicious / max(1, malicious_nodes)
            
            # Resilience will depend on detection rate and number of malicious nodes
            network_resilience = detection_rate * (1 - malicious_ratio/0.7) * security
            network_resilience = max(0, min(0.6, network_resilience))  # Maximum is 0.6 for 51% attack
        elif malicious_ratio > 0.3:
            # For high malicious ratio but not yet 51%
            network_resilience = (1 - malicious_ratio) * security * 0.7
        else:
            # For low malicious ratio
            network_resilience = (1 - malicious_ratio) * security
        
        network_resilience = max(0, min(1, network_resilience))  # Maximum is 0-1
        
        # Calculate average block size (simulated)
        avg_block_size = sum(tx.size for tx in self.processed_transactions[-100:]) / min(100, len(self.processed_transactions))
        
        # Number of network partition events (simulated)
        network_partition_events = int(10 * (1 - network_stability))
        
        return {
            'throughput': throughput,
            'latency': avg_latency,
            'energy': avg_energy,
            'security': security,
            'cross_shard_ratio': cross_shard_ratio,
            'transaction_success_rate': success_rate,
            'network_stability': network_stability,
            'resource_utilization': resource_utilization,
            'consensus_efficiency': consensus_efficiency,
            'shard_health': shard_health,
            'avg_hops': avg_hops,
            'network_resilience': network_resilience,
            'avg_block_size': avg_block_size,
            'network_partition_events': network_partition_events
        }
    
    def _update_metrics(self):
        metrics = self._calculate_metrics()
        
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
    
    def run_simulation(self, num_steps=1000, transactions_per_step=50):
        print(f"Starting simulation with {num_steps} steps, {transactions_per_step} transactions/step")
        
        for step in tqdm(range(num_steps)):
            self.current_step = step + 1
            
            # Generate new transactions
            self._generate_transactions(transactions_per_step)
            
            # Process transactions
            processed, blocked, avg_hops = self._process_transactions()
            
            # Update metrics
            self._update_metrics()
            
            # Print every 100 steps
            if (step + 1) % 100 == 0:
                metrics = self._calculate_metrics()
                print(f"\nStep {step + 1}/{num_steps}:")
                print(f"  Throughput: {metrics['throughput']:.2f} tx/s")
                print(f"  Average latency: {metrics['latency']:.2f} ms")
                print(f"  Transaction success rate: {metrics['transaction_success_rate']:.2f}")
                print(f"  Network stability: {metrics['network_stability']:.2f}")
                print(f"  Shard health: {metrics['shard_health']:.2f}")
                print(f"  Transactions processed: {processed}, blocked: {blocked}")
        
        print("\nSimulation complete!")
        return self.metrics_history
    
    def plot_metrics(self, save_dir=None):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Set style for plot
            plt.style.use('dark_background')
            sns.set(style="darkgrid")
            
            # Create custom color map
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=len(colors))
            
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(20, 16))
            
            # Set GridSpec
            gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
            
            # 1. Throughput
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(self.metrics_history['throughput'], color=colors[0], linewidth=2)
            ax1.set_title('Throughput (tx/s)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('tx/s')
            ax1.grid(True, alpha=0.3)
            ax1.fill_between(range(len(self.metrics_history['throughput'])), 
                             self.metrics_history['throughput'], 
                             alpha=0.3, 
                             color=colors[0])
            
            # 2. Latency
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(self.metrics_history['latency'], color=colors[1], linewidth=2)
            ax2.set_title('Latency (ms)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('ms')
            ax2.grid(True, alpha=0.3)
            ax2.fill_between(range(len(self.metrics_history['latency'])), 
                             self.metrics_history['latency'], 
                             alpha=0.3, 
                             color=colors[1])
            
            # 3. Energy
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.plot(self.metrics_history['energy'], color=colors[2], linewidth=2)
            ax3.set_title('Energy Consumption', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Energy Units')
            ax3.grid(True, alpha=0.3)
            ax3.fill_between(range(len(self.metrics_history['energy'])), 
                             self.metrics_history['energy'], 
                             alpha=0.3, 
                             color=colors[2])
            
            # 4. Security
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.plot(self.metrics_history['security'], color=colors[3], linewidth=2)
            ax4.set_title('Security', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Score (0-1)')
            ax4.grid(True, alpha=0.3)
            ax4.fill_between(range(len(self.metrics_history['security'])), 
                             self.metrics_history['security'], 
                             alpha=0.3, 
                             color=colors[3])
            
            # 5. Cross-shard ratio
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.plot(self.metrics_history['cross_shard_ratio'], color=colors[4], linewidth=2)
            ax5.set_title('Cross-shard Transaction Ratio', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Step')
            ax5.set_ylabel('Ratio')
            ax5.grid(True, alpha=0.3)
            ax5.fill_between(range(len(self.metrics_history['cross_shard_ratio'])), 
                             self.metrics_history['cross_shard_ratio'], 
                             alpha=0.3, 
                             color=colors[4])
            
            # 6. Transaction success rate
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.plot(self.metrics_history['transaction_success_rate'], color=colors[5], linewidth=2)
            ax6.set_title('Transaction Success Rate', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Step')
            ax6.set_ylabel('Ratio')
            ax6.grid(True, alpha=0.3)
            ax6.fill_between(range(len(self.metrics_history['transaction_success_rate'])), 
                             self.metrics_history['transaction_success_rate'], 
                             alpha=0.3, 
                             color=colors[5])
            
            # 7. Network stability
            ax7 = fig.add_subplot(gs[2, 0])
            ax7.plot(self.metrics_history['network_stability'], color=colors[0], linewidth=2)
            ax7.set_title('Network Stability', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Step')
            ax7.set_ylabel('Score (0-1)')
            ax7.grid(True, alpha=0.3)
            ax7.fill_between(range(len(self.metrics_history['network_stability'])), 
                             self.metrics_history['network_stability'], 
                             alpha=0.3, 
                             color=colors[0])
            
            # 8. Resource utilization
            ax8 = fig.add_subplot(gs[2, 1])
            ax8.plot(self.metrics_history['resource_utilization'], color=colors[1], linewidth=2)
            ax8.set_title('Resource Utilization', fontsize=14, fontweight='bold')
            ax8.set_xlabel('Step')
            ax8.set_ylabel('Ratio')
            ax8.grid(True, alpha=0.3)
            ax8.fill_between(range(len(self.metrics_history['resource_utilization'])), 
                             self.metrics_history['resource_utilization'], 
                             alpha=0.3, 
                             color=colors[1])
            
            # 9. Consensus efficiency
            ax9 = fig.add_subplot(gs[2, 2])
            ax9.plot(self.metrics_history['consensus_efficiency'], color=colors[2], linewidth=2)
            ax9.set_title('Consensus Efficiency', fontsize=14, fontweight='bold')
            ax9.set_xlabel('Step')
            ax9.set_ylabel('Score (0-1)')
            ax9.grid(True, alpha=0.3)
            ax9.fill_between(range(len(self.metrics_history['consensus_efficiency'])), 
                             self.metrics_history['consensus_efficiency'], 
                             alpha=0.3, 
                             color=colors[2])
            
            # 10. Shard health
            ax10 = fig.add_subplot(gs[3, 0])
            ax10.plot(self.metrics_history['shard_health'], color=colors[3], linewidth=2)
            ax10.set_title('Shard Health', fontsize=14, fontweight='bold')
            ax10.set_xlabel('Step')
            ax10.set_ylabel('Score (0-1)')
            ax10.grid(True, alpha=0.3)
            ax10.fill_between(range(len(self.metrics_history['shard_health'])), 
                             self.metrics_history['shard_health'], 
                             alpha=0.3, 
                             color=colors[3])
            
            # 11. Network resilience
            ax11 = fig.add_subplot(gs[3, 1])
            ax11.plot(self.metrics_history['network_resilience'], color=colors[4], linewidth=2)
            ax11.set_title('Network Resilience', fontsize=14, fontweight='bold')
            ax11.set_xlabel('Step')
            ax11.set_ylabel('Score (0-1)')
            ax11.grid(True, alpha=0.3)
            ax11.fill_between(range(len(self.metrics_history['network_resilience'])), 
                             self.metrics_history['network_resilience'], 
                             alpha=0.3, 
                             color=colors[4])
            
            # 12. Average block size
            ax12 = fig.add_subplot(gs[3, 2])
            ax12.plot(self.metrics_history['avg_block_size'], color=colors[5], linewidth=2)
            ax12.set_title('Average Block Size', fontsize=14, fontweight='bold')
            ax12.set_xlabel('Step')
            ax12.set_ylabel('Size')
            ax12.grid(True, alpha=0.3)
            ax12.fill_between(range(len(self.metrics_history['avg_block_size'])), 
                             self.metrics_history['avg_block_size'], 
                             alpha=0.3, 
                             color=colors[5])
            
            # Main title
            fig.suptitle(f'QTrust Blockchain Metrics - {self.num_shards} Shards, {self.nodes_per_shard} Nodes/Shard', 
                        fontsize=20, fontweight='bold', y=0.98)
            
            # Add note about attack scenario
            attack_text = f"Attack Scenario: {self.attack_scenario}" if self.attack_scenario else "No Attack"
            plt.figtext(0.5, 0.01, attack_text, ha="center", fontsize=16, bbox={"facecolor":"red", "alpha":0.2, "pad":5})
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save plot
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = os.path.join(save_dir, f"detailed_metrics_{self.num_shards}shards_{self.nodes_per_shard}nodes_{timestamp}.png")
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Saved detailed metrics to: {save_file}")
            
            plt.close(fig)
            
            # Create radar chart
            self._plot_radar_chart(save_dir)
            
            # Create heatmap for congestion level
            self._plot_congestion_heatmap(save_dir)
    
    def _plot_congestion_heatmap(self, save_dir=None):
        # Check if there is congestion data
        has_congestion_data = True
        for shard in self.shards:
            if not hasattr(shard, 'historical_congestion') or not shard.historical_congestion:
                has_congestion_data = False
                break
        
        if not has_congestion_data:
            return
        
        # Prepare data for heatmap
        # Get historical congestion data from each shard
        congestion_data = []
        for shard in self.shards:
            # Ensure all shards have the same number of data points
            # by taking the last 100 points
            if len(shard.historical_congestion) > 100:
                congestion_data.append(shard.historical_congestion[-100:])
            else:
                # If less than 100 points, add 0 at the beginning to make 100 points
                padding = [0] * (100 - len(shard.historical_congestion))
                congestion_data.append(padding + shard.historical_congestion)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Create heatmap
        congestion_array = np.array(congestion_data)
        
        # Reverse order so shard 0 is at the bottom
        congestion_array = np.flip(congestion_array, axis=0)
        
        # Customize color map
        cmap = LinearSegmentedColormap.from_list("custom", ["#1a9850", "#ffffbf", "#d73027"], N=256)
        
        # Draw heatmap
        im = ax.imshow(congestion_array, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Congestion Level', rotation=270, labelpad=15)
        
        # Set labels
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Shard ID')
        
        # Set y-axis ticks to display shard IDs
        shard_ids = [f'Shard {self.num_shards - i - 1}' for i in range(self.num_shards)]
        ax.set_yticks(np.arange(len(shard_ids)))
        ax.set_yticklabels(shard_ids)
        
        # Reduce number of x-axis labels to avoid overlap
        step = max(1, len(congestion_array[0]) // 10)
        ax.set_xticks(np.arange(0, len(congestion_array[0]), step))
        ax.set_xticklabels(np.arange(0, len(congestion_array[0]), step))
        
        # Title
        plt.title(f'Shard Congestion Analysis - {self.num_shards} Shards, {self.nodes_per_shard} Nodes/Shard', fontsize=14)
        
        plt.tight_layout()
        
        # Save chart
        if save_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = os.path.join(save_dir, f"congestion_heatmap_{self.num_shards}shards_{self.nodes_per_shard}nodes_{timestamp}.png")
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Congestion heatmap saved to: {save_file}")
        
        plt.close(fig)
    
    def generate_report(self, save_dir=None):
        """Generate comprehensive report of simulation results and save to specified directory."""
        print("\n" + "=" * 80)
        print(" " * 30 + "SIMULATION REPORT")
        print("=" * 80 + "\n")
        
        print("CONFIGURATION:")
        print(f"- Number of shards: {self.num_shards}")
        print(f"- Nodes per shard: {self.nodes_per_shard}")
        print(f"- Malicious node percentage: {self.malicious_percentage}%")
        print(f"- Attack scenario: {self.attack_scenario}")
        print()
        
        # Performance metrics
        metrics = self._calculate_metrics()
        
        print("PERFORMANCE:")
        print(f"- Throughput: {metrics['throughput']:.2f} tx/s")
        print(f"- Latency: {metrics['latency']:.2f} ms")
        print(f"- Energy: {metrics['energy']:.2f} units")
        print(f"- Security score: {metrics['security']:.2f}")
        print(f"- Cross-shard transaction ratio: {metrics['cross_shard_ratio']:.2f}")
        print(f"- Transaction success rate: {metrics['transaction_success_rate']:.2f}")
        print()
        
        # Pipeline performance
        print("PIPELINE PERFORMANCE:")
        print(f"- Pipeline throughput: {len(self.processed_transactions)/max(1, self.current_step):.2f} tx/s")
        
        # Find bottleneck
        pipeline_stages = self.pipeline.pipeline_metrics
        max_stage = max(pipeline_stages, key=pipeline_stages.get)
        print(f"- Bottleneck stage: {max_stage}")
        
        # Calculate parallel efficiency
        parallel_efficiency = 1.0  # Default value if can't calculate
        print(f"- Parallel processing efficiency: {parallel_efficiency:.2f}")
        print()
        
        # Pipeline stage performance
        print("  Stage performance:")
        total = max(1, sum(pipeline_stages.values()))
        for stage, count in pipeline_stages.items():
            print(f"  - {stage}: {count/max(1, pipeline_stages[max_stage]):.2f}")
        
        print()
        print("  Transactions processed by stage:")
        for stage, count in pipeline_stages.items():
            print(f"  - {stage}: {count}")
        
        print()
        
        # Per-shard statistics
        print("SHARD STATISTICS:")
        print()
        
        for shard_id, shard in enumerate(self.shards):
            shard_stats = shard.get_shard_health()
            print(f"SHARD_{shard_id}:")
            print(f"- Nodes: {shard_stats['total_nodes']}")
            print(f"- Malicious nodes: {shard_stats['malicious_nodes']}")
            print(f"- Congestion level: {shard_stats['congestion_level']:.2f}")
            print(f"- Processed transactions: {len(shard.processed_transactions)}")
            print(f"- Blocked transactions: {len(shard.blocked_transactions)}")
            
            # Calculate health score from the shard_stats dictionary
            health_score = (
                (1 - shard_stats['congestion_level']) * 0.3 +
                shard_stats['network_stability'] * 0.3 +
                (1 - shard_stats['resource_utilization']) * 0.2 +
                (shard_stats['non_malicious_nodes'] / max(1, shard_stats['total_nodes'])) * 0.2
            )
            print(f"- Health score: {health_score:.2f}")
            print()
        
        # Network statistics
        print("NETWORK STATISTICS:")
        print(f"- Network stability: {metrics['network_stability']:.2f}")
        print(f"- Resource utilization: {metrics['resource_utilization']:.2f}")
        print(f"- Network partition events: {metrics['network_partition_events']}")
        print(f"- Average number of hops: {metrics['avg_hops']:.2f}")
        print(f"- Network resilience: {metrics['network_resilience']:.2f}")
        print()
        
        # Save report if directory specified
        if save_dir:
            report_file = os.path.join(save_dir, "simulation_report.txt")
            with open(report_file, "w") as f:
                f.write("QTrust Large Scale Blockchain Simulation Report\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Configuration:\n")
                f.write(f"- Number of shards: {self.num_shards}\n")
                f.write(f"- Nodes per shard: {self.nodes_per_shard}\n")
                f.write(f"- Malicious node percentage: {self.malicious_percentage}%\n")
                f.write(f"- Attack scenario: {self.attack_scenario}\n\n")
                
                f.write("Performance Metrics:\n")
                for name, value in metrics.items():
                    f.write(f"- {name}: {value:.4f}\n")
                
                f.write("\nPipeline Performance:\n")
                for stage, count in pipeline_stages.items():
                    f.write(f"- {stage}: {count}\n")
                
                # Tính tổng số giao dịch
                total_transactions = len(self.transactions)
                cross_shard_count = sum(1 for tx in self.processed_transactions if tx.is_cross_shard)
                
                f.write("\nTransaction Statistics:\n")
                f.write(f"- Total transactions: {total_transactions}\n")
                f.write(f"- Successful transactions: {len(self.processed_transactions)}\n")
                f.write(f"- Failed transactions: {len(self.blocked_transactions)}\n")
                f.write(f"- Cross-shard transactions: {cross_shard_count}\n")
                
            print(f"Report saved to {report_file}")
        
        return metrics
    
    def _save_metrics_json(self, filename):
        """Save metrics as JSON for reuse."""
        import json
        
        # Prepare data
        data = {
            "config": {
                "num_shards": self.num_shards,
                "nodes_per_shard": self.nodes_per_shard,
                "total_nodes": self.num_shards * self.nodes_per_shard,
                "malicious_percentage": self.malicious_percentage,
                "attack_scenario": self.attack_scenario
            },
            "metrics": self.metrics_history,
            "processed_transactions": len(self.processed_transactions)
        }
        
        # Save file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Metrics saved to file: {filename}")
    
    def parallel_save_results(self, save_dir):
        """Save all simulation results in parallel using multiple threads."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Define tasks for parallel execution
        tasks = [
            ('metrics', lambda: self._save_metrics_json(os.path.join(save_dir, 'metrics.json'))),
            ('report', lambda: self._save_report_json(os.path.join(save_dir, 'report.json'))),
            ('transactions', lambda: self._save_transaction_stats(os.path.join(save_dir, 'transactions.json'))),
            ('nodes', lambda: self._save_node_stats(os.path.join(save_dir, 'nodes.json'))),
            ('radar_chart', lambda: self._plot_radar_chart(save_dir)),
            ('congestion_heatmap', lambda: self._plot_congestion_heatmap(save_dir)),
            ('metrics_chart', lambda: self.plot_metrics(save_dir))
        ]
        
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(task[1]): task[0] for task in tasks}
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Saving results"):
                task_name = future_to_task[future]
                try:
                    results[task_name] = future.result()
                except Exception as e:
                    print(f"Error saving {task_name}: {e}")
                    results[task_name] = False
        
        # Report success/failure
        success_count = sum(1 for result in results.values() if result)
        print(f"\nCompleted! Results saved at: {save_dir}")
    
    def _save_report_json(self, filename):
        """Save report as JSON."""
        import json
        
        # Create report
        report = self.generate_report(None)  # Create report without saving and plotting
        
        # Save file
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Report saved to file: {filename}")
    
    def _save_transaction_stats(self, filename):
        """Save transaction statistics."""
        import json
        
        # Calculate statistics
        processed_count = len(self.processed_transactions)
        cross_shard_count = sum(1 for tx in self.processed_transactions if tx.is_cross_shard)
        avg_latency = sum(tx.latency for tx in self.processed_transactions) / max(1, processed_count)
        avg_energy = sum(tx.energy for tx in self.processed_transactions) / max(1, processed_count)
        avg_hops = sum(tx.hops for tx in self.processed_transactions) / max(1, processed_count)
        
        # Latency distribution
        latency_distribution = {}
        for tx in self.processed_transactions:
            latency_range = int(tx.latency / 10) * 10  # Round to nearest 10ms
            latency_distribution[latency_range] = latency_distribution.get(latency_range, 0) + 1
        
        # Statistical data
        data = {
            "total_transactions": len(self.transactions),
            "processed_transactions": processed_count,
            "cross_shard_transactions": cross_shard_count,
            "cross_shard_ratio": cross_shard_count / max(1, processed_count),
            "avg_latency": avg_latency,
            "avg_energy": avg_energy,
            "avg_hops": avg_hops,
            "latency_distribution": {str(k): v for k, v in sorted(latency_distribution.items())}
        }
        
        # Save file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Transaction statistics saved to file: {filename}")
    
    def _save_node_stats(self, filename):
        """Save node statistics."""
        import json
        
        # Calculate statistics
        node_stats = []
        for shard in self.shards:
            for node in shard.nodes:
                node_stats.append({
                    "node_id": node.node_id,
                    "shard_id": node.shard_id,
                    "is_malicious": node.is_malicious,
                    "attack_type": node.attack_type,
                    "trust_score": node.trust_score,
                    "processing_power": node.processing_power,
                    "transactions_processed": node.transactions_processed,
                    "uptime": node.uptime,
                    "energy_efficiency": node.energy_efficiency
                })
        
        # Save file
        with open(filename, 'w') as f:
            json.dump(node_stats, f, indent=2)
            
        print(f"Node statistics saved to file: {filename}")
    
    def save_final_metrics(self, result_subdir):
        """Save and return final performance metrics"""
        print(f"\nCompleted! Results saved at: {result_subdir}")
        
        # Return final performance metrics
        final_metrics = self._calculate_metrics()
        return final_metrics

    def _plot_radar_chart(self, save_dir=None):
        """Plot radar chart for performance metrics."""
        if not save_dir:
            return
            
        # Get metrics from simulation
        metrics = self.metrics_history
        
        if not metrics or not all(key in metrics for key in ['throughput', 'latency', 'energy', 'security', 'cross_shard_ratio', 'transaction_success_rate']):
            return
            
        # Get the last value from each metric
        categories = ['Throughput', 'Latency', 'Energy',
                 'Security', 'Cross-shard Transactions', 
                 'Success Rate', 'Network Stability',
                 'Resource Utilization', 'Consensus', 
                 'Shard Health']
        
        # Normalize values
        values = []
        if metrics['throughput']:
            values.append(min(1.0, metrics['throughput'][-1] / 50))  # Assume max 50 tx/s is 1.0
        else:
            values.append(0)
            
        if metrics['latency']:
            values.append(max(0, 1.0 - (metrics['latency'][-1] / 50)))  # Lower is better
        else:
            values.append(0)
            
        if metrics['energy']:
            values.append(max(0, 1.0 - (metrics['energy'][-1] / 5.0)))  # Lower is better
        else:
            values.append(0)
            
        for key in ['security', 'cross_shard_ratio', 'transaction_success_rate', 'network_stability']:
            if metrics[key]:
                values.append(metrics[key][-1])
            else:
                values.append(0)
        
        if metrics['resource_utilization']:
            values.append(max(0, 1.0 - metrics['resource_utilization'][-1]))  # Lower is better
        else:
            values.append(0)
            
        if metrics['consensus_efficiency']:
            values.append(metrics['consensus_efficiency'][-1])
        else:
            values.append(0)
            
        if metrics['shard_health']:
            values.append(metrics['shard_health'][-1])
        else:
            values.append(0)
        
        # Create chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of categories
        N = len(categories)
        
        # Angle for each category (divide 360 degrees evenly)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the last value to close the chart
        values += values[:1]
        
        # Draw the chart
        ax.plot(angles, values, linewidth=2, linestyle='solid', color='#1f77b4')
        ax.fill(angles, values, color='#1f77b4', alpha=0.4)
        
        # Set labels
        plt.xticks(angles[:-1], categories, size=12)
        
        # Create level labels (0.2, 0.4, 0.6, 0.8, 1.0)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], 
                  color="grey", size=10)
        plt.ylim(0, 1)
        
        # Title
        plt.title(f'Blockchain Performance Radar - {self.num_shards} Shards', 
                 size=15, y=1.05)
        
        # Save chart
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = os.path.join(save_dir, f"radar_chart_{self.num_shards}shards_{self.nodes_per_shard}nodes_{timestamp}.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Radar chart saved to: {save_file}")
        
        plt.close(fig)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="QTrust Large Scale Blockchain Simulation")
    
    # Simulation parameters
    parser.add_argument("--num-shards", type=int, default=10, help="Number of shards (default: 10)")
    parser.add_argument("--nodes-per-shard", type=int, default=20, help="Number of nodes per shard (default: 20)")
    parser.add_argument("--malicious-percentage", type=float, default=10, help="Percentage of malicious nodes (default: 10%)")
    parser.add_argument("--attack-scenario", type=str, choices=["51_percent", "sybil", "eclipse", "selfish_mining", "random", None],
                        default=None, help="Attack scenario to simulate")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps (default: 1000)")
    parser.add_argument("--transactions-per-step", type=int, default=50, help="Number of transactions per step (default: 50)")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of workers for parallel processing")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save results (default: 'results')")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating plots (save JSON data only)")
    parser.add_argument("--parallel-save", action="store_true", help="Save results in parallel using multiple threads")
    
    args = parser.parse_args()
    
    # Print configuration
    print("\nQTrust Large Scale Blockchain Simulation")
    print("=======================================")
    print(f"Number of shards: {args.num_shards}")
    print(f"Number of nodes per shard: {args.nodes_per_shard}")
    print(f"Percentage of malicious nodes: {args.malicious_percentage}%")
    print(f"Attack scenario: {args.attack_scenario or 'No Attack'}")
    print(f"Number of simulation steps: {args.steps}")
    print(f"Number of transactions per step: {args.transactions_per_step}")
    if args.max_workers:
        print(f"Maximum number of workers: {args.max_workers}")
    print("=======================================\n")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Initialize and run simulation
    start_time = time.time()
    
    simulation = LargeScaleBlockchainSimulation(
        num_shards=args.num_shards,
        nodes_per_shard=args.nodes_per_shard,
        malicious_percentage=args.malicious_percentage,
        attack_scenario=args.attack_scenario,
        max_workers=args.max_workers
    )
    
    print(f"Initializing blockchain with {args.num_shards} shards, each with {args.nodes_per_shard} nodes")
    print(f"Percentage of malicious nodes: {args.malicious_percentage}%")
    
    metrics = simulation.run_simulation(num_steps=args.steps, 
                                     transactions_per_step=args.transactions_per_step)
    
    # Calculate running time
    elapsed_time = time.time() - start_time
    print(f"\nTotal simulation time: {elapsed_time:.2f} seconds")
    
    # Create subdirectory for results based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_subdir = os.path.join(args.save_dir, f"sim_{timestamp}")
    os.makedirs(result_subdir, exist_ok=True)
    
    # Save results and generate report
    if args.parallel_save:
        # Save results in parallel
        simulation.parallel_save_results(result_subdir)
    else:
        # Generate report and save results sequentially
        simulation.generate_report(result_subdir)
        
        # Only draw charts if not skipped
        if not args.skip_plots:
            simulation.plot_metrics(result_subdir)
    
    print(f"\nCompleted! Results saved at: {result_subdir}")
    
    # Return final performance metrics
    final_metrics = simulation.save_final_metrics(result_subdir)
    return final_metrics

if __name__ == "__main__":
    main() 