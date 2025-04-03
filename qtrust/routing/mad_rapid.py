import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set
import heapq
import random
import time
from collections import defaultdict, deque
import math

"""
Multi-Agent Dynamic Routing and Adaptive Path Intelligence Distribution (MAD-RAPID)

This module implements an intelligent cross-shard transaction routing algorithm for 
blockchain networks that optimizes for latency, congestion, energy efficiency, and security.
Key features include:
- Proximity-aware routing using geographical/logical coordinates
- Dynamic mesh connections for high-traffic shard pairs
- Predictive routing based on historical transaction patterns
- Congestion detection and avoidance
- Adaptive weight optimization based on network conditions
- Time-based traffic pattern analysis

The MAD-RAPID router provides significant improvements over traditional routing algorithms
by considering both network topology and transaction characteristics to find optimal paths.
"""

class MADRAPIDRouter:
    """
    Multi-Agent Dynamic Routing and Adaptive Path Intelligence Distribution (MAD-RAPID).
    Intelligent routing algorithm for cross-shard transactions in blockchain networks.
    Includes proximity-aware routing, dynamic mesh connections and predictive routing.
    """
    
    def __init__(self, 
                 network: nx.Graph,
                 shards: List[List[int]],
                 congestion_weight: float = 0.4,
                 latency_weight: float = 0.3,
                 energy_weight: float = 0.1,
                 trust_weight: float = 0.1,
                 prediction_horizon: int = 5,
                 congestion_threshold: float = 0.7,
                 proximity_weight: float = 0.3,  # Increased proximity influence
                 use_dynamic_mesh: bool = True,
                 predictive_window: int = 15,  # Increased prediction window
                 max_cache_size: int = 2000,  # Increased cache size
                 geo_awareness: bool = True,  # New: Geographical awareness
                 traffic_history_length: int = 100,  # New: Traffic history length
                 dynamic_connections_limit: int = 20,  # New: Dynamic connections limit
                 update_interval: int = 30):  # New: More frequent updates
        """
        Initialize MAD-RAPID router.
        
        Args:
            network: Blockchain network graph
            shards: List of shards and nodes in each shard
            congestion_weight: Weight for congestion level
            latency_weight: Weight for latency
            energy_weight: Weight for energy consumption
            trust_weight: Weight for trust scores
            prediction_horizon: Number of steps for future congestion prediction
            congestion_threshold: Threshold to consider congestion significant
            proximity_weight: Weight for geographical/logical proximity
            use_dynamic_mesh: Whether to use dynamic mesh connections
            predictive_window: Transaction steps to build prediction model
            max_cache_size: Maximum cache size
            geo_awareness: Geographical location awareness feature
            traffic_history_length: Number of steps to store traffic history
            dynamic_connections_limit: Maximum number of dynamic connections
            update_interval: Update interval (number of steps)
        """
        self.network = network
        self.shards = shards
        self.num_shards = len(shards)
        
        # Weights for different factors in routing decisions
        self.congestion_weight = congestion_weight
        self.latency_weight = latency_weight
        self.energy_weight = energy_weight
        self.trust_weight = trust_weight
        self.proximity_weight = proximity_weight
        
        # Congestion prediction parameters
        self.prediction_horizon = prediction_horizon
        self.congestion_threshold = congestion_threshold
        self.use_dynamic_mesh = use_dynamic_mesh
        self.predictive_window = predictive_window
        
        # New parameters
        self.geo_awareness = geo_awareness
        self.traffic_history_length = traffic_history_length
        self.dynamic_connections_limit = dynamic_connections_limit
        self.mesh_update_interval = update_interval
        
        # Congestion history for future congestion prediction
        self.congestion_history = [np.zeros(self.num_shards) for _ in range(prediction_horizon)]
        
        # Path cache and cache expiration time
        self.path_cache = {}
        self.cache_expire_time = 10  # Steps before cache expires
        self.max_cache_size = max_cache_size
        self.current_step = 0
        
        # Track same-shard transaction ratio
        self.same_shard_ratio = 0.8  # Target ratio of same-shard transactions
        
        # Build shard-level graph from network
        self.shard_graph = self._build_shard_graph()
        
        # Affinity matrix between shards - updated periodically
        self.shard_affinity = np.ones((self.num_shards, self.num_shards)) - np.eye(self.num_shards)
        
        # Geographic location matrix for shards - new
        self.geo_distance_matrix = np.zeros((self.num_shards, self.num_shards))
        self._calculate_geo_distance_matrix()
        
        # Storage for historical transactions
        self.transaction_history = deque(maxlen=self.traffic_history_length)
        
        # Dynamic mesh connections
        self.dynamic_connections = set()
        
        # Traffic statistics between shard pairs
        self.shard_pair_traffic = defaultdict(int)
        
        # Traffic history between shard pairs - new
        self.traffic_history = defaultdict(lambda: deque(maxlen=self.traffic_history_length))
        
        # Predictive routing model - new
        self.shard_traffic_pattern = {}  # Store traffic patterns for prediction
        
        # Temporal localization - new
        self.temporal_locality = defaultdict(lambda: defaultdict(float))  # (source, dest) -> time -> frequency
        
        # Last time dynamic mesh was updated
        self.last_mesh_update = 0
        
        # Time-based traffic history - new  
        self.time_based_traffic = defaultdict(lambda: {})  # time_bucket -> (source, dest) -> count
        
        # Previous optimal paths
        self.last_optimal_paths = {}
    
    def _calculate_geo_distance_matrix(self):
        """Calculate geographical distance matrix between shards."""
        for i in range(self.num_shards):
            for j in range(self.num_shards):
                if i == j:
                    self.geo_distance_matrix[i, j] = 0
                else:
                    pos_i = self.shard_graph.nodes[i]['position']
                    pos_j = self.shard_graph.nodes[j]['position']
                    distance = math.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                    
                    # Normalize distance to [0,1]
                    normalized_distance = min(1.0, distance / 100.0)
                    self.geo_distance_matrix[i, j] = normalized_distance
    
    def _build_shard_graph(self) -> nx.Graph:
        """
        Build shard-level graph from network graph with geographical/logical awareness.
        
        Returns:
            nx.Graph: Shard-level graph
        """
        shard_graph = nx.Graph()
        
        # Add shards as nodes
        for shard_id in range(self.num_shards):
            # Add position information for each shard (simulating logical or geographical coordinates)
            # Create random logical coordinates for shards to simulate geographical/logical position
            x_pos = random.uniform(0, 100)
            y_pos = random.uniform(0, 100)
            
            # Calculate zone based on geographical position - new
            zone_x = int(x_pos / 25)  # Divide space into 4x4 zones
            zone_y = int(y_pos / 25)
            zone = zone_x + zone_y * 4  # 16 zones total
            
            shard_graph.add_node(shard_id, 
                               congestion=0.0,
                               trust_score=0.0,
                               position=(x_pos, y_pos),
                               capacity=len(self.shards[shard_id]),
                               zone=zone,  # Add zone - new
                               processing_power=random.uniform(0.7, 1.0),  # Add processing power - new
                               stability=1.0)  # Shard stability - new
        
        # Calculate geographical distances between shards (based on logical coordinates)
        geographical_distances = {}
        for i in range(self.num_shards):
            pos_i = shard_graph.nodes[i]['position']
            zone_i = shard_graph.nodes[i]['zone']
            
            for j in range(i + 1, self.num_shards):
                pos_j = shard_graph.nodes[j]['position']
                zone_j = shard_graph.nodes[j]['zone']
                
                # Calculate Euclidean distance
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                geographical_distances[(i, j)] = distance
                geographical_distances[(j, i)] = distance
                
                # If in same zone, increase proximity factor - new
                same_zone_bonus = 0.3 if zone_i == zone_j else 0.0
        
        # Find connections between shards and calculate average latency/bandwidth
        for i in range(self.num_shards):
            for j in range(i + 1, self.num_shards):
                cross_shard_edges = []
                
                # Find all edges between nodes in the two shards
                for node_i in self.shards[i]:
                    for node_j in self.shards[j]:
                        if self.network.has_edge(node_i, node_j):
                            cross_shard_edges.append((node_i, node_j))
                
                # Proximity factor based on geographical distance
                geo_dist = geographical_distances.get((i, j), 50)
                norm_geo_dist = min(1.0, geo_dist / 100.0)
                
                # Check zone - new
                zone_i = shard_graph.nodes[i]['zone']
                zone_j = shard_graph.nodes[j]['zone']
                same_zone = zone_i == zone_j
                
                if cross_shard_edges:
                    # Add latency and bandwidth if not already present
                    for u, v in cross_shard_edges:
                        if 'latency' not in self.network.edges[u, v]:
                            # Calculate latency based on geographical distance
                            # Decrease latency when in same zone - new
                            zone_factor = 0.7 if same_zone else 1.0
                            base_latency = random.uniform(3, 18)  # Decrease base latency
                            geo_factor = norm_geo_dist  # Normalized to [0,1]
                            # Latency increases with geographical distance
                            self.network.edges[u, v]['latency'] = base_latency * (1 + geo_factor) * zone_factor
                        
                        if 'bandwidth' not in self.network.edges[u, v]:
                            # Bandwidth decreases with geographical distance
                            # Increase bandwidth when in same zone - new
                            zone_factor = 1.3 if same_zone else 1.0
                            base_bandwidth = random.uniform(10, 30)  # Increase base bandwidth
                            # Bandwidth decreases with geographical distance
                            self.network.edges[u, v]['bandwidth'] = base_bandwidth * (1 - 0.4 * norm_geo_dist) * zone_factor
                    
                    # Calculate average latency and bandwidth of connections
                    avg_latency = np.mean([self.network.edges[u, v]['latency'] for u, v in cross_shard_edges])
                    avg_bandwidth = np.mean([self.network.edges[u, v]['bandwidth'] for u, v in cross_shard_edges])
                    
                    # Calculate proximity factor based on geographical distance and number of connections
                    num_connections = len(cross_shard_edges)
                    
                    # Proximity factor increases when distance is short and number of connections is high
                    # Increase proximity influence and same zone - new
                    proximity_factor = (1 - norm_geo_dist * 0.8) * min(1.0, num_connections / 8.0)
                    if same_zone:
                        proximity_factor += 0.2  # Bonus for same zone
                    proximity_factor = min(1.0, proximity_factor)
                    
                    # Add edge between two shards with extended attributes
                    shard_graph.add_edge(i, j, 
                                       latency=avg_latency,
                                       bandwidth=avg_bandwidth,
                                       connection_count=len(cross_shard_edges),
                                       geographical_distance=geo_dist,
                                       proximity_factor=proximity_factor,
                                       historical_traffic=0,
                                       is_dynamic=False,
                                       stability=1.0,  # Stability of connection
                                       same_zone=same_zone,  # New: flag for same zone
                                       last_updated=time.time())
                
                # Automatically add virtual connection between shards in the same zone - new
                elif same_zone and self.geo_awareness:
                    virtual_latency = 25 + norm_geo_dist * 30  # Higher latency but still reasonable
                    virtual_bandwidth = 5 + (1 - norm_geo_dist) * 10
                    
                    shard_graph.add_edge(i, j,
                                       latency=virtual_latency,
                                       bandwidth=virtual_bandwidth,
                                       connection_count=0,  # No real connection
                                       geographical_distance=geo_dist,
                                       proximity_factor=0.3,  # Lower proximity
                                       historical_traffic=0,
                                       is_dynamic=False,
                                       is_virtual=True,  # Mark as virtual connection
                                       stability=0.7,  # Lower stability
                                       same_zone=True,
                                       last_updated=time.time())
        
        return shard_graph
    
    def update_network_state(self, shard_congestion: np.ndarray, node_trust_scores: Dict[int, float], transaction_batch: List[Dict[str, Any]] = None):
        """
        Update network state with new data and update dynamic mesh if needed.
        
        Args:
            shard_congestion: Array containing congestion levels of each shard
            node_trust_scores: Dictionary mapping node ID to trust score
            transaction_batch: Recent transactions to update historical data
        """
        # Update current step
        self.current_step += 1
        
        # Update congestion history
        self.congestion_history.pop(0)
        self.congestion_history.append(shard_congestion.copy())
        
        # Update current congestion level for each shard in shard_graph
        for shard_id in range(self.num_shards):
            self.shard_graph.nodes[shard_id]['congestion'] = shard_congestion[shard_id]
            
            # Calculate average trust score for shard
            shard_nodes = self.shards[shard_id]
            avg_trust = np.mean([node_trust_scores.get(node, 0.5) for node in shard_nodes])
            self.shard_graph.nodes[shard_id]['trust_score'] = avg_trust
        
        # Update historical transaction data if available
        if transaction_batch:
            # Limit historical data size
            max_history_size = self.predictive_window * 100
            if len(self.transaction_history) >= max_history_size:
                self.transaction_history = self.transaction_history[-max_history_size:]
            
            # Add new transactions to historical data
            self.transaction_history.extend(transaction_batch)
            
            # Update traffic statistics for shard pairs
            for tx in transaction_batch:
                if 'source_shard' in tx and 'destination_shard' in tx:
                    src = tx['source_shard']
                    dst = tx['destination_shard']
                    if src != dst:  # Only care about cross-shard transactions
                        pair = tuple(sorted([src, dst]))  # Ensure consistent order
                        self.shard_pair_traffic[pair] += 1
        
        # Clear cache if network state changes
        # Only clear old cache entries, not entire cache
        current_time = self.current_step
        expired_keys = [k for k, (path, timestamp) in self.path_cache.items() 
                        if current_time - timestamp > self.cache_expire_time]
        for k in expired_keys:
            del self.path_cache[k]
        
        # If cache is too large, remove oldest entries
        if len(self.path_cache) > self.max_cache_size:
            sorted_cache = sorted(self.path_cache.items(), key=lambda x: x[1][1])  # Sort by timestamp
            entries_to_remove = len(self.path_cache) - self.max_cache_size
            for k, _ in sorted_cache[:entries_to_remove]:
                del self.path_cache[k]
        
        # Cập nhật dynamic mesh
        self.update_dynamic_mesh()
    
    def _predict_congestion(self, shard_id: int) -> float:
        """
        Predict the future congestion level of a shard based on history.
        
        Args:
            shard_id: ID of the shard to predict
            
        Returns:
            float: Predicted congestion level
        """
        # Get congestion history for the specific shard
        congestion_values = [history[shard_id] for history in self.congestion_history]
        
        if not congestion_values:
            return 0.0
        
        # Advanced prediction with multiple models
        
        # Model 1: Time-weighted average
        # Calculate weights decreasing with time (more recent = more important)
        weights = np.exp(np.linspace(0, 2, len(congestion_values)))  # Increase slope from 1 to 2
        weights = weights / np.sum(weights)
        predicted_congestion_1 = np.sum(weights * congestion_values)
        
        # Model 2: Linear trend prediction
        if len(congestion_values) >= 3:
            # Get differences between consecutive values
            diffs = np.diff(congestion_values[-3:])
            # Calculate average trend
            avg_trend = np.mean(diffs)
            # Predict next based on the last value and trend
            predicted_congestion_2 = congestion_values[-1] + avg_trend
            # Limit to [0, 1] range
            predicted_congestion_2 = np.clip(predicted_congestion_2, 0.0, 1.0)
        else:
            predicted_congestion_2 = congestion_values[-1]
        
        # Model 3: Simplified ARMA(1,1)
        if len(congestion_values) >= 3:
            # AR and MA parameters
            ar_param = 0.7
            ma_param = 0.3
            error = congestion_values[-1] - congestion_values[-2] if len(congestion_values) >= 2 else 0
            predicted_congestion_3 = ar_param * congestion_values[-1] + ma_param * error
            predicted_congestion_3 = np.clip(predicted_congestion_3, 0.0, 1.0)
        else:
            predicted_congestion_3 = congestion_values[-1]
        
        # Model 4: Double Exponential Smoothing (Holt's method)
        if len(congestion_values) >= 4:
            alpha = 0.3  # Smoothing parameter
            beta = 0.2   # Trend parameter
            
            # Initial smoothed values
            s_prev = congestion_values[-3]
            b_prev = congestion_values[-2] - congestion_values[-3]
            
            # Update for final step
            s_curr = alpha * congestion_values[-1] + (1 - alpha) * (s_prev + b_prev)
            b_curr = beta * (s_curr - s_prev) + (1 - beta) * b_prev
            
            # Predict next step
            predicted_congestion_4 = s_curr + b_curr
            predicted_congestion_4 = np.clip(predicted_congestion_4, 0.0, 1.0)
        else:
            predicted_congestion_4 = congestion_values[-1]
        
        # Combine predictions with weights based on history length
        # Longer history = more trust in complex models
        if len(congestion_values) >= 5:
            # Enough data for complex models
            final_prediction = 0.2 * predicted_congestion_1 + 0.3 * predicted_congestion_2 + 0.2 * predicted_congestion_3 + 0.3 * predicted_congestion_4
        elif len(congestion_values) >= 3:
            # Medium data
            final_prediction = 0.3 * predicted_congestion_1 + 0.3 * predicted_congestion_2 + 0.2 * predicted_congestion_3 + 0.2 * predicted_congestion_4
        else:
            # Limited data
            final_prediction = 0.6 * predicted_congestion_1 + 0.4 * predicted_congestion_2
        
        # Adjust based on volatility
        if len(congestion_values) >= 3:
            variance = np.var(congestion_values[-3:])
            # If high volatility, add safety factor
            if variance > 0.05:
                final_prediction += 0.1 * variance
        
        # Network factor analysis
        # Check number of connections to this shard
        num_connections = sum(1 for u, v, data in self.shard_graph.edges(data=True) if u == shard_id or v == shard_id)
        
        # Capacity factor - each shard has different number of nodes
        shard_capacity = len(self.shards[shard_id])
        capacity_factor = max(0.0, 0.1 * (1.0 - shard_capacity / max(shard_capacity for shard in self.shards)))
        
        # Adjust prediction based on connections and capacity
        connection_factor = max(0.0, 0.1 * (1.0 - num_connections / max(1, self.num_shards)))
        final_prediction += connection_factor + capacity_factor
        
        # Check if connected shards are congested
        connected_shards = [node for node in self.shard_graph.neighbors(shard_id)]
        if connected_shards:
            neighbor_congestion = np.mean([self.shard_graph.nodes[s]['congestion'] for s in connected_shards])
            # If neighboring shards are congested, this shard is likely to be affected
            if neighbor_congestion > 0.6:
                final_prediction += 0.1 * neighbor_congestion
        
        # Ensure value is within [0, 1] range
        return np.clip(final_prediction, 0.0, 1.0)
    
    def _calculate_path_cost(self, path: List[int], transaction: Dict[str, Any]) -> float:
        """
        Calculate path cost based on performance factors and proximity.
        
        Args:
            path: The path as a list of shard IDs
            transaction: The transaction to be routed
            
        Returns:
            float: The combined cost of the path
        """
        if len(path) < 2:
            return 0.0
        
        # Penalize long paths to minimize cross-shard transactions
        # Strongly increase penalty for longer paths to favor shorter paths
        path_length_penalty = (len(path) - 2) * 0.5  # Increased from 0.3 to 0.5
        
        total_latency = 0.0
        total_energy = 0.0
        total_congestion = 0.0
        total_trust = 0.0
        total_proximity = 0.0
        
        for i in range(len(path) - 1):
            shard_from = path[i]
            shard_to = path[i + 1]
            
            # If there's no direct connection between the two shards, return high cost
            if not self.shard_graph.has_edge(shard_from, shard_to):
                return float('inf')
            
            # Get edge data
            edge_data = self.shard_graph.edges[shard_from, shard_to]
            
            # Calculate edge latency
            latency = edge_data['latency']
            total_latency += latency
            
            # Calculate energy consumption (based on latency and bandwidth)
            bandwidth = edge_data['bandwidth']
            energy = (latency / 10.0) * (10.0 / bandwidth)
            total_energy += energy
            
            # Calculate predicted congestion for each shard
            congestion_from = self._predict_congestion(shard_from)
            congestion_to = self._predict_congestion(shard_to)
            
            # Calculate average edge congestion
            edge_congestion = (congestion_from + congestion_to) / 2.0
            total_congestion += edge_congestion
            
            # Calculate average trust scores for the two shards
            trust_from = self.shard_graph.nodes[shard_from].get('trust_score', 0.5)
            trust_to = self.shard_graph.nodes[shard_to].get('trust_score', 0.5)
            
            # High trust score reduces cost, so we take 1 - trust
            edge_risk = 1.0 - ((trust_from + trust_to) / 2.0)
            total_trust += edge_risk
            
            # Add proximity factor cost
            proximity_factor = edge_data.get('proximity_factor', 0.5)
            
            # High proximity reduces cost, so we take 1 - proximity_factor
            edge_proximity_cost = 1.0 - proximity_factor
            total_proximity += edge_proximity_cost
            
            # Consider connection stability parameter
            stability = edge_data.get('stability', 1.0)
            
            # Favor stable connections
            if stability < 0.8:
                total_latency *= (2.0 - stability)  # Increase cost if connection is unstable
            
            # Favor dynamic connections created to optimize for high traffic
            if edge_data.get('is_dynamic', False):
                total_latency *= 0.8  # Reduce cost for dynamic connections
        
        # If there's special information in the transaction, adjust costs accordingly
        # For example: transactions requiring low latency, high security, or energy efficiency
        tx_priority = transaction.get('priority', 'normal')
        if tx_priority == 'low_latency':
            total_latency *= 1.5  # Increase weight for latency
        elif tx_priority == 'high_security':
            total_trust *= 1.5  # Increase weight for security
        elif tx_priority == 'energy_efficient':
            total_energy *= 1.5  # Increase weight for energy
        
        # Calculate combined cost
        total_cost = (
            self.latency_weight * total_latency +
            self.energy_weight * total_energy +
            self.congestion_weight * total_congestion +
            self.trust_weight * total_trust +
            self.proximity_weight * total_proximity +
            path_length_penalty
        )
        
        return total_cost
    
    def _dijkstra(self, source_shard: int, dest_shard: int, transaction: Dict[str, Any]) -> List[int]:
        """
        Modified Dijkstra algorithm to find optimal path between shards.
        
        Args:
            source_shard: Source shard
            dest_shard: Destination shard
            transaction: Transaction to be routed
            
        Returns:
            List[int]: Optimal path as a list of shard IDs
        """
        # Check cache
        cache_key = (source_shard, dest_shard, transaction['value'])
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Initialization
        distances = {shard: float('inf') for shard in range(self.num_shards)}
        distances[source_shard] = 0
        previous = {shard: None for shard in range(self.num_shards)}
        priority_queue = [(0, source_shard)]
        
        while priority_queue:
            current_distance, current_shard = heapq.heappop(priority_queue)
            
            # If we've reached the destination, exit the loop
            if current_shard == dest_shard:
                break
            
            # If current distance is greater than known distance, skip
            if current_distance > distances[current_shard]:
                continue
            
            # Explore adjacent shards
            for neighbor in self.shard_graph.neighbors(current_shard):
                # Build temporary path to neighbor
                temp_path = self._reconstruct_path(previous, current_shard)
                temp_path.append(neighbor)
                
                # Calculate path cost
                path_cost = self._calculate_path_cost(temp_path, transaction)
                
                # If we found a better path
                if path_cost < distances[neighbor]:
                    distances[neighbor] = path_cost
                    previous[neighbor] = current_shard
                    heapq.heappush(priority_queue, (path_cost, neighbor))
        
        # Construct path from source to destination
        path = self._reconstruct_path(previous, dest_shard)
        
        # Save to cache
        self.path_cache[cache_key] = path
        
        return path
    
    def _reconstruct_path(self, previous: Dict[int, int], end: int) -> List[int]:
        """
        Reconstruct path from previous dict.
        
        Args:
            previous: Dictionary mapping from node to its predecessor in the path
            end: The last node in the path
            
        Returns:
            List[int]: Complete path
        """
        path = []
        current = end
        
        while current is not None:
            path.append(current)
            current = previous[current]
        
        # Reverse the path to get source to destination
        return path[::-1]
    
    def find_optimal_path(self, 
                         transaction: Dict[str, Any], 
                         source_shard: int = None, 
                         destination_shard: int = None,
                         prioritize_security: bool = False) -> List[int]:
        """
        Find optimal path for a transaction between two shards.
        
        Args:
            transaction: Transaction to be routed
            source_shard: Source shard (if not provided, will use prediction)
            destination_shard: Destination shard (if not provided, will use prediction)
            prioritize_security: Prioritize security over performance
            
        Returns:
            List[int]: List of shard IDs forming the optimal path
        """
        # Use predictive routing if source or destination not provided
        if source_shard is None or destination_shard is None:
            src, dst, confidence = self.predictive_routing(transaction)
            source_shard = src if source_shard is None else source_shard
            destination_shard = dst if destination_shard is None else destination_shard
        
        # Ensure source and destination are in valid range
        source_shard = max(0, min(source_shard, self.num_shards - 1))
        destination_shard = max(0, min(destination_shard, self.num_shards - 1))
        
        # Special case: source and destination are the same shard
        if source_shard == destination_shard:
            return [source_shard]
        
        # Check cache first
        cache_key = (source_shard, destination_shard, str(prioritize_security), 
                    transaction.get('priority', 'normal'))
        
        if cache_key in self.path_cache:
            path, timestamp = self.path_cache[cache_key]
            # Check if cache is still valid
            if self.current_step - timestamp <= self.cache_expire_time:
                return path
        
        # Prepare temporary weights if we need to prioritize security
        tmp_weights = {
            'congestion': self.congestion_weight,
            'latency': self.latency_weight,
            'energy': self.energy_weight,
            'trust': self.trust_weight,
            'proximity': self.proximity_weight
        }
        
        if prioritize_security:
            # Increase weight for trust, decrease for performance
            tmp_weights['trust'] *= 2.0
            tmp_weights['latency'] *= 0.5
            tmp_weights['energy'] *= 0.5
        
        # Consider priority in transaction
        tx_priority = transaction.get('priority', 'normal')
        if tx_priority == 'low_latency':
            tmp_weights['latency'] *= 1.5
        elif tx_priority == 'high_security':
            tmp_weights['trust'] *= 1.5
        elif tx_priority == 'energy_efficient':
            tmp_weights['energy'] *= 1.5
        
        # Save current weights
        old_weights = {
            'congestion': self.congestion_weight,
            'latency': self.latency_weight,
            'energy': self.energy_weight,
            'trust': self.trust_weight,
            'proximity': self.proximity_weight
        }
        
        # Set temporary weights
        self.congestion_weight = tmp_weights['congestion'] 
        self.latency_weight = tmp_weights['latency']
        self.energy_weight = tmp_weights['energy']
        self.trust_weight = tmp_weights['trust']
        self.proximity_weight = tmp_weights['proximity']
        
        # Find optimal path
        path = self._dijkstra(source_shard, destination_shard, transaction)
        
        # Restore weights
        self.congestion_weight = old_weights['congestion']
        self.latency_weight = old_weights['latency']
        self.energy_weight = old_weights['energy']
        self.trust_weight = old_weights['trust']
        self.proximity_weight = old_weights['proximity']
        
        # Save to cache
        self.path_cache[cache_key] = (path, self.current_step)
        
        # Control cache size
        if len(self.path_cache) > self.max_cache_size:
            # Remove oldest entry
            oldest_entry = min(self.path_cache.items(), key=lambda x: x[1][1])
            del self.path_cache[oldest_entry[0]]
        
        return path
    
    def detect_congestion_hotspots(self) -> List[int]:
        """
        Detect congestion hotspots in the network.
        
        Returns:
            List[int]: List of shard IDs that are congested
        """
        hotspots = []
        
        for shard_id in range(self.num_shards):
            # Predict congestion
            predicted_congestion = self._predict_congestion(shard_id)
            
            # If predicted congestion exceeds threshold, consider it a hotspot
            if predicted_congestion > self.congestion_threshold:
                hotspots.append(shard_id)
        
        return hotspots
    
    def find_optimal_paths_for_transactions(self, transaction_pool: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """
        Find optimal paths for a set of transactions.
        
        Improvements implemented:
        1. Proximity-aware routing: Uses geographical/logical position information to optimize routing
        2. Dynamic mesh connections: Creates direct connections between high-traffic shard pairs
        3. Predictive routing: Predicts optimal routes based on transaction history
        4. Enhanced congestion prediction: Uses multiple combined prediction models
        5. Transaction pattern analysis: Optimizes routing based on traffic patterns
        
        Args:
            transaction_pool: List of transactions to route
            
        Returns:
            Dict[int, List[int]]: Dictionary mapping from transaction ID to optimal path
        """
        # Ensure transaction_history is initialized as a list
        if not hasattr(self, 'transaction_history') or not isinstance(self.transaction_history, list):
            self.transaction_history = []
        
        # Update transaction history and analyze patterns
        self.transaction_history.extend(transaction_pool)
        if len(self.transaction_history) > 1000:
            self.transaction_history = self.transaction_history[-1000:]
        
        # Analyze transaction patterns to optimize routing
        pattern_analysis = self.analyze_transaction_patterns()
        
        # Check pattern_analysis results
        if isinstance(pattern_analysis, dict) and pattern_analysis.get('patterns_found') and pattern_analysis.get('high_traffic_pairs'):
            for pair in pattern_analysis['high_traffic_pairs']:
                if hasattr(self, 'shard_pair_traffic'):
                    self.shard_pair_traffic[pair] += 5  # Increase priority for high-traffic shard pairs
        
        # Update dynamic mesh if needed
        if self.use_dynamic_mesh and (self.current_step % 10 == 0):
            self.update_dynamic_mesh()
        
        # Separate intra-shard and cross-shard transactions
        intra_shard_txs = []
        cross_shard_txs = []
        
        for tx in transaction_pool:
            if 'source_shard' in tx and 'destination_shard' in tx:
                if tx['source_shard'] == tx['destination_shard']:
                    intra_shard_txs.append(tx)
                else:
                    cross_shard_txs.append(tx)
            else:
                # Use predictive routing to predict source and destination
                src, dst, confidence = self.predictive_routing(tx)
                
                # Update transaction with predicted information
                tx['source_shard'] = src
                tx['destination_shard'] = dst
                tx['predicted_route'] = True
                tx['prediction_confidence'] = confidence
                
                if src == dst:
                    intra_shard_txs.append(tx)
                else:
                    cross_shard_txs.append(tx)
        
        # Sort cross-shard transactions by priority level
        # and group by source-destination shard pair to avoid redundant calculations
        cross_shard_txs.sort(key=lambda tx: tx.get('priority_level', 0), reverse=True)
        
        # Group transactions by source-destination shard pair
        route_groups = defaultdict(list)
        for tx in cross_shard_txs:
            route_key = (tx['source_shard'], tx['destination_shard'])
            route_groups[route_key].append(tx)
        
        # Dictionary to store optimal path for each transaction
        tx_paths = {}
        
        # Process intra-shard transactions (only 1 shard in path)
        for tx in intra_shard_txs:
            if 'id' in tx:
                tx_paths[tx['id']] = [tx['source_shard']]
        
        # Group transactions by priority for load balancing
        high_priority_txs = []
        normal_priority_txs = []
        low_priority_txs = []
        
        for txs in route_groups.values():
            for tx in txs:
                priority_level = tx.get('priority_level', 0)
                if priority_level > 8:
                    high_priority_txs.append(tx)
                elif priority_level > 4:
                    normal_priority_txs.append(tx)
                else:
                    low_priority_txs.append(tx)
        
        # Process transactions in order of priority
        # High priority transactions processed first with focus on performance
        for tx in high_priority_txs:
            path = self.find_optimal_path(
                tx, 
                tx['source_shard'], 
                tx['destination_shard'],
                prioritize_security=False
            )
            if 'id' in tx:
                tx_paths[tx['id']] = path
        
        # Medium priority transactions processed next with balanced performance and security
        for tx in normal_priority_txs:
            path = self.find_optimal_path(
                tx, 
                tx['source_shard'], 
                tx['destination_shard'],
                prioritize_security=tx.get('prioritize_security', False)
            )
            if 'id' in tx:
                tx_paths[tx['id']] = path
        
        # Low priority transactions processed last with focus on security
        for tx in low_priority_txs:
            path = self.find_optimal_path(
                tx, 
                tx['source_shard'], 
                tx['destination_shard'],
                prioritize_security=True
            )
            if 'id' in tx:
                tx_paths[tx['id']] = path
        
        # Detect congestion hotspots to alert the system
        congestion_hotspots = self.detect_congestion_hotspots()
        if congestion_hotspots:
            print(f"Alert: Congestion hotspots detected at shards: {congestion_hotspots}")
        
        return tx_paths
    
    def optimize_routing_weights(self, 
                               recent_metrics: Dict[str, List[float]], 
                               target_latency: float = 0.0, 
                               target_energy: float = 0.0):
        """
        Optimize routing weights based on recent metrics and targets.
        
        Args:
            recent_metrics: Dictionary containing recent performance metrics
            target_latency: Latency target (0.0 = no limit)
            target_energy: Energy consumption target (0.0 = no limit)
        """
        # If recent latency is high and target_latency > 0
        if target_latency > 0 and 'latency' in recent_metrics:
            avg_latency = np.mean(recent_metrics['latency'])
            if avg_latency > target_latency:
                # Increase weight for latency
                self.latency_weight = min(0.6, self.latency_weight * 1.2)
                
                # Reduce other weights so total = 1.0
                total_other = self.congestion_weight + self.energy_weight + self.trust_weight
                scale_factor = (1.0 - self.latency_weight) / total_other
                
                self.congestion_weight *= scale_factor
                self.energy_weight *= scale_factor
                self.trust_weight *= scale_factor
        
        # If recent energy consumption is high and target_energy > 0
        if target_energy > 0 and 'energy_consumption' in recent_metrics:
            avg_energy = np.mean(recent_metrics['energy_consumption'])
            if avg_energy > target_energy:
                # Increase weight for energy
                self.energy_weight = min(0.5, self.energy_weight * 1.2)
                
                # Reduce other weights so total = 1.0
                total_other = self.congestion_weight + self.latency_weight + self.trust_weight
                scale_factor = (1.0 - self.energy_weight) / total_other
                
                self.congestion_weight *= scale_factor
                self.latency_weight *= scale_factor
                self.trust_weight *= scale_factor
        
        # Ensure total weights = 1.0
        total_weight = self.congestion_weight + self.latency_weight + self.energy_weight + self.trust_weight
        if abs(total_weight - 1.0) > 1e-6:
            scale = 1.0 / total_weight
            self.congestion_weight *= scale
            self.latency_weight *= scale
            self.energy_weight *= scale
            self.trust_weight *= scale
        
        # Clear cache when weights change
        self.path_cache = {} 
    
    def update_dynamic_mesh(self):
        """
        Update dynamic mesh connections based on transaction traffic.
        Creates direct connections between high-traffic shard pairs.
        """
        current_time = time.time()
        
        # Only update mesh after a certain time interval
        if not self.use_dynamic_mesh or (current_time - self.last_mesh_update < self.mesh_update_interval):
            return
        
        self.last_mesh_update = current_time
        
        # Perform transaction pattern analysis before updating mesh
        pattern_analysis = self.analyze_transaction_patterns(window_size=min(self.traffic_history_length, 50))
        
        # Identify shard pairs with highest traffic
        top_traffic_pairs = sorted(self.shard_pair_traffic.items(), key=lambda x: x[1], reverse=True)
        
        # Consider time-based patterns to detect temporal hotspots
        time_based_hotspots = []
        for time_bucket, traffic_map in self.time_based_traffic.items():
            if traffic_map:  # Check if not empty
                # Get top 3 shard pairs for each time bucket
                hotspots = sorted(traffic_map.items(), key=lambda x: x[1], reverse=True)[:3]
                time_based_hotspots.extend([pair for pair, _ in hotspots])
        
        # Limit maximum number of dynamic connections
        max_dynamic_connections = min(self.dynamic_connections_limit, 
                                     self.num_shards * (self.num_shards - 1) // 3)
        
        # Remove old dynamic connections
        old_connections = list(self.dynamic_connections)
        for i, j in old_connections:
            if self.shard_graph.has_edge(i, j):
                # If there's a direct connection between the 2 shards, update attributes
                if 'is_dynamic' in self.shard_graph.edges[i, j]:
                    self.shard_graph.edges[i, j]['is_dynamic'] = False
            self.dynamic_connections.remove((i, j))
        
        # Add new connections for high-traffic shard pairs
        new_connections_count = 0
        
        # Prioritize pairs from time_based_hotspots
        priority_pairs = []
        for pair in time_based_hotspots:
            if pair not in priority_pairs:
                priority_pairs.append(pair)
        
        # Add pairs from top_traffic_pairs
        for (i, j), traffic in top_traffic_pairs:
            if (i, j) not in priority_pairs and (j, i) not in priority_pairs:
                priority_pairs.append((i, j))
        
        # Go through the priority list and add dynamic connections
        for i, j in priority_pairs:
            # If we've reached the maximum number of connections, stop
            if new_connections_count >= max_dynamic_connections:
                break
            
            # Sort to ensure i < j
            if i > j:
                i, j = j, i
                
            # Get transaction traffic
            traffic = self.shard_pair_traffic.get((i, j), 0) + self.shard_pair_traffic.get((j, i), 0)
            
            # Skip if traffic is too low
            if traffic < 5:  # Minimum threshold for creating dynamic connection
                continue
                
            # Skip if there's already a direct connection between the 2 shards
            if self.shard_graph.has_edge(i, j):
                self.shard_graph.edges[i, j]['is_dynamic'] = True
                self.shard_graph.edges[i, j]['historical_traffic'] = traffic
                
                # Update connection attributes based on traffic data
                if traffic > 20:  # High traffic
                    # Reduce latency and increase bandwidth
                    self.shard_graph.edges[i, j]['latency'] *= 0.8
                    self.shard_graph.edges[i, j]['bandwidth'] *= 1.2
                
                self.dynamic_connections.add((i, j))
                new_connections_count += 1
                continue
                
            # Calculate attributes for the new connection
            pos_i = self.shard_graph.nodes[i]['position']
            pos_j = self.shard_graph.nodes[j]['position']
            geo_dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
            
            # Check zone - new
            zone_i = self.shard_graph.nodes[i].get('zone', -1)
            zone_j = self.shard_graph.nodes[j].get('zone', -1)
            same_zone = zone_i == zone_j and zone_i != -1
            
            # Calculate proximity factor - new
            normalized_dist = min(1.0, geo_dist / 100.0)
            proximity_factor = 0.5 + 0.5 * (1 - normalized_dist)
            if same_zone:
                proximity_factor += 0.2  # Bonus for same zone
            proximity_factor = min(1.0, proximity_factor)
            
            # Better latency and bandwidth for dynamic connections as they're optimized
            zone_factor = 0.8 if same_zone else 1.0
            latency = (5 + normalized_dist * 15) * zone_factor  # Lower latency for dynamic connection
            bandwidth = (15 + (1 - normalized_dist) * 15) * (1.2 if same_zone else 1.0)  # Higher bandwidth
            
            # Calculate stability based on traffic stability
            stability = 0.7
            if (i, j) in self.traffic_history or (j, i) in self.traffic_history:
                # If there's traffic history, calculate stability based on variation
                history = self.traffic_history.get((i, j), self.traffic_history.get((j, i), []))
                if len(history) > 5:
                    # Calculate coefficient of variation (CV)
                    std_dev = np.std(history)
                    mean = np.mean(history)
                    if mean > 0:
                        cv = std_dev / mean
                        # High stability when CV is low
                        stability = max(0.5, min(0.95, 1.0 - cv))
            
            # Add new edge to shard graph
            self.shard_graph.add_edge(i, j,
                                     latency=latency,
                                     bandwidth=bandwidth,
                                     connection_count=1,
                                     geographical_distance=geo_dist,
                                     proximity_factor=proximity_factor,
                                     historical_traffic=traffic,
                                     is_dynamic=True,
                                     same_zone=same_zone,  # New
                                     stability=stability,
                                     last_updated=current_time)
            
            # Add to dynamic connections list
            self.dynamic_connections.add((i, j))
            new_connections_count += 1
        
        # If we've created at least one new connection, log info
        if new_connections_count > 0:
            dynamic_edges = [(i, j, self.shard_graph.edges[i, j]['latency'], 
                             self.shard_graph.edges[i, j]['bandwidth']) 
                            for i, j in self.dynamic_connections]
            try:
                # Try to print without raising error if in test mode
                print(f"Created {new_connections_count} dynamic mesh connections: {dynamic_edges}")
            except:
                pass
        
        # Update dynamic weights for routing algorithm based on traffic - new
        self._update_routing_weights(pattern_analysis)
    
    def _update_routing_weights(self, pattern_analysis):
        """
        Update routing weights based on traffic pattern analysis.
        
        Args:
            pattern_analysis: Traffic pattern analysis results
        """
        if not pattern_analysis.get('patterns_found', False):
            return
            
        # Adjust weights based on same-shard transaction ratio
        same_shard_ratio = pattern_analysis.get('same_shard_ratio', 0.8)
        
        # If same-shard transaction ratio is low, increase proximity_weight
        if same_shard_ratio < 0.5:
            new_proximity_weight = min(0.4, self.proximity_weight + 0.05)
            print(f"Adjusting proximity_weight: {self.proximity_weight:.2f} -> {new_proximity_weight:.2f}")
            self.proximity_weight = new_proximity_weight
            
            # Decrease congestion_weight correspondingly
            self.congestion_weight = max(0.2, 1.0 - self.proximity_weight - self.latency_weight - self.energy_weight - self.trust_weight)
        
        # If there are many cross-shard transactions between zones, increase geo_awareness
        high_traffic_pairs = pattern_analysis.get('high_traffic_pairs', [])
        if high_traffic_pairs:
            # Check if high-traffic pairs are in the same zone
            different_zone_pairs = 0
            for i, j in high_traffic_pairs:
                if i < len(self.shard_graph.nodes) and j < len(self.shard_graph.nodes):
                    zone_i = self.shard_graph.nodes[i].get('zone', -1)
                    zone_j = self.shard_graph.nodes[j].get('zone', -1)
                    if zone_i != zone_j or zone_i == -1:
                        different_zone_pairs += 1
            
            # If many pairs are in different zones, increase geo_awareness
            if different_zone_pairs > len(high_traffic_pairs) / 2:
                self.geo_awareness = True
    
    def predictive_routing(self, transaction: Dict[str, Any]) -> Tuple[int, int, bool]:
        """
        Predict optimal path based on transaction history and patterns.
        
        Args:
            transaction: Transaction to route
            
        Returns:
            Tuple[int, int, bool]: (next_shard, final_shard, is_direct_route)
            - next_shard: Next shard to forward the transaction
            - final_shard: Final destination shard
            - is_direct_route: True if direct route (no intermediate shards)
        """
        source_shard = transaction['source_shard']
        dest_shard = transaction['destination_shard']
        tx_value = transaction.get('value', 0.0)
        tx_priority = transaction.get('priority', 0.5)
        tx_type = transaction.get('type', 'intra_shard')
        tx_category = transaction.get('category', 'simple_transfer')
        
        # Create ID for transaction pair
        tx_pair_id = f"{source_shard}_{dest_shard}"
        
        # If intra-shard transaction, return source_shard as destination
        if source_shard == dest_shard:
            return source_shard, dest_shard, True
        
        # Check cache first
        cache_key = f"{source_shard}_{dest_shard}_{tx_category}_{round(tx_value)}"
        if cache_key in self.path_cache and self.current_step - self.path_cache[cache_key]['step'] < self.cache_expire_time:
            cached_path = self.path_cache[cache_key]['path']
            if len(cached_path) > 1:
                return cached_path[1], cached_path[-1], len(cached_path) == 2
        
        # Check proximity first
        if self.geo_awareness:
            geo_distance = self.geo_distance_matrix[source_shard, dest_shard]
            
            # If two shards are geographically close and have a direct connection
            if geo_distance < 0.4 and self.shard_graph.has_edge(source_shard, dest_shard):
                direct_edge = self.shard_graph.edges[source_shard, dest_shard]
                
                # Check conditions to use direct connection
                if direct_edge.get('bandwidth', 0) > 5 and direct_edge.get('stability', 0) > 0.5:
                    return dest_shard, dest_shard, True
        
        # Check temporal patterns
        current_time_bucket = self.current_step % (24 * 60)  # Simulate 24 hours with 1 minute per step
        time_slot = current_time_bucket // 60  # Divide into 24 time slots
        
        # If there's a temporal pattern for this shard pair
        if tx_pair_id in self.temporal_locality and time_slot in self.temporal_locality[tx_pair_id]:
            frequency = self.temporal_locality[tx_pair_id][time_slot]
            
            # If high frequency, find fastest path
            if frequency > 0.5:
                # Find fastest path
                fast_path = self._find_fastest_path(source_shard, dest_shard)
                if fast_path and len(fast_path) > 1:
                    # Save to cache
                    self.path_cache[cache_key] = {
                        'path': fast_path,
                        'step': self.current_step,
                        'type': 'temporal_pattern'
                    }
                    return fast_path[1], fast_path[-1], len(fast_path) == 2
        
        # Check recent transaction history
        if len(self.transaction_history) > 0:
            # Filter similar recent transactions
            similar_txs = []
            for tx in self.transaction_history[-100:]:
                if (tx['source_shard'] == source_shard and 
                    tx['destination_shard'] == dest_shard and
                    tx.get('category', '') == tx_category and
                    abs(tx.get('value', 0) - tx_value) / max(1, tx_value) < 0.3):  # Value differs <30%
                    similar_txs.append(tx)
            
            if similar_txs:
                # Find most recent transaction with same source and destination
                latest_similar_tx = max(similar_txs, key=lambda tx: tx.get('created_at', 0))
                
                # If it has a routed_path and was successful
                if 'routed_path' in latest_similar_tx and latest_similar_tx.get('status', '') == 'completed':
                    historical_path = latest_similar_tx['routed_path']
                    
                    # Check if this path is still valid
                    is_valid_path = True
                    for i in range(len(historical_path) - 1):
                        if not self.shard_graph.has_edge(historical_path[i], historical_path[i+1]):
                            is_valid_path = False
                            break
                    
                    if is_valid_path and len(historical_path) > 1:
                        # Save to cache
                        self.path_cache[cache_key] = {
                            'path': historical_path,
                            'step': self.current_step,
                            'type': 'historical'
                        }
                        return historical_path[1], historical_path[-1], len(historical_path) == 2
        
        # Consider dynamic mesh connections
        if self.use_dynamic_mesh and len(self.dynamic_connections) > 0:
            # Check if there's a direct dynamic connection
            if (source_shard, dest_shard) in self.dynamic_connections or (dest_shard, source_shard) in self.dynamic_connections:
                return dest_shard, dest_shard, True
            
            # Find path through dynamic mesh
            dynamic_path = self._find_dynamic_mesh_path(source_shard, dest_shard)
            if dynamic_path and len(dynamic_path) > 1:
                # Save to cache
                self.path_cache[cache_key] = {
                    'path': dynamic_path,
                    'step': self.current_step,
                    'type': 'dynamic_mesh'
                }
                return dynamic_path[1], dynamic_path[-1], len(dynamic_path) == 2
        
        # If all methods above fail, use traditional Dijkstra
        optimal_path = self._dijkstra(source_shard, dest_shard, transaction)
        
        # Save path to cache
        if optimal_path and len(optimal_path) > 1:
            self.path_cache[cache_key] = {
                'path': optimal_path,
                'step': self.current_step,
                'type': 'dijkstra'
            }
            return optimal_path[1], optimal_path[-1], len(optimal_path) == 2
        
        # If no path found, return direct destination
        return dest_shard, dest_shard, True
    
    def _find_fastest_path(self, source_shard: int, dest_shard: int) -> List[int]:
        """
        Find the fastest path between two shards.
        
        Args:
            source_shard: Source shard
            dest_shard: Destination shard
            
        Returns:
            List[int]: Fastest path (list of shard IDs)
        """
        # If there's a direct path, use it
        if self.shard_graph.has_edge(source_shard, dest_shard):
            return [source_shard, dest_shard]
        
        # Use Dijkstra algorithm focused on latency
        visited = set()
        distances = {source_shard: 0}
        previous = {}
        priority_queue = [(0, source_shard)]
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            if current_node == dest_shard:
                # Found path to destination
                return self._reconstruct_path(previous, dest_shard)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            for neighbor in self.shard_graph.neighbors(current_node):
                if neighbor in visited:
                    continue
                
                # Get connection latency
                edge_latency = self.shard_graph.edges[current_node, neighbor].get('latency', 50)
                
                # Prioritize dynamic mesh connections
                if self.shard_graph.edges[current_node, neighbor].get('is_dynamic', False):
                    edge_latency *= 0.7  # Reduce latency for dynamic connections
                
                # Calculate new distance
                new_distance = current_distance + edge_latency
                
                # If we found a shorter path or no path exists yet
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))
        
        # If no path found
        return []
    
    def _find_dynamic_mesh_path(self, source_shard: int, dest_shard: int) -> List[int]:
        """
        Find path through dynamic mesh connections.
        
        Args:
            source_shard: Source shard
            dest_shard: Destination shard
            
        Returns:
            List[int]: Path through dynamic mesh (list of shard IDs)
        """
        # Create dynamic mesh graph
        dynamic_graph = nx.Graph()
        
        # Add all nodes
        for i in range(self.num_shards):
            dynamic_graph.add_node(i)
        
        # Add dynamic connections
        for i, j in self.dynamic_connections:
            dynamic_graph.add_edge(i, j)
        
        # Add direct connections between shards
        for i, j in self.shard_graph.edges():
            dynamic_graph.add_edge(i, j)
        
        # Use networkx shortest path algorithm
        try:
            # Check if path exists
            if nx.has_path(dynamic_graph, source_shard, dest_shard):
                path = nx.shortest_path(dynamic_graph, source=source_shard, target=dest_shard)
                return path
        except Exception as e:
            print(f"Error finding dynamic mesh path: {e}")
        
        return []
    
    def analyze_transaction_patterns(self, window_size: int = 100) -> Dict[str, Any]:
        """
        Analyze transaction patterns to optimize routing.
        
        Args:
            window_size: Number of most recent transactions to analyze
            
        Returns:
            Dict[str, Any]: Transaction pattern analysis results
        """
        # Check if transaction_history is a list and has enough data
        if not isinstance(self.transaction_history, list):
            return {'patterns_found': False, 'error': 'transaction_history is not a list'}
            
        if len(self.transaction_history) < window_size:
            return {'patterns_found': False, 'message': 'Not enough transactions for analysis'}
        
        # Get the most recent transactions for analysis
        recent_txs = self.transaction_history[-window_size:]
        
        # Analyze ratio of same-shard and cross-shard transactions
        same_shard_count = 0
        cross_shard_count = 0
        
        # Count transactions by shard pair
        shard_pair_counts = defaultdict(int)
        
        # Analyze by transaction type
        tx_type_counts = defaultdict(int)
        
        for tx in recent_txs:
            tx_type = tx.get('type', 'default')
            tx_type_counts[tx_type] += 1
            
            if 'source_shard' in tx and 'destination_shard' in tx:
                src = tx['source_shard']
                dst = tx['destination_shard']
                
                if src == dst:
                    same_shard_count += 1
                else:
                    cross_shard_count += 1
                    pair = tuple(sorted([src, dst]))
                    shard_pair_counts[pair] += 1
        
        # Calculate same-shard transaction ratio
        total_tx_with_route = same_shard_count + cross_shard_count
        same_shard_ratio = same_shard_count / max(1, total_tx_with_route)
        
        # Find high-traffic shard pairs
        high_traffic_pairs = []
        if shard_pair_counts:
            avg_count = sum(shard_pair_counts.values()) / len(shard_pair_counts)
            high_traffic_pairs = [pair for pair, count in shard_pair_counts.items() 
                                if count > 2 * avg_count]
        
        # Update proximity matrix based on transaction traffic
        for (i, j), count in shard_pair_counts.items():
            # Calculate factor based on traffic
            traffic_factor = min(1.0, count / max(1, window_size * 0.1))
            
            # Update proximity matrix - increase affinity for high-traffic shard pairs
            current_affinity = self.shard_affinity[i, j]
            # Affinity gradually increases with traffic, but doesn't exceed 1.0
            new_affinity = min(1.0, current_affinity + 0.1 * traffic_factor)
            
            self.shard_affinity[i, j] = new_affinity
            self.shard_affinity[j, i] = new_affinity
        
        # Update target same-shard transaction ratio
        target_ratio = max(0.4, min(0.9, same_shard_ratio))
        self.same_shard_ratio = 0.8 * self.same_shard_ratio + 0.2 * target_ratio
        
        # Return analysis results
        return {
            'patterns_found': True,
            'same_shard_ratio': same_shard_ratio,
            'high_traffic_pairs': high_traffic_pairs,
            'tx_type_distribution': dict(tx_type_counts)
        } 