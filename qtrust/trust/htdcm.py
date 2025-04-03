"""
Hierarchical Trust-based Data Center Mechanism (HTDCM) Module

This module implements a hierarchical trust evaluation system for blockchain networks.
It provides mechanisms to assess and maintain trust scores for nodes and shards based on
their transaction history, response times, peer ratings, and detected anomalies.
The module offers advanced attack detection capabilities including identification of
Eclipse, Sybil, and 51% attacks, as well as ML-based anomaly detection for suspicious activities.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque
from .anomaly_detection import MLBasedAnomalyDetectionSystem
import time

class HTDCMNode:
    """
    Represents the trust information of a node in the system.
    """
    def __init__(self, node_id: int, shard_id: int, initial_trust: float = 0.7):
        """
        Initialize trust information for a node.
        
        Args:
            node_id: ID of the node
            shard_id: ID of the shard the node belongs to
            initial_trust: Initial trust score (0.0-1.0)
        """
        self.node_id = node_id
        self.shard_id = shard_id
        self.trust_score = initial_trust
        
        # Store activity history
        self.successful_txs = 0
        self.failed_txs = 0
        self.malicious_activities = 0
        self.response_times = []
        
        # Parameters for trust calculation
        self.alpha = 0.8  # Weight for trust history
        self.beta = 0.2   # Weight for new evaluation
        
        # Store ratings from other nodes
        self.peer_ratings = defaultdict(lambda: 0.5)  # Node ID -> Rating
        
        # Behavior history
        self.activity_history = deque(maxlen=100)  # Store 100 most recent activities

        # Add: ML anomaly score
        self.ml_anomaly_score = 0.0
        self.detected_as_anomaly = False
        self.anomaly_history = []
    
    def update_trust_score(self, new_rating: float):
        """
        Update trust score based on new rating.
        
        Args:
            new_rating: New rating (0.0-1.0)
        """
        # Handle extreme cases immediately
        if new_rating >= 1.0:
            self.trust_score = 1.0
            return
        elif new_rating <= 0.0:
            self.trust_score = 0.0
            return
            
        # Update trust score using weighted average function
        new_trust = self.alpha * self.trust_score + self.beta * new_rating
        
        # Ensure trust score is within range [0.0, 1.0]
        self.trust_score = max(0.0, min(1.0, new_trust))
    
    def record_transaction_result(self, success: bool, response_time: float, is_validator: bool):
        """
        Record the result of a transaction in which the node participated.
        
        Args:
            success: Whether the transaction was successful
            response_time: Response time (ms)
            is_validator: Whether the node was a validator for the transaction
        """
        if success:
            self.successful_txs += 1
            self.activity_history.append(('success', response_time, is_validator))
        else:
            self.failed_txs += 1
            self.activity_history.append(('fail', response_time, is_validator))
        
        self.response_times.append(response_time)
        
        # Limit the number of stored responses
        if len(self.response_times) > 100:
            self.response_times.pop(0)
    
    def record_peer_rating(self, peer_id: int, rating: float):
        """
        Record a rating from another node.
        
        Args:
            peer_id: ID of the rating node
            rating: Rating (0.0-1.0)
        """
        self.peer_ratings[peer_id] = rating
    
    def record_malicious_activity(self, activity_type: str):
        """
        Record a detected malicious activity.
        
        Args:
            activity_type: Type of malicious activity
        """
        self.malicious_activities += 1
        self.activity_history.append(('malicious', activity_type, True))
        
        # Apply severe penalty for malicious activity - directly set to 0
        self.trust_score = 0.0
    
    def update_anomaly_score(self, anomaly_score: float, is_anomaly: bool):
        """
        Update anomaly score from ML-based detection.
        
        Args:
            anomaly_score: Anomaly score (higher = more likely to be anomalous)
            is_anomaly: Whether it's detected as an anomaly
        """
        self.ml_anomaly_score = anomaly_score
        
        if is_anomaly:
            self.detected_as_anomaly = True
            self.anomaly_history.append(anomaly_score)
            
            # Adjust trust score based on anomaly severity
            if anomaly_score > 2.0:  # Severe anomaly
                self.trust_score = max(0.0, self.trust_score - 0.3)
            else:  # Mild anomaly
                self.trust_score = max(0.0, self.trust_score - 0.1)
    
    def get_average_response_time(self) -> float:
        """
        Get the recent average response time.
        
        Returns:
            float: Average response time
        """
        if not self.response_times:
            return 0.0
        return np.mean(self.response_times)
    
    def get_success_rate(self) -> float:
        """
        Get the success rate of transactions.
        
        Returns:
            float: Success rate
        """
        total = self.successful_txs + self.failed_txs
        if total == 0:
            return 0.0
        return self.successful_txs / total
    
    def get_peer_trust(self) -> float:
        """
        Get the average trust rating from other nodes.
        
        Returns:
            float: Average trust rating from peers
        """
        if not self.peer_ratings:
            return 0.5
        return np.mean(list(self.peer_ratings.values()))
    
    def get_state_for_ml(self) -> Dict[str, Any]:
        """
        Create node state for ML model.
        
        Returns:
            Dict[str, Any]: Node attributes for feature extraction
        """
        return {
            'node_id': self.node_id,
            'shard_id': self.shard_id,
            'trust_score': self.trust_score,
            'successful_txs': self.successful_txs,
            'failed_txs': self.failed_txs,
            'malicious_activities': self.malicious_activities,
            'response_times': self.response_times,
            'peer_ratings': dict(self.peer_ratings),
            'activity_history': list(self.activity_history)
        }

class HTDCM:
    """
    Hierarchical Trust-based Data Center Mechanism (HTDCM).
    Multi-level trust evaluation mechanism for blockchain networks.
    """
    
    def __init__(self, 
                 network = None,
                 shards = None,
                 num_nodes = None,
                 tx_success_weight: float = 0.4,
                 response_time_weight: float = 0.2,
                 peer_rating_weight: float = 0.3,
                 history_weight: float = 0.1,
                 malicious_threshold: float = 0.25,
                 suspicious_pattern_window: int = 8,
                 use_ml_detection: bool = True):
        """
        Initialize the HTDCM trust evaluation system.
        
        Args:
            network: Blockchain network graph (optional)
            shards: List of shards and nodes in each shard (optional)
            num_nodes: Total number of nodes in the network (optional)
            tx_success_weight: Weight for transaction success rate
            response_time_weight: Weight for response time
            peer_rating_weight: Weight for ratings from other nodes
            history_weight: Weight for behavior history
            malicious_threshold: Trust score threshold for malicious classification
            suspicious_pattern_window: Window size for suspicious pattern detection
            use_ml_detection: Whether to use ML-based anomaly detection
        """
        # Weights for different factors in trust evaluation
        self.tx_success_weight = tx_success_weight
        self.response_time_weight = response_time_weight
        self.peer_rating_weight = peer_rating_weight
        self.history_weight = history_weight
        
        # Threshold and parameters for malicious detection
        self.malicious_threshold = malicious_threshold
        self.suspicious_pattern_window = suspicious_pattern_window
        
        # If using simple constructor with num_nodes
        if num_nodes is not None and (network is None or shards is None):
            self.network = None
            self.num_shards = 1  # Default
            
            # Trust scores of shards
            self.shard_trust_scores = np.ones(self.num_shards) * 0.7
            
            # Initialize trust information for each node
            self.nodes = {}
            for node_id in range(num_nodes):
                shard_id = node_id % self.num_shards
                self.nodes[node_id] = HTDCMNode(node_id, shard_id, 0.7)
            
            # Assume shards have equal number of nodes
            self.shards = []
            for i in range(self.num_shards):
                self.shards.append([node_id for node_id in range(num_nodes) if node_id % self.num_shards == i])
        else:
            # Original constructor
            self.network = network
            self.shards = shards
            self.num_shards = len(shards) if shards else 1
            
            # Trust scores of shards
            self.shard_trust_scores = np.ones(self.num_shards) * 0.7
            
            # Initialize trust information for each node
            self.nodes = {}
            if network and shards:
                for shard_id, shard_nodes in enumerate(shards):
                    for node_id in shard_nodes:
                        initial_trust = self.network.nodes[node_id].get('trust_score', 0.7)
                        self.nodes[node_id] = HTDCMNode(node_id, shard_id, initial_trust)
            
        # Global rating history
        self.global_ratings_history = []
        
        # Global tracking for coordinated attack detection
        self.suspected_nodes = set()  
        self.attack_patterns = {} # Store detected attack patterns
        
        # Flag indicating global attack state
        self.under_attack = False
        
        # Initialize ML-based anomaly detection system
        self.use_ml_detection = use_ml_detection
        if use_ml_detection:
            self.ml_anomaly_detector = MLBasedAnomalyDetectionSystem(input_features=20)
            self.ml_detection_stats = {
                "total_detections": 0,
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0
            }
    
    def update_node_trust(self, 
                        node_id: int, 
                        tx_success: bool, 
                        response_time: float, 
                        is_validator: bool):
        """
        Update trust score for a node based on transaction results.
        
        Args:
            node_id: Node ID
            tx_success: Whether the transaction was successful
            response_time: Response time (ms)
            is_validator: Whether the node was a validator for the transaction
        """
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Record transaction result
        node.record_transaction_result(tx_success, response_time, is_validator)
        
        # Detect suspicious behavior
        suspicious = self._detect_suspicious_behavior(node_id)
        
        # Don't update trust score if node is detected with suspicious behavior
        if suspicious:
            return
        
        # Calculate new rating
        new_rating = self._calculate_node_rating(node)
        
        # Update trust score
        node.update_trust_score(new_rating)
        
        # Save to global rating history
        self.global_ratings_history.append((node_id, new_rating))
        
        # Update trust score in network
        if self.network is not None:
            self.network.nodes[node_id]['trust_score'] = node.trust_score
        
        # Update shard trust score
        shard_id = node.shard_id
        self._update_shard_trust(shard_id)
        
        # Add: ML-based anomaly detection
        if self.use_ml_detection:
            # Convert node information to format suitable for ML
            node_data = node.get_state_for_ml()
            
            # Detect anomalies
            is_anomaly, anomaly_score, details = self.ml_anomaly_detector.process_node_data(node_id, node_data)
            
            # Update anomaly score for node
            node.update_anomaly_score(anomaly_score, is_anomaly)
            
            # Update statistics
            if is_anomaly:
                self.ml_detection_stats["total_detections"] += 1
                
                # If node was already detected as malicious, this is a true positive
                if node.trust_score < self.malicious_threshold or node.malicious_activities > 0:
                    self.ml_detection_stats["true_positives"] += 1
                else:
                    self.ml_detection_stats["false_positives"] += 1
                    
                # Add to list of suspected nodes
                self.suspected_nodes.add(node_id)
                self._check_for_coordinated_attack()
    
    def _calculate_node_rating(self, node: HTDCMNode) -> float:
        """
        Calculate trust rating for a node based on multiple criteria.
        
        Args:
            node: Node to calculate rating for
            
        Returns:
            float: Trust rating (0.0-1.0)
        """
        # Success rate
        success_rate = node.get_success_rate()
        
        # Response time
        avg_response_time = node.get_average_response_time()
        response_time_rating = 1.0 - min(1.0, avg_response_time / 50.0)  # Assume 50ms is maximum good response time
        
        # Peer rating
        peer_rating = node.get_peer_trust()
        
        # Calculate composite rating
        rating = (
            self.tx_success_weight * success_rate +
            self.response_time_weight * response_time_rating +
            self.peer_rating_weight * peer_rating
        )
        
        # Add: penalty based on anomaly score
        if node.detected_as_anomaly and node.ml_anomaly_score > 1.0:
            penalty = min(0.3, node.ml_anomaly_score / 10.0)
            rating -= penalty
            
        # Ensure rating is within range [0.0, 1.0]
        return max(0.0, min(1.0, rating))
    
    def _update_shard_trust(self, shard_id: int):
        """
        Update trust score for a shard based on nodes in that shard.
        
        Args:
            shard_id: Shard ID
        """
        # Get all nodes in the shard
        shard_nodes = [self.nodes[node_id] for node_id in self.shards[shard_id]]
        
        # Calculate average trust score
        avg_trust = np.mean([node.trust_score for node in shard_nodes])
        
        # Update shard trust score
        self.shard_trust_scores[shard_id] = avg_trust
    
    def _detect_suspicious_behavior(self, node_id: int):
        """
        Detect suspicious behaviors of a node.
        
        Args:
            node_id: ID of the node to check
            
        Returns:
            bool: True if suspicious activity is detected, False otherwise
        """
        node = self.nodes[node_id]
        
        # If trust score is below threshold, consider malicious - improve performance by checking immediately
        if node.trust_score < self.malicious_threshold:
            node.record_malicious_activity('low_trust')
            self.suspected_nodes.add(node_id)
            self._check_for_coordinated_attack()
            return True
        
        # Use cache to avoid checking too frequently - check every 3-5 transactions
        recent_tx_count = node.successful_txs + node.failed_txs
        last_check = getattr(node, 'last_suspicion_check', 0)
        if recent_tx_count - last_check < 4 and node.trust_score > 0.4:
            return False
        
        node.last_suspicion_check = recent_tx_count
        
        # Detect suspicious patterns in activity history
        if len(node.activity_history) >= self.suspicious_pattern_window:
            recent_activities = list(node.activity_history)[-self.suspicious_pattern_window:]
            
            # 1. Check if node consistently fails in transactions - use counter for optimization
            fail_count = sum(1 for act in recent_activities if act[0] == 'fail')
            if fail_count >= self.suspicious_pattern_window * 0.65:  # Reduced threshold from 0.7 to 0.65
                node.record_malicious_activity('consistent_failure')
                self.suspected_nodes.add(node_id)
                self._check_for_coordinated_attack()
                return True
            
            # 2. Check if node has abnormal response times - only if fail_count is below threshold
            if fail_count < self.suspicious_pattern_window * 0.5:
                # Get response times from activity history
                response_times = [act[1] for act in recent_activities if isinstance(act[1], (int, float))]
                if response_times and len(response_times) >= 6:
                    # More efficient calculation by storing mean and standard deviation
                    mean_time = np.mean(response_times)
                    std_time = np.std(response_times)
                    if std_time > 2.0 * mean_time:
                        node.record_malicious_activity('erratic_response_time')
                        self.suspected_nodes.add(node_id)
                        self._check_for_coordinated_attack()
                        return True
            
            # 3. Detect oscillating activity pattern - only if other patterns haven't been detected
            success_fail_pattern = [1 if act[0] == 'success' else 0 for act in recent_activities]
            if len(success_fail_pattern) >= 6:
                # Optimize correlation calculation
                if self._check_alternating_pattern(success_fail_pattern, 0.5):
                    node.record_malicious_activity('oscillating_behavior')
                    self.suspected_nodes.add(node_id)
                    self._check_for_coordinated_attack()
                    return True
            
            # 4 & 5. Skip some more complex checks if node has high trust score
            if node.trust_score < 0.65:
                # Get response times from activity history (if not already obtained in previous step)
                if not 'response_times' in locals():
                    response_times = [act[1] for act in recent_activities if isinstance(act[1], (int, float))]
                
                # 4. Detect abnormal activity based on response time distribution
                if len(response_times) >= 6:  # Reduced from 8 to 6
                    # Check maximum distance between times
                    sorted_times = sorted(response_times)
                    differences = np.diff(sorted_times)
                    if differences.size > 0 and np.max(differences) > 4 * np.mean(differences):
                        node.record_malicious_activity('bimodal_response_times')
                        self.suspected_nodes.add(node_id)
                        self._check_for_coordinated_attack()
                        return True
                
                # 5. Detect "sleeper agent" behavior - node starts good then becomes bad
                if len(response_times) >= 8:  # Reduced from 10 to 8
                    n = len(recent_activities) // 2
                    early_success_rate = sum(1 for act in recent_activities[:n] if act[0] == 'success') / n
                    late_success_rate = sum(1 for act in recent_activities[n:] if act[0] == 'success') / n
                    
                    if early_success_rate > 0.7 and late_success_rate < 0.5 and early_success_rate - late_success_rate > 0.3:
                        node.record_malicious_activity('sleeper_agent')
                        self.suspected_nodes.add(node_id)
                        self._check_for_coordinated_attack()
                        return True
        
        # 6. Analyze compared to other nodes - only check if node has trust score below threshold
        if node.trust_score < 0.6:
            shard_id = node.shard_id
            shard_nodes = [self.nodes[n_id] for n_id in self.shards[shard_id] if n_id != node_id]
            
            if shard_nodes:
                other_nodes_avg_trust = np.mean([n.trust_score for n in shard_nodes])
                if node.trust_score < other_nodes_avg_trust * 0.65 and other_nodes_avg_trust > 0.45:
                    node.record_malicious_activity('significant_trust_deviation')
                    self.suspected_nodes.add(node_id)
                    self._check_for_coordinated_attack()
                    return True
                    
        # 7. Add: Detect nodes with abnormal trust score volatility - only with long history
        if len(self.global_ratings_history) > 10:
            # Filter rating history for current node - only take 5 most recent data points
            node_history = [(idx, rating) for idx, (rated_node, rating) in enumerate(self.global_ratings_history[-20:]) 
                           if rated_node == node_id]
            
            if len(node_history) >= 5:
                # Calculate rate of change in trust score
                ratings = [r for (_, r) in node_history[-5:]]
                changes = np.abs(np.diff(ratings))
                
                # If there's a sudden change
                if np.max(changes) > 0.25:
                    node.record_malicious_activity('trust_score_volatility')
                    self.suspected_nodes.add(node_id)
                    self._check_for_coordinated_attack()
                    return True
        
        return False
    
    def _check_alternating_pattern(self, pattern, threshold=0.5):
        """
        Check for alternating 0/1 pattern in a sequence.
        
        Args:
            pattern: List to check
            threshold: Threshold for pattern determination (0.0-1.0)
            
        Returns:
            bool: True if alternating pattern is detected
        """
        if len(pattern) < 4:
            return False
            
        # Count transitions between 0 and 1
        alternations = sum(1 for i in range(1, len(pattern)) if pattern[i] != pattern[i-1])
        max_alternations = len(pattern) - 1
        
        # If alternation ratio is high, could be suspicious pattern
        return alternations / max_alternations >= threshold
    
    def _check_for_coordinated_attack(self):
        """
        Detect coordinated attacks based on suspicious activity patterns from multiple nodes.
        """
        # If too many nodes are suspected, could be a coordinated attack
        suspected_ratio = len(self.suspected_nodes) / sum(len(shard) for shard in self.shards)
        
        if suspected_ratio > 0.15:  # More than 15% of nodes are suspected
            self.under_attack = True
            
            # Analyze distribution of suspected nodes
            nodes_per_shard = defaultdict(int)
            for node_id in self.suspected_nodes:
                shard_id = self.nodes[node_id].shard_id
                nodes_per_shard[shard_id] += 1
            
            # Identify most heavily affected shards
            shard_sizes = {i: len(shard) for i, shard in enumerate(self.shards)}
            shard_ratios = {shard_id: count / shard_sizes[shard_id] 
                          for shard_id, count in nodes_per_shard.items()}
            
            # Save attack pattern for monitoring
            self.attack_patterns['affected_shards'] = [shard_id for shard_id, ratio in shard_ratios.items() 
                                                   if ratio > 0.3]  # Shards with > 30% suspected nodes
            
            # Mark all nodes in heavily affected shards
            for shard_id in self.attack_patterns['affected_shards']:
                for node_id in self.shards[shard_id]:
                    # Reduce score of all nodes in affected shard
                    if node_id not in self.suspected_nodes:
                        self.nodes[node_id].trust_score = max(0.4, self.nodes[node_id].trust_score * 0.8)
                    # Update trust_score in network
                    if self.network is not None:
                        self.network.nodes[node_id]['trust_score'] = self.nodes[node_id].trust_score
                    
            # Update trust scores of affected shards
            for shard_id in self.attack_patterns['affected_shards']:
                self._update_shard_trust(shard_id)
                
            # Record attack detection time
            self.attack_patterns['detection_time'] = len(self.global_ratings_history)
            
            # Implement response measures
            self._attack_response()
    
    def _attack_response(self):
        """
        Implement response measures when an attack is detected.
        """
        if not self.under_attack:
            return
            
        # 1. Increase monitoring - reduce detection window size
        self.suspicious_pattern_window = max(4, self.suspicious_pattern_window - 2)
        
        # 2. Adjust evaluation weights to rely more on history
        self.history_weight = min(0.4, self.history_weight * 2)
        self.tx_success_weight = max(0.3, self.tx_success_weight * 0.8)
        
        # 3. Identify most trustworthy nodes in each shard
        trusted_nodes_per_shard = {}
        for shard_id in range(self.num_shards):
            trusted_validators = self.recommend_trusted_validators(shard_id, count=max(3, len(self.shards[shard_id]) // 3))
            
            # Extract node IDs from the dictionaries returned by recommend_trusted_validators
            trusted_node_ids = []
            for validator in trusted_validators:
                if isinstance(validator, dict) and 'node_id' in validator:
                    trusted_node_ids.append(validator['node_id'])
                elif isinstance(validator, int):
                    trusted_node_ids.append(validator)
            
            trusted_nodes_per_shard[shard_id] = trusted_validators
            
            # Boost trust scores for trusted nodes
            for node_id in trusted_node_ids:
                if node_id in self.nodes:
                    self.nodes[node_id].trust_score = min(1.0, self.nodes[node_id].trust_score * 1.2)
                    if self.network is not None:
                        self.network.nodes[node_id]['trust_score'] = self.nodes[node_id].trust_score
        
        # 4. Save list of trusted nodes for reference
        self.attack_patterns['trusted_nodes'] = trusted_nodes_per_shard
    
    def rate_peers(self, observer_id: int, transactions: List[Dict[str, Any]]):
        """
        Allow a node to rate other nodes based on shared transactions.
        
        Args:
            observer_id: ID of the observing node
            transactions: List of transactions that the observing node participated in
        """
        if observer_id not in self.nodes:
            return
        
        # Trust score of the observing node
        observer_trust = self.nodes[observer_id].trust_score
        
        # Create dictionary to track observed nodes
        observed_nodes = defaultdict(list)
        
        for tx in transactions:
            # Get list of nodes participating in the transaction (excluding observer)
            participant_nodes = []
            if 'source_node' in tx and tx['source_node'] != observer_id:
                participant_nodes.append(tx['source_node'])
            if 'destination_node' in tx and tx['destination_node'] != observer_id:
                participant_nodes.append(tx['destination_node'])
            
            # Add information about each node's participation in this transaction
            for node_id in participant_nodes:
                if node_id in self.nodes:
                    observed_nodes[node_id].append({
                        'success': tx['status'] == 'completed',
                        'response_time': tx.get('completion_time', 0) - tx.get('created_at', 0)
                    })
        
        # Rate each node based on performance in shared transactions
        for node_id, observations in observed_nodes.items():
            if not observations:
                continue
            
            # Calculate success rate and average response time
            success_rate = sum(1 for obs in observations if obs['success']) / len(observations)
            avg_response_time = np.mean([obs['response_time'] for obs in observations])
            
            # Normalize response time
            normalized_response_time = 1.0 - min(1.0, avg_response_time / 100.0)
            
            # Calculate composite rating
            rating = 0.7 * success_rate + 0.3 * normalized_response_time
            
            # Save rating to observed node
            self.nodes[node_id].record_peer_rating(observer_id, rating)
            
            # Update trust score of observed node (with lower weight)
            peer_influence = min(0.1, observer_trust * 0.2)  # Weight of peer rating
            self.nodes[node_id].update_trust_score(
                self.nodes[node_id].trust_score * (1 - peer_influence) + rating * peer_influence
            )
    
    def get_node_trust_scores(self) -> Dict[int, float]:
        """
        Get the list of trust scores for all nodes.
        
        Returns:
            Dict[int, float]: Dictionary mapping node ID to trust score
        """
        return {node_id: node.trust_score for node_id, node in self.nodes.items()}
    
    def get_shard_trust_scores(self) -> np.ndarray:
        """
        Get the list of trust scores for all shards.
        
        Returns:
            np.ndarray: Array containing trust scores of shards
        """
        return self.shard_trust_scores
    
    def identify_malicious_nodes(self, min_malicious_activities: int = 2, advanced_filtering: bool = True) -> List[int]:
        """
        Identify malicious nodes in the network.
        
        Args:
            min_malicious_activities: Minimum number of recorded malicious activities
                                    to classify a node as malicious
            advanced_filtering: If True, apply advanced filters to reduce false positives
        
        Returns:
            List[int]: List of IDs of nodes identified as malicious
        """
        malicious_nodes = []
        
        for node_id, node in self.nodes.items():
            # Basic condition: Trust score below threshold
            trust_below_threshold = node.trust_score < self.malicious_threshold
            
            if not trust_below_threshold:
                continue  # Skip node if trust score is high
            
            # Condition 2: Has enough minimum malicious activities
            enough_malicious_activities = node.malicious_activities >= min_malicious_activities
            
            # Condition 3: Success rate too low (if has enough transactions)
            total_txs = node.successful_txs + node.failed_txs
            low_success_rate = False
            if total_txs >= 5:  # Only calculate when enough data
                success_rate = node.successful_txs / total_txs if total_txs > 0 else 0
                low_success_rate = success_rate < 0.4  # Success rate below 40%
            
            if not advanced_filtering:
                # Simple version: only consider trust score and malicious activities
                if trust_below_threshold and enough_malicious_activities:
                    malicious_nodes.append(node_id)
            else:
                # Advanced version:
                
                # Check response time
                high_response_time = False
                if node.response_times and len(node.response_times) >= 3:
                    avg_response_time = np.mean(node.response_times)
                    high_response_time = avg_response_time > 20  # High response time
                
                # Check peer feedback
                poor_peer_rating = False
                if node.peer_ratings:
                    avg_peer_rating = np.mean(list(node.peer_ratings.values()))
                    poor_peer_rating = avg_peer_rating < 0.4  # Low peer rating
                
                # Combined condition:
                # 1. Low trust score AND at least 2 of the following conditions:
                #    - Enough malicious activities
                #    - Low success rate
                #    - High response time
                #    - Poor peer rating
                evidence_count = sum([
                    enough_malicious_activities,
                    low_success_rate,
                    high_response_time,
                    poor_peer_rating
                ])
                
                if evidence_count >= 2:
                    malicious_nodes.append(node_id)
        
        return malicious_nodes
    
    def recommend_trusted_validators(self, shard_id: int, count: int = 3, 
                                    include_ml_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Recommend trusted validators based on reputation, supplemented with ML scores if available.
        
        Args:
            shard_id: ID of the shard for which to recommend validators
            count: Number of validators to recommend
            include_ml_scores: Whether to consider ML scores
            
        Returns:
            List[Dict[str, Any]]: List of trusted validators with detailed information
        """
        # Ensure shard ID is valid
        if shard_id >= self.num_shards:
            return []
            
        # Get list of nodes in the shard
        nodes_in_shard = self.shards[shard_id] if self.shards else []
        if not nodes_in_shard:
            return []
            
        # Calculate composite score for each node
        node_scores = []
        for node_id in nodes_in_shard:
            if node_id not in self.nodes:
                continue
                
            node = self.nodes[node_id]
            
            # Base score = trust score
            base_score = node.trust_score
            
            # Calculate performance score
            success_rate = node.get_success_rate()
            avg_response_time = node.get_average_response_time()
            response_time_factor = 1.0 - min(1.0, avg_response_time / 50.0)
            
            # Calculate composite score
            composite_score = base_score * 0.5 + success_rate * 0.3 + response_time_factor * 0.2
            
            # Reduce score if has malicious activities
            if node.malicious_activities > 0:
                composite_score *= max(0.1, 1.0 - node.malicious_activities * 0.2)
            
            # Consider ML scores if requested
            if include_ml_scores and self.use_ml_detection and node.detected_as_anomaly:
                # Reduce score depending on anomaly severity
                anomaly_penalty = min(0.5, node.ml_anomaly_score / 5.0)
                composite_score *= (1.0 - anomaly_penalty)
            
            # Create detailed information about node
            node_detail = {
                "node_id": node_id,
                "trust_score": node.trust_score,
                "success_rate": success_rate,
                "response_time": avg_response_time,
                "malicious_activities": node.malicious_activities,
                "composite_score": composite_score
            }
            
            # Add ML information if available
            if include_ml_scores and self.use_ml_detection:
                node_detail["ml_anomaly_score"] = node.ml_anomaly_score
                node_detail["detected_as_anomaly"] = node.detected_as_anomaly
                
            node_scores.append(node_detail)
        
        # Sort by composite score
        sorted_nodes = sorted(node_scores, key=lambda x: x["composite_score"], reverse=True)
        
        # Return best nodes
        return sorted_nodes[:count]
        
    def dynamic_malicious_threshold(self, network_congestion: float = 0.5, 
                                   attack_probability: float = 0.0) -> float:
        """
        Calculate dynamic malicious detection threshold based on network conditions.
        
        Args:
            network_congestion: Level of network congestion (0.0-1.0)
            attack_probability: Estimated probability of being under attack (0.0-1.0)
            
        Returns:
            float: Dynamic malicious detection threshold
        """
        # Base threshold
        base_threshold = self.malicious_threshold
        
        # Adjust based on network congestion
        # When network is congested, we relax the threshold to reduce false positives
        congestion_adjustment = network_congestion * 0.1  # Maximum +0.1
        
        # Adjust based on attack probability
        # When there's high probability of attack, we tighten the threshold
        attack_adjustment = -attack_probability * 0.15  # Maximum -0.15
        
        # Adjust based on ML statistics
        ml_adjustment = 0.0
        if self.use_ml_detection:
            # If false positive rate is high, relax threshold
            total_detections = self.ml_detection_stats["total_detections"]
            if total_detections > 0:
                false_positive_rate = self.ml_detection_stats["false_positives"] / total_detections
                ml_adjustment = false_positive_rate * 0.05  # Maximum +0.05
        
        # Final threshold
        final_threshold = base_threshold + congestion_adjustment + attack_adjustment + ml_adjustment
        
        # Ensure threshold is within reasonable range
        return max(0.1, min(0.4, final_threshold))
    
    def detect_advanced_attacks(self, transaction_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect advanced attack types based on transaction history.
        
        Args:
            transaction_history: Recent transaction history
            
        Returns:
            Dict[str, Any]: Detection results with attack types and confidence
        """
        results = {
            "under_attack": False,
            "attack_types": [],
            "confidence": 0.0,
            "suspect_nodes": [],
            "recommended_actions": []
        }
        
        if not transaction_history:
            return results
            
        # 1. Analyze transaction patterns
        txs_per_node = defaultdict(list)
        # Classify transactions by node
        for tx in transaction_history:
            node_id = tx.get("node_id")
            if node_id is not None:
                txs_per_node[node_id].append(tx)
        
        # 2. Detect Eclipse attack
        eclipse_candidates = []
        for node_id, txs in txs_per_node.items():
            if node_id not in self.nodes:
                continue
                
            # Eclipse attack: node isolates other nodes
            # Check if this node rejects too many transactions from a fixed set of other nodes
            rejected_from = defaultdict(int)
            for tx in txs:
                if not tx.get("success", True) and tx.get("rejected_by") == node_id:
                    source_node = tx.get("source_node")
                    if source_node is not None:
                        rejected_from[source_node] += 1
            
            # If rejected too many from specific nodes
            if rejected_from and max(rejected_from.values()) > 5:
                eclipse_candidates.append({
                    "node_id": node_id,
                    "rejection_pattern": dict(rejected_from),
                    "confidence": min(0.9, max(rejected_from.values()) / 10.0)
                })
        
        if eclipse_candidates:
            results["attack_types"].append("Eclipse Attack")
            results["suspect_nodes"].extend([c["node_id"] for c in eclipse_candidates])
            results["under_attack"] = True
            results["confidence"] = max([c["confidence"] for c in eclipse_candidates])
            
            # Suggest actions
            results["recommended_actions"].append("Enhance mesh connectivity between nodes")
            results["recommended_actions"].append("Reduce trust of suspicious nodes")
        
        # 3. Detect Sybil attack
        if len(self.suspected_nodes) > 0.2 * sum(len(shard) for shard in self.shards):
            # Analyze common validation patterns
            validation_patterns = defaultdict(list)
            for tx in transaction_history:
                validators = tx.get("validators", [])
                if validators:
                    key = tuple(sorted(validators))
                    validation_patterns[key].append(tx)
            
            # Find fixed validation groups appearing too frequently
            pattern_frequency = {k: len(v) for k, v in validation_patterns.items()}
            if pattern_frequency:
                max_pattern = max(pattern_frequency.items(), key=lambda x: x[1])
                max_frequency = max_pattern[1] / len(transaction_history)
                
                if max_frequency > 0.4:  # If one pattern accounts for > 40% of transactions
                    results["attack_types"].append("Sybil Attack")
                    results["suspect_nodes"].extend(max_pattern[0])
                    results["under_attack"] = True
                    results["confidence"] = max(results["confidence"], max_frequency)
                    
                    # Suggest actions
                    results["recommended_actions"].append("Increase consensus threshold")
                    results["recommended_actions"].append("Implement immediate validator rotation")
        
        # 4. Detect 51% attack
        for shard_id in range(self.num_shards):
            node_ids = self.shards[shard_id]
            # Get list of suspected nodes in shard
            suspected_in_shard = [n for n in node_ids if n in self.suspected_nodes]
            
            # If > 40% nodes in shard are suspected
            if len(suspected_in_shard) > 0.4 * len(node_ids):
                results["attack_types"].append(f"51% Attack on Shard {shard_id}")
                results["suspect_nodes"].extend(suspected_in_shard)
                results["under_attack"] = True
                results["confidence"] = max(results["confidence"], len(suspected_in_shard) / len(node_ids))
                
                # Suggest actions
                results["recommended_actions"].append(f"Perform resharding for shard {shard_id}")
                results["recommended_actions"].append("Increase number of validators from other shards")
        
        # 5. Use ML to assist detection
        if self.use_ml_detection:
            # Get statistics from anomaly detection system
            ml_stats = self.ml_anomaly_detector.get_statistics()
            
            # If too many anomalies detected
            if ml_stats["total_detections"] > 0.15 * len(self.nodes):
                # Consider this a new unidentified attack type
                results["attack_types"].append("ML-Detected Novel Attack")
                results["suspect_nodes"].extend(ml_stats.get("top_anomalous_nodes", []))
                results["under_attack"] = True
                results["confidence"] = max(results["confidence"], 0.7)
                
                # Suggest actions
                results["recommended_actions"].append("Activate automated defense mechanisms")
                results["recommended_actions"].append("Enhance network-wide monitoring")
        
        return results
        
    def enhance_security_posture(self, attack_detection_result: Dict[str, Any]):
        """
        Enhance security posture based on attack detection results.
        
        Args:
            attack_detection_result: Results from detect_advanced_attacks function
        """
        if not attack_detection_result.get("under_attack", False):
            return
            
        # Mark network as under attack
        self.under_attack = True
        
        # 1. Reduce trust scores of suspicious nodes
        suspect_nodes = attack_detection_result.get("suspect_nodes", [])
        for node_id in suspect_nodes:
            if node_id in self.nodes:
                # Reduce trust score corresponding to detection confidence
                confidence = attack_detection_result.get("confidence", 0.5)
                penalty = min(0.8, confidence * 1.5)  # Maximum 80% reduction
                self.nodes[node_id].trust_score *= (1.0 - penalty)
                
                # Mark malicious activity
                attack_types = ", ".join(attack_detection_result.get("attack_types", ["Unknown"]))
                self.nodes[node_id].record_malicious_activity(f"detected_in_{attack_types}")
                
                # Add to global suspect list
                self.suspected_nodes.add(node_id)
        
        # 2. Adjust malicious detection threshold
        self.malicious_threshold = max(0.15, self.malicious_threshold * 0.8)  # Reduce threshold
        
        # 3. Activate enhanced defense mode if sufficient data
        if self.use_ml_detection and attack_detection_result.get("confidence", 0) > 0.7:
            # Predict nodes that might be part of the attack but not yet detected
            all_nodes = list(self.nodes.keys())
            for node_id in all_nodes:
                if node_id not in suspect_nodes and node_id not in self.suspected_nodes:
                    node = self.nodes[node_id]
                    
                    # Check connections to suspicious nodes
                    connections_to_suspects = 0
                    if self.network:
                        for suspect in suspect_nodes:
                            if self.network.has_edge(node_id, suspect):
                                connections_to_suspects += 1
                    
                    # If too many connections to suspicious nodes
                    if connections_to_suspects > 3:
                        # Reduce trust score mildly
                        node.trust_score *= 0.9
                        self.suspected_nodes.add(node_id)
        
        # 4. Save attack pattern for future reference
        attack_pattern = {
            "time": time.time(),
            "types": attack_detection_result.get("attack_types", []),
            "confidence": attack_detection_result.get("confidence", 0),
            "suspects": suspect_nodes,
            "affected_shards": list(set(self.nodes[n].shard_id for n in suspect_nodes if n in self.nodes))
        }
        
        # Save attack pattern
        pattern_key = f"attack_{len(self.attack_patterns)}"
        self.attack_patterns[pattern_key] = attack_pattern
    
    def get_ml_security_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from ML-based anomaly detection system.
        
        Returns:
            Dict[str, Any]: Security statistics
        """
        if not self.use_ml_detection:
            return {"ml_detection_enabled": False}
            
        # Get statistics from ML anomaly detector
        detector_stats = self.ml_anomaly_detector.get_statistics()
        
        # Calculate performance metrics
        total_pos = self.ml_detection_stats["true_positives"] + self.ml_detection_stats["false_positives"]
        total_neg = self.ml_detection_stats["true_negatives"] + self.ml_detection_stats["false_negatives"]
        
        precision = self.ml_detection_stats["true_positives"] / max(1, total_pos)
        recall = self.ml_detection_stats["true_positives"] / max(1, self.ml_detection_stats["true_positives"] + self.ml_detection_stats["false_negatives"])
        f1_score = 2 * precision * recall / max(0.001, precision + recall)
        
        return {
            "ml_detection_enabled": True,
            "ml_detection_stats": self.ml_detection_stats,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "detector_stats": detector_stats,
            "under_attack": self.under_attack,
            "suspected_nodes_count": len(self.suspected_nodes),
            "attack_patterns_detected": len(self.attack_patterns)
        }
    
    def reset(self):
        """Reset trust scores for all nodes to default values."""
        for node_id, node in self.nodes.items():
            node.trust_score = 0.7
            node.successful_txs = 0
            node.failed_txs = 0
            node.malicious_activities = 0
            node.response_times = []
            node.peer_ratings = defaultdict(lambda: 0.5)
            node.activity_history.clear()
        
        # Reset shard information
        self.shard_trust_scores = np.ones(self.num_shards) * 0.7
        self.under_attack = False
        self.suspected_nodes.clear()
        self.attack_patterns.clear() 