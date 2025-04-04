"""
Adaptive Consensus Module

This module provides a dynamic and flexible consensus mechanism selection system for blockchain networks.
It implements multiple Byzantine Fault Tolerance consensus protocols that adapt to different network conditions,
transaction values, and trust levels.

Key features:
- Multiple consensus protocols: FastBFT, PBFT, RobustBFT, LightBFT and BLS-based consensus
- Automatic protocol selection based on transaction value, network congestion, trust scores, etc.
- Integration with Adaptive Proof-of-Stake (PoS) for validator selection and rotation
- Energy optimization through lightweight cryptography and selective validator activation
- Byzantine fault tolerance with different security-performance trade-offs
- Real-time adaptation to changing network conditions

The main class `AdaptiveConsensus` coordinates protocol selection and execution, optimizing for the
specific requirements of each transaction or block, resulting in optimal balance between security,
performance, and energy efficiency.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import random
from .bls_signatures import BLSBasedConsensus
from .adaptive_pos import AdaptivePoSManager
from qtrust.consensus.lightweight_crypto import AdaptiveCryptoManager

class ConsensusProtocol:
    """
    Base class for consensus protocols.
    """
    def __init__(self, name: str, latency_factor: float, energy_factor: float, security_factor: float):
        """
        Initialize consensus protocol.
        
        Args:
            name: Protocol name
            latency_factor: Latency factor (1.0 = base latency)
            energy_factor: Energy consumption factor (1.0 = base energy)
            security_factor: Security factor (1.0 = maximum security)
        """
        self.name = name
        self.latency_factor = latency_factor
        self.energy_factor = energy_factor
        self.security_factor = security_factor
    
    def execute(self, transaction_value: float, trust_scores: Dict[int, float]) -> Tuple[bool, float, float]:
        """
        Execute consensus protocol on a transaction.
        
        Args:
            transaction_value: Transaction value
            trust_scores: Trust scores of nodes
            
        Returns:
            Tuple[bool, float, float]: (Consensus result, latency, energy consumption)
        """
        raise NotImplementedError("This method must be implemented by subclass")

class FastBFT(ConsensusProtocol):
    """
    Fast Byzantine Fault Tolerance - Quick consensus protocol with lower security.
    Suitable for low-value transactions requiring fast processing.
    """
    def __init__(self, latency_factor: float = 0.2, energy_factor: float = 0.3, security_factor: float = 0.7):
        super().__init__(
            name="FastBFT",
            latency_factor=latency_factor,
            energy_factor=energy_factor,
            security_factor=security_factor
        )
    
    def execute(self, transaction_value: float, trust_scores: Dict[int, float]) -> Tuple[bool, float, float]:
        """
        Execute Fast BFT.
        
        Args:
            transaction_value: Transaction value
            trust_scores: Trust scores of nodes
            
        Returns:
            Tuple[bool, float, float]: (Consensus result, latency, energy consumption)
        """
        # Calculate average trust score of participating nodes
        avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 0.5
        
        # Success probability based on average trust score
        success_prob = min(0.98, avg_trust * 0.9 + 0.1)
        
        # Determine consensus result
        consensus_achieved = bool(random.random() < success_prob)
        
        # Calculate latency
        latency = self.latency_factor * (5.0 + 0.1 * transaction_value)
        
        # Calculate energy consumption
        energy = self.energy_factor * (10.0 + 0.2 * transaction_value)
        
        return consensus_achieved, latency, energy

class PBFT(ConsensusProtocol):
    """
    Practical Byzantine Fault Tolerance - Balanced consensus protocol for security and performance.
    Suitable for most regular transactions.
    """
    def __init__(self, latency_factor: float = 0.5, energy_factor: float = 0.6, security_factor: float = 0.85):
        super().__init__(
            name="PBFT",
            latency_factor=latency_factor,
            energy_factor=energy_factor,
            security_factor=security_factor
        )
    
    def execute(self, transaction_value: float, trust_scores: Dict[int, float]) -> Tuple[bool, float, float]:
        """
        Execute PBFT.
        
        Args:
            transaction_value: Transaction value
            trust_scores: Trust scores of nodes
            
        Returns:
            Tuple[bool, float, float]: (Consensus result, latency, energy consumption)
        """
        # Calculate average trust score of participating nodes
        avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 0.5
        
        # Success probability based on average trust score
        success_prob = min(0.99, avg_trust * 0.95 + 0.05)
        
        # Determine consensus result
        consensus_achieved = bool(random.random() < success_prob)
        
        # Calculate latency
        latency = self.latency_factor * (10.0 + 0.2 * transaction_value)
        
        # Calculate energy consumption
        energy = self.energy_factor * (20.0 + 0.4 * transaction_value)
        
        return consensus_achieved, latency, energy

class RobustBFT(ConsensusProtocol):
    """
    Robust Byzantine Fault Tolerance - Highest security consensus protocol.
    Suitable for high-value transactions requiring maximum security.
    """
    def __init__(self, latency_factor: float = 0.8, energy_factor: float = 0.8, security_factor: float = 0.95):
        super().__init__(
            name="RobustBFT",
            latency_factor=latency_factor,
            energy_factor=energy_factor,
            security_factor=security_factor
        )
    
    def execute(self, transaction_value: float, trust_scores: Dict[int, float]) -> Tuple[bool, float, float]:
        """
        Execute Robust BFT.
        
        Args:
            transaction_value: Transaction value
            trust_scores: Trust scores of nodes
            
        Returns:
            Tuple[bool, float, float]: (Consensus result, latency, energy consumption)
        """
        # Calculate average trust score of participating nodes
        avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 0.5
        
        # Success probability based on average trust score
        success_prob = min(0.995, avg_trust * 0.98 + 0.02)
        
        # Determine consensus result
        consensus_achieved = bool(random.random() < success_prob)
        
        # Calculate latency
        latency = self.latency_factor * (20.0 + 0.3 * transaction_value)
        
        # Calculate energy consumption
        energy = self.energy_factor * (30.0 + 0.5 * transaction_value)
        
        return consensus_achieved, latency, energy

class LightBFT(ConsensusProtocol):
    """
    Light Byzantine Fault Tolerance - Lightweight consensus protocol for stable networks.
    Minimizes energy consumption and latency for networks with high stability.
    """
    def __init__(self, latency_factor: float = 0.15, energy_factor: float = 0.2, security_factor: float = 0.75):
        super().__init__(
            name="LightBFT",
            latency_factor=latency_factor,
            energy_factor=energy_factor,
            security_factor=security_factor
        )
    
    def execute(self, transaction_value: float, trust_scores: Dict[int, float]) -> Tuple[bool, float, float]:
        """
        Execute Light BFT.
        
        Args:
            transaction_value: Transaction value
            trust_scores: Trust scores of nodes
            
        Returns:
            Tuple[bool, float, float]: (Consensus result, latency, energy consumption)
        """
        # Calculate average trust score of participating nodes
        avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 0.5
        
        # LightBFT should only be used in high-trust environments
        # If trust is low, success probability will decrease significantly
        if avg_trust < 0.7:
            success_prob = 0.6 + avg_trust * 0.3  # Lower success probability if trust is low
        else:
            # High success probability for high-trust environments
            success_prob = min(0.99, 0.9 + avg_trust * 0.09)
        
        # Determine consensus result
        consensus_achieved = bool(random.random() < success_prob)
        
        # Calculate latency
        latency = self.latency_factor * (4.0 + 0.05 * transaction_value)
        
        # Calculate energy consumption
        energy = self.energy_factor * (8.0 + 0.1 * transaction_value)
        
        return consensus_achieved, latency, energy

class AdaptiveConsensus:
    """
    Class to select consensus mechanism based on different factors.
    """
    def __init__(self, 
                transaction_threshold_low: float = 10.0, 
                transaction_threshold_high: float = 50.0,
                congestion_threshold: float = 0.7,
                min_trust_threshold: float = 0.3,
                transaction_history_size: int = 100,
                network_stability_weight: float = 0.3,
                transaction_value_weight: float = 0.3,
                congestion_weight: float = 0.2,
                trust_weight: float = 0.2,
                high_stability_threshold: float = 0.8,
                high_trust_threshold: float = 0.7,
                enable_bls: bool = True,
                num_validators_per_shard: int = 10,
                enable_adaptive_pos: bool = True,
                enable_lightweight_crypto: bool = True,
                active_validator_ratio: float = 0.7,
                rotation_period: int = 50):
        """
        Initialize AdaptiveConsensus.
        
        Args:
            transaction_threshold_low: Low transaction value threshold
            transaction_threshold_high: High transaction value threshold
            congestion_threshold: Congestion threshold
            min_trust_threshold: Minimum trust threshold
            transaction_history_size: Transaction history size for analysis
            network_stability_weight: Network stability weight
            transaction_value_weight: Transaction value weight
            congestion_weight: Congestion weight
            trust_weight: Trust weight
            high_stability_threshold: High stability threshold to use LightBFT
            high_trust_threshold: High trust threshold to use LightBFT
            enable_bls: Enable BLS signature aggregation
            num_validators_per_shard: Number of validators per shard
            enable_adaptive_pos: Enable Adaptive PoS
            enable_lightweight_crypto: Enable/disable Lightweight Cryptography
            active_validator_ratio: Active validator ratio (0.0-1.0)
            rotation_period: Rotation period before considering rotation
        """
        self.transaction_threshold_low = transaction_threshold_low
        self.transaction_threshold_high = transaction_threshold_high
        self.congestion_threshold = congestion_threshold
        self.min_trust_threshold = min_trust_threshold
        self.transaction_history_size = transaction_history_size
        self.high_stability_threshold = high_stability_threshold
        self.high_trust_threshold = high_trust_threshold
        self.enable_bls = enable_bls
        self.num_validators_per_shard = num_validators_per_shard
        self.enable_adaptive_pos = enable_adaptive_pos
        self.enable_lightweight_crypto = enable_lightweight_crypto
        
        # Set weights for different factors
        self.network_stability_weight = network_stability_weight
        self.transaction_value_weight = transaction_value_weight
        self.congestion_weight = congestion_weight
        self.trust_weight = trust_weight
        
        # Initialize consensus protocols
        self.consensus_protocols = {
            "FastBFT": FastBFT(),
            "PBFT": PBFT(),
            "RobustBFT": RobustBFT(),
            "LightBFT": LightBFT()
        }
        
        # Add BLS-based Consensus if enabled
        if self.enable_bls:
            self.consensus_protocols["BLS_Consensus"] = BLSBasedConsensus(
                num_validators=self.num_validators_per_shard,
                threshold_percent=0.7,
                latency_factor=0.4,
                energy_factor=0.5,
                security_factor=0.9
            )
        
        # Initialize Adaptive PoS Managers if enabled
        self.pos_managers = {}
        if self.enable_adaptive_pos:
            # Create a PoS Manager for each shard
            for shard_id in range(10):  # Assuming maximum 10 shards
                self.pos_managers[shard_id] = AdaptivePoSManager(
                    num_validators=num_validators_per_shard,
                    active_validator_ratio=active_validator_ratio,
                    rotation_period=rotation_period,
                    min_stake=10.0,
                    energy_threshold=30.0,
                    performance_threshold=0.3,
                    seed=42 + shard_id  # Different seeds for each shard
                )
        
        # Transaction history performance metrics
        self.protocol_performance = {}
        for name in self.consensus_protocols.keys():
            self.protocol_performance[name] = {
                "total_count": 10,       # Total transaction count
                "success_count": 8,      # Successful transaction count
                "latency_sum": 500.0,    # Total latency (ms)
                "energy_sum": 250.0      # Total energy consumption
            }
        
        # Transaction history
        self.transaction_history = []
        
        # Protocol assignments for each shard
        self.shard_protocols = {}
        
        # Protocol usage statistics
        self.protocol_usage = {name: 0 for name in self.consensus_protocols.keys()}
        
        # Energy savings from Adaptive PoS
        self.total_energy_saved = 0.0
        self.total_rotations = 0
        
        # Initialize crypto manager
        if enable_lightweight_crypto:
            self.crypto_manager = AdaptiveCryptoManager()
        else:
            self.crypto_manager = None
        
        # Energy optimization statistics
        self.energy_optimization_stats = {
            "total_energy_saved_crypto": 0.0,
            "total_operations": 0,
            "security_level_distribution": {"low": 0, "medium": 0, "high": 0}
        }
        
        # Trust scores for validators
        self.validator_trust_scores = {}
        for i in range(num_validators_per_shard):
            # Khởi tạo với trust scores mặc định giữa 0.5 và 0.9
            self.validator_trust_scores[i] = 0.5 + (0.4 * random.random())
    
    def update_consensus_mechanism(self, congestion_levels: Dict[int, float], trust_scores: Dict[int, float], 
                              network_stability: float = 0.5, cross_shard_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Update consensus mechanism for each shard based on current network conditions.
        
        Args:
            congestion_levels: Dict mapping shard ID to congestion level
            trust_scores: Dict mapping node ID to trust score
            network_stability: Overall network stability (0-1)
            cross_shard_ratio: Cross-shard transaction ratio
            
        Returns:
            Dict[str, Any]: Information about consensus mechanism update
        """
        protocol_assignments = {}
        changes_made = 0
        
        # Analyze performance of current consensus protocols
        protocol_metrics = self._analyze_protocol_performance()
        
        # Calculate average trust score for each shard
        shard_trust_scores = {}
        for node_id, trust in trust_scores.items():
            shard_id = node_id // self.num_validators_per_shard
            if shard_id not in shard_trust_scores:
                shard_trust_scores[shard_id] = []
            shard_trust_scores[shard_id].append(trust)
        
        avg_shard_trust = {shard_id: sum(scores)/len(scores) if scores else 0.5 
                          for shard_id, scores in shard_trust_scores.items()}
        
        # Select appropriate protocol for each shard
        for shard_id, congestion in congestion_levels.items():
            # Get average trust score of shard
            trust = avg_shard_trust.get(shard_id, 0.5)
            
            # Determine importance of shard based on cross-shard ratio
            is_important_shard = congestion > 0.5 or cross_shard_ratio > 0.4
            
            # Select appropriate protocol
            if network_stability > self.high_stability_threshold and trust > self.high_trust_threshold:
                # Stable environment and high trust -> use lightweight protocol
                if "LightBFT" in self.consensus_protocols:
                    selected_protocol = "LightBFT"
                else:
                    selected_protocol = "FastBFT"
            elif is_important_shard and trust < self.min_trust_threshold:
                # Important shard but low trust -> use strongest protocol
                selected_protocol = "RobustBFT"
            elif congestion > self.congestion_threshold:
                # High congestion -> use BLS if enabled or FastBFT
                if self.enable_bls and "BLS_Consensus" in self.consensus_protocols:
                    selected_protocol = "BLS_Consensus"
                else:
                    selected_protocol = "FastBFT"
            elif cross_shard_ratio > 0.4:
                # High cross-shard transaction ratio -> use BLS if enabled
                if self.enable_bls and "BLS_Consensus" in self.consensus_protocols:
                    selected_protocol = "BLS_Consensus"
                else:
                    selected_protocol = "PBFT"
            else:
                # Regular case -> use PBFT
                selected_protocol = "PBFT"
            
            # Check if there's a change compared to current protocol
            current_protocol = self.shard_protocols.get(shard_id, "PBFT")
            if current_protocol != selected_protocol:
                changes_made += 1
            
            # Update protocol for shard
            self.shard_protocols[shard_id] = selected_protocol
            protocol_assignments[shard_id] = {
                "protocol": selected_protocol,
                "congestion": congestion,
                "trust": trust,
                "changed": current_protocol != selected_protocol
            }
            
            # Update protocol usage statistics
            self.protocol_usage[selected_protocol] = self.protocol_usage.get(selected_protocol, 0) + 1
        
        # Update protocol usage statistics
        protocol_distribution = {name: count/sum(self.protocol_usage.values()) 
                               for name, count in self.protocol_usage.items() if count > 0}
        
        return {
            "assignments": protocol_assignments,
            "changes_made": changes_made,
            "protocol_distribution": protocol_distribution,
            "protocol_metrics": protocol_metrics,
            "bls_enabled": self.enable_bls and "BLS_Consensus" in self.consensus_protocols
        }
    
    def _analyze_protocol_performance(self) -> Dict[str, float]:
        """
        Analyze performance of protocols based on history.
        
        Returns:
            Dict[str, float]: Performance score for each protocol (0.0-1.0)
        """
        scores = {}
        
        for protocol, stats in self.protocol_performance.items():
            if stats["total_count"] == 0:
                scores[protocol] = 0.33  # Default score
                continue
                
            # Calculate success rate
            success_rate = stats["success_count"] / stats["total_count"] if stats["total_count"] > 0 else 0
            
            # Calculate average latency and energy
            avg_latency = stats["latency_sum"] / stats["total_count"] if stats["total_count"] > 0 else 0
            avg_energy = stats["energy_sum"] / stats["total_count"] if stats["total_count"] > 0 else 0
            
            # Normalize latency and energy (lower is better)
            # Assuming maximum value is 100ms and 100mJ
            norm_latency = 1.0 - min(1.0, avg_latency / 100.0)
            norm_energy = 1.0 - min(1.0, avg_energy / 100.0)
            
            # Calculate overall score
            # Weights: success_rate (0.5), latency (0.3), energy (0.2)
            scores[protocol] = 0.5 * success_rate + 0.3 * norm_latency + 0.2 * norm_energy
        
        return scores
    
    def select_protocol(self, transaction_value: float, congestion: float, trust_scores: Dict[int, float],
                      network_stability: float = 0.5, cross_shard: bool = False) -> ConsensusProtocol:
        """
        Select appropriate consensus protocol based on factors.
        
        Args:
            transaction_value: Transaction value
            congestion: Congestion level (0-1)
            trust_scores: Trust scores of nodes
            network_stability: Network stability (0-1)
            cross_shard: Is it a cross-shard transaction
            
        Returns:
            ConsensusProtocol: Selected consensus protocol
        """
        # Calculate score for each protocol
        protocol_scores = {}
        
        # Calculate average trust score
        avg_trust = sum(trust_scores.values()) / len(trust_scores) if trust_scores else 0.5
        
        # Check conditions to select protocol
        for name, protocol in self.consensus_protocols.items():
            # Start with basic score
            score = 5.0
            
            # Adjust based on transaction value
            if transaction_value <= self.transaction_threshold_low:
                # Low value transactions prioritize speed
                if name in ["FastBFT", "LightBFT"]:
                    score += 2.0
                elif name == "BLS_Consensus":
                    score += 1.5
                elif name == "PBFT":
                    score += 1.0
                # RobustBFT is not suitable for low-value transactions
            elif transaction_value >= self.transaction_threshold_high:
                # High value transactions prioritize security
                if name == "RobustBFT":
                    score += 2.5
                elif name == "PBFT":
                    score += 1.5
                elif name == "BLS_Consensus":
                    score += 1.0
                # FastBFT/LightBFT is not suitable for high-value transactions
            else:
                # Medium value transaction
                if name == "PBFT":
                    score += 1.5
                elif name == "BLS_Consensus":
                    score += 1.2
                else:
                    score += 0.8
                
            # Adjust based on congestion level
            if congestion > self.congestion_threshold:
                # High congestion prioritize performance
                if name in ["FastBFT", "BLS_Consensus"]:
                    score += 1.5
                elif name == "LightBFT":
                    score += 1.0
                # Other protocols are less effective in high congestion
            else:
                # Low congestion, consider security more
                if name == "RobustBFT":
                    score += 0.8
                elif name == "PBFT":
                    score += 0.5
                    
            # Adjust based on trust
            if avg_trust < self.min_trust_threshold:
                # Low trust prioritize security
                if name == "RobustBFT":
                    score += 2.0
                elif name == "PBFT":
                    score += 1.0
                # Other protocols are less secure in low-trust environments
            elif avg_trust > self.high_trust_threshold:
                # High trust allows using less secure protocol
                if name == "LightBFT":
                    score += 1.5
                elif name == "FastBFT":
                    score += 1.0
                elif name == "BLS_Consensus":
                    score += 0.8
                    
            # Adjust based on network stability
            if network_stability > self.high_stability_threshold:
                # Stable network prioritize performance
                if name == "LightBFT":
                    score += 1.5
                elif name in ["FastBFT", "BLS_Consensus"]:
                    score += 1.0
            elif network_stability < 0.3:
                # Unstable network prioritize security
                if name == "RobustBFT":
                    score += 1.5
                elif name == "PBFT":
                    score += 0.8
                    
            # Adjust for cross-shard transactions
            if cross_shard:
                # Cross-shard transactions are usually more complex
                if name == "BLS_Consensus":
                    score += 2.0  # BLS is best for cross-shard due to reduced overhead
                elif name == "RobustBFT":
                    score += 1.0  # High security for cross-shard
                elif name == "PBFT":
                    score += 0.5  # Balanced for cross-shard
                # FastBFT/LightBFT may not be secure enough for cross-shard
                    
            # Save score
            protocol_scores[name] = score
        
        # Select protocol with highest score
        selected_protocol_name = max(protocol_scores.items(), key=lambda x: x[1])[0]
        selected_protocol = self.consensus_protocols[selected_protocol_name]
        
        # Update protocol usage statistics
        self.protocol_usage[selected_protocol_name] = self.protocol_usage.get(selected_protocol_name, 0) + 1
        
        return selected_protocol
    
    def execute_consensus(self, 
                        transaction_value: float, 
                        shard_id: int,
                        trust_scores: Dict[int, float],
                        transaction_data: Any = None) -> Tuple[bool, float, float, str]:
        """
        Execute consensus for a transaction.
        
        Args:
            transaction_value: Value of the transaction
            shard_id: Shard ID
            trust_scores: Trust scores of nodes
            transaction_data: Transaction data (optional)
            
        Returns:
            Tuple[bool, float, float, str]: (Success, latency, energy consumption, protocol name)
        """
        # Record transaction start time
        start_time = time.time()
        
        # Apply adaptive PoS for validator selection if enabled
        pos_energy_saved = 0.0
        if self.enable_adaptive_pos and shard_id in self.pos_managers:
            # Get the validator IDs for this transaction
            validator_ids = self.pos_managers[shard_id].select_validators_for_committee(
                committee_size=self.num_validators_per_shard,
                trust_scores=trust_scores
            )
            pos_energy_saved = 0.0  # Giả sử không có tiết kiệm năng lượng nào
            
            # Filter the trust scores to only include active validators
            active_trust_scores = {vid: trust_scores.get(vid, 0.5) for vid in validator_ids}
            
            # Update total energy saved from adaptive PoS
            self.total_energy_saved += pos_energy_saved
        else:
            active_trust_scores = trust_scores
        
        # Determine the best consensus protocol
        protocol_name = self.select_consensus_protocol(
            transaction_value=transaction_value,
            shard_id=shard_id,
            trust_scores=active_trust_scores
        )
        
        # Get the selected protocol
        if protocol_name in self.consensus_protocols:
            protocol = self.consensus_protocols[protocol_name]
        else:
            # Default to PBFT if protocol not found
            protocol = self.consensus_protocols["PBFT"]
            protocol_name = "PBFT"
        
        # Apply lightweight cryptography if enabled
        crypto_energy_saved = 0.0
        if self.enable_lightweight_crypto and self.crypto_manager:
            # Determine security level based on transaction value
            if transaction_value <= self.transaction_threshold_low:
                security_level = "low"
            elif transaction_value <= self.transaction_threshold_high:
                security_level = "medium"
            else:
                security_level = "high"
            
            # Apply appropriate cryptography and get energy savings
            message = f"tx_{transaction_value}_{time.time()}"
            crypto_result = self.crypto_manager.execute_crypto_operation(
                operation="hash",
                params={"message": message},
                transaction_value=transaction_value,
                network_congestion=0.5,  # Giả sử mức độ tắc nghẽn trung bình
                remaining_energy=50.0,  # Giả sử năng lượng còn lại trung bình
                is_critical=(transaction_value > self.transaction_threshold_high)
            )
            
            # Lấy năng lượng đã tiết kiệm từ kết quả
            crypto_energy_saved = crypto_result["energy_saved"]
            
            # Update energy optimization statistics
            self.energy_optimization_stats["total_energy_saved_crypto"] += crypto_energy_saved
            self.energy_optimization_stats["total_operations"] += 1
            self.energy_optimization_stats["security_level_distribution"][security_level] += 1
        
        # Execute the selected consensus protocol
        success, latency, energy = protocol.execute(transaction_value, active_trust_scores)
        
        # Record transaction in history
        self.transaction_history.append({
            "value": transaction_value,
            "shard_id": shard_id,
            "protocol": protocol_name,
            "success": success,
            "latency": latency,
            "energy": energy,
            "timestamp": time.time()
        })
        
        # Keep transaction history to a limited size
        if len(self.transaction_history) > self.transaction_history_size:
            self.transaction_history = self.transaction_history[-self.transaction_history_size:]
        
        # Update protocol performance metrics
        self.update_protocol_performance(protocol_name, success, latency, energy)
        
        # Return success, adjusted latency, adjusted energy (accounting for savings) and the protocol used
        total_energy_saved = pos_energy_saved + crypto_energy_saved
        adjusted_energy = max(1.0, energy - total_energy_saved)
        
        return success, latency, adjusted_energy, protocol_name
    
    def _update_protocol_performance(self, protocol_name: str, success: bool, latency: float, energy: float):
        """
        Update protocol performance statistics.
        
        Args:
            protocol_name: Protocol name
            success: Transaction successful or not
            latency: Transaction latency
            energy: Transaction energy consumption
        """
        if protocol_name in self.protocol_performance:
            stats = self.protocol_performance[protocol_name]
            stats["total_count"] += 1
            if success:
                stats["success_count"] += 1
            stats["latency_sum"] += latency
            stats["energy_sum"] += energy
            
            # Keep statistics in recent history range
            if stats["total_count"] > self.transaction_history_size:
                # Reduce all values proportionally
                ratio = self.transaction_history_size / stats["total_count"]
                stats["total_count"] = self.transaction_history_size
                stats["success_count"] = int(stats["success_count"] * ratio)
                stats["latency_sum"] *= ratio
                stats["energy_sum"] *= ratio
    
    def get_protocol_factors(self, protocol_name: str) -> Tuple[float, float, float]:
        """
        Get performance factors of a protocol.
        
        Args:
            protocol_name: Protocol name
            
        Returns:
            Tuple[float, float, float]: (latency_factor, energy_factor, security_factor)
        """
        if protocol_name in self.consensus_protocols:
            protocol = self.consensus_protocols[protocol_name]
            return protocol.latency_factor, protocol.energy_factor, protocol.security_factor
        else:
            # Default value if protocol not found
            return 0.5, 0.6, 0.8
    
    def get_bls_metrics(self) -> Dict[str, float]:
        """
        Get BLS signature aggregation performance metrics.
        
        Returns:
            Dict[str, float]: BLS metrics or None if not enabled
        """
        if self.enable_bls and "BLS_Consensus" in self.consensus_protocols:
            return self.consensus_protocols["BLS_Consensus"].get_performance_metrics()
        return None
    
    def get_pos_statistics(self) -> Dict[str, Any]:
        """
        Get Adaptive PoS performance statistics.
        
        Returns:
            Dict[str, Any]: Adaptive PoS statistics
        """
        if not self.enable_adaptive_pos:
            return {"enabled": False}
        
        result = {
            "enabled": True,
            "total_energy_saved": self.total_energy_saved,
            "total_rotations": self.total_rotations,
            "shard_stats": {}
        }
        
        # Collect statistics from each shard
        for shard_id, pos_manager in self.pos_managers.items():
            result["shard_stats"][shard_id] = {
                "energy": pos_manager.get_energy_statistics(),
                "validators": pos_manager.get_validator_statistics()
            }
        
        return result
    
    def select_committee_for_shard(self, shard_id: int, committee_size: int, 
                                  trust_scores: Dict[int, float] = None) -> List[int]:
        """
        Select validator committee for a shard using Adaptive PoS.
        
        Args:
            shard_id: ID of shard
            committee_size: Committee size
            trust_scores: Trust scores of validators
            
        Returns:
            List[int]: List of selected validator IDs
        """
        if not self.enable_adaptive_pos or shard_id not in self.pos_managers:
            # If not using Adaptive PoS, return random list
            return list(range(1, min(committee_size + 1, self.num_validators_per_shard + 1)))
        
        # Use Adaptive PoS to select committee
        pos_manager = self.pos_managers[shard_id]
        return pos_manager.select_validators_for_committee(committee_size, trust_scores)

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """
        Get optimization statistics.
        
        Returns:
            Dict[str, Any]: Optimization statistics
        """
        stats = {
            "adaptive_pos": {
                "enabled": self.enable_adaptive_pos,
                "total_energy_saved": self.total_energy_saved,
                "total_rotations": self.total_rotations
            },
            "lightweight_crypto": {
                "enabled": self.enable_lightweight_crypto,
                "total_energy_saved": self.energy_optimization_stats["total_energy_saved_crypto"],
                "total_operations": self.energy_optimization_stats["total_operations"],
                "security_distribution": self.energy_optimization_stats["security_level_distribution"]
            }
        }
        
        # Add detailed statistics from crypto manager if enabled
        if self.enable_lightweight_crypto and self.crypto_manager is not None:
            crypto_detailed_stats = self.crypto_manager.get_crypto_statistics()
            stats["lightweight_crypto"]["detailed"] = crypto_detailed_stats
        
        # Add BLS statistics
        if self.enable_bls and "BLS_Consensus" in self.consensus_protocols:
            bls_stats = self.get_bls_metrics()
            stats["bls_signature_aggregation"] = {
                "enabled": self.enable_bls,
                "metrics": bls_stats
            }
        
        # Calculate total energy savings from all mechanisms
        total_savings = self.total_energy_saved
        if self.enable_lightweight_crypto:
            total_savings += self.energy_optimization_stats["total_energy_saved_crypto"]
        
        stats["total_energy_saved"] = total_savings
        
        return stats

    def update_protocol_performance(self, protocol_name: str, success: bool, latency: float, energy: float) -> None:
        """
        Update performance metrics for a protocol.
        
        Args:
            protocol_name: Name of the protocol
            success: Whether consensus was successful
            latency: Latency of this execution
            energy: Energy consumed by this execution
        """
        # Update performance stats
        if protocol_name in self.protocol_performance:
            perf = self.protocol_performance[protocol_name]
            perf["total_count"] += 1
            if success:
                perf["success_count"] += 1
            perf["latency_sum"] += latency
            perf["energy_sum"] += energy
    
    def get_protocol_success_rate(self, protocol_name: str) -> float:
        """
        Get success rate for a protocol.
        
        Args:
            protocol_name: Name of the protocol
        
        Returns:
            float: Success rate
        """
        if protocol_name in self.protocol_performance:
            perf = self.protocol_performance[protocol_name]
            if perf["total_count"] > 0:
                return perf["success_count"] / perf["total_count"]
        return 0.8  # Default success rate
    
    def get_protocol_avg_latency(self, protocol_name: str) -> float:
        """
        Get average latency for a protocol.
        
        Args:
            protocol_name: Name of the protocol
        
        Returns:
            float: Average latency
        """
        if protocol_name in self.protocol_performance:
            perf = self.protocol_performance[protocol_name]
            if perf["total_count"] > 0:
                return perf["latency_sum"] / perf["total_count"]
        return 50.0  # Default average latency
    
    def get_protocol_avg_energy(self, protocol_name: str) -> float:
        """
        Get average energy consumption for a protocol.
        
        Args:
            protocol_name: Name of the protocol
        
        Returns:
            float: Average energy consumption
        """
        if protocol_name in self.protocol_performance:
            perf = self.protocol_performance[protocol_name]
            if perf["total_count"] > 0:
                return perf["energy_sum"] / perf["total_count"]
        return 25.0  # Default average energy consumption
    
    def calculate_network_stability(self) -> float:
        """
        Calculate network stability based on transaction history.
        
        Returns:
            float: Network stability factor (0.0-1.0)
        """
        # Calculate success rates of all recent transactions
        if not self.transaction_history:
            return 0.7  # Default stability
        
        # Get success rates from most recent 100 transactions
        recent_history = self.transaction_history[-min(len(self.transaction_history), 100):]
        stability = sum(1 for tx in recent_history if tx.get("success", False)) / len(recent_history)
        
        return min(1.0, max(0.0, stability))
    
    def calculate_congestion_level(self) -> float:
        """
        Calculate network congestion level based on transaction history.
        
        Returns:
            float: Congestion level (0.0-1.0)
        """
        # Default congestion level if no history
        if not self.transaction_history:
            return 0.4  # Default congestion
        
        # Calculate congestion level based on the number of 
        # transactions in the last 60 seconds
        current_time = time.time()
        recent_tx_count = sum(1 for tx in self.transaction_history 
                            if current_time - tx.get("timestamp", 0) <= 60)
        
        # Normalize to get congestion level
        # Assuming more than 100 transactions in 60 seconds is high congestion
        congestion = min(1.0, max(0.0, recent_tx_count / 100.0))
        
        return congestion
    
    def select_consensus_protocol(self, 
                                transaction_value: float,
                                shard_id: int,
                                trust_scores: Dict[int, float]) -> str:
        """
        Select the most appropriate consensus protocol based on factors.
        
        Args:
            transaction_value: Value of the transaction
            shard_id: Shard ID
            trust_scores: Trust scores of nodes
            
        Returns:
            str: Name of the selected consensus protocol
        """
        # If BLS consensus is enabled and this is a specific case for it, use it
        if (self.enable_bls and
            len(trust_scores) >= self.num_validators_per_shard * 0.7 and
            transaction_value > self.transaction_threshold_low and
            transaction_value < self.transaction_threshold_high):
            return "BLS_Consensus"
        
        # Calculate network stability
        network_stability = self.calculate_network_stability()
        
        # Calculate network congestion
        congestion_level = self.calculate_congestion_level()
        
        # Calculate average trust score
        avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 0.5
        
        # Check if light BFT can be used
        if (network_stability >= self.high_stability_threshold and
            avg_trust >= self.high_trust_threshold and
            transaction_value <= self.transaction_threshold_low and
            congestion_level < self.congestion_threshold):
            return "LightBFT"
        
        # Calculate scores for each protocol
        protocol_scores = {}
        
        for name, protocol in self.consensus_protocols.items():
            # Skip BLS Consensus which is handled above
            if name == "BLS_Consensus":
                continue
                
            # Protocol basic scores
            success_rate = self.get_protocol_success_rate(name)
            latency_score = 1.0 - (self.get_protocol_avg_latency(name) / 100.0)
            energy_score = 1.0 - (self.get_protocol_avg_energy(name) / 50.0)
            
            # Combine factors with weights
            stability_factor = network_stability_weight = (
                0.5 * protocol.security_factor + 
                0.5 * (1.0 - protocol.latency_factor)
            )
            
            value_factor = transaction_value_weight = (
                transaction_value / self.transaction_threshold_high
                if transaction_value < self.transaction_threshold_high
                else 1.0
            )
            
            congestion_factor = congestion_weight = congestion_level
            trust_factor = trust_weight = (
                (1.0 - avg_trust) 
                if avg_trust >= self.min_trust_threshold
                else 1.0
            )
            
            # Calculate final score
            score = (
                success_rate * 0.2 +
                latency_score * 0.1 +
                energy_score * 0.1 +
                (1.0 - abs(stability_factor - protocol.security_factor)) * self.network_stability_weight +
                (1.0 - abs(value_factor - protocol.security_factor)) * self.transaction_value_weight +
                (1.0 - abs(congestion_factor - (1.0 - protocol.latency_factor))) * self.congestion_weight +
                (1.0 - abs(trust_factor - protocol.security_factor)) * self.trust_weight
            )
            
            protocol_scores[name] = score
        
        # Select protocol with highest score
        best_protocol = max(protocol_scores, key=protocol_scores.get)
        
        # Update protocol usage statistics
        self.protocol_usage[best_protocol] = self.protocol_usage.get(best_protocol, 0) + 1
        
        # Update shard protocol for performance tracking
        if shard_id is not None:
            self.shard_protocols[shard_id] = best_protocol
        
        return best_protocol
    
    def get_protocol_usage_statistics(self) -> Dict[str, float]:
        """
        Get statistics about protocol usage.
        
        Returns:
            Dict[str, float]: Protocol usage percentage
        """
        total_usage = sum(self.protocol_usage.values())
        if total_usage == 0:
            return {name: 0.0 for name in self.protocol_usage.keys()}
        
        return {name: count / total_usage for name, count in self.protocol_usage.items()}
    
    def get_energy_optimization_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about energy optimization.
        
        Returns:
            Dict[str, Any]: Energy optimization statistics
        """
        stats = {
            "adaptive_pos_energy_saved": self.total_energy_saved,
            "lightweight_crypto_energy_saved": (
                self.energy_optimization_stats["total_energy_saved_crypto"]
                if self.enable_lightweight_crypto
                else 0.0
            ),
            "total_energy_saved": (
                self.total_energy_saved + 
                self.energy_optimization_stats["total_energy_saved_crypto"]
                if self.enable_lightweight_crypto
                else self.total_energy_saved
            ),
            "security_level_distribution": (
                self.energy_optimization_stats["security_level_distribution"]
                if self.enable_lightweight_crypto
                else {"low": 0, "medium": 0, "high": 0}
            )
        }
        
        return stats
    
    def reset_statistics(self) -> None:
        """
        Reset performance statistics.
        """
        self.protocol_usage = {name: 0 for name in self.consensus_protocols.keys()}
        self.transaction_history = []
        self.total_energy_saved = 0.0
        self.total_rotations = 0
        
        for name in self.consensus_protocols.keys():
            self.protocol_performance[name] = {
                "total_count": 10,
                "success_count": 8,
                "latency_sum": 500.0,
                "energy_sum": 250.0
            }
        
        self.energy_optimization_stats = {
            "total_energy_saved_crypto": 0.0,
            "total_operations": 0,
            "security_level_distribution": {"low": 0, "medium": 0, "high": 0}
        }

    def get_trust_scores(self) -> Dict[str, float]:
        """
        Trả về điểm tin cậy của tất cả các validator.
        
        Returns:
            Dict[str, float]: Từ điển validator_id -> trust_score
        """
        if not hasattr(self, 'validators') or not self.validators:
            # Khởi tạo giá trị mặc định nếu chưa có dữ liệu
            return {f"validator_{i}": 0.7 + (0.3 * random.random()) for i in range(self.num_validators_per_shard)}
            
        # Nếu đã có dữ liệu về validators, sử dụng điểm tin cậy thực tế
        trust_scores = {}
        for validator_id, validator_data in self.validators.items():
            trust_scores[validator_id] = validator_data.get('trust_score', 0.7 + (0.3 * random.random()))
            
        return trust_scores
        
    def update_trust_scores(self, validator_updates: Dict[str, float]) -> None:
        """
        Cập nhật điểm tin cậy cho các validator.
        
        Args:
            validator_updates: Từ điển validator_id -> trust_score_mới
        """
        if not hasattr(self, 'validators'):
            self.validators = {}
            
        for validator_id, new_score in validator_updates.items():
            # Đảm bảo validator tồn tại
            if validator_id not in self.validators:
                self.validators[validator_id] = {
                    'trust_score': 0.7,  # Giá trị mặc định
                    'transactions_processed': 0,
                    'successful_transactions': 0
                }
                
            # Cập nhật điểm tin cậy mới và đảm bảo nằm trong khoảng [0,1]
            self.validators[validator_id]['trust_score'] = max(0.0, min(1.0, new_score))
            
        logger.info(f"Updated trust scores for {len(validator_updates)} validators")