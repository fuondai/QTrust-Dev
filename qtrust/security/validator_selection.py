"""
Validator Selection Module for QTrust Blockchain System

This module provides mechanisms for selecting validators in a blockchain system based on
reputation, security, and performance metrics. It implements various selection policies
and validator rotation strategies to ensure decentralization and system integrity.

Key features:
- Reputation-based selection of validators
- Support for various selection policies (random, reputation, stake-weighted, performance, hybrid)
- Validator rotation to prevent concentration of power
- Integration with Zero-Knowledge Proofs for secure and verifiable selection
- Performance tracking and statistics generation
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from .zk_proofs import ZKProofSystem, ProofType

class ValidatorSelectionPolicy:
    """Validator selection policies."""
    RANDOM = "random"  # Random selection
    REPUTATION = "reputation"  # Based on reputation
    STAKE_WEIGHTED = "stake_weighted"  # Based on stake
    PERFORMANCE = "performance"  # Based on performance
    HYBRID = "hybrid"  # Combination of criteria

class ReputationBasedValidatorSelection:
    """
    Reputation-based validator selection system with security features.
    
    Integrates with Zero-Knowledge Proofs to select validators in a 
    fair and secure manner.
    """
    
    def __init__(self, 
                trust_manager, 
                policy: str = ValidatorSelectionPolicy.HYBRID,
                zk_enabled: bool = True,
                security_level: str = "medium",
                use_rotation: bool = True,
                rotation_period: int = 20):
        """
        Initialize the validator selection system.
        
        Args:
            trust_manager: System trust manager
            policy: Validator selection policy
            zk_enabled: Enable ZK Proofs
            security_level: Security level ("low", "medium", "high")
            use_rotation: Enable validator rotation
            rotation_period: Rotation period (number of blocks)
        """
        self.trust_manager = trust_manager
        self.policy = policy
        self.zk_enabled = zk_enabled
        self.use_rotation = use_rotation
        self.rotation_period = rotation_period
        
        # Initialize ZK proof system if enabled
        if zk_enabled:
            self.zk_system = ZKProofSystem(security_level=security_level)
        else:
            self.zk_system = None
        
        # Track current validators and history
        self.active_validators = {}  # Shard ID -> List of validator IDs
        self.validator_history = defaultdict(list)  # Shard ID -> List of (round, validators) tuples
        self.current_round = 0
        
        # Performance statistics
        self.stats = {
            "selections": 0,
            "rotations": 0,
            "energy_saved": 0.0,
            "validator_diversity": 0.0,
            "avg_selection_time": 0.0,
            "total_selection_time": 0.0
        }
    
    def select_validators(self, shard_id: int, block_num: int, num_validators: int = 3) -> List[int]:
        """
        Select validators for a specific block/shard.
        
        Args:
            shard_id: Shard ID
            block_num: Block number
            num_validators: Number of validators to select
            
        Returns:
            List[int]: List of selected validator IDs
        """
        start_time = time.time()
        self.current_round = block_num
        
        # Check if validators need to be rotated
        needs_rotation = self.use_rotation and block_num > 0 and block_num % self.rotation_period == 0
        
        # If validators already exist and no rotation needed, keep current ones
        if shard_id in self.active_validators and not needs_rotation:
            selected_validators = self.active_validators[shard_id]
            
            # Record history
            self.validator_history[shard_id].append((block_num, selected_validators.copy()))
            
            # Update statistics
            end_time = time.time()
            selection_time = end_time - start_time
            self.stats["selections"] += 1
            self.stats["total_selection_time"] += selection_time
            self.stats["avg_selection_time"] = self.stats["total_selection_time"] / self.stats["selections"]
            
            return selected_validators
        
        # Select new validators (rotation or first time)
        if needs_rotation:
            self.stats["rotations"] += 1
        
        # Get list of trusted nodes from trust manager
        trusted_nodes = self.trust_manager.recommend_trusted_validators(
            shard_id, count=min(10, num_validators * 3)
        )
        
        # Apply selection policy
        if not trusted_nodes:
            # If no trusted information, randomly select from all nodes in shard
            if hasattr(self.trust_manager, "shards") and self.trust_manager.shards:
                all_shard_nodes = self.trust_manager.shards[shard_id]
                selected_validators = np.random.choice(
                    all_shard_nodes, 
                    size=min(num_validators, len(all_shard_nodes)), 
                    replace=False
                ).tolist()
            else:
                # No information about shards, return empty list
                selected_validators = []
        else:
            # Filter nodes by policy
            filtered_nodes = self._apply_selection_policy(trusted_nodes, block_num)
            
            # If rotation needed, add stronger randomness
            if needs_rotation and len(filtered_nodes) > num_validators:
                # Save current validators to ensure new ones are different
                current_validators = set()
                if shard_id in self.active_validators:
                    current_validators = set(self.active_validators[shard_id])
                
                # When rotating, prioritize nodes that haven't been validators recently
                if shard_id in self.validator_history and len(self.validator_history[shard_id]) > 0:
                    recent_validators = set()
                    for _, validators in self.validator_history[shard_id][-3:]:  # 3 most recent blocks
                        recent_validators.update(validators)
                    
                    # Prioritize nodes that haven't been validators recently
                    non_recent = [node for node in filtered_nodes if node["node_id"] not in recent_validators]
                    
                    # If enough new nodes, prioritize them
                    if len(non_recent) >= num_validators:
                        filtered_nodes = non_recent
                
                # Add stronger randomness
                randomness = np.random.random(len(filtered_nodes)) * 0.3  # Add up to 30% randomness
                adjusted_scores = [node["composite_score"] * (1 + rand) for node, rand in zip(filtered_nodes, randomness)]
                nodes_with_scores = list(zip(filtered_nodes, adjusted_scores))
                nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
                filtered_nodes = [node for node, _ in nodes_with_scores]
            
            # Select best nodes
            selected_validators = [node["node_id"] for node in filtered_nodes[:num_validators]]
            
            # Ensure validators change during rotation
            if needs_rotation and shard_id in self.active_validators:
                current_set = set(self.active_validators[shard_id])
                new_set = set(selected_validators)
                
                # If no change, force at least 1 validator change
                if current_set == new_set and len(filtered_nodes) > num_validators:
                    # Choose a current validator to replace
                    validator_to_replace = np.random.choice(list(current_set))
                    
                    # Find new validator not in current set
                    new_candidates = [node["node_id"] for node in filtered_nodes[num_validators:] 
                                     if node["node_id"] not in current_set]
                    
                    if new_candidates:
                        # Replace old validator with new one
                        selected_validators = list(new_set)
                        selected_validators.remove(validator_to_replace)
                        selected_validators.append(np.random.choice(new_candidates))
            
            # Generate ZK proof if enabled
            if self.zk_enabled and self.zk_system:
                self._generate_selection_proof(shard_id, block_num, selected_validators)
        
        # Update current validators and history
        self.active_validators[shard_id] = selected_validators
        self.validator_history[shard_id].append((block_num, selected_validators.copy()))
        
        # Update statistics
        end_time = time.time()
        selection_time = end_time - start_time
        self.stats["selections"] += 1
        self.stats["total_selection_time"] += selection_time
        self.stats["avg_selection_time"] = self.stats["total_selection_time"] / self.stats["selections"]
        
        # Calculate validator diversity
        if len(self.validator_history[shard_id]) >= 2:
            self._calculate_validator_diversity(shard_id)
        
        return selected_validators
    
    def _apply_selection_policy(self, trusted_nodes: List[Dict[str, Any]], block_num: int) -> List[Dict[str, Any]]:
        """
        Apply validator selection policy.
        
        Args:
            trusted_nodes: List of trusted nodes with detailed information
            block_num: Block number
            
        Returns:
            List[Dict[str, Any]]: List of nodes filtered by policy
        """
        if self.policy == ValidatorSelectionPolicy.RANDOM:
            # Random selection, regardless of score
            np.random.shuffle(trusted_nodes)
            return trusted_nodes
            
        elif self.policy == ValidatorSelectionPolicy.REPUTATION:
            # Sort by trust score
            return sorted(trusted_nodes, key=lambda x: x["trust_score"], reverse=True)
            
        elif self.policy == ValidatorSelectionPolicy.STAKE_WEIGHTED:
            # If stake information exists, sort by it
            if all("stake" in node for node in trusted_nodes):
                return sorted(trusted_nodes, key=lambda x: x["stake"], reverse=True)
            else:
                # Otherwise, use composite score
                return sorted(trusted_nodes, key=lambda x: x["composite_score"], reverse=True)
                
        elif self.policy == ValidatorSelectionPolicy.PERFORMANCE:
            # Sort by success rate and response time
            return sorted(trusted_nodes, 
                         key=lambda x: (x["success_rate"], -x["response_time"]), 
                         reverse=True)
                
        elif self.policy == ValidatorSelectionPolicy.HYBRID:
            # Combine multiple criteria with different weights
            # Use composite score calculated by trust manager
            nodes = sorted(trusted_nodes, key=lambda x: x["composite_score"], reverse=True)
            
            # Add small randomness to avoid always selecting the same nodes
            # Only apply when more nodes than needed
            if len(nodes) > 3:
                randomness = np.random.random(len(nodes)) * 0.05  # Add up to 5% randomness
                adjusted_scores = [node["composite_score"] * (1 + rand) for node, rand in zip(nodes, randomness)]
                nodes_with_scores = list(zip(nodes, adjusted_scores))
                nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
                nodes = [node for node, _ in nodes_with_scores]
            
            return nodes
        
        # Default, return original list
        return trusted_nodes
    
    def _generate_selection_proof(self, shard_id: int, block_num: int, selected_validators: List[int]):
        """
        Generate ZK proof for validator selection.
        
        Args:
            shard_id: Shard ID
            block_num: Block number
            selected_validators: List of selected validators
        """
        if not self.zk_system:
            return
            
        # Create data for proof
        proof_data = {
            "shard_id": shard_id,
            "block_num": block_num,
            "validators": selected_validators,
            "policy": self.policy,
            "timestamp": time.time()
        }
        
        # Generate proof
        proof = self.zk_system.generate_proof(
            data=proof_data,
            proof_type=ProofType.SET_MEMBERSHIP
        )
        
        # Update energy saved
        if "energy_cost" in proof:
            self.stats["energy_saved"] += proof.get("energy_cost", 0) * 0.3
    
    def _calculate_validator_diversity(self, shard_id: int):
        """
        Calculate validator diversity across rounds.
        
        Args:
            shard_id: Shard ID
        """
        if shard_id not in self.validator_history or len(self.validator_history[shard_id]) < 2:
            return
            
        # Get recent validator sets
        recent_history = self.validator_history[shard_id][-10:]  # Maximum 10 most recent rounds
        
        # Calculate proportion of unique validators
        unique_validators = set()
        for _, validators in recent_history:
            unique_validators.update(validators)
        
        total_slots = sum(len(validators) for _, validators in recent_history)
        diversity = len(unique_validators) / total_slots if total_slots > 0 else 0
        
        # Update statistics
        self.stats["validator_diversity"] = diversity
    
    def verify_selection(self, shard_id: int, block_num: int, validators: List[int]) -> bool:
        """
        Verify validity of a selected validator set.
        
        Args:
            shard_id: Shard ID
            block_num: Block number
            validators: List of validators to verify
            
        Returns:
            bool: True if validator set is valid
        """
        # Check if matches with history
        if shard_id in self.validator_history:
            for round_num, round_validators in self.validator_history[shard_id]:
                if round_num == block_num:
                    return set(validators) == set(round_validators)
        
        # If not found in history, perform validity check
        # Principle: validators must have high trust scores
        trusted_nodes = self.trust_manager.recommend_trusted_validators(
            shard_id, count=len(validators) * 2
        )
        
        if not trusted_nodes:
            return False
            
        trusted_ids = {node["node_id"] for node in trusted_nodes}
        return all(validator in trusted_ids for validator in validators)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the validator selection system.
        
        Returns:
            Dict[str, Any]: Usage statistics
        """
        stats = self.stats.copy()
        
        # Add info from ZK system if present
        if self.zk_enabled and self.zk_system:
            zk_stats = self.zk_system.get_statistics()
            stats["zk_proofs_generated"] = zk_stats["proofs_generated"]
            stats["zk_proofs_verified"] = zk_stats["proofs_verified"]
            stats["zk_energy_saved"] = zk_stats["energy_saved"]
        
        # Add info about configuration
        stats.update({
            "policy": self.policy,
            "zk_enabled": self.zk_enabled,
            "use_rotation": self.use_rotation,
            "rotation_period": self.rotation_period,
            "active_shards": len(self.active_validators),
            "total_rounds": self.current_round
        })
        
        return stats 