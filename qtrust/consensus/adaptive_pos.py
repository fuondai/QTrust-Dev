"""
Adaptive Proof of Stake (PoS) implementation for QTrust blockchain.

This module provides energy-efficient validator management with automatic rotation
based on performance, energy state, and trust scores.
"""
import random
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import math


class ValidatorStakeInfo:
    """
    Stores information about a validator's stake and status.
    """
    def __init__(self, id: int, initial_stake: float = 100.0, max_energy: float = 100.0):
        """
        Initialize validator information.
        
        Args:
            id: Validator ID
            initial_stake: Initial stake amount
            max_energy: Maximum energy (used for battery energy simulation)
        """
        self.id = id
        self.stake = initial_stake
        self.max_energy = max_energy
        self.current_energy = max_energy
        self.active = True
        self.last_active_time = time.time()
        self.active_rounds = 0
        self.total_rounds = 0
        self.successful_validations = 0
        self.failed_validations = 0
        self.energy_consumption_history = []
        self.performance_score = 0.5  # performance score (0.0-1.0)
        self.participation_rate = 1.0  # participation rate (0.0-1.0)
        self.last_rotation_time = time.time()
        
    def update_stake(self, delta: float):
        """Update stake amount."""
        self.stake = max(0.0, self.stake + delta)
        
    def consume_energy(self, amount: float) -> bool:
        """
        Consume energy and return True if there is enough energy.
        """
        if self.current_energy >= amount:
            self.current_energy -= amount
            self.energy_consumption_history.append(amount)
            if len(self.energy_consumption_history) > 100:
                self.energy_consumption_history.pop(0)
            return True
        return False
    
    def recharge_energy(self, amount: float = None):
        """
        Recharge validator's energy.
        
        Args:
            amount: Amount of energy to recharge, if None, recharge to full
        """
        if amount is None:
            self.current_energy = self.max_energy
        else:
            self.current_energy = min(self.max_energy, self.current_energy + amount)
    
    def update_performance(self, success: bool):
        """
        Update performance score based on validation results.
        """
        self.total_rounds += 1
        alpha = 0.1  # learning rate
        
        if success:
            self.successful_validations += 1
            self.performance_score = (1 - alpha) * self.performance_score + alpha * 1.0
        else:
            self.failed_validations += 1
            self.performance_score = (1 - alpha) * self.performance_score
        
        if self.active:
            self.active_rounds += 1
        
        self.participation_rate = self.active_rounds / max(1, self.total_rounds)
    
    def get_average_energy_consumption(self) -> float:
        """Return average energy consumption."""
        if not self.energy_consumption_history:
            return 0.0
        return sum(self.energy_consumption_history) / len(self.energy_consumption_history)


class AdaptivePoSManager:
    """
    Manager for adaptive Proof of Stake (PoS) mechanism with validator rotation.
    """
    def __init__(self, 
                 num_validators: int = 20,
                 active_validator_ratio: float = 0.7,
                 rotation_period: int = 50,
                 min_stake: float = 10.0,
                 energy_threshold: float = 30.0,
                 performance_threshold: float = 0.3,
                 energy_optimization_level: str = "balanced",
                 enable_smart_energy_management: bool = True,
                 seed: int = 42):
        """
        Initialize AdaptivePoSManager.
        
        Args:
            num_validators: Total number of validators
            active_validator_ratio: Ratio of active validators (0.0-1.0)
            rotation_period: Number of rounds before considering rotation
            min_stake: Minimum stake required to be selected as validator
            energy_threshold: Low energy threshold for replacement
            performance_threshold: Minimum performance threshold
            energy_optimization_level: Energy optimization level ("low", "balanced", "aggressive")
            enable_smart_energy_management: Enable/disable smart energy management
            seed: Seed value for random calculations
        """
        self.num_validators = num_validators
        self.active_validator_ratio = active_validator_ratio
        self.num_active_validators = max(1, int(num_validators * active_validator_ratio))
        self.rotation_period = rotation_period
        self.min_stake = min_stake
        self.energy_threshold = energy_threshold
        self.performance_threshold = performance_threshold
        self.energy_optimization_level = energy_optimization_level
        self.enable_smart_energy_management = enable_smart_energy_management
        self.seed = seed
        random.seed(seed)
        
        # Initialize validators
        self.validators = {i: ValidatorStakeInfo(id=i) for i in range(1, num_validators + 1)}
        
        # Set of active validators
        self.active_validators = set()
        
        # Perform initial selection
        self._select_initial_validators()
        
        # Statistics
        self.rounds_since_rotation = 0
        self.total_rotations = 0
        self.total_rounds = 0
        self.energy_saved = 0.0
        
        # Energy information and predictions
        self.energy_prediction_model = {}  # validator_id -> predicted_energy
        self.energy_efficiency_rankings = {}  # validator_id -> efficiency_rank
        self.historical_energy_usage = []  # energy usage history
        self.energy_optimization_weights = self._get_optimization_weights()
        
    def _get_optimization_weights(self) -> Dict[str, float]:
        """
        Get energy optimization weights based on configuration.
        """
        if self.energy_optimization_level == "low":
            return {
                "energy_weight": 0.2,
                "performance_weight": 0.5,
                "stake_weight": 0.3,
                "rotation_aggressiveness": 0.3
            }
        elif self.energy_optimization_level == "aggressive":
            return {
                "energy_weight": 0.6,
                "performance_weight": 0.2,
                "stake_weight": 0.2,
                "rotation_aggressiveness": 0.8
            }
        else:  # balanced
            return {
                "energy_weight": 0.4,
                "performance_weight": 0.3,
                "stake_weight": 0.3,
                "rotation_aggressiveness": 0.5
            }

    def _select_initial_validators(self):
        """
        Select initial validators based on stake.
        """
        # Sort by stake in descending order
        sorted_validators = sorted(self.validators.items(), 
                                  key=lambda x: x[1].stake, reverse=True)
        
        # Select validators with highest stake
        self.active_validators = set()
        for i in range(min(self.num_active_validators, len(sorted_validators))):
            validator_id = sorted_validators[i][0]
            self.validators[validator_id].active = True
            self.active_validators.add(validator_id)
    
    def select_validator_for_block(self, trust_scores: Dict[int, float] = None) -> int:
        """
        Select validator for next block creation based on stake, trust scores, and energy.
        
        Args:
            trust_scores: Trust scores of validators (can be None)
            
        Returns:
            int: ID of selected validator
        """
        if not self.active_validators:
            self._select_initial_validators()
            if not self.active_validators:
                return None  # No suitable validator found
        
        # Calculate selection probabilities based on stake, trust scores, and energy efficiency
        selection_weights = {}
        total_weight = 0.0
        
        for validator_id in self.active_validators:
            validator = self.validators[validator_id]
            # Check minimum energy
            if validator.current_energy < self.energy_threshold / 2:
                continue  # Skip validator with too low energy
                
            # Combine stake with trust score (if available)
            trust_factor = trust_scores.get(validator_id, 0.5) if trust_scores else 0.5
            
            # Get energy efficiency
            energy_rank = self.energy_efficiency_rankings.get(validator_id, self.num_validators)
            energy_efficiency = 1.0 - (energy_rank / self.num_validators)  # 0.0-1.0, higher is better
            
            # Calculate weights based on factors
            stake_component = validator.stake * self.energy_optimization_weights["stake_weight"]
            performance_component = validator.performance_score * self.energy_optimization_weights["performance_weight"]
            energy_component = energy_efficiency * self.energy_optimization_weights["energy_weight"]
            trust_component = trust_factor * 0.1  # Small weight for trust
            
            weight = stake_component + performance_component + energy_component + trust_component
            
            selection_weights[validator_id] = weight
            total_weight += weight
        
        # Select validator based on weighted probability
        if total_weight <= 0 or not selection_weights:
            # If total weight is 0 or no suitable validator, select randomly
            return random.choice(list(self.active_validators))
        
        # Select based on weighted probability
        selection_point = random.uniform(0, total_weight)
        current_sum = 0.0
        
        for validator_id, weight in selection_weights.items():
            current_sum += weight
            if current_sum >= selection_point:
                return validator_id
        
        # Fallback if calculation error
        return random.choice(list(self.active_validators))
    
    def select_validators_for_committee(self, 
                                      committee_size: int, 
                                      trust_scores: Dict[int, float] = None) -> List[int]:
        """
        Select a validator committee for consensus.
        
        Args:
            committee_size: Committee size
            trust_scores: Trust scores of validators
            
        Returns:
            List[int]: List of selected validator IDs
        """
        actual_committee_size = min(committee_size, len(self.active_validators))
        
        if actual_committee_size <= 0:
            return []
            
        # Create list of validators with weights
        weighted_validators = []
        for validator_id in self.active_validators:
            validator = self.validators[validator_id]
            
            # Calculate weights based on stake, performance, and trust score
            trust_factor = trust_scores.get(validator_id, 0.5) if trust_scores else 0.5
            weight = validator.stake * validator.performance_score * (0.5 + 0.5 * trust_factor)
            
            # Check remaining energy
            if validator.current_energy >= self.energy_threshold:
                weighted_validators.append((validator_id, weight))
        
        # If not enough validators with energy, add low energy validator
        if len(weighted_validators) < actual_committee_size:
            for validator_id in self.active_validators:
                if validator_id not in [v[0] for v in weighted_validators]:
                    validator = self.validators[validator_id]
                    trust_factor = trust_scores.get(validator_id, 0.5) if trust_scores else 0.5
                    weight = validator.stake * validator.performance_score * (0.5 + 0.5 * trust_factor)
                    weighted_validators.append((validator_id, weight))
        
        # Sort by weights and select
        weighted_validators.sort(key=lambda x: x[1], reverse=True)
        committee = [validator_id for validator_id, _ in weighted_validators[:actual_committee_size]]
        
        # Ensure selection, if necessary select randomly from remaining validators
        remaining = actual_committee_size - len(committee)
        if remaining > 0:
            remaining_validators = [v_id for v_id in self.active_validators if v_id not in committee]
            if remaining_validators:
                committee.extend(random.sample(remaining_validators, min(remaining, len(remaining_validators))))
        
        return committee
    
    def update_validator_energy(self, 
                               validator_id: int, 
                               energy_consumed: float, 
                               transaction_success: bool):
        """
        Update validator's energy and evaluate performance.
        
        Args:
            validator_id: ID of validator
            energy_consumed: Energy consumed
            transaction_success: Transaction successful or not
        """
        if validator_id not in self.validators:
            return
            
        validator = self.validators[validator_id]
        
        # Consume energy
        sufficient_energy = validator.consume_energy(energy_consumed)
        
        # Update energy prediction
        self._update_energy_prediction(validator_id, energy_consumed)
        
        # Update performance
        validator.update_performance(transaction_success)
        
        # Reward stake when transaction successful
        if transaction_success:
            reward = 0.1  # Small reward for each successful transaction
            validator.update_stake(reward)
        
        # Update energy statistics
        self.historical_energy_usage.append({
            "validator_id": validator_id,
            "energy_consumed": energy_consumed,
            "success": transaction_success,
            "remaining_energy": validator.current_energy
        })
        
        # Limit history size
        if len(self.historical_energy_usage) > 1000:
            self.historical_energy_usage = self.historical_energy_usage[-1000:]
        
        # Recalculate energy efficiency
        self._recalculate_energy_efficiency()
        
        # Consider smart energy management
        if self.enable_smart_energy_management:
            self._apply_smart_energy_management(validator_id)
    
    def _update_energy_prediction(self, validator_id: int, energy_consumed: float):
        """
        Update energy prediction model for validator.
        """
        if validator_id not in self.energy_prediction_model:
            self.energy_prediction_model[validator_id] = energy_consumed
        else:
            # Update with weight 0.2 for new value
            current_prediction = self.energy_prediction_model[validator_id]
            self.energy_prediction_model[validator_id] = 0.8 * current_prediction + 0.2 * energy_consumed
    
    def _recalculate_energy_efficiency(self):
        """
        Recalculate energy efficiency for all validators.
        """
        efficiency_scores = {}
        
        for validator_id, validator in self.validators.items():
            # Get validator's energy history
            history = [entry for entry in self.historical_energy_usage 
                      if entry["validator_id"] == validator_id]
            
            if not history:
                efficiency_scores[validator_id] = 0.5  # Default value
                continue
                
            # Calculate success/energy ratio
            total_energy = sum(entry["energy_consumed"] for entry in history)
            successful_txs = sum(1 for entry in history if entry["success"])
            
            if total_energy > 0:
                # Efficiency score = successful transactions / total energy * performance
                efficiency = (successful_txs / total_energy) * validator.performance_score
            else:
                efficiency = 0.0
                
            efficiency_scores[validator_id] = efficiency
        
        # Rank validators by efficiency
        sorted_validators = sorted(efficiency_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Assign rank
        for rank, (validator_id, _) in enumerate(sorted_validators):
            self.energy_efficiency_rankings[validator_id] = rank + 1
    
    def _apply_smart_energy_management(self, validator_id: int):
        """
        Apply smart energy management for validator.
        """
        validator = self.validators[validator_id]
        
        # Check if energy is low
        if validator.current_energy < self.energy_threshold:
            # If validator is active, consider temporarily pausing to save energy
            if validator_id in self.active_validators and len(self.active_validators) > self.num_active_validators / 2:
                self.active_validators.remove(validator_id)
                validator.active = False
                
                # Calculate saved energy
                predicted_energy = self.energy_prediction_model.get(validator_id, 10.0)
                self.energy_saved += predicted_energy
                
                # Replace with higher energy validator
                self._add_replacement_validator()
    
    def _add_replacement_validator(self):
        """
        Add replacement validator based on energy efficiency.
        """
        # Find inactive validator with good energy efficiency and enough energy
        candidates = []
        
        for validator_id, validator in self.validators.items():
            if validator_id not in self.active_validators:
                if validator.stake >= self.min_stake and validator.current_energy >= 2 * self.energy_threshold:  # Ensure enough energy
                    efficiency_rank = self.energy_efficiency_rankings.get(validator_id, float('inf'))
                    candidates.append((validator_id, efficiency_rank, validator.current_energy))
        
        if not candidates:
            return
            
        # Select validator based on efficiency rank and remaining energy
        candidates.sort(key=lambda x: (x[1], -x[2]))  # Sort by rank, then energy (highest)
        
        if candidates:
            new_validator_id = candidates[0][0]
            self.active_validators.add(new_validator_id)
            self.validators[new_validator_id].active = True

    def rotate_validators(self, trust_scores: Dict[int, float] = None) -> int:
        """
        Rotate validators to balance energy and performance.
        
        Args:
            trust_scores: Trust scores of validators
            
        Returns:
            int: Number of rotated validators
        """
        self.rounds_since_rotation += 1
        
        # Check if rotation is needed
        if self.rounds_since_rotation < self.rotation_period:
            return 0
            
        self.rounds_since_rotation = 0
        rotations = 0
        
        # Identify validators to replace
        validators_to_replace = []
        
        # Step 1: Identify low energy validators
        for validator_id in list(self.active_validators):
            validator = self.validators[validator_id]
            
            # Criteria for replacement:
            # 1. Low energy
            # 2. Poor performance
            # 3. Low energy efficiency rank
            
            energy_criteria = validator.current_energy < self.energy_threshold
            performance_criteria = validator.performance_score < self.performance_threshold
            
            energy_rank = self.energy_efficiency_rankings.get(validator_id, float('inf'))
            efficiency_criteria = energy_rank > 0.7 * self.num_validators  # Bottom 30%
            
            # Weight deciding based on energy optimization level
            if energy_criteria:
                replace_score = 0.6
            else:
                replace_score = 0
                
            if performance_criteria:
                replace_score += 0.3
                
            if efficiency_criteria:
                replace_score += 0.3 * self.energy_optimization_weights["rotation_aggressiveness"]
                
            # Decide to replace if score exceeds threshold
            if replace_score >= 0.5:
                validators_to_replace.append(validator_id)
        
        # Limit number of validators to replace in one rotation
        max_rotations = max(1, int(self.num_active_validators * 0.3))  # Max 30%
        if len(validators_to_replace) > max_rotations:
            # Prioritize lowest energy validator
            validators_to_replace.sort(
                key=lambda v_id: (
                    self.validators[v_id].current_energy,
                    self.validators[v_id].performance_score
                )
            )
            validators_to_replace = validators_to_replace[:max_rotations]
        
        # Step 2: Perform replacement
        for validator_id in validators_to_replace:
            # Remove from active list
            self.active_validators.remove(validator_id)
            self.validators[validator_id].active = False
            
            # Find replacement validator
            # Prioritize validator with good energy efficiency
            replacement_found = self._find_replacement_validator()
            
            if replacement_found:
                rotations += 1
                # Calculate saved energy
                predicted_energy = self.energy_prediction_model.get(validator_id, 10.0)
                self.energy_saved += predicted_energy
        
        self.total_rotations += rotations
        return rotations
    
    def _find_replacement_validator(self) -> bool:
        """
        Find suitable replacement validator.
        
        Returns:
            bool: True if replacement found
        """
        candidates = []
        
        for validator_id, validator in self.validators.items():
            # Check validator not active
            if validator_id not in self.active_validators:
                # Check sufficient stake and energy
                if validator.stake >= self.min_stake and validator.current_energy >= 2 * self.energy_threshold:
                    # Get necessary information
                    energy_score = validator.current_energy / 100.0  # 0.0-1.0
                    energy_rank = self.energy_efficiency_rankings.get(validator_id, self.num_validators)
                    rank_score = 1.0 - (energy_rank / self.num_validators)  # 0.0-1.0
                    performance_score = validator.performance_score
                    
                    # Calculate candidate score
                    candidate_score = (
                        energy_score * 0.4 +
                        rank_score * 0.4 +
                        performance_score * 0.1
                    )
                    
                    candidates.append((validator_id, candidate_score))
        
        if candidates:
            # Sort by score in descending order
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Select validator with highest score
            best_candidate_id = candidates[0][0]
            self.active_validators.add(best_candidate_id)
            self.validators[best_candidate_id].active = True
            return True
            
        return False

    def update_energy_recharge(self, recharge_rate: float = 0.02):
        """
        Update energy recharge rate for inactive validators.
        
        Args:
            recharge_rate: Maximum recharge rate allowed per call
        """
        for validator_id, validator in self.validators.items():
            if not validator.active:
                # Recharge energy for inactive validator
                recharge_amount = validator.max_energy * recharge_rate
                validator.recharge_energy(recharge_amount)
                
                # Check if validator meets conditions to join again
                if (validator.current_energy >= 0.9 * validator.max_energy and 
                    time.time() - validator.last_rotation_time >= self.rotation_period):
                    # Mark validator as candidate ready to join again
                    validator.performance_score = max(0.4, validator.performance_score)
    
    def get_energy_statistics(self) -> Dict[str, float]:
        """
        Get energy statistics of validators.
        
        Returns:
            Dict[str, float]: Energy statistics
        """
        total_energy = 0.0
        active_energy = 0.0
        inactive_energy = 0.0
        energy_levels = []
        active_energy_levels = []
        
        for validator_id, validator in self.validators.items():
            energy = validator.current_energy
            total_energy += energy
            energy_levels.append(energy)
            
            if validator_id in self.active_validators:
                active_energy += energy
                active_energy_levels.append(energy)
            else:
                inactive_energy += energy
        
        # Calculate additional energy indicators
        avg_energy = total_energy / len(self.validators) if self.validators else 0
        avg_active_energy = active_energy / len(self.active_validators) if self.active_validators else 0
        
        # Predict future consumption rate
        predicted_consumption_rate = 0.0
        if self.historical_energy_usage:
            recent_usage = self.historical_energy_usage[-min(10, len(self.historical_energy_usage)):]
            if recent_usage:
                predicted_consumption_rate = sum(entry["energy_consumed"] for entry in recent_usage) / len(recent_usage)
        
        # Calculate energy difference between validators
        energy_std_dev = np.std(energy_levels) if energy_levels else 0.0
        
        return {
            "total_energy": total_energy,
            "active_energy": active_energy,
            "inactive_energy": inactive_energy,
            "avg_energy": avg_energy,
            "avg_active_energy": avg_active_energy,
            "energy_saved": self.energy_saved,
            "predicted_consumption_rate": predicted_consumption_rate,
            "energy_distribution_std": energy_std_dev,
            "optimization_level": self.energy_optimization_level
        }
    
    def get_validator_statistics(self) -> Dict[str, Any]:
        """
        Get validator statistics.
        
        Returns:
            Dict[str, Any]: Validator statistics
        """
        active_count = len(self.active_validators)
        inactive_count = self.num_validators - active_count
        
        # Calculate average performance
        avg_performance = sum(v.performance_score for v in self.validators.values()) / self.num_validators
        
        # Calculate average energy efficiency score
        avg_energy_efficiency = 0.0
        if self.energy_efficiency_rankings:
            efficiency_scores = [1.0 - (rank / self.num_validators) 
                               for rank in self.energy_efficiency_rankings.values()]
            avg_energy_efficiency = sum(efficiency_scores) / len(efficiency_scores)
        
        # Combine top validators' information about energy efficiency
        top_efficient_validators = sorted(
            [(v_id, 1.0 - (self.energy_efficiency_rankings.get(v_id, float('inf')) / self.num_validators))
             for v_id in self.validators.keys()],
            key=lambda x: x[1], reverse=True
        )[:5]  # Top 5
        
        return {
            "total_validators": self.num_validators,
            "active_validators": active_count,
            "inactive_validators": inactive_count,
            "avg_performance_score": avg_performance,
            "avg_energy_efficiency": avg_energy_efficiency,
            "total_rotations": self.total_rotations,
            "top_energy_efficient": top_efficient_validators,
            "energy_optimization_level": self.energy_optimization_level
        }
    
    def simulate_round(self, trust_scores: Dict[int, float] = None, transaction_value: float = 10.0) -> Dict[str, Any]:
        """
        Simulate a PoS round.
        
        Args:
            trust_scores: Trust scores of validators
            transaction_value: Transaction value
            
        Returns:
            Dict[str, Any]: Simulation result
        """
        # Update round count
        self.total_rounds += 1
        
        # Select validator
        validator_id = self.select_validator_for_block(trust_scores)
        
        if validator_id is None:
            return {
                "success": False,
                "error": "No suitable validator found",
                "rotations": 0,
                "energy_saved": 0.0
            }
            
        validator = self.validators[validator_id]
        
        # Calculate energy consumption for this transaction
        base_energy = 5.0 + 0.1 * transaction_value
        # Reduce energy based on efficiency
        energy_rank = self.energy_efficiency_rankings.get(validator_id, self.num_validators)
        energy_efficiency_factor = 1.0 - 0.3 * (1.0 - energy_rank / self.num_validators)
        energy_consumed = base_energy * energy_efficiency_factor
        
        # Consume energy
        sufficient_energy = validator.consume_energy(energy_consumed)
        
        # Success probability
        success_probability = 0.95
        
        # If energy is not enough, reduce success probability
        if not sufficient_energy:
            success_probability *= 0.5
        
        # Simulate transaction result
        success = random.random() < success_probability
        
        # Update performance
        validator.update_performance(success)
        
        # Update energy prediction model
        self._update_energy_prediction(validator_id, energy_consumed)
        
        # Consider validator rotation
        rotations = self.rotate_validators(trust_scores)
        
        # Recharge energy for inactive validator
        self.update_energy_recharge(0.02)  # 2% per round
        
        # Update energy usage statistics
        self.historical_energy_usage.append({
            "validator_id": validator_id,
            "energy_consumed": energy_consumed,
            "success": success,
            "remaining_energy": validator.current_energy
        })
        
        # Reward/penalty stake
        if success:
            reward = 0.1 * transaction_value
            validator.update_stake(reward)
        
        # Update energy efficiency periodic
        if self.total_rounds % 10 == 0:
            self._recalculate_energy_efficiency()
        
        return {
            "success": success,
            "validator": validator_id,
            "stake": validator.stake,
            "energy_consumed": energy_consumed,
            "remaining_energy": validator.current_energy,
            "performance_score": validator.performance_score,
            "rotations": rotations,
            "energy_saved": self.energy_saved,
        } 