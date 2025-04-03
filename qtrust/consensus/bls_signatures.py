"""
BLS Signatures Module

This module implements Boneh-Lynn-Shacham (BLS) signature aggregation for blockchain consensus protocols.
BLS signatures allow multiple signatures to be combined into a single compact signature,
significantly reducing data size and verification time in distributed consensus.

Key features:
- BLS signature generation and verification (simulated for testing environments)
- Signature aggregation to reduce communications overhead
- Performance metrics tracking
- BLS-based consensus protocol implementation

Note: This is a simulation version for testing environments only and should not be used in production.
"""

import random
from typing import Dict, List, Tuple, Set, Optional
import hashlib
import time

class BLSSignatureManager:
    """
    BLS signature aggregation manager.
    
    BLS (Boneh-Lynn-Shacham) allows aggregating multiple signatures into a single signature,
    significantly reducing data size and verification time in consensus protocols.
    
    Note: This is a simulation version for testing environments and should not be used in production.
    """
    
    def __init__(self, num_validators: int = 10, threshold: int = 7, seed: int = 42):
        """
        Initialize BLS signature manager.
        
        Args:
            num_validators: Number of validators
            threshold: Minimum number of validators required to create a valid signature
            seed: Seed value for reproducible random calculations
        """
        self.num_validators = num_validators
        self.threshold = threshold
        self.seed = seed
        random.seed(seed)
        
        # Simulate keys and IDs
        self.validator_ids = list(range(1, num_validators + 1))
        self.validator_keys = {vid: hashlib.sha256(f"key_{vid}_{seed}".encode()).hexdigest() for vid in self.validator_ids}
        
        # Store performance information
        self.verification_times = []
        self.signature_sizes = []
        
    def sign_message(self, message: str, validator_id: int) -> str:
        """
        Simulate signing a message by a validator.
        
        Args:
            message: Message to sign
            validator_id: ID of the validator performing the signing
            
        Returns:
            str: Simulated signature
        """
        if validator_id not in self.validator_ids:
            raise ValueError(f"Invalid validator ID {validator_id}")
            
        key = self.validator_keys[validator_id]
        # Simulate signature by combining key, ID and message
        signature = hashlib.sha256(f"{key}_{message}".encode()).hexdigest()
        
        # Simulate signing delay
        time.sleep(0.001)
        
        return signature
    
    def aggregate_signatures(self, message: str, signatures: Dict[int, str]) -> Tuple[str, int, float]:
        """
        Aggregate multiple signatures into a single signature.
        
        Args:
            message: Original message
            signatures: Dict mapping from validator ID to signature
            
        Returns:
            Tuple[str, int, float]: (Aggregated signature, size reduction, aggregation time)
        """
        start_time = time.time()
        
        # Check if there are enough signatures
        if len(signatures) < self.threshold:
            raise ValueError(f"Not enough signatures: {len(signatures)}/{self.threshold}")
        
        # Simulate aggregation process
        combined = "_".join([f"{vid}:{sig}" for vid, sig in sorted(signatures.items())])
        aggregated_signature = hashlib.sha256(combined.encode()).hexdigest()
        
        # Calculate time and size
        aggregate_time = time.time() - start_time
        
        # Actual size of multiple separate signatures (assuming 64 bytes per signature)
        original_size = len(signatures) * 64
        
        # Size of aggregated signature (64 bytes) + validator info (2 bytes/validator)
        aggregated_size = 64 + len(signatures) * 2
        
        # Size reduction
        size_reduction = original_size - aggregated_size
        
        # Store performance information
        self.signature_sizes.append((original_size, aggregated_size))
        
        return aggregated_signature, size_reduction, aggregate_time
    
    def verify_aggregated_signature(self, message: str, aggregated_signature: str, signer_ids: Set[int]) -> Tuple[bool, float]:
        """
        Verify aggregated signature.
        
        Args:
            message: Original message
            aggregated_signature: Aggregated signature
            signer_ids: Set of validator IDs who signed
            
        Returns:
            Tuple[bool, float]: (Verification result, verification time)
        """
        start_time = time.time()
        
        # In simulation, we assume signature is always valid if number of signers >= threshold
        if len(signer_ids) < self.threshold:
            verification_time = time.time() - start_time
            self.verification_times.append(verification_time)
            return False, verification_time
            
        # Simulate verification time (increases logarithmically with number of validators)
        verification_delay = 0.001 * (1 + 0.2 * (len(signer_ids) / self.num_validators))
        time.sleep(verification_delay)
        
        verification_time = time.time() - start_time
        self.verification_times.append(verification_time)
        
        return True, verification_time
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics of BLS signature aggregation.
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        if not self.verification_times:
            return {
                "avg_verification_time": 0.0,
                "avg_size_reduction_percent": 0.0,
                "verification_speedup": 0.0
            }
            
        avg_verification_time = sum(self.verification_times) / len(self.verification_times)
        
        # Calculate average size reduction (%)
        if self.signature_sizes:
            size_reductions = [(original - aggregated) / original * 100 
                               for original, aggregated in self.signature_sizes]
            avg_size_reduction = sum(size_reductions) / len(size_reductions)
        else:
            avg_size_reduction = 0.0
            
        # Estimate verification speedup compared to separate verification
        # Assume separate verification takes O(n) time, while aggregated is O(1)
        verification_speedup = self.num_validators / 2.0  # Assumption
        
        return {
            "avg_verification_time": avg_verification_time,
            "avg_size_reduction_percent": avg_size_reduction,
            "verification_speedup": verification_speedup
        }

class BLSBasedConsensus:
    """
    Consensus protocol using BLS signature aggregation.
    """
    
    def __init__(self, 
                 num_validators: int = 10, 
                 threshold_percent: float = 0.7,
                 latency_factor: float = 0.4,
                 energy_factor: float = 0.5,
                 security_factor: float = 0.9,
                 seed: int = 42):
        """
        Initialize BLS-based consensus protocol.
        
        Args:
            num_validators: Number of validators
            threshold_percent: Percentage of validators required for consensus
            latency_factor: Latency factor
            energy_factor: Energy factor
            security_factor: Security factor
            seed: Seed for random calculations
        """
        self.name = "BLS_Consensus"
        self.latency_factor = latency_factor
        self.energy_factor = energy_factor
        self.security_factor = security_factor
        
        self.num_validators = num_validators
        self.threshold = max(1, int(num_validators * threshold_percent))
        
        # Initialize BLS signature manager
        self.bls_manager = BLSSignatureManager(
            num_validators=num_validators,
            threshold=self.threshold,
            seed=seed
        )
    
    def execute(self, transaction_value: float, trust_scores: Dict[int, float]) -> Tuple[bool, float, float]:
        """
        Execute consensus using BLS signature aggregation.
        
        Args:
            transaction_value: Transaction value
            trust_scores: Trust scores of validators
            
        Returns:
            Tuple[bool, float, float]: (Consensus result, latency, energy consumption)
        """
        message = f"tx_{random.randint(0, 1000000)}_{transaction_value}"
        
        # Select validators based on trust scores
        selected_validators = set()
        for vid in range(1, self.num_validators + 1):
            trust = trust_scores.get(vid, 0.5)
            if random.random() < trust:
                selected_validators.add(vid)
        
        # If not enough trusted validators, consensus fails
        if len(selected_validators) < self.threshold:
            latency = self.latency_factor * (5.0 + 0.1 * transaction_value)
            energy = self.energy_factor * (10.0 + 0.1 * transaction_value)
            return False, latency, energy
        
        # Collect signatures from selected validators
        signatures = {}
        for vid in selected_validators:
            signatures[vid] = self.bls_manager.sign_message(message, vid)
        
        # Aggregate signatures
        try:
            aggregated_signature, size_reduction, aggregate_time = self.bls_manager.aggregate_signatures(message, signatures)
            
            # Verify aggregated signature
            consensus_achieved, verification_time = self.bls_manager.verify_aggregated_signature(
                message, aggregated_signature, selected_validators
            )
            
            # Calculate latency: aggregation time + verification time
            latency = self.latency_factor * (verification_time + aggregate_time) * 1000  # Convert to ms
            
            # Calculate energy consumption (assume energy savings proportional to size reduction)
            base_energy = (15.0 + 0.2 * transaction_value)
            energy_reduction_factor = 1.0 - (size_reduction / (self.num_validators * 64)) * 0.5
            energy = self.energy_factor * base_energy * energy_reduction_factor
            
            return consensus_achieved, latency, energy
            
        except ValueError:
            # If signatures cannot be aggregated
            latency = self.latency_factor * (10.0 + 0.2 * transaction_value)
            energy = self.energy_factor * (20.0 + 0.3 * transaction_value)
            return False, latency, energy
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics of the protocol.
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        return self.bls_manager.get_performance_metrics() 