"""
Tests for BLS Signatures Module

This module tests the functionality of the BLS signature aggregation
implementation in the qtrust/consensus/bls_signatures.py module.
"""

import unittest
import random
from typing import Dict, Set
import hashlib
import time
from qtrust.consensus.bls_signatures import BLSSignatureManager, BLSBasedConsensus

class TestBLSSignatureManager(unittest.TestCase):
    """Tests for BLSSignatureManager class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.num_validators = 10
        self.threshold = 7
        self.seed = 42
        self.bls_manager = BLSSignatureManager(
            num_validators=self.num_validators,
            threshold=self.threshold,
            seed=self.seed
        )
        
        # Create a test message
        self.test_message = "test_transaction_123"
        
    def test_initialization(self):
        """Test proper initialization of BLSSignatureManager."""
        # Check basic attributes
        self.assertEqual(self.bls_manager.num_validators, self.num_validators)
        self.assertEqual(self.bls_manager.threshold, self.threshold)
        self.assertEqual(self.bls_manager.seed, self.seed)
        
        # Check validator IDs and keys
        self.assertEqual(len(self.bls_manager.validator_ids), self.num_validators)
        self.assertEqual(len(self.bls_manager.validator_keys), self.num_validators)
        for i in range(1, self.num_validators + 1):
            self.assertIn(i, self.bls_manager.validator_ids)
            self.assertIn(i, self.bls_manager.validator_keys)
    
    def test_sign_message(self):
        """Test signing a message by a validator."""
        # Sign with a valid validator ID
        signature = self.bls_manager.sign_message(self.test_message, 1)
        self.assertIsInstance(signature, str)
        self.assertEqual(len(signature), 64)  # SHA-256 hash is 64 hex characters
        
        # Ensure consistency for the same inputs
        signature2 = self.bls_manager.sign_message(self.test_message, 1)
        self.assertEqual(signature, signature2)
        
        # Verify different validators produce different signatures
        signature3 = self.bls_manager.sign_message(self.test_message, 2)
        self.assertNotEqual(signature, signature3)
        
        # Verify invalid validator ID raises ValueError
        with self.assertRaises(ValueError):
            self.bls_manager.sign_message(self.test_message, self.num_validators + 1)
    
    def test_aggregate_signatures(self):
        """Test aggregating signatures."""
        # Create signatures from multiple validators
        signatures = {}
        for i in range(1, self.threshold + 1):
            signatures[i] = self.bls_manager.sign_message(self.test_message, i)
        
        # Aggregate signatures
        aggregated_signature, size_reduction, aggregate_time = self.bls_manager.aggregate_signatures(
            self.test_message, signatures
        )
        
        # Check results
        self.assertIsInstance(aggregated_signature, str)
        self.assertGreater(size_reduction, 0)
        self.assertIsInstance(aggregate_time, float)  # Just check it's a float, not its value
        
        # Check size reduction is as expected
        expected_reduction = len(signatures) * 64 - (64 + len(signatures) * 2)
        self.assertEqual(size_reduction, expected_reduction)
        
        # Test with too few signatures
        with self.assertRaises(ValueError):
            self.bls_manager.aggregate_signatures(
                self.test_message, {1: signatures[1]}
            )
    
    def test_verify_aggregated_signature(self):
        """Test verifying an aggregated signature."""
        # Create and aggregate signatures
        signatures = {}
        signer_ids = set()
        for i in range(1, self.threshold + 1):
            signatures[i] = self.bls_manager.sign_message(self.test_message, i)
            signer_ids.add(i)
        
        aggregated_signature, _, _ = self.bls_manager.aggregate_signatures(
            self.test_message, signatures
        )
        
        # Verify aggregated signature
        result, verification_time = self.bls_manager.verify_aggregated_signature(
            self.test_message, aggregated_signature, signer_ids
        )
        
        # Check results
        self.assertTrue(result)
        self.assertIsInstance(verification_time, float)  # Just check it's a float, not its value
        
        # Test with too few signers
        insufficient_signers = set([1, 2])
        result2, _ = self.bls_manager.verify_aggregated_signature(
            self.test_message, aggregated_signature, insufficient_signers
        )
        self.assertFalse(result2)
    
    def test_performance_metrics(self):
        """Test getting performance metrics."""
        # Perform some operations to generate metrics
        signatures = {}
        signer_ids = set()
        for i in range(1, self.threshold + 1):
            signatures[i] = self.bls_manager.sign_message(self.test_message, i)
            signer_ids.add(i)
        
        aggregated_signature, _, _ = self.bls_manager.aggregate_signatures(
            self.test_message, signatures
        )
        
        self.bls_manager.verify_aggregated_signature(
            self.test_message, aggregated_signature, signer_ids
        )
        
        # Get metrics
        metrics = self.bls_manager.get_performance_metrics()
        
        # Check metrics structure
        self.assertIn("avg_verification_time", metrics)
        self.assertIn("avg_size_reduction_percent", metrics)
        self.assertIn("verification_speedup", metrics)
        
        # Check values are of correct type
        self.assertIsInstance(metrics["avg_verification_time"], float)
        self.assertIsInstance(metrics["avg_size_reduction_percent"], float)
        self.assertIsInstance(metrics["verification_speedup"], float)

class TestBLSBasedConsensus(unittest.TestCase):
    """Tests for BLSBasedConsensus class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.num_validators = 10
        self.threshold_percent = 0.7
        self.latency_factor = 0.4
        self.energy_factor = 0.5
        self.security_factor = 0.9
        self.seed = 42
        
        self.consensus = BLSBasedConsensus(
            num_validators=self.num_validators,
            threshold_percent=self.threshold_percent,
            latency_factor=self.latency_factor,
            energy_factor=self.energy_factor,
            security_factor=self.security_factor,
            seed=self.seed
        )
        
        # Set up trust scores for testing
        self.high_trust_scores = {i: 0.9 for i in range(1, self.num_validators + 1)}
        self.low_trust_scores = {i: 0.3 for i in range(1, self.num_validators + 1)}
    
    def test_initialization(self):
        """Test proper initialization of BLSBasedConsensus."""
        # Check attributes
        self.assertEqual(self.consensus.name, "BLS_Consensus")
        self.assertEqual(self.consensus.latency_factor, self.latency_factor)
        self.assertEqual(self.consensus.energy_factor, self.energy_factor)
        self.assertEqual(self.consensus.security_factor, self.security_factor)
        self.assertEqual(self.consensus.num_validators, self.num_validators)
        self.assertEqual(self.consensus.threshold, int(self.num_validators * self.threshold_percent))
        
        # Check BLS manager was initialized
        self.assertIsInstance(self.consensus.bls_manager, BLSSignatureManager)
    
    def test_execute_high_trust(self):
        """Test executing consensus with high trust scores."""
        # Execute consensus with high trust
        result, latency, energy = self.consensus.execute(
            transaction_value=10.0,
            trust_scores=self.high_trust_scores
        )
        
        # Check results
        self.assertIsInstance(result, bool)
        self.assertIsInstance(latency, float)
        self.assertIsInstance(energy, float)
        self.assertGreater(latency, 0)
        self.assertGreater(energy, 0)
        
        # With high trust, consensus should typically succeed
        # (though it's probabilistic, so we don't assert this)
    
    def test_execute_low_trust(self):
        """Test executing consensus with low trust scores."""
        # Execute consensus with low trust
        result, latency, energy = self.consensus.execute(
            transaction_value=10.0,
            trust_scores=self.low_trust_scores
        )
        
        # Check results
        self.assertIsInstance(latency, float)
        self.assertIsInstance(energy, float)
        self.assertGreater(latency, 0)
        self.assertGreater(energy, 0)
        
        # With low trust, consensus may fail, but we just check that it runs
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        # Execute consensus to generate metrics
        self.consensus.execute(
            transaction_value=10.0,
            trust_scores=self.high_trust_scores
        )
        
        # Get metrics
        metrics = self.consensus.get_performance_metrics()
        
        # Check metrics structure
        self.assertIn("avg_verification_time", metrics)
        self.assertIn("avg_size_reduction_percent", metrics)
        self.assertIn("verification_speedup", metrics)
        
        # Check values are of correct type
        self.assertIsInstance(metrics["avg_verification_time"], float)
        self.assertIsInstance(metrics["avg_size_reduction_percent"], float)
        self.assertIsInstance(metrics["verification_speedup"], float)

if __name__ == '__main__':
    unittest.main() 