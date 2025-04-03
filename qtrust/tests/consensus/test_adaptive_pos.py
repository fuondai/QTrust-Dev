"""
Unit tests for the Adaptive Proof of Stake (PoS) components in the QTrust blockchain system.
This file tests the ValidatorStakeInfo and AdaptivePoSManager classes, as well as their integration
with the AdaptiveConsensus system.
"""
import unittest
import random
import time
import numpy as np
from qtrust.consensus.adaptive_pos import AdaptivePoSManager, ValidatorStakeInfo
from qtrust.consensus.adaptive_consensus import AdaptiveConsensus


class TestValidatorStakeInfo(unittest.TestCase):
    """Test the ValidatorStakeInfo class."""
    
    def test_init(self):
        """Test validator initialization."""
        validator = ValidatorStakeInfo(id=1, initial_stake=100.0, max_energy=200.0)
        self.assertEqual(validator.id, 1)
        self.assertEqual(validator.stake, 100.0)
        self.assertEqual(validator.max_energy, 200.0)
        self.assertEqual(validator.current_energy, 200.0)
        self.assertTrue(validator.active)
    
    def test_update_stake(self):
        """Test stake update."""
        validator = ValidatorStakeInfo(id=1, initial_stake=100.0)
        validator.update_stake(50.0)
        self.assertEqual(validator.stake, 150.0)
        validator.update_stake(-200.0)
        self.assertEqual(validator.stake, 0.0)  # Not negative
    
    def test_consume_energy(self):
        """Test energy consumption."""
        validator = ValidatorStakeInfo(id=1, initial_stake=100.0, max_energy=100.0)
        self.assertTrue(validator.consume_energy(50.0))
        self.assertEqual(validator.current_energy, 50.0)
        self.assertTrue(validator.consume_energy(50.0))
        self.assertEqual(validator.current_energy, 0.0)
        self.assertFalse(validator.consume_energy(10.0))
        self.assertEqual(validator.current_energy, 0.0)
    
    def test_recharge_energy(self):
        """Test energy recharging."""
        validator = ValidatorStakeInfo(id=1, max_energy=100.0)
        validator.consume_energy(80.0)
        self.assertEqual(validator.current_energy, 20.0)
        validator.recharge_energy(30.0)
        self.assertEqual(validator.current_energy, 50.0)
        validator.recharge_energy(None)  # Full recharge
        self.assertEqual(validator.current_energy, 100.0)
    
    def test_update_performance(self):
        """Test performance updates."""
        validator = ValidatorStakeInfo(id=1)
        initial_score = validator.performance_score
        validator.update_performance(True)  # Success
        self.assertGreater(validator.performance_score, initial_score)
        self.assertEqual(validator.successful_validations, 1)
        
        high_score = validator.performance_score
        validator.update_performance(False)  # Failure
        self.assertLess(validator.performance_score, high_score)
        self.assertEqual(validator.failed_validations, 1)
    
    def test_average_energy_consumption(self):
        """Test average energy consumption calculation."""
        validator = ValidatorStakeInfo(id=1, max_energy=100.0)
        self.assertEqual(validator.get_average_energy_consumption(), 0.0)
        
        validator.consume_energy(10.0)
        validator.consume_energy(20.0)
        self.assertEqual(validator.get_average_energy_consumption(), 15.0)


class TestAdaptivePoSManager(unittest.TestCase):
    """Test the AdaptivePoSManager class."""
    
    def setUp(self):
        """Prepare test environment."""
        self.pos_manager = AdaptivePoSManager(
            num_validators=20,
            active_validator_ratio=0.7,
            rotation_period=50,
            min_stake=10.0,
            energy_threshold=30.0,
            performance_threshold=0.3,
            seed=42
        )
    
    def test_init(self):
        """Test PoS manager initialization."""
        self.assertEqual(len(self.pos_manager.validators), 20)
        self.assertEqual(self.pos_manager.num_active_validators, 14)  # 70% of 20
        self.assertEqual(len(self.pos_manager.active_validators), 14)
    
    def test_select_validator_for_block(self):
        """Test validator selection for block."""
        selected = self.pos_manager.select_validator_for_block()
        self.assertIsNotNone(selected)
        self.assertIn(selected, self.pos_manager.active_validators)
        
        # Test with trust scores
        trust_scores = {v_id: random.random() for v_id in self.pos_manager.validators}
        selected_with_trust = self.pos_manager.select_validator_for_block(trust_scores)
        self.assertIsNotNone(selected_with_trust)
        self.assertIn(selected_with_trust, self.pos_manager.active_validators)
    
    def test_select_validators_for_committee(self):
        """Test committee validators selection."""
        committee = self.pos_manager.select_validators_for_committee(5)
        self.assertEqual(len(committee), 5)
        for v_id in committee:
            self.assertIn(v_id, self.pos_manager.active_validators)
        
        # Test with large committee size
        large_committee = self.pos_manager.select_validators_for_committee(30)
        self.assertLessEqual(len(large_committee), len(self.pos_manager.active_validators))
    
    def test_update_validator_energy(self):
        """Test validator energy updates."""
        validator_id = next(iter(self.pos_manager.active_validators))
        initial_energy = self.pos_manager.validators[validator_id].current_energy
        initial_stake = self.pos_manager.validators[validator_id].stake
        
        # Update with successful transaction
        self.pos_manager.update_validator_energy(validator_id, 10.0, True)
        self.assertLess(self.pos_manager.validators[validator_id].current_energy, initial_energy)
        self.assertGreater(self.pos_manager.validators[validator_id].stake, initial_stake)
        
        # Update non-existent validator (no error)
        self.pos_manager.update_validator_energy(999, 10.0, True)
    
    def test_rotate_validators(self):
        """Test validator rotation."""
        # Initialize a new PoS with small rotation period for easier testing
        pos = AdaptivePoSManager(
            num_validators=20,
            rotation_period=1,  # Rotate every round
            energy_threshold=30.0
        )
        
        # Make some validators have low energy
        for v_id in list(pos.active_validators)[:3]:
            pos.validators[v_id].consume_energy(80.0)  # 20 energy remaining
        
        # Perform rotation
        rotations = pos.rotate_validators()
        self.assertGreaterEqual(rotations, 1)  # At least one validator rotated
    
    def test_update_energy_recharge(self):
        """Test validator energy recharging."""
        # Make some validators inactive
        active_ids = list(self.pos_manager.active_validators)
        for v_id in active_ids[:3]:
            self.pos_manager.validators[v_id].active = False
            self.pos_manager.active_validators.remove(v_id)
            self.pos_manager.validators[v_id].current_energy = 10.0
        
        # Recharge energy
        self.pos_manager.update_energy_recharge(0.1)  # 10% each time
        
        # Check inactive validators have been recharged
        for v_id in active_ids[:3]:
            self.assertGreater(self.pos_manager.validators[v_id].current_energy, 10.0)
    
    def test_get_energy_statistics(self):
        """Test energy statistics retrieval."""
        stats = self.pos_manager.get_energy_statistics()
        self.assertIn("total_energy", stats)
        self.assertIn("active_energy", stats)
        self.assertIn("inactive_energy", stats)
        self.assertIn("avg_energy", stats)
        self.assertIn("avg_active_energy", stats)
        self.assertIn("energy_saved", stats)
    
    def test_get_validator_statistics(self):
        """Test validator statistics retrieval."""
        stats = self.pos_manager.get_validator_statistics()
        self.assertEqual(stats["total_validators"], 20)
        self.assertEqual(stats["active_validators"], 14)
        self.assertEqual(stats["inactive_validators"], 6)
    
    def test_simulate_round(self):
        """Test PoS round simulation."""
        result = self.pos_manager.simulate_round()
        self.assertIn("success", result)
        self.assertIn("validator", result)
        self.assertIn("stake", result)
        self.assertIn("energy_consumed", result)
        self.assertIn("performance_score", result)


class TestAdaptiveConsensusWithPoS(unittest.TestCase):
    """Test the integration of AdaptiveConsensus with AdaptivePoS."""
    
    def setUp(self):
        """Prepare test environment."""
        self.consensus = AdaptiveConsensus(
            enable_adaptive_pos=True,
            num_validators_per_shard=10,
            active_validator_ratio=0.6,
            rotation_period=5
        )
    
    def test_execute_consensus_with_pos(self):
        """Test consensus execution with AdaptivePoS."""
        # Create mock trust scores
        trust_scores = {i: random.random() for i in range(1, 21)}
        
        # Execute consensus
        result, protocol, latency, energy = self.consensus.execute_consensus(
            transaction_value=20.0,
            congestion=0.3,
            trust_scores=trust_scores,
            shard_id=0  # Shard ID 0
        )
        
        # Check return values
        self.assertIsInstance(result, bool)
        self.assertIsInstance(protocol, str)
        self.assertGreater(latency, 0)
        self.assertGreater(energy, 0)
    
    def test_pos_statistics(self):
        """Test AdaptivePoS statistics."""
        # Execute some consensus
        trust_scores = {i: random.random() for i in range(1, 21)}
        for _ in range(10):
            self.consensus.execute_consensus(
                transaction_value=15.0,
                congestion=0.4,
                trust_scores=trust_scores,
                shard_id=0
            )
        
        # Get statistics
        stats = self.consensus.get_pos_statistics()
        
        # Check statistics structure
        self.assertTrue(stats["enabled"])
        self.assertIn("total_energy_saved", stats)
        self.assertIn("total_rotations", stats)
        self.assertIn("shard_stats", stats)
        self.assertIn(0, stats["shard_stats"])
    
    def test_select_committee_for_shard(self):
        """Test validator committee selection for a shard."""
        committee = self.consensus.select_committee_for_shard(
            shard_id=0,
            committee_size=4
        )
        
        self.assertEqual(len(committee), 4)
        
        # Test with trust scores
        trust_scores = {i: random.random() for i in range(1, 21)}
        committee = self.consensus.select_committee_for_shard(
            shard_id=0,
            committee_size=4,
            trust_scores=trust_scores
        )
        
        self.assertEqual(len(committee), 4)


if __name__ == "__main__":
    unittest.main() 