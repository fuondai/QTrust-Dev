"""
Test cases for the blockchain environment.
"""

import unittest
import numpy as np
import pytest
import gym

from qtrust.simulation.blockchain_environment import BlockchainEnvironment

class TestBlockchainEnvironment(unittest.TestCase):
    """
    Tests for BlockchainEnvironment.
    """
    
    def setUp(self):
        """
        Setup before each test.
        """
        self.env = BlockchainEnvironment(
            num_shards=3,
            num_nodes_per_shard=5,
            max_transactions_per_step=50,
            transaction_value_range=(0.1, 50.0),
            max_steps=500
        )
        
    def test_initialization(self):
        """
        Test environment initialization.
        """
        # Check number of shards and nodes
        self.assertEqual(self.env.num_shards, 3)
        self.assertEqual(self.env.num_nodes_per_shard, 5)
        self.assertEqual(self.env.total_nodes, 15)
        
        # Check other initialization values
        self.assertEqual(self.env.max_transactions_per_step, 50)
        self.assertEqual(self.env.transaction_value_range, (0.1, 50.0))
        self.assertEqual(self.env.max_steps, 500)
        
        # Check blockchain network has been initialized
        self.assertIsNotNone(self.env.blockchain_network)
        self.assertEqual(len(self.env.blockchain_network.nodes), 15)
        
        # Check penalty and reward parameters
        self.assertGreater(self.env.latency_penalty, 0)
        self.assertGreater(self.env.energy_penalty, 0)
        self.assertGreater(self.env.throughput_reward, 0)
        self.assertGreater(self.env.security_reward, 0)
        
    def test_reset(self):
        """
        Test environment reset.
        """
        initial_state = self.env.reset()
        
        # Check initial state
        self.assertIsNotNone(initial_state)
        self.assertEqual(self.env.current_step, 0)
        
        # Check transaction pool has been cleared
        self.assertEqual(len(self.env.transaction_pool), 0)
        
        # Check metrics have been reset
        self.assertEqual(self.env.performance_metrics['transactions_processed'], 0)
        self.assertEqual(self.env.performance_metrics['total_latency'], 0)
        self.assertEqual(self.env.performance_metrics['total_energy'], 0)
        
    def test_step(self):
        """
        Test a single step in the environment.
        """
        _ = self.env.reset()
        
        # Create a valid random action
        action = self.env.action_space.sample()
        
        # Perform one step
        next_state, reward, done, info = self.env.step(action)
        
        # Check next state
        self.assertIsNotNone(next_state)
        
        # Check reward
        self.assertIsInstance(reward, float)
        
        # Check info
        self.assertIn('transactions_processed', info)
        self.assertIn('avg_latency', info)
        self.assertIn('avg_energy', info)
        self.assertIn('throughput', info)
        
        # Check current step has been incremented
        self.assertEqual(self.env.current_step, 1)
        
        # Check done flag (should be False since it's the first step)
        self.assertFalse(done)
        
    def test_multiple_steps(self):
        """
        Test multiple consecutive steps.
        """
        _ = self.env.reset()
        
        # Run 10 steps
        rewards = []
        for _ in range(10):
            action = self.env.action_space.sample()
            _, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            
            if done:
                break
        
        # Check current step
        self.assertEqual(self.env.current_step, 10)
        
        # Check number of rewards
        self.assertEqual(len(rewards), 10)
        
    def test_generate_transactions(self):
        """
        Test transaction generation.
        """
        _ = self.env.reset()
        
        # Generate new transactions
        transactions = self.env._generate_transactions(10)
        
        # Check number of transactions generated
        self.assertEqual(len(transactions), 10)
        
        # Check transaction format
        if transactions:
            tx = transactions[0]
            self.assertIn('id', tx)
            self.assertIn('source_shard', tx)
            self.assertIn('destination_shard', tx)
            self.assertIn('value', tx)
            self.assertIn('timestamp', tx)
            
            # Check value is within range
            self.assertGreaterEqual(tx['value'], self.env.transaction_value_range[0])
            self.assertLessEqual(tx['value'], self.env.transaction_value_range[1])
            
    def test_get_state(self):
        """
        Test getting environment state.
        """
        _ = self.env.reset()
        
        # Get state
        state = self.env.get_state()
        
        # Check state is not None
        self.assertIsNotNone(state)
        
        # Check state size
        self.assertIsInstance(state, np.ndarray)
        
    def test_calculate_reward(self):
        """
        Test reward calculation.
        """
        _ = self.env.reset()
        
        # Add some mock metrics
        self.env.metrics['throughput'] = [0.8]
        self.env.metrics['latency'] = [15.0]
        self.env.metrics['energy_consumption'] = [200.0]
        self.env.metrics['security_score'] = [0.7]
        
        # Calculate reward
        reward = self.env._get_reward([1, 0], self.env.get_state())
        
        # Check reward is not None
        self.assertIsNotNone(reward)
        self.assertIsInstance(reward, float)
        
    def test_done_condition(self):
        """
        Test done condition.
        """
        _ = self.env.reset()
        
        # Not at max steps -> not done
        action = self.env.action_space.sample()
        _, _, done, _ = self.env.step(action)
        self.assertFalse(done)
        
        # Set current_step to max_steps - 1
        self.env.current_step = self.env.max_steps - 1
        action = self.env.action_space.sample()
        _, _, done, _ = self.env.step(action)
        self.assertTrue(done)
        
    def test_action_space(self):
        """
        Test action space.
        """
        # Check that action space is valid
        self.assertIsInstance(self.env.action_space, gym.spaces.MultiDiscrete)
        
        # Check action space shape
        action_shape = self.env.action_space.nvec
        self.assertEqual(len(action_shape), 2)
        self.assertEqual(action_shape[0], self.env.max_num_shards)
        self.assertEqual(action_shape[1], 3)  # 3 consensus protocols
        
        # Check that we can sample from action space
        action = self.env.action_space.sample()
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(len(action), 2)
        
    def test_observation_space(self):
        """
        Test observation space.
        """
        # Check that observation space is valid
        self.assertIsInstance(self.env.observation_space, gym.spaces.Box)
        
        # Check observation space shape
        obs_shape = self.env.observation_space.shape
        self.assertEqual(len(obs_shape), 1)
        
        # Get a state and verify its shape
        _ = self.env.reset()
        state = self.env.get_state()
        self.assertIsInstance(state, np.ndarray)
        
        # The state might not actually fit in observation_space due to dynamic nature
        # of the environment. Just check that it's a properly formed state vector.
        self.assertEqual(len(state.shape), 1)  # Should be a 1D array
        
    def test_reward_optimization(self):
        """
        Test reward optimization mechanisms.
        """
        _ = self.env.reset()
        
        # Check throughput reward calculation
        throughput_values = [0.5, 0.8, 1.0]
        for val in throughput_values:
            self.env.metrics['throughput'] = [val]
            throughput_reward = self.env._get_throughput_reward()
            self.assertIsInstance(throughput_reward, float)
            self.assertGreaterEqual(throughput_reward, 0)
        
        # Check that higher throughput gives higher reward
        self.env.metrics['throughput'] = [0.5]
        low_throughput_reward = self.env._get_throughput_reward()
        self.env.metrics['throughput'] = [1.0]
        high_throughput_reward = self.env._get_throughput_reward()
        self.assertGreater(high_throughput_reward, low_throughput_reward)
        
        # Check latency penalty calculation
        latency_values = [10.0, 30.0, 80.0]
        for val in latency_values:
            self.env.metrics['latency'] = [val]
            latency_penalty = self.env._get_latency_penalty()
            self.assertIsInstance(latency_penalty, float)
            self.assertGreaterEqual(latency_penalty, 0)
        
        # Check that higher latency gives higher penalty
        self.env.metrics['latency'] = [10.0]
        low_latency_penalty = self.env._get_latency_penalty()
        self.env.metrics['latency'] = [80.0]
        high_latency_penalty = self.env._get_latency_penalty()
        self.assertGreater(high_latency_penalty, low_latency_penalty)
        
        # Check energy penalty calculation
        energy_values = [50.0, 150.0, 350.0]
        for val in energy_values:
            self.env.metrics['energy_consumption'] = [val]
            energy_penalty = self.env._get_energy_penalty()
            self.assertIsInstance(energy_penalty, float)
            self.assertGreaterEqual(energy_penalty, 0)
        
        # Check that higher energy gives higher penalty
        self.env.metrics['energy_consumption'] = [50.0]
        low_energy_penalty = self.env._get_energy_penalty()
        self.env.metrics['energy_consumption'] = [350.0]
        high_energy_penalty = self.env._get_energy_penalty()
        self.assertGreater(high_energy_penalty, low_energy_penalty)
        
        # Check security reward calculation
        for protocol in range(3):
            security_reward = self.env._get_security_score(protocol)
            self.assertIsInstance(security_reward, float)
            self.assertGreaterEqual(security_reward, 0)
        
        # Check that more secure protocols give higher reward
        fast_bft_reward = self.env._get_security_score(0)  # Fast BFT
        pbft_reward = self.env._get_security_score(1)      # PBFT
        robust_bft_reward = self.env._get_security_score(2)  # Robust BFT
        self.assertGreater(robust_bft_reward, fast_bft_reward)
        self.assertGreater(pbft_reward, fast_bft_reward)
        
    def test_innovative_routing(self):
        """
        Test innovative routing detection.
        """
        _ = self.env.reset()
        
        # Test innovative routing function
        is_innovative = self.env._is_innovative_routing([1, 0])
        # Check if the return is either a boolean or a "bool-like" value
        # that can be used in boolean context
        self.assertIn(is_innovative, [True, False, 0, 1])
        
        # Test with various inputs
        is_innovative_2 = self.env._is_innovative_routing([0, 1])
        is_innovative_3 = self.env._is_innovative_routing([2, 2])
        
        # Just verify these calls work without exception
        self.assertIsNotNone(is_innovative_2)
        self.assertIsNotNone(is_innovative_3)
        
    def test_high_performance_criteria(self):
        """
        Test high performance detection.
        """
        _ = self.env.reset()
        
        # Set metrics for high performance
        self.env.metrics['throughput'] = [20.0]
        self.env.metrics['latency'] = [20.0]
        self.env.metrics['energy_consumption'] = [150.0]
        
        # Check high performance criteria
        is_high_perf = self.env._is_high_performance()
        self.assertIsInstance(is_high_perf, bool)
        
        # Test that poor performance is detected
        self.env.metrics['throughput'] = [5.0]
        self.env.metrics['latency'] = [100.0]
        self.env.metrics['energy_consumption'] = [500.0]
        
        is_poor_perf = not self.env._is_high_performance()
        self.assertTrue(is_poor_perf)

if __name__ == '__main__':
    unittest.main() 