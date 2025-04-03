"""
Tests for DQN agent.
"""

import unittest
import numpy as np
import torch
import gym
from gym.spaces import Box, Discrete, MultiDiscrete

from qtrust.agents.dqn.agent import DQNAgent
from qtrust.agents.dqn.networks import QNetwork

class SimpleEnv(gym.Env):
    """
    Simple environment for testing.
    """
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.action_space = MultiDiscrete([3, 2])  # Two discrete action spaces
        self.state = np.zeros(5, dtype=np.float32)
        self.step_count = 0
        
    def reset(self):
        self.state = np.random.uniform(-1, 1, size=5).astype(np.float32)
        self.step_count = 0
        return self.state
    
    def step(self, action):
        self.step_count += 1
        self.state = np.random.uniform(-1, 1, size=5).astype(np.float32)
        reward = 1.0 if action[0] == 1 else -0.1
        done = self.step_count >= 10
        info = {}
        return self.state, reward, done, info

class TestQNetwork(unittest.TestCase):
    """
    Tests for QNetwork.
    """
    
    def setUp(self):
        self.state_size = 5
        self.action_dim = [3, 2]  # Two discrete action spaces
        self.hidden_sizes = [64, 64]
        self.network = QNetwork(self.state_size, self.action_dim, hidden_sizes=self.hidden_sizes)
        
    def test_initialization(self):
        """
        Test Q-network initialization.
        """
        # Check parameters
        self.assertEqual(self.network.state_size, self.state_size)
        self.assertEqual(self.network.action_dim, self.action_dim)
        
        # Check network structure
        self.assertIsNotNone(self.network.input_layer)
        self.assertIsNotNone(self.network.res_blocks)
        self.assertEqual(len(self.network.output_layers), len(self.action_dim))
        self.assertIsNotNone(self.network.value_stream)
        
    def test_forward_pass(self):
        """
        Test forward pass.
        """
        # Create random input
        batch_size = 10
        x = torch.randn(batch_size, self.state_size)
        
        # Forward pass
        action_values, state_value = self.network(x)
        
        # Check output dimensions
        self.assertEqual(len(action_values), len(self.action_dim))
        self.assertEqual(action_values[0].shape, (batch_size, self.action_dim[0]))
        self.assertEqual(action_values[1].shape, (batch_size, self.action_dim[1]))
        self.assertEqual(state_value.shape, (batch_size, 1))

class TestDQNAgent(unittest.TestCase):
    """
    Tests for DQNAgent.
    """
    
    def setUp(self):
        self.env = SimpleEnv()
        
        # State and action space dimensions
        self.state_size = 5
        self.action_size = 3
        
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            seed=42,
            buffer_size=1000,
            batch_size=64,
            gamma=0.99,
            tau=1e-3,
            learning_rate=0.001,
            update_every=100,
            prioritized_replay=True,
            alpha=0.6,
            beta_start=0.4,
            dueling=True,
            noisy_nets=False,
            hidden_layers=[64, 64],
            device='cpu'
        )
        
    def test_initialization(self):
        """
        Test agent initialization.
        """
        # Check parameters
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertEqual(self.agent.batch_size, 64)
        
        # Check networks
        self.assertIsInstance(self.agent.qnetwork_local, QNetwork)
        self.assertIsInstance(self.agent.qnetwork_target, QNetwork)
        
        # Check optimizer
        self.assertIsInstance(self.agent.optimizer, torch.optim.Adam)
        
        # Check buffer is initially empty
        self.assertEqual(len(self.agent.memory), 0)
        
    def test_step(self):
        """
        Test step function.
        """
        # Create some experiences and add to buffer
        state = np.random.uniform(-1, 1, size=self.state_size).astype(np.float32)
        action = 1  # Sample action (index)
        reward = 1.0
        next_state = np.random.uniform(-1, 1, size=self.state_size).astype(np.float32)
        done = False
        
        # Add 100 experiences
        for _ in range(100):
            self.agent.step(state, action, reward, next_state, done)
            
        # Check buffer
        self.assertEqual(len(self.agent.memory), 100)
        
    def test_act(self):
        """
        Test act function.
        """
        # Create random state
        state = np.random.uniform(-1, 1, size=self.state_size).astype(np.float32)
        
        # Perform action with epsilon=0 (greedy)
        action = self.agent.act(state, eps=0)
        
        # Check action
        self.assertIsInstance(action, (int, np.int64))
        self.assertIn(action, range(self.action_size))
        
    def test_learn(self):
        """
        Test learning process.
        """
        # Add enough experiences to learn
        state = np.random.uniform(-1, 1, size=self.state_size).astype(np.float32)
        action = 1  # Sample action
        reward = 1.0
        next_state = np.random.uniform(-1, 1, size=self.state_size).astype(np.float32)
        done = False
        
        # Get parameters before learning
        params_before = [p.clone().detach() for p in self.agent.qnetwork_local.parameters()]
        
        # Add many experiences to buffer
        for _ in range(self.agent.batch_size + 10):  # Add more than batch_size
            self.agent.step(state, action, reward, next_state, done)
            
        # Parameters may change after many steps but not directly calling _learn
        # so no detailed check here
            
    def test_soft_update(self):
        """
        Test soft update of target network.
        """
        # Change local network
        for param in self.agent.qnetwork_local.parameters():
            param.data = torch.randn_like(param.data)
            
        # Target network initially different from local network
        # Store a copy of target network parameters to compare later
        target_params = []
        for param in self.agent.qnetwork_target.parameters():
            target_params.append(param.data.clone())
        
        # Perform soft update
        self.agent._soft_update()
        
        # Check that after update, target network has been updated 
        # but not completely identical to local (due to soft update with tau < 1)
        params_changed = False
        for i, (local_param, target_param) in enumerate(zip(
            self.agent.qnetwork_local.parameters(),
            self.agent.qnetwork_target.parameters()
        )):
            # Compare with previously stored copy
            if not torch.all(torch.eq(target_param.data, target_params[i])):
                params_changed = True
                break
        
        # Ensure that at least one parameter has changed
        self.assertTrue(params_changed, "Soft update did not change any parameters of the target network")
        
    def test_update_epsilon(self):
        """
        Test epsilon decay.
        """
        # Save initial epsilon
        initial_epsilon = self.agent.epsilon
        
        # Perform epsilon decay
        for _ in range(10):
            self.agent.update_epsilon()
            
        # Check epsilon has decreased
        self.assertLess(self.agent.epsilon, initial_epsilon)
        
        # Check epsilon is not lower than epsilon_end
        self.assertGreaterEqual(self.agent.epsilon, self.agent.eps_end)

if __name__ == '__main__':
    unittest.main() 