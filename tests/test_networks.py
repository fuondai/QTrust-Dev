"""
Test suite for neural network architectures in DQN agents.

This module tests the functionality of various neural network architectures
used in deep reinforcement learning implementations.
"""

import unittest
import torch
import numpy as np

from qtrust.agents.dqn.networks import (
    NoisyLinear,
    ResidualBlock,
    QNetwork,
    DuelingQNetwork,
    CategoricalQNetwork,
    ActorNetwork,
    CriticNetwork
)

class TestNoisyLinear(unittest.TestCase):
    """Test NoisyLinear layer functionality."""
    
    def setUp(self):
        self.in_features = 10
        self.out_features = 5
        self.layer = NoisyLinear(self.in_features, self.out_features)
        
    def test_initialization(self):
        """Test if layer initializes correctly."""
        self.assertEqual(self.layer.in_features, self.in_features)
        self.assertEqual(self.layer.out_features, self.out_features)
        
    def test_forward(self):
        """Test forward pass."""
        x = torch.randn(32, self.in_features)
        out = self.layer(x)
        self.assertEqual(out.shape, (32, self.out_features))
        
    def test_reset_noise(self):
        """Test noise reset."""
        old_weight = self.layer.weight_epsilon.clone()
        self.layer.reset_noise()
        self.assertFalse(torch.allclose(old_weight, self.layer.weight_epsilon))

class TestQNetwork(unittest.TestCase):
    """Test QNetwork functionality."""
    
    def setUp(self):
        self.state_size = 4
        self.action_size = 2
        self.network = QNetwork(self.state_size, self.action_size)
        
    def test_initialization(self):
        """Test if network initializes correctly."""
        self.assertEqual(self.network.state_size, self.state_size)
        self.assertEqual(self.network.action_dim[0], self.action_size)
        
    def test_forward(self):
        """Test forward pass."""
        state = torch.randn(16, self.state_size)
        out = self.network(state)
        self.assertEqual(out.shape, (16, self.action_size))
        
    def test_noisy_network(self):
        """Test noisy network variant."""
        noisy_net = QNetwork(self.state_size, self.action_size, noisy=True)
        state = torch.randn(16, self.state_size)
        out = noisy_net(state)
        self.assertEqual(out.shape, (16, self.action_size))

class TestDuelingQNetwork(unittest.TestCase):
    """Test DuelingQNetwork functionality."""
    
    def setUp(self):
        self.state_size = 4
        self.action_size = 2
        self.network = DuelingQNetwork(self.state_size, self.action_size)
        
    def test_forward(self):
        """Test forward pass."""
        state = torch.randn(16, self.state_size)
        out = self.network(state)
        self.assertEqual(out.shape, (16, self.action_size))

class TestCategoricalQNetwork(unittest.TestCase):
    """Test CategoricalQNetwork functionality."""
    
    def setUp(self):
        self.state_size = 4
        self.action_dim = [2]
        self.n_atoms = 51
        self.network = CategoricalQNetwork(self.state_size, self.action_dim, self.n_atoms)
        
    def test_forward(self):
        """Test forward pass."""
        state = torch.randn(16, self.state_size)
        distributions, values = self.network(state)
        self.assertEqual(len(distributions), len(self.action_dim))
        self.assertEqual(distributions[0].shape, (16, self.action_dim[0], self.n_atoms))

class TestActorCriticNetworks(unittest.TestCase):
    """Test Actor and Critic networks functionality."""
    
    def setUp(self):
        self.state_size = 4
        self.action_dim = [2, 3]
        self.actor = ActorNetwork(self.state_size, self.action_dim)
        self.critic = CriticNetwork(self.state_size, self.action_dim)
        
    def test_actor_forward(self):
        """Test actor forward pass."""
        state = torch.randn(16, self.state_size)
        action_probs = self.actor(state)
        self.assertEqual(len(action_probs), len(self.action_dim))
        self.assertEqual(action_probs[0].shape, (16, self.action_dim[0]))
        self.assertEqual(action_probs[1].shape, (16, self.action_dim[1]))
        
    def test_critic_forward(self):
        """Test critic forward pass."""
        state = torch.randn(16, self.state_size)
        actions = [torch.randn(16, dim) for dim in self.action_dim]
        q_values = self.critic(state, actions)
        self.assertEqual(len(q_values), len(self.action_dim))
        self.assertEqual(q_values[0].shape, (16, 1))

if __name__ == '__main__':
    unittest.main() 