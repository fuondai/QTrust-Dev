import unittest
import torch
import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qtrust.trust.models import QNetwork, ActorCriticNetwork, TrustNetwork

class TestQNetwork(unittest.TestCase):
    """Test cases for QNetwork class"""
    
    def setUp(self):
        """Set up test environment before each test method"""
        self.state_size = 8
        self.action_dims = [4]
        self.batch_size = 5
        self.seed = 42
        self.hidden_layers = [64, 32]
        
        # Create standard QNetwork
        self.qnet = QNetwork(
            state_size=self.state_size,
            action_dims=self.action_dims,
            seed=self.seed,
            hidden_layers=self.hidden_layers,
            dueling=False
        )
        
        # Create dueling QNetwork
        self.dueling_qnet = QNetwork(
            state_size=self.state_size,
            action_dims=self.action_dims,
            seed=self.seed,
            hidden_layers=self.hidden_layers,
            dueling=True
        )
        
        # Sample state tensor
        self.state = torch.rand((self.batch_size, self.state_size))

    def test_initialization(self):
        """Test proper initialization of QNetwork"""
        # Standard QNetwork
        self.assertEqual(self.qnet.state_size, self.state_size)
        self.assertEqual(self.qnet.action_dims, self.action_dims)
        self.assertEqual(self.qnet.dueling, False)
        self.assertEqual(self.qnet.total_actions, self.action_dims[0])
        
        # Dueling QNetwork
        self.assertEqual(self.dueling_qnet.state_size, self.state_size)
        self.assertEqual(self.dueling_qnet.action_dims, self.action_dims)
        self.assertEqual(self.dueling_qnet.dueling, True)
        
        # Check for proper layer creation
        self.assertIsInstance(self.qnet.feature_layer, torch.nn.Sequential)
        self.assertIsInstance(self.qnet.output_layers, torch.nn.ModuleList)
        self.assertIsInstance(self.dueling_qnet.feature_layer, torch.nn.Sequential)
        self.assertIsInstance(self.dueling_qnet.value_stream, torch.nn.Sequential)
        self.assertIsInstance(self.dueling_qnet.advantage_streams, torch.nn.ModuleList)

    def test_forward(self):
        """Test forward pass of QNetwork"""
        # Standard QNetwork
        q_values, state_value = self.qnet.forward(self.state)
        
        # Check output types
        self.assertIsInstance(q_values, list)
        self.assertIsNone(state_value)
        
        # Check output shapes
        self.assertEqual(len(q_values), len(self.action_dims))
        self.assertEqual(q_values[0].shape, (self.batch_size, self.action_dims[0]))
        
        # Dueling QNetwork
        dueling_q_values, dueling_state_value = self.dueling_qnet.forward(self.state)
        
        # Check output types
        self.assertIsInstance(dueling_q_values, list)
        self.assertIsInstance(dueling_state_value, torch.Tensor)
        
        # Check output shapes
        self.assertEqual(len(dueling_q_values), len(self.action_dims))
        self.assertEqual(dueling_q_values[0].shape, (self.batch_size, self.action_dims[0]))
        self.assertEqual(dueling_state_value.shape, (self.batch_size, 1))
    
    def test_multihead_output(self):
        """Test QNetwork with multiple action dimensions"""
        multi_action_dims = [3, 4, 2]
        
        # Create QNetwork with multiple action dimensions
        multi_qnet = QNetwork(
            state_size=self.state_size,
            action_dims=multi_action_dims,
            seed=self.seed,
            hidden_layers=self.hidden_layers,
            dueling=False
        )
        
        # Test forward pass
        q_values, _ = multi_qnet.forward(self.state)
        
        # Check output
        self.assertEqual(len(q_values), len(multi_action_dims))
        for i, dim in enumerate(multi_action_dims):
            self.assertEqual(q_values[i].shape, (self.batch_size, dim))


class TestActorCriticNetwork(unittest.TestCase):
    """Test cases for ActorCriticNetwork class"""
    
    def setUp(self):
        """Set up test environment before each test method"""
        self.state_size = 10
        self.action_dims = [4, 3]
        self.batch_size = 6
        self.seed = 42
        self.hidden_layers = [128, 64]
        
        # Create ActorCriticNetwork
        self.ac_net = ActorCriticNetwork(
            state_size=self.state_size,
            action_dims=self.action_dims,
            hidden_layers=self.hidden_layers,
            seed=self.seed
        )
        
        # Sample state tensor
        self.state = torch.rand((self.batch_size, self.state_size))

    def test_initialization(self):
        """Test proper initialization of ActorCriticNetwork"""
        self.assertEqual(self.ac_net.state_size, self.state_size)
        self.assertEqual(self.ac_net.action_dims, self.action_dims)
        
        # Check for proper layer creation
        self.assertIsInstance(self.ac_net.shared_layers, torch.nn.Sequential)
        self.assertIsInstance(self.ac_net.actor_hidden, torch.nn.Sequential)
        self.assertIsInstance(self.ac_net.actor_heads, torch.nn.ModuleList)
        self.assertIsInstance(self.ac_net.critic, torch.nn.Sequential)
        
        # Check number of actor heads
        self.assertEqual(len(self.ac_net.actor_heads), len(self.action_dims))

    def test_forward(self):
        """Test forward pass of ActorCriticNetwork"""
        action_probs, state_value = self.ac_net.forward(self.state)
        
        # Check output types
        self.assertIsInstance(action_probs, list)
        self.assertIsInstance(state_value, torch.Tensor)
        
        # Check output shapes
        self.assertEqual(len(action_probs), len(self.action_dims))
        for i, dim in enumerate(self.action_dims):
            self.assertEqual(action_probs[i].shape, (self.batch_size, dim))
        self.assertEqual(state_value.shape, (self.batch_size, 1))
        
        # Check that action probabilities sum to 1
        for probs in action_probs:
            sums = torch.sum(probs, dim=1)
            for sum_val in sums:
                self.assertAlmostEqual(sum_val.item(), 1.0, delta=1e-6)


class TestTrustNetwork(unittest.TestCase):
    """Test cases for TrustNetwork class"""
    
    def setUp(self):
        """Set up test environment before each test method"""
        self.input_size = 12
        self.batch_size = 8
        self.seed = 42
        self.hidden_layers = [32, 16]
        
        # Create TrustNetwork
        self.trust_net = TrustNetwork(
            input_size=self.input_size,
            hidden_layers=self.hidden_layers,
            seed=self.seed
        )
        
        # Sample input features
        self.features = torch.rand((self.batch_size, self.input_size))

    def test_initialization(self):
        """Test proper initialization of TrustNetwork"""
        # Check model structure
        self.assertIsInstance(self.trust_net.model, torch.nn.Sequential)
        
        # Make sure last layer has Sigmoid activation
        last_layer = list(self.trust_net.model.children())[-1]
        self.assertIsInstance(last_layer, torch.nn.Sigmoid)

    def test_forward(self):
        """Test forward pass of TrustNetwork"""
        trust_scores = self.trust_net.forward(self.features)
        
        # Check output type and shape
        self.assertIsInstance(trust_scores, torch.Tensor)
        self.assertEqual(trust_scores.shape, (self.batch_size, 1))
        
        # Check that all values are in [0, 1] range (trust scores)
        self.assertTrue(torch.all(trust_scores >= 0).item())
        self.assertTrue(torch.all(trust_scores <= 1).item())

    def test_calculate_trust(self):
        """Test calculate_trust method of TrustNetwork"""
        # Set model to eval mode
        self.trust_net.train()  # Make sure it's in train mode first
        
        # Call calculate_trust
        trust_scores = self.trust_net.calculate_trust(self.features)
        
        # Check that model was set to eval mode
        self.assertFalse(self.trust_net.training)
        
        # Check output type and shape
        self.assertIsInstance(trust_scores, torch.Tensor)
        self.assertEqual(trust_scores.shape, (self.batch_size, 1))
        
        # Check that all values are in [0, 1] range
        self.assertTrue(torch.all(trust_scores >= 0).item())
        self.assertTrue(torch.all(trust_scores <= 1).item())


if __name__ == '__main__':
    unittest.main() 