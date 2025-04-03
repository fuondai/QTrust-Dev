"""
Simple test script for DQN agent functionality.

This script tests the DQN agent with scalar action spaces
to ensure correct functionality.
"""

import torch
import numpy as np
import unittest

from qtrust.agents.dqn.agent import DQNAgent


class TestDQNAgent(unittest.TestCase):
    """Test DQN agent basic functionality."""
    
    def test_dqn_agent_scalar_action(self):
        """Test DQNAgent with scalar action space."""
        state_size = 4
        action_size = 2
        
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_size=32,
            buffer_size=1000,
            batch_size=16
        )
        
        # Test action selection
        state = np.random.random(state_size)
        action = agent.act(state)
        
        # DQNAgent.act() may return a numpy int64/int32, not Python int
        self.assertTrue(isinstance(action, (int, np.integer)))
        self.assertTrue(0 <= action < action_size)
        
        # Test step function
        next_state = np.random.random(state_size)
        agent.step(state, action, 1.0, next_state, False)
        
        # Test learning
        if len(agent.memory) >= agent.batch_size:
            agent.learn()


if __name__ == "__main__":
    unittest.main() 