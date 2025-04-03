import unittest
import numpy as np
import torch
from qtrust.agents.dqn.replay_buffer import (
    ReplayBuffer, PrioritizedReplayBuffer, 
    EfficientReplayBuffer, NStepReplayBuffer,
    NStepPrioritizedReplayBuffer
)

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 100
        self.batch_size = 32
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # Sample data
        self.state = np.array([1.0, 2.0, 3.0, 4.0])
        self.action = 1
        self.reward = 1.0
        self.next_state = np.array([2.0, 3.0, 4.0, 5.0])
        self.done = False
        
    def test_push(self):
        self.buffer.push(self.state, self.action, self.reward, self.next_state, self.done)
        self.assertEqual(len(self.buffer), 1)
        
    def test_sample(self):
        # Add multiple experiences
        for _ in range(self.batch_size):
            self.buffer.push(self.state, self.action, self.reward, self.next_state, self.done)
            
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        self.assertEqual(states.shape, (self.batch_size, 4))
        self.assertEqual(actions.shape, (self.batch_size,))
        self.assertEqual(rewards.shape, (self.batch_size,))
        self.assertEqual(next_states.shape, (self.batch_size, 4))
        self.assertEqual(dones.shape, (self.batch_size,))

class TestPrioritizedReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 100
        self.batch_size = 32
        self.buffer = PrioritizedReplayBuffer(self.buffer_size, self.batch_size)
        
        # Sample data
        self.state = np.array([1.0, 2.0, 3.0, 4.0])
        self.action = 1
        self.reward = 1.0
        self.next_state = np.array([2.0, 3.0, 4.0, 5.0])
        self.done = False
        
    def test_push_with_priority(self):
        self.buffer.push(self.state, self.action, self.reward, self.next_state, self.done, error=1.0)
        self.assertEqual(len(self.buffer), 1)
        
    def test_sample_with_priorities(self):
        # Add experiences with different priorities
        for i in range(self.batch_size):
            self.buffer.push(self.state, self.action, self.reward, self.next_state, self.done, error=float(i))
            
        result = self.buffer.sample()
        self.assertIsNotNone(result)
        if result:
            states, actions, rewards, next_states, dones, indices, weights = result
            self.assertEqual(states.shape, (self.batch_size, 4))
            self.assertEqual(weights.shape, (self.batch_size,))

class TestEfficientReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 100
        self.batch_size = 32
        self.state_shape = (4,)
        self.action_shape = 1
        self.buffer = EfficientReplayBuffer(
            self.buffer_size, self.batch_size, 
            self.state_shape, self.action_shape
        )
        
        # Sample data
        self.state = np.array([1.0, 2.0, 3.0, 4.0])
        self.action = 1
        self.reward = 1.0
        self.next_state = np.array([2.0, 3.0, 4.0, 5.0])
        self.done = False
        
    def test_efficient_storage(self):
        self.buffer.push(self.state, self.action, self.reward, self.next_state, self.done)
        self.assertEqual(self.buffer.count, 1)
        
    def test_efficient_sample(self):
        # Add multiple experiences
        for _ in range(self.batch_size):
            self.buffer.push(self.state, self.action, self.reward, self.next_state, self.done)
            
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        self.assertEqual(states.shape, (self.batch_size, 4))
        self.assertEqual(actions.shape, (self.batch_size,))

class TestNStepReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 100
        self.batch_size = 32
        self.n_step = 3
        self.gamma = 0.99
        self.buffer = NStepReplayBuffer(
            self.buffer_size, self.batch_size,
            n_step=self.n_step, gamma=self.gamma
        )
        
        # Sample data
        self.state = np.array([1.0, 2.0, 3.0, 4.0])
        self.action = 1
        self.reward = 1.0
        self.next_state = np.array([2.0, 3.0, 4.0, 5.0])
        self.done = False
        
    def test_n_step_storage(self):
        # Add n_step experiences
        for _ in range(self.n_step):
            self.buffer.push(self.state, self.action, self.reward, self.next_state, self.done)
        self.assertEqual(len(self.buffer), 1)
        
    def test_n_step_sample(self):
        # Add multiple n-step experiences
        for _ in range(self.batch_size * self.n_step):
            self.buffer.push(self.state, self.action, self.reward, self.next_state, self.done)
            
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        self.assertEqual(states.shape, (self.batch_size, 4))
        self.assertEqual(actions.shape, (self.batch_size,))

class TestNStepPrioritizedReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 100
        self.batch_size = 8
        self.n_step = 3
        self.gamma = 0.99
        self.alpha = 0.6
        self.beta_start = 0.4
        self.device = 'cpu'
        self.buffer = NStepPrioritizedReplayBuffer(
            self.buffer_size, 
            self.batch_size,
            self.n_step,
            self.gamma,
            self.alpha,
            self.beta_start,
            device=self.device
        )
        
        # Sample data for testing
        self.state_shape = (4,)
        self.state = np.zeros(self.state_shape)
        self.action = 0
        self.reward = 1.0
        self.next_state = np.ones(self.state_shape)
        self.done = False

    def test_n_step_prioritized_storage(self):
        # Push n_step experiences
        for _ in range(self.n_step - 1):
            self.buffer.push(self.state, self.action, self.reward, self.next_state, self.done)
            self.assertEqual(len(self.buffer), 0)  # Buffer shouldn't store until n_step experiences are collected
            
        # Push one more experience to complete n_step sequence
        self.buffer.push(self.state, self.action, self.reward, self.next_state, self.done)
        self.assertEqual(len(self.buffer), 1)  # Now buffer should store the first experience
        
        # Push another experience
        self.buffer.push(self.state, self.action, self.reward, self.next_state, self.done)
        self.assertEqual(len(self.buffer), 2)  # Buffer should now have 2 experiences

    def test_n_step_prioritized_sample(self):
        # Fill buffer with enough experiences for sampling
        for _ in range(self.batch_size * (self.n_step + 2)):
            self.buffer.push(self.state, self.action, self.reward, self.next_state, self.done)
            
        # Sample from buffer
        one_step, n_step, indices, weights = self.buffer.sample()
        
        # Check sample sizes
        self.assertEqual(len(indices), self.batch_size)
        self.assertEqual(weights.shape[0], self.batch_size)
        self.assertEqual(len(one_step), 5)  # states, actions, rewards, next_states, dones
        self.assertEqual(len(n_step), 5)  # states, actions, n_rewards, n_next_states, n_dones
        
        # Check tensor shapes
        states, actions, rewards, next_states, dones = one_step
        self.assertEqual(states.shape[0], self.batch_size)
        self.assertEqual(actions.shape[0], self.batch_size)
        self.assertEqual(rewards.shape[0], self.batch_size)
        self.assertEqual(next_states.shape[0], self.batch_size)
        self.assertEqual(dones.shape[0], self.batch_size)
        
        # Check n-step returns
        n_states, n_actions, n_rewards, n_next_states, n_dones = n_step
        self.assertEqual(n_states.shape[0], self.batch_size)
        self.assertEqual(n_actions.shape[0], self.batch_size)
        self.assertEqual(n_rewards.shape[0], self.batch_size)
        self.assertEqual(n_next_states.shape[0], self.batch_size)
        self.assertEqual(n_dones.shape[0], self.batch_size)

if __name__ == '__main__':
    unittest.main() 