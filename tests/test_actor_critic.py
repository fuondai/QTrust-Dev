#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import torch
import numpy as np
from qtrust.agents.dqn.actor_critic_agent import ActorCriticAgent

class TestActorCriticAgent(unittest.TestCase):
    def setUp(self):
        self.state_size = 4
        self.action_size = 2
        self.batch_size = 32
        self.hidden_layers = [64, 32]  # Use consistent hidden layer sizes
        self.agent = ActorCriticAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            seed=42,
            batch_size=self.batch_size,
            buffer_size=1000,
            hidden_layers=self.hidden_layers,
            warm_up_steps=0,  # Disable warm up for testing
            update_every=1  # Update every step for testing
        )
        
    def test_initialization(self):
        """Test if agent is initialized correctly."""
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertEqual(self.agent.batch_size, self.batch_size)
        self.assertIsNotNone(self.agent.actor)
        self.assertIsNotNone(self.agent.critic)
        self.assertIsNotNone(self.agent.target_actor)
        self.assertIsNotNone(self.agent.target_critic)
        
    def test_act(self):
        """Test if agent can select actions."""
        state = np.random.rand(self.state_size)
        action = self.agent.act(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_size)
        
        # Test deterministic action selection
        action_deterministic = self.agent.act(state, explore=False)
        self.assertIsInstance(action_deterministic, int)
        self.assertTrue(0 <= action_deterministic < self.action_size)
        
    def test_step(self):
        """Test if agent can process experience steps."""
        # Reset memory and step counter
        self.agent.memory.memory.clear()
        self.agent.t_step = 0
        
        # Add n_step experiences to fill the n_step buffer
        for i in range(self.agent.memory.n_step):
            state = np.random.rand(self.state_size)
            action = self.agent.act(state)
            next_state = np.random.rand(self.state_size)
            reward = 1.0
            done = False if i < self.agent.memory.n_step - 1 else True
            
            # Add experience to memory
            self.agent.step(state, action, reward, next_state, done)
            
        # Verify memory has the experience
        self.assertEqual(len(self.agent.memory), 1)
        
        # Sample from memory to verify the experience
        states, actions, rewards, next_states, dones = self.agent.memory.sample()
        
        # Verify the sampled experience
        self.assertEqual(states.shape[0], 1)  # Should have 1 experience
        self.assertEqual(actions.shape[0], 1)
        self.assertEqual(rewards.shape[0], 1)
        self.assertEqual(next_states.shape[0], 1)
        self.assertEqual(dones.shape[0], 1)
        
    def test_learn(self):
        """Test if agent can learn from experiences."""
        # Reset memory and step counter
        self.agent.memory.memory.clear()
        self.agent.t_step = 0
        
        # Fill memory with some experiences
        for _ in range(self.batch_size):
            state = np.random.rand(self.state_size)
            action = self.agent.act(state)
            next_state = np.random.rand(self.state_size)
            reward = np.random.rand()
            done = bool(np.random.randint(2))
            self.agent.step(state, action, reward, next_state, done)
        
        # Get initial loss values
        initial_actor_loss = self.agent.actor_loss_history[-1] if self.agent.actor_loss_history else None
        initial_critic_loss = self.agent.critic_loss_history[-1] if self.agent.critic_loss_history else None
        
        # Sample experiences
        states, actions, rewards, next_states, dones = self.agent.memory.sample()
        
        # Print tensor shapes for debugging
        print(f"States shape: {states.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"Rewards shape: {rewards.shape}")
        print(f"Next states shape: {next_states.shape}")
        print(f"Dones shape: {dones.shape}")
        
        # Convert actions to one-hot for critic
        actions_one_hot = torch.zeros(len(actions), self.action_size, device=self.agent.device)
        actions_one_hot.scatter_(1, actions.unsqueeze(1), 1)
        print(f"Actions one-hot shape: {actions_one_hot.shape}")
        
        # Force learning step with both action forms
        self.agent._learn((states, actions, rewards, next_states, dones))
        
        # Verify that losses were updated
        self.assertGreater(len(self.agent.actor_loss_history), 0)
        self.assertGreater(len(self.agent.critic_loss_history), 0)
        
        if initial_actor_loss is not None:
            self.assertNotEqual(initial_actor_loss, self.agent.actor_loss_history[-1])
        if initial_critic_loss is not None:
            self.assertNotEqual(initial_critic_loss, self.agent.critic_loss_history[-1])
            
    def test_save_load(self):
        """Test if agent can save and load its state."""
        import tempfile
        import os
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filepath = tmp.name
            
        try:
            # Save the model
            self.agent.save(filepath)
            self.assertTrue(os.path.exists(filepath))
            
            # Create a new agent with identical architecture
            new_agent = ActorCriticAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                seed=42,
                hidden_layers=self.hidden_layers,  # Use same hidden layer sizes
                warm_up_steps=0,
                update_every=1
            )
            
            # Load the saved model
            success = new_agent.load(filepath)
            self.assertTrue(success)
            
            # Compare parameters
            for p1, p2 in zip(self.agent.actor.parameters(), new_agent.actor.parameters()):
                self.assertTrue(torch.equal(p1, p2))
                
            for p1, p2 in zip(self.agent.critic.parameters(), new_agent.critic.parameters()):
                self.assertTrue(torch.equal(p1, p2))
                
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
                
    def test_cache_functionality(self):
        """Test the caching functionality."""
        state = np.random.rand(self.state_size)
        
        # First call should miss cache
        initial_misses = self.agent.cache_misses
        _ = self.agent.act(state)
        self.assertEqual(self.agent.cache_misses, initial_misses + 1)
        
        # Second call with same state should hit cache
        initial_hits = self.agent.cache_hits
        _ = self.agent.act(state)
        self.assertEqual(self.agent.cache_hits, initial_hits + 1)
        
        # Test cache clearing
        self.agent.clear_cache()
        self.assertEqual(len(self.agent.action_cache), 0)
        self.assertEqual(len(self.agent.value_cache), 0)

if __name__ == '__main__':
    unittest.main() 