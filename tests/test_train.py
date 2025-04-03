import unittest
import numpy as np
import torch
import gym
from qtrust.agents.dqn.train import train_dqn, evaluate_dqn, plot_dqn_rewards, compare_dqn_variants
from qtrust.agents.dqn.agent import DQNAgent

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('CartPole-v1', render_mode=None)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        # Create a small DQN agent for testing
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=32,
            buffer_size=100,
            batch_size=4,
            learning_rate=1e-3,
            update_every=1
        )
        
        # Set seeds
        np.random.seed(42)
        torch.manual_seed(42)
        
    def test_train_dqn(self):
        # Test training with minimal episodes
        results = train_dqn(
            agent=self.agent,
            env=self.env,
            n_episodes=2,
            max_t=50,
            checkpoint_freq=1,
            early_stopping=False,
            eval_interval=1,
            verbose=False
        )
        
        # Check results structure
        self.assertIn('rewards', results)
        self.assertIn('validation_rewards', results)
        self.assertIn('best_reward', results)
        self.assertIn('training_time', results)
        self.assertIn('episodes', results)
        
        # Check data types and shapes
        self.assertIsInstance(results['rewards'], list)
        self.assertEqual(len(results['rewards']), 2)
        
    def test_evaluate_dqn(self):
        # Test evaluation
        avg_reward = evaluate_dqn(
            agent=self.agent,
            env=self.env,
            n_episodes=2,
            max_t=50,
            render=False
        )
        
        self.assertIsInstance(avg_reward, float)
        
    def test_plot_dqn_rewards(self):
        # Test plotting with minimal data
        rewards = [1.0, 2.0, 3.0]
        val_rewards = [1.5, 2.5]
        
        # Should not raise any exceptions
        plot_dqn_rewards(rewards, val_rewards, window_size=2)
        
    def test_compare_dqn_variants(self):
        # Test comparison with two simple variants
        variants = [
            {
                'name': 'Small',
                'state_size': self.state_size,
                'action_size': self.action_size,
                'hidden_size': 32,
                'buffer_size': 100,
                'batch_size': 4,
                'learning_rate': 1e-3
            },
            {
                'name': 'Tiny',
                'state_size': self.state_size,
                'action_size': self.action_size,
                'hidden_size': 16,
                'buffer_size': 50,
                'batch_size': 2,
                'learning_rate': 2e-3
            }
        ]
        
        results = compare_dqn_variants(
            env=self.env,
            variants=variants,
            n_episodes=2,
            max_t=50
        )
        
        # Check results structure
        self.assertEqual(len(results), 2)
        for name in ['Small', 'Tiny']:
            self.assertIn(name, results)
            self.assertIn('rewards', results[name])
            self.assertIn('validation_rewards', results[name])
            self.assertIn('best_reward', results[name])
            self.assertIn('agent', results[name])
    
    def tearDown(self):
        self.env.close()

if __name__ == '__main__':
    unittest.main() 