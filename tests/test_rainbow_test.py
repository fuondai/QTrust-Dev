import unittest
import numpy as np
import torch
import gym
from qtrust.agents.dqn.test_rainbow import test_rainbow_dqn, test_actor_critic, compare_methods
from qtrust.agents.dqn.agent import DQNAgent
from qtrust.agents.dqn.rainbow_agent import RainbowDQNAgent
from qtrust.agents.dqn.actor_critic_agent import ActorCriticAgent

class TestRainbowTest(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('CartPole-v1', render_mode=None)
        self.num_episodes = 5  # More episodes
        self.max_steps = 100  # Fewer steps per episode
        self.seed = 42
        
        # Get state and action space dimensions
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        # Initialize agent with small buffer and batch size for testing
        self.agent = RainbowDQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            buffer_size=200,  # Smaller buffer for testing
            batch_size=4,  # Smaller batch size
            n_step=2,  # Smaller n-step
            n_atoms=11,  # Fewer atoms
            v_min=-10.0,
            v_max=10.0,
            hidden_layers=[64, 32],  # Smaller network
            learning_rate=1e-3,
            update_every=1,  # Update every step
            warm_up_steps=4  # Smaller warm up period
        )
        
        # Set seeds
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def test_rainbow_dqn_function(self):
        # Test Rainbow DQN training
        agent, scores, avg_scores = test_rainbow_dqn(
            env=self.env,
            num_episodes=self.num_episodes,
            max_steps=self.max_steps,
            render=False,
            seed=self.seed
        )
        
        # Check outputs
        self.assertIsNotNone(agent)
        self.assertEqual(len(scores), self.num_episodes)
        self.assertEqual(len(avg_scores), self.num_episodes)
        
        # Check if agent has loss history
        self.assertTrue(hasattr(agent, 'loss_history'))
        
        # Check if agent has learned something (loss history should be populated)
        self.assertTrue(len(agent.loss_history) > 0, "Agent should have learned something")
        
    def test_actor_critic_function(self):
        # Test Actor-Critic training
        agent, scores, avg_scores = test_actor_critic(
            env=self.env,
            num_episodes=self.num_episodes,
            max_steps=self.max_steps,
            render=False,
            seed=self.seed
        )
        
        # Check outputs
        self.assertIsNotNone(agent)
        self.assertEqual(len(scores), self.num_episodes)
        self.assertEqual(len(avg_scores), self.num_episodes)
        
        # Check if agent has loss histories
        self.assertTrue(hasattr(agent, 'actor_loss_history'))
        self.assertTrue(hasattr(agent, 'critic_loss_history'))
        
        # Check if agent has learned something (loss histories should be populated)
        self.assertTrue(len(agent.actor_loss_history) > 0, "Agent should have learned something (actor)")
        self.assertTrue(len(agent.critic_loss_history) > 0, "Agent should have learned something (critic)")
        
    def test_compare_methods_function(self):
        # Test comparison of methods
        agents, all_scores, all_avg_scores = compare_methods(
            env=self.env,
            num_episodes=self.num_episodes,
            max_steps=self.max_steps,
            seed=self.seed
        )
        
        # Check outputs
        self.assertEqual(len(agents), 3)  # DQN, Rainbow DQN, Actor-Critic
        self.assertEqual(len(all_scores), 3)
        self.assertEqual(len(all_avg_scores), 3)
        
        # Check each agent's scores
        for scores in all_scores:
            self.assertEqual(len(scores), self.num_episodes)
        
        # Check each agent's average scores
        for avg_scores in all_avg_scores:
            self.assertEqual(len(avg_scores), self.num_episodes)
        
        # Check if each agent has learned something
        for agent in agents:
            if isinstance(agent, DQNAgent):
                self.assertTrue(len(agent.loss_history) > 0, f"{type(agent).__name__} should have learned something")
            elif isinstance(agent, RainbowDQNAgent):
                self.assertTrue(len(agent.loss_history) > 0, f"{type(agent).__name__} should have learned something")
            elif isinstance(agent, ActorCriticAgent):
                self.assertTrue(len(agent.actor_loss_history) > 0, f"{type(agent).__name__} should have learned something (actor)")
                self.assertTrue(len(agent.critic_loss_history) > 0, f"{type(agent).__name__} should have learned something (critic)")
    
    def tearDown(self):
        self.env.close()

if __name__ == '__main__':
    unittest.main() 