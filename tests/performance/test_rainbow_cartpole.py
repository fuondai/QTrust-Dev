"""
Rainbow DQN CartPole Test Script

This script tests the Rainbow DQN algorithm implementation on the CartPole environment.
It demonstrates how to set up a Rainbow DQN agent, train it on a simple environment,
and visualize the learning progress through performance metrics.

The script includes functionality for:
- Setting up the CartPole environment
- Initializing a Rainbow DQN agent with configurable parameters
- Training the agent for a specified number of episodes
- Monitoring and visualizing learning progress
- Saving performance plots
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from qtrust.agents.dqn.rainbow_agent import RainbowDQNAgent
import time

def test_rainbow_dqn(env_name='CartPole-v1', num_episodes=10, max_steps=500, render=False):
    """
    This script tests the Rainbow DQN algorithm in a specified environment.
    
    Args:
        env_name: Name of the gym environment
        num_episodes: Number of test episodes
        max_steps: Maximum steps per episode
        render: Whether to display the environment
    """
    env = gym.make(env_name)
    
    # Get state and action space dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Initialize Rainbow DQN Agent
    agent = RainbowDQNAgent(
        state_size=state_size,
        action_size=action_size,
        n_step=3,
        n_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        hidden_layers=[512, 256],
        learning_rate=5e-4,
        batch_size=32,
        buffer_size=10000
    )
    
    # Performance monitoring
    scores = []
    avg_scores = []
    start_time = time.time()
    
    # Train agent across episodes
    for i_episode in range(1, num_episodes+1):
        state, _ = env.reset()
        score = 0
        for t in range(max_steps):
            # Render environment if requested
            if render:
                env.render()
                
            # Select action
            action = agent.act(state)
            
            # Execute action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Learn from experience
            agent.step(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            if done or truncated:
                break
        
        # Save score
        scores.append(score)
        
        # Calculate average score
        avg_score = np.mean(scores)
        avg_scores.append(avg_score)
        
        # Print information
        elapsed_time = time.time() - start_time
        print(f"Episode {i_episode}/{num_episodes} | Score: {score:.2f} | Avg Score: {avg_score:.2f} | Time: {elapsed_time:.2f}s")
    
    # Plot learning curve
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, num_episodes+1), scores, alpha=0.6, label='Score')
    plt.plot(range(1, num_episodes+1), avg_scores, label='Avg Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Rainbow DQN Learning Curve')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(agent.loss_history, alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Rainbow DQN Loss')
    
    plt.tight_layout()
    plt.savefig('rainbow_dqn_cartpole_learning_curve.png')
    plt.show()
    
    env.close()
    return agent, scores, avg_scores

if __name__ == "__main__":
    test_rainbow_dqn(num_episodes=20) 