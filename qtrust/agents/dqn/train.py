"""
DQN Training Module

This module provides utility functions for training and evaluating DQN agents.
Key features:
- Training with early stopping and checkpointing
- Evaluation with rendering option
- Performance plotting and comparison tools
- Support for different epsilon decay strategies
"""

import numpy as np
import torch
import time
import os
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import matplotlib.pyplot as plt
from .utils import format_time, logger, plot_learning_curve
from .agent import DQNAgent

def train_dqn(agent: DQNAgent, 
             env, 
             n_episodes: int = 1000, 
             max_t: int = 1000, 
             eps_decay_type: str = 'exponential', 
             checkpoint_freq: int = 100,
             early_stopping: bool = True,
             patience: int = 50,
             min_improvement: float = 0.01,
             eval_interval: int = 10,
             render: bool = False,
             verbose: bool = True):
    """
    Train a DQN agent with early stopping and checkpointing.
    
    Args:
        agent: DQN agent to train
        env: Training environment
        n_episodes: Maximum number of training episodes
        max_t: Maximum steps per episode
        eps_decay_type: Type of epsilon decay ('exponential', 'linear', or 'custom')
        checkpoint_freq: Episodes between checkpoints
        early_stopping: Whether to use early stopping
        patience: Episodes to wait before early stopping
        min_improvement: Minimum improvement threshold
        eval_interval: Episodes between evaluations
        render: Whether to render environment
        verbose: Whether to print progress
        
    Returns:
        Dict: Training results including rewards, validation rewards, best reward, and training time
    """
    train_start_time = time.time()
    rewards = []
    best_avg_reward = -float('inf')
    episodes_without_improvement = 0
    window_size = 100
    
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated) and steps < max_t:
            # Choose action
            action = agent.act(state, agent.epsilon)
            
            # Execute action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience and update policy
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            steps += 1
            
            if render:
                env.render()
        
        # Update epsilon
        agent.update_epsilon(eps_decay_type)
        
        # Add reward
        rewards.append(score)
        agent.training_rewards.append(score)
        
        # Calculate moving average over window
        avg_reward = np.mean(rewards[-min(len(rewards), window_size):])
        
        # Print progress information
        if verbose and (episode % 10 == 0 or episode == 1):
            elapsed_time = time.time() - train_start_time
            time_str = format_time(elapsed_time)
            
            remaining_episodes = n_episodes - episode
            if episode > 1:
                time_per_episode = elapsed_time / episode
                remaining_time = remaining_episodes * time_per_episode
                eta_str = format_time(remaining_time)
            else:
                eta_str = "N/A"
            
            logger.info(f"Episode {episode}/{n_episodes} | Score: {score:.2f} | Avg Score: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f} | Time: {time_str} | ETA: {eta_str}")
        
        # Save checkpoint
        if episode % checkpoint_freq == 0:
            is_best = avg_reward > best_avg_reward
            if is_best:
                best_avg_reward = avg_reward
                episodes_without_improvement = 0
            agent.save_checkpoint(episode, avg_reward, is_best)
        
        # Evaluate agent
        if episode % eval_interval == 0:
            eval_reward = evaluate_dqn(agent, env, 5, max_t, render=False)
            agent.validation_rewards.append(eval_reward)
            if verbose:
                logger.info(f"Evaluation at episode {episode}: Average reward = {eval_reward:.2f}")
            
            # Update best score and early stopping
            if eval_reward > best_avg_reward + min_improvement:
                best_avg_reward = eval_reward
                episodes_without_improvement = 0
                agent.save_checkpoint(episode, eval_reward, True)
            else:
                episodes_without_improvement += eval_interval
        
        # Early stopping
        if early_stopping and episodes_without_improvement >= patience:
            logger.info(f"Early stopping at episode {episode}. No improvement after {patience} episodes.")
            break
    
    # End training
    agent.load_best_model()  # Load best model
    total_time = time.time() - train_start_time
    logger.info(f"Training completed after {episode} episodes.")
    logger.info(f"Training time: {format_time(total_time)}")
    logger.info(f"Best score: {best_avg_reward:.2f}")
    
    # Return results
    return {
        'rewards': rewards,
        'validation_rewards': agent.validation_rewards,
        'best_reward': best_avg_reward,
        'training_time': total_time,
        'episodes': episode
    }

def evaluate_dqn(agent: DQNAgent, env, n_episodes: int = 5, max_t: int = 1000, render: bool = False):
    """
    Evaluate a trained DQN agent's performance.
    
    Args:
        agent: Trained DQN agent to evaluate
        env: Evaluation environment
        n_episodes: Number of evaluation episodes
        max_t: Maximum steps per episode
        render: Whether to render environment
        
    Returns:
        float: Average reward across evaluation episodes
    """
    rewards = []
    
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated) and steps < max_t:
            # Choose action using current policy (no epsilon)
            action = agent.act(state, eps=0.0)
            
            # Execute action
            next_state, reward, done, truncated, _ = env.step(action)
            
            state = next_state
            score += reward
            steps += 1
            
            if render:
                env.render()
        
        rewards.append(score)
    
    avg_reward = np.mean(rewards)
    return avg_reward

def plot_dqn_rewards(rewards: List[float], 
                    val_rewards: Optional[List[float]] = None,
                    window_size: int = 20,
                    title: str = "DQN Training Rewards", 
                    save_path: Optional[str] = None):
    """
    Plot training and validation rewards with moving average.
    
    Args:
        rewards: List of training episode rewards
        val_rewards: List of validation episode rewards
        window_size: Window size for moving average
        title: Plot title
        save_path: Path to save plot (displays if None)
    """
    plt.figure(figsize=(12, 6))
    
    # Plot individual episode rewards
    plt.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Calculate moving average
    if len(rewards) > 1:
        avg_rewards = []
        for i in range(len(rewards)):
            if i < window_size:
                avg_rewards.append(np.mean(rewards[:i+1]))
            else:
                avg_rewards.append(np.mean(rewards[i-window_size+1:i+1]))
        
        plt.plot(avg_rewards, color='blue', linewidth=2, 
                 label=f'Moving Avg ({window_size} ep)')
    
    # Plot validation rewards if provided
    if val_rewards is not None and len(val_rewards) > 0:
        # Calculate evaluation episode numbers
        eval_episodes = np.linspace(0, len(rewards), len(val_rewards), endpoint=False)
        eval_episodes = [int(ep) for ep in eval_episodes]
        
        plt.plot(eval_episodes, val_rewards, 'ro-', label='Evaluation')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()

def compare_dqn_variants(env, variants: List[Dict[str, Any]], 
                         n_episodes: int = 500, 
                         max_t: int = 1000):
    """
    Compare different DQN variants on the same environment.
    
    Args:
        env: Environment for comparison
        variants: List of dictionaries with variant parameters
        n_episodes: Number of training episodes per variant
        max_t: Maximum steps per episode
        
    Returns:
        Dict: Results for each variant including rewards and best performance
    """
    results = {}
    
    for variant in variants:
        name = variant.pop('name', f"Variant-{len(results)}")
        logger.info(f"\nTraining variant: {name}")
        
        # Create agent with variant parameters
        agent = DQNAgent(**variant)
        
        # Train agent
        train_result = train_dqn(
            agent=agent,
            env=env,
            n_episodes=n_episodes,
            max_t=max_t,
            early_stopping=True,
            verbose=True
        )
        
        # Store results
        results[name] = {
            'rewards': train_result['rewards'],
            'validation_rewards': train_result['validation_rewards'],
            'best_reward': train_result['best_reward'],
            'agent': agent
        }
        
        logger.info(f"Variant {name} completed. Best reward: {train_result['best_reward']:.2f}")
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    
    for name, result in results.items():
        rewards = result['rewards']
        
        # Calculate moving average
        avg_rewards = []
        window_size = 20
        for i in range(len(rewards)):
            if i < window_size:
                avg_rewards.append(np.mean(rewards[:i+1]))
            else:
                avg_rewards.append(np.mean(rewards[i-window_size+1:i+1]))
        
        plt.plot(avg_rewards, label=f"{name} (Best: {result['best_reward']:.2f})")
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('DQN Variants Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('dqn_variants_comparison.png')
    logger.info("Comparison plot saved to dqn_variants_comparison.png")
    
    return results 