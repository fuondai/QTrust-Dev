#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QTrust - Optimized Blockchain Sharding with Deep Reinforcement Learning

This file implements an optimized training pipeline for DQN Agent to efficiently 
learn optimal sharding strategies in blockchain networks. It features prioritized 
experience replay, dueling DQN architecture, and early stopping mechanisms to achieve 
better performance with reduced training time.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import time
import random
from pathlib import Path
import json

# Add current directory to PYTHONPATH to ensure modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from qtrust.simulation.blockchain_environment import BlockchainEnvironment
from qtrust.agents.dqn.agent import DQNAgent
from qtrust.agents.dqn.train import train_dqn, evaluate_dqn, plot_dqn_rewards
from qtrust.consensus.adaptive_consensus import AdaptiveConsensus
from qtrust.routing.mad_rapid import MADRAPIDRouter
from qtrust.trust.htdcm import HTDCM
from qtrust.federated.federated_learning import FederatedLearning, FederatedModel, FederatedClient
from qtrust.utils.metrics import (
    calculate_throughput, 
    calculate_latency_metrics,
    calculate_energy_efficiency,
    calculate_security_metrics,
    calculate_cross_shard_transaction_ratio,
    plot_performance_metrics,
    plot_comparison_charts
)
from qtrust.utils.data_generation import (
    generate_network_topology,
    assign_nodes_to_shards,
    generate_transactions
)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments with optimal values for training."""
    parser = argparse.ArgumentParser(description='QTrust - Optimized DQN Agent Training')
    
    parser.add_argument('--num-shards', type=int, default=12, 
                        help='Number of shards in the network (default: 12)')
    parser.add_argument('--nodes-per-shard', type=int, default=16, 
                        help='Number of nodes in each shard (default: 16)')
    parser.add_argument('--episodes', type=int, default=2000, 
                        help='Number of episodes for training (default: 2000)')
    parser.add_argument('--max-steps', type=int, default=1000, 
                        help='Maximum steps in each episode (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='Batch size for DQN model (default: 128)')
    parser.add_argument('--hidden-size', type=int, default=256, 
                        help='Hidden layer size for DQN model (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0005, 
                        help='Learning rate (default: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--epsilon-start', type=float, default=1.0, 
                        help='Initial epsilon value (default: 1.0)')
    parser.add_argument('--epsilon-end', type=float, default=0.01, 
                        help='Final epsilon value (default: 0.01)')
    parser.add_argument('--epsilon-decay', type=float, default=0.998, 
                        help='Epsilon decay rate (default: 0.998)')
    parser.add_argument('--memory-size', type=int, default=100000, 
                        help='Replay memory size (default: 100000)')
    parser.add_argument('--target-update', type=int, default=10, 
                        help='Update target model every N episodes (default: 10)')
    parser.add_argument('--save-dir', type=str, default='models/optimized_dqn', 
                        help='Directory to save models')
    parser.add_argument('--log-interval', type=int, default=10, 
                        help='Episodes between printing results')
    parser.add_argument('--eval-interval', type=int, default=50,
                        help='Episodes between evaluations')
    parser.add_argument('--patience', type=int, default=100,
                        help='Episodes without improvement for early stopping')
    parser.add_argument('--enable-federated', action='store_true', 
                        help='Enable Federated Learning mode')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Training device (cuda or cpu)')
    parser.add_argument('--attack-scenario', type=str, 
                        choices=['51_percent', 'sybil', 'eclipse', 'selfish_mining', 'bribery', 'ddos', 'finney', 'mixed', 'none'], 
                        default='none', help='Attack scenario')
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the blockchain environment."""
    print("Initializing blockchain environment...")
    
    env = BlockchainEnvironment(
        num_shards=args.num_shards,
        num_nodes_per_shard=args.nodes_per_shard,
        max_steps=args.max_steps,
        latency_penalty=0.5,
        energy_penalty=0.3,
        throughput_reward=1.0,
        security_reward=0.8
    )
    
    return env

def setup_dqn_agent(env, args):
    """Set up the DQN Agent with optimal configuration."""
    print("Initializing DQN Agent with optimal configuration...")
    
    # Get state size from environment
    state = env.reset()
    state_size = len(state)
    print(f"Actual state size: {state_size}")
    
    # Calculate total possible actions
    total_actions = env.num_shards * 3  # num_shards * num_consensus_protocols
    
    # Create agent with optimal configuration
    agent = DQNAgent(
        state_size=state_size,
        action_size=total_actions,
        seed=SEED,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=0.001,  # Target network update coefficient
        hidden_layers=[args.hidden_size, args.hidden_size//2],
        update_every=4,  # Update network every 4 steps
        device=args.device,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.epsilon_end,
        buffer_size=args.memory_size,
        prioritized_replay=True,  # Use Prioritized Experience Replay
        alpha=0.6,  # Alpha for prioritized replay
        beta_start=0.4,   # Beta for prioritized replay
        dueling=True,  # Use Dueling DQN for better learning efficiency
        clip_gradients=True,  # Limit gradients to stabilize learning
        grad_clip_value=1.0  # Gradient clipping value
    )
    
    print(f"Created DQN Agent with {state_size} states and {total_actions} actions.")
    print(f"Using Prioritized Replay: {agent.prioritized_replay}")
    
    # Create wrapper for agent to convert from single action to MultiDiscrete action
    class DQNAgentWrapper:
        def __init__(self, agent, num_shards, num_consensus_protocols=3):
            self.agent = agent
            self.num_shards = num_shards
            self.num_consensus_protocols = num_consensus_protocols
            
        def act(self, state, eps=None):
            # Get action from base agent
            action_idx = self.agent.act(state, eps)
            
            # Convert action_idx to MultiDiscrete action [shard_idx, consensus_idx]
            shard_idx = action_idx % self.num_shards
            consensus_idx = (action_idx // self.num_shards) % self.num_consensus_protocols
            
            return np.array([shard_idx, consensus_idx], dtype=np.int32)
            
        def step(self, state, action, reward, next_state, done):
            # Convert MultiDiscrete action to single action
            # Ensure action is an array
            if isinstance(action, np.ndarray) and len(action) >= 2:
                action_idx = action[0] + action[1] * self.num_shards
            else:
                # If action is an integer, process directly
                action_idx = action
            
            # Call step of base agent
            self.agent.step(state, action_idx, reward, next_state, done)
            
        def save(self, path):
            return self.agent.save(path)
            
        def load(self, path):
            return self.agent.load(path)
            
        # Forward properties
        @property
        def epsilon(self):
            return self.agent.epsilon
        
        @property
        def device(self):
            return self.agent.device
            
    # Wrap agent in wrapper
    wrapped_agent = DQNAgentWrapper(agent, args.num_shards)
    
    return wrapped_agent, agent

def train_optimized_dqn(env, agent, base_agent, args):
    """
    Train the DQN agent with optimized parameters and early stopping.
    
    Args:
        env: The blockchain environment
        agent: The wrapped DQN agent
        base_agent: The underlying DQN agent
        args: Command line arguments
        
    Returns:
        dict: Dictionary containing training metrics and results
    """
    print("\nStarting optimized training...\n")
    start_time = time.time()
    
    # Set up logging
    train_rewards = []
    eval_rewards = []
    eval_metrics = []
    training_times = []
    best_reward = -float('inf')
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    patience_counter = 0
    
    # Create directory for saving models
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize pbar for tracking episodes
    pbar = tqdm(range(1, args.episodes + 1), desc="Training Episode")
    
    # Training loop
    for episode in pbar:
        state = env.reset()
        total_reward = 0
        episode_start_time = time.time()
        
        # Run a single episode
        for step in range(args.max_steps):
            # Determine epsilon for exploration
            current_eps = max(
                args.epsilon_end, 
                args.epsilon_start * (args.epsilon_decay ** episode)
            )
            
            # Get action based on current state
            action = agent.act(state, current_eps)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience in memory
            agent.step(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Record episode duration
        episode_duration = time.time() - episode_start_time
        training_times.append(episode_duration)
        
        # Record reward
        train_rewards.append(total_reward)
        
        # Print progress
        if episode % args.log_interval == 0:
            avg_reward = np.mean(train_rewards[-args.log_interval:])
            pbar.set_postfix({
                'Reward': f'{avg_reward:.2f}', 
                'Epsilon': f'{base_agent.epsilon:.2f}'
            })
            
        # Evaluate agent
        if episode % args.eval_interval == 0:
            # Run evaluation
            eval_reward, eval_info = evaluate_agent_wrapper(agent, env, n_episodes=5)
            eval_rewards.append(eval_reward)
            eval_metrics.append(eval_info)
            
            print(f"\nEvaluation at episode {episode}: Avg Reward = {eval_reward:.2f}")
            print(f"Throughput: {eval_info['throughput']:.2f} tx/s, Latency: {eval_info['avg_latency']:.2f} ms")
            print(f"Energy efficiency: {eval_info['energy_efficiency']:.4f}, Security: {eval_info['security_score']:.4f}")
            
            # Check if we have a new best model
            if eval_reward > best_reward:
                print(f"New best model with reward: {eval_reward:.2f} (previous: {best_reward:.2f})")
                best_reward = eval_reward
                agent.save(best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} evaluations (patience: {args.patience})")
            
            # Early stopping check
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {episode} episodes due to no improvement")
                break
    
    # Calculate total training time
    total_training_time = time.time() - start_time
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    agent.save(final_model_path)
    
    # Save training metrics
    metrics = {
        'train_rewards': train_rewards,
        'eval_rewards': eval_rewards,
        'eval_metrics': eval_metrics,
        'training_times': training_times,
        'total_training_time': total_training_time,
        'best_reward': best_reward,
        'episodes_completed': episode,
        'final_epsilon': base_agent.epsilon
    }
    
    with open(os.path.join(args.save_dir, 'training_metrics.json'), 'w') as f:
        json.dump({k: v if not isinstance(v, list) or not isinstance(v[0], dict) else str(v) 
                 for k, v in metrics.items()}, f, indent=4)
    
    # Plot training rewards
    plt.figure(figsize=(12, 6))
    plt.plot(train_rewards, label='Training Rewards')
    plt.plot([i * args.eval_interval for i in range(len(eval_rewards))], 
             eval_rewards, 'r-', label='Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training and Evaluation Rewards')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, 'rewards.png'))
    
    print(f"\nTraining completed in {total_training_time:.2f} seconds")
    print(f"Best model saved to {best_model_path} with reward {best_reward:.2f}")
    
    return metrics

def evaluate_agent_wrapper(agent, env, n_episodes=10, max_t=1000):
    """
    Evaluate the wrapped DQN agent.
    
    Args:
        agent: The wrapped DQN agent
        env: The blockchain environment
        n_episodes: Number of evaluation episodes
        max_t: Maximum steps per episode
        
    Returns:
        tuple: (average reward, additional metrics)
    """
    rewards = []
    throughputs = []
    latencies = []
    energy_scores = []
    security_scores = []
    
    for i in range(n_episodes):
        state = env.reset()
        total_reward = 0
        
        for t in range(max_t):
            # Get deterministic action
            action = agent.act(state, eps=0)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Collect metrics
        rewards.append(total_reward)
        throughputs.append(info.get('throughput', 0))
        latencies.append(info.get('avg_latency', 0))
        energy_scores.append(info.get('energy_efficiency', 0))
        security_scores.append(info.get('security_score', 0))
    
    # Calculate averages
    avg_reward = np.mean(rewards)
    avg_throughput = np.mean(throughputs)
    avg_latency = np.mean(latencies)
    avg_energy = np.mean(energy_scores)
    avg_security = np.mean(security_scores)
    
    return avg_reward, {
        'throughput': avg_throughput,
        'avg_latency': avg_latency,
        'energy_efficiency': avg_energy,
        'security_score': avg_security
    }

def main():
    """Main function for running optimized training."""
    # Parse arguments
    args = parse_args()
    
    # Setup environment and agent
    env = setup_environment(args)
    agent_wrapper, base_agent = setup_dqn_agent(env, args)
    
    # Train agent
    training_metrics = train_optimized_dqn(env, agent_wrapper, base_agent, args)
    
    # Display final metrics
    print("\nFinal Training Metrics:")
    print(f"Total Episodes: {training_metrics['episodes_completed']}")
    print(f"Best Reward: {training_metrics['best_reward']:.2f}")
    print(f"Final Epsilon: {training_metrics['final_epsilon']:.4f}")
    print(f"Total Training Time: {training_metrics['total_training_time'] / 60:.2f} minutes")
    
    # Load best model
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    agent_wrapper.load(best_model_path)
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    final_reward, final_metrics = evaluate_agent_wrapper(
        agent_wrapper, env, n_episodes=20, max_t=args.max_steps
    )
    
    print("\nFinal Evaluation Metrics:")
    print(f"Average Reward: {final_reward:.2f}")
    print(f"Throughput: {final_metrics['throughput']:.2f} transactions/second")
    print(f"Average Latency: {final_metrics['avg_latency']:.2f} ms")
    print(f"Energy Efficiency: {final_metrics['energy_efficiency']:.4f}")
    print(f"Security Score: {final_metrics['security_score']:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 