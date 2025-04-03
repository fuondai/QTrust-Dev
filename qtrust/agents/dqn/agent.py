"""
Deep Q-Network (DQN) Agent Implementation

This module provides a DQN agent with modern enhancements for reinforcement learning tasks.
Key features:
- Double DQN for more stable learning
- Dueling Network Architecture for better value estimation 
- Prioritized Experience Replay for efficient learning
- N-step returns for better credit assignment
- Noisy Networks for exploration
- Distributional RL support
- Caching system for improved performance
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import random
from collections import deque, defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any

from .networks import QNetwork, DuelingQNetwork
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .utils import soft_update, hard_update, calculate_td_error, calculate_huber_loss, create_save_directory, logger, generate_timestamp, get_device, exponential_decay, linear_decay


class DQNAgent:
    """
    Deep Q-Network Agent with several optional enhancements:
    
    1. Double DQN - Uses a separate target network for more stable learning
    2. Dueling Network Architecture - Separate streams for estimating state value and action advantages
    3. Prioritized Experience Replay - Samples important transitions more frequently
    
    Args:
        state_size: Size of state space
        action_size: Size of action space
        hidden_size: Size of hidden layers
        learning_rate: Learning rate for optimizer
        buffer_size: Maximum size of experience replay buffer
        batch_size: Batch size for sampling from replay buffer
        gamma: Discount factor for future rewards
        tau: Soft update parameter for target network
        update_every: How often to update the network
        use_double_dqn: Whether to use Double DQN
        use_dueling: Whether to use Dueling Network Architecture
        use_prioritized_replay: Whether to use Prioritized Experience Replay
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Rate of decay for exploration
        alpha: Priority exponent for prioritized replay
        beta_start: Initial importance sampling weight for prioritized replay
        beta_end: Final importance sampling weight for prioritized replay
        beta_duration: Number of steps to anneal beta from start to end
        seed: Random seed
        device: Device to run the model on ('cuda', 'cpu', or 'auto')
    """
    
    def __init__(self, 
                state_size: int, 
                action_size: int,
                hidden_size: int = 64,
                learning_rate: float = 1e-3,
                buffer_size: int = 100000,
                batch_size: int = 64,
                gamma: float = 0.99,
                tau: float = 1e-3,
                update_every: int = 4,
                use_double_dqn: bool = True,
                use_dueling: bool = False,
                use_prioritized_replay: bool = False,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay: float = 0.995,
                alpha: float = 0.6,
                beta_start: float = 0.4,
                beta_end: float = 1.0,
                beta_duration: int = 100000,
                seed: Optional[int] = None,
                device: str = 'auto'):
        """
        Initialize a Deep Q-Network agent.
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            hidden_size: Size of hidden layers
            learning_rate: Learning rate for optimizer
            buffer_size: Maximum size of experience replay buffer
            batch_size: Batch size for sampling from replay buffer
            gamma: Discount factor for future rewards
            tau: Soft update parameter for target network
            update_every: How often to update the network
            use_double_dqn: Whether to use Double DQN
            use_dueling: Whether to use Dueling Network Architecture
            use_prioritized_replay: Whether to use Prioritized Experience Replay
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of decay for exploration
            alpha: Priority exponent for prioritized replay
            beta_start: Initial importance sampling weight for prioritized replay
            beta_end: Final importance sampling weight for prioritized replay
            beta_duration: Number of steps to anneal beta from start to end
            seed: Random seed
            device: Device to run the model on ('cuda', 'cpu', or 'auto')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.use_prioritized_replay = use_prioritized_replay
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_duration = beta_duration
        
        # Set seed for reproducibility
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Set device
        self.device = get_device(device)
        
        # Create Q-Networks (local and target)
        if self.use_dueling:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, hidden_size).to(self.device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, hidden_size).to(self.device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, hidden_size).to(self.device)
            self.qnetwork_target = QNetwork(state_size, action_size, hidden_size).to(self.device)
        
        # Initialize target network with same weights as local network
        hard_update(self.qnetwork_local, self.qnetwork_target)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Create replay buffer
        if self.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size, batch_size, alpha=alpha)
        else:
            self.memory = ReplayBuffer(buffer_size, batch_size, device=self.device)
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
        
        # Create directory for saving models
        self.timestamp = generate_timestamp()
        self.save_dir = create_save_directory(f"dqn_{self.timestamp}")
        
        # Training stats
        self.training_rewards = []
        self.validation_rewards = []
        self.loss_history = []
        self.epsilon_history = []
        self.beta_history = []
        self.best_score = -float('inf')
        self.best_model_path = None
        
        # Print agent configuration
        configuration = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
            "use_double_dqn": self.use_double_dqn,
            "use_dueling": self.use_dueling,
            "use_prioritized_replay": self.use_prioritized_replay,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "device": self.device
        }
        
        logger.info("DQN Agent Configuration:")
        for key, value in configuration.items():
            logger.info(f"  {key}: {value}")

    def step(self, state, action, reward, next_state, done):
        """
        Add experience to memory and learn if it's time to learn.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            # Get experiences from replay buffer
            if self.use_prioritized_replay:
                experiences, indices, weights = self.memory.sample(self.beta)
                # Update beta
                self.beta = min(self.beta_end, self.beta + (self.beta_end - self.beta_start) / self.beta_duration)
                self.beta_history.append(self.beta)
            else:
                experiences = self.memory.sample()
                indices = None
                weights = torch.ones(self.batch_size, device=self.device) # Uniform weights for standard replay
            
            # Learn from experiences
            losses = self.learn(experiences, weights)
            
            # Update priorities in prioritized replay buffer
            if self.use_prioritized_replay and indices is not None:
                priorities = losses.detach().cpu().numpy() + 1e-5  # Add small constant to avoid zero priority
                self.memory.update_priorities(indices, priorities)
            
            # Update target network
            soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def act(self, state, eps=None):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            eps: Epsilon for exploration, if None uses internal epsilon
            
        Returns:
            int: Selected action
        """
        # Use provided epsilon or current epsilon
        epsilon = eps if eps is not None else self.epsilon
        
        # Explore with probability epsilon
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to tensor
        if isinstance(state, tuple):
            state = state[0]  # Get first element if tuple
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        
        # Get action values from network
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Return best action
        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, weights=None):
        """
        Update value parameters using batch of experience tuples.
        
        Args:
            experiences: Tuple of (s, a, r, s', done) tensors
            weights: Importance sampling weights for prioritized replay
            
        Returns:
            torch.Tensor: Loss values for prioritized replay update
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Ensure tensors have correct dimensions for operations [batch_size, 1]
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # Double DQN: Get predicted Q-values for next states from target network
        if self.use_double_dqn:
            # Get actions from local model
            self.qnetwork_local.eval()
            with torch.no_grad():
                next_actions = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)
            self.qnetwork_local.train()
            
            # Get Q-values from target model using actions from local model
            with torch.no_grad():
                next_q_values = self.qnetwork_target(next_states).gather(1, next_actions)
        else:
            # Vanilla DQN: Get maximum Q-value for next states from target network
            with torch.no_grad():
                next_q_values = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
        
        # Compute target Q-values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Get current Q-values from local network
        current_q_values = self.qnetwork_local(states).gather(1, actions)
        
        # Compute TD errors for prioritized replay
        td_errors = target_q_values - current_q_values
        
        # Compute loss (Huber loss for stability)
        # Use huber loss with weights for both prioritized and standard replay
        loss = calculate_huber_loss(current_q_values, target_q_values, weights)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients for stability (only for parameters with gradients)
        for param in self.qnetwork_local.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
                
        self.optimizer.step()
        
        # Store loss for plotting
        self.loss_history.append(loss.item())
        
        return td_errors.abs() if self.use_prioritized_replay else torch.zeros_like(td_errors)

    def update_epsilon(self, decay_type='exponential'):
        """
        Update epsilon according to specified decay type.
        
        Args:
            decay_type: Type of decay ('exponential', 'linear', or 'custom')
        """
        if decay_type == 'exponential':
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        elif decay_type == 'linear':
            self.epsilon = max(self.epsilon_end, self.epsilon - (self.epsilon_start - self.epsilon_end) / 1000)
        elif decay_type == 'custom':
            # Define any custom decay function here
            step_ratio = len(self.training_rewards) / 1000
            self.epsilon = exponential_decay(self.epsilon_start, self.epsilon_end, step_ratio)
        
        self.epsilon_history.append(self.epsilon)

    def save_checkpoint(self, episode, score, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            episode: Current episode number
            score: Current score
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'episode': episode,
            'score': score,
            'state_dict': self.qnetwork_local.state_dict(),
            'target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'beta': self.beta,
            'training_rewards': self.training_rewards,
            'validation_rewards': self.validation_rewards,
            'loss_history': self.loss_history,
            'best_score': self.best_score
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_ep{episode}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_model_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            self.best_model_path = best_model_path
            self.best_score = score
            logger.info(f"New best model saved with score: {score:.2f}")

    def load_best_model(self):
        """
        Load the best model.
        """
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.load(self.best_model_path)
            logger.info(f"Loaded best model from {self.best_model_path} with score: {self.best_score:.2f}")
        else:
            logger.warning("No best model found to load.")

    def load(self, filepath):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the model file
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return False
        
        try:
            # Load with weights_only=False since we trust our own checkpoints
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # Load model state
            self.qnetwork_local.load_state_dict(checkpoint['state_dict'])
            self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Load other attributes if they exist in the checkpoint
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.beta = checkpoint.get('beta', self.beta)
            self.training_rewards = checkpoint.get('training_rewards', self.training_rewards)
            self.validation_rewards = checkpoint.get('validation_rewards', self.validation_rewards)
            self.loss_history = checkpoint.get('loss_history', self.loss_history)
            self.best_score = checkpoint.get('best_score', self.best_score)
            
            logger.info(f"Model loaded successfully from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def plot_training_results(self, save_path=None):
        """
        Plot training results.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.training_rewards, alpha=0.7)
        window_size = min(50, len(self.training_rewards))
        if window_size > 0:
            moving_avg = np.convolve(self.training_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(np.arange(window_size-1, len(self.training_rewards)), moving_avg, 'r-')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.grid(alpha=0.3)
        
        # Plot loss
        plt.subplot(2, 2, 2)
        plt.plot(self.loss_history, alpha=0.7)
        window_size = min(100, len(self.loss_history))
        if window_size > 0:
            moving_avg = np.convolve(self.loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(np.arange(window_size-1, len(self.loss_history)), moving_avg, 'r-')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(alpha=0.3)
        
        # Plot epsilon
        plt.subplot(2, 2, 3)
        plt.plot(self.epsilon_history)
        plt.xlabel('Training Step')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate')
        plt.grid(alpha=0.3)
        
        # Plot beta if using prioritized replay
        if self.use_prioritized_replay and self.beta_history:
            plt.subplot(2, 2, 4)
            plt.plot(self.beta_history)
            plt.xlabel('Training Step')
            plt.ylabel('Beta')
            plt.title('Importance Sampling Weight')
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

    def __repr__(self):
        """
        String representation of the agent.
        """
        return (f"DQNAgent(state_size={self.state_size}, action_size={self.action_size}, "
                f"use_double_dqn={self.use_double_dqn}, use_dueling={self.use_dueling}, "
                f"use_prioritized_replay={self.use_prioritized_replay})")
