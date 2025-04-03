"""
Rainbow DQN Agent Implementation

A complete implementation of Rainbow DQN that combines multiple improvements to DQN:
- Double DQN for more stable learning
- Dueling Network Architecture for better value estimation
- Prioritized Experience Replay for efficient learning
- Noisy Networks for exploration
- Distributional RL (Categorical DQN) for value distribution learning
- Multi-step learning for better credit assignment

Features:
- Automatic hyperparameter tuning
- Learning rate scheduling
- Gradient clipping
- Warm-up period
- Model checkpointing
- Performance tracking
- Caching system for improved efficiency
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

from qtrust.agents.dqn.networks import CategoricalQNetwork
from qtrust.agents.dqn.replay_buffer import NStepPrioritizedReplayBuffer
from qtrust.agents.dqn.utils import (
    soft_update, hard_update, calculate_td_error, calculate_huber_loss,
    exponential_decay, linear_decay, generate_timestamp, create_save_directory,
    plot_learning_curve, format_time, get_device, logger, SAVE_DIR
)
from qtrust.utils.cache import tensor_cache, lru_cache, compute_hash

class RainbowDQNAgent:
    """
    Rainbow DQN Agent - Complete implementation with all Rainbow DQN improvements.
    """
    def __init__(self, 
                state_size: int, 
                action_size: int, 
                seed: int = 42,
                buffer_size: int = 100000,
                batch_size: int = 64,
                gamma: float = 0.99,
                tau: float = 1e-3,
                learning_rate: float = 5e-4,
                update_every: int = 4,
                n_step: int = 3,
                alpha: float = 0.6,
                beta_start: float = 0.4,
                n_atoms: int = 51,
                v_min: float = -10.0,
                v_max: float = 10.0,
                hidden_layers: List[int] = [512, 256],
                device: str = 'auto',
                min_epsilon: float = 0.01,
                epsilon_decay: float = 0.995,
                clip_gradients: bool = True,
                grad_clip_value: float = 1.0,
                warm_up_steps: int = 1000,
                save_dir: str = SAVE_DIR):
        """
        Initialize Rainbow DQN Agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            seed: Random seed for reproducibility
            buffer_size: Maximum size of replay buffer
            batch_size: Size of training batch
            gamma: Discount factor
            tau: Soft update rate for target network
            learning_rate: Learning rate for optimizer
            update_every: Network update frequency
            n_step: Number of steps for n-step returns
            alpha: Priority exponent for PER
            beta_start: Initial beta for importance sampling
            n_atoms: Number of atoms for categorical DQN
            v_min: Minimum value for support
            v_max: Maximum value for support
            hidden_layers: Sizes of hidden layers
            device: Computing device ('cpu', 'cuda', 'auto')
            min_epsilon: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            clip_gradients: Whether to clip gradients
            grad_clip_value: Maximum gradient value
            warm_up_steps: Steps before starting learning
            save_dir: Directory for saving models
        """
        self.device = get_device(device)
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.n_step = n_step
        self.clip_gradients = clip_gradients
        self.grad_clip_value = grad_clip_value
        self.warm_up_steps = warm_up_steps
        self.save_dir = create_save_directory(save_dir)
        
        # Parameters for Categorical DQN
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Initialize Q-Networks - Always use Noisy Networks and Dueling in Rainbow
        self.qnetwork_local = CategoricalQNetwork(
            state_size, [action_size], n_atoms=n_atoms, v_min=v_min, v_max=v_max,
            hidden_sizes=hidden_layers, noisy=True
        ).to(self.device)
        
        self.qnetwork_target = CategoricalQNetwork(
            state_size, [action_size], n_atoms=n_atoms, v_min=v_min, v_max=v_max,
            hidden_sizes=hidden_layers, noisy=True
        ).to(self.device)
        
        # Copy parameters from local to target
        hard_update(self.qnetwork_target, self.qnetwork_local)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Initialize replay memory - Always use N-step and PER in Rainbow
        self.memory = NStepPrioritizedReplayBuffer(
            buffer_size, batch_size, n_step=n_step, alpha=alpha, beta_start=beta_start,
            gamma=gamma, device=self.device
        )
            
        # Learning step counter
        self.t_step = 0
        self.total_steps = 0
        
        # Initialize epsilon for exploration
        self.eps_start = 1.0
        self.eps_end = min_epsilon
        self.eps_decay = epsilon_decay
        self.epsilon = self.eps_start
        
        # Add variables for tracking performance
        self.training_rewards = []
        self.validation_rewards = []
        self.loss_history = []
        self.best_score = -float('inf')
        self.best_model_path = None
        self.train_start_time = None

    def step(self, state, action, reward, next_state, done):
        """
        Take a step in the environment and learn from it.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Done flag
        """
        self.total_steps += 1
        
        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        if self.total_steps % self.update_every == 0 and len(self.memory) > self.batch_size:
            # Get experiences from memory
            one_step, n_step, indices, weights = self.memory.sample()
            states, actions, rewards, next_states, dones = one_step
            
            # Learn from experiences
            self._learn((states, actions, rewards, next_states, dones, indices, weights))
    
    def act(self, state, eps=None):
        """
        Select an action based on current policy.
        
        Args:
            state: Current state
            eps: Epsilon for epsilon-greedy, defaults to None (uses internal epsilon)
            
        Returns:
            int: Selected action
        """
        # Use provided epsilon or current epsilon
        epsilon = eps if eps is not None else self.epsilon
        
        # Explore with probability epsilon
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        
        # Reset noise for Noisy Networks - ensure it's always called
        self.qnetwork_local.reset_noise()
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(np.asarray(state)).unsqueeze(0).to(self.device)
        
        # Disable training mode for network
        self.qnetwork_local.eval()
        with torch.no_grad():
            # Get Q distribution
            action_distributions, _ = self.qnetwork_local(state_tensor)
            action_dist = action_distributions[0]  # [batch_size, action_size, n_atoms]
            
            # Calculate expected value
            expected_q = torch.sum(action_dist * self.support.unsqueeze(0).unsqueeze(0), dim=2)
            
        # Re-enable training mode
        self.qnetwork_local.train()
        
        # Return best action
        return np.argmax(expected_q.cpu().data.numpy())
    
    @tensor_cache
    def _calculate_target_distribution(self, rewards, next_dist, dones, batch_idx):
        """
        Calculate target distribution for Distributional RL.
        Separate function for caching.
        
        Args:
            rewards: Rewards for transition
            next_dist: Q distribution of next state
            dones: Done flag
            batch_idx: Index of current batch being processed
            
        Returns:
            torch.Tensor: Target distribution
        """
        if dones:
            # When done, target distribution is delta function at reward
            target_idx = torch.clamp(torch.floor((rewards - self.v_min) / self.delta_z), 0, self.n_atoms - 1).long()
            target_dist = torch.zeros_like(next_dist)
            target_dist[target_idx] = 1.0
            return target_dist
        else:
            # Project Bellman distribution
            # Tz_j = r + gamma * z_j
            Tz = rewards + self.gamma * self.support
            
            # Calculate projection
            Tz = torch.clamp(Tz, self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Ensure tensor
            target_dist = torch.zeros_like(next_dist)
            
            # Distribute probabilities safely
            for j in range(self.n_atoms):
                # Ensure indices are within valid range
                l_idx = min(l[j].item(), self.n_atoms - 1)
                u_idx = min(u[j].item(), self.n_atoms - 1)
                
                # Get value from next_dist safely
                next_prob = next_dist[j].item() if j < next_dist.size(0) else 0
                
                # Update target_dist
                target_dist[l_idx] += next_prob * (u[j] - b[j]).item()
                target_dist[u_idx] += next_prob * (b[j] - l[j]).item()
            
            return target_dist
    
    def _learn(self, experiences):
        """
        Update Q values based on batch of experiences.
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        states, actions, rewards, next_states, dones, indices, weights = experiences
        
        # Calculate Q distribution for current state
        action_distributions, _ = self.qnetwork_local(states)
        action_dist = action_distributions[0]  # [batch_size, action_size, n_atoms]
        
        # Get distribution for taken actions
        action_indices = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, self.n_atoms)  # [batch_size, 1, n_atoms]
        current_dist = action_dist.gather(1, action_indices).squeeze(1)  # [batch_size, n_atoms]
        
        # Calculate target Q distribution - Double DQN
        next_action_distributions, _ = self.qnetwork_local(next_states)
        next_actions = next_action_distributions[0].sum(dim=2).argmax(dim=1)  # Get actions from local network
        
        next_dist_target, _ = self.qnetwork_target(next_states)
        next_dist = next_dist_target[0].gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms))
        next_dist = next_dist.squeeze(1)  # [batch_size, n_atoms]
        
        # Calculate target distribution - Distributional RL
        target_dist = torch.zeros_like(current_dist)
        
        # Use cache to calculate target distribution
        for idx in range(self.batch_size):
            # Use cached version of target distribution calculation
            target_dist[idx] = self._calculate_target_distribution(
                rewards[idx], next_dist[idx], dones[idx], idx
            )
        
        # Calculate loss - Cross entropy loss
        log_current_dist = current_dist.clamp(1e-10, 1.0).log()
        loss = -(target_dist * log_current_dist).sum(1)
        
        # Calculate TD errors for PER
        with torch.no_grad():
            td_errors = torch.abs(target_dist - current_dist).mean(1).cpu().numpy()
        
        # Apply importance sampling weights from PER
        loss = (loss * weights).mean()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors)
        
        # Update beta
        self.memory.update_beta(self.t_step, self.total_steps)
        
        # Optimize loss
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients if needed
        if self.clip_gradients:
            for param in self.qnetwork_local.parameters():
                param.grad.data.clamp_(-self.grad_clip_value, self.grad_clip_value)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update target network
        soft_update(self.qnetwork_target, self.qnetwork_local, self.tau)
        
        # Store loss value
        self.loss_history.append(loss.item())
    
    def save(self, filepath=None, episode=None):
        """
        Save the model.
        
        Args:
            filepath: Path to save model, if None uses default path
            episode: Current episode number, used for filename
            
        Returns:
            str: Path where model was saved
        """
        if filepath is None:
            timestamp = generate_timestamp()
            if episode is not None:
                filepath = os.path.join(self.save_dir, f"rainbow_dqn_ep{episode}_{timestamp}.pth")
            else:
                filepath = os.path.join(self.save_dir, f"rainbow_dqn_{timestamp}.pth")
        
        torch.save({
            'local_state_dict': self.qnetwork_local.state_dict(),
            'target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_history': self.loss_history,
            'training_rewards': self.training_rewards,
            'validation_rewards': self.validation_rewards,
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'hyperparams': {
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'tau': self.tau,
                'state_size': self.state_size,
                'action_size': self.action_size,
                'n_atoms': self.n_atoms,
                'v_min': self.v_min,
                'v_max': self.v_max,
                'n_step': self.n_step
            }
        }, filepath)
        
        logger.info(f"Model saved at: {filepath}")
        return filepath
    
    def load(self, filepath):
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
                
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])
            self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            if 'loss_history' in checkpoint:
                self.loss_history = checkpoint['loss_history']
                
            if 'training_rewards' in checkpoint:
                self.training_rewards = checkpoint['training_rewards']
                
            if 'validation_rewards' in checkpoint:
                self.validation_rewards = checkpoint['validation_rewards']
                
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
                
            if 'total_steps' in checkpoint:
                self.total_steps = checkpoint['total_steps']
            
            logger.info(f"Model loaded from: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False 