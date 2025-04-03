"""
Actor-Critic Agent Module

This module implements an Actor-Critic agent for deep reinforcement learning in the QTrust blockchain system.
The agent combines policy-based (Actor) and value-based (Critic) learning approaches to optimize decision making.

Key features:
- Actor-Critic architecture with advantage calculation
- Experience replay with optional n-step returns
- Entropy regularization for exploration
- Noisy Networks option for exploration
- Distributional RL support
- Caching system for action and value predictions
- Model saving/loading capabilities
- Performance tracking and visualization
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

from qtrust.agents.dqn.networks import ActorNetwork, CriticNetwork
from qtrust.agents.dqn.replay_buffer import ReplayBuffer, NStepReplayBuffer
from qtrust.agents.dqn.utils import (
    soft_update, hard_update, calculate_td_error, calculate_huber_loss,
    exponential_decay, linear_decay, generate_timestamp, create_save_directory,
    plot_learning_curve, format_time, get_device, logger, SAVE_DIR
)
from qtrust.utils.cache import tensor_cache, lru_cache, compute_hash

class ActorCriticAgent:
    """
    Actor-Critic Agent - combines Actor and Critic for deep reinforcement learning.
    """
    def __init__(self, 
                state_size: int, 
                action_size: int, 
                seed: int = 42,
                buffer_size: int = 100000,
                batch_size: int = 64,
                gamma: float = 0.99,
                tau: float = 1e-3,
                actor_lr: float = 1e-4,
                critic_lr: float = 5e-4,
                update_every: int = 4,
                use_n_step: bool = True,
                n_step: int = 3,
                hidden_layers: List[int] = [512, 256],
                device: str = 'auto',
                entropy_coef: float = 0.01,
                value_loss_coef: float = 0.5,
                use_noisy_nets: bool = False,
                clip_gradients: bool = True,
                grad_clip_value: float = 1.0,
                warm_up_steps: int = 1000,
                distributional: bool = False,
                n_atoms: int = 51,
                v_min: float = -10.0,
                v_max: float = 10.0,
                save_dir: str = SAVE_DIR):
        """
        Initialize Actor-Critic Agent.
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            seed: Seed for random values
            buffer_size: Maximum size of replay buffer
            batch_size: Batch size when sampling from replay buffer
            gamma: Discount factor
            tau: Parameter update rate for target network
            actor_lr: Learning rate for Actor
            critic_lr: Learning rate for Critic
            update_every: Update frequency (number of steps)
            use_n_step: Whether to use n-step returns
            n_step: Number of steps for n-step returns
            hidden_layers: Size of hidden layers
            device: Device to use ('cpu', 'cuda', 'auto')
            entropy_coef: Coefficient for entropy regularization
            value_loss_coef: Coefficient for value loss
            use_noisy_nets: Use Noisy Networks for exploration
            clip_gradients: Whether to limit gradients during learning
            grad_clip_value: Gradient clipping value
            warm_up_steps: Number of warm-up steps before starting learning
            distributional: Use Distributional RL for Critic
            n_atoms: Number of atoms in distribution (only used when distributional=True)
            v_min: Minimum Q value (only used when distributional=True)
            v_max: Maximum Q value (only used when distributional=True)
            save_dir: Directory to save model
        """
        self.device = get_device(device)
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.use_noisy_nets = use_noisy_nets
        self.clip_gradients = clip_gradients
        self.grad_clip_value = grad_clip_value
        self.max_grad_norm = grad_clip_value
        self.warm_up_steps = warm_up_steps
        self.save_dir = create_save_directory(save_dir)
        self.distributional = distributional
        
        # Parameters for Distributional RL (if used)
        if distributional:
            self.n_atoms = n_atoms
            self.v_min = v_min
            self.v_max = v_max
            self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)
            self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Initialize Actor network
        self.actor = ActorNetwork(
            state_size, [action_size], hidden_sizes=hidden_layers, noisy=use_noisy_nets
        ).to(self.device)
        
        # Initialize target Actor network
        self.target_actor = ActorNetwork(
            state_size, [action_size], hidden_sizes=hidden_layers, noisy=use_noisy_nets
        ).to(self.device)
        
        # Copy parameters from actor to target_actor
        hard_update(self.target_actor, self.actor)
        
        # Initialize Critic network
        self.critic = CriticNetwork(
            state_size, [action_size], hidden_sizes=hidden_layers, noisy=use_noisy_nets,
            n_atoms=n_atoms if distributional else 1, v_min=v_min, v_max=v_max
        ).to(self.device)
        
        # Create target networks
        self.target_critic = CriticNetwork(
            state_size, [action_size], hidden_sizes=hidden_layers, noisy=use_noisy_nets,
            n_atoms=n_atoms if distributional else 1, v_min=v_min, v_max=v_max
        ).to(self.device)
        
        # Copy parameters from critic to target_critic
        hard_update(self.target_critic, self.critic)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Add learning rate scheduler
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.95)
        
        # Initialize replay memory
        if use_n_step:
            self.memory = NStepReplayBuffer(buffer_size, batch_size, n_step=n_step, gamma=gamma, device=self.device)
        else:
            self.memory = ReplayBuffer(buffer_size, batch_size, device=self.device)
            
        # Learning step counter
        self.t_step = 0
        self.total_steps = 0
        
        # Add variables to track performance
        self.training_rewards = []
        self.validation_rewards = []
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.entropy_history = []
        self.best_score = -float('inf')
        self.best_model_path = None
        self.train_start_time = None

        # Cache for actions and values
        self.action_cache = {}
        self.value_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 10000  # Limit cache size
        
        # Training statistics
        self.train_count = 0
        self.losses = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0
        }

    def step(self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor, done: bool):
        """
        Perform a learning step from experience
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)
        
        # Increment step counter
        self.t_step = (self.t_step + 1) % self.update_every
        self.total_steps += 1
        
        # If enough samples are available, learn
        if len(self.memory) > self.batch_size and self.total_steps > self.warm_up_steps:
            if self.t_step == 0:
                experiences = self.memory.sample()
                self._learn(experiences)
                
                # Update target networks
                soft_update(self.target_actor, self.actor, self.tau)
                soft_update(self.target_critic, self.critic, self.tau)
                
                # Update learning rates
                self.actor_scheduler.step()
                self.critic_scheduler.step()
                
                # Reset noise if using noisy networks
                if self.use_noisy_nets:
                    self.actor.reset_noise()
                    self.critic.reset_noise()
                    self.target_actor.reset_noise()
                    self.target_critic.reset_noise()

    def act(self, state, explore=True, use_target=False):
        """
        Select action based on current policy
        
        Args:
            state: Current state
            explore: Whether to explore or exploit
            use_target: Whether to use target networks (for double learning)
            
        Returns:
            int: Selected action
        """
        # Convert state to tensor and cache key
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state_key = compute_hash(tuple(np.asarray(state).flatten()))
        
        # Check if in cache
        if state_key in self.action_cache:
            self.cache_hits += 1
            return self.action_cache[state_key]
        else:
            self.cache_misses += 1
        
        # Reset noise for Noisy Networks
        if self.use_noisy_nets:
            self.actor.reset_noise()
        
        # Get action probabilities from actor network
        action_probs = self._get_action_probs(state_tensor)
        
        if explore:
            # Sample action from probability distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
        else:
            # Select action with highest probability
            action = torch.argmax(action_probs).item()
        
        # Add to cache if not full
        if len(self.action_cache) < self.max_cache_size:
            self.action_cache[state_key] = action
            
        return action
        
    def _learn(self, experiences: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """
        Update actor and critic networks based on batch of experiences
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones)
        """
        self.train_count += 1
        
        states, actions, rewards, next_states, dones = experiences
        
        # Convert actions to one-hot for critic input
        actions_one_hot = F.one_hot(actions.long(), self.action_size).float()
        
        # Get current Q values from critic
        if self.distributional:
            current_distributions, current_values = self.critic(states, [actions_one_hot])
        else:
            current_values = self.critic(states, [actions_one_hot])
            current_distributions = None
            
        # Print shapes for debugging
        print(f"Current values shape: {current_values[0].shape}")
        
        # Get next actions from target actor
        with torch.no_grad():
            next_action_probs = self.target_actor(next_states)
            next_actions = torch.argmax(next_action_probs[0], dim=1)  # Take first action dimension
            next_actions_one_hot = F.one_hot(next_actions.long(), self.action_size).float()
            
            # Get next Q values from target critic
            if self.distributional:
                next_distributions, next_values = self.target_critic(next_states, [next_actions_one_hot])
                target_values = rewards + (1 - dones) * self.gamma * next_values[0]
                critic_loss = self.calculate_distributional_loss(current_distributions[0], target_values, next_distributions[0])
            else:
                next_values = self.target_critic(next_states, [next_actions_one_hot])
                # Print shapes for debugging
                print(f"Next values shape: {next_values[0].shape}")
                print(f"Rewards shape: {rewards.shape}")
                print(f"Dones shape: {dones.shape}")
                
                # Ensure rewards and dones have correct shape
                rewards = rewards.view(-1, 1)
                dones = dones.view(-1, 1)
                
                target_values = rewards + (1 - dones) * self.gamma * next_values[0]
                # Print target values shape
                print(f"Target values shape: {target_values.shape}")
                
                # Ensure target_values has same shape as current_values[0]
                target_values = target_values.view(-1, 1)  # [batch_size, 1]
                
        # Ensure current_values requires grad
        current_values = [v.detach().requires_grad_(True) for v in current_values]
        critic_loss = F.mse_loss(current_values[0], target_values)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Update actor
        action_probs = self.actor(states)
        actions_pred = torch.argmax(action_probs[0], dim=1)  # Take first action dimension
        actions_pred_one_hot = F.one_hot(actions_pred.long(), self.action_size).float()
        actor_values = self.critic(states, [actions_pred_one_hot])
        if self.distributional:
            actor_values = actor_values[1][0]  # Use mean values for actor update
        else:
            actor_values = actor_values[0]
        actor_loss = -actor_values.mean()
        
        # Add entropy regularization if enabled
        if self.entropy_coef > 0:
            probs = action_probs[0]  # Take first action dimension
            log_probs = torch.log(probs + 1e-10)  # Add small constant for numerical stability
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            actor_loss -= self.entropy_coef * entropy
        else:
            entropy = torch.tensor(0.0)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Record losses
        self.losses["critic_loss"] = critic_loss.item()
        self.losses["actor_loss"] = actor_loss.item()
        self.losses["entropy"] = entropy.item()
        
        # Update loss histories
        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())
        
        return self.losses

    def clear_cache(self):
        """Clear action and value caches."""
        self.action_cache.clear()
        self.value_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    @tensor_cache
    def _get_action_probs(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities from actor network
        
        Args:
            state_tensor: Input state tensor
            
        Returns:
            torch.Tensor: Action probabilities
        """
        action_probs = self.actor(state_tensor)[0]
        return action_probs
    
    @tensor_cache
    def _get_state_value(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get state value from critic network
        
        Args:
            state_tensor: Input state tensor
            
        Returns:
            torch.Tensor: State value
        """
        # Create dummy action for critic
        action = torch.zeros(state_tensor.size(0), self.action_size).to(self.device)
        value, _ = self.critic(state_tensor, [action])
        return value[0]
    
    @tensor_cache
    def _get_target_value(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get state value from target critic network
        
        Args:
            state_tensor: Input state tensor
            
        Returns:
            torch.Tensor: Target state value
        """
        # Create dummy action for critic
        action = torch.zeros(state_tensor.size(0), self.action_size).to(self.device)
        value, _ = self.critic_target(state_tensor, [action])
        return value[0]
    
    def save(self, filepath: str) -> bool:
        """
        Save agent state to file.
        
        Args:
            filepath: Path to save file
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'target_actor_state_dict': self.target_actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'target_critic_state_dict': self.target_critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
                'critic_scheduler_state_dict': self.critic_scheduler.state_dict(),
                'losses': self.losses,
                'train_count': self.train_count,
                'total_steps': self.total_steps,
                'best_score': self.best_score
            }, filepath)
            logger.info(f"Model saved to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load(self, filepath: str) -> bool:
        """
        Load agent state from file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"No file found at {filepath}")
                return False
            
            checkpoint = torch.load(filepath)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
            self.losses = checkpoint['losses']
            self.train_count = checkpoint['train_count']
            self.total_steps = checkpoint['total_steps']
            self.best_score = checkpoint['best_score']
            
            logger.info(f"Model loaded from: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get agent performance statistics
        
        Returns:
            Dict: Dictionary with performance statistics
        """
        stats = {
            'actor_loss': self.actor_loss_history[-1] if self.actor_loss_history else float('nan'),
            'critic_loss': self.critic_loss_history[-1] if self.critic_loss_history else float('nan'),
            'entropy': self.entropy_history[-1] if self.entropy_history else float('nan'),
            'avg_actor_loss': np.mean(self.actor_loss_history[-100:]) if self.actor_loss_history else float('nan'),
            'avg_critic_loss': np.mean(self.critic_loss_history[-100:]) if self.critic_loss_history else float('nan'),
            'avg_entropy': np.mean(self.entropy_history[-100:]) if self.entropy_history else float('nan'),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'total_steps': self.total_steps,
            'train_count': self.train_count
        }
        return stats 