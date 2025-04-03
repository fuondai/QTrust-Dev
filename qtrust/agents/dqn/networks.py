"""
Neural Network Architectures for Deep Reinforcement Learning

This module implements various neural network architectures used in deep reinforcement learning:

Core Components:
- NoisyLinear: Linear layer with parametric noise for efficient exploration
- ResidualBlock: Residual connections with normalization for stable training

Network Architectures:
- QNetwork: Basic DQN network with attention and residual connections
- DuelingQNetwork: Separates state value and action advantages
- CategoricalQNetwork: Distributional RL with categorical value distribution
- ActorNetwork: Policy network for actor-critic methods
- CriticNetwork: Value network for actor-critic methods

Features:
- Attention mechanisms for feature importance
- Residual connections for deep architectures
- Layer normalization for stable training
- Dropout for regularization
- Support for both deterministic and noisy networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Optional, Union

class NoisyLinear(nn.Module):
    """
    NoisyLinear layer for efficient exploration.
    Replaces epsilon-greedy with noisy parameters.
    """
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Initialize the NoisyLinear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            std_init: Initial standard deviation for noise parameters
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Initialize layer parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameter values."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        """Generate noise from factorized Gaussian distribution."""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def reset_noise(self):
        """Regenerate noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        """Forward pass."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class ResidualBlock(nn.Module):
    """
    Residual block with normalization to improve convergence.
    """
    def __init__(self, in_features: int, out_features: int, noisy: bool = False, use_layer_norm: bool = False):
        super(ResidualBlock, self).__init__()
        
        # Choose layer based on noisy parameter
        linear_layer = NoisyLinear if noisy else nn.Linear
        
        # Main layer
        self.main = nn.Sequential(
            linear_layer(in_features, out_features),
            nn.LayerNorm(out_features) if use_layer_norm else nn.BatchNorm1d(out_features),
            nn.ReLU(),
            linear_layer(out_features, out_features),
            nn.LayerNorm(out_features) if use_layer_norm else nn.BatchNorm1d(out_features)
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                linear_layer(in_features, out_features),
                nn.LayerNorm(out_features) if use_layer_norm else nn.BatchNorm1d(out_features)
            )
        
        self.relu = nn.ReLU()
        self.noisy = noisy
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block."""
        identity = self.shortcut(x)
        out = self.main(x)
        out += identity
        out = self.relu(out)
        return out
    
    def reset_noise(self):
        """Reset noise for NoisyLinear layers in the block."""
        if not self.noisy:
            return
            
        # Reset noise in main block
        for module in self.main:
            if isinstance(module, NoisyLinear):
                module.reset_noise()
        
        # Reset noise in shortcut
        for module in self.shortcut:
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class QNetwork(nn.Module):
    """
    Neural network for Deep Q-Network with residual connections.
    """
    def __init__(self, state_size: int, action_size: Union[int, List[int]], hidden_size: Union[int, List[int]] = 64, seed: int = None, noisy: bool = False):
        """
        Initialize the Q network.
        
        Args:
            state_size: Size of the state space
            action_size: Size of the action space (scalar or list)
            hidden_size: Size of hidden layers (can be int or list)
            seed: Random seed for initialization
            noisy: Use NoisyLinear instead of Linear
        """
        super(QNetwork, self).__init__()
        
        self.state_size = state_size
        self.noisy = noisy
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Handle both scalar and list action dimensions
        if isinstance(action_size, int):
            self.action_dim = [action_size]
        else:
            self.action_dim = action_size
            
        self.action_dim_product = np.prod(self.action_dim)
        
        # Convert hidden_size to hidden_sizes list if it's an int
        if isinstance(hidden_size, int):
            self.hidden_sizes = [hidden_size, hidden_size]
        else:
            self.hidden_sizes = hidden_size
        
        # Choose layer based on noisy parameter
        linear_layer = NoisyLinear if noisy else nn.Linear
        
        # Input embedding layer to encode the state
        self.input_layer = linear_layer(state_size, self.hidden_sizes[0])
        
        # Replace BatchNorm1d with LayerNorm for capability with batch size = 1
        self.input_norm = nn.LayerNorm(self.hidden_sizes[0])
        
        # Residual blocks - update ResidualBlock below
        self.res_blocks = nn.ModuleList()
        for i in range(len(self.hidden_sizes) - 1):
            self.res_blocks.append(
                ResidualBlock(self.hidden_sizes[i], self.hidden_sizes[i+1], noisy=noisy, use_layer_norm=True)
            )
        
        # Output layer for each action dimension
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                linear_layer(self.hidden_sizes[-1], self.hidden_sizes[-1] // 2),
                nn.ReLU(),
                linear_layer(self.hidden_sizes[-1] // 2, dim)
            ) for dim in self.action_dim
        ])
        
        # State value assessment (V(s))
        self.value_stream = nn.Sequential(
            linear_layer(self.hidden_sizes[-1], self.hidden_sizes[-1] // 2),
            nn.ReLU(),
            linear_layer(self.hidden_sizes[-1] // 2, 1)
        )
        
        # Attention layer for important features
        self.attention = nn.Sequential(
            linear_layer(self.hidden_sizes[-1], self.hidden_sizes[-1]),
            nn.Sigmoid()
        )
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            List of action values for each action dimension or single tensor
        """
        # Process through input layer
        x = self.input_layer(state)
        x = self.input_norm(x)
        x = F.relu(x)
        
        # Process through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Apply attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Get state value
        state_value = self.value_stream(x)
        
        # Get advantage for each action dimension
        action_values = []
        
        for i, output_layer in enumerate(self.output_layers):
            action_val = output_layer(x)
            action_values.append(action_val)
        
        # If there's only one action dimension, return it directly
        if len(action_values) == 1:
            return action_values[0]
        
        # Otherwise, return list of action values
        return action_values
    
    def reset_noise(self):
        """Reset noise for all NoisyLinear layers."""
        if not self.noisy:
            return
            
        # Reset noise in input layer if it's NoisyLinear
        if isinstance(self.input_layer, NoisyLinear):
            self.input_layer.reset_noise()
        
        # Reset noise in residual blocks
        for res_block in self.res_blocks:
            res_block.reset_noise()
        
        # Reset noise in output layers
        for output_layer in self.output_layers:
            for module in output_layer:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
        
        # Reset noise in value stream
        for module in self.value_stream:
            if isinstance(module, NoisyLinear):
                module.reset_noise()
        
        # Reset noise in attention
        for module in self.attention:
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture that separates state value and action advantages.
    """
    def __init__(self, state_size: int, action_size: Union[int, List[int]], 
                 hidden_size: Union[int, List[int]] = 64, seed: int = None, noisy: bool = False):
        """
        Initialize the Dueling Q-Network.
        
        Args:
            state_size: Size of the state space
            action_size: Size of the action space (scalar or list)
            hidden_size: Size of hidden layers (can be int or list)
            seed: Random seed for initialization
            noisy: Use NoisyLinear instead of Linear
        """
        super(DuelingQNetwork, self).__init__()
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            
        # Handle both scalar and list action dimensions
        if isinstance(action_size, int):
            self.action_size = action_size
            self.action_dim = [action_size]
        else:
            self.action_size = np.prod(action_size)
            self.action_dim = action_size
            
        self.noisy = noisy
        
        # Convert hidden_size to hidden_layers list if it's an int
        if isinstance(hidden_size, int):
            self.hidden_layers = [hidden_size, hidden_size]
        else:
            self.hidden_layers = hidden_size
        
        # Choose layer based on noisy parameter
        linear_layer = NoisyLinear if noisy else nn.Linear
        
        # Feature extraction layers
        layers = []
        prev_size = state_size
        for h_size in self.hidden_layers:
            layers.append(linear_layer(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        
        self.feature_layer = nn.Sequential(*layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            linear_layer(self.hidden_layers[-1], self.hidden_layers[-1] // 2),
            nn.ReLU(),
            linear_layer(self.hidden_layers[-1] // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            linear_layer(self.hidden_layers[-1], self.hidden_layers[-1] // 2),
            nn.ReLU(),
            linear_layer(self.hidden_layers[-1] // 2, self.action_size)
        )
    
    def forward(self, state):
        """
        Forward pass to build dueling Q network.
        
        Args:
            state: Current state
            
        Returns:
            Q-values for all actions
        """
        features = self.feature_layer(state)
        
        # Calculate state value V(s)
        values = self.value_stream(features)
        
        # Calculate advantages A(s,a)
        advantages = self.advantage_stream(features)
        
        # Combine to get Q(s,a) = V(s) + A(s,a) - mean(A(s,*))
        # Subtraction of mean advantage ensures identifiability
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class CategoricalQNetwork(nn.Module):
    """
    Neural network for Distributional DQN (C51) with residual connections.
    Uses probability distribution instead of a single Q value.
    """
    def __init__(self, state_size: int, action_dim: List[int], n_atoms: int = 51,
                 v_min: float = -10.0, v_max: float = 10.0, 
                 hidden_sizes: List[int] = [256, 256], noisy: bool = False):
        """
        Initialize the Distributional Q network.
        
        Args:
            state_size: Size of the state space
            action_dim: List of dimensions of the action space
            n_atoms: Number of atoms in the distribution
            v_min: Minimum Q value
            v_max: Maximum Q value
            hidden_sizes: Size of hidden layers
            noisy: Use NoisyLinear instead of Linear
        """
        super(CategoricalQNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_dim = action_dim
        self.action_dim_product = np.prod(action_dim)
        self.noisy = noisy
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Choose layer based on noisy parameter
        linear_layer = NoisyLinear if noisy else nn.Linear
        
        # Input embedding layer to encode the state
        self.input_layer = linear_layer(state_size, hidden_sizes[0])
        self.input_norm = nn.LayerNorm(hidden_sizes[0])
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_sizes[i], hidden_sizes[i+1], noisy=noisy, use_layer_norm=True)
            )
        
        # State value assessment (V(s))
        self.value_stream = nn.Sequential(
            linear_layer(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            linear_layer(hidden_sizes[-1] // 2, n_atoms)
        )
        
        # Output layer for each action × number of atoms
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                linear_layer(hidden_sizes[-1], hidden_sizes[-1] // 2),
                nn.ReLU(),
                linear_layer(hidden_sizes[-1] // 2, dim * n_atoms)
            ) for dim in action_dim
        ])
        
        # Attention layer
        self.attention = nn.Sequential(
            linear_layer(hidden_sizes[-1], hidden_sizes[-1]),
            nn.Sigmoid()
        )
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: 
                - List of probability distributions for each action
                - State value (V(s))
        """
        batch_size = state.size(0)
        
        # Encode input - use LayerNorm
        x = F.relu(self.input_norm(self.input_layer(state)))
        
        # Pass through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Apply attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Apply dropout - disable during evaluation
        if self.training:
            x = self.dropout(x)
            
        # Calculate state value (V(s))
        state_value = self.value_stream(x)  # [batch_size, n_atoms]
        
        # Calculate advantage of each action (A(s,a))
        action_distributions = []
        
        for i, layer in enumerate(self.output_layers):
            advantages = layer(x)  # [batch_size, action_dim * n_atoms]
            advantages = advantages.view(batch_size, self.action_dim[i], self.n_atoms)
            
            # Zero-center advantages within each atom
            advantages = advantages - advantages.mean(dim=1, keepdim=True)
            
            # Combine value and advantages for each atom
            q_atoms = state_value.unsqueeze(1) + advantages
            
            # Apply softmax over atoms to get probabilities
            q_dist = F.softmax(q_atoms, dim=2)
            
            action_distributions.append(q_dist)
            
        return action_distributions, state_value
    
    def reset_noise(self):
        """Reset noise for all NoisyLinear layers."""
        if not self.noisy:
            return
            
        if isinstance(self.input_layer, NoisyLinear):
            self.input_layer.reset_noise()
            
        for res_block in self.res_blocks:
            res_block.reset_noise()
            
        for output_layer in self.output_layers:
            for module in output_layer:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
                    
        for module in self.value_stream:
            if isinstance(module, NoisyLinear):
                module.reset_noise()
                
        for module in self.attention:
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class ActorNetwork(nn.Module):
    """
    Actor Network for Actor-Critic architecture.
    Outputs probability distribution over action space.
    """
    def __init__(self, state_size: int, action_dim: List[int], 
                 hidden_sizes: List[int] = [256, 256], noisy: bool = False):
        """
        Initialize the Actor network.
        
        Args:
            state_size: Size of the state space
            action_dim: List of dimensions of the action space
            hidden_sizes: Size of hidden layers
            noisy: Use NoisyLinear instead of Linear
        """
        super(ActorNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_dim = action_dim
        self.noisy = noisy
        self.hidden_sizes = hidden_sizes
        
        # Choose layer based on noisy parameter
        linear_layer = NoisyLinear if noisy else nn.Linear
        
        # Input embedding layer
        self.input_layer = linear_layer(state_size, hidden_sizes[0])
        self.input_norm = nn.LayerNorm(hidden_sizes[0])
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_sizes[i], hidden_sizes[i+1], noisy=noisy, use_layer_norm=True)
            )
        
        # Policy output for each action dimension
        self.policy_layers = nn.ModuleList([
            nn.Sequential(
                linear_layer(hidden_sizes[-1], hidden_sizes[-1] // 2),
                nn.ReLU(),
                linear_layer(hidden_sizes[-1] // 2, dim)
            ) for dim in action_dim
        ])
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the Actor network.
        
        Args:
            state: Input state tensor
            
        Returns:
            List[torch.Tensor]: List of action probability distributions
        """
        # Encode input
        x = F.relu(self.input_norm(self.input_layer(state)))
        
        # Pass through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Apply dropout
        if self.training:
            x = self.dropout(x)
        
        # Calculate action probability distributions
        action_probs = []
        for layer in self.policy_layers:
            logits = layer(x)
            # Apply softmax to get probability distribution
            probs = F.softmax(logits, dim=-1)
            action_probs.append(probs)
        
        return action_probs
    
    def reset_noise(self):
        """Reset noise for all NoisyLinear layers."""
        if not self.noisy:
            return
            
        if isinstance(self.input_layer, NoisyLinear):
            self.input_layer.reset_noise()
            
        for res_block in self.res_blocks:
            res_block.reset_noise()
            
        for policy_layer in self.policy_layers:
            for module in policy_layer:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class CriticNetwork(nn.Module):
    """
    Critic Network for Actor-Critic architecture.
    Evaluates state-action pairs.
    """
    def __init__(self, state_size: int, action_dim: List[int],
                 hidden_sizes: List[int] = [256, 256], noisy: bool = False,
                 n_atoms: int = 1, v_min: float = -10.0, v_max: float = 10.0):
        """
        Initialize the Critic network.
        
        Args:
            state_size: Size of the state space
            action_dim: List of dimensions of the action space
            hidden_sizes: Size of hidden layers
            noisy: Use NoisyLinear instead of Linear
            n_atoms: Number of atoms in the distribution
            v_min: Minimum Q value
            v_max: Maximum Q value
        """
        super(CriticNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_dim = action_dim
        self.noisy = noisy
        self.distributional = n_atoms > 1
        self.n_atoms = n_atoms
        self.hidden_sizes = hidden_sizes
        
        if self.distributional:
            self.v_min = v_min
            self.v_max = v_max
            self.support = torch.linspace(v_min, v_max, n_atoms)
            self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Choose layer based on noisy parameter
        linear_layer = NoisyLinear if noisy else nn.Linear
        
        # Input embedding layer
        self.input_layer = linear_layer(state_size, hidden_sizes[0])
        self.input_norm = nn.LayerNorm(hidden_sizes[0])
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_sizes[i], hidden_sizes[i+1], noisy=noisy, use_layer_norm=True)
            )
        
        # Q value output for each state-action pair
        self.q_layers = nn.ModuleList([
            nn.Sequential(
                linear_layer(hidden_sizes[-1] + dim, hidden_sizes[-1] // 2),  # Concatenate state features and action
                nn.ReLU(),
                linear_layer(hidden_sizes[-1] // 2, n_atoms),
                nn.Linear(n_atoms, 1) if not self.distributional else nn.Identity()  # Ensure output is [batch_size, 1]
            ) for dim in action_dim
        ])
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state: torch.Tensor, actions: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        Forward pass through the Critic network.
        
        Args:
            state: Input state tensor
            actions: List of action tensors that were taken
            
        Returns:
            Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]: 
                - If distributional: (list of probability distributions, list of mean values)
                - If not: list of Q values
        """
        # Ensure input tensors require grad
        state = state.detach().requires_grad_(True)
        actions = [a.detach().requires_grad_(True) for a in actions]
        
        # Encode input
        x = F.relu(self.input_norm(self.input_layer(state)))
        
        # Pass through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Apply dropout
        if self.training:
            x = self.dropout(x)
        
        # Calculate Q values
        q_values = []
        q_distributions = []
        
        for i, (layer, action) in enumerate(zip(self.q_layers, actions)):
            # Combine state features and action
            sa_features = torch.cat([x, action], dim=-1)
            q_output = layer(sa_features)
            
            if self.distributional:
                # Apply softmax to get probability distribution
                q_dist = F.softmax(q_output, dim=-1)
                q_distributions.append(q_dist)
                
                # Calculate mean Q value
                support = self.support.to(state.device)
                q_val = torch.sum(q_dist * support, dim=-1, keepdim=True)
                q_values.append(q_val)
            else:
                q_values.append(q_output)
        
        if self.distributional:
            return q_distributions, q_values
        else:
            return q_values
    
    def reset_noise(self):
        """Reset noise for all NoisyLinear layers."""
        if not self.noisy:
            return
            
        if isinstance(self.input_layer, NoisyLinear):
            self.input_layer.reset_noise()
            
        for res_block in self.res_blocks:
            res_block.reset_noise()
            
        for q_layer in self.q_layers:
            for module in q_layer:
                if isinstance(module, NoisyLinear):
                    module.reset_noise() 