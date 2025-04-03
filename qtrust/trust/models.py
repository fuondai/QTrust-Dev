"""
Neural Network Models for Trust and Reinforcement Learning in QTrust Blockchain

This module contains neural network model implementations used in the QTrust blockchain system:
- QNetwork: Deep Q-learning network with optional dueling architecture for RL-based decision making
- ActorCriticNetwork: Actor-critic architecture for policy-based reinforcement learning
- TrustNetwork: Neural network for evaluating node trustworthiness based on behavioral features

These models provide the foundation for AI-driven trust evaluation and decision making in the
QTrust blockchain environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional

class QNetwork(nn.Module):
    """
    Deep Q-Network with optional Dueling Network architecture
    """
    def __init__(self, state_size: int, action_dims: List[int], seed: int = 42, 
                 hidden_layers: List[int] = [128, 128], dueling: bool = False):
        """
        Initialize Q-Network.

        Args:
            state_size: Dimension of the state space
            action_dims: List of dimensions for each dimension of the action space
            seed: Seed for random generation
            hidden_layers: Size of hidden layers
            dueling: Whether to use Dueling Network architecture
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_dims = action_dims
        self.dueling = dueling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Total number of outputs
        self.total_actions = action_dims[0] if len(action_dims) == 1 else sum(action_dims)
        
        # Feature layer
        layers = []
        in_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        
        self.feature_layer = nn.Sequential(*layers)
        
        if dueling:
            # Dueling Network Architecture - separate state value and action advantage estimation
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_layers[-1], hidden_layers[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_layers[-1] // 2, 1)
            )
            
            # Create advantage network for each action dimension
            self.advantage_streams = nn.ModuleList()
            for action_dim in action_dims:
                self.advantage_streams.append(nn.Sequential(
                    nn.Linear(hidden_layers[-1], hidden_layers[-1] // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_layers[-1] // 2, action_dim)
                ))
        else:
            # Standard architecture - each action dimension has its own output
            self.output_layers = nn.ModuleList()
            for action_dim in action_dims:
                self.output_layers.append(nn.Linear(hidden_layers[-1], action_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the network.

        Args:
            state: Input state, shape (batch_size, state_size)

        Returns:
            Tuple[List[torch.Tensor], Optional[torch.Tensor]]: 
                - List of Q-value tensors, each tensor has shape (batch_size, action_dim)
                - State value (if using dueling network)
        """
        x = self.feature_layer(state)
        
        if self.dueling:
            # Calculate state value
            value = self.value_stream(x)
            
            # Calculate action advantage for each dimension
            advantages = [advantage_stream(x) for advantage_stream in self.advantage_streams]
            
            # Combine value and advantage to get Q-values
            q_values = []
            for advantage in advantages:
                # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a'))) to ensure stability
                q = value + (advantage - advantage.mean(dim=1, keepdim=True))
                q_values.append(q)
            
            return q_values, value
        else:
            # Direct output is Q-values for each action dimension
            q_values = [output_layer(x) for output_layer in self.output_layers]
            return q_values, None

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for Advantage Actor-Critic (A2C/A3C) learning methods
    """
    def __init__(self, state_size: int, action_dims: List[int], 
                 hidden_layers: List[int] = [128, 128], seed: int = 42):
        """
        Initialize Actor-Critic Network.

        Args:
            state_size: Dimension of the state space
            action_dims: List of dimensions for each dimension of the action space
            hidden_layers: Size of hidden layers
            seed: Seed for random generation
        """
        super(ActorCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_dims = action_dims
        
        # Shared feature layers
        layers = []
        in_size = state_size
        
        for hidden_size in hidden_layers[:-1]:  # All except the last layer
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor layer - creates policy
        self.actor_hidden = nn.Sequential(
            nn.Linear(in_size, hidden_layers[-1]),
            nn.ReLU()
        )
        
        # Policy output for each action dimension
        self.actor_heads = nn.ModuleList()
        for action_dim in action_dims:
            self.actor_heads.append(nn.Sequential(
                nn.Linear(hidden_layers[-1], action_dim),
                nn.Softmax(dim=1)
            ))
        
        # Critic layer - evaluates state value
        self.critic = nn.Sequential(
            nn.Linear(in_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: Input state, shape (batch_size, state_size)

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: 
                - List of action probabilities for each dimension
                - State value
        """
        shared_features = self.shared_layers(state)
        
        # Actor - calculate action probabilities
        actor_features = self.actor_hidden(shared_features)
        action_probs = [head(actor_features) for head in self.actor_heads]
        
        # Critic - calculate state value
        state_value = self.critic(shared_features)
        
        return action_probs, state_value

class TrustNetwork(nn.Module):
    """
    Neural network for evaluating trustworthiness of nodes in the blockchain
    """
    def __init__(self, input_size: int, hidden_layers: List[int] = [64, 32], seed: int = 42):
        """
        Initialize trust evaluation network.

        Args:
            input_size: Dimension of input (usually node features)
            hidden_layers: Size of hidden layers
            seed: Seed for random generation
        """
        super(TrustNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Build network
        layers = []
        in_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        
        # Output layer - trust score in the range [0, 1]
        layers.append(nn.Linear(in_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input features, shape (batch_size, input_size)

        Returns:
            torch.Tensor: Trust score, shape (batch_size, 1)
        """
        return self.model(x)
        
    def calculate_trust(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calculate trust score based on features.

        Args:
            features: Node features, shape (batch_size, input_size)

        Returns:
            torch.Tensor: Trust score, shape (batch_size, 1)
        """
        self.eval()
        with torch.no_grad():
            trust_scores = self.forward(features)
        return trust_scores 