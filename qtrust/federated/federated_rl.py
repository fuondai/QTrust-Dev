"""
Federated Reinforcement Learning Module for QTrust Blockchain

This module implements a Federated Reinforcement Learning (FRL) system for the QTrust blockchain.
It extends the base Federated Learning module to support RL tasks, allowing agents to share
learned policies across shards while maintaining privacy and efficiency.

Key features:
- Federated client for RL tasks (FRLClient)
- Federated Reinforcement Learning system (FederatedRL)
- Support for differential privacy
- Personalization options for client models
- Optimized model aggregation methods
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict
import copy
import random

from qtrust.agents.dqn.agent import DQNAgent
from qtrust.agents.dqn.networks import QNetwork
from qtrust.federated.federated_learning import FederatedLearning, FederatedClient
from qtrust.federated.model_aggregation import ModelAggregationManager

class FRLClient(FederatedClient):
    """
    Represents a client in Federated Reinforcement Learning.
    Extends FederatedClient to support RL tasks.
    """
    def __init__(self, 
                 client_id: int, 
                 agent: DQNAgent,
                 model: nn.Module, 
                 optimizer_class: torch.optim.Optimizer = optim.Adam,
                 learning_rate: float = 0.001,
                 local_epochs: int = 5,
                 batch_size: int = 32,
                 trust_score: float = 0.7,
                 device: str = 'cpu'):
        """
        Initialize a client for Federated Reinforcement Learning.
        
        Args:
            client_id: Unique ID of the client
            agent: DQNAgent used for learning process
            model: Model to be learned
            optimizer_class: Optimizer class used for learning process
            learning_rate: Learning rate
            local_epochs: Local training epochs
            batch_size: Batch size
            trust_score: Trust score of the client
            device: Device to use (CPU or GPU)
        """
        super(FRLClient, self).__init__(
            client_id, model, optimizer_class, learning_rate, 
            local_epochs, batch_size, trust_score, device
        )
        
        self.agent = agent
        
        # Local experience data
        self.local_experiences = []
        self.environment = None
        
    def set_environment(self, env):
        """
        Set the environment for the client.
        
        Args:
            env: RL environment
        """
        self.environment = env
    
    def collect_experiences(self, num_steps: int = 1000, epsilon: float = 0.1):
        """
        Collect experiences from the environment.
        
        Args:
            num_steps: Number of steps to interact with the environment
            epsilon: Epsilon value for epsilon-greedy policy
            
        Returns:
            List: Collected experiences
        """
        if self.environment is None:
            raise ValueError("Environment not set for client!")
        
        experiences = []
        state = self.environment.reset()
        
        for _ in range(num_steps):
            action = self.agent.act(state, eps=epsilon)
            next_state, reward, done, _ = self.environment.step(action)
            
            # Store experience
            experiences.append((state, action, reward, next_state, done))
            
            # Update state
            state = next_state
            
            if done:
                state = self.environment.reset()
        
        self.local_experiences.extend(experiences)
        return experiences
    
    def update_agent_model(self):
        """
        Update agent's model with learned model.
        """
        # Update qnetwork_local of agent from model of client
        self.agent.qnetwork_local.load_state_dict(self.model.state_dict())
        
        # Update target network
        if not self.agent.tau or self.agent.tau > 0:
            self.agent.qnetwork_target.load_state_dict(self.agent.qnetwork_local.state_dict())
    
    def train_local_model(self, loss_fn: Callable = nn.MSELoss()):
        """
        Train the local model with client data.
        
        Args:
            loss_fn: Loss function for training
            
        Returns:
            Dict: Dictionary containing training loss history and number of samples trained
        """
        if not self.local_experiences:
            raise ValueError("Client has no local experience data to train!")
        
        # Add experiences to agent's replay buffer
        for exp in self.local_experiences:
            state, action, reward, next_state, done = exp
            self.agent.step(state, action, reward, next_state, done)
        
        # Perform update steps
        losses = []
        for _ in range(self.local_epochs * len(self.local_experiences) // self.batch_size):
            if len(self.agent.memory) > self.batch_size:
                if self.agent.use_prioritized_replay:
                    experiences, indices, weights = self.agent.memory.sample(self.agent.beta)
                    self.agent.learn(experiences, weights)
                else:
                    experiences = self.agent.memory.sample()
                    self.agent.learn(experiences)
                
                if self.agent.loss_history:
                    losses.append(self.agent.loss_history[-1])
        
        # Update client model from agent
        self.model.load_state_dict(self.agent.qnetwork_local.state_dict())
        
        return {
            'client_id': self.client_id,
            'train_loss': losses,
            'val_loss': None,
            'samples': len(self.local_experiences),
            'trust_score': self.trust_score
        }

class FederatedRL(FederatedLearning):
    """
    Manage Federated Reinforcement Learning training process.
    """
    def __init__(self, 
                 global_model: QNetwork,
                 aggregation_method: str = 'fedavg',
                 client_selection_method: str = 'random',
                 min_clients_per_round: int = 2,
                 min_samples_per_client: int = 10,
                 device: str = 'cpu',
                 personalized: bool = True,
                 personalization_alpha: float = 0.3,
                 privacy_preserving: bool = False,
                 privacy_epsilon: float = 0.1,
                 optimized_aggregation: bool = False):
        """
        Initialize Federated Reinforcement Learning system.
        
        Args:
            global_model: Global Q Network
            aggregation_method: Aggregation method ('fedavg', 'fedtrust', 'fedadam')
            client_selection_method: Client selection method ('random', 'trust_based', 'performance_based')
            min_clients_per_round: Minimum number of clients needed for each round
            min_samples_per_client: Minimum number of samples each client needs
            device: Device to use
            personalized: Whether to use client-specific customization
            personalization_alpha: Weight for client-specific customization (0-1)
            privacy_preserving: Whether to enable privacy preserving
            privacy_epsilon: Epsilon parameter for differential privacy
            optimized_aggregation: Whether to use optimized aggregation
        """
        super(FederatedRL, self).__init__(
            global_model, aggregation_method, client_selection_method,
            min_clients_per_round, min_samples_per_client, device,
            personalized, personalization_alpha
        )
        
        # Add additional parameters for RL across shards
        self.privacy_preserving = privacy_preserving
        self.privacy_epsilon = privacy_epsilon
        
        # Store additional information for RL
        self.client_rewards = defaultdict(list)
        self.global_environment = None
        
        # Optimized aggregation
        self.optimized_aggregation = optimized_aggregation
        if optimized_aggregation:
            self.aggregation_manager = ModelAggregationManager(default_method='weighted_average')
            
            # Map old method names to new method names
            self.method_mapping = {
                'fedavg': 'weighted_average',
                'fedtrust': 'adaptive_fedavg',
                'fedadam': 'fedprox'
            }
        
    def add_client(self, client: FRLClient):
        """
        Add a client to the Federated RL system.
        
        Args:
            client: FRLClient to add
        """
        if not isinstance(client, FRLClient):
            raise TypeError("Client must be of type FRLClient")
            
        super().add_client(client)
    
    def set_global_environment(self, env):
        """
        Set the global environment for evaluation.
        
        Args:
            env: Global RL environment
        """
        self.global_environment = env
    
    def _apply_differential_privacy(self, model_updates):
        """
        Apply differential privacy to model updates.
        
        Args:
            model_updates: Model parameters updates
            
        Returns:
            Tensor: Updated parameters with added noise
        """
        if not self.privacy_preserving:
            return model_updates
            
        # Add Laplace noise for privacy preserving
        sensitivity = 2.0  # Maximum sensitivity of gradient updates
        scale = sensitivity / self.privacy_epsilon
        
        noise = torch.distributions.laplace.Laplace(
            torch.zeros_like(model_updates),
            torch.ones_like(model_updates) * scale
        ).sample()
        
        return model_updates + noise
    
    def train_round(self, 
                   round_num: int, 
                   client_fraction: float = 0.5,
                   steps_per_client: int = 1000,
                   exploration_epsilon: float = 0.1,
                   global_eval_episodes: int = 5) -> Dict:
        """
        Perform one training round in Federated RL.
        
        Args:
            round_num: Training round number
            client_fraction: Percentage of clients participating
            steps_per_client: Number of steps each client interacts with the environment
            exploration_epsilon: Epsilon for exploration policy
            global_eval_episodes: Number of episodes to evaluate global model
            
        Returns:
            Dict: Training round information and metrics
        """
        self.round_counter = round_num
        
        # Select participating clients for this round
        selected_clients = self.select_clients(client_fraction)
        
        if len(selected_clients) < self.min_clients_per_round:
            print(f"Not enough clients: {len(selected_clients)} < {self.min_clients_per_round}")
            return {'round': round_num, 'error': 'Not enough clients'}
        
        client_updates = {}
        
        # Send global model to selected clients
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            if self.personalized:
                # If using customization, combine global model with client-specific model
                self._personalize_client_model(client)
            else:
                # Update client model with global model
                client.set_model_params(self.global_model.state_dict())
            
            # Update agent's model
            client.update_agent_model()
            
            # Collect experiences from environment
            client.collect_experiences(steps_per_client, exploration_epsilon)
            
            # Train local model
            client_metrics = client.train_local_model()
            
            # If client has enough data, collect update
            if client_metrics['samples'] >= self.min_samples_per_client:
                client_updates[client_id] = {
                    'params': client.get_model_params(),
                    'metrics': client_metrics
                }
                
                # Apply differential privacy if enabled
                if self.privacy_preserving:
                    for key in client_updates[client_id]['params']:
                        client_updates[client_id]['params'][key] = self._apply_differential_privacy(
                            client_updates[client_id]['params'][key]
                        )
            else:
                print(f"Skipping client {client_id} because not enough data: "
                     f"{client_metrics['samples']} < {self.min_samples_per_client}")
        
        # Aggregate updates from selected clients
        if client_updates:
            self.aggregate_updates(client_updates)
        
        # Evaluate global model if there's a global environment
        global_reward = None
        if self.global_environment is not None:
            global_reward = self._evaluate_global_model(global_eval_episodes)
            self.global_val_loss.append(-global_reward)  # Use negative reward as loss
        
        # Calculate average training loss across clients
        avg_train_loss = np.mean([
            np.mean(update['metrics']['train_loss']) if update['metrics']['train_loss'] else 0
            for _, update in client_updates.items()
        ])
        self.global_train_loss.append(avg_train_loss)
        
        # Store evaluation metrics for current round
        round_metrics = {
            'round': round_num,
            'clients': selected_clients,
            'avg_train_loss': avg_train_loss,
            'global_reward': global_reward,
            'client_metrics': {
                cid: update['metrics'] for cid, update in client_updates.items()
            }
        }
        
        self.round_metrics.append(round_metrics)
        return round_metrics
    
    def _evaluate_global_model(self, num_episodes: int = 5) -> float:
        """
        Evaluate global model on global environment.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            float: Average reward
        """
        if self.global_environment is None:
            return None
        
        # Create temporary agent with global model
        state_size = next(iter(self.global_model.parameters())).size(1)
        action_size = self.global_model.action_dim[0]
        
        temp_agent = DQNAgent(state_size, action_size, device=self.device)
        temp_agent.qnetwork_local.load_state_dict(self.global_model.state_dict())
        temp_agent.qnetwork_target.load_state_dict(self.global_model.state_dict())
        
        # Evaluate 
        total_rewards = []
        
        for _ in range(num_episodes):
            state = self.global_environment.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = temp_agent.act(state, eps=0.0)  # No exploration in evaluation
                next_state, reward, done, _ = self.global_environment.step(action)
                episode_reward += reward
                state = next_state
                
            total_rewards.append(episode_reward)
            
        return np.mean(total_rewards)
    
    def train(self, 
             num_rounds: int,
             client_fraction: float = 0.5,
             steps_per_client: int = 1000,
             exploration_schedule = lambda r: max(0.05, 0.5 * (0.99 ** r)),
             global_eval_episodes: int = 5,
             save_path: Optional[str] = None,
             early_stopping_rounds: int = 10,
             early_stopping_tolerance: float = 0.001,
             verbose: bool = True):
        """
        Perform Federated RL training through multiple rounds.
        
        Args:
            num_rounds: Number of training rounds
            client_fraction: Client participation ratio in each round
            steps_per_client: Number of steps each client interacts with the environment
            exploration_schedule: Function to get epsilon from round number
            global_eval_episodes: Number of episodes to evaluate global model
            save_path: Path to save the best model
            early_stopping_rounds: Number of rounds to wait before stopping early
            early_stopping_tolerance: Minimum improvement threshold
            verbose: Whether to display detailed information in training process
            
        Returns:
            Dict: Dictionary containing training history
        """
        best_reward = float('-inf')
        rounds_without_improvement = 0
        
        for round_idx in range(1, num_rounds + 1):
            # Get epsilon from exploration schedule
            exploration_epsilon = exploration_schedule(round_idx)
            
            # Perform one training round
            round_metrics = self.train_round(
                round_idx, 
                client_fraction, 
                steps_per_client,
                exploration_epsilon,
                global_eval_episodes
            )
            
            global_reward = round_metrics.get('global_reward')
            
            # Check for improvement
            if global_reward is not None:
                if global_reward > best_reward + early_stopping_tolerance:
                    best_reward = global_reward
                    rounds_without_improvement = 0
                    
                    if save_path:
                        self.save_global_model(save_path)
                else:
                    rounds_without_improvement += 1
            else:
                rounds_without_improvement += 1
            
            # Display information if requested
            if verbose:
                try:
                    print(f"Round {round_idx}/{num_rounds} | "
                         f"Loss: {round_metrics['avg_train_loss']:.4f} | "
                         f"Reward: {global_reward if global_reward is not None else 'N/A':.2f} | "
                         f"Epsilon: {exploration_epsilon:.3f} | "
                         f"Clients: {len(round_metrics['clients'])}")
                except:
                    pass
            
            # Early stopping
            if rounds_without_improvement >= early_stopping_rounds:
                try:
                    print(f"Early stopping triggered after {round_idx} rounds")
                except:
                    pass
                break
        
        # Read best model if available
        if save_path:
            try:
                self.load_global_model(save_path)
                try:
                    print(f"Read best model from {save_path}")
                except:
                    pass
            except:
                try:
                    print(f"Could not read model from {save_path}, using last model")
                except:
                    pass
        
        return {
            'rounds_completed': min(num_rounds, round_idx),
            'best_reward': best_reward,
            'train_loss_history': self.global_train_loss,
            'reward_history': [-loss for loss in self.global_val_loss] if self.global_val_loss else None,
            'round_metrics': self.round_metrics
        }
    
    def aggregate_updates(self, client_updates: Dict) -> None:
        """
        Aggregate parameter updates from clients according to selected aggregation method.
        
        Args:
            client_updates: Dictionary containing updates from each client
        """
        if not client_updates:
            return
        
        # Use optimized aggregation if enabled
        if self.optimized_aggregation:
            # Get list of model parameters
            params_list = [update['params'] for _, update in client_updates.items()]
            
            # Get weights based on sample counts
            sample_counts = [update['metrics']['samples'] for _, update in client_updates.items()]
            weights = [count / sum(sample_counts) if sum(sample_counts) > 0 else 1.0 / len(sample_counts) 
                      for count in sample_counts]
            
            # Additional parameters based on aggregation method
            kwargs = {'weights': weights}
            
            if self.aggregation_method == 'fedtrust':
                # Get trust and performance scores for adaptive_fedavg
                trust_scores = [update['metrics']['trust_score'] for _, update in client_updates.items()]
                
                # Calculate performance score (inverse of loss)
                performance_scores = []
                for _, update in client_updates.items():
                    if update['metrics']['val_loss'] is not None:
                        perf = 1.0 / (update['metrics']['val_loss'] + 1e-10)
                    else:
                        perf = 1.0 / (update['metrics']['train_loss'][-1] if update['metrics']['train_loss'] else 1.0 + 1e-10)
                    performance_scores.append(perf)
                
                kwargs.update({
                    'trust_scores': trust_scores,
                    'performance_scores': performance_scores
                })
            
            elif self.aggregation_method == 'fedadam':
                # Add global model parameters for FedProx
                kwargs.update({
                    'global_params': self.global_model.state_dict(),
                    'mu': 0.01  # Default regularization coefficient
                })
            
            # Check for security concerns
            suspected_byzantine = any(client.trust_score < 0.3 for client in self.clients.values())
            
            # Propose best aggregation method if not specified
            if self.aggregation_method == 'auto':
                method = self.aggregation_manager.recommend_method(
                    num_clients=len(params_list),
                    has_trust_scores=(self.aggregation_method == 'fedtrust'),
                    suspected_byzantine=suspected_byzantine
                )
            else:
                # Map old method names to new method names
                method = self.method_mapping.get(self.aggregation_method, 'weighted_average')
            
            # Aggregate model with selected method
            aggregated_params = self.aggregation_manager.aggregate(method, params_list, **kwargs)
            
            # Update global model
            self.global_model.load_state_dict(aggregated_params)
            
            # Update client performance
            for client_id, update in client_updates.items():
                # Store average reward
                if 'rewards' in update['metrics']:
                    self.client_rewards[client_id].append(np.mean(update['metrics']['rewards']))
        else:
            # Use default aggregation if not enabled
            super().aggregate_updates(client_updates) 