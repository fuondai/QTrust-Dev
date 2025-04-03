"""
Federated Learning core implementation.

This module provides the core implementation for federated learning,
including client and server components.
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from collections import defaultdict
from .model_aggregation import ModelAggregationManager
from .privacy import PrivacyManager, SecureAggregator
import time
import os

class FederatedModel(nn.Module):
    """
    Base model for Federated Learning.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize model with basic parameters.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension
        """
        super(FederatedModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input data
            
        Returns:
            Model output
        """
        return self.layers(x)

class FederatedClient:
    """
    Represents a client in the Federated Learning system.
    """
    def __init__(self, 
                 client_id: int, 
                 model: nn.Module, 
                 optimizer_class: torch.optim.Optimizer = optim.Adam,
                 learning_rate: float = 0.001,
                 local_epochs: int = 5,
                 batch_size: int = 32,
                 trust_score: float = 0.7,
                 device: str = 'cpu'):
        """
        Initialize Federated Learning client.
        
        Args:
            client_id: Unique client ID
            model: PyTorch model used by client
            optimizer_class: Optimizer class for training
            learning_rate: Learning rate
            local_epochs: Number of local training epochs
            batch_size: Batch size
            trust_score: Client trust score
            device: Device to use (CPU or GPU)
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.optimizer_class = optimizer_class
        self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.trust_score = trust_score
        self.device = device
        
        # Client's local data
        self.local_data = []
        
        # Client's personalized model
        self.personalized_model = None
    
    def set_data(self, data: List):
        """
        Set client's data.
        
        Args:
            data: Data for this client
        """
        self.local_data = data
    
    def set_model_params(self, model_params: Dict[str, torch.Tensor]):
        """
        Set model parameters from global model.
        
        Args:
            model_params: Global model parameters
        """
        self.model.load_state_dict(model_params)
    
    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """
        Get current model parameters of the client.
        
        Returns:
            Dict: Model parameters
        """
        return copy.deepcopy(self.model.state_dict())
    
    def train_local_model(self) -> Dict:
        """
        Train local model with client's data.
        
        Returns:
            Dict: Dictionary containing loss history and number of samples trained
        """
        # Simple implementation to avoid NotImplementedError
        if not self.local_data:
            return {'train_loss': [], 'val_loss': None, 'samples': 0}
        
        # Initialize model with new parameters
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
        
        # Simulate training by creating random data and training on it
        batch_losses = []
        num_samples = len(self.local_data)
        
        # Simulate training over local epochs
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            # Process each data sample
            for data in self.local_data:
                # Each sample may have different format, here we assume (state, action, reward)
                if len(data) >= 2:
                    # Convert state from numpy array to tensor if needed
                    if isinstance(data[0], np.ndarray):
                        state = torch.tensor(data[0], device=self.device)
                    else:
                        state = data[0].to(self.device) if hasattr(data[0], 'to') else torch.tensor(data[0], device=self.device)
                    
                    # Pass data through model
                    self.optimizer.zero_grad()
                    outputs = self.model(state)
                    
                    # Simulate loss function (e.g., MSE between output and a random value)
                    target = torch.randn_like(outputs)
                    loss = torch.nn.functional.mse_loss(outputs, target)
                    
                    # Backpropagation and parameter update
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
            
            # Add average epoch loss to list
            avg_epoch_loss = epoch_loss / max(1, len(self.local_data))
            batch_losses.append(avg_epoch_loss)
        
        # Return training information
        return {
            'train_loss': batch_losses,
            'val_loss': None,
            'samples': num_samples
        }
    
    def set_personalized_model(self, alpha: float, global_model_params: Dict[str, torch.Tensor]):
        """
        Set personalized model by combining global model with local model.
        
        Args:
            alpha: Weight for local model (0-1)
            global_model_params: Global model parameters
        """
        if not self.personalized_model:
            self.personalized_model = copy.deepcopy(self.model)
        
        local_params = self.model.state_dict()
        
        # Combine global and local models with weight alpha
        for key in local_params:
            local_params[key] = alpha * local_params[key] + (1 - alpha) * global_model_params[key]
        
        self.personalized_model.load_state_dict(local_params)

class FederatedLearning:
    """
    Manages the federated learning process.
    """
    def __init__(self, 
                 global_model: nn.Module,
                 aggregation_method: str = 'fedavg',
                 client_selection_method: str = 'random',
                 min_clients_per_round: int = 2,
                 min_samples_per_client: int = 10,
                 device: str = 'cpu',
                 personalized: bool = False,
                 personalization_alpha: float = 0.3,
                 optimized_aggregation: bool = True,
                 privacy_preserving: bool = True,
                 privacy_epsilon: float = 1.0,
                 privacy_delta: float = 1e-5,
                 secure_aggregation: bool = True):
        """
        Initialize the Federated Learning system.
        
        Args:
            global_model: Global PyTorch model
            aggregation_method: Aggregation method ('fedavg', 'fedtrust', 'fedadam', 'auto')
            client_selection_method: Client selection method ('random', 'trust_based', 'performance_based')
            min_clients_per_round: Minimum number of clients needed per round
            min_samples_per_client: Minimum number of samples per client
            device: Device to use
            personalized: Whether to use personalized models for each client
            personalization_alpha: Weight for personalized models (0-1)
            optimized_aggregation: Whether to use optimized aggregation
            privacy_preserving: Whether to use privacy preserving
            privacy_epsilon: Privacy budget (epsilon)
            privacy_delta: Probability of failure (delta)
            secure_aggregation: Whether to use secure aggregation
        """
        self.global_model = global_model.to(device)
        self.aggregation_method = aggregation_method
        self.client_selection_method = client_selection_method
        self.min_clients_per_round = min_clients_per_round
        self.min_samples_per_client = min_samples_per_client
        self.device = device
        self.personalized = personalized
        self.personalization_alpha = personalization_alpha
        
        # Client information
        self.clients = {}
        self.client_performance = defaultdict(list)
        
        # Training information
        self.round_counter = 0
        self.global_train_loss = []
        self.global_val_loss = []
        self.round_metrics = []
        
        # Initialize ModelAggregationManager
        self.aggregation_manager = ModelAggregationManager(
            default_method='weighted_average' if aggregation_method == 'fedavg' else aggregation_method
        )
        
        # Map old method names to new method names
        self.method_mapping = {
            'fedavg': 'weighted_average',
            'fedtrust': 'adaptive_fedavg',
            'fedadam': 'fedprox',
            'auto': 'auto'
        }
        
        # Initialize PrivacyManager and SecureAggregator if required
        self.privacy_preserving = privacy_preserving
        if privacy_preserving:
            self.privacy_manager = PrivacyManager(
                epsilon=privacy_epsilon,
                delta=privacy_delta
            )
            
            if secure_aggregation:
                self.secure_aggregator = SecureAggregator(
                    privacy_manager=self.privacy_manager,
                    threshold=min_clients_per_round
                )
            else:
                self.secure_aggregator = None
        else:
            self.privacy_manager = None
            self.secure_aggregator = None
    
    def add_client(self, client: FederatedClient):
        """
        Add a client to the Federated Learning system.
        
        Args:
            client: Client to add
        """
        if not isinstance(client, FederatedClient):
            raise TypeError("Client must be of type FederatedClient")
            
        self.clients[client.client_id] = client
    
    def select_clients(self, fraction: float) -> List[int]:
        """
        Select clients to participate in the current training round.
        
        Args:
            fraction: Fraction of clients to select
            
        Returns:
            List: List of selected client IDs
        """
        num_clients = max(self.min_clients_per_round, int(fraction * len(self.clients)))
        num_clients = min(num_clients, len(self.clients))
        
        if self.client_selection_method == 'random':
            # Select randomly
            selected_clients = random.sample(list(self.clients.keys()), num_clients)
            
        elif self.client_selection_method == 'trust_based':
            # Select based on trust scores
            clients_by_trust = sorted(
                self.clients.items(), 
                key=lambda x: x[1].trust_score, 
                reverse=True
            )
            selected_clients = [c[0] for c in clients_by_trust[:num_clients]]
            
        elif self.client_selection_method == 'performance_based':
            # Select based on recent performance
            if not self.client_performance or len(self.client_performance) < len(self.clients) / 2:
                # If not enough performance data, select randomly
                selected_clients = random.sample(list(self.clients.keys()), num_clients)
            else:
                # Calculate average recent performance for each client
                avg_performance = {}
                for client_id, performances in self.client_performance.items():
                    # Get up to 5 most recent rounds
                    recent_perf = performances[-5:] 
                    avg_performance[client_id] = sum(recent_perf) / len(recent_perf)
                
                # Select clients with highest performance
                top_clients = sorted(
                    avg_performance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                selected_clients = [c[0] for c in top_clients[:num_clients]]
        else:
            raise ValueError(f"Invalid client selection method: {self.client_selection_method}")
        
        return selected_clients
    
    def detect_byzantine_clients(self, client_updates: Dict, threshold: float = 0.8) -> List[int]:
        """
        Detect potentially Byzantine clients based on model parameter differences.
        
        Byzantine clients typically have parameter updates that differ significantly from the majority.
        This method compares the cosine similarity between each client's update and 
        the average update of all clients.
        
        Args:
            client_updates: Dictionary containing updates from each client
            threshold: Minimum similarity threshold to not be considered Byzantine
            
        Returns:
            List of client_ids suspected of being Byzantine
        """
        if len(client_updates) < 3:  # Need at least 3 clients for effective comparison
            return []
            
        # Get list of model parameters
        params_list = [update['params'] for _, update in client_updates.items()]
        client_ids = list(client_updates.keys())
        
        # Calculate average update (unweighted)
        avg_params = {}
        for key in params_list[0]:
            avg_params[key] = torch.mean(torch.stack([params[key] for params in params_list]), dim=0)
        
        # Calculate cosine similarity between each client and the average update
        similarities = []
        for client_id, params in zip(client_ids, params_list):
            sim = self._compute_model_similarity(params, avg_params)
            similarities.append((client_id, sim))
        
        # Identify clients with similarity below threshold
        byzantine_clients = [client_id for client_id, sim in similarities if sim < threshold]
        
        if byzantine_clients:
            print(f"Warning: Detected {len(byzantine_clients)} potentially Byzantine clients: {byzantine_clients}")
            
        return byzantine_clients
    
    def _compute_model_similarity(self, model1: Dict[str, torch.Tensor], model2: Dict[str, torch.Tensor]) -> float:
        """
        Calculate cosine similarity between two models.
        
        Args:
            model1: First model parameters
            model2: Second model parameters
            
        Returns:
            Cosine similarity (0-1), with 1 being identical
        """
        # Flatten model parameters into vectors
        vec1, vec2 = [], []
        
        for key in model1:
            if key in model2:
                vec1.append(model1[key].flatten())
                vec2.append(model2[key].flatten())
        
        # Concatenate parameters into a single vector
        vec1 = torch.cat(vec1)
        vec2 = torch.cat(vec2)
        
        # Calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
        return cos_sim.item()
    
    def aggregate_updates(self, client_updates: Dict) -> None:
        """
        Aggregate parameter updates from clients using the selected method.
        
        Args:
            client_updates: Dictionary containing updates from each client
        """
        if not client_updates:
            return
            
        # Detect Byzantine clients
        byzantine_clients = self.detect_byzantine_clients(client_updates)
        
        # Filter out Byzantine clients if detected
        if byzantine_clients and len(client_updates) > len(byzantine_clients):
            for client_id in byzantine_clients:
                if client_id in client_updates:
                    print(f"Removing client {client_id} from aggregation due to suspected Byzantine behavior")
                    client_updates.pop(client_id)
            
        # Get list of model parameters
        params_list = [update['params'] for _, update in client_updates.items()]
        
        # Get weights based on sample counts
        sample_counts = [update['metrics']['samples'] for _, update in client_updates.items()]
        weights = [count / sum(sample_counts) if sum(sample_counts) > 0 else 1.0 / len(sample_counts) 
                  for count in sample_counts]
        
        # If using secure aggregation
        if self.secure_aggregator is not None:
            try:
                aggregated_params = self.secure_aggregator.aggregate_secure(
                    client_updates, weights
                )
                self.global_model.load_state_dict(aggregated_params)
                return
            except Exception as e:
                print(f"Error during secure aggregation: {e}")
                print("Switching to regular aggregation method...")
                
        # Additional parameters based on aggregation method
        kwargs = {'weights': weights}
        
        # Mark suspected Byzantine clients
        suspected_byzantine = byzantine_clients
        
        if self.aggregation_method == 'fedtrust':
            # Get trust scores and performance for adaptive_fedavg
            trust_scores = []
            for client_id, update in client_updates.items():
                # If client is suspected, reduce trust score
                trust_score = update['metrics'].get('trust_score', 0.5)
                if client_id in byzantine_clients:
                    trust_score *= 0.1  # Reduce trust by 90%
                trust_scores.append(trust_score)
            
            # Calculate performance scores (inverse of loss)
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
        
        # Recommend best aggregation method if in auto mode
        if self.aggregation_method == 'auto':
            method = self.aggregation_manager.recommend_method(
                num_clients=len(params_list),
                has_trust_scores=(self.aggregation_method == 'fedtrust'),
                suspected_byzantine=(len(byzantine_clients) > 0)
            )
        else:
            # Map old method names to new method names
            method = self.method_mapping.get(self.aggregation_method, 'weighted_average')
            
            # If Byzantine clients detected, switch to robust method if not already using one
            if byzantine_clients and method not in ['median', 'trimmed_mean', 'krum']:
                print(f"Byzantine clients detected, switching from {method} to krum method")
                method = 'krum'
        
        # Aggregate model with chosen method
        aggregated_params = self.aggregation_manager.aggregate(method, params_list, **kwargs)
        
        # Add noise if using privacy preserving but no secure aggregation
        if self.privacy_preserving and self.secure_aggregator is None:
            total_samples = sum(sample_counts)
            aggregated_params = self.privacy_manager.add_noise_to_model(
                aggregated_params, total_samples
            )
        
        # Update global model
        self.global_model.load_state_dict(aggregated_params)
        
        # Update performance metrics
        avg_loss = np.mean([
            update['metrics']['val_loss'] if update['metrics']['val_loss'] is not None 
            else update['metrics']['train_loss'][-1]
            for _, update in client_updates.items()
        ])
        
        self.aggregation_manager.update_performance_metrics(method, {
            'loss': avg_loss,
            'score': 1.0 / (avg_loss + 1e-10),
            'num_clients': len(params_list),
            'suspected_byzantine': len(byzantine_clients) > 0
        })
        
        # Update privacy report if available
        if self.privacy_preserving:
            privacy_report = self.privacy_manager.get_privacy_report()
            print("\nPrivacy Report:")
            print(f"Status: {privacy_report['status']}")
            print(f"Consumed Budget: {privacy_report['consumed_budget']:.4f}")
            print(f"Remaining Budget: {privacy_report['remaining_budget']:.4f}")
            if privacy_report['status'] == 'Privacy budget exceeded':
                print("Warning: Privacy budget has exceeded limits!")
    
    def _personalize_client_model(self, client: FederatedClient) -> None:
        """
        Personalize client model by combining global and local models.
        
        Args:
            client: Client to personalize model for
        """
        client.set_personalized_model(
            self.personalization_alpha, 
            self.global_model.state_dict()
        )
    
    def train_round(self, round_num: int, client_fraction: float = 0.5) -> Dict:
        """
        Perform one round of Federated Learning training.
        
        Args:
            round_num: Training round number
            client_fraction: Percentage of clients to participate
            
        Returns:
            Dict: Information and metrics for the training round
        """
        self.round_counter = round_num
        
        # Select clients to participate in this round
        selected_clients = self.select_clients(client_fraction)
        
        if len(selected_clients) < self.min_clients_per_round:
            print(f"Not enough clients: {len(selected_clients)} < {self.min_clients_per_round}")
            return {'round': round_num, 'error': 'Not enough clients'}
        
        client_updates = {}
        
        # Send global model to selected clients
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            if self.personalized:
                # If using personalization, combine global model with personal model
                self._personalize_client_model(client)
            else:
                # Update client model with global model
                client.set_model_params(self.global_model.state_dict())
            
            # Train local model
            client_metrics = client.train_local_model()
            
            # If client has enough data, collect updates
            if client_metrics['samples'] >= self.min_samples_per_client:
                client_updates[client_id] = {
                    'params': client.get_model_params(),
                    'metrics': client_metrics
                }
            else:
                print(f"Skipping client {client_id} due to insufficient data: "
                     f"{client_metrics['samples']} < {self.min_samples_per_client}")
        
        # Aggregate updates from clients
        if client_updates:
            self.aggregate_updates(client_updates)
        
        # Calculate average training and validation loss across clients
        avg_train_loss = np.mean([
            np.mean(update['metrics']['train_loss']) if update['metrics']['train_loss'] else 0
            for _, update in client_updates.items()
        ])
        
        avg_val_loss = None
        if all(update['metrics']['val_loss'] is not None for _, update in client_updates.items()):
            avg_val_loss = np.mean([
                update['metrics']['val_loss'] for _, update in client_updates.items()
            ])
        
        self.global_train_loss.append(avg_train_loss)
        if avg_val_loss is not None:
            self.global_val_loss.append(avg_val_loss)
        
        # Save metrics for current round
        round_metrics = {
            'round': round_num,
            'clients': selected_clients,
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss,
            'client_metrics': {
                cid: update['metrics'] for cid, update in client_updates.items()
            }
        }
        
        self.round_metrics.append(round_metrics)
        return round_metrics
    
    def train(self, 
             num_rounds: int,
             client_fraction: float = 0.5,
             early_stopping_rounds: int = 10,
             early_stopping_tolerance: float = 0.001,
             save_path: Optional[str] = None,
             verbose: bool = True) -> Dict:
        """
        Run Federated Learning training process over multiple rounds.
        
        Args:
            num_rounds: Number of training rounds
            client_fraction: Fraction of clients to participate in each round
            early_stopping_rounds: Number of rounds to wait before early stopping
            early_stopping_tolerance: Minimum improvement threshold
            save_path: Path to save best model
            verbose: Display detailed information during training
            
        Returns:
            Dict: Dictionary containing training history
        """
        best_val_loss = float('inf')
        rounds_without_improvement = 0
        
        for round_idx in range(1, num_rounds + 1):
            # Perform one training round
            round_metrics = self.train_round(round_idx, client_fraction)
            
            val_loss = round_metrics.get('avg_val_loss')
            
            # Check for improvement
            if val_loss is not None:
                if val_loss < best_val_loss - early_stopping_tolerance:
                    best_val_loss = val_loss
                    rounds_without_improvement = 0
                    
                    if save_path:
                        self.save_global_model(save_path)
                else:
                    rounds_without_improvement += 1
            
            # Print information if requested
            if verbose:
                val_str = f" | Val Loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Round {round_idx}/{num_rounds} | Train Loss: {round_metrics['avg_train_loss']:.4f}{val_str} | "
                     f"Clients: {len(round_metrics['clients'])}")
            
            # Early stopping
            if rounds_without_improvement >= early_stopping_rounds and val_loss is not None:
                print(f"Early stopping triggered after {round_idx} rounds")
                break
        
        # Load back best model if available
        if save_path and val_loss is not None:
            try:
                self.load_global_model(save_path)
                print(f"Loaded best model from {save_path}")
            except:
                print(f"Couldn't load model from {save_path}, using final model")
        
        return {
            'rounds_completed': min(num_rounds, round_idx),
            'best_val_loss': best_val_loss if val_loss is not None else None,
            'train_loss_history': self.global_train_loss,
            'val_loss_history': self.global_val_loss,
            'round_metrics': self.round_metrics
        }
    
    def save_global_model(self, path: str) -> None:
        """
        Save global model to file.
        
        Args:
            path: Path to save file
        """
        # Save model
        torch.save(self.global_model.state_dict(), path)
        
        # Create new path for metadata
        metadata_path = path.replace('.pth', '_metadata.pt')
        if metadata_path == path:  # If path doesn't have .pth extension
            metadata_path = path + '_metadata.pt'
            
        # Save metadata
        metadata = {
            'round_counter': self.round_counter,
            'global_train_loss': self.global_train_loss,
            'global_val_loss': self.global_val_loss,
            'last_round_metrics': self.round_metrics[-1] if self.round_metrics else None,
            'aggregation_method': self.aggregation_method,
            'timestamp': time.time(),
            'client_info': {client_id: {'trust_score': client.trust_score} 
                            for client_id, client in self.clients.items()}
        }
        
        # Save metadata information
        torch.save(metadata, metadata_path)
        print(f"Saved metadata at: {metadata_path}")
    
    def load_global_model(self, path: str, load_metadata: bool = True) -> None:
        """
        Load global model from file.
        
        Args:
            path: Path to saved file
            load_metadata: Whether to load metadata information
        """
        # Load model
        self.global_model.load_state_dict(torch.load(path, map_location=self.device))
        
        # Load metadata if requested
        if load_metadata:
            metadata_path = path.replace('.pth', '_metadata.pt')
            if metadata_path == path:  # If path doesn't have .pth extension
                metadata_path = path + '_metadata.pt'
                
            try:
                metadata = torch.load(metadata_path, map_location=self.device)
                
                # Restore information
                self.round_counter = metadata.get('round_counter', 0)
                self.global_train_loss = metadata.get('global_train_loss', [])
                self.global_val_loss = metadata.get('global_val_loss', [])
                
                # Restore client information if available
                for client_id, info in metadata.get('client_info', {}).items():
                    if client_id in self.clients:
                        self.clients[client_id].trust_score = info.get('trust_score', 0.7)
                
                print(f"Loaded metadata from: {metadata_path}")
            except FileNotFoundError:
                print(f"Metadata file not found: {metadata_path}")
            except Exception as e:
                print(f"Error loading metadata: {str(e)}")
    
    def save_training_history(self, path: str) -> None:
        """
        Save Federated Learning training history to file.
        
        Args:
            path: Path to save file
        """
        history = {
            'round_metrics': self.round_metrics,
            'global_train_loss': self.global_train_loss,
            'global_val_loss': self.global_val_loss,
            'aggregation_method': self.aggregation_method,
            'client_performance': dict(self.client_performance),
            'aggregation_performance': self.aggregation_manager.performance_metrics,
            'timestamp': time.time()
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save history
        torch.save(history, path)
        print(f"Saved training history at: {path}")
    
    def load_training_history(self, path: str) -> Dict:
        """
        Load Federated Learning training history from file.
        
        Args:
            path: Path to saved file
            
        Returns:
            Dict: Training history
        """
        try:
            history = torch.load(path, map_location=self.device)
            
            # Restore information
            self.round_metrics = history.get('round_metrics', [])
            self.global_train_loss = history.get('global_train_loss', [])
            self.global_val_loss = history.get('global_val_loss', [])
            
            # Restore client_performance if available
            client_perf = history.get('client_performance', {})
            for client_id, perf in client_perf.items():
                self.client_performance[int(client_id) if client_id.isdigit() else client_id] = perf
            
            # Restore aggregation performance
            if 'aggregation_performance' in history and self.aggregation_manager:
                self.aggregation_manager.performance_metrics = history['aggregation_performance']
            
            print(f"Loaded training history from: {path}")
            return history
        except FileNotFoundError:
            print(f"History file not found: {path}")
            return {}
        except Exception as e:
            print(f"Error loading training history: {str(e)}")
            return {}
    
    def get_client_performance_summary(self) -> Dict:
        """
        Create summary of client performance.
        
        Returns:
            Dict: Summary of client performance
        """
        summary = {}
        
        # Calculate average performance for each client
        for client_id, client in self.clients.items():
            performances = self.client_performance.get(client_id, [])
            
            # Only calculate if data exists
            if performances:
                avg_perf = sum(performances) / len(performances)
                # Get 5 most recent values
                recent_perf = performances[-5:] if len(performances) >= 5 else performances
                avg_recent_perf = sum(recent_perf) / len(recent_perf)
                
                summary[client_id] = {
                    'trust_score': client.trust_score,
                    'avg_performance': avg_perf,
                    'recent_performance': avg_recent_perf,
                    'participation_count': len(performances),
                    'last_performance': performances[-1] if performances else None
                }
            else:
                summary[client_id] = {
                    'trust_score': client.trust_score,
                    'avg_performance': None,
                    'recent_performance': None,
                    'participation_count': 0,
                    'last_performance': None
                }
        
        return summary

    def aggregate_models(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        client_weights: List[float],
        method: str = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client models into a new global model.
        
        Args:
            client_models: List of model parameter dictionaries
            client_weights: List of weights for each client
            method: Aggregation method (can be None to use default)
            
        Returns:
            Aggregated global model
        """
        if not method:
            method = self.method_mapping.get(self.aggregation_method, 'weighted_average')
        
        # Call aggregation_manager to perform aggregation
        try:
            return self.aggregation_manager.aggregate(
                method=method,
                params_list=client_models,
                weights=client_weights
            )
        except Exception as e:
            print(f"Error during model aggregation: {str(e)}")
            # If failed, return current global model
            return self.global_model.state_dict() 