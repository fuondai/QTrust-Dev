"""
Federated Learning Manager Module

This module provides a comprehensive federated learning management system for distributed model training.
It handles client selection, model distribution, collection, aggregation, and evaluation across nodes.
The manager supports various aggregation methods including weighted average, median, trimmed mean, and 
Byzantine-resistant approaches like Krum. The module also supports trust-weighted client selection 
and performance tracking.

Key features:
- Client selection based on trust and performance metrics
- Model distribution and collection across distributed nodes
- Multiple aggregation methods for different security requirements
- Support for performance tracking and metrics collection
- Secure model saving and versioning
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from qtrust.federated.model_aggregation import ModelAggregator
from qtrust.federated.protocol import FederatedProtocol


class FederatedLearningManager:
    """
    Manager for federated learning operations.
    
    This class coordinates the federated learning process, including client selection,
    model distribution, aggregation, and evaluation across distributed nodes.
    """
    
    def __init__(
        self,
        num_clients: int = 10,
        client_fraction: float = 0.8,
        local_epochs: int = 5,
        aggregation_method: str = "fedavg",
        protocol: Optional[FederatedProtocol] = None,
        model_path: str = "models/federated",
        enable_trust_weighting: bool = True
    ):
        """
        Initialize a federated learning manager.
        
        Args:
            num_clients: Total number of federated clients
            client_fraction: Fraction of clients to select each round
            local_epochs: Number of local training epochs per round
            aggregation_method: Method for model aggregation
            protocol: Communication protocol
            model_path: Path to save federated models
            enable_trust_weighting: Whether to use trust scores in aggregation
        """
        self.num_clients = num_clients
        self.client_fraction = client_fraction
        self.local_epochs = local_epochs
        self.round = 0
        self.model_path = model_path
        self.enable_trust_weighting = enable_trust_weighting
        self.aggregation_method = aggregation_method
        
        # Create model path if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize aggregator
        self.aggregator = ModelAggregator()
        
        # Initialize protocol
        self.protocol = protocol or FederatedProtocol()
        
        # Client metrics
        self.client_metrics = {i: {'trust': 1.0, 'performance': 1.0} for i in range(num_clients)}
        
        # Keep track of global model performance
        self.global_metrics = {
            'loss': [],
            'accuracy': [],
            'convergence_rate': 0.0
        }
    
    def select_clients(self) -> List[int]:
        """
        Select clients for the current round.
        
        Returns:
            List of selected client IDs
        """
        num_selected = max(1, int(self.num_clients * self.client_fraction))
        
        # Use trust scores to weight client selection if enabled
        if self.enable_trust_weighting:
            trust_scores = np.array([self.client_metrics[i]['trust'] for i in range(self.num_clients)])
            selection_probs = trust_scores / trust_scores.sum()
            selected_clients = np.random.choice(
                self.num_clients, 
                size=num_selected, 
                replace=False, 
                p=selection_probs
            )
        else:
            selected_clients = np.random.choice(self.num_clients, size=num_selected, replace=False)
        
        return selected_clients.tolist()
    
    def distribute_model(self, global_model: Dict[str, np.ndarray], selected_clients: List[int]) -> bool:
        """
        Distribute the global model to selected clients.
        
        Args:
            global_model: Global model parameters
            selected_clients: List of selected client IDs
            
        Returns:
            Success status
        """
        for client_id in selected_clients:
            success = self.protocol.send_model(global_model, client_id)
            if not success:
                return False
                
        return True
    
    def collect_models(self, selected_clients: List[int]) -> Tuple[List[Dict[str, np.ndarray]], List[float]]:
        """
        Collect updated models from clients.
        
        Args:
            selected_clients: List of selected client IDs
            
        Returns:
            Tuple of (list of client models, list of client weights)
        """
        client_models = []
        client_weights = []
        
        for client_id in selected_clients:
            client_model = self.protocol.receive_model(client_id)
            if client_model:
                client_models.append(client_model)
                
                # Use trust and performance metrics for weighting
                if self.enable_trust_weighting:
                    weight = self.client_metrics[client_id]['trust'] * self.client_metrics[client_id]['performance']
                else:
                    weight = 1.0
                    
                client_weights.append(weight)
        
        return client_models, client_weights
    
    def aggregate_models(
        self, 
        client_models: List[Dict[str, np.ndarray]], 
        client_weights: List[float]
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate client models into a new global model.
        
        Args:
            client_models: List of client models
            client_weights: List of client weights for aggregation
            
        Returns:
            Aggregated global model
        """
        if not client_models:
            return {}
        
        # Choose appropriate method for aggregation
        if self.aggregation_method == 'weighted_average' or self.aggregation_method == 'fedavg':
            return self.aggregator.weighted_average(client_models, client_weights)
        elif self.aggregation_method == 'median':
            return self.aggregator.median(client_models)
        elif self.aggregation_method == 'trimmed_mean':
            return self.aggregator.trimmed_mean(client_models, trim_ratio=0.1)
        elif self.aggregation_method == 'krum':
            return self.aggregator.krum(client_models, num_byzantine=int(len(client_models) * 0.1))
        elif self.aggregation_method == 'adaptive':
            trust_scores = [self.client_metrics[i]['trust'] for i in range(len(client_models))]
            performance_scores = [self.client_metrics[i]['performance'] for i in range(len(client_models))]
            return self.aggregator.adaptive_federated_averaging(client_models, trust_scores, performance_scores)
        else:
            # Default to weighted average if method not recognized
            return self.aggregator.weighted_average(client_models, client_weights)
    
    def update_client_metrics(self, client_id: int, metrics: Dict[str, float]):
        """
        Update metrics for a specific client.
        
        Args:
            client_id: Client ID
            metrics: Dictionary with performance metrics
        """
        if 'trust' in metrics:
            self.client_metrics[client_id]['trust'] = metrics['trust']
        
        if 'performance' in metrics:
            self.client_metrics[client_id]['performance'] = metrics['performance']
    
    def save_model(self, global_model: Dict[str, np.ndarray], suffix: str = None):
        """
        Save the global model to disk.
        
        Args:
            global_model: Global model parameters
            suffix: Optional suffix for filename
        """
        filename = f"global_model_round_{self.round}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".npz"
        
        path = os.path.join(self.model_path, filename)
        np.savez(path, **global_model)
    
    def run_round(self, global_model: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run a complete federated learning round.
        
        Args:
            global_model: Current global model parameters
            
        Returns:
            Updated global model
        """
        self.round += 1
        start_time = time.time()
        
        # Select clients for this round
        selected_clients = self.select_clients()
        print(f"Round {self.round}: Selected {len(selected_clients)} clients")
        
        # Distribute model to selected clients
        distribution_success = self.distribute_model(global_model, selected_clients)
        if not distribution_success:
            print("Error: Failed to distribute model to clients")
            return global_model
        
        # Simulate local training (in a real system, clients train locally)
        # Here we just wait a bit to simulate training time
        time.sleep(0.1 * self.local_epochs)
        
        # Collect updated models from clients
        client_models, client_weights = self.collect_models(selected_clients)
        
        if not client_models:
            print("Error: No client models received")
            return global_model
        
        # Aggregate models
        new_global_model = self.aggregate_models(client_models, client_weights)
        
        if not new_global_model:
            print("Error: Model aggregation failed")
            return global_model
        
        # Save the new global model
        self.save_model(new_global_model)
        
        # Update metrics
        round_time = time.time() - start_time
        print(f"Round {self.round} completed in {round_time:.2f} seconds")
        
        return new_global_model 