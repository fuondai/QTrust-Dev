"""
Privacy Management Module for Federated Learning in QTrust

This module provides privacy-preserving mechanisms for Federated Learning in the QTrust blockchain system.
It includes differential privacy techniques to protect sensitive data during model aggregation
and secure aggregation protocols to enable collaborative learning without exposing individual updates.

Key components:
- PrivacyManager: Manages differential privacy mechanisms with configurable privacy budgets 
- SecureAggregator: Implements secure aggregation protocols for combining model updates privately
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import copy

class PrivacyManager:
    """
    Manages privacy protection mechanisms in Federated Learning.
    """
    def __init__(self, 
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 clip_norm: float = 1.0,
                 noise_multiplier: float = 1.1):
        """
        Initialize Privacy Manager.
        
        Args:
            epsilon: Privacy budget parameter
            delta: Failure probability parameter
            clip_norm: Gradient clipping threshold
            noise_multiplier: Noise multiplier coefficient
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        
        # Track privacy budget
        self.consumed_budget = 0.0
        self.privacy_metrics = []
    
    def add_noise_to_gradients(self, 
                             gradients: torch.Tensor,
                             num_samples: int) -> torch.Tensor:
        """
        Add Gaussian noise to gradients for privacy protection.
        
        Args:
            gradients: Tensor containing gradients
            num_samples: Number of samples in batch
            
        Returns:
            Tensor: Gradients with added noise
        """
        # Clip gradients
        total_norm = torch.norm(gradients)
        scale = torch.min(torch.tensor(1.0), self.clip_norm / (total_norm + 1e-10))
        gradients = gradients * scale
        
        # Calculate noise standard deviation
        noise_scale = self.clip_norm * self.noise_multiplier / np.sqrt(num_samples)
        
        # Add Gaussian noise
        noise = torch.normal(0, noise_scale, size=gradients.shape)
        noisy_gradients = gradients + noise
        
        # Update consumed privacy budget
        self._update_privacy_accounting(num_samples)
        
        return noisy_gradients
    
    def add_noise_to_model(self, 
                          model_params: Dict[str, torch.Tensor],
                          num_samples: int) -> Dict[str, torch.Tensor]:
        """
        Add noise to model parameters for privacy protection.
        
        Args:
            model_params: Dictionary containing model parameters
            num_samples: Number of samples used
            
        Returns:
            Dict: Model parameters with added noise
        """
        noisy_params = {}
        for key, param in model_params.items():
            noisy_params[key] = self.add_noise_to_gradients(param, num_samples)
        return noisy_params
    
    def _update_privacy_accounting(self, num_samples: int) -> None:
        """
        Update consumed privacy budget.
        
        Args:
            num_samples: Number of samples in batch
        """
        # Calculate privacy loss for this update
        q = 1.0 / num_samples  # Sampling ratio
        steps = 1
        
        # Use RDP accountant to calculate privacy loss
        rdp = self._compute_rdp(q, self.noise_multiplier, steps)
        
        # Convert RDP to (ε, δ)-DP
        eps = self._rdp_to_dp(rdp, self.delta)
        
        self.consumed_budget += eps
        
        # Save metrics
        self.privacy_metrics.append({
            'epsilon': eps,
            'total_budget': self.consumed_budget,
            'remaining_budget': max(0, self.epsilon - self.consumed_budget),
            'num_samples': num_samples
        })
    
    def _compute_rdp(self, q: float, noise_multiplier: float, steps: int) -> float:
        """
        Calculate Rényi Differential Privacy (RDP) for Gaussian mechanism.
        
        Args:
            q: Sampling ratio
            noise_multiplier: Noise multiplier coefficient
            steps: Number of steps
            
        Returns:
            float: RDP value
        """
        # Calculate RDP for Gaussian mechanism with subsampling
        c = noise_multiplier
        alpha = 10  # RDP order
        
        # Calculate RDP for one step
        rdp_step = q**2 * alpha / (2 * c**2)
        
        # Calculate total RDP for all steps
        return rdp_step * steps
    
    def _rdp_to_dp(self, rdp: float, delta: float) -> float:
        """
        Convert RDP to (ε, δ)-DP.
        
        Args:
            rdp: RDP value
            delta: Desired δ parameter
            
        Returns:
            float: Corresponding ε value
        """
        # Convert using formula from theorem 3.1 in RDP paper
        alpha = 10  # RDP order
        return rdp + np.sqrt(2 * np.log(1/delta) / alpha)
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """
        Generate privacy status report.
        
        Returns:
            Dict: Detailed privacy report
        """
        if not self.privacy_metrics:
            return {
                'status': 'No privacy metrics available',
                'consumed_budget': 0.0,
                'remaining_budget': self.epsilon
            }
            
        latest = self.privacy_metrics[-1]
        return {
            'status': 'Privacy budget exceeded' if self.consumed_budget > self.epsilon else 'Active',
            'consumed_budget': self.consumed_budget,
            'remaining_budget': max(0, self.epsilon - self.consumed_budget),
            'noise_multiplier': self.noise_multiplier,
            'clip_norm': self.clip_norm,
            'last_update': {
                'epsilon': latest['epsilon'],
                'num_samples': latest['num_samples']
            },
            'total_updates': len(self.privacy_metrics)
        }

class SecureAggregator:
    """
    Implements secure model aggregation with privacy mechanisms.
    """
    def __init__(self, 
                 privacy_manager: PrivacyManager,
                 secure_communication: bool = True,
                 threshold: int = 3):
        """
        Initialize Secure Aggregator.
        
        Args:
            privacy_manager: PrivacyManager for privacy management
            secure_communication: Enable/disable secure communication
            threshold: Minimum number of clients required for aggregation
        """
        self.privacy_manager = privacy_manager
        self.secure_communication = secure_communication
        self.threshold = threshold
        
        # Store keys and shares
        self.key_shares = {}
        self.masked_models = {}
    
    def generate_key_shares(self, num_clients: int) -> List[bytes]:
        """
        Generate key shares for Secure Aggregation.
        
        Args:
            num_clients: Number of participating clients
            
        Returns:
            List[bytes]: List of key shares
        """
        # TODO: Implement Shamir's Secret Sharing
        pass
    
    def aggregate_secure(self,
                        client_updates: Dict[int, Dict[str, torch.Tensor]],
                        weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """
        Perform secure model aggregation.
        
        Args:
            client_updates: Dictionary containing updates from clients
            weights: Weights for each client
            
        Returns:
            Dict: Securely aggregated model parameters
        """
        if len(client_updates) < self.threshold:
            raise ValueError(f"Not enough clients: {len(client_updates)} < {self.threshold}")
            
        # Normalize weights
        if weights is None:
            weights = [1.0 / len(client_updates)] * len(client_updates)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Add noise to each client update
        noisy_updates = {}
        for client_id, update in client_updates.items():
            num_samples = update.get('num_samples', 1)
            noisy_updates[client_id] = self.privacy_manager.add_noise_to_model(
                update['params'], num_samples
            )
        
        # Aggregate the protected updates
        result = {}
        for key in noisy_updates[list(noisy_updates.keys())[0]].keys():
            result[key] = sum(
                weights[i] * updates[key]
                for i, (_, updates) in enumerate(noisy_updates.items())
            )
            
        return result 