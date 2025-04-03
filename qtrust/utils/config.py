"""
Configuration management module for the QTrust system.

This module provides tools for loading, saving, and managing configuration settings
for all components of the QTrust blockchain system. It supports YAML and JSON formats,
command-line arguments parsing, and nested configuration access.
"""

import os
import yaml
import json
import argparse
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

class QTrustConfig:
    """
    Configuration manager for the QTrust system.
    """
    
    DEFAULT_CONFIG = {
        # Environment parameters
        "environment": {
            "num_shards": 4,
            "num_nodes_per_shard": 10,
            "max_transactions_per_step": 100,
            "transaction_value_range": [0.1, 100.0],
            "max_steps": 1000,
            "latency_penalty": 0.1,
            "energy_penalty": 0.1,
            "throughput_reward": 1.0,
            "security_reward": 1.0,
            "cross_shard_reward": 0.5,
            "seed": 42
        },
        
        # DQN Agent parameters
        "dqn_agent": {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 0.995,
            "buffer_size": 10000,
            "batch_size": 64,
            "target_update": 10,
            "hidden_dim": 128,
            "num_episodes": 500
        },
        
        # Adaptive Consensus parameters
        "consensus": {
            "transaction_threshold_low": 10.0,
            "transaction_threshold_high": 50.0,
            "congestion_threshold": 0.7,
            "min_trust_threshold": 0.3,
            "fastbft_latency": 0.2,
            "fastbft_energy": 0.2,
            "fastbft_security": 0.5,
            "pbft_latency": 0.5,
            "pbft_energy": 0.5,
            "pbft_security": 0.8,
            "robustbft_latency": 0.8,
            "robustbft_energy": 0.8,
            "robustbft_security": 0.95
        },
        
        # MAD-RAPID Router parameters
        "routing": {
            "congestion_weight": 1.0,
            "latency_weight": 0.7,
            "energy_weight": 0.5,
            "trust_weight": 0.8,
            "prediction_horizon": 5,
            "congestion_threshold": 0.8,
            "weight_adjustment_rate": 0.1
        },
        
        # HTDCM parameters
        "trust": {
            "local_update_weight": 0.7,
            "global_update_weight": 0.3,
            "initial_trust": 0.5,
            "trust_threshold": 0.3,
            "penalty_factor": 0.2,
            "reward_factor": 0.1,
            "observation_window": 50,
            "suspicious_threshold": 0.7,
            "malicious_threshold": 0.9
        },
        
        # Federated Learning parameters
        "federated": {
            "num_rounds": 20,
            "local_epochs": 5,
            "fraction_fit": 0.8,
            "min_fit_clients": 3,
            "min_available_clients": 4,
            "batch_size": 32,
            "learning_rate": 0.01,
            "trust_threshold": 0.4
        },
        
        # Visualization parameters
        "visualization": {
            "save_plots": True,
            "plot_frequency": 50,
            "output_dir": "results"
        },
        
        # Simulator parameters
        "simulation": {
            "num_transactions": 1000,
            "cross_shard_prob": 0.3,
            "honest_node_prob": 0.9,
            "num_network_events": 20,
            "num_malicious_activities": 10
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration object.
        
        Args:
            config_path: Path to the configuration file. If None,
                        use the default configuration.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path is not None:
            self.load_config(config_path)
            
        # Ensure output directory exists
        output_dir = self.config["visualization"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
            
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to the configuration file
        """
        extension = os.path.splitext(config_path)[1].lower()
        
        if not os.path.exists(config_path):
            try:
                print(f"Warning: Configuration file {config_path} does not exist. Using default configuration.")
            except:
                pass
            return
        
        try:
            if extension == '.yaml' or extension == '.yml':
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif extension == '.json':
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
                
            # Update configuration with data from file
            self._update_nested_dict(self.config, config_data)
            
        except Exception as e:
            try:
                print(f"Error loading configuration file: {e}")
            except:
                pass
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update dictionary recursively, preserving nested structure.
        
        Args:
            d: Target dictionary
            u: Source dictionary
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def save_config(self, config_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to the configuration file
        """
        extension = os.path.splitext(config_path)[1].lower()
        
        try:
            os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
            
            if extension == '.yaml' or extension == '.yml':
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif extension == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
                
        except Exception as e:
            print(f"Error saving configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get the value of a configuration parameter.
        
        Args:
            key: Parameter key (can use dots for nested keys)
            default: Default value if key is not found
            
        Returns:
            Parameter value or default value
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
                
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set the value of a configuration parameter.
        
        Args:
            key: Parameter key (can use dots for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        current = self.config
        
        for i, k in enumerate(keys[:-1]):
            if isinstance(current, dict):
                if k not in current:
                    current[k] = {}
                current = current[k]
            else:
                raise ValueError(f"Cannot set value for {key}: {'.'.join(keys[:i+1])} is not a dictionary")
                
        if isinstance(current, dict):
            current[keys[-1]] = value
        else:
            raise ValueError(f"Cannot set value for {key}: {'.'.join(keys[:-1])} is not a dictionary")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration.
        
        Returns:
            Dictionary containing the entire configuration
        """
        return self.config.copy()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a section of the configuration.
        
        Args:
            section: Name of the configuration section
            
        Returns:
            Dictionary containing the requested configuration section
        """
        if section in self.config:
            return self.config[section].copy()
        return {}

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Namespace containing parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='QTrust - Smart Blockchain System with Deep Reinforcement Learning')
    
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to configuration file')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--num-shards', type=int, default=None,
                        help='Number of shards in blockchain network')
    
    parser.add_argument('--num-nodes-per-shard', type=int, default=None,
                        help='Number of nodes in each shard')
    
    parser.add_argument('--num-episodes', type=int, default=None,
                        help='Number of episodes for DQN training')
    
    parser.add_argument('--num-transactions', type=int, default=None,
                        help='Number of transactions to simulate')
    
    parser.add_argument('--cross-shard-prob', type=float, default=None,
                        help='Probability of cross-shard transactions')
    
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate for DQN Agent')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory to save results')
    
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'simulate'],
                        default='train', help='Run mode (train, evaluate, simulate)')
    
    parser.add_argument('--save-model', action='store_true',
                        help='Save model after training')
    
    parser.add_argument('--load-model', type=str, default=None,
                        help='Load model from given path')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations during run')
    
    return parser.parse_args()

def update_config_from_args(config: QTrustConfig, args: argparse.Namespace) -> None:
    """
    Update configuration from command line arguments.
    
    Args:
        config: Configuration object to update
        args: Namespace containing command line arguments
    """
    # Map arguments to configuration keys
    arg_to_config = {
        'seed': 'environment.seed',
        'num_shards': 'environment.num_shards',
        'num_nodes_per_shard': 'environment.num_nodes_per_shard',
        'num_episodes': 'dqn_agent.num_episodes',
        'num_transactions': 'simulation.num_transactions',
        'cross_shard_prob': 'simulation.cross_shard_prob',
        'learning_rate': 'dqn_agent.learning_rate',
        'output_dir': 'visualization.output_dir'
    }
    
    # Update configuration if argument is provided
    for arg_name, config_key in arg_to_config.items():
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            config.set(config_key, arg_value)
            
    # Handle other arguments
    if args.visualize:
        config.set('visualization.save_plots', True)

def load_config_from_args() -> QTrustConfig:
    """
    Load configuration from command line arguments.
    
    Returns:
        Updated configuration object
    """
    args = parse_arguments()
    config = QTrustConfig(args.config)
    update_config_from_args(config, args)
    return config, args 