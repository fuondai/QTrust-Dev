"""
DQN Utilities Module

This module provides utility functions and constants for DQN-based reinforcement learning agents.
Key features:
- Model saving and loading utilities
- Network update functions (soft/hard)
- Loss calculation functions
- Exploration rate decay functions
- Plotting and visualization utilities
- Logging configuration
- Device management
"""

import numpy as np
import torch
import torch.nn.functional as F
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dqn_agent')

# Constants
SAVE_DIR = 'models'  # Default directory to save models

def create_save_directory(dir_path: str = SAVE_DIR) -> str:
    """
    Create directory for saving models if it doesn't exist.
    
    Args:
        dir_path: Path to create directory
        
    Returns:
        str: Path of created directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Created directory {dir_path} to save models")
    return dir_path

def soft_update(target_model: torch.nn.Module, 
               local_model: torch.nn.Module, 
               tau: float = 1e-3):
    """
    Perform soft update of target network parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    
    Args:
        target_model: Target network to update
        local_model: Local network with new parameters
        tau: Interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        
def hard_update(target_model: torch.nn.Module, local_model: torch.nn.Module):
    """
    Perform complete update of target network with local network parameters.
    
    Args:
        target_model: Target network to update
        local_model: Local network with new parameters
    """
    target_model.load_state_dict(local_model.state_dict())

def calculate_td_error(current_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
    """
    Calculate temporal difference error between current and target Q values.
    
    Args:
        current_q: Current Q value
        target_q: Target Q value
        
    Returns:
        torch.Tensor: TD error
    """
    return torch.abs(target_q - current_q)

def calculate_huber_loss(current_q: torch.Tensor, 
                         target_q: torch.Tensor, 
                         weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate Huber loss between current and target Q values.
    
    Args:
        current_q: Current Q value
        target_q: Target Q value
        weights: Sample weights for prioritized replay
        
    Returns:
        torch.Tensor: Huber loss
    """
    elementwise_loss = F.huber_loss(current_q, target_q, reduction='none')
    
    if weights is not None:
        loss = torch.mean(elementwise_loss * weights)
    else:
        loss = torch.mean(elementwise_loss)
        
    return loss

def exponential_decay(start_value: float, end_value: float, decay_rate: float, step: int) -> float:
    """
    Calculate exponentially decayed value.
    
    Args:
        start_value: Initial value
        end_value: Minimum value
        decay_rate: Decay rate (0 < decay_rate < 1)
        step: Current step
    
    Returns:
        float: Decayed value
    """
    return max(end_value, start_value * (decay_rate ** step))

def linear_decay(start_value: float, end_value: float, decay_steps: int, step: int) -> float:
    """
    Calculate linearly decayed value.
    
    Args:
        start_value: Initial value
        end_value: Minimum value
        decay_steps: Steps to reach minimum
        step: Current step
    
    Returns:
        float: Decayed value
    """
    fraction = min(float(step) / float(decay_steps), 1.0)
    return start_value + fraction * (end_value - start_value)

def generate_timestamp() -> str:
    """
    Generate timestamp string for file naming.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_learning_curve(rewards: List[float], 
                        avg_window: int = 20,
                        title: str = "Learning Curve", 
                        save_path: Optional[str] = None):
    """
    Plot learning curve with moving average.
    
    Args:
        rewards: List of episode rewards
        avg_window: Window size for moving average
        title: Plot title
        save_path: Path to save plot (None for display)
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(rewards, label='Reward per Episode', alpha=0.3)
    
    avg_rewards = []
    for i in range(len(rewards)):
        if i < avg_window:
            avg_rewards.append(np.mean(rewards[:i+1]))
        else:
            avg_rewards.append(np.mean(rewards[i-avg_window+1:i+1]))
    
    plt.plot(avg_rewards, label=f'Average Reward (window={avg_window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved at: {save_path}")
    else:
        plt.show()
        
def format_time(seconds: float) -> str:
    """
    Format time duration from seconds.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    elif m > 0:
        return f"{int(m)}m {int(s)}s"
    else:
        return f"{s:.1f}s"

def get_device(device: str = 'auto') -> torch.device:
    """
    Get PyTorch device (CPU/CUDA).
    
    Args:
        device: Device selection ('auto'/'cpu'/'cuda')
        
    Returns:
        torch.device: Selected device
    """
    if device == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) 