"""
Experience replay memory used in DQN Agent.

This file contains the following classes:
- ReplayBuffer: Standard experience memory for DQN
- PrioritizedReplayBuffer: Prioritized experience memory for DQN
- EfficientReplayBuffer: Memory optimized for performance
- NStepReplayBuffer: Memory supporting N-step returns for Rainbow DQN
- NStepPrioritizedReplayBuffer: Combining N-step and PER for Rainbow DQN
"""

import numpy as np
import torch
import random
from collections import deque, namedtuple
from typing import Tuple, Optional, List, Union, Dict, Any

# Define namedtuple Experience to store experience in replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Standard experience memory.
    """
    def __init__(self, buffer_size: int, batch_size: int, device: str = 'cpu'):
        """
        Initialize ReplayBuffer.
        
        Args:
            buffer_size: Maximum size of buffer
            batch_size: Batch size when sampling
            device: Device to transfer tensors (cpu/cuda)
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
    
    def push(self, state, action, reward, next_state, done):
        """
        Store an experience in memory.
        
        Args:
            state: State
            action: Action
            reward: Reward
            next_state: Next state
            done: Terminal state flag (boolean)
        """
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def sample(self):
        """
        Sample randomly from memory.
        
        Returns:
            Tuple containing (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([e.done for e in experiences], dtype=np.uint8)).to(self.device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Returns the number of experiences in memory."""
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer (alias for push method).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Done flag
        """
        self.push(state, action, reward, next_state, done)

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer
    Prioritizes sampling experiences with high TD error
    """
    def __init__(self, buffer_size: int, batch_size: int, alpha: float = 0.6, 
                 beta_start: float = 0.4, beta_end: float = 1.0, 
                 beta_frames: int = 100000, device: str = 'cpu'):
        """
        Initialize Prioritized Replay Buffer
        
        Args:
            buffer_size: Maximum size of buffer
            batch_size: Batch size when sampling
            alpha: Coefficient determining priority level (0 = uniform sampling, 1 = fully prioritized)
            beta_start: Initial beta value for importance sampling weight
            beta_end: Final beta value
            beta_frames: Number of frames for beta to increase from start to end
            device: Device to transfer tensors (cpu/cuda)
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.beta_increment = (beta_end - beta_start) / beta_frames
        self.frame = 0
        self.device = device
        
        self.memory = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0
        
    def update_beta(self, batch_size: Optional[int] = None):
        """
        Update beta based on current frame
        
        Args:
            batch_size: New batch size if needed
        """
        self.frame += 1
        self.beta = min(self.beta_end, self.beta + self.beta_increment)
        
        if batch_size is not None:
            self.batch_size = batch_size
            
    def push(self, state, action, reward, next_state, done, error=None):
        """
        Add an experience to buffer with priority
        
        Args:
            state: State
            action: Action
            reward: Reward
            next_state: Next state
            done: Done flag
            error: TD error or priority, if None then max priority is used
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.memory) < self.buffer_size:
            self.memory.append(experience)
        else:
            self.memory[self.pos] = experience
            
        if error is None:
            # If no error is provided, use max priority or 1 if buffer is empty
            max_priority = self.priorities.max() if self.memory else 1.0
            self.priorities[self.pos] = max_priority
        else:
            # Add a small amount to ensure all experiences have a chance to be sampled
            self.priorities[self.pos] = (abs(error) + 1e-5) ** self.alpha
            
        self.pos = (self.pos + 1) % self.buffer_size
    
    def sample(self):
        """
        Sample a batch of experiences based on priority
        
        Returns:
            Tuple containing batch states, actions, rewards, next_states, dones, indices and weights
        """
        N = len(self.memory)
        if N == 0:
            return None
        
        if N < self.batch_size:
            batch_size = N
        else:
            batch_size = self.batch_size
            
        # Calculate sampling probabilities based on priority
        if N < self.buffer_size:
            priorities = self.priorities[:N]
        else:
            priorities = self.priorities
            
        probs = priorities[:N] / priorities[:N].sum()
        
        # Sample according to probability
        indices = np.random.choice(N, batch_size, replace=False, p=probs)
        
        # Calculate importance sampling weights
        weights = (N * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        batch = [self.memory[idx] for idx in indices]
        
        # Separate batch into components
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, errors):
        """
        Update priorities for experiences
        
        Args:
            indices: Indices to update
            errors: New TD errors
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
    
    def __len__(self):
        """Returns the number of experiences in memory."""
        return len(self.memory)

    def add(self, state, action, reward, next_state, done, error=None):
        """
        Add an experience to the buffer with priority (alias for push method).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Done flag
            error: TD error or priority, if None then max priority is used
        """
        self.push(state, action, reward, next_state, done, error)

class EfficientReplayBuffer:
    """
    Memory-efficient experience buffer using numpy arrays instead of deque.
    Optimizes performance and memory usage compared to standard ReplayBuffer.
    """
    def __init__(self, buffer_size: int, batch_size: int, 
                 state_shape: tuple, action_shape: tuple,
                 device: str = 'cpu'):
        """
        Initialize EfficientReplayBuffer.
        
        Args:
            buffer_size: Maximum size of buffer
            batch_size: Batch size when sampling
            state_shape: Shape of state (e.g., (84, 84, 4) for Atari)
            action_shape: Shape of action
            device: Device to transfer tensors (cpu/cuda)
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.count = 0
        self.current = 0
        
        # Initialize buffer with numpy arrays
        self.states = np.zeros((buffer_size,) + state_shape, dtype=np.float32)
        if isinstance(action_shape, int):
            self.actions = np.zeros((buffer_size,), dtype=np.int64)
        else:
            self.actions = np.zeros((buffer_size,) + action_shape, dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.next_states = np.zeros((buffer_size,) + state_shape, dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.uint8)
    
    def push(self, state, action, reward, next_state, done):
        """
        Store an experience in memory.
        
        Args:
            state: State
            action: Action
            reward: Reward
            next_state: Next state
            done: Terminal state flag (boolean)
        """
        # Store transition
        self.states[self.current] = state
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.next_states[self.current] = next_state
        self.dones[self.current] = done
        
        self.current = (self.current + 1) % self.buffer_size
        self.count = min(self.count + 1, self.buffer_size)
    
    def sample(self):
        """
        Sample randomly from memory.
        
        Returns:
            Tuple containing (states, actions, rewards, next_states, dones)
        """
        # Sample randomly from buffer
        if self.count < self.batch_size:
            indices = np.random.choice(self.count, self.count, replace=False)
        else:
            indices = np.random.choice(self.count, self.batch_size, replace=False)
        
        # Get data from buffer and convert to tensors
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Returns the number of experiences in memory."""
        return self.count 

class NStepReplayBuffer:
    """
    Experience memory supporting N-step returns.
    Stores sequences of n consecutive steps to calculate n-step return values.
    """
    def __init__(self, buffer_size: int, batch_size: int, n_step: int = 3, 
                 gamma: float = 0.99, device: str = 'cpu'):
        """
        Initialize NStepReplayBuffer.
        
        Args:
            buffer_size: Maximum size of buffer
            batch_size: Batch size when sampling
            n_step: Number of steps for n-step returns
            gamma: Discount factor
            device: Device to transfer tensors (cpu/cuda)
        """
        self.memory = deque(maxlen=buffer_size)
        self.n_step_buffer = deque(maxlen=n_step)
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.device = device
    
    def _get_n_step_info(self):
        """
        Calculate n-step reward and terminal state after n steps.
        
        Returns:
            Tuple: (reward, next_state, done)
        """
        reward, next_state, done = self.n_step_buffer[-1][2:]
        
        # If terminal, no need for further calculation
        if done:
            return reward, next_state, done
            
        # Calculate n-step return: reward + gamma * reward' + gamma^2 * reward'' + ...
        for idx in range(len(self.n_step_buffer) - 1, 0, -1):
            r, s, d = self.n_step_buffer[idx-1][2:]
            reward = r + self.gamma * (1 - d) * reward
            next_state, done = (s, d) if d else (next_state, done)
            
        return reward, next_state, done
        
    def push(self, state, action, reward, next_state, done):
        """
        Store an experience in n-step memory.
        
        Args:
            state: State
            action: Action
            reward: Reward
            next_state: Next state
            done: Terminal state flag (boolean)
        """
        experience = (state, action, reward, next_state, done)
        self.n_step_buffer.append(experience)
        
        # If n_step_buffer doesn't have enough elements, can't calculate n-step return yet
        if len(self.n_step_buffer) < self.n_step:
            return
            
        # Calculate n-step reward and state after n steps
        reward, next_state, done = self._get_n_step_info()
        
        # Get original state and action
        state, action = self.n_step_buffer[0][:2]
        
        # Store n-step experience
        self.memory.append((state, action, reward, next_state, done))
        
        # If terminal, clear n-step buffer
        if done:
            self.n_step_buffer.clear()
    
    def sample(self):
        """
        Sample randomly from memory.
        
        Returns:
            Tuple containing (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        states = torch.FloatTensor(np.array([e[0] for e in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in experiences])).to(self.device)
        dones = torch.FloatTensor(np.array([e[4] for e in experiences], dtype=np.uint8)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Returns the number of experiences in memory."""
        return len(self.memory)


class NStepPrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer with N-step returns.
    Combines multi-step returns with prioritized sampling based on TD error.
    
    Args:
        buffer_size: Maximum size of buffer
        batch_size: Size of each training batch
        n_step: Number of steps for n-step returns
        gamma: Discount factor
        alpha: Exponent determining how much prioritization is used
        beta_start: Initial value of beta for importance sampling weights
        beta_end: Final value of beta after annealing
        seed: Random seed
        device: Device to store tensors on
    """
    
    def __init__(self, buffer_size: int, batch_size: int, 
                n_step: int = 3, gamma: float = 0.99,
                alpha: float = 0.6, beta_start: float = 0.4, beta_end: float = 1.0,
                seed: Optional[int] = None, device: str = "cpu"):
        """Initialize a NStepPrioritizedReplayBuffer object."""
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        # Initialize buffers
        self.memory = []
        self.n_step_buffer = deque(maxlen=n_step)
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # For tracking metrics
        self.max_priority = 1.0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _get_n_step_info(self, reward: float, next_state, done: bool):
        """
        Get N-step return and final state.
        
        Args:
            reward: Current reward
            next_state: Current next state
            done: Current terminal flag
            
        Returns:
            Tuple of n-step reward, n-step next state, n-step done flag
        """
        # Add state transition to n-step buffer
        self.n_step_buffer.append((reward, next_state, done))
        
        # If buffer not full yet, return None
        if len(self.n_step_buffer) < self.n_step:
            return None, None, None
        
        # Calculate n-step rewards with discount
        n_reward = 0
        for i in range(self.n_step):
            r_i = self.n_step_buffer[i][0]
            n_reward += (self.gamma ** i) * r_i
        
        # Get final state and done flag
        _, n_next_state, n_done = self.n_step_buffer[-1]
        
        return n_reward, n_next_state, n_done
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory with n-step returns.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Done flag
        """
        # Get n-step returns
        n_reward, n_next_state, n_done = self._get_n_step_info(reward, next_state, done)
        
        # If n-step buffer not full yet, simply add to it and return
        if n_reward is None:
            return
            
        # Create experiences with 1-step and n-step returns
        experience = (state, action, reward, next_state, done, n_reward, n_next_state, n_done)
        
        # Add with max priority for new experiences
        priority = self.max_priority
        
        if self.size < self.buffer_size:
            self.memory.append(experience)
            self.size += 1
        else:
            self.memory[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, beta: Optional[float] = None):
        """
        Sample a batch of experiences from memory with priority weights.
        
        Args:
            beta: Beta parameter for importance sampling weights
            
        Returns:
            Tuple of experiences batch, n-step experiences batch, indices, importance sampling weights
        """
        if beta is None:
            beta = self.beta
        
        # Cannot sample more than we have in memory
        batch_size = min(self.batch_size, self.size)
        
        if self.size == 0:
            return None
        
        # Calculate sampling probabilities based on priorities
        if self.size < self.buffer_size:
            priorities = self.priorities[:self.size]
        else:
            priorities = self.priorities
        
        # Normalize by sum and apply exponent alpha
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs[:self.size])
        
        # Get experiences from memory
        experiences = [self.memory[i] for i in indices]
        
        # Prepare for converting to tensors
        states = np.vstack([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.vstack([e[2] for e in experiences])
        next_states = np.vstack([e[3] for e in experiences])
        dones = np.vstack([e[4] for e in experiences]).astype(np.uint8)
        
        # n-step returns
        n_rewards = np.vstack([e[5] for e in experiences])
        n_next_states = np.vstack([e[6] for e in experiences])
        n_dones = np.vstack([e[7] for e in experiences]).astype(np.uint8)
        
        # Convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        
        n_rewards = torch.from_numpy(n_rewards).float().to(self.device)
        n_next_states = torch.from_numpy(n_next_states).float().to(self.device)
        n_dones = torch.from_numpy(n_dones).float().to(self.device)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        weights = torch.from_numpy(weights).float().to(self.device)
        
        # Return 1-step experiences, n-step experiences, indices, and weights
        one_step = (states, actions, rewards, next_states, dones)
        n_step = (states, actions, n_rewards, n_next_states, n_dones)
        
        return one_step, n_step, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of sampled experiences
            priorities: New priorities for these experiences
        """
        priorities = priorities.squeeze()
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            if priority > self.max_priority:
                self.max_priority = priority
    
    def update_beta(self, step: int, total_steps: int):
        """
        Update beta parameter for importance sampling weights.
        
        Args:
            step: Current step
            total_steps: Total steps for annealing
        """
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * (step / total_steps)
        self.beta = min(1.0, self.beta)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return self.size 