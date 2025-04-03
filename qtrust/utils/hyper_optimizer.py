"""
Hyperparameter optimization module for QTrust system.

This module provides Bayesian optimization tools for hyperparameter tuning of machine learning models,
particularly for Deep Q-Networks (DQN) in reinforcement learning settings. It includes implementations
of Bayesian optimization with Gaussian Processes, acquisition functions, and utilities for tracking 
and visualizing the optimization process.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Callable
import os
import json
from datetime import datetime
import torch

class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameters.
    """
    def __init__(self,
                param_ranges: Dict[str, Tuple[float, float]],
                objective_function: Callable,
                n_initial_points: int = 5,
                exploration_weight: float = 0.1,
                log_dir: str = 'logs/hyperopt',
                maximize: bool = True):
        """
        Initialize BayesianOptimizer.
        
        Args:
            param_ranges: Dictionary containing hyperparameter ranges
            objective_function: Objective function to optimize
            n_initial_points: Number of initial points
            exploration_weight: Weight for exploration
            log_dir: Directory to save logs
            maximize: True if maximizing objective, False if minimizing
        """
        self.param_ranges = param_ranges
        self.objective_function = objective_function
        self.n_initial_points = n_initial_points
        self.exploration_weight = exploration_weight
        self.log_dir = log_dir
        self.maximize = maximize
        
        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Save search history
        self.X = []  # Points tried
        self.y = []  # Corresponding objective values
        
        # GP model parameters
        self.length_scale = 1.0
        self.noise = 1e-5
        
        # Optimization information
        self.best_params = None
        self.best_value = -float('inf') if maximize else float('inf')
    
    def optimize(self, n_iterations: int = 30) -> Dict[str, float]:
        """
        Perform Bayesian optimization.
        
        Args:
            n_iterations: Number of optimization iterations
            
        Returns:
            Dict[str, float]: Best hyperparameters
        """
        # Initialize with starting points
        self._initialize()
        
        # Main loop for Bayesian optimization
        for i in range(n_iterations):
            # Select next point to evaluate
            next_point = self._select_next_point()
            
            # Evaluate objective function at that point
            params_dict = self._vector_to_params(next_point)
            value = self.objective_function(params_dict)
            
            # Update history
            self.X.append(next_point)
            self.y.append(value)
            
            # Update best point
            if (self.maximize and value > self.best_value) or \
               (not self.maximize and value < self.best_value):
                self.best_value = value
                self.best_params = params_dict
            
            # Log progress
            self._log_iteration(i, params_dict, value)
            
            # Update exploration weight
            self.exploration_weight *= 0.95  # Gradually reduce exploration
        
        # Save and return best results
        self._save_results()
        return self.best_params
    
    def _initialize(self):
        """
        Initialize with random starting points.
        """
        for _ in range(self.n_initial_points):
            # Generate a random point in parameter space
            point = self._sample_random_point()
            
            # Evaluate objective function
            params_dict = self._vector_to_params(point)
            value = self.objective_function(params_dict)
            
            # Update history
            self.X.append(point)
            self.y.append(value)
            
            # Update best point
            if (self.maximize and value > self.best_value) or \
               (not self.maximize and value < self.best_value):
                self.best_value = value
                self.best_params = params_dict
    
    def _sample_random_point(self) -> np.ndarray:
        """
        Sample a random point in parameter space.
        
        Returns:
            np.ndarray: Random point
        """
        point = []
        for param_name, (lower, upper) in self.param_ranges.items():
            # Sample from uniform distribution
            value = np.random.uniform(lower, upper)
            point.append(value)
        
        return np.array(point)
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, float]:
        """
        Convert vector to parameter dict.
        
        Args:
            vector: Vector of parameter values
            
        Returns:
            Dict[str, float]: Parameter dictionary
        """
        params = {}
        for i, (param_name, _) in enumerate(self.param_ranges.items()):
            params[param_name] = vector[i]
        
        return params
    
    def _params_to_vector(self, params: Dict[str, float]) -> np.ndarray:
        """
        Convert parameter dict to vector.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            np.ndarray: Vector of parameter values
        """
        vector = []
        for param_name in self.param_ranges.keys():
            vector.append(params[param_name])
        
        return np.array(vector)
    
    def _select_next_point(self) -> np.ndarray:
        """
        Select next point to evaluate based on acquisition function.
        
        Returns:
            np.ndarray: Next point
        """
        if len(self.X) == 0:
            return self._sample_random_point()
        
        # Convert to numpy arrays
        X = np.array(self.X)
        y = np.array(self.y)
        
        # If minimizing, negate y
        if not self.maximize:
            y = -y
        
        # Implement model-based decision support algorithm
        # Use Expected Improvement (EI) as acquisition function
        
        # Sample random points
        n_samples = 1000
        candidates = np.random.uniform(
            low=[lower for _, (lower, _) in self.param_ranges.items()],
            high=[upper for _, (_, upper) in self.param_ranges.items()],
            size=(n_samples, len(self.param_ranges))
        )
        
        # Calculate GP posterior for each candidate
        ei_values = []
        for candidate in candidates:
            mean, std = self._gp_posterior(candidate, X, y)
            
            # Calculate Expected Improvement
            improvement = mean - np.max(y)
            Z = improvement / (std + 1e-9)
            ei = improvement * (0.5 + 0.5 * np.tanh(Z / np.sqrt(2))) + \
                 std * np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi)
            
            # Add penalty term to encourage exploration
            distance_penalty = -self._min_distance(candidate, X)
            
            # Combine
            acquisition = ei + self.exploration_weight * distance_penalty
            ei_values.append(acquisition)
        
        # Select best point
        best_idx = np.argmax(ei_values)
        return candidates[best_idx]
    
    def _gp_posterior(self, x: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Calculate posterior from Gaussian Process.
        
        Args:
            x: Point to calculate posterior for
            X: Evaluated points
            y: Evaluated values
            
        Returns:
            Tuple[float, float]: (mean, standard deviation)
        """
        # Calculate kernel matrix
        K = self._rbf_kernel(X, X)
        K += np.eye(len(X)) * self.noise  # Add noise
        
        # Calculate kernel between x and X
        k = self._rbf_kernel(x.reshape(1, -1), X).flatten()
        
        # Calculate kernel of x with itself
        kxx = self._rbf_kernel(x.reshape(1, -1), x.reshape(1, -1)).flatten()[0]
        
        # Calculate posterior
        K_inv = np.linalg.inv(K)
        mean = k.dot(K_inv).dot(y)
        var = kxx - k.dot(K_inv).dot(k)
        std = np.sqrt(max(1e-8, var))  # Avoid negative values due to numerical errors
        
        return mean, std
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        RBF kernel function.
        
        Args:
            X1: First set of points
            X2: Second set of points
            
        Returns:
            np.ndarray: Kernel matrix
        """
        sq_dists = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return np.exp(-0.5 * sq_dists / self.length_scale**2)
    
    def _min_distance(self, x: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate minimum distance from x to set X.
        
        Args:
            x: Point to calculate from
            X: Set of points
            
        Returns:
            float: Minimum distance
        """
        if len(X) == 0:
            return float('inf')
        
        dists = np.sqrt(np.sum((X - x)**2, axis=1))
        return np.min(dists)
    
    def _log_iteration(self, iteration: int, params: Dict[str, float], value: float):
        """
        Log information for each iteration.
        
        Args:
            iteration: Iteration number
            params: Evaluated parameters
            value: Objective value
        """
        print(f"Iteration {iteration+1}:")
        for param_name, param_value in params.items():
            print(f"  {param_name} = {param_value:.6f}")
        print(f"  Value = {value:.6f}")
        if (self.maximize and value == self.best_value) or (not self.maximize and value == self.best_value):
            print(f"  (New best value!)")
        print()
    
    def _save_results(self):
        """
        Save optimization results.
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save optimization information
        results = {
            'best_params': self.best_params,
            'best_value': float(self.best_value),
            'history': {
                'params': [self._vector_to_params(x) for x in self.X],
                'values': [float(y) for y in self.y]
            },
            'timestamp': timestamp,
            'maximize': self.maximize
        }
        
        # Save to file
        results_path = os.path.join(self.log_dir, f"hyperopt_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save plot
        self._plot_optimization_history(os.path.join(self.log_dir, f"hyperopt_history_{timestamp}.png"))
        
        print(f"Optimization results have been saved to {results_path}")
    
    def _plot_optimization_history(self, save_path: str):
        """
        Plot optimization history.
        
        Args:
            save_path: Path to save the figure
        """
        # Copy y values
        y_values = self.y.copy()
        
        # If minimizing, negate y
        if not self.maximize:
            best_vals = [-float('inf')]
            for y in y_values:
                best_vals.append(min(best_vals[-1], y) if best_vals[-1] != -float('inf') else y)
            best_vals = best_vals[1:]
        else:
            best_vals = [-float('inf')]
            for y in y_values:
                best_vals.append(max(best_vals[-1], y) if best_vals[-1] != -float('inf') else y)
            best_vals = best_vals[1:]
        
        plt.figure(figsize=(12, 6))
        
        # Plot values at each iteration
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(y_values) + 1), y_values, 'o-', label='Observed values')
        plt.xlabel('Iteration')
        plt.ylabel('Objective value')
        plt.title('Objective Value at Each Iteration')
        plt.grid(True)
        
        # Plot best value over iterations
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(best_vals) + 1), best_vals, 'o-', color='green', label='Best value')
        plt.xlabel('Iteration')
        plt.ylabel('Best objective value')
        plt.title('Best Objective Value Over Iterations')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


class HyperParameterOptimizer:
    """
    Hyperparameter optimization for DQN Agent.
    """
    def __init__(self, 
                env_creator: Callable,
                agent_creator: Callable,
                param_ranges: Dict[str, Tuple[float, float]] = None,
                n_episodes_per_trial: int = 200,
                log_dir: str = 'logs/hyperopt',
                n_eval_episodes: int = 20,
                maximize: bool = True,
                device: str = 'auto'):
        """
        Initialize HyperParameterOptimizer.
        
        Args:
            env_creator: Environment creator function
            agent_creator: Agent creator function with parameters
            param_ranges: Dictionary containing hyperparameter ranges
            n_episodes_per_trial: Number of episodes for each trial
            log_dir: Directory to save logs
            n_eval_episodes: Number of episodes for evaluation
            maximize: True if maximizing score, False if minimizing loss
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        self.env_creator = env_creator
        self.agent_creator = agent_creator
        self.n_episodes_per_trial = n_episodes_per_trial
        self.log_dir = log_dir
        self.n_eval_episodes = n_eval_episodes
        self.maximize = maximize
        
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Default parameter ranges if not provided
        if param_ranges is None:
            self.param_ranges = {
                'learning_rate': (1e-5, 1e-2),
                'gamma': (0.9, 0.999),
                'tau': (1e-4, 1e-2),
                'alpha': (0.3, 0.9),
                'beta_start': (0.2, 0.6),
                'eps_end': (0.01, 0.2),
                'eps_decay': (0.9, 0.999)
            }
        else:
            self.param_ranges = param_ranges
        
        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create optimizer
        self.optimizer = BayesianOptimizer(
            param_ranges=self.param_ranges,
            objective_function=self._objective_function,
            log_dir=log_dir,
            maximize=maximize
        )
    
    def optimize(self, n_iterations: int = 20) -> Dict[str, float]:
        """
        Perform hyperparameter optimization.
        
        Args:
            n_iterations: Number of optimization iterations
            
        Returns:
            Dict[str, float]: Best hyperparameters
        """
        return self.optimizer.optimize(n_iterations)
    
    def _objective_function(self, params: Dict[str, float]) -> float:
        """
        Objective function to optimize.
        
        Args:
            params: Dictionary containing hyperparameters
            
        Returns:
            float: Objective value (reward or negative loss)
        """
        # Create environment
        env = self.env_creator()
        
        # Create agent with params
        agent = self.agent_creator(params, self.device)
        
        # Train agent
        total_rewards = []
        
        for episode in range(self.n_episodes_per_trial):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                agent.step(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            
            # Update epsilon
            agent.update_epsilon()
        
        # Evaluate agent
        eval_rewards = []
        
        for _ in range(self.n_eval_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.act(state, 0.0)  # No exploration during evaluation
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
        
        avg_eval_reward = np.mean(eval_rewards)
        
        # Log results
        print(f"Params: {params}")
        print(f"Average Training Reward: {np.mean(total_rewards):.2f}")
        print(f"Average Evaluation Reward: {avg_eval_reward:.2f}")
        
        if not self.maximize:
            # If minimizing loss, use negative reward
            return -avg_eval_reward
        
        return avg_eval_reward

def optimize_dqn_hyperparameters(env_creator, agent_creator, param_ranges=None, 
                               n_iterations=20, n_episodes_per_trial=200,
                               log_dir='logs/hyperopt'):
    """
    Helper function to optimize hyperparameters for DQN.
    
    Args:
        env_creator: Environment creator function
        agent_creator: Agent creator function
        param_ranges: Parameter ranges
        n_iterations: Number of optimization iterations
        n_episodes_per_trial: Number of episodes per trial
        log_dir: Directory to save logs
        
    Returns:
        Dict[str, float]: Best hyperparameters
    """
    optimizer = HyperParameterOptimizer(
        env_creator=env_creator,
        agent_creator=agent_creator,
        param_ranges=param_ranges,
        n_episodes_per_trial=n_episodes_per_trial,
        log_dir=log_dir
    )
    
    best_params = optimizer.optimize(n_iterations)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(log_dir, f"best_dqn_params_{timestamp}.json")
    
    with open(results_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Best hyperparameters have been saved to {results_path}")
    return best_params 