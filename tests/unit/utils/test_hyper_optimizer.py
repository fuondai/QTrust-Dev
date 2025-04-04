"""
Test file for the QTrust hyperparameter optimization module.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
from typing import Dict, Any, Tuple

# Add the project root directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrust.utils.hyper_optimizer import BayesianOptimizer, HyperParameterOptimizer

class SimpleDummyEnv:
    """Simple dummy environment for testing."""
    
    def __init__(self):
        self.state = np.zeros(4)
        self.step_count = 0
        self.max_steps = 100
    
    def reset(self):
        """Reset the environment."""
        self.state = np.random.rand(4)
        self.step_count = 0
        return self.state
    
    def step(self, action):
        """Take a step in the environment."""
        self.step_count += 1
        
        # Simple reward function that depends on the action
        if action == 0:
            reward = 0.1
        elif action == 1:
            reward = 0.2
        else:
            reward = -0.1
            
        # Move state randomly
        self.state = np.random.rand(4)
        
        # Check if done
        done = self.step_count >= self.max_steps
        
        return self.state, reward, done, {}

class SimpleDummyAgent:
    """Simple dummy agent for testing."""
    
    def __init__(self, params: Dict[str, float], device=None):
        self.params = params
        # Extract parameters with defaults
        self.learning_rate = params.get('learning_rate', 0.001)
        self.gamma = params.get('gamma', 0.99)
        self.epsilon = params.get('epsilon_start', 1.0)
        self.epsilon_decay = params.get('epsilon_decay', 0.995)
        self.epsilon_min = params.get('epsilon_end', 0.01)
    
    def act(self, state, epsilon=None):
        """Choose an action."""
        eps = epsilon if epsilon is not None else self.epsilon
        
        # Simple epsilon-greedy policy
        if np.random.rand() < eps:
            return np.random.randint(0, 3)  # Random action
        else:
            # Simple "optimal" policy based on parameters
            # For testing, we'll make better params give better actions
            if self.params.get('learning_rate', 0.001) > 0.005:
                return 1  # High reward action
            else:
                return 0  # Medium reward action
    
    def step(self, state, action, reward, next_state, done):
        """Process a step."""
        # Do nothing in this dummy agent
        pass
    
    def update_epsilon(self):
        """Update exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def dummy_objective(params: Dict[str, float]) -> float:
    """Simple objective function for testing BayesianOptimizer."""
    # Simple quadratic function with a known optimum
    x = params.get('x', 0)
    y = params.get('y', 0)
    
    # Optimum at (2, 3) with value 0
    return -((x - 2)**2 + (y - 3)**2)

class TestBayesianOptimizer(unittest.TestCase):
    """Tests for BayesianOptimizer class."""
    
    def setUp(self):
        """Set up for each test."""
        # Create a temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        
        # Define parameter ranges
        self.param_ranges = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        }
        
        # Create optimizer
        self.optimizer = BayesianOptimizer(
            param_ranges=self.param_ranges,
            objective_function=dummy_objective,
            n_initial_points=3,
            exploration_weight=0.1,
            log_dir=self.test_dir
        )
    
    def test_initialization(self):
        """Test initialization of BayesianOptimizer."""
        self.assertEqual(self.optimizer.param_ranges, self.param_ranges)
        self.assertEqual(self.optimizer.n_initial_points, 3)
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertEqual(len(self.optimizer.X), 0)
        self.assertEqual(len(self.optimizer.y), 0)
    
    def test_sample_random_point(self):
        """Test sampling random points."""
        point = self.optimizer._sample_random_point()
        self.assertEqual(len(point), 2)  # Two parameters: x and y
        
        # Check that points are within range
        self.assertGreaterEqual(point[0], -5.0)
        self.assertLessEqual(point[0], 5.0)
        self.assertGreaterEqual(point[1], -5.0)
        self.assertLessEqual(point[1], 5.0)
    
    def test_vector_to_params(self):
        """Test conversion from vector to params dict."""
        vector = np.array([1.5, 2.5])
        params = self.optimizer._vector_to_params(vector)
        
        self.assertEqual(params['x'], 1.5)
        self.assertEqual(params['y'], 2.5)
    
    def test_optimize(self):
        """Test optimization process with small number of iterations."""
        # Run optimization with few iterations for testing
        best_params = self.optimizer.optimize(n_iterations=2)
        
        # Check that we have results
        self.assertIsNotNone(best_params)
        self.assertIn('x', best_params)
        self.assertIn('y', best_params)
        
        # Check that history is updated
        self.assertEqual(len(self.optimizer.X), 2 + 3)  # 3 initial + 2 iterations
        self.assertEqual(len(self.optimizer.y), 2 + 3)
        
        # Check that results are saved
        files = os.listdir(self.test_dir)
        self.assertGreaterEqual(len(files), 1)  # At least one results file

class TestHyperParameterOptimizer(unittest.TestCase):
    """Tests for HyperParameterOptimizer class."""
    
    def setUp(self):
        """Set up for each test."""
        # Create a temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        
        # Define parameter ranges
        self.param_ranges = {
            'learning_rate': (0.001, 0.01),
            'gamma': (0.9, 0.99),
            'epsilon_start': (0.5, 1.0),
            'epsilon_decay': (0.9, 0.99),
            'epsilon_end': (0.01, 0.1)
        }
        
        # Create optimizer with reduced episode count for faster testing
        self.optimizer = HyperParameterOptimizer(
            env_creator=lambda: SimpleDummyEnv(),
            agent_creator=lambda params, device: SimpleDummyAgent(params, device),
            param_ranges=self.param_ranges,
            n_episodes_per_trial=2,  # Use very few episodes for testing
            log_dir=self.test_dir,
            n_eval_episodes=2
        )
    
    def test_initialization(self):
        """Test initialization of HyperParameterOptimizer."""
        self.assertEqual(self.optimizer.param_ranges, self.param_ranges)
        self.assertEqual(self.optimizer.n_episodes_per_trial, 2)
        self.assertTrue(os.path.exists(self.test_dir))
    
    def test_objective_function(self):
        """Test the objective function."""
        # Create a fixed params dictionary for testing
        params = {
            'learning_rate': 0.005,
            'gamma': 0.95,
            'epsilon_start': 0.8,
            'epsilon_decay': 0.95,
            'epsilon_end': 0.05
        }
        
        # Run the objective function once
        result = self.optimizer._objective_function(params)
        
        # Just check that it returns a value without error
        self.assertIsInstance(result, float)
    
    def test_optimize(self):
        """Test optimization process with minimal iterations."""
        # Run optimization with minimal iterations for quick testing
        best_params = self.optimizer.optimize(n_iterations=1)
        
        # Check that we have results
        self.assertIsNotNone(best_params)
        self.assertIn('learning_rate', best_params)
        self.assertIn('gamma', best_params)
        self.assertIn('epsilon_start', best_params)
        self.assertIn('epsilon_decay', best_params)
        self.assertIn('epsilon_end', best_params)

if __name__ == "__main__":
    unittest.main() 