import unittest
import torch
import numpy as np
import os
import tempfile
from qtrust.agents.dqn.utils import (
    create_save_directory,
    soft_update,
    hard_update,
    calculate_td_error,
    calculate_huber_loss,
    exponential_decay,
    linear_decay,
    generate_timestamp,
    format_time,
    get_device
)

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def test_create_save_directory(self):
        test_dir = os.path.join(self.temp_dir, "test_models")
        created_dir = create_save_directory(test_dir)
        self.assertTrue(os.path.exists(created_dir))
        self.assertEqual(test_dir, created_dir)
        
    def test_soft_update(self):
        target = torch.nn.Linear(2, 2)
        local = torch.nn.Linear(2, 2)
        tau = 0.5
        
        # Save initial parameters
        initial_target_params = target.weight.data.clone()
        local_params = local.weight.data.clone()
        
        soft_update(target, local, tau)
        
        # Check if parameters were updated correctly
        expected_params = tau * local_params + (1 - tau) * initial_target_params
        self.assertTrue(torch.allclose(target.weight.data, expected_params))
        
    def test_hard_update(self):
        target = torch.nn.Linear(2, 2)
        local = torch.nn.Linear(2, 2)
        
        hard_update(target, local)
        
        # Check if parameters match exactly
        self.assertTrue(torch.equal(target.weight.data, local.weight.data))
        
    def test_calculate_td_error(self):
        current_q = torch.tensor([1.0, 2.0, 3.0])
        target_q = torch.tensor([1.5, 2.0, 2.5])
        
        td_error = calculate_td_error(current_q, target_q)
        
        expected_error = torch.tensor([0.5, 0.0, 0.5])
        self.assertTrue(torch.allclose(td_error, expected_error))
        
    def test_calculate_huber_loss(self):
        current_q = torch.tensor([1.0, 2.0, 3.0])
        target_q = torch.tensor([1.5, 2.0, 2.5])
        weights = torch.tensor([1.0, 0.5, 1.0])
        
        # Test without weights
        loss = calculate_huber_loss(current_q, target_q)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar output
        
        # Test with weights
        weighted_loss = calculate_huber_loss(current_q, target_q, weights)
        self.assertIsInstance(weighted_loss, torch.Tensor)
        self.assertEqual(weighted_loss.dim(), 0)  # Scalar output
        
    def test_exponential_decay(self):
        start_value = 1.0
        end_value = 0.1
        decay_rate = 0.95
        step = 10
        
        value = exponential_decay(start_value, end_value, decay_rate, step)
        
        self.assertGreaterEqual(value, end_value)
        self.assertLessEqual(value, start_value)
        
    def test_linear_decay(self):
        start_value = 1.0
        end_value = 0.1
        decay_steps = 10
        
        # Test at start
        value = linear_decay(start_value, end_value, decay_steps, 0)
        self.assertEqual(value, start_value)
        
        # Test at end
        value = linear_decay(start_value, end_value, decay_steps, decay_steps)
        self.assertAlmostEqual(value, end_value, places=7)  # Use assertAlmostEqual for floating point comparison
        
    def test_generate_timestamp(self):
        timestamp = generate_timestamp()
        self.assertIsInstance(timestamp, str)
        self.assertEqual(len(timestamp), 15)  # Format: YYYYMMDD_HHMMSS
        
    def test_format_time(self):
        # Test seconds
        self.assertEqual(format_time(45.5), "45.5s")
        
        # Test minutes
        self.assertEqual(format_time(125), "2m 5s")
        
        # Test hours
        self.assertEqual(format_time(3665), "1h 1m 5s")
        
    def test_get_device(self):
        # Test auto device selection
        device = get_device('auto')
        self.assertIsInstance(device, torch.device)
        
        # Test CPU device
        device = get_device('cpu')
        self.assertEqual(device, torch.device('cpu'))

if __name__ == '__main__':
    unittest.main() 