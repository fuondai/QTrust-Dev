"""
Test file for the QTrust anomaly detection module.
"""

import os
import sys
import unittest
import numpy as np
import torch
import tempfile
from typing import Dict, List

# Add the project root directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrust.trust.anomaly_detection import (
    AnomalyDetector,
    AutoEncoder,
    LSTMAnomalyDetector,
    MLBasedAnomalyDetectionSystem
)

class TestAnomalyDetection(unittest.TestCase):
    """Tests for anomaly detection module."""
    
    def setUp(self):
        """Set up for each test."""
        # Disable CUDA for testing to ensure reproducibility
        self.original_cuda_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        
        # Create a detector with a small memory size for testing
        self.detector = AnomalyDetector(
            input_features=10,
            hidden_size=16,
            memory_size=100,
            learning_rate=0.01
        )
        
        # Create ML system
        self.ml_system = MLBasedAnomalyDetectionSystem(input_features=10)
        
        # Create a temporary directory for output files
        self.test_dir = tempfile.mkdtemp()
        
        # Generate sample node data
        self.normal_node_data = self._generate_node_data(successful_txs=90, failed_txs=10, is_normal=True)
        self.anomalous_node_data = self._generate_node_data(successful_txs=10, failed_txs=90, is_normal=False)
    
    def tearDown(self):
        """Clean up after each test."""
        # Restore original CUDA availability check
        torch.cuda.is_available = self.original_cuda_available
    
    def _generate_node_data(self, successful_txs=50, failed_txs=10, is_normal=True) -> Dict[str, any]:
        """Generate sample node data for testing."""
        # Generate random response times (normal nodes have lower response times)
        if is_normal:
            response_times = np.random.normal(20, 5, 50).tolist()  # Faster responses
        else:
            response_times = np.random.normal(100, 30, 50).tolist()  # Slower responses
        
        # Generate random peer ratings (normal nodes have higher ratings)
        peer_ratings = {}
        for i in range(10):
            if is_normal:
                peer_ratings[i] = 0.7 + 0.3 * np.random.random()  # Higher ratings
            else:
                peer_ratings[i] = 0.1 + 0.3 * np.random.random()  # Lower ratings
        
        # Generate random activity history (normal nodes have more successes)
        activity_history = []
        for i in range(20):
            if is_normal:
                success_prob = 0.9  # 90% success
            else:
                success_prob = 0.3  # 30% success
            
            if np.random.random() < success_prob:
                activity_history.append(('success', np.random.normal(20, 5), True))
            else:
                activity_history.append(('fail', np.random.normal(50, 10), True))
        
        # Create node data dictionary
        return {
            'node_id': 1,
            'shard_id': 0,
            'trust_score': 0.7 if is_normal else 0.3,
            'successful_txs': successful_txs,
            'failed_txs': failed_txs,
            'malicious_activities': 0 if is_normal else np.random.randint(1, 5),
            'response_times': response_times,
            'peer_ratings': peer_ratings,
            'activity_history': activity_history
        }
    
    def test_autoencoder_initialization(self):
        """Test AutoEncoder initialization and forward pass."""
        input_size = 10
        hidden_size = 8
        
        # Initialize AutoEncoder
        autoencoder = AutoEncoder(input_size, hidden_size)
        
        # Check layers
        self.assertEqual(autoencoder.input_size, input_size)
        self.assertEqual(autoencoder.hidden_size, hidden_size)
        
        # Test forward pass
        batch_size = 5
        x = torch.rand(batch_size, input_size)
        output = autoencoder(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, input_size))
    
    def test_lstm_anomaly_detector_initialization(self):
        """Test LSTMAnomalyDetector initialization and forward pass."""
        input_size = 10
        hidden_size = 8
        num_layers = 2
        
        # Initialize LSTM detector
        lstm_detector = LSTMAnomalyDetector(input_size, hidden_size, num_layers)
        
        # Check layers
        self.assertEqual(lstm_detector.input_size, input_size)
        self.assertEqual(lstm_detector.hidden_size, hidden_size)
        self.assertEqual(lstm_detector.num_layers, num_layers)
        
        # Test forward pass
        batch_size = 5
        seq_length = 20
        x = torch.rand(batch_size, seq_length, input_size)
        output = lstm_detector(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, input_size))
    
    def test_feature_extraction(self):
        """Test feature extraction from node data."""
        # Extract features
        features = self.detector.extract_features(self.normal_node_data)
        
        # Check feature properties
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (self.detector.input_features,))
        self.assertEqual(features.dtype, np.float32)
        
        # Features should contain success rate
        success_rate = self.normal_node_data['successful_txs'] / (self.normal_node_data['successful_txs'] + self.normal_node_data['failed_txs'])
        self.assertAlmostEqual(features[0], success_rate, places=5)
    
    def test_add_sample_and_memory(self):
        """Test adding samples to memory."""
        # Add multiple samples
        for _ in range(10):
            self.detector.add_sample(self.normal_node_data)
        
        for _ in range(5):
            self.detector.add_sample(self.anomalous_node_data, is_anomaly=True)
        
        # Check memory size
        self.assertEqual(len(self.detector.memory), 15)
        
        # Check sample format in memory
        for features, is_anomaly in self.detector.memory:
            self.assertIsInstance(features, np.ndarray)
            self.assertIsInstance(is_anomaly, bool)
    
    def test_detector_training(self):
        """Test training of the anomaly detector."""
        # Add samples to memory
        for _ in range(70):
            self.detector.add_sample(self.normal_node_data)
        
        for _ in range(10):
            self.detector.add_sample(self.anomalous_node_data, is_anomaly=True)
        
        # Train the detector
        loss = self.detector.train(epochs=2)
        
        # Check training results
        self.assertGreater(len(self.detector.training_loss_history), 0)
        self.assertTrue(self.detector.is_trained)
        self.assertGreater(self.detector.anomaly_threshold, 0)
    
    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        # Train the detector first
        for _ in range(70):
            self.detector.add_sample(self.normal_node_data)
        
        self.detector.train(epochs=3)
        
        # Test detection on normal data
        is_anomaly, score, details = self.detector.detect_anomaly(self.normal_node_data)
        self.assertIn(is_anomaly, [True, False])  # Check if is_anomaly is a boolean value
        self.assertIsInstance(score, float)
        self.assertIsInstance(details, dict)
        
        # Test detection on anomalous data
        is_anomaly, score, details = self.detector.detect_anomaly(self.anomalous_node_data)
        self.assertIn(is_anomaly, [True, False])  # Check if is_anomaly is a boolean value
        self.assertIsInstance(score, float)
        self.assertIsInstance(details, dict)
    
    def test_model_statistics(self):
        """Test model statistics functionality."""
        # Add samples to memory
        for _ in range(10):
            self.detector.add_sample(self.normal_node_data)
        
        # Get statistics
        stats = self.detector.get_statistics()
        
        # Check statistics structure
        self.assertIn("detected_anomalies", stats)
        self.assertIn("false_positives", stats)
        self.assertIn("is_trained", stats)
        self.assertIn("memory_size", stats)
        self.assertIn("anomaly_threshold", stats)
        
        # Check statistics values
        self.assertEqual(stats["memory_size"], 10)
        self.assertEqual(stats["is_trained"], self.detector.is_trained)
    
    def test_ml_system_integration(self):
        """Test ML-based anomaly detection system."""
        # Process multiple node data points
        for i in range(5):
            node_id = i
            is_anomaly, score, details = self.ml_system.process_node_data(node_id, self.normal_node_data)
            
            self.assertIsInstance(is_anomaly, bool)
            self.assertIsInstance(score, float)
            self.assertIsInstance(details, dict)
            self.assertEqual(details["node_id"], node_id)
        
        # Process anomalous data
        node_id = 99
        is_anomaly, score, details = self.ml_system.process_node_data(node_id, self.anomalous_node_data)
        
        # Check statistics
        stats = self.ml_system.get_statistics()
        self.assertIn("total_detections", stats)
        self.assertIn("nodes_with_anomalies", stats)
        self.assertIn("detector_stats", stats)
        self.assertIn("top_anomalous_nodes", stats)
    
    def test_periodic_training(self):
        """Test periodic training functionality."""
        # Add samples
        for i in range(70):
            self.ml_system.process_node_data(i % 5, self.normal_node_data)
        
        # Force training
        self.ml_system._periodic_training()
        
        # Check if detector is trained
        self.assertTrue(self.ml_system.anomaly_detector.is_trained)

if __name__ == "__main__":
    unittest.main() 