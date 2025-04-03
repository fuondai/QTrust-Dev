"""
Anomaly Detection Module for Blockchain Systems

This module provides machine learning-based anomaly detection capabilities for blockchain networks.
It implements various detection models including AutoEncoder and LSTM architectures to identify
abnormal node behaviors, potential attacks, and suspicious activities in the network.
The module supports both batch and real-time anomaly detection with features for training,
model persistence, and statistical analysis of detected anomalies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Any, Union
from collections import deque
import random
import os
import time

class AnomalyDetector:
    """
    Anomaly detection class using Machine Learning for the HTDCM system.
    """
    def __init__(self, input_features: int = 10, hidden_size: int = 64, 
                 anomaly_threshold: float = 0.85, memory_size: int = 1000,
                 learning_rate: float = 0.001, device: str = None):
        """
        Initialize Anomaly Detector.
        
        Args:
            input_features: Number of input features from node behavior
            hidden_size: Hidden layer size of the model
            anomaly_threshold: Threshold for anomaly detection
            memory_size: Memory size for storing samples
            learning_rate: Learning rate of the model
            device: Computing device (None for auto-detection)
        """
        # Set computing device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        self.encoder = AutoEncoder(input_features, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Anomaly detection parameters
        self.input_features = input_features
        self.anomaly_threshold = anomaly_threshold
        
        # Memory for training
        self.memory = deque(maxlen=memory_size)
        self.batch_size = 32
        
        # Statistics
        self.anomaly_history = []
        self.normal_history = []
        self.detected_anomalies = 0
        self.false_positives = 0
        self.training_loss_history = []
        
        # State
        self.is_trained = False
        self.min_samples_for_training = 64
        self.reconstruction_errors = []
    
    def extract_features(self, node_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from node data for anomaly detection.
        
        Args:
            node_data: Data about node activities
            
        Returns:
            np.ndarray: Feature vector for the model input
        """
        features = []
        
        # 1. Transaction success rate
        total_txs = node_data.get('successful_txs', 0) + node_data.get('failed_txs', 0)
        success_rate = node_data.get('successful_txs', 0) / max(1, total_txs)
        features.append(success_rate)
        
        # 2. Response time statistics
        response_times = node_data.get('response_times', [])
        if response_times:
            features.append(np.mean(response_times))
            features.append(np.std(response_times))
            features.append(np.max(response_times))
            features.append(np.min(response_times))
        else:
            features.extend([0, 0, 0, 0])  # Default values
        
        # 3. Peer ratings
        peer_ratings = list(node_data.get('peer_ratings', {}).values())
        if peer_ratings:
            features.append(np.mean(peer_ratings))
            features.append(np.std(peer_ratings))
        else:
            features.extend([0.5, 0])  # Default values
        
        # 4. Number of malicious activities
        features.append(min(1.0, node_data.get('malicious_activities', 0) / 10.0))  # Normalize
        
        # 5. Recent activity pattern
        recent_activities = node_data.get('activity_history', [])
        success_history = [1.0 if act[0] == 'success' else 0.0 for act in recent_activities[-10:]]
        # Ensure there are 10 values
        success_history = success_history + [0.5] * (10 - len(success_history))
        features.extend(success_history)
        
        # Normalize and ensure correct size
        features = np.array(features, dtype=np.float32)
        if len(features) < self.input_features:
            features = np.pad(features, (0, self.input_features - len(features)), 'constant')
        elif len(features) > self.input_features:
            features = features[:self.input_features]
            
        return features
    
    def add_sample(self, node_data: Dict[str, Any], is_anomaly: bool = False):
        """
        Add a sample to memory for training.
        
        Args:
            node_data: Data about node activities
            is_anomaly: Whether it's a known anomaly
        """
        features = self.extract_features(node_data)
        self.memory.append((features, is_anomaly))
    
    def train(self, epochs: int = 5) -> float:
        """
        Train the anomaly detection model.
        
        Args:
            epochs: Number of epochs for training
            
        Returns:
            float: Average loss in the last training session
        """
        if len(self.memory) < self.min_samples_for_training:
            return 0.0  # Not enough data
            
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = 0
            
            # Get mostly normal samples (not anomalies) for training
            normal_samples = [(features, label) for features, label in self.memory if not label]
            if len(normal_samples) < self.batch_size // 2:
                samples = random.sample(self.memory, min(self.batch_size, len(self.memory)))
            else:
                samples = random.sample(normal_samples, min(self.batch_size, len(normal_samples)))
            
            # Batch division
            for i in range(0, len(samples), self.batch_size):
                batch = samples[i:i+self.batch_size]
                if len(batch) < 2:  # Too few samples
                    continue
                    
                features_batch = np.array([sample[0] for sample in batch])
                features_tensor = torch.FloatTensor(features_batch).to(self.device)
                
                # Train autoencoder
                self.optimizer.zero_grad()
                reconstructed = self.encoder(features_tensor)
                loss = self.criterion(reconstructed, features_tensor)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1
            
            # Save average epoch loss
            if batches > 0:
                avg_epoch_loss = epoch_loss / batches
                self.training_loss_history.append(avg_epoch_loss)
                losses.append(avg_epoch_loss)
        
        # Calculate anomaly detection thresholds based on reconstruction errors
        self._calculate_reconstruction_thresholds()
        
        self.is_trained = True
        return np.mean(losses) if losses else 0.0
    
    def _calculate_reconstruction_thresholds(self):
        """
        Calculate reconstruction error thresholds based on memory data.
        """
        if len(self.memory) < self.min_samples_for_training:
            return
            
        # Get normal samples
        normal_samples = [(features, label) for features, label in self.memory if not label]
        if not normal_samples:
            return
            
        features_batch = np.array([sample[0] for sample in normal_samples])
        features_tensor = torch.FloatTensor(features_batch).to(self.device)
        
        self.encoder.eval()
        with torch.no_grad():
            reconstructed = self.encoder(features_tensor)
            # Calculate reconstruction errors
            reconstruction_errors = F.mse_loss(reconstructed, features_tensor, reduction='none').mean(dim=1).cpu().numpy()
        self.encoder.train()
        
        self.reconstruction_errors = reconstruction_errors
        # Threshold is mean plus x times standard deviation
        self.anomaly_threshold = np.mean(reconstruction_errors) + 2.0 * np.std(reconstruction_errors)
    
    def detect_anomaly(self, node_data: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect anomalies from node data.
        
        Args:
            node_data: Data about node activities
            
        Returns:
            Tuple[bool, float, Dict[str, Any]]: 
                - Whether it's an anomaly
                - Anomaly score (higher means more likely to be an anomaly)
                - Detailed information about the detection
        """
        if not self.is_trained:
            # Train the model if not trained yet
            if len(self.memory) >= self.min_samples_for_training:
                self.train()
            else:
                # If not enough data for training, add sample and return False
                self.add_sample(node_data)
                return False, 0.0, {"message": "Not enough data for anomaly detection"}
        
        # Extract features
        features = self.extract_features(node_data)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Calculate reconstruction error
        self.encoder.eval()
        with torch.no_grad():
            reconstructed = self.encoder(features_tensor)
            reconstruction_error = F.mse_loss(reconstructed, features_tensor).item()
        self.encoder.train()
        
        # Detect anomaly if reconstruction error is greater than threshold
        is_anomaly = reconstruction_error > self.anomaly_threshold
        anomaly_score = reconstruction_error / self.anomaly_threshold if self.anomaly_threshold > 0 else 0.0
        
        # Add result to history
        if is_anomaly:
            self.anomaly_history.append((features, anomaly_score))
            self.detected_anomalies += 1
        else:
            self.normal_history.append(features)
            
        # Add sample to memory
        self.add_sample(node_data, is_anomaly)
        
        # Calculate additional information
        details = {
            "reconstruction_error": reconstruction_error,
            "anomaly_threshold": self.anomaly_threshold,
            "anomaly_score": anomaly_score,
            "is_trained": self.is_trained,
            "memory_size": len(self.memory)
        }
        
        return is_anomaly, anomaly_score, details
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about anomaly detection.
        
        Returns:
            Dict[str, Any]: Statistics of the anomaly detector
        """
        return {
            "detected_anomalies": self.detected_anomalies,
            "false_positives": self.false_positives,
            "is_trained": self.is_trained,
            "training_loss": self.training_loss_history[-1] if self.training_loss_history else 0.0,
            "memory_size": len(self.memory),
            "anomaly_threshold": self.anomaly_threshold,
            "recent_normal_samples": len(self.normal_history),
            "recent_anomaly_samples": len(self.anomaly_history)
        }
    
    def save_model(self, path: str = "models/anomaly_detector.pt"):
        """
        Save model to file.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'anomaly_threshold': self.anomaly_threshold,
            'input_features': self.input_features,
            'detected_anomalies': self.detected_anomalies,
            'is_trained': self.is_trained
        }, path)
    
    def load_model(self, path: str = "models/anomaly_detector.pt"):
        """
        Load model from file.
        
        Args:
            path: Path to load the model
        """
        if not os.path.exists(path):
            return False
            
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate model with correct number of features
        self.input_features = checkpoint['input_features']
        hidden_size = self.encoder.hidden_size
        self.encoder = AutoEncoder(self.input_features, hidden_size).to(self.device)
        
        # Load state
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = optim.Adam(self.encoder.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.anomaly_threshold = checkpoint['anomaly_threshold']
        self.detected_anomalies = checkpoint['detected_anomalies']
        self.is_trained = checkpoint['is_trained']
        
        return True


class AutoEncoder(nn.Module):
    """
    AutoEncoder model for anomaly detection.
    """
    def __init__(self, input_size: int, hidden_size: int = 64):
        """
        Initialize AutoEncoder.
        
        Args:
            input_size: Input size
            hidden_size: Hidden layer size
        """
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, input_size),
            nn.Sigmoid()  # Output values in range [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass data through autoencoder.
        
        Args:
            x: Input data
            
        Returns:
            torch.Tensor: Reconstructed data
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class LSTMAnomalyDetector(nn.Module):
    """
    LSTM model for time-series based anomaly detection.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        """
        Initialize LSTM Anomaly Detector.
        
        Args:
            input_size: Input size
            hidden_size: Hidden state size of LSTM
            num_layers: Number of stacked LSTM layers
        """
        super(LSTMAnomalyDetector, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Prediction layer
        self.linear = nn.Linear(hidden_size, input_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass data through LSTM.
        
        Args:
            x: Input time series (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Prediction for next time series
        """
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Get output from last time step
        last_time_step = lstm_out[:, -1, :]
        
        # Predict next value
        predictions = self.linear(last_time_step)
        
        return predictions


class MLBasedAnomalyDetectionSystem:
    """
    ML-based anomaly detection system integrating multiple methods.
    """
    def __init__(self, input_features: int = 20, time_series_length: int = 10):
        """
        Initialize anomaly detection system.
        
        Args:
            input_features: Number of features for AutoEncoder model
            time_series_length: Time series length for LSTM model
        """
        # AutoEncoder-based anomaly detection
        self.anomaly_detector = AnomalyDetector(input_features=input_features)
        
        # Time series based anomaly detection (to be implemented later)
        self.time_series_length = time_series_length
        
        # Store data history
        self.node_history = {}  # node_id -> data history
        
        # Statistics
        self.total_detections = 0
        self.detection_per_node = {}
        self.last_training_time = 0
        
    def process_node_data(self, node_id: int, node_data: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Process data from a node and detect anomalies.
        
        Args:
            node_id: Node ID
            node_data: Data about node activities
            
        Returns:
            Tuple[bool, float, Dict[str, Any]]: 
                - Whether it's an anomaly
                - Anomaly score (higher means more likely to be an anomaly)
                - Detailed information about the detection
        """
        # Save node data to history
        if node_id not in self.node_history:
            self.node_history[node_id] = []
        self.node_history[node_id].append(node_data)
        
        # Limit history
        if len(self.node_history[node_id]) > 100:
            self.node_history[node_id] = self.node_history[node_id][-100:]
            
        # Periodic training if needed
        current_time = time.time()
        if current_time - self.last_training_time > 300:  # 5 minutes
            self._periodic_training()
            self.last_training_time = current_time
            
        # Detect anomalies
        is_anomaly, score, details = self.anomaly_detector.detect_anomaly(node_data)
        
        # Update statistics
        if is_anomaly:
            self.total_detections += 1
            if node_id not in self.detection_per_node:
                self.detection_per_node[node_id] = 0
            self.detection_per_node[node_id] += 1
            
        # Add detailed information
        details["node_id"] = node_id
        details["detection_count"] = self.detection_per_node.get(node_id, 0)
        
        return is_anomaly, score, details
        
    def _periodic_training(self):
        """
        Periodically train anomaly detection models.
        """
        # Train autoencoder anomaly detector
        if len(self.anomaly_detector.memory) >= self.anomaly_detector.min_samples_for_training:
            self.anomaly_detector.train(epochs=3)
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the anomaly detection system.
        
        Returns:
            Dict[str, Any]: Statistics of the anomaly detection system
        """
        detector_stats = self.anomaly_detector.get_statistics()
        
        # Combined statistics
        return {
            "total_detections": self.total_detections,
            "nodes_with_anomalies": len(self.detection_per_node),
            "detector_stats": detector_stats,
            "top_anomalous_nodes": sorted(self.detection_per_node.items(), key=lambda x: x[1], reverse=True)[:5]
        } 