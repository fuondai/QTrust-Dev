"""
Performance evaluation metrics module for QTrust system.

This module provides functions for evaluating the performance of the QTrust blockchain system,
including transaction throughput, latency, energy efficiency, security metrics, 
cross-shard transaction analysis, and various visualization utilities for performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union
import os

from .paths import get_chart_path  # Added import

def calculate_transaction_throughput(successful_txs: int, total_time: float) -> float:
    """
    Calculate transaction throughput.
    
    Args:
        successful_txs: Number of successful transactions
        total_time: Total time (ms)
        
    Returns:
        float: Transaction throughput (transactions/ms)
    """
    if total_time == 0:
        return 0.0
    return successful_txs / total_time

def calculate_latency_metrics(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate latency metrics.
    
    Args:
        latencies: List of transaction latencies (ms)
        
    Returns:
        Dict: Latency metrics (average, median, maximum, etc.)
    """
    if not latencies:
        return {
            'avg_latency': 0.0,
            'median_latency': 0.0,
            'min_latency': 0.0,
            'max_latency': 0.0,
            'p95_latency': 0.0,
            'p99_latency': 0.0
        }
    
    return {
        'avg_latency': np.mean(latencies),
        'median_latency': np.median(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies),
        'p95_latency': np.percentile(latencies, 95),
        'p99_latency': np.percentile(latencies, 99)
    }

def calculate_energy_efficiency(energy_consumption: float, successful_txs: int) -> float:
    """
    Calculate energy efficiency.
    
    Args:
        energy_consumption: Total energy consumption
        successful_txs: Number of successful transactions
        
    Returns:
        float: Energy consumption per successful transaction
    """
    if successful_txs == 0:
        return float('inf')
    return energy_consumption / successful_txs

def calculate_security_metrics(
    trust_scores: Dict[int, float], 
    malicious_nodes: List[int]
) -> Dict[str, float]:
    """
    Calculate security metrics.
    
    Args:
        trust_scores: Trust scores of nodes
        malicious_nodes: List of IDs of malicious nodes
        
    Returns:
        Dict: Security metrics
    """
    total_nodes = len(trust_scores)
    if total_nodes == 0:
        return {
            'avg_trust': 0.0,
            'malicious_ratio': 0.0,
            'trust_variance': 0.0
        }
    
    avg_trust = np.mean(list(trust_scores.values()))
    trust_variance = np.var(list(trust_scores.values()))
    malicious_ratio = len(malicious_nodes) / total_nodes if total_nodes > 0 else 0
    
    return {
        'avg_trust': avg_trust,
        'malicious_ratio': malicious_ratio,
        'trust_variance': trust_variance
    }

def calculate_cross_shard_metrics(
    cross_shard_txs: int, 
    total_txs: int, 
    cross_shard_latencies: List[float],
    intra_shard_latencies: List[float]
) -> Dict[str, float]:
    """
    Calculate cross-shard transaction metrics.
    
    Args:
        cross_shard_txs: Number of cross-shard transactions
        total_txs: Total number of transactions
        cross_shard_latencies: Latencies of cross-shard transactions
        intra_shard_latencies: Latencies of intra-shard transactions
        
    Returns:
        Dict: Cross-shard transaction metrics
    """
    cross_shard_ratio = cross_shard_txs / total_txs if total_txs > 0 else 0
    
    cross_shard_avg_latency = np.mean(cross_shard_latencies) if cross_shard_latencies else 0
    intra_shard_avg_latency = np.mean(intra_shard_latencies) if intra_shard_latencies else 0
    
    latency_overhead = (cross_shard_avg_latency / intra_shard_avg_latency) if intra_shard_avg_latency > 0 else 0
    
    return {
        'cross_shard_ratio': cross_shard_ratio,
        'cross_shard_avg_latency': cross_shard_avg_latency,
        'intra_shard_avg_latency': intra_shard_avg_latency,
        'latency_overhead': latency_overhead
    }

def generate_performance_report(metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate performance report from metrics.
    
    Args:
        metrics: Dictionary containing performance metrics
        
    Returns:
        pd.DataFrame: Performance report in tabular format
    """
    report = pd.DataFrame({
        'Metric': [
            'Throughput (tx/s)',
            'Average Latency (ms)',
            'Median Latency (ms)',
            'P95 Latency (ms)',
            'Energy per Transaction',
            'Average Trust Score',
            'Malicious Node Ratio',
            'Cross-Shard Transaction Ratio',
            'Cross-Shard Latency Overhead'
        ],
        'Value': [
            metrics.get('throughput', 0) * 1000,  # Convert from tx/ms to tx/s
            metrics.get('latency', {}).get('avg_latency', 0),
            metrics.get('latency', {}).get('median_latency', 0),
            metrics.get('latency', {}).get('p95_latency', 0),
            metrics.get('energy_per_tx', 0),
            metrics.get('security', {}).get('avg_trust', 0),
            metrics.get('security', {}).get('malicious_ratio', 0),
            metrics.get('cross_shard', {}).get('cross_shard_ratio', 0),
            metrics.get('cross_shard', {}).get('latency_overhead', 0)
        ]
    })
    
    return report

def plot_trust_distribution(trust_scores: Dict[int, float], 
                           malicious_nodes: List[int], 
                           title: str = "Trust Score Distribution",
                           save_path: Optional[str] = None):
    """
    Plot trust score distribution of nodes.
    
    Args:
        trust_scores: Trust scores of nodes
        malicious_nodes: List of IDs of malicious nodes
        title: Title of the plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create trust score lists for each node group
    normal_nodes = [node_id for node_id in trust_scores if node_id not in malicious_nodes]
    
    normal_scores = [trust_scores[node_id] for node_id in normal_nodes]
    malicious_scores = [trust_scores[node_id] for node_id in malicious_nodes if node_id in trust_scores]
    
    # Plot histogram
    sns.histplot(normal_scores, color='green', alpha=0.5, label='Normal Nodes', bins=20)
    if malicious_scores:
        sns.histplot(malicious_scores, color='red', alpha=0.5, label='Malicious Nodes', bins=20)
    
    plt.title(title)
    plt.xlabel('Trust Score')
    plt.ylabel('Number of Nodes')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_performance_comparison(results: Dict[str, Dict[str, List[float]]], 
                               metric_name: str,
                               title: str,
                               ylabel: str,
                               save_path: Optional[str] = None):
    """
    Plot performance comparison between methods.
    
    Args:
        results: Dictionary containing results from different methods
        metric_name: Name of the metric to compare
        title: Title of the plot
        ylabel: Y-axis label
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create data for boxplot
    data = []
    labels = []
    
    for method_name, method_results in results.items():
        if metric_name in method_results:
            data.append(method_results[metric_name])
            labels.append(method_name)
    
    # Plot boxplot
    box = plt.boxplot(data, patch_artist=True, labels=labels)
    
    # Set colors for boxes
    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
    for patch, color in zip(box['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_time_series(data: List[float], 
                    title: str, 
                    xlabel: str, 
                    ylabel: str,
                    window: int = 10,
                    save_path: Optional[str] = None):
    """
    Plot time series data.
    
    Args:
        data: Data to plot
        title: Title of the plot
        xlabel: X-axis label
        ylabel: Y-axis label
        window: Window size for moving average
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot raw data
    plt.plot(data, alpha=0.5, label='Raw Data')
    
    # Plot moving average
    if len(data) >= window:
        moving_avg = pd.Series(data).rolling(window=window).mean().values
        plt.plot(range(window-1, len(data)), moving_avg[window-1:], 'r-', linewidth=2, label=f'Moving Average (window={window})')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_heatmap(data: np.ndarray, 
                x_labels: List[str], 
                y_labels: List[str],
                title: str,
                save_path: Optional[str] = None):
    """
    Plot heatmap.
    
    Args:
        data: Matrix data
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        title: Title of the plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(data, annot=True, cmap='viridis', xticklabels=x_labels, yticklabels=y_labels)
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def calculate_throughput(successful_txs: int, total_time: float) -> float:
    """
    Calculate transaction throughput.
    
    Args:
        successful_txs: Number of successful transactions
        total_time: Total time (s)
        
    Returns:
        float: Transaction throughput (transactions/second)
    """
    if total_time == 0:
        return 0.0
    return successful_txs / total_time

def calculate_cross_shard_transaction_ratio(cross_shard_txs: int, total_txs: int) -> float:
    """
    Calculate cross-shard transaction ratio.
    
    Args:
        cross_shard_txs: Number of cross-shard transactions
        total_txs: Total number of transactions
        
    Returns:
        float: Cross-shard transaction ratio
    """
    if total_txs == 0:
        return 0.0
    return cross_shard_txs / total_txs

def plot_performance_metrics(metrics: Dict[str, List[float]], 
                            title: str = "Performance Metrics",
                            figsize: Tuple[int, int] = (10, 8),
                            save_path: Optional[str] = None):
    """
    Plot performance metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Title of the plot
        figsize: Figure size (width, height)
        save_path: Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    n_metrics = len(metrics)
    if n_metrics == 0:
        return
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_metrics))
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.subplot(n_metrics, 1, i + 1)
        plt.plot(values, marker='o', linestyle='-', color=colors[i])
        plt.title(f"{metric_name}")
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        # Use get_chart_path to get the full path
        full_path = get_chart_path(save_path, "metrics")
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Performance metrics plot saved to {full_path}")
    else:
        plt.show()

def plot_comparison_charts(comparison_data: Dict[str, Dict[str, float]], 
                          metrics: List[str],
                          title: str = "Performance Comparison",
                          save_path: Optional[str] = None):
    """
    Plot performance comparison between methods.
    
    Args:
        comparison_data: Dictionary containing comparison data between methods
        metrics: List of metrics to compare
        title: Title of the plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    num_metrics = len(metrics)
    rows = (num_metrics + 1) // 2
    cols = min(2, num_metrics)
    
    method_names = list(comparison_data.keys())
    
    for i, metric in enumerate(metrics):
        plt.subplot(rows, cols, i+1)
        
        # Prepare data
        values = [comparison_data[method][metric] for method in method_names]
        
        # Plot bar chart
        bars = plt.bar(method_names, values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title(metric)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_metrics_over_time(metrics_over_time: Dict[str, List[float]],
                           labels: List[str],
                           title: str = "Metrics Over Time",
                           xlabel: str = "Step",
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None):
    """
    Plot metrics over time.
    
    Args:
        metrics_over_time: Dictionary containing time series data for metrics
        labels: Labels for each metric
        title: Title of the plot
        xlabel: X-axis label
        figsize: Size of the figure
        save_path: Path to save the plot
    """
    plt.figure(figsize=figsize)
    
    for i, (metric_name, values) in enumerate(metrics_over_time.items()):
        plt.subplot(len(metrics_over_time), 1, i+1)
        plt.plot(values)
        plt.ylabel(labels[i] if i < len(labels) else metric_name)
        if i == len(metrics_over_time) - 1:
            plt.xlabel(xlabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
    
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

class SecurityMetrics:
    """
    Class for tracking and analyzing security metrics in the QTrust system.
    """
    def __init__(self, window_size: int = 20):
        """
        Initialize SecurityMetrics object.
        
        Args:
            window_size: Analysis window size for security metrics
        """
        self.analysis_window = window_size
        self.thresholds = {
            "51_percent": 0.7,
            "ddos": 0.65,
            "mixed": 0.6,
            "selfish_mining": 0.75,
            "bribery": 0.7
        }
        self.weights = {
            "51_percent": {
                "failed_tx_ratio": 0.3,
                "high_value_tx_failure": 0.5,
                "node_trust_variance": 0.2
            },
            "ddos": {
                "latency_deviation": 0.5,
                "failed_tx_ratio": 0.3,
                "network_congestion": 0.2
            },
            "mixed": {
                "failed_tx_ratio": 0.25,
                "latency_deviation": 0.25,
                "node_trust_variance": 0.25,
                "high_value_tx_failure": 0.25
            },
            "selfish_mining": {
                "block_withholding": 0.6,
                "fork_rate": 0.4
            },
            "bribery": {
                "voting_deviation": 0.5,
                "trust_inconsistency": 0.5
            }
        }
        
        # History of attack metrics
        self.history = {
            "attack_indicators": [],  # List of attack indicators over time
            "detected_attacks": [],   # List of detected attacks
            "node_trust_variance": [],  # History of trust variance
            "latency_deviation": [],  # History of latency deviation
            "failed_tx_ratio": [],    # History of failed transaction ratio
            "security_metrics": []    # History of security metrics
        }
        
        # Current attack indicators
        self.attack_indicators = {
            "51_percent": 0.0,
            "ddos": 0.0,
            "mixed": 0.0,
            "selfish_mining": 0.0,
            "bribery": 0.0
        }
        
        # Current attack state
        self.current_attack = None
        self.attack_confidence = 0.0
        
    def calculate_attack_indicators(self, 
                                  failed_tx_ratio: float,
                                  node_trust_variance: float,
                                  latency_deviation: float,
                                  network_metrics: Dict[str, Any],
                                  transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate indicators for each attack type.
        
        Args:
            failed_tx_ratio: Ratio of failed transactions
            node_trust_variance: Variance of trust scores between nodes
            latency_deviation: Deviation of latency
            network_metrics: Network metrics
            transactions: List of recent transactions
            
        Returns:
            Dict[str, float]: Attack indicators
        """
        indicators = {}
        
        # Calculate 51% attack indicator
        indicators['51_percent'] = self._calculate_51_percent_indicator(
            failed_tx_ratio, node_trust_variance, transactions)
        
        # Calculate DDoS attack indicator
        indicators['ddos'] = self._calculate_ddos_indicator(
            failed_tx_ratio, latency_deviation, network_metrics)
        
        # Calculate mixed attack indicator
        indicators['mixed'] = self._calculate_mixed_indicator(
            failed_tx_ratio, node_trust_variance, latency_deviation, network_metrics, transactions)
        
        # Calculate selfish mining attack indicator
        indicators['selfish_mining'] = self._calculate_selfish_mining_indicator(
            network_metrics, transactions)
        
        # Calculate bribery attack indicator
        indicators['bribery'] = self._calculate_bribery_indicator(
            node_trust_variance, network_metrics, transactions)
        
        # Update history
        self.history["attack_indicators"].append(indicators.copy())
        self.history["node_trust_variance"].append(node_trust_variance)
        self.history["latency_deviation"].append(latency_deviation)
        self.history["failed_tx_ratio"].append(failed_tx_ratio)
        
        # Limit history size
        if len(self.history["attack_indicators"]) > self.analysis_window * 2:
            self.history["attack_indicators"].pop(0)
            self.history["node_trust_variance"].pop(0)
            self.history["latency_deviation"].pop(0)
            self.history["failed_tx_ratio"].pop(0)
        
        # Update current attack indicators
        self.attack_indicators = indicators
        
        return indicators
    
    def detect_attack(self) -> Tuple[Optional[str], float]:
        """
        Detect attack type based on current indicators.
        
        Returns:
            Tuple[Optional[str], float]: (Attack type, confidence)
        """
        # Find attack type with highest indicator
        highest_indicator = 0.0
        detected_attack = None
        
        for attack_type, indicator in self.attack_indicators.items():
            if indicator > highest_indicator:
                highest_indicator = indicator
                detected_attack = attack_type
        
        # Check if above threshold
        if detected_attack and highest_indicator >= self.thresholds.get(detected_attack, 0.7):
            attack_confidence = highest_indicator
        else:
            detected_attack = None
            attack_confidence = 0.0
        
        # Update current attack state
        self.current_attack = detected_attack
        self.attack_confidence = attack_confidence
        
        # Save to history
        self.history["detected_attacks"].append((detected_attack, attack_confidence))
        
        return detected_attack, attack_confidence
    
    def _calculate_51_percent_indicator(self, 
                                      failed_tx_ratio: float, 
                                      node_trust_variance: float,
                                      transactions: List[Dict[str, Any]]) -> float:
        """
        Calculate 51% attack indicator.
        
        Args:
            failed_tx_ratio: Ratio of failed transactions
            node_trust_variance: Variance of trust scores between nodes
            transactions: List of recent transactions
            
        Returns:
            float: 51% attack indicator (0.0-1.0)
        """
        # Calculate failure ratio of high-value transactions
        high_value_txs = [tx for tx in transactions if tx.get('value', 0) > 50]
        high_value_failure_ratio = 0.0
        if high_value_txs:
            high_value_failure_ratio = sum(1 for tx in high_value_txs if tx.get('status') != 'completed') / len(high_value_txs)
        
        # Calculate 51% attack indicator based on weights
        weights = self.weights["51_percent"]
        indicator = (
            weights["failed_tx_ratio"] * failed_tx_ratio +
            weights["node_trust_variance"] * min(1.0, node_trust_variance * 10) +
            weights["high_value_tx_failure"] * high_value_failure_ratio
        )
        
        return min(1.0, indicator)
    
    def _calculate_ddos_indicator(self, 
                                failed_tx_ratio: float, 
                                latency_deviation: float,
                                network_metrics: Dict[str, Any]) -> float:
        """
        Calculate DDoS attack indicator.
        
        Args:
            failed_tx_ratio: Ratio of failed transactions
            latency_deviation: Deviation of latency
            network_metrics: Network metrics
            
        Returns:
            float: DDoS attack indicator (0.0-1.0)
        """
        # Calculate network congestion index
        network_congestion = network_metrics.get('congestion', latency_deviation)
        
        # Calculate DDoS attack indicator based on weights
        weights = self.weights["ddos"]
        indicator = (
            weights["failed_tx_ratio"] * failed_tx_ratio +
            weights["latency_deviation"] * min(1.0, latency_deviation * 2) +
            weights["network_congestion"] * network_congestion
        )
        
        return min(1.0, indicator)
    
    def _calculate_mixed_indicator(self, 
                                 failed_tx_ratio: float, 
                                 node_trust_variance: float,
                                 latency_deviation: float,
                                 network_metrics: Dict[str, Any],
                                 transactions: List[Dict[str, Any]]) -> float:
        """
        Calculate mixed attack indicator.
        
        Args:
            failed_tx_ratio: Ratio of failed transactions
            node_trust_variance: Variance of trust scores between nodes
            latency_deviation: Deviation of latency
            network_metrics: Network metrics
            transactions: List of recent transactions
            
        Returns:
            float: Mixed attack indicator (0.0-1.0)
        """
        # Calculate failure ratio of high-value transactions
        high_value_txs = [tx for tx in transactions if tx.get('value', 0) > 50]
        high_value_failure_ratio = 0.0
        if high_value_txs:
            high_value_failure_ratio = sum(1 for tx in high_value_txs if tx.get('status') != 'completed') / len(high_value_txs)
        
        # Calculate mixed attack indicator based on weights
        weights = self.weights["mixed"]
        indicator = (
            weights["failed_tx_ratio"] * failed_tx_ratio +
            weights["latency_deviation"] * latency_deviation +
            weights["node_trust_variance"] * min(1.0, node_trust_variance * 5) +
            weights["high_value_tx_failure"] * high_value_failure_ratio
        )
        
        # Calculate trust entropy
        trust_entropy = self._calculate_trust_entropy(network_metrics)
        
        # Mixed attacks usually have inconsistent indicators and high variability
        # Detect diverse patterns
        mixed_patterns = 0
        if failed_tx_ratio > 0.4:
            mixed_patterns += 1
        if latency_deviation > 0.5:
            mixed_patterns += 1
        if node_trust_variance > 0.1:  # Lower threshold to increase sensitivity
            mixed_patterns += 1
        if high_value_failure_ratio > 0.6:
            mixed_patterns += 1
        if trust_entropy > 0.6:
            mixed_patterns += 1
            
        # Increase indicator if multiple patterns characteristic of mixed attack are detected
        if mixed_patterns >= 3:
            indicator *= 1.0 + (mixed_patterns - 2) * 0.1
            
        return min(1.0, indicator)
    
    def _calculate_selfish_mining_indicator(self, 
                                          network_metrics: Dict[str, Any],
                                          transactions: List[Dict[str, Any]]) -> float:
        """
        Calculate selfish mining attack indicator.
        
        Args:
            network_metrics: Network metrics
            transactions: List of recent transactions
            
        Returns:
            float: Selfish mining attack indicator (0.0-1.0)
        """
        # Selfish mining typically causes high fork rate and block withholding
        fork_rate = network_metrics.get('fork_rate', 0.0)
        block_withholding = network_metrics.get('block_withholding', 0.0)
        
        # Calculate selfish mining attack indicator based on weights
        weights = self.weights["selfish_mining"]
        indicator = (
            weights["fork_rate"] * fork_rate +
            weights["block_withholding"] * block_withholding
        )
        
        return min(1.0, indicator)
    
    def _calculate_bribery_indicator(self, 
                                   node_trust_variance: float,
                                   network_metrics: Dict[str, Any],
                                   transactions: List[Dict[str, Any]]) -> float:
        """
        Calculate bribery attack indicator.
        
        Args:
            node_trust_variance: Variance of trust scores between nodes
            network_metrics: Network metrics
            transactions: List of recent transactions
            
        Returns:
            float: Bribery attack indicator (0.0-1.0)
        """
        # Bribery usually causes inconsistency in voting and trust
        voting_deviation = network_metrics.get('voting_deviation', 0.0)
        trust_inconsistency = min(1.0, node_trust_variance * 5)
        
        # Calculate bribery attack indicator based on weights
        weights = self.weights["bribery"]
        indicator = (
            weights["voting_deviation"] * voting_deviation +
            weights["trust_inconsistency"] * trust_inconsistency
        )
        
        return min(1.0, indicator)

    def _calculate_trust_entropy(self, network_metrics: Dict[str, Any]) -> float:
        """
        Calculate entropy of trust variation.
        
        Args:
            network_metrics: Network metrics
            
        Returns:
            float: Entropy of trust variation (0.0-1.0)
        """
        # Get trust variance history
        if len(self.history["node_trust_variance"]) < 2:
            return 0.0
            
        # Calculate variation
        trust_variances = self.history["node_trust_variance"][-self.analysis_window:]
        trust_variance_changes = [abs(trust_variances[i] - trust_variances[i-1]) 
                                  for i in range(1, len(trust_variances))]
        
        if not trust_variance_changes:
            return 0.0
            
        # Calculate simple entropy based on variation level
        avg_change = np.mean(trust_variance_changes)
        max_change = max(trust_variance_changes)
        entropy = avg_change / max(0.001, max_change)
        
        return min(1.0, entropy * 2)  # Multiply by 2 to increase sensitivity

    def update_security_metrics(self, 
                              detected_attack: str,
                              attack_confidence: float,
                              network_metrics: Dict[str, Any],
                              previous_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Update and return security metrics based on detected attacks.
        
        Args:
            detected_attack: Type of detected attack
            attack_confidence: Confidence of attack detection
            network_metrics: Current network metrics
            previous_state: Previous security state
            
        Returns:
            Dict[str, float]: Updated security metrics
        """
        security_metrics = {}
        
        # Update overall metric
        security_metrics['overall_security'] = 1.0 - (attack_confidence if detected_attack else 0.0)
        
        # Update recovery rate
        prev_security = previous_state.get('overall_security', 1.0)
        security_metrics['recovery_rate'] = max(0.0, (security_metrics['overall_security'] - prev_security) / max(0.01, 1.0 - prev_security))
        
        # Update detection level
        security_metrics['detection_level'] = attack_confidence if detected_attack else 0.0
        
        # Update protection score for each attack type
        security_metrics['51_percent_protection'] = 1.0 - self.attack_indicators.get('51_percent', 0.0)
        security_metrics['ddos_protection'] = 1.0 - self.attack_indicators.get('ddos', 0.0)
        security_metrics['mixed_protection'] = 1.0 - self.attack_indicators.get('mixed', 0.0)
        
        # Network stability index
        security_metrics['network_stability'] = max(0.0, 1.0 - network_metrics.get('latency_deviation', 0.0))
        
        # Transaction reliability index
        security_metrics['transaction_reliability'] = max(0.0, 1.0 - network_metrics.get('failed_tx_ratio', 0.0))
        
        return security_metrics

def plot_attack_detection_results(security_metrics, output_dir=None):
    """
    Plot attack detection results.
    
    Args:
        security_metrics: SecurityMetrics object containing attack detection history
        output_dir: Output directory for plots
    """
    # Create data for plot
    history = security_metrics.history
    
    if len(history["attack_indicators"]) < 2:
        return
    
    # Prepare data
    time_points = list(range(len(history["attack_indicators"])))
    attack_types = list(history["attack_indicators"][0].keys())
    
    # Create dataframe for easier plotting
    attack_data = {attack_type: [indicators[attack_type] for indicators in history["attack_indicators"]] 
                  for attack_type in attack_types}
    attack_data['time'] = time_points
    df = pd.DataFrame(attack_data)
    
    # Prepare data for detected attacks
    detected_data = []
    for i, (attack, confidence) in enumerate(history["detected_attacks"]):
        if attack:
            detected_data.append((i, attack, confidence))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot attack indicators
    for attack_type in attack_types:
        plt.plot(df['time'], df[attack_type], label=f'{attack_type} Indicator')
    
    # Plot detection thresholds
    for attack_type, threshold in security_metrics.thresholds.items():
        if attack_type in attack_types:
            plt.axhline(y=threshold, linestyle='--', alpha=0.5, color='gray', 
                       label=f'{attack_type} Threshold')
    
    # Mark detected attack points
    if detected_data:
        for time_point, attack, confidence in detected_data:
            plt.scatter(time_point, confidence, marker='*', s=150, 
                       color='red', label=f'Detected {attack}' if attack != detected_data[0][1] else None)
            plt.annotate(f'{attack}', (time_point, confidence), 
                        xytext=(5, 5), textcoords='offset points')
    
    plt.title('Attack Detection Results Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Attack Indicator Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if needed
    if output_dir:
        plt.savefig(f"{output_dir}/attack_detection_results.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_security_metrics_comparison(security_metrics, output_dir=None):
    """
    Compare security metrics between different time points.
    
    Args:
        security_metrics: List of security metrics over time
        output_dir: Output directory for plots
    """
    # Check input data
    if not isinstance(security_metrics, list) or len(security_metrics) < 2:
        return
    
    # Prepare data
    metrics_keys = ['overall_security', 'recovery_rate', 'detection_level', 
                   '51_percent_protection', 'ddos_protection', 'mixed_protection',
                   'network_stability', 'transaction_reliability']
    
    # Create dataframe with data from different time points
    data = []
    for i, metrics in enumerate(security_metrics):
        metrics_values = {key: metrics.get(key, 0.0) for key in metrics_keys if key in metrics}
        metrics_values['time_point'] = i
        data.append(metrics_values)
    
    df = pd.DataFrame(data)
    
    # Create comparison chart
    plt.figure(figsize=(14, 10))
    
    # Plot each metric over time
    for key in metrics_keys:
        if key in df.columns:
            plt.plot(df['time_point'], df[key], marker='o', label=key)
    
    plt.title('Security Metrics Comparison Over Time')
    plt.xlabel('Time Points')
    plt.ylabel('Metric Value')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if needed
    if output_dir:
        plt.savefig(f"{output_dir}/security_metrics_comparison.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
    
    # Heatmap for metric changes over time
    if len(df) > 5:
        plt.figure(figsize=(12, 8))
        
        # Prepare data for heatmap
        heatmap_data = df[metrics_keys].T if all(key in df.columns for key in metrics_keys) else df.drop('time_point', axis=1).T
        
        # Plot heatmap
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=0.5)
        
        plt.title('Security Metrics Change Over Time')
        plt.xlabel('Time Points')
        plt.ylabel('Security Metrics')
        plt.tight_layout()
        
        # Save plot if needed
        if output_dir:
            plt.savefig(f"{output_dir}/security_metrics_heatmap.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

def plot_attack_impact_radar(attack_metrics, output_dir=None):
    """
    Plot radar chart to compare impact of different attack types.
    
    Args:
        attack_metrics: Dict containing impact metrics for different attack types
        output_dir: Output directory for plots
    """
    # Check input data
    if not isinstance(attack_metrics, dict) or not attack_metrics:
        return
    
    # Prepare data
    attack_types = list(attack_metrics.keys())
    metrics = list(attack_metrics[attack_types[0]].keys()) if attack_types else []
    
    if not metrics:
        return
    
    # Create dataframe
    data = {metric: [attack_metrics[attack][metric] for attack in attack_types] for metric in metrics}
    df = pd.DataFrame(data, index=attack_types)
    
    # Prepare data for radar chart
    categories = metrics
    N = len(categories)
    
    # Create angles for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add data for each attack type
    for attack in attack_types:
        values = df.loc[attack].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=attack)
        ax.fill(angles, values, alpha=0.25)
    
    # Customize plot
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Attack Impact Comparison')
    
    # Save plot if needed
    if output_dir:
        plt.savefig(f"{output_dir}/attack_impact_radar.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_blockchain_metrics(metrics: Dict[str, Dict[str, float]],
                           title: str = "Blockchain Metrics",
                           figsize: Tuple[int, int] = (12, 10),
                           save_path: Optional[str] = None):
    """
    Plot blockchain metrics in a radar chart.
    
    Args:
        metrics: Dictionary mapping system names to metric dictionaries
        title: Title of the plot
        figsize: Figure size
        save_path: Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Get all unique metric names
    all_metrics = set()
    for system_metrics in metrics.values():
        all_metrics.update(system_metrics.keys())
    
    all_metrics = sorted(list(all_metrics))
    n_metrics = len(all_metrics)
    
    if n_metrics < 3:
        # Not enough metrics for a radar chart
        return
    
    # Set up radar chart
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Set up plot
    ax = plt.subplot(111, polar=True)
    
    # Plot each system
    for i, (system, system_metrics) in enumerate(metrics.items()):
        values = [system_metrics.get(metric, 0) for metric in all_metrics]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=system)
        ax.fill(angles, values, alpha=0.25)
    
    # Set labels and title
    plt.xticks(angles[:-1], all_metrics)
    plt.title(title)
    plt.legend(loc='upper right')
    
    if save_path:
        # Use get_chart_path to get the full path
        full_path = get_chart_path(save_path, "blockchain")
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Blockchain metrics plot saved to {full_path}")
    else:
        plt.show()

def plot_transaction_distribution(tx_data: Dict[str, List[float]],
                                 title: str = "Transaction Distribution",
                                 figsize: Tuple[int, int] = (10, 6),
                                 save_path: Optional[str] = None):
    """
    Plot transaction distribution.
    
    Args:
        tx_data: Dictionary with transaction data
        title: Title of the plot
        figsize: Figure size
        save_path: Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    for data_name, values in tx_data.items():
        plt.hist(values, alpha=0.7, label=data_name, bins=20)
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        # Use get_chart_path to get the full path
        full_path = get_chart_path(save_path, "transactions")
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Transaction distribution plot saved to {full_path}")
    else:
        plt.show() 