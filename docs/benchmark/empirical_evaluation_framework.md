# QTrust Empirical Evaluation Framework

This document outlines the comprehensive framework for empirically evaluating QTrust against real-world blockchain systems, focusing on performance, security, and scalability metrics.

## 1. Evaluation Objectives

The primary objectives of this framework are to:

1. Provide quantitative comparison between QTrust and existing blockchain systems
2. Evaluate performance under realistic workloads and network conditions
3. Assess security and attack resistance through standardized attack scenarios
4. Measure scalability across varying network sizes and transaction volumes
5. Create reproducible benchmarks for ongoing development

## 2. Comparison Systems

QTrust will be evaluated against the following blockchain systems:

| System | Type | Consensus | Sharding |
|--------|------|-----------|----------|
| Ethereum 2.0 | Public | Proof of Stake | Yes |
| Polkadot | Public | GRANDPA/BABE | Parachains |
| Cosmos | Public | Tendermint | Zones |
| Zilliqa | Public | PBFT/PoW | Yes |
| Harmony | Public | FBFT | Yes |
| Solana | Public | Proof of History/Stake | No (high TPS) |
| Hyperledger Fabric | Private | PBFT variants | Channels |
| R3 Corda | Private | Notary-based | Flows |

## 3. Evaluation Metrics

### 3.1. Performance Metrics

Performance is evaluated across the following dimensions:

#### 3.1.1. Throughput

```python
def calculate_throughput(transactions_processed, time_elapsed):
    """Calculate transactions per second."""
    return transactions_processed / time_elapsed
```

- **TPS (Transactions Per Second)**: Primary measure of system capacity
- **Sustained TPS**: Average TPS over a 24-hour period
- **Peak TPS**: Maximum TPS achievable in optimal conditions
- **Cross-shard TPS**: TPS for transactions spanning multiple shards

#### 3.1.2. Latency

```python
def calculate_latency_metrics(transaction_times):
    """Calculate various latency metrics."""
    latencies = [tx['confirmation_time'] - tx['submission_time'] for tx in transaction_times]
    return {
        'average': np.mean(latencies),
        'median': np.median(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'max': np.max(latencies)
    }
```

- **Average Latency**: Mean time from submission to confirmation
- **Median Latency**: 50th percentile of confirmation time
- **P95/P99 Latency**: 95th/99th percentile (tail latency)
- **Cross-shard Latency**: Latency for cross-shard transactions

#### 3.1.3. Resource Utilization

```python
def calculate_resource_utilization(cpu_usage, memory_usage, network_usage, storage_growth):
    """Calculate system resource efficiency."""
    return {
        'cpu_per_tx': cpu_usage / transactions_processed,
        'memory_per_node': memory_usage / num_nodes,
        'network_bandwidth': network_usage,
        'storage_growth_rate': storage_growth
    }
```

- **CPU Utilization**: Processing power per transaction
- **Memory Usage**: RAM requirements per node
- **Network Bandwidth**: Data transfer between nodes
- **Storage Growth Rate**: Blockchain size increase over time

#### 3.1.4. Energy Efficiency

```python
def calculate_energy_efficiency(energy_consumed, transactions_processed):
    """Calculate energy efficiency metrics."""
    return {
        'energy_per_tx': energy_consumed / transactions_processed,
        'tx_per_joule': transactions_processed / energy_consumed,
        'relative_efficiency': baseline_energy_per_tx / energy_per_tx
    }
```

- **Energy Per Transaction**: Joules required per transaction
- **Transactions Per Joule**: Inverse of energy per transaction
- **Relative Efficiency**: Comparison to baseline systems

### 3.2. Security Metrics

Security evaluation focuses on:

#### 3.2.1. Attack Resistance

```python
def calculate_attack_resistance(attack_detection_rate, successful_attack_ratio, 
                               recovery_time, affected_transactions):
    """Calculate attack resistance metrics."""
    return {
        'detection_rate': attack_detection_rate,
        'success_prevention': 1 - successful_attack_ratio,
        'recovery_time': recovery_time,
        'affected_tx_ratio': affected_transactions / total_transactions
    }
```

- **Attack Detection Rate**: Percentage of attacks detected
- **Attack Prevention Rate**: Percentage of attacks prevented
- **Recovery Time**: Time to recover from successful attacks
- **Impact Scope**: Percentage of affected transactions during attack

#### 3.2.2. Byzantine Fault Tolerance

```python
def evaluate_bft_threshold(num_malicious_nodes, num_total_nodes, consensus_success):
    """Evaluate Byzantine fault tolerance threshold."""
    f = num_malicious_nodes
    n = num_total_nodes
    return {
        'theoretical_limit': (n - 1) / 3,
        'actual_limit': f if consensus_success else f - 1,
        'resilience_ratio': f / n if consensus_success else (f - 1) / n
    }
```

- **Theoretical BFT Limit**: Maximum tolerable faulty nodes (n-1)/3
- **Empirical BFT Limit**: Actual measured tolerance in testing
- **Fault Tolerance Ratio**: Ratio of faulty nodes to total nodes

#### 3.2.3. Attack Scenario Performance

For each attack type (Sybil, 51%, Eclipse, etc.), measure:

- **Time to Detection**: How quickly attack is detected
- **Prevention Success Rate**: Percentage of prevented attacks
- **System Degradation**: Performance impact during attack
- **Recovery Speed**: Time to return to normal operation

### 3.3. Scalability Metrics

Scalability is evaluated by measuring how metrics change as the system scales:

#### 3.3.1. Horizontal Scalability

```python
def calculate_horizontal_scalability(tps_values, node_counts):
    """Calculate how throughput scales with node count."""
    # Fit scaling factors
    log_tps = np.log(tps_values)
    log_nodes = np.log(node_counts)
    slope, _, r_value, _, _ = stats.linregress(log_nodes, log_tps)
    
    return {
        'scaling_factor': slope,  # 1.0 = linear, <1 = sublinear, >1 = superlinear
        'correlation': r_value**2,
        'scaling_efficiency': tps_values[-1] / tps_values[0] / (node_counts[-1] / node_counts[0])
    }
```

- **Linear Scaling Factor**: How close to linear scaling (ideal = 1.0)
- **Scaling Efficiency**: Ratio of performance increase to resource increase
- **Maximum Effective Size**: Point of diminishing returns

#### 3.3.2. Transaction Volume Scalability

```python
def calculate_volume_scalability(latency_values, tps_values):
    """Calculate how latency scales with throughput."""
    return {
        'critical_point': tps_values[np.argmin(np.gradient(latency_values) < threshold)],
        'degradation_slope': np.polyfit(tps_values, latency_values, 1)[0],
        'queue_stability': all(latency < max_latency for latency in latency_values)
    }
```

- **Critical TPS**: Point where latency begins to increase rapidly
- **Latency-TPS Curve**: Relationship between latency and throughput
- **Queue Stability**: Whether transaction queue remains bounded

#### 3.3.3. Shard Scalability

```python
def calculate_shard_scalability(tps_values, shard_counts, cross_shard_ratio):
    """Calculate how performance scales with shard count."""
    # Theoretical scaling model accounting for cross-shard overhead
    expected_tps = [base_tps * (s * (1 - cross_shard_ratio) + 
                              cross_shard_ratio * s / np.log(s)) 
                   for s in shard_counts]
    
    # Calculate actual vs. theoretical ratio
    scaling_efficiency = [actual/expected for actual, expected 
                        in zip(tps_values, expected_tps)]
    
    return {
        'scaling_efficiency': scaling_efficiency,
        'cross_shard_overhead': 1 - min(scaling_efficiency),
        'optimal_shard_count': shard_counts[np.argmax(tps_values)]
    }
```

- **Shard Scaling Efficiency**: How performance scales with shard count
- **Cross-shard Overhead**: Performance cost of cross-shard transactions
- **Optimal Shard Count**: Number of shards maximizing throughput

## 4. Workload Models

### 4.1. Transaction Distribution Models

The following transaction patterns will be used:

#### 4.1.1. Synthetic Workloads

- **Uniform Random**: Transactions uniformly distributed across shards
- **Zipfian Distribution**: Transactions following power law distribution
- **Bursty Traffic**: Periods of high load followed by low activity
- **Heavy-tail Value**: Transaction values following Pareto distribution

#### 4.1.2. Real-world Transaction Patterns

- **Ethereum Trace**: Based on Ethereum mainnet transaction patterns
- **DeFi Simulation**: Mimicking decentralized finance operations
- **NFT Marketplace**: Simulating NFT minting and trading
- **Token Exchange**: Modeling exchange transactions

### 4.2. Network Condition Models

Tests will be conducted under:

- **Ideal Network**: Low latency, high bandwidth, no packet loss
- **Continental Network**: 50-100ms latency, moderate bandwidth
- **Global Network**: 100-300ms latency, variable bandwidth
- **Degraded Network**: High latency (300ms+), packet loss (1-5%)
- **Partitioned Network**: Temporary network partitions between nodes

## 5. Benchmark Methodology

### 5.1. Benchmark Procedure

Each benchmark follows a standard procedure:

1. **System Setup**: Deploy blockchain nodes according to specified topology
2. **Warm-up Phase**: Run transactions to reach steady state (10-30 minutes)
3. **Measurement Phase**: Collect metrics during active benchmarking (1-24 hours)
4. **Cool-down Phase**: Allow pending transactions to complete
5. **Data Collection**: Gather logs and performance data

### 5.2. Parameter Sweep

For each system, perform parameter sweeps across:

- **Node Count**: 10, 20, 50, 100, 200, 500, 1000 nodes
- **Shard Count**: 1, 2, 4, 8, 16, 32, 64 shards (if applicable)
- **Transaction Rate**: 10, 100, 1k, 10k, 100k TPS (until saturation)
- **Cross-shard Ratio**: 0%, 10%, 30%, 50%, 70% cross-shard transactions
- **Attack Intensity**: 0%, 5%, 10%, 20%, 33% malicious nodes

### 5.3. Statistical Rigor

To ensure valid results:

- **Multiple Runs**: Minimum 5 runs per configuration
- **Statistical Analysis**: Report mean, median, standard deviation, 95% confidence intervals
- **Outlier Handling**: Identify and analyze outliers, but include in results
- **Reproducibility**: Publish all parameters, seeds, and methodologies

## 6. Implementation of Benchmark System

### 6.1. Benchmark Runner Architecture

```python
class BenchmarkRunner:
    """Main orchestrator for blockchain benchmarks."""
    
    def __init__(self, system_name, config, output_dir):
        self.system_name = system_name
        self.config = config
        self.output_dir = output_dir
        self.metrics = {}
        
    def setup_system(self):
        """Deploy and configure the blockchain system."""
        pass
        
    def generate_workload(self, workload_type, transaction_count, **params):
        """Generate transaction workload according to specified model."""
        pass
        
    def execute_benchmark(self, duration, transaction_rate):
        """Run the actual benchmark with specified parameters."""
        pass
        
    def collect_metrics(self):
        """Collect and process performance metrics."""
        pass
        
    def simulate_attack(self, attack_type, intensity):
        """Simulate specific attack with given intensity."""
        pass
        
    def analyze_results(self):
        """Analyze collected metrics and generate reports."""
        pass
        
    def cleanup(self):
        """Clean up resources after benchmark completion."""
        pass
```

### 6.2. Comparative Analysis System

```python
class BlockchainComparator:
    """Comparative analysis of different blockchain systems."""
    
    def __init__(self, systems, benchmark_configs):
        self.systems = systems
        self.benchmark_configs = benchmark_configs
        self.results = {}
        
    def run_comparative_benchmarks(self):
        """Execute same benchmarks across all systems."""
        for system in self.systems:
            for config in self.benchmark_configs:
                runner = BenchmarkRunner(system, config, f"results/{system}")
                self.results[(system, config['name'])] = runner.execute_benchmark()
                
    def generate_comparison_report(self):
        """Generate comparative analysis report."""
        pass
        
    def plot_comparison_charts(self):
        """Create visualizations comparing system performance."""
        pass
```

## 7. Attack Simulation Framework

### 7.1. Simulated Attack Types

The following attacks will be simulated:

#### 7.1.1. Sybil Attack

```python
def simulate_sybil_attack(network, sybil_percentage, coordination_level):
    """Simulate Sybil attack with coordinated malicious nodes."""
    num_sybil_nodes = int(len(network.nodes) * sybil_percentage)
    sybil_nodes = []
    
    # Create Sybil identities
    for i in range(num_sybil_nodes):
        node = SybilNode(
            coordination_level=coordination_level,
            malicious_behavior=SybilBehavior()
        )
        network.add_node(node)
        sybil_nodes.append(node)
    
    # Create coordination between Sybil nodes
    if coordination_level > 0:
        for i in range(len(sybil_nodes)):
            for j in range(i+1, len(sybil_nodes)):
                if random.random() < coordination_level:
                    network.add_edge(sybil_nodes[i], sybil_nodes[j], 
                                    latency=1.0, bandwidth=1000)
    
    return sybil_nodes
```

#### 7.1.2. Eclipse Attack

```python
def simulate_eclipse_attack(network, target_shard, isolation_strength):
    """Simulate Eclipse attack targeting a specific shard."""
    target_nodes = [n for n in network.nodes if n.shard_id == target_shard]
    other_nodes = [n for n in network.nodes if n.shard_id != target_shard]
    
    # Remove or degrade connections
    for target in target_nodes:
        for other in other_nodes:
            if network.has_edge(target, other):
                if random.random() < isolation_strength:
                    network.remove_edge(target, other)
                else:
                    # Degrade connection
                    network.edges[target, other]['latency'] *= (1 + isolation_strength * 10)
                    network.edges[target, other]['bandwidth'] /= (1 + isolation_strength * 10)
    
    return target_nodes
```

#### 7.1.3. 51% Attack

```python
def simulate_51_percent_attack(network, target_shard, control_percentage, 
                              double_spend_attempts):
    """Simulate 51% attack with double spending attempts."""
    shard_nodes = [n for n in network.nodes if n.shard_id == target_shard]
    num_malicious = int(len(shard_nodes) * control_percentage)
    
    # Select nodes to compromise
    malicious_nodes = random.sample(shard_nodes, num_malicious)
    
    # Convert to malicious behavior
    for node in malicious_nodes:
        node.behavior = MaliciousValidatorBehavior(
            double_spend_attempts=double_spend_attempts,
            fork_length=3
        )
    
    return malicious_nodes
```

#### 7.1.4. DDoS Attack

```python
def simulate_ddos_attack(network, target_percentage, attack_intensity):
    """Simulate DDoS attack on a percentage of nodes."""
    num_targets = int(len(network.nodes) * target_percentage)
    targets = random.sample(list(network.nodes), num_targets)
    
    # Apply DDoS effect to targets
    for node in targets:
        # Reduce computational capacity
        node.processing_power *= (1 - attack_intensity)
        
        # Increase latency for all connections
        for neighbor in network.neighbors(node):
            network.edges[node, neighbor]['latency'] *= (1 + attack_intensity * 5)
            network.edges[node, neighbor]['bandwidth'] /= (1 + attack_intensity * 5)
    
    return targets
```

### 7.2. Attack Detection and Evaluation

```python
class AttackEvaluator:
    """Evaluate system response to attacks."""
    
    def __init__(self, network, security_monitor):
        self.network = network
        self.security_monitor = security_monitor
        self.attack_metrics = {}
        
    def measure_attack_impact(self, before_metrics, during_metrics, after_metrics):
        """Measure the impact of an attack on system performance."""
        return {
            'throughput_degradation': 1 - during_metrics['throughput'] / before_metrics['throughput'],
            'latency_increase': during_metrics['latency'] / before_metrics['latency'],
            'recovery_time': after_metrics['recovery_time'],
            'data_loss': after_metrics['data_loss']
        }
    
    def evaluate_detection_capability(self, attack_type, attack_params, actual_malicious_nodes):
        """Evaluate the system's ability to detect the attack."""
        detected_nodes = self.security_monitor.detect_malicious_nodes()
        
        true_positives = len(set(detected_nodes) & set(actual_malicious_nodes))
        false_positives = len([n for n in detected_nodes if n not in actual_malicious_nodes])
        false_negatives = len([n for n in actual_malicious_nodes if n not in detected_nodes])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'detection_time': self.security_monitor.detection_time
        }
```

## 8. Visualization and Reporting

### 8.1. Standard Performance Charts

```python
def plot_performance_comparison(systems, metrics, metric_name, title):
    """Create bar chart comparing systems on specific metric."""
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(systems))
    width = 0.35
    
    plt.bar(x, [metrics[system][metric_name] for system in systems], width)
    plt.xlabel('Blockchain System')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.xticks(x, systems)
    
    plt.tight_layout()
    plt.savefig(f"comparison_{metric_name}.png", dpi=300)
```

### 8.2. Scalability Visualization

```python
def plot_scalability_curves(system_results, x_param, y_metric, systems):
    """Plot how system performance scales with parameter changes."""
    plt.figure(figsize=(10, 8))
    
    for system in systems:
        x_values = system_results[system][x_param]
        y_values = system_results[system][y_metric]
        
        plt.plot(x_values, y_values, marker='o', label=system)
    
    plt.xlabel(x_param)
    plt.ylabel(y_metric)
    plt.title(f"{y_metric} vs {x_param} Scaling")
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"scalability_{y_metric}_vs_{x_param}.png", dpi=300)
```

### 8.3. Attack Impact Visualization

```python
def plot_attack_impact(before, during, after, metrics, systems, attack_type):
    """Visualize system metrics before, during, and after attacks."""
    phases = ['Before', 'During', 'After']
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        data = {
            'Before': [before[system][metric] for system in systems],
            'During': [during[system][metric] for system in systems],
            'After': [after[system][metric] for system in systems]
        }
        
        x = np.arange(len(systems))
        width = 0.25
        
        for i, phase in enumerate(phases):
            plt.bar(x + (i-1)*width, data[phase], width, label=phase)
        
        plt.xlabel('Blockchain System')
        plt.ylabel(metric)
        plt.title(f'Impact of {attack_type} Attack on {metric}')
        plt.xticks(x, systems)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"attack_impact_{attack_type}_{metric}.png", dpi=300)
```

## 9. Reproducibility Package

To ensure reproducibility, each benchmark will include:

### 9.1. Configuration Files

- System configuration parameters for each blockchain
- Network topology definitions
- Transaction workload parameters
- Attack simulation parameters

### 9.2. Raw Data

- Transaction logs
- Performance metrics time series
- Resource utilization metrics
- Attack detection logs

### 9.3. Analysis Scripts

- Data processing scripts
- Statistical analysis code
- Visualization generation code
- Comparative analysis tools

### 9.4. Environment Description

- Hardware specifications
- Network configuration
- Software versions and dependencies
- Seed values for random number generators

## 10. Future Extensions

The framework will be extended to include:

1. **Smart Contract Performance**: Metrics for smart contract execution
2. **Cross-chain Evaluation**: Measurements of cross-chain transaction capabilities
3. **Long-term Stability Testing**: Extended runs (weeks/months) to assess long-term performance
4. **Real-network Deployment**: Moving from simulation to real-network testing
5. **User Experience Metrics**: Including metrics related to user experience and developer usability 