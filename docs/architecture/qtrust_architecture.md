# QTrust Architecture - Blockchain Sharding Optimized with DRL and FL

## Architecture Overview

QTrust is designed with a modular architecture and tight interaction between components. Below is an overview of the architectural model:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│                           QTrust Framework                                │
│                                                                           │
└───────────────┬───────────────────────────────────┬───────────────────────┘
                │                                   │
                ▼                                   ▼
┌───────────────────────────────┐       ┌─────────────────────────────────┐
│                               │       │                                 │
│   BlockchainEnvironment       │◄──────┤        DQN Agents              │
│   - Sharding Simulation       │       │   - Rainbow DQN                │
│   - Dynamic Resharding        │       │   - Actor-Critic               │
│   - Cross-Shard Transactions  │       │   - Policy Networks            │
│   - Performance Monitoring    │       │   - Prioritized Experience     │
│                               │       │                                 │
└──────────────┬────────────────┘       └──────────────┬──────────────────┘
               │                                       │
               │                                       │
               ▼                                       │
┌──────────────────────────────┐                       │
│                              │                       │
│    AdaptiveConsensus         │◄──────────────────────┘
│    - Fast BFT                │                       
│    - PBFT                    │                       ┌────────────────────────────┐
│    - Robust BFT              │◄─────────────────────►│                            │
│    - Protocol Selection      │                       │     FederatedLearning      │
└──────────────┬───────────────┘                       │     - FedAvg               │
               │                                       │     - FedTrust             │
               │                                       │     - Secure Aggregation   │
               ▼                                       │                            │
┌──────────────────────────────┐                       └──────────────┬─────────────┘
│                              │                                      │
│     MADRAPIDRouter           │◄─────────────────────────────────────┘
│     - Transaction Routing    │                                      
│     - Load Balancing         │                       ┌────────────────────────────┐
│     - Congestion Avoidance   │◄─────────────────────►│                            │
│     - Predictive Routing     │                       │        HTDCM               │
└──────────────────────────────┘                       │     - Trust Evaluation     │
                                                       │     - Anomaly Detection    │
                                                       │     - Security Monitoring  │
                                                       │                            │
                                                       └────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│                           Caching System                                  │
│       ┌─────────────┐     ┌─────────────┐      ┌──────────────┐          │
│       │  LRU Cache  │     │  TTL Cache  │      │ Tensor Cache │          │
│       └─────────────┘     └─────────────┘      └──────────────┘          │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Description

### 1. BlockchainEnvironment

Blockchain simulation environment with the ability to recreate various network scenarios. This component implements:

- **Sharding Framework**: Models the system with multiple shards (24-32)
- **Network Simulation**: Simulates latency, bandwidth, and P2P connections
- **Transaction Generator**: Creates and processes cross-shard transactions
- **Dynamic Resharding**: Dynamic resharding mechanism based on network load
- **Performance Metrics**: Collects and reports performance metrics

### 2. DQN Agents

Deep Reinforcement Learning (DRL) agents implementing advanced techniques to optimize decisions:

- **Rainbow DQN**: Combines 6 DQN improvements (Double DQN, Dueling, PER, Multi-step, Distributional RL, Noisy Nets)
- **Actor-Critic**: Architecture for learning policy and value simultaneously
- **Multi-objective Optimization**: Multi-objective optimization (throughput, latency, energy, security)
- **Experience Replay**: Stores and reuses experiences for efficient learning

### 3. AdaptiveConsensus

Module for dynamic selection of consensus protocols based on network state:

- **Fast BFT**: Fast protocol for stable networks and low-value transactions
- **PBFT**: Balance between performance and security
- **Robust BFT**: Optimized for high security when networks are unstable
- **Consensus Selection Algorithm**: Algorithm to decide optimal protocol
- **BLS Signature Aggregation**: Reduces signature size and communication costs

### 4. MADRAPIDRouter

Intelligent router for transaction routing between shards:

- **Proximity-aware Routing**: Routing based on network distance
- **Dynamic Mesh Connections**: Flexible connections between shards
- **Predictive Routing**: Predicts congestion and adjusts in advance
- **Cross-shard Optimization**: Optimization of cross-shard transactions
- **Load Balancing**: Load balancing between shards

### 5. HTDCM (Hierarchical Trust-based Data Center Mechanism)

Hierarchical trust mechanism to evaluate node trustworthiness:

- **Multi-level Trust Evaluation**: Trust evaluation at node, shard, and network levels
- **ML-based Anomaly Detection**: Anomaly detection using machine learning
- **Attack Classification**: Classification of multiple attack types
- **Trust Scoring**: Trust scoring for nodes and shards
- **Reputation Management**: Long-term reputation management

### 6. FederatedLearning

Distributed training system with privacy protection:

- **Federated DRL**: Federated reinforcement learning between nodes
- **FedTrust**: Model aggregation based on trust scores
- **Secure Aggregation**: Secure aggregation without revealing local data
- **Differential Privacy**: Privacy protection during learning
- **Model Personalization**: Model adaptation for individual nodes

### 7. Caching System

Intelligent caching system to optimize performance:

- **LRU Cache**: Stores recently used values
- **TTL Cache**: Caching with expiration time
- **Tensor Cache**: Specially optimized for PyTorch tensors
- **Cache Statistics**: Monitors cache hit/miss rates
- **Intelligent Eviction**: Smart eviction strategies

## Data Flow and Interaction

1. **BlockchainEnvironment** provides current network state to DQN Agents
2. **DQN Agents** make decisions about routing and consensus protocols
3. **AdaptiveConsensus** selects consensus protocol based on agent decisions
4. **MADRAPIDRouter** routes transactions based on agent decisions
5. **HTDCM** provides trust information to router and consensus
6. **FederatedLearning** aggregates learning experiences from multiple nodes
7. **Caching System** supports all modules with intelligent caching

## Architectural Advantages

1. **High Modularity**: Easy to replace or improve individual components
2. **Scalability**: Supports from dozens to thousands of nodes
3. **Flexible Adaptation**: Self-adjusts based on network conditions
4. **Decentralization**: No single point of failure
5. **High Security**: Multiple layers of attack detection and prevention
6. **Energy Efficiency**: Optimizes energy consumption
7. **Continuous Learning**: Continuously improves through reinforcement learning and federated learning 