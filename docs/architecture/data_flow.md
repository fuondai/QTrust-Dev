# QTrust Data Flow Documentation

## Core System Architecture

The QTrust system is built around several key components that interact in a structured manner to optimize blockchain sharding using reinforcement learning. This document visualizes these relationships and data flows.

## Core Components

1. **BlockchainEnvironment** (qtrust/simulation/blockchain_environment.py)
   - Central simulation environment implementing the OpenAI Gym interface
   - Manages shards, nodes, transactions, and network topology
   - Provides state observations and processes actions from agents

2. **DQNAgent** (qtrust/agents/dqn/agent.py)
   - Implements Deep Q-Network for learning optimal shard routing and consensus selection
   - Processes environment states and outputs actions
   - Stores experiences in replay buffer for training

3. **MADRAPIDRouter** (qtrust/routing/mad_rapid.py)
   - Handles cross-shard transaction routing
   - Optimizes paths based on congestion, latency, energy, and trust

4. **HTDCM** (qtrust/trust/htdcm.py)
   - Hierarchical Trust and Data-Centric Management system
   - Manages trust scores for nodes and detects anomalies

5. **AdaptiveConsensus** (qtrust/consensus/adaptive_consensus.py)
   - Manages consensus protocol selection and adaptation
   - Integrates with BLS signatures, adaptive PoS, and lightweight crypto

6. **FederatedLearning** (qtrust/federated/federated_learning.py)
   - Enables decentralized model training across shards
   - Aggregates models securely while preserving privacy

## Data Flow Diagrams

### 1. Main Execution Flow

```
main.py
  │
  ├─► setup_simulation()
  │     ├─► BlockchainEnvironment
  │     ├─► MADRAPIDRouter
  │     ├─► HTDCM
  │     └─► AdaptiveConsensus
  │
  ├─► setup_dqn_agent()
  │     └─► DQNAgent
  │
  ├─► setup_federated_learning()
  │     └─► FederatedLearning
  │
  ├─► train_qtrust() or evaluate_qtrust()
  │     ├─► env.reset()
  │     └─► Training/Evaluation Loop
  │           ├─► agent.act(state)
  │           ├─► env.step(action)
  │           ├─► router.optimize_routing()
  │           ├─► htdcm.update_trust_scores()
  │           ├─► consensus.select_protocol()
  │           └─► agent.step(state, action, reward, next_state, done)
  │
  └─► plot_results()
```

### 2. BlockchainEnvironment Data Flow

```
BlockchainEnvironment
  │
  ├─► reset()
  │     └─► Returns initial state observation
  │
  ├─► step(action)
  │     ├─► _generate_transactions()
  │     ├─► _process_transaction()
  │     │     ├─► _calculate_transaction_latency()
  │     │     ├─► _calculate_energy_consumption()
  │     │     └─► _determine_transaction_success()
  │     │
  │     ├─► _get_reward()
  │     │     ├─► _get_throughput_reward()
  │     │     ├─► _get_latency_penalty()
  │     │     ├─► _get_energy_penalty()
  │     │     └─► _get_security_score()
  │     │
  │     ├─► _update_shard_congestion()
  │     ├─► _check_and_perform_resharding()
  │     └─► Returns (next_state, reward, done, info)
  │
  └─► get_state()
        ├─► _get_network_stability()
        ├─► _get_average_transaction_value()
        ├─► _get_average_trust_score()
        └─► _get_success_rate()
```

### 3. DQNAgent Data Flow

```
DQNAgent
  │
  ├─► act(state)
  │     ├─► QNetwork forward pass
  │     └─► Returns action based on epsilon-greedy policy
  │
  ├─► step(state, action, reward, next_state, done)
  │     ├─► Store experience in ReplayBuffer
  │     └─► If sufficient samples, call learn()
  │
  └─► learn()
        ├─► Sample batch from ReplayBuffer
        ├─► Compute TD targets using target network
        ├─► Update policy network
        └─► Periodically update target network
```

### 4. MADRAPIDRouter Data Flow

```
MADRAPIDRouter
  │
  ├─► optimize_routing(transaction, state)
  │     ├─► compute_path_metrics()
  │     │     ├─► calculate_congestion_metric()
  │     │     ├─► calculate_latency_metric()
  │     │     ├─► calculate_energy_metric()
  │     │     └─► get_trust_metric()
  │     │
  │     └─► select_optimal_path()
  │
  └─► update_network_state(shard_congestion)
```

### 5. HTDCM Data Flow

```
HTDCM
  │
  ├─► update_trust_scores(node_ids, transaction_success)
  │     └─► Update trust scores based on transaction success/failure
  │
  ├─► detect_anomalies(transactions, node_behaviors)
  │     ├─► MLBasedAnomalyDetectionSystem.detect()
  │     └─► Flag suspicious activities
  │
  └─► get_trust_scores()
        └─► Returns current trust scores for all nodes
```

### 6. AdaptiveConsensus Data Flow

```
AdaptiveConsensus
  │
  ├─► select_protocol(transaction_value, congestion, trust_score)
  │     └─► Returns optimal consensus protocol based on conditions
  │
  ├─► verify_consensus(shard_id, consensus_result)
  │     ├─► BLSBasedConsensus.verify() (if enabled)
  │     └─► Returns verification result
  │
  └─► rotate_validators() (if adaptive PoS enabled)
        ├─► AdaptivePoSManager.select_validators()
        └─► Update active validators based on performance
```

### 7. FederatedLearning Data Flow

```
FederatedLearning
  │
  ├─► initialize_clients(num_clients)
  │     └─► Create FederatedClient instances for each shard
  │
  ├─► train_round(local_data)
  │     ├─► Each client trains on local data
  │     └─► Returns local model updates
  │
  ├─► aggregate_models(model_updates)
  │     ├─► ModelAggregationManager.aggregate()
  │     └─► Returns global model
  │
  └─► distribute_global_model(global_model)
        └─► Update each client with aggregated model
```

## Input/Output Data Types

### BlockchainEnvironment
- **Inputs**: Actions from DQNAgent (routing decisions and consensus selection)
- **Outputs**: State observations (shard congestion, transaction values, trust scores), rewards

### DQNAgent
- **Inputs**: State observations from environment
- **Outputs**: Actions (shard selection and consensus protocol)

### MADRAPIDRouter
- **Inputs**: Network topology, current congestion, transaction details
- **Outputs**: Optimized routing paths

### HTDCM
- **Inputs**: Transaction outcomes, node behaviors
- **Outputs**: Updated trust scores, anomaly detection results

### AdaptiveConsensus
- **Inputs**: Transaction properties, network state
- **Outputs**: Selected consensus protocol, validation results

### FederatedLearning
- **Inputs**: Local training data, local model updates
- **Outputs**: Aggregated global model

## Performance Optimization
- Extensive use of caching (`@lru_cache`, `@ttl_cache`) for computation-intensive operations
- Batch processing of transactions for efficiency
- Optimized network topology management during resharding

## Future Improvements
- Decoupling of tightly integrated components for better modularity
- Clearer interfaces between components
- More explicit data flow documentation in code
- Enhanced error handling at component boundaries 