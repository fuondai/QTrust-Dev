# QTrust Sharding System Documentation

## Overview

The QTrust sharding system is designed to horizontally scale blockchain networks by partitioning nodes and transactions into separate but interconnected groups called shards. This approach achieves higher throughput and reduced latency compared to traditional monolithic blockchain systems while maintaining security properties.

## Key Components

### 1. Shard Structure

Each shard in QTrust consists of:
- A fixed number of validator nodes (default: 10 per shard)
- A dedicated transaction pool
- A local state database
- Cross-shard communication channels
- A consensus mechanism (adaptively selected)

### 2. Network Topology

The network topology is modeled as a graph where:
- Nodes within the same shard are fully connected (intra-shard connections)
- Select nodes between different shards maintain inter-shard connections
- Edge properties include latency and bandwidth metrics
- Node properties include processing power, trust scores, and energy efficiency

## Shard Assignment Algorithm

### Initial Assignment

1. **Network Initialization**:
   ```python
   def _init_blockchain_network(self):
       self.network = nx.Graph()
       self.shards = []
       total_nodes = 0
       
       # Create nodes for each shard
       for shard_id in range(self.num_shards):
           shard_nodes = []
           for i in range(self.num_nodes_per_shard):
               node_id = total_nodes + i
               self.network.add_node(
                   node_id, 
                   shard_id=shard_id,
                   trust_score=np.random.uniform(0.6, 1.0),
                   processing_power=np.random.uniform(0.8, 1.0),
                   energy_efficiency=np.random.uniform(0.7, 0.95)
               )
               shard_nodes.append(node_id)
           
           self.shards.append(shard_nodes)
           total_nodes += self.num_nodes_per_shard
   ```

2. **Connection Establishment**:
   - Intra-shard: Full connectivity with low latency (1-5ms)
   - Inter-shard: Partial connectivity with higher latency (5-30ms)

### Dynamic Resharding

QTrust implements dynamic resharding based on network conditions:

1. **Shard Addition**: When network congestion exceeds a high threshold
   ```python
   def _split_shard(self, shard_id):
       # Divide nodes in the shard into two groups
       nodes = self.shards[shard_id]
       mid_point = len(nodes) // 2
       
       # Create new shard with second half of nodes
       new_shard_id = len(self.shards)
       self.shards.append(nodes[mid_point:])
       
       # Update original shard to contain only first half
       self.shards[shard_id] = nodes[:mid_point]
       
       # Update node shard_id attributes
       for node_id in self.shards[new_shard_id]:
           self.network.nodes[node_id]['shard_id'] = new_shard_id
       
       # Create new inter-shard connections
       self._establish_inter_shard_connections(shard_id, new_shard_id)
   ```

2. **Shard Merging**: When network congestion falls below a low threshold
   ```python
   def _merge_shards(self, shard_id1, shard_id2):
       # Combine nodes from both shards
       combined_nodes = self.shards[shard_id1] + self.shards[shard_id2]
       
       # Update first shard to contain all nodes
       self.shards[shard_id1] = combined_nodes
       
       # Update node shard_id attributes for second shard's nodes
       for node_id in self.shards[shard_id2]:
           self.network.nodes[node_id]['shard_id'] = shard_id1
       
       # Remove the second shard
       self.shards.pop(shard_id2)
       
       # Update shard_id for all subsequent shards
       for i in range(shard_id2, len(self.shards)):
           for node_id in self.shards[i]:
               self.network.nodes[node_id]['shard_id'] = i
   ```

3. **Resharding Trigger**:
   ```python
   def _check_and_perform_resharding(self):
       if not self.enable_dynamic_resharding:
           return
       
       if self.current_step - self.last_resharding_step < self.resharding_interval:
           return
       
       shard_congestion = self.get_shard_congestion()
       max_congestion = max(shard_congestion)
       min_congestion = min(shard_congestion)
       
       resharded = False
       
       # Check if we need to split a shard (add more shards)
       if max_congestion > self.congestion_threshold_high and self.num_shards < self.max_num_shards:
           most_congested_shard = np.argmax(shard_congestion)
           self._split_shard(most_congested_shard)
           self.num_shards += 1
           resharded = True
           
       # Check if we need to merge shards (reduce number of shards)
       elif min_congestion < self.congestion_threshold_low and self.num_shards > self.min_num_shards:
           least_congested_shard = np.argmin(shard_congestion)
           # Find neighbor shard with lowest congestion
           neighbor_shard = self._find_neighbor_with_lowest_congestion(least_congested_shard)
           if neighbor_shard is not None:
               self._merge_shards(least_congested_shard, neighbor_shard)
               self.num_shards -= 1
               resharded = True
       
       if resharded:
           self.last_resharding_step = self.current_step
           self.resharding_history.append((self.current_step, self.num_shards))
   ```

## Cross-Shard Transaction Handling

QTrust implements a two-phase commit protocol for cross-shard transactions:

### 1. Transaction Routing

The MADRAPIDRouter optimizes transaction routing between shards:

```python
def optimize_routing(self, transaction, network_state):
    # Extract source and potential destination shards
    source_shard = transaction['source_shard']
    possible_destinations = list(range(self.num_shards))
    
    best_shard = source_shard  # Default to own shard
    best_score = float('-inf')
    
    # Evaluate each potential destination shard
    for dest_shard in possible_destinations:
        # Skip if same as source for cross-shard evaluation
        if dest_shard == source_shard and len(possible_destinations) > 1:
            continue
            
        # Compute metrics for this path
        congestion = self.calculate_congestion_metric(dest_shard)
        latency = self.calculate_latency_metric(source_shard, dest_shard)
        energy = self.calculate_energy_metric(dest_shard)
        trust = self.get_trust_metric(dest_shard)
        
        # Weighted score calculation
        score = (
            self.congestion_weight * (1 - congestion) +
            self.latency_weight * (1 - latency) +
            self.energy_weight * (1 - energy) +
            self.trust_weight * trust
        )
        
        # Update best destination if better score found
        if score > best_score:
            best_score = score
            best_shard = dest_shard
    
    return best_shard
```

### 2. Transaction Processing

Cross-shard transactions follow these steps:

1. **Initial Processing**:
   - Transaction is routed to destination shard
   - Source shard locks relevant state

2. **Validation**:
   - Destination shard validates transaction
   - Consensus is reached within destination shard

3. **Commit or Abort**:
   - If validated, both shards update their states
   - If rejected, source shard releases locks

4. **Confirmation**:
   - Transaction result is recorded in both shards

```python
def _process_transaction(self, transaction, action):
    # Extract routing decision and consensus protocol selection
    destination_shard = action[0]
    consensus_protocol = action[1]
    
    # Limit to actual number of shards
    destination_shard = min(destination_shard, self.num_shards - 1)
    
    # Calculate latency for this transaction processing
    latency = self._calculate_transaction_latency(
        transaction, destination_shard, consensus_protocol
    )
    
    # Calculate energy consumption
    energy = self._calculate_energy_consumption(
        transaction, destination_shard, consensus_protocol
    )
    
    # Determine if transaction is successful
    success = self._determine_transaction_success(
        transaction, destination_shard, consensus_protocol, latency
    )
    
    # Update shard congestion levels
    self._update_shard_congestion(transaction, destination_shard)
    
    return success, latency, energy
```

## Shard Synchronization

Shards maintain synchronization through:

1. **Periodic State Digests**:
   - Each shard periodically generates a state digest
   - Digests are shared with all other shards

2. **Cross-Shard References**:
   - Transactions can reference state in other shards
   - References are verified through Merkle proofs

3. **Consensus Checkpoints**:
   - Global checkpoints are established periodically
   - All shards must agree on checkpoint integrity

## Performance Considerations

### Load Balancing

The system balances load across shards through:

1. **Transaction Routing Optimization**:
   - MADRAPIDRouter directs transactions to less congested shards
   - Trust scores influence routing decisions

2. **Dynamic Resharding**:
   - High congestion triggers shard splitting
   - Low utilization triggers shard merging

### Data Locality

QTrust optimizes for data locality:

1. **Address-based Sharding**:
   - Related addresses are assigned to the same shard when possible
   - Reduces cross-shard transaction frequency

2. **Temporal Locality**:
   - Recently interacting addresses are kept in the same shard
   - Adaptive resharding respects interaction patterns

## Security Considerations

### Shard Security

Each shard must maintain security properties:

1. **Minimum Validator Requirement**:
   - Each shard maintains a minimum number of validators
   - Ensures Byzantine fault tolerance within the shard

2. **Trust-weighted Consensus**:
   - Validators with higher trust scores have more influence
   - HTDCM system continuously updates trust scores

3. **Cross-shard Validation**:
   - Critical transactions require validation from multiple shards
   - Threshold signatures combine validations efficiently

### Attack Resistance

QTrust implements specific protections against shard-targeted attacks:

1. **Sybil Attack Prevention**:
   - Trust scores and stake requirements limit influence of fake nodes
   - Anomaly detection identifies coordinated behavior

2. **Eclipse Attack Mitigation**:
   - Randomly selected cross-shard connections prevent isolation
   - Regular topology shuffling prevents targeted attacks

3. **Cross-shard Consistency Attacks**:
   - Global checkpoints verify cross-shard consistency
   - BLS signature aggregation ensures efficient verification

## Theoretical Limits and Scalability

QTrust's sharding approach provides theoretical scalability improvements:

1. **Linear Throughput Scaling**:
   - System throughput scales approximately linearly with shard count
   - Limited by cross-shard transaction overhead

2. **Communication Complexity**:
   - Communication complexity grows as O(n + m) where:
     - n = number of nodes
     - m = number of cross-shard transactions

3. **Latency Considerations**:
   - Intra-shard latency remains constant regardless of network size
   - Cross-shard latency increases logarithmically with shard count

## System Parameters

The sharding system is highly configurable through these parameters:

1. **Structural Parameters**:
   - `num_shards`: Initial number of shards (default: 4)
   - `num_nodes_per_shard`: Nodes per shard (default: 10)
   - `max_num_shards`: Maximum allowed shards (default: 32)
   - `min_num_shards`: Minimum required shards (default: 2)

2. **Resharding Parameters**:
   - `enable_dynamic_resharding`: Toggle resharding (default: True)
   - `congestion_threshold_high`: Trigger for shard splitting (default: 0.85)
   - `congestion_threshold_low`: Trigger for shard merging (default: 0.15)
   - `resharding_interval`: Minimum steps between resharding (default: 50)

3. **Cross-shard Parameters**:
   - `cross_shard_transaction_ratio`: Target ratio of cross-shard transactions
   - `cross_shard_penalty`: Performance penalty for cross-shard operations

## Future Enhancements

Planned improvements to the sharding system include:

1. **Hierarchical Sharding**:
   - Implement multiple levels of shards
   - Allow for more efficient routing hierarchy

2. **State Execution Sharding**:
   - Separate execution from consensus for higher parallelism
   - Implement speculative execution across shards

3. **Adaptive Shard Size**:
   - Dynamically adjust nodes per shard based on load
   - Optimize for security vs. performance trade-offs

4. **Inter-Shard Synchronization Optimization**:
   - Reduce coordination overhead through improved protocols
   - Implement zero-knowledge proofs for efficient verification 