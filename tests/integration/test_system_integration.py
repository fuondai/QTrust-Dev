"""
QTrust System Integration Test

This file tests the integration between core components of the QTrust system,
including Adaptive PoS, Adaptive Consensus, MAD-RAPID Router, and the blockchain
simulation environment.
"""

import time
import random
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from qtrust.consensus.adaptive_pos import AdaptivePoSManager
from qtrust.consensus.adaptive_consensus import AdaptiveConsensus
from qtrust.routing.mad_rapid import MADRAPIDRouter
from qtrust.simulation.blockchain_environment import BlockchainEnvironment
from qtrust.trust.models import TrustNetwork
import torch

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def test_adaptive_pos_consensus_integration():
    """Test integration between Adaptive PoS and Adaptive Consensus."""
    print("\n=== Testing Adaptive PoS and Consensus Integration ===")
    
    # Initialize Adaptive Consensus with PoS
    consensus = AdaptiveConsensus(
        enable_adaptive_pos=True,
        num_validators_per_shard=10,
        active_validator_ratio=0.6,
        rotation_period=5
    )
    
    # Create mock trust scores
    trust_scores = {i: random.random() for i in range(1, 21)}
    
    # Execute consensus for multiple rounds
    rounds = 20
    results = {
        "success_rate": 0,
        "energy_savings": [],
        "latency": [],
        "rotation_count": 0
    }
    
    for i in range(rounds):
        print(f"Round {i+1}/{rounds}")
        
        # Execute consensus
        result, protocol, latency, energy = consensus.execute_consensus(
            transaction_value=random.uniform(5.0, 50.0),
            shard_id=0,  # Shard ID 0
            trust_scores=trust_scores
        )
        
        results["success_rate"] += 1 if result else 0
        results["latency"].append(latency)
        
        # Check rotation statistics
        if i > 0 and i % 5 == 0:
            stats = consensus.get_pos_statistics()
            rotations = stats["total_rotations"]
            results["rotation_count"] = rotations
            results["energy_savings"].append(stats["total_energy_saved"])
            print(f"Rotations: {rotations}, Energy saved: {stats['total_energy_saved']:.2f}")
    
    # Print summary
    print("\nSummary:")
    print(f"Success rate: {results['success_rate']/rounds*100:.2f}%")
    print(f"Average latency: {np.mean(results['latency']):.2f}")
    print(f"Total rotations: {results['rotation_count']}")
    print(f"Total energy saved: {results['energy_savings'][-1] if results['energy_savings'] else 0:.2f}")
    
    # Thay vì trả về kết quả, hãy kiểm tra các giá trị
    assert results["success_rate"] >= 0
    assert len(results["latency"]) == rounds
    assert results["rotation_count"] >= 0

def test_routing_integration():
    """Test integration with MAD-RAPID Router."""
    print("\n=== Testing MAD-RAPID Router Integration ===")
    
    # Create simulation environment
    env = BlockchainEnvironment(
        num_shards=4,
        num_nodes_per_shard=10,
        enable_dynamic_resharding=True,
        congestion_threshold_high=0.8,
        congestion_threshold_low=0.2
    )
    
    # Get the network and shard information
    network = env.network
    shards = []
    for shard_id in range(env.num_shards):
        shard_nodes = [node for node, data in network.nodes(data=True) if data.get('shard_id') == shard_id]
        shards.append(shard_nodes)
    
    # Initialize MAD-RAPID Router
    router = MADRAPIDRouter(
        network=network,
        shards=shards,
        congestion_weight=0.4,
        latency_weight=0.3,
        energy_weight=0.2,
        trust_weight=0.1
    )
    
    # Test cross-shard routing
    num_transactions = 50
    routes = []
    congestion_levels = {i: random.uniform(0.1, 0.9) for i in range(env.num_shards)}
    
    # Update network state with congestion levels
    router.update_network_state(
        shard_congestion=np.array([congestion_levels[i] for i in range(env.num_shards)]),
        node_trust_scores={i: random.random() for i in range(env.total_nodes)}
    )
    
    for _ in range(num_transactions):
        # Select random source and destination shards
        source_shard = random.randint(0, env.num_shards-1)
        potential_destinations = list(range(env.num_shards))
        potential_destinations.remove(source_shard)
        destination_shard = random.choice(potential_destinations)
        
        # Create transaction object
        transaction = {
            'value': random.uniform(1.0, 100.0),
            'priority': random.choice(['normal', 'low_latency', 'high_security', 'energy_efficient'])
        }
        
        # Route transaction using find_optimal_path
        route = router.find_optimal_path(
            transaction=transaction,
            source_shard=source_shard,
            destination_shard=destination_shard
        )
        
        routes.append((source_shard, destination_shard, route))
    
    # Analyze routes
    hop_counts = [len(route) - 1 for _, _, route in routes]
    print(f"Average number of hops: {np.mean(hop_counts):.2f}")
    print(f"Max hops: {max(hop_counts)}")
    print(f"Min hops: {min(hop_counts)}")
    
    # Test route optimization over time
    print("\nRouting optimization over time test:")
    source = 0
    dest = 2
    routes_over_time = []
    
    # Simulate changing congestion levels
    for i in range(20):
        # Update congestion levels
        congestion_levels = {
            0: 0.2 + 0.6 * abs(np.sin(i * 0.1)),
            1: 0.3 + 0.5 * abs(np.sin(i * 0.2 + 1)),
            2: 0.1 + 0.7 * abs(np.sin(i * 0.15 + 2)),
            3: 0.4 + 0.4 * abs(np.sin(i * 0.25 + 3)),
        }
        
        # Update network state
        router.update_network_state(
            shard_congestion=np.array([congestion_levels[i] for i in range(env.num_shards)]),
            node_trust_scores={i: random.random() for i in range(env.total_nodes)}
        )
        
        transaction = {
            'value': 50.0,
            'priority': 'normal'
        }
        
        route = router.find_optimal_path(
            transaction=transaction,
            source_shard=source,
            destination_shard=dest
        )
        
        routes_over_time.append((i, route, congestion_levels.copy()))
        
        print(f"Iteration {i+1}: Route: {source} -> {' -> '.join(map(str, route[1:-1]))} -> {dest}")
    
    # Kiểm tra thay vì trả về kết quả
    assert len(hop_counts) == num_transactions
    assert len(routes_over_time) == 20
    assert all(len(route) >= 2 for _, route, _ in routes_over_time)

def test_full_system_integration():
    """Test full system integration."""
    print("\n=== Testing Full System Integration ===")
    
    # Create blockchain environment
    env = BlockchainEnvironment(
        num_shards=3,
        num_nodes_per_shard=8,
        enable_dynamic_resharding=True,
        max_transactions_per_step=20
    )
    
    # Initialize consensus for each shard
    consensus_managers = {}
    for shard_id in range(env.num_shards):
        consensus_managers[shard_id] = AdaptiveConsensus(
            enable_adaptive_pos=True,
            num_validators_per_shard=8,
            active_validator_ratio=0.7,
            rotation_period=10
        )
    
    # Get network and shards for routing
    network = env.network
    shards = []
    for shard_id in range(env.num_shards):
        shard_nodes = [node for node, data in network.nodes(data=True) if data.get('shard_id') == shard_id]
        shards.append(shard_nodes)
    
    # Initialize router
    router = MADRAPIDRouter(
        network=network,
        shards=shards
    )
    
    # Initialize simple trust model
    trust_scores = {}
    for node_id in range(env.total_nodes):
        trust_scores[node_id] = 0.7  # Initial trust score
    
    # Run simulation
    num_steps = 30
    metrics = {
        "throughput": [],
        "energy_consumption": [],
        "latency": [],
        "success_rate": []
    }
    
    print(f"Running {num_steps} simulation steps...")
    for step in range(num_steps):
        # Generate transactions
        num_transactions = random.randint(5, 20)
        transactions_processed = 0
        transactions_successful = 0
        total_energy = 0
        total_latency = 0
        
        # Get current congestion levels
        congestion_levels = {}
        for shard_id in range(env.num_shards):
            # Simulate congestion based on recent transaction volume
            congestion_levels[shard_id] = random.uniform(0.2, 0.8)
        
        # Update router with current network state
        router.update_network_state(
            shard_congestion=np.array([congestion_levels[i] for i in range(env.num_shards)]),
            node_trust_scores=trust_scores
        )
        
        # Process transactions
        for _ in range(num_transactions):
            # Select random source and destination shards
            source_shard = random.randint(0, env.num_shards-1)
            dest_shard = random.randint(0, env.num_shards-1)
            
            # Generate transaction value
            transaction_value = random.uniform(1.0, 100.0)
            
            # Create transaction object
            transaction = {
                'value': transaction_value,
                'priority': 'normal',
                'source_shard': source_shard,
                'destination_shard': dest_shard
            }
            
            # Route transaction if cross-shard
            if source_shard != dest_shard:
                route = router.find_optimal_path(
                    transaction=transaction,
                    source_shard=source_shard,
                    destination_shard=dest_shard
                )
                
                # Process through each shard in the route
                success = True
                route_latency = 0
                route_energy = 0
                
                for i in range(len(route) - 1):
                    current_shard = route[i]
                    success, latency, energy, protocol = consensus_managers[current_shard].execute_consensus(
                        transaction_value=transaction_value,
                        shard_id=current_shard,
                        trust_scores={i: trust_scores.get(i, 0.5) for i in range(1, 21)}
                    )
                    
                    route_latency += float(latency)
                    route_energy += float(energy)
                    
                    if not success:
                        success = False
                        break
            else:
                # Same shard transaction
                success, latency, energy, protocol = consensus_managers[source_shard].execute_consensus(
                    transaction_value=transaction_value,
                    shard_id=source_shard,
                    trust_scores={i: trust_scores.get(i, 0.5) for i in range(1, 21)}
                )
                
                route_latency = float(latency)
                route_energy = float(energy)
            
            # Update metrics
            transactions_processed += 1
            transactions_successful += 1 if success else 0
            total_energy += route_energy
            total_latency += route_latency
            
            # Update trust scores based on transaction success
            if source_shard != dest_shard:
                for shard_id in route:
                    for node in shards[shard_id]:
                        if success:
                            trust_scores[node] = min(1.0, trust_scores.get(node, 0.5) + 0.01)
                        else:
                            trust_scores[node] = max(0.1, trust_scores.get(node, 0.5) - 0.02)
        
        # Calculate metrics for this step
        success_rate = transactions_successful / max(1, transactions_processed)
        avg_latency = total_latency / max(1, transactions_processed)
        avg_energy = total_energy / max(1, transactions_processed)
        
        metrics["throughput"].append(transactions_processed)
        metrics["success_rate"].append(success_rate)
        metrics["latency"].append(avg_latency)
        metrics["energy_consumption"].append(avg_energy)
        
        # Print progress
        if step % 5 == 0 or step == num_steps - 1:
            print(f"Step {step+1}/{num_steps}: " +
                  f"Throughput: {transactions_processed}, " +
                  f"Success rate: {success_rate*100:.1f}%, " +
                  f"Avg latency: {avg_latency:.2f}, " +
                  f"Avg energy: {avg_energy:.2f}")
            
            # Print PoS statistics
            for shard_id, consensus in consensus_managers.items():
                pos_stats = consensus.get_pos_statistics()
                if "shard_stats" in pos_stats and shard_id in pos_stats["shard_stats"]:
                    shard_pos = pos_stats["shard_stats"][shard_id]
                    print(f"  Shard {shard_id} PoS: " +
                          f"Rotations: {shard_pos.get('rotations', 0)}, " +
                          f"Energy saved: {shard_pos.get('energy_saved', 0):.2f}")
    
    # Print summary
    print("\nSimulation complete. Summary:")
    print(f"Average throughput: {np.mean(metrics['throughput']):.2f} tx/step")
    print(f"Average success rate: {np.mean(metrics['success_rate'])*100:.2f}%")
    print(f"Average latency: {np.mean(metrics['latency']):.2f}")
    print(f"Average energy consumption: {np.mean(metrics['energy_consumption']):.2f}")
    
    # Kiểm tra thay vì trả về kết quả
    assert len(metrics["throughput"]) == num_steps
    assert len(metrics["success_rate"]) == num_steps
    assert len(metrics["latency"]) == num_steps
    assert len(metrics["energy_consumption"]) == num_steps
    assert np.mean(metrics["success_rate"]) > 0
    
    # Trả về metrics để có thể sử dụng trong if __name__ == "__main__"
    global _system_metrics
    _system_metrics = metrics

def plot_metrics(metrics, title="QTrust System Metrics"):
    """Plot system metrics."""
    plt.figure(figsize=(14, 10))
    
    # Plot throughput
    plt.subplot(2, 2, 1)
    plt.plot(metrics["throughput"], 'b-')
    plt.title("Transactions Per Step")
    plt.xlabel("Step")
    plt.ylabel("Count")
    plt.grid(True)
    
    # Plot success rate
    plt.subplot(2, 2, 2)
    plt.plot([r*100 for r in metrics["success_rate"]], 'g-')
    plt.title("Transaction Success Rate")
    plt.xlabel("Step")
    plt.ylabel("Success Rate (%)")
    plt.grid(True)
    
    # Plot latency
    plt.subplot(2, 2, 3)
    plt.plot(metrics["latency"], 'r-')
    plt.title("Average Transaction Latency")
    plt.xlabel("Step")
    plt.ylabel("Latency")
    plt.grid(True)
    
    # Plot energy consumption
    plt.subplot(2, 2, 4)
    plt.plot(metrics["energy_consumption"], 'y-')
    plt.title("Average Energy Consumption")
    plt.xlabel("Step")
    plt.ylabel("Energy Units")
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.savefig("qtrust_system_metrics.png")
    print("Metrics plotted and saved to 'qtrust_system_metrics.png'")
    
if __name__ == "__main__":
    print("QTrust System Integration Test")
    print("==============================")
    
    # Biến toàn cục để lưu metrics từ test_full_system_integration
    _system_metrics = None
    
    # Test Adaptive PoS and Consensus integration
    test_adaptive_pos_consensus_integration()
    
    # Test MAD-RAPID Router integration
    test_routing_integration()
    
    # Test full system integration
    test_full_system_integration()
    
    # Plot metrics
    if _system_metrics:
        plot_metrics(_system_metrics, "QTrust Integrated System Performance") 