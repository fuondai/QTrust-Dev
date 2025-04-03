"""
Cache Optimization Test Script for Blockchain Environment

This script evaluates the performance impact of caching in a blockchain environment by comparing 
execution with and without caching enabled. It provides performance metrics including execution time, 
cache hit ratios, and transaction processing statistics, and generates visualization charts.

Note: In current implementation, enabling caching might show reduced performance (higher execution time).
This counter-intuitive result can occur when the overhead of cache management exceeds the benefits,
especially in small test scenarios or with suboptimal cache parameters. The test is still valuable
to identify these scenarios and optimize caching strategies.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from qtrust.simulation.blockchain_environment import BlockchainEnvironment

def test_blockchain_environment_performance(num_shards=24, nodes_per_shard=20, 
                                           num_steps=100, tx_per_step=200,
                                           with_cache=True):
    """
    Test the performance of BlockchainEnvironment.
    
    Args:
        num_shards: Number of shards
        nodes_per_shard: Number of nodes per shard
        num_steps: Number of simulation steps
        tx_per_step: Number of transactions per step
        with_cache: Whether to use caching
        
    Returns:
        Dict: Performance statistics
    """
    # Create environment
    env = BlockchainEnvironment(
        num_shards=num_shards,
        num_nodes_per_shard=nodes_per_shard,
        max_transactions_per_step=tx_per_step,
        enable_caching=with_cache
    )
    
    # Reset environment
    env.reset()
    
    # Performance statistics
    performance_stats = {
        "total_time": 0,
        "step_times": [],
        "transactions_processed": 0,
        "successful_transactions": 0,
        "cache_stats": []
    }
    
    # Execute steps in the environment
    for step in tqdm(range(num_steps), desc=f"Testing {'with' if with_cache else 'without'} cache"):
        start_time = time.time()
        
        # Generate random action
        action = np.array([
            np.random.randint(0, num_shards),  # Target shard
            np.random.randint(0, 3)            # Consensus protocol
        ])
        
        # Execute step in the environment
        _, reward, done, info = env.step(action)
        
        # Record time
        step_time = time.time() - start_time
        performance_stats["step_times"].append(step_time)
        performance_stats["total_time"] += step_time
        
        # Update transaction statistics
        performance_stats["transactions_processed"] += info.get("transactions_processed", 0)
        performance_stats["successful_transactions"] += info.get("successful_transactions", 0)
        
        # Save cache statistics if available
        if with_cache:
            cache_stats = env.get_cache_stats()
            performance_stats["cache_stats"].append(cache_stats)
        
        if done:
            break
    
    # Calculate aggregate statistics
    performance_stats["avg_step_time"] = np.mean(performance_stats["step_times"])
    performance_stats["std_step_time"] = np.std(performance_stats["step_times"])
    
    if performance_stats["transactions_processed"] > 0:
        success_rate = performance_stats["successful_transactions"] / performance_stats["transactions_processed"]
        performance_stats["transaction_success_rate"] = success_rate
    else:
        performance_stats["transaction_success_rate"] = 0.0
    
    return performance_stats

def compare_performance(num_shards=24, nodes_per_shard=20, num_steps=100, tx_per_step=200):
    """
    Compare performance with and without caching.
    
    Args:
        num_shards: Number of shards
        nodes_per_shard: Number of nodes per shard
        num_steps: Number of simulation steps
        tx_per_step: Number of transactions per step
        
    Returns:
        Tuple: (stats_with_cache, stats_without_cache)
    """
    print(f"Testing blockchain environment with {num_shards} shards, {nodes_per_shard} nodes per shard")
    print(f"Running {num_steps} steps with {tx_per_step} transactions per step")
    
    # Test performance without caching
    print("\nTesting without caching...")
    stats_without_cache = test_blockchain_environment_performance(
        num_shards=num_shards,
        nodes_per_shard=nodes_per_shard,
        num_steps=num_steps,
        tx_per_step=tx_per_step,
        with_cache=False
    )
    
    # Test performance with caching
    print("\nTesting with caching...")
    stats_with_cache = test_blockchain_environment_performance(
        num_shards=num_shards,
        nodes_per_shard=nodes_per_shard,
        num_steps=num_steps,
        tx_per_step=tx_per_step,
        with_cache=True
    )
    
    # Calculate performance improvement
    time_improvement = (stats_without_cache["total_time"] - stats_with_cache["total_time"]) / stats_without_cache["total_time"] * 100
    
    # Print results
    print("\n--------- PERFORMANCE COMPARISON ---------")
    print(f"Total time without cache: {stats_without_cache['total_time']:.2f}s")
    print(f"Total time with cache: {stats_with_cache['total_time']:.2f}s")
    print(f"Time improvement: {time_improvement:.2f}%")
    print(f"Average step time without cache: {stats_without_cache['avg_step_time']*1000:.2f}ms")
    print(f"Average step time with cache: {stats_with_cache['avg_step_time']*1000:.2f}ms")
    
    if stats_with_cache["cache_stats"]:
        last_cache_stats = stats_with_cache["cache_stats"][-1]
        print("\n--------- CACHE STATISTICS ---------")
        print(f"Cache hit ratio: {last_cache_stats.get('hit_ratio', 0)*100:.2f}%")
        print(f"Total cache hits: {last_cache_stats.get('total_hits', 0)}")
        print(f"Total cache misses: {last_cache_stats.get('total_misses', 0)}")
        
        # Detailed cache hits
        if 'detailed_hits' in last_cache_stats:
            print("\nDetailed cache hits:")
            for cache_type, hits in last_cache_stats['detailed_hits'].items():
                print(f"  {cache_type}: {hits}")
    
    return stats_with_cache, stats_without_cache

def plot_performance_comparison(stats_with_cache, stats_without_cache):
    """
    Plot performance comparison charts.
    
    Args:
        stats_with_cache: Performance statistics with cache
        stats_without_cache: Performance statistics without cache
    """
    plt.figure(figsize=(15, 10))
    
    # Execution time chart
    plt.subplot(2, 2, 1)
    steps = range(1, len(stats_with_cache["step_times"]) + 1)
    plt.plot(steps, stats_with_cache["step_times"], label="With Cache")
    plt.plot(steps, stats_without_cache["step_times"], label="Without Cache")
    plt.xlabel("Step")
    plt.ylabel("Time (s)")
    plt.title("Step Execution Time")
    plt.legend()
    plt.grid(True)
    
    # Cache hit ratio chart
    if stats_with_cache["cache_stats"]:
        plt.subplot(2, 2, 2)
        hit_ratios = [stats.get('hit_ratio', 0) * 100 for stats in stats_with_cache["cache_stats"]]
        plt.plot(steps, hit_ratios)
        plt.xlabel("Step")
        plt.ylabel("Hit Ratio (%)")
        plt.title("Cache Hit Ratio Over Time")
        plt.grid(True)
        
        # Detailed cache hits chart
        plt.subplot(2, 2, 3)
        detailed_hits = {}
        for stats in stats_with_cache["cache_stats"]:
            if 'detailed_hits' in stats:
                for cache_type, hits in stats['detailed_hits'].items():
                    if cache_type not in detailed_hits:
                        detailed_hits[cache_type] = []
                    detailed_hits[cache_type].append(hits)
        
        for cache_type, hits in detailed_hits.items():
            if len(hits) < len(steps):
                hits.extend([hits[-1]] * (len(steps) - len(hits)))
            plt.plot(steps, hits, label=cache_type)
        
        plt.xlabel("Step")
        plt.ylabel("Hits")
        plt.title("Detailed Cache Hits")
        plt.legend()
        plt.grid(True)
    
    # Total time comparison chart
    plt.subplot(2, 2, 4)
    plt.bar(["Without Cache", "With Cache"], 
            [stats_without_cache["total_time"], stats_with_cache["total_time"]])
    plt.ylabel("Total Time (s)")
    plt.title("Total Execution Time Comparison")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("cache_performance_comparison.png")
    print("Performance comparison chart saved to 'cache_performance_comparison.png'")

def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Test blockchain environment caching performance")
    parser.add_argument("--shards", type=int, default=24, help="Number of shards")
    parser.add_argument("--nodes", type=int, default=20, help="Nodes per shard")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--tx", type=int, default=200, help="Transactions per step")
    
    args = parser.parse_args()
    
    # Compare performance
    stats_with_cache, stats_without_cache = compare_performance(
        num_shards=args.shards,
        nodes_per_shard=args.nodes,
        num_steps=args.steps,
        tx_per_step=args.tx
    )
    
    # Plot comparison charts
    plot_performance_comparison(stats_with_cache, stats_without_cache)

if __name__ == "__main__":
    main() 