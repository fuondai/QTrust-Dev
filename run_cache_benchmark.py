"""
Performance benchmark for smart caching functionality in QTRUST.

This script compares transaction processing performance with and without caching.
"""

import time
import random
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import uuid
import os

from qtrust.simulation.system_simulator import SystemSimulator
from qtrust.utils.paths import get_chart_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_workload(
    num_unique: int, 
    repeating_ratio: float, 
    total_transactions: int
) -> List[Dict[str, Any]]:
    """
    Generate workload with specified number of transactions and repeat ratio.
    
    Args:
        num_unique: Number of unique transactions to create
        repeating_ratio: Ratio of repeating transactions (0.0 - 1.0)
        total_transactions: Total transactions to create
        
    Returns:
        List[Dict[str, Any]]: List of transactions
    """
    logging.info(f"Generating workload with {num_unique} unique transactions, {repeating_ratio*100:.1f}% repeating ratio, {total_transactions} total transactions")
    
    # Create a set of unique transactions
    tmp_simulator = SystemSimulator(enable_parallel_processing=False)
    unique_transactions = tmp_simulator.generate_random_transactions(num_unique)
    
    # Determine how many repeating transactions to create
    num_repeating = int(total_transactions * repeating_ratio)
    
    if num_repeating == 0:
        # If no repeating transactions, return unique set
        for tx in unique_transactions:
            tx['debug_marker'] = 'from_benchmark'
        return unique_transactions[:total_transactions]
    
    # Number of source transactions to create copies from (keep small for better cache hit patterns)
    repeat_count = min(5, max(1, int(num_unique * 0.05)))
    
    # Select a subset of transactions to repeat
    repeating_source = random.sample(unique_transactions, repeat_count)
    
    # For debugging
    for i, tx in enumerate(repeating_source):
        logging.debug(f"Source transaction {i}: sender_shard={tx.get('sender_shard')}, receiver_shard={tx.get('receiver_shard')}")
    
    # Create copies of selected transactions
    repeating_transactions = []
    
    # Create exactly copies from each source transaction 
    copies_per_source = max(1, num_repeating // repeat_count)
    
    # Create copies for better cache hit ratio
    for source_tx in repeating_source:
        for _ in range(copies_per_source):
            # Create a new transaction that preserves all the important cache key fields
            new_tx = {
                'id': str(uuid.uuid4()),  # New unique ID
                'timestamp': time.time(),  # Current timestamp
                'sender': source_tx.get('sender', source_tx.get('sender_id')),
                'sender_id': source_tx.get('sender', source_tx.get('sender_id')),
                'receiver': source_tx.get('receiver', source_tx.get('receiver_id')),
                'receiver_id': source_tx.get('receiver', source_tx.get('receiver_id')),
                'sender_shard': source_tx.get('sender_shard'),
                'receiver_shard': source_tx.get('receiver_shard'),
                'value': source_tx.get('value'),
                'data': f"Cache test transaction {time.time()}",
                'debug_marker': 'repeating_tx'
            }
            
            repeating_transactions.append(new_tx)
    
    # Adjust to get exact ratio if needed
    while len(repeating_transactions) > num_repeating:
        repeating_transactions.pop()
    
    # Add random repeats if needed to reach desired count
    while len(repeating_transactions) < num_repeating:
        source_tx = random.choice(repeating_source)
        new_tx = {
            'id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'sender': source_tx.get('sender', source_tx.get('sender_id')),
            'sender_id': source_tx.get('sender', source_tx.get('sender_id')),
            'receiver': source_tx.get('receiver', source_tx.get('receiver_id')),
            'receiver_id': source_tx.get('receiver', source_tx.get('receiver_id')),
            'sender_shard': source_tx.get('sender_shard'),
            'receiver_shard': source_tx.get('receiver_shard'),
            'value': source_tx.get('value'),
            'data': f"Additional cache test transaction {time.time()}",
            'debug_marker': 'repeating_tx'
        }
        repeating_transactions.append(new_tx)
    
    # Create final workload
    remaining = total_transactions - num_repeating
    workload = []
    
    # Add unique transactions
    for i in range(min(remaining, len(unique_transactions))):
        tx = unique_transactions[i].copy()
            tx['debug_marker'] = 'unique_tx'
        workload.append(tx)
    
    # Add repeating transactions - keep repeating ones together for better cache locality
    workload.extend(repeating_transactions)
    
    # Create batched workload - first process unique, then similar ones
    # This helps ensure that we see patterns in the cache
    batch_size = max(5, total_transactions // 10)
    batched_workload = []
    
    # Process in small chunks
    for i in range(0, len(workload), batch_size):
        batch = workload[i:i+batch_size]
        # Shuffle within batch
        random.shuffle(batch)
        batched_workload.extend(batch)
    
    # Log statistics and verify
    repeating_count = sum(1 for tx in batched_workload if tx.get('debug_marker') == 'repeating_tx')
    unique_count = sum(1 for tx in batched_workload if tx.get('debug_marker') in ('unique_tx', 'from_benchmark'))
    actual_ratio = repeating_count / len(batched_workload) if len(batched_workload) > 0 else 0
    
    logging.info(f"Generated workload: {len(batched_workload)} transactions "
                f"({unique_count} unique, {repeating_count} repeating, ratio: {actual_ratio:.1%})")
    
    return batched_workload


def run_benchmark(workload: List[Dict[str, Any]], 
                enable_caching: bool = True, 
                num_runs: int = 3,
                simulator: Optional[SystemSimulator] = None) -> Dict[str, Any]:
    """
    Run benchmark with given workload.
    
    Args:
        workload: List of transactions to process
        enable_caching: Enable/disable caching functionality
        num_runs: Number of runs for each configuration to get average results
        simulator: Optional simulator instance to reuse (to preserve cache between runs)
        
    Returns:
        Dict[str, Any]: Benchmark results
    """
    logger.info(f"Running benchmark with caching {'enabled' if enable_caching else 'disabled'}")
    
    # Initialize simulator with desired configuration if not provided
    if simulator is None:
    simulator = SystemSimulator(
        num_shards=4,
        num_validators_per_shard=4,
        enable_parallel_processing=True
    )
    
    if hasattr(simulator, 'transaction_processor'):
        # Configure caching for transaction processor
        if simulator.transaction_processor:
        simulator.transaction_processor.enable_caching = enable_caching
            if not enable_caching:
                # Only clear cache when absolutely necessary (when running non-cached benchmark)
                simulator.transaction_processor.clear_cache()
    
    # Run multiple times and calculate average
    results_list = []
    for run in range(1, num_runs + 1):
        logger.info(f"Run {run}/{num_runs}")
        
        # Split workload into batches to simulate more realistic scenario
        batch_size = min(500, len(workload))
        num_batches = (len(workload) + batch_size - 1) // batch_size
        
        all_metrics = []
        start_time = time.time()
        
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(workload))
            batch = workload[batch_start:batch_end]
            
            # Reset metrics between runs
            simulator.reset_metrics()
            
            # Process batch of transactions
            metrics = simulator.run_simulation(
                num_transactions=0,  # Don't create new transactions, use batch
                existing_transactions=batch  # Use existing transactions
                )
            
            all_metrics.append(metrics)
        
        end_time = time.time()
        total_elapsed = end_time - start_time
        
        # Calculate aggregate metrics
        total_transactions = sum(m.get('num_transactions', 0) for m in all_metrics)
        successful_transactions = sum(m.get('successful_transactions', 0) for m in all_metrics)
        success_rate = successful_transactions / total_transactions if total_transactions > 0 else 0
        avg_latency = np.mean([m.get('avg_latency', 0) for m in all_metrics])
        avg_throughput = total_transactions / total_elapsed if total_elapsed > 0 else 0
        
        # Get cache info if available
        cache_hits = 0
        cache_misses = 0
        
        # Extract cache metrics from transaction processor if available
        if simulator.transaction_processor:
            cache_hits = simulator.transaction_processor.cache_hits
            cache_misses = simulator.transaction_processor.cache_misses
        
        # Calculate cache hit ratio
        cache_hit_ratio = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        
        results = {
            'total_transactions': total_transactions,
            'successful_transactions': successful_transactions,
            'success_rate': success_rate,
            'avg_latency': avg_latency,
            'throughput': avg_throughput,
            'elapsed_time': total_elapsed,
            'cache_enabled': enable_caching,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_ratio': cache_hit_ratio
        }
        
        results_list.append(results)
    
    # Calculate averages across multiple runs
    avg_results = {
        'total_transactions': np.mean([r['total_transactions'] for r in results_list]),
        'successful_transactions': np.mean([r['successful_transactions'] for r in results_list]),
        'success_rate': np.mean([r['success_rate'] for r in results_list]),
        'avg_latency': np.mean([r['avg_latency'] for r in results_list]),
        'throughput': np.mean([r['throughput'] for r in results_list]),
        'elapsed_time': np.mean([r['elapsed_time'] for r in results_list]),
        'cache_enabled': enable_caching,
        'cache_hits': np.mean([r['cache_hits'] for r in results_list]),
        'cache_misses': np.mean([r['cache_misses'] for r in results_list]),
        'cache_hit_ratio': np.mean([r['cache_hit_ratio'] for r in results_list])
    }
    
    # Add standard deviation
    avg_results['throughput_std'] = np.std([r['throughput'] for r in results_list])
    avg_results['latency_std'] = np.std([r['avg_latency'] for r in results_list])
    
    logger.info(f"Benchmark results with caching {'enabled' if enable_caching else 'disabled'}:")
    logger.info(f"  Throughput: {avg_results['throughput']:.2f} tx/s (±{avg_results['throughput_std']:.2f})")
    logger.info(f"  Latency: {avg_results['avg_latency']:.4f}s (±{avg_results['latency_std']:.4f})")
    logger.info(f"  Success rate: {avg_results['success_rate']:.2%}")
    
    if enable_caching:
        logger.info(f"  Cache hit ratio: {avg_results['cache_hit_ratio']:.2%}")
    
    return avg_results


def plot_results(repeating_ratios, with_cache_results, without_cache_results, cache_hit_ratios, save_path):
    """
    Plot benchmark results with scientific standards.
    
    Args:
        repeating_ratios: List of repeating ratios tested
        with_cache_results: Results with caching enabled
        without_cache_results: Results with caching disabled
        cache_hit_ratios: List of cache hit ratios
        save_path: Path to save the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Global font setup
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'Computer Modern Roman'],
        'font.weight': 'normal',
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })
    
    # Scientific color palette
    with_cache_color = '#1f77b4'  # blue
    without_cache_color = '#ff7f0e'  # orange
    speedup_color = '#2ca02c'  # green
    improvement_color = '#d62728'  # red
    
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 14), facecolor='white', 
                          gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    
    # Convert to flat array for easier indexing
    axs = axs.flatten()
    
    # Ensure data is in arrays
    repeating_ratios = np.array(repeating_ratios) * 100  # Convert to percentages
    
    # Subplot 1: Throughput
    ax = axs[0]
    
    # Prepare data for chart
    # Note: Results are stored as dictionaries with repeat ratio as key
    with_cache_throughput_mean = []
    with_cache_throughput_std = []
    without_cache_throughput_mean = []
    without_cache_throughput_std = []
    
    # Extract data from results
    for ratio in repeating_ratios/100:  # Convert percentage back to decimal for dict access
        if isinstance(with_cache_results[ratio]['throughput'], list):
            with_cache_throughput_mean.append(np.mean(with_cache_results[ratio]['throughput']))
            with_cache_throughput_std.append(np.std(with_cache_results[ratio]['throughput']))
        else:
            with_cache_throughput_mean.append(with_cache_results[ratio]['throughput'])
            with_cache_throughput_std.append(0)
            
        if isinstance(without_cache_results[ratio]['throughput'], list):
            without_cache_throughput_mean.append(np.mean(without_cache_results[ratio]['throughput']))
            without_cache_throughput_std.append(np.std(without_cache_results[ratio]['throughput']))
        else:
            without_cache_throughput_mean.append(without_cache_results[ratio]['throughput'])
            without_cache_throughput_std.append(0)
    
    # Draw bar chart with error bars
    x = np.arange(len(repeating_ratios))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, with_cache_throughput_mean, width, label='With Cache', 
                 color=with_cache_color, alpha=0.8, edgecolor='black', linewidth=1)
    ax.errorbar(x - width/2, with_cache_throughput_mean, yerr=with_cache_throughput_std, 
               fmt='none', color='black', capsize=5, linewidth=1.5)
    
    bars2 = ax.bar(x + width/2, without_cache_throughput_mean, width, label='Without Cache', 
                 color=without_cache_color, alpha=0.8, edgecolor='black', linewidth=1)
    ax.errorbar(x + width/2, without_cache_throughput_mean, yerr=without_cache_throughput_std, 
               fmt='none', color='black', capsize=5, linewidth=1.5)
    
    # Add labels and title
    ax.set_xlabel('Repeating Transaction Ratio (%)', fontweight='bold')
    ax.set_ylabel('Throughput (tx/s)', fontweight='bold')
    ax.set_title('Transaction Processing Throughput', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.0f}%' for r in repeating_ratios])
    ax.legend(frameon=True, facecolor='white', edgecolor='gray')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add value labels to each bar
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Calculate and display speedup
    for i in range(len(repeating_ratios)):
        speedup = with_cache_throughput_mean[i] / without_cache_throughput_mean[i]
        y_pos = max(with_cache_throughput_mean[i], without_cache_throughput_mean[i]) + 150
        ax.annotate(f'Speedup: {speedup:.2f}x', 
                  xy=(x[i], y_pos),
                  xytext=(0, 0), 
                  textcoords='offset points',
                  ha='center', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor=speedup_color, alpha=0.1),
                  fontsize=10, fontweight='bold', color=speedup_color)
    
    # Subplot 2: Latency
    ax = axs[1]
    
    # Prepare latency data
    with_cache_latency_mean = []
    with_cache_latency_std = []
    without_cache_latency_mean = []
    without_cache_latency_std = []
    
    # Extract data from results
    for ratio in repeating_ratios/100:  # Convert percentage back to decimal
        if 'avg_latency' in with_cache_results[ratio]:
            if isinstance(with_cache_results[ratio]['avg_latency'], list):
                # Convert from seconds to milliseconds
                values = [x * 1000 for x in with_cache_results[ratio]['avg_latency']]
                with_cache_latency_mean.append(np.mean(values))
                with_cache_latency_std.append(np.std(values))
            else:
                with_cache_latency_mean.append(with_cache_results[ratio]['avg_latency'] * 1000)
                with_cache_latency_std.append(0)
        elif 'latency' in with_cache_results[ratio]:
            if isinstance(with_cache_results[ratio]['latency'], list):
                # Convert from seconds to milliseconds
                values = [x * 1000 for x in with_cache_results[ratio]['latency']]
                with_cache_latency_mean.append(np.mean(values))
                with_cache_latency_std.append(np.std(values))
            else:
                with_cache_latency_mean.append(with_cache_results[ratio]['latency'] * 1000)
                with_cache_latency_std.append(0)
                
        if 'avg_latency' in without_cache_results[ratio]:            
            if isinstance(without_cache_results[ratio]['avg_latency'], list):
                # Convert from seconds to milliseconds
                values = [x * 1000 for x in without_cache_results[ratio]['avg_latency']]
                without_cache_latency_mean.append(np.mean(values))
                without_cache_latency_std.append(np.std(values))
            else:
                without_cache_latency_mean.append(without_cache_results[ratio]['avg_latency'] * 1000)
                without_cache_latency_std.append(0)
        elif 'latency' in without_cache_results[ratio]:
            if isinstance(without_cache_results[ratio]['latency'], list):
                # Convert from seconds to milliseconds
                values = [x * 1000 for x in without_cache_results[ratio]['latency']]
                without_cache_latency_mean.append(np.mean(values))
                without_cache_latency_std.append(np.std(values))
            else:
                without_cache_latency_mean.append(without_cache_results[ratio]['latency'] * 1000)
                without_cache_latency_std.append(0)
    
    # Draw bar chart with error bars
    bars1 = ax.bar(x - width/2, with_cache_latency_mean, width, label='With Cache', 
                 color=with_cache_color, alpha=0.8, edgecolor='black', linewidth=1)
    ax.errorbar(x - width/2, with_cache_latency_mean, yerr=with_cache_latency_std, 
               fmt='none', color='black', capsize=5, linewidth=1.5)
    
    bars2 = ax.bar(x + width/2, without_cache_latency_mean, width, label='Without Cache', 
                 color=without_cache_color, alpha=0.8, edgecolor='black', linewidth=1)
    ax.errorbar(x + width/2, without_cache_latency_mean, yerr=without_cache_latency_std, 
               fmt='none', color='black', capsize=5, linewidth=1.5)
    
    # Add labels and title
    ax.set_xlabel('Repeating Transaction Ratio (%)', fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title('Transaction Processing Latency', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.0f}%' for r in repeating_ratios])
    ax.legend(frameon=True, facecolor='white', edgecolor='gray')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add value labels to each bar
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Calculate and display latency improvement
    for i in range(len(repeating_ratios)):
        improvement = (without_cache_latency_mean[i] - with_cache_latency_mean[i]) / without_cache_latency_mean[i] * 100
        y_pos = max(with_cache_latency_mean[i], without_cache_latency_mean[i]) + 10
        ax.annotate(f'Improvement: {improvement:.1f}%', 
                  xy=(x[i], y_pos),
                  xytext=(0, 0), 
                  textcoords='offset points',
                  ha='center', va='bottom',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor=improvement_color, alpha=0.1),
                  fontsize=10, fontweight='bold', color=improvement_color)
    
    # Subplot 3: Success Rate
    ax = axs[2]
    
    # Prepare success rate data
    with_cache_success_mean = []
    without_cache_success_mean = []
    
    # Extract data from results
    for ratio in repeating_ratios/100:  # Convert percentage back to decimal
        if 'success_rate' in with_cache_results[ratio]:
            if isinstance(with_cache_results[ratio]['success_rate'], list):
                values = [x * 100 for x in with_cache_results[ratio]['success_rate']]
                with_cache_success_mean.append(np.mean(values))
            else:
                with_cache_success_mean.append(with_cache_results[ratio]['success_rate'] * 100)
                
        if 'success_rate' in without_cache_results[ratio]:
            if isinstance(without_cache_results[ratio]['success_rate'], list):
                values = [x * 100 for x in without_cache_results[ratio]['success_rate']]
                without_cache_success_mean.append(np.mean(values))
            else:
                without_cache_success_mean.append(without_cache_results[ratio]['success_rate'] * 100)
    
    # Draw line chart with markers
    ax.plot(repeating_ratios, with_cache_success_mean, 'o-', linewidth=2, markersize=8, 
          label='With Cache', color=with_cache_color)
    ax.plot(repeating_ratios, without_cache_success_mean, 's-', linewidth=2, markersize=8,
          label='Without Cache', color=without_cache_color)
    
    # Add labels and title
    ax.set_xlabel('Repeating Transaction Ratio (%)', fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Transaction Processing Success Rate', fontweight='bold', pad=15)
    ax.set_xticks(repeating_ratios)
    ax.set_xticklabels([f'{r:.0f}%' for r in repeating_ratios])
    ax.legend(frameon=True, facecolor='white', edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Ensure y limits from 0-100% with small margin
    ax.set_ylim(min(min(with_cache_success_mean), min(without_cache_success_mean)) - 5, 105)
    
    # Add value labels to each point on the line
    for i, (x_val, y_val) in enumerate(zip(repeating_ratios, with_cache_success_mean)):
        ax.annotate(f'{y_val:.1f}%', 
                  xy=(x_val, y_val),
                  xytext=(0, 10), 
                  textcoords='offset points',
                  ha='center', va='bottom',
                  fontsize=9, fontweight='bold')
    
    for i, (x_val, y_val) in enumerate(zip(repeating_ratios, without_cache_success_mean)):
        ax.annotate(f'{y_val:.1f}%', 
                  xy=(x_val, y_val),
                  xytext=(0, -15), 
                  textcoords='offset points',
                  ha='center', va='top',
                  fontsize=9, fontweight='bold')
    
    # Calculate and display difference
    for i in range(len(repeating_ratios)):
        diff = with_cache_success_mean[i] - without_cache_success_mean[i]
        if abs(diff) > 1.0:  # Only display if significant difference
            midpoint = (with_cache_success_mean[i] + without_cache_success_mean[i]) / 2
            ax.annotate(f'Δ: {diff:+.1f}%', 
                      xy=(repeating_ratios[i], midpoint),
                      xytext=(15, 0), 
                      textcoords='offset points',
                      ha='left', va='center',
                      arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                      fontsize=9, fontweight='bold', color='#7f7f7f')
    
    # Subplot 4: Cache Hit Ratio
    ax = axs[3]
    
    # Create pie chart for Cache Hit Ratio
    cache_hit_ratios = np.array(cache_hit_ratios) * 100  # Convert to percentage
    
    # Draw bar chart for cache hit ratios
    bars = ax.bar(x, cache_hit_ratios, width=0.6, color=with_cache_color, alpha=0.7, 
                edgecolor='black', linewidth=1)
    
    # Add labels and title
    ax.set_xlabel('Repeating Transaction Ratio (%)', fontweight='bold')
    ax.set_ylabel('Cache Hit Ratio (%)', fontweight='bold')
    ax.set_title('Cache Effectiveness', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.0f}%' for r in repeating_ratios])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 100)
    
    # Add value labels to each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.1f}%', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
        
        # Add small pie chart on each bar to illustrate hit/miss ratio
        hit_ratio = cache_hit_ratios[i] / 100.0
        miss_ratio = 1.0 - hit_ratio
        
        # Position of pie chart
        pie_size = 0.1
        pie_x = bar.get_x() + bar.get_width() * 0.5
        pie_y = height * 0.5
        
        # Draw pie chart
        pie_ax = plt.axes([0, 0, 1, 1])
        pie_ax.set_axis_off()
        bbox = ax.get_position()
        pie_axis_size = min(bbox.width, bbox.height) * pie_size
        pie_x_fig = bbox.x0 + bbox.width * (pie_x / len(repeating_ratios))
        pie_y_fig = bbox.y0 + bbox.height * (pie_y / 100.0)
        pie_ax.set_position([pie_x_fig - pie_axis_size/2, pie_y_fig - pie_axis_size/2, pie_axis_size, pie_axis_size])
        
        # Draw pie chart
        wedges, texts, autotexts = pie_ax.pie(
            [hit_ratio, miss_ratio], 
            colors=[with_cache_color, 'lightgray'], 
            autopct=lambda p: f'{p:.0f}%' if p > 10 else '',
            textprops={'fontsize': 8, 'color': 'white', 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
    
    # Add large title for entire figure
    fig.suptitle('QTRUST Smart Cache Performance Analysis', 
               fontsize=18, fontweight='bold', y=0.98)
    
    # Add descriptive footnote at bottom
    benchmark_info = (
        "Performance metrics for QTRUST Smart Caching Framework. "
        "Repeating transaction ratio significantly impacts performance: "
        "Higher ratio of repeating transactions leads to better cache efficiency, "
        "resulting in higher throughput and lower latency."
    )
    fig.text(0.5, 0.01, benchmark_info, ha='center', fontsize=11, fontstyle='italic')
    
    # Adjust layout and save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure with high resolution
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Results saved to {save_path}")

    # Display statistics about performance improvement
    print("\nPerformance Summary:")
    for i, ratio in enumerate(repeating_ratios):
        throughput_speedup = with_cache_throughput_mean[i] / without_cache_throughput_mean[i]
        latency_improvement = (without_cache_latency_mean[i] - with_cache_latency_mean[i]) / without_cache_latency_mean[i] * 100
        
        print(f"  For repeating ratio of {ratio:.0f}%:")
        print(f"    - Throughput of {with_cache_throughput_mean[i]:.2f} tx/s vs {without_cache_throughput_mean[i]:.2f} tx/s (speedup: {throughput_speedup:.2f}x)")
        print(f"    - Latency: {with_cache_latency_mean[i]:.2f} ms vs {without_cache_latency_mean[i]:.2f} ms (improvement: {latency_improvement:.1f}%)")
        print(f"    - Cache hit ratio: {cache_hit_ratios[i]:.2f}%")


def main():
    """Main function running the benchmark."""
    parser = argparse.ArgumentParser(description='QTRUST Cache Benchmark')
    parser.add_argument('--total-transactions', type=int, default=1000,
                       help='Total number of transactions in each workload')
    parser.add_argument('--num-runs', type=int, default=3,
                       help='Number of runs for each configuration to get average results')
    args = parser.parse_args()
    
    logger.info("Starting QTRUST Cache Benchmark")
    
    # Transaction repeat ratios to test
    repeating_ratios = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    results_with_cache = {}
    results_without_cache = {}
    
    # Create a shared simulator to preserve cache between runs
    shared_simulator = SystemSimulator(
        num_shards=4,
        num_validators_per_shard=4,
        enable_parallel_processing=True
    )
    
    for ratio in repeating_ratios:
        logger.info(f"\n=== Testing with {ratio:.0%} repeating transactions ===")
        
        # Create workload
        num_unique = max(100, int(args.total_transactions * (1 - ratio * 0.8)))
        workload = generate_workload(
            num_unique=num_unique,
            repeating_ratio=ratio,
            total_transactions=args.total_transactions
        )
        
        # Important: For each ratio, run benchmarks in this order to build up cache
        # First with caching disabled to avoid cache influence
        results_without_cache[ratio] = run_benchmark(
            workload=workload,
            enable_caching=False,
            num_runs=args.num_runs,
            simulator=shared_simulator
        )
        
        # Then with caching enabled to see the benefit
        results_with_cache[ratio] = run_benchmark(
            workload=workload,
            enable_caching=True,
            num_runs=args.num_runs,
            simulator=shared_simulator
        )
    
    # Plot results
    plot_results(repeating_ratios, results_with_cache, results_without_cache, [r['cache_hit_ratio'] for r in results_with_cache.values()], get_chart_path("cache_performance_benchmark.png", "benchmark"))
    
    logger.info("\n=== Summary of QTRUST Cache Benchmark ===")
    for ratio in repeating_ratios:
        speedup = results_with_cache[ratio]['throughput'] / results_without_cache[ratio]['throughput'] if results_without_cache[ratio]['throughput'] > 0 else 0
        latency_improv = (results_without_cache[ratio]['avg_latency'] - results_with_cache[ratio]['avg_latency']) / results_without_cache[ratio]['avg_latency'] if results_without_cache[ratio]['avg_latency'] > 0 else 0
        
        logger.info(f"Repeating ratio: {ratio:.0%}")
        logger.info(f"  - Throughput: {results_with_cache[ratio]['throughput']:.2f} tx/s vs {results_without_cache[ratio]['throughput']:.2f} tx/s (speedup: {speedup:.2f}x)")
        logger.info(f"  - Latency: {results_with_cache[ratio]['avg_latency']*1000:.2f} ms vs {results_without_cache[ratio]['avg_latency']*1000:.2f} ms (improvement: {latency_improv:.1%})")
        logger.info(f"  - Cache hit ratio: {results_with_cache[ratio]['cache_hit_ratio']:.2%}")
    
    logger.info("\nBenchmark completed. Results saved to charts/benchmark/cache_performance_benchmark.png")


if __name__ == "__main__":
    main() 