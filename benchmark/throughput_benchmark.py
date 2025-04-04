#!/usr/bin/env python
"""
Benchmark script to compare sequential vs parallel transaction processing.
"""
import os
import sys
import time
import logging
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

# Add root directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qtrust.utils.performance_optimizer import ParallelTransactionProcessor
from qtrust.utils.paths import get_chart_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_transactions(num_transactions: int, 
                             min_value: float = 1.0, 
                             max_value: float = 1000.0) -> List[Dict[str, Any]]:
    """
    Generate mock transactions for benchmark purposes.
    
    Args:
        num_transactions: Number of transactions
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        List[Dict[str, Any]]: List of transactions
    """
    transactions = []
    for i in range(num_transactions):
        tx = {
            'id': f'tx_{int(time.time())}_{i}',
            'sender': f'sender_{random.randint(1, 100)}',
            'receiver': f'receiver_{random.randint(1, 100)}',
            'value': random.uniform(min_value, max_value),
            'timestamp': time.time(),
            'data': f'Data payload {i}',
            'sender_shard': random.randint(0, 3),
            'receiver_shard': random.randint(0, 3)
        }
        transactions.append(tx)
    return transactions


def process_transaction_sequentially(tx: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Process a transaction sequentially (for benchmark purposes).
    
    Args:
        tx: Transaction to process
        
    Returns:
        Tuple[bool, float]: (success, latency)
    """
    # Simulate basic processing with variable latency based on value
    base_latency = 0.005  # 5ms base
    value_factor = tx['value'] / 10000.0  # Factor scaled by value
    processing_time = base_latency + random.uniform(0.002, 0.01) * (1 + value_factor)
    
    # Simulate processing time
    time.sleep(processing_time)
    
    # Simulate success rate (95% success)
    success = random.random() < 0.95
    
    # Simulate actual latency (including random factor)
    actual_latency = processing_time * (1.0 + random.uniform(-0.1, 0.1))
    
    return success, actual_latency


def run_benchmark(
    transaction_counts: List[int],
    num_runs: int = 3
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Run benchmark to compare sequential and parallel processing.
    
    Args:
        transaction_counts: List of transaction counts to test
        num_runs: Number of times to repeat each test
        
    Returns:
        Tuple containing sequential and parallel results
    """
    sequential_results = {
        'transaction_counts': transaction_counts,
        'throughput': [],
        'latency': [],
        'success_rate': [],
        'run_time': []
    }
    
    parallel_results = {
        'transaction_counts': transaction_counts,
        'throughput': [],
        'latency': [],
        'success_rate': [],
        'run_time': []
    }
    
    for tx_count in transaction_counts:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing with {tx_count} transactions")
        logger.info(f"{'='*50}")
        
        # Arrays to store temporary results for multiple runs
        seq_throughputs = []
        seq_latencies = []
        seq_success_rates = []
        seq_run_times = []
        
        par_throughputs = []
        par_latencies = []
        par_success_rates = []
        par_run_times = []
        
        for run in range(num_runs):
            logger.info(f"Run {run+1}/{num_runs}")
            
            # Generate transactions for each run
            transactions = generate_test_transactions(tx_count)
            
            # Run sequential processing
            logger.info("Running sequential processing...")
            seq_start_time = time.time()
            
            # Perform sequential processing
            seq_successful = 0
            seq_total_latency = 0.0
            
            for tx in transactions:
                success, latency = process_transaction_sequentially(tx)
                if success:
                    seq_successful += 1
                    seq_total_latency += latency
            
            seq_end_time = time.time()
            seq_elapsed = seq_end_time - seq_start_time
            seq_throughput = tx_count / seq_elapsed if seq_elapsed > 0 else 0
            seq_avg_latency = seq_total_latency / seq_successful if seq_successful > 0 else 0
            seq_success_rate = seq_successful / tx_count if tx_count > 0 else 0
            
            # Save results
            seq_throughputs.append(seq_throughput)
            seq_latencies.append(seq_avg_latency)
            seq_success_rates.append(seq_success_rate)
            seq_run_times.append(seq_elapsed)
            
            logger.info(f"Sequential: {seq_throughput:.2f} tx/s, "
                      f"Latency: {seq_avg_latency:.4f}s, "
                      f"Success: {seq_success_rate:.2%}")
            
            # Run parallel processing
            logger.info("Running parallel processing...")
            
            # Initialize parallel processor with 8 workers
            parallel_processor = ParallelTransactionProcessor(max_workers=8)
            
            # Start timing
            par_start_time = time.time()
            
            # Perform parallel processing
            par_result = parallel_processor.process_transactions(
                transactions, process_transaction_sequentially
            )
            
            par_end_time = time.time()
            par_elapsed = par_end_time - par_start_time
            
            # Get metrics from result
            par_throughput = par_result['throughput']
            par_avg_latency = par_result['avg_latency']
            par_success_rate = par_result['success_rate']
            
            # Save results
            par_throughputs.append(par_throughput)
            par_latencies.append(par_avg_latency)
            par_success_rates.append(par_success_rate)
            par_run_times.append(par_elapsed)
            
            logger.info(f"Parallel: {par_throughput:.2f} tx/s, "
                      f"Latency: {par_avg_latency:.4f}s, "
                      f"Success: {par_success_rate:.2%}")
            
            # Short break between runs to avoid CPU impact
            time.sleep(1)
        
        # Calculate averages for each transaction size
        sequential_results['throughput'].append(np.mean(seq_throughputs))
        sequential_results['latency'].append(np.mean(seq_latencies))
        sequential_results['success_rate'].append(np.mean(seq_success_rates))
        sequential_results['run_time'].append(np.mean(seq_run_times))
        
        parallel_results['throughput'].append(np.mean(par_throughputs))
        parallel_results['latency'].append(np.mean(par_latencies))
        parallel_results['success_rate'].append(np.mean(par_success_rates))
        parallel_results['run_time'].append(np.mean(par_run_times))
        
        logger.info(f"Average results for {tx_count} transactions:")
        logger.info(f"Sequential: {np.mean(seq_throughputs):.2f} tx/s, "
                  f"Latency: {np.mean(seq_latencies):.4f}s")
        logger.info(f"Parallel: {np.mean(par_throughputs):.2f} tx/s, "
                  f"Latency: {np.mean(par_latencies):.4f}s")
        logger.info(f"Speedup: {np.mean(par_throughputs)/np.mean(seq_throughputs):.2f}x")
    
    return sequential_results, parallel_results


def plot_results(sequential_results: Dict[str, List[float]], 
                parallel_results: Dict[str, List[float]]) -> None:
    """
    Plot benchmark results.
    
    Args:
        sequential_results: Results of sequential processing
        parallel_results: Results of parallel processing
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # X-axis data
    x = sequential_results['transaction_counts']
    x_ticks = [str(count) for count in x]
    
    # Plot throughput chart
    ax1.plot(x, sequential_results['throughput'], 'o-', label='Sequential', color='blue')
    ax1.plot(x, parallel_results['throughput'], 's-', label='Parallel', color='green')
    ax1.set_title('Transaction Throughput Comparison')
    ax1.set_xlabel('Number of Transactions')
    ax1.set_ylabel('Throughput (tx/s)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_ticks)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Create second axis for speedup
    ax1_right = ax1.twinx()
    speedup = [p / s if s > 0 else 0 for p, s in zip(parallel_results['throughput'], sequential_results['throughput'])]
    ax1_right.plot(x, speedup, 'd-', label='Speedup', color='red')
    ax1_right.set_ylabel('Speedup (x times)')
    ax1_right.legend(loc='upper right')
    
    # Plot latency chart
    ax2.plot(x, sequential_results['latency'], 'o-', label='Sequential', color='blue')
    ax2.plot(x, parallel_results['latency'], 's-', label='Parallel', color='green')
    ax2.set_title('Transaction Latency Comparison')
    ax2.set_xlabel('Number of Transactions')
    ax2.set_ylabel('Average Latency (s)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_ticks)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.suptitle('Parallel vs Sequential Transaction Processing Benchmark', fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    # Save chart
    chart_path = get_chart_path('benchmark/parallel_processing_benchmark.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    logger.info(f"Chart saved to {chart_path}")
    
    # Plot additional chart for success rate
    plt.figure(figsize=(10, 6), facecolor='white')
    
    # Use scientific style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Configure global font settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })
    
    # Define professional scientific color palette
    sequential_color = '#1f77b4'  # Blue
    parallel_color = '#2ca02c'    # Green
    reference_color = '#d62728'   # Red
    
    # Create plot with enhanced styling
    plt.plot(x, sequential_results['success_rate'], 'o-', 
            label='Sequential Processing', 
            color=sequential_color, 
            linewidth=2, 
            markersize=8,
            alpha=0.8)
    
    plt.plot(x, parallel_results['success_rate'], 's-', 
            label='Parallel Processing', 
            color=parallel_color, 
            linewidth=2, 
            markersize=8,
            alpha=0.8)
    
    # Add 100% reference line
    plt.axhline(y=1.0, color=reference_color, linestyle='--', 
               alpha=0.5, label='Ideal Success Rate (100%)')
    
    # Add value annotations at last point
    last_idx = len(x) - 1
    plt.annotate(f"{sequential_results['success_rate'][last_idx]:.1%}", 
                xy=(x[last_idx], sequential_results['success_rate'][last_idx]),
                xytext=(10, -20), 
                textcoords='offset points',
                fontsize=10,
                color=sequential_color,
                fontweight='bold')
    
    plt.annotate(f"{parallel_results['success_rate'][last_idx]:.1%}", 
                xy=(x[last_idx], parallel_results['success_rate'][last_idx]),
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=10,
                color=parallel_color,
                fontweight='bold')
    
    # Add shaded regions for success rate quality
    plt.axhspan(0.98, 1.01, alpha=0.1, color='green', label='Excellent (>98%)')
    plt.axhspan(0.95, 0.98, alpha=0.1, color='lightgreen', label='Good (95-98%)')
    plt.axhspan(0.90, 0.95, alpha=0.1, color='yellow', label='Fair (90-95%)')
    plt.axhspan(0.0, 0.90, alpha=0.1, color='red', label='Poor (<90%)')
    
    # Enhance chart elements
    plt.title('Transaction Success Rate Comparison', fontweight='bold', pad=15)
    plt.xlabel('Number of Transactions', fontweight='bold')
    plt.ylabel('Success Rate', fontweight='bold')
    plt.xticks(x, x_ticks)
    plt.yticks(np.arange(0, 1.1, 0.1), [f"{i*10}%" for i in range(11)])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', frameon=True, framealpha=0.95)
    
    # Set axis limits with padding
    plt.ylim(min(min(sequential_results['success_rate']), 
                min(parallel_results['success_rate'])) * 0.9, 
            1.05)
    
    # Add descriptive annotation
    success_diff = abs(sequential_results['success_rate'][last_idx] - 
                      parallel_results['success_rate'][last_idx])
    
    if sequential_results['success_rate'][last_idx] > parallel_results['success_rate'][last_idx]:
        better_method = "Sequential processing"
        worse_method = "parallel processing"
    else:
        better_method = "Parallel processing"
        worse_method = "sequential processing"
    
    if success_diff > 0.01:  # Only show if difference is greater than 1%
        plt.figtext(0.5, 0.01,
                   f"{better_method} shows {success_diff:.1%} better success rate than {worse_method} at {x[last_idx]} transactions.\n"
                   "Success rate represents the percentage of transactions that completed without errors.",
                   ha='center', fontsize=10, fontstyle='italic')
    else:
        plt.figtext(0.5, 0.01,
                   "Both processing methods show similar success rates across different transaction volumes.\n"
                   "Success rate represents the percentage of transactions that completed without errors.",
                   ha='center', fontsize=10, fontstyle='italic')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    # Save chart with high quality
    chart_path = get_chart_path('benchmark/success_rate_benchmark.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    logger.info(f"Success rate chart saved to {chart_path}")


def main():
    """Main function running the benchmark."""
    parser = argparse.ArgumentParser(description='Transaction processing benchmark')
    parser.add_argument('--transactions', type=str, default='100,500,1000',
                      help='Comma-separated list of transaction counts to test')
    parser.add_argument('--runs', type=int, default=3,
                      help='Number of runs for each benchmark')
    
    args = parser.parse_args()
    
    # Convert transaction counts string to integer list
    transaction_counts = [int(x.strip()) for x in args.transactions.split(',')]
    
    logger.info(f"Starting benchmark with transaction counts: {transaction_counts}")
    logger.info(f"Each benchmark will be run {args.runs} times")
    
    # Run benchmark
    sequential_results, parallel_results = run_benchmark(
        transaction_counts=transaction_counts,
        num_runs=args.runs
    )
    
    # Display overall benchmark results
    logger.info("\n\nBenchmark Results Summary:")
    logger.info("============================")
    for i, count in enumerate(transaction_counts):
        seq_throughput = sequential_results['throughput'][i]
        par_throughput = parallel_results['throughput'][i]
        speedup = par_throughput / seq_throughput if seq_throughput > 0 else 0
        
        logger.info(f"\nTransaction count: {count}")
        logger.info(f"Sequential throughput: {seq_throughput:.2f} tx/s")
        logger.info(f"Parallel throughput: {par_throughput:.2f} tx/s")
        logger.info(f"Speedup: {speedup:.2f}x")
        logger.info(f"Sequential latency: {sequential_results['latency'][i]:.4f}s")
        logger.info(f"Parallel latency: {parallel_results['latency'][i]:.4f}s")
    
    # Plot benchmark results
    plot_results(sequential_results, parallel_results)


if __name__ == '__main__':
    main() 