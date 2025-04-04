#!/usr/bin/env python
"""
System Simulation Test Script

This script tests the SystemSimulator with both sequential and parallel transaction processing.
It compares performance metrics between the two processing methods, including:
- Throughput (transactions per second)
- Success rate
- Average latency
- Average energy consumption
- Execution time
- Speedup ratio

The script also analyzes the distribution of consensus protocols used during simulation.
"""
import time
import logging
from qtrust.simulation.system_simulator import SystemSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the system simulation."""
    logger.info("Starting system simulation test...")
    
    # Initialize simulators with and without parallel processing
    sequential_simulator = SystemSimulator(enable_parallel_processing=False)
    parallel_simulator = SystemSimulator(enable_parallel_processing=True)
    
    num_transactions = 100
    
    # Run sequential simulation
    logger.info(f"Running sequential simulation with {num_transactions} transactions...")
    start_time = time.time()
    sequential_results = sequential_simulator.run_simulation(
        num_transactions=num_transactions
    )
    sequential_time = time.time() - start_time
    
    # Run parallel simulation
    logger.info(f"Running parallel simulation with {num_transactions} transactions...")
    start_time = time.time()
    parallel_results = parallel_simulator.run_simulation(
        num_transactions=num_transactions
    )
    parallel_time = time.time() - start_time
    
    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    
    # Display results
    logger.info("\n" + "="*50)
    logger.info("SYSTEM SIMULATION RESULTS COMPARISON")
    logger.info("="*50)
    
    logger.info("\nSequential Processing Results:")
    logger.info(f"- Throughput: {sequential_results['throughput']:.2f} tx/s")
    logger.info(f"- Success Rate: {sequential_results['success_rate']:.2%}")
    logger.info(f"- Avg Latency: {sequential_results['avg_latency']:.4f}s")
    logger.info(f"- Avg Energy: {sequential_results['avg_energy']:.4f}")
    logger.info(f"- Execution Time: {sequential_time:.2f}s")
    
    logger.info("\nParallel Processing Results:")
    logger.info(f"- Throughput: {parallel_results['throughput']:.2f} tx/s")
    logger.info(f"- Success Rate: {parallel_results['success_rate']:.2%}")
    logger.info(f"- Avg Latency: {parallel_results['avg_latency']:.4f}s")
    logger.info(f"- Avg Energy: {parallel_results['avg_energy']:.4f}")
    logger.info(f"- Execution Time: {parallel_time:.2f}s")
    
    logger.info(f"\nSpeedup: {speedup:.2f}x")
    
    # Display consensus protocols used
    protocols_used = {}
    for tx in parallel_results['transactions']:
        if 'protocol' in tx:
            protocol = tx['protocol']
            if protocol in protocols_used:
                protocols_used[protocol] += 1
            else:
                protocols_used[protocol] = 1
    
    logger.info("\nConsensus Protocols Used:")
    for protocol, count in protocols_used.items():
        percentage = count / num_transactions * 100
        logger.info(f"- {protocol}: {count} times ({percentage:.1f}%)")


if __name__ == "__main__":
    main() 