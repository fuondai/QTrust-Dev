#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark Runner - A tool for running benchmark scenarios for QTrust

This file provides tools to execute predefined benchmark scenarios
and collect results for analysis and comparison.
"""

import os
import sys
import time
import subprocess
import argparse
import json
import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure encoding for output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add root directory to PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from qtrust.benchmark.benchmark_scenarios import (
    get_scenario, get_all_scenario_ids, get_all_scenarios,
    BenchmarkScenario, NetworkCondition, AttackProfile,
    WorkloadProfile, NodeProfile
)

# Results directory
RESULTS_DIR = os.path.join(project_root, "benchmark_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_benchmark(scenario_id: str, output_dir: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Run a benchmark scenario and collect results.
    
    Args:
        scenario_id: ID of the scenario to run
        output_dir: Output directory, if not specified a subdirectory will be created in RESULTS_DIR
        verbose: Print detailed information during execution
        
    Returns:
        Dict containing metadata and benchmark results
    """
    start_time = time.time()
    scenario = get_scenario(scenario_id)
    
    if verbose:
        print(f"Running benchmark scenario: {scenario.name} ({scenario_id})")
        print(f"Description: {scenario.description}")
        print(f"Parameters: {scenario.get_command_line_args()}")
        print("-" * 80)
    
    # Create output directory if not specified
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, f"{scenario_id}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create JSON file with scenario configuration
    scenario_config_path = os.path.join(output_dir, "scenario_config.json")
    with open(scenario_config_path, "w", encoding="utf-8") as f:
        # Convert dataclasses to dict
        scenario_dict = {
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
            "num_shards": scenario.num_shards,
            "nodes_per_shard": scenario.nodes_per_shard,
            "max_steps": scenario.max_steps,
            "network_conditions": {
                "latency_base": scenario.network_conditions.latency_base,
                "latency_variance": scenario.network_conditions.latency_variance,
                "packet_loss_rate": scenario.network_conditions.packet_loss_rate,
                "bandwidth_limit": scenario.network_conditions.bandwidth_limit,
                "congestion_probability": scenario.network_conditions.congestion_probability,
                "jitter": scenario.network_conditions.jitter
            },
            "attack_profile": {
                "attack_type": scenario.attack_profile.attack_type,
                "malicious_node_percentage": scenario.attack_profile.malicious_node_percentage,
                "attack_intensity": scenario.attack_profile.attack_intensity,
                "attack_target": scenario.attack_profile.attack_target,
                "attack_duration": scenario.attack_profile.attack_duration,
                "attack_start_step": scenario.attack_profile.attack_start_step
            },
            "workload_profile": {
                "transactions_per_step_base": scenario.workload_profile.transactions_per_step_base,
                "transactions_per_step_variance": scenario.workload_profile.transactions_per_step_variance,
                "cross_shard_transaction_ratio": scenario.workload_profile.cross_shard_transaction_ratio,
                "transaction_value_mean": scenario.workload_profile.transaction_value_mean,
                "transaction_value_variance": scenario.workload_profile.transaction_value_variance,
                "transaction_size_mean": scenario.workload_profile.transaction_size_mean,
                "transaction_size_variance": scenario.workload_profile.transaction_size_variance,
                "bursty_traffic": scenario.workload_profile.bursty_traffic,
                "burst_interval": scenario.workload_profile.burst_interval,
                "burst_multiplier": scenario.workload_profile.burst_multiplier
            },
            "node_profile": {
                "processing_power_mean": scenario.node_profile.processing_power_mean,
                "processing_power_variance": scenario.node_profile.processing_power_variance,
                "energy_efficiency_mean": scenario.node_profile.energy_efficiency_mean,
                "energy_efficiency_variance": scenario.node_profile.energy_efficiency_variance,
                "reliability_mean": scenario.node_profile.reliability_mean,
                "reliability_variance": scenario.node_profile.reliability_variance,
                "node_failure_rate": scenario.node_profile.node_failure_rate,
                "node_recovery_rate": scenario.node_profile.node_recovery_rate
            },
            "enable_dynamic_resharding": scenario.enable_dynamic_resharding,
            "min_shards": scenario.min_shards,
            "max_shards": scenario.max_shards,
            "enable_adaptive_consensus": scenario.enable_adaptive_consensus,
            "enable_bls": scenario.enable_bls,
            "enable_adaptive_pos": scenario.enable_adaptive_pos,
            "enable_lightweight_crypto": scenario.enable_lightweight_crypto,
            "enable_federated": scenario.enable_federated,
            "seed": scenario.seed
        }
        json.dump(scenario_dict, f, indent=4)
    
    # Build command
    save_path = os.path.join(output_dir, "results")
    
    cmd = [
        "py", "-3.10", "-m", "main",
        "--eval",  # Evaluation mode
        "--save-dir", save_path,
    ]
    
    # Add parameters from scenario
    scenario_args = scenario.get_command_line_args().split()
    cmd.extend(scenario_args)
    
    # Log file
    log_file_path = os.path.join(output_dir, "benchmark_log.txt")
    
    # Run command and log
    if verbose:
        print(f"Running command: {' '.join(cmd)}")
        print(f"Log file: {log_file_path}")
    
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8"
        )
        
        # Read output and write to log file
        for line in process.stdout:
            log_file.write(line)
            if verbose:
                print(line.strip())
        
        process.wait()
    
    # Read results from JSON file (if exists)
    results_json_path = os.path.join(save_path, "final_metrics.json")
    results = {}
    
    if os.path.exists(results_json_path):
        with open(results_json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    
    # Add execution time information
    end_time = time.time()
    execution_time = end_time - start_time
    
    benchmark_results = {
        "scenario_id": scenario_id,
        "scenario_name": scenario.name,
        "execution_time": execution_time,
        "timestamp": timestamp,
        "output_dir": output_dir,
        "results": results,
        "exit_code": process.returncode
    }
    
    # Save summary results
    with open(os.path.join(output_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=4)
    
    if verbose:
        print(f"Benchmark completed in {execution_time:.2f} seconds")
        print(f"Results saved to: {output_dir}")
    
    return benchmark_results

def run_all_benchmarks(
    scenario_ids: Optional[List[str]] = None,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Run multiple benchmark scenarios, optionally in parallel.
    
    Args:
        scenario_ids: List of scenario IDs to run, if None will run all
        parallel: Allow parallel execution
        max_workers: Maximum number of workers if running in parallel
        verbose: Print detailed information during execution
        
    Returns:
        Dict containing results of all benchmarks, with scenario_id as key
    """
    if scenario_ids is None:
        scenario_ids = get_all_scenario_ids()
    
    if verbose:
        print(f"Preparing to run {len(scenario_ids)} benchmark scenarios")
        if parallel:
            num_workers = multiprocessing.cpu_count() if max_workers is None else max_workers
            print(f"Parallel mode with {num_workers} workers")
    
    # Create output directory for batch
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_dir = os.path.join(RESULTS_DIR, f"batch_{timestamp}")
    os.makedirs(batch_output_dir, exist_ok=True)
    
    all_results = {}
    
    if parallel:
        # Run in parallel with ProcessPoolExecutor
        num_workers = multiprocessing.cpu_count() if max_workers is None else max_workers
        
        def _run_benchmark_wrapper(scenario_id):
            output_dir = os.path.join(batch_output_dir, scenario_id)
            return run_benchmark(scenario_id, output_dir, verbose=False)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_scenario = {
                executor.submit(_run_benchmark_wrapper, scenario_id): scenario_id
                for scenario_id in scenario_ids
            }
            
            for future in future_to_scenario:
                scenario_id = future_to_scenario[future]
                try:
                    result = future.result()
                    all_results[scenario_id] = result
                    if verbose:
                        print(f"Completed: {scenario_id} in {result['execution_time']:.2f} seconds")
                except Exception as e:
                    if verbose:
                        print(f"Error running {scenario_id}: {str(e)}")
    else:
        # Run sequentially
        for scenario_id in scenario_ids:
            output_dir = os.path.join(batch_output_dir, scenario_id)
            try:
                result = run_benchmark(scenario_id, output_dir, verbose)
                all_results[scenario_id] = result
            except Exception as e:
                if verbose:
                    print(f"Error running {scenario_id}: {str(e)}")
    
    # Save summary of all results
    summary_path = os.path.join(batch_output_dir, "all_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    
    if verbose:
        print(f"All benchmarks completed. Summary results at: {summary_path}")
    
    return all_results

def generate_comparison_report(results_dir: Optional[str] = None, output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Generate comparison report from benchmark results.
    
    Args:
        results_dir: Directory containing results (batch directory), if None will use latest batch directory
        output_file: Path to save report, if None will create filename based on timestamp
        
    Returns:
        DataFrame containing comparison data
    """
    if results_dir is None:
        # Find latest batch directory
        batch_dirs = [d for d in os.listdir(RESULTS_DIR) if d.startswith("batch_")]
        if not batch_dirs:
            raise ValueError("No batch results directories found")
        batch_dirs.sort(reverse=True)  # Sort by time descending
        results_dir = os.path.join(RESULTS_DIR, batch_dirs[0])
    
    # Read summary results file
    summary_path = os.path.join(results_dir, "all_results_summary.json")
    if not os.path.exists(summary_path):
        # Find subdirectories with individual results
        result_data = {}
        for scenario_dir in os.listdir(results_dir):
            scenario_path = os.path.join(results_dir, scenario_dir)
            if os.path.isdir(scenario_path):
                result_file = os.path.join(scenario_path, "benchmark_results.json")
                if os.path.exists(result_file):
                    with open(result_file, "r", encoding="utf-8") as f:
                        result_data[scenario_dir] = json.load(f)
    else:
        with open(summary_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)
    
    # Create DataFrame from results
    data = []
    for scenario_id, result in result_data.items():
        scenario_name = result.get("scenario_name", scenario_id)
        execution_time = result.get("execution_time", 0)
        
        # Extract key metrics from results
        metrics = result.get("results", {})
        throughput = metrics.get("average_throughput", 0)
        latency = metrics.get("average_latency", 0)
        energy = metrics.get("average_energy", 0)
        security_score = metrics.get("security_score", 0)
        cross_shard_ratio = metrics.get("cross_shard_ratio", 0)
        
        # Scenario configuration information
        config_file = os.path.join(results_dir, scenario_id, "scenario_config.json")
        config = {}
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        
        attack_type = config.get("attack_profile", {}).get("attack_type", "none")
        num_shards = config.get("num_shards", 0)
        nodes_per_shard = config.get("nodes_per_shard", 0)
        
        # Create data row
        row = {
            "Scenario ID": scenario_id,
            "Scenario Name": scenario_name,
            "Num Shards": num_shards,
            "Nodes per Shard": nodes_per_shard,
            "Attack Type": attack_type,
            "Throughput (tx/s)": throughput,
            "Latency (ms)": latency,
            "Energy (mJ/tx)": energy,
            "Security Score": security_score,
            "Cross-Shard Ratio": cross_shard_ratio,
            "Execution Time (s)": execution_time
        }
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save report if output_file specified
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(results_dir, f"benchmark_comparison_{timestamp}.csv")
    
    df.to_csv(output_file, index=False)
    print(f"Comparison report saved to: {output_file}")
    
    # Plot comparison charts
    output_dir = os.path.dirname(output_file)
    plot_comparison_charts(df, output_dir)
    
    return df

def plot_comparison_charts(df: pd.DataFrame, output_dir: str):
    """
    Plot comparison charts from benchmark data.
    
    Args:
        df: DataFrame containing benchmark data
        output_dir: Output directory to save charts
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Throughput comparison chart
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df.sort_values("Throughput (tx/s)", ascending=False), 
                x="Scenario Name", y="Throughput (tx/s)")
    plt.xticks(rotation=45, ha="right")
    plt.title("Throughput Comparison Between Scenarios", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"throughput_comparison_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 2. Latency comparison chart
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df.sort_values("Latency (ms)"), 
                x="Scenario Name", y="Latency (ms)")
    plt.xticks(rotation=45, ha="right")
    plt.title("Latency Comparison Between Scenarios", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"latency_comparison_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 3. Security Score comparison chart
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df.sort_values("Security Score", ascending=False), 
                x="Scenario Name", y="Security Score")
    plt.xticks(rotation=45, ha="right")
    plt.title("Security Score Comparison Between Scenarios", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"security_score_comparison_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 4. Energy comparison chart
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df.sort_values("Energy (mJ/tx)"), 
                x="Scenario Name", y="Energy (mJ/tx)")
    plt.xticks(rotation=45, ha="right")
    plt.title("Energy Comparison Between Scenarios", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"energy_comparison_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 5. Heat map chart
    plt.figure(figsize=(16, 12))
    metrics = ["Throughput (tx/s)", "Latency (ms)", "Energy (mJ/tx)", "Security Score", "Cross-Shard Ratio"]
    
    # Normalize data for heat map
    df_heatmap = df[["Scenario Name"] + metrics].set_index("Scenario Name")
    for col in df_heatmap.columns:
        if col in ["Latency (ms)", "Energy (mJ/tx)"]:  # Metrics where lower is better
            df_heatmap[col] = (df_heatmap[col].max() - df_heatmap[col]) / (df_heatmap[col].max() - df_heatmap[col].min())
        else:  # Metrics where higher is better
            df_heatmap[col] = (df_heatmap[col] - df_heatmap[col].min()) / (df_heatmap[col].max() - df_heatmap[col].min())
    
    sns.heatmap(df_heatmap, annot=True, cmap="viridis", linewidths=.5)
    plt.title("Heat Matrix of Metrics Between Scenarios (normalized)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"metrics_heatmap_{timestamp}.png"), dpi=300)
    plt.close()
    
    # 6. 3D scatter plot: Throughput vs Latency vs Security
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x = df["Throughput (tx/s)"]
        y = df["Latency (ms)"]
        z = df["Security Score"]
        
        ax.scatter(x, y, z, c=df["Energy (mJ/tx)"], cmap="plasma", s=100, alpha=0.7)
        
        for i, scenario in enumerate(df["Scenario Name"]):
            ax.text(x[i], y[i], z[i], scenario, fontsize=8)
        
        ax.set_xlabel("Throughput (tx/s)")
        ax.set_ylabel("Latency (ms)")
        ax.set_zlabel("Security Score")
        plt.title("3D Scatter: Throughput vs Latency vs Security", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"3d_scatter_{timestamp}.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not create 3D chart: {str(e)}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark Runner for QTrust")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # list-scenarios command
    list_parser = subparsers.add_parser("list-scenarios", help="List available benchmark scenarios")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run a benchmark scenario")
    run_parser.add_argument("scenario_id", help="ID of the scenario to run")
    run_parser.add_argument("--output-dir", help="Optional output directory")
    run_parser.add_argument("--quiet", action="store_true", help="Quiet mode (no detailed printing)")
    
    # run-all command
    run_all_parser = subparsers.add_parser("run-all", help="Run all benchmark scenarios")
    run_all_parser.add_argument("--scenario-ids", nargs="+", help="List of scenario IDs to run")
    run_all_parser.add_argument("--parallel", action="store_true", help="Run scenarios in parallel")
    run_all_parser.add_argument("--max-workers", type=int, help="Maximum number of workers if running in parallel")
    run_all_parser.add_argument("--quiet", action="store_true", help="Quiet mode (no detailed printing)")
    
    # compare command
    compare_parser = subparsers.add_parser("compare", help="Generate comparison report from benchmark results")
    compare_parser.add_argument("--results-dir", help="Batch results directory")
    compare_parser.add_argument("--output-file", help="Path to save CSV report")
    
    return parser.parse_args()

def main():
    """Main entry point for the benchmark tool."""
    args = parse_args()
    
    if args.command == "list-scenarios":
        # List scenarios
        print("Available benchmark scenarios:")
        print("-" * 80)
        
        for scenario_id, scenario in get_all_scenarios().items():
            print(f"ID: {scenario_id}")
            print(f"Name: {scenario.name}")
            print(f"Description: {scenario.description}")
            print(f"Parameters: {scenario.get_command_line_args()}")
            print("-" * 80)
    
    elif args.command == "run":
        # Run a scenario
        verbose = not args.quiet
        run_benchmark(args.scenario_id, args.output_dir, verbose)
    
    elif args.command == "run-all":
        # Run all scenarios
        verbose = not args.quiet
        scenario_ids = args.scenario_ids
        run_all_benchmarks(scenario_ids, args.parallel, args.max_workers, verbose)
    
    elif args.command == "compare":
        # Generate comparison report
        generate_comparison_report(args.results_dir, args.output_file)
    
    else:
        print("Invalid command. Run 'benchmark_runner.py -h' for help.")

if __name__ == "__main__":
    main() 