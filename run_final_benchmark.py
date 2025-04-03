#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QTrust Final Benchmark Runner

This script runs comprehensive benchmarks for the QTrust blockchain platform and saves the results
in a structured format. It executes various tests including cache optimization, benchmark comparisons
with other blockchain systems, attack simulations, and HTDCM performance tests. All results are
organized and saved with timestamps for future reference and analysis.
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def create_directories():
    """Create necessary directories to store results."""
    directories = [
        'cleaned_results',
        'cleaned_results/cache',
        'cleaned_results/attack',
        'cleaned_results/benchmark',
        'cleaned_results/htdcm',
        'cleaned_results/charts'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return directories

def run_command(command, description=None, save_output=True, output_file=None):
    """Run command and log the execution."""
    print(f"\n{'='*80}")
    if description:
        print(f"{description}")
    print(f"Command: {command}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    if save_output and output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                command,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
    else:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        if result.stderr:
            print(f"Error: {result.stderr}")
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds. Exit code: {result.returncode}\n")
    
    return result.returncode == 0, result.stdout if not save_output else None

def clean_old_results():
    """Clean up old results."""
    dirs_to_clean = [
        'results',
        'logs',
        'benchmark_results',
        'energy_results'
    ]
    
    for directory in dirs_to_clean:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"Deleted directory: {directory}")
            except Exception as e:
                print(f"Could not delete directory {directory}: {e}")
                
    # Delete log files and temporary png files
    for file in os.listdir('.'):
        if file.endswith('.log') or (file.endswith('.png') and file != 'qtrust_logo.png'):
            try:
                os.remove(file)
                print(f"Deleted file: {file}")
            except Exception as e:
                print(f"Could not delete file {file}: {e}")

def run_final_benchmark():
    """Run final benchmark and save results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    create_directories()
    
    # Clean old results
    clean_old_results()
    
    # Create temporary results directory
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/cache", exist_ok=True)
    os.makedirs("results/attack", exist_ok=True)
    os.makedirs("results/benchmark", exist_ok=True)
    os.makedirs("results/charts", exist_ok=True)
    
    all_results = {
        "timestamp": timestamp,
        "tests": {}
    }
    
    # 1. Run cache optimization test - optimal configuration
    command = "py -3.10 tests/cache_optimization_test.py --shards 32 --nodes 24 --steps 50 --tx 500"
    description = "Running cache optimization test with optimal configuration (32 shards, 24 nodes/shard)"
    output_file = f"results/cache/cache_optimization_{timestamp}.log"
    
    success, _ = run_command(command, description, True, output_file)
    all_results["tests"]["cache_optimization"] = {
        "success": success,
        "command": command,
        "output_file": output_file
    }
    
    # Copy cache charts if they exist
    cache_charts = [f for f in os.listdir('.') if f.startswith('cache_') and f.endswith('.png')]
    for chart in cache_charts:
        try:
            shutil.copy(chart, f"cleaned_results/cache/{timestamp}_{chart}")
            all_results["tests"]["cache_optimization"]["charts"] = [f"cleaned_results/cache/{timestamp}_{chart}"]
        except Exception as e:
            print(f"Could not copy cache chart {chart}: {e}")
    
    # 2. Run benchmark comparison with other systems
    command = "py -3.10 tests/benchmark_comparison_systems.py --output-dir results/benchmark"
    description = "Comparing QTrust with other blockchain systems"
    output_file = f"results/benchmark/benchmark_comparison_{timestamp}.log"
    
    success, _ = run_command(command, description, True, output_file)
    all_results["tests"]["benchmark_comparison"] = {
        "success": success,
        "command": command,
        "output_file": output_file
    }
    
    # Copy benchmark results
    benchmark_files = [f for f in os.listdir('results/benchmark') if f.endswith('.png') or f.endswith('.json') or f.endswith('.csv')]
    for file in benchmark_files:
        try:
            shutil.copy(f"results/benchmark/{file}", f"cleaned_results/benchmark/{timestamp}_{file}")
            if "charts" not in all_results["tests"]["benchmark_comparison"]:
                all_results["tests"]["benchmark_comparison"]["charts"] = []
            all_results["tests"]["benchmark_comparison"]["charts"].append(f"cleaned_results/benchmark/{timestamp}_{file}")
        except Exception as e:
            print(f"Could not copy benchmark file {file}: {e}")
    
    # 3. Run attack simulation
    command = "py -3.10 tests/attack_simulation_runner.py --num-shards 32 --nodes-per-shard 24 --attack-type all --output-dir results/attack"
    description = "Simulating attacks on QTrust (32 shards, 24 nodes/shard)"
    output_file = f"results/attack/attack_simulation_{timestamp}.log"
    
    success, _ = run_command(command, description, True, output_file)
    all_results["tests"]["attack_simulation"] = {
        "success": success,
        "command": command,
        "output_file": output_file
    }
    
    # 4. Run plot attack comparison
    command = "py -3.10 tests/plot_attack_comparison.py --output-dir results/charts"
    description = "Plotting attack comparison charts"
    output_file = f"results/attack/plot_attack_{timestamp}.log"
    
    success, _ = run_command(command, description, True, output_file)
    all_results["tests"]["attack_comparison"] = {
        "success": success,
        "command": command,
        "output_file": output_file
    }
    
    # Copy attack analysis results
    attack_files = []
    if os.path.exists('results/attack'):
        attack_files = [f for f in os.listdir('results/attack') if f.endswith('.png') or f.endswith('.json') or f.endswith('.csv')]
    chart_files = []
    if os.path.exists('results/charts'):
        chart_files = [f for f in os.listdir('results/charts') if f.endswith('.png') or f.endswith('.json') or f.endswith('.csv')]
    
    for file in attack_files:
        try:
            shutil.copy(f"results/attack/{file}", f"cleaned_results/attack/{timestamp}_{file}")
            if "charts" not in all_results["tests"]["attack_simulation"]:
                all_results["tests"]["attack_simulation"]["charts"] = []
            all_results["tests"]["attack_simulation"]["charts"].append(f"cleaned_results/attack/{timestamp}_{file}")
        except Exception as e:
            print(f"Could not copy attack file {file}: {e}")
            
    for file in chart_files:
        try:
            shutil.copy(f"results/charts/{file}", f"cleaned_results/charts/{timestamp}_{file}")
            if "charts" not in all_results["tests"]["attack_comparison"]:
                all_results["tests"]["attack_comparison"]["charts"] = []
            all_results["tests"]["attack_comparison"]["charts"].append(f"cleaned_results/charts/{timestamp}_{file}")
        except Exception as e:
            print(f"Could not copy chart file {file}: {e}")
    
    # 5. Run HTDCM performance test
    command = "py -3.10 tests/htdcm_performance_test.py"
    description = "Testing HTDCM performance"
    output_file = f"results/htdcm/htdcm_performance_{timestamp}.log"
    
    success, _ = run_command(command, description, True, output_file)
    all_results["tests"]["htdcm_performance"] = {
        "success": success,
        "command": command,
        "output_file": output_file
    }
    
    # Save results summary
    with open(f"cleaned_results/benchmark_summary_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    
    # Create README file in cleaned_results directory
    with open("cleaned_results/README.md", 'w', encoding='utf-8') as f:
        f.write(f"# QTrust Benchmark Results\n\n")
        f.write(f"Results from benchmark run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Test | Status | Description |\n")
        f.write("|------|--------|-------------|\n")
        
        for test_name, test_info in all_results["tests"].items():
            status = "✅ Success" if test_info["success"] else "❌ Failed"
            description = test_info["command"]
            f.write(f"| {test_name} | {status} | `{description}` |\n")
        
        f.write("\n## Details\n\n")
        for test_name, test_info in all_results["tests"].items():
            f.write(f"### {test_name}\n\n")
            f.write(f"- Command: `{test_info['command']}`\n")
            f.write(f"- Status: {'Success' if test_info['success'] else 'Failed'}\n")
            
            if "charts" in test_info and test_info["charts"]:
                f.write("\n#### Generated Charts\n\n")
                for chart in test_info["charts"]:
                    f.write(f"- [{os.path.basename(chart)}]({chart})\n")
            
            f.write("\n")
    
    print(f"\n{'='*80}")
    print(f"Benchmark completed. Results have been saved to the cleaned_results directory.")
    print(f"Results summary: cleaned_results/benchmark_summary_{timestamp}.json")
    print(f"{'='*80}\n")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run final benchmark for QTrust.')
    parser.add_argument('--clean-only', action='store_true', help='Only clean old results without running benchmarks')
    
    args = parser.parse_args()
    
    if args.clean_only:
        clean_old_results()
    else:
        run_final_benchmark() 