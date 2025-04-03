#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QTrust Complete Workflow Runner

This script orchestrates the entire QTrust workflow, executing all steps from testing to benchmarking
and visualization. It manages the process flow, handles errors, and provides summary information
about the execution. The script allows for customization of the workflow through command-line options.
"""

import os
import sys  
import time
import argparse
import subprocess
from datetime import datetime

def run_command(command, description=None):
    """Run command and log execution."""
    print(f"\n{'='*80}")
    if description:
        print(f"{description}")
    print(f"Command: {command}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
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
    
    return result.returncode == 0

def run_all(args):
    """Run the entire workflow."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting QTrust complete workflow at {timestamp}")
    
    # Clean if needed
    if args.clean:
        run_command("py -3.10 run_final_benchmark.py --clean-only", "Cleaning results directories")
    
    # 1. Run all tests
    if not args.skip_tests:
        success = run_command("py -3.10 tests/run_all_tests.py", "Running all tests")
        if not success and not args.ignore_failures:
            print("Tests failed. Stopping process.")
            return False
    
    # 2. Run final benchmark
    if not args.skip_benchmark:
        success = run_command("py -3.10 run_final_benchmark.py", "Running final benchmark")
        if not success and not args.ignore_failures:
            print("Benchmark failed. Stopping process.")
            return False
    
    # 3. Generate results charts
    if not args.skip_charts:
        success = run_command("py -3.10 generate_final_charts.py", "Generating results charts")
        if not success and not args.ignore_failures:
            print("Chart generation failed. Stopping process.")
            return False
    
    # 4. Display overview information
    end_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*80}")
    print(f"QTrust has completed all steps.")
    print(f"Start: {timestamp}")
    print(f"End: {end_timestamp}")
    print(f"{'='*80}")
    
    print("\nImportant documents:")
    print("- README.md: Project overview")
    print("- docs/architecture/qtrust_architecture.md: QTrust architecture")
    print("- docs/methodology/qtrust_methodology.md: Research methodology")
    print("- docs/exported_charts/index.html: Results charts")
    print("- cleaned_results/README.md: Benchmark results")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run all steps from start to finish for QTrust.')
    parser.add_argument('--clean', action='store_true', help='Clean results directories before running')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--skip-benchmark', action='store_true', help='Skip running benchmark')
    parser.add_argument('--skip-charts', action='store_true', help='Skip generating charts')
    parser.add_argument('--ignore-failures', action='store_true', help='Continue even if errors occur')
    
    args = parser.parse_args()
    
    success = run_all(args)
    sys.exit(0 if success else 1) 