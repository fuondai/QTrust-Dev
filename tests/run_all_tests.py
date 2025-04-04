#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated Test Runner for QTrust Blockchain System

This script provides a comprehensive test framework that automatically discovers and executes
all test modules in the tests directory. It offers:

1. Automated test discovery and execution
2. Detailed reporting of test results including execution time and success rates
3. Visualization of test performance with customizable output options
4. Support for running specific test modules through command line arguments
5. Clear error reporting for failed tests

Example usage:
    # Run all tests
    python run_all_tests.py
    
    # Run specific test modules
    python run_all_tests.py --modules test_metrics test_config
    
    # Generate visualization plot
    python run_all_tests.py --plot
    
    # Save visualization to file
    python run_all_tests.py --save-plot test_results.png

The script returns exit code 0 if all tests pass, 1 otherwise, making it
suitable for integration into CI/CD pipelines.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import unittest
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import json

# Xóa các import không cần thiết để tránh lỗi
# import test_data_generation
# import test_hyper_optimizer
# import test_metrics
# import test_logging
# import test_visualization
# import test_anomaly_detection
# import test_trust_models
# import test_config
# import test_security
# import test_mad_rapid
# import test_federated_rl
# import test_privacy
# import test_model_aggregation
# import test_federated_manager
# import test_import
# import test_caching
# import simple_cache_test
# import test_adaptive_consensus
# import test_bls_signatures
# import test_lightweight_crypto
# import test_benchmark_runner
# import test_benchmark_scenarios
# import test_blockchain_comparison_utils
# import test_system_comparison
# import test_actor_critic

# Thay vào đó, chúng ta sẽ tự động tìm các module test có sẵn
def find_test_modules():
    """Find all available test modules in tests directory"""
    test_modules = []
    test_path = Path(__file__).parent
    for file_path in test_path.glob("**/*.py"):
        if file_path.name.startswith("test_") and file_path.name != "test_all.py" and file_path.name != "run_all_tests.py":
            # Convert path to module name
            rel_path = file_path.relative_to(test_path)
            module_parts = list(rel_path.parts)
            module_parts[-1] = module_parts[-1][:-3]  # Remove .py extension
            module_name = ".".join(module_parts)
            test_modules.append(module_name)
    
    return test_modules

# Add the root directory to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

def create_directories():
    """Create necessary directories to store results."""
    directories = [
        'results',
        'results/cache',
        'results/attack',
        'results/benchmark',
        'results/htdcm',
        'results/energy',
        'results/charts'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return directories

def run_command(command, description=None, save_output=True, output_file=None):
    """Run a command and log the output."""
    print(f"\n{'='*80}")
    if description:
        print(f"EXECUTING: {description}")
    print(f"COMMAND: {command}")
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
    
    return result.returncode == 0

def run_tests():
    """Run all tests that exist in the file system."""
    print("Starting test execution...")
    create_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_tests = []
    
    # 1. Xuất ra danh sách các module test cụ thể
    available_tests = []
    test_path = Path(__file__).parent
    
    # Tìm các file test ở thư mục tests
    for file_path in test_path.glob("**/*.py"):
        if file_path.name.startswith("test_") and file_path.name != "run_all_tests.py":
            test_name = file_path.stem  # Lấy tên file không có extension
            rel_path = file_path.relative_to(test_path.parent)
            str_path = str(rel_path).replace("\\", "/")
            available_tests.append((test_name, str_path))
    
    print(f"Found {len(available_tests)} test modules:")
    for name, path in available_tests:
        print(f" - {name} ({path})")
    
    # 2. Tạo các test command
    for test_name, test_path in available_tests:
        # Tạo lệnh chạy test
        test = {
            'name': test_name,
            'command': f'py -3.10 {test_path}',
            'output': f'results/{test_name}_{timestamp}.log',
            'description': f'Running {test_name}'
        }
        all_tests.append(test)
    
    # 3. Thêm lệnh chạy benchmark nếu file tồn tại
    benchmark_path = Path(__file__).parent.parent / "benchmark_comparison_systems.py"
    if benchmark_path.exists():
        test = {
            'name': 'Benchmark Comparison',
            'command': f'py -3.10 benchmark_comparison_systems.py --output-dir results/benchmark',
            'output': f'results/benchmark/benchmark_comparison_{timestamp}.log',
            'description': 'Compare QTrust with other blockchain systems'
        }
        all_tests.append(test)
    
    # Kiểm tra nếu không có test nào được tìm thấy, chạy pytest
    if not all_tests:
        test = {
            'name': 'Pytest Run All',
            'command': f'py -3.10 -m pytest tests/',
            'output': f'results/pytest_all_{timestamp}.log',
            'description': 'Run all tests using pytest'
        }
        all_tests.append(test)
    
    # 4. Chạy các test
    success_count = 0
    for test in all_tests:
        print(f"\nRunning test: {test['name']}")
        print(f"Description: {test['description']}")
        success = run_command(test['command'], test['description'], True, test['output'])
        if success:
            success_count += 1
            print(f"{test['name']} - SUCCESS ✅")
        else:
            print(f"{test['name']} - FAILED ❌")
    
    # 5. In kết quả
    print(f"\n{'='*80}")
    print(f"Test Summary: {success_count}/{len(all_tests)} tests passed")
    print(f"{'='*80}")
    
    if success_count == len(all_tests):
        print("\nAll tests passed! ✅")
        return True
    else:
        print("\nSome tests failed! ❌")
        return False

class TestResult:
    """Class to hold test results."""
    
    def __init__(self, name: str, success: bool, execution_time: float, error_message: str = None):
        self.name = name
        self.success = success
        self.execution_time = execution_time
        self.error_message = error_message
        
    def __str__(self):
        status = "PASS" if self.success else "FAIL"
        return f"{self.name}: {status} ({self.execution_time:.2f}s)"

def discover_test_modules() -> List[str]:
    """
    Discover test modules in the tests directory.
    
    Returns:
        List[str]: List of test module names
    """
    test_modules = []
    test_files = [f for f in os.listdir(".") if f.startswith("test_") and f.endswith(".py")]
    
    for file in test_files:
        module_name = file[:-3]  # Remove .py extension
        test_modules.append(module_name)
    
    return test_modules

def run_test_module(module_name: str) -> Tuple[bool, float, str]:
    """
    Run a test module and return results.
    
    Args:
        module_name: Name of the test module
        
    Returns:
        Tuple[bool, float, str]: Success status, execution time, and error message
    """
    start_time = time.time()
    success = True
    error_message = None
    
    try:
        # Import the module
        __import__(module_name)
        module = sys.modules[module_name]
        
        # Create a test suite from the module
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        
        # Run the test suite
        test_runner = unittest.TextTestRunner(verbosity=0)
        result = test_runner.run(suite)
        
        # Check if test was successful
        if result.failures or result.errors:
            success = False
            if result.failures:
                error_message = str(result.failures[0][1])
            else:
                error_message = str(result.errors[0][1])
    except Exception as e:
        success = False
        error_message = str(e)
    
    execution_time = time.time() - start_time
    
    return success, execution_time, error_message

def run_all_tests(modules: List[str] = None) -> List[TestResult]:
    """
    Run all test modules.
    
    Args:
        modules: List of test module names to run (optional)
        
    Returns:
        List[TestResult]: List of test results
    """
    if modules is None:
        modules = discover_test_modules()
    
    results = []
    
    print(f"Running {len(modules)} test modules...")
    print("-" * 80)
    
    for module_name in modules:
        print(f"Running {module_name}...", end="", flush=True)
        success, execution_time, error_message = run_test_module(module_name)
        result = TestResult(module_name, success, execution_time, error_message)
        results.append(result)
        
        status = "PASSED" if success else "FAILED"
        print(f" {status} ({execution_time:.3f}s)")
    
    print("-" * 80)
    
    return results

def generate_test_report(results: List[TestResult]) -> Dict[str, Any]:
    """
    Generate a report from test results.
    
    Args:
        results: List of test results
        
    Returns:
        Dict[str, Any]: Test report containing stats and results
    """
    successful_tests = [r for r in results if r.success]
    failed_tests = [r for r in results if not r.success]
    
    total_time = sum(r.execution_time for r in results)
    
    report = {
        "total_tests": len(results),
        "successful_tests": len(successful_tests),
        "failed_tests": len(failed_tests),
        "total_execution_time": total_time,
        "success_rate": len(successful_tests) / len(results) if results else 0,
        "results": results
    }
    
    return report

def print_test_report(report: Dict[str, Any]):
    """
    Print test report.
    
    Args:
        report: Test report
    """
    try:
        print("\nTest Report:")
        print(f"Total tests: {report['total_tests']}")
        print(f"Successful tests: {report['successful_tests']}")
        print(f"Failed tests: {report['failed_tests']}")
        print(f"Total execution time: {report['total_execution_time']:.3f}s")
        print(f"Success rate: {report['success_rate'] * 100:.2f}%")
        
        if report['failed_tests'] > 0 and 'results' in report and report['results']:
            print("\nFailed tests:")
            for result in report['results']:
                if not result.success:
                    print(f"  {result}")
    except:
        # Nếu có lỗi I/O, bỏ qua
        pass

def plot_test_results(report: Dict[str, Any], save_path: str = None):
    """
    Plot test results.
    
    Args:
        report: Test report
        save_path: Path to save the plot
    """
    try:
        # Get execution times and sort
        if "results" not in report or not report["results"]:
            return

        sorted_results = sorted(report['results'], key=lambda r: r.execution_time if hasattr(r, "execution_time") else 0.0, reverse=True)
        names = [r.name if hasattr(r, "name") else "unknown" for r in sorted_results]
        times = [r.execution_time if hasattr(r, "execution_time") else 0.0 for r in sorted_results]
        colors = ['green' if hasattr(r, "success") and r.success else 'red' for r in sorted_results]
        
        # Plot execution times
        plt.figure(figsize=(12, 8))
        plt.barh(names, times, color=colors)
        plt.xlabel('Execution Time (s)')
        plt.ylabel('Test Module')
        plt.title('Test Execution Times')
        
        # Add success rate and total time in text
        success_rate = report['success_rate'] * 100
        total_time = report['total_execution_time']
        plt.figtext(0.5, 0.01, f"Success Rate: {success_rate:.2f}% | Total Time: {total_time:.3f}s", 
                    ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    except Exception as e:
        try:
            print(f"Error plotting test results: {e}")
        except:
            pass

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='QTrust test runner')
    parser.add_argument('--modules', type=str, nargs='*', 
                        help='List of test modules to run')
    parser.add_argument('--plot', action='store_true',
                        help='Plot test execution times')
    parser.add_argument('--save-plot', type=str, 
                        help='Save plot to file')
    
    return parser.parse_args()

def save_results_to_json(report: Dict[str, Any], filename: str = "test_results.json"):
    """
    Save test results to a JSON file.
    
    Args:
        report: Test report
        filename: Name of the JSON file
    """
    try:
        # Convert TestResult objects to dictionaries
        serializable_report = report.copy()
        if "results" in serializable_report and serializable_report["results"]:
            serializable_report["results"] = [
                {
                    "name": r.name if hasattr(r, "name") else "unknown",
                    "success": r.success if hasattr(r, "success") else False,
                    "execution_time": r.execution_time if hasattr(r, "execution_time") else 0.0,
                    "error_message": r.error_message if hasattr(r, "error_message") else None
                }
                for r in report["results"]
            ]
        
        # Save to JSON file
        try:
            os.makedirs("results", exist_ok=True)
            with open(os.path.join("results", filename), "w") as f:
                json.dump(serializable_report, f, indent=2)
        except Exception as e:
            print(f"Error saving results to JSON: {e}")
    except:
        # Nếu có lỗi, bỏ qua
        pass

def load_results_from_json(filename: str = "test_results.json") -> Dict[str, Any]:
    """
    Load test results from a JSON file.
    
    Args:
        filename: Name of the JSON file
        
    Returns:
        Dict[str, Any]: Test report
    """
    json_path = os.path.join("results", filename)
    
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, "r") as f:
        serialized_report = json.load(f)
    
    # Convert dictionaries to TestResult objects
    report = serialized_report.copy()
    report["results"] = [
        TestResult(
            r["name"],
            r["success"],
            r["execution_time"],
            r["error_message"]
        )
        for r in serialized_report["results"]
    ]
    
    return report

def main():
    """Main function."""
    args = parse_arguments()
    
    if args.modules:
        # Run only specified modules
        results = run_all_tests(args.modules)
    else:
        # Run all discovered modules
        modules = discover_test_modules()
        results = run_all_tests(modules)
    
    # Generate report
    report = generate_test_report(results)
    
    # Print report
    print_test_report(report)
    
    # Save results to JSON if specified
    if args.save_json:
        save_results_to_json(report, args.save_json)
    
    # Plot results if specified
    if args.plot or args.save_plot:
        save_path = args.save_plot if args.save_plot else None
        plot_test_results(report, save_path)
    
    # Return exit code based on success
    return 0 if report["overall_success"] else 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Parse command line arguments when provided
        parser = argparse.ArgumentParser(description='Run all tests or specific test modules.')
        parser.add_argument('--modules', nargs='+', help='Run only specified test modules')
        parser.add_argument('--skip-long', action='store_true', help='Skip tests marked as long-running')
        args = parser.parse_args()
        
        if args.modules:
            print(f"Running specified test modules: {', '.join(args.modules)}")
            success = run_tests()
        else:
            print("Running all tests")
            success = run_tests()
    else:
        # Run all tests if no arguments provided
        print("Running all tests")
        success = run_tests()
    
    sys.exit(0 if success else 1) 