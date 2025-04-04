"""
QTrust Benchmark Tool

Script này thực hiện benchmark hiệu năng của hệ thống QTrust.
Nó đo lường throughput, độ trễ, tỷ lệ thành công, và tiêu thụ năng lượng
trong các kịch bản khác nhau.
"""

import argparse
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os

from qtrust.simulation.system_simulator import SystemSimulator
from qtrust.utils.paths import get_chart_path
from qtrust.utils.metrics import calculate_transaction_throughput

def run_benchmark(total_transactions: int = 1000, 
                 num_shards: int = 4,
                 nodes_per_shard: int = 5,
                 enable_parallel: bool = True,
                 enable_caching: bool = True,
                 verbose: bool = True) -> Dict[str, Any]:
    """
    Chạy benchmark cho hệ thống QTrust.
    
    Args:
        total_transactions: Tổng số giao dịch cần xử lý
        num_shards: Số lượng shard
        nodes_per_shard: Số node trong mỗi shard
        enable_parallel: Bật xử lý song song
        enable_caching: Bật cache
        verbose: In kết quả chi tiết
        
    Returns:
        Dict[str, Any]: Kết quả benchmark
    """
    start_time = time.time()
    
    # Khởi tạo simulator
    simulator = SystemSimulator(
        num_shards=num_shards,
        num_validators_per_shard=nodes_per_shard,
        enable_parallel_processing=enable_parallel
    )
    
    # Thiết lập caching cho transaction processor nếu có
    if hasattr(simulator, 'transaction_processor') and simulator.transaction_processor is not None:
        simulator.transaction_processor.enable_caching = enable_caching
    
    # Tạo danh sách giao dịch
    transactions = simulator.generate_random_transactions(total_transactions)
    
    if verbose:
        print(f"Generated {len(transactions)} transactions")
        print(f"Cross-shard transactions: {sum(1 for tx in transactions if tx.get('cross_shard', False))}")
    
    # Bắt đầu benchmark
    benchmark_start = time.time()
    results = simulator.run_simulation(
        num_transactions=total_transactions,
        existing_transactions=transactions
    )
    benchmark_end = time.time()
    
    # Tính kết quả
    elapsed_time = benchmark_end - benchmark_start
    throughput = calculate_transaction_throughput(results.get("successful_transactions", 0), elapsed_time)
    success_rate = results.get("successful_transactions", 0) / total_transactions if total_transactions > 0 else 0
    
    # Lấy thông tin cache hit ratio nếu có
    cache_hit_ratio = 0
    if hasattr(simulator, 'transaction_processor') and simulator.transaction_processor is not None:
        processor_stats = simulator.transaction_processor.get_processing_stats()
        cache_hit_ratio = processor_stats.get("cache_hit_ratio", 0)
    
    # Tạo kết quả benchmark
    benchmark_results = {
        "total_time": elapsed_time,
        "throughput": throughput,
        "success_rate": success_rate * 100,
        "avg_latency": results.get("avg_latency", 0),
        "total_energy": results.get("total_energy", 0),
        "transactions_processed": total_transactions,
        "successful_transactions": results.get("successful_transactions", 0),
        "cross_shard_ratio": sum(1 for tx in transactions if tx.get('cross_shard', False)) / total_transactions,
        "cache_hit_ratio": cache_hit_ratio * 100 if enable_caching else 0,
        "parallel_enabled": enable_parallel,
        "caching_enabled": enable_caching,
        "num_shards": num_shards,
        "nodes_per_shard": nodes_per_shard
    }
    
    if verbose:
        print("\n===== BENCHMARK RESULTS =====")
        print(f"Configuration: {num_shards} shards, {nodes_per_shard} nodes per shard")
        print(f"Features: {'Parallel' if enable_parallel else 'Sequential'} processing, {'With' if enable_caching else 'Without'} caching")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Throughput: {throughput:.2f} tx/s")
        print(f"Success rate: {success_rate * 100:.2f}%")
        print(f"Average latency: {results.get('avg_latency', 0):.2f} ms")
        
        if hasattr(simulator, 'transaction_processor') and simulator.transaction_processor is not None:
            print(f"Cache hit ratio: {cache_hit_ratio * 100:.2f}%")
        else:
            print("Cache statistics not available")
            
        print(f"Cross-shard ratio: {benchmark_results['cross_shard_ratio'] * 100:.2f}%")
        print("============================")
    
    return benchmark_results

def compare_configurations(transaction_counts: List[int] = [100, 500, 1000], 
                          num_runs: int = 3,
                          output_chart: bool = True) -> pd.DataFrame:
    """
    So sánh hiệu năng giữa chế độ xử lý tuần tự và song song.
    
    Args:
        transaction_counts: Danh sách các số lượng giao dịch để kiểm tra
        num_runs: Số lần chạy cho mỗi cấu hình để lấy trung bình
        output_chart: Xuất biểu đồ so sánh
        
    Returns:
        pd.DataFrame: Kết quả so sánh dạng bảng
    """
    results = []
    
    for tx_count in transaction_counts:
        print(f"\nRunning benchmark with {tx_count} transactions...")
        
        # Kết quả trung bình cho mỗi cấu hình
        sequential_results = {"throughput": [], "success_rate": [], "avg_latency": [], "cache_hit_ratio": []}
        parallel_results = {"throughput": [], "success_rate": [], "avg_latency": [], "cache_hit_ratio": []}
        
        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs}...")
            
            # Chạy với xử lý tuần tự
            seq_result = run_benchmark(
                total_transactions=tx_count,
                enable_parallel=False,
                enable_caching=True,
                verbose=False
            )
            
            # Chạy với xử lý song song
            par_result = run_benchmark(
                total_transactions=tx_count,
                enable_parallel=True,
                enable_caching=True,
                verbose=False
            )
            
            # Thu thập kết quả
            for key in sequential_results.keys():
                sequential_results[key].append(seq_result.get(key, 0))
                parallel_results[key].append(par_result.get(key, 0))
        
        # Tính trung bình cho mỗi cấu hình
        seq_avg = {k: np.mean(v) for k, v in sequential_results.items()}
        par_avg = {k: np.mean(v) for k, v in parallel_results.items()}
        
        # Tính speedup
        speedup = par_avg["throughput"] / seq_avg["throughput"] if seq_avg["throughput"] > 0 else 0
        
        # In kết quả
        print(f"\n===== RESULTS FOR {tx_count} TRANSACTIONS =====")
        print(f"Sequential throughput: {seq_avg['throughput']:.2f} tx/s")
        print(f"Parallel throughput: {par_avg['throughput']:.2f} tx/s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Sequential success rate: {seq_avg['success_rate']:.2f}%")
        print(f"Parallel success rate: {par_avg['success_rate']:.2f}%")
        print(f"Sequential latency: {seq_avg['avg_latency']:.2f} ms")
        print(f"Parallel latency: {par_avg['avg_latency']:.2f} ms")
        print("=======================================")
        
        # Thêm vào kết quả
        results.append({
            "tx_count": tx_count,
            "seq_throughput": seq_avg["throughput"],
            "par_throughput": par_avg["throughput"],
            "speedup": speedup,
            "seq_success_rate": seq_avg["success_rate"],
            "par_success_rate": par_avg["success_rate"],
            "seq_latency": seq_avg["avg_latency"],
            "par_latency": par_avg["avg_latency"],
            "seq_cache_hit": seq_avg["cache_hit_ratio"],
            "par_cache_hit": par_avg["cache_hit_ratio"]
        })
    
    # Tạo DataFrame từ kết quả
    results_df = pd.DataFrame(results)
    
    # Tạo biểu đồ so sánh nếu được yêu cầu
    if output_chart:
        plt.figure(figsize=(15, 10))
        
        # Biểu đồ throughput
        plt.subplot(2, 2, 1)
        plt.bar(
            np.arange(len(transaction_counts)) - 0.2, 
            results_df["seq_throughput"], 
            width=0.4, 
            label="Sequential"
        )
        plt.bar(
            np.arange(len(transaction_counts)) + 0.2, 
            results_df["par_throughput"], 
            width=0.4, 
            label="Parallel"
        )
        plt.xticks(np.arange(len(transaction_counts)), transaction_counts)
        plt.xlabel("Transaction Count")
        plt.ylabel("Throughput (tx/s)")
        plt.title("Throughput Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Biểu đồ speedup
        plt.subplot(2, 2, 2)
        plt.bar(np.arange(len(transaction_counts)), results_df["speedup"])
        plt.xticks(np.arange(len(transaction_counts)), transaction_counts)
        plt.xlabel("Transaction Count")
        plt.ylabel("Speedup (x)")
        plt.title("Speedup from Parallel Processing")
        plt.grid(True, alpha=0.3)
        
        # Biểu đồ success rate
        plt.subplot(2, 2, 3)
        plt.bar(
            np.arange(len(transaction_counts)) - 0.2, 
            results_df["seq_success_rate"], 
            width=0.4, 
            label="Sequential"
        )
        plt.bar(
            np.arange(len(transaction_counts)) + 0.2, 
            results_df["par_success_rate"], 
            width=0.4, 
            label="Parallel"
        )
        plt.xticks(np.arange(len(transaction_counts)), transaction_counts)
        plt.xlabel("Transaction Count")
        plt.ylabel("Success Rate (%)")
        plt.title("Success Rate Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Biểu đồ latency
        plt.subplot(2, 2, 4)
        plt.bar(
            np.arange(len(transaction_counts)) - 0.2, 
            results_df["seq_latency"], 
            width=0.4, 
            label="Sequential"
        )
        plt.bar(
            np.arange(len(transaction_counts)) + 0.2, 
            results_df["par_latency"], 
            width=0.4, 
            label="Parallel"
        )
        plt.xticks(np.arange(len(transaction_counts)), transaction_counts)
        plt.xlabel("Transaction Count")
        plt.ylabel("Latency (ms)")
        plt.title("Latency Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        chart_path = get_chart_path("throughput_benchmark.png", "benchmark")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"\nBenchmark chart saved to: {chart_path}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="QTrust Benchmark Tool")
    parser.add_argument("--total-transactions", type=int, default=1000, help="Total transactions to process")
    parser.add_argument("--num-shards", type=int, default=4, help="Number of shards")
    parser.add_argument("--nodes-per-shard", type=int, default=5, help="Nodes per shard")
    parser.add_argument("--compare", action="store_true", help="Compare sequential vs parallel processing")
    parser.add_argument("--tx-counts", type=int, nargs="+", default=[100, 500, 1000], help="Transaction counts for comparison")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs for each configuration")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    if args.compare:
        print("Comparing sequential vs parallel processing...")
        results_df = compare_configurations(
            transaction_counts=args.tx_counts,
            num_runs=args.num_runs
        )
        print("\nComparison results:")
        print(results_df.to_string(index=False))
    else:
        print(f"Running benchmark with {args.total_transactions} transactions...")
        results = run_benchmark(
            total_transactions=args.total_transactions,
            num_shards=args.num_shards,
            nodes_per_shard=args.nodes_per_shard,
            enable_parallel=not args.no_parallel,
            enable_caching=not args.no_cache,
            verbose=True
        )

if __name__ == "__main__":
    main() 