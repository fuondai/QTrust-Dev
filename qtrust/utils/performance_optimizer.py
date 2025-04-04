"""
Performance optimization tools for the QTrust system.

This module provides tools and classes to optimize blockchain system performance,
including parallel transaction processing, intelligent caching, and performance analysis.
"""

import time
import logging
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Callable, Optional
from functools import lru_cache
import numpy as np
import hashlib
import json

from qtrust.utils.logging import get_logger

# Set up logger
logger = get_logger(__name__)

# Decorator to measure execution time
def timing_decorator(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper


class ParallelTransactionProcessor:
    """
    Process transactions in parallel to increase throughput.
    
    This class enables processing multiple transactions simultaneously using multithreading
    or multiprocessing, maximizing the use of available CPU resources.
    """
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False, enable_caching: bool = True):
        """
        Initialize parallel transaction processor.
        
        Args:
            max_workers: Maximum number of workers. If None, uses CPU count * 2
            use_processes: If True, uses multiprocessing instead of multithreading
            enable_caching: Enable intelligent caching for similar transactions
        """
        self.max_workers = max_workers or min(32, multiprocessing.cpu_count() * 2)
        self.use_processes = use_processes
        self.executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        self.enable_caching = enable_caching
        
        # Initialize transaction result cache
        self.transaction_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 1000  # Limit cache size
        
        self.processing_stats = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "throughput": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_ratio": 0.0
        }
        
        logger.info(f"Initialized ParallelTransactionProcessor with {self.max_workers} workers "
                   f"using {'processes' if use_processes else 'threads'}"
                   f" and caching {'enabled' if enable_caching else 'disabled'}")
        
    def _cache_key(self, transaction):
        """
        Generate a cache key for a transaction.
        
        Creates a deterministic key based on transaction attributes that affect the 
        execution result. Ignores mutable fields like ID and timestamp to optimize
        cache hits for repeating transactions.
        
        Args:
            transaction: The transaction to generate a key for
            
        Returns:
            str: A unique cache key string based on transaction content
        """
        import hashlib
        
        # Extract core fields that determine transaction outcome
        sender = transaction.get('sender', '')
        receiver = transaction.get('receiver', '')
        sender_shard = transaction.get('sender_shard', '')
        receiver_shard = transaction.get('receiver_shard', '')
        value = transaction.get('value', 0)
        tx_type = transaction.get('type', 'transfer')
        
        # Start with core attributes that define transaction behavior
        key_parts = [
            f"sender:{sender}",
            f"receiver:{receiver}",
            f"sender_shard:{sender_shard}", 
            f"receiver_shard:{receiver_shard}",
            f"value:{value}",
            f"type:{tx_type}"
        ]
        
        # Add additional fields if present, limiting data field to first 32 bytes to keep key manageable
        if 'data' in transaction:
            data = transaction['data']
            # Limit data size to prevent excessively large keys
            if isinstance(data, str) and len(data) > 32:
                data = data[:32]
            key_parts.append(f"data:{data}")
        
        if 'gas_price' in transaction:
            key_parts.append(f"gas_price:{transaction['gas_price']}")
        
        if 'nonce' in transaction:
            key_parts.append(f"nonce:{transaction['nonce']}")
        
        # Create deterministic key string from all parts
        key_str = "|".join(key_parts)
        
        # Generate MD5 hash for a consistent and compact key format
        cache_key = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        
        # Debug logging
        if transaction.get('debug_marker') == 'repeating_tx':
            logger.debug(f"Generated cache key for repeating transaction: {cache_key}")
        
        return cache_key
    
    def _check_cache(self, transaction: Dict[str, Any]) -> Optional[Tuple[bool, float]]:
        """
        Check cache for a transaction.
        
        Args:
            transaction: Transaction to check
            
        Returns:
            Optional[Tuple[bool, float]]: Result from cache or None if not present
        """
        if not self.enable_caching:
            return None
        
        # Calculate cache key
        cache_key = self._cache_key(transaction)
        
        # Check and update cache hits/misses
        if cache_key in self.transaction_cache:
            self.cache_hits += 1
            # Get the cached result
            result = self.transaction_cache[cache_key]
            
            # Debug logging for cache hits
            if transaction.get('debug_marker') == 'repeating_tx':
                logger.info(f"Cache HIT for repeating transaction [{transaction.get('id', 'unknown')}] with key {cache_key[:8]}...")
                logger.debug(f"Full cache key: {cache_key}, Result: {result}")
            else:
                logger.debug(f"Cache HIT for transaction with key {cache_key[:8]}...")
                
            # Prioritize this entry by moving it to the end (most recently used)
            self._refresh_cache_entry(cache_key)
            return result
        
        self.cache_misses += 1
        # Debug logging for cache misses
        if transaction.get('debug_marker') == 'repeating_tx':
            logger.info(f"Cache MISS for repeating transaction [{transaction.get('id', 'unknown')}] with key {cache_key[:8]}...")
        else:
            logger.debug(f"Cache MISS for transaction with key {cache_key[:8]}...")
        return None
    
    def _refresh_cache_entry(self, cache_key: str) -> None:
        """
        Move a cache entry to the end of the cache to mark it as recently used.
        
        Args:
            cache_key: Key of the cache entry to refresh
        """
        if not self.enable_caching or cache_key not in self.transaction_cache:
            return
            
        # Get the value
        value = self.transaction_cache[cache_key]
        
        # Remove and re-add to make it the newest entry
        del self.transaction_cache[cache_key]
        self.transaction_cache[cache_key] = value
    
    def _update_cache(self, transaction: Dict[str, Any], result: Tuple[bool, float]) -> None:
        """
        Update cache with new result.
        
        Args:
            transaction: Transaction to update
            result: Processing result
        """
        if not self.enable_caching:
            return
        
        # Calculate cache key
        cache_key = self._cache_key(transaction)
        
        # Protect against memory overflow
        if len(self.transaction_cache) >= self.max_cache_size:
            # Remove 20% oldest records (simply delete from the beginning of the list)
            items_to_remove = int(self.max_cache_size * 0.2)
            keys_to_remove = list(self.transaction_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                if key in self.transaction_cache:
                    del self.transaction_cache[key]
            logger.debug(f"Cache cleanup: removed {len(keys_to_remove)} oldest entries")
        
        # Update cache and log for repeating transactions
        self.transaction_cache[cache_key] = result
        
        if transaction.get('debug_marker') == 'repeating_tx':
            logger.info(f"Cached result for repeating transaction [{transaction.get('id', 'unknown')}] with key {cache_key[:8]}...")
            logger.debug(f"Full cache key: {cache_key}, Result: {result}")
        else:
            logger.debug(f"Cached result for transaction with key {cache_key[:8]}...")
    
    @timing_decorator
    def process_transactions(self, 
                           transactions: List[Dict[str, Any]], 
                           process_func: Callable[[Dict[str, Any]], Tuple[bool, float]],
                           **kwargs) -> Dict[str, Any]:
        """
        Process a list of transactions in parallel.
        
        Args:
            transactions: List of transactions to process
            process_func: Function to process each transaction, takes transaction and returns (success, latency)
            **kwargs: Additional parameters for process_func
            
        Returns:
            Dict[str, Any]: Processing results including performance metrics
        """
        start_time = time.time()
        num_transactions = len(transactions)
        results = []
        cached_results = []
        
        # Reset hits and misses for this processing batch
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Check cache before processing
        transactions_to_process = []
        for tx in transactions:
            cached_result = self._check_cache(tx)
            if cached_result:
                cached_results.append(cached_result)
                # Add information to transaction
                tx['from_cache'] = True
            else:
                transactions_to_process.append(tx)
                tx['from_cache'] = False
        
        if cached_results:
            logger.info(f"Cache hits: {len(cached_results)}/{num_transactions} transactions ({len(cached_results)/num_transactions:.1%})")
        
        # If all are from cache, no further processing needed
        if not transactions_to_process:
            results = cached_results
        else:
            # Classify transactions by complexity (cross-shard or not)
            simple_transactions = [tx for tx in transactions_to_process if tx.get('sender_shard') == tx.get('receiver_shard')]
            complex_transactions = [tx for tx in transactions_to_process if tx.get('sender_shard') != tx.get('receiver_shard')]
            
            # Optimize batch size based on transaction count and complexity
            simple_batch_size = max(1, min(100, len(simple_transactions) // (self.max_workers // 2 or 1)))
            complex_batch_size = max(1, min(50, len(complex_transactions) // (self.max_workers // 2 or 1)))
            
            # Group transactions into batches for better performance
            simple_batches = [
                simple_transactions[i:i + simple_batch_size] 
                for i in range(0, len(simple_transactions), simple_batch_size)
            ] if simple_transactions else []
            
            complex_batches = [
                complex_transactions[i:i + complex_batch_size] 
                for i in range(0, len(complex_transactions), complex_batch_size)
            ] if complex_transactions else []
            
            # Combine all batches, prioritizing simple transactions
            all_batches = simple_batches + complex_batches
            
            logger.info(f"Processing {len(transactions_to_process)} transactions in {len(all_batches)} "
                      f"batches (simple: {len(simple_transactions)}, complex: {len(complex_transactions)})")
            
            with self.executor_class(max_workers=self.max_workers) as executor:
                # Submit batches to executor
                futures = []
                for batch in all_batches:
                    future = executor.submit(self._process_batch, batch, process_func, **kwargs)
                    futures.append(future)
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        
            # Combine with results from cache
            results.extend(cached_results)
        
        # Calculate performance metrics
        end_time = time.time()
        elapsed_time = end_time - start_time
        successful = sum(1 for r in results if r[0])
        success_rate = successful / num_transactions if num_transactions > 0 else 0
        avg_latency = np.mean([r[1] for r in results]) if results else 0
        throughput = num_transactions / elapsed_time if elapsed_time > 0 else 0
        
        # Update processing statistics
        self.processing_stats["total_transactions"] += num_transactions
        self.processing_stats["successful_transactions"] += successful
        self.processing_stats["total_processing_time"] += elapsed_time
        self.processing_stats["cache_hits"] += self.cache_hits
        self.processing_stats["cache_misses"] += self.cache_misses
        
        total_processed = self.processing_stats["total_transactions"]
        total_time = self.processing_stats["total_processing_time"]
        total_cache_requests = self.processing_stats["cache_hits"] + self.processing_stats["cache_misses"]
        
        self.processing_stats["avg_processing_time"] = total_time / total_processed if total_processed > 0 else 0
        self.processing_stats["throughput"] = self.processing_stats["total_transactions"] / total_time if total_time > 0 else 0
        self.processing_stats["cache_hit_ratio"] = self.processing_stats["cache_hits"] / total_cache_requests if total_cache_requests > 0 else 0
        
        # Tính tỷ lệ hit đúng cách dựa trên tổng số lần truy cập cache
        total_cache_accesses = self.cache_hits + self.cache_misses
        cache_hit_ratio = self.cache_hits / total_cache_accesses if total_cache_accesses > 0 else 0
        
        metrics = {
            "num_transactions": num_transactions,
            "successful_transactions": successful,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "elapsed_time": elapsed_time,
            "throughput": throughput,
            "transactions_per_second": throughput,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": cache_hit_ratio
        }
        
        logger.info(f"Processed {num_transactions} transactions in {elapsed_time:.2f}s "
                   f"(throughput: {throughput:.2f} tx/s, success rate: {success_rate:.2%}, "
                   f"cache hits: {cache_hit_ratio:.1%})")
        
        return metrics
    
    def _process_batch(self, 
                     batch: List[Dict[str, Any]], 
                     process_func: Callable[[Dict[str, Any]], Tuple[bool, float]],
                     **kwargs) -> List[Tuple[bool, float]]:
        """
        Process a batch of transactions.
        
        Args:
            batch: Batch of transactions to process
            process_func: Function to process each transaction
            **kwargs: Additional parameters for process_func
            
        Returns:
            List[Tuple[bool, float]]: List of results (success, latency)
        """
        results = []
        for tx in batch:
            try:
                result = process_func(tx, **kwargs)
                results.append(result)
                
                # Update cache with new result
                if self.enable_caching and not tx.get('from_cache', False):
                    self._update_cache(tx, result)
                    
            except Exception as e:
                logger.error(f"Error processing transaction {tx.get('id', 'unknown')}: {e}")
                results.append((False, 0.0))
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "throughput": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_ratio": 0.0
        }
        
    def clear_cache(self) -> None:
        """Clear transaction cache."""
        self.transaction_cache = {}
        logger.info("Transaction cache cleared")


class PerformanceOptimizer:
    """
    System performance optimizer.
    
    This class provides methods to automatically optimize blockchain system performance
    based on system conditions and workload requirements.
    """
    
    def __init__(self):
        """Initialize system performance optimizer."""
        self.transaction_processor = ParallelTransactionProcessor()
        self.performance_metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "throughput": [],
            "latency": []
        }
        self.optimization_params = {
            "thread_count": multiprocessing.cpu_count(),
            "batch_size": 64,
            "cache_size": 1000,
            "prefetch_enabled": True
        }
    
    def analyze_system_performance(self) -> Dict[str, Any]:
        """
        Analyze current system performance.
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Simplified implementation
        avg_throughput = np.mean(self.performance_metrics["throughput"]) if self.performance_metrics["throughput"] else 0
        avg_latency = np.mean(self.performance_metrics["latency"]) if self.performance_metrics["latency"] else 0
        
        analysis = {
            "avg_throughput": avg_throughput,
            "avg_latency": avg_latency,
            "bottlenecks": self._identify_bottlenecks(),
            "optimization_recommendations": self._generate_recommendations()
        }
        
        return analysis
    
    def _identify_bottlenecks(self) -> List[str]:
        """
        Identify system bottlenecks.
        
        Returns:
            List[str]: List of potential bottlenecks
        """
        bottlenecks = []
        
        # Check CPU usage
        if len(self.performance_metrics["cpu_usage"]) > 0:
            avg_cpu = np.mean(self.performance_metrics["cpu_usage"])
            if avg_cpu > 80:
                bottlenecks.append("CPU usage is high (> 80%)")
        
        # Check memory usage
        if len(self.performance_metrics["memory_usage"]) > 0:
            avg_mem = np.mean(self.performance_metrics["memory_usage"])
            if avg_mem > 80:
                bottlenecks.append("Memory usage is high (> 80%)")
        
        # Check throughput
        if len(self.performance_metrics["throughput"]) > 1:
            throughput_trend = self.performance_metrics["throughput"][-1] - self.performance_metrics["throughput"][0]
            if throughput_trend < 0:
                bottlenecks.append("Throughput is decreasing")
        
        # Check latency
        if len(self.performance_metrics["latency"]) > 1:
            latency_trend = self.performance_metrics["latency"][-1] - self.performance_metrics["latency"][0]
            if latency_trend > 0:
                bottlenecks.append("Latency is increasing")
        
        return bottlenecks
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate system optimization recommendations.
        
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        bottlenecks = self._identify_bottlenecks()
        
        if "CPU usage is high (> 80%)" in bottlenecks:
            recommendations.append("Consider reducing batch size")
            recommendations.append("Consider enabling caching mechanisms")
        
        if "Memory usage is high (> 80%)" in bottlenecks:
            recommendations.append("Optimize memory utilization with streaming techniques")
            recommendations.append("Consider reducing cache size")
        
        if "Throughput is decreasing" in bottlenecks or "Latency is increasing" in bottlenecks:
            recommendations.append("Consider increasing thread count")
            recommendations.append("Optimize database queries")
            recommendations.append("Implement data preprocessing techniques")
        
        # Add general recommendations
        recommendations.append("Regularly monitor system performance")
        recommendations.append("Implement adaptive scaling based on workload")
        
        return recommendations
    
    def optimize_throughput(self, transactions: List[Dict[str, Any]], 
                          process_func: Callable[[Dict[str, Any]], Tuple[bool, float]],
                          **kwargs) -> Dict[str, Any]:
        """
        Optimize throughput for a list of transactions.
        
        Args:
            transactions: List of transactions to process
            process_func: Function to process each transaction
            **kwargs: Additional parameters for process_func
            
        Returns:
            Dict[str, Any]: Processing results
        """
        # Test different configurations
        configurations = [
            {"max_workers": 4, "use_processes": False},
            {"max_workers": 8, "use_processes": False},
            {"max_workers": 16, "use_processes": False},
            {"max_workers": min(32, multiprocessing.cpu_count() * 4), "use_processes": False},
            {"max_workers": 4, "use_processes": True},
            {"max_workers": 8, "use_processes": True}
        ]
        
        # Use a small sample for testing configurations
        sample_size = min(100, len(transactions))
        sample_transactions = transactions[:sample_size]
        
        best_throughput = 0
        best_config = None
        
        logger.info(f"Testing {len(configurations)} configurations on {sample_size} transactions")
        
        # Test each configuration
        for config in configurations:
            processor = ParallelTransactionProcessor(**config)
            result = processor.process_transactions(sample_transactions, process_func, **kwargs)
            
            if result["throughput"] > best_throughput:
                best_throughput = result["throughput"]
                best_config = config
        
        logger.info(f"Selected optimal configuration: {best_config}")
        
        # Reset to best configuration
        self.transaction_processor = ParallelTransactionProcessor(
            max_workers=best_config["max_workers"],
            use_processes=best_config["use_processes"]
        )
        
        # Process all transactions with optimal configuration
        result = self.transaction_processor.process_transactions(
            transactions, process_func, **kwargs
        )
        
        # Update performance metrics
        self.performance_metrics["throughput"].append(result["throughput"])
        self.performance_metrics["latency"].append(result["avg_latency"])
        
        return result 