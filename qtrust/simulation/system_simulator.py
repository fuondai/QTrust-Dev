"""
System simulation for blockchain network testing.
"""
import time
import random
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from qtrust.consensus.adaptive_consensus import AdaptiveConsensus
from qtrust.utils.performance_optimizer import ParallelTransactionProcessor

logger = logging.getLogger(__name__)

class SystemSimulator:
    """
    Simulates the complete blockchain system to test performance and parameters.
    """
    
    def __init__(self, 
                num_shards: int = 4, 
                num_validators_per_shard: int = 4,
                enable_parallel_processing: bool = True,
                max_workers: Optional[int] = None):
        """
        Initialize system simulator.
        
        Args:
            num_shards: Number of shards in the system
            num_validators_per_shard: Number of validators per shard
            enable_parallel_processing: Use parallel processing to increase throughput
            max_workers: Number of parallel workers (None = automatic)
        """
        self.num_shards = num_shards
        self.num_validators_per_shard = num_validators_per_shard
        self.enable_parallel_processing = enable_parallel_processing
        
        # Initialize network (simplified)
        # self.network = NetworkSimulator(num_shards=num_shards)
        
        # Initialize shards with separate consensus (adjust parameters)
        self.shards = {}
        for i in range(num_shards):
            # Initialize AdaptiveConsensus with correct parameters
            self.shards[i] = AdaptiveConsensus(
                num_validators_per_shard=num_validators_per_shard,
                enable_adaptive_pos=True,
                enable_lightweight_crypto=True
            )
        
        # Initialize parallel processor
        if enable_parallel_processing:
            # Tạo processor một lần và giữ nguyên cache
            self.transaction_processor = ParallelTransactionProcessor(
                max_workers=max_workers,
                enable_caching=True
            )
            logger.info(f"Parallel transaction processing enabled with {self.transaction_processor.max_workers} workers")
        else:
            self.transaction_processor = None
            logger.info("Parallel transaction processing disabled")
        
        # Metrics
        self.total_transactions = 0
        self.successful_transactions = 0
        self.total_throughput = 0
        self.total_latency = 0
        self.total_energy = 0
        self.metrics_history = {
            'throughput': [],
            'success_rate': [],
            'latency': [],
            'energy': []
        }
    
    def generate_random_transactions(self, 
                                   num_transactions: int, 
                                   min_value: float = 1.0, 
                                   max_value: float = 1000.0) -> List[Dict[str, Any]]:
        """
        Generate random transactions for testing.
        
        Args:
            num_transactions: Number of transactions to create
            min_value: Minimum transaction value
            max_value: Maximum transaction value
            
        Returns:
            List[Dict[str, Any]]: List of transactions
        """
        transactions = []
        for i in range(num_transactions):
            # Select random source and destination shards
            sender_shard = random.randint(0, self.num_shards - 1)
            # 30% chance of intra-shard transaction
            same_shard = random.random() < 0.3
            receiver_shard = sender_shard if same_shard else random.randint(0, self.num_shards - 1)
            
            # Create random IDs for sender and receiver
            sender_id = f"user_{sender_shard}_{random.randint(1, 100)}"
            receiver_id = f"user_{receiver_shard}_{random.randint(1, 100)}"
            
            # Create random transaction value within allowed range
            value = random.uniform(min_value, max_value)
            
            # Create transaction
            transaction = {
                'id': f"tx_{int(time.time())}_{i}",
                'sender': sender_id,
                'sender_shard': sender_shard,
                'receiver': receiver_id, 
                'receiver_shard': receiver_shard,
                'value': value,
                'timestamp': time.time(),
                'data': f"Transaction data {i}"
            }
            
            transactions.append(transaction)
        
        return transactions
    
    def process_transaction(self, tx: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Process a single transaction through the system.
        
        Args:
            tx: Transaction to process
            
        Returns:
            Tuple[bool, float]: (success, latency)
        """
        start_time = time.time()
        sender_shard = tx['sender_shard']
        receiver_shard = tx['receiver_shard']
        transaction_value = tx['value']
        
        # Get trust scores from the system
        trust_scores = {}
        for shard_id, shard in self.shards.items():
            trust_scores[shard_id] = shard.get_trust_scores()
        
        try:
            # Check if it's an intra-shard transaction
            if sender_shard == receiver_shard:
                # Intra-shard transaction
                success, latency, energy, protocol = self.shards[sender_shard].execute_consensus(
                    transaction_value=transaction_value,
                    shard_id=sender_shard,
                    trust_scores=trust_scores[sender_shard]
                )
            else:
                # Cross-shard transaction - needs processing through both shards
                # Source shard
                success1, latency1, energy1, protocol1 = self.shards[sender_shard].execute_consensus(
                    transaction_value=transaction_value,
                    shard_id=sender_shard,
                    trust_scores=trust_scores[sender_shard]
                )
                
                # If first phase succeeds, continue with destination shard
                if success1:
                    success2, latency2, energy2, protocol2 = self.shards[receiver_shard].execute_consensus(
                        transaction_value=transaction_value,
                        shard_id=receiver_shard,
                        trust_scores=trust_scores[receiver_shard]
                    )
                    success = success2
                    latency = latency1 + latency2
                    energy = energy1 + energy2
                    protocol = f"{protocol1}->{protocol2}"
                else:
                    success = False
                    latency = latency1
                    energy = energy1
                    protocol = protocol1
            
            end_time = time.time()
            total_latency = end_time - start_time
            
            # Ensure we have numerical values
            latency = float(latency)
            energy = float(energy)
            
            # For tracking, we store the results in the transaction
            tx['success'] = success
            tx['latency'] = latency
            tx['energy'] = energy
            tx['protocol'] = protocol
            tx['total_latency'] = total_latency
            
            return success, total_latency
            
        except Exception as e:
            logger.error(f"Error processing transaction {tx['id']}: {str(e)}")
            return False, 0.0
    
    def run_simulation(self, 
                     num_transactions: int, 
                     min_value: float = 1.0, 
                     max_value: float = 1000.0,
                     existing_transactions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run simulation with given number of transactions.
        
        Args:
            num_transactions: Number of transactions to process
            min_value: Minimum transaction value
            max_value: Maximum transaction value
            existing_transactions: List of existing transactions (if None, num_transactions new transactions will be created)
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        # If there are existing transactions, use them instead of creating new ones
        if existing_transactions is not None:
            transactions = existing_transactions
            num_transactions = len(transactions)
            logger.info(f"Using {num_transactions} existing transactions for simulation")
        else:
            # Create random transactions
            logger.info(f"Starting simulation with {num_transactions} transactions")
            transactions = self.generate_random_transactions(
                num_transactions, min_value, max_value
            )
        
        start_time = time.time()
        
        # Process transactions sequentially or in parallel
        if self.enable_parallel_processing and self.transaction_processor:
            # Parallel processing
            logger.info("Using parallel transaction processing")
            
            # Lưu các giao dịch thành batch nếu có giao dịch lặp lại để tối ưu cache hit
            # Gom các giao dịch giống nhau vào cùng một batch
            repeating_txs = [tx for tx in transactions if tx.get('debug_marker') == 'repeating_tx']
            unique_txs = [tx for tx in transactions if tx.get('debug_marker') != 'repeating_tx']
            
            if repeating_txs:
                # Xử lý từng loại giao dịch riêng biệt để cải thiện cache hit
                logger.info(f"Processing {len(repeating_txs)} repeating transactions first for better cache pattern")
                
                # Xử lý giao dịch lặp lại trước để build cache
                result_repeating = self.transaction_processor.process_transactions(
                    repeating_txs, self.process_transaction
                )
                
                # Xử lý giao dịch không lặp lại
                if unique_txs:
                    result_unique = self.transaction_processor.process_transactions(
                        unique_txs, self.process_transaction
                    )
                    
                    # Kết hợp kết quả
                    result = {
                        'num_transactions': result_repeating['num_transactions'] + result_unique['num_transactions'],
                        'successful_transactions': result_repeating['successful_transactions'] + result_unique['successful_transactions'],
                        'avg_latency': (result_repeating['avg_latency'] * result_repeating['successful_transactions'] + 
                                      result_unique['avg_latency'] * result_unique['successful_transactions']) / 
                                      (result_repeating['successful_transactions'] + result_unique['successful_transactions'] 
                                       if (result_repeating['successful_transactions'] + result_unique['successful_transactions']) > 0 else 1),
                        'throughput': (result_repeating['num_transactions'] + result_unique['num_transactions']) / 
                                     (result_repeating['elapsed_time'] + result_unique['elapsed_time'] 
                                      if (result_repeating['elapsed_time'] + result_unique['elapsed_time']) > 0 else 1),
                        'elapsed_time': result_repeating['elapsed_time'] + result_unique['elapsed_time'],
                        'cache_hits': result_repeating.get('cache_hits', 0) + result_unique.get('cache_hits', 0),
                        'cache_misses': result_repeating.get('cache_misses', 0) + result_unique.get('cache_misses', 0),
                        'cache_hit_ratio': (result_repeating.get('cache_hits', 0) + result_unique.get('cache_hits', 0)) / 
                                          (result_repeating['num_transactions'] + result_unique['num_transactions'])
                                          if (result_repeating['num_transactions'] + result_unique['num_transactions']) > 0 else 0
                    }
                else:
                    result = result_repeating
            else:
                # Không có giao dịch lặp lại, xử lý bình thường
                result = self.transaction_processor.process_transactions(
                    transactions, self.process_transaction
                )
            
            # Update metrics from parallel results
            num_transactions = result['num_transactions']
            successful = result['successful_transactions']
            avg_latency = result['avg_latency']
            throughput = result['throughput']
            
            # Calculate energy metrics from processed transactions
            total_energy = sum(float(tx.get('energy', 0)) for tx in transactions if 'energy' in tx)
            avg_energy = total_energy / num_transactions if num_transactions > 0 else 0
            
            # Initialize total_latency to avoid reference error before definition
            total_latency = avg_latency * successful if successful > 0 else 0
            
        else:
            # Sequential processing
            logger.info("Using sequential transaction processing")
            successful = 0
            total_latency = 0
            total_energy = 0
            
            for tx in transactions:
                success, latency = self.process_transaction(tx)
                if success:
                    successful += 1
                    total_latency += latency
                    total_energy += float(tx.get('energy', 0))
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            throughput = num_transactions / elapsed_time if elapsed_time > 0 else 0
            avg_latency = total_latency / successful if successful > 0 else 0
            avg_energy = total_energy / successful if successful > 0 else 0
        
        # Update total metrics
        self.total_transactions += num_transactions
        self.successful_transactions += successful
        success_rate = successful / num_transactions if num_transactions > 0 else 0
        
        # Calculate metrics
        self.total_throughput = self.total_transactions / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        self.total_latency += total_latency
        self.total_energy += total_energy
        
        # Save metrics to history
        self.metrics_history['throughput'].append(throughput)
        self.metrics_history['success_rate'].append(success_rate)
        self.metrics_history['latency'].append(avg_latency)
        self.metrics_history['energy'].append(avg_energy)
        
        # Create result
        results = {
            'num_transactions': num_transactions,
            'successful_transactions': successful,
            'success_rate': success_rate,
            'throughput': throughput,
            'avg_latency': avg_latency,
            'avg_energy': avg_energy,
            'total_energy': total_energy,
            'transactions': transactions
        }
        
        logger.info(f"Simulation completed: {successful}/{num_transactions} transactions successful")
        logger.info(f"Throughput: {throughput:.2f} tx/s, Success rate: {success_rate:.2%}")
        logger.info(f"Avg latency: {avg_latency:.4f}s, Avg energy: {avg_energy:.4f}")
        
        return results
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """
        Get metrics history from simulations.
        
        Returns:
            Dict[str, List[float]]: Metrics history
        """
        return self.metrics_history
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """
        Get overall metrics summary.
        
        Returns:
            Dict[str, float]: Metrics summary
        """
        avg_throughput = np.mean(self.metrics_history['throughput']) if self.metrics_history['throughput'] else 0
        avg_success_rate = np.mean(self.metrics_history['success_rate']) if self.metrics_history['success_rate'] else 0
        avg_latency = np.mean(self.metrics_history['latency']) if self.metrics_history['latency'] else 0
        avg_energy = np.mean(self.metrics_history['energy']) if self.metrics_history['energy'] else 0
        
        return {
            'total_transactions': self.total_transactions,
            'successful_transactions': self.successful_transactions,
            'success_rate': self.successful_transactions / self.total_transactions if self.total_transactions > 0 else 0,
            'avg_throughput': avg_throughput,
            'avg_latency': avg_latency,
            'avg_energy': avg_energy,
            'total_energy': self.total_energy
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.total_transactions = 0
        self.successful_transactions = 0
        self.total_throughput = 0
        self.total_latency = 0
        self.total_energy = 0
        self.metrics_history = {
            'throughput': [],
            'success_rate': [],
            'latency': [],
            'energy': []
        }
        
        # Reset metrics in shards
        for shard in self.shards.values():
            shard.reset_statistics() 