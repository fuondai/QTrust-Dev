"""
Integration tests for parallel transaction processing.
"""
import pytest
import time
import random
from typing import Dict, Any, List

from qtrust.utils.performance_optimizer import ParallelTransactionProcessor

# Tạm thời bỏ bài kiểm tra liên quan đến SystemSimulator
# Chỉ giữ bài kiểm tra cho ParallelTransactionProcessor


def test_parallel_transaction_processor_integration():
    """
    Kiểm tra tích hợp của ParallelTransactionProcessor với dữ liệu mẫu.
    """
    # Tạo bộ xử lý giao dịch song song
    processor = ParallelTransactionProcessor(max_workers=4)
    
    # Tạo dữ liệu giao dịch mẫu
    transactions = [
        {
            'id': f'tx_{i}',
            'value': random.uniform(1, 100),
            'timestamp': time.time()
        }
        for i in range(100)
    ]
    
    # Hàm xử lý đơn giản để giả lập xử lý giao dịch
    def process_tx(tx: Dict[str, Any]) -> tuple:
        time.sleep(0.01)  # Giả lập thời gian xử lý
        success = random.random() < 0.95  # 95% tỷ lệ thành công
        latency = random.uniform(0.01, 0.05)
        return success, latency
    
    # Xử lý giao dịch bằng bộ xử lý song song
    results = processor.process_transactions(transactions, process_tx)
    
    # Kiểm tra kết quả
    assert 'num_transactions' in results
    assert results['num_transactions'] == 100
    assert 'throughput' in results
    assert results['throughput'] > 0
    assert 'success_rate' in results
    assert 0.8 <= results['success_rate'] <= 1.0
    assert 'avg_latency' in results
    
    # Kiểm tra thống kê
    stats = processor.get_processing_stats()
    assert stats['total_transactions'] == 100
    assert stats['total_processing_time'] > 0 