import pytest
import time
import random
from typing import Dict, Any, Tuple
import numpy as np

from qtrust.utils.performance_optimizer import ParallelTransactionProcessor, PerformanceOptimizer


def dummy_process_func(tx: Dict[str, Any], **kwargs) -> Tuple[bool, float]:
    """Hàm mô phỏng xử lý giao dịch để kiểm tra."""
    # Mô phỏng thời gian xử lý giao dịch
    processing_time = random.uniform(0.001, 0.05)
    time.sleep(processing_time)
    
    # Mô phỏng tỷ lệ thành công
    success_rate = kwargs.get('success_rate', 0.95)
    success = random.random() < success_rate
    
    # Mô phỏng độ trễ
    latency = processing_time + random.uniform(0.001, 0.01)
    
    return success, latency


def generate_test_transactions(num_txs: int) -> list:
    """Tạo các giao dịch mẫu để kiểm tra."""
    return [
        {
            'id': f'tx_{i}',
            'sender': f'sender_{random.randint(1, 100)}',
            'receiver': f'receiver_{random.randint(1, 100)}',
            'value': random.uniform(1, 1000),
            'data': f'data_{i}',
            'timestamp': time.time()
        }
        for i in range(num_txs)
    ]


class TestParallelTransactionProcessor:
    
    def test_process_transactions_throughput(self):
        """Kiểm tra throughput của trình xử lý giao dịch song song."""
        processor = ParallelTransactionProcessor(max_workers=4)
        
        # Tạo 1000 giao dịch mẫu
        transactions = generate_test_transactions(1000)
        
        # Xử lý giao dịch và đo throughput
        result = processor.process_transactions(
            transactions, dummy_process_func, success_rate=0.95
        )
        
        # Kiểm tra kết quả
        assert result['num_transactions'] == 1000
        assert result['throughput'] > 0
        assert 0.85 <= result['success_rate'] <= 1.0
        
        # Kiểm tra thống kê
        stats = processor.get_processing_stats()
        assert stats['total_transactions'] == 1000
        assert stats['throughput'] > 0
        
        print(f"Throughput: {result['throughput']:.2f} tx/s")
    
    def test_process_transactions_reliability(self):
        """Kiểm tra độ tin cậy của trình xử lý giao dịch song song."""
        processor = ParallelTransactionProcessor(max_workers=4)
        
        # Tạo 500 giao dịch mẫu
        transactions = generate_test_transactions(500)
        
        # Xử lý lô đầu tiên với tỷ lệ thành công cao
        result1 = processor.process_transactions(
            transactions, dummy_process_func, success_rate=0.95
        )
        
        # Xử lý lô thứ hai với tỷ lệ thành công thấp
        result2 = processor.process_transactions(
            transactions, dummy_process_func, success_rate=0.6
        )
        
        # Kiểm tra thống kê
        stats = processor.get_processing_stats()
        assert stats['total_transactions'] == 1000
        
        # Kiểm tra tỷ lệ thành công
        assert result1['success_rate'] > result2['success_rate']
        assert 0.5 <= result2['success_rate'] <= 0.7  # Cho phép một ít chênh lệch
    
    def test_thread_vs_process_performance(self):
        """So sánh hiệu suất giữa đa luồng và đa xử lý."""
        # Tạo 200 giao dịch mẫu
        transactions = generate_test_transactions(200)
        
        # Sử dụng đa luồng
        thread_processor = ParallelTransactionProcessor(max_workers=4, use_processes=False)
        thread_result = thread_processor.process_transactions(
            transactions, dummy_process_func
        )
        
        # Sử dụng đa xử lý
        process_processor = ParallelTransactionProcessor(max_workers=4, use_processes=True)
        process_result = process_processor.process_transactions(
            transactions, dummy_process_func
        )
        
        # Ghi lại kết quả (không so sánh trực tiếp vì kết quả có thể khác nhau giữa các hệ thống)
        print(f"Thread throughput: {thread_result['throughput']:.2f} tx/s")
        print(f"Process throughput: {process_result['throughput']:.2f} tx/s")
        
        # Kiểm tra cả hai phương pháp đều thành công
        assert thread_result['num_transactions'] == 200
        assert process_result['num_transactions'] == 200
    
    def test_batch_size_optimization(self):
        """Kiểm tra tối ưu hóa kích thước lô."""
        optimizer = PerformanceOptimizer()
        
        # Tạo 300 giao dịch mẫu
        transactions = generate_test_transactions(300)
        
        # Tối ưu hóa throughput
        result = optimizer.optimize_throughput(
            transactions, dummy_process_func
        )
        
        # Kiểm tra kết quả
        assert result['num_transactions'] == 300
        assert result['throughput'] > 0
        
        # Kiểm tra số liệu hiệu suất được lưu lại
        assert len(optimizer.performance_metrics['throughput']) == 1
        assert len(optimizer.performance_metrics['latency']) == 1 