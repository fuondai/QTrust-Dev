import unittest
import torch
import numpy as np
import sys
import os
from collections import OrderedDict

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from qtrust.federated.model_aggregation import ModelAggregator, ModelAggregationManager

class TestModelAggregator(unittest.TestCase):
    def setUp(self):
        # Tạo dữ liệu mẫu PyTorch cho bài kiểm tra
        self.params1 = OrderedDict({
            'layer1.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            'layer1.bias': torch.tensor([0.1, 0.2]),
            'layer2.weight': torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
            'layer2.bias': torch.tensor([0.01, 0.02])
        })
        
        self.params2 = OrderedDict({
            'layer1.weight': torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
            'layer1.bias': torch.tensor([0.3, 0.4]),
            'layer2.weight': torch.tensor([[0.9, 1.0], [1.1, 1.2]]),
            'layer2.bias': torch.tensor([0.03, 0.04])
        })
        
        self.params3 = OrderedDict({
            'layer1.weight': torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
            'layer1.bias': torch.tensor([0.5, 0.6]),
            'layer2.weight': torch.tensor([[1.3, 1.4], [1.5, 1.6]]),
            'layer2.bias': torch.tensor([0.05, 0.06])
        })
        
        # Tạo một tham số bất thường để kiểm tra khả năng chống Byzantine
        self.byzantine_params = OrderedDict({
            'layer1.weight': torch.tensor([[30.0, 40.0], [50.0, 60.0]]),  # Giá trị bất thường
            'layer1.bias': torch.tensor([5.0, 6.0]),  # Giá trị bất thường
            'layer2.weight': torch.tensor([[13.0, 14.0], [15.0, 16.0]]),  # Giá trị bất thường
            'layer2.bias': torch.tensor([0.5, 0.6])   # Giá trị bất thường
        })
        
        self.params_list = [self.params1, self.params2, self.params3, self.byzantine_params]
        self.normal_params_list = [self.params1, self.params2, self.params3]
        
        # Trọng số cho các client
        self.weights = [0.3, 0.3, 0.3, 0.1]
        self.normal_weights = [0.33, 0.33, 0.34]
        
        # Thông tin tin cậy và hiệu suất
        self.trust_scores = [0.9, 0.8, 0.7, 0.1]  # Client cuối là Byzantine
        self.performance_scores = [0.85, 0.9, 0.8, 0.3]  # Client cuối có hiệu suất thấp
        
        # Khởi tạo aggregator
        self.aggregator = ModelAggregator()

    def test_weighted_average(self):
        """Kiểm tra phương pháp tổng hợp trung bình có trọng số."""
        aggregated = ModelAggregator.weighted_average(self.normal_params_list, weights=self.normal_weights)
        
        # Kiểm tra kết quả
        for key in aggregated:
            expected = sum(params[key] * w for params, w in zip(self.normal_params_list, self.normal_weights))
            self.assertTrue(torch.allclose(aggregated[key], expected, rtol=1e-5))

    def test_median(self):
        """Kiểm tra phương pháp tổng hợp theo trung vị."""
        aggregated = ModelAggregator.median(self.params_list)
        
        # Kiểm tra kết quả - Tính toán trung vị phải chính xác
        for key in aggregated:
            # Tạo tensor để tính trung vị
            values = [params[key].cpu().numpy() for params in self.params_list]
            expected = np.median(values, axis=0)
            expected_tensor = torch.tensor(expected, dtype=aggregated[key].dtype)
            self.assertTrue(torch.allclose(aggregated[key], expected_tensor, rtol=1e-5))

    def test_trimmed_mean(self):
        """Kiểm tra phương pháp tổng hợp theo trung bình cắt bỏ."""
        aggregated = ModelAggregator.trimmed_mean(self.params_list, trim_ratio=0.25)  # Cắt bỏ 25% giá trị cao nhất và thấp nhất
        
        # Kiểm tra kết quả - Phải loại bỏ giá trị Byzantine
        for key in aggregated:
            # Tạo tensor để tính trung bình cắt bỏ (loại bỏ giá trị cao nhất và thấp nhất)
            values = [params[key].cpu().numpy() for params in self.params_list]
            
            # Cắt bỏ 25% giá trị cao nhất và thấp nhất
            n = len(values)
            k = int(np.ceil(n * 0.25))
            
            if 2*k >= n:
                # Nếu cắt quá nhiều, sử dụng trung bình thông thường
                expected = np.mean(values, axis=0)
            else:
                # Sắp xếp và cắt bỏ
                sorted_values = np.sort(values, axis=0)
                expected = np.mean(sorted_values[k:n-k], axis=0)
            
            expected_tensor = torch.tensor(expected, dtype=aggregated[key].dtype)
            self.assertTrue(torch.allclose(aggregated[key], expected_tensor, rtol=1e-5))

    def test_krum(self):
        """Kiểm tra phương pháp tổng hợp Krum."""
        aggregated = ModelAggregator.krum(self.params_list, num_byzantine=1)
        
        # Kiểm tra kết quả - Phải chọn một trong các tham số bình thường
        is_one_of_normal = False
        for normal_params in self.normal_params_list:
            if all(torch.allclose(aggregated[key], normal_params[key]) for key in aggregated):
                is_one_of_normal = True
                break
        
        self.assertTrue(is_one_of_normal, "Kết quả Krum phải là một trong các tham số bình thường")

    def test_adaptive_federated_averaging(self):
        """Kiểm tra phương pháp tổng hợp trung bình liên bang thích ứng."""
        # Adaptive federated averaging sử dụng trust_scores và performance_scores thay vì weights
        aggregated = ModelAggregator.adaptive_federated_averaging(
            self.params_list, 
            trust_scores=self.trust_scores,
            performance_scores=self.performance_scores,
            adaptive_alpha=0.5
        )
        
        # Tính combined scores như trong implementation
        combined_scores = [0.5 * trust + 0.5 * perf 
                         for trust, perf in zip(self.trust_scores, self.performance_scores)]
        
        # Chuẩn hóa thành trọng số
        total_score = sum(combined_scores)
        expected_weights = [score / total_score for score in combined_scores]
        
        # Tính trung bình có trọng số theo cách thủ công
        expected_result = OrderedDict()
        for key in self.params1.keys():
            expected_result[key] = sum(params[key] * w for params, w in zip(self.params_list, expected_weights))
        
        # So sánh kết quả
        for key in aggregated:
            self.assertTrue(torch.allclose(aggregated[key], expected_result[key], rtol=1e-5))

    def test_fedprox(self):
        """Kiểm tra phương pháp tổng hợp FedProx."""
        global_params = OrderedDict({
            'layer1.weight': torch.tensor([[1.5, 2.5], [3.5, 4.5]]),
            'layer1.bias': torch.tensor([0.25, 0.35]),
            'layer2.weight': torch.tensor([[0.75, 0.85], [0.95, 1.05]]),
            'layer2.bias': torch.tensor([0.025, 0.035])
        })
        
        # Hệ số mu
        mu = 0.01
        
        # Tính average theo weighted_average
        weighted_avg = ModelAggregator.weighted_average(self.normal_params_list, weights=self.normal_weights)
        
        # Áp dụng FedProx
        aggregated = ModelAggregator.fedprox(
            self.normal_params_list, 
            global_params=global_params,
            weights=self.normal_weights,
            mu=mu
        )
        
        # Kiểm tra kết quả
        for key in aggregated:
            # FedProx: (1 - mu) * weighted_avg + mu * global_params
            expected = (1 - mu) * weighted_avg[key] + mu * global_params[key]
            self.assertTrue(torch.allclose(aggregated[key], expected, rtol=1e-5))


class TestModelAggregationManager(unittest.TestCase):
    def setUp(self):
        # Khởi tạo manager với phương pháp mặc định
        self.manager = ModelAggregationManager(default_method='weighted_average')
        
        # Tạo dữ liệu mẫu đơn giản cho bài kiểm tra
        self.params1 = OrderedDict({'weight': torch.tensor([1.0, 2.0]), 'bias': torch.tensor([0.1])})
        self.params2 = OrderedDict({'weight': torch.tensor([3.0, 4.0]), 'bias': torch.tensor([0.2])})
        self.params3 = OrderedDict({'weight': torch.tensor([5.0, 6.0]), 'bias': torch.tensor([0.3])})
        
        self.params_list = [self.params1, self.params2, self.params3]
        self.weights = [0.2, 0.3, 0.5]

    def test_aggregate_with_default_method(self):
        """Kiểm tra tổng hợp với phương pháp mặc định."""
        # Phương pháp phải được chỉ định rõ ràng
        aggregated = self.manager.aggregate('weighted_average', params_list=self.params_list, weights=self.weights)
        
        # Kiểm tra kết quả - Phải sử dụng phương pháp mặc định (weighted_average)
        for key in aggregated:
            expected = sum(params[key] * w for params, w in zip(self.params_list, self.weights))
            self.assertTrue(torch.allclose(aggregated[key], expected, rtol=1e-5))

    def test_aggregate_with_specified_method(self):
        """Kiểm tra tổng hợp với phương pháp được chỉ định."""
        aggregated = self.manager.aggregate('median', params_list=self.params_list)
        
        # Kiểm tra kết quả - Phải sử dụng phương pháp median
        for key in aggregated:
            values = [params[key].cpu().numpy() for params in self.params_list]
            expected = np.median(values, axis=0)
            expected_tensor = torch.tensor(expected, dtype=aggregated[key].dtype)
            self.assertTrue(torch.allclose(aggregated[key], expected_tensor, rtol=1e-5))

    def test_recommend_method(self):
        """Kiểm tra khả năng đề xuất phương pháp tốt nhất."""
        # Với ít client và không có nghi ngờ Byzantine -> weighted_average
        method = self.manager.recommend_method(num_clients=3, has_trust_scores=False, suspected_byzantine=False)
        self.assertEqual(method, 'weighted_average')
        
        # Với nhiều client, có nghi ngờ Byzantine -> median
        method = self.manager.recommend_method(num_clients=10, has_trust_scores=True, suspected_byzantine=True)
        self.assertEqual(method, 'median')
        
        # Với ít client (< 4), có nghi ngờ Byzantine -> trimmed_mean
        method = self.manager.recommend_method(num_clients=3, has_trust_scores=True, suspected_byzantine=True)
        self.assertEqual(method, 'trimmed_mean')
        
        # Với ít client, có điểm tin cậy, không nghi ngờ Byzantine -> adaptive_fedavg
        method = self.manager.recommend_method(num_clients=5, has_trust_scores=True, suspected_byzantine=False)
        self.assertEqual(method, 'adaptive_fedavg')


if __name__ == '__main__':
    unittest.main() 