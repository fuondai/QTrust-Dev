"""
Bài kiểm thử cho HTDCM (Hierarchical Trust-based Data Center Mechanism).
"""

import unittest
import numpy as np
from qtrust.trust.htdcm import HTDCM, HTDCMNode

class TestHTDCMNode(unittest.TestCase):
    """
    Kiểm thử cho lớp HTDCMNode.
    """
    
    def setUp(self):
        """Khởi tạo node tin cậy cho kiểm thử."""
        self.node = HTDCMNode(node_id=1, shard_id=0, initial_trust=0.7)
    
    def test_initialization(self):
        """Kiểm thử khởi tạo node."""
        self.assertEqual(self.node.node_id, 1)
        self.assertEqual(self.node.shard_id, 0)
        self.assertEqual(self.node.trust_score, 0.7)
        self.assertEqual(self.node.successful_txs, 0)
        self.assertEqual(self.node.failed_txs, 0)
        self.assertEqual(self.node.malicious_activities, 0)
    
    def test_update_trust_score(self):
        """Kiểm thử cập nhật điểm tin cậy."""
        # Ghi lại điểm tin cậy ban đầu
        initial_score = self.node.trust_score
        
        # Kiểm tra cập nhật bình thường
        self.node.update_trust_score(0.9)
        expected_score = self.node.alpha * initial_score + self.node.beta * 0.9
        self.assertAlmostEqual(self.node.trust_score, expected_score)
        
        # Đặt lại node để kiểm tra giá trị cực đoan
        self.node = HTDCMNode(node_id=1, shard_id=0, initial_trust=0.7)
        
        # Kiểm tra giá trị cực đoan
        self.node.update_trust_score(2.0)  # Vượt quá 1.0
        self.assertEqual(self.node.trust_score, 1.0)  # Nên được giới hạn ở 1.0
        
        # Đặt lại node một lần nữa
        self.node = HTDCMNode(node_id=1, shard_id=0, initial_trust=0.7)
        
        self.node.update_trust_score(-1.0)  # Dưới 0.0
        self.assertEqual(self.node.trust_score, 0.0)  # Nên được giới hạn ở 0.0
    
    def test_record_transaction_result(self):
        """Kiểm thử ghi lại kết quả giao dịch."""
        # Ghi lại giao dịch thành công
        self.node.record_transaction_result(success=True, response_time=10.0, is_validator=True)
        self.assertEqual(self.node.successful_txs, 1)
        self.assertEqual(self.node.failed_txs, 0)
        self.assertEqual(len(self.node.response_times), 1)
        self.assertEqual(self.node.response_times[0], 10.0)
        
        # Ghi lại giao dịch thất bại
        self.node.record_transaction_result(success=False, response_time=20.0, is_validator=False)
        self.assertEqual(self.node.successful_txs, 1)
        self.assertEqual(self.node.failed_txs, 1)
        self.assertEqual(len(self.node.response_times), 2)
        self.assertEqual(self.node.response_times[1], 20.0)
    
    def test_record_malicious_activity(self):
        """Kiểm thử ghi lại hoạt động độc hại."""
        initial_score = self.node.trust_score
        self.node.record_malicious_activity("double_spending")
        
        self.assertEqual(self.node.malicious_activities, 1)
        self.assertLess(self.node.trust_score, initial_score)  # Điểm tin cậy nên giảm
        
        # Kiểm tra điểm tin cậy bị giảm mạnh khi phát hiện hành vi độc hại
        self.assertEqual(self.node.trust_score, 0.0)
    
    def test_get_success_rate(self):
        """Kiểm thử lấy tỷ lệ thành công."""
        # Ban đầu, chưa có giao dịch nào
        self.assertEqual(self.node.get_success_rate(), 0.0)
        
        # Thêm giao dịch thành công và thất bại
        self.node.record_transaction_result(success=True, response_time=10.0, is_validator=True)
        self.node.record_transaction_result(success=True, response_time=15.0, is_validator=True)
        self.node.record_transaction_result(success=False, response_time=20.0, is_validator=True)
        
        # Tỷ lệ thành công nên là 2/3
        self.assertAlmostEqual(self.node.get_success_rate(), 2/3)
    
    def test_get_average_response_time(self):
        """Kiểm thử lấy thời gian phản hồi trung bình."""
        # Ban đầu, chưa có phản hồi nào
        self.assertEqual(self.node.get_average_response_time(), 0.0)
        
        # Thêm các thời gian phản hồi
        self.node.record_transaction_result(success=True, response_time=10.0, is_validator=True)
        self.node.record_transaction_result(success=True, response_time=20.0, is_validator=True)
        
        # Thời gian trung bình nên là (10+20)/2 = 15
        self.assertEqual(self.node.get_average_response_time(), 15.0)

class TestHTDCM(unittest.TestCase):
    """
    Kiểm thử cho lớp HTDCM.
    """
    
    def setUp(self):
        """Khởi tạo HTDCM cho kiểm thử."""
        self.num_nodes = 10
        self.htdcm = HTDCM(num_nodes=self.num_nodes)
    
    def test_initialization(self):
        """Kiểm thử khởi tạo HTDCM."""
        self.assertEqual(len(self.htdcm.nodes), self.num_nodes)
        self.assertEqual(self.htdcm.num_shards, 1)  # Mặc định
        self.assertEqual(len(self.htdcm.shard_trust_scores), 1)
        
        # Kiểm tra trọng số
        self.assertAlmostEqual(self.htdcm.tx_success_weight, 0.4)
        self.assertAlmostEqual(self.htdcm.response_time_weight, 0.2)
        self.assertAlmostEqual(self.htdcm.peer_rating_weight, 0.3)
        self.assertAlmostEqual(self.htdcm.history_weight, 0.1)
    
    def test_update_node_trust(self):
        """Kiểm thử cập nhật điểm tin cậy node."""
        # Lưu điểm tin cậy ban đầu của node 0
        initial_trust = self.htdcm.nodes[0].trust_score
        
        # Cập nhật với một giao dịch thành công
        self.htdcm.update_node_trust(
            node_id=0, 
            tx_success=True, 
            response_time=10.0, 
            is_validator=True
        )
        
        # Điểm tin cậy nên tăng sau giao dịch thành công
        self.assertGreaterEqual(self.htdcm.nodes[0].trust_score, initial_trust)
        
        # Kiểm tra các thuộc tính khác được cập nhật
        self.assertEqual(self.htdcm.nodes[0].successful_txs, 1)
        self.assertEqual(self.htdcm.nodes[0].failed_txs, 0)
        self.assertEqual(len(self.htdcm.nodes[0].response_times), 1)
    
    def test_identify_malicious_nodes(self):
        """Kiểm thử nhận diện node độc hại."""
        # Ban đầu, không có node nào bị coi là độc hại
        self.assertEqual(len(self.htdcm.identify_malicious_nodes()), 0)
        
        # Tạo một node độc hại
        self.htdcm.nodes[3].trust_score = 0.1  # Dưới ngưỡng độc hại mặc định (0.25)
        
        # Chỉ điểm tin cậy thấp không đủ để xác định nút độc hại với bộ lọc nâng cao
        malicious_nodes = self.htdcm.identify_malicious_nodes()
        self.assertEqual(len(malicious_nodes), 0)
        
        # Thêm minh chứng cho việc nút là độc hại: số hoạt động độc hại
        self.htdcm.nodes[3].malicious_activities = 2  # Đủ hoạt động độc hại
        
        # Thêm minh chứng: tỉ lệ thành công thấp
        self.htdcm.nodes[3].successful_txs = 1
        self.htdcm.nodes[3].failed_txs = 9  # 10% tỉ lệ thành công
        
        # Kiểm tra nhận diện (sử dụng bộ lọc nâng cao)
        malicious_nodes = self.htdcm.identify_malicious_nodes()
        self.assertEqual(len(malicious_nodes), 1)
        self.assertEqual(malicious_nodes[0], 3)
        
        # Kiểm tra khi tắt bộ lọc nâng cao (chỉ xét điểm tin cậy và số hoạt động độc hại)
        malicious_nodes = self.htdcm.identify_malicious_nodes(advanced_filtering=False)
        self.assertEqual(len(malicious_nodes), 1)
        self.assertEqual(malicious_nodes[0], 3)
    
    def test_recommend_trusted_validators(self):
        """Kiểm thử đề xuất validator tin cậy."""
        # Đặt điểm tin cậy cho một số node
        self.htdcm.nodes[1].trust_score = 0.9
        self.htdcm.nodes[2].trust_score = 0.8
        self.htdcm.nodes[3].trust_score = 0.95
        
        # Lấy đề xuất 2 validator tin cậy nhất
        validators = self.htdcm.recommend_trusted_validators(shard_id=0, count=2)
        
        # Nên đề xuất node 3 và 1 (có điểm tin cậy cao nhất)
        self.assertEqual(len(validators), 2)
        self.assertIn(3, validators)
        self.assertIn(1, validators)
    
    def test_get_node_trust_scores(self):
        """Kiểm thử lấy tất cả điểm tin cậy của node."""
        # Đặt điểm tin cậy cho một số node
        self.htdcm.nodes[0].trust_score = 0.5
        self.htdcm.nodes[1].trust_score = 0.6
        
        # Lấy tất cả điểm tin cậy
        trust_scores = self.htdcm.get_node_trust_scores()
        
        self.assertEqual(len(trust_scores), self.num_nodes)
        self.assertEqual(trust_scores[0], 0.5)
        self.assertEqual(trust_scores[1], 0.6)
    
    def test_reset(self):
        """Kiểm thử đặt lại HTDCM."""
        # Thay đổi một số giá trị
        self.htdcm.nodes[0].trust_score = 0.1
        self.htdcm.shard_trust_scores[0] = 0.2
        
        # Đặt lại
        self.htdcm.reset()
        
        # Kiểm tra các giá trị đã được đặt lại
        self.assertEqual(self.htdcm.nodes[0].trust_score, 0.7)  # Giá trị mặc định
        self.assertEqual(self.htdcm.shard_trust_scores[0], 0.7)  # Giá trị mặc định

if __name__ == '__main__':
    unittest.main() 