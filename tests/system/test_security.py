import unittest
import numpy as np
import networkx as nx
from collections import defaultdict
from qtrust.security import ZKProofSystem, ProofType, SecurityLevel
from qtrust.security import ReputationBasedValidatorSelection, ValidatorSelectionPolicy
from qtrust.security import AttackResistanceSystem, AttackType
from qtrust.trust.htdcm import HTDCM, HTDCMNode

class MockHTDCM:
    """Mock của HTDCM để kiểm thử."""
    
    def __init__(self, num_shards=3, num_nodes=10):
        self.num_shards = num_shards
        self.nodes = {}
        self.shards = []
        
        # Tạo shards và nodes
        for shard_id in range(num_shards):
            shard_nodes = []
            for i in range(num_nodes // num_shards):
                node_id = shard_id * (num_nodes // num_shards) + i
                shard_nodes.append(node_id)
                self.nodes[node_id] = HTDCMNode(node_id, shard_id, np.random.random() * 0.5 + 0.5)
            self.shards.append(shard_nodes)
    
    def recommend_trusted_validators(self, shard_id, count=3, include_ml_scores=True):
        """Đề xuất validators đáng tin cậy."""
        if shard_id >= self.num_shards:
            return []
            
        nodes_in_shard = self.shards[shard_id]
        node_scores = []
        
        for node_id in nodes_in_shard:
            node = self.nodes[node_id]
            composite_score = node.trust_score
            
            node_detail = {
                "node_id": node_id,
                "trust_score": node.trust_score,
                "success_rate": 0.9,
                "response_time": 10.0,
                "malicious_activities": 0,
                "composite_score": composite_score
            }
            
            node_scores.append(node_detail)
        
        return sorted(node_scores, key=lambda x: x["composite_score"], reverse=True)[:count]
    
    def detect_advanced_attacks(self, transaction_history):
        """Phát hiện tấn công nâng cao."""
        if not transaction_history:
            return {
                "under_attack": False,
                "attack_types": [],
                "confidence": 0.0,
                "suspect_nodes": [],
                "recommended_actions": []
            }
            
        # Giả lập phát hiện tấn công dựa trên dữ liệu
        under_attack = any(tx.get("suspicious", False) for tx in transaction_history)
        
        if under_attack:
            suspect_nodes = [tx["node_id"] for tx in transaction_history if tx.get("suspicious", False)]
            return {
                "under_attack": True,
                "attack_types": ["Simulated Attack"],
                "confidence": 0.7,
                "suspect_nodes": suspect_nodes,
                "recommended_actions": ["Giả lập hành động phòng thủ"]
            }
        
        return {
            "under_attack": False,
            "attack_types": [],
            "confidence": 0.0,
            "suspect_nodes": [],
            "recommended_actions": []
        }
    
    def enhance_security_posture(self, attack_detection_result):
        """Tăng cường tư thế bảo mật."""
        pass


class TestZKProofSystem(unittest.TestCase):
    """Kiểm thử hệ thống ZK Proof."""
    
    def setUp(self):
        """Khởi tạo môi trường kiểm thử."""
        self.zk_system_low = ZKProofSystem(security_level="low", energy_optimization=True)
        self.zk_system_medium = ZKProofSystem(security_level="medium", energy_optimization=True)
        self.zk_system_high = ZKProofSystem(security_level="high", energy_optimization=False)
        
        # Dữ liệu kiểm thử
        self.test_data = {
            "transaction": {
                "tx_id": "0x1234",
                "sender": "alice",
                "receiver": "bob",
                "amount": 100
            },
            "ownership": {
                "public_key": "pk_alice",
                "signature": "sig_data",
                "asset_id": "asset_1"
            },
            "range": {
                "value": 50,
                "min": 0,
                "max": 100
            }
        }
    
    def test_proof_generation(self):
        """Kiểm tra tạo bằng chứng."""
        # Tạo các loại bằng chứng
        tx_proof = self.zk_system_medium.generate_proof(
            self.test_data["transaction"], 
            ProofType.TRANSACTION_VALIDITY
        )
        
        # Kiểm tra kết quả
        self.assertIn("tx_hash", tx_proof)
        self.assertIn("witness", tx_proof)
        self.assertIn("security_level", tx_proof)
        self.assertEqual(tx_proof["security_level"], "medium")
        
        # Kiểm tra bằng chứng quyền sở hữu
        ownership_proof = self.zk_system_medium.generate_proof(
            self.test_data["ownership"], 
            ProofType.OWNERSHIP
        )
        
        self.assertIn("ownership_hash", ownership_proof)
        self.assertIn("commitment", ownership_proof)
    
    def test_proof_verification(self):
        """Kiểm tra xác minh bằng chứng."""
        # Tạo bằng chứng
        range_proof = self.zk_system_medium.generate_proof(
            self.test_data["range"], 
            ProofType.RANGE_PROOF
        )
        
        # Xác minh bằng chứng
        verification_result = self.zk_system_medium.verify_proof(
            self.test_data["range"], 
            range_proof
        )
        
        self.assertTrue(verification_result)
        
        # Kiểm tra xác minh với dữ liệu không khớp
        invalid_data = self.test_data["range"].copy()
        invalid_data["value"] = 150  # Ngoài khoảng
        
        invalid_verification = self.zk_system_medium.verify_proof(
            invalid_data, 
            range_proof
        )
        
        self.assertFalse(invalid_verification)
    
    def test_energy_optimization(self):
        """Kiểm tra tối ưu hóa năng lượng."""
        # So sánh năng lượng giữa các mức độ bảo mật
        data = self.test_data["transaction"]
        
        # Tạo bằng chứng ở các mức bảo mật khác nhau
        proof_low = self.zk_system_low.generate_proof(data, ProofType.TRANSACTION_VALIDITY)
        proof_medium = self.zk_system_medium.generate_proof(data, ProofType.TRANSACTION_VALIDITY)
        proof_high = self.zk_system_high.generate_proof(data, ProofType.TRANSACTION_VALIDITY)
        
        # Kiểm tra chi phí năng lượng
        self.assertLess(proof_low["energy_cost"], proof_medium["energy_cost"])
        self.assertLess(proof_medium["energy_cost"], proof_high["energy_cost"])
        
        # Kiểm tra năng lượng tiết kiệm
        self.assertGreater(self.zk_system_low.stats["energy_saved"], 0)
        self.assertGreater(self.zk_system_medium.stats["energy_saved"], 0)
        
        # ZK System cao không tối ưu hóa năng lượng
        self.assertEqual(self.zk_system_high.stats["energy_saved"], 0)
    
    def test_statistics(self):
        """Kiểm tra thống kê."""
        # Tạo một số bằng chứng
        for _ in range(5):
            self.zk_system_medium.generate_proof(
                self.test_data["transaction"], 
                ProofType.TRANSACTION_VALIDITY
            )
        
        # Lấy thống kê
        stats = self.zk_system_medium.get_statistics()
        
        # Kiểm tra số lượng bằng chứng đã tạo
        self.assertEqual(stats["proofs_generated"], 5)
        self.assertGreater(stats["energy_saved"], 0)
        self.assertEqual(stats["security_level"], "medium")


class TestValidatorSelection(unittest.TestCase):
    """Kiểm thử hệ thống chọn validator."""
    
    def setUp(self):
        """Khởi tạo môi trường kiểm thử."""
        self.htdcm = MockHTDCM(num_shards=3, num_nodes=15)
        
        # Tạo các chính sách selection khác nhau
        self.random_selector = ReputationBasedValidatorSelection(
            self.htdcm,
            policy=ValidatorSelectionPolicy.RANDOM,
            zk_enabled=True,
            use_rotation=True,
            rotation_period=10
        )
        
        self.reputation_selector = ReputationBasedValidatorSelection(
            self.htdcm,
            policy=ValidatorSelectionPolicy.REPUTATION,
            zk_enabled=True,
            use_rotation=True,
            rotation_period=5
        )
        
        self.hybrid_selector = ReputationBasedValidatorSelection(
            self.htdcm,
            policy=ValidatorSelectionPolicy.HYBRID,
            zk_enabled=False,
            use_rotation=True,
            rotation_period=8
        )
    
    def test_validator_selection(self):
        """Kiểm tra lựa chọn validator."""
        # Chọn validator cho shard 0, block 1
        validators_random = self.random_selector.select_validators(0, 1, 3)
        
        # Kiểm tra số lượng validator
        self.assertEqual(len(validators_random), 3)
        
        # Chọn validator dựa trên reputation
        validators_reputation = self.reputation_selector.select_validators(0, 1, 3)
        
        # Kiểm tra số lượng validator
        self.assertEqual(len(validators_reputation), 3)
        
        # Kiểm tra xem validator đã được lưu vào lịch sử
        self.assertIn(0, self.reputation_selector.validator_history)
        self.assertEqual(len(self.reputation_selector.validator_history[0]), 1)
    
    def test_validator_rotation(self):
        """Kiểm tra luân chuyển validator."""
        # Chọn validator cho block 1
        validators_block1 = self.reputation_selector.select_validators(0, 1, 3)
        
        # Chọn validator cho block 2 (trước khi đến chu kỳ luân chuyển)
        # Vì các yếu tố ngẫu nhiên trong lựa chọn, chúng ta không thể đảm bảo rằng
        # validators_block1 == validators_block2
        # Nhưng chúng ta có thể so sánh kết quả luân chuyển ở block 5
        validators_block2 = self.reputation_selector.select_validators(0, 2, 3)
        
        # Lưu validator từ block 2 để sau đó so sánh
        current_validators = validators_block2
        
        # Chọn validator cho block 5 (kích hoạt luân chuyển)
        validators_block5 = self.reputation_selector.select_validators(0, 5, 3)
        
        # Kiểm tra xem validator có thay đổi khi đến block xoay vòng
        self.assertNotEqual(set(current_validators), set(validators_block5))
        
        # Kiểm tra xem các validator block 5 có được lưu vào lịch sử không
        has_block5_history = False
        for block_num, validators in self.reputation_selector.validator_history[0]:
            if block_num == 5:
                has_block5_history = True
                self.assertEqual(set(validators), set(validators_block5))
                break
                
        self.assertTrue(has_block5_history, "Không tìm thấy lịch sử validator cho block 5")
    
    def test_statistics(self):
        """Kiểm tra thống kê."""
        # Thực hiện một số lần chọn validator
        for block in range(1, 11):
            self.hybrid_selector.select_validators(0, block, 3)
        
        # Lấy thống kê
        stats = self.hybrid_selector.get_statistics()
        
        # Kiểm tra số lần lựa chọn
        self.assertEqual(stats["selections"], 10)
        
        # Kiểm tra số lần luân chuyển (với rotation_period=8, chỉ có 1 lần luân chuyển ở block 8)
        self.assertEqual(stats["rotations"], 1)
        
        # Kiểm tra chính sách
        self.assertEqual(stats["policy"], ValidatorSelectionPolicy.HYBRID)


class TestAttackResistance(unittest.TestCase):
    """Kiểm thử hệ thống chống tấn công."""
    
    def setUp(self):
        """Khởi tạo môi trường kiểm thử."""
        self.htdcm = MockHTDCM(num_shards=2, num_nodes=10)
        
        # Tạo network mô phỏng
        self.network = nx.Graph()
        for shard_id, shard_nodes in enumerate(self.htdcm.shards):
            for node_id in shard_nodes:
                self.network.add_node(node_id, shard=shard_id, trust_score=self.htdcm.nodes[node_id].trust_score)
        
        # Thêm edges mô phỏng
        for node in self.network.nodes:
            # Kết nối với node trong cùng shard
            shard = self.network.nodes[node]["shard"]
            for other_node in self.htdcm.shards[shard]:
                if node != other_node:
                    self.network.add_edge(node, other_node)
        
        # Tạo hệ thống chống tấn công
        self.attack_system = AttackResistanceSystem(
            self.htdcm,
            network=self.network,
            detection_threshold=0.6,
            auto_response=True
        )
    
    def test_scan_for_attacks(self):
        """Kiểm tra quét tấn công."""
        # Tạo lịch sử giao dịch bình thường
        normal_transactions = [
            {"tx_id": f"tx_{i}", "node_id": i % 10, "successful": True}
            for i in range(20)
        ]
        
        # Quét tấn công với dữ liệu bình thường
        normal_result = self.attack_system.scan_for_attacks(normal_transactions)
        
        # Kiểm tra kết quả
        self.assertFalse(normal_result["under_attack"])
        self.assertEqual(len(normal_result["new_attacks_detected"]), 0)
        
        # Tạo lịch sử giao dịch đáng ngờ
        suspicious_transactions = normal_transactions + [
            {"tx_id": f"sus_tx_{i}", "node_id": i % 5, "suspicious": True}
            for i in range(5)
        ]
        
        # Quét tấn công với dữ liệu đáng ngờ
        suspicious_result = self.attack_system.scan_for_attacks(suspicious_transactions)
        
        # Kiểm tra kết quả
        self.assertTrue(suspicious_result["under_attack"])
        self.assertGreater(len(suspicious_result["new_attacks_detected"]), 0)
    
    def test_attack_report(self):
        """Kiểm tra báo cáo tấn công."""
        # Tạo lịch sử giao dịch đáng ngờ
        suspicious_transactions = [
            {"tx_id": f"sus_tx_{i}", "node_id": i % 5, "suspicious": True}
            for i in range(5)
        ]
        
        # Quét tấn công trước
        self.attack_system.scan_for_attacks(suspicious_transactions)
        
        # Lấy báo cáo tấn công
        report = self.attack_system.get_attack_report()
        
        # Kiểm tra báo cáo
        self.assertTrue(report["under_attack"])
        self.assertGreater(report["active_attacks"], 0)
        self.assertGreater(len(report["attack_types"]), 0)
        
        # Kiểm tra thống kê
        self.assertEqual(report["stats"]["total_scans"], 1)
        self.assertGreater(report["stats"]["attacks_detected"], 0)


if __name__ == '__main__':
    unittest.main() 