"""
Unit tests for HTDCM (Hierarchical Trust-based Data Center Mechanism).

This module contains test cases for the HTDCM trust management system,
verifying the behavior of nodes and trust scoring mechanisms.
"""

import unittest
import numpy as np
from qtrust.trust.htdcm import HTDCM, HTDCMNode

class TestHTDCMNode(unittest.TestCase):
    """
    Test cases for the HTDCMNode class.
    """
    
    def setUp(self):
        """Initialize a trust node for testing."""
        self.node = HTDCMNode(node_id=1, shard_id=0, initial_trust=0.7)
    
    def test_initialization(self):
        """Test node initialization."""
        self.assertEqual(self.node.node_id, 1)
        self.assertEqual(self.node.shard_id, 0)
        self.assertEqual(self.node.trust_score, 0.7)
        self.assertEqual(self.node.successful_txs, 0)
        self.assertEqual(self.node.failed_txs, 0)
        self.assertEqual(self.node.malicious_activities, 0)
    
    def test_update_trust_score(self):
        """Test trust score updating."""
        # Record initial trust score
        initial_score = self.node.trust_score
        
        # Test normal update
        self.node.update_trust_score(0.9)
        expected_score = self.node.alpha * initial_score + self.node.beta * 0.9
        self.assertAlmostEqual(self.node.trust_score, expected_score)
        
        # Reset node to test extreme values
        self.node = HTDCMNode(node_id=1, shard_id=0, initial_trust=0.7)
        
        # Test extreme values
        self.node.update_trust_score(2.0)  # Exceeds 1.0
        self.assertEqual(self.node.trust_score, 1.0)  # Should be limited to 1.0
        
        # Reset node one more time
        self.node = HTDCMNode(node_id=1, shard_id=0, initial_trust=0.7)
        
        self.node.update_trust_score(-1.0)  # Below 0.0
        self.assertEqual(self.node.trust_score, 0.0)  # Should be limited to 0.0
    
    def test_record_transaction_result(self):
        """Test recording transaction results."""
        # Record successful transaction
        self.node.record_transaction_result(success=True, response_time=10.0, is_validator=True)
        self.assertEqual(self.node.successful_txs, 1)
        self.assertEqual(self.node.failed_txs, 0)
        self.assertEqual(len(self.node.response_times), 1)
        self.assertEqual(self.node.response_times[0], 10.0)
        
        # Record failed transaction
        self.node.record_transaction_result(success=False, response_time=20.0, is_validator=False)
        self.assertEqual(self.node.successful_txs, 1)
        self.assertEqual(self.node.failed_txs, 1)
        self.assertEqual(len(self.node.response_times), 2)
        self.assertEqual(self.node.response_times[1], 20.0)
    
    def test_record_malicious_activity(self):
        """Test recording malicious activity."""
        initial_score = self.node.trust_score
        self.node.record_malicious_activity("double_spending")
        
        self.assertEqual(self.node.malicious_activities, 1)
        self.assertLess(self.node.trust_score, initial_score)  # Trust score should decrease
        
        # Check that trust score is significantly reduced on malicious activity detection
        self.assertEqual(self.node.trust_score, 0.0)
    
    def test_get_success_rate(self):
        """Test getting success rate."""
        # Initially, no transactions
        self.assertEqual(self.node.get_success_rate(), 0.0)
        
        # Add successful and failed transactions
        self.node.record_transaction_result(success=True, response_time=10.0, is_validator=True)
        self.node.record_transaction_result(success=True, response_time=15.0, is_validator=True)
        self.node.record_transaction_result(success=False, response_time=20.0, is_validator=True)
        
        # Success rate should be 2/3
        self.assertAlmostEqual(self.node.get_success_rate(), 2/3)
    
    def test_get_average_response_time(self):
        """Test getting average response time."""
        # Initially, no responses
        self.assertEqual(self.node.get_average_response_time(), 0.0)
        
        # Add response times
        self.node.record_transaction_result(success=True, response_time=10.0, is_validator=True)
        self.node.record_transaction_result(success=True, response_time=20.0, is_validator=True)
        
        # Average time should be (10+20)/2 = 15
        self.assertEqual(self.node.get_average_response_time(), 15.0)

class TestHTDCM(unittest.TestCase):
    """
    Test cases for the HTDCM class.
    """
    
    def setUp(self):
        """Initialize HTDCM for testing."""
        self.num_nodes = 10
        self.htdcm = HTDCM(num_nodes=self.num_nodes)
    
    def test_initialization(self):
        """Test HTDCM initialization."""
        self.assertEqual(len(self.htdcm.nodes), self.num_nodes)
        self.assertEqual(self.htdcm.num_shards, 1)  # Default
        self.assertEqual(len(self.htdcm.shard_trust_scores), 1)
        
        # Check weights
        self.assertAlmostEqual(self.htdcm.tx_success_weight, 0.4)
        self.assertAlmostEqual(self.htdcm.response_time_weight, 0.2)
        self.assertAlmostEqual(self.htdcm.peer_rating_weight, 0.3)
        self.assertAlmostEqual(self.htdcm.history_weight, 0.1)
    
    def test_update_node_trust(self):
        """Test updating node trust."""
        # Save initial trust score of node 0
        initial_trust = self.htdcm.nodes[0].trust_score
        
        # Update with a successful transaction
        self.htdcm.update_node_trust(
            node_id=0, 
            tx_success=True, 
            response_time=10.0, 
            is_validator=True
        )
        
        # Trust score should increase after successful transaction
        self.assertGreaterEqual(self.htdcm.nodes[0].trust_score, initial_trust)
        
        # Check other attributes are updated
        self.assertEqual(self.htdcm.nodes[0].successful_txs, 1)
        self.assertEqual(self.htdcm.nodes[0].failed_txs, 0)
        self.assertEqual(len(self.htdcm.nodes[0].response_times), 1)
    
    def test_identify_malicious_nodes(self):
        """Test identifying malicious nodes."""
        # Initially, no nodes are considered malicious
        self.assertEqual(len(self.htdcm.identify_malicious_nodes()), 0)
        
        # Create a malicious node
        self.htdcm.nodes[3].trust_score = 0.1  # Below default malicious threshold (0.25)
        
        # Low trust score alone is not sufficient to identify malicious nodes with advanced filtering
        malicious_nodes = self.htdcm.identify_malicious_nodes()
        self.assertEqual(len(malicious_nodes), 0)
        
        # Add evidence for node being malicious: number of malicious activities
        self.htdcm.nodes[3].malicious_activities = 2  # Sufficient malicious activities
        
        # Add evidence: low success rate
        self.htdcm.nodes[3].successful_txs = 1
        self.htdcm.nodes[3].failed_txs = 9  # 10% success rate
        
        # Check identification (using advanced filtering)
        malicious_nodes = self.htdcm.identify_malicious_nodes()
        self.assertEqual(len(malicious_nodes), 1)
        self.assertEqual(malicious_nodes[0], 3)
        
        # Check when advanced filtering is turned off (only consider trust score and malicious activities)
        malicious_nodes = self.htdcm.identify_malicious_nodes(advanced_filtering=False)
        self.assertEqual(len(malicious_nodes), 1)
        self.assertEqual(malicious_nodes[0], 3)
    
    def test_recommend_trusted_validators(self):
        """Test recommending trusted validators."""
        # Set trust scores for some nodes
        self.htdcm.nodes[1].trust_score = 0.9
        self.htdcm.nodes[2].trust_score = 0.8
        self.htdcm.nodes[3].trust_score = 0.95
        
        # Get 2 most trusted validator recommendations
        validators = self.htdcm.recommend_trusted_validators(shard_id=0, count=2)
        
        # Should recommend nodes 3 and 1 (highest trust scores)
        self.assertEqual(len(validators), 2)
        # Check that node_ids 3 and 1 are in the validators list
        validator_ids = [validator["node_id"] for validator in validators]
        self.assertIn(3, validator_ids)
        self.assertIn(1, validator_ids)
    
    def test_get_node_trust_scores(self):
        """Test getting all node trust scores."""
        # Set trust scores for some nodes
        self.htdcm.nodes[0].trust_score = 0.5
        self.htdcm.nodes[1].trust_score = 0.6
        
        # Get all trust scores
        trust_scores = self.htdcm.get_node_trust_scores()
        
        self.assertEqual(len(trust_scores), self.num_nodes)
        self.assertEqual(trust_scores[0], 0.5)
        self.assertEqual(trust_scores[1], 0.6)
    
    def test_reset(self):
        """Test resetting HTDCM."""
        # Change some values
        self.htdcm.nodes[0].trust_score = 0.1
        self.htdcm.shard_trust_scores[0] = 0.2
        
        # Reset
        self.htdcm.reset()
        
        # Check values have been reset
        self.assertEqual(self.htdcm.nodes[0].trust_score, 0.7)  # Default value
        self.assertEqual(self.htdcm.shard_trust_scores[0], 0.7)  # Default value

if __name__ == '__main__':
    unittest.main() 