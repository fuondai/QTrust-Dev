"""
Test module for blockchain sharding simulation.
"""
import unittest
import random
from unittest.mock import patch, MagicMock
from qtrust.simulation.shard import BlockchainNode, Shard

class TestBlockchainNode(unittest.TestCase):
    """
    Tests for the BlockchainNode class.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Create test nodes
        self.honest_node = BlockchainNode(node_id=1, shard_id=0, is_malicious=False)
        self.malicious_node = BlockchainNode(node_id=2, shard_id=0, is_malicious=True)
        
        # Set up attack behavior for malicious node
        self.malicious_node.attack_behaviors = [
            {
                'type': '51_percent',
                'probability': 1.0,  # 100% probability for testing
                'actions': ['reject_valid_tx', 'validate_invalid_tx', 'double_spend']
            },
            {
                'type': 'ddos',
                'probability': 1.0,  # 100% probability for testing
                'actions': ['flood_requests', 'resource_exhaustion', 'connection_overload']
            },
            {
                'type': 'bribery',
                'probability': 1.0,  # 100% probability for testing
                'actions': ['bribe_validators', 'incentivize_forks', 'corrupt_consensus']
            },
            {
                'type': 'selfish_mining',
                'probability': 1.0,  # 100% probability for testing
                'actions': ['withhold_blocks', 'release_selectively', 'fork_chain']
            }
        ]
    
    def test_node_initialization(self):
        """Test that nodes are properly initialized."""
        # Check honest node properties
        self.assertEqual(self.honest_node.node_id, 1)
        self.assertEqual(self.honest_node.shard_id, 0)
        self.assertFalse(self.honest_node.is_malicious)
        self.assertEqual(self.honest_node.transactions_processed, 0)
        self.assertEqual(self.honest_node.blocks_created, 0)
        self.assertEqual(len(self.honest_node.attack_behaviors), 0)
        
        # Check malicious node properties
        self.assertEqual(self.malicious_node.node_id, 2)
        self.assertEqual(self.malicious_node.shard_id, 0)
        self.assertTrue(self.malicious_node.is_malicious)
        self.assertEqual(self.malicious_node.transactions_processed, 0)
        self.assertEqual(self.malicious_node.blocks_created, 0)
        self.assertEqual(len(self.malicious_node.attack_behaviors), 4)
    
    def test_process_transaction(self):
        """Test transaction processing behavior."""
        # Create test transactions
        valid_tx = {'valid': True, 'value': 10.0}
        invalid_tx = {'valid': False, 'value': 10.0}
        
        # Test honest node with valid transaction
        self.assertTrue(self.honest_node.process_transaction(valid_tx))
        self.assertEqual(self.honest_node.transactions_processed, 1)
        
        # Test honest node with invalid transaction
        self.assertFalse(self.honest_node.process_transaction(invalid_tx))
        self.assertEqual(self.honest_node.transactions_processed, 2)
        
        # Test malicious node with valid transaction (should reject it)
        with patch('random.random', return_value=0.5):  # Ensure attack happens
            self.assertFalse(self.malicious_node.process_transaction(valid_tx))
        
        # Test malicious node with invalid transaction (should validate it)
        with patch('random.random', return_value=0.5):  # Ensure attack happens
            self.assertTrue(self.malicious_node.process_transaction(invalid_tx))
    
    def test_create_block(self):
        """Test block creation behavior."""
        # Create test transactions
        transactions = [{'id': 1, 'value': 10.0}, {'id': 2, 'value': 20.0}]
        
        # Test honest node block creation
        honest_block = self.honest_node.create_block(transactions)
        self.assertEqual(honest_block['creator'], 1)
        self.assertEqual(honest_block['shard_id'], 0)
        self.assertEqual(honest_block['transactions'], transactions)
        self.assertFalse(honest_block['is_withheld'])
        self.assertEqual(self.honest_node.blocks_created, 1)
        
        # Test malicious node block creation (should withhold block)
        with patch('random.random', return_value=0.5):  # Ensure attack happens
            malicious_block = self.malicious_node.create_block(transactions)
            self.assertEqual(malicious_block['creator'], 2)
            self.assertEqual(malicious_block['shard_id'], 0)
            self.assertEqual(malicious_block['transactions'], transactions)
            self.assertTrue(malicious_block['is_withheld'])
            self.assertEqual(self.malicious_node.blocks_created, 1)
    
    def test_execute_ddos_attack(self):
        """Test DDoS attack execution."""
        # Honest node should not execute DDoS attack
        self.assertEqual(self.honest_node.execute_ddos_attack(), 0.0)
        
        # Malicious node should execute DDoS attack
        with patch('random.random', return_value=0.5):  # Ensure attack happens
            with patch('random.uniform', return_value=0.8):  # Fixed intensity for testing
                attack_intensity = self.malicious_node.execute_ddos_attack()
                self.assertEqual(attack_intensity, 0.8)
    
    def test_attempt_bribery(self):
        """Test bribery attack attempt."""
        # Create test target nodes
        target_nodes = [
            BlockchainNode(node_id=3, shard_id=0, is_malicious=False),
            BlockchainNode(node_id=4, shard_id=0, is_malicious=False),
            BlockchainNode(node_id=5, shard_id=0, is_malicious=True)  # Malicious nodes can't be bribed
        ]
        
        # Honest node should not attempt bribery
        bribed_by_honest = self.honest_node.attempt_bribery(target_nodes)
        self.assertEqual(len(bribed_by_honest), 0)
        
        # Malicious node should attempt bribery
        with patch('random.random', return_value=0.2):  # Ensure bribery succeeds (below 0.3)
            bribed_by_malicious = self.malicious_node.attempt_bribery(target_nodes)
            # Should have bribed the two honest nodes but not the malicious one
            self.assertEqual(len(bribed_by_malicious), 2)
            self.assertIn(3, bribed_by_malicious)
            self.assertIn(4, bribed_by_malicious)
            self.assertNotIn(5, bribed_by_malicious)


class TestShard(unittest.TestCase):
    """
    Tests for the Shard class.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Create a shard with 10 nodes, 30% malicious
        self.shard = Shard(
            shard_id=0,
            num_nodes=10,
            malicious_percentage=30.0,
            attack_types=['51_percent', 'ddos', 'bribery', 'finney']
        )
        
        # Create a shard with only honest nodes
        self.honest_shard = Shard(
            shard_id=1,
            num_nodes=10,
            malicious_percentage=0.0
        )
        
        # Create test transactions
        self.transactions = [
            {'id': 1, 'value': 10.0, 'valid': True},
            {'id': 2, 'value': 20.0, 'valid': True},
            {'id': 3, 'value': 15.0, 'valid': False},
            {'id': 4, 'value': 25.0, 'valid': True},
            {'id': 5, 'value': 30.0, 'valid': True}
        ]
    
    def test_shard_initialization(self):
        """Test that shards are properly initialized."""
        # Check shard properties
        self.assertEqual(self.shard.shard_id, 0)
        self.assertEqual(self.shard.num_nodes, 10)
        self.assertEqual(self.shard.malicious_percentage, 30.0)
        self.assertEqual(len(self.shard.attack_types), 4)
        
        # Check node creation
        self.assertEqual(len(self.shard.nodes), 10)
        
        # Check malicious nodes (should be 30% = 3 nodes)
        self.assertEqual(len(self.shard.malicious_nodes), 3)
        
        # Check honest shard
        self.assertEqual(len(self.honest_shard.nodes), 10)
        self.assertEqual(len(self.honest_shard.malicious_nodes), 0)
    
    def test_shard_attack_setup(self):
        """Test that attack behaviors are properly set up."""
        # Check that malicious nodes have attack behaviors configured
        for node in self.shard.malicious_nodes:
            self.assertTrue(node.is_malicious)
            self.assertGreater(len(node.attack_behaviors), 0)
            
            # Check that all specified attack types are set up
            attack_types = [behavior['type'] for behavior in node.attack_behaviors]
            for attack_type in self.shard.attack_types:
                self.assertIn(attack_type, attack_types)
    
    def test_process_transactions(self):
        """Test transaction processing in the shard."""
        # Test transaction processing in honest shard (should succeed)
        with patch('random.sample', side_effect=lambda list, k: list[:k]):  # Deterministic selection
            result_honest = self.honest_shard.process_transactions(self.transactions)
        
        # In honest shard, 4 of 5 transactions should succeed (all except the invalid one)
        self.assertEqual(result_honest['processed'], 5)
        self.assertEqual(result_honest['successful'], 4)
        self.assertEqual(result_honest['rejected'], 1)
        
        # Test transaction processing in malicious shard
        with patch('random.sample', side_effect=lambda list, k: list[:k]):  # Deterministic selection
            with patch('random.random', return_value=0.5):  # Ensure attacks happen
                result_malicious = self.shard.process_transactions(self.transactions)
        
        # In malicious shard, fewer transactions should succeed due to attacks
        self.assertEqual(result_malicious['processed'], 5)
        # Can't deterministically predict successful count due to voting dynamics
        self.assertLessEqual(result_malicious['successful'], 4)
    
    def test_get_shard_health(self):
        """Test calculation of shard health."""
        # Honest shard should have better health
        honest_health = self.honest_shard.get_shard_health()
        self.assertGreaterEqual(honest_health, 0.9)  # 100% honest nodes, high stability
        
        # Shard with malicious nodes should have worse health
        malicious_health = self.shard.get_shard_health()
        # 70% honest nodes
        self.assertLess(malicious_health, honest_health)
        
        # Test effect of DDoS attack on health
        for node in self.shard.malicious_nodes:
            # Configure node for DDoS
            node.attack_behaviors = [{
                'type': 'ddos',
                'probability': 1.0,
                'actions': ['flood_requests']
            }]
        
        with patch('random.random', return_value=0.5):
            with patch('random.uniform', return_value=0.9):
                # Process transactions to trigger DDoS
                self.shard.process_transactions(self.transactions)
                
                # Health should decrease after DDoS
                post_ddos_health = self.shard.get_shard_health()
                self.assertLess(post_ddos_health, malicious_health)
    
    def test_get_malicious_nodes(self):
        """Test retrieving malicious nodes."""
        # Honest shard should return empty list
        self.assertEqual(len(self.honest_shard.get_malicious_nodes()), 0)
        
        # Shard with malicious nodes should return them
        malicious_nodes = self.shard.get_malicious_nodes()
        self.assertEqual(len(malicious_nodes), 3)
        for node in malicious_nodes:
            self.assertTrue(node.is_malicious)


if __name__ == '__main__':
    unittest.main() 