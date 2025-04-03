"""
Blockchain sharding simulation module for the QTrust system.

This module provides classes to simulate blockchain shards and nodes within a sharded network.
It models honest and malicious nodes, various attack behaviors, and transaction processing within shards.
The simulation supports multiple attack types including 51% attacks, Sybil attacks, eclipse attacks,
selfish mining, bribery attacks, DDoS attacks, and Finney attacks.
"""

import random
from typing import List, Optional, Dict, Any

class BlockchainNode:
    """
    Represents a node in the blockchain network.
    """
    def __init__(self, node_id: int, shard_id: int, is_malicious: bool = False):
        """
        Initialize a blockchain node.
        
        Args:
            node_id: ID of the node
            shard_id: ID of the shard that the node belongs to
            is_malicious: True if the node is malicious
        """
        self.node_id = node_id
        self.shard_id = shard_id
        self.is_malicious = is_malicious
        self.connections = []  # List of connected nodes
        self.transactions_processed = 0
        self.blocks_created = 0
        self.attack_behaviors = []  # List of attack behaviors
        self.resource_usage = 0.0  # Resource usage (0.0-1.0)
        
    def process_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Process a transaction.
        
        Args:
            transaction: Transaction information
            
        Returns:
            bool: True if the transaction was processed successfully
        """
        # If malicious node, may perform an attack
        if self.is_malicious and self.attack_behaviors:
            for behavior in self.attack_behaviors:
                # Check probability of performing attack
                if random.random() < behavior['probability']:
                    # Perform attack behavior
                    if 'reject_valid_tx' in behavior['actions'] and transaction.get('valid', True):
                        return False
                    elif 'validate_invalid_tx' in behavior['actions'] and not transaction.get('valid', True):
                        return True
                    elif 'double_spend' in behavior['actions']:
                        # Simulate double-spending
                        pass
                    # Simulate other behaviors based on attack type
                    
        # Normal processing for honest nodes or malicious nodes not performing attacks
        self.transactions_processed += 1
        
        return transaction.get('valid', True)
    
    def create_block(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a new block.
        
        Args:
            transactions: List of transactions to include in the block
            
        Returns:
            Dict[str, Any]: Information about the created block
        """
        self.blocks_created += 1
        
        # Check if node is performing selfish mining attack
        is_withholding = False
        if self.is_malicious:
            for behavior in self.attack_behaviors:
                if behavior['type'] == 'selfish_mining' and 'withhold_blocks' in behavior['actions']:
                    if random.random() < behavior['probability']:
                        is_withholding = True
        
        return {
            'creator': self.node_id,
            'shard_id': self.shard_id,
            'transactions': transactions,
            'is_withheld': is_withholding,
            'timestamp': random.randint(100000, 999999)  # Simulate timestamp
        }
    
    def execute_ddos_attack(self) -> float:
        """
        Execute a DDoS attack if the node is configured to do so.
        
        Returns:
            float: Attack intensity (0.0-1.0)
        """
        if not self.is_malicious:
            return 0.0
            
        for behavior in self.attack_behaviors:
            if behavior['type'] == 'ddos' and random.random() < behavior['probability']:
                # Simulate DDoS attack
                attack_intensity = random.uniform(0.5, 1.0)
                return attack_intensity
                
        return 0.0
        
    def attempt_bribery(self, target_nodes: List['BlockchainNode']) -> List[int]:
        """
        Attempt a bribery attack if the node is configured to do so.
        
        Args:
            target_nodes: List of nodes that could be bribed
            
        Returns:
            List[int]: IDs of nodes that were successfully bribed
        """
        bribed_nodes = []
        
        if not self.is_malicious:
            return bribed_nodes
            
        for behavior in self.attack_behaviors:
            if behavior['type'] == 'bribery' and random.random() < behavior['probability']:
                # Simulate bribery attack
                for target in target_nodes:
                    # Only honest nodes can be bribed
                    if not target.is_malicious and random.random() < 0.3:  # 30% chance of successful bribery
                        bribed_nodes.append(target.node_id)
                
        return bribed_nodes

class Shard:
    """
    Represents a shard in the blockchain network.
    """
    def __init__(self, shard_id: int, num_nodes: int, 
                 malicious_percentage: float = 0.0,
                 attack_types: List[str] = None):
        """
        Initialize a shard.
        
        Args:
            shard_id: ID of the shard
            num_nodes: Number of nodes in the shard
            malicious_percentage: Percentage of malicious nodes
            attack_types: List of attack types
        """
        self.shard_id = shard_id
        self.num_nodes = num_nodes
        self.malicious_percentage = malicious_percentage
        self.attack_types = attack_types or []
        
        # Initialize nodes
        self.nodes = []
        self.malicious_nodes = []
        self._initialize_nodes()
        
        # Shard statistics
        self.transactions_pool = []
        self.confirmed_transactions = []
        self.blocks = []
        self.network_stability = 1.0
        self.resource_utilization = 0.0
        
    def _initialize_nodes(self):
        """Initialize nodes in the shard."""
        # Number of malicious nodes
        num_malicious = int(self.num_nodes * self.malicious_percentage / 100.0)
        
        # Create honest nodes
        for i in range(self.num_nodes - num_malicious):
            node_id = self.shard_id * self.num_nodes + i
            node = BlockchainNode(node_id, self.shard_id, is_malicious=False)
            self.nodes.append(node)
        
        # Create malicious nodes
        for i in range(num_malicious):
            node_id = self.shard_id * self.num_nodes + (self.num_nodes - num_malicious) + i
            node = BlockchainNode(node_id, self.shard_id, is_malicious=True)
            self.nodes.append(node)
            self.malicious_nodes.append(node)
        
        # Set up attack behaviors
        if self.malicious_nodes and self.attack_types:
            self._setup_attack_behavior()
    
    def _setup_attack_behavior(self):
        """Set up attack behaviors for malicious nodes."""
        for node in self.malicious_nodes:
            # Attack behaviors based on attack type
            if '51_percent' in self.attack_types:
                node.attack_behaviors.append({
                    'type': '51_percent',
                    'probability': 0.8,
                    'actions': ['reject_valid_tx', 'validate_invalid_tx', 'double_spend']
                })
            
            if 'sybil' in self.attack_types:
                node.attack_behaviors.append({
                    'type': 'sybil',
                    'probability': 0.7,
                    'actions': ['create_fake_identities', 'vote_manipulation']
                })
            
            if 'eclipse' in self.attack_types:
                node.attack_behaviors.append({
                    'type': 'eclipse',
                    'probability': 0.75,
                    'actions': ['isolate_node', 'filter_transactions']
                })
                
            if 'selfish_mining' in self.attack_types:
                node.attack_behaviors.append({
                    'type': 'selfish_mining',
                    'probability': 0.6,
                    'actions': ['withhold_blocks', 'release_selectively', 'fork_chain']
                })
                
            if 'bribery' in self.attack_types:
                node.attack_behaviors.append({
                    'type': 'bribery',
                    'probability': 0.5,
                    'actions': ['bribe_validators', 'incentivize_forks', 'corrupt_consensus']
                })
                
            if 'ddos' in self.attack_types:
                node.attack_behaviors.append({
                    'type': 'ddos',
                    'probability': 0.9,
                    'actions': ['flood_requests', 'resource_exhaustion', 'connection_overload']
                })
                
            if 'finney' in self.attack_types:
                node.attack_behaviors.append({
                    'type': 'finney',
                    'probability': 0.65,
                    'actions': ['prepare_hidden_chain', 'double_spend_attack', 'revert_transactions']
                })
    
    def process_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of transactions in the shard.
        
        Args:
            transactions: List of transactions to process
            
        Returns:
            Dict[str, Any]: Processing results
        """
        # Simulate DDoS attack
        ddos_intensity = 0.0
        for node in self.malicious_nodes:
            ddos_intensity = max(ddos_intensity, node.execute_ddos_attack())
        
        # If DDoS attack, reduce processing performance
        performance_factor = max(0.1, 1.0 - ddos_intensity * 0.8)
        
        # Randomly select validator nodes
        available_nodes = [node for node in self.nodes 
                          if not (node.is_malicious and any(b['type'] == 'ddos' for b in node.attack_behaviors))]
        
        validator_count = max(3, int(len(available_nodes) * 0.3))  # At least 3 validator nodes
        validators = random.sample(available_nodes, min(validator_count, len(available_nodes)))
        
        # Process each transaction
        processed_count = 0
        successful_count = 0
        rejected_count = 0
        
        for tx in transactions:
            # Process through validators
            votes = []
            for validator in validators:
                # Simulate bribery attack
                if any(node.is_malicious and any(b['type'] == 'bribery' for b in node.attack_behaviors) 
                      for node in self.malicious_nodes):
                    # Check if validator is bribed
                    for attacker in [n for n in self.malicious_nodes 
                                  if any(b['type'] == 'bribery' for b in n.attack_behaviors)]:
                        bribed_nodes = attacker.attempt_bribery([validator])
                        if validator.node_id in bribed_nodes:
                            # Bribed node will vote as the attacker wants
                            votes.append(not tx.get('valid', True))
                            break
                    else:
                        # Not bribed, normal voting
                        votes.append(validator.process_transaction(tx))
                else:
                    # No bribery attack, normal processing
                    votes.append(validator.process_transaction(tx))
            
            # Check result
            if sum(votes) > len(votes) / 2:  # Majority agrees
                successful_count += 1
                tx['status'] = 'completed'
                self.confirmed_transactions.append(tx)
            else:
                rejected_count += 1
                tx['status'] = 'rejected'
            
            processed_count += 1
        
        # Process Finney attack if present
        if any(node.is_malicious and any(b['type'] == 'finney' for b in node.attack_behaviors) 
              for node in self.malicious_nodes):
            # Chance to reverse some completed transactions
            finney_attacker = next((node for node in self.malicious_nodes 
                                if any(b['type'] == 'finney' for b in node.attack_behaviors)), None)
            
            if finney_attacker and self.confirmed_transactions:
                # Finney attack success probability
                finney_probability = 0.0
                for behavior in finney_attacker.attack_behaviors:
                    if behavior['type'] == 'finney':
                        finney_probability = behavior['probability']
                
                if random.random() < finney_probability:
                    # Randomly select some transactions to revert
                    revert_count = min(3, len(self.confirmed_transactions))
                    transactions_to_revert = random.sample(self.confirmed_transactions, revert_count)
                    
                    for tx in transactions_to_revert:
                        tx['status'] = 'reverted'
                        self.confirmed_transactions.remove(tx)
                        successful_count -= 1
        
        # Update shard statistics
        self.network_stability = max(0.2, self.network_stability - ddos_intensity * 0.3)
        self.resource_utilization = min(1.0, self.resource_utilization + ddos_intensity * 0.4)
        
        return {
            'processed': processed_count,
            'successful': successful_count,
            'rejected': rejected_count,
            'network_stability': self.network_stability,
            'resource_utilization': self.resource_utilization
        }
    
    def get_shard_health(self) -> float:
        """
        Calculate the health score of the shard based on multiple factors.
        
        Returns:
            float: Health score (0.0-1.0)
        """
        # Ratio of honest nodes
        honest_ratio = 1.0 - len(self.malicious_nodes) / max(1, len(self.nodes))
        
        # Combine with network stability
        health = 0.6 * honest_ratio + 0.3 * self.network_stability + 0.1 * (1.0 - self.resource_utilization)
        
        return health
    
    def get_malicious_nodes(self) -> List[BlockchainNode]:
        """
        Get the list of malicious nodes in the shard.
        
        Returns:
            List[BlockchainNode]: List of malicious nodes
        """
        return self.malicious_nodes 