
import pytest
import numpy as np
import torch
import random
from pathlib import Path

# Set fixed seeds for reproducibility
@pytest.fixture(scope="session")
def set_random_seeds():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
    return 42

@pytest.fixture(scope="session")
def test_data_dir():
    return Path("tests/data")

# Common blockchain environment configuration
@pytest.fixture
def blockchain_env_config():
    return {
        "num_shards": 2,
        "num_nodes_per_shard": 3,
        "max_transactions_per_step": 10,
        "transaction_value_range": (0.1, 10.0),
        "max_steps": 100
    }

# Common network topology fixture
@pytest.fixture
def simple_network_topology(set_random_seeds):
    import networkx as nx
    G = nx.Graph()
    
    # Add nodes
    for i in range(6):  # 2 shards with 3 nodes each
        shard_id = i // 3
        G.add_node(
            i, 
            shard_id=shard_id,
            trust_score=0.8,
            processing_power=0.9,
            energy_efficiency=0.85
        )
    
    # Add intra-shard connections
    for shard in range(2):
        nodes = [i for i in range(6) if i // 3 == shard]
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                G.add_edge(
                    nodes[i], 
                    nodes[j], 
                    latency=2.0,
                    bandwidth=100.0
                )
    
    # Add inter-shard connections
    for i in range(3):
        G.add_edge(
            i, 
            i+3, 
            latency=10.0,
            bandwidth=50.0
        )
    
    return G

# Simple transaction generator
@pytest.fixture
def transaction_generator(set_random_seeds):
    def _generate_transactions(num_transactions, shards=2):
        transactions = []
        for i in range(num_transactions):
            source_shard = i % shards
            dest_shard = (i + 1) % shards if i % 3 == 0 else source_shard
            
            tx = {
                'id': f'tx_{i}',
                'source_shard': source_shard,
                'destination_shard': dest_shard,
                'value': 1.0 + (i % 10),
                'timestamp': i * 10
            }
            transactions.append(tx)
        return transactions
    return _generate_transactions
