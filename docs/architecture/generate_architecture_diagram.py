#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QTrust Architecture Diagram Generator

This module creates a visual representation of the QTrust blockchain architecture
for use in documentation and scientific papers. The diagram illustrates the key
components and their relationships within the QTrust framework, including the
blockchain environment, DQN agents, consensus mechanisms, and other critical
subsystems.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np

def create_architecture_diagram():
    """Create the QTrust architecture diagram."""
    # Create figure
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    nodes = {
        "QTrust": {"pos": (6, 9), "color": "#3498db", "width": 10, "height": 1},
        "BlockchainEnv": {"pos": (3, 7), "color": "#2ecc71", "width": 4, "height": 1.5},
        "DQNAgents": {"pos": (9, 7), "color": "#9b59b6", "width": 4, "height": 1.5},
        "AdaptiveConsensus": {"pos": (3, 4.5), "color": "#e74c3c", "width": 4, "height": 1.5},
        "MADRAPIDRouter": {"pos": (3, 2), "color": "#f39c12", "width": 4, "height": 1.5},
        "HTDCM": {"pos": (9, 2), "color": "#1abc9c", "width": 4, "height": 1.5},
        "FederatedLearning": {"pos": (9, 4.5), "color": "#d35400", "width": 4, "height": 1.5},
        "CachingSystem": {"pos": (6, 0.5), "color": "#7f8c8d", "width": 10, "height": 1}
    }
    
    # Add edges
    edges = [
        ("QTrust", "BlockchainEnv"),
        ("QTrust", "DQNAgents"),
        ("BlockchainEnv", "AdaptiveConsensus"),
        ("BlockchainEnv", "DQNAgents"),
        ("DQNAgents", "AdaptiveConsensus"),
        ("DQNAgents", "FederatedLearning"),
        ("AdaptiveConsensus", "MADRAPIDRouter"),
        ("AdaptiveConsensus", "FederatedLearning"),
        ("MADRAPIDRouter", "HTDCM"),
        ("FederatedLearning", "MADRAPIDRouter"),
        ("FederatedLearning", "HTDCM")
    ]
    
    # Draw nodes as boxes
    for node, attrs in nodes.items():
        x, y = attrs["pos"]
        width = attrs["width"]
        height = attrs["height"]
        rect = patches.Rectangle((x - width/2, y - height/2), width, height, 
                                linewidth=2, edgecolor='black', 
                                facecolor=attrs["color"], alpha=0.7)
        ax.add_patch(rect)
        plt.text(x, y, node, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Add to graph for drawing edges
        G.add_node(node, pos=attrs["pos"])
    
    # Add edges
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, 
                        arrowsize=20, arrowstyle='->', alpha=0.7)
    
    # Add descriptions below modules
    descriptions = {
        "BlockchainEnv": "Sharding Simulation\nDynamic Resharding\nTransaction Processing",
        "DQNAgents": "Rainbow DQN\nActor-Critic\nPolicy Networks",
        "AdaptiveConsensus": "Fast BFT / PBFT\nRobust BFT\nProtocol Selection",
        "MADRAPIDRouter": "Transaction Routing\nLoad Balancing\nCongestion Avoidance",
        "HTDCM": "Trust Evaluation\nAnomaly Detection\nSecurity Monitoring",
        "FederatedLearning": "FedAvg / FedTrust\nSecure Aggregation\nPrivacy Preservation",
        "CachingSystem": "LRU Cache | TTL Cache | Tensor Cache"
    }
    
    for node, desc in descriptions.items():
        x, y = nodes[node]["pos"]
        height = nodes[node]["height"]
        lines = desc.count('\n') + 1
        plt.text(x, y - height/2 - 0.1 - 0.15*lines, desc, ha='center', va='top', 
                fontsize=8, color='black', alpha=0.9, 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Title and configuration
    plt.text(6, 9.6, "QTrust Blockchain Sharding Framework", ha='center', fontsize=18, fontweight='bold')
    plt.text(6, 9.2, "Deep Reinforcement Learning & Federated Learning", ha='center', fontsize=14)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    # Save the image
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtrust_architecture.png"), 
               dpi=300, bbox_inches='tight')
    
    print("QTrust architecture diagram created and saved at:", 
         os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtrust_architecture.png"))

if __name__ == "__main__":
    create_architecture_diagram() 