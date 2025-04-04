#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Architecture Diagram Generator

This module generates architecture diagrams for the QTrust project, including:
- Main component relationship diagram showing how different modules interact
- System overview diagram illustrating the core structure of the QTrust framework
- Visual representation of the project's architectural layers

The diagrams are saved in both the architecture documentation directory and
the exported charts directory for use in reports and presentations.
"""

import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def ensure_chart_dirs():
    """Ensure all chart directories exist."""
    chart_dirs = [
        'charts',
        'charts/architecture',
        'charts/visualization',
        'charts/benchmark',
        'charts/simulation'
    ]
    for directory in chart_dirs:
        os.makedirs(directory, exist_ok=True)
    print("Chart directories created or verified.")

def get_chart_path(filename, category='architecture'):
    """Get the path for saving a chart."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chart_dir = os.path.join(base_dir, 'charts', category)
    os.makedirs(chart_dir, exist_ok=True)
    return os.path.join(chart_dir, filename)

def create_architecture_diagram():
    """Create the architecture diagram for QTrust system."""
    # Create figure
    plt.figure(figsize=(16, 12))
    ax = plt.gca()
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes - repositioned to create clearer diagrams
    nodes = [
        ('QTrust', {'pos': (0, 0)}),
        ('BlockchainEnv', {'pos': (-4, -2)}),
        ('DQNAgents', {'pos': (0, -2)}),
        ('AdaptiveConsensus', {'pos': (4, -2)}),
        ('MADRAPIDRouter', {'pos': (-4, -4)}),
        ('HTDCM', {'pos': (0, -4)}),
        ('FederatedLearning', {'pos': (4, -4)}),
        ('CachingSystem', {'pos': (0, -6)})
    ]
    
    G.add_nodes_from(nodes)
    
    # Define node positions, colors, and sizes
    pos = {node: data['pos'] for node, data in nodes}
    
    node_colors = {
        'QTrust': '#3498DB',
        'BlockchainEnv': '#2ECC71',
        'DQNAgents': '#9B59B6',
        'AdaptiveConsensus': '#F1C40F',
        'MADRAPIDRouter': '#E74C3C',
        'HTDCM': '#1ABC9C',
        'FederatedLearning': '#34495E',
        'CachingSystem': '#F39C12'
    }
    
    node_sizes = {
        'QTrust': 3000,
        'BlockchainEnv': 2500,
        'DQNAgents': 2500,
        'AdaptiveConsensus': 2500,
        'MADRAPIDRouter': 2200,
        'HTDCM': 2200,
        'FederatedLearning': 2200,
        'CachingSystem': 2200
    }
    
    # Draw nodes using different node types - increased size for better visibility
    for node, data in nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[node],
            node_color=node_colors[node],
            node_size=node_sizes[node],
            alpha=0.8,
            node_shape='o',  # Chuyển sang hình tròn để rõ ràng hơn
            edgecolors='black',
            linewidths=2
        )
    
    # Add edges - improved connection structure
    edges = [
        ('QTrust', 'BlockchainEnv'),
        ('QTrust', 'DQNAgents'),
        ('QTrust', 'AdaptiveConsensus'),
        ('BlockchainEnv', 'MADRAPIDRouter'),
        ('BlockchainEnv', 'DQNAgents'),
        ('DQNAgents', 'AdaptiveConsensus'),
        ('DQNAgents', 'FederatedLearning'),
        ('AdaptiveConsensus', 'FederatedLearning'),
        ('MADRAPIDRouter', 'HTDCM'),
        ('HTDCM', 'CachingSystem'),
        ('FederatedLearning', 'CachingSystem')
    ]
    
    G.add_edges_from(edges)
    
    # Classify edges
    primary_edges = edges[:3]  # Core connections
    secondary_edges = edges[3:8]  # Main functional connections
    tertiary_edges = edges[8:]  # Support connections
    
    # Draw edges với độ rõ ràng khác nhau theo tầng
    nx.draw_networkx_edges(
        G, pos,
        edgelist=primary_edges,
        width=3.5,
        alpha=0.9,
        edge_color='#2C3E50',
        style='solid',
        arrowsize=25,
        arrowstyle='-|>', # Clearer arrows
        connectionstyle='arc3,rad=0.1'  # Slight curve to avoid overlapping
    )
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=secondary_edges,
        width=2.5,
        alpha=0.8,
        edge_color='#7F8C8D',
        style='solid',
        arrowsize=20,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1'
    )
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=tertiary_edges,
        width=2,
        alpha=0.7,
        edge_color='#95A5A6',
        style='dashed',
        arrowsize=15,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1'
    )
    
    # Add labels - increased size and clarity
    nx.draw_networkx_labels(
        G, pos,
        font_size=16,
        font_family='sans-serif',
        font_weight='bold',
        font_color='white',  # White to stand out against node background
        bbox=dict(facecolor='black', alpha=0.2, edgecolor='none')
    )
    
    # Brief description for each module
    descriptions = {
        'QTrust': 'Core Framework',
        'BlockchainEnv': 'Sharding Blockchain Environment',
        'DQNAgents': 'Reinforcement Learning Agents',
        'AdaptiveConsensus': 'Adaptive Protocol Selection',
        'MADRAPIDRouter': 'Transaction Routing System',
        'HTDCM': 'Trust-based Security Mechanism',
        'FederatedLearning': 'Decentralized Training System',
        'CachingSystem': 'Performance Optimization Cache'
    }
    
    # Add descriptions next to nodes instead of below them
    for node, desc in descriptions.items():
        x, y = pos[node]
        offset_x = 0
        offset_y = 0
        
        # Adjust text position based on node position
        if 'QTrust' in node:
            offset_y = 0.7
        elif 'BlockchainEnv' in node:
            offset_x = -1.0
        elif 'DQNAgents' in node:
            offset_y = 0.7
        elif 'AdaptiveConsensus' in node:
            offset_x = 1.0
        elif 'MADRAPIDRouter' in node:
            offset_x = -1.0
        elif 'HTDCM' in node:
            offset_y = -0.7
        elif 'FederatedLearning' in node:
            offset_x = 1.0
        elif 'CachingSystem' in node:
            offset_y = -0.7
            
        plt.text(
            x + offset_x, y + offset_y,
            desc,
            fontsize=12,
            ha='center',
            va='center',
            color='black',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5')
        )
    
    # Define graph boundaries
    plt.xlim(-6, 6)
    plt.ylim(-7, 1.5)
    plt.axis('off')
    
    # Add title
    plt.title('QTrust Architecture', fontsize=22, pad=20, fontweight='bold')
    
    # Add legend for component types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', 
                  markersize=15, label='Core Component'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71', 
                  markersize=15, label='Environment'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9B59B6', 
                  markersize=15, label='Intelligence'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F1C40F', 
                  markersize=15, label='Consensus'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', 
                  markersize=15, label='Routing'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1ABC9C', 
                  markersize=15, label='Security'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#34495E', 
                  markersize=15, label='Learning'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F39C12', 
                  markersize=15, label='Caching')
    ]
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
               ncol=4, fancybox=True, shadow=True, fontsize=12)
    
    # Save figure
    plt.tight_layout()
    
    # Save diagram to charts/architecture directory
    architecture_path = get_chart_path('qtrust_architecture.png', 'architecture')
    plt.savefig(architecture_path, dpi=300, bbox_inches='tight')
    
    print(f"Architecture diagram created: {architecture_path}")

def create_system_overview():
    """Create a system overview diagram for QTrust."""
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create central area for 'QTrust Core'
    core_box = plt.Rectangle((-5, -4), 10, 8, fill=True, alpha=0.1, 
                             facecolor='#3498db', edgecolor='#2980b9', lw=3, 
                             zorder=0, linestyle='--')
    plt.gca().add_patch(core_box)
    plt.text(0, 3, 'QTrust Framework', fontsize=22, fontweight='bold', 
             horizontalalignment='center', color='#2980b9')
    
    # Main modules - improved position and information
    modules = [
        {'name': 'Blockchain\nEnvironment', 'pos': (-3, 1), 'color': '#e74c3c', 'size': (3, 2)},
        {'name': 'Reinforcement\nLearning', 'pos': (3, 1), 'color': '#2ecc71', 'size': (3, 2)},
        {'name': 'Federated\nLearning', 'pos': (-3, -2), 'color': '#34495e', 'size': (3, 2)},
        {'name': 'Security\nLayer', 'pos': (3, -2), 'color': '#1abc9c', 'size': (3, 2)},
    ]
    
    # Draw main modules with clearer shapes
    for module in modules:
        x, y = module['pos']
        width, height = module['size']
        rect = plt.Rectangle((x-width/2, y-height/2), width, height, fill=True, alpha=0.8,
                           facecolor=module['color'], edgecolor='black', lw=2,
                           zorder=1)
        plt.gca().add_patch(rect)
        plt.text(x, y, module['name'], fontsize=14, fontweight='bold',
                horizontalalignment='center', verticalalignment='center', color='white')
    
    # External components 
    ext_components = [
        {'name': 'Network', 'pos': (-7, 0), 'color': '#8e44ad'},
        {'name': 'Clients', 'pos': (0, -6), 'color': '#d35400'},
        {'name': 'Validators', 'pos': (7, 0), 'color': '#16a085'},
        {'name': 'Data\nStores', 'pos': (0, 6), 'color': '#c0392b'}
    ]
    
    # Draw external components
    for comp in ext_components:
        x, y = comp['pos']
        circle = plt.Circle((x, y), 1.2, fill=True, alpha=0.7,
                           facecolor=comp['color'], edgecolor='black', lw=1.5, zorder=1)
        plt.gca().add_patch(circle)
        plt.text(x, y, comp['name'], fontsize=12, fontweight='bold',
                horizontalalignment='center', verticalalignment='center', color='white')
    
    # Connect components với arrows
    connections = [
        # External to internal
        {'start': (-7, 0), 'end': (-3, 1), 'color': '#8e44ad', 'style': '-', 'width': 2},
        {'start': (7, 0), 'end': (3, 1), 'color': '#16a085', 'style': '-', 'width': 2},
        {'start': (0, 6), 'end': (-3, 1), 'color': '#c0392b', 'style': '-', 'width': 2},
        {'start': (0, -6), 'end': (3, -2), 'color': '#d35400', 'style': '-', 'width': 2},
        
        # Internal connections
        {'start': (-3, 1), 'end': (3, 1), 'color': 'gray', 'style': '--', 'width': 1.5},
        {'start': (-3, 1), 'end': (-3, -2), 'color': 'gray', 'style': '--', 'width': 1.5},
        {'start': (3, 1), 'end': (3, -2), 'color': 'gray', 'style': '--', 'width': 1.5},
        {'start': (-3, -2), 'end': (3, -2), 'color': 'gray', 'style': '--', 'width': 1.5},
    ]
    
    # Draw connections with clearer arrows
    for conn in connections:
        plt.annotate('', 
                    xy=conn['end'], xycoords='data',
                    xytext=conn['start'], textcoords='data',
                    arrowprops=dict(arrowstyle='-|>', color=conn['color'], 
                                   lw=conn['width'], ls=conn['style'],
                                   connectionstyle='arc3,rad=0.1'),
                    zorder=0)
    
    # Add labels for main relationships
    relation_labels = [
        {'pos': (-5, 0.5), 'text': 'Transactions', 'color': '#8e44ad'},
        {'pos': (5, 0.5), 'text': 'Validation', 'color': '#16a085'},
        {'pos': (-1.5, 4), 'text': 'Data Flow', 'color': '#c0392b'},
        {'pos': (1.5, -4), 'text': 'User Interface', 'color': '#d35400'},
    ]
    
    for label in relation_labels:
        plt.text(label['pos'][0], label['pos'][1], label['text'], fontsize=10,
                color=label['color'], ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3', edgecolor=label['color']))
    
    # Add title
    plt.title('QTrust System Overview', fontsize=24, fontweight='bold', pad=20)
    
    # Define graph boundaries
    plt.xlim(-9, 9)
    plt.ylim(-7, 7)
    plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    
    # Save diagram to charts/architecture directory
    overview_path = get_chart_path('system_overview.png', 'architecture')
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    
    print(f"System overview created: {overview_path}")

def main():
    """Run all diagram generators."""
    ensure_chart_dirs()
    create_architecture_diagram()
    create_system_overview()
    print("All diagrams created successfully.")

if __name__ == "__main__":
    main() 