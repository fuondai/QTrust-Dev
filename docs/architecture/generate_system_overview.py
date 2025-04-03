#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QTrust System Overview Generator

This module creates a comprehensive visual representation of the QTrust blockchain system.
It generates a detailed diagram illustrating key components of the QTrust architecture,
including the blockchain environment, sharding system, DRL agents, and federated learning
components. The diagram uses custom icons to represent each subsystem and their
interconnections.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

def create_blockchain_icon(ax, x, y, width, height, color='#3498DB'):
    """Create a blockchain icon representation."""
    block_height = height / 5
    for i in range(5):
        block = patches.Rectangle(
            (x, y + i * block_height), 
            width, 
            block_height * 0.9,
            linewidth=1,
            edgecolor='black',
            facecolor=color,
            alpha=0.7
        )
        ax.add_patch(block)
        
        # Add connections between blocks
        if i > 0:
            arrow = patches.FancyArrowPatch(
                (x + width/2, y + (i-1) * block_height + block_height/2),
                (x + width/2, y + i * block_height + block_height/2),
                arrowstyle='-|>',
                mutation_scale=15,
                linewidth=1.5,
                color='black'
            )
            ax.add_patch(arrow)

def create_shard_icon(ax, x, y, width, height, color='#2ECC71', num_shards=4):
    """Create a shard icon representation."""
    shard_width = width / num_shards
    for i in range(num_shards):
        shard = patches.Rectangle(
            (x + i * shard_width, y), 
            shard_width * 0.9, 
            height,
            linewidth=1,
            edgecolor='black',
            facecolor=color,
            alpha=0.7
        )
        ax.add_patch(shard)

def create_agent_icon(ax, x, y, width, height, color='#E74C3C'):
    """Create an agent icon representation."""
    # Draw agent representation (robot/AI)
    head = patches.Circle(
        (x + width/2, y + height * 0.7),
        radius=height * 0.25,
        linewidth=1,
        edgecolor='black',
        facecolor=color,
        alpha=0.8
    )
    
    body = patches.Rectangle(
        (x + width * 0.3, y + height * 0.2),
        width * 0.4,
        height * 0.4,
        linewidth=1,
        edgecolor='black',
        facecolor=color,
        alpha=0.8
    )
    
    # Arms
    arm1 = patches.Rectangle(
        (x + width * 0.1, y + height * 0.4),
        width * 0.2,
        height * 0.1,
        linewidth=1,
        edgecolor='black',
        facecolor=color,
        alpha=0.8
    )
    
    arm2 = patches.Rectangle(
        (x + width * 0.7, y + height * 0.4),
        width * 0.2,
        height * 0.1,
        linewidth=1,
        edgecolor='black',
        facecolor=color,
        alpha=0.8
    )
    
    # Legs
    leg1 = patches.Rectangle(
        (x + width * 0.35, y),
        width * 0.1,
        height * 0.2,
        linewidth=1,
        edgecolor='black',
        facecolor=color,
        alpha=0.8
    )
    
    leg2 = patches.Rectangle(
        (x + width * 0.55, y),
        width * 0.1,
        height * 0.2,
        linewidth=1,
        edgecolor='black',
        facecolor=color,
        alpha=0.8
    )
    
    ax.add_patch(head)
    ax.add_patch(body)
    ax.add_patch(arm1)
    ax.add_patch(arm2)
    ax.add_patch(leg1)
    ax.add_patch(leg2)

def create_neural_network_icon(ax, x, y, width, height, color='#9B59B6'):
    """Create a neural network icon representation."""
    # Number of nodes and layers
    n_layers = 3
    nodes_per_layer = [4, 6, 4]
    
    # Node size
    node_radius = min(width / (2 * max(nodes_per_layer)), height / (2 * max(nodes_per_layer) * n_layers)) * 0.8
    
    # Draw nodes
    node_positions = []
    for l in range(n_layers):
        layer_nodes = []
        layer_width = width
        layer_x = x
        layer_y = y + height * (l / (n_layers - 1))
        
        for n in range(nodes_per_layer[l]):
            node_x = layer_x + layer_width * ((n + 1) / (nodes_per_layer[l] + 1))
            node_y = layer_y
            
            node = patches.Circle(
                (node_x, node_y),
                radius=node_radius,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.8
            )
            ax.add_patch(node)
            layer_nodes.append((node_x, node_y))
        
        node_positions.append(layer_nodes)
    
    # Draw connections
    for l in range(n_layers - 1):
        for node1 in node_positions[l]:
            for node2 in node_positions[l + 1]:
                line = plt.Line2D(
                    [node1[0], node2[0]],
                    [node1[1], node2[1]],
                    linewidth=0.5,
                    color='gray',
                    alpha=0.6
                )
                ax.add_line(line)

def create_system_overview():
    """Create a system overview diagram for QTrust."""
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Set axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    
    # Turn off axes
    ax.set_axis_off()
    
    # Title
    ax.set_title('QTrust System Overview', fontsize=24, fontweight='bold', pad=20)
    
    # Draw main frame
    main_frame = patches.Rectangle(
        (0.5, 0.5), 9, 6,
        linewidth=2,
        edgecolor='black',
        facecolor='none',
        alpha=0.8
    )
    ax.add_patch(main_frame)
    
    # Draw main components
    # 1. Blockchain Environment
    env_box = patches.Rectangle(
        (1, 4), 3, 2,
        linewidth=1.5,
        edgecolor='black',
        facecolor='#F9EBEA',
        alpha=0.7
    )
    ax.add_patch(env_box)
    ax.text(2.5, 6.2, 'Blockchain Environment', fontsize=14, fontweight='bold', ha='center')
    create_blockchain_icon(ax, 1.5, 4.2, 2, 1.5, '#5DADE2')
    
    # 2. Sharding System
    shard_box = patches.Rectangle(
        (6, 4), 3, 2,
        linewidth=1.5,
        edgecolor='black',
        facecolor='#E8F8F5',
        alpha=0.7
    )
    ax.add_patch(shard_box)
    ax.text(7.5, 6.2, 'Sharding System', fontsize=14, fontweight='bold', ha='center')
    create_shard_icon(ax, 6.5, 4.2, 2, 1.5, '#58D68D')
    
    # 3. DRL Agent
    agent_box = patches.Rectangle(
        (1, 1), 3, 2,
        linewidth=1.5,
        edgecolor='black',
        facecolor='#FDEDEC',
        alpha=0.7
    )
    ax.add_patch(agent_box)
    ax.text(2.5, 3.2, 'DRL Agent', fontsize=14, fontweight='bold', ha='center')
    create_agent_icon(ax, 1.5, 1.2, 2, 1.5, '#E74C3C')
    
    # 4. Federated Learning
    fl_box = patches.Rectangle(
        (6, 1), 3, 2,
        linewidth=1.5,
        edgecolor='black',
        facecolor='#F4ECF7',
        alpha=0.7
    )
    ax.add_patch(fl_box)
    ax.text(7.5, 3.2, 'Federated Learning', fontsize=14, fontweight='bold', ha='center')
    create_neural_network_icon(ax, 6.5, 1.2, 2, 1.5, '#9B59B6')
    
    # Draw connecting arrows
    # Blockchain -> DRL
    arrow1 = patches.FancyArrowPatch(
        (2.5, 4),
        (2.5, 3),
        arrowstyle='<|-|>',
        mutation_scale=20,
        linewidth=2,
        color='#5D6D7E'
    )
    ax.add_patch(arrow1)
    ax.text(2.15, 3.5, 'State/Action', fontsize=10, rotation=90)
    
    # DRL -> Sharding
    arrow2 = patches.FancyArrowPatch(
        (4, 2),
        (6, 5),
        connectionstyle="arc3,rad=0.3",
        arrowstyle='<|-|>',
        mutation_scale=20,
        linewidth=2,
        color='#5D6D7E'
    )
    ax.add_patch(arrow2)
    ax.text(4.8, 3.3, 'Optimize', fontsize=10, rotation=45)
    
    # Sharding -> Blockchain
    arrow3 = patches.FancyArrowPatch(
        (6, 5),
        (4, 5),
        arrowstyle='<|-|>',
        mutation_scale=20,
        linewidth=2,
        color='#5D6D7E'
    )
    ax.add_patch(arrow3)
    ax.text(5, 5.2, 'Transactions', fontsize=10)
    
    # DRL <-> Federated Learning
    arrow4 = patches.FancyArrowPatch(
        (4, 2),
        (6, 2),
        arrowstyle='<|-|>',
        mutation_scale=20,
        linewidth=2,
        color='#5D6D7E'
    )
    ax.add_patch(arrow4)
    ax.text(5, 2.2, 'Models', fontsize=10)
    
    # Add annotations
    ax.text(0.5, 0.2, 'QTrust integrates DRL with Federated Learning to optimize blockchain sharding',
           fontsize=12, ha='left')
    
    # Save image
    output_dir = '../exported_charts'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(f'{output_dir}/system_overview.png', dpi=300, bbox_inches='tight')
    print(f"System overview diagram created at: {output_dir}/system_overview.png")
    plt.close()

if __name__ == "__main__":
    create_system_overview() 