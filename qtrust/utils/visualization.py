"""
Blockchain Visualization Module

This module contains functions for visualizing blockchain networks, transaction flows,
shard performance, consensus protocol comparisons, and learning curves for DQN agents.
It provides comprehensive visualization tools to help understand and analyze the
behavior and performance of blockchain systems and their components.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional, Union
import os
import time
from .paths import get_chart_path  # Import the new get_chart_path function

def plot_blockchain_network(network: nx.Graph, 
                           shards: List[List[int]], 
                           trust_scores: Optional[Dict[int, float]] = None,
                           title: str = "Blockchain Network",
                           figsize: Tuple[int, int] = (12, 10),
                           save_path: Optional[str] = None):
    """
    Draw blockchain network with shards and nodes.
    
    Args:
        network: Blockchain network graph
        shards: List of shards and nodes in each shard
        trust_scores: Trust scores of nodes (if available)
        title: Title of the graph
        figsize: Figure size
        save_path: Path to save the graph
    """
    # Cấu hình style khoa học
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Cấu hình font toàn cục
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Create figure
    plt.figure(figsize=figsize, facecolor='white')
    
    # Define color palette for shards 
    # Use colorblind-friendly palette
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 0.9, len(shards)))
    
    # Define positions for nodes
    pos = nx.spring_layout(network, seed=42)
    
    # Draw nodes and edges
    for shard_id, shard_nodes in enumerate(shards):
        # Draw nodes in this shard
        node_size = 300
        if trust_scores is not None:
            # Node size based on trust score
            node_size = [trust_scores.get(node_id, 0.5) * 500 for node_id in shard_nodes]
        
        nx.draw_networkx_nodes(
            network, pos,
            nodelist=shard_nodes,
            node_color=[colors[shard_id]] * len(shard_nodes),
            node_size=node_size,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5,
            label=f"Shard {shard_id}"
        )
    
    # Draw intra-shard edges (darker)
    intra_shard_edges = []
    for shard_nodes in shards:
        for i, node1 in enumerate(shard_nodes):
            for node2 in shard_nodes[i+1:]:
                if network.has_edge(node1, node2):
                    intra_shard_edges.append((node1, node2))
    
    nx.draw_networkx_edges(
        network, pos,
        edgelist=intra_shard_edges,
        width=1.5,
        alpha=0.7,
        edge_color='#333333'
    )
    
    # Draw cross-shard edges (lighter, dashed)
    cross_shard_edges = []
    for edge in network.edges():
        if edge not in intra_shard_edges and (edge[1], edge[0]) not in intra_shard_edges:
            cross_shard_edges.append(edge)
    
    nx.draw_networkx_edges(
        network, pos,
        edgelist=cross_shard_edges,
        width=1.0,
        alpha=0.4,
        style='dashed',
        edge_color='#777777'
    )
    
    # Add labels for nodes
    nx.draw_networkx_labels(
        network, pos,
        font_size=9,
        font_family='serif',
        font_weight='bold',
        font_color='black'
    )
    
    # Configure plot
    plt.title(title, fontweight='bold', pad=15)
    
    # Create custom legend with larger markers
    legend = plt.legend(frameon=True, loc='upper right', 
                      title="Shard Assignments", 
                      title_fontsize=12,
                      markerscale=1.5)
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('gray')
    
    plt.axis('off')
    
    # Add descriptive footer
    plt.figtext(0.5, 0.01, 
               f"Network visualization of {len(network.nodes())} nodes distributed across {len(shards)} shards. "
               f"Intra-shard connections shown as solid lines; cross-shard connections as dashed lines.",
               ha='center', fontsize=9, fontstyle='italic')
    
    # Save with high resolution
    if save_path:
        # Use get_chart_path to get the full path
        full_path = get_chart_path(save_path, "visualization")
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Blockchain network plot saved to {full_path}")
    else:
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

def plot_transaction_flow(network: nx.Graph, 
                         transactions: List[Dict[str, Any]], 
                         paths: Dict[int, List[int]],
                         title: str = "Transaction Flow",
                         figsize: Tuple[int, int] = (12, 10),
                         save_path: Optional[str] = None):
    """
    Draw transaction flows on the blockchain network.
    
    Args:
        network: Blockchain network graph
        transactions: List of transactions
        paths: Dictionary mapping transaction ID to path
        title: Title of the graph
        figsize: Figure size
        save_path: Path to save the graph
    """
    # Cấu hình style khoa học
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Cấu hình font toàn cục
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Create figure
    plt.figure(figsize=figsize, facecolor='white')
    
    # Define positions for nodes
    pos = nx.spring_layout(network, seed=42)
    
    # Draw background network graph
    nx.draw_networkx_nodes(
        network, pos,
        node_size=250,
        node_color='#e6e6e6',
        edgecolors='#cccccc',
        linewidths=0.5,
        alpha=0.7
    )
    
    nx.draw_networkx_edges(
        network, pos,
        width=0.8,
        alpha=0.2,
        edge_color='#999999'
    )
    
    # Draw transaction flows
    # Use colorblind-friendly palette
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0, 0.8, min(10, len(transactions))))
    
    # Create dummy lines for legend
    legend_handles = []
    legend_labels = []
    
    # Track nodes that are part of transactions
    tx_nodes = set()
    
    # For better transparency in visualization
    shown_txs = 0
    max_txs_to_show = min(8, len(paths))  # Limit to avoid visual clutter
    
    for idx, (tx_id, path) in enumerate(paths.items()):
        if len(path) < 2 or shown_txs >= max_txs_to_show:
            continue
        
        # Find transaction information
        tx_info = next((tx for tx in transactions if tx.get('id') == tx_id), None)
        if not tx_info:
            continue
        
        # Create path
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        
        # Color for this transaction path
        color = colors[idx % len(colors)]
        
        # Draw path with curved arrows for clarity
        nx.draw_networkx_edges(
            network, pos,
            edgelist=edges,
            width=2.0,
            alpha=0.85,
            edge_color=[color] * len(edges),
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15,
            connectionstyle='arc3,rad=0.15'
        )
        
        # Add source and destination nodes to the tracking set
        tx_nodes.update([path[0], path[-1]])
        
        # Create a custom legend entry for this transaction
        from matplotlib.lines import Line2D
        legend_line = Line2D([0], [0], color=color, lw=2, marker='', 
                         label=f"Tx {tx_id[-6:]} (Path: {len(path)} hops)")
        legend_handles.append(legend_line)
        legend_labels.append(f"Tx {tx_id[-6:]} ({len(path)} hops)")
        
        shown_txs += 1
    
    # Redraw source and destination nodes with special colors
    sources = {path[0] for tx_id, path in paths.items() if len(path) >= 2}
    destinations = {path[-1] for tx_id, path in paths.items() if len(path) >= 2}
    
    # Draw source nodes
    if sources:
        nx.draw_networkx_nodes(
            network, pos,
            nodelist=list(sources),
            node_color='#2ca02c',  # Green
            node_size=300,
            alpha=0.9,
            edgecolors='black',
            linewidths=0.8
        )
        
    # Draw destination nodes
    if destinations:
        nx.draw_networkx_nodes(
            network, pos,
            nodelist=list(destinations),
            node_color='#d62728',  # Red
            node_size=300,
            alpha=0.9,
            edgecolors='black',
            linewidths=0.8
        )
    
    # Add node label only for transaction nodes for clarity
    node_labels = {node: str(node) for node in tx_nodes}
    nx.draw_networkx_labels(
        network, pos,
        labels=node_labels,
        font_size=9,
        font_family='serif',
        font_weight='bold',
        font_color='black'
    )
    
    # Configure plot
    plt.title(title, fontweight='bold', pad=15)
    
    # Add legend for transactions
    if legend_handles:
        # Add source and destination markers to legend
        from matplotlib.lines import Line2D
        legend_handles.append(Line2D([0], [0], marker='o', color='white', 
                            markerfacecolor='#2ca02c', markersize=10, 
                            label='Source Node'))
        legend_labels.append('Source Node')
        
        legend_handles.append(Line2D([0], [0], marker='o', color='white', 
                             markerfacecolor='#d62728', markersize=10, 
                             label='Destination Node'))
        legend_labels.append('Destination Node')
        
        plt.legend(handles=legend_handles, labels=legend_labels, 
                 loc='upper right', frameon=True, title="Transaction Paths")
    
    plt.axis('off')
    
    # Add descriptive footer
    plt.figtext(0.5, 0.01, 
               f"Visualization of {shown_txs} transaction paths through a {len(network.nodes())}-node network. "
               f"Green nodes represent transaction sources; red nodes represent destinations.",
               ha='center', fontsize=9, fontstyle='italic')
    
    # Save with high resolution
    if save_path:
        full_path = get_chart_path(save_path, "visualization")
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Transaction flow plot saved to {full_path}")
    else:
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

def plot_shard_graph(network: nx.Graph,
                    shards: List[List[int]],
                    shard_metrics: Dict[int, Dict[str, float]],
                    metric_name: str,
                    title: str = "Shard Performance",
                    figsize: Tuple[int, int] = (10, 8),
                    save_path: Optional[str] = None):
    """
    Draw blockchain network with nodes colored by shard performance metrics.
    
    Args:
        network: Blockchain network graph
        shards: List of shards and nodes in each shard
        shard_metrics: Dictionary of metrics for each shard
        metric_name: Name of the metric to visualize
        title: Title of the graph
        figsize: Figure size
        save_path: Path to save the graph
    """
    # Cấu hình style khoa học
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Cấu hình font toàn cục
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Create figure with 2x1 layout: network graph and bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                  gridspec_kw={'width_ratios': [2, 1]}, 
                                  facecolor='white')
    
    plt.sca(ax1)  # Set the left subplot as active
    
    # Collect metric values for all shards
    metric_values = [shard_metrics.get(shard_id, {}).get(metric_name, 0) 
                   for shard_id in range(len(shards))]
    
    # Create a normalized colormap for the metric values
    vmin = min(metric_values) if metric_values else 0
    vmax = max(metric_values) if metric_values else 1
    
    # Use a perceptually uniform colormap
    cmap = plt.cm.viridis
    
    # Define positions for nodes
    pos = nx.spring_layout(network, seed=42)
    
    # Draw the nodes for each shard, colored by their metric value
    for shard_id, shard_nodes in enumerate(shards):
        metric_value = shard_metrics.get(shard_id, {}).get(metric_name, 0)
        
        # Normalize the metric value to [0, 1] for color mapping
        normalized_value = (metric_value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        color = cmap(normalized_value)
        
        # Draw nodes in this shard
        nx.draw_networkx_nodes(
            network, pos,
            nodelist=shard_nodes,
            node_color=[color] * len(shard_nodes),
            node_size=300,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5,
            label=f"Shard {shard_id}: {metric_value:.2f}"
        )
    
    # Draw edges with reduced alpha for clarity
    nx.draw_networkx_edges(
        network, pos,
        width=1.0,
        alpha=0.3,
        edge_color='#555555'
    )
    
    # Add node labels
    nx.draw_networkx_labels(
        network, pos,
        font_size=8,
        font_family='serif',
        font_weight='bold'
    )
    
    ax1.set_title(f"Network Visualization: {metric_name} by Shard", fontweight='bold')
    ax1.axis('off')
    
    # Create a bar chart of the metrics in the right subplot
    plt.sca(ax2)  # Set the right subplot as active
    
    shard_ids = list(range(len(shards)))
    bars = ax2.bar(shard_ids, metric_values, color=[cmap(
        (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    ) for val in metric_values])
    
    # Add bar value labels
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02 * (vmax - vmin),
                f'{value:.2f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    
    ax2.set_title(f"Shard {metric_name} Comparison", fontweight='bold')
    ax2.set_xlabel("Shard ID")
    ax2.set_ylabel(metric_name.replace('_', ' ').title())
    ax2.set_xticks(shard_ids)
    ax2.set_xticklabels([f"Shard {i}" for i in shard_ids])
    ax2.set_ylim(vmin - 0.1 * (vmax - vmin), vmax + 0.15 * (vmax - vmin))
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)
    
    # Rotate x-axis labels if there are many shards
    if len(shard_ids) > 5:
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add a colorbar to show the mapping of colors to values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label(metric_name.replace('_', ' ').title())
    
    # Add descriptive footer
    plt.figtext(0.5, 0.01, 
               f"Visualization of {metric_name.replace('_', ' ').title()} across {len(shards)} shards with {len(network.nodes())} total nodes. "
               f"Color intensity indicates metric value, ranging from {vmin:.2f} to {vmax:.2f}.",
               ha='center', fontsize=9, fontstyle='italic')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Save with high resolution
    if save_path:
        full_path = get_chart_path(save_path, "visualization")
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Shard performance plot saved to {full_path}")
    else:
        plt.show()

def plot_consensus_comparison(consensus_results: Dict[str, Dict[str, List[float]]],
                             figsize: Tuple[int, int] = (15, 10),
                             save_path: Optional[str] = None):
    """
    Compare different consensus algorithms on multiple metrics.
    
    Args:
        consensus_results: Dictionary mapping algorithm names to metrics
        figsize: Figure size
        save_path: Path to save the graph
    """
    # Cấu hình style khoa học
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Cấu hình font toàn cục
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Extract algorithm names and metrics
    algorithms = list(consensus_results.keys())
    metrics = []
    
    # Find common metrics across all algorithms
    for algo_metrics in consensus_results.values():
        for metric in algo_metrics.keys():
            if metric not in metrics:
                metrics.append(metric)
    
    # Sắp xếp metrics cho thống nhất
    metrics = sorted(metrics)
    
    # Create a figure with a subplot for each metric
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, facecolor='white')
    
    # Flatten axes array for easy indexing
    if n_metrics > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Color palette
    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(len(algorithms))]
    
    # Plot each metric in its own subplot
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Collect data for this metric across all algorithms
        algo_values = []
        algo_names = []
        
        for j, algo in enumerate(algorithms):
            if metric in consensus_results[algo]:
                values = consensus_results[algo][metric]
                algo_values.append(values)
                algo_names.append(algo)
        
        # Calculate statistics for box plots
        if algo_values:
            # If values are lists (multiple runs), create box plots
            if all(isinstance(vals, list) and len(vals) > 1 for vals in algo_values):
                # Create box plot
                bp = ax.boxplot(algo_values, patch_artist=True, showfliers=False)
                
                # Customize box appearance
                for j, box in enumerate(bp['boxes']):
                    box.set(facecolor=colors[j], alpha=0.7)
                    box.set(edgecolor='black', linewidth=1.5)
                
                # Customize whiskers
                for j, whisker in enumerate(bp['whiskers']):
                    whisker.set(color='black', linewidth=1.5)
                
                # Customize caps
                for j, cap in enumerate(bp['caps']):
                    cap.set(color='black', linewidth=1.5)
                
                # Customize medians
                for j, median in enumerate(bp['medians']):
                    median.set(color='black', linewidth=2)
                
                # Add value annotations for medians
                for j, vals in enumerate(algo_values):
                    median_val = np.median(vals)
                    ax.text(j+1, median_val, f'{median_val:.3f}', 
                           ha='center', va='bottom', fontsize=9, 
                           fontweight='bold', color='black')
                
                ax.set_xticklabels(algo_names, rotation=30, ha='right')
            else:
                # If single values, create bar chart
                bar_values = [vals[0] if isinstance(vals, list) else vals for vals in algo_values]
                bars = ax.bar(algo_names, bar_values, color=colors[:len(algo_names)], 
                            alpha=0.7, edgecolor='black', linewidth=1)
                
                # Add value labels
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 * max(bar_values),
                           f'{height:.3f}', ha='center', va='bottom', fontsize=10,
                           fontweight='bold')
        
        # Set title and labels
        metric_title = metric.replace('_', ' ').title()
        ax.set_title(metric_title, fontweight='bold')
        ax.set_ylabel(metric_title)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Set y-axis to start from 0 for most metrics
        if not metric.lower().startswith('time') and all(np.array(values) >= 0 for values in algo_values):
            ax.set_ylim(bottom=0)
    
    # Hide any unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    # Add an overall title
    plt.suptitle('Consensus Algorithm Performance Comparison', fontsize=16, fontweight='bold')
    
    # Add descriptive footer
    plt.figtext(0.5, 0.01, 
               f"Comparative analysis of {len(algorithms)} consensus algorithms across {n_metrics} performance metrics. "
               f"Higher values are better for throughput and success rate; lower values are better for latency and energy consumption.",
               ha='center', fontsize=9, fontstyle='italic')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_path:
        full_path = get_chart_path(save_path, "visualization")
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Consensus comparison plot saved to {full_path}")
    else:
        plt.show()

def plot_learning_curve(rewards: List[float], 
                       avg_window: int = 20,
                       title: str = "Learning Curve",
                       figsize: Tuple[int, int] = (10, 6),
                       save_path: Optional[str] = None):
    """
    Plot learning curve for reinforcement learning agents.
    
    Args:
        rewards: List of rewards per episode
        avg_window: Window size for moving average
        title: Title of the graph
        figsize: Figure size
        save_path: Path to save the graph
    """
    # Cấu hình style khoa học
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Cấu hình font toàn cục
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Plot raw rewards as a scatter plot with reduced alpha for clarity
    episodes = np.arange(1, len(rewards) + 1)
    ax.scatter(episodes, rewards, s=10, alpha=0.3, color='#1f77b4', label='Episode Reward')
    
    # Calculate and plot moving average
    if len(rewards) >= avg_window:
        moving_avg = np.convolve(rewards, np.ones(avg_window)/avg_window, mode='valid')
        moving_avg_episodes = np.arange(avg_window, len(rewards) + 1)
        
        ax.plot(moving_avg_episodes, moving_avg, color='#ff7f0e', linewidth=2,
              label=f'{avg_window}-Episode Moving Average')
        
        # Add shaded area to represent standard deviation if there's enough data
        if len(rewards) >= 2 * avg_window:
            # Calculate moving standard deviation
            std_dev = []
            for i in range(len(moving_avg)):
                window_slice = rewards[i:i+avg_window]
                std_dev.append(np.std(window_slice))
            
            std_dev = np.array(std_dev)
            
            # Plot shaded area for standard deviation
            ax.fill_between(moving_avg_episodes, moving_avg - std_dev, moving_avg + std_dev,
                          color='#ff7f0e', alpha=0.2)
    
    # Calculate and display statistics
    if rewards:
        max_reward = max(rewards)
        min_reward = min(rewards)
        avg_reward = np.mean(rewards)
        
        # Draw horizontal lines for statistics
        ax.axhline(y=max_reward, color='#2ca02c', linestyle='--', alpha=0.7, 
                 label=f'Max: {max_reward:.1f}')
        ax.axhline(y=avg_reward, color='#9467bd', linestyle='--', alpha=0.7, 
                 label=f'Avg: {avg_reward:.1f}')
        
        # Add text annotations for statistics
        ax.text(len(rewards) * 0.02, max_reward, f'Max: {max_reward:.1f}', 
               verticalalignment='bottom', horizontalalignment='left',
               color='#2ca02c', fontweight='bold')
        
        ax.text(len(rewards) * 0.02, avg_reward, f'Avg: {avg_reward:.1f}', 
               verticalalignment='bottom', horizontalalignment='left',
               color='#9467bd', fontweight='bold')
        
        # Add final performance annotation
        final_avg = np.mean(rewards[-avg_window:]) if len(rewards) >= avg_window else rewards[-1]
        ax.text(len(rewards) * 0.98, final_avg, f'Final: {final_avg:.1f}', 
               verticalalignment='bottom', horizontalalignment='right',
               color='#d62728', fontweight='bold')
    
    # Set title and labels
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper left', frameon=True)
    
    # Add descriptive footer
    if rewards:
        plt.figtext(0.5, 0.01, 
                  f"Learning curve over {len(rewards)} episodes. "
                  f"Initial reward: {rewards[0]:.1f}, Final reward: {rewards[-1]:.1f}, "
                  f"Improvement: {(rewards[-1] - rewards[0]):.1f} ({(rewards[-1] - rewards[0]) / max(abs(rewards[0]), 1) * 100:.1f}%).",
                  ha='center', fontsize=9, fontstyle='italic')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_path:
        full_path = get_chart_path(save_path, "visualization")
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve plot saved to {full_path}")
    else:
        plt.show()

def plot_performance_heatmap(data: Dict[str, Dict[str, float]],
                           metric: str,
                           title: str = "Performance Heatmap",
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None):
    """
    Create a heatmap visualization of performance metrics.
    
    Args:
        data: Dictionary of dictionaries with performance data
        metric: Name of the metric to visualize
        title: Title of the graph
        figsize: Figure size
        save_path: Path to save the graph
    """
    # Cấu hình style khoa học
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Cấu hình font toàn cục
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Extract parameters and values
    param1_values = sorted(list(data.keys()))
    param2_values = sorted(list(next(iter(data.values())).keys()))
    
    # Create data matrix
    matrix = np.zeros((len(param1_values), len(param2_values)))
    
    for i, p1 in enumerate(param1_values):
        for j, p2 in enumerate(param2_values):
            matrix[i, j] = data[p1][p2]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Create heatmap with perceptually uniform colormap
    cmap = plt.cm.viridis
    im = ax.imshow(matrix, cmap=cmap)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(param2_values)))
    ax.set_yticks(np.arange(len(param1_values)))
    ax.set_xticklabels(param2_values)
    ax.set_yticklabels(param1_values)
    
    # Rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add value annotations
    for i in range(len(param1_values)):
        for j in range(len(param2_values)):
            value = matrix[i, j]
            # Choose text color based on background darkness
            # for better contrast and readability
            color = "white" if value > np.mean(matrix) else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                  color=color, fontweight="bold")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric.replace('_', ' ').title(), rotation=-90, va="bottom")
    
    # Set title and labels
    ax.set_title(f"{title}: {metric.replace('_', ' ').title()}", fontweight='bold', pad=15)
    ax.set_xlabel("Parameter 2")
    ax.set_ylabel("Parameter 1")
    
    # Add descriptive footer
    plt.figtext(0.5, 0.01, 
               f"Performance heatmap showing {metric.replace('_', ' ').title()} across different parameter combinations. "
               f"Values range from {np.min(matrix):.2f} to {np.max(matrix):.2f}, with an average of {np.mean(matrix):.2f}.",
               ha='center', fontsize=9, fontstyle='italic')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_path:
        full_path = get_chart_path(save_path, "visualization")
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Performance heatmap saved to {full_path}")
    else:
        plt.show() 