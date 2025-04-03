#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Blockchain Comparison Utilities

This module provides utility functions for comparing the performance of different
blockchain systems. It includes functions to create heatmap charts, scatter plots,
comparison tables and reports for benchmarking blockchain systems based on metrics
such as throughput, latency, security, energy efficiency, and decentralization.
"""

import os
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Any

def plot_heatmap_comparison(systems: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """
    Create a heatmap chart comparing blockchain systems.
    
    Args:
        systems: Dict containing blockchain systems data
        output_dir: Output directory to save the chart
    """
    # Convert data to DataFrame for easier processing
    df = pd.DataFrame(systems).T
    
    # Normalize data for fair comparison
    normalized_df = df.copy()
    for col in df.columns:
        if col in ['latency', 'energy']:
            # For metrics where lower values are better
            # Use 1 - normalized to reverse (higher value = better)
            normalized_df[col] = 1 - ((df[col] - df[col].min()) / 
                                    (df[col].max() - df[col].min()) 
                                    if df[col].max() != df[col].min() else 0)
        else:
            # For metrics where higher values are better
            normalized_df[col] = ((df[col] - df[col].min()) / 
                                (df[col].max() - df[col].min())
                                if df[col].max() != df[col].min() else 1.0)
    
    # Create a large figure
    plt.figure(figsize=(14, 10))
    
    # Create a colormap - from red (low) to green (high)
    cmap = LinearSegmentedColormap.from_list('green_red', ['#d73027', '#f46d43', '#fdae61', 
                                                         '#fee08b', '#d9ef8b', '#a6d96a', 
                                                         '#66bd63', '#1a9850'])
    
    # Draw heatmap
    ax = sns.heatmap(normalized_df, annot=df.round(2), fmt=".2f", 
                   cmap=cmap, linewidths=0.5, 
                   cbar_kws={'label': 'Score (normalized)'})
    
    # Customize
    plt.title('Blockchain Systems Comparison Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Performance Metrics', fontsize=12)
    plt.ylabel('Systems', fontsize=12)
    
    # Rename columns for better appearance
    ax.set_xticklabels([col.replace('_', ' ').title() for col in df.columns], rotation=45, ha='right')
    
    # Save chart
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"blockchain_heatmap_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save normalized DataFrame for reference
    normalized_df.to_csv(os.path.join(output_dir, f"normalized_metrics_{timestamp}.csv"))

def plot_relationship_comparison(systems: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """
    Create a scatter plot showing the relationship between throughput, latency and security.
    
    Args:
        systems: Dict containing blockchain systems data
        output_dir: Output directory to save the chart
    """
    # Create DataFrame
    data = []
    for system, values in systems.items():
        data.append({
            'System': system,
            'Throughput': values['throughput'],
            'Latency': values['latency'],
            'Security': values['security'],
            'Energy': values['energy'],
            'Decentralization': values['decentralization']
        })
    
    df = pd.DataFrame(data)
    
    # Create scatter plot
    plt.figure(figsize=(14, 10))
    
    # Use point size based on security score
    sizes = df['Security'] * 1000
    
    # Color based on decentralization level
    colors = df['Decentralization']
    
    # If throughput has too large a difference, use logarithmic scale
    if max(df['Throughput']) / min(df['Throughput']) > 100:
        scatter = plt.scatter(np.log10(df['Throughput']), df['Latency'], 
                           s=sizes, c=colors, cmap='viridis', alpha=0.7, edgecolors='k')
        plt.xlabel('Throughput (log10 tx/s)', fontsize=12)
    else:
        scatter = plt.scatter(df['Throughput'], df['Latency'], 
                           s=sizes, c=colors, cmap='viridis', alpha=0.7, edgecolors='k')
        plt.xlabel('Throughput (tx/s)', fontsize=12)
    
    # Similar for latency
    if max(df['Latency']) / min(df['Latency']) > 100:
        plt.yscale('log')
        plt.ylabel('Latency (log ms)', fontsize=12)
    else:
        plt.ylabel('Latency (ms)', fontsize=12)
    
    # Add label for each point
    for i, system in enumerate(df['System']):
        plt.annotate(system, 
                   (np.log10(df['Throughput'][i]) if max(df['Throughput']) / min(df['Throughput']) > 100 
                    else df['Throughput'][i], 
                    df['Latency'][i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Decentralization Score', fontsize=10)
    
    # Add size legend
    plt.annotate('Bubble size represents Security Score', xy=(0.05, 0.95),
               xycoords='axes fraction', fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title('Relationship between Throughput, Latency, and Security', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save chart
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"blockchain_relationship_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_table(systems: Dict[str, Dict[str, float]], output_dir: str) -> pd.DataFrame:
    """
    Create a comparison table of blockchain systems and calculate composite scores.
    
    Args:
        systems: Dict containing blockchain systems data
        output_dir: Output directory to save the table
        
    Returns:
        DataFrame containing the comparison table with scores
    """
    # Create DataFrame
    df = pd.DataFrame(systems).T
    
    # Rename columns
    df.columns = [col.replace('_', ' ').title() for col in df.columns]
    
    # Add Performance Score column with weights for each metric
    df['Performance Score'] = (
        # Throughput: 25%
        df['Throughput'] / df['Throughput'].max() * 0.25 + 
        # Latency: 25% (lower values are better)
        (1 - df['Latency'] / df['Latency'].max()) * 0.25 + 
        # Security: 20%
        df['Security'] * 0.20 + 
        # Energy: 10% (lower values are better)
        (1 - df['Energy'] / df['Energy'].max()) * 0.10 +
        # Scalability: 10%
        df['Scalability'] * 0.10 +
        # Cross-shard efficiency: 5%
        df['Cross Shard Efficiency'] * 0.05 +
        # Attack resistance: 5%
        df['Attack Resistance'] * 0.05
    )
    
    # Calculate overall rank
    df['Rank'] = df['Performance Score'].rank(ascending=False).astype(int)
    
    # Sort by Performance Score
    df = df.sort_values('Performance Score', ascending=False)
    
    # Round values
    df = df.round(3)
    
    # Save table as CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"blockchain_comparison_table_{timestamp}.csv")
    df.to_csv(csv_path)
    
    # Create HTML with nice formatting
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Blockchain Systems Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #ddd; }}
            .highlight {{ background-color: #e6ffe6; }}
            .top-score {{ font-weight: bold; color: #006600; }}
            caption {{ font-size: 1.5em; margin-bottom: 10px; }}
            .footer {{ font-size: 0.8em; margin-top: 20px; color: #666; }}
        </style>
    </head>
    <body>
        <h1>Blockchain Systems Performance Comparison</h1>
        <p>Created on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        {df.to_html(classes='highlight')}
        <p class="footer">
            Note: Performance Score is calculated based on metric weights:<br>
            Throughput (25%), Latency (25%), Security (20%), Energy (10%), 
            Scalability (10%), Cross-shard Efficiency (5%), Attack Resistance (5%)
        </p>
    </body>
    </html>
    """
    
    # Save HTML
    html_path = os.path.join(output_dir, f"blockchain_comparison_table_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(styled_html)
    
    return df

def run_all_comparisons(output_dir: Optional[str] = None) -> None:
    """
    Run all comparison analyses and generate a comprehensive report.
    
    Args:
        output_dir: Output directory to save the report
    """
    import importlib
    blockchain_comp = importlib.import_module('qtrust.benchmark.blockchain_comparison')
    
    # Generate full comparison report
    blockchain_comp.generate_comparison_report(output_dir)
    
    # Create completion message
    print("All blockchain comparison analyses completed!")

if __name__ == "__main__":
    # Run all analyses when directly called
    run_all_comparisons() 