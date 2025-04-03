#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Attack Comparison Analysis Tool for QTrust

This script performs comprehensive analysis of QTrust's performance under various attack scenarios.
It generates multiple visualization formats including radar charts, bar charts, and security-performance
trade-off plots, as well as summary tables for comparison. The tool supports customizable simulation 
parameters and can focus on specific attack subsets for targeted analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add current directory to PYTHONPATH to import other modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from attack_simulation_runner import run_attack_comparison
from large_scale_simulation import LargeScaleBlockchainSimulation

def plot_comparison_radar(metrics_data, output_dir='results_comparison'):
    """Draw a radar chart comparing performance across different attack types."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics for comparison
    metrics = ['throughput', 'latency', 'energy', 'security', 'cross_shard_ratio']
    
    # Normalize data for radar chart comparison
    normalized_data = {}
    
    # Find min/max values for each metric
    min_values = {m: float('inf') for m in metrics}
    max_values = {m: float('-inf') for m in metrics}
    
    for attack, values in metrics_data.items():
        for m in metrics:
            if values[m] < min_values[m]:
                min_values[m] = values[m]
            if values[m] > max_values[m]:
                max_values[m] = values[m]
    
    # Normalize
    for attack, values in metrics_data.items():
        normalized_data[attack] = {}
        for m in metrics:
            # For latency and energy, lower values are better, so reverse
            if m in ['latency', 'energy']:
                if max_values[m] == min_values[m]:
                    normalized_data[attack][m] = 1.0
                else:
                    normalized_data[attack][m] = 1 - (values[m] - min_values[m]) / (max_values[m] - min_values[m])
            else:
                if max_values[m] == min_values[m]:
                    normalized_data[attack][m] = 1.0
                else:
                    normalized_data[attack][m] = (values[m] - min_values[m]) / (max_values[m] - min_values[m])
    
    # Number of variables
    N = len(metrics)
    
    # Angle for each axis
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Metric names
    labels = [m.replace('_', ' ').title() for m in metrics]
    labels += labels[:1]  # Close the circle
    
    # Colors and styles for each attack type
    colors = sns.color_palette("Set1", len(normalized_data))
    
    # Create a larger figure
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, polar=True)
    
    # Plot for each attack type
    for i, (attack, values) in enumerate(normalized_data.items()):
        # Prepare data for plotting
        attack_values = [values[m] for m in metrics]
        attack_values += attack_values[:1]  # Close the circle
        
        # Plot line and fill color
        ax.plot(angles, attack_values, linewidth=2, linestyle='solid', 
                label=attack.replace('_', ' ').title(), color=colors[i])
        ax.fill(angles, attack_values, alpha=0.1, color=colors[i])
    
    # Customize radar chart
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)  # Go clockwise
    
    # Set labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=12)
    
    # Add fade levels
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add title and legend
    plt.title('QTrust Performance Comparison Across Attack Scenarios', fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/attack_comparison_radar_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison_bars(metrics_data, output_dir='results_comparison'):
    """Draw bar charts comparing performance metrics across attack types."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert data for easier plotting with seaborn
    df_data = []
    for attack, values in metrics_data.items():
        for metric, value in values.items():
            df_data.append({
                'Attack': attack.replace('_', ' ').title(),
                'Metric': metric.replace('_', ' ').title(),
                'Value': value
            })
    
    df = pd.DataFrame(df_data)
    
    # Create large figure
    plt.figure(figsize=(15, 12))
    
    # Plot for each metric
    metrics = ['Throughput', 'Latency', 'Energy', 'Security', 'Cross Shard Ratio']
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i+1)
        metric_data = df[df['Metric'] == metric]
        
        # Sort attacks by value
        if metric in ['Latency', 'Energy']:  # Lower values are better
            metric_data = metric_data.sort_values('Value')
        else:  # Higher values are better
            metric_data = metric_data.sort_values('Value', ascending=False)
        
        # Plot bar chart
        sns.barplot(data=metric_data, x='Attack', y='Value', 
                    palette='viridis', alpha=0.8)
        
        # Add annotations
        for j, row in enumerate(metric_data.itertuples()):
            plt.text(j, row.Value + (max(metric_data['Value']) * 0.02), 
                     f"{row.Value:.2f}", ha='center', fontsize=9)
        
        # Customize axes and title
        plt.title(f"{metric} Comparison", fontsize=13, fontweight='bold')
        plt.xlabel('')
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/attack_comparison_bars_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_security_vs_performance(metrics_data, output_dir='results_comparison'):
    """Draw a chart comparing security and performance trade-offs."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Prepare data for scatter plot
    attacks = []
    throughputs = []
    latencies = []
    securities = []
    
    for attack, values in metrics_data.items():
        attacks.append(attack.replace('_', ' ').title())
        throughputs.append(values['throughput'])
        latencies.append(values['latency'])
        securities.append(values['security'])
    
    # Colors based on latency (lower = better)
    norm_latencies = [l/max(latencies) for l in latencies]
    colors = plt.cm.cool(norm_latencies)
    
    # Size based on throughput (higher = better)
    norm_throughputs = [t/max(throughputs) * 500 for t in throughputs]
    
    # Draw scatter plot
    plt.scatter(securities, latencies, s=norm_throughputs, c=colors, alpha=0.6)
    
    # Add annotations
    for i, attack in enumerate(attacks):
        plt.annotate(attack, (securities[i], latencies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Customize chart
    plt.title('Security vs. Latency Trade-off Across Attack Scenarios', fontsize=14, fontweight='bold')
    plt.xlabel('Security Score', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add size legend
    plt.annotate('Bubble size represents throughput', xy=(0.05, 0.95), 
                xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                                                fc="white", ec="gray", alpha=0.8))
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/security_vs_performance_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table(metrics_data, output_dir='results_comparison'):
    """Create a summary table comparing attack types."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame from metrics data
    df = pd.DataFrame(metrics_data).T
    
    # Rename columns for readability
    df.columns = [c.replace('_', ' ').title() for c in df.columns]
    
    # Rename index (attack names)
    df.index = [idx.replace('_', ' ').title() for idx in df.index]
    
    # Create Performance Score column by calculating composite score
    # Formula: increase throughput + security, decrease latency + energy
    df['Performance Score'] = (
        df['Throughput'] / df['Throughput'].max() * 0.3 + 
        (1 - df['Latency'] / df['Latency'].max()) * 0.3 + 
        (1 - df['Energy'] / df['Energy'].max()) * 0.1 + 
        df['Security'] / df['Security'].max() * 0.3
    )
    
    # Sort by Performance Score
    df = df.sort_values('Performance Score', ascending=False)
    
    # Round values
    df = df.round(3)
    
    # Save table as HTML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create style for HTML table
    html = df.to_html()
    
    # Add CSS
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attack Comparison Results</title>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                font-family: Arial, sans-serif;
            }}
            th, td {{
                text-align: left;
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #ddd;
            }}
            caption {{
                font-size: 1.5em;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>QTrust Performance Under Different Attack Scenarios</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        {html}
    </body>
    </html>
    """
    
    # Save HTML
    with open(f"{output_dir}/attack_comparison_table_{timestamp}.html", "w") as f:
        f.write(styled_html)
    
    # Save CSV
    df.to_csv(f"{output_dir}/attack_comparison_table_{timestamp}.csv")
    
    print(f"Summary table has been saved in {output_dir}")
    
    return df

def run_comprehensive_analysis(args):
    """Run comprehensive analysis comparing different attack types."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Starting comprehensive analysis of QTrust under various attack types ===")
    
    # Run attack comparison
    all_metrics = run_attack_comparison(args, output_dir)
    
    # Save metrics data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{output_dir}/attack_metrics_{timestamp}.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    
    # Draw comparison charts
    plot_comparison_radar(all_metrics, output_dir)
    plot_comparison_bars(all_metrics, output_dir)
    plot_security_vs_performance(all_metrics, output_dir)
    
    # Create summary table
    summary_df = generate_summary_table(all_metrics, output_dir)
    
    # Print summary results
    print("\n=== SUMMARY RESULTS ===")
    print(summary_df)
    
    print(f"\nAnalysis completed. Results saved in directory: {output_dir}")
    
    return summary_df

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='QTrust Attack Comparison Analysis')
    
    parser.add_argument('--num-shards', type=int, default=4, help='Number of shards')
    parser.add_argument('--nodes-per-shard', type=int, default=10, help='Number of nodes per shard')
    parser.add_argument('--steps', type=int, default=500, help='Number of simulation steps for each attack type')
    parser.add_argument('--tx-per-step', type=int, default=20, help='Number of transactions per step')
    parser.add_argument('--malicious', type=float, default=20, help='Percentage of malicious nodes (default)')
    parser.add_argument('--output-dir', type=str, default='results_comparison', help='Directory to save results')
    parser.add_argument('--attack-subset', type=str, nargs='+', 
                      choices=['none', '51_percent', 'sybil', 'eclipse', 'selfish_mining', 
                              'bribery', 'ddos', 'finney', 'mixed'], 
                      default=None, 
                      help='Optional subset of attacks to analyze')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_comprehensive_analysis(args) 