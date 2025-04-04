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
    
    # Set scientific style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Configure global font settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
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
    
    # Scientific color palette for each attack type
    colors = plt.cm.tab10(np.linspace(0, 1, len(normalized_data)))
    
    # Create a figure with white background
    plt.figure(figsize=(12, 10), facecolor='white')
    ax = plt.subplot(111, polar=True)
    
    # Plot for each attack type
    for i, (attack, values) in enumerate(normalized_data.items()):
        # Prepare data for plotting
        attack_values = [values[m] for m in metrics]
        attack_values += attack_values[:1]  # Close the circle
        
        # Plot line and fill color with enhanced styling
        ax.plot(angles, attack_values, linewidth=2.5, linestyle='solid', 
                label=attack.replace('_', ' ').title(), color=colors[i])
        ax.fill(angles, attack_values, alpha=0.2, color=colors[i])
    
    # Customize radar chart
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)  # Go clockwise
    
    # Set labels for each axis with enhanced styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=12, fontweight='bold')
    
    # Add fade levels with enhanced styling
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.set_ylim(0, 1)
    
    # Add grid with enhanced styling
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add title and legend with enhanced positioning and styling
    plt.title('QTrust Performance Comparison Across Attack Scenarios', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Better legend positioning and styling
    legend = plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1),
                       framealpha=0.9, edgecolor='gray')
    
    # Add background color to every other metric for better readability
    for i in range(N):
        if i % 2 == 0:
            angle = angles[i]
            ax.fill_between([angle, angles[i+1]], 0, 1, 
                           color='gray', alpha=0.05)
    
    # Add metric explanations
    plt.figtext(0.5, 0.01, 
               "Normalized metrics where 1.0 represents the best performance across attack types.\n"
               "For latency and energy consumption, lower original values are normalized to higher values.",
               ha='center', fontsize=10, fontstyle='italic')
    
    # Apply tight layout and save with high quality
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/attack_comparison_radar_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison_bars(metrics_data, output_dir='results_comparison'):
    """Draw bar charts comparing performance metrics across attack types."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set scientific style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Configure global font settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
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
    
    # Create large figure with white background
    plt.figure(figsize=(15, 12), facecolor='white')
    
    # Plot for each metric
    metrics = ['Throughput', 'Latency', 'Energy', 'Security', 'Cross Shard Ratio']
    
    # Scientific color palettes for each metric
    palettes = {
        'Throughput': 'viridis',
        'Latency': 'magma',
        'Energy': 'cividis', 
        'Security': 'plasma',
        'Cross Shard Ratio': 'inferno'
    }
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i+1)
        metric_data = df[df['Metric'] == metric]
        
        # Sort attacks by value
        if metric in ['Latency', 'Energy']:  # Lower values are better
            metric_data = metric_data.sort_values('Value')
        else:  # Higher values are better
            metric_data = metric_data.sort_values('Value', ascending=False)
        
        # Plot bar chart with enhanced styling
        ax = sns.barplot(data=metric_data, x='Attack', y='Value', 
                    palette=palettes[metric], alpha=0.85, edgecolor='black', linewidth=1)
        
        # Add value annotations on bars
        for j, row in enumerate(metric_data.itertuples()):
            plt.text(j, row.Value + (max(metric_data['Value']) * 0.02), 
                     f"{row.Value:.2f}", ha='center', fontsize=9, fontweight='bold')
        
        # Customize axes and title
        plt.title(f"{metric} Comparison", fontsize=14, fontweight='bold', pad=10)
        plt.xlabel('Attack Type', fontweight='bold')
        plt.ylabel(f"{metric} Value", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Enhanced grid styling
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)  # Put grid behind bars
        
        # Add band for context if applicable
        if metric == 'Security':
            plt.axhspan(0.8, 1.0, alpha=0.1, color='green', label='High Security')
            plt.axhspan(0.6, 0.8, alpha=0.1, color='yellow', label='Medium Security')
            plt.axhspan(0.0, 0.6, alpha=0.1, color='red', label='Low Security')
            plt.legend(loc='lower right')
    
    # Add overall title
    plt.suptitle('QTrust Performance Metrics Under Different Attack Scenarios', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add figure description
    plt.figtext(0.5, 0.01, 
               "Performance comparison across multiple metrics. For Latency and Energy, lower values indicate better performance.\n"
               "For Throughput, Security, and Cross-Shard Ratio, higher values indicate better performance.",
               ha='center', fontsize=10, fontstyle='italic')
    
    # Adjust layout and save with high quality
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/attack_comparison_bars_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_security_vs_performance(metrics_data, output_dir='results_comparison'):
    """Draw a chart comparing security and performance trade-offs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set scientific style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Configure global font settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Create figure with white background
    plt.figure(figsize=(12, 9), facecolor='white')
    
    # Prepare data for scatter plot
    attacks = []
    throughputs = []
    latencies = []
    securities = []
    energies = []
    
    for attack, values in metrics_data.items():
        attacks.append(attack.replace('_', ' ').title())
        throughputs.append(values['throughput'])
        latencies.append(values['latency'])
        securities.append(values['security'])
        energies.append(values['energy'])
    
    # Normalize for sizing
    norm_energies = [1.0 - (e/max(energies)) for e in energies]  # Invert so lower energy = larger point
    sizes = [100 + 400 * ne for ne in norm_energies]  # Scale for visibility
    
    # Colors based on latency (lower = better)
    norm_latencies = [1.0 - (l/max(latencies)) for l in latencies]  # Invert so lower latency = better color
    colors = plt.cm.viridis(norm_latencies)
    
    # Create scatter plot with enhanced styling
    scatter = plt.scatter(throughputs, securities, s=sizes, c=colors, alpha=0.7, 
                        edgecolor='black', linewidth=1)
    
    # Add attack labels with adjusted positions
    for i, attack in enumerate(attacks):
        plt.annotate(attack, 
                   (throughputs[i], securities[i]),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    # Add grid with enhanced styling
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_axisbelow(True)  # Put grid behind points
    
    # Add performance quadrants with subtle background shading
    avg_throughput = sum(throughputs) / len(throughputs)
    avg_security = sum(securities) / len(securities)
    
    plt.axvline(x=avg_throughput, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=avg_security, color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    plt.text(max(throughputs)*0.95, max(securities)*0.95, 'High Performance\nHigh Security', 
             ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='green', alpha=0.1))
    plt.text(min(throughputs)*1.05, max(securities)*0.95, 'Low Performance\nHigh Security', 
             ha='left', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.1))
    plt.text(max(throughputs)*0.95, min(securities)*1.05, 'High Performance\nLow Security', 
             ha='right', va='bottom', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.1))
    plt.text(min(throughputs)*1.05, min(securities)*1.05, 'Low Performance\nLow Security', 
             ha='left', va='bottom', fontsize=10, bbox=dict(boxstyle='round', facecolor='red', alpha=0.1))
    
    # Add colorbar for latency interpretation
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Latency Performance (darker = better)', fontweight='bold')
    
    # Add size legend for energy interpretation
    handles, labels = [], []
    energy_levels = [min(energies), (min(energies) + max(energies))/2, max(energies)]
    for energy in energy_levels:
        norm_energy = 1.0 - (energy/max(energies))
        size = 100 + 400 * norm_energy
        handles.append(plt.scatter([], [], s=size, color='gray', alpha=0.7, edgecolor='black', linewidth=1))
        labels.append(f'Energy: {energy:.2f}')
    
    plt.legend(handles, labels, title="Energy Consumption\n(larger = lower energy)", 
              title_fontsize=10, loc='upper left', framealpha=0.9)
    
    # Set axis labels and title with enhanced styling
    plt.xlabel('Throughput (tx/s)', fontweight='bold')
    plt.ylabel('Security Score', fontweight='bold')
    plt.title('Security vs. Performance Trade-off Under Various Attacks', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Add descriptive annotation
    plt.figtext(0.5, 0.01, 
               "Bubble size represents energy efficiency (larger bubbles = lower energy consumption).\n"
               "Color represents latency performance (darker = lower latency = better performance).",
               ha='center', fontsize=10, fontstyle='italic')
    
    # Apply tight layout and save with high quality
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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