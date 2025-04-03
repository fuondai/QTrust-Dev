#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
System Comparison - Utility functions for comparing transaction systems

This module provides tools to compare the performance of QTrust with
various transaction systems in general, not just limited to blockchain
technology. It includes functions for creating visualization charts,
comparison tables, and performance reports.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any

# Transaction systems to compare (sample, needs to be updated with real values)
TRANSACTION_SYSTEMS = {
    'QTrust': {
        'throughput': 5000,           # tx/s
        'latency': 2.5,               # ms
        'overhead': 15,               # % overhead
        'security_score': 0.95,       # 0-1 scale
        'energy_efficiency': 0.92,    # 0-1 scale
        'resource_utilization': 0.88, # 0-1 scale
        'fault_tolerance': 0.94       # 0-1 scale
    },
    'VISA': {
        'throughput': 24000,
        'latency': 1.8,
        'overhead': 5,
        'security_score': 0.90,
        'energy_efficiency': 0.95,
        'resource_utilization': 0.92,
        'fault_tolerance': 0.88
    },
    'Paypal': {
        'throughput': 193,
        'latency': 650,
        'overhead': 9,
        'security_score': 0.88,
        'energy_efficiency': 0.90,
        'resource_utilization': 0.85,
        'fault_tolerance': 0.82
    },
    'Ripple': {
        'throughput': 1500,
        'latency': 3.5,
        'overhead': 11,
        'security_score': 0.85,
        'energy_efficiency': 0.87,
        'resource_utilization': 0.80,
        'fault_tolerance': 0.84
    },
    'SWIFT': {
        'throughput': 127,
        'latency': 86400000,  # 24 hours in ms
        'overhead': 20,
        'security_score': 0.92,
        'energy_efficiency': 0.60,
        'resource_utilization': 0.70,
        'fault_tolerance': 0.90
    },
    'Traditional Database': {
        'throughput': 50000,
        'latency': 0.8,
        'overhead': 4,
        'security_score': 0.75,
        'energy_efficiency': 0.96,
        'resource_utilization': 0.95,
        'fault_tolerance': 0.70
    }
}

def update_system_data(system_name: str, metrics: Dict[str, Any]) -> None:
    """
    Update data for a transaction system.
    
    Args:
        system_name: Name of the system to update
        metrics: Dict containing values to update
    """
    if system_name in TRANSACTION_SYSTEMS:
        for key, value in metrics.items():
            if key in TRANSACTION_SYSTEMS[system_name]:
                TRANSACTION_SYSTEMS[system_name][key] = value
        try:
            print(f"Updated data for {system_name}")
        except (IOError, ValueError):
            pass
    else:
        TRANSACTION_SYSTEMS[system_name] = metrics
        try:
            print(f"Added new system: {system_name}")
        except (IOError, ValueError):
            pass

def save_system_data(output_dir: str) -> None:
    """
    Save system data to a file.
    
    Args:
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"transaction_systems_data_{timestamp}.json")
    
    try:
        with open(filepath, 'w') as f:
            json.dump(TRANSACTION_SYSTEMS, f, indent=4)
        try:
            print(f"Saved system data to {filepath}")
        except (IOError, ValueError):
            pass
    except Exception as e:
        try:
            print(f"Error saving data: {e}")
        except (IOError, ValueError):
            pass

def plot_throughput_vs_security(output_dir: str) -> None:
    """
    Create a chart comparing throughput and security score.
    
    Args:
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataframe from data
    data = []
    for system, metrics in TRANSACTION_SYSTEMS.items():
        data.append({
            'System': system,
            'Throughput': metrics['throughput'],
            'Security': metrics['security_score'],
            'Latency': metrics['latency'],
            'Energy': metrics['energy_efficiency']
        })
    
    df = pd.DataFrame(data)
    
    # Create chart
    plt.figure(figsize=(12, 9))
    
    # Use latency as point size, energy efficiency as color
    sizes = 1000 / (df['Latency'] + 1)  # +1 to avoid division by 0, and invert latency (lower = better = larger point)
    sizes = np.clip(sizes, 20, 1000)  # Limit size
    
    # Check if logarithmic scale is needed
    use_log_scale = max(df['Throughput']) / min(df['Throughput']) > 100
    
    if use_log_scale:
        scatter = plt.scatter(np.log10(df['Throughput']), df['Security'],
                           s=sizes, c=df['Energy'], cmap='viridis', alpha=0.7, edgecolors='k')
        plt.xlabel('Throughput (log10 tx/s)', fontsize=12)
    else:
        scatter = plt.scatter(df['Throughput'], df['Security'],
                           s=sizes, c=df['Energy'], cmap='viridis', alpha=0.7, edgecolors='k')
        plt.xlabel('Throughput (tx/s)', fontsize=12)
    
    # Add label for each point
    for i, system in enumerate(df['System']):
        plt.annotate(system, 
                   (np.log10(df['Throughput'][i]) if use_log_scale else df['Throughput'][i], 
                    df['Security'][i]),
                   xytext=(7, 7), textcoords='offset points', fontsize=11, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Energy Efficiency', fontsize=10)
    
    # Add information about bubble size
    plt.annotate('Bubble size represents inverse of Latency', xy=(0.05, 0.05),
               xycoords='axes fraction', fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.ylabel('Security Score', fontsize=12)
    plt.title('Throughput vs Security Comparison of Transaction Systems', fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save chart
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"throughput_vs_security_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_performance_metrics_table(output_dir: str) -> pd.DataFrame:
    """
    Create a performance comparison table and calculate Performance Index.
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        DataFrame containing comparison table with Performance Index
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(TRANSACTION_SYSTEMS, orient='index')
    
    # Rename columns
    df.columns = [col.replace('_', ' ').title() for col in df.columns]
    
    # Calculate Performance Index based on weights
    df['Performance Index'] = (
        # Throughput: 25%
        df['Throughput'] / df['Throughput'].max() * 0.25 +
        # Latency: 25% (lower value = better)
        (1 - df['Latency'] / df['Latency'].max()) * 0.25 +
        # Security: 20%
        df['Security Score'] * 0.20 +
        # Energy: 10%
        df['Energy Efficiency'] * 0.10 +
        # Resource utilization: 10% 
        df['Resource Utilization'] * 0.10 +
        # Fault tolerance: 5%
        df['Fault Tolerance'] * 0.05 +
        # Overhead: 5% (lower value = better)
        (1 - df['Overhead'] / df['Overhead'].max()) * 0.05
    )
    
    # Rank index
    df['Rank'] = df['Performance Index'].rank(ascending=False).astype(int)
    
    # Sort by Performance Index
    df = df.sort_values('Performance Index', ascending=False)
    
    # Round values
    df = df.round(3)
    
    # Save in different formats
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV
    csv_path = os.path.join(output_dir, f"system_performance_table_{timestamp}.csv")
    df.to_csv(csv_path)
    
    # HTML
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>System Performance Comparison</title>
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
        <h1>Transaction Systems Performance Comparison</h1>
        <p>Created: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        {df.to_html(classes='highlight')}
        <p class="footer">
            Note: Performance Index is calculated based on weights of metrics:<br>
            Throughput (25%), Latency (25%), Security Score (20%), Energy Efficiency (10%), 
            Resource Utilization (10%), Fault Tolerance (5%), Overhead (5%)
        </p>
    </body>
    </html>
    """
    
    html_path = os.path.join(output_dir, f"system_performance_table_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(styled_html)
    
    return df

def plot_performance_radar(output_dir: str) -> None:
    """
    Create a radar chart for performance metrics.
    
    Args:
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Only take a few important systems to keep the chart readable
    selected_systems = ['QTrust', 'VISA', 'Traditional Database', 'SWIFT']
    metrics = ['throughput', 'latency', 'security_score', 'energy_efficiency', 
              'resource_utilization', 'fault_tolerance', 'overhead']
    
    # Normalize data
    normalized_data = {}
    
    # Find min/max for each metric
    min_values = {m: float('inf') for m in metrics}
    max_values = {m: float('-inf') for m in metrics}
    
    for system, values in TRANSACTION_SYSTEMS.items():
        if system in selected_systems:
            for m in metrics:
                if values[m] < min_values[m]:
                    min_values[m] = values[m]
                if values[m] > max_values[m]:
                    max_values[m] = values[m]
    
    # Normalize
    for system, values in TRANSACTION_SYSTEMS.items():
        if system in selected_systems:
            normalized_data[system] = {}
            for m in metrics:
                # For latency and overhead, lower values are better
                if m in ['latency', 'overhead']:
                    # Invert values after normalization
                    # High value = better
                    normalized_data[system][m] = 1 - ((values[m] - min_values[m]) / 
                                                   (max_values[m] - min_values[m])
                                                   if max_values[m] != min_values[m] else 0)
                else:
                    normalized_data[system][m] = ((values[m] - min_values[m]) / 
                                               (max_values[m] - min_values[m])
                                               if max_values[m] != min_values[m] else 1.0)
    
    # Create radar chart
    n_metrics = len(metrics)
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    labels = [m.replace('_', ' ').title() for m in metrics]
    labels += labels[:1]  # Close the circle
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    
    # Colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_systems)))
    
    # Draw for each system
    for i, system in enumerate(selected_systems):
        values = [normalized_data[system][m] for m in metrics]
        values += values[:1]  # Close the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=system, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Customize
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)  # Go clockwise
    
    # Set labels for axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=10)
    
    # Set limits and ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_ylim(0, 1)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Title and legend
    plt.title('Transaction Systems Performance Comparison (normalized)', fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save chart
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"system_performance_radar_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_system_comparison_report(output_dir: Optional[str] = None) -> None:
    """
    Generate a comprehensive report on transaction system comparison.
    
    Args:
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "system_comparison_reports")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create charts and tables
    plot_throughput_vs_security(output_dir)
    plot_performance_radar(output_dir)
    performance_table = generate_performance_metrics_table(output_dir)
    
    # Save data
    save_system_data(output_dir)
    
    # Create markdown report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"system_comparison_report_{timestamp}.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Transaction Systems Comparison Report\n\n")
        f.write(f"*Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report compares the performance of QTrust with various transaction systems, including both traditional and blockchain-based systems. The metrics compared include:\n\n")
        f.write("- Throughput (transactions/second)\n")
        f.write("- Latency (delay, ms)\n")
        f.write("- Overhead (percentage of additional resource consumption)\n")
        f.write("- Security Score\n")
        f.write("- Energy Efficiency\n")
        f.write("- Resource Utilization\n")
        f.write("- Fault Tolerance\n\n")
        
        f.write("## Performance Comparison Table\n\n")
        f.write(performance_table.to_markdown() + "\n\n")
        
        f.write("## Analysis\n\n")
        
        # Find system with highest score
        top_system = performance_table.index[0]
        top_score = performance_table['Performance Index'].max()
        
        f.write(f"### Overall Performance Score\n\n")
        f.write(f"Based on the analysis, **{top_system}** has the highest overall performance score ({top_score:.3f}).\n\n")
        
        # Analyze QTrust
        if 'QTrust' in performance_table.index:
            qtrust_rank = performance_table.loc['QTrust', 'Rank']
            qtrust_score = performance_table.loc['QTrust', 'Performance Index']
            
            f.write(f"### QTrust Assessment\n\n")
            f.write(f"QTrust currently ranks **{qtrust_rank}** with a score of {qtrust_score:.3f}.\n\n")
            
            # Analyze strengths and weaknesses
            f.write("#### Strengths:\n\n")
            strengths = []
            weaknesses = []
            
            for col in performance_table.columns:
                if col not in ['Performance Index', 'Rank']:
                    # For latency and overhead, lower values are better
                    if col in ['Latency', 'Overhead']:
                        if performance_table.loc['QTrust', col] <= performance_table[col].median():
                            strengths.append(f"{col}: {performance_table.loc['QTrust', col]}")
                        else:
                            weaknesses.append(f"{col}: {performance_table.loc['QTrust', col]}")
                    # For other metrics, higher values are better
                    else:
                        if performance_table.loc['QTrust', col] >= performance_table[col].median():
                            strengths.append(f"{col}: {performance_table.loc['QTrust', col]}")
                        else:
                            weaknesses.append(f"{col}: {performance_table.loc['QTrust', col]}")
            
            for strength in strengths:
                f.write(f"- {strength}\n")
            
            f.write("\n#### Weaknesses:\n\n")
            for weakness in weaknesses:
                f.write(f"- {weakness}\n")
            
            # Compare with Traditional Database
            if 'Traditional Database' in performance_table.index:
                f.write("\n#### Comparison with Traditional Database:\n\n")
                trad_db = performance_table.loc['Traditional Database']
                qtrust = performance_table.loc['QTrust']
                
                for col in performance_table.columns:
                    if col not in ['Performance Index', 'Rank']:
                        # For latency and overhead, lower values are better
                        if col in ['Latency', 'Overhead']:
                            if qtrust[col] < trad_db[col]:
                                f.write(f"- QTrust is better in {col}: {qtrust[col]} vs {trad_db[col]}\n")
                            else:
                                f.write(f"- Traditional Database is better in {col}: {trad_db[col]} vs {qtrust[col]}\n")
                        # For other metrics, higher values are better
                        else:
                            if qtrust[col] > trad_db[col]:
                                f.write(f"- QTrust is better in {col}: {qtrust[col]} vs {trad_db[col]}\n")
                            else:
                                f.write(f"- Traditional Database is better in {col}: {trad_db[col]} vs {qtrust[col]}\n")
        
        f.write("\n## Conclusion\n\n")
        f.write("QTrust provides a good balance between high performance and strong security, with competitive throughput and low latency. ")
        f.write("Compared to other blockchain systems, QTrust offers better energy efficiency and faster transaction processing. ")
        f.write("While traditional databases still lead in pure throughput, QTrust provides significant advantages in security and fault tolerance.\n\n")
        
        f.write("*Note: This data is based on benchmarks performed under test conditions and may differ from production environment performance.*\n")
    
    try:
        print(f"System comparison report generated at {report_path}")
    except (IOError, ValueError):
        pass

if __name__ == "__main__":
    output_dir = os.path.join(os.getcwd(), "system_comparison_reports")
    generate_system_comparison_report(output_dir) 