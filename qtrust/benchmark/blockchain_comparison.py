#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Blockchain Comparison - Tools for comparing performance with other blockchain systems

This file provides functions to compare QTrust with other blockchain systems
based on various performance metrics.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any

# Import from blockchain_comparison_utils
from qtrust.benchmark.blockchain_comparison_utils import (
    plot_heatmap_comparison,
    plot_relationship_comparison,
    generate_comparison_table
)

# Comparison data (sample, needs to be updated with actual benchmark data)
BLOCKCHAIN_SYSTEMS = {
    'QTrust': {
        'throughput': 5000,        # transactions/second
        'latency': 2.5,            # ms
        'security': 0.95,          # 0-1 scale
        'energy': 15,              # watt-hours/tx (estimated)
        'scalability': 0.9,        # 0-1 scale
        'decentralization': 0.85,  # 0-1 scale
        'cross_shard_efficiency': 0.92,  # 0-1 scale
        'attack_resistance': 0.94  # 0-1 scale
    },
    'Ethereum 2.0': {
        'throughput': 3000,
        'latency': 12,
        'security': 0.92,
        'energy': 35,
        'scalability': 0.85,
        'decentralization': 0.80,
        'cross_shard_efficiency': 0.85,
        'attack_resistance': 0.90
    },
    'Solana': {
        'throughput': 65000,
        'latency': 0.4,
        'security': 0.80,
        'energy': 5,
        'scalability': 0.95,
        'decentralization': 0.65,
        'cross_shard_efficiency': 0.89,
        'attack_resistance': 0.78
    },
    'Algorand': {
        'throughput': 1200,
        'latency': 4.5,
        'security': 0.90,
        'energy': 8,
        'scalability': 0.82,
        'decentralization': 0.75,
        'cross_shard_efficiency': 0.88,
        'attack_resistance': 0.87
    },
    'Avalanche': {
        'throughput': 4500,
        'latency': 2,
        'security': 0.87,
        'energy': 20,
        'scalability': 0.88,
        'decentralization': 0.72,
        'cross_shard_efficiency': 0.90,
        'attack_resistance': 0.85
    },
    'Polkadot': {
        'throughput': 1500,
        'latency': 6,
        'security': 0.89,
        'energy': 25,
        'scalability': 0.87,
        'decentralization': 0.78,
        'cross_shard_efficiency': 0.94,
        'attack_resistance': 0.88
    },
    'Cardano': {
        'throughput': 1000,
        'latency': 10,
        'security': 0.91,
        'energy': 18,
        'scalability': 0.80,
        'decentralization': 0.82,
        'cross_shard_efficiency': 0.83,
        'attack_resistance': 0.89
    }
}

# Define resistance to various attack types (0-1, higher value = better)
ATTACK_RESISTANCE = {
    'QTrust': {
        'sybil_attack': 0.95,
        'ddos_attack': 0.90,
        '51_percent': 0.93,
        'eclipse_attack': 0.92,
        'smart_contract_exploit': 0.88
    },
    'Ethereum 2.0': {
        'sybil_attack': 0.92,
        'ddos_attack': 0.88,
        '51_percent': 0.94,
        'eclipse_attack': 0.85,
        'smart_contract_exploit': 0.80
    },
    'Solana': {
        'sybil_attack': 0.85,
        'ddos_attack': 0.92,
        '51_percent': 0.80,
        'eclipse_attack': 0.78,
        'smart_contract_exploit': 0.82
    },
    'Algorand': {
        'sybil_attack': 0.90,
        'ddos_attack': 0.85,
        '51_percent': 0.92,
        'eclipse_attack': 0.88,
        'smart_contract_exploit': 0.85
    },
    'Avalanche': {
        'sybil_attack': 0.88,
        'ddos_attack': 0.87,
        '51_percent': 0.89,
        'eclipse_attack': 0.84,
        'smart_contract_exploit': 0.79
    },
    'Polkadot': {
        'sybil_attack': 0.91,
        'ddos_attack': 0.86,
        '51_percent': 0.88,
        'eclipse_attack': 0.90,
        'smart_contract_exploit': 0.83
    },
    'Cardano': {
        'sybil_attack': 0.93,
        'ddos_attack': 0.84,
        '51_percent': 0.91,
        'eclipse_attack': 0.87,
        'smart_contract_exploit': 0.90
    }
}

def update_with_benchmark_results(benchmark_results: Dict[str, Any], system_name: str = 'QTrust') -> None:
    """
    Update simulation data with actual benchmark results.
    
    Args:
        benchmark_results: Dict containing benchmark results
        system_name: Name of the system to update
    """
    if system_name in BLOCKCHAIN_SYSTEMS:
        for key, value in benchmark_results.items():
            if key in BLOCKCHAIN_SYSTEMS[system_name]:
                BLOCKCHAIN_SYSTEMS[system_name][key] = value
                
        print(f"Updated benchmark data for {system_name}")
    else:
        BLOCKCHAIN_SYSTEMS[system_name] = benchmark_results
        print(f"Added new benchmark data for {system_name}")

def import_blockchain_data(filepath: str) -> None:
    """
    Import blockchain data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Check data structure
        for system_name, values in data.items():
            required_keys = ['throughput', 'latency', 'security']
            if all(key in values for key in required_keys):
                # Update if system exists or add new
                if system_name in BLOCKCHAIN_SYSTEMS:
                    for key, value in values.items():
                        BLOCKCHAIN_SYSTEMS[system_name][key] = value
                else:
                    BLOCKCHAIN_SYSTEMS[system_name] = values
            else:
                print(f"Skipping system {system_name} due to missing required information")
                
        print(f"Successfully imported data from {filepath}")
    except Exception as e:
        print(f"Error importing data: {e}")

def export_benchmark_data(output_dir: str) -> None:
    """
    Export current benchmark data to a JSON file.
    
    Args:
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"blockchain_data_{timestamp}.json")
    
    try:
        with open(filepath, 'w') as f:
            json.dump(BLOCKCHAIN_SYSTEMS, f, indent=4)
            
        print(f"Successfully exported benchmark data to {filepath}")
    except Exception as e:
        print(f"Error exporting data: {e}")

def plot_attack_resistance(output_dir: str) -> None:
    """
    Create a bar chart to compare attack resistance across blockchain systems.
    
    Args:
        output_dir: Output directory to save the chart
    """
    # Create DataFrame from attack resistance data
    systems = list(ATTACK_RESISTANCE.keys())
    attack_types = list(next(iter(ATTACK_RESISTANCE.values())).keys())
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(systems))  # System positions
    width = 0.15  # Width of the bars
    
    # Plot bars for each attack type
    for i, attack in enumerate(attack_types):
        values = [ATTACK_RESISTANCE[system][attack] for system in systems]
        plt.bar(x + (i - len(attack_types)/2 + 0.5) * width, values, width, 
               label=attack.replace('_', ' ').title())
    
    # Customize plot
    plt.xlabel('Blockchain Systems', fontsize=12)
    plt.ylabel('Resistance Score (0-1)', fontsize=12)
    plt.title('Blockchain Systems Attack Resistance Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x, systems)
    plt.ylim(0, 1.0)
    plt.legend(title='Attack Types', loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    """
    for i, attack in enumerate(attack_types):
        values = [ATTACK_RESISTANCE[system][attack] for system in systems]
        for j, value in enumerate(values):
            plt.text(j + (i - len(attack_types)/2 + 0.5) * width, value + 0.02, 
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
    """
    
    # Save chart
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"attack_resistance_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Attack resistance chart saved to {output_dir}")

def generate_comparison_report(output_dir: Optional[str] = None) -> None:
    """
    Generate a comprehensive comparison report with multiple visualizations.
    
    Args:
        output_dir: Output directory to save the report
    """
    # Create output directory if not provided
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('results', 'comparison', f'report_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create report index file
    report_path = os.path.join(output_dir, 'report.html')
    
    # Generate all visualizations
    print(f"Generating heatmap comparison...")
    plot_heatmap_comparison(BLOCKCHAIN_SYSTEMS, output_dir)
    
    print(f"Generating relationship comparison...")
    plot_relationship_comparison(BLOCKCHAIN_SYSTEMS, output_dir)
    
    print(f"Generating comparison table...")
    comparison_df = generate_comparison_table(BLOCKCHAIN_SYSTEMS, output_dir)
    
    print(f"Generating attack resistance chart...")
    plot_attack_resistance(output_dir)
    
    # Export raw data for reference
    export_benchmark_data(output_dir)
    
    # List all generated files
    files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    
    # Create an HTML index page with links to all generated files
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QTrust Blockchain Systems Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; }}
            .files {{ margin-top: 20px; }}
            .file-item {{ margin: 10px 0; }}
            .charts {{ display: flex; flex-wrap: wrap; }}
            .chart {{ margin: 10px; max-width: 45%; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .timestamp {{ color: #7f8c8d; font-size: 0.8em; }}
        </style>
    </head>
    <body>
        <h1>QTrust Blockchain Systems Comparison Report</h1>
        <p class="timestamp">Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Results Summary</h2>
        <p>This report compares QTrust with {len(BLOCKCHAIN_SYSTEMS) - 1} other blockchain systems across various performance metrics.</p>
        
        <div class="summary">
            <h3>Top Performers</h3>
            <ul>
                <li><strong>Overall Performance:</strong> {comparison_df.index[0]} (Score: {comparison_df.iloc[0]['Performance Score']:.3f})</li>
                <li><strong>Throughput:</strong> {comparison_df.sort_values('Throughput', ascending=False).index[0]} ({int(comparison_df.sort_values('Throughput', ascending=False).iloc[0]['Throughput'])} tx/s)</li>
                <li><strong>Latency:</strong> {comparison_df.sort_values('Latency').index[0]} ({comparison_df.sort_values('Latency').iloc[0]['Latency']} ms)</li>
                <li><strong>Security:</strong> {comparison_df.sort_values('Security', ascending=False).index[0]} ({comparison_df.sort_values('Security', ascending=False).iloc[0]['Security']:.2f}/1.0)</li>
            </ul>
        </div>
        
        <h2>Charts and Visualizations</h2>
        <div class="charts">
    """
    
    # Add links to all PNG files
    png_files = [f for f in files if f.endswith('.png')]
    for png_file in png_files:
        html_content += f"""
            <div class="chart">
                <h3>{png_file.split('_')[0].title()} {png_file.split('_')[1].title()}</h3>
                <img src="{png_file}" style="width: 100%; border: 1px solid #ddd;" />
            </div>
        """
    
    html_content += """
        </div>
        
        <h2>Generated Files</h2>
        <div class="files">
    """
    
    # Add links to all generated files
    for file in files:
        if file != 'report.html':
            file_type = file.split('.')[-1].upper()
            html_content += f"""
                <div class="file-item">
                    <a href="{file}" target="_blank">{file}</a> ({file_type})
                </div>
            """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nComparison report generated successfully!")
    print(f"Report location: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    # Generate comparison report when called directly
    generate_comparison_report() 