#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Visualization Generator

This module generates visualizations related to blockchain performance for the QTrust project.
It creates multiple charts showing:
- Throughput comparison between different blockchain systems
- Multi-dimensional comparison using radar charts
- Latency comparison across systems
- Security and attack resistance metrics
- Energy consumption comparison

These visualizations demonstrate the performance advantages of the QTrust system
compared to other blockchain implementations based on comparative analysis data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Ensure the output directory exists
os.makedirs('docs/exported_charts', exist_ok=True)

def load_data():
    """Load data from CSV file."""
    # Tìm đường dẫn hiện tại của script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Đi lên một cấp từ thư mục visualization để đến thư mục data
    parent_dir = os.path.dirname(current_dir)
    # Xây dựng đường dẫn tuyệt đối đến file CSV
    csv_path = os.path.join(parent_dir, 'sources', 'comparative_analysis.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found at {csv_path}")
        print("Creating sample data for demonstration...")
        # Tạo dữ liệu mẫu nếu file không tồn tại
        return create_sample_data()
    
    return pd.read_csv(csv_path)

def create_sample_data():
    """Create sample data if CSV file is not found."""
    systems = ['QTrust', 'Ethereum 2.0', 'Polkadot', 'Solana', 'Algorand', 'Cardano']
    
    data = {
        'System': systems,
        'Throughput (tx/s)': [5000, 2500, 1500, 4000, 1200, 1000],
        'Latency (s)': [1.2, 3.5, 2.8, 1.5, 3.2, 4.0],
        'Energy Consumption': [0.5, 1.0, 0.8, 0.7, 0.6, 1.2],
        'Security': [0.95, 0.88, 0.90, 0.82, 0.92, 0.89],
        'Attack Resistance': [0.94, 0.86, 0.89, 0.81, 0.90, 0.88],
        'Reference': ['QTrust Labs, 2024', 'Ethereum Foundation, 2023', 'Polkadot Research, 2023', 
                      'Solana Labs, 2023', 'Algorand Inc., 2024', 'Cardano Foundation, 2023']
    }
    
    return pd.DataFrame(data)

def create_performance_comparison_chart(df):
    """Create chart comparing performance metrics."""
    plt.figure(figsize=(12, 7))
    
    # Get data
    systems = df['System']
    throughput = df['Throughput (tx/s)']
    
    # Create bar chart
    bars = plt.bar(systems, throughput, color=['#2C82C9', '#EF4836', '#8E44AD', '#F89406', '#16A085', '#7F8C8D'])
    
    # Add values on each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 20,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Throughput Comparison Between Blockchain Systems', fontsize=16, fontweight='bold')
    plt.ylabel('Throughput (Transactions/second)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(throughput) * 1.15)  # Add space for values above bars
    
    # Add data source
    references = df['Reference'].tolist()
    source_text = "Source: " + "; ".join(set([ref.split(',')[0] for ref in references]))
    plt.figtext(0.5, 0.01, source_text, ha='center', fontsize=10, style='italic')
    
    # Save chart
    plt.tight_layout(pad=3.0)
    plt.savefig('docs/exported_charts/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Performance comparison chart created: docs/exported_charts/performance_comparison.png")
    plt.close()

def create_radar_chart(df):
    """Create radar chart for multi-dimensional comparison."""
    # Prepare data
    categories = ['Throughput (tx/s)', 'Latency (s)', 'Energy Consumption', 'Security', 'Attack Resistance']
    
    # Normalize data (0-1)
    normalized_df = df.copy()
    
    # Invert values for latency and energy consumption (lower is better)
    max_latency = normalized_df['Latency (s)'].max()
    max_energy = normalized_df['Energy Consumption'].max()
    normalized_df['Latency (s)'] = (max_latency - normalized_df['Latency (s)']) / max_latency
    normalized_df['Energy Consumption'] = (max_energy - normalized_df['Energy Consumption']) / max_energy
    
    # Normalize other columns
    for col in ['Throughput (tx/s)', 'Security', 'Attack Resistance']:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    
    # Number of variables
    N = len(categories)
    
    # Angle for each axis
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the chart
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add each system to the chart
    colors = ['#2C82C9', '#EF4836', '#8E44AD', '#F89406', '#16A085', '#7F8C8D']
    for i, system in enumerate(normalized_df['System']):
        values = normalized_df.loc[i, categories].values.tolist()
        values += values[:1]  # Close the chart
        ax.plot(angles, values, linewidth=2, label=system, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set up axes
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    # Set y-ticks
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
    ax.grid(True)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Title
    plt.title('Multi-dimensional Comparison Between Blockchain Systems', size=15, fontweight='bold', y=1.1)
    
    # Add data source
    references = df['Reference'].tolist()
    source_text = "Source: " + "; ".join(set([ref.split(',')[0] for ref in references]))
    plt.figtext(0.5, 0.01, source_text, ha='center', fontsize=10, style='italic')
    
    # Save chart
    plt.tight_layout()
    plt.savefig('docs/exported_charts/radar_chart.png', dpi=300, bbox_inches='tight')
    print("Radar chart created: docs/exported_charts/radar_chart.png")
    plt.close()

def create_latency_chart(df):
    """Create chart comparing latency."""
    plt.figure(figsize=(12, 7))
    
    # Get data
    systems = df['System']
    latency = df['Latency (s)']
    
    # Create horizontal bar chart
    bars = plt.barh(systems, latency, color=['#2C82C9', '#EF4836', '#8E44AD', '#F89406', '#16A085', '#7F8C8D'])
    
    # Add values next to each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                 f'{width}s', va='center', fontweight='bold')
    
    plt.title('Latency Comparison Between Blockchain Systems', fontsize=16, fontweight='bold')
    plt.xlabel('Latency (seconds)', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim(0, max(latency) * 1.2)  # Add space for values
    
    # Add data source
    references = df['Reference'].tolist()
    source_text = "Source: " + "; ".join(set([ref.split(',')[0] for ref in references]))
    plt.figtext(0.5, 0.01, source_text, ha='center', fontsize=10, style='italic')
    
    # Save chart
    plt.tight_layout(pad=3.0)
    plt.savefig('docs/exported_charts/latency_chart.png', dpi=300, bbox_inches='tight')
    print("Latency chart created: docs/exported_charts/latency_chart.png")
    plt.close()

def create_security_chart(df):
    """Create chart comparing security and attack resistance."""
    plt.figure(figsize=(12, 8))
    
    # Get data
    systems = df['System']
    security = df['Security']
    attack_resistance = df['Attack Resistance']
    
    # Set up chart
    x = np.arange(len(systems))
    width = 0.35
    
    # Create grouped bar chart
    bar1 = plt.bar(x - width/2, security, width, label='Security', color='#2C82C9')
    bar2 = plt.bar(x + width/2, attack_resistance, width, label='Attack Resistance', color='#EF4836')
    
    # Add values on each bar
    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Security Comparison Between Blockchain Systems', fontsize=16, fontweight='bold')
    plt.ylabel('Rating (0-1)', fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(x, systems)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add data source
    references = df['Reference'].tolist()
    source_text = "Source: " + "; ".join(set([ref.split(',')[0] for ref in references[:3]]))
    plt.figtext(0.5, 0.01, source_text, ha='center', fontsize=10, style='italic')
    
    # Save chart
    plt.tight_layout(pad=3.0)
    plt.savefig('docs/exported_charts/security_chart.png', dpi=300, bbox_inches='tight')
    print("Security comparison chart created: docs/exported_charts/security_chart.png")
    plt.close()

def create_energy_chart(df):
    """Create chart comparing energy consumption."""
    plt.figure(figsize=(10, 6))
    
    # Get data
    systems = df['System']
    energy = df['Energy Consumption']
    
    # Create chart
    bars = plt.bar(systems, energy, color=['#2C82C9', '#EF4836', '#8E44AD', '#F89406', '#16A085', '#7F8C8D'])
    
    # Add values on each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Relative Energy Consumption Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Energy Consumption (relative units)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(energy) * 1.15)
    
    # Add data source
    references = df['Reference'].tolist()
    source_text = "Source: " + "; ".join(set([ref.split(',')[0] for ref in references[:2]]))
    plt.figtext(0.5, 0.01, source_text, ha='center', fontsize=10, style='italic')
    
    # Save chart
    plt.tight_layout(pad=3.0)
    plt.savefig('docs/exported_charts/energy_chart.png', dpi=300, bbox_inches='tight')
    print("Energy efficiency chart created: docs/exported_charts/energy_chart.png")
    plt.close()

def main():
    """Run all chart generation functions."""
    df = load_data()
    create_performance_comparison_chart(df)
    create_radar_chart(df)
    create_latency_chart(df)
    create_security_chart(df)
    create_energy_chart(df)
    print("All performance charts have been generated successfully.")

if __name__ == "__main__":
    main() 