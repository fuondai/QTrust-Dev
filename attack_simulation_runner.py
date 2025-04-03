"""
Attack Simulation Runner for QTrust Blockchain

This script simulates various attack scenarios on the QTrust blockchain system and
evaluates performance metrics under different scaling configurations. It generates
detailed reports and visualization charts to analyze system resilience against attacks
and scalability characteristics.

Features:
- Multiple attack type simulations (51%, Sybil, Eclipse, etc.)
- Network scaling comparison with different shard and node configurations
- Performance metrics tracking (throughput, latency, energy, security)
- Detailed report generation and visualization
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Add current directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from large_scale_simulation import LargeScaleBlockchainSimulation

def run_attack_comparison(base_args, output_dir='results_attack_comparison'):
    """Run comparison of different attack types on the blockchain."""
    attack_types = [None, '51_percent', 'sybil', 'eclipse', 'selfish_mining', 'bribery', 'ddos', 'finney', 'mixed']
    all_metrics = {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Starting attack comparison ===")
    
    # Run simulation for each attack type
    for attack_type in attack_types:
        attack_name = attack_type if attack_type else "no_attack"
        print(f"\n== Running simulation with attack: {attack_name} ==")
        
        # Set malicious node percentage based on attack type
        malicious_percentage = base_args.malicious
        if attack_type == '51_percent':
            malicious_percentage = 51
        elif attack_type == 'sybil':
            malicious_percentage = 30
        elif attack_type == 'selfish_mining':
            malicious_percentage = 20
        elif attack_type == 'bribery':
            malicious_percentage = 25
        elif attack_type == 'ddos':
            malicious_percentage = 15
        elif attack_type == 'finney':
            malicious_percentage = 20
        elif attack_type == 'mixed':
            malicious_percentage = 40
        
        # Create simulation
        simulation = LargeScaleBlockchainSimulation(
            num_shards=base_args.num_shards,
            nodes_per_shard=base_args.nodes_per_shard,
            malicious_percentage=malicious_percentage,
            attack_scenario=attack_type
        )
        
        # Run simulation with fewer steps to save time for attack scenarios
        steps = base_args.steps // 2 if attack_type else base_args.steps
        metrics = simulation.run_simulation(
            num_steps=steps,
            transactions_per_step=base_args.tx_per_step
        )
        
        # Save metrics
        all_metrics[attack_name] = {
            'throughput': np.mean(metrics['throughput'][-100:]),
            'latency': np.mean(metrics['latency'][-100:]),
            'energy': np.mean(metrics['energy'][-100:]),
            'security': np.mean(metrics['security'][-100:]),
            'cross_shard_ratio': np.mean(metrics['cross_shard_ratio'][-100:])
        }
        
        # Create charts and reports
        print(f"Generating detailed charts for attack {attack_name}...")
        simulation.plot_metrics(save_dir=output_dir)
        simulation.generate_report(save_dir=output_dir)
        print(f"Completed simulation with attack {attack_name}")
    
    # Create comparison report
    print("\nGenerating comparison report between attack types...")
    generate_comparison_report(all_metrics, output_dir)
    
    # Create comparison charts
    print("Generating comparison charts between attack types...")
    plot_comparison_charts(all_metrics, output_dir)
    
    print(f"Completed attack comparison. Results saved in {output_dir}")
    
    return all_metrics

def run_scale_comparison(base_args, output_dir='results_scale_comparison'):
    """Run comparison of different network scales."""
    # Different scale configurations
    scale_configs = [
        {"name": "small", "num_shards": 4, "nodes_per_shard": 10},
        {"name": "medium", "num_shards": 8, "nodes_per_shard": 20},
        {"name": "large", "num_shards": 16, "nodes_per_shard": 30},
        {"name": "xlarge", "num_shards": 32, "nodes_per_shard": 40}
    ]
    
    all_metrics = {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Starting network scale comparison ===")
    
    # Run simulation for each scale
    for config in scale_configs:
        print(f"\n== Running simulation with scale: {config['name']} ==")
        print(f"Configuration: {config['num_shards']} shards, {config['nodes_per_shard']} nodes/shard")
        
        # Create simulation
        simulation = LargeScaleBlockchainSimulation(
            num_shards=config['num_shards'],
            nodes_per_shard=config['nodes_per_shard'],
            malicious_percentage=base_args.malicious,
            attack_scenario=base_args.attack
        )
        
        # Adjust step count based on scale
        scale_factor = config['num_shards'] / base_args.num_shards
        steps = int(base_args.steps / scale_factor)
        steps = max(steps, 200)  # At least 200 steps
        
        # Run simulation
        metrics = simulation.run_simulation(
            num_steps=steps,
            transactions_per_step=base_args.tx_per_step
        )
        
        # Save metrics
        all_metrics[config['name']] = {
            'throughput': np.mean(metrics['throughput'][-100:]),
            'latency': np.mean(metrics['latency'][-100:]),
            'energy': np.mean(metrics['energy'][-100:]),
            'security': np.mean(metrics['security'][-100:]),
            'cross_shard_ratio': np.mean(metrics['cross_shard_ratio'][-100:]),
            'num_nodes': config['num_shards'] * config['nodes_per_shard']
        }
        
        # Create charts and reports
        print(f"Generating detailed charts for scale {config['name']}...")
        simulation.plot_metrics(save_dir=output_dir)
        simulation.generate_report(save_dir=output_dir)
        print(f"Completed simulation with scale {config['name']}")
    
    # Create comparison report
    print("\nGenerating comparison report between network scales...")
    generate_scale_comparison_report(all_metrics, output_dir)
    
    # Create comparison charts
    print("Generating comparison charts between network scales...")
    plot_scale_comparison_charts(all_metrics, output_dir)
    
    print(f"Completed scale comparison. Results saved in {output_dir}")
    
    return all_metrics

def generate_comparison_report(metrics, output_dir):
    """Generate comparison report for different attack types."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/attack_comparison_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Comparison of Attack Types in QTrust\n")
        f.write("=====================================\n\n")
        
        # Column headers
        f.write(f"{'Attack Type':<20} {'Throughput':<12} {'Latency':<12} {'Energy':<12} {'Security':<12} {'Cross-shard':<12}\n")
        f.write(f"{'-'*76}\n")
        
        # Data rows
        for attack_name, attack_metrics in metrics.items():
            display_name = "No Attack" if attack_name == "no_attack" else attack_name
            f.write(f"{display_name:<20} {attack_metrics['throughput']:<12.2f} {attack_metrics['latency']:<12.2f} ")
            f.write(f"{attack_metrics['energy']:<12.2f} {attack_metrics['security']:<12.2f} ")
            f.write(f"{attack_metrics['cross_shard_ratio']:<12.2f}\n")
        
        f.write("\nAnalysis:\n\n")
        
        # Find best/worst values
        best_throughput = max(metrics.items(), key=lambda x: x[1]['throughput'])
        worst_latency = max(metrics.items(), key=lambda x: x[1]['latency'])
        best_security = max(metrics.items(), key=lambda x: x[1]['security'])
        
        f.write(f"1. Best throughput: {best_throughput[0]} ({best_throughput[1]['throughput']:.2f} tx/s)\n")
        f.write(f"2. Highest latency: {worst_latency[0]} ({worst_latency[1]['latency']:.2f} ms)\n")
        f.write(f"3. Best security: {best_security[0]} ({best_security[1]['security']:.2f})\n\n")
        
        f.write("Impact assessment of attack types:\n\n")
        
        # Compare with no attack baseline
        baseline = metrics["no_attack"]
        for attack_name, attack_metrics in metrics.items():
            if attack_name == "no_attack":
                continue
                
            throughput_change = ((attack_metrics['throughput'] - baseline['throughput']) / baseline['throughput']) * 100
            latency_change = ((attack_metrics['latency'] - baseline['latency']) / baseline['latency']) * 100
            security_change = ((attack_metrics['security'] - baseline['security']) / baseline['security']) * 100
            
            f.write(f"Attack {attack_name}:\n")
            f.write(f"  - Throughput: {throughput_change:.2f}%\n")
            f.write(f"  - Latency: {latency_change:.2f}%\n")
            f.write(f"  - Security: {security_change:.2f}%\n\n")
    
    print(f"Comparison report saved at: {filename}")

def generate_scale_comparison_report(metrics, output_dir):
    """Generate comparison report for different network scales."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/scale_comparison_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("QTrust Performance Comparison Across Network Scales\n")
        f.write("===========================================\n\n")
        
        # Column headers
        f.write(f"{'Scale':<12} {'Nodes':<12} {'Throughput':<12} {'Latency':<12} {'Energy':<12} {'Security':<12} {'Cross-shard':<12}\n")
        f.write(f"{'-'*84}\n")
        
        # Data rows
        for scale_name, scale_metrics in metrics.items():
            f.write(f"{scale_name:<12} {scale_metrics['num_nodes']:<12d} {scale_metrics['throughput']:<12.2f} ")
            f.write(f"{scale_metrics['latency']:<12.2f} {scale_metrics['energy']:<12.2f} ")
            f.write(f"{scale_metrics['security']:<12.2f} {scale_metrics['cross_shard_ratio']:<12.2f}\n")
        
        f.write("\nScale Analysis:\n\n")
        
        # Calculate relationship between scale and performance
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1]['num_nodes'])
        
        smallest = sorted_metrics[0][1]
        largest = sorted_metrics[-1][1]
        
        node_scale = largest['num_nodes'] / smallest['num_nodes']
        throughput_scale = largest['throughput'] / smallest['throughput']
        latency_scale = largest['latency'] / smallest['latency']
        
        f.write(f"1. Node scaling ratio: {node_scale:.2f}x\n")
        f.write(f"2. Throughput scaling ratio: {throughput_scale:.2f}x\n")
        f.write(f"3. Latency increase ratio: {latency_scale:.2f}x\n\n")
        
        # Calculate scaling efficiency
        scaling_efficiency = throughput_scale / node_scale * 100
        f.write(f"Scaling efficiency: {scaling_efficiency:.2f}%\n")
        f.write(f"(100% is perfect linear scaling, >100% is super-linear, <100% is sub-linear)\n\n")
        
        f.write("Relationship between scale and latency:\n")
        for i in range(1, len(sorted_metrics)):
            prev = sorted_metrics[i-1][1]
            curr = sorted_metrics[i][1]
            
            node_increase = (curr['num_nodes'] - prev['num_nodes']) / prev['num_nodes'] * 100
            latency_increase = (curr['latency'] - prev['latency']) / prev['latency'] * 100
            
            f.write(f"  - When increasing nodes by {node_increase:.2f}%, latency increases by {latency_increase:.2f}%\n")
    
    print(f"Scale comparison report saved at: {filename}")

def plot_comparison_charts(metrics, output_dir):
    """Create comparison charts for different attack types."""
    # Prepare data
    attack_names = list(metrics.keys())
    display_names = ["No Attack" if name == "no_attack" else name for name in attack_names]
    
    throughputs = [metrics[name]['throughput'] for name in attack_names]
    latencies = [metrics[name]['latency'] for name in attack_names]
    securities = [metrics[name]['security'] for name in attack_names]
    energies = [metrics[name]['energy'] for name in attack_names]
    
    # Setup style
    plt.style.use('dark_background')
    sns.set(style="darkgrid")
    
    # Create nice color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    # Create bar charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Throughput
    bars1 = axes[0, 0].bar(display_names, throughputs, color=colors[0], alpha=0.7)
    axes[0, 0].set_title('Throughput (tx/s)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('tx/s', fontsize=12)
    axes[0, 0].grid(axis='y', alpha=0.3)
    # Add values on top of bars
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Latency
    bars2 = axes[0, 1].bar(display_names, latencies, color=colors[1], alpha=0.7)
    axes[0, 1].set_title('Latency (ms)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('ms', fontsize=12)
    axes[0, 1].grid(axis='y', alpha=0.3)
    # Add values on top of bars
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Security
    bars3 = axes[1, 0].bar(display_names, securities, color=colors[2], alpha=0.7)
    axes[1, 0].set_title('Security Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Score (0-1)', fontsize=12)
    axes[1, 0].grid(axis='y', alpha=0.3)
    # Add values on top of bars
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Energy
    bars4 = axes[1, 1].bar(display_names, energies, color=colors[3], alpha=0.7)
    axes[1, 1].set_title('Energy Consumption', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Energy Units', fontsize=12)
    axes[1, 1].grid(axis='y', alpha=0.3)
    # Add values on top of bars
    for bar in bars4:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Main title
    fig.suptitle('Performance Comparison Under Different Attack Types', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/attack_comparison_chart_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison chart saved at: {filename}")
    
    # Close chart to free memory
    plt.close(fig)
    
    # Create radar chart to compare all metrics
    plot_radar_comparison(metrics, output_dir)

def plot_scale_comparison_charts(metrics, output_dir):
    """Create comparison charts for different network scales."""
    # Prepare data
    scale_names = list(metrics.keys())
    num_nodes = [metrics[name]['num_nodes'] for name in scale_names]
    throughputs = [metrics[name]['throughput'] for name in scale_names]
    latencies = [metrics[name]['latency'] for name in scale_names]
    securities = [metrics[name]['security'] for name in scale_names]
    
    # Sort data by node count
    sorted_data = sorted(zip(num_nodes, scale_names, throughputs, latencies, securities))
    num_nodes = [d[0] for d in sorted_data]
    scale_names = [d[1] for d in sorted_data]
    throughputs = [d[2] for d in sorted_data]
    latencies = [d[3] for d in sorted_data]
    securities = [d[4] for d in sorted_data]
    
    # Setup style
    plt.style.use('dark_background')
    sns.set(style="darkgrid")
    
    # Better color palette
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    
    # Create chart
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Throughput vs Number of Nodes
    axes[0].plot(num_nodes, throughputs, 'o-', color=colors[0], linewidth=3, markersize=10)
    axes[0].set_title('Throughput vs Number of Nodes', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Number of Nodes', fontsize=14)
    axes[0].set_ylabel('Throughput (tx/s)', fontsize=14, color=colors[0])
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(num_nodes, throughputs, alpha=0.2, color=colors[0])
    
    # Mark points with labels
    for i, (x, y) in enumerate(zip(num_nodes, throughputs)):
        axes[0].annotate(f"{scale_names[i]}: {y:.2f}",
                        (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=12)
    
    # Latency vs Number of Nodes
    axes[1].plot(num_nodes, latencies, 'o-', color=colors[1], linewidth=3, markersize=10)
    axes[1].set_title('Latency vs Number of Nodes', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Number of Nodes', fontsize=14)
    axes[1].set_ylabel('Latency (ms)', fontsize=14, color=colors[1])
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(num_nodes, latencies, alpha=0.2, color=colors[1])
    
    # Mark points with labels
    for i, (x, y) in enumerate(zip(num_nodes, latencies)):
        axes[1].annotate(f"{scale_names[i]}: {y:.2f}",
                        (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=12)
    
    # Security vs Number of Nodes
    axes[2].plot(num_nodes, securities, 'o-', color=colors[2], linewidth=3, markersize=10)
    axes[2].set_title('Security vs Number of Nodes', fontsize=16, fontweight='bold')
    axes[2].set_xlabel('Number of Nodes', fontsize=14)
    axes[2].set_ylabel('Security Score', fontsize=14, color=colors[2])
    axes[2].grid(True, alpha=0.3)
    axes[2].fill_between(num_nodes, securities, alpha=0.2, color=colors[2])
    
    # Mark points with labels
    for i, (x, y) in enumerate(zip(num_nodes, securities)):
        axes[2].annotate(f"{scale_names[i]}: {y:.2f}",
                        (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=12)
    
    plt.tight_layout()
    
    # Main title
    fig.suptitle('QTrust Blockchain Scalability Assessment', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/scale_comparison_chart_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Scale comparison chart saved at: {filename}")
    
    # Close chart to free memory
    plt.close(fig)
    
    # Create scaling efficiency chart
    plot_scaling_efficiency(metrics, output_dir)

def plot_radar_comparison(metrics, output_dir):
    """Create radar chart to compare all attack types across metrics."""
    # Prepare data
    attack_names = list(metrics.keys())
    display_names = ["No Attack" if name == "no_attack" else name for name in attack_names]
    
    # Normalize metrics
    max_throughput = max(metrics[name]['throughput'] for name in attack_names)
    max_latency = max(metrics[name]['latency'] for name in attack_names)
    max_energy = max(metrics[name]['energy'] for name in attack_names)
    
    normalized_metrics = {}
    for name in attack_names:
        normalized_metrics[name] = {
            'throughput': metrics[name]['throughput'] / max_throughput,
            # Invert latency and energy since lower values are better
            'latency': 1 - (metrics[name]['latency'] / max_latency),
            'energy': 1 - (metrics[name]['energy'] / max_energy),
            'security': metrics[name]['security']
        }
    
    # Result categories
    categories = ['Throughput', 'Latency', 'Energy', 'Security']
    
    # Number of variables
    N = len(categories)
    
    # Angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Setup style
    plt.style.use('dark_background')
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Better color palette
    colors = plt.cm.rainbow(np.linspace(0, 1, len(attack_names)))
    
    # Add data for each attack type
    for i, name in enumerate(attack_names):
        values = [
            normalized_metrics[name]['throughput'],
            normalized_metrics[name]['latency'],
            normalized_metrics[name]['energy'],
            normalized_metrics[name]['security']
        ]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=display_names[i], color=colors[i])
        ax.fill(angles, values, alpha=0.2, color=colors[i])
    
    # Set labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=14)
    
    # Set y-ticks
    ax.set_ylim(0, 1)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_rlabel_position(0)
    ax.tick_params(axis='y', labelsize=10, colors='gray')
    
    # Add title and legend
    ax.set_title("Performance Comparison Under Different Attack Types", size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/attack_radar_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Radar comparison chart saved at: {filename}")
    
    # Close chart to free memory
    plt.close(fig)

def plot_scaling_efficiency(metrics, output_dir):
    """Create scaling efficiency chart."""
    # Prepare data
    scale_names = list(metrics.keys())
    sorted_data = sorted([(metrics[name]['num_nodes'], name) for name in scale_names])
    
    num_nodes = [d[0] for d in sorted_data]
    scale_names = [d[1] for d in sorted_data]
    
    # Calculate scaling efficiency
    base_nodes = num_nodes[0]
    base_throughput = metrics[scale_names[0]]['throughput']
    
    ideal_scaling = [base_throughput * (n / base_nodes) for n in num_nodes]
    actual_throughput = [metrics[name]['throughput'] for name in scale_names]
    
    scaling_efficiency = [(actual / ideal) * 100 for actual, ideal in zip(actual_throughput, ideal_scaling)]
    
    # Setup style
    plt.style.use('dark_background')
    
    # Create chart
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Primary axis: Throughput
    line1 = ax1.plot(num_nodes, actual_throughput, 'o-', color='#3498db', linewidth=3, markersize=10, label='Actual Throughput')
    line2 = ax1.plot(num_nodes, ideal_scaling, '--', color='#95a5a6', linewidth=2, label='Ideal Linear Scaling')
    ax1.set_xlabel('Number of Nodes', fontsize=14)
    ax1.set_ylabel('Throughput (tx/s)', color='#3498db', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.grid(True, alpha=0.3)
    
    # Mark points with labels
    for i, (x, y) in enumerate(zip(num_nodes, actual_throughput)):
        ax1.annotate(f"{scale_names[i]}",
                    (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=12)
    
    # Secondary axis: Scaling Efficiency
    ax2 = ax1.twinx()
    line3 = ax2.plot(num_nodes, scaling_efficiency, 'o-', color='#e74c3c', linewidth=3, markersize=10, label='Scaling Efficiency')
    ax2.set_ylabel('Scaling Efficiency (%)', color='#e74c3c', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Add horizontal line at 100%
    ax2.axhline(y=100, linestyle='--', color='#2ecc71', alpha=0.7, linewidth=2)
    
    # Mark scaling efficiency
    for i, (x, y) in enumerate(zip(num_nodes, scaling_efficiency)):
        ax2.annotate(f"{y:.1f}%",
                    (x, y), xytext=(0, -20),
                    textcoords='offset points', fontsize=12, color='#e74c3c')
    
    # Combine lines for legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=12)
    
    # Title
    plt.title('QTrust Blockchain Scaling Efficiency', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/scaling_efficiency_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Scaling efficiency chart saved at: {filename}")
    
    # Close chart to free memory
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='QTrust Attack and Scale Simulation Runner')
    parser.add_argument('--mode', type=str, choices=['attack', 'scale', 'both'], default='both',
                        help='Simulation mode: attack (attack comparison), scale (scale comparison), both (both types)')
    parser.add_argument('--num-shards', type=int, default=10, help='Base number of shards')
    parser.add_argument('--nodes-per-shard', type=int, default=20, help='Base number of nodes per shard')
    parser.add_argument('--steps', type=int, default=500, help='Number of simulation steps')
    parser.add_argument('--tx-per-step', type=int, default=50, help='Transactions per step')
    parser.add_argument('--malicious', type=float, default=10, help='Percentage of malicious nodes')
    parser.add_argument('--attack', type=str, choices=['51_percent', 'sybil', 'eclipse', 'mixed', None], 
                        default=None, help='Base attack scenario')
    parser.add_argument('--output-dir', type=str, default='results_comparison', help='Output directory for results')
    parser.add_argument('--no-display', action='store_true', help='Do not display detailed results on screen')
    parser.add_argument('--high-quality', action='store_true', help='Generate high quality charts (higher dpi)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup dpi for high quality charts
    if args.high_quality:
        plt.rcParams['figure.dpi'] = 300
    
    # Setup to not display charts on screen
    if args.no_display:
        plt.switch_backend('Agg')
    
    if args.mode == 'attack' or args.mode == 'both':
        attack_output_dir = os.path.join(args.output_dir, 'attack_comparison')
        attack_metrics = run_attack_comparison(args, attack_output_dir)
    
    if args.mode == 'scale' or args.mode == 'both':
        scale_output_dir = os.path.join(args.output_dir, 'scale_comparison')
        scale_metrics = run_scale_comparison(args, scale_output_dir)
    
    print("\nSimulation completed! Results have been saved in directory:", args.output_dir)

if __name__ == "__main__":
    main() 