"""
Utilities for the QTrust system.
"""

from qtrust.utils.metrics import (
    calculate_transaction_throughput,
    calculate_latency_metrics,
    calculate_energy_efficiency,
    calculate_security_metrics,
    calculate_cross_shard_metrics,
    plot_performance_comparison,
    plot_metrics_over_time,
    calculate_throughput,
    calculate_cross_shard_transaction_ratio,
    plot_performance_metrics,
    plot_comparison_charts
)

from qtrust.utils.visualization import (
    plot_blockchain_network,
    plot_transaction_flow,
    plot_shard_graph,
    plot_consensus_comparison,
    plot_learning_curve
)

from qtrust.utils.data_generation import (
    generate_network_topology,
    assign_nodes_to_shards,
    generate_transactions,
    generate_network_events,
    generate_malicious_activities,
    assign_trust_scores
)

from qtrust.utils.config import (
    QTrustConfig,
    parse_arguments,
    update_config_from_args,
    load_config_from_args
)

__all__ = [
    # Metrics
    'calculate_transaction_throughput',
    'calculate_latency_metrics',
    'calculate_energy_efficiency',
    'calculate_security_metrics',
    'calculate_cross_shard_metrics',
    'plot_performance_comparison',
    'plot_metrics_over_time',
    'calculate_throughput',
    'calculate_cross_shard_transaction_ratio',
    'plot_performance_metrics',
    'plot_comparison_charts',
    
    # Visualization
    'plot_blockchain_network',
    'plot_transaction_flow',
    'plot_shard_graph',
    'plot_consensus_comparison',
    'plot_learning_curve',
    
    # Data generation
    'generate_network_topology',
    'assign_nodes_to_shards',
    'generate_transactions',
    'generate_network_events',
    'generate_malicious_activities',
    'assign_trust_scores',
    
    # Config management
    'QTrustConfig',
    'parse_arguments',
    'update_config_from_args',
    'load_config_from_args'
] 